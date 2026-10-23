# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 86 of 197

## 目录 (Table of Contents)

1. [Breaking the Factorization Barrier in Diffusion Language Models](#paper-2806)
2. [InteractBench: Benchmarking LLMs on Competitive Programming under Unrevealed Information](#paper-2807)
3. [Bridging the Perceptual Gap: Residual-Enhanced Downscaling and Manifold-Aware Perception Alignment Adaptation for NR-IQA](#paper-2808)
4. [DenseMLLM: Standard Multimodal LLMs for Dense Prediction](#paper-2809)
5. [An Asymmetric Latent Factorization-of-Tensors Model for Relation Analysis](#paper-2810)
6. [NeuroMamba: A Universal Spatiotemporal Module for Robust Perception in Degraded Sensory Streams](#paper-2811)
7. [SemRep : Generative Code Representation Learning with Code Transformations](#paper-2812)
8. [A Direct Approach for Handling Contextual Bandits with Latent State Dynamics](#paper-2813)
9. [VLA-ATTC: Adaptive Test-Time Compute for VLA Models with Relative Action Critic Model](#paper-2814)
10. [Sentinel-VLA: A Metacognitive VLA Model with Active Status Monitoring for Dynamic Reasoning and Error Recovery](#paper-2815)
11. [Trajectory Consistency for One-Step Generation on Euler Mean Flows](#paper-2816)
12. [Controllable Molecule Generation via Sparse Representation Editing: An Interpretability-Driven Perspective](#paper-2817)
13. [RTInfer: Real-Time Inference of Multiple DNNs on Edge GPUs](#paper-2818)
14. [ChartE$^3$: A Comprehensive Benchmark for End-to-End Chart Editing](#paper-2819)
15. [DECO: Decoupled Multimodal Diffusion Transformer for Bimanual Dexterous Manipulation with a Plugin Tactile Adapter](#paper-2820)
16. [MIMOMamba: From Scalar Duality to Matrix-Valued Attention](#paper-2821)
17. [Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting](#paper-2822)
18. [FedRGL: Robust Federated Graph Learning under Label Noise](#paper-2823)
19. [Private and Stable Test-time Adaptation with Differential Privacy](#paper-2824)
20. [HeraSys: Collaborative Serving of Multiple LLM Workflows via Fine-Grained End-to-End Optimization](#paper-2825)
21. [Learning from Comparison: Constrained Projection Policy Optimization for Pareto-Front Improvement](#paper-2826)
22. [The Implicit Bias of Steepest Descent with Mini-batch Stochastic Gradient](#paper-2827)
23. [Compile to Compress: Boosting Formal Theorem Provers by Compiler Outputs](#paper-2828)
24. [Data- and Variance-dependent Regret Bounds for Online Tabular MDPs](#paper-2829)
25. [Causes and Consequences of Representational Similarity in Machine Learning Models](#paper-2830)
26. [Large Language Models as Topological Thinkers: A Benchmark on Graph Persistent Homology](#paper-2831)
27. [ARC-Decode: Accelerated Decoding with Risk-Bounded Acceptance](#paper-2832)
28. [Anchor-guided Hypergraph Condensation with Dual-level Discrimination](#paper-2833)
29. [Self-Prompting Diffusion Transformer for Open-Vocabulary Scene Text Editing via In-Context Learning](#paper-2834)
30. [OpenSage: Self-programming Agent Generation Engine](#paper-2835)
31. [Generative Adaptation of Dynamics to Environmental Shifts via Weight-space Diffusion](#paper-2836)
32. [BandPO: Bridging Trust Regions and Ratio Clipping via Probability-Aware Bounds for LLM Reinforcement Learning](#paper-2837)
33. [Prefix-Cache-Aware Data Reordering for LLM-Augmented Database Analytics](#paper-2838)

---


<a id="paper-2806"></a>

## Breaking the Factorization Barrier in Diffusion Language Models

### Images

![02411fe9efa51fd4d0f1162a8a507a6edd57e2ebae71a8949e09857150c436d0.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/02411fe9efa51fd4d0f1162a8a507a6edd57e2ebae71a8949e09857150c436d0.jpg)

![265f7bb628099f2fc0064dc175c892cc7d107d293d2ab0a0bdb9b8ad94df46de.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/265f7bb628099f2fc0064dc175c892cc7d107d293d2ab0a0bdb9b8ad94df46de.jpg)

![30a50c7dd773a4fddc7bcc508a45d9a1045a869d77100d200829b2694088655e.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/30a50c7dd773a4fddc7bcc508a45d9a1045a869d77100d200829b2694088655e.jpg)

![4ee8bea456f0cdd8d2a563de25d8554daa00fcf195fcb9f2d20e542ee2ac6fc9.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/4ee8bea456f0cdd8d2a563de25d8554daa00fcf195fcb9f2d20e542ee2ac6fc9.jpg)

![60f788f6487aa3224884ae00698e117af217e385b12a716527706a6a464596c5.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/60f788f6487aa3224884ae00698e117af217e385b12a716527706a6a464596c5.jpg)

![76437b344fadee97402d94f9415978498bd0890ba236569e2b61c53ddd0c4aee.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/76437b344fadee97402d94f9415978498bd0890ba236569e2b61c53ddd0c4aee.jpg)

![94c3c7615aa5802fdeb9f8e0a9648597f6ec52b1cfa77eeae1e600dc4cea5513.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/94c3c7615aa5802fdeb9f8e0a9648597f6ec52b1cfa77eeae1e600dc4cea5513.jpg)

![a58db4f7c729ebe60524ef8bde9e62b550a139de55b316ffea75efffffd706e8.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/a58db4f7c729ebe60524ef8bde9e62b550a139de55b316ffea75efffffd706e8.jpg)

![abe65a193e80db7fc80f3a1f825b3ada801858c20f2d58b2b54b34317be9ca56.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/abe65a193e80db7fc80f3a1f825b3ada801858c20f2d58b2b54b34317be9ca56.jpg)

![bce6c7fcb7875eea0e4b797358c8db9ae3ddf98550deb7b8c6bf3b303b252a6b.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/bce6c7fcb7875eea0e4b797358c8db9ae3ddf98550deb7b8c6bf3b303b252a6b.jpg)

![f01ec8a7b962d8d6270c09d63a2cd873b57540af0a0fab97fc6056fa6c6db675.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/f01ec8a7b962d8d6270c09d63a2cd873b57540af0a0fab97fc6056fa6c6db675.jpg)

![f6c1642c8288b0e27a5a2d800fe56d51d71c4f0d519a2c43447fbc693e29fee3.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/images/f6c1642c8288b0e27a5a2d800fe56d51d71c4f0d519a2c43447fbc693e29fee3.jpg)

### Tables

![0a9df3ba5175d0484af180746baf746a052e2372a92824c9751240ee28f247f1.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/0a9df3ba5175d0484af180746baf746a052e2372a92824c9751240ee28f247f1.jpg)

![4bc534a291577d5c2ddc1246cee13f21b0eed4be4084127310939eb734db29b8.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/4bc534a291577d5c2ddc1246cee13f21b0eed4be4084127310939eb734db29b8.jpg)

![4be235addd46f76af697117bffb804852115a22daa9a45866d2266efd6705fdc.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/4be235addd46f76af697117bffb804852115a22daa9a45866d2266efd6705fdc.jpg)

![5170f7302018d13fad8ac5d712cb5456534eaeab7cd4138ac8715f9bdd9c0f21.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/5170f7302018d13fad8ac5d712cb5456534eaeab7cd4138ac8715f9bdd9c0f21.jpg)

![51a0154fb43d62d091cb2a2bbbeb8fcf1b6b31b173ff7fc14e6dcd55651b568d.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/51a0154fb43d62d091cb2a2bbbeb8fcf1b6b31b173ff7fc14e6dcd55651b568d.jpg)

![61b63fb28ffab00f79ce38ae258c1b4343a0b9e9a72b6ad63a4eebfbdb16a938.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/61b63fb28ffab00f79ce38ae258c1b4343a0b9e9a72b6ad63a4eebfbdb16a938.jpg)

![67c230d3679bda63006d2db5244cfaa5d64ed5d7593cea32d4ed0d9b9b76c204.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/67c230d3679bda63006d2db5244cfaa5d64ed5d7593cea32d4ed0d9b9b76c204.jpg)

![9a26ba235318735f4e730e09b1a7b34f736b15b6eeafab96c392b46679da6dbd.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/9a26ba235318735f4e730e09b1a7b34f736b15b6eeafab96c392b46679da6dbd.jpg)

![e4a261a03f87021ceb434e0166c5f12878bf2d117cf149356e3ba6c40aba5f8a.jpg](../icml_results/2842_SHAP-Guided%20Kernel%20Actor-Critic%20for%20Explainable%20Reinforcement%20Learning/tables/e4a261a03f87021ceb434e0166c5f12878bf2d117cf149356e3ba6c40aba5f8a.jpg)

## Breaking the Factorization Barrier in Diffusion Language Models


### Images

![0ace1a77b68dea482721e2dddb54237ff1ab36edd0e211835af91f706c96d035.jpg](../icml_results/2843_Breaking%20the%20Factorization%20Barrier%20in%20Diffusion%20Language%20Models/images/0ace1a77b68dea482721e2dddb54237ff1ab36edd0e211835af91f706c96d035.jpg)

![1d473209dadc45ba8f4a199a317771087f5c6fff0221fe0b05d69d735a702c33.jpg](../icml_results/2843_Breaking%20the%20Factorization%20Barrier%20in%20Diffusion%20Language%20Models/images/1d473209dadc45ba8f4a199a317771087f5c6fff0221fe0b05d69d735a702c33.jpg)

![5962aae3700aa199bd9c2356ba22bfba86386b0727ebe1004d9cc8e8994efc7c.jpg](../icml_results/2843_Breaking%20the%20Factorization%20Barrier%20in%20Diffusion%20Language%20Models/images/5962aae3700aa199bd9c2356ba22bfba86386b0727ebe1004d9cc8e8994efc7c.jpg)

![e6c16dff8377dc75bc4b6b10268145e6d2951bec545e2f364cec2ba2bbce53f6.jpg](../icml_results/2843_Breaking%20the%20Factorization%20Barrier%20in%20Diffusion%20Language%20Models/images/e6c16dff8377dc75bc4b6b10268145e6d2951bec545e2f364cec2ba2bbce53f6.jpg)

### Tables

![13e1b29d7ddb7dacb904206fbe4fd6c7b48dd3a332b71a5af45122a25b5080d5.jpg](../icml_results/2843_Breaking%20the%20Factorization%20Barrier%20in%20Diffusion%20Language%20Models/tables/13e1b29d7ddb7dacb904206fbe4fd6c7b48dd3a332b71a5af45122a25b5080d5.jpg)

![3ab882544ded432acf7759b17835b55fdef814da25a23cbff390771c9f54b192.jpg](../icml_results/2843_Breaking%20the%20Factorization%20Barrier%20in%20Diffusion%20Language%20Models/tables/3ab882544ded432acf7759b17835b55fdef814da25a23cbff390771c9f54b192.jpg)

![4198516fa25d9cd12782a97cba4c9c490e18979d97a0bd2f608b3f481b54ea67.jpg](../icml_results/2843_Breaking%20the%20Factorization%20Barrier%20in%20Diffusion%20Language%20Models/tables/4198516fa25d9cd12782a97cba4c9c490e18979d97a0bd2f608b3f481b54ea67.jpg)

![598c888709350c3b2421c0e47eac2d97cc35121c9f2d376d5765be1383bfd021.jpg](../icml_results/2843_Breaking%20the%20Factorization%20Barrier%20in%20Diffusion%20Language%20Models/tables/598c888709350c3b2421c0e47eac2d97cc35121c9f2d376d5765be1383bfd021.jpg)

![6e80909cfc89b0cabfefb03c0124e4a096d243303960a84a7be2541d1295e236.jpg](../icml_results/2843_Breaking%20the%20Factorization%20Barrier%20in%20Diffusion%20Language%20Models/tables/6e80909cfc89b0cabfefb03c0124e4a096d243303960a84a7be2541d1295e236.jpg)

![86116c490c61b2de0b6e3b2caf2545064773d05f927328e323f32a1c3b7d232f.jpg](../icml_results/2843_Breaking%20the%20Factorization%20Barrier%20in%20Diffusion%20Language%20Models/tables/86116c490c61b2de0b6e3b2caf2545064773d05f927328e323f32a1c3b7d232f.jpg)

![f9ef26334201c3d9de499c33ff6556b7b302fd6b75461436ee6d7866702a2a42.jpg](../icml_results/2843_Breaking%20the%20Factorization%20Barrier%20in%20Diffusion%20Language%20Models/tables/f9ef26334201c3d9de499c33ff6556b7b302fd6b75461436ee6d7866702a2a42.jpg)

<a id="paper-2807"></a>

## InteractBench: Benchmarking LLMs on Competitive Programming under Unrevealed Information


### Images

![252d13c446579e9078e1c7c35465bb7a3a68ffa9e1e2d384eb72a4dc00923249.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/images/252d13c446579e9078e1c7c35465bb7a3a68ffa9e1e2d384eb72a4dc00923249.jpg)

![9d8c7bfc313607243b95634b14424299b84f59d63b64e66ad7f9c8fdbd7607b5.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/images/9d8c7bfc313607243b95634b14424299b84f59d63b64e66ad7f9c8fdbd7607b5.jpg)

![a7e813f9f6a077e3d0a39f3032fb9d800d5484094756d329771fc021c3eacb3e.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/images/a7e813f9f6a077e3d0a39f3032fb9d800d5484094756d329771fc021c3eacb3e.jpg)

![f826797cc2d54c929931683acc7bd85f600e4f75aee8aba93b8dd50a57c97391.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/images/f826797cc2d54c929931683acc7bd85f600e4f75aee8aba93b8dd50a57c97391.jpg)

![f915c139b21d3371909693c9743791cf9bebb93f9a4276ffa75e362d074fdd9e.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/images/f915c139b21d3371909693c9743791cf9bebb93f9a4276ffa75e362d074fdd9e.jpg)

### Tables

![177bbeb353f68a29f49f2ea31b4bbac7e7c954db4f6cafb9ffc2132c7ee01867.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/177bbeb353f68a29f49f2ea31b4bbac7e7c954db4f6cafb9ffc2132c7ee01867.jpg)

![1f748e0fac6d21b7d2639ce91da5b1ba324c13ec7a714078756139531549871e.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/1f748e0fac6d21b7d2639ce91da5b1ba324c13ec7a714078756139531549871e.jpg)

![27d9de1493c747810c664ee144150ec78cbcd04a8a9d94f3a84189d5d2692a35.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/27d9de1493c747810c664ee144150ec78cbcd04a8a9d94f3a84189d5d2692a35.jpg)

![2b20baa76b2c8018b673046a06e81a2749b3bce816230e8c9442b5aa64050b91.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/2b20baa76b2c8018b673046a06e81a2749b3bce816230e8c9442b5aa64050b91.jpg)

![304f9f56cfa6239b444a5835765d65aa8d551d31473726144e659b43c69ca883.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/304f9f56cfa6239b444a5835765d65aa8d551d31473726144e659b43c69ca883.jpg)

![305c442fb597d029e34d2ba8d329c4c1658f2ac394f06616fae1a07941044f90.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/305c442fb597d029e34d2ba8d329c4c1658f2ac394f06616fae1a07941044f90.jpg)

![32acbba737706506589b709b35c69f279f4fa2a0624b1c66e6c451cd518f0f2d.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/32acbba737706506589b709b35c69f279f4fa2a0624b1c66e6c451cd518f0f2d.jpg)

![340baae761042c5889e16bbfc1ef66f235d527ab9351caf3b85061ee68ba8fb3.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/340baae761042c5889e16bbfc1ef66f235d527ab9351caf3b85061ee68ba8fb3.jpg)

![3cef26e293b545b04b4b7bf2059a6559dde03482db3cee3a74531decc6c5da2f.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/3cef26e293b545b04b4b7bf2059a6559dde03482db3cee3a74531decc6c5da2f.jpg)

![413e1e459c15e02cf27519bcf69827079b75876c8a920fceabf12be6870e75c2.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/413e1e459c15e02cf27519bcf69827079b75876c8a920fceabf12be6870e75c2.jpg)

![436b7c7d50216251f1954ef14745674df4518eb7882c9fea7a82e3fb9c506863.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/436b7c7d50216251f1954ef14745674df4518eb7882c9fea7a82e3fb9c506863.jpg)

![53c2f6d7fe4dd1bb5900c4b70cd388009514afdc748e7ded9a0cb2683b5e4827.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/53c2f6d7fe4dd1bb5900c4b70cd388009514afdc748e7ded9a0cb2683b5e4827.jpg)

![5647848298eb6b31bd7c57c0b45f6770ed1b7e6fe7aa5d8b073217ffe1756365.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/5647848298eb6b31bd7c57c0b45f6770ed1b7e6fe7aa5d8b073217ffe1756365.jpg)

![6c96fc3abae32173f3c2bd2cb13fe57079bb01ae3c41b854ac2d7c3243d77179.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/6c96fc3abae32173f3c2bd2cb13fe57079bb01ae3c41b854ac2d7c3243d77179.jpg)

![6e805b4f4188889a57b2ed9ac552a16f7d280fe3fa7ff04a10792bf267880b0b.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/6e805b4f4188889a57b2ed9ac552a16f7d280fe3fa7ff04a10792bf267880b0b.jpg)

![94cc3b8174c78dfc920dd56d9321bdaaf5a43f424dd3220e67a06ac9c151d528.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/94cc3b8174c78dfc920dd56d9321bdaaf5a43f424dd3220e67a06ac9c151d528.jpg)

![9e38c156cc477f226698a3788af6b38ad06ce767c615ef320d21e2bfcc069c24.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/9e38c156cc477f226698a3788af6b38ad06ce767c615ef320d21e2bfcc069c24.jpg)

![9e48cddebbeb1dc0043df485b7e9076b2ab00e8e02638be8c708976ebc765fe3.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/9e48cddebbeb1dc0043df485b7e9076b2ab00e8e02638be8c708976ebc765fe3.jpg)

![a102efd3ed38021209eabfa4f2099e571190baa647035945328b27e3c4ecc7fe.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/a102efd3ed38021209eabfa4f2099e571190baa647035945328b27e3c4ecc7fe.jpg)

![a2bbfe8fa73d48a68466cbc47fc74a260b2b1193c913ca8e36aa9991ac41482c.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/a2bbfe8fa73d48a68466cbc47fc74a260b2b1193c913ca8e36aa9991ac41482c.jpg)

![ad1050f83422eef774d809fddf487a4ea160f7168820122061374362df1b3a49.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/ad1050f83422eef774d809fddf487a4ea160f7168820122061374362df1b3a49.jpg)

![b1929bd575d7d47a909de7fba718e271b36ad93259280b7a52a1d1e114b78608.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/b1929bd575d7d47a909de7fba718e271b36ad93259280b7a52a1d1e114b78608.jpg)

![b244d2d5623ac5a415d77c891d0335cc07b09418d832b733be89a3ca72092300.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/b244d2d5623ac5a415d77c891d0335cc07b09418d832b733be89a3ca72092300.jpg)

![d0e673dbe885f13bd7d9ea73007abdff58fa5b7cc93b94998e01efadb0174058.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/d0e673dbe885f13bd7d9ea73007abdff58fa5b7cc93b94998e01efadb0174058.jpg)

![d26a7258739e9482aa8d3dd4a20579b17c9bcb33ebd67479a32a61a8879bd3d1.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/d26a7258739e9482aa8d3dd4a20579b17c9bcb33ebd67479a32a61a8879bd3d1.jpg)

![eb11cde6018bb9bc51e50752b05842d3ff608f1ab98d43012f18195d313ac67c.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/eb11cde6018bb9bc51e50752b05842d3ff608f1ab98d43012f18195d313ac67c.jpg)

![eb54c4271895096d9cf1f8d058226713faad7497c30d1f74f3b6862a1bb3ee22.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/eb54c4271895096d9cf1f8d058226713faad7497c30d1f74f3b6862a1bb3ee22.jpg)

![ec61c11a896df89dda77f0159fa1f1a868cae975653b0a1d1a593f0e27b2d5b5.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/ec61c11a896df89dda77f0159fa1f1a868cae975653b0a1d1a593f0e27b2d5b5.jpg)

![f2e2db732f8e5d4e088ace5f8c8f226efc8506f0403c5366392385526b16b119.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/f2e2db732f8e5d4e088ace5f8c8f226efc8506f0403c5366392385526b16b119.jpg)

![f7690ac7e2ff90eccd4eb11f8bf54dd98f40f22420d259b09577b1ff08a6d63c.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/f7690ac7e2ff90eccd4eb11f8bf54dd98f40f22420d259b09577b1ff08a6d63c.jpg)

![fc194bf0ea3ff647ca3db9ff194af05548fa24171ec714c228432e0bfbf2ef48.jpg](../icml_results/2844_InteractBench_%20Benchmarking%20LLMs%20on%20Competitive%20Programming%20under%20Unrevealed%20Information/tables/fc194bf0ea3ff647ca3db9ff194af05548fa24171ec714c228432e0bfbf2ef48.jpg)

<a id="paper-2808"></a>

## Bridging the Perceptual Gap: Residual-Enhanced Downscaling and Manifold-Aware Perception Alignment Adaptation for NR-IQA


### Images

![193ba4f7bf9b3fa6443e237baa2e1a44ca87ec1499396d02864713614861241d.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/images/193ba4f7bf9b3fa6443e237baa2e1a44ca87ec1499396d02864713614861241d.jpg)

![569d5783fd4628b58d0397aa58362f57a5bcf7b1ef1967d8892983c50aa47ed3.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/images/569d5783fd4628b58d0397aa58362f57a5bcf7b1ef1967d8892983c50aa47ed3.jpg)

![7c9e7805a02793411cab8f20dd0aae2a4867c37d254468616e651c9a91e3e7ee.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/images/7c9e7805a02793411cab8f20dd0aae2a4867c37d254468616e651c9a91e3e7ee.jpg)

![81ec71c5b540e4e3fa8d9b5d097cc6607a7d18a3373593599f22d00589f0dbdf.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/images/81ec71c5b540e4e3fa8d9b5d097cc6607a7d18a3373593599f22d00589f0dbdf.jpg)

![a9758fcb4a3c55eb3a5cb2d6b5e9c92415ddbd50f38f07a05e2bf47d6d3c6d6e.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/images/a9758fcb4a3c55eb3a5cb2d6b5e9c92415ddbd50f38f07a05e2bf47d6d3c6d6e.jpg)

![ae06af53afe028a09bc58e898e79220493fdb70f4aefb4941634b20b6db8fddd.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/images/ae06af53afe028a09bc58e898e79220493fdb70f4aefb4941634b20b6db8fddd.jpg)

![bf16b9ead46ea44cc7ddfc75cfa0d5877eb7269cfed648f43dca4595c9cd1584.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/images/bf16b9ead46ea44cc7ddfc75cfa0d5877eb7269cfed648f43dca4595c9cd1584.jpg)

![cc1d9291b2aed7d28e1470ecf013383fe1a789ec5b2668e667d3c57113e361d4.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/images/cc1d9291b2aed7d28e1470ecf013383fe1a789ec5b2668e667d3c57113e361d4.jpg)

![d8abb489e1c3d5a5905ef6b1a994e3c5ea9de3bde27ac073f16575fbe19294ea.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/images/d8abb489e1c3d5a5905ef6b1a994e3c5ea9de3bde27ac073f16575fbe19294ea.jpg)

![da0ee5266a2566f35fc2cbe4e7748a7222a93b2f8a7676d2ae827a3700d4f82e.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/images/da0ee5266a2566f35fc2cbe4e7748a7222a93b2f8a7676d2ae827a3700d4f82e.jpg)

![e32817adde1dd6351963bc1d4fa0d3d62ce79b24a128e21c7aca6b6e20de230d.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/images/e32817adde1dd6351963bc1d4fa0d3d62ce79b24a128e21c7aca6b6e20de230d.jpg)

![fa330b5fbb2451b71df5c1f3dd888f8689eb63aa4409b53af50840c679f12959.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/images/fa330b5fbb2451b71df5c1f3dd888f8689eb63aa4409b53af50840c679f12959.jpg)

### Tables

![10f3d0b2e0fcbcfe8a2baadbb1a1c1bc953ce09d78f6f1424ab7b23a408722b6.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/10f3d0b2e0fcbcfe8a2baadbb1a1c1bc953ce09d78f6f1424ab7b23a408722b6.jpg)

![466e5b3d62f3de82d40dcfdd5255587db932f65b526fd0e936828622209796e4.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/466e5b3d62f3de82d40dcfdd5255587db932f65b526fd0e936828622209796e4.jpg)

![468b5ce69c071c385a76ac1a203e016925b7231d1f62fd9c7dc7dcbb5286b302.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/468b5ce69c071c385a76ac1a203e016925b7231d1f62fd9c7dc7dcbb5286b302.jpg)

![4a487e4f5eaeffdc729ce913b2fa97298f1e40984259147d6be2e7179354c6b7.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/4a487e4f5eaeffdc729ce913b2fa97298f1e40984259147d6be2e7179354c6b7.jpg)

![4d1b5c0564d8453a8169f0a6301b3dd7a5355c089b4e4a9c34aee495f1622e26.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/4d1b5c0564d8453a8169f0a6301b3dd7a5355c089b4e4a9c34aee495f1622e26.jpg)

![52c2f18bc37510d91e151c1952e592a19d16d43542ba758214957dbf65dd2325.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/52c2f18bc37510d91e151c1952e592a19d16d43542ba758214957dbf65dd2325.jpg)

![5c4fef04f3565aed89de65bf406598bbb5bba32a77118cbd82071c067d96a7ab.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/5c4fef04f3565aed89de65bf406598bbb5bba32a77118cbd82071c067d96a7ab.jpg)

![7115aef568598dc1c3b65cabae8b68891cc9f75c2c422435780478521b3d96f8.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/7115aef568598dc1c3b65cabae8b68891cc9f75c2c422435780478521b3d96f8.jpg)

![7e1ba8182e6580d9d5301824ec54adc8eadfdc619db4a142a33ec75c7200b8a0.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/7e1ba8182e6580d9d5301824ec54adc8eadfdc619db4a142a33ec75c7200b8a0.jpg)

![9332d7d74283f637a4989580a6a6f43890a2020b5d936623330803eb72359859.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/9332d7d74283f637a4989580a6a6f43890a2020b5d936623330803eb72359859.jpg)

![958a8ba3ec9970676fec95110a4e352f7f5c6ec4fe6a51dda625391615b547af.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/958a8ba3ec9970676fec95110a4e352f7f5c6ec4fe6a51dda625391615b547af.jpg)

![9ffba14f672e54238c290d39deca4b26ee415fd12bb06e91550b4357e7c29d44.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/9ffba14f672e54238c290d39deca4b26ee415fd12bb06e91550b4357e7c29d44.jpg)

![a769dedf79132f8cfbc7c373ffe2a3b6e69be31a120ba461b25104d8d800a720.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/a769dedf79132f8cfbc7c373ffe2a3b6e69be31a120ba461b25104d8d800a720.jpg)

![c72d451eeb256c8b66abdad62134282e9bbb72171acd6cb80882fc226b40343f.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/c72d451eeb256c8b66abdad62134282e9bbb72171acd6cb80882fc226b40343f.jpg)

![c94cc9d0d9b5c7305a7dd7a4861fc7aeb21dfd20e7ff67b5583656ab6ab7c01f.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/c94cc9d0d9b5c7305a7dd7a4861fc7aeb21dfd20e7ff67b5583656ab6ab7c01f.jpg)

![cc7ee7e30e22188a9693ec6fe8c87d99717b7292c2d73eaddb9cea7da93dd60e.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/cc7ee7e30e22188a9693ec6fe8c87d99717b7292c2d73eaddb9cea7da93dd60e.jpg)

![e619f1cb3b9230803c5348770fcfbb601b33962cde6f6168a113005303386df5.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/e619f1cb3b9230803c5348770fcfbb601b33962cde6f6168a113005303386df5.jpg)

![e80d6f4b1ec5741cc666c140155b0eb1e38deec299c1e4c570e4d02c2c7a34fe.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/e80d6f4b1ec5741cc666c140155b0eb1e38deec299c1e4c570e4d02c2c7a34fe.jpg)

![ede00438b5f5e7c6e3bbbbb9b6de512c9004508cbab23484cd5737cd4e5e45a7.jpg](../icml_results/2845_Bridging%20the%20Perceptual%20Gap_%20Residual-Enhanced%20Downscaling%20and%20Manifold-Aware%20Perception%20Alignment%20A/tables/ede00438b5f5e7c6e3bbbbb9b6de512c9004508cbab23484cd5737cd4e5e45a7.jpg)

<a id="paper-2809"></a>

## DenseMLLM: Standard Multimodal LLMs for Dense Prediction


### Images

![09c22be34588189b10b5d77979500188af398e75df6daaaa2e84554d0a70f081.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/images/09c22be34588189b10b5d77979500188af398e75df6daaaa2e84554d0a70f081.jpg)

![337a88a0f1ffc90d6f767a25c613efd53bf0c2f52a4d73678fa20408c60b1a35.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/images/337a88a0f1ffc90d6f767a25c613efd53bf0c2f52a4d73678fa20408c60b1a35.jpg)

![4419fc4467fadc188d489eff26af1a28979462de6899b2a80738062648e063d8.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/images/4419fc4467fadc188d489eff26af1a28979462de6899b2a80738062648e063d8.jpg)

![59cd0f0bd5a905470e0b755f3aec66cf085a1575d9ee2473c8aeac30a800430f.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/images/59cd0f0bd5a905470e0b755f3aec66cf085a1575d9ee2473c8aeac30a800430f.jpg)

![64077f83cf789e6e179a2d8b041b5a8b57788a34fc6309d90153aa5d0036484c.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/images/64077f83cf789e6e179a2d8b041b5a8b57788a34fc6309d90153aa5d0036484c.jpg)

![7758d3ff9f0b7c54ea6f34d13a43024a2e49a97bbfd102b007cbbde93aecc8a8.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/images/7758d3ff9f0b7c54ea6f34d13a43024a2e49a97bbfd102b007cbbde93aecc8a8.jpg)

![8fa373d297c060264f2d61b615b6056318b61095c429bb85973d415ba68823cf.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/images/8fa373d297c060264f2d61b615b6056318b61095c429bb85973d415ba68823cf.jpg)

![91f54d66776b0a0c32b2c4d648b24e89572e3907e36b0618aaa3f22ca8d110c4.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/images/91f54d66776b0a0c32b2c4d648b24e89572e3907e36b0618aaa3f22ca8d110c4.jpg)

![97edf91b90230cacff0fe05ad59c7764c2302634fb69c4dda5e34f91a7ba30f3.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/images/97edf91b90230cacff0fe05ad59c7764c2302634fb69c4dda5e34f91a7ba30f3.jpg)

![bf14b8b44226d4235a7cdd192bab4f50e84f0870d1581f0806d6d8c7fb44235e.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/images/bf14b8b44226d4235a7cdd192bab4f50e84f0870d1581f0806d6d8c7fb44235e.jpg)

![d26ace021f33654ab4de153f5e3527022e4c57af7a856fdc1a5d279b6c73bddf.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/images/d26ace021f33654ab4de153f5e3527022e4c57af7a856fdc1a5d279b6c73bddf.jpg)

### Tables

![0fb41f1104c6e09017f3c5c34f4c62b342b7964a223cfef4ff97358a01e84567.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/tables/0fb41f1104c6e09017f3c5c34f4c62b342b7964a223cfef4ff97358a01e84567.jpg)

![10057f9df52b145d8a834623589d5d27876108fb2ac1c9c7e05273f88a0f77e7.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/tables/10057f9df52b145d8a834623589d5d27876108fb2ac1c9c7e05273f88a0f77e7.jpg)

![1b49976becdb76250ba4ad53db30cc0865e469977002ed161c422947f70c9b7a.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/tables/1b49976becdb76250ba4ad53db30cc0865e469977002ed161c422947f70c9b7a.jpg)

![1dadb1eed8718a0b34c3d067614ec47784e1b4838b90b98bf56c629f69a702b7.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/tables/1dadb1eed8718a0b34c3d067614ec47784e1b4838b90b98bf56c629f69a702b7.jpg)

![1fc9b922dabb72ba103a7bdf2c9f6fbcf638a9d8ee913032763b84f2247ae3a4.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/tables/1fc9b922dabb72ba103a7bdf2c9f6fbcf638a9d8ee913032763b84f2247ae3a4.jpg)

![2250b9247ba82a0e074bb84eb689e25bec13fa244de6075733459f697d325799.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/tables/2250b9247ba82a0e074bb84eb689e25bec13fa244de6075733459f697d325799.jpg)

![292af35e137132d4f6cbf95b56493ad44eb615154d4a8cc25dffbc772d3d36ee.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/tables/292af35e137132d4f6cbf95b56493ad44eb615154d4a8cc25dffbc772d3d36ee.jpg)

![2cf658147aa6a5a7dc892f5bd82815d8b15d9485e9e875f4db0ddb410629b712.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/tables/2cf658147aa6a5a7dc892f5bd82815d8b15d9485e9e875f4db0ddb410629b712.jpg)

![5ecc7b9ed63f8ddf91528b583932df38f62113dc035e01ca042fd23be4ce0d63.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/tables/5ecc7b9ed63f8ddf91528b583932df38f62113dc035e01ca042fd23be4ce0d63.jpg)

![650912caaeb76b1ecf6a3b227eb99fbe32432bc1e5ac5cf189498a5287eb826b.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/tables/650912caaeb76b1ecf6a3b227eb99fbe32432bc1e5ac5cf189498a5287eb826b.jpg)

![6bafd6feb484b7270c7bb6954b198ae2ef4558b2144c65408e27a4762c5d4e44.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/tables/6bafd6feb484b7270c7bb6954b198ae2ef4558b2144c65408e27a4762c5d4e44.jpg)

![863ad5d9c79553946df4261faf8ac36acd21628174e57ffb21ea6cb1f0ff7d44.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/tables/863ad5d9c79553946df4261faf8ac36acd21628174e57ffb21ea6cb1f0ff7d44.jpg)

![a17400ac84c594c465211fd8a35cf1d91fe0378733597095e1ea3b56f82f9e7b.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/tables/a17400ac84c594c465211fd8a35cf1d91fe0378733597095e1ea3b56f82f9e7b.jpg)

![a1a6e506744f17b485f61644f6373f507cf754defcb54b45b8b30d182094c891.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/tables/a1a6e506744f17b485f61644f6373f507cf754defcb54b45b8b30d182094c891.jpg)

![bca6f5597ed6da0d6d04b62f41b563d48a6484032ca7ab28adf74af156bb0a5a.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/tables/bca6f5597ed6da0d6d04b62f41b563d48a6484032ca7ab28adf74af156bb0a5a.jpg)

![c521457bc3baafe979d984806c3bc286ac12dd3c8215fe4972304baa942b1a97.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/tables/c521457bc3baafe979d984806c3bc286ac12dd3c8215fe4972304baa942b1a97.jpg)

![d024be2c3a3751d64c2a241c0c93e2ef9bfd8ab31ffe39822eb01c0b3f14bae6.jpg](../icml_results/2846_DenseMLLM_%20Standard%20Multimodal%20LLMs%20for%20Dense%20Prediction/tables/d024be2c3a3751d64c2a241c0c93e2ef9bfd8ab31ffe39822eb01c0b3f14bae6.jpg)

<a id="paper-2810"></a>

## An Asymmetric Latent Factorization-of-Tensors Model for Relation Analysis


### Images

![876830e97f90d888981f671f28b4d025bea78f4fd85f4bdf5fadcffab6e09fbb.jpg](../icml_results/2847_An%20Asymmetric%20Latent%20Factorization-of-Tensors%20Model%20for%20Relation%20Analysis/images/876830e97f90d888981f671f28b4d025bea78f4fd85f4bdf5fadcffab6e09fbb.jpg)

![8ec126f8504b4b4b1a852e8facea429b6684bb26bb38aad9e367de53b54d346c.jpg](../icml_results/2847_An%20Asymmetric%20Latent%20Factorization-of-Tensors%20Model%20for%20Relation%20Analysis/images/8ec126f8504b4b4b1a852e8facea429b6684bb26bb38aad9e367de53b54d346c.jpg)

![9af64bfb0c43ab12c17d4d56dc1e68ecbd3f891ca4ac9082c4da4adf85c11607.jpg](../icml_results/2847_An%20Asymmetric%20Latent%20Factorization-of-Tensors%20Model%20for%20Relation%20Analysis/images/9af64bfb0c43ab12c17d4d56dc1e68ecbd3f891ca4ac9082c4da4adf85c11607.jpg)

![c0594a5d5f05dc7b6795bcc1fe1a5e51420e3d82f6cdede9f7a6393cca81b12a.jpg](../icml_results/2847_An%20Asymmetric%20Latent%20Factorization-of-Tensors%20Model%20for%20Relation%20Analysis/images/c0594a5d5f05dc7b6795bcc1fe1a5e51420e3d82f6cdede9f7a6393cca81b12a.jpg)

![cde375a4b677e6c71dc78a0256a5deda2081811a61eaa48d36257ec9b837278d.jpg](../icml_results/2847_An%20Asymmetric%20Latent%20Factorization-of-Tensors%20Model%20for%20Relation%20Analysis/images/cde375a4b677e6c71dc78a0256a5deda2081811a61eaa48d36257ec9b837278d.jpg)

![d5e450c21db577b49bcba1c1aa26ce8cee0061266f1922fbc12660a809cb3cc6.jpg](../icml_results/2847_An%20Asymmetric%20Latent%20Factorization-of-Tensors%20Model%20for%20Relation%20Analysis/images/d5e450c21db577b49bcba1c1aa26ce8cee0061266f1922fbc12660a809cb3cc6.jpg)

### Tables

![02e21a162ee73afe158e072a380f75a1b768857b112016edf01beb63362175eb.jpg](../icml_results/2847_An%20Asymmetric%20Latent%20Factorization-of-Tensors%20Model%20for%20Relation%20Analysis/tables/02e21a162ee73afe158e072a380f75a1b768857b112016edf01beb63362175eb.jpg)

![0db51b7645e645eaa0aa5d51610bde1289900ae93b881ff795f835cd796bacaf.jpg](../icml_results/2847_An%20Asymmetric%20Latent%20Factorization-of-Tensors%20Model%20for%20Relation%20Analysis/tables/0db51b7645e645eaa0aa5d51610bde1289900ae93b881ff795f835cd796bacaf.jpg)

![3a484d7ba667e5eb69dc31b9d232fc17c6b52aa549243076b2beab143aecc5d5.jpg](../icml_results/2847_An%20Asymmetric%20Latent%20Factorization-of-Tensors%20Model%20for%20Relation%20Analysis/tables/3a484d7ba667e5eb69dc31b9d232fc17c6b52aa549243076b2beab143aecc5d5.jpg)

![4d764eee78bb9941f6e5753f5081c413339b6e2a873a4c26d54e7bf8344561cb.jpg](../icml_results/2847_An%20Asymmetric%20Latent%20Factorization-of-Tensors%20Model%20for%20Relation%20Analysis/tables/4d764eee78bb9941f6e5753f5081c413339b6e2a873a4c26d54e7bf8344561cb.jpg)

![5adb8273c58678c92f3f7eb5180e6dc1f2f58c5f323f8e4cf0d53e89a7c94dfc.jpg](../icml_results/2847_An%20Asymmetric%20Latent%20Factorization-of-Tensors%20Model%20for%20Relation%20Analysis/tables/5adb8273c58678c92f3f7eb5180e6dc1f2f58c5f323f8e4cf0d53e89a7c94dfc.jpg)

![5e890655c52d8cd8fbb67deb5c9985097d3ba014be93150296385b73d684162c.jpg](../icml_results/2847_An%20Asymmetric%20Latent%20Factorization-of-Tensors%20Model%20for%20Relation%20Analysis/tables/5e890655c52d8cd8fbb67deb5c9985097d3ba014be93150296385b73d684162c.jpg)

![cfbce9ecd2bea8dbcc60202477a916fe91afe8488378802841346887b6e74717.jpg](../icml_results/2847_An%20Asymmetric%20Latent%20Factorization-of-Tensors%20Model%20for%20Relation%20Analysis/tables/cfbce9ecd2bea8dbcc60202477a916fe91afe8488378802841346887b6e74717.jpg)

![d3f08be9fe7f212a7b5f6ca7d5713a5aa12d313a24dd85977464241f9e52c6c5.jpg](../icml_results/2847_An%20Asymmetric%20Latent%20Factorization-of-Tensors%20Model%20for%20Relation%20Analysis/tables/d3f08be9fe7f212a7b5f6ca7d5713a5aa12d313a24dd85977464241f9e52c6c5.jpg)

![ed2d6e391e11de34f6ec6d73cc0fa3167371683b5ac83b2ddcfce3f6e25f030e.jpg](../icml_results/2847_An%20Asymmetric%20Latent%20Factorization-of-Tensors%20Model%20for%20Relation%20Analysis/tables/ed2d6e391e11de34f6ec6d73cc0fa3167371683b5ac83b2ddcfce3f6e25f030e.jpg)

<a id="paper-2811"></a>

## NeuroMamba: A Universal Spatiotemporal Module for Robust Perception in Degraded Sensory Streams


### Images

![0253a3ad58b2dde7fc836a9cfdf0bc4b13a29275a9dcecb4f1cb5021629d2248.jpg](../icml_results/2848_NeuroMamba_%20A%20Universal%20Spatiotemporal%20Module%20for%20Robust%20Perception%20in%20Degraded%20Sensory%20Streams/images/0253a3ad58b2dde7fc836a9cfdf0bc4b13a29275a9dcecb4f1cb5021629d2248.jpg)

![35c89ac4916fd9c9955c22486235704ec866b68d29a4f8719e199d9543f5a646.jpg](../icml_results/2848_NeuroMamba_%20A%20Universal%20Spatiotemporal%20Module%20for%20Robust%20Perception%20in%20Degraded%20Sensory%20Streams/images/35c89ac4916fd9c9955c22486235704ec866b68d29a4f8719e199d9543f5a646.jpg)

![d4a37155d68a0b9e8ab9d48577fb09f00805eb6ad3186d68bdf1103a5cdebe64.jpg](../icml_results/2848_NeuroMamba_%20A%20Universal%20Spatiotemporal%20Module%20for%20Robust%20Perception%20in%20Degraded%20Sensory%20Streams/images/d4a37155d68a0b9e8ab9d48577fb09f00805eb6ad3186d68bdf1103a5cdebe64.jpg)

### Tables

![09e7e79ffddb55e2d33ff0e56edfd6718e07c90ecc63d85b5646ad17684e0b12.jpg](../icml_results/2848_NeuroMamba_%20A%20Universal%20Spatiotemporal%20Module%20for%20Robust%20Perception%20in%20Degraded%20Sensory%20Streams/tables/09e7e79ffddb55e2d33ff0e56edfd6718e07c90ecc63d85b5646ad17684e0b12.jpg)

![242577e35fbc850de4630f644f16a3f2949e390f44e09acd5b9bcda742bca6ed.jpg](../icml_results/2848_NeuroMamba_%20A%20Universal%20Spatiotemporal%20Module%20for%20Robust%20Perception%20in%20Degraded%20Sensory%20Streams/tables/242577e35fbc850de4630f644f16a3f2949e390f44e09acd5b9bcda742bca6ed.jpg)

![3d25899d9d265c13ab180eb08e7c0b0ff8939223a5a3a1ddc69ddf3da648d130.jpg](../icml_results/2848_NeuroMamba_%20A%20Universal%20Spatiotemporal%20Module%20for%20Robust%20Perception%20in%20Degraded%20Sensory%20Streams/tables/3d25899d9d265c13ab180eb08e7c0b0ff8939223a5a3a1ddc69ddf3da648d130.jpg)

![427c4bec115264672fad48e37c394466e009eadb677c1c0ac64ae3f545627268.jpg](../icml_results/2848_NeuroMamba_%20A%20Universal%20Spatiotemporal%20Module%20for%20Robust%20Perception%20in%20Degraded%20Sensory%20Streams/tables/427c4bec115264672fad48e37c394466e009eadb677c1c0ac64ae3f545627268.jpg)

![589502cac2b9de75581a1bed99d2dbd0e2c868175308780ebd89c0b5f418890c.jpg](../icml_results/2848_NeuroMamba_%20A%20Universal%20Spatiotemporal%20Module%20for%20Robust%20Perception%20in%20Degraded%20Sensory%20Streams/tables/589502cac2b9de75581a1bed99d2dbd0e2c868175308780ebd89c0b5f418890c.jpg)

![b957bc13fcec69da181ea287c58674321c52ec9751978f4ecb0a2218735445eb.jpg](../icml_results/2848_NeuroMamba_%20A%20Universal%20Spatiotemporal%20Module%20for%20Robust%20Perception%20in%20Degraded%20Sensory%20Streams/tables/b957bc13fcec69da181ea287c58674321c52ec9751978f4ecb0a2218735445eb.jpg)

![bc18a0b4362c97422dc4a1e448c6f2dfe5202e860f3d72ae6345828915c294cd.jpg](../icml_results/2848_NeuroMamba_%20A%20Universal%20Spatiotemporal%20Module%20for%20Robust%20Perception%20in%20Degraded%20Sensory%20Streams/tables/bc18a0b4362c97422dc4a1e448c6f2dfe5202e860f3d72ae6345828915c294cd.jpg)

![c97dc333e2cecbb548afbcff083d69d7567a39994cddfa22f78e8149e430558e.jpg](../icml_results/2848_NeuroMamba_%20A%20Universal%20Spatiotemporal%20Module%20for%20Robust%20Perception%20in%20Degraded%20Sensory%20Streams/tables/c97dc333e2cecbb548afbcff083d69d7567a39994cddfa22f78e8149e430558e.jpg)

![d1d11dd0e016fca9bbffe55762df8930ddefb905df0c5671b612a42b29271be3.jpg](../icml_results/2848_NeuroMamba_%20A%20Universal%20Spatiotemporal%20Module%20for%20Robust%20Perception%20in%20Degraded%20Sensory%20Streams/tables/d1d11dd0e016fca9bbffe55762df8930ddefb905df0c5671b612a42b29271be3.jpg)

<a id="paper-2812"></a>

## SemRep : Generative Code Representation Learning with Code Transformations


### Images

![39424e24c715c65893c1a840bb40f1c185903d05629634ed0fc8a72249465d9b.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/images/39424e24c715c65893c1a840bb40f1c185903d05629634ed0fc8a72249465d9b.jpg)

![3e9a59bb56bcb9499b9a1a78bd334a5b05ee3fad5484eaac7dfd5799e3ece638.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/images/3e9a59bb56bcb9499b9a1a78bd334a5b05ee3fad5484eaac7dfd5799e3ece638.jpg)

![4c77b89ecd38a4dfefdd9d081f15eeab4db40978a7c546aac3f1fc2224fd5385.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/images/4c77b89ecd38a4dfefdd9d081f15eeab4db40978a7c546aac3f1fc2224fd5385.jpg)

![54b008e484e2be00ea050f8a8e6fbd2d1c096eaaea9c7739b6ddfc46f05ade62.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/images/54b008e484e2be00ea050f8a8e6fbd2d1c096eaaea9c7739b6ddfc46f05ade62.jpg)

![5578769cf0e838de39ab5d1425868ab8071d094119a4a0d6522556cd23839b10.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/images/5578769cf0e838de39ab5d1425868ab8071d094119a4a0d6522556cd23839b10.jpg)

![594e19872062124a3102712a698388b64a7e5f9a4202e90f7b4d02ed9f0c5765.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/images/594e19872062124a3102712a698388b64a7e5f9a4202e90f7b4d02ed9f0c5765.jpg)

![6eb5becea41ab7c0ae97d35ed63287914a5386624ae416cf7b153194092e64b0.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/images/6eb5becea41ab7c0ae97d35ed63287914a5386624ae416cf7b153194092e64b0.jpg)

![84f6c391a35cc488b03457edcf492a72d13089b80f927d052327513d33941844.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/images/84f6c391a35cc488b03457edcf492a72d13089b80f927d052327513d33941844.jpg)

![8e810abfaa6c9af973c2e64239db36098ebfdf3f63001089270b43c9c5fcf245.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/images/8e810abfaa6c9af973c2e64239db36098ebfdf3f63001089270b43c9c5fcf245.jpg)

![ab82206682f17cb71745dbc879b6da066fe79c320c13b4f1a2774ae4d6eaf3db.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/images/ab82206682f17cb71745dbc879b6da066fe79c320c13b4f1a2774ae4d6eaf3db.jpg)

![cc51146f38b2f55ef7aced6c77a32bb44fe01be1804f315443ade9c9c137ac91.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/images/cc51146f38b2f55ef7aced6c77a32bb44fe01be1804f315443ade9c9c137ac91.jpg)

![ceec4f20c2e9a23db6c4a746da370aa3ff702f3c22c5238145ae6053d2635688.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/images/ceec4f20c2e9a23db6c4a746da370aa3ff702f3c22c5238145ae6053d2635688.jpg)

![f4b5845ce18011c472c3d0a8ef9829544b4f3d44bf629ee93735d0f28c329fc5.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/images/f4b5845ce18011c472c3d0a8ef9829544b4f3d44bf629ee93735d0f28c329fc5.jpg)

### Tables

![083ce2b43910d2b58beaf730be12fd1f179496bb2f92917038f388151d3130ab.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/tables/083ce2b43910d2b58beaf730be12fd1f179496bb2f92917038f388151d3130ab.jpg)

![08f6d2a6673f659a93b1e159e8ba116a3c88a440f37e9930b15a149f84398869.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/tables/08f6d2a6673f659a93b1e159e8ba116a3c88a440f37e9930b15a149f84398869.jpg)

![104f6ba4f022568982c0f53746c363a555081ac9657685806b1c0e28bb121770.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/tables/104f6ba4f022568982c0f53746c363a555081ac9657685806b1c0e28bb121770.jpg)

![11becca92a16cd843408747102a24b59a1c5715cc686b30060d6a910a1aba22d.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/tables/11becca92a16cd843408747102a24b59a1c5715cc686b30060d6a910a1aba22d.jpg)

![4d75e13d7531c3c05c620eb6416c0381382e8ac4e551dbfd3fa1930f2bcf1943.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/tables/4d75e13d7531c3c05c620eb6416c0381382e8ac4e551dbfd3fa1930f2bcf1943.jpg)

![6e62a9c21d7a94ebd7617d26509f7a2c8112f5c0f9902e651918363dc358281f.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/tables/6e62a9c21d7a94ebd7617d26509f7a2c8112f5c0f9902e651918363dc358281f.jpg)

![7a4310fc8d296702589f773e6cdccb8d354800a7c1ae0de0632cb5389cbacffb.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/tables/7a4310fc8d296702589f773e6cdccb8d354800a7c1ae0de0632cb5389cbacffb.jpg)

![8a95ea3ed77c303cfe887743fef5149c409c6d8d3dcbbe8fd19116633d72ca8f.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/tables/8a95ea3ed77c303cfe887743fef5149c409c6d8d3dcbbe8fd19116633d72ca8f.jpg)

![a76e08406cd0e459f583741433f00dc8e5c009ee115743951fbed6acf5025062.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/tables/a76e08406cd0e459f583741433f00dc8e5c009ee115743951fbed6acf5025062.jpg)

![df1d8e023fd1a9e7afb8d47eb942ccc2be42b5711c9bf1d6c5cd747fcec9a5fc.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/tables/df1d8e023fd1a9e7afb8d47eb942ccc2be42b5711c9bf1d6c5cd747fcec9a5fc.jpg)

![df8f76891f6a391766dc1d7a48015c7e161efdeeea1052978f6dfa5d7554c9cc.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/tables/df8f76891f6a391766dc1d7a48015c7e161efdeeea1052978f6dfa5d7554c9cc.jpg)

![f5f1b8ca1240f3547e59ac9bfedc4c251a743eca0419001129ee88e0ba07d5a1.jpg](../icml_results/2849_SemRep%20_%20Generative%20Code%20Representation%20Learning%20with%20Code%20Transformations/tables/f5f1b8ca1240f3547e59ac9bfedc4c251a743eca0419001129ee88e0ba07d5a1.jpg)

<a id="paper-2813"></a>

## A Direct Approach for Handling Contextual Bandits with Latent State Dynamics


### Images

![472d63d75dc928ccac4754c1848f048ef3dca1b7515923c2ace4d42fa601588e.jpg](../icml_results/2850_A%20Direct%20Approach%20for%20Handling%20Contextual%20Bandits%20with%20Latent%20State%20Dynamics/images/472d63d75dc928ccac4754c1848f048ef3dca1b7515923c2ace4d42fa601588e.jpg)

![62f3335a51d90b1fb742cff73a327c0cbc077a9e3e7db85cd8b87249c9b32266.jpg](../icml_results/2850_A%20Direct%20Approach%20for%20Handling%20Contextual%20Bandits%20with%20Latent%20State%20Dynamics/images/62f3335a51d90b1fb742cff73a327c0cbc077a9e3e7db85cd8b87249c9b32266.jpg)

![68a380622f8d0d9b87c2b8c29c711a1909f9b2e77430d6a361336e8f747e11d4.jpg](../icml_results/2850_A%20Direct%20Approach%20for%20Handling%20Contextual%20Bandits%20with%20Latent%20State%20Dynamics/images/68a380622f8d0d9b87c2b8c29c711a1909f9b2e77430d6a361336e8f747e11d4.jpg)

![f0a34d2faa7ad66b2c8e8eb7930e37b5f4dc9802aef68245004942431077644d.jpg](../icml_results/2850_A%20Direct%20Approach%20for%20Handling%20Contextual%20Bandits%20with%20Latent%20State%20Dynamics/images/f0a34d2faa7ad66b2c8e8eb7930e37b5f4dc9802aef68245004942431077644d.jpg)

### Tables

![23d83ef54fb6c1cb424bc2f94b853daa5a7ac5495552a6e9b0acee86d1b7d431.jpg](../icml_results/2850_A%20Direct%20Approach%20for%20Handling%20Contextual%20Bandits%20with%20Latent%20State%20Dynamics/tables/23d83ef54fb6c1cb424bc2f94b853daa5a7ac5495552a6e9b0acee86d1b7d431.jpg)

![5e1f6cc409ba9f089bb0b4d9bd7a20958355bf263335842f414d5d44bb6beaf3.jpg](../icml_results/2850_A%20Direct%20Approach%20for%20Handling%20Contextual%20Bandits%20with%20Latent%20State%20Dynamics/tables/5e1f6cc409ba9f089bb0b4d9bd7a20958355bf263335842f414d5d44bb6beaf3.jpg)

![74c17f1408b2b448a6d86de3b31c543d5129854dabf45aad447513c8049d58ec.jpg](../icml_results/2850_A%20Direct%20Approach%20for%20Handling%20Contextual%20Bandits%20with%20Latent%20State%20Dynamics/tables/74c17f1408b2b448a6d86de3b31c543d5129854dabf45aad447513c8049d58ec.jpg)

![aec74ebf3cb283b4e23ace8a74377273ebacbb503b107001b5ad3d1410a99281.jpg](../icml_results/2850_A%20Direct%20Approach%20for%20Handling%20Contextual%20Bandits%20with%20Latent%20State%20Dynamics/tables/aec74ebf3cb283b4e23ace8a74377273ebacbb503b107001b5ad3d1410a99281.jpg)

![b1d31c7c9e4db066439338d722429fb363721006b5bb7e2f12573afdc229a2bd.jpg](../icml_results/2850_A%20Direct%20Approach%20for%20Handling%20Contextual%20Bandits%20with%20Latent%20State%20Dynamics/tables/b1d31c7c9e4db066439338d722429fb363721006b5bb7e2f12573afdc229a2bd.jpg)

![e1f9acad71d87dd05a5b9117eb9aa7e0b271d98dc28889a49ecc804a832c670f.jpg](../icml_results/2850_A%20Direct%20Approach%20for%20Handling%20Contextual%20Bandits%20with%20Latent%20State%20Dynamics/tables/e1f9acad71d87dd05a5b9117eb9aa7e0b271d98dc28889a49ecc804a832c670f.jpg)

![fc32667f71e5cb2388e49e3bb53567f0bec0d0c3af2dee239d2f9b841d7473c3.jpg](../icml_results/2850_A%20Direct%20Approach%20for%20Handling%20Contextual%20Bandits%20with%20Latent%20State%20Dynamics/tables/fc32667f71e5cb2388e49e3bb53567f0bec0d0c3af2dee239d2f9b841d7473c3.jpg)

<a id="paper-2814"></a>

## VLA-ATTC: Adaptive Test-Time Compute for VLA Models with Relative Action Critic Model


### Images

![240c9ac18fb2470fe6b9b047c74e6d6cfb3e2f388669f7b15151997830e4fe53.jpg](../icml_results/2851_VLA-ATTC_%20Adaptive%20Test-Time%20Compute%20for%20VLA%20Models%20with%20Relative%20Action%20Critic%20Model/images/240c9ac18fb2470fe6b9b047c74e6d6cfb3e2f388669f7b15151997830e4fe53.jpg)

![7b06bc98b955bddecf334fddfa119ccf89ea1cd63550a266a201c72f7710ecd0.jpg](../icml_results/2851_VLA-ATTC_%20Adaptive%20Test-Time%20Compute%20for%20VLA%20Models%20with%20Relative%20Action%20Critic%20Model/images/7b06bc98b955bddecf334fddfa119ccf89ea1cd63550a266a201c72f7710ecd0.jpg)

![a23bf12a2c466b1177632f452c43bbb991bc9570705db072c856aa3c0443c2ba.jpg](../icml_results/2851_VLA-ATTC_%20Adaptive%20Test-Time%20Compute%20for%20VLA%20Models%20with%20Relative%20Action%20Critic%20Model/images/a23bf12a2c466b1177632f452c43bbb991bc9570705db072c856aa3c0443c2ba.jpg)

![c66355c68095853ce41e737d23b110dac7c66cc271116eca92964b34197fecff.jpg](../icml_results/2851_VLA-ATTC_%20Adaptive%20Test-Time%20Compute%20for%20VLA%20Models%20with%20Relative%20Action%20Critic%20Model/images/c66355c68095853ce41e737d23b110dac7c66cc271116eca92964b34197fecff.jpg)

![d24b67a0196e18a32d7d82f696d66fd1f0232eeca991b418fdc143b9a53fc3f0.jpg](../icml_results/2851_VLA-ATTC_%20Adaptive%20Test-Time%20Compute%20for%20VLA%20Models%20with%20Relative%20Action%20Critic%20Model/images/d24b67a0196e18a32d7d82f696d66fd1f0232eeca991b418fdc143b9a53fc3f0.jpg)

![e385eae99f34b1826ad3ec27a1e36a14718cff1f947635aa02fb9f36aeac13bf.jpg](../icml_results/2851_VLA-ATTC_%20Adaptive%20Test-Time%20Compute%20for%20VLA%20Models%20with%20Relative%20Action%20Critic%20Model/images/e385eae99f34b1826ad3ec27a1e36a14718cff1f947635aa02fb9f36aeac13bf.jpg)

### Tables

![052a554b6d3b0b7c13025d0620ed5d17f1f997cec4850f02768415b699332d00.jpg](../icml_results/2851_VLA-ATTC_%20Adaptive%20Test-Time%20Compute%20for%20VLA%20Models%20with%20Relative%20Action%20Critic%20Model/tables/052a554b6d3b0b7c13025d0620ed5d17f1f997cec4850f02768415b699332d00.jpg)

![0c944ecb3ba29a677607b71433368a20fda413734b1159371f3fc90b81ec5f71.jpg](../icml_results/2851_VLA-ATTC_%20Adaptive%20Test-Time%20Compute%20for%20VLA%20Models%20with%20Relative%20Action%20Critic%20Model/tables/0c944ecb3ba29a677607b71433368a20fda413734b1159371f3fc90b81ec5f71.jpg)

![17441ae90fa6f911570c8f5ecbcbeaba50b8c8c4c157ddfe2311bccbf05b70be.jpg](../icml_results/2851_VLA-ATTC_%20Adaptive%20Test-Time%20Compute%20for%20VLA%20Models%20with%20Relative%20Action%20Critic%20Model/tables/17441ae90fa6f911570c8f5ecbcbeaba50b8c8c4c157ddfe2311bccbf05b70be.jpg)

![77c63573efd507250bb03814a5247b37499e6792182a70454e6bf45aaa80235b.jpg](../icml_results/2851_VLA-ATTC_%20Adaptive%20Test-Time%20Compute%20for%20VLA%20Models%20with%20Relative%20Action%20Critic%20Model/tables/77c63573efd507250bb03814a5247b37499e6792182a70454e6bf45aaa80235b.jpg)

![aaf2d07883c6b2a6337e038f54f77aecc3b8e3cd3e482f9736adce23c2e98e61.jpg](../icml_results/2851_VLA-ATTC_%20Adaptive%20Test-Time%20Compute%20for%20VLA%20Models%20with%20Relative%20Action%20Critic%20Model/tables/aaf2d07883c6b2a6337e038f54f77aecc3b8e3cd3e482f9736adce23c2e98e61.jpg)

![aaf53bfd0f2a0a4053bf25e0c723e9ebd6063fd91f88d973c1eb7efb7ab1867d.jpg](../icml_results/2851_VLA-ATTC_%20Adaptive%20Test-Time%20Compute%20for%20VLA%20Models%20with%20Relative%20Action%20Critic%20Model/tables/aaf53bfd0f2a0a4053bf25e0c723e9ebd6063fd91f88d973c1eb7efb7ab1867d.jpg)

![bd16f21015f23fbe908eee532d332d9333bea11693749ac5705ff4ebd9980d12.jpg](../icml_results/2851_VLA-ATTC_%20Adaptive%20Test-Time%20Compute%20for%20VLA%20Models%20with%20Relative%20Action%20Critic%20Model/tables/bd16f21015f23fbe908eee532d332d9333bea11693749ac5705ff4ebd9980d12.jpg)

![fc71429aef5d80d8abd7738cfacce216c0976f8f7ac7fc55bf33f8ebfdf950df.jpg](../icml_results/2851_VLA-ATTC_%20Adaptive%20Test-Time%20Compute%20for%20VLA%20Models%20with%20Relative%20Action%20Critic%20Model/tables/fc71429aef5d80d8abd7738cfacce216c0976f8f7ac7fc55bf33f8ebfdf950df.jpg)

<a id="paper-2815"></a>

## Sentinel-VLA: A Metacognitive VLA Model with Active Status Monitoring for Dynamic Reasoning and Error Recovery


### Images

![327ac282de2d82d640005cdaf04c02921fa73af6b63de9755c39f0b28185a4ce.jpg](../icml_results/2852_Sentinel-VLA_%20A%20Metacognitive%20VLA%20Model%20with%20Active%20Status%20Monitoring%20for%20Dynamic%20Reasoning%20and%20Erro/images/327ac282de2d82d640005cdaf04c02921fa73af6b63de9755c39f0b28185a4ce.jpg)

![3da15cfe70c40b1856927e4bdefeb3e9fd4f2e459cafada6ed063f2f5b057297.jpg](../icml_results/2852_Sentinel-VLA_%20A%20Metacognitive%20VLA%20Model%20with%20Active%20Status%20Monitoring%20for%20Dynamic%20Reasoning%20and%20Erro/images/3da15cfe70c40b1856927e4bdefeb3e9fd4f2e459cafada6ed063f2f5b057297.jpg)

![62e84f7d7fa098079c82a9150620db8eebc5d9fa479dc3a08fe14ffe93293b77.jpg](../icml_results/2852_Sentinel-VLA_%20A%20Metacognitive%20VLA%20Model%20with%20Active%20Status%20Monitoring%20for%20Dynamic%20Reasoning%20and%20Erro/images/62e84f7d7fa098079c82a9150620db8eebc5d9fa479dc3a08fe14ffe93293b77.jpg)

![6aab6923808885eacb652b2887fc9590d59f3a86a57624350c6a08665d0f5ea0.jpg](../icml_results/2852_Sentinel-VLA_%20A%20Metacognitive%20VLA%20Model%20with%20Active%20Status%20Monitoring%20for%20Dynamic%20Reasoning%20and%20Erro/images/6aab6923808885eacb652b2887fc9590d59f3a86a57624350c6a08665d0f5ea0.jpg)

![763a99276144b88d08477734e6c8489229f5cfb7e618f502121f2ab033eefde3.jpg](../icml_results/2852_Sentinel-VLA_%20A%20Metacognitive%20VLA%20Model%20with%20Active%20Status%20Monitoring%20for%20Dynamic%20Reasoning%20and%20Erro/images/763a99276144b88d08477734e6c8489229f5cfb7e618f502121f2ab033eefde3.jpg)

![dc0d152670037c8b1324ae3470359915ef0e4abc9cd1b622ad259d8a9bae96d6.jpg](../icml_results/2852_Sentinel-VLA_%20A%20Metacognitive%20VLA%20Model%20with%20Active%20Status%20Monitoring%20for%20Dynamic%20Reasoning%20and%20Erro/images/dc0d152670037c8b1324ae3470359915ef0e4abc9cd1b622ad259d8a9bae96d6.jpg)

### Tables

![6d3560e555b86e4f3a08336ea05860f873e3cc319b6e62bcbda77a824f5e4d3a.jpg](../icml_results/2852_Sentinel-VLA_%20A%20Metacognitive%20VLA%20Model%20with%20Active%20Status%20Monitoring%20for%20Dynamic%20Reasoning%20and%20Erro/tables/6d3560e555b86e4f3a08336ea05860f873e3cc319b6e62bcbda77a824f5e4d3a.jpg)

![801e61e99f393918c14e55a353885b32d036c06d5c72d2e338f969a842109d86.jpg](../icml_results/2852_Sentinel-VLA_%20A%20Metacognitive%20VLA%20Model%20with%20Active%20Status%20Monitoring%20for%20Dynamic%20Reasoning%20and%20Erro/tables/801e61e99f393918c14e55a353885b32d036c06d5c72d2e338f969a842109d86.jpg)

![8b08bf45c78a638d9670ec6f1d7c2d089f192f6cf9cae2da11c58d1cb9516f3e.jpg](../icml_results/2852_Sentinel-VLA_%20A%20Metacognitive%20VLA%20Model%20with%20Active%20Status%20Monitoring%20for%20Dynamic%20Reasoning%20and%20Erro/tables/8b08bf45c78a638d9670ec6f1d7c2d089f192f6cf9cae2da11c58d1cb9516f3e.jpg)

![8e1f0248c540a6db36718bfa86efcccafe07cd7bd6a9659737f23a5141ef7152.jpg](../icml_results/2852_Sentinel-VLA_%20A%20Metacognitive%20VLA%20Model%20with%20Active%20Status%20Monitoring%20for%20Dynamic%20Reasoning%20and%20Erro/tables/8e1f0248c540a6db36718bfa86efcccafe07cd7bd6a9659737f23a5141ef7152.jpg)

![af5e5679aecb6cb9ea94d13f2f34f51acff9513ce450a233eb724d807a5fc3ed.jpg](../icml_results/2852_Sentinel-VLA_%20A%20Metacognitive%20VLA%20Model%20with%20Active%20Status%20Monitoring%20for%20Dynamic%20Reasoning%20and%20Erro/tables/af5e5679aecb6cb9ea94d13f2f34f51acff9513ce450a233eb724d807a5fc3ed.jpg)

![ce567ee184e561cdaa2dfd0dbaa83900cf4c32326546195608cb235c3e551869.jpg](../icml_results/2852_Sentinel-VLA_%20A%20Metacognitive%20VLA%20Model%20with%20Active%20Status%20Monitoring%20for%20Dynamic%20Reasoning%20and%20Erro/tables/ce567ee184e561cdaa2dfd0dbaa83900cf4c32326546195608cb235c3e551869.jpg)

<a id="paper-2816"></a>

## Trajectory Consistency for One-Step Generation on Euler Mean Flows


### Images

![0469ea0618109e26f61cd4a4ca4afa73653c2ca52f3ab8e797717e659905e693.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/0469ea0618109e26f61cd4a4ca4afa73653c2ca52f3ab8e797717e659905e693.jpg)

![0ebcb8dab61a4db38dd74d25027fc5453a46e32ec8492df6c9469f13dfb30556.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/0ebcb8dab61a4db38dd74d25027fc5453a46e32ec8492df6c9469f13dfb30556.jpg)

![2b46bc57daf721043347b7566b0f1512008f98cd8f080fe1f2dfe30ba4a2ef5c.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/2b46bc57daf721043347b7566b0f1512008f98cd8f080fe1f2dfe30ba4a2ef5c.jpg)

![314634127ced26448d7382a7aa16bf4c6f4b64ba0d080ddee153e3c3faca379d.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/314634127ced26448d7382a7aa16bf4c6f4b64ba0d080ddee153e3c3faca379d.jpg)

![320839461a37aa7dbb0842a6f403386ee81ff362a8dd7668a262f87fdc76a118.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/320839461a37aa7dbb0842a6f403386ee81ff362a8dd7668a262f87fdc76a118.jpg)

![36cf1fe334f09956bafaf29de24175f83018f54fe0f98c084f2bfd1320455f40.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/36cf1fe334f09956bafaf29de24175f83018f54fe0f98c084f2bfd1320455f40.jpg)

![4acb5dab0b00c26a47c742eacdd34cb9adce30f81fe1dbeff515ed14c449b16b.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/4acb5dab0b00c26a47c742eacdd34cb9adce30f81fe1dbeff515ed14c449b16b.jpg)

![4f80f76adc5d2237423d9435f8e3d3cdc6e9a17236341c94d44f2026d5d37827.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/4f80f76adc5d2237423d9435f8e3d3cdc6e9a17236341c94d44f2026d5d37827.jpg)

![533ac4bd69aec7288608238296187765c8128f916d982b196eab115778b0f057.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/533ac4bd69aec7288608238296187765c8128f916d982b196eab115778b0f057.jpg)

![6d188defa4657dec8650fe32a62af95cacd570d5d4afc5225aa146b3f5c0c4d7.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/6d188defa4657dec8650fe32a62af95cacd570d5d4afc5225aa146b3f5c0c4d7.jpg)

![78da20d13ffffa5dcabf51e0d3928b2fa0da8bfbc194c05ea278aa1e06e6307b.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/78da20d13ffffa5dcabf51e0d3928b2fa0da8bfbc194c05ea278aa1e06e6307b.jpg)

![7b8efdce3dbe75ddfe1fc412621dbf78c7b39dc513c7f1bafd70b4db6503a55b.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/7b8efdce3dbe75ddfe1fc412621dbf78c7b39dc513c7f1bafd70b4db6503a55b.jpg)

![83b4f8da7086b23fdd2b7d675a3488c7c331f54ff2f868e3126410485b049894.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/83b4f8da7086b23fdd2b7d675a3488c7c331f54ff2f868e3126410485b049894.jpg)

![86c212f5baf3115897c08e690f85235296ec84c00ae97438d2de26567454b074.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/86c212f5baf3115897c08e690f85235296ec84c00ae97438d2de26567454b074.jpg)

![99a21a052b3b758affdd838fdfb1d8e18ed4d57f31990a1dee0f635da2e0c8e8.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/99a21a052b3b758affdd838fdfb1d8e18ed4d57f31990a1dee0f635da2e0c8e8.jpg)

![acb017a615778756a8507b0c0ed2979edc12f11b1563892f88410925956c0f7c.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/acb017a615778756a8507b0c0ed2979edc12f11b1563892f88410925956c0f7c.jpg)

![b2e4a069ea6ed6e0929b8268ac31439db6864150b7f14b559e3710dd7470b0dc.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/b2e4a069ea6ed6e0929b8268ac31439db6864150b7f14b559e3710dd7470b0dc.jpg)

![c16b2d7de6621089eb1ded01ae5e378302b379aad31559a73aacf11a0c7fa7b8.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/c16b2d7de6621089eb1ded01ae5e378302b379aad31559a73aacf11a0c7fa7b8.jpg)

![ca1c1e3bf572e0e8e4cc6d14ef25f22ffc6d468375efdea3eb1b330e15aa4cb4.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/ca1c1e3bf572e0e8e4cc6d14ef25f22ffc6d468375efdea3eb1b330e15aa4cb4.jpg)

![d69f0e627a8c768f6cffb1257e8af8e349ed9767bccde717098f95ee5b2aa2f6.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/d69f0e627a8c768f6cffb1257e8af8e349ed9767bccde717098f95ee5b2aa2f6.jpg)

![d84f106f7fa4bd4647b8bd9eec2d0400c499d2f96817674453d4f89a3147147d.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/d84f106f7fa4bd4647b8bd9eec2d0400c499d2f96817674453d4f89a3147147d.jpg)

![d85f298fd1cd0fb5b3d7de82b171380103056419d061b9111d123beaf3f4110e.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/d85f298fd1cd0fb5b3d7de82b171380103056419d061b9111d123beaf3f4110e.jpg)

![e1b121213720040ddab875bb544204fe5ab2519c63478e25b06733d04060f7b9.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/e1b121213720040ddab875bb544204fe5ab2519c63478e25b06733d04060f7b9.jpg)

![f24bda38061c37d7a5f850f2898a189f7ddb3cc486f5964d636ec487fc4cde0d.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/f24bda38061c37d7a5f850f2898a189f7ddb3cc486f5964d636ec487fc4cde0d.jpg)

![fe92832dccd027aa39789b12bf405f64197ad3f47e4018c791f1a50c85cbb477.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/images/fe92832dccd027aa39789b12bf405f64197ad3f47e4018c791f1a50c85cbb477.jpg)

### Tables

![263dc6e9900c029ad08a8429a5d163c165f227098159fbf474b354998cce7573.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/tables/263dc6e9900c029ad08a8429a5d163c165f227098159fbf474b354998cce7573.jpg)

![2ebe3a81d051a1d1e3e66729035f6386325f315afb85de98a3a8a4e5fabad9d7.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/tables/2ebe3a81d051a1d1e3e66729035f6386325f315afb85de98a3a8a4e5fabad9d7.jpg)

![3d28ba01382cb4fe44a8d23eac7a854d4fa2215cce2f426eefa0b74efc53dcda.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/tables/3d28ba01382cb4fe44a8d23eac7a854d4fa2215cce2f426eefa0b74efc53dcda.jpg)

![4664afef35f1d45beb6783f68b7de681b51a386611d96d4cf5e9076a6e0f5dd5.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/tables/4664afef35f1d45beb6783f68b7de681b51a386611d96d4cf5e9076a6e0f5dd5.jpg)

![64544fdd5f314a5a9bbd9bca90e574be0e4080bc1801624ed2ad5cabcaf51682.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/tables/64544fdd5f314a5a9bbd9bca90e574be0e4080bc1801624ed2ad5cabcaf51682.jpg)

![893303b5ce6b4576405e4be078645366e74993b336e3490de0cf7f963de065bd.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/tables/893303b5ce6b4576405e4be078645366e74993b336e3490de0cf7f963de065bd.jpg)

![abefe757d4f8951fe84e2c4a5988df37b098c66337f512e216e73633d26cfdbd.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/tables/abefe757d4f8951fe84e2c4a5988df37b098c66337f512e216e73633d26cfdbd.jpg)

![b116999ab12e194d41243cf6835db1cad494179ecbccae49d83de080eb97ca6a.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/tables/b116999ab12e194d41243cf6835db1cad494179ecbccae49d83de080eb97ca6a.jpg)

![b673c55381e848c704879f3534ab6032a271ddc704317853866ebe2c69eafdeb.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/tables/b673c55381e848c704879f3534ab6032a271ddc704317853866ebe2c69eafdeb.jpg)

![c52ac0a35945c438510052e6299e5d722c4b2e1f432ebc6f05bb692e4502bb20.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/tables/c52ac0a35945c438510052e6299e5d722c4b2e1f432ebc6f05bb692e4502bb20.jpg)

![f98b8708d00540bc623513c7ab38abaaeceffb6be0d801c74325bce0e1b488ba.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/tables/f98b8708d00540bc623513c7ab38abaaeceffb6be0d801c74325bce0e1b488ba.jpg)

![fb21bdfd74ac46c37de260ce741f4d18719e8ae91115e6d7dce4f858a002a785.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/tables/fb21bdfd74ac46c37de260ce741f4d18719e8ae91115e6d7dce4f858a002a785.jpg)

![fe163a7c5158eca2e9f7ea3718a60953e02142e46967caa032b4161507d75668.jpg](../icml_results/2853_Trajectory%20Consistency%20for%20One-Step%20Generation%20on%20Euler%20Mean%20Flows/tables/fe163a7c5158eca2e9f7ea3718a60953e02142e46967caa032b4161507d75668.jpg)

<a id="paper-2817"></a>

## Controllable Molecule Generation via Sparse Representation Editing: An Interpretability-Driven Perspective


### Images

![2089abfd3a86ad76648fee41f5f8ac2c9b4eb6ff659cd26f1dd246954f554aaf.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/2089abfd3a86ad76648fee41f5f8ac2c9b4eb6ff659cd26f1dd246954f554aaf.jpg)

![24ebd4b1e562fa19d6ab4ee8535d4a37c17e01106cce44c9e443d4ac46e6cc10.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/24ebd4b1e562fa19d6ab4ee8535d4a37c17e01106cce44c9e443d4ac46e6cc10.jpg)

![2833b0e7eb4ae61525a18e1cf1d195f601f282e112247385e5b317a134865add.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/2833b0e7eb4ae61525a18e1cf1d195f601f282e112247385e5b317a134865add.jpg)

![2d29320d30a32c0674d1ef0c9d1de6be506c1d952c3ddc1b6ac25b212d40d445.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/2d29320d30a32c0674d1ef0c9d1de6be506c1d952c3ddc1b6ac25b212d40d445.jpg)

![2f32403f2b12598a6deec8bad66d40ee2241056d5b221b2987e2b350e2223dda.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/2f32403f2b12598a6deec8bad66d40ee2241056d5b221b2987e2b350e2223dda.jpg)

![3011b39b1e720f12a65c8c79bcf31e68f076035a8e196f3ff4ed789465d142d6.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/3011b39b1e720f12a65c8c79bcf31e68f076035a8e196f3ff4ed789465d142d6.jpg)

![3156e3c3391e55ba95bf1fc6d34cc8440e8e5d822e5549afed9a621656bfa5dd.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/3156e3c3391e55ba95bf1fc6d34cc8440e8e5d822e5549afed9a621656bfa5dd.jpg)

![328d9d89586e4ebd18293843af9ee2c85d4ca3b2a013dcc5ee241a7735c8d6dc.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/328d9d89586e4ebd18293843af9ee2c85d4ca3b2a013dcc5ee241a7735c8d6dc.jpg)

![347e475c8e714e59255d14cc522cb639b2d8c278db1211d7ba75eba5abfd676b.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/347e475c8e714e59255d14cc522cb639b2d8c278db1211d7ba75eba5abfd676b.jpg)

![353abe9f056e93e38d5f859c5837bb32f896aa518820002d574321d4bda55d94.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/353abe9f056e93e38d5f859c5837bb32f896aa518820002d574321d4bda55d94.jpg)

![374cf7765de0a8aa64189aa00c2912eb365b9992147a23b1f55493a0acff0d57.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/374cf7765de0a8aa64189aa00c2912eb365b9992147a23b1f55493a0acff0d57.jpg)

![3de9d0a93fa8049dc6c25403f911820618b348d6ecece1e0261a1bed49b32936.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/3de9d0a93fa8049dc6c25403f911820618b348d6ecece1e0261a1bed49b32936.jpg)

![40f0aa697fa2ff2f35fc794b26fa24095c131744d2fbe300e93b457596c83935.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/40f0aa697fa2ff2f35fc794b26fa24095c131744d2fbe300e93b457596c83935.jpg)

![421bfbe11d42e310855e4ea977671d70da636cba39a1ed333d180247541cf7e5.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/421bfbe11d42e310855e4ea977671d70da636cba39a1ed333d180247541cf7e5.jpg)

![62d572b64be7c8c4db5f443e1156ebf7412471f8e4a41c35ccd944d9bf06584f.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/62d572b64be7c8c4db5f443e1156ebf7412471f8e4a41c35ccd944d9bf06584f.jpg)

![64ba8128524a71ae615392bf1e02e5ab2d6cbb38c03517111d1c60bdb63a9e0e.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/64ba8128524a71ae615392bf1e02e5ab2d6cbb38c03517111d1c60bdb63a9e0e.jpg)

![6f4df815d6a3f7e9512e9625c79bef84caa5ac83ad4261416f36d0c7d7522b05.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/6f4df815d6a3f7e9512e9625c79bef84caa5ac83ad4261416f36d0c7d7522b05.jpg)

![7dfe6af7261ad3ed4e98c7d56f0a3a3b6153c5cc61b64e544aae880ffe882c95.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/7dfe6af7261ad3ed4e98c7d56f0a3a3b6153c5cc61b64e544aae880ffe882c95.jpg)

![89d9808ce9d95d9d6b2fa0dbb1af7f35c063372caa92fdfeebf3b3798a748f7a.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/89d9808ce9d95d9d6b2fa0dbb1af7f35c063372caa92fdfeebf3b3798a748f7a.jpg)

![89db57c551b7f6432d599a4fc4deaacf5e8342e85325d6cb3364ee56588841b5.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/89db57c551b7f6432d599a4fc4deaacf5e8342e85325d6cb3364ee56588841b5.jpg)

![8d975101cdca6063aa4de652c9c6773e99fe1c554b59532bf1224b04dc3cd9cb.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/8d975101cdca6063aa4de652c9c6773e99fe1c554b59532bf1224b04dc3cd9cb.jpg)

![98193fc8142e3ad89c40812b068ae40f48f986993d97b30eaf2164fa1f9bb7e9.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/98193fc8142e3ad89c40812b068ae40f48f986993d97b30eaf2164fa1f9bb7e9.jpg)

![9c6e2ded9ce41323388b39b9b757714004b197c433b202c33ca29a19ec3d4c57.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/9c6e2ded9ce41323388b39b9b757714004b197c433b202c33ca29a19ec3d4c57.jpg)

![9f35d23a1206db969bedc281fd33c5a49ed2e68ef03c0b22853c8bc7c1b9a156.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/9f35d23a1206db969bedc281fd33c5a49ed2e68ef03c0b22853c8bc7c1b9a156.jpg)

![a589e8ab00b4ebf7969bfeec97cd28b037562010dd3eac5347c18ab5a817004a.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/a589e8ab00b4ebf7969bfeec97cd28b037562010dd3eac5347c18ab5a817004a.jpg)

![a60df8aee5d81f8e140301d1d36f9c1afe4240474ec97edde10fce6a48b8deb4.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/a60df8aee5d81f8e140301d1d36f9c1afe4240474ec97edde10fce6a48b8deb4.jpg)

![b27dd77f886a7fce864c40a16af21430ac165bc03d60afa21c61cf144e8c2f59.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/b27dd77f886a7fce864c40a16af21430ac165bc03d60afa21c61cf144e8c2f59.jpg)

![b481dbad0272bc80e8927f66613f5e5a2baafefef390a59cd88fe63cf305dfaa.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/b481dbad0272bc80e8927f66613f5e5a2baafefef390a59cd88fe63cf305dfaa.jpg)

![be1afb908c5542665d3a82f9f85c690b1e455977ccb270f1e8fc97f5b92d1fc5.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/be1afb908c5542665d3a82f9f85c690b1e455977ccb270f1e8fc97f5b92d1fc5.jpg)

![bf92335b5a8775ea984f7e1d31c5b591d68966f2c9923a355129d31067abc701.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/bf92335b5a8775ea984f7e1d31c5b591d68966f2c9923a355129d31067abc701.jpg)

![cc4ed2a875e610ea0cb73e006d2bc81b55c82c95c95398eb22d9d66a32f30129.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/cc4ed2a875e610ea0cb73e006d2bc81b55c82c95c95398eb22d9d66a32f30129.jpg)

![d94fbf4833760439dd5225a14c8c4af4beafdb261ddc8f477041d18d97ebe383.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/d94fbf4833760439dd5225a14c8c4af4beafdb261ddc8f477041d18d97ebe383.jpg)

![df0295caab1ef58759743eb8914f11db372a58bf92d8a3def7a5bcb867ce7465.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/df0295caab1ef58759743eb8914f11db372a58bf92d8a3def7a5bcb867ce7465.jpg)

![e21c102ba570b56a8ee9c17beed167293bebefd4d9d6803d5a6b54cd2bca8365.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/e21c102ba570b56a8ee9c17beed167293bebefd4d9d6803d5a6b54cd2bca8365.jpg)

![e31327b0d34137f1a80107725a69df967999beb217aece2d6048108b637685d0.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/e31327b0d34137f1a80107725a69df967999beb217aece2d6048108b637685d0.jpg)

![e3d8d05d65dc0ccef61a19cb19d1bd720dfb62abc7e9f0356d58e8a89f840dca.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/e3d8d05d65dc0ccef61a19cb19d1bd720dfb62abc7e9f0356d58e8a89f840dca.jpg)

![e6df737c7bd528a7e11e1b3e53f4068cd9b9a6f55f289a08c7001c3a24f63204.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/e6df737c7bd528a7e11e1b3e53f4068cd9b9a6f55f289a08c7001c3a24f63204.jpg)

![e6ec35617ff035d44f6121fe74423568872bac5aa78a06fa74507a22ef4bd6a7.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/e6ec35617ff035d44f6121fe74423568872bac5aa78a06fa74507a22ef4bd6a7.jpg)

![e77d6f6d2cee40f3343b1d8f54e45239dce560bc64c3520bafae9da63b5f9e7d.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/e77d6f6d2cee40f3343b1d8f54e45239dce560bc64c3520bafae9da63b5f9e7d.jpg)

![ecde8717c8ca36498bd2d0383b6d371ff1fb9d03f09eb65fe7941adbb09feae6.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/ecde8717c8ca36498bd2d0383b6d371ff1fb9d03f09eb65fe7941adbb09feae6.jpg)

![f29c04dc24ad2141f858bf0f6a3a5e697e8609a916a33b3ddab050b0cdcbdb43.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/f29c04dc24ad2141f858bf0f6a3a5e697e8609a916a33b3ddab050b0cdcbdb43.jpg)

![fa9a3333a427b8a2a0a9be4753bab94210d7df183ca6b6fbaf6c703ac767b011.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/images/fa9a3333a427b8a2a0a9be4753bab94210d7df183ca6b6fbaf6c703ac767b011.jpg)

### Tables

![07809dddc411ce69ad18338970b80ce099777be4b04da03c91d15d0f5f0656fc.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/07809dddc411ce69ad18338970b80ce099777be4b04da03c91d15d0f5f0656fc.jpg)

![0af8ea67f1484887b34c09953a57e15dbe6fd82e6122c34a644b78f4bf73ea5f.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/0af8ea67f1484887b34c09953a57e15dbe6fd82e6122c34a644b78f4bf73ea5f.jpg)

![1b86c96618db8712dfd52408486bfcd153d0e86eb83d75e4d0303c3a5d175248.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/1b86c96618db8712dfd52408486bfcd153d0e86eb83d75e4d0303c3a5d175248.jpg)

![2eb5ef53e05c74426ac61aac241b2825c196e6f19aff6fcf4bb7ca93003e5447.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/2eb5ef53e05c74426ac61aac241b2825c196e6f19aff6fcf4bb7ca93003e5447.jpg)

![471d9525ad451d4b41b211a55ff7ada0275da756df7db2c129c75276132024f9.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/471d9525ad451d4b41b211a55ff7ada0275da756df7db2c129c75276132024f9.jpg)

![4ee094b303aabc02580871630876cf2945ee30b5dd1817210a2da72e1446fb35.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/4ee094b303aabc02580871630876cf2945ee30b5dd1817210a2da72e1446fb35.jpg)

![5caaa4678c279d8a4808938284f0a013e55a49da4e36c6be671b9e6ce29c63bd.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/5caaa4678c279d8a4808938284f0a013e55a49da4e36c6be671b9e6ce29c63bd.jpg)

![632e7c5e992729f8fa446ee7f3ccc5b7d31a877e43392eb5344249fe3cc52590.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/632e7c5e992729f8fa446ee7f3ccc5b7d31a877e43392eb5344249fe3cc52590.jpg)

![8bca5e3a48d825c7cbc098468e1e3f8de8ab264b4e3af3cda8a07e5f02a86f51.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/8bca5e3a48d825c7cbc098468e1e3f8de8ab264b4e3af3cda8a07e5f02a86f51.jpg)

![90667ef74431b977476ffd9cbf4ac23a6ac22c4ee43cd7bb71757d22e8012030.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/90667ef74431b977476ffd9cbf4ac23a6ac22c4ee43cd7bb71757d22e8012030.jpg)

![91b8fd525c95258373a06be0a78b2f38a52be056981e498571f70e2f8d56f758.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/91b8fd525c95258373a06be0a78b2f38a52be056981e498571f70e2f8d56f758.jpg)

![99a09974e6db90a1ad661283dee08d3b26699037a12cea2bba40c8f36ce741d5.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/99a09974e6db90a1ad661283dee08d3b26699037a12cea2bba40c8f36ce741d5.jpg)

![ae6f99e84231bf7c464a6ab9c2181d36ecb703015e142206c87f98c25ca77d76.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/ae6f99e84231bf7c464a6ab9c2181d36ecb703015e142206c87f98c25ca77d76.jpg)

![b69625e49f4049f8157fdacebb3d5ab108607b412bac69c7fd7dc653dcd98713.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/b69625e49f4049f8157fdacebb3d5ab108607b412bac69c7fd7dc653dcd98713.jpg)

![bdbdb0a2718ffa769e34b93fd6e98902fd269b91ce9bdb7fee60e0d42f944b21.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/bdbdb0a2718ffa769e34b93fd6e98902fd269b91ce9bdb7fee60e0d42f944b21.jpg)

![c4fed95dbd5ba7d0f7058688c4dd09e4eed9e538233ba3c97f5b68fe81036740.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/c4fed95dbd5ba7d0f7058688c4dd09e4eed9e538233ba3c97f5b68fe81036740.jpg)

![c9bae2edb9b6878a3c5667d60be388c488d94fd9d8363c769dbc07b96604887e.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/c9bae2edb9b6878a3c5667d60be388c488d94fd9d8363c769dbc07b96604887e.jpg)

![cbf9fa55099631f47ee4203b65aaeff421b7895ab9c567cd267bede8f7cd4ea6.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/cbf9fa55099631f47ee4203b65aaeff421b7895ab9c567cd267bede8f7cd4ea6.jpg)

![d4a788d68a314fd3ceda4e752e0f522b8b2a4bb314d725c5d791ac73cb3e1ffe.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/d4a788d68a314fd3ceda4e752e0f522b8b2a4bb314d725c5d791ac73cb3e1ffe.jpg)

![d5bb9f933ff0d54f7e590d275271a626e46139e62d3ea205583d444ab37903b1.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/d5bb9f933ff0d54f7e590d275271a626e46139e62d3ea205583d444ab37903b1.jpg)

![d7ba85346d00a341934d82d8af6416665bc4e36318fa0f13eda85310ae8717a3.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/d7ba85346d00a341934d82d8af6416665bc4e36318fa0f13eda85310ae8717a3.jpg)

![fa3156141d612a9cd2b0f29ae5e65b83344e19990281ebe8657a42a04c7a5ab0.jpg](../icml_results/2854_Controllable%20Molecule%20Generation%20via%20Sparse%20Representation%20Editing_%20An%20Interpretability-Driven%20Persp/tables/fa3156141d612a9cd2b0f29ae5e65b83344e19990281ebe8657a42a04c7a5ab0.jpg)

<a id="paper-2818"></a>

## RTInfer: Real-Time Inference of Multiple DNNs on Edge GPUs


### Images

![0b75ca0c9deb2af3e306287e5f2b29c9fab6e1ca13648bb460e249eba348d14e.jpg](../icml_results/2856_RTInfer_%20Real-Time%20Inference%20of%20Multiple%20DNNs%20on%20Edge%20GPUs/images/0b75ca0c9deb2af3e306287e5f2b29c9fab6e1ca13648bb460e249eba348d14e.jpg)

![2718d794271fd6bc5ef8f1622b98303340329307fdb2fd5343e0c29dbef36f1e.jpg](../icml_results/2856_RTInfer_%20Real-Time%20Inference%20of%20Multiple%20DNNs%20on%20Edge%20GPUs/images/2718d794271fd6bc5ef8f1622b98303340329307fdb2fd5343e0c29dbef36f1e.jpg)

![3a0b48432b498c6a31da86398b59a7d2933372e17c286d5fcf78ea425d7892fc.jpg](../icml_results/2856_RTInfer_%20Real-Time%20Inference%20of%20Multiple%20DNNs%20on%20Edge%20GPUs/images/3a0b48432b498c6a31da86398b59a7d2933372e17c286d5fcf78ea425d7892fc.jpg)

![480df15ce37322bff07e5fd0f45220985ff504b2c5cd3a98c76aa25abae6cf2f.jpg](../icml_results/2856_RTInfer_%20Real-Time%20Inference%20of%20Multiple%20DNNs%20on%20Edge%20GPUs/images/480df15ce37322bff07e5fd0f45220985ff504b2c5cd3a98c76aa25abae6cf2f.jpg)

![62a301673a45ce8b61db05c78b36a3fee157f107ede3f43e73a00945646db85f.jpg](../icml_results/2856_RTInfer_%20Real-Time%20Inference%20of%20Multiple%20DNNs%20on%20Edge%20GPUs/images/62a301673a45ce8b61db05c78b36a3fee157f107ede3f43e73a00945646db85f.jpg)

![7053053a137a25b9cfdcad951b1928e8b308d52c2702548b0ff46261606cfdcf.jpg](../icml_results/2856_RTInfer_%20Real-Time%20Inference%20of%20Multiple%20DNNs%20on%20Edge%20GPUs/images/7053053a137a25b9cfdcad951b1928e8b308d52c2702548b0ff46261606cfdcf.jpg)

![750aa2908e03eace1c803eb112e4f8611afb1e07ac7388e605081d99deb146b9.jpg](../icml_results/2856_RTInfer_%20Real-Time%20Inference%20of%20Multiple%20DNNs%20on%20Edge%20GPUs/images/750aa2908e03eace1c803eb112e4f8611afb1e07ac7388e605081d99deb146b9.jpg)

![8ee26162b26c5231325e12fe1829c202c1227b662b60fd3c5e4f6083a9a2ac10.jpg](../icml_results/2856_RTInfer_%20Real-Time%20Inference%20of%20Multiple%20DNNs%20on%20Edge%20GPUs/images/8ee26162b26c5231325e12fe1829c202c1227b662b60fd3c5e4f6083a9a2ac10.jpg)

![ae76e10146653fd81d7e0d552e74a46d3f85e8c90222077578086a79767c6039.jpg](../icml_results/2856_RTInfer_%20Real-Time%20Inference%20of%20Multiple%20DNNs%20on%20Edge%20GPUs/images/ae76e10146653fd81d7e0d552e74a46d3f85e8c90222077578086a79767c6039.jpg)

![b53d6fd8211b264af4df3a8617a17e6a8aadb3494e2f773f1b36ece973a369e3.jpg](../icml_results/2856_RTInfer_%20Real-Time%20Inference%20of%20Multiple%20DNNs%20on%20Edge%20GPUs/images/b53d6fd8211b264af4df3a8617a17e6a8aadb3494e2f773f1b36ece973a369e3.jpg)

![be88062c290e6f814027eb7fd8225df868cb4efc0a18816e094a94239728b9b2.jpg](../icml_results/2856_RTInfer_%20Real-Time%20Inference%20of%20Multiple%20DNNs%20on%20Edge%20GPUs/images/be88062c290e6f814027eb7fd8225df868cb4efc0a18816e094a94239728b9b2.jpg)

![c36249a298be8cb2bae15e004a89e60caed51a5b8be724b00735c4b6033877b9.jpg](../icml_results/2856_RTInfer_%20Real-Time%20Inference%20of%20Multiple%20DNNs%20on%20Edge%20GPUs/images/c36249a298be8cb2bae15e004a89e60caed51a5b8be724b00735c4b6033877b9.jpg)

![de5d4d0e9882addc2ebe9879ba226af64a2b55de7415819339ca3961ddc3e988.jpg](../icml_results/2856_RTInfer_%20Real-Time%20Inference%20of%20Multiple%20DNNs%20on%20Edge%20GPUs/images/de5d4d0e9882addc2ebe9879ba226af64a2b55de7415819339ca3961ddc3e988.jpg)

![fc56321ff44eb43550bd2c51ef179afcd2c1ef51408b92789ea3fee637847869.jpg](../icml_results/2856_RTInfer_%20Real-Time%20Inference%20of%20Multiple%20DNNs%20on%20Edge%20GPUs/images/fc56321ff44eb43550bd2c51ef179afcd2c1ef51408b92789ea3fee637847869.jpg)

### Tables

![c6ae78d6f069d60c600c6cb06b3a098545a07d0e472446747b7a7d9f7b6b45b3.jpg](../icml_results/2856_RTInfer_%20Real-Time%20Inference%20of%20Multiple%20DNNs%20on%20Edge%20GPUs/tables/c6ae78d6f069d60c600c6cb06b3a098545a07d0e472446747b7a7d9f7b6b45b3.jpg)

<a id="paper-2819"></a>

## ChartE$^3$: A Comprehensive Benchmark for End-to-End Chart Editing


### Images

![11a04c0b943b676f86fb1c33d4b35eccfe1adb8de5e08bcd215d1e82ca9464c2.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/images/11a04c0b943b676f86fb1c33d4b35eccfe1adb8de5e08bcd215d1e82ca9464c2.jpg)

![4b2a4f05dbf91faed55d382772ee42cc7fd2f38c6035392093c8c9df297fd76c.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/images/4b2a4f05dbf91faed55d382772ee42cc7fd2f38c6035392093c8c9df297fd76c.jpg)

![5abfcf638949b5b43eff21789b65d1efea307eee8fa8dc86d552a396963b57fa.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/images/5abfcf638949b5b43eff21789b65d1efea307eee8fa8dc86d552a396963b57fa.jpg)

![650bd184ac1c8ac471083155db01a592fde98aee9721d270ed5f044d917670fd.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/images/650bd184ac1c8ac471083155db01a592fde98aee9721d270ed5f044d917670fd.jpg)

![6650de58743f9d2ef8dbf5ffa242a6337504606f2aa9945d3bfc5a7adc8ee1bb.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/images/6650de58743f9d2ef8dbf5ffa242a6337504606f2aa9945d3bfc5a7adc8ee1bb.jpg)

![7fca6b0f4e01f639b66849278be40826fcc65df80f5a250360d863f6a3597109.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/images/7fca6b0f4e01f639b66849278be40826fcc65df80f5a250360d863f6a3597109.jpg)

![a711128b0f125780d9be70b439f7feaf780c80a706d486c76b8ce6f24db01b56.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/images/a711128b0f125780d9be70b439f7feaf780c80a706d486c76b8ce6f24db01b56.jpg)

![b854c0902b1e2446cb00e3c7b23220b89186c8febe07a081b674c11069ce4c9b.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/images/b854c0902b1e2446cb00e3c7b23220b89186c8febe07a081b674c11069ce4c9b.jpg)

![bdfc32bea1719573dbf1dfab63b0df5b29ba76534e39f22e0ba7e1caa97816c4.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/images/bdfc32bea1719573dbf1dfab63b0df5b29ba76534e39f22e0ba7e1caa97816c4.jpg)

![d8beb29244109db0dc74291178b2ca72a82571e5fe8decad6deafabb4b42e7ae.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/images/d8beb29244109db0dc74291178b2ca72a82571e5fe8decad6deafabb4b42e7ae.jpg)

![d8c5c2322870b5361b1ff1463f63e2209bc70fcd9e3cf4e57624320199d16bae.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/images/d8c5c2322870b5361b1ff1463f63e2209bc70fcd9e3cf4e57624320199d16bae.jpg)

![dfa1afa6e3f23c8dd57f5ca6bfe51945cb018e8e62d06da4b8e41773778a4ff0.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/images/dfa1afa6e3f23c8dd57f5ca6bfe51945cb018e8e62d06da4b8e41773778a4ff0.jpg)

![e7c1e5fc23f432223bfa2cce0766d7e3231bb17bac38f82447df90f2c511a6ff.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/images/e7c1e5fc23f432223bfa2cce0766d7e3231bb17bac38f82447df90f2c511a6ff.jpg)

![f6c6ff43180c30c35c475aa79467be6f98e85222291367b003d89b572606a699.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/images/f6c6ff43180c30c35c475aa79467be6f98e85222291367b003d89b572606a699.jpg)

![fc1304bac7295575fe375f0836c3e2d76234ce2fbfcefbff23f483aba5d55e01.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/images/fc1304bac7295575fe375f0836c3e2d76234ce2fbfcefbff23f483aba5d55e01.jpg)

### Tables

![1e4416e451e3c436bd6243de14338224503cc9a5976d7ded54ee33b117f31872.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/tables/1e4416e451e3c436bd6243de14338224503cc9a5976d7ded54ee33b117f31872.jpg)

![253893b97f114c7ea79ca5be6868f95ce77807ef01892fbf2645bd75b03ec700.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/tables/253893b97f114c7ea79ca5be6868f95ce77807ef01892fbf2645bd75b03ec700.jpg)

![2a3d2ce2540d88bd9a291195a40c964eba0a3ea9cfcebba9840b1d000502bf3a.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/tables/2a3d2ce2540d88bd9a291195a40c964eba0a3ea9cfcebba9840b1d000502bf3a.jpg)

![6a88ce720326073b1046c1f0004410c63412a0adc0937d2662dea38d116146da.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/tables/6a88ce720326073b1046c1f0004410c63412a0adc0937d2662dea38d116146da.jpg)

![98d8fe682b2317d4c821c1171f92e56abf26e669412a21e771c08f436357d3ab.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/tables/98d8fe682b2317d4c821c1171f92e56abf26e669412a21e771c08f436357d3ab.jpg)

![acd1e38c9b1e1acdb2bd141e0b39d51f5414bd26208e4b2327e30bd6135404f3.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/tables/acd1e38c9b1e1acdb2bd141e0b39d51f5414bd26208e4b2327e30bd6135404f3.jpg)

![b9dceceb27838a167fe79a7b672b8068447b26af3dfbc05c9f63d36225eda186.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/tables/b9dceceb27838a167fe79a7b672b8068447b26af3dfbc05c9f63d36225eda186.jpg)

![cc7c2e757c96eb2fe6318bbaffb28337c28c98d19f5c75b4c66fb671e31e6481.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/tables/cc7c2e757c96eb2fe6318bbaffb28337c28c98d19f5c75b4c66fb671e31e6481.jpg)

![d80a3669a1a635925f7a0dca1d7c71339eb1f6aa2cc75747fba7cc6e03b8974a.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/tables/d80a3669a1a635925f7a0dca1d7c71339eb1f6aa2cc75747fba7cc6e03b8974a.jpg)

![ea54cb1e346712c0dbf27da629e658482b8ca38016d7f0619f7a8e50a3fd152a.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/tables/ea54cb1e346712c0dbf27da629e658482b8ca38016d7f0619f7a8e50a3fd152a.jpg)

![ef12cd72f848d0918eafbc119ad42724ce535065b9b925f70d398b4eea8fa9ec.jpg](../icml_results/2857_ChartE%24%5E3%24_%20A%20Comprehensive%20Benchmark%20for%20End-to-End%20Chart%20Editing/tables/ef12cd72f848d0918eafbc119ad42724ce535065b9b925f70d398b4eea8fa9ec.jpg)

<a id="paper-2820"></a>

## DECO: Decoupled Multimodal Diffusion Transformer for Bimanual Dexterous Manipulation with a Plugin Tactile Adapter


### Images

![0e35ef506b856572f0f0434ec5f09b41da70ce4143878ac99135a86d61c6ce64.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/images/0e35ef506b856572f0f0434ec5f09b41da70ce4143878ac99135a86d61c6ce64.jpg)

![369cef5bbfe4bd378cb226940850468c7c697832189862385afc2236b19fb22c.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/images/369cef5bbfe4bd378cb226940850468c7c697832189862385afc2236b19fb22c.jpg)

![4d3ebca311e2bc6fe8fab141d998d6731d7817b1ab583f098530c00b35e6b8fa.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/images/4d3ebca311e2bc6fe8fab141d998d6731d7817b1ab583f098530c00b35e6b8fa.jpg)

![67864238525ea04bf76129380e50484924e98843cf64065e9e877b1fb79f2804.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/images/67864238525ea04bf76129380e50484924e98843cf64065e9e877b1fb79f2804.jpg)

![79836dc52cdaddb6c6c112f039fd5dcb4674a5ad8620f9cdfe10eba5166f1bee.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/images/79836dc52cdaddb6c6c112f039fd5dcb4674a5ad8620f9cdfe10eba5166f1bee.jpg)

![7e9f264eec628a3e9a006849a3a8ca0c46c1c37d7443253cf73fcab2bfda3214.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/images/7e9f264eec628a3e9a006849a3a8ca0c46c1c37d7443253cf73fcab2bfda3214.jpg)

![b2b54d6a528e5167269f3adb2d78da46489ec65431ac6ce64f0541d00743e9ba.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/images/b2b54d6a528e5167269f3adb2d78da46489ec65431ac6ce64f0541d00743e9ba.jpg)

![dedc94581e76dc0d414b5ed74eefc4719f14c716d4449d1803d25d8e7f8699a8.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/images/dedc94581e76dc0d414b5ed74eefc4719f14c716d4449d1803d25d8e7f8699a8.jpg)

### Tables

![02ff0628ebdbccbccbfb63013beb41f2d8297c5553c07b48c76a261edc05184e.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/02ff0628ebdbccbccbfb63013beb41f2d8297c5553c07b48c76a261edc05184e.jpg)

![11cff171ad03e3476fdfba72c69a5f59498b3a5e4a7dfa0f20f951c128bd4dc7.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/11cff171ad03e3476fdfba72c69a5f59498b3a5e4a7dfa0f20f951c128bd4dc7.jpg)

![2969bc3ec13fdb1be1031ddc8fb55d064db917686bd5a298e28ab3f5eaf5c5f4.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/2969bc3ec13fdb1be1031ddc8fb55d064db917686bd5a298e28ab3f5eaf5c5f4.jpg)

![33c577978d91b8132dc99d7ce1dd17fce3cace5e4fc7583716bec52f95e4adf5.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/33c577978d91b8132dc99d7ce1dd17fce3cace5e4fc7583716bec52f95e4adf5.jpg)

![4abc30a792b3cf034a779ff81204434f4cd8942229de3124e04523b88924115b.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/4abc30a792b3cf034a779ff81204434f4cd8942229de3124e04523b88924115b.jpg)

![4e47652a4ec6183c3df7c5cd3c55778a2027b35d69157972169d300db44bc092.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/4e47652a4ec6183c3df7c5cd3c55778a2027b35d69157972169d300db44bc092.jpg)

![5ba0457e1a972e1953c010e788dd7efec81ef1853422d51fc95bf1d5aef88832.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/5ba0457e1a972e1953c010e788dd7efec81ef1853422d51fc95bf1d5aef88832.jpg)

![66a558af33b986e9108c78e44d402761c88f00b4820b0402ba19d263fcf2cdec.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/66a558af33b986e9108c78e44d402761c88f00b4820b0402ba19d263fcf2cdec.jpg)

![74f4627acc7e135ff45fb24023e186661f5ee51dbce3c8d0c2eadc8a356952af.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/74f4627acc7e135ff45fb24023e186661f5ee51dbce3c8d0c2eadc8a356952af.jpg)

![9f9f92b2a484ca394ee15f7731cacd5a79e7a51bb96d333a69abcc6fdcb63d6e.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/9f9f92b2a484ca394ee15f7731cacd5a79e7a51bb96d333a69abcc6fdcb63d6e.jpg)

![a57c371a0f761e08c4a0f2ec1a44a14b0b96754a090f72279d14d1dd36c29f4f.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/a57c371a0f761e08c4a0f2ec1a44a14b0b96754a090f72279d14d1dd36c29f4f.jpg)

![a5b7927c21673022fd3cff5771505ffc625d567193086b66dd2c4f9f73af6b40.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/a5b7927c21673022fd3cff5771505ffc625d567193086b66dd2c4f9f73af6b40.jpg)

![ac6a8683849484174928d01c9e45a692cb817f54e5ff95092e27fbeade441075.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/ac6a8683849484174928d01c9e45a692cb817f54e5ff95092e27fbeade441075.jpg)

![ae378e70320341bb551c00bcde14a3015f905d1a7d2c66d14196b2b04a8d842e.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/ae378e70320341bb551c00bcde14a3015f905d1a7d2c66d14196b2b04a8d842e.jpg)

![b8d75f8abab1a14f565495881b0f53dad83129690ec30e8889dbbb0a058100c2.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/b8d75f8abab1a14f565495881b0f53dad83129690ec30e8889dbbb0a058100c2.jpg)

![dba5d72ab021572acecda83f77027c0d1abdf3c52e1b5efe279c1f591c487005.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/dba5d72ab021572acecda83f77027c0d1abdf3c52e1b5efe279c1f591c487005.jpg)

![dc5b4f570751faecf276852af4de603179791f1f5c21e81df8480aad40b9d5e1.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/dc5b4f570751faecf276852af4de603179791f1f5c21e81df8480aad40b9d5e1.jpg)

![df5b4e19a53ad972e3fdb2c035bb68bd0ecc3a3a9d74b106bb9a0736fcaaa5ea.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/df5b4e19a53ad972e3fdb2c035bb68bd0ecc3a3a9d74b106bb9a0736fcaaa5ea.jpg)

![ef3bd2c64036e98feafe530cc72459af421763b3c15f6812c418e0478ad8e8c2.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/ef3bd2c64036e98feafe530cc72459af421763b3c15f6812c418e0478ad8e8c2.jpg)

![fe785d337e1ca2dee87c4f1e0339d6ca19caa77dbc35f63e21a4158d058e3e15.jpg](../icml_results/2858_DECO_%20Decoupled%20Multimodal%20Diffusion%20Transformer%20for%20Bimanual%20Dexterous%20Manipulation%20with%20a%20Plugin%20T/tables/fe785d337e1ca2dee87c4f1e0339d6ca19caa77dbc35f63e21a4158d058e3e15.jpg)

<a id="paper-2821"></a>

## MIMOMamba: From Scalar Duality to Matrix-Valued Attention


### Images

![266bf99941c1564d513b7da0d507949a3dc047c7a41b6f82dc2d687fad8aae88.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/images/266bf99941c1564d513b7da0d507949a3dc047c7a41b6f82dc2d687fad8aae88.jpg)

![3150c53277247a7ec8953c62116efd1d3c189d4ea33bacb776bbbd98e332fcb5.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/images/3150c53277247a7ec8953c62116efd1d3c189d4ea33bacb776bbbd98e332fcb5.jpg)

![47a29ad7713cfb51e4d6c9501725c0b670aa515656c8be12e6ef888497e6eff7.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/images/47a29ad7713cfb51e4d6c9501725c0b670aa515656c8be12e6ef888497e6eff7.jpg)

![4f7ae3f5b21626fca3e7fd92076b6c1f9664de33e1c48110542ad8e79412be1d.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/images/4f7ae3f5b21626fca3e7fd92076b6c1f9664de33e1c48110542ad8e79412be1d.jpg)

![7b62d3ab1a7acf95489941cb478b97e5e1ad27ec483a9c49b917b79fe2158852.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/images/7b62d3ab1a7acf95489941cb478b97e5e1ad27ec483a9c49b917b79fe2158852.jpg)

![cb44b04cf75b4457ecec96e3b94ad385d184ef510fb7aa22313d9b957de04fb8.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/images/cb44b04cf75b4457ecec96e3b94ad385d184ef510fb7aa22313d9b957de04fb8.jpg)

![e508840f38eaaa5e07659002135f1fa96e4b3cf2c750e7f507595635050490c4.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/images/e508840f38eaaa5e07659002135f1fa96e4b3cf2c750e7f507595635050490c4.jpg)

### Tables

![13b29c40dbba065ad7fd7ad1ba8f10d18e5ca15557aa3d61b629f0e11bdc0f8a.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/tables/13b29c40dbba065ad7fd7ad1ba8f10d18e5ca15557aa3d61b629f0e11bdc0f8a.jpg)

![34c94f49c6db71e83bb9eb258d88e151ab63b950f14e8f305bd04cf5204d83ab.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/tables/34c94f49c6db71e83bb9eb258d88e151ab63b950f14e8f305bd04cf5204d83ab.jpg)

![465d009f82de1ea53600f1f4e642b7c1b9a6e2cbbf670f5280b5042dece26751.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/tables/465d009f82de1ea53600f1f4e642b7c1b9a6e2cbbf670f5280b5042dece26751.jpg)

![4d480662cddd4887ae7e6019caac4870fbdbdfc71c473843ce3044941d4d3b8f.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/tables/4d480662cddd4887ae7e6019caac4870fbdbdfc71c473843ce3044941d4d3b8f.jpg)

![71ad979c6b141b688e128eda2b5c5f8352a48a6f21a6920bbb1fbe4791721593.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/tables/71ad979c6b141b688e128eda2b5c5f8352a48a6f21a6920bbb1fbe4791721593.jpg)

![7f49b1df4f278d407a9a1b5f97caae0324eaa45e2f22f34f8df333d89486d788.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/tables/7f49b1df4f278d407a9a1b5f97caae0324eaa45e2f22f34f8df333d89486d788.jpg)

![97aedae049067e67665965769a8339678a80e29704861944c0c1437fb923aa8e.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/tables/97aedae049067e67665965769a8339678a80e29704861944c0c1437fb923aa8e.jpg)

![bfd4d321e465874acba3eb8a3fc69c14ac7955f0dd090b90e31f06923ae2c467.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/tables/bfd4d321e465874acba3eb8a3fc69c14ac7955f0dd090b90e31f06923ae2c467.jpg)

![cf5b0e4b7ec34fc361a7f6cf7bae578867c3f8d494285e8b5e41818afd326241.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/tables/cf5b0e4b7ec34fc361a7f6cf7bae578867c3f8d494285e8b5e41818afd326241.jpg)

![d331818722ecefcece249f62e1a6a2e72e6396f8b17e58337e04e5861bce80df.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/tables/d331818722ecefcece249f62e1a6a2e72e6396f8b17e58337e04e5861bce80df.jpg)

![e6c167df9701696fe4c4213beef0af85798f1fdc743cc2334247741ed2125ee9.jpg](../icml_results/2859_MIMOMamba_%20From%20Scalar%20Duality%20to%20Matrix-Valued%20Attention/tables/e6c167df9701696fe4c4213beef0af85798f1fdc743cc2334247741ed2125ee9.jpg)

<a id="paper-2822"></a>

## Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting


### Images

![1fb402526cc927e5ff95f3582fddca2d796f60d2bb9b75d6288178456403b9ce.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/images/1fb402526cc927e5ff95f3582fddca2d796f60d2bb9b75d6288178456403b9ce.jpg)

![3cca2be5443ff6aec71479db851af91eb003dba3cbe72082604203733cdbdc1e.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/images/3cca2be5443ff6aec71479db851af91eb003dba3cbe72082604203733cdbdc1e.jpg)

![45dbe04bc4a2c23fc04506e8c597ccd9fdf0aacb9b7b2b592ccadf9fef41507d.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/images/45dbe04bc4a2c23fc04506e8c597ccd9fdf0aacb9b7b2b592ccadf9fef41507d.jpg)

![68f331730ba574736e39b9969a10b26411e039264495e72e7cd48b8f9555f461.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/images/68f331730ba574736e39b9969a10b26411e039264495e72e7cd48b8f9555f461.jpg)

![77d049351431c86899da2305ebd5c044208628a88ec344f0be1d594bcd31be14.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/images/77d049351431c86899da2305ebd5c044208628a88ec344f0be1d594bcd31be14.jpg)

![7a899b54b4257da269a971fb0dc5e36dcd7b73f57e140025cb4a28abc30309cb.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/images/7a899b54b4257da269a971fb0dc5e36dcd7b73f57e140025cb4a28abc30309cb.jpg)

![a9fd7cd36402f23a8a4b5800af03a9f9b244a9aac9710bac147ba282572a8a3b.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/images/a9fd7cd36402f23a8a4b5800af03a9f9b244a9aac9710bac147ba282572a8a3b.jpg)

![c6c874807186865c84ada5ef48e9b7db3b0df9863fa9e2682c7dec465712b382.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/images/c6c874807186865c84ada5ef48e9b7db3b0df9863fa9e2682c7dec465712b382.jpg)

![df83c20742dd6098b0471c82e1307a0282f1ee9da31fe137798d524fead003b2.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/images/df83c20742dd6098b0471c82e1307a0282f1ee9da31fe137798d524fead003b2.jpg)

![ed73d7b7e6a7e2b04b7e4c59207b426022e359fbb31a659b06000265f1c8dddd.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/images/ed73d7b7e6a7e2b04b7e4c59207b426022e359fbb31a659b06000265f1c8dddd.jpg)

### Tables

![0b993024568269e43cea118ce67c86f404298482429201c699001fa4185a1cfb.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/tables/0b993024568269e43cea118ce67c86f404298482429201c699001fa4185a1cfb.jpg)

![328624851dc1adbd65b104b77acf3951a81d2161e2d8948633c7f865bc5c2689.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/tables/328624851dc1adbd65b104b77acf3951a81d2161e2d8948633c7f865bc5c2689.jpg)

![342d7641afa4d811d54a6d24536242552a538d04cdb32a0a0a8a58febe03e115.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/tables/342d7641afa4d811d54a6d24536242552a538d04cdb32a0a0a8a58febe03e115.jpg)

![3e380e3edd7eea86c2871d686684d9bbfcf9fc70eeb1c7d192fef24a61c5056d.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/tables/3e380e3edd7eea86c2871d686684d9bbfcf9fc70eeb1c7d192fef24a61c5056d.jpg)

![4d79025d035f1f216dbb59c4521456de63f579b7fa21967ca10c0a720ffe593c.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/tables/4d79025d035f1f216dbb59c4521456de63f579b7fa21967ca10c0a720ffe593c.jpg)

![7053e2c44b4412c32bb319cec88a01cce51106c83efb694350898a09385b2c44.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/tables/7053e2c44b4412c32bb319cec88a01cce51106c83efb694350898a09385b2c44.jpg)

![732ae3d661d245fbcb77adba351b68c57eca7efb322f2391bea3800a80a0603f.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/tables/732ae3d661d245fbcb77adba351b68c57eca7efb322f2391bea3800a80a0603f.jpg)

![749c9c29ae41ba4ccbe509ff629e4cc46ca6f3edc79d3d1e919528a09af6c14f.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/tables/749c9c29ae41ba4ccbe509ff629e4cc46ca6f3edc79d3d1e919528a09af6c14f.jpg)

![a85b3d877857bd930428ba9e7e7460f22466b433f4d54dec3d9eda16b5e99570.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/tables/a85b3d877857bd930428ba9e7e7460f22466b433f4d54dec3d9eda16b5e99570.jpg)

![ce5a9a4e4e0baf812cf688e9fe06d49a863d21ff253edba773c7f16a9911af9f.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/tables/ce5a9a4e4e0baf812cf688e9fe06d49a863d21ff253edba773c7f16a9911af9f.jpg)

![eb1b2dd3fb1c58fc8a44ade20504017181f7878626ab2b5191e03077891934a3.jpg](../icml_results/2860_Parametric%20Prior%20Mapping%20Framework%20for%20Non-stationary%20Probabilistic%20Time%20Series%20Forecasting/tables/eb1b2dd3fb1c58fc8a44ade20504017181f7878626ab2b5191e03077891934a3.jpg)

<a id="paper-2823"></a>

## FedRGL: Robust Federated Graph Learning under Label Noise


### Images

![0f9169eed46ad07dc52d05a04d7ff232cdc38e0c090f1ac919c00c02332c3c6d.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/images/0f9169eed46ad07dc52d05a04d7ff232cdc38e0c090f1ac919c00c02332c3c6d.jpg)

![2bca37d291699522c1e3d1b0675ab683b1947363cc4f9a011e3e37cc89cc6cb7.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/images/2bca37d291699522c1e3d1b0675ab683b1947363cc4f9a011e3e37cc89cc6cb7.jpg)

![352fa43c69778d684625ee00c395aa08166d959df208dca29c10faa02c747f4c.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/images/352fa43c69778d684625ee00c395aa08166d959df208dca29c10faa02c747f4c.jpg)

![5e8bc0ddc134dca3a9c3c4f13e15025d9012879e8003454c9775f59889b78946.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/images/5e8bc0ddc134dca3a9c3c4f13e15025d9012879e8003454c9775f59889b78946.jpg)

![923092e08925624e71478beeb9705b7657fd66861d18ef34a246d1049393b017.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/images/923092e08925624e71478beeb9705b7657fd66861d18ef34a246d1049393b017.jpg)

![a8e05525c9cb940209548719265ce3735e1943e2ca99ce64414aef15b21bffea.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/images/a8e05525c9cb940209548719265ce3735e1943e2ca99ce64414aef15b21bffea.jpg)

![c8ff33a070569143d73d058ab1bfc66c51c30eb0f4d9ebada0cc96cbc04a7b41.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/images/c8ff33a070569143d73d058ab1bfc66c51c30eb0f4d9ebada0cc96cbc04a7b41.jpg)

![cdedfd235a142dd883cd14707f0619e8940a293edfa1c01abb96469f0d2b6e04.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/images/cdedfd235a142dd883cd14707f0619e8940a293edfa1c01abb96469f0d2b6e04.jpg)

![e5a50395d2aed7f5d29aed0f9a1d8391f8fb2a77cf30db546adb90ec9dad5bcc.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/images/e5a50395d2aed7f5d29aed0f9a1d8391f8fb2a77cf30db546adb90ec9dad5bcc.jpg)

![f7007d9ddeffaadf745b1d84ab6c84945704dfd9f9705a746a0b5346106ad3ce.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/images/f7007d9ddeffaadf745b1d84ab6c84945704dfd9f9705a746a0b5346106ad3ce.jpg)

![fcbcf8ba79798903af7384118f282556a0fbaccf2615ffcb3b63f52e3ce2aa44.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/images/fcbcf8ba79798903af7384118f282556a0fbaccf2615ffcb3b63f52e3ce2aa44.jpg)

### Tables

![2834998179489da4683258a2e0333761e75eb04c144e4db9bb427359198b90bc.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/tables/2834998179489da4683258a2e0333761e75eb04c144e4db9bb427359198b90bc.jpg)

![348844db6cf41edfe66621686489c11b8cb91cd0f99c3c5bb1fe82a2154958e9.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/tables/348844db6cf41edfe66621686489c11b8cb91cd0f99c3c5bb1fe82a2154958e9.jpg)

![3899770132ac58c9c851b6d7aa73a13e92cde33b6eec3dd76e65f5598fda1502.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/tables/3899770132ac58c9c851b6d7aa73a13e92cde33b6eec3dd76e65f5598fda1502.jpg)

![8ef5536433db72927ee33b411ecc698dd974453339f7f168fe9e27af88b96aa2.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/tables/8ef5536433db72927ee33b411ecc698dd974453339f7f168fe9e27af88b96aa2.jpg)

![9cc70517539518a2863fd89f1344d36353d8f8aa4e1e14b6153116b4fd579dfa.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/tables/9cc70517539518a2863fd89f1344d36353d8f8aa4e1e14b6153116b4fd579dfa.jpg)

![9e03ffbeef7d38d0faa687cd60f3f5dd258e6157f4b2a736204b96a00ba2e687.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/tables/9e03ffbeef7d38d0faa687cd60f3f5dd258e6157f4b2a736204b96a00ba2e687.jpg)

![d450b127b0568739c40d2d6aeb2547990124458408f9cae5fecfe4ab820cb5e3.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/tables/d450b127b0568739c40d2d6aeb2547990124458408f9cae5fecfe4ab820cb5e3.jpg)

![dd6e56dc25530dd3998e9355369e3c639b40145da86dc44f91ba5133581dfd65.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/tables/dd6e56dc25530dd3998e9355369e3c639b40145da86dc44f91ba5133581dfd65.jpg)

![ee25e978e35257ecf6e620bb5fed958419a4a80aa07878f4827135a3fcdfa664.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/tables/ee25e978e35257ecf6e620bb5fed958419a4a80aa07878f4827135a3fcdfa664.jpg)

![ff884bbeda32e0d9019a7d993464f57afee22c3b6ec7cc0987f69f49b82542fb.jpg](../icml_results/2861_FedRGL_%20Robust%20Federated%20Graph%20Learning%20under%20Label%20Noise/tables/ff884bbeda32e0d9019a7d993464f57afee22c3b6ec7cc0987f69f49b82542fb.jpg)

<a id="paper-2824"></a>

## Private and Stable Test-time Adaptation with Differential Privacy


### Images

![534f0d4b3311af1c55914fbc582ab2f38cdb62e26f7056d21d9a6c6de7092eec.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/images/534f0d4b3311af1c55914fbc582ab2f38cdb62e26f7056d21d9a6c6de7092eec.jpg)

![93ecf52cce5c04493b98f161b3726056bad74d64e8020d21476e610a86468723.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/images/93ecf52cce5c04493b98f161b3726056bad74d64e8020d21476e610a86468723.jpg)

![b8830c2c2922a72eb09d2ee075adb5d66b7892abc4aab5d1209418576773d3a8.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/images/b8830c2c2922a72eb09d2ee075adb5d66b7892abc4aab5d1209418576773d3a8.jpg)

### Tables

![00cf5973ac9952862623191dddd836b7ffa63a551e8e684065aeae56d439e995.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/00cf5973ac9952862623191dddd836b7ffa63a551e8e684065aeae56d439e995.jpg)

![12cb0c99671ef08c3551b7c04e6ea2c557136168ff9bd66e3c08dbcd0d8424b5.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/12cb0c99671ef08c3551b7c04e6ea2c557136168ff9bd66e3c08dbcd0d8424b5.jpg)

![16851a52088e50f24abafd1c31a3cfef977a2f81e332e0adec19aeabbda5c886.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/16851a52088e50f24abafd1c31a3cfef977a2f81e332e0adec19aeabbda5c886.jpg)

![3907d977a5ae1444d59fc581173ab0262ef0a651e5001f89d0a1f3e73a4ab4a3.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/3907d977a5ae1444d59fc581173ab0262ef0a651e5001f89d0a1f3e73a4ab4a3.jpg)

![735a4c555dffd34a878c3520608179b7e775a0f81ce41944124466279399cbb2.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/735a4c555dffd34a878c3520608179b7e775a0f81ce41944124466279399cbb2.jpg)

![73846c39b4bf1b5d6b90586e5d2e8563fb30716b360c40e30f50202e029cb091.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/73846c39b4bf1b5d6b90586e5d2e8563fb30716b360c40e30f50202e029cb091.jpg)

![7aba3893091efd6e1cef314e3d144ace91772f83af294798fe7cdb1e5c432d37.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/7aba3893091efd6e1cef314e3d144ace91772f83af294798fe7cdb1e5c432d37.jpg)

![7f6d0aec230ce56ea4d22900efc1e59e4b5a610019af6a37979e1b3076561267.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/7f6d0aec230ce56ea4d22900efc1e59e4b5a610019af6a37979e1b3076561267.jpg)

![89dbc3f8445f9895b7f13bc8d311d1037fc029d6b6c416ad0928b4728233585b.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/89dbc3f8445f9895b7f13bc8d311d1037fc029d6b6c416ad0928b4728233585b.jpg)

![91d9032aaabe37fabacce925601a1253ec7401f4d24c9ca0c17fb02842b1cc22.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/91d9032aaabe37fabacce925601a1253ec7401f4d24c9ca0c17fb02842b1cc22.jpg)

![995f1f17c5a7210ad5adc96c1b1479b1d70f46226e8477c97504d039a50baaa9.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/995f1f17c5a7210ad5adc96c1b1479b1d70f46226e8477c97504d039a50baaa9.jpg)

![9f15b7da8def982e4abd7555d63b0b071b9a7408ba2566fc6161628067604eb9.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/9f15b7da8def982e4abd7555d63b0b071b9a7408ba2566fc6161628067604eb9.jpg)

![a9dddf48df7eefd78efdc2814725ffc266501a8e8f0e189670c622bd14fb9d4d.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/a9dddf48df7eefd78efdc2814725ffc266501a8e8f0e189670c622bd14fb9d4d.jpg)

![b2e588e4ac2919fe9216c191a1c1ad1b69f2c3c0dfb7db3bea85a335663eee48.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/b2e588e4ac2919fe9216c191a1c1ad1b69f2c3c0dfb7db3bea85a335663eee48.jpg)

![bc258a94f41b4b7fde8a1d84c99b00cf07dd2c4f1464d823b05914a22233f61e.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/bc258a94f41b4b7fde8a1d84c99b00cf07dd2c4f1464d823b05914a22233f61e.jpg)

![c993af525f98ef78942e9ee9e9e27d836b900d8c621d826f4c13758003888df4.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/c993af525f98ef78942e9ee9e9e27d836b900d8c621d826f4c13758003888df4.jpg)

![caa7a5773115579d8277c6f58d180315cdc5143f58aa2b4c618825d6de5d6e24.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/caa7a5773115579d8277c6f58d180315cdc5143f58aa2b4c618825d6de5d6e24.jpg)

![e8314fc5b83709ecb40098cda7c590cd7cae5c20636985fde03ea772e5626244.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/e8314fc5b83709ecb40098cda7c590cd7cae5c20636985fde03ea772e5626244.jpg)

![ecb8d4f71cf71c418c65d2ae756ea0977b681517b68cc049816a2532399ea28e.jpg](../icml_results/2862_Private%20and%20Stable%20Test-time%20Adaptation%20with%20Differential%20Privacy/tables/ecb8d4f71cf71c418c65d2ae756ea0977b681517b68cc049816a2532399ea28e.jpg)

<a id="paper-2825"></a>

## HeraSys: Collaborative Serving of Multiple LLM Workflows via Fine-Grained End-to-End Optimization


### Images

![010316edf3e4b2882e0a046ac912c9e6f01b00ec89bc4406da2321cdd3933fa3.jpg](../icml_results/2863_HeraSys_%20Collaborative%20Serving%20of%20Multiple%20LLM%20Workflows%20via%20Fine-Grained%20End-to-End%20Optimization/images/010316edf3e4b2882e0a046ac912c9e6f01b00ec89bc4406da2321cdd3933fa3.jpg)

![0b19caf51ec7f8fbdfb7f52ea48e9eb3956ab8d43cbdb44dfac6fbf8bc9931a2.jpg](../icml_results/2863_HeraSys_%20Collaborative%20Serving%20of%20Multiple%20LLM%20Workflows%20via%20Fine-Grained%20End-to-End%20Optimization/images/0b19caf51ec7f8fbdfb7f52ea48e9eb3956ab8d43cbdb44dfac6fbf8bc9931a2.jpg)

![2172f21f9a1f8f680130105549084db59ae96ccdb84168e7a1934b715946377a.jpg](../icml_results/2863_HeraSys_%20Collaborative%20Serving%20of%20Multiple%20LLM%20Workflows%20via%20Fine-Grained%20End-to-End%20Optimization/images/2172f21f9a1f8f680130105549084db59ae96ccdb84168e7a1934b715946377a.jpg)

![5f75bb645977943fae9c43b6abdcfec9b622ab9d7acb3eee56d1bfbc8b207440.jpg](../icml_results/2863_HeraSys_%20Collaborative%20Serving%20of%20Multiple%20LLM%20Workflows%20via%20Fine-Grained%20End-to-End%20Optimization/images/5f75bb645977943fae9c43b6abdcfec9b622ab9d7acb3eee56d1bfbc8b207440.jpg)

![6843f34c2cfb081bf062eeeabe4b5eebd75aa4541d9b317c7b2c268dfcfaf6e0.jpg](../icml_results/2863_HeraSys_%20Collaborative%20Serving%20of%20Multiple%20LLM%20Workflows%20via%20Fine-Grained%20End-to-End%20Optimization/images/6843f34c2cfb081bf062eeeabe4b5eebd75aa4541d9b317c7b2c268dfcfaf6e0.jpg)

![72e88673bb964ff1204ec48e76efca596fa65407cee6ac65681b227550705fd3.jpg](../icml_results/2863_HeraSys_%20Collaborative%20Serving%20of%20Multiple%20LLM%20Workflows%20via%20Fine-Grained%20End-to-End%20Optimization/images/72e88673bb964ff1204ec48e76efca596fa65407cee6ac65681b227550705fd3.jpg)

![9d59ee8df5b46f1807c3d5fe8f23b5bc9ecedf8fb63527c5b714558d99fa6d40.jpg](../icml_results/2863_HeraSys_%20Collaborative%20Serving%20of%20Multiple%20LLM%20Workflows%20via%20Fine-Grained%20End-to-End%20Optimization/images/9d59ee8df5b46f1807c3d5fe8f23b5bc9ecedf8fb63527c5b714558d99fa6d40.jpg)

![b685153e53ff0cb1e1b12da8dbb3fe845fc209d2bdb77d001c9660ac9722f060.jpg](../icml_results/2863_HeraSys_%20Collaborative%20Serving%20of%20Multiple%20LLM%20Workflows%20via%20Fine-Grained%20End-to-End%20Optimization/images/b685153e53ff0cb1e1b12da8dbb3fe845fc209d2bdb77d001c9660ac9722f060.jpg)

![f1005afc2957223a13af3a7af8f0ab9644986069fc49d8eb7895767c70d12fd1.jpg](../icml_results/2863_HeraSys_%20Collaborative%20Serving%20of%20Multiple%20LLM%20Workflows%20via%20Fine-Grained%20End-to-End%20Optimization/images/f1005afc2957223a13af3a7af8f0ab9644986069fc49d8eb7895767c70d12fd1.jpg)

### Tables

![1ceef937c24abc405edff20d8bf0e1a79c21cc93c62bcb3b338580fed202d30d.jpg](../icml_results/2863_HeraSys_%20Collaborative%20Serving%20of%20Multiple%20LLM%20Workflows%20via%20Fine-Grained%20End-to-End%20Optimization/tables/1ceef937c24abc405edff20d8bf0e1a79c21cc93c62bcb3b338580fed202d30d.jpg)

![71cb214a3c95264366ece18f2b1051489c01fb6aad1af55b36089d879764ad7b.jpg](../icml_results/2863_HeraSys_%20Collaborative%20Serving%20of%20Multiple%20LLM%20Workflows%20via%20Fine-Grained%20End-to-End%20Optimization/tables/71cb214a3c95264366ece18f2b1051489c01fb6aad1af55b36089d879764ad7b.jpg)

![9affed6bb56dd3301f2ee985dd500488ba17824488308946113f8482d338099d.jpg](../icml_results/2863_HeraSys_%20Collaborative%20Serving%20of%20Multiple%20LLM%20Workflows%20via%20Fine-Grained%20End-to-End%20Optimization/tables/9affed6bb56dd3301f2ee985dd500488ba17824488308946113f8482d338099d.jpg)

<a id="paper-2826"></a>

## Learning from Comparison: Constrained Projection Policy Optimization for Pareto-Front Improvement


### Images

![26509301dc58bc96895b6e51cc12d31184f7e93e0dc0629f889cd911e451afd2.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/images/26509301dc58bc96895b6e51cc12d31184f7e93e0dc0629f889cd911e451afd2.jpg)

![5fd63b90f04e23c50c289b08706177db395c578a93be87ba0eb172e31452b73b.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/images/5fd63b90f04e23c50c289b08706177db395c578a93be87ba0eb172e31452b73b.jpg)

![7fe1070293ca1c8cdfee603cfa25965ef971c0d9c5401669e74ddee63d61b625.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/images/7fe1070293ca1c8cdfee603cfa25965ef971c0d9c5401669e74ddee63d61b625.jpg)

![80b7204f2f4dddd49b81dfa66a4f8db75e0766466534ce8a17f0f745591bf385.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/images/80b7204f2f4dddd49b81dfa66a4f8db75e0766466534ce8a17f0f745591bf385.jpg)

![88edc4e07c707552274cefd08955e59fddc9808e69c6d097f6bf36816a656a4d.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/images/88edc4e07c707552274cefd08955e59fddc9808e69c6d097f6bf36816a656a4d.jpg)

![8ca3188e6b15c811a4125a4d9b07dcf100ed6a89ead7e01fd276f2e596a06b37.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/images/8ca3188e6b15c811a4125a4d9b07dcf100ed6a89ead7e01fd276f2e596a06b37.jpg)

![a5e0503aba4d9d246673196012bb8ba128635ffdf4a8356b50dc7d3521dc4603.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/images/a5e0503aba4d9d246673196012bb8ba128635ffdf4a8356b50dc7d3521dc4603.jpg)

![b0860028c44800367061279123a02e2d78903b3c8938378a94a507c377a0ceea.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/images/b0860028c44800367061279123a02e2d78903b3c8938378a94a507c377a0ceea.jpg)

![d62bf55f366be7b160fabe4e7378945c1dac1ca62f81fdd493ea4810ee510680.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/images/d62bf55f366be7b160fabe4e7378945c1dac1ca62f81fdd493ea4810ee510680.jpg)

![e2351d74f1bc0eecb31b0d4e3779f879e1ec5f2dd8bc2d9779c68e85645933f7.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/images/e2351d74f1bc0eecb31b0d4e3779f879e1ec5f2dd8bc2d9779c68e85645933f7.jpg)

### Tables

![00bbd0c442144c1dc80f2c23b52eec4d4b9a692ee0dd6a9427ccd4b1131f4cb3.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/tables/00bbd0c442144c1dc80f2c23b52eec4d4b9a692ee0dd6a9427ccd4b1131f4cb3.jpg)

![10689a8970164c615831bf3e102b6c0b54e48b946646b8180beb1722fde38c45.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/tables/10689a8970164c615831bf3e102b6c0b54e48b946646b8180beb1722fde38c45.jpg)

![536b31e92b16a61d1286cd7499d9cc45dd66601b9a81df80a29584acf3b3ce0f.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/tables/536b31e92b16a61d1286cd7499d9cc45dd66601b9a81df80a29584acf3b3ce0f.jpg)

![5aac8761fc084e93d2cfc37805a98f27a1c19070a223539c4deb6c337714cdb8.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/tables/5aac8761fc084e93d2cfc37805a98f27a1c19070a223539c4deb6c337714cdb8.jpg)

![676945243c170dbdcc9d3ddaa5b84c94ce98edb54a3261bea01e4493b63aa81b.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/tables/676945243c170dbdcc9d3ddaa5b84c94ce98edb54a3261bea01e4493b63aa81b.jpg)

![a0f9b9cc63961f9edc53b98c0f29832e14eefc4196b17f9cd20ff175390569d4.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/tables/a0f9b9cc63961f9edc53b98c0f29832e14eefc4196b17f9cd20ff175390569d4.jpg)

![bb22f066485b25c7c04c93ad07448a434dcb1bc2ace195c2413222f357838906.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/tables/bb22f066485b25c7c04c93ad07448a434dcb1bc2ace195c2413222f357838906.jpg)

![c05ecc545de05b1eeac1b381a9621babc1cd42b69b4269c5b3eca2a2595478da.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/tables/c05ecc545de05b1eeac1b381a9621babc1cd42b69b4269c5b3eca2a2595478da.jpg)

![d30f6b02b871b6c76e3e770c94eafa78d579c4a689802bf2cf1c4eb20ed52373.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/tables/d30f6b02b871b6c76e3e770c94eafa78d579c4a689802bf2cf1c4eb20ed52373.jpg)

![de4546e2c42227fac24bec8afd7dbc9a04a79337e175c8cf1cb7a610acf33143.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/tables/de4546e2c42227fac24bec8afd7dbc9a04a79337e175c8cf1cb7a610acf33143.jpg)

![feeefe1bc79fc0381d3d2f89dbb6907e56b99b853535c2b29cca36766e14c559.jpg](../icml_results/2864_Learning%20from%20Comparison_%20Constrained%20Projection%20Policy%20Optimization%20for%20Pareto-Front%20Improvement/tables/feeefe1bc79fc0381d3d2f89dbb6907e56b99b853535c2b29cca36766e14c559.jpg)

<a id="paper-2827"></a>

## The Implicit Bias of Steepest Descent with Mini-batch Stochastic Gradient


### Images

![218f5ec4dd11da494bab80de7251b8d253398013b89429dab25a5b0411edd00c.jpg](../icml_results/2865_The%20Implicit%20Bias%20of%20Steepest%20Descent%20with%20Mini-batch%20Stochastic%20Gradient/images/218f5ec4dd11da494bab80de7251b8d253398013b89429dab25a5b0411edd00c.jpg)

![2e9d62078fa8ababaf9009fbc687116176feb67ccf379f318c88e90b6d9a4312.jpg](../icml_results/2865_The%20Implicit%20Bias%20of%20Steepest%20Descent%20with%20Mini-batch%20Stochastic%20Gradient/images/2e9d62078fa8ababaf9009fbc687116176feb67ccf379f318c88e90b6d9a4312.jpg)

![3f76ce6943692007ca7a13aac5575d949b51f36b21ef6bbad01adf19fcd8d4c3.jpg](../icml_results/2865_The%20Implicit%20Bias%20of%20Steepest%20Descent%20with%20Mini-batch%20Stochastic%20Gradient/images/3f76ce6943692007ca7a13aac5575d949b51f36b21ef6bbad01adf19fcd8d4c3.jpg)

![a883ad6df2e80891d4289e833bf9daa8e0265bfbe8c16871af2583cee246963c.jpg](../icml_results/2865_The%20Implicit%20Bias%20of%20Steepest%20Descent%20with%20Mini-batch%20Stochastic%20Gradient/images/a883ad6df2e80891d4289e833bf9daa8e0265bfbe8c16871af2583cee246963c.jpg)

![e276258df24f3318bd6aeef636f7a6b7c10696266be2522eb5433a67c7716d71.jpg](../icml_results/2865_The%20Implicit%20Bias%20of%20Steepest%20Descent%20with%20Mini-batch%20Stochastic%20Gradient/images/e276258df24f3318bd6aeef636f7a6b7c10696266be2522eb5433a67c7716d71.jpg)

<a id="paper-2828"></a>

## Compile to Compress: Boosting Formal Theorem Provers by Compiler Outputs


### Images

![168dbe1b7e0dbb9c1b74065833c1222fe47c324a171e4de8b38c468a13208a07.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/images/168dbe1b7e0dbb9c1b74065833c1222fe47c324a171e4de8b38c468a13208a07.jpg)

![58f3b123efdd0885185328614d8a063cfaa0f4b1e564d4dbd4f7ff311eadb268.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/images/58f3b123efdd0885185328614d8a063cfaa0f4b1e564d4dbd4f7ff311eadb268.jpg)

![8ed566fd9718c6d166ddad8f5c489d05e16a03aeb8a0e6ad9c03420bcd264dc4.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/images/8ed566fd9718c6d166ddad8f5c489d05e16a03aeb8a0e6ad9c03420bcd264dc4.jpg)

![a19590ed17b982cc78263b5073375b139f55522e12005acaaae9b590c3175c28.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/images/a19590ed17b982cc78263b5073375b139f55522e12005acaaae9b590c3175c28.jpg)

![c532114876736362fae7f3af254dd3994770d7952c6136115d328d69857ae968.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/images/c532114876736362fae7f3af254dd3994770d7952c6136115d328d69857ae968.jpg)

![d45f330f1821a6bde8ec869d91793e1e7ae85887ef52b8e0fb0c4b156ca3b987.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/images/d45f330f1821a6bde8ec869d91793e1e7ae85887ef52b8e0fb0c4b156ca3b987.jpg)

![d6f08f1e7577afd38a3f1610a2596dbd094d58f7ed88147f63938f3901cd96b8.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/images/d6f08f1e7577afd38a3f1610a2596dbd094d58f7ed88147f63938f3901cd96b8.jpg)

![dc2fa68db5c5f073142fd99e9f82af2bcaf50d7f85167753c435f9202705f2c0.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/images/dc2fa68db5c5f073142fd99e9f82af2bcaf50d7f85167753c435f9202705f2c0.jpg)

![e1e943f313dab97974a5e4f13a73952d8f78c4fd6abeaad3172078a126baa84d.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/images/e1e943f313dab97974a5e4f13a73952d8f78c4fd6abeaad3172078a126baa84d.jpg)

![e86e95c2d035fc80feaf4b0bc2717a6cb35eaa17d54e6fd2a2f4cdd1c150f48a.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/images/e86e95c2d035fc80feaf4b0bc2717a6cb35eaa17d54e6fd2a2f4cdd1c150f48a.jpg)

![eb8ca55a56fe046fefd49ce21ff47a77495bb6d184575ffa4d46d7d2c224f7a4.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/images/eb8ca55a56fe046fefd49ce21ff47a77495bb6d184575ffa4d46d7d2c224f7a4.jpg)

![f6995bc6b84b94a06b26f6cf7da226815b11742df3e67224fc4cab33d17be68d.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/images/f6995bc6b84b94a06b26f6cf7da226815b11742df3e67224fc4cab33d17be68d.jpg)

![fa2cfcc7e57ac1e01893533667e8bafeeca02e0f391d5cba18f1bc7fe9562b8a.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/images/fa2cfcc7e57ac1e01893533667e8bafeeca02e0f391d5cba18f1bc7fe9562b8a.jpg)

![fb97a8db71c2412f0176c3e6b657409db0fb6a7115d61dbfb805d4d3465f46e9.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/images/fb97a8db71c2412f0176c3e6b657409db0fb6a7115d61dbfb805d4d3465f46e9.jpg)

### Tables

![117492734a522a8cf7fe30c380b70259493c029da9bb9360d03e30e0f79cedad.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/tables/117492734a522a8cf7fe30c380b70259493c029da9bb9360d03e30e0f79cedad.jpg)

![32c78f7240e85c3ef7102bd9d75e900302f1e0cb42875654a93bf66d9168d9d7.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/tables/32c78f7240e85c3ef7102bd9d75e900302f1e0cb42875654a93bf66d9168d9d7.jpg)

![357f6f8991922d5eac3e29de36e7e1d927515090fa7b257a8f370a29746e86a3.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/tables/357f6f8991922d5eac3e29de36e7e1d927515090fa7b257a8f370a29746e86a3.jpg)

![50e8bc67b680c2ccaf08bbf6a50f0392f4474098f778893b62e4c237c15cd824.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/tables/50e8bc67b680c2ccaf08bbf6a50f0392f4474098f778893b62e4c237c15cd824.jpg)

![5211397014e38404fb8a6e6d20445e973177ce9c47174ba6e9b833f3dc4c7d55.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/tables/5211397014e38404fb8a6e6d20445e973177ce9c47174ba6e9b833f3dc4c7d55.jpg)

![8100473e2ff6a89d7c5f31fd805852ecd843bce99dded77b5414b256edb7b9b1.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/tables/8100473e2ff6a89d7c5f31fd805852ecd843bce99dded77b5414b256edb7b9b1.jpg)

![834d5c473167c460524a0d66a192c3d011c590de27ba4abbba32cf3dad7c34b7.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/tables/834d5c473167c460524a0d66a192c3d011c590de27ba4abbba32cf3dad7c34b7.jpg)

![84cc3066b26edc0ca134dcf97a7c4eddb6a8084e57b88ea826b3901ec47c2ad3.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/tables/84cc3066b26edc0ca134dcf97a7c4eddb6a8084e57b88ea826b3901ec47c2ad3.jpg)

![a7a9f5a27c4871fbc98e00238f4e7dd50b60fc9ad699f8332837fac38c85740a.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/tables/a7a9f5a27c4871fbc98e00238f4e7dd50b60fc9ad699f8332837fac38c85740a.jpg)

![b3bb46cec096fbabc2db1a1461680a5fde6f5317e9a33301c75126e481d3ddd6.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/tables/b3bb46cec096fbabc2db1a1461680a5fde6f5317e9a33301c75126e481d3ddd6.jpg)

![c26435b489892b186e8bb1f60e8ac1dfb7c76f152b510bf1fc7008ed8658752b.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/tables/c26435b489892b186e8bb1f60e8ac1dfb7c76f152b510bf1fc7008ed8658752b.jpg)

![d20243ac8396831af4f0094e69062797bb9d0ccebd17439371305da2f60d23c8.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/tables/d20243ac8396831af4f0094e69062797bb9d0ccebd17439371305da2f60d23c8.jpg)

![ef8500971a1fade49de4c90320391226b279415ad181e66530fb65152b3d8e08.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/tables/ef8500971a1fade49de4c90320391226b279415ad181e66530fb65152b3d8e08.jpg)

![f594037902bdb181db5820c86764c9e66aae54d10ee627af2b21a4a8506af8b6.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/tables/f594037902bdb181db5820c86764c9e66aae54d10ee627af2b21a4a8506af8b6.jpg)

![f9caaae4ee1e52f42320dd0c6e1529f999e2d5ec8780d5664f2d0fda0caf141d.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/tables/f9caaae4ee1e52f42320dd0c6e1529f999e2d5ec8780d5664f2d0fda0caf141d.jpg)

![fb00062030ec2dc9a239d15b4cf08aee5c0cf1c4ad4f95a59d3c105ce8ee9a61.jpg](../icml_results/2866_Compile%20to%20Compress_%20Boosting%20Formal%20Theorem%20Provers%20by%20Compiler%20Outputs/tables/fb00062030ec2dc9a239d15b4cf08aee5c0cf1c4ad4f95a59d3c105ce8ee9a61.jpg)

<a id="paper-2829"></a>

## Data- and Variance-dependent Regret Bounds for Online Tabular MDPs


### Tables

![060f83915f5c637a5e6216d476849be62b6d61734106e2353e382dce8042e0dd.jpg](../icml_results/2867_Data-%20and%20Variance-dependent%20Regret%20Bounds%20for%20Online%20Tabular%20MDPs/tables/060f83915f5c637a5e6216d476849be62b6d61734106e2353e382dce8042e0dd.jpg)

![167b957e420414b870005588553f3f5d2c1d439855cf996dbaa4e112d3e30b7c.jpg](../icml_results/2867_Data-%20and%20Variance-dependent%20Regret%20Bounds%20for%20Online%20Tabular%20MDPs/tables/167b957e420414b870005588553f3f5d2c1d439855cf996dbaa4e112d3e30b7c.jpg)

![ab4d60ef3eb989e9ebd0e6eb8976df97604286259fc1fcb5a9dafedb97a99bc2.jpg](../icml_results/2867_Data-%20and%20Variance-dependent%20Regret%20Bounds%20for%20Online%20Tabular%20MDPs/tables/ab4d60ef3eb989e9ebd0e6eb8976df97604286259fc1fcb5a9dafedb97a99bc2.jpg)

![c0280d5a401b122b7d4f9f7a4064be32cae763a7cd629ed2c1fbdfe774b294cd.jpg](../icml_results/2867_Data-%20and%20Variance-dependent%20Regret%20Bounds%20for%20Online%20Tabular%20MDPs/tables/c0280d5a401b122b7d4f9f7a4064be32cae763a7cd629ed2c1fbdfe774b294cd.jpg)

<a id="paper-2830"></a>

## Causes and Consequences of Representational Similarity in Machine Learning Models


### Images

![029b50b08247414cbde6a20fc6d90b7fd4d77244e9f3a41cfec614fc2512feb4.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/029b50b08247414cbde6a20fc6d90b7fd4d77244e9f3a41cfec614fc2512feb4.jpg)

![0d73412784ca316f5e9e6f85605b55e644ac2bff889612968597f8fb8fa20a8d.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/0d73412784ca316f5e9e6f85605b55e644ac2bff889612968597f8fb8fa20a8d.jpg)

![1005be396e6686fc120525f2fc4eba87b5c6f517110efa9e46cccf14cf7b3fba.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/1005be396e6686fc120525f2fc4eba87b5c6f517110efa9e46cccf14cf7b3fba.jpg)

![1da592180fe958558153b95dfd308a88a0e17d1356a5e5155e0203283398ace6.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/1da592180fe958558153b95dfd308a88a0e17d1356a5e5155e0203283398ace6.jpg)

![2fde96b19b80eafa68acf0b0ba31eb92a5c2ea1bbee121833919298ef23d3a37.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/2fde96b19b80eafa68acf0b0ba31eb92a5c2ea1bbee121833919298ef23d3a37.jpg)

![3f5b60342e5e12b824c60515d57ca1ecbbe83d286f26521a9cf3806237ee3e6a.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/3f5b60342e5e12b824c60515d57ca1ecbbe83d286f26521a9cf3806237ee3e6a.jpg)

![436541411ffcc7491363bf2221c6894396bdc9778c2c54c7ceab648c19bdcdbe.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/436541411ffcc7491363bf2221c6894396bdc9778c2c54c7ceab648c19bdcdbe.jpg)

![56f6852662e91a3802872a5cf07fa99943996666f9c927353a3bc1044910fbbd.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/56f6852662e91a3802872a5cf07fa99943996666f9c927353a3bc1044910fbbd.jpg)

![59b52e96a345cc99b4ef97e9405e1d4ebeee32895654903204a04b8430b55405.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/59b52e96a345cc99b4ef97e9405e1d4ebeee32895654903204a04b8430b55405.jpg)

![5b7338f9b192948dcde562fe6e516a4685d37eabd4b01e88b361a48b2515cc78.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/5b7338f9b192948dcde562fe6e516a4685d37eabd4b01e88b361a48b2515cc78.jpg)

![665f55c14164758440b7cb44f7d0fac060ed81a2ff629866677f22acbcb6ce9b.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/665f55c14164758440b7cb44f7d0fac060ed81a2ff629866677f22acbcb6ce9b.jpg)

![66fc78bf6f149c2f5b52c26dbebcc63e7203204565bfe6e17fa824c24b366b73.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/66fc78bf6f149c2f5b52c26dbebcc63e7203204565bfe6e17fa824c24b366b73.jpg)

![6b4e245f6c8a67d2926dd097bd98334b5fb77b6671c4234326e1e751f8adecb2.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/6b4e245f6c8a67d2926dd097bd98334b5fb77b6671c4234326e1e751f8adecb2.jpg)

![6b827d1440a7fd08977de44c7212671428761ca02792497cbac3d46db1a381e3.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/6b827d1440a7fd08977de44c7212671428761ca02792497cbac3d46db1a381e3.jpg)

![7853a333e427bec1efd523496c28d6725dc0294f9aaf2c615bebfa63747e491a.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/7853a333e427bec1efd523496c28d6725dc0294f9aaf2c615bebfa63747e491a.jpg)

![7abde2ceb84667b01c9555416a68a7ebe53b2c151ab917c87f33f50121558ea3.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/7abde2ceb84667b01c9555416a68a7ebe53b2c151ab917c87f33f50121558ea3.jpg)

![806affcff83005addd451db371e2603798923626d5075237aa233681dcdbfc1b.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/806affcff83005addd451db371e2603798923626d5075237aa233681dcdbfc1b.jpg)

![896bbfe0f2ff705a5fc491c03bb71e90aa88ac9f6396e41203ad61dbf647fb80.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/896bbfe0f2ff705a5fc491c03bb71e90aa88ac9f6396e41203ad61dbf647fb80.jpg)

![8f50ef04f34e8c4fd3619331ebb447a8c90144c373b6abf08f21aad8e91d4e9d.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/8f50ef04f34e8c4fd3619331ebb447a8c90144c373b6abf08f21aad8e91d4e9d.jpg)

![90e5e96af1956785e8e55177a495444032a93c655d4e5a5eab14d689b9fa1f93.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/90e5e96af1956785e8e55177a495444032a93c655d4e5a5eab14d689b9fa1f93.jpg)

![94da30b51c343f13b889392d89213fa8570460607478d984925e4321e354f268.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/94da30b51c343f13b889392d89213fa8570460607478d984925e4321e354f268.jpg)

![9c03cfa8412656754e5dff07c2fb665645d0f986df25a62d0668373222817b17.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/9c03cfa8412656754e5dff07c2fb665645d0f986df25a62d0668373222817b17.jpg)

![a0d41d320dca75c27bc655a3cbc182e18f696e905474e13c61ac639e44da21c8.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/a0d41d320dca75c27bc655a3cbc182e18f696e905474e13c61ac639e44da21c8.jpg)

![a9e44e755b6b0db1fb4e11b2ac308b0da221b30bb41c2036faddfa8209ed1b95.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/a9e44e755b6b0db1fb4e11b2ac308b0da221b30bb41c2036faddfa8209ed1b95.jpg)

![bbba44554187da69f888f10727d00812e08ec73e5477a1381165aa3308fa406d.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/bbba44554187da69f888f10727d00812e08ec73e5477a1381165aa3308fa406d.jpg)

![bfad8a2b453b9b22fe23f43989276286c49cd0207e9337242f90783404b7e46d.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/bfad8a2b453b9b22fe23f43989276286c49cd0207e9337242f90783404b7e46d.jpg)

![c5cb0b067382e2a9ec94de9aecbabf8a3ee8be47e5082b4ab3f6928401d0d32b.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/c5cb0b067382e2a9ec94de9aecbabf8a3ee8be47e5082b4ab3f6928401d0d32b.jpg)

![caf64d6a8ced7e830f1ec4ac4ce423c74acafe46d735143ccc7f05a00e04d5a1.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/caf64d6a8ced7e830f1ec4ac4ce423c74acafe46d735143ccc7f05a00e04d5a1.jpg)

![cdc2c7b6d995fe4e00490681c1f5ba83160a4db8da0df69061a2e06f77fd1d01.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/cdc2c7b6d995fe4e00490681c1f5ba83160a4db8da0df69061a2e06f77fd1d01.jpg)

![e2d1de4af1e6a643eb4aa2a071bd2cef7bf773a771ba612b550fed569ab2f1c0.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/e2d1de4af1e6a643eb4aa2a071bd2cef7bf773a771ba612b550fed569ab2f1c0.jpg)

![edbfa81c743a9739141cdff34a9e4c0e21b3e636392a035a8c17250314ae50c4.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/images/edbfa81c743a9739141cdff34a9e4c0e21b3e636392a035a8c17250314ae50c4.jpg)

### Tables

![07c9b2db6593c6d080c6fa7c2969f043d8d669bfd983f0c52e239393191a6761.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/07c9b2db6593c6d080c6fa7c2969f043d8d669bfd983f0c52e239393191a6761.jpg)

![0c0b8d647e64444da54ffd2adbbcedeb1e3b0b912ddd1d0e2d2ceb97cd97a2d2.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/0c0b8d647e64444da54ffd2adbbcedeb1e3b0b912ddd1d0e2d2ceb97cd97a2d2.jpg)

![1e7ca5eb4571789dda1a18df7a083e46cf634760004e4c8c56df8a53cd9605ee.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/1e7ca5eb4571789dda1a18df7a083e46cf634760004e4c8c56df8a53cd9605ee.jpg)

![39c8001f6c4f38cef6cb75ad19e64f4e8b574ca6a89d1bc938baf312f3d009d4.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/39c8001f6c4f38cef6cb75ad19e64f4e8b574ca6a89d1bc938baf312f3d009d4.jpg)

![3b1721cb2772d4a44e192a5ca7705f4641449e1cc2ceb318dfa58fe462c0510b.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/3b1721cb2772d4a44e192a5ca7705f4641449e1cc2ceb318dfa58fe462c0510b.jpg)

![46f22d8c08d028b58e768afde78c4be3de272342281fcf715884dd8abb44a4be.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/46f22d8c08d028b58e768afde78c4be3de272342281fcf715884dd8abb44a4be.jpg)

![4a9b92d8db8b54b0b0c7d0ad2aac1a29c59ceafd67b88e8e0f2548b8546449a5.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/4a9b92d8db8b54b0b0c7d0ad2aac1a29c59ceafd67b88e8e0f2548b8546449a5.jpg)

![552acb3bdb4233d7520692729dd4bcd297f98def396058e7c67991bbbcc69043.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/552acb3bdb4233d7520692729dd4bcd297f98def396058e7c67991bbbcc69043.jpg)

![5b3d7d56dad6846d905d56a3c4571fee320bae705d7a65df4e1662bd19086837.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/5b3d7d56dad6846d905d56a3c4571fee320bae705d7a65df4e1662bd19086837.jpg)

![63e05c9d0bba46e821878469a928d919b748ae13fbcdd9892f812bb8abffe73f.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/63e05c9d0bba46e821878469a928d919b748ae13fbcdd9892f812bb8abffe73f.jpg)

![696f8cf080c84e40d72365d866c210843d5504cf2c2cbcd90dd74b92ac8e44ef.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/696f8cf080c84e40d72365d866c210843d5504cf2c2cbcd90dd74b92ac8e44ef.jpg)

![8ff34c3c1ed0e3e1db2ae48f8e2d3edbc81c420f72a343d9564b7f19c4bf0af9.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/8ff34c3c1ed0e3e1db2ae48f8e2d3edbc81c420f72a343d9564b7f19c4bf0af9.jpg)

![9abd1339ed17e7a31572467504d0acf5ea78d3b92eba1d426cbf2117ea146070.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/9abd1339ed17e7a31572467504d0acf5ea78d3b92eba1d426cbf2117ea146070.jpg)

![9bd0cbe154be7bbeacfe24cef7f988fbc7dccfbd0098da1ef852d9450db5dddf.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/9bd0cbe154be7bbeacfe24cef7f988fbc7dccfbd0098da1ef852d9450db5dddf.jpg)

![afae05f6de699f83bace717dc3d26b1f6e7ea66bfbd5e7ac64843fc92dcc0b13.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/afae05f6de699f83bace717dc3d26b1f6e7ea66bfbd5e7ac64843fc92dcc0b13.jpg)

![b3d8f887c84cd6f2901901597b223e314b6326cb5a933216b1d70f65b50fd131.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/b3d8f887c84cd6f2901901597b223e314b6326cb5a933216b1d70f65b50fd131.jpg)

![b45e3067df54ddfbeec428b98e007290acedeccc716ee1ba65130bf9b3ec9c8d.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/b45e3067df54ddfbeec428b98e007290acedeccc716ee1ba65130bf9b3ec9c8d.jpg)

![b60953ced61285261add82c002ffd2a1e362d90ea378b9e815ae2bce0f9e9d84.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/b60953ced61285261add82c002ffd2a1e362d90ea378b9e815ae2bce0f9e9d84.jpg)

![c8aa7a10e1df9286cc701868a449329238764c881ce50596674cd84553ccf557.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/c8aa7a10e1df9286cc701868a449329238764c881ce50596674cd84553ccf557.jpg)

![d831ad1b5e83e43dcb47f7d4d976b63f04253bcdbac36a2202416ab684c13841.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/d831ad1b5e83e43dcb47f7d4d976b63f04253bcdbac36a2202416ab684c13841.jpg)

![db539c32e2b0bfc8d08dfff023ea22334ff9d955eba3469bd076ce3131113138.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/db539c32e2b0bfc8d08dfff023ea22334ff9d955eba3469bd076ce3131113138.jpg)

![ebe6ec12a03eca93f4e77c43176d53a1bd6b1bc9904790d111284be1fc9b203c.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/ebe6ec12a03eca93f4e77c43176d53a1bd6b1bc9904790d111284be1fc9b203c.jpg)

![f33732f260447bd6d514d1052c794301a6bec8b24a16075efbbbd8f6e55869d3.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/f33732f260447bd6d514d1052c794301a6bec8b24a16075efbbbd8f6e55869d3.jpg)

![f6ba8709c9cec3dfcba4809f30a981921b7af0986d09740620ef8285dbf35358.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/f6ba8709c9cec3dfcba4809f30a981921b7af0986d09740620ef8285dbf35358.jpg)

![f9aa37940ba6a1887ddc7cac109018dbfda04ee3edda590f793e7a6753905064.jpg](../icml_results/2868_Causes%20and%20Consequences%20of%20Representational%20Similarity%20in%20Machine%20Learning%20Models/tables/f9aa37940ba6a1887ddc7cac109018dbfda04ee3edda590f793e7a6753905064.jpg)

<a id="paper-2831"></a>

## Large Language Models as Topological Thinkers: A Benchmark on Graph Persistent Homology


### Images

![6c5b77ee702c4e7b9a4622d08a66372ce8a8d3d2607c009e5a6e5cf35b99062f.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/images/6c5b77ee702c4e7b9a4622d08a66372ce8a8d3d2607c009e5a6e5cf35b99062f.jpg)

![ea15e6317dac3f42114a4b2b1cff51d50c5dad7f042a24d03c236e40847fc7df.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/images/ea15e6317dac3f42114a4b2b1cff51d50c5dad7f042a24d03c236e40847fc7df.jpg)

### Tables

![35d4bcab34060d9ed416f47b8527e4d87a7eb9529383fa1f6e114bd9a2b2d678.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/35d4bcab34060d9ed416f47b8527e4d87a7eb9529383fa1f6e114bd9a2b2d678.jpg)

![51cdd839ff30b9f67095c12181fa1cdff892a1ae0e8e93d6401737e0705d3536.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/51cdd839ff30b9f67095c12181fa1cdff892a1ae0e8e93d6401737e0705d3536.jpg)

![520328f4cda5f4a8042cc0678f55cb1df70cf7df5e0575805cedd42e766e954a.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/520328f4cda5f4a8042cc0678f55cb1df70cf7df5e0575805cedd42e766e954a.jpg)

![584624c40d3dd7addaa22d08e2486db80c8dc0fa3cc16b88c7566b41dca872f5.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/584624c40d3dd7addaa22d08e2486db80c8dc0fa3cc16b88c7566b41dca872f5.jpg)

![5ee016fbc1cbbc09663f353c066b862a11ed0d57151482f7a2ec855afc29c0b8.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/5ee016fbc1cbbc09663f353c066b862a11ed0d57151482f7a2ec855afc29c0b8.jpg)

![612f70a21526725558c8f7c4ef6c23920bbf599b39876184241019dd5cdb0c54.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/612f70a21526725558c8f7c4ef6c23920bbf599b39876184241019dd5cdb0c54.jpg)

![638a08dca907044cdf8be433442a4883827471e012bfdb596ae7ac42cb8f7b9d.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/638a08dca907044cdf8be433442a4883827471e012bfdb596ae7ac42cb8f7b9d.jpg)

![8ded02ec37d1f8738d40f563c19dbd90e6d5aeacb5b57ede270cad0de52c9d00.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/8ded02ec37d1f8738d40f563c19dbd90e6d5aeacb5b57ede270cad0de52c9d00.jpg)

![8e846c89c7043fd6753f3ddf4ceb2610f2e8b18a3c3d99cc42d57c5aed355b5e.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/8e846c89c7043fd6753f3ddf4ceb2610f2e8b18a3c3d99cc42d57c5aed355b5e.jpg)

![bc9e2d7cb807dc3fe8fc66227ff6c785041b6d99abc644a6ef87d9ad1c4ff209.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/bc9e2d7cb807dc3fe8fc66227ff6c785041b6d99abc644a6ef87d9ad1c4ff209.jpg)

![d89c39b7f9ff4de8e282b1635a945eb34a887cac0a2de8578b103c03fc1d5f93.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/d89c39b7f9ff4de8e282b1635a945eb34a887cac0a2de8578b103c03fc1d5f93.jpg)

![e33eb4ab93ab6d186008d9924013a20085c7eb89b2107b6098345f6317552a15.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/e33eb4ab93ab6d186008d9924013a20085c7eb89b2107b6098345f6317552a15.jpg)

![e3bdb8736cf1c79cd54ce33b26c0c9c612aa342dba7c152ac4d7e10967da45c6.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/e3bdb8736cf1c79cd54ce33b26c0c9c612aa342dba7c152ac4d7e10967da45c6.jpg)

![ee07b9705c594d4b099fb614eacf87462dc97b0544e6de549c99bba4cec3a31b.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/ee07b9705c594d4b099fb614eacf87462dc97b0544e6de549c99bba4cec3a31b.jpg)

![ef01ea25aad0053cfe5e70cebed78c5df1ad45fd575cce56aa9661da8706fc75.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/ef01ea25aad0053cfe5e70cebed78c5df1ad45fd575cce56aa9661da8706fc75.jpg)

![f36666acfa978b075f33f855214b77b104497d4c712854d28b9501eae0f230e5.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/f36666acfa978b075f33f855214b77b104497d4c712854d28b9501eae0f230e5.jpg)

![f505bdc0b6f33be84614963a2db659299756dafe4b68a40c7b3affc6105a6294.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/f505bdc0b6f33be84614963a2db659299756dafe4b68a40c7b3affc6105a6294.jpg)

![f5271e7474d9ba32ed53469a5d2cffb9a1b6c2fbec00feec08cce32dae5f0643.jpg](../icml_results/2869_Large%20Language%20Models%20as%20Topological%20Thinkers_%20A%20Benchmark%20on%20Graph%20Persistent%20Homology/tables/f5271e7474d9ba32ed53469a5d2cffb9a1b6c2fbec00feec08cce32dae5f0643.jpg)

<a id="paper-2832"></a>

## ARC-Decode: Accelerated Decoding with Risk-Bounded Acceptance


### Images

![184e65ee775cdff7a531783695959eac1c1265b44c5dd23630196cec7b372692.jpg](../icml_results/2870_ARC-Decode_%20Accelerated%20Decoding%20with%20Risk-Bounded%20Acceptance/images/184e65ee775cdff7a531783695959eac1c1265b44c5dd23630196cec7b372692.jpg)

![1b02ff126d2a05301e366648463f4fadc3c92e07976b88d4e72df1acb84b4f18.jpg](../icml_results/2870_ARC-Decode_%20Accelerated%20Decoding%20with%20Risk-Bounded%20Acceptance/images/1b02ff126d2a05301e366648463f4fadc3c92e07976b88d4e72df1acb84b4f18.jpg)

![4dca850fd8ac75d4f468d9feb94a7adfee5c2ff9c708a6defd5b6de52bf8fb3c.jpg](../icml_results/2870_ARC-Decode_%20Accelerated%20Decoding%20with%20Risk-Bounded%20Acceptance/images/4dca850fd8ac75d4f468d9feb94a7adfee5c2ff9c708a6defd5b6de52bf8fb3c.jpg)

![a17fbde14e2ca92910f354fd2a6d0f03811637a8b7fed77dac843784d77ab47d.jpg](../icml_results/2870_ARC-Decode_%20Accelerated%20Decoding%20with%20Risk-Bounded%20Acceptance/images/a17fbde14e2ca92910f354fd2a6d0f03811637a8b7fed77dac843784d77ab47d.jpg)

![bdab652a44b22a097c2e09f2452986eefdb0a2035983bffd59befff1a1d0e38e.jpg](../icml_results/2870_ARC-Decode_%20Accelerated%20Decoding%20with%20Risk-Bounded%20Acceptance/images/bdab652a44b22a097c2e09f2452986eefdb0a2035983bffd59befff1a1d0e38e.jpg)

### Tables

![07b24164a68140aad998eac301962b869f17cae503b6883ae1c812d03b2a6bb9.jpg](../icml_results/2870_ARC-Decode_%20Accelerated%20Decoding%20with%20Risk-Bounded%20Acceptance/tables/07b24164a68140aad998eac301962b869f17cae503b6883ae1c812d03b2a6bb9.jpg)

![1f1a80a4b5f0619aff1638806bc0a7b7fea27314db923b26c5578270d36bbdd7.jpg](../icml_results/2870_ARC-Decode_%20Accelerated%20Decoding%20with%20Risk-Bounded%20Acceptance/tables/1f1a80a4b5f0619aff1638806bc0a7b7fea27314db923b26c5578270d36bbdd7.jpg)

![56241e8da7a1c1a9da913ed33d4774fc31eceae92f9d114ee3a562425bbed3b6.jpg](../icml_results/2870_ARC-Decode_%20Accelerated%20Decoding%20with%20Risk-Bounded%20Acceptance/tables/56241e8da7a1c1a9da913ed33d4774fc31eceae92f9d114ee3a562425bbed3b6.jpg)

![581242adec1bf362f366c1dfe0f4c36b2e3b86efa45e394c1d110ec52cb603e2.jpg](../icml_results/2870_ARC-Decode_%20Accelerated%20Decoding%20with%20Risk-Bounded%20Acceptance/tables/581242adec1bf362f366c1dfe0f4c36b2e3b86efa45e394c1d110ec52cb603e2.jpg)

![7403192db84375e8fd4f9aa548f289780e5b2192910547836d27c09de69034f9.jpg](../icml_results/2870_ARC-Decode_%20Accelerated%20Decoding%20with%20Risk-Bounded%20Acceptance/tables/7403192db84375e8fd4f9aa548f289780e5b2192910547836d27c09de69034f9.jpg)

![a6c2d13829d09b85354f93244511ac082a52a22bd678423da537bb617eb6f92e.jpg](../icml_results/2870_ARC-Decode_%20Accelerated%20Decoding%20with%20Risk-Bounded%20Acceptance/tables/a6c2d13829d09b85354f93244511ac082a52a22bd678423da537bb617eb6f92e.jpg)

![bbd3df3abbfffbfa2d7de69d5136a57ef13ef37f77de6a4dc591d612277aa060.jpg](../icml_results/2870_ARC-Decode_%20Accelerated%20Decoding%20with%20Risk-Bounded%20Acceptance/tables/bbd3df3abbfffbfa2d7de69d5136a57ef13ef37f77de6a4dc591d612277aa060.jpg)

![db0f9e39d8e428453098b1ab6fb345e937ef3d6860df40f99c22b8612d66751b.jpg](../icml_results/2870_ARC-Decode_%20Accelerated%20Decoding%20with%20Risk-Bounded%20Acceptance/tables/db0f9e39d8e428453098b1ab6fb345e937ef3d6860df40f99c22b8612d66751b.jpg)

<a id="paper-2833"></a>

## Anchor-guided Hypergraph Condensation with Dual-level Discrimination


### Images

![01dfc615f055979670c559f4e029375316ea8db5ab281734202b751292dcd65e.jpg](../icml_results/2871_Anchor-guided%20Hypergraph%20Condensation%20with%20Dual-level%20Discrimination/images/01dfc615f055979670c559f4e029375316ea8db5ab281734202b751292dcd65e.jpg)

![93f91ba2260d137ca56a7ea486de869ad9bff723e4c38ee3c8a8934169a8339b.jpg](../icml_results/2871_Anchor-guided%20Hypergraph%20Condensation%20with%20Dual-level%20Discrimination/images/93f91ba2260d137ca56a7ea486de869ad9bff723e4c38ee3c8a8934169a8339b.jpg)

![db879004ae8d7a4c31a8bdf42527a2b65b87d8ac0604ac889e428855686fd0cc.jpg](../icml_results/2871_Anchor-guided%20Hypergraph%20Condensation%20with%20Dual-level%20Discrimination/images/db879004ae8d7a4c31a8bdf42527a2b65b87d8ac0604ac889e428855686fd0cc.jpg)

![e200093099af4a71c5be4e563b6e44ec6fd4f5d515a2ed597e2d320cc13c9468.jpg](../icml_results/2871_Anchor-guided%20Hypergraph%20Condensation%20with%20Dual-level%20Discrimination/images/e200093099af4a71c5be4e563b6e44ec6fd4f5d515a2ed597e2d320cc13c9468.jpg)

### Tables

![13d57cb9c791a929b392973c52e9d24eade21f89847ff6aecb3f6fe640cd3649.jpg](../icml_results/2871_Anchor-guided%20Hypergraph%20Condensation%20with%20Dual-level%20Discrimination/tables/13d57cb9c791a929b392973c52e9d24eade21f89847ff6aecb3f6fe640cd3649.jpg)

![35f7bbcb2ffdb9966c2da061295a02ef3edd87a09b4b01b3d9582438af4938b6.jpg](../icml_results/2871_Anchor-guided%20Hypergraph%20Condensation%20with%20Dual-level%20Discrimination/tables/35f7bbcb2ffdb9966c2da061295a02ef3edd87a09b4b01b3d9582438af4938b6.jpg)

![43eee65b5f1102db1c0ecc1bb1cc436dd11e615226ff20c8cdd4f1d6c331d485.jpg](../icml_results/2871_Anchor-guided%20Hypergraph%20Condensation%20with%20Dual-level%20Discrimination/tables/43eee65b5f1102db1c0ecc1bb1cc436dd11e615226ff20c8cdd4f1d6c331d485.jpg)

![5d6aeca317fbfa942985f54a71126aa4c33ed8656664c7abc190440285065474.jpg](../icml_results/2871_Anchor-guided%20Hypergraph%20Condensation%20with%20Dual-level%20Discrimination/tables/5d6aeca317fbfa942985f54a71126aa4c33ed8656664c7abc190440285065474.jpg)

![6b2655ce9539a581088d39e8fa003bdbb9e710dbf7534102588cff1a107d9ccf.jpg](../icml_results/2871_Anchor-guided%20Hypergraph%20Condensation%20with%20Dual-level%20Discrimination/tables/6b2655ce9539a581088d39e8fa003bdbb9e710dbf7534102588cff1a107d9ccf.jpg)

![780dcfee7a847da4fee4a7cc7b96155475d4cdd45f4cac8a1f6de0a143bda4b7.jpg](../icml_results/2871_Anchor-guided%20Hypergraph%20Condensation%20with%20Dual-level%20Discrimination/tables/780dcfee7a847da4fee4a7cc7b96155475d4cdd45f4cac8a1f6de0a143bda4b7.jpg)

![df62976eeeb33369faf4ac5e2f7ca63d166fa2c8c03dd5dbc317f57bcb2d9fe7.jpg](../icml_results/2871_Anchor-guided%20Hypergraph%20Condensation%20with%20Dual-level%20Discrimination/tables/df62976eeeb33369faf4ac5e2f7ca63d166fa2c8c03dd5dbc317f57bcb2d9fe7.jpg)

![f992ad884e9d87bbe5344ec7490445d3cc9a22d1eb76fa7c6bf9b12b954a683c.jpg](../icml_results/2871_Anchor-guided%20Hypergraph%20Condensation%20with%20Dual-level%20Discrimination/tables/f992ad884e9d87bbe5344ec7490445d3cc9a22d1eb76fa7c6bf9b12b954a683c.jpg)

![fb992195c2a7cbaa92c9cf7bfa7b208aac09f590a44d9fec112add15d791dc5c.jpg](../icml_results/2871_Anchor-guided%20Hypergraph%20Condensation%20with%20Dual-level%20Discrimination/tables/fb992195c2a7cbaa92c9cf7bfa7b208aac09f590a44d9fec112add15d791dc5c.jpg)

<a id="paper-2834"></a>

## Self-Prompting Diffusion Transformer for Open-Vocabulary Scene Text Editing via In-Context Learning


### Images

![0883fea02ab59c4ee87df27b7f0e6822d57502def7bcdcf2ea990e59f4d5bb3d.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/images/0883fea02ab59c4ee87df27b7f0e6822d57502def7bcdcf2ea990e59f4d5bb3d.jpg)

![17f9c4d9bb734d14fd322cbe62904a4d32d7eebf71f036de5871e11d35250087.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/images/17f9c4d9bb734d14fd322cbe62904a4d32d7eebf71f036de5871e11d35250087.jpg)

![1b3e1c8b3b8f520a5cbc36217c5dbb7606f2071755c7c59f5cb895f0d0de8167.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/images/1b3e1c8b3b8f520a5cbc36217c5dbb7606f2071755c7c59f5cb895f0d0de8167.jpg)

![24f98c562488072772711a882c69ff8865df8ece483f31f3e0e473b0da5553a3.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/images/24f98c562488072772711a882c69ff8865df8ece483f31f3e0e473b0da5553a3.jpg)

![3727a5c251269f13b1f21fc25ab305b71812363f1684aba1d732359a613dcfec.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/images/3727a5c251269f13b1f21fc25ab305b71812363f1684aba1d732359a613dcfec.jpg)

![4892d3c562b25d4a3bba88c949c5c9fa42339b0c5996cb52083cbabde4fc283a.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/images/4892d3c562b25d4a3bba88c949c5c9fa42339b0c5996cb52083cbabde4fc283a.jpg)

![5b6b3df1049028a1a993da98151bf77a6011ab48ad08ad279a2974c2ac62e48c.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/images/5b6b3df1049028a1a993da98151bf77a6011ab48ad08ad279a2974c2ac62e48c.jpg)

![62a20b2bb6ce71cde6d3be22119c5980153a6a9ddb4f1d4ef5e7422673902a12.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/images/62a20b2bb6ce71cde6d3be22119c5980153a6a9ddb4f1d4ef5e7422673902a12.jpg)

![8af6c9cd3012e116db0273b9931f9a7f393af8563db0fc8a19b4fa28223b8a8f.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/images/8af6c9cd3012e116db0273b9931f9a7f393af8563db0fc8a19b4fa28223b8a8f.jpg)

![9ffed2a31b171cfc6a695fa37d11f72d1b0d767e6d462f0ee0bba9d7793e418f.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/images/9ffed2a31b171cfc6a695fa37d11f72d1b0d767e6d462f0ee0bba9d7793e418f.jpg)

![a0b02d81f80f159dfbe875c43e4568d3d65c1da0d7dd287155be59c73d499bb8.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/images/a0b02d81f80f159dfbe875c43e4568d3d65c1da0d7dd287155be59c73d499bb8.jpg)

![a8e09d3ad073fe114f299b7f1d01fbf28968634bfff9706fb3ac40a0b9b73854.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/images/a8e09d3ad073fe114f299b7f1d01fbf28968634bfff9706fb3ac40a0b9b73854.jpg)

![b2f038fc99a6e53dca1a4dd8f19d3e676cb7dcb2163306b42ec44fa3cbbc2172.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/images/b2f038fc99a6e53dca1a4dd8f19d3e676cb7dcb2163306b42ec44fa3cbbc2172.jpg)

![da7a30d24d658568fdb930f224bd97c0cb16602e038b48484dd7852ad7961cdb.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/images/da7a30d24d658568fdb930f224bd97c0cb16602e038b48484dd7852ad7961cdb.jpg)

![daad54430e6ca0f2d2750b6ad196f5e5643edd14899977a8673413c81ec067f7.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/images/daad54430e6ca0f2d2750b6ad196f5e5643edd14899977a8673413c81ec067f7.jpg)

![fc70b64731b9b31d40972ca9eb87b708a92fe716a73d7a6b544ff8384b7e95bd.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/images/fc70b64731b9b31d40972ca9eb87b708a92fe716a73d7a6b544ff8384b7e95bd.jpg)

### Tables

![21ccade15a4622e4f450b7b51d20b339c7a587186ae570f966f59695e0df8481.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/tables/21ccade15a4622e4f450b7b51d20b339c7a587186ae570f966f59695e0df8481.jpg)

![6838c788a4791241a84207b590e5a4018b5d5903e36264280655eff3c3aca096.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/tables/6838c788a4791241a84207b590e5a4018b5d5903e36264280655eff3c3aca096.jpg)

![83475c454f1958c1936ec29b9cdbee5d550a00f336f779beea12efce54860673.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/tables/83475c454f1958c1936ec29b9cdbee5d550a00f336f779beea12efce54860673.jpg)

![9dbdc1d3b0a5bfe119cf7959d700bcc17e0dd2287f6c3e257644c09f357a51d9.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/tables/9dbdc1d3b0a5bfe119cf7959d700bcc17e0dd2287f6c3e257644c09f357a51d9.jpg)

![b743e58e5ad5bf2f7fed1081e1f7394038466aa6350de09fccc8f597ac88a50f.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/tables/b743e58e5ad5bf2f7fed1081e1f7394038466aa6350de09fccc8f597ac88a50f.jpg)

![c2644c483e2d931fd5354d2b465c22cfd775e787da205dc346adc32ab2c53aad.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/tables/c2644c483e2d931fd5354d2b465c22cfd775e787da205dc346adc32ab2c53aad.jpg)

![d34cafc312f4c7d0c36f4e2b3c9b57fe74e9aaded0140ec626cab1eeadcd93d1.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/tables/d34cafc312f4c7d0c36f4e2b3c9b57fe74e9aaded0140ec626cab1eeadcd93d1.jpg)

![e9c7bed127b8b763d088464353bc1a4b08620f594294d92fdd216c973b938433.jpg](../icml_results/2872_Self-Prompting%20Diffusion%20Transformer%20for%20Open-Vocabulary%20Scene%20Text%20Editing%20via%20In-Context%20Learning/tables/e9c7bed127b8b763d088464353bc1a4b08620f594294d92fdd216c973b938433.jpg)

<a id="paper-2835"></a>

## OpenSage: Self-programming Agent Generation Engine


### Images

![111dbbb0fb223ef29a4c0f57da47bb6e5ecc71b0b6a19c8c8143219015308a45.jpg](../icml_results/2873_OpenSage_%20Self-programming%20Agent%20Generation%20Engine/images/111dbbb0fb223ef29a4c0f57da47bb6e5ecc71b0b6a19c8c8143219015308a45.jpg)

![98671306bee38e03ad5a5de0bfd05f4ba69015a0970f402f8e390e437aaade5d.jpg](../icml_results/2873_OpenSage_%20Self-programming%20Agent%20Generation%20Engine/images/98671306bee38e03ad5a5de0bfd05f4ba69015a0970f402f8e390e437aaade5d.jpg)

![ad7d5f96a006b55670b5264628c5b62d1ee2d37119d8ce0a58a5dc787abec9ce.jpg](../icml_results/2873_OpenSage_%20Self-programming%20Agent%20Generation%20Engine/images/ad7d5f96a006b55670b5264628c5b62d1ee2d37119d8ce0a58a5dc787abec9ce.jpg)

![b120754be48b375561c61015fffaba7b3e292e533691569ce2ccce237ed17d80.jpg](../icml_results/2873_OpenSage_%20Self-programming%20Agent%20Generation%20Engine/images/b120754be48b375561c61015fffaba7b3e292e533691569ce2ccce237ed17d80.jpg)

![cb2cd6b8150d4b88bff04c890bb04902df6c546f45ccaa9575044d066541cfb1.jpg](../icml_results/2873_OpenSage_%20Self-programming%20Agent%20Generation%20Engine/images/cb2cd6b8150d4b88bff04c890bb04902df6c546f45ccaa9575044d066541cfb1.jpg)

### Tables

![303428e7140fcb3d6a4111dcc690b468f8c39068a430f234156b30e7758b986f.jpg](../icml_results/2873_OpenSage_%20Self-programming%20Agent%20Generation%20Engine/tables/303428e7140fcb3d6a4111dcc690b468f8c39068a430f234156b30e7758b986f.jpg)

![3c634517373491c92f87b49e8a27ab022ab02fddbdaa6ee5cb731d7e4e8ec2cb.jpg](../icml_results/2873_OpenSage_%20Self-programming%20Agent%20Generation%20Engine/tables/3c634517373491c92f87b49e8a27ab022ab02fddbdaa6ee5cb731d7e4e8ec2cb.jpg)

![402d1b84c27cd6c9e518222cf8a9908e242df40cc6225ecd8abc7189bf903e0e.jpg](../icml_results/2873_OpenSage_%20Self-programming%20Agent%20Generation%20Engine/tables/402d1b84c27cd6c9e518222cf8a9908e242df40cc6225ecd8abc7189bf903e0e.jpg)

![5af0436e548f0a1c9e671ae7d7868aed88135a0fe0f27b3efdbc93c297b9267d.jpg](../icml_results/2873_OpenSage_%20Self-programming%20Agent%20Generation%20Engine/tables/5af0436e548f0a1c9e671ae7d7868aed88135a0fe0f27b3efdbc93c297b9267d.jpg)

![8ab849b2fc6244a035bf16a90474999b07e98dd787e633a1bd30afbac1d1bf9a.jpg](../icml_results/2873_OpenSage_%20Self-programming%20Agent%20Generation%20Engine/tables/8ab849b2fc6244a035bf16a90474999b07e98dd787e633a1bd30afbac1d1bf9a.jpg)

![f33cf18600ba85157e13f492947fbcda7da8f94bec8b18203d1844d1df597029.jpg](../icml_results/2873_OpenSage_%20Self-programming%20Agent%20Generation%20Engine/tables/f33cf18600ba85157e13f492947fbcda7da8f94bec8b18203d1844d1df597029.jpg)

<a id="paper-2836"></a>

## Generative Adaptation of Dynamics to Environmental Shifts via Weight-space Diffusion


### Images

![1e919536777875b6d8e376673d7e2b1025729776257685629a0be0fa75750041.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/images/1e919536777875b6d8e376673d7e2b1025729776257685629a0be0fa75750041.jpg)

![2b4cd174aa070dd5c55358f0f72ab074e8d3a40eeec1fe136a43a1bb1576003b.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/images/2b4cd174aa070dd5c55358f0f72ab074e8d3a40eeec1fe136a43a1bb1576003b.jpg)

![4c3bbb66b91c156b9c49f3f102057837be42ab7a972add6f683b560ac9944d50.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/images/4c3bbb66b91c156b9c49f3f102057837be42ab7a972add6f683b560ac9944d50.jpg)

![4c60125dbb32880b273894fb9410f2fec4ee032e0fda5136337fd3d6e3112e62.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/images/4c60125dbb32880b273894fb9410f2fec4ee032e0fda5136337fd3d6e3112e62.jpg)

![4d749ce636d0e2832b15dfa1068442940e947c401dd12d0e2dbf13cd930c4c5e.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/images/4d749ce636d0e2832b15dfa1068442940e947c401dd12d0e2dbf13cd930c4c5e.jpg)

![51ba573141dcb7e872d3c828ec00f07eba5dd05578db12005e6feccd547e61e0.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/images/51ba573141dcb7e872d3c828ec00f07eba5dd05578db12005e6feccd547e61e0.jpg)

![6f4edd2a16e314169b6367d2be4b12b0dbe561c07f1947a93ec18fe160457a3e.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/images/6f4edd2a16e314169b6367d2be4b12b0dbe561c07f1947a93ec18fe160457a3e.jpg)

![74e1fab49e5e2f4ff3fdb1324b56e54a8b235b2670246c7734efbbf5579dc9ea.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/images/74e1fab49e5e2f4ff3fdb1324b56e54a8b235b2670246c7734efbbf5579dc9ea.jpg)

![99cc327d980dd5495fa3e6bd86b1fc58e1d1325f3965e6d3f55c06bd0273e9e3.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/images/99cc327d980dd5495fa3e6bd86b1fc58e1d1325f3965e6d3f55c06bd0273e9e3.jpg)

![a73f455bec0452de3c3fed8ad058660f486108f2063d350502bf201e943dc934.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/images/a73f455bec0452de3c3fed8ad058660f486108f2063d350502bf201e943dc934.jpg)

![ad8e4eac025b1fd1157d98bb2ecaa7db3d0a3cb183bac4b9758434bc59bb6a66.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/images/ad8e4eac025b1fd1157d98bb2ecaa7db3d0a3cb183bac4b9758434bc59bb6a66.jpg)

![c254049a69c97bdfe6d066063c18b79a23904349795c2b38d7b80443eede495e.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/images/c254049a69c97bdfe6d066063c18b79a23904349795c2b38d7b80443eede495e.jpg)

![d859c8ca1ad9421ed02efbb697e87783e50b228abacbe3bd7d08226a17abb975.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/images/d859c8ca1ad9421ed02efbb697e87783e50b228abacbe3bd7d08226a17abb975.jpg)

![df4672ae7c7512acbf6d998761eafa2a11db90fe1cdbf522b1ecc2e3a74ecda8.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/images/df4672ae7c7512acbf6d998761eafa2a11db90fe1cdbf522b1ecc2e3a74ecda8.jpg)

### Tables

![09d7e82997b8049927f25e36791ccbf3faeefbc2f489c013342f4120084db080.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/09d7e82997b8049927f25e36791ccbf3faeefbc2f489c013342f4120084db080.jpg)

![2f4f3b69b4fa73765fa19ad22547431acb6435df25e476c7ddbbce036b7802d0.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/2f4f3b69b4fa73765fa19ad22547431acb6435df25e476c7ddbbce036b7802d0.jpg)

![3019eff7d855c4fa2c71611006685862b0a14ec28e57a1448bbc4b5131597d3f.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/3019eff7d855c4fa2c71611006685862b0a14ec28e57a1448bbc4b5131597d3f.jpg)

![46468852e01ebaa54147e30619d6ab672013baa648d0f83dc7a42ad032824a3e.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/46468852e01ebaa54147e30619d6ab672013baa648d0f83dc7a42ad032824a3e.jpg)

![5fe1558416b1fd5eb3affb7080719ddb79e329ea41db6227925fec3d0361ca99.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/5fe1558416b1fd5eb3affb7080719ddb79e329ea41db6227925fec3d0361ca99.jpg)

![64390d11b023830861f4d9d1b670e82c08c45cfa7439d73d7ea4303947cf6a0b.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/64390d11b023830861f4d9d1b670e82c08c45cfa7439d73d7ea4303947cf6a0b.jpg)

![69403b287822767eb02621c6f83b85c584de8ef42ab47c5f951544ff09800565.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/69403b287822767eb02621c6f83b85c584de8ef42ab47c5f951544ff09800565.jpg)

![6b46a3b94b905f24c851715172d40cdcedd8fbb24e97687942fd064cffc17a4d.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/6b46a3b94b905f24c851715172d40cdcedd8fbb24e97687942fd064cffc17a4d.jpg)

![72403fce96ac223a70f498934e03dc162d932ddbda02462f8aa2964cb62cd6d0.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/72403fce96ac223a70f498934e03dc162d932ddbda02462f8aa2964cb62cd6d0.jpg)

![829dc46cf0ffbfe2f6c4e777a83311f0b138f29a1a580481561373ad41211b6d.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/829dc46cf0ffbfe2f6c4e777a83311f0b138f29a1a580481561373ad41211b6d.jpg)

![85aa030693a3b9f894ef9262d9d35594e4829600f7574119628be548ddb84e13.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/85aa030693a3b9f894ef9262d9d35594e4829600f7574119628be548ddb84e13.jpg)

![93aa3f1ae9b82c1a7e68550087bf3972a2a35254fe181186df3df7634f4f84bd.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/93aa3f1ae9b82c1a7e68550087bf3972a2a35254fe181186df3df7634f4f84bd.jpg)

![9c348db4fc83174640a6a2e2861566007dfa86a89cdfe55f925d120197bc0d4b.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/9c348db4fc83174640a6a2e2861566007dfa86a89cdfe55f925d120197bc0d4b.jpg)

![a4d2fc0253067f72cfbd0b35ea33855df0ea71c30522dd6fa8ccc176ba5700d9.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/a4d2fc0253067f72cfbd0b35ea33855df0ea71c30522dd6fa8ccc176ba5700d9.jpg)

![a6195673dadc83802c6c1b5b34c97f4a1eb01fc12842538e2f2e83945db7da46.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/a6195673dadc83802c6c1b5b34c97f4a1eb01fc12842538e2f2e83945db7da46.jpg)

![b03be2a6ad15084c4d93bb157f7ff1087686675762ec06dc8bb184cf08c01f53.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/b03be2a6ad15084c4d93bb157f7ff1087686675762ec06dc8bb184cf08c01f53.jpg)

![c8984940d4825b9df1b0988cb9e2e862b1c9d0f2552a04e911dd40614b7667e4.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/c8984940d4825b9df1b0988cb9e2e862b1c9d0f2552a04e911dd40614b7667e4.jpg)

![ce34138a577cf099d80423b3b88d85139a7c2f8c49769263eca9e176685adf7f.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/ce34138a577cf099d80423b3b88d85139a7c2f8c49769263eca9e176685adf7f.jpg)

![e20cdf8237880ed90b8f4f8b0072c13b3571064992f6b76d8c6c6c4c752c47b6.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/e20cdf8237880ed90b8f4f8b0072c13b3571064992f6b76d8c6c6c4c752c47b6.jpg)

![e35c24bfb9c5efb9758d7b62e7d5d44c88b632bfffee45392589207f445e2910.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/e35c24bfb9c5efb9758d7b62e7d5d44c88b632bfffee45392589207f445e2910.jpg)

![ef518bef5c9798ec4073ab8e67f5ae0e8bf0bef08d436309f8cffe862cd00bf9.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/ef518bef5c9798ec4073ab8e67f5ae0e8bf0bef08d436309f8cffe862cd00bf9.jpg)

![f98c00c521e9084b2f5b77848c35738a41abdd79ab2328d95c18a9653f7826ea.jpg](../icml_results/2874_Generative%20Adaptation%20of%20Dynamics%20to%20Environmental%20Shifts%20via%20Weight-space%20Diffusion/tables/f98c00c521e9084b2f5b77848c35738a41abdd79ab2328d95c18a9653f7826ea.jpg)

<a id="paper-2837"></a>

## BandPO: Bridging Trust Regions and Ratio Clipping via Probability-Aware Bounds for LLM Reinforcement Learning


### Images

![05daaea10327d909e607d2bd8e5096153727b782564b982a6c495e7e278a00fb.jpg](../icml_results/2875_BandPO_%20Bridging%20Trust%20Regions%20and%20Ratio%20Clipping%20via%20Probability-Aware%20Bounds%20for%20LLM%20Reinforcement/images/05daaea10327d909e607d2bd8e5096153727b782564b982a6c495e7e278a00fb.jpg)

![5b804acfeac0d96ccf1376721b4b5daa81ead512e202145178c02e9a991f3912.jpg](../icml_results/2875_BandPO_%20Bridging%20Trust%20Regions%20and%20Ratio%20Clipping%20via%20Probability-Aware%20Bounds%20for%20LLM%20Reinforcement/images/5b804acfeac0d96ccf1376721b4b5daa81ead512e202145178c02e9a991f3912.jpg)

![bcafd4c4e68fb5347210ca551f87646d1b94422ede55db79b3c536714ba89a15.jpg](../icml_results/2875_BandPO_%20Bridging%20Trust%20Regions%20and%20Ratio%20Clipping%20via%20Probability-Aware%20Bounds%20for%20LLM%20Reinforcement/images/bcafd4c4e68fb5347210ca551f87646d1b94422ede55db79b3c536714ba89a15.jpg)

![fc347b3fd9f4ce496116eca394acda14a40ccb5b46f2560ae29e8f1805577ea5.jpg](../icml_results/2875_BandPO_%20Bridging%20Trust%20Regions%20and%20Ratio%20Clipping%20via%20Probability-Aware%20Bounds%20for%20LLM%20Reinforcement/images/fc347b3fd9f4ce496116eca394acda14a40ccb5b46f2560ae29e8f1805577ea5.jpg)

### Tables

![cde619a7b9c6f004b16a3ec46d2003007391241a90d8406f28f86e2410c56c8b.jpg](../icml_results/2875_BandPO_%20Bridging%20Trust%20Regions%20and%20Ratio%20Clipping%20via%20Probability-Aware%20Bounds%20for%20LLM%20Reinforcement/tables/cde619a7b9c6f004b16a3ec46d2003007391241a90d8406f28f86e2410c56c8b.jpg)

![e4b4733b33f745fb91122827967f0a6f835a192515c0153d2993eb17b4abf520.jpg](../icml_results/2875_BandPO_%20Bridging%20Trust%20Regions%20and%20Ratio%20Clipping%20via%20Probability-Aware%20Bounds%20for%20LLM%20Reinforcement/tables/e4b4733b33f745fb91122827967f0a6f835a192515c0153d2993eb17b4abf520.jpg)

<a id="paper-2838"></a>

## Prefix-Cache-Aware Data Reordering for LLM-Augmented Database Analytics

### Images

![3be7dcc518c193bdb3e3fbec61caf60fc9390cf8192557ff18539a38df719bd8.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/3be7dcc518c193bdb3e3fbec61caf60fc9390cf8192557ff18539a38df719bd8.jpg)

![533ec01102d13fd4220656684493ffa20974e5b3906f9bb2b6cb52377dfbfec4.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/533ec01102d13fd4220656684493ffa20974e5b3906f9bb2b6cb52377dfbfec4.jpg)

![8ab40db10fea2ae42af363ae4ea52d7e6f2abe4ce35357f4ca3e839a9eedfe71.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/8ab40db10fea2ae42af363ae4ea52d7e6f2abe4ce35357f4ca3e839a9eedfe71.jpg)

![91a16100df644b1ba4c1df50c5268b8ec9994c10b68828ba2dcb0cc0fa054e5c.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/91a16100df644b1ba4c1df50c5268b8ec9994c10b68828ba2dcb0cc0fa054e5c.jpg)

![925e169652a9794689a3478b5df15f394f86cacb4c0a0a9e049a7c5c4b0ba7f9.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/925e169652a9794689a3478b5df15f394f86cacb4c0a0a9e049a7c5c4b0ba7f9.jpg)

![a3f477bd534f09af24f559d14dcb64daa773df17b213ecbb89a3f61843e0309e.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/a3f477bd534f09af24f559d14dcb64daa773df17b213ecbb89a3f61843e0309e.jpg)

![a4213671feafdbcc7c62689a93c5c3a12fb0fcdedf07af3e7e8a8baa8ed1fedb.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/a4213671feafdbcc7c62689a93c5c3a12fb0fcdedf07af3e7e8a8baa8ed1fedb.jpg)

![c9c42aca8e790a6192c2e2df5f54378b16365c9c67966ea6cf839a45fa405efd.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/c9c42aca8e790a6192c2e2df5f54378b16365c9c67966ea6cf839a45fa405efd.jpg)

![c9f6bd5dcff53c2eeeb9100b47716736b21509c37e84f86deead6c531f40ed5d.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/c9f6bd5dcff53c2eeeb9100b47716736b21509c37e84f86deead6c531f40ed5d.jpg)

![da48c5d1f1fa0abe3c15bf1c5ffdde73dbb80caece5281ff8920b9cd3280033a.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/da48c5d1f1fa0abe3c15bf1c5ffdde73dbb80caece5281ff8920b9cd3280033a.jpg)

![e28ccd8e1a36a331398fce124c302214d9035d3e0f70e9030681be3ffc266973.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/e28ccd8e1a36a331398fce124c302214d9035d3e0f70e9030681be3ffc266973.jpg)

![e3e96aa042bc8f2d3792d8aec184158f53fcc3e7ed099efbb3cc434a3628ba81.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/images/e3e96aa042bc8f2d3792d8aec184158f53fcc3e7ed099efbb3cc434a3628ba81.jpg)

### Tables

![0d32ecb9325834d65d3a7299d8adf9f88ce4e90d5d8ca3d17975798a82c93081.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/0d32ecb9325834d65d3a7299d8adf9f88ce4e90d5d8ca3d17975798a82c93081.jpg)

![10b72fce971699af6df02570a905f1e704e41d74f6e476bdbbdfb078eff3c980.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/10b72fce971699af6df02570a905f1e704e41d74f6e476bdbbdfb078eff3c980.jpg)

![1650fffa0b17e3db73ac627eb2415877fa572942ea9a4ed36ae461d969a273d7.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/1650fffa0b17e3db73ac627eb2415877fa572942ea9a4ed36ae461d969a273d7.jpg)

![25a9aa27e94358d1fab1a155a89a5acddbc148eef0ce3cfadd2ac9b51aea3b39.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/25a9aa27e94358d1fab1a155a89a5acddbc148eef0ce3cfadd2ac9b51aea3b39.jpg)

![3ad28466a2f0a6d675e4b9bcc017a6bf3e66f5e07834ab211a539e7ec99eb2c2.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/3ad28466a2f0a6d675e4b9bcc017a6bf3e66f5e07834ab211a539e7ec99eb2c2.jpg)

![4153c8a41d950b5b34eadd51dd84ed2ca5b164640d4f7d3ce04ebe9056ad0ae1.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/4153c8a41d950b5b34eadd51dd84ed2ca5b164640d4f7d3ce04ebe9056ad0ae1.jpg)

![4fa0772b7c12cb3b46caae9bf036434e4c0ab63a7dfd949d1ac3714a554dbbd0.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/4fa0772b7c12cb3b46caae9bf036434e4c0ab63a7dfd949d1ac3714a554dbbd0.jpg)

![62090a1339bdf13b53c18da96374d3fa2dd88dde21eb1fa1b166b0ba27b679ba.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/62090a1339bdf13b53c18da96374d3fa2dd88dde21eb1fa1b166b0ba27b679ba.jpg)

![62e001143c7126f8ec42810d7e5fb9d73606ce1dfe91c240ffdc257510e65b72.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/62e001143c7126f8ec42810d7e5fb9d73606ce1dfe91c240ffdc257510e65b72.jpg)

![6316120e8d004e7003f91b25bce0c96e98ccd3ed6c4ec94c128bc6a3c3342074.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/6316120e8d004e7003f91b25bce0c96e98ccd3ed6c4ec94c128bc6a3c3342074.jpg)

![664c537085c4a1df91b3194be9372236b74d2c8899a4259870131deb73178551.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/664c537085c4a1df91b3194be9372236b74d2c8899a4259870131deb73178551.jpg)

![6d2f35ed2513af73f148c25e9299e9b6186428ed38234b09f33e0637f8e1bf5b.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/6d2f35ed2513af73f148c25e9299e9b6186428ed38234b09f33e0637f8e1bf5b.jpg)

![7215c3c74dac8c49a0162fc04f70358eac3af2d955beee8eae3b464ab2044803.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/7215c3c74dac8c49a0162fc04f70358eac3af2d955beee8eae3b464ab2044803.jpg)

![79b6ab9d4c6d8dfb182c4735cb7d20ee6c94c3631037e763faf9bf0c379bde0d.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/79b6ab9d4c6d8dfb182c4735cb7d20ee6c94c3631037e763faf9bf0c379bde0d.jpg)

![85178ce68a148bd9e85773d841f747e0b73403d1e03b353c6e480f8b4f9f86ed.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/85178ce68a148bd9e85773d841f747e0b73403d1e03b353c6e480f8b4f9f86ed.jpg)

![8eb0d849a2463e67dac54f3a192730d9efae18840af2151093d2107b741cf1cc.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/8eb0d849a2463e67dac54f3a192730d9efae18840af2151093d2107b741cf1cc.jpg)

![922f1a23da71dfe9d32bb8271bfdc5f6483be9846e6f2f2b93a73a90a0895df0.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/922f1a23da71dfe9d32bb8271bfdc5f6483be9846e6f2f2b93a73a90a0895df0.jpg)

![ac6e6e18442211d5d4e0c7dc4525d0580b00305aba94b4a57f73a275c288cd8a.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/ac6e6e18442211d5d4e0c7dc4525d0580b00305aba94b4a57f73a275c288cd8a.jpg)

![ad1dda1ca0c5e0bbc4cbb1ba63525c8eb96673ecc702f625d26746d06b2d9fde.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/ad1dda1ca0c5e0bbc4cbb1ba63525c8eb96673ecc702f625d26746d06b2d9fde.jpg)

![c3ea61419e980f7af4490c104d12d553e48d0121ee634c7fef440906e2a9d4c4.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/c3ea61419e980f7af4490c104d12d553e48d0121ee634c7fef440906e2a9d4c4.jpg)

![c6d28a85620cd19ea850f9c779d6a81754d45ed6207f88b0aa1eb6105226da4d.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/c6d28a85620cd19ea850f9c779d6a81754d45ed6207f88b0aa1eb6105226da4d.jpg)

![c91a745f98294c8b8a38860a7bd495fa696acb90d2780cdebb5fb2a365d33b58.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/c91a745f98294c8b8a38860a7bd495fa696acb90d2780cdebb5fb2a365d33b58.jpg)

![cc5f780c1f32653a4a3fd1abffb49c01e5eff40f65076c866c1f7d34953337ae.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/cc5f780c1f32653a4a3fd1abffb49c01e5eff40f65076c866c1f7d34953337ae.jpg)

![d0b93589c5720f300491a4a11f94f5a0462c5c2b9b7e4c6ba92403f46103ef2c.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/d0b93589c5720f300491a4a11f94f5a0462c5c2b9b7e4c6ba92403f46103ef2c.jpg)

![d0c432f4b76727083cde3371539884d6f0dce2629afb35b4edec7b7e035d6a80.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/d0c432f4b76727083cde3371539884d6f0dce2629afb35b4edec7b7e035d6a80.jpg)

![d4dee49bfea333cf807c3f5de5d5bc940ed668d3e80667269e0e53d2f8f1717e.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/d4dee49bfea333cf807c3f5de5d5bc940ed668d3e80667269e0e53d2f8f1717e.jpg)

![e309791bbf33eaffc317bceba0e05d789c8a83a5852f6a408d9a60d9f0ba5106.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/e309791bbf33eaffc317bceba0e05d789c8a83a5852f6a408d9a60d9f0ba5106.jpg)

![e94915d9eed77e539ee3215b8ba6f8930419e4efde85ab8f8020010d4972dc00.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/e94915d9eed77e539ee3215b8ba6f8930419e4efde85ab8f8020010d4972dc00.jpg)

![fbfabfc4742e82479b05dd54e9f8ba8f632741b6da3c2640d4ca84efab675f96.jpg](../icml_results/2876_Prefix-Cache-Aware%20Data%20Reordering%20for%20LLM-Augmented%20Database%20Analytics/tables/fbfabfc4742e82479b05dd54e9f8ba8f632741b6da3c2640d4ca84efab675f96.jpg)
