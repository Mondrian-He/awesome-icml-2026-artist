# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 89 of 197

## 目录 (Table of Contents)

1. [MedScope: Incentivizing "Think with Videos" for Clinical Reasoning via Coarse-to-Fine Tool Calling](#paper-2905)
2. [SetPO: Set-Level Policy Optimization for Diversity-Preserving LLM Reasoning](#paper-2906)
3. [FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents](#paper-2907)
4. [ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment](#paper-2908)
5. [Geometrically Constrained Stenosis Editing in Coronary Angiography via Entropic Optimal Transport](#paper-2909)
6. [State-Dependent Safety Failures in Multi-Turn Language Model Interaction](#paper-2910)
7. [Task-Aware Preference Calibration for Direct Preference Optimization](#paper-2911)
8. [RePo: Language Models with Context Re-Positioning](#paper-2912)
9. [UAV$^2$: A Unified and Adaptive Scheduling Framework for UAV Autopilot Systems with Reinforcement Learning](#paper-2913)
10. [Rethinking Temporal Consistency in Video Object-Centric Learning: From Prediction to Correspondence](#paper-2914)
11. [Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training](#paper-2915)
12. [On Revisiting Entropy for Identifying Mislabeled Images](#paper-2916)
13. [Diffusing to Coordinate: Efficient Online Multi-Agent Diffusion Policies](#paper-2917)
14. [Multi-Label Test-Time Adaptation with Bayesian Conditional Priors](#paper-2918)
15. [Beyond Logits: Metastable Latent Dynamics for Sample-Efficient Best-of-N Selection in LLMs](#paper-2919)
16. [Self-Guidance: Enhancing Neural Codecs via Decoder Manifold Alignment](#paper-2920)
17. [Lavida-R1: Advancing Reasoning for Unified Multimodal Diffusion Language Models](#paper-2921)
18. [TransLight: Image-Guided Customized Lighting Control with Generative Decoupling](#paper-2922)
19. [COPF: An Online Framework for Deployment-Stable Counterfactual Fairness in Evolving Graphs](#paper-2923)
20. [Enhancing Membership Inference Attacks on Diffusion Models from a Frequency-Domain Perspective](#paper-2924)
21. [MapDream: Task-Driven Map Learning for Vision-Language Navigation](#paper-2925)
22. [LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries](#paper-2926)
23. [Evidential Reasoning Advances Interpretable Real-World Disease Screening](#paper-2927)
24. [Bipartite Graph Attention-based Clustering for Large-scale scRNA-seq Data](#paper-2928)
25. [Hyperspectral Image Fusion with Spectral-Band and Fusion-Scale Agnosticism](#paper-2929)
26. [Geometric Decoupling: Diagnosing the Structural Instability of Latent](#paper-2930)
27. [An Evidential Route to Asymptotic Bayes Optimality under Sparsity](#paper-2931)
28. [Causal Representation Learning with Optimal Compression under Complex Treatments](#paper-2932)
29. [Seg-ReSearch: Segmentation with Interleaved Reasoning and External Search](#paper-2933)
30. [CURVE: Learning Causality-Inspired Invariant Representations for Robust Scene Understanding via Uncertainty-Guided Regularization](#paper-2934)
31. [SMM Transformer: Leveraging Spiking Neural Networks for Multimodal Tasks](#paper-2935)
32. [Spike-HTR: Spiking Neural Transformer for Handwritten Text Recognition](#paper-2936)
33. [AnyMod-LLVE: Low-Light Video Enhancement with Modality-Agnostic Inference](#paper-2937)

---


<a id="paper-2905"></a>

## MedScope: Incentivizing "Think with Videos" for Clinical Reasoning via Coarse-to-Fine Tool Calling

### Images

![0a65cacfbe810dafea088dd2de9742cf63e9072b6a32b8d5b8004c1d0147269f.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/0a65cacfbe810dafea088dd2de9742cf63e9072b6a32b8d5b8004c1d0147269f.jpg)

![0b145398a9700f46f368c949d088322ed38c41058aca0a8655ab5037028264fc.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/0b145398a9700f46f368c949d088322ed38c41058aca0a8655ab5037028264fc.jpg)

![19b16d790108e86419781d59dd2286e7098a47be90c035ba2edc45b74a9c4622.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/19b16d790108e86419781d59dd2286e7098a47be90c035ba2edc45b74a9c4622.jpg)

![1bf3c65439deea9ba8cc51859edbd7538faf5331745f15f353da3e5340608be2.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/1bf3c65439deea9ba8cc51859edbd7538faf5331745f15f353da3e5340608be2.jpg)

![21ea9a43d5863e6423800174cde7db95b63a3011223dce04161c19689e37c26b.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/21ea9a43d5863e6423800174cde7db95b63a3011223dce04161c19689e37c26b.jpg)

![22b66033065414598c974caeb426dfe90eb73d03bc3fd23c25b0c6149f0ba15e.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/22b66033065414598c974caeb426dfe90eb73d03bc3fd23c25b0c6149f0ba15e.jpg)

![2a302a61ec7ee303a9bd867751b89f1d9227a08500ec29425c3cd1f2c7d6380b.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/2a302a61ec7ee303a9bd867751b89f1d9227a08500ec29425c3cd1f2c7d6380b.jpg)

![30cfd2717f8f50bb0a4e13132bd756bc572f3a2bec02ca2978cb87bd1344b5b5.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/30cfd2717f8f50bb0a4e13132bd756bc572f3a2bec02ca2978cb87bd1344b5b5.jpg)

![310770226fa899fc6e9936fc06724186b2cdccfe6a0b5a0110298a439f45b22c.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/310770226fa899fc6e9936fc06724186b2cdccfe6a0b5a0110298a439f45b22c.jpg)

![3fe3e0fc7be9dd7bcfdd0da551df4ece3d5b20356590ecce5eac80d58dd8d084.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/3fe3e0fc7be9dd7bcfdd0da551df4ece3d5b20356590ecce5eac80d58dd8d084.jpg)

![416a1bc506f512373c52df72b179331111f3c3523b8c395e69f220f3100d2fc9.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/416a1bc506f512373c52df72b179331111f3c3523b8c395e69f220f3100d2fc9.jpg)

![45d73e060e5eee39e97463c2da24bb55b9dc627e89f92ae26ec0834671aac75b.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/45d73e060e5eee39e97463c2da24bb55b9dc627e89f92ae26ec0834671aac75b.jpg)

![4f13bba6edc4101c7e14a5eab55110e79f424e81f164476a02931b673e6ac217.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/4f13bba6edc4101c7e14a5eab55110e79f424e81f164476a02931b673e6ac217.jpg)

![676c0bebeb2b990867936fd616edfb87e82058797745fc237a1b2e37d257ed99.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/676c0bebeb2b990867936fd616edfb87e82058797745fc237a1b2e37d257ed99.jpg)

![7bcbd8eb335900b053b556d294b94688693739e92da1a61e589dc3e35cc82205.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/7bcbd8eb335900b053b556d294b94688693739e92da1a61e589dc3e35cc82205.jpg)

![8bd42222f3f9070d6dc9ceda9bfc3efbb7b8a7073b36564feec27544f0e2ec94.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/8bd42222f3f9070d6dc9ceda9bfc3efbb7b8a7073b36564feec27544f0e2ec94.jpg)

![9cea405a267194fdb2a9b4271825d68786e987abd725cde6a3a82157f50b1dbe.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/9cea405a267194fdb2a9b4271825d68786e987abd725cde6a3a82157f50b1dbe.jpg)

![9ee8e97ca37b86e1a557e8c82932118e1687fc6898b86bf514108aab26cae6ec.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/9ee8e97ca37b86e1a557e8c82932118e1687fc6898b86bf514108aab26cae6ec.jpg)

![a0f21583385f60ed742fd897c6e4c580cd2130d0e474ec60107d88720bb84642.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/a0f21583385f60ed742fd897c6e4c580cd2130d0e474ec60107d88720bb84642.jpg)

![ade7011d4f299e9a9c7df013abb1a5a2293aee515d3d8ee08b771965a5043390.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/ade7011d4f299e9a9c7df013abb1a5a2293aee515d3d8ee08b771965a5043390.jpg)

![b84db6bb267c7c83053260d0319d8b336b4c2f3e2b7e0ed9bf93fa29b3ce4d83.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/b84db6bb267c7c83053260d0319d8b336b4c2f3e2b7e0ed9bf93fa29b3ce4d83.jpg)

![bcdc601fd40659e1bde12c5c38549d115afbfd59e5bc8ffcab00a14c3facff60.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/bcdc601fd40659e1bde12c5c38549d115afbfd59e5bc8ffcab00a14c3facff60.jpg)

![bfe1115ce1f3ace40d4188c052b7e5ad73e16df13f205a875f93b7a9e80eb230.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/bfe1115ce1f3ace40d4188c052b7e5ad73e16df13f205a875f93b7a9e80eb230.jpg)

![c1d9283a92b4b95cc8aa4ccfe96a34d092ca102f13a0bb2d1db56cfc3e314cdd.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/c1d9283a92b4b95cc8aa4ccfe96a34d092ca102f13a0bb2d1db56cfc3e314cdd.jpg)

![e7f0b58f6ac7ce3b080a7b9942579ea04145bffb2702770eb3550437bdb7eafa.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/e7f0b58f6ac7ce3b080a7b9942579ea04145bffb2702770eb3550437bdb7eafa.jpg)

![f41a909060f8e7572280a74399062ae533938103c4698200b4c6fb0625bb38cf.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/f41a909060f8e7572280a74399062ae533938103c4698200b4c6fb0625bb38cf.jpg)

![f7f0843092b30b952c5d4783dce625e6b905fef746dba6b107cf622431810596.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/f7f0843092b30b952c5d4783dce625e6b905fef746dba6b107cf622431810596.jpg)

![fda1858328273a12036d681706b402975ff89064ddbb35f9c716aff83e6be30d.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/fda1858328273a12036d681706b402975ff89064ddbb35f9c716aff83e6be30d.jpg)

### Tables

![1fcb3cb2ba8e9731c138465ae1c812df7cf48aab7996e0d7ac4c31c2b71caa71.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/1fcb3cb2ba8e9731c138465ae1c812df7cf48aab7996e0d7ac4c31c2b71caa71.jpg)

![38d641fe928cac9f8af0887ecf4d2b527087858bff2a5d5c2bc776831a3c7775.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/38d641fe928cac9f8af0887ecf4d2b527087858bff2a5d5c2bc776831a3c7775.jpg)

![486ff668b1c6857b1b8bff4da32f34dd30cb7b9f6eebb5a329a57679de256084.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/486ff668b1c6857b1b8bff4da32f34dd30cb7b9f6eebb5a329a57679de256084.jpg)

![48bec6965c87fdf66a5c904aa52a05d24d34abbdf1adc2d72c6b472547bd5d97.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/48bec6965c87fdf66a5c904aa52a05d24d34abbdf1adc2d72c6b472547bd5d97.jpg)

![71029f4dff94651f5b5c5989eb31703ac7816e7fd9b218510344b08e61195eb6.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/71029f4dff94651f5b5c5989eb31703ac7816e7fd9b218510344b08e61195eb6.jpg)

![775a815e0e5b3c5a1f45dc5001cdb2b4cb37136a998723c314096ca36d345e15.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/775a815e0e5b3c5a1f45dc5001cdb2b4cb37136a998723c314096ca36d345e15.jpg)

![a430b279dc34d53265d98c3fc4afb162b37775c08cdb8e4b9169f990d662989f.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/a430b279dc34d53265d98c3fc4afb162b37775c08cdb8e4b9169f990d662989f.jpg)

![c8db1896a42022866dbd5a592ba49c7226b821e0b892e7be89e8f91585576613.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/c8db1896a42022866dbd5a592ba49c7226b821e0b892e7be89e8f91585576613.jpg)

![daec1b8ca174fee7d7119e4758f836699128dc4e602fa1a0f43adce63972f350.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/daec1b8ca174fee7d7119e4758f836699128dc4e602fa1a0f43adce63972f350.jpg)

![e3957c68a3ab04094bbd675c93a78b3b9b86e45c6f62ae4162bb563186023895.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/e3957c68a3ab04094bbd675c93a78b3b9b86e45c6f62ae4162bb563186023895.jpg)

![e74c09e10f4b6ec8345949321d804921f1f782b3b39caf41392feacbaefa1b86.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/e74c09e10f4b6ec8345949321d804921f1f782b3b39caf41392feacbaefa1b86.jpg)

## MedScope: Incentivizing "Think with Videos" for Clinical Reasoning via Coarse-to-Fine Tool Calling


### Images

![007557dad08ca35a79740b0f96508257f2e54a4726c8aab6bf1e4fae3ff16a8c.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/007557dad08ca35a79740b0f96508257f2e54a4726c8aab6bf1e4fae3ff16a8c.jpg)

![021526a1d74c2311ace4578ecc47345a6534094f7bb2b86003a16a35778098ab.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/021526a1d74c2311ace4578ecc47345a6534094f7bb2b86003a16a35778098ab.jpg)

![021e25921553035ed07d19109be33239e5ad7f4adf70f43f2c623ddaf68ba709.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/021e25921553035ed07d19109be33239e5ad7f4adf70f43f2c623ddaf68ba709.jpg)

![095f292e98b4bd3ebf8386997b5dd3c2b9f25a9eb90630d67135c90af1a04798.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/095f292e98b4bd3ebf8386997b5dd3c2b9f25a9eb90630d67135c90af1a04798.jpg)

![0a295253cba9cc62ad86050e063843f1e2c27ee59ecc8e8b5335c90fac00f807.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/0a295253cba9cc62ad86050e063843f1e2c27ee59ecc8e8b5335c90fac00f807.jpg)

![0c6334d25dbe90b0e2c3223f713194aee993ab4cb0484eca6954ddf8c2ad8c87.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/0c6334d25dbe90b0e2c3223f713194aee993ab4cb0484eca6954ddf8c2ad8c87.jpg)

![0edee88f086bb7e30af8a9a234ab75da40ba87ba3dd6c40ce175733dd4f412c2.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/0edee88f086bb7e30af8a9a234ab75da40ba87ba3dd6c40ce175733dd4f412c2.jpg)

![123ec3adcf42de16c5299a4baa96dab0bb24edb15a2605fc48bcb513056d69d4.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/123ec3adcf42de16c5299a4baa96dab0bb24edb15a2605fc48bcb513056d69d4.jpg)

![15f276048a49270241fb30e773da6e6cc357810875c5738d68cb95955d231c82.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/15f276048a49270241fb30e773da6e6cc357810875c5738d68cb95955d231c82.jpg)

![17175d13c0a45575b2727f4286459f1371c2b1d1e7dac6dd32710024eaa74b97.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/17175d13c0a45575b2727f4286459f1371c2b1d1e7dac6dd32710024eaa74b97.jpg)

![195d30f6faa12e56d26307f2f5ecd92b77dbc16a63d82323e1bbfdc48c5082f2.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/195d30f6faa12e56d26307f2f5ecd92b77dbc16a63d82323e1bbfdc48c5082f2.jpg)

![1b0ec05ff1742a2c53f232b003760695529c1de2546323936737f9988a7fcd26.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/1b0ec05ff1742a2c53f232b003760695529c1de2546323936737f9988a7fcd26.jpg)

![1ebc5ff49846a17a52f3aa1ae82429055a66316ec96fc2ea37c7373d9cbdc243.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/1ebc5ff49846a17a52f3aa1ae82429055a66316ec96fc2ea37c7373d9cbdc243.jpg)

![21c45b160aaaefe530674a7e60f3ac5b60b0a478bd775ce7ac233b44e1500ba4.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/21c45b160aaaefe530674a7e60f3ac5b60b0a478bd775ce7ac233b44e1500ba4.jpg)

![21f3fef5dfd5a92f64ffd2900ee53adf145b616f28e466da901f8addcbeae753.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/21f3fef5dfd5a92f64ffd2900ee53adf145b616f28e466da901f8addcbeae753.jpg)

![23f68fce0427777110cbc95639eb3e5784d3b886f01c7dc05eea23f13504d84f.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/23f68fce0427777110cbc95639eb3e5784d3b886f01c7dc05eea23f13504d84f.jpg)

![26b0b6df9f1c0eeaaf605b3334f8adb83c8feb22c36c0ce5b4069adf508ee32e.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/26b0b6df9f1c0eeaaf605b3334f8adb83c8feb22c36c0ce5b4069adf508ee32e.jpg)

![276fc28b867b5b25a05270f734264b94f7afa4d49a03feef2964666dc44319b6.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/276fc28b867b5b25a05270f734264b94f7afa4d49a03feef2964666dc44319b6.jpg)

![278b8eb2d476a3ae7d9b113b209a67741575614545cc4253ee1f3b48fdc1e1dd.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/278b8eb2d476a3ae7d9b113b209a67741575614545cc4253ee1f3b48fdc1e1dd.jpg)

![29c284a126607051bf037931e380f4869d608c14a11ebb37f8f2be4eb72ef4b9.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/29c284a126607051bf037931e380f4869d608c14a11ebb37f8f2be4eb72ef4b9.jpg)

![3edfe49000e9f2c4ed7bcb36ddc2e2561ee5cd627b58165fbb7fb90c08a74d73.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/3edfe49000e9f2c4ed7bcb36ddc2e2561ee5cd627b58165fbb7fb90c08a74d73.jpg)

![47a4cd871377ca9799f96cc74180f274000cb29b6fc90bd07e0a6271c92717aa.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/47a4cd871377ca9799f96cc74180f274000cb29b6fc90bd07e0a6271c92717aa.jpg)

![560fa8da40525279c0fad9e9d283d7bfd23d7560a700bfdba310ea6ef17cebe8.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/560fa8da40525279c0fad9e9d283d7bfd23d7560a700bfdba310ea6ef17cebe8.jpg)

![58ce4cee03dbcceff49a9ba015846b75b3e452a10df21ca99983c21241f18aea.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/58ce4cee03dbcceff49a9ba015846b75b3e452a10df21ca99983c21241f18aea.jpg)

![5d72bc6850f3b29b018617180d3136aead2fb961d8698bf1f054108bd710ef39.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/5d72bc6850f3b29b018617180d3136aead2fb961d8698bf1f054108bd710ef39.jpg)

![78506750815f57857128bab11c1a432a10405b91ca7034654be58eeb28a3658b.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/78506750815f57857128bab11c1a432a10405b91ca7034654be58eeb28a3658b.jpg)

![7ac38ce557e5cc12854f24d79f94baf4153930c016e8c3107cfd213884236000.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/7ac38ce557e5cc12854f24d79f94baf4153930c016e8c3107cfd213884236000.jpg)

![814149c33f7772fe573db1d49eae156cfca92270324961e8ca2669cd95362a01.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/814149c33f7772fe573db1d49eae156cfca92270324961e8ca2669cd95362a01.jpg)

![895fc084780251920a69fbf313310d0d2b765cd1ebf597a7e38703c3a0fe01a2.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/895fc084780251920a69fbf313310d0d2b765cd1ebf597a7e38703c3a0fe01a2.jpg)

![8ab5bbe197207d0e5203c2d6e2e4c82875b12b35d5fc68000b96f3801dbf7a26.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/8ab5bbe197207d0e5203c2d6e2e4c82875b12b35d5fc68000b96f3801dbf7a26.jpg)

![9bee72e64c7954029f80b5d1d6a354dfc68c35e3f69061dadc8a282439cfd88e.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/9bee72e64c7954029f80b5d1d6a354dfc68c35e3f69061dadc8a282439cfd88e.jpg)

![a191133518d6a6e26b99e12121c19c4e3452614bacc7831621bd877fa8f3ea77.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/a191133518d6a6e26b99e12121c19c4e3452614bacc7831621bd877fa8f3ea77.jpg)

![b1176d8b01e3a1fab0036f3a9b97846f6102495c230d4d489c593addfd652329.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/b1176d8b01e3a1fab0036f3a9b97846f6102495c230d4d489c593addfd652329.jpg)

![bba01fefef256ec5e8e68922d0d6d3c4b7b901b0e92c607f13c1871008b655fd.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/bba01fefef256ec5e8e68922d0d6d3c4b7b901b0e92c607f13c1871008b655fd.jpg)

![cdb649b4a33c9a705fe181c9b3bae77576c53872c0c08c69905f7389f3c11871.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/cdb649b4a33c9a705fe181c9b3bae77576c53872c0c08c69905f7389f3c11871.jpg)

![d0101bbb0a519b5499bbd5d6cf33e2833148fff480c8809d1199af02ec554c5e.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/d0101bbb0a519b5499bbd5d6cf33e2833148fff480c8809d1199af02ec554c5e.jpg)

![d0e186ef82aba5d0764ffffdb9206597061036d52bb916f85824608e05f3882c.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/d0e186ef82aba5d0764ffffdb9206597061036d52bb916f85824608e05f3882c.jpg)

![d32027507654f7930d0bd4396a6ebfb98032beea3c3bf427326670c399239417.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/d32027507654f7930d0bd4396a6ebfb98032beea3c3bf427326670c399239417.jpg)

![dfedffca8bf70fb390cf3f1fa01811c4d9ba2b0b6a40c78e9d4730387f06e3d4.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/dfedffca8bf70fb390cf3f1fa01811c4d9ba2b0b6a40c78e9d4730387f06e3d4.jpg)

![e3851453ce1fb7f5e10187bd86bc12042cc383b223a4ee223b52b986dca29e4f.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/e3851453ce1fb7f5e10187bd86bc12042cc383b223a4ee223b52b986dca29e4f.jpg)

![e51661f15df7110da704832936e38e11538cdbbd2aa08a9a10fb947a43730e0d.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/e51661f15df7110da704832936e38e11538cdbbd2aa08a9a10fb947a43730e0d.jpg)

![e5f579f58923b3f716dfb0f9acf41fae14ba59a49bf116e2ca71c010f731e99a.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/e5f579f58923b3f716dfb0f9acf41fae14ba59a49bf116e2ca71c010f731e99a.jpg)

![e8af30916f9b1b947c56579899412d1a42e06f83ee160b9c1252072b5259fc61.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/e8af30916f9b1b947c56579899412d1a42e06f83ee160b9c1252072b5259fc61.jpg)

![edc5550b5f392828816d60a5877ba32edbf9a2aae028f1ffad74208f029595f9.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/edc5550b5f392828816d60a5877ba32edbf9a2aae028f1ffad74208f029595f9.jpg)

![f0f37fee77cf59e4ebe3147599e83c0f8b0f6b3aa17c33072a8c587ecbc9cc47.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/f0f37fee77cf59e4ebe3147599e83c0f8b0f6b3aa17c33072a8c587ecbc9cc47.jpg)

![f810b2e87d402941e5844853bdd38f0b2e38b801aee7fce0e6af8575bf93576d.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/f810b2e87d402941e5844853bdd38f0b2e38b801aee7fce0e6af8575bf93576d.jpg)

![f9474f155af9b45d9a61fdd74814b079334738f205456539331a180999180fd7.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/f9474f155af9b45d9a61fdd74814b079334738f205456539331a180999180fd7.jpg)

![fe634e5fc6e0c92d699376f009f1621b7e5f3c069d308c2f6f3301bcffd72f34.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/images/fe634e5fc6e0c92d699376f009f1621b7e5f3c069d308c2f6f3301bcffd72f34.jpg)

### Tables

![157edde3ac438fd77bee14848aba5c62336e3731074e41d398fa907e9184ab04.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/tables/157edde3ac438fd77bee14848aba5c62336e3731074e41d398fa907e9184ab04.jpg)

![25d5a2c93f9273bf8338e1dea1ddb8c97544bba37f400465debaef6146705664.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/tables/25d5a2c93f9273bf8338e1dea1ddb8c97544bba37f400465debaef6146705664.jpg)

![4a42fa3469414db7933ca32fc5f2638d5ee87a16ce3359f3b41b2f8978f25c4e.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/tables/4a42fa3469414db7933ca32fc5f2638d5ee87a16ce3359f3b41b2f8978f25c4e.jpg)

![99cf7afc891a068a3bfab08837d601d229c6c7eb5bc24ac80a65eb674e63d67b.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/tables/99cf7afc891a068a3bfab08837d601d229c6c7eb5bc24ac80a65eb674e63d67b.jpg)

![a9272b5cade461549b5f2d2d59652899cba80d0c898171ca05faf9df01a41326.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/tables/a9272b5cade461549b5f2d2d59652899cba80d0c898171ca05faf9df01a41326.jpg)

![b970dfda4121b6f6f22f6764fc8c5cc16afb8de477db2b9971c328f5cc2e3fbe.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/tables/b970dfda4121b6f6f22f6764fc8c5cc16afb8de477db2b9971c328f5cc2e3fbe.jpg)

![ba4c170beb34ce4ed08e1422cdd18a448ac88f49680c477e4184c26a88675934.jpg](../icml_results/2943_MedScope_%20Incentivizing%20_Think%20with%20Videos_%20for%20Clinical%20Reasoning%20via%20Coarse-to-Fine%20Tool%20Calling/tables/ba4c170beb34ce4ed08e1422cdd18a448ac88f49680c477e4184c26a88675934.jpg)

<a id="paper-2906"></a>

## SetPO: Set-Level Policy Optimization for Diversity-Preserving LLM Reasoning


### Images

![1e498d69e25a9ed8507ea6b54679131e0544451f73059f7887c0b5458439d7b2.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/images/1e498d69e25a9ed8507ea6b54679131e0544451f73059f7887c0b5458439d7b2.jpg)

![3497a46400f590b641392e3b7d8814055e20914e171c64ffae42fefeb52c4334.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/images/3497a46400f590b641392e3b7d8814055e20914e171c64ffae42fefeb52c4334.jpg)

![b240f0963ff711b8f0c3f6362a1384def09b7fc4879296398ae5874f834fedec.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/images/b240f0963ff711b8f0c3f6362a1384def09b7fc4879296398ae5874f834fedec.jpg)

![bdfcb707b2b8ed661fa719c4e77b769351cc44b5e48c186cb0daf449a86efab9.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/images/bdfcb707b2b8ed661fa719c4e77b769351cc44b5e48c186cb0daf449a86efab9.jpg)

![c00f2c1baef6008dc176e00e94d635d9344017cba0f1268a2556120806eb0cfb.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/images/c00f2c1baef6008dc176e00e94d635d9344017cba0f1268a2556120806eb0cfb.jpg)

![d474608d13d1de7d9872dd2f42ded461e8a5896ff7ad253f966a92618421618d.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/images/d474608d13d1de7d9872dd2f42ded461e8a5896ff7ad253f966a92618421618d.jpg)

![d4d711f0d9c843dadd5ddc440c00a6ce88e955c8c2c5d8665a6b649c8fc9b374.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/images/d4d711f0d9c843dadd5ddc440c00a6ce88e955c8c2c5d8665a6b649c8fc9b374.jpg)

![e92debc3b9195efbb368f8b3b8592c43109129c070d5bb9292858eed21df0f43.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/images/e92debc3b9195efbb368f8b3b8592c43109129c070d5bb9292858eed21df0f43.jpg)

![f154aa6b26be01dd9b3b9a9b610d15bf51cdd3d0c5de8b2216c3594ba1c62845.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/images/f154aa6b26be01dd9b3b9a9b610d15bf51cdd3d0c5de8b2216c3594ba1c62845.jpg)

![f97479c4452733e55a6b67d45fee6b52b1047f12c29f7a4049166bfbc09a9970.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/images/f97479c4452733e55a6b67d45fee6b52b1047f12c29f7a4049166bfbc09a9970.jpg)

### Tables

![2d923ddcc1fac206bc3fd3f06d136fdb7b67078c89ce9083266d59fe161d2c7f.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/tables/2d923ddcc1fac206bc3fd3f06d136fdb7b67078c89ce9083266d59fe161d2c7f.jpg)

![49d91ce277b76754827eb3b1bfa47a7868f641fcc1cc739f4239bb10ec4bc1c0.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/tables/49d91ce277b76754827eb3b1bfa47a7868f641fcc1cc739f4239bb10ec4bc1c0.jpg)

![69f7bda9965af843aedc05abbadc6f89b33d354987669d43e8f0cb40af2a0c35.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/tables/69f7bda9965af843aedc05abbadc6f89b33d354987669d43e8f0cb40af2a0c35.jpg)

![733e369f623ef3b53e77e437d9819a8142582dfe481a32c7ca56004dffa1c358.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/tables/733e369f623ef3b53e77e437d9819a8142582dfe481a32c7ca56004dffa1c358.jpg)

![7db3bff3750029418c7f67db77bf6f762a02452d9954f812613edfa64cfe0c9c.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/tables/7db3bff3750029418c7f67db77bf6f762a02452d9954f812613edfa64cfe0c9c.jpg)

![a3721a5f1f9313bdf38019f5fd3e2a225706c936dd77c748ea05956f58a8d356.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/tables/a3721a5f1f9313bdf38019f5fd3e2a225706c936dd77c748ea05956f58a8d356.jpg)

![ab36c13c88f4c953a84a82c2cdbf7bfdeeaa6564dc4ec3c57b7a4c3f75bc52a7.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/tables/ab36c13c88f4c953a84a82c2cdbf7bfdeeaa6564dc4ec3c57b7a4c3f75bc52a7.jpg)

![e0f44cdb8699b6a6ac8479d3f30def4b687d1c355b52db40233eebe35d9e1f33.jpg](../icml_results/2944_SetPO_%20Set-Level%20Policy%20Optimization%20for%20Diversity-Preserving%20LLM%20Reasoning/tables/e0f44cdb8699b6a6ac8479d3f30def4b687d1c355b52db40233eebe35d9e1f33.jpg)

<a id="paper-2907"></a>

## FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents


### Images

![14d34500fb07fe126f7becb97dfa33e295901600718de6c775e660f1481fc744.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/images/14d34500fb07fe126f7becb97dfa33e295901600718de6c775e660f1481fc744.jpg)

![197461042db57e3ffcd204a5f7b0b7dcc7378196175797e5fa18db4462fb40f2.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/images/197461042db57e3ffcd204a5f7b0b7dcc7378196175797e5fa18db4462fb40f2.jpg)

![201e35f97d9225e4b3e339e6b6f76600063734a25716dbf4d0d0a39ff75aaba9.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/images/201e35f97d9225e4b3e339e6b6f76600063734a25716dbf4d0d0a39ff75aaba9.jpg)

![288aec1c717c6fe85167622b9fb4241c4ff0dfcd88fc9fc396a855ae34c64b9a.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/images/288aec1c717c6fe85167622b9fb4241c4ff0dfcd88fc9fc396a855ae34c64b9a.jpg)

![5442fa66d0891af6e74adb4ff86b3f36919f7c8f58389e8534ac9a4c79458763.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/images/5442fa66d0891af6e74adb4ff86b3f36919f7c8f58389e8534ac9a4c79458763.jpg)

![6be5a83216c5e598e838400e7b3c5523cf464991578c113d68586b681535888c.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/images/6be5a83216c5e598e838400e7b3c5523cf464991578c113d68586b681535888c.jpg)

![af7793b3c4ea5528be482590eb7732666452d0d612c90a7a6cccbfeaa7eab8ed.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/images/af7793b3c4ea5528be482590eb7732666452d0d612c90a7a6cccbfeaa7eab8ed.jpg)

### Tables

![13398f1a19ca3c6e119b98cdd0ec501f911ff0522c4a9ad38db210c66b6628fe.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/tables/13398f1a19ca3c6e119b98cdd0ec501f911ff0522c4a9ad38db210c66b6628fe.jpg)

![2ff6347c5856e182d66caa39297d1bd68b8816b2cf0090affa052808811f2a61.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/tables/2ff6347c5856e182d66caa39297d1bd68b8816b2cf0090affa052808811f2a61.jpg)

![5243dba6293e94047b4c68c168f41f73f97f227c64cdc11ebc7bd53da379308b.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/tables/5243dba6293e94047b4c68c168f41f73f97f227c64cdc11ebc7bd53da379308b.jpg)

![6e849db466e3e2e7de8cd155f5c9caa698e377f8d952c332d4522ed04d4ee87d.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/tables/6e849db466e3e2e7de8cd155f5c9caa698e377f8d952c332d4522ed04d4ee87d.jpg)

![72d9b0f5a782d7369e9663cc9b6991c57904bda3512e792413dd78a4fee9ae68.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/tables/72d9b0f5a782d7369e9663cc9b6991c57904bda3512e792413dd78a4fee9ae68.jpg)

![8b66dc2c7967cd2447d07c0116df957fefba9f3193d9782e7fbac33122666618.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/tables/8b66dc2c7967cd2447d07c0116df957fefba9f3193d9782e7fbac33122666618.jpg)

![9036a1714c7cd6f9cb6e12a45d7617e45ebafd0713d39fae2bf80ffdf70d58e8.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/tables/9036a1714c7cd6f9cb6e12a45d7617e45ebafd0713d39fae2bf80ffdf70d58e8.jpg)

![b12e558b0524735a667db8af91629a6f4629cda208f724f61c8307cb049a8332.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/tables/b12e558b0524735a667db8af91629a6f4629cda208f724f61c8307cb049a8332.jpg)

![bbc1b2be4966df52e25b7c11918d96945a067da57a62cdc81320d5e1d9b79a94.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/tables/bbc1b2be4966df52e25b7c11918d96945a067da57a62cdc81320d5e1d9b79a94.jpg)

![d454a5ab329696936288238479cae2beda3b5ee79306bd2b41ef7bbf85fdc15f.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/tables/d454a5ab329696936288238479cae2beda3b5ee79306bd2b41ef7bbf85fdc15f.jpg)

![f4aa258aa8550fb0cedd02ced321c0d895a53a128ec0ef2f0ebc9b6ff6e140ac.jpg](../icml_results/2945_FT-Dojo_%20Towards%20Autonomous%20LLM%20Fine-Tuning%20with%20Language%20Agents/tables/f4aa258aa8550fb0cedd02ced321c0d895a53a128ec0ef2f0ebc9b6ff6e140ac.jpg)

<a id="paper-2908"></a>

## ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment


### Images

![1d8a73957cd46f973de179b2e697c674beca6cadf283c78fd76e78dc4ac019fa.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/images/1d8a73957cd46f973de179b2e697c674beca6cadf283c78fd76e78dc4ac019fa.jpg)

![528d1b417440426de21bd416bedf4dbd36329a6b47fdd56db62126c412bc412c.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/images/528d1b417440426de21bd416bedf4dbd36329a6b47fdd56db62126c412bc412c.jpg)

![61cf3b55a118f221ff5981c604e6eef5ed52543590ee40f99cdf2b7bf6bf1dc9.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/images/61cf3b55a118f221ff5981c604e6eef5ed52543590ee40f99cdf2b7bf6bf1dc9.jpg)

![62c985b38e63c8bc62cd363a3b495f84bdcf60f3aef344a980fe49b37e6218c1.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/images/62c985b38e63c8bc62cd363a3b495f84bdcf60f3aef344a980fe49b37e6218c1.jpg)

![8616e548bdc24afaff78df0c9a499110f87bd38c99da83f07e929e77b4ba9ad0.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/images/8616e548bdc24afaff78df0c9a499110f87bd38c99da83f07e929e77b4ba9ad0.jpg)

![96d319c9bd348d983464e0fc4650ad83829ccf0e6cd7f32070b1a096a457d64f.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/images/96d319c9bd348d983464e0fc4650ad83829ccf0e6cd7f32070b1a096a457d64f.jpg)

![a6c550544c124c5eb56e81d2f30adf505c7aefde8d35f70166ea0005958d5308.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/images/a6c550544c124c5eb56e81d2f30adf505c7aefde8d35f70166ea0005958d5308.jpg)

![aca8ce76d456150f0a2ac17b03a395ed76ce04f27b7397a322c73098717c5ddb.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/images/aca8ce76d456150f0a2ac17b03a395ed76ce04f27b7397a322c73098717c5ddb.jpg)

![f145323bbf86d1048fce77dbca534734db52f3b3b4f56efd5ec606df71f9859d.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/images/f145323bbf86d1048fce77dbca534734db52f3b3b4f56efd5ec606df71f9859d.jpg)

![fab91935ebe62cc2b59414af9b3472e0c4b956ab622730fa8da4042779cd43ad.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/images/fab91935ebe62cc2b59414af9b3472e0c4b956ab622730fa8da4042779cd43ad.jpg)

### Tables

![06751b1c526336249e91d9f75e482404f58e16fcbf021d32fc0767042c55ff66.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/tables/06751b1c526336249e91d9f75e482404f58e16fcbf021d32fc0767042c55ff66.jpg)

![0ceabdb25e653b733cf64c92bbd4e0dee631e529ebedd1a0bb8daa6752d64834.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/tables/0ceabdb25e653b733cf64c92bbd4e0dee631e529ebedd1a0bb8daa6752d64834.jpg)

![10d65e8b9e5e4cb8104b1a1b7d7c8dfb92a4d272b79ed97c504616701fc74c64.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/tables/10d65e8b9e5e4cb8104b1a1b7d7c8dfb92a4d272b79ed97c504616701fc74c64.jpg)

![1d1ff1e998f0cf84643e2d391f38ca4bcc9e5c2d7a63d3191c5854710dffeb1f.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/tables/1d1ff1e998f0cf84643e2d391f38ca4bcc9e5c2d7a63d3191c5854710dffeb1f.jpg)

![5c48a7c89b1afdef5271d5d8dd8ffbdb13fa92c47b2727601c0157c2d4cc2794.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/tables/5c48a7c89b1afdef5271d5d8dd8ffbdb13fa92c47b2727601c0157c2d4cc2794.jpg)

![6c22036864ab75909222887f766e4ed6044490def3ffc1d8d5da31b843ef5307.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/tables/6c22036864ab75909222887f766e4ed6044490def3ffc1d8d5da31b843ef5307.jpg)

![8db7f14ad3ee2736981298b26d55c02a12a341d94b996bc5a98640c48e9b0beb.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/tables/8db7f14ad3ee2736981298b26d55c02a12a341d94b996bc5a98640c48e9b0beb.jpg)

![a53e676e13ca55ec6e11a62c7fd70f1c19a7fa6fbd53113a397f76ff719dcb43.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/tables/a53e676e13ca55ec6e11a62c7fd70f1c19a7fa6fbd53113a397f76ff719dcb43.jpg)

![ca2511d33af6c3377685037dfef3eac3a34aa00367d34df4fba160b6506d5469.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/tables/ca2511d33af6c3377685037dfef3eac3a34aa00367d34df4fba160b6506d5469.jpg)

![f5157320059bf8e2417672e17a55244b0e937f8972a3afbac6424e1d3f5fb8d9.jpg](../icml_results/2946_ETS_%20Energy-Guided%20Test-Time%20Scaling%20for%20Training-Free%20RL%20Alignment/tables/f5157320059bf8e2417672e17a55244b0e937f8972a3afbac6424e1d3f5fb8d9.jpg)

<a id="paper-2909"></a>

## Geometrically Constrained Stenosis Editing in Coronary Angiography via Entropic Optimal Transport


### Images

![0e31e6ebdfa09eb546fd3b0fb3bb18af6add4c10ed53821ff08df0c0f425a228.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/images/0e31e6ebdfa09eb546fd3b0fb3bb18af6add4c10ed53821ff08df0c0f425a228.jpg)

![110db4880a792c9ac43b14a82f7061f8f76700477fac289335e923df003b468b.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/images/110db4880a792c9ac43b14a82f7061f8f76700477fac289335e923df003b468b.jpg)

![226232dfbb852633163e9b410699bb391e4d4842b9ceaccc4734b7a86c29217e.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/images/226232dfbb852633163e9b410699bb391e4d4842b9ceaccc4734b7a86c29217e.jpg)

![25d25812b2f3e8ad201f8991b2c3290e52b1a5853aaac2cf58074c320e31a208.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/images/25d25812b2f3e8ad201f8991b2c3290e52b1a5853aaac2cf58074c320e31a208.jpg)

![2a6307d8f111a8b420fa037faa881921429f652072823d15b96c75c11733880c.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/images/2a6307d8f111a8b420fa037faa881921429f652072823d15b96c75c11733880c.jpg)

![4578f717156f1d272c10a1932c34da367e12186cdd2d7bff642620eb70f7da22.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/images/4578f717156f1d272c10a1932c34da367e12186cdd2d7bff642620eb70f7da22.jpg)

![58dde1a75c94fa8a2f834d4b58a55df25c89f8902557a100599c9d567b0d5f4b.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/images/58dde1a75c94fa8a2f834d4b58a55df25c89f8902557a100599c9d567b0d5f4b.jpg)

![60d2158da9cac5ecfe8e0c518a699072c82d214e5307ea7f204d742c51f858a9.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/images/60d2158da9cac5ecfe8e0c518a699072c82d214e5307ea7f204d742c51f858a9.jpg)

![61b5cb155d99092d4d7672ca554dc8f89b945b4b15b2b1cdc651bfddf0935f3a.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/images/61b5cb155d99092d4d7672ca554dc8f89b945b4b15b2b1cdc651bfddf0935f3a.jpg)

![b975143d27d404c126795b18bdaabe75349cb05007a292524005ce22e5977b4a.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/images/b975143d27d404c126795b18bdaabe75349cb05007a292524005ce22e5977b4a.jpg)

![cefb8ea086ba51ed8aa0464e4a823c3ca340af0f0af68dd858d67a504cd46461.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/images/cefb8ea086ba51ed8aa0464e4a823c3ca340af0f0af68dd858d67a504cd46461.jpg)

### Tables

![099909f7d999a19e2770b5b626e36c81116590ab3ca5251b70ea5c1a9c7eba0b.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/tables/099909f7d999a19e2770b5b626e36c81116590ab3ca5251b70ea5c1a9c7eba0b.jpg)

![1134105723c28449bdadeae77668e5512bdee7ba4cfe68455091a5a391eb19d1.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/tables/1134105723c28449bdadeae77668e5512bdee7ba4cfe68455091a5a391eb19d1.jpg)

![1d98fd350534fe697e0a22048765b3d056e5a55b72a4a58edee051826f12e4ee.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/tables/1d98fd350534fe697e0a22048765b3d056e5a55b72a4a58edee051826f12e4ee.jpg)

![1e86f3bb6876e30b024b99a343d75e1b703031f19e402402c4069f84e53563b2.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/tables/1e86f3bb6876e30b024b99a343d75e1b703031f19e402402c4069f84e53563b2.jpg)

![450c834081192b8aa0258eba34f63f09876138b7231c767993d901a9ef02ed39.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/tables/450c834081192b8aa0258eba34f63f09876138b7231c767993d901a9ef02ed39.jpg)

![4f724f7c6b67dedaf67f219beebcdaa969f6c41aefccad3fda1994c4a1ac18a0.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/tables/4f724f7c6b67dedaf67f219beebcdaa969f6c41aefccad3fda1994c4a1ac18a0.jpg)

![58ea8eb9fadc9f1dae10376df64b73388110dd85311b262a98af6448b1c33ac3.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/tables/58ea8eb9fadc9f1dae10376df64b73388110dd85311b262a98af6448b1c33ac3.jpg)

![71087eee94cf625d9905c895df2b9f10ce952283224d7413ac299ef9f569d743.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/tables/71087eee94cf625d9905c895df2b9f10ce952283224d7413ac299ef9f569d743.jpg)

![a0b2bd2a770ebf05ff0072e4fe0f7688419d29d6c81998044dec9a9be6ed9f3c.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/tables/a0b2bd2a770ebf05ff0072e4fe0f7688419d29d6c81998044dec9a9be6ed9f3c.jpg)

![d175c8b6380c5d02dcc31132128a367e7544c9e983cb574be6f0fd6f2927c60e.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/tables/d175c8b6380c5d02dcc31132128a367e7544c9e983cb574be6f0fd6f2927c60e.jpg)

![d240e377fe3d234665f26602ac41717707906f379af3ccbb4afe43a6c387ae40.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/tables/d240e377fe3d234665f26602ac41717707906f379af3ccbb4afe43a6c387ae40.jpg)

![f80028d45864bcfe057f3be3da03cc1ee02599ae0f3e5a38768aea3d6d2be14c.jpg](../icml_results/2947_Geometrically%20Constrained%20Stenosis%20Editing%20in%20Coronary%20Angiography%20via%20Entropic%20Optimal%20Transport/tables/f80028d45864bcfe057f3be3da03cc1ee02599ae0f3e5a38768aea3d6d2be14c.jpg)

<a id="paper-2910"></a>

## State-Dependent Safety Failures in Multi-Turn Language Model Interaction


### Images

![06b773f83e90f92cb998fb470b1ece42348cfd37f5737afbf38cd52e18649d51.jpg](../icml_results/2948_State-Dependent%20Safety%20Failures%20in%20Multi-Turn%20Language%20Model%20Interaction/images/06b773f83e90f92cb998fb470b1ece42348cfd37f5737afbf38cd52e18649d51.jpg)

![195b57d241fa5529fa8fa3192bd046a5a09c315b9459842cb729ec4ca6fe214c.jpg](../icml_results/2948_State-Dependent%20Safety%20Failures%20in%20Multi-Turn%20Language%20Model%20Interaction/images/195b57d241fa5529fa8fa3192bd046a5a09c315b9459842cb729ec4ca6fe214c.jpg)

![39837dcb340dc9257b0602b8a17fc8fe3e8cff1419b180dc6b5fe758e67abc0c.jpg](../icml_results/2948_State-Dependent%20Safety%20Failures%20in%20Multi-Turn%20Language%20Model%20Interaction/images/39837dcb340dc9257b0602b8a17fc8fe3e8cff1419b180dc6b5fe758e67abc0c.jpg)

![45b0dc8356978c93b2b4cfed568fcff8ebc4964c784c0956bee3d98dda3796b1.jpg](../icml_results/2948_State-Dependent%20Safety%20Failures%20in%20Multi-Turn%20Language%20Model%20Interaction/images/45b0dc8356978c93b2b4cfed568fcff8ebc4964c784c0956bee3d98dda3796b1.jpg)

![615d9962af370f5c2a6ca617eb5c88b3ab34e46c1edc21c4983c0a69d66df8fb.jpg](../icml_results/2948_State-Dependent%20Safety%20Failures%20in%20Multi-Turn%20Language%20Model%20Interaction/images/615d9962af370f5c2a6ca617eb5c88b3ab34e46c1edc21c4983c0a69d66df8fb.jpg)

![62988c100f676cf1db4d4b17784f4ce609f9f7de627378b08a6a5b6063d03950.jpg](../icml_results/2948_State-Dependent%20Safety%20Failures%20in%20Multi-Turn%20Language%20Model%20Interaction/images/62988c100f676cf1db4d4b17784f4ce609f9f7de627378b08a6a5b6063d03950.jpg)

![c198b823a2d697aa43bee67aef322ac8f3301ced81eed463f4550ad90a114705.jpg](../icml_results/2948_State-Dependent%20Safety%20Failures%20in%20Multi-Turn%20Language%20Model%20Interaction/images/c198b823a2d697aa43bee67aef322ac8f3301ced81eed463f4550ad90a114705.jpg)

![e6d0b21440b4ad71d0133f083ffe2b9e5aedbd6717fd30b181296aebf0a49c8d.jpg](../icml_results/2948_State-Dependent%20Safety%20Failures%20in%20Multi-Turn%20Language%20Model%20Interaction/images/e6d0b21440b4ad71d0133f083ffe2b9e5aedbd6717fd30b181296aebf0a49c8d.jpg)

![ed80358e669dce0f2577dd9f02097ea6c8b555661ee6f83228953f7b9f23f4b0.jpg](../icml_results/2948_State-Dependent%20Safety%20Failures%20in%20Multi-Turn%20Language%20Model%20Interaction/images/ed80358e669dce0f2577dd9f02097ea6c8b555661ee6f83228953f7b9f23f4b0.jpg)

![fe669a496e087c8700d57c00355a5d7ab917930f94d08faf0b26f9d6b2d968f2.jpg](../icml_results/2948_State-Dependent%20Safety%20Failures%20in%20Multi-Turn%20Language%20Model%20Interaction/images/fe669a496e087c8700d57c00355a5d7ab917930f94d08faf0b26f9d6b2d968f2.jpg)

### Tables

![407990ccf57062b4ef3a6a96767509c4ffddde082d5f0d8d518b2ec0013c38be.jpg](../icml_results/2948_State-Dependent%20Safety%20Failures%20in%20Multi-Turn%20Language%20Model%20Interaction/tables/407990ccf57062b4ef3a6a96767509c4ffddde082d5f0d8d518b2ec0013c38be.jpg)

![4be67b426ac223a74e366b6cf3c8c3985ee866ee2a4b1af798e02cab52d3f1d7.jpg](../icml_results/2948_State-Dependent%20Safety%20Failures%20in%20Multi-Turn%20Language%20Model%20Interaction/tables/4be67b426ac223a74e366b6cf3c8c3985ee866ee2a4b1af798e02cab52d3f1d7.jpg)

![df5fef94b0f0816e0a3912ed30348755a8feb701736c6555a21d2782a34b72ea.jpg](../icml_results/2948_State-Dependent%20Safety%20Failures%20in%20Multi-Turn%20Language%20Model%20Interaction/tables/df5fef94b0f0816e0a3912ed30348755a8feb701736c6555a21d2782a34b72ea.jpg)

![f19907b7f01efd1c165a41e0b73e00efc63593c569b5110c076dc8c4e46ac0c2.jpg](../icml_results/2948_State-Dependent%20Safety%20Failures%20in%20Multi-Turn%20Language%20Model%20Interaction/tables/f19907b7f01efd1c165a41e0b73e00efc63593c569b5110c076dc8c4e46ac0c2.jpg)

<a id="paper-2911"></a>

## Task-Aware Preference Calibration for Direct Preference Optimization


### Images

![552c71971a2189f83430bfdd3e202f0bd80de646a5b0741a3526f456b8119889.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/images/552c71971a2189f83430bfdd3e202f0bd80de646a5b0741a3526f456b8119889.jpg)

![7355a7873d3d74229625917874536240a7bda9c67e9a6f7564290871524e86de.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/images/7355a7873d3d74229625917874536240a7bda9c67e9a6f7564290871524e86de.jpg)

![9d5206901deb0405aa9df44120a220b292d5d08a3bc3ea4ae5f82b362b315246.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/images/9d5206901deb0405aa9df44120a220b292d5d08a3bc3ea4ae5f82b362b315246.jpg)

![cd039c94915899c13ecde5ea15e590b3c84cd1a8ab77f7cd591eb5a67793d940.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/images/cd039c94915899c13ecde5ea15e590b3c84cd1a8ab77f7cd591eb5a67793d940.jpg)

![cf5f2fda8e043e691d4c1c97ce54094b85a3d3e302b1bc8fdb8cba2aae534ea4.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/images/cf5f2fda8e043e691d4c1c97ce54094b85a3d3e302b1bc8fdb8cba2aae534ea4.jpg)

![f2f019d8985a2c5bdeb92655d31af86903b7e673ae349a54ce962e33ce26ed09.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/images/f2f019d8985a2c5bdeb92655d31af86903b7e673ae349a54ce962e33ce26ed09.jpg)

### Tables

![16e4502d71def175839c73f7159e802e36e5b3e2477d6375a37437abdca7ccdd.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/16e4502d71def175839c73f7159e802e36e5b3e2477d6375a37437abdca7ccdd.jpg)

![308e36e268d287108ca8c6c1cd7a8d8aee2f482f4511723eada1635960a6bfee.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/308e36e268d287108ca8c6c1cd7a8d8aee2f482f4511723eada1635960a6bfee.jpg)

![32b004d450fe2941a85cbbb1e14c675303b2004e5ee927909a6cd7f81da630ba.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/32b004d450fe2941a85cbbb1e14c675303b2004e5ee927909a6cd7f81da630ba.jpg)

![3a6b59adb63071a04e44066c23cf3c5af278c2904f82e40011fe415c01326107.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/3a6b59adb63071a04e44066c23cf3c5af278c2904f82e40011fe415c01326107.jpg)

![3adaf1c42103a063cabc55024af232cd10b81bf3085a1aced0c4dc71d4dbbea8.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/3adaf1c42103a063cabc55024af232cd10b81bf3085a1aced0c4dc71d4dbbea8.jpg)

![6267b66a73d821ccf256cbc875780ab590a7fc296816501193d1d0536e6ed9c1.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/6267b66a73d821ccf256cbc875780ab590a7fc296816501193d1d0536e6ed9c1.jpg)

![7a5b82e5817fa5f1b9accd637472d960f060caefcb31005eddabfa6ca72fe800.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/7a5b82e5817fa5f1b9accd637472d960f060caefcb31005eddabfa6ca72fe800.jpg)

![7dec3fe6bf8105ad62e3de475db9e891a0e75290ac8cfe18f094b676d95d4367.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/7dec3fe6bf8105ad62e3de475db9e891a0e75290ac8cfe18f094b676d95d4367.jpg)

![8504017cdeab7b9416302adc624e668c5e5c2019124248a12f310da92ba366ee.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/8504017cdeab7b9416302adc624e668c5e5c2019124248a12f310da92ba366ee.jpg)

![92b4f511910f61ab9d0fbdb5a0ee904d3ccf17bd1e9b40fb567513e78dbf12d6.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/92b4f511910f61ab9d0fbdb5a0ee904d3ccf17bd1e9b40fb567513e78dbf12d6.jpg)

![9fbc591bec31f96ae74f9282c92f090bd9876a3350ff9f00ddb89613b5e65995.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/9fbc591bec31f96ae74f9282c92f090bd9876a3350ff9f00ddb89613b5e65995.jpg)

![af41849148ef69f5222d7c0200fa31af5612ebc2f4f463b015717ee9ce1328bb.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/af41849148ef69f5222d7c0200fa31af5612ebc2f4f463b015717ee9ce1328bb.jpg)

![b09c395d6746762d841121489a89d50e295e26eaaae334b6d281d688d4afeb44.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/b09c395d6746762d841121489a89d50e295e26eaaae334b6d281d688d4afeb44.jpg)

![b31bbb502cb3290d33563f35afbee8fdc570c28524720b40ea3a299da5cbee18.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/b31bbb502cb3290d33563f35afbee8fdc570c28524720b40ea3a299da5cbee18.jpg)

![c43c550735bf3308b6b02270084abf1c33c9d1819341fb62ac13573bf4f5c13b.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/c43c550735bf3308b6b02270084abf1c33c9d1819341fb62ac13573bf4f5c13b.jpg)

![dda3772ec7c19d081b424815473c2991bd46b861f2cb2b307a070eaeead5210f.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/dda3772ec7c19d081b424815473c2991bd46b861f2cb2b307a070eaeead5210f.jpg)

![dff247bf7a739d1850933b63f9af1aef1ed38a15ea08a139d7be139a92d0fcd1.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/dff247bf7a739d1850933b63f9af1aef1ed38a15ea08a139d7be139a92d0fcd1.jpg)

![e323cfd139e849679b16b13c64290ef020dc527275c876c124829af64766634d.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/e323cfd139e849679b16b13c64290ef020dc527275c876c124829af64766634d.jpg)

![ee67ff067d11d19820a4cc48afe3aa418ca2e805e054702e2caac206a5591f93.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/ee67ff067d11d19820a4cc48afe3aa418ca2e805e054702e2caac206a5591f93.jpg)

![fa3128c65793d7e100f6505f4652f72198aa2f61713780eac6891189b09fcb03.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/fa3128c65793d7e100f6505f4652f72198aa2f61713780eac6891189b09fcb03.jpg)

![ff01022fe8d1d1e0fa3aa5e70acb531695c0689fd7e1965b9aa1e3a62f4d8e19.jpg](../icml_results/2949_Task-Aware%20Preference%20Calibration%20for%20Direct%20Preference%20Optimization/tables/ff01022fe8d1d1e0fa3aa5e70acb531695c0689fd7e1965b9aa1e3a62f4d8e19.jpg)

<a id="paper-2912"></a>

## RePo: Language Models with Context Re-Positioning


### Images

![12b04dbc94af98e42e33316b8ffc78b7e08fdcc8331d7bce6e8c0dfc6c76ea60.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/images/12b04dbc94af98e42e33316b8ffc78b7e08fdcc8331d7bce6e8c0dfc6c76ea60.jpg)

![190dfd830157a9731e851ff67bc479a287e158079ed900022474d1643aa3b608.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/images/190dfd830157a9731e851ff67bc479a287e158079ed900022474d1643aa3b608.jpg)

![50ca89ad74b463488cc0912b9840f93500536141bcbbc9ed378bb4181cc20457.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/images/50ca89ad74b463488cc0912b9840f93500536141bcbbc9ed378bb4181cc20457.jpg)

![6518b41a6865286c6437a808aa344a5a7464a183187f8d1ef0a16035abf06dfa.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/images/6518b41a6865286c6437a808aa344a5a7464a183187f8d1ef0a16035abf06dfa.jpg)

![655f358788fffc1f4fc15d9da6802f54ae35ba1a240e2253a51fa2793dd0bf7e.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/images/655f358788fffc1f4fc15d9da6802f54ae35ba1a240e2253a51fa2793dd0bf7e.jpg)

![84775c4e225c99b75ca46eb323ba7b5183403acf348bbfcd64b048539a5372e0.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/images/84775c4e225c99b75ca46eb323ba7b5183403acf348bbfcd64b048539a5372e0.jpg)

![863964fc3a5d578fdadc1054f1e613d475fc738c0e04516f7d12726498f144bc.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/images/863964fc3a5d578fdadc1054f1e613d475fc738c0e04516f7d12726498f144bc.jpg)

![a60aefd1520baa2860b4a8fa2a77b3be524d3495e2f37d0556aa05a926d29026.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/images/a60aefd1520baa2860b4a8fa2a77b3be524d3495e2f37d0556aa05a926d29026.jpg)

![aa1c9748825218031ec7e923491da769666a2e813192aa37dca0635339b45512.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/images/aa1c9748825218031ec7e923491da769666a2e813192aa37dca0635339b45512.jpg)

![aa85182da1f45294b1a0e69fb1333eb4afd4adf5cf65ec3fac4978a870952301.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/images/aa85182da1f45294b1a0e69fb1333eb4afd4adf5cf65ec3fac4978a870952301.jpg)

![d421ced3c4c20e20cac457fba462e12a50018571ca84fac68a5ac78f8ee201b3.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/images/d421ced3c4c20e20cac457fba462e12a50018571ca84fac68a5ac78f8ee201b3.jpg)

### Tables

![739813461ec39964de9e0e3481dcd9e1af28ce77e9fc68337769303443814e5d.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/tables/739813461ec39964de9e0e3481dcd9e1af28ce77e9fc68337769303443814e5d.jpg)

![83ffb43b6905bc253ebe26af23ce35660653ecae35e953b69d2c1a33415c1ea2.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/tables/83ffb43b6905bc253ebe26af23ce35660653ecae35e953b69d2c1a33415c1ea2.jpg)

![a7582e8ecb79968c675bc16a02d674b40ed3fdbf0a5710faffc2ae4e1f8722c2.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/tables/a7582e8ecb79968c675bc16a02d674b40ed3fdbf0a5710faffc2ae4e1f8722c2.jpg)

![c0e12bf523e3220818307ab70b0a6ac9a4dc231f6fa827d852e27d8bb9deff61.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/tables/c0e12bf523e3220818307ab70b0a6ac9a4dc231f6fa827d852e27d8bb9deff61.jpg)

![c12df1f99c080c72c2897b98a76ffc26047d87ebf13e4657f14f3a702de4cfae.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/tables/c12df1f99c080c72c2897b98a76ffc26047d87ebf13e4657f14f3a702de4cfae.jpg)

![c8998adcaa2b468e261e3f5787de5de7a74a92cf363620a29d8138b00fe0a37e.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/tables/c8998adcaa2b468e261e3f5787de5de7a74a92cf363620a29d8138b00fe0a37e.jpg)

![d595b60676fc4e4249168f234600d46727fac99adebe8e7e165deaff789912fe.jpg](../icml_results/2950_RePo_%20Language%20Models%20with%20Context%20Re-Positioning/tables/d595b60676fc4e4249168f234600d46727fac99adebe8e7e165deaff789912fe.jpg)

<a id="paper-2913"></a>

## UAV$^2$: A Unified and Adaptive Scheduling Framework for UAV Autopilot Systems with Reinforcement Learning


### Images

![0067c9623ed91058c9fc9b48e58bb87210d364c3ae8e846afbaa402a5b317151.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/0067c9623ed91058c9fc9b48e58bb87210d364c3ae8e846afbaa402a5b317151.jpg)

![02522770a42c24b077c398b217be06b41e9c42b72a1ff3410bc472faf9261feb.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/02522770a42c24b077c398b217be06b41e9c42b72a1ff3410bc472faf9261feb.jpg)

![0ed4aff673b58789ec9abc0adc0a46205672fbdaa25f26c799fe04964f459662.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/0ed4aff673b58789ec9abc0adc0a46205672fbdaa25f26c799fe04964f459662.jpg)

![2a49cfc42cae62ef7af15f63d276cc9a40d288c2019dbe7e80cc99d7175f3d47.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/2a49cfc42cae62ef7af15f63d276cc9a40d288c2019dbe7e80cc99d7175f3d47.jpg)

![2bea4dae76ce4b25cce4d5f424ddfc126a2d32b4289b9d75ed3cfd5ea1279694.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/2bea4dae76ce4b25cce4d5f424ddfc126a2d32b4289b9d75ed3cfd5ea1279694.jpg)

![3c99d29d29540abd1bc0dae2758c5b3dadbf3bbe79003efd89055699e87da214.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/3c99d29d29540abd1bc0dae2758c5b3dadbf3bbe79003efd89055699e87da214.jpg)

![4ef8b15a44efca32786026243bb5dc3d6df766fdebc4af1d584dcebd28bf4fc3.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/4ef8b15a44efca32786026243bb5dc3d6df766fdebc4af1d584dcebd28bf4fc3.jpg)

![56a43de68f177a14f5eb7b00d452a2bae3d7c06b98b78abdeb1007e0876abe1b.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/56a43de68f177a14f5eb7b00d452a2bae3d7c06b98b78abdeb1007e0876abe1b.jpg)

![62897b59ac216fc3f77bfb43f574799268fad5489957622133a37daf682ab1c4.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/62897b59ac216fc3f77bfb43f574799268fad5489957622133a37daf682ab1c4.jpg)

![6499eadbafae10b57b77f5f4904da6764913a2ce502695b22beceec1e37d453a.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/6499eadbafae10b57b77f5f4904da6764913a2ce502695b22beceec1e37d453a.jpg)

![67eaf84c1aa94b7e1d523874f8b2ebf1d3b7efdb587e692c932b160ed3ba8fd4.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/67eaf84c1aa94b7e1d523874f8b2ebf1d3b7efdb587e692c932b160ed3ba8fd4.jpg)

![6a2fe5372bd3646d5652337d7da09d9e97f08fbe5fa1b27b72f0537ef7d21361.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/6a2fe5372bd3646d5652337d7da09d9e97f08fbe5fa1b27b72f0537ef7d21361.jpg)

![766eb39948ab49d42b037aace532f393fa15fa39b4491189f6773433b97e6f9b.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/766eb39948ab49d42b037aace532f393fa15fa39b4491189f6773433b97e6f9b.jpg)

![86f45ee45e96af1304f5ff79a543a0205b7d4e509cf1dd1c2f585237676f8257.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/86f45ee45e96af1304f5ff79a543a0205b7d4e509cf1dd1c2f585237676f8257.jpg)

![9844670e18be761a37f9deaa370480f8d2070d38b51957c1df572a30dbb34be6.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/9844670e18be761a37f9deaa370480f8d2070d38b51957c1df572a30dbb34be6.jpg)

![cc3cc83daec0298e79f993f300c8786f9043819749e285b3b05525119ea20c36.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/cc3cc83daec0298e79f993f300c8786f9043819749e285b3b05525119ea20c36.jpg)

![e16e0bf0bc848a1208ec6eb1c924a133b931c4e85f31fa642c2cee41573228b6.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/e16e0bf0bc848a1208ec6eb1c924a133b931c4e85f31fa642c2cee41573228b6.jpg)

![e17d6c1cf2094dab2d95a38e4a7aa08ee00f2ed2652038ed58afef9e7c0452f3.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/e17d6c1cf2094dab2d95a38e4a7aa08ee00f2ed2652038ed58afef9e7c0452f3.jpg)

![ec4c69d67d5eb4f96f5ea576151d339f8f097cdd444c53873438d96f998e2c6a.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/ec4c69d67d5eb4f96f5ea576151d339f8f097cdd444c53873438d96f998e2c6a.jpg)

![edd316293865bab4196f04e85c06a4793bb5267b33342ae7341b430c727cd4cb.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/edd316293865bab4196f04e85c06a4793bb5267b33342ae7341b430c727cd4cb.jpg)

![f970d70f14cc77c53d0422bb6a0714e81f6f1b2548250e5103042f66ab717168.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/images/f970d70f14cc77c53d0422bb6a0714e81f6f1b2548250e5103042f66ab717168.jpg)

### Tables

![17a02379853d5ee2d270ad7cca84197aaa6bf88de01ff1824e57c30cf53bb46f.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/tables/17a02379853d5ee2d270ad7cca84197aaa6bf88de01ff1824e57c30cf53bb46f.jpg)

![309698d6feab72dd914298ad5a921820b01d941c60cfdb321d717cbabb579288.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/tables/309698d6feab72dd914298ad5a921820b01d941c60cfdb321d717cbabb579288.jpg)

![3b13c7340ca7e74e30241e131a70fc76b0c923ce9e460e9d62ad5a60351044d2.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/tables/3b13c7340ca7e74e30241e131a70fc76b0c923ce9e460e9d62ad5a60351044d2.jpg)

![535680cfc3a9003c7263f4b9f3724bfee7b29298a4c2ed1ba6e9a24e83b5969d.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/tables/535680cfc3a9003c7263f4b9f3724bfee7b29298a4c2ed1ba6e9a24e83b5969d.jpg)

![65b2f67d1de2b04704e23b23402126ef0728c8e1fd0d8cc885ef75169a4de9ac.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/tables/65b2f67d1de2b04704e23b23402126ef0728c8e1fd0d8cc885ef75169a4de9ac.jpg)

![ad62f26ceeb900d27e560bf6548748c5e7ba792a6884cc6f596d3fa6f95e9073.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/tables/ad62f26ceeb900d27e560bf6548748c5e7ba792a6884cc6f596d3fa6f95e9073.jpg)

![af22b7743333f43d68253781e9e7bfad2076c1b2b46a02f811616419b51e12d7.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/tables/af22b7743333f43d68253781e9e7bfad2076c1b2b46a02f811616419b51e12d7.jpg)

![e27dd769d9f500a9aebcc49e54775a1cbb6d2c41e5ae99ade40af12e395c5b21.jpg](../icml_results/2951_UAV%24%5E2%24_%20A%20Unified%20and%20Adaptive%20Scheduling%20Framework%20for%20UAV%20Autopilot%20Systems%20with%20Reinforcement%20Le/tables/e27dd769d9f500a9aebcc49e54775a1cbb6d2c41e5ae99ade40af12e395c5b21.jpg)

<a id="paper-2914"></a>

## Rethinking Temporal Consistency in Video Object-Centric Learning: From Prediction to Correspondence


### Images

![08eae1ebb4de1becbdc0985d7f671c263d6e5a9a60d842b955cb30381821d3fb.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/images/08eae1ebb4de1becbdc0985d7f671c263d6e5a9a60d842b955cb30381821d3fb.jpg)

![1f23e48d172431c7d156f4e89fcf518bb0cbc32189bd1723c34d238af51b1a32.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/images/1f23e48d172431c7d156f4e89fcf518bb0cbc32189bd1723c34d238af51b1a32.jpg)

![2715fc301f538610007d094af7a340603943415e4d63d58102b7c41a4c365375.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/images/2715fc301f538610007d094af7a340603943415e4d63d58102b7c41a4c365375.jpg)

![2a095291f809b737a1ee914d079a9db6693011e236f813ed9e9024958a914fa1.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/images/2a095291f809b737a1ee914d079a9db6693011e236f813ed9e9024958a914fa1.jpg)

![3e51c4689cf710f53993242173695f279219055e18a8b71b0c48c659e32adf22.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/images/3e51c4689cf710f53993242173695f279219055e18a8b71b0c48c659e32adf22.jpg)

![7856a1260118b887e923711f81c4caa3a24836719e5cbd2b61eb710c4924201d.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/images/7856a1260118b887e923711f81c4caa3a24836719e5cbd2b61eb710c4924201d.jpg)

![c43a691803230871b647e596775b1d71be07ba7f08b92a27f0ac32b06253fa81.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/images/c43a691803230871b647e596775b1d71be07ba7f08b92a27f0ac32b06253fa81.jpg)

![c68da1572d3e8fe398a6fb0e067e131a618ec5682f0a590877cd91f5e5342a8f.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/images/c68da1572d3e8fe398a6fb0e067e131a618ec5682f0a590877cd91f5e5342a8f.jpg)

![d35429600a9b9917831f6d37991fcbe875809c7ba17cabb92469b33523b9499a.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/images/d35429600a9b9917831f6d37991fcbe875809c7ba17cabb92469b33523b9499a.jpg)

![e1bd5bd98685ddd8d8d5d1da2a714992589b75e778f7a957b23a3399e129c325.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/images/e1bd5bd98685ddd8d8d5d1da2a714992589b75e778f7a957b23a3399e129c325.jpg)

![f4ab215885e294f6040b44322b6c92225cc7bb77d5804c405c0d4a7230defbaf.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/images/f4ab215885e294f6040b44322b6c92225cc7bb77d5804c405c0d4a7230defbaf.jpg)

![fb78f74ea6e5bfc0718880ee08c50af8274b0549bbbcfa3256ca4100f31630f0.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/images/fb78f74ea6e5bfc0718880ee08c50af8274b0549bbbcfa3256ca4100f31630f0.jpg)

### Tables

![3262e0d2b5f39be4ad34493d5eff6c7ce843757e04a1ef53e9a7682c1c6ceb8e.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/tables/3262e0d2b5f39be4ad34493d5eff6c7ce843757e04a1ef53e9a7682c1c6ceb8e.jpg)

![367c19b2f1ba70cb7a7c7c4a8b816d923a069f0ea900926561b43d6fc696c53c.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/tables/367c19b2f1ba70cb7a7c7c4a8b816d923a069f0ea900926561b43d6fc696c53c.jpg)

![4afcda9b1e144dba629e5877315f99fe8121b2e46d3d1824de19e52c3a0fd617.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/tables/4afcda9b1e144dba629e5877315f99fe8121b2e46d3d1824de19e52c3a0fd617.jpg)

![6523e85485b09e69508dc66c7010afe65c1cf23fa41792db4a7d055f5ae92567.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/tables/6523e85485b09e69508dc66c7010afe65c1cf23fa41792db4a7d055f5ae92567.jpg)

![6cbaa06c865c3e13052350abee886d958901405cb85f38992482e5c59d4ba58b.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/tables/6cbaa06c865c3e13052350abee886d958901405cb85f38992482e5c59d4ba58b.jpg)

![74dc9eacaeb24714e63cc826b6184b6acf2a1021a57ae8aecb24ecec94c84701.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/tables/74dc9eacaeb24714e63cc826b6184b6acf2a1021a57ae8aecb24ecec94c84701.jpg)

![7b579d2ae092e8244af11c0ee6ffff7e05feea64f86315b186cb8a476e5f02f0.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/tables/7b579d2ae092e8244af11c0ee6ffff7e05feea64f86315b186cb8a476e5f02f0.jpg)

![a7b57e99573b60521db5032fe8c7adf07ae274fa2bee7416ae25a8aebd3f21af.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/tables/a7b57e99573b60521db5032fe8c7adf07ae274fa2bee7416ae25a8aebd3f21af.jpg)

![f804f261fdec9ec026d3a6c73128653d99d6b964b9fee015f38d69bf536279f2.jpg](../icml_results/2952_Rethinking%20Temporal%20Consistency%20in%20Video%20Object-Centric%20Learning_%20From%20Prediction%20to%20Correspondence/tables/f804f261fdec9ec026d3a6c73128653d99d6b964b9fee015f38d69bf536279f2.jpg)

<a id="paper-2915"></a>

## Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training


### Images

![1a59b38b6707b2f7b922180fe43e8cf8580f4f1f5ccb3e91e93ee8ffbe261b50.jpg](../icml_results/2953_Decouple%20Searching%20from%20Training_%20Scaling%20Data%20Mixing%20via%20Model%20Merging%20for%20Large%20Language%20Model%20Pre/images/1a59b38b6707b2f7b922180fe43e8cf8580f4f1f5ccb3e91e93ee8ffbe261b50.jpg)

![672257a9fca26836997b08b48dd8f96e00da22f8c03a75f7e44a62c28f2c2069.jpg](../icml_results/2953_Decouple%20Searching%20from%20Training_%20Scaling%20Data%20Mixing%20via%20Model%20Merging%20for%20Large%20Language%20Model%20Pre/images/672257a9fca26836997b08b48dd8f96e00da22f8c03a75f7e44a62c28f2c2069.jpg)

![8cb60692fdf307ebfe3e59f0504bbfbd04086f61646fc72a4d078515c3f805f1.jpg](../icml_results/2953_Decouple%20Searching%20from%20Training_%20Scaling%20Data%20Mixing%20via%20Model%20Merging%20for%20Large%20Language%20Model%20Pre/images/8cb60692fdf307ebfe3e59f0504bbfbd04086f61646fc72a4d078515c3f805f1.jpg)

![ab916a2e11e5972cd4b6e5819fea413218bfd881b5bed0fd5a5fd2c761fb1f55.jpg](../icml_results/2953_Decouple%20Searching%20from%20Training_%20Scaling%20Data%20Mixing%20via%20Model%20Merging%20for%20Large%20Language%20Model%20Pre/images/ab916a2e11e5972cd4b6e5819fea413218bfd881b5bed0fd5a5fd2c761fb1f55.jpg)

![b22942f3cc5a6ef4c2c524a6e74d252141aece71efb866a93c656f1dca2d627d.jpg](../icml_results/2953_Decouple%20Searching%20from%20Training_%20Scaling%20Data%20Mixing%20via%20Model%20Merging%20for%20Large%20Language%20Model%20Pre/images/b22942f3cc5a6ef4c2c524a6e74d252141aece71efb866a93c656f1dca2d627d.jpg)

### Tables

![3b531da71eec248442b7a685fccc4fbc0cade6c5c83cb369f3aac783a29bd31a.jpg](../icml_results/2953_Decouple%20Searching%20from%20Training_%20Scaling%20Data%20Mixing%20via%20Model%20Merging%20for%20Large%20Language%20Model%20Pre/tables/3b531da71eec248442b7a685fccc4fbc0cade6c5c83cb369f3aac783a29bd31a.jpg)

![3d1e1bc3ff9ea4e461099795b56ccdd2ce4a5bed46862388197e14e16b585c59.jpg](../icml_results/2953_Decouple%20Searching%20from%20Training_%20Scaling%20Data%20Mixing%20via%20Model%20Merging%20for%20Large%20Language%20Model%20Pre/tables/3d1e1bc3ff9ea4e461099795b56ccdd2ce4a5bed46862388197e14e16b585c59.jpg)

![4290c57b6718084baf921ba29858a8e734bab2ba8d80d4e344ee7a3fb6b9f348.jpg](../icml_results/2953_Decouple%20Searching%20from%20Training_%20Scaling%20Data%20Mixing%20via%20Model%20Merging%20for%20Large%20Language%20Model%20Pre/tables/4290c57b6718084baf921ba29858a8e734bab2ba8d80d4e344ee7a3fb6b9f348.jpg)

![4bed397dfa4ec8320fe8821e24dd5bfac6487c7619f32434cb0f0092d2d01dad.jpg](../icml_results/2953_Decouple%20Searching%20from%20Training_%20Scaling%20Data%20Mixing%20via%20Model%20Merging%20for%20Large%20Language%20Model%20Pre/tables/4bed397dfa4ec8320fe8821e24dd5bfac6487c7619f32434cb0f0092d2d01dad.jpg)

![5348a102a0734b790fa8167c0ddd6fd839c6c25e42e4c4c8fb3c98966c9e012e.jpg](../icml_results/2953_Decouple%20Searching%20from%20Training_%20Scaling%20Data%20Mixing%20via%20Model%20Merging%20for%20Large%20Language%20Model%20Pre/tables/5348a102a0734b790fa8167c0ddd6fd839c6c25e42e4c4c8fb3c98966c9e012e.jpg)

![57b4a9dc99e703ee07e8fe3a0425473cec34aae619dcf1f78e0c30994168bc4c.jpg](../icml_results/2953_Decouple%20Searching%20from%20Training_%20Scaling%20Data%20Mixing%20via%20Model%20Merging%20for%20Large%20Language%20Model%20Pre/tables/57b4a9dc99e703ee07e8fe3a0425473cec34aae619dcf1f78e0c30994168bc4c.jpg)

![6bd5862a0eee0d9e44a264722244ffc5e77ab22791680dcd6e76bc29dfa93f5b.jpg](../icml_results/2953_Decouple%20Searching%20from%20Training_%20Scaling%20Data%20Mixing%20via%20Model%20Merging%20for%20Large%20Language%20Model%20Pre/tables/6bd5862a0eee0d9e44a264722244ffc5e77ab22791680dcd6e76bc29dfa93f5b.jpg)

![821fa70ece3ad6a9f4ad6308e775cddc3c81d71dfe8c625e783ebf55df099c53.jpg](../icml_results/2953_Decouple%20Searching%20from%20Training_%20Scaling%20Data%20Mixing%20via%20Model%20Merging%20for%20Large%20Language%20Model%20Pre/tables/821fa70ece3ad6a9f4ad6308e775cddc3c81d71dfe8c625e783ebf55df099c53.jpg)

![896215fdaaefb7620c1477bf1db3cf4ca9d2d2b80135731c3ece77c7cd8a4005.jpg](../icml_results/2953_Decouple%20Searching%20from%20Training_%20Scaling%20Data%20Mixing%20via%20Model%20Merging%20for%20Large%20Language%20Model%20Pre/tables/896215fdaaefb7620c1477bf1db3cf4ca9d2d2b80135731c3ece77c7cd8a4005.jpg)

![a937b9133a57674f4adfe807ff7d64ace147c35f1fada4dc7597359285b89ed9.jpg](../icml_results/2953_Decouple%20Searching%20from%20Training_%20Scaling%20Data%20Mixing%20via%20Model%20Merging%20for%20Large%20Language%20Model%20Pre/tables/a937b9133a57674f4adfe807ff7d64ace147c35f1fada4dc7597359285b89ed9.jpg)

![adc25df8c010ddcf2c53c5e37b52b06cf41bdd2c1e67f455d54f9973871898c0.jpg](../icml_results/2953_Decouple%20Searching%20from%20Training_%20Scaling%20Data%20Mixing%20via%20Model%20Merging%20for%20Large%20Language%20Model%20Pre/tables/adc25df8c010ddcf2c53c5e37b52b06cf41bdd2c1e67f455d54f9973871898c0.jpg)

![dcedd4934b3456bb577e2ffe570e69cded266a548bf44311aef6f98c55e323f4.jpg](../icml_results/2953_Decouple%20Searching%20from%20Training_%20Scaling%20Data%20Mixing%20via%20Model%20Merging%20for%20Large%20Language%20Model%20Pre/tables/dcedd4934b3456bb577e2ffe570e69cded266a548bf44311aef6f98c55e323f4.jpg)

<a id="paper-2916"></a>

## On Revisiting Entropy for Identifying Mislabeled Images


### Images

![09d57f37aed084a0b0170f8d2b1bb4ce08b14f95d2a19fad10c24e40166752ce.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/images/09d57f37aed084a0b0170f8d2b1bb4ce08b14f95d2a19fad10c24e40166752ce.jpg)

![0b2acc3b11212473778d974fe7c304505c7b6d3868454654a22c5f383c2a64b4.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/images/0b2acc3b11212473778d974fe7c304505c7b6d3868454654a22c5f383c2a64b4.jpg)

![154cae7af986130af7aeba350404f823a185b994dda94eefe718e90f18d0975d.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/images/154cae7af986130af7aeba350404f823a185b994dda94eefe718e90f18d0975d.jpg)

![2303269cbb51459a1ed4e623a4b7b369e41d2a450bd943c122ae82d0b052faca.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/images/2303269cbb51459a1ed4e623a4b7b369e41d2a450bd943c122ae82d0b052faca.jpg)

![27afdacd8254a387547623e840b14f0f1419041bd4e685d259f4e8a76c80d21a.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/images/27afdacd8254a387547623e840b14f0f1419041bd4e685d259f4e8a76c80d21a.jpg)

![289da231581ea903448c300802217d81061b6a2c0c7de06e2665a991782dd1d4.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/images/289da231581ea903448c300802217d81061b6a2c0c7de06e2665a991782dd1d4.jpg)

![681c70437aa30c2e8a7debcd553e352953c6013d126e42384e276f786da9d3bf.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/images/681c70437aa30c2e8a7debcd553e352953c6013d126e42384e276f786da9d3bf.jpg)

![693cf70b0a36f717c42d46c2f3d02eae1b3d8272a27717a0487816528b0bbd27.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/images/693cf70b0a36f717c42d46c2f3d02eae1b3d8272a27717a0487816528b0bbd27.jpg)

![71ab76839216df05123b889e8d8523eccf81f9c9c5216e35842636eaa846cd59.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/images/71ab76839216df05123b889e8d8523eccf81f9c9c5216e35842636eaa846cd59.jpg)

![89bf197bf74b23124c5d37de6690af5c13d909a2c74440b21efb857f6917f6e9.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/images/89bf197bf74b23124c5d37de6690af5c13d909a2c74440b21efb857f6917f6e9.jpg)

![97fe4d629d313f50a0325fa9ebdcaa726fefe384c2e4907758ec9eb4779a59c1.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/images/97fe4d629d313f50a0325fa9ebdcaa726fefe384c2e4907758ec9eb4779a59c1.jpg)

![bde038dc90cc1535b79cb9001a25a5f36f0b24858dadd1a773c96e4642b3b9a5.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/images/bde038dc90cc1535b79cb9001a25a5f36f0b24858dadd1a773c96e4642b3b9a5.jpg)

![d9408a5c401b3017f039f0ac5aac16c22e2f3a5d11ee7f2b016d69831be7f478.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/images/d9408a5c401b3017f039f0ac5aac16c22e2f3a5d11ee7f2b016d69831be7f478.jpg)

![e7730f1996585323dcc585e04318e9d0944015062a82c5fe66c7daaa09b24c9f.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/images/e7730f1996585323dcc585e04318e9d0944015062a82c5fe66c7daaa09b24c9f.jpg)

![ea36482bc226e3d03da515326cce247072b7df1ba205bcb6fcff5cb4b2101fe0.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/images/ea36482bc226e3d03da515326cce247072b7df1ba205bcb6fcff5cb4b2101fe0.jpg)

![f74b47e46f95ec04ad67b17ebd8d26a6e2095f724dfd2d1359c090c7d9a2dc40.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/images/f74b47e46f95ec04ad67b17ebd8d26a6e2095f724dfd2d1359c090c7d9a2dc40.jpg)

### Tables

![128fd182eed352522de2b4ab0267b57fb4fc675c09baf72cd063b1d730a2b530.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/128fd182eed352522de2b4ab0267b57fb4fc675c09baf72cd063b1d730a2b530.jpg)

![200f4b27e7ca479a45a1b41e170b419c9769f8fa6b43d15005c5955d78222660.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/200f4b27e7ca479a45a1b41e170b419c9769f8fa6b43d15005c5955d78222660.jpg)

![2b9b350b3cfabbe8e9f97e020c49aa2016a365e4cfb9fce4a33097ac23ee2bd4.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/2b9b350b3cfabbe8e9f97e020c49aa2016a365e4cfb9fce4a33097ac23ee2bd4.jpg)

![2dec5a24263691252652d8d90094044047d33b1702db25117ee60d5c942de24d.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/2dec5a24263691252652d8d90094044047d33b1702db25117ee60d5c942de24d.jpg)

![3daa0c2ba7e3d28d385c99b62798da9de8bd7dfa66b6d831058f8464f841d75a.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/3daa0c2ba7e3d28d385c99b62798da9de8bd7dfa66b6d831058f8464f841d75a.jpg)

![500c71ffe297b6f619098ee9fbcb5d17f6c15501414f4ff77eb05be55b33a6c9.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/500c71ffe297b6f619098ee9fbcb5d17f6c15501414f4ff77eb05be55b33a6c9.jpg)

![5a961985a4a07293aba4e659dd59fad5d06718afcbec9be15950137a0dfbb480.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/5a961985a4a07293aba4e659dd59fad5d06718afcbec9be15950137a0dfbb480.jpg)

![693da328e839de23e665fccb4b5807375ff87cdb86e6bef24758cbf32cc4814e.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/693da328e839de23e665fccb4b5807375ff87cdb86e6bef24758cbf32cc4814e.jpg)

![7219d50ce8878091e46b7647cf461802c648bb3a41da8badc74ef0e445e3f9c1.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/7219d50ce8878091e46b7647cf461802c648bb3a41da8badc74ef0e445e3f9c1.jpg)

![75c6ffc9f701cca978c2fbd8d4f832f14df165b5fcb9351d274653e20af5571d.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/75c6ffc9f701cca978c2fbd8d4f832f14df165b5fcb9351d274653e20af5571d.jpg)

![785b2070245b871c72f8f39ca1d0cd18d82ad575801528203a3f1e7381910049.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/785b2070245b871c72f8f39ca1d0cd18d82ad575801528203a3f1e7381910049.jpg)

![8628d93b0eccbb7a848b42fd94af7876e08e7118b71a550cb33e419e8607cc1d.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/8628d93b0eccbb7a848b42fd94af7876e08e7118b71a550cb33e419e8607cc1d.jpg)

![a3b30c9174e65a67ee57d2dd8f1e1e6a0173635bee82100b73ed935784c08324.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/a3b30c9174e65a67ee57d2dd8f1e1e6a0173635bee82100b73ed935784c08324.jpg)

![a84b80c7518d893ec2b06d954b212242d523253fe503ae6c9dc42cf81775cb83.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/a84b80c7518d893ec2b06d954b212242d523253fe503ae6c9dc42cf81775cb83.jpg)

![b02b1017800853d3c6e2bf2c32fa3ad0a72592d229060fb6b56f3c091f729e07.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/b02b1017800853d3c6e2bf2c32fa3ad0a72592d229060fb6b56f3c091f729e07.jpg)

![bbbbd61fed53e441258b1982a5c784c3a16654e8b9e877bce03427bc9b9c801c.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/bbbbd61fed53e441258b1982a5c784c3a16654e8b9e877bce03427bc9b9c801c.jpg)

![daeb800c34b3945c9e5ce34fa41435d2538272985db5facc7ad68614dd8375cf.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/daeb800c34b3945c9e5ce34fa41435d2538272985db5facc7ad68614dd8375cf.jpg)

![dcebb645e69430b61858131403ac9694899d3e398db0af56f03f58b75d965aef.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/dcebb645e69430b61858131403ac9694899d3e398db0af56f03f58b75d965aef.jpg)

![de473824ffdd88a5e2b39ff55a95772a5a26707e8ec8f222b3b9a698daefef12.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/de473824ffdd88a5e2b39ff55a95772a5a26707e8ec8f222b3b9a698daefef12.jpg)

![eb4c1558308e4bacfbc853f858960d238482c5ca76c909df0297e605afe1dbdd.jpg](../icml_results/2954_On%20Revisiting%20Entropy%20for%20Identifying%20Mislabeled%20Images/tables/eb4c1558308e4bacfbc853f858960d238482c5ca76c909df0297e605afe1dbdd.jpg)

<a id="paper-2917"></a>

## Diffusing to Coordinate: Efficient Online Multi-Agent Diffusion Policies


### Images

![13109a131a0d05227e757e08d556ec90c8f259c349dff8c4002894a89d9b4b56.jpg](../icml_results/2955_Diffusing%20to%20Coordinate_%20Efficient%20Online%20Multi-Agent%20Diffusion%20Policies/images/13109a131a0d05227e757e08d556ec90c8f259c349dff8c4002894a89d9b4b56.jpg)

![277015db355b76fc649f74c95cdaa5fb1090e1d1f2a319f0af6dcf38671c9081.jpg](../icml_results/2955_Diffusing%20to%20Coordinate_%20Efficient%20Online%20Multi-Agent%20Diffusion%20Policies/images/277015db355b76fc649f74c95cdaa5fb1090e1d1f2a319f0af6dcf38671c9081.jpg)

![4ecbef893ba61da2ad9a899dfdfafb3317ad79fa621fbd7fedc730b65d88dcf4.jpg](../icml_results/2955_Diffusing%20to%20Coordinate_%20Efficient%20Online%20Multi-Agent%20Diffusion%20Policies/images/4ecbef893ba61da2ad9a899dfdfafb3317ad79fa621fbd7fedc730b65d88dcf4.jpg)

![5f13b42bd0d481f86216a7f89d7b908540764a96e01b218b9481bbfa16d32314.jpg](../icml_results/2955_Diffusing%20to%20Coordinate_%20Efficient%20Online%20Multi-Agent%20Diffusion%20Policies/images/5f13b42bd0d481f86216a7f89d7b908540764a96e01b218b9481bbfa16d32314.jpg)

![74a84d5b3c3e957c11c70ad7fa09f3555f6b78f64e83a4e2ebe891dd5c70dc61.jpg](../icml_results/2955_Diffusing%20to%20Coordinate_%20Efficient%20Online%20Multi-Agent%20Diffusion%20Policies/images/74a84d5b3c3e957c11c70ad7fa09f3555f6b78f64e83a4e2ebe891dd5c70dc61.jpg)

![7e8c9d3704de8b9188896a5e79d47700e556b4310d2a14a8ae4862563cfd87df.jpg](../icml_results/2955_Diffusing%20to%20Coordinate_%20Efficient%20Online%20Multi-Agent%20Diffusion%20Policies/images/7e8c9d3704de8b9188896a5e79d47700e556b4310d2a14a8ae4862563cfd87df.jpg)

![8a7438bc67e42ec2190bba4c5d3f7c0a04c6f12481372be6762654a2e2075e72.jpg](../icml_results/2955_Diffusing%20to%20Coordinate_%20Efficient%20Online%20Multi-Agent%20Diffusion%20Policies/images/8a7438bc67e42ec2190bba4c5d3f7c0a04c6f12481372be6762654a2e2075e72.jpg)

![8fa36fe13cb115e1fc9f5a0830ec4ee4d88432d65fbf28a67074794aeb71c594.jpg](../icml_results/2955_Diffusing%20to%20Coordinate_%20Efficient%20Online%20Multi-Agent%20Diffusion%20Policies/images/8fa36fe13cb115e1fc9f5a0830ec4ee4d88432d65fbf28a67074794aeb71c594.jpg)

![a40d92e81e9a831a7c52657240084519eeb48449abe6b511641c6136a974c9a6.jpg](../icml_results/2955_Diffusing%20to%20Coordinate_%20Efficient%20Online%20Multi-Agent%20Diffusion%20Policies/images/a40d92e81e9a831a7c52657240084519eeb48449abe6b511641c6136a974c9a6.jpg)

![e6371d10e2784eca66160a8759d6ef61593fdad2fdce69c1e972357907153a00.jpg](../icml_results/2955_Diffusing%20to%20Coordinate_%20Efficient%20Online%20Multi-Agent%20Diffusion%20Policies/images/e6371d10e2784eca66160a8759d6ef61593fdad2fdce69c1e972357907153a00.jpg)

### Tables

![067039641270240ee1ec096be035006d0f64c3abd909fefec391c85f1d7b2a4e.jpg](../icml_results/2955_Diffusing%20to%20Coordinate_%20Efficient%20Online%20Multi-Agent%20Diffusion%20Policies/tables/067039641270240ee1ec096be035006d0f64c3abd909fefec391c85f1d7b2a4e.jpg)

![281baf396646c5f0a5888a0af8f7fc84d5e6252ee288a9886ef422e3b530e48c.jpg](../icml_results/2955_Diffusing%20to%20Coordinate_%20Efficient%20Online%20Multi-Agent%20Diffusion%20Policies/tables/281baf396646c5f0a5888a0af8f7fc84d5e6252ee288a9886ef422e3b530e48c.jpg)

![41c5430d0050224e389dd47d5e3fd231a53ac6b1dfd442f519a99c0f287fe6ff.jpg](../icml_results/2955_Diffusing%20to%20Coordinate_%20Efficient%20Online%20Multi-Agent%20Diffusion%20Policies/tables/41c5430d0050224e389dd47d5e3fd231a53ac6b1dfd442f519a99c0f287fe6ff.jpg)

![73f72e28d0f254de16c483b65b627fb2af10da6ad4555a6455fbefbda58a0c50.jpg](../icml_results/2955_Diffusing%20to%20Coordinate_%20Efficient%20Online%20Multi-Agent%20Diffusion%20Policies/tables/73f72e28d0f254de16c483b65b627fb2af10da6ad4555a6455fbefbda58a0c50.jpg)

![97b2ac2ab9ac880646a2a5425f76fde797270f9d09be63da3c86c044f674166e.jpg](../icml_results/2955_Diffusing%20to%20Coordinate_%20Efficient%20Online%20Multi-Agent%20Diffusion%20Policies/tables/97b2ac2ab9ac880646a2a5425f76fde797270f9d09be63da3c86c044f674166e.jpg)

![b16cd1ea3bee44f5a89b0735b6e1a557f70e1e6807994166245f21ccffe25a48.jpg](../icml_results/2955_Diffusing%20to%20Coordinate_%20Efficient%20Online%20Multi-Agent%20Diffusion%20Policies/tables/b16cd1ea3bee44f5a89b0735b6e1a557f70e1e6807994166245f21ccffe25a48.jpg)

![ba9fe261e3aabe277c64c87bbdcab44dc798f8c28ad2b263c3a1cccc73496bc9.jpg](../icml_results/2955_Diffusing%20to%20Coordinate_%20Efficient%20Online%20Multi-Agent%20Diffusion%20Policies/tables/ba9fe261e3aabe277c64c87bbdcab44dc798f8c28ad2b263c3a1cccc73496bc9.jpg)

<a id="paper-2918"></a>

## Multi-Label Test-Time Adaptation with Bayesian Conditional Priors


### Images

![18e9db39a6ef7f61f4f7257dbe60baa206af5218f8ccdfdaf3486061b2f09093.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/images/18e9db39a6ef7f61f4f7257dbe60baa206af5218f8ccdfdaf3486061b2f09093.jpg)

![3b4ca3c14635cce781d20032f0dc90ca4c134e0de8bf91bc7634c190f170caa8.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/images/3b4ca3c14635cce781d20032f0dc90ca4c134e0de8bf91bc7634c190f170caa8.jpg)

![3c91a7a685643144f6bc1b00410f79a1069605ecab94bbbdc3aa2cfe98852172.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/images/3c91a7a685643144f6bc1b00410f79a1069605ecab94bbbdc3aa2cfe98852172.jpg)

![83f3d29a4774e1b8a2204b0c6f0d6c411ddd7eecf13281e9cf0061837f220f9b.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/images/83f3d29a4774e1b8a2204b0c6f0d6c411ddd7eecf13281e9cf0061837f220f9b.jpg)

![e3155d0dd89397c3cc17fde74b9befc31fabc11911475d29795fe418e296b3d5.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/images/e3155d0dd89397c3cc17fde74b9befc31fabc11911475d29795fe418e296b3d5.jpg)

### Tables

![0634a77bd81dcfbeeb6d59512f3b3a433e2aeef3bb77079f899944b821d9f57d.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/tables/0634a77bd81dcfbeeb6d59512f3b3a433e2aeef3bb77079f899944b821d9f57d.jpg)

![32c95087e5b4a663d47ddc51f7dd6bc97a7bd56f9178acdd41a7c23f04c8d3c6.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/tables/32c95087e5b4a663d47ddc51f7dd6bc97a7bd56f9178acdd41a7c23f04c8d3c6.jpg)

![4529b5986743f807abbc4fa4b3a8cd60aeecbce778e9e53774ff582134da38d8.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/tables/4529b5986743f807abbc4fa4b3a8cd60aeecbce778e9e53774ff582134da38d8.jpg)

![4e2687c40699aec1bc33187962c94ec0a3ce16ca830c5410751e087087bf4bb5.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/tables/4e2687c40699aec1bc33187962c94ec0a3ce16ca830c5410751e087087bf4bb5.jpg)

![5de3a4b456329845d768c561cba5125fec6b424cf2109cc48bdec66204408dcb.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/tables/5de3a4b456329845d768c561cba5125fec6b424cf2109cc48bdec66204408dcb.jpg)

![61b489051139c43c8fa879c664107518e946fafef07c3c6d3ba9693a62c71a9f.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/tables/61b489051139c43c8fa879c664107518e946fafef07c3c6d3ba9693a62c71a9f.jpg)

![669a3fb2857824c48c7d80b0d2deeacafd810ca2e48724ed743bb3d2003d0df9.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/tables/669a3fb2857824c48c7d80b0d2deeacafd810ca2e48724ed743bb3d2003d0df9.jpg)

![8c39bdc947c4d7f129103781492f36f0d89ee3cf4c04b524b32463b3595643c5.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/tables/8c39bdc947c4d7f129103781492f36f0d89ee3cf4c04b524b32463b3595643c5.jpg)

![b32912aa8295c6d49a054f9fb2fde65213e479f0e51c35119379f5dff48c7706.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/tables/b32912aa8295c6d49a054f9fb2fde65213e479f0e51c35119379f5dff48c7706.jpg)

![b876ee98d5af1e924b7a6f22bbd3e4a9b8aad0b88cca6300342597eef643a3b3.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/tables/b876ee98d5af1e924b7a6f22bbd3e4a9b8aad0b88cca6300342597eef643a3b3.jpg)

![bc5fea5eadeb4856e31ecbe0126360be1308be48820a0064bf30f13ac525e851.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/tables/bc5fea5eadeb4856e31ecbe0126360be1308be48820a0064bf30f13ac525e851.jpg)

![cdaeec9d3339b815e6561f0184032ff5521ab6e70c01782fcae204ba0bd21555.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/tables/cdaeec9d3339b815e6561f0184032ff5521ab6e70c01782fcae204ba0bd21555.jpg)

![cdc1f8799154e030f860efe98e02ffd3533f93d48d3c1d07af6ca104cee6f6a8.jpg](../icml_results/2956_Multi-Label%20Test-Time%20Adaptation%20with%20Bayesian%20Conditional%20Priors/tables/cdc1f8799154e030f860efe98e02ffd3533f93d48d3c1d07af6ca104cee6f6a8.jpg)

<a id="paper-2919"></a>

## Beyond Logits: Metastable Latent Dynamics for Sample-Efficient Best-of-N Selection in LLMs


### Images

![004b818ffaeb67a1474e0b0980e6cf6cb169d9b4cf4d5f67cbb585098fd24845.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/images/004b818ffaeb67a1474e0b0980e6cf6cb169d9b4cf4d5f67cbb585098fd24845.jpg)

![05cb874c1662f4f1e6512cdcfe699f8488718a3decdd40470635ab40fdb08944.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/images/05cb874c1662f4f1e6512cdcfe699f8488718a3decdd40470635ab40fdb08944.jpg)

![0d5b7cd1b2071033d3ff6cb1b6dcafe2eb366f1a2d23769f90c0158000fd6abe.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/images/0d5b7cd1b2071033d3ff6cb1b6dcafe2eb366f1a2d23769f90c0158000fd6abe.jpg)

![421811f5693f361f3f204d901faeabfaef2b06723568553621114162229a48da.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/images/421811f5693f361f3f204d901faeabfaef2b06723568553621114162229a48da.jpg)

![8695a3d8d408130ba86c03750f158408ede173499fb2377f490663a52be36794.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/images/8695a3d8d408130ba86c03750f158408ede173499fb2377f490663a52be36794.jpg)

![ac692ff3a221688205d754f346d11ec218201f2320c4638df2007246489c89d6.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/images/ac692ff3a221688205d754f346d11ec218201f2320c4638df2007246489c89d6.jpg)

![cb5bbd3048dbab35182bc418893d6b13822f45171a14a36598293725d9efadeb.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/images/cb5bbd3048dbab35182bc418893d6b13822f45171a14a36598293725d9efadeb.jpg)

![ce7e16556c633256ab921c76cc83637a28904f74d895f214ff1458b84ae14da5.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/images/ce7e16556c633256ab921c76cc83637a28904f74d895f214ff1458b84ae14da5.jpg)

![d4bceaf37b70eaa4141ef845a02f99cc784dfc35bdf5f1cd15cf7557561f10fe.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/images/d4bceaf37b70eaa4141ef845a02f99cc784dfc35bdf5f1cd15cf7557561f10fe.jpg)

![e73e537ed69135599305bb25db889d341909378ab29b91de506147beb02d1a2c.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/images/e73e537ed69135599305bb25db889d341909378ab29b91de506147beb02d1a2c.jpg)

![f518ea39009cb94164a1925ccc35ddfada0b474b20ce2f092edfec793d80cd72.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/images/f518ea39009cb94164a1925ccc35ddfada0b474b20ce2f092edfec793d80cd72.jpg)

![f78d26ac54f61e2a3cf2501fd74009ce3696ed9df567ea18bb0c2e356d61c564.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/images/f78d26ac54f61e2a3cf2501fd74009ce3696ed9df567ea18bb0c2e356d61c564.jpg)

![f85230bc33e639084d0151140e95535fcdcd39b30da8f370aefdd44f2c02ecdb.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/images/f85230bc33e639084d0151140e95535fcdcd39b30da8f370aefdd44f2c02ecdb.jpg)

### Tables

![866d8c97f3f8d90a282b7e6355e1d345b231b58f769f264f448f11cb95a228a4.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/tables/866d8c97f3f8d90a282b7e6355e1d345b231b58f769f264f448f11cb95a228a4.jpg)

![b69ec0224511167a281af8967502dc53ab533b1ff0f83b8ddc6a4ea31777c764.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/tables/b69ec0224511167a281af8967502dc53ab533b1ff0f83b8ddc6a4ea31777c764.jpg)

![b6a57c4b6d039f537823caadfabc9447eda9fbe289e1fc690b65382cbf60e41d.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/tables/b6a57c4b6d039f537823caadfabc9447eda9fbe289e1fc690b65382cbf60e41d.jpg)

![d50f6f0a048d984a7dabc0ffc7e15975ab5378a7140597ca4024a4db054949e2.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/tables/d50f6f0a048d984a7dabc0ffc7e15975ab5378a7140597ca4024a4db054949e2.jpg)

![ea53f84d2e7846894ef6f0bace458b3ddfa1701b891b0bf3db21a9885c967b47.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/tables/ea53f84d2e7846894ef6f0bace458b3ddfa1701b891b0bf3db21a9885c967b47.jpg)

![f12e09a96d937c3fba14f60d155141a37e3f883f1b81415c16eb4352b96f3018.jpg](../icml_results/2957_Beyond%20Logits_%20Metastable%20Latent%20Dynamics%20for%20Sample-Efficient%20Best-of-N%20Selection%20in%20LLMs/tables/f12e09a96d937c3fba14f60d155141a37e3f883f1b81415c16eb4352b96f3018.jpg)

<a id="paper-2920"></a>

## Self-Guidance: Enhancing Neural Codecs via Decoder Manifold Alignment


### Images

![498fcefb2e6f8f98afe71eebac9e4150ab3ed7fada9ca92b77b43e856a59a050.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/images/498fcefb2e6f8f98afe71eebac9e4150ab3ed7fada9ca92b77b43e856a59a050.jpg)

![4efce22c01dd2c187f07c977ae4927ab440044a6cf72ed238bd006e4be73dfac.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/images/4efce22c01dd2c187f07c977ae4927ab440044a6cf72ed238bd006e4be73dfac.jpg)

![5c5fd5a2b5f8e7e9565ac110668fe71e8b7bab1ff77ca5203854693e010d9f3a.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/images/5c5fd5a2b5f8e7e9565ac110668fe71e8b7bab1ff77ca5203854693e010d9f3a.jpg)

![61266a64745cfbfbedd121de5f6674f9a3c34e488de6a052983704724c67e589.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/images/61266a64745cfbfbedd121de5f6674f9a3c34e488de6a052983704724c67e589.jpg)

![8cfe13a686644628d5a146cd87bcb24e4f6709e5340e7ddef3bb0c37c1543e0a.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/images/8cfe13a686644628d5a146cd87bcb24e4f6709e5340e7ddef3bb0c37c1543e0a.jpg)

![9bac321c30b66ce483b69dda6a449f6222942ba272f051882a042e5914ac4457.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/images/9bac321c30b66ce483b69dda6a449f6222942ba272f051882a042e5914ac4457.jpg)

![c59258116e6c3d36db9e2673361afbc3cfc27f3742789b63b1c81df821bd10c2.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/images/c59258116e6c3d36db9e2673361afbc3cfc27f3742789b63b1c81df821bd10c2.jpg)

![ef05f5730acfcf5b22775857a74530343603d7503b1b0c2ce016a3f062ed2e0a.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/images/ef05f5730acfcf5b22775857a74530343603d7503b1b0c2ce016a3f062ed2e0a.jpg)

![fc42daf8c314db079ffdaaa759124e1a16ef54e6fe87f49cdb17656ffbac4f85.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/images/fc42daf8c314db079ffdaaa759124e1a16ef54e6fe87f49cdb17656ffbac4f85.jpg)

### Tables

![03e787da53b77647377650389ac88e5e233a6d1acd42ee27c2b5d06ee9ccc02b.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/tables/03e787da53b77647377650389ac88e5e233a6d1acd42ee27c2b5d06ee9ccc02b.jpg)

![119d767e1ce7fd32d52903143e9a132880c990b577b3051caf8acba7e82f2704.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/tables/119d767e1ce7fd32d52903143e9a132880c990b577b3051caf8acba7e82f2704.jpg)

![4da3477e7c7df8291fa0ef7842ea02ea94507f6ce09e673027ab3d7d179f0bd2.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/tables/4da3477e7c7df8291fa0ef7842ea02ea94507f6ce09e673027ab3d7d179f0bd2.jpg)

![4e76d170e51ad6d1b470713cef1c00ac2bcf2e14f6170b79c36072c878cc3d27.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/tables/4e76d170e51ad6d1b470713cef1c00ac2bcf2e14f6170b79c36072c878cc3d27.jpg)

![75b25d9d18929370b3610023af3f5a97ee3967709c6a8e330478436a0661c50e.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/tables/75b25d9d18929370b3610023af3f5a97ee3967709c6a8e330478436a0661c50e.jpg)

![883079b78e823ba145a01050757a9534bcebf464ea8b1314090ced630c3b49c6.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/tables/883079b78e823ba145a01050757a9534bcebf464ea8b1314090ced630c3b49c6.jpg)

![b6699f47b8cbec359c8197b21d73fb079212bb7052ddedfdc851c404264853a1.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/tables/b6699f47b8cbec359c8197b21d73fb079212bb7052ddedfdc851c404264853a1.jpg)

![bef0efb8504a91666d1653fddaf4bfc298c7a328f341a67dfa61825d9e81e4a2.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/tables/bef0efb8504a91666d1653fddaf4bfc298c7a328f341a67dfa61825d9e81e4a2.jpg)

![bf3ab293393642f402fcf2a422cd338bbbaf3dea6c415ca9317ae8b6da9da391.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/tables/bf3ab293393642f402fcf2a422cd338bbbaf3dea6c415ca9317ae8b6da9da391.jpg)

![c655b4f2986b6dce04f8e66b918be02c851c56e55402ff176a1043cad9cfd463.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/tables/c655b4f2986b6dce04f8e66b918be02c851c56e55402ff176a1043cad9cfd463.jpg)

![c713d6f75e23dad3cb390c3ecd0574e12483bd685622c643cdc0922d52e0ba72.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/tables/c713d6f75e23dad3cb390c3ecd0574e12483bd685622c643cdc0922d52e0ba72.jpg)

![e0218457cf0927d41bb87b92bc715567060d2ce286b1159ee40337772bd31d47.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/tables/e0218457cf0927d41bb87b92bc715567060d2ce286b1159ee40337772bd31d47.jpg)

![e4fa37891685a746eaa4eb3df5adfca545cdad9d6b09bc06475fcbc464410a15.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/tables/e4fa37891685a746eaa4eb3df5adfca545cdad9d6b09bc06475fcbc464410a15.jpg)

![eb12c6090a6b7dd771cb0aa1706472bfdf164a27f26bfff2ceb90b8c1a80a36d.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/tables/eb12c6090a6b7dd771cb0aa1706472bfdf164a27f26bfff2ceb90b8c1a80a36d.jpg)

![eb6e1f7c35712f641c774b006ef8f1c605790714bed98da818bfc015176e9c10.jpg](../icml_results/2958_Self-Guidance_%20Enhancing%20Neural%20Codecs%20via%20Decoder%20Manifold%20Alignment/tables/eb6e1f7c35712f641c774b006ef8f1c605790714bed98da818bfc015176e9c10.jpg)

<a id="paper-2921"></a>

## Lavida-R1: Advancing Reasoning for Unified Multimodal Diffusion Language Models


### Images

![01025a4c030298ca8e7a98705575f3b72ef6efefc30bedd9e343964fd3123dbf.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/01025a4c030298ca8e7a98705575f3b72ef6efefc30bedd9e343964fd3123dbf.jpg)

![08e9b8d32819e43af845c5a82a4364e791060b56fd47ee60f7ad20d6d0d457e8.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/08e9b8d32819e43af845c5a82a4364e791060b56fd47ee60f7ad20d6d0d457e8.jpg)

![1f2f00c9dc73bef58faf3c04b51c61fcc6a63e4b04d06fe1d9a5c8e57f098cb7.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/1f2f00c9dc73bef58faf3c04b51c61fcc6a63e4b04d06fe1d9a5c8e57f098cb7.jpg)

![23d39cca07e246ec5aab9a1bf551481a3bf51aa787d370adb03a4ccab800b5b1.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/23d39cca07e246ec5aab9a1bf551481a3bf51aa787d370adb03a4ccab800b5b1.jpg)

![3ca2459adf000b9883932d4dcb466ffe2173e71162cb218869210263687eec0f.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/3ca2459adf000b9883932d4dcb466ffe2173e71162cb218869210263687eec0f.jpg)

![3db932d29c75c396ef709d1c6c8ce28bf10307827c734930a4a5f2fa7b446bfb.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/3db932d29c75c396ef709d1c6c8ce28bf10307827c734930a4a5f2fa7b446bfb.jpg)

![3dd239e66f7df8913bf93ff859a940d0c9294e0f278bb2eb9a2b51804ee07ab0.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/3dd239e66f7df8913bf93ff859a940d0c9294e0f278bb2eb9a2b51804ee07ab0.jpg)

![3f61fbd25e8054c683bb6f8b10bfd2ddea78f8e6db77ca9b8d23756e1ce60a4f.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/3f61fbd25e8054c683bb6f8b10bfd2ddea78f8e6db77ca9b8d23756e1ce60a4f.jpg)

![4465b82ee8462c95506240f387f3b67fbae1e2209770b32787408ff621d02fe4.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/4465b82ee8462c95506240f387f3b67fbae1e2209770b32787408ff621d02fe4.jpg)

![7f2e10198cf7f9b4c8e34d20cb1bbf47a228f98bebf0a13b89c4bc17ee3d395b.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/7f2e10198cf7f9b4c8e34d20cb1bbf47a228f98bebf0a13b89c4bc17ee3d395b.jpg)

![81ddf589451aedf6270ad78d09bdebfb1ea9ead93023b57b3041ed90b7c9a115.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/81ddf589451aedf6270ad78d09bdebfb1ea9ead93023b57b3041ed90b7c9a115.jpg)

![82a3a9091670e77a9a50cd5508c4e1432fddc73817e727146dd19ddb5306080d.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/82a3a9091670e77a9a50cd5508c4e1432fddc73817e727146dd19ddb5306080d.jpg)

![95620f72e301c89eeed4fdd63e6d922cb8eb80c558c535d2d0cb90ae45c22ec3.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/95620f72e301c89eeed4fdd63e6d922cb8eb80c558c535d2d0cb90ae45c22ec3.jpg)

![a1e302123d2eb07e8a667f2c53ebaedc52381996744d4368705fc6c9305c3810.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/a1e302123d2eb07e8a667f2c53ebaedc52381996744d4368705fc6c9305c3810.jpg)

![a7b83edd9977ff511103937fef8233388d2806c1efc3176ecd74b8cc118e51fb.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/a7b83edd9977ff511103937fef8233388d2806c1efc3176ecd74b8cc118e51fb.jpg)

![ab34b55148a3b4ddb0bcefa6fcec01efc635a2b909c5dc6abc3955adbe5c9218.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/ab34b55148a3b4ddb0bcefa6fcec01efc635a2b909c5dc6abc3955adbe5c9218.jpg)

![abd662e545526e8c1c4317fc3c180b4bd19e6135286384b14d7f8f0749905e3b.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/abd662e545526e8c1c4317fc3c180b4bd19e6135286384b14d7f8f0749905e3b.jpg)

![ac6e9c0c6fdcb25e41e94210436bd45d9416cada89e30855eee4bb984a5d81cc.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/ac6e9c0c6fdcb25e41e94210436bd45d9416cada89e30855eee4bb984a5d81cc.jpg)

![ca7bee4d37317196cb8beb5fe93165d745f01cd8d01252a2084a17a73c2e13cd.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/ca7bee4d37317196cb8beb5fe93165d745f01cd8d01252a2084a17a73c2e13cd.jpg)

![d3256f58c6734839ba67530f01e813d95f5601449a257108337f055ed9aee213.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/d3256f58c6734839ba67530f01e813d95f5601449a257108337f055ed9aee213.jpg)

![d5534009f58c98cc3c5266ad45502a5d8bb95e192b91620f92a16d14e67bfd85.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/d5534009f58c98cc3c5266ad45502a5d8bb95e192b91620f92a16d14e67bfd85.jpg)

![dd0b43e1b04829b9c82e5b732137cddf6e31cf0a8aa6566be2040065270279f2.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/dd0b43e1b04829b9c82e5b732137cddf6e31cf0a8aa6566be2040065270279f2.jpg)

![fc61ea474df8d615ecd0e87828567d35f329f264c9781e020b10a8c78ffb418d.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/images/fc61ea474df8d615ecd0e87828567d35f329f264c9781e020b10a8c78ffb418d.jpg)

### Tables

![2a105bd1d424f91a5e16db7dd4a08818e24fedca46c5993d47b790b15a805638.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/tables/2a105bd1d424f91a5e16db7dd4a08818e24fedca46c5993d47b790b15a805638.jpg)

![3910620df563e70f795b21971e43b7b169f5255c8765a6a0cace8fdf70a48c09.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/tables/3910620df563e70f795b21971e43b7b169f5255c8765a6a0cace8fdf70a48c09.jpg)

![5214794a7bf3b02e6ff3c81ec0e5fb7a82e5d20739f5ad08d065d214f7cf8b91.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/tables/5214794a7bf3b02e6ff3c81ec0e5fb7a82e5d20739f5ad08d065d214f7cf8b91.jpg)

![57254547e2b9df1738a19d86a7685f035ce94e100610949545ccc80110b3d39b.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/tables/57254547e2b9df1738a19d86a7685f035ce94e100610949545ccc80110b3d39b.jpg)

![89adb3af6f9c4f832a06dd62f542d410d72468b41d99be4e48cffa16991a436c.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/tables/89adb3af6f9c4f832a06dd62f542d410d72468b41d99be4e48cffa16991a436c.jpg)

![9ade1f549b14e721cce378905f57293eac9b8762914bc16244902e3b76ca6f99.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/tables/9ade1f549b14e721cce378905f57293eac9b8762914bc16244902e3b76ca6f99.jpg)

![a2919c570d7b3b95d7602411b4409d5daed29590147760a972341a9047a62708.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/tables/a2919c570d7b3b95d7602411b4409d5daed29590147760a972341a9047a62708.jpg)

![c3c63664ce0ce264db1eac5613e89e83099ed5108bd052dbd208932a342a10d9.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/tables/c3c63664ce0ce264db1eac5613e89e83099ed5108bd052dbd208932a342a10d9.jpg)

![ca58b9d815fd066ce11a17c6ec53720601c71b8ae029cc64e5b3a1c45db09997.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/tables/ca58b9d815fd066ce11a17c6ec53720601c71b8ae029cc64e5b3a1c45db09997.jpg)

![dc0b1c3cd30509f439bc549242941170fa4e781d3ff1204cbb021368b8e81d11.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/tables/dc0b1c3cd30509f439bc549242941170fa4e781d3ff1204cbb021368b8e81d11.jpg)

![f08f57dbddb6aecd3e08cb572ab7772b5ba6d9d94e82fcc335eb76948bf2047d.jpg](../icml_results/2959_Lavida-R1_%20Advancing%20Reasoning%20for%20Unified%20Multimodal%20Diffusion%20Language%20Models/tables/f08f57dbddb6aecd3e08cb572ab7772b5ba6d9d94e82fcc335eb76948bf2047d.jpg)

<a id="paper-2922"></a>

## TransLight: Image-Guided Customized Lighting Control with Generative Decoupling


### Images

![00061a4ff46b076d0341025ea05138ce7c6a894df977697525bd55271c40f234.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/images/00061a4ff46b076d0341025ea05138ce7c6a894df977697525bd55271c40f234.jpg)

![21484a6908c132a4c68a5e7c71ccefabfb67d0b67f9ece19d4ea4a84b6e98434.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/images/21484a6908c132a4c68a5e7c71ccefabfb67d0b67f9ece19d4ea4a84b6e98434.jpg)

![25a8dcef93b48df6b4081b2b435d8b2434585ebd04ddc3a35a7246acf1f004ba.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/images/25a8dcef93b48df6b4081b2b435d8b2434585ebd04ddc3a35a7246acf1f004ba.jpg)

![40d62a618d1568d3f9fd8f17c5816bca04d86386a418a09ae1096e9eac3502a1.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/images/40d62a618d1568d3f9fd8f17c5816bca04d86386a418a09ae1096e9eac3502a1.jpg)

![497dd526acbecb9037e426f6b4dfec33ffc7ab25d457fce4f623e3cbab3e0248.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/images/497dd526acbecb9037e426f6b4dfec33ffc7ab25d457fce4f623e3cbab3e0248.jpg)

![6aecced0786da8c464c98f186a2b0aa45a1cc66f68860ec1861eebbbe7f94daf.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/images/6aecced0786da8c464c98f186a2b0aa45a1cc66f68860ec1861eebbbe7f94daf.jpg)

![738d3399999df39f18c40221500286aebbcb4d59e694808d24374928eb2edad1.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/images/738d3399999df39f18c40221500286aebbcb4d59e694808d24374928eb2edad1.jpg)

![7c9d77a1be8f887235bf5c1533e143fe35b10a63ee01a9a0387719178f57bbf1.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/images/7c9d77a1be8f887235bf5c1533e143fe35b10a63ee01a9a0387719178f57bbf1.jpg)

![81171d5bdbad1e1c96e9af3625baffb66a127a5b7eb03816cdf0a7deae16c4fb.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/images/81171d5bdbad1e1c96e9af3625baffb66a127a5b7eb03816cdf0a7deae16c4fb.jpg)

![b0c1438e212deabbfaa3ff9a21ffdd8078467f8a95931cbe0182c17ccbffb59a.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/images/b0c1438e212deabbfaa3ff9a21ffdd8078467f8a95931cbe0182c17ccbffb59a.jpg)

![b8e60e6bffd620ec6b29850962fc78ca22de2a9c81bcaba413eaa9e2327069bc.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/images/b8e60e6bffd620ec6b29850962fc78ca22de2a9c81bcaba413eaa9e2327069bc.jpg)

![bf25f98ccdd2aa996b647a5eb0024a2094409fc6b955bae9a043ce19ed6f9023.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/images/bf25f98ccdd2aa996b647a5eb0024a2094409fc6b955bae9a043ce19ed6f9023.jpg)

![c478cc6fba01272e27ee4fd64da487ec71cb8bc7de8f442ce980e15fd88c31c2.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/images/c478cc6fba01272e27ee4fd64da487ec71cb8bc7de8f442ce980e15fd88c31c2.jpg)

![ddf4a41005066ed28b5220e3b4269b98b637b10ba63a8c19b9fe3b8b61ff9eb1.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/images/ddf4a41005066ed28b5220e3b4269b98b637b10ba63a8c19b9fe3b8b61ff9eb1.jpg)

### Tables

![366d1361029ad9fc5e49fef73cb31299f341b7bbb727dfd1e90af642a1b4aa1d.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/tables/366d1361029ad9fc5e49fef73cb31299f341b7bbb727dfd1e90af642a1b4aa1d.jpg)

![46e4ec446d5d9f06f6a464f5a30a7697c243a3778c80519fb753aeb158a08acb.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/tables/46e4ec446d5d9f06f6a464f5a30a7697c243a3778c80519fb753aeb158a08acb.jpg)

![5306c6c3b84ebe9c9aa8e91ae7bbb3336a5f5184cfdd981088df26497e45d1a6.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/tables/5306c6c3b84ebe9c9aa8e91ae7bbb3336a5f5184cfdd981088df26497e45d1a6.jpg)

![8c8802b7460fea0cf8984f7f99f0f429c75248f388faedcb9cb49da38f620ce7.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/tables/8c8802b7460fea0cf8984f7f99f0f429c75248f388faedcb9cb49da38f620ce7.jpg)

![a4cd21ef7210ddb6ed21204e5cf3abdc88c145f211a467acc0ace4d0b5e9e105.jpg](../icml_results/2960_TransLight_%20Image-Guided%20Customized%20Lighting%20Control%20with%20Generative%20Decoupling/tables/a4cd21ef7210ddb6ed21204e5cf3abdc88c145f211a467acc0ace4d0b5e9e105.jpg)

<a id="paper-2923"></a>

## COPF: An Online Framework for Deployment-Stable Counterfactual Fairness in Evolving Graphs


### Images

![27dee36e6ee5274758afacbdd8d65f291463021839517dc99dc1411b45e1bf81.jpg](../icml_results/2961_COPF_%20An%20Online%20Framework%20for%20Deployment-Stable%20Counterfactual%20Fairness%20in%20Evolving%20Graphs/images/27dee36e6ee5274758afacbdd8d65f291463021839517dc99dc1411b45e1bf81.jpg)

![300c1d4871d9de3f9dd6eb1f87899dd432568d7e029c26ee7cb923ce6526cc90.jpg](../icml_results/2961_COPF_%20An%20Online%20Framework%20for%20Deployment-Stable%20Counterfactual%20Fairness%20in%20Evolving%20Graphs/images/300c1d4871d9de3f9dd6eb1f87899dd432568d7e029c26ee7cb923ce6526cc90.jpg)

![67fdac11262d1e3558ef90ecfa23f3139c701bb1f7bbdb2254da87ce9b1f7839.jpg](../icml_results/2961_COPF_%20An%20Online%20Framework%20for%20Deployment-Stable%20Counterfactual%20Fairness%20in%20Evolving%20Graphs/images/67fdac11262d1e3558ef90ecfa23f3139c701bb1f7bbdb2254da87ce9b1f7839.jpg)

![7aeeaa21ce78dd3442eca6c03dffacd3db42078ba90ec7cd9074292ad552a853.jpg](../icml_results/2961_COPF_%20An%20Online%20Framework%20for%20Deployment-Stable%20Counterfactual%20Fairness%20in%20Evolving%20Graphs/images/7aeeaa21ce78dd3442eca6c03dffacd3db42078ba90ec7cd9074292ad552a853.jpg)

![9ba5bfb84b93ac73d36a90c294f5ac91677410b97fa833cb77af671e3b8fbfd0.jpg](../icml_results/2961_COPF_%20An%20Online%20Framework%20for%20Deployment-Stable%20Counterfactual%20Fairness%20in%20Evolving%20Graphs/images/9ba5bfb84b93ac73d36a90c294f5ac91677410b97fa833cb77af671e3b8fbfd0.jpg)

![b446c423f1a4969024fc585cd30a4119eb32140bebdbc9f14ff88f9b56cf3ab3.jpg](../icml_results/2961_COPF_%20An%20Online%20Framework%20for%20Deployment-Stable%20Counterfactual%20Fairness%20in%20Evolving%20Graphs/images/b446c423f1a4969024fc585cd30a4119eb32140bebdbc9f14ff88f9b56cf3ab3.jpg)

![e4b9aebe51d703f66fc6213f374c5fc7d435854fa629cdb009363812624b95d0.jpg](../icml_results/2961_COPF_%20An%20Online%20Framework%20for%20Deployment-Stable%20Counterfactual%20Fairness%20in%20Evolving%20Graphs/images/e4b9aebe51d703f66fc6213f374c5fc7d435854fa629cdb009363812624b95d0.jpg)

![f748474e6a8e91b3d6d55dad21eb1cf3b2b50bc55b83fe1c493cdac39cebd709.jpg](../icml_results/2961_COPF_%20An%20Online%20Framework%20for%20Deployment-Stable%20Counterfactual%20Fairness%20in%20Evolving%20Graphs/images/f748474e6a8e91b3d6d55dad21eb1cf3b2b50bc55b83fe1c493cdac39cebd709.jpg)

### Tables

![35df0e7d8455e4949edfa1cd07e70d4fe0a8fbe70250e22cac097382abc63868.jpg](../icml_results/2961_COPF_%20An%20Online%20Framework%20for%20Deployment-Stable%20Counterfactual%20Fairness%20in%20Evolving%20Graphs/tables/35df0e7d8455e4949edfa1cd07e70d4fe0a8fbe70250e22cac097382abc63868.jpg)

![4b748c6d50d816863edb6ef6a17147f479244a9ae1f2044c0631e82d03988f33.jpg](../icml_results/2961_COPF_%20An%20Online%20Framework%20for%20Deployment-Stable%20Counterfactual%20Fairness%20in%20Evolving%20Graphs/tables/4b748c6d50d816863edb6ef6a17147f479244a9ae1f2044c0631e82d03988f33.jpg)

![7298d0993f2a8b8a955eb13f470c0e1b82bf156e63412ea21e3406e625005cba.jpg](../icml_results/2961_COPF_%20An%20Online%20Framework%20for%20Deployment-Stable%20Counterfactual%20Fairness%20in%20Evolving%20Graphs/tables/7298d0993f2a8b8a955eb13f470c0e1b82bf156e63412ea21e3406e625005cba.jpg)

![7f6e2e8b4f7c96e8713b5d400219f7d8724ded7109df82ccdec4be107ae2b9c6.jpg](../icml_results/2961_COPF_%20An%20Online%20Framework%20for%20Deployment-Stable%20Counterfactual%20Fairness%20in%20Evolving%20Graphs/tables/7f6e2e8b4f7c96e8713b5d400219f7d8724ded7109df82ccdec4be107ae2b9c6.jpg)

![aae876fb3ea9d4abfa95181cb01283cf626a499382c7f43094064ab7c6b4f639.jpg](../icml_results/2961_COPF_%20An%20Online%20Framework%20for%20Deployment-Stable%20Counterfactual%20Fairness%20in%20Evolving%20Graphs/tables/aae876fb3ea9d4abfa95181cb01283cf626a499382c7f43094064ab7c6b4f639.jpg)

![b8d701ed39f29968833c6fe8ec4694df9ebd6cda2c221c980f294fbc038513bd.jpg](../icml_results/2961_COPF_%20An%20Online%20Framework%20for%20Deployment-Stable%20Counterfactual%20Fairness%20in%20Evolving%20Graphs/tables/b8d701ed39f29968833c6fe8ec4694df9ebd6cda2c221c980f294fbc038513bd.jpg)

![f3eebc728800406b14a81936b400bbc029037369e7b64b44bba645e1596c7b11.jpg](../icml_results/2961_COPF_%20An%20Online%20Framework%20for%20Deployment-Stable%20Counterfactual%20Fairness%20in%20Evolving%20Graphs/tables/f3eebc728800406b14a81936b400bbc029037369e7b64b44bba645e1596c7b11.jpg)

<a id="paper-2924"></a>

## Enhancing Membership Inference Attacks on Diffusion Models from a Frequency-Domain Perspective


### Images

![0f2364c77ba9bb90a51aabaddac06f8ad6ad035d11cf1274e3d660a90bc95729.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/images/0f2364c77ba9bb90a51aabaddac06f8ad6ad035d11cf1274e3d660a90bc95729.jpg)

![175c6ba503231fbc3e222e5d066fb6cae5dc158b4939cf6abf3e2eaa9682487b.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/images/175c6ba503231fbc3e222e5d066fb6cae5dc158b4939cf6abf3e2eaa9682487b.jpg)

![1ba5296a9af80aec5900b4ae1dced37a793ea0e797215601875884c8f7c20bb2.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/images/1ba5296a9af80aec5900b4ae1dced37a793ea0e797215601875884c8f7c20bb2.jpg)

![21ff94adac7bfa612e9d8d10da9e7a311b60b923d8ef34e10ac84d58035b6cd9.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/images/21ff94adac7bfa612e9d8d10da9e7a311b60b923d8ef34e10ac84d58035b6cd9.jpg)

![282d784c1bc0753251dc1fc35c260cb887f921ac55a875bd50f3ea4d540441ef.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/images/282d784c1bc0753251dc1fc35c260cb887f921ac55a875bd50f3ea4d540441ef.jpg)

![39a08d7091d1a4464a59ae01f22fad7618428c1f1581ec9146fcb20480d69ca0.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/images/39a08d7091d1a4464a59ae01f22fad7618428c1f1581ec9146fcb20480d69ca0.jpg)

![4279cd9651a4733b103adc09d1855e15e899e42206065f0a1730052045458609.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/images/4279cd9651a4733b103adc09d1855e15e899e42206065f0a1730052045458609.jpg)

![a87b2e539ffb6286dc3e7bb4e72d036df0403e9102fd8e8ec004589214573e6b.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/images/a87b2e539ffb6286dc3e7bb4e72d036df0403e9102fd8e8ec004589214573e6b.jpg)

![aa40851a69bc85b943da2d6b2083f538392bfefd257089b66e9586ec1c0406da.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/images/aa40851a69bc85b943da2d6b2083f538392bfefd257089b66e9586ec1c0406da.jpg)

### Tables

![126665ee93c8fe5ad8bab09aac55a9356afa4ddba512c00e1855ad366b348c66.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/126665ee93c8fe5ad8bab09aac55a9356afa4ddba512c00e1855ad366b348c66.jpg)

![12ce794490e17c8479c701057f9202cc6ad09861b1de6d0aabc313a4842d2298.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/12ce794490e17c8479c701057f9202cc6ad09861b1de6d0aabc313a4842d2298.jpg)

![1464518a1374cab8ee4d53221b12e7789689aed99cdfb50ccf5e40690cfc6ebd.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/1464518a1374cab8ee4d53221b12e7789689aed99cdfb50ccf5e40690cfc6ebd.jpg)

![1f6e2d2467b2819bc6e85836a0602a4fc30928743352cdd61b0102e72e7cd021.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/1f6e2d2467b2819bc6e85836a0602a4fc30928743352cdd61b0102e72e7cd021.jpg)

![285c6508015d362ef592b251f46c7c67b3b6d7e90ef3a72615d241d993485473.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/285c6508015d362ef592b251f46c7c67b3b6d7e90ef3a72615d241d993485473.jpg)

![2e7bbeb74d06112b2017ba0b7fbf3673d44e7125e17d90ba227e11c823157749.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/2e7bbeb74d06112b2017ba0b7fbf3673d44e7125e17d90ba227e11c823157749.jpg)

![35c1d4858891270842899ab49f796ff52f95ac6d31c832d104a8f117c2a175d6.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/35c1d4858891270842899ab49f796ff52f95ac6d31c832d104a8f117c2a175d6.jpg)

![4210122cc188a9fedeecb1f226cc6ebfbdc4ed0d4a8a9a2909ff8eb8a5d19077.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/4210122cc188a9fedeecb1f226cc6ebfbdc4ed0d4a8a9a2909ff8eb8a5d19077.jpg)

![4512089f22c9996fc5ad87ba587bc41244eb430076a94d13855a76189ea9779c.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/4512089f22c9996fc5ad87ba587bc41244eb430076a94d13855a76189ea9779c.jpg)

![57428fdf7b6ba03f2ad08efd24e85db5cd197097abf58b63eec88a6ad7ea0d86.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/57428fdf7b6ba03f2ad08efd24e85db5cd197097abf58b63eec88a6ad7ea0d86.jpg)

![5e5960f82fdc0d429e791ad0f166c7ef21eddef22aa5f534bd70828c43809cac.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/5e5960f82fdc0d429e791ad0f166c7ef21eddef22aa5f534bd70828c43809cac.jpg)

![7777e32c67e1701cbf14cb01cebfa69ee4e6f839bad2aa7c258d2ef0cc349cb3.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/7777e32c67e1701cbf14cb01cebfa69ee4e6f839bad2aa7c258d2ef0cc349cb3.jpg)

![79e12090b03e019bb9206b72bb8f7dff8fdc4cb4bdcea971d97229020357524d.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/79e12090b03e019bb9206b72bb8f7dff8fdc4cb4bdcea971d97229020357524d.jpg)

![a3d3b4b635119ffcba394269cbc21b8230ec069a456d91a1531b6ec64cca2b84.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/a3d3b4b635119ffcba394269cbc21b8230ec069a456d91a1531b6ec64cca2b84.jpg)

![b252ab0cbf016d26691cb992f7d948ce82c8c0b731cfbda469637e5478ac4c75.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/b252ab0cbf016d26691cb992f7d948ce82c8c0b731cfbda469637e5478ac4c75.jpg)

![cef4f9881e143bd3a5f46f36afa905153a53797f78540485ec072b1625cd9afd.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/cef4f9881e143bd3a5f46f36afa905153a53797f78540485ec072b1625cd9afd.jpg)

![d07263aac46cc8acc829eaa78f8db5f8e59cc032c2cccf8ad7631e9772f5f56f.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/d07263aac46cc8acc829eaa78f8db5f8e59cc032c2cccf8ad7631e9772f5f56f.jpg)

![d5e0a945b3461e4adee2d9b9616265a6c7df779869de664a49da6c9e52f18a55.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/d5e0a945b3461e4adee2d9b9616265a6c7df779869de664a49da6c9e52f18a55.jpg)

![decbcf79000df27a88a99ab9e15ca0e64810817442034428ee68b15905d06ac9.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/decbcf79000df27a88a99ab9e15ca0e64810817442034428ee68b15905d06ac9.jpg)

![e9646a00d01c1ccd42d65729a9c19838e713e1acd429f8f3760c43c4ef54e1ab.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/e9646a00d01c1ccd42d65729a9c19838e713e1acd429f8f3760c43c4ef54e1ab.jpg)

![ee3c6eb08ef2f256109b6317d721e81bb879375e2bbcfcc44d4c705d956dfc40.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/ee3c6eb08ef2f256109b6317d721e81bb879375e2bbcfcc44d4c705d956dfc40.jpg)

![f85a8a8be310866b35c6c85c0d8c6acfe774bf4b1a37ea8a843fce6636bfc06a.jpg](../icml_results/2962_Enhancing%20Membership%20Inference%20Attacks%20on%20Diffusion%20Models%20from%20a%20Frequency-Domain%20Perspective/tables/f85a8a8be310866b35c6c85c0d8c6acfe774bf4b1a37ea8a843fce6636bfc06a.jpg)

<a id="paper-2925"></a>

## MapDream: Task-Driven Map Learning for Vision-Language Navigation


### Images

![0579590a31d99ba68c7ecaf4843ad3fbdc0992bc93a0fa1e18d8d6272b6dba18.jpg](../icml_results/2963_MapDream_%20Task-Driven%20Map%20Learning%20for%20Vision-Language%20Navigation/images/0579590a31d99ba68c7ecaf4843ad3fbdc0992bc93a0fa1e18d8d6272b6dba18.jpg)

![8c880e29bdfd6a75d6a51dcbb112b23f821b84dc8c9f5b292f4a0a54d8bf6c99.jpg](../icml_results/2963_MapDream_%20Task-Driven%20Map%20Learning%20for%20Vision-Language%20Navigation/images/8c880e29bdfd6a75d6a51dcbb112b23f821b84dc8c9f5b292f4a0a54d8bf6c99.jpg)

![b3cab7dcc71006c9f883b9dbe841a5c9a4b0913901f684d8b5eaada16c128c5d.jpg](../icml_results/2963_MapDream_%20Task-Driven%20Map%20Learning%20for%20Vision-Language%20Navigation/images/b3cab7dcc71006c9f883b9dbe841a5c9a4b0913901f684d8b5eaada16c128c5d.jpg)

![d46f5055f3c485b39a3f28742fce96593d662996afd1d4a606432eb1ecc62e36.jpg](../icml_results/2963_MapDream_%20Task-Driven%20Map%20Learning%20for%20Vision-Language%20Navigation/images/d46f5055f3c485b39a3f28742fce96593d662996afd1d4a606432eb1ecc62e36.jpg)

### Tables

![3bfa2c1999151b6d6c1a4938eb46c678108ed57c23e582f576e89bad153bfed4.jpg](../icml_results/2963_MapDream_%20Task-Driven%20Map%20Learning%20for%20Vision-Language%20Navigation/tables/3bfa2c1999151b6d6c1a4938eb46c678108ed57c23e582f576e89bad153bfed4.jpg)

![593469539d1a92f1c3573efd061ddc3da36b451e5a010e285363920237595f6b.jpg](../icml_results/2963_MapDream_%20Task-Driven%20Map%20Learning%20for%20Vision-Language%20Navigation/tables/593469539d1a92f1c3573efd061ddc3da36b451e5a010e285363920237595f6b.jpg)

![7f90c54810b9ca22051b40fc9d63921b429693177bc4a8d8c06ff7cc0ac59ae1.jpg](../icml_results/2963_MapDream_%20Task-Driven%20Map%20Learning%20for%20Vision-Language%20Navigation/tables/7f90c54810b9ca22051b40fc9d63921b429693177bc4a8d8c06ff7cc0ac59ae1.jpg)

![98cbbc78015287fb44bab0232b8212da98833028ecee520f8bfc1e18e7f9e88c.jpg](../icml_results/2963_MapDream_%20Task-Driven%20Map%20Learning%20for%20Vision-Language%20Navigation/tables/98cbbc78015287fb44bab0232b8212da98833028ecee520f8bfc1e18e7f9e88c.jpg)

![ee2e6ffac5176d3fe74249a526e40f327fb00290d9323c9a85021805b315e268.jpg](../icml_results/2963_MapDream_%20Task-Driven%20Map%20Learning%20for%20Vision-Language%20Navigation/tables/ee2e6ffac5176d3fe74249a526e40f327fb00290d9323c9a85021805b315e268.jpg)

<a id="paper-2926"></a>

## LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries


### Images

![187755dc1ec3bd73ba66cd2491ad9da770c3c98768459ae2fb670933e462c9ce.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/images/187755dc1ec3bd73ba66cd2491ad9da770c3c98768459ae2fb670933e462c9ce.jpg)

![18d944129bc32b04cfc10fd4e5605137ccb8b495177346eab694a448271f1d46.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/images/18d944129bc32b04cfc10fd4e5605137ccb8b495177346eab694a448271f1d46.jpg)

![2366518e1bf93a065aaae6572d6031ba5a5ab66cd15ddb2c0d28a596ef19a1a0.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/images/2366518e1bf93a065aaae6572d6031ba5a5ab66cd15ddb2c0d28a596ef19a1a0.jpg)

![39eb26686fc9eac0469a028cf53a3cc3fa75719bc7d3c5bc13a70fa613b44095.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/images/39eb26686fc9eac0469a028cf53a3cc3fa75719bc7d3c5bc13a70fa613b44095.jpg)

![6f7cf330c27218682422c813ca1f7241edf89ad9bfa021868d4f353259f968fe.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/images/6f7cf330c27218682422c813ca1f7241edf89ad9bfa021868d4f353259f968fe.jpg)

![b2f55b694a48b5419a6d30e67e647e97b6bdf4fe6b0ce37bc8b417883de5db82.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/images/b2f55b694a48b5419a6d30e67e647e97b6bdf4fe6b0ce37bc8b417883de5db82.jpg)

![bf483e85d9dd900b4f331f0d5c16ad6ce881eb4bd54014fc5327de684435268e.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/images/bf483e85d9dd900b4f331f0d5c16ad6ce881eb4bd54014fc5327de684435268e.jpg)

![c242ff70b49b38e469f6d48c7c0fd1d65f02236c3ce2a4cc90ea978852abbabe.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/images/c242ff70b49b38e469f6d48c7c0fd1d65f02236c3ce2a4cc90ea978852abbabe.jpg)

### Tables

![05d81b1b0e4ca0f73645dfa05b251f198fe9e5edca7f402b6b70217acd9414be.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/tables/05d81b1b0e4ca0f73645dfa05b251f198fe9e5edca7f402b6b70217acd9414be.jpg)

![1709787553d45400904c6f1d9a1d95cdc9e7d7a24323cadbc59551e8025c810a.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/tables/1709787553d45400904c6f1d9a1d95cdc9e7d7a24323cadbc59551e8025c810a.jpg)

![2b94353809964378975cab57392e0ed8a58b4b80d39d46f71b7154e78d892f6b.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/tables/2b94353809964378975cab57392e0ed8a58b4b80d39d46f71b7154e78d892f6b.jpg)

![6ed14ad79f2fad47547cf4754c7957925dbddd8c504827005b67e4219445499e.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/tables/6ed14ad79f2fad47547cf4754c7957925dbddd8c504827005b67e4219445499e.jpg)

![b0cff9d13442da4423ad5b4d07d664ae3989521e27de66d8a2849dc5a1025da9.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/tables/b0cff9d13442da4423ad5b4d07d664ae3989521e27de66d8a2849dc5a1025da9.jpg)

![b1f6cca1c39e6875ced62635d715fcf74a68ea2876a889dec950a878c8882c4d.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/tables/b1f6cca1c39e6875ced62635d715fcf74a68ea2876a889dec950a878c8882c4d.jpg)

![ce7c7dd3089b6383896438f54bc810a6b2ece898293b677f28585f3530316d76.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/tables/ce7c7dd3089b6383896438f54bc810a6b2ece898293b677f28585f3530316d76.jpg)

![d8bfbcdbc7a1149fa795f83f6a7e126b0b602ef618b989b1f3931f87fd8e6091.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/tables/d8bfbcdbc7a1149fa795f83f6a7e126b0b602ef618b989b1f3931f87fd8e6091.jpg)

![e5d4bf600cb5d0a2ca7c14887b24f1f8a6819894def532c55c383438a0cd223e.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/tables/e5d4bf600cb5d0a2ca7c14887b24f1f8a6819894def532c55c383438a0cd223e.jpg)

![eb383627da27e0354fa2ecfe4f7050f06e7752effbca0d4e298288a905686fa7.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/tables/eb383627da27e0354fa2ecfe4f7050f06e7752effbca0d4e298288a905686fa7.jpg)

![ef068d7c06c3229b95f1872fb61c1d3825377e72a03942d001c9cfcdc9370040.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/tables/ef068d7c06c3229b95f1872fb61c1d3825377e72a03942d001c9cfcdc9370040.jpg)

![f596017ac6caf929e34d221d7ab1a3e54913c37c8631f4fb98bf680cc9ee3d89.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/tables/f596017ac6caf929e34d221d7ab1a3e54913c37c8631f4fb98bf680cc9ee3d89.jpg)

![fe5caab355005102d56777d86e7899528d124e74244ec84a467ae3197b97ef6f.jpg](../icml_results/2964_LangForce_%20Bayesian%20Decomposition%20of%20Vision%20Language%20Action%20Models%20via%20Latent%20Action%20Queries/tables/fe5caab355005102d56777d86e7899528d124e74244ec84a467ae3197b97ef6f.jpg)

<a id="paper-2927"></a>

## Evidential Reasoning Advances Interpretable Real-World Disease Screening


### Images

![18ba9315ddfc3c254a242b9945634ba9628fa966457814b2ca6d011ab9eba754.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/images/18ba9315ddfc3c254a242b9945634ba9628fa966457814b2ca6d011ab9eba754.jpg)

![231544b10aa3cc48d7a7e12fd1e2b9bd1fa95bc502cd42e83f36312efaae61a3.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/images/231544b10aa3cc48d7a7e12fd1e2b9bd1fa95bc502cd42e83f36312efaae61a3.jpg)

![289f5fcf2d7db2c9c6c3c181a04a149cb8607f86225050bf909a0b9274ec8782.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/images/289f5fcf2d7db2c9c6c3c181a04a149cb8607f86225050bf909a0b9274ec8782.jpg)

![3aa7112c3ef8e52c0aa48d3d76c42fc149bf80d0dd78b96823192d4142a18bd9.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/images/3aa7112c3ef8e52c0aa48d3d76c42fc149bf80d0dd78b96823192d4142a18bd9.jpg)

![50b9d9db71e9cc3f994d0f5675c6639446e60d871d6412f7654a5bfad52586ea.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/images/50b9d9db71e9cc3f994d0f5675c6639446e60d871d6412f7654a5bfad52586ea.jpg)

![9eee3ee37310f9ee7aee2e6379ff10d0cbe1a978b8c14002b0750cf50f466e56.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/images/9eee3ee37310f9ee7aee2e6379ff10d0cbe1a978b8c14002b0750cf50f466e56.jpg)

![9f3df7a2ce4a9ecfabec1b0057b4094418b10c651e634c95b8cbcbe68e38f02e.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/images/9f3df7a2ce4a9ecfabec1b0057b4094418b10c651e634c95b8cbcbe68e38f02e.jpg)

![b4a799985be725f8a4b5c46ea1e867064db3f07c61ca1237a5be89c7879bf8d2.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/images/b4a799985be725f8a4b5c46ea1e867064db3f07c61ca1237a5be89c7879bf8d2.jpg)

![c1cc0a16cf6bc768aed152a499cf48daa7b3ee89acf31df3b2f729d8b957248a.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/images/c1cc0a16cf6bc768aed152a499cf48daa7b3ee89acf31df3b2f729d8b957248a.jpg)

![c22bb933e666d6ca21cac8c34263daa14fd56911319840be8325acbcb0d3b2cb.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/images/c22bb933e666d6ca21cac8c34263daa14fd56911319840be8325acbcb0d3b2cb.jpg)

### Tables

![2598b9bb01eded868d44394212fd46fb06e93df35378fc3561eef26e0179b0b3.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/tables/2598b9bb01eded868d44394212fd46fb06e93df35378fc3561eef26e0179b0b3.jpg)

![34156e5dde571683741e2912d9cc4df8d83e565eba720aa45eaa1a03d1c1da42.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/tables/34156e5dde571683741e2912d9cc4df8d83e565eba720aa45eaa1a03d1c1da42.jpg)

![614409e6eb5bd02f722674919c99868218814cf23815387c275f23092fc9df85.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/tables/614409e6eb5bd02f722674919c99868218814cf23815387c275f23092fc9df85.jpg)

![b80288fa8412408b151f2ebc6aec531d7575557d3cc8ece873e823f7ec7b2b9f.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/tables/b80288fa8412408b151f2ebc6aec531d7575557d3cc8ece873e823f7ec7b2b9f.jpg)

![cbb7756f9e82bd24cbf0e6ba1d457e4f878baa7e14ff47fe18e075e84563cfd0.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/tables/cbb7756f9e82bd24cbf0e6ba1d457e4f878baa7e14ff47fe18e075e84563cfd0.jpg)

![cce717e54d9396150cf37ea9b0cf3f888519d73378519a4193efc3f1babe2c06.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/tables/cce717e54d9396150cf37ea9b0cf3f888519d73378519a4193efc3f1babe2c06.jpg)

![da6618cb6019517ad9a182969e338fe21e3b7ba3697bf5064366f6fe937057f1.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/tables/da6618cb6019517ad9a182969e338fe21e3b7ba3697bf5064366f6fe937057f1.jpg)

![fe7e69d6851fd332a6ca5c128e0b13b8bf49800390d8edf7a42e7adece6fb52e.jpg](../icml_results/2965_Evidential%20Reasoning%20Advances%20Interpretable%20Real-World%20Disease%20Screening/tables/fe7e69d6851fd332a6ca5c128e0b13b8bf49800390d8edf7a42e7adece6fb52e.jpg)

<a id="paper-2928"></a>

## Bipartite Graph Attention-based Clustering for Large-scale scRNA-seq Data


### Images

![0e3b6d6a2f778a7bc215035d8bbeef975ad8f2955088475450a0014c7a32cca1.jpg](../icml_results/2966_Bipartite%20Graph%20Attention-based%20Clustering%20for%20Large-scale%20scRNA-seq%20Data/images/0e3b6d6a2f778a7bc215035d8bbeef975ad8f2955088475450a0014c7a32cca1.jpg)

![2143cab708ebfc999cd3436dbb40cd5f1fb8a4042ad464fab115feb4a0933ec2.jpg](../icml_results/2966_Bipartite%20Graph%20Attention-based%20Clustering%20for%20Large-scale%20scRNA-seq%20Data/images/2143cab708ebfc999cd3436dbb40cd5f1fb8a4042ad464fab115feb4a0933ec2.jpg)

![3bf0e0c297fa38a2fcf812a990652956ed31aa03d58f25a724aabce35b462978.jpg](../icml_results/2966_Bipartite%20Graph%20Attention-based%20Clustering%20for%20Large-scale%20scRNA-seq%20Data/images/3bf0e0c297fa38a2fcf812a990652956ed31aa03d58f25a724aabce35b462978.jpg)

![5a3c7a02da458941b421a6dbfa3460dff23792eb1f657b72537ad9ca25283df2.jpg](../icml_results/2966_Bipartite%20Graph%20Attention-based%20Clustering%20for%20Large-scale%20scRNA-seq%20Data/images/5a3c7a02da458941b421a6dbfa3460dff23792eb1f657b72537ad9ca25283df2.jpg)

![786e651f6cd0aea72e6e8f28fe7e19022e6a3764f246aea90f53da8be4f5ef32.jpg](../icml_results/2966_Bipartite%20Graph%20Attention-based%20Clustering%20for%20Large-scale%20scRNA-seq%20Data/images/786e651f6cd0aea72e6e8f28fe7e19022e6a3764f246aea90f53da8be4f5ef32.jpg)

![7efcedb87bc4fbca6ca6907400ace09e9fda78a0553bbd56a3af8dda0dddc654.jpg](../icml_results/2966_Bipartite%20Graph%20Attention-based%20Clustering%20for%20Large-scale%20scRNA-seq%20Data/images/7efcedb87bc4fbca6ca6907400ace09e9fda78a0553bbd56a3af8dda0dddc654.jpg)

![98ac0f740284911b978973745653eeee51531e4f49141e479cbf3fe8f7a68a40.jpg](../icml_results/2966_Bipartite%20Graph%20Attention-based%20Clustering%20for%20Large-scale%20scRNA-seq%20Data/images/98ac0f740284911b978973745653eeee51531e4f49141e479cbf3fe8f7a68a40.jpg)

![9b491c53884c6dcd57ce57796bf56f5af26c4c939982c4ec2a6df2ea172df5c8.jpg](../icml_results/2966_Bipartite%20Graph%20Attention-based%20Clustering%20for%20Large-scale%20scRNA-seq%20Data/images/9b491c53884c6dcd57ce57796bf56f5af26c4c939982c4ec2a6df2ea172df5c8.jpg)

### Tables

![3f3e16cbd155efa83bdcbc466e99259ee11a4a5d4694872256624e87a8fa17be.jpg](../icml_results/2966_Bipartite%20Graph%20Attention-based%20Clustering%20for%20Large-scale%20scRNA-seq%20Data/tables/3f3e16cbd155efa83bdcbc466e99259ee11a4a5d4694872256624e87a8fa17be.jpg)

![4267f25be2e4831e7beac119fa83702ac9edd1a48fea963c43fb0b56b13ddac9.jpg](../icml_results/2966_Bipartite%20Graph%20Attention-based%20Clustering%20for%20Large-scale%20scRNA-seq%20Data/tables/4267f25be2e4831e7beac119fa83702ac9edd1a48fea963c43fb0b56b13ddac9.jpg)

![487b886cca008e8eb50e7a643ecf24e710c9fdacd142277252d2ce97c6fe884d.jpg](../icml_results/2966_Bipartite%20Graph%20Attention-based%20Clustering%20for%20Large-scale%20scRNA-seq%20Data/tables/487b886cca008e8eb50e7a643ecf24e710c9fdacd142277252d2ce97c6fe884d.jpg)

![5c0df08073f97849ae6eed1cd60eac703178b4c08e67d1afce951dadafc7c035.jpg](../icml_results/2966_Bipartite%20Graph%20Attention-based%20Clustering%20for%20Large-scale%20scRNA-seq%20Data/tables/5c0df08073f97849ae6eed1cd60eac703178b4c08e67d1afce951dadafc7c035.jpg)

![5c38dba85423ae8586d1c370ce88a3453407749e1a2de04c34d15ee5795ad556.jpg](../icml_results/2966_Bipartite%20Graph%20Attention-based%20Clustering%20for%20Large-scale%20scRNA-seq%20Data/tables/5c38dba85423ae8586d1c370ce88a3453407749e1a2de04c34d15ee5795ad556.jpg)

![6f6d8b3c0017e4b408c73179968e19e7fc6cfcd102e8482a2f9224bc7165c277.jpg](../icml_results/2966_Bipartite%20Graph%20Attention-based%20Clustering%20for%20Large-scale%20scRNA-seq%20Data/tables/6f6d8b3c0017e4b408c73179968e19e7fc6cfcd102e8482a2f9224bc7165c277.jpg)

![72b0f75adae7b86c262310d53b98cf06f8d2430dec7a46f75f04e59ec007f186.jpg](../icml_results/2966_Bipartite%20Graph%20Attention-based%20Clustering%20for%20Large-scale%20scRNA-seq%20Data/tables/72b0f75adae7b86c262310d53b98cf06f8d2430dec7a46f75f04e59ec007f186.jpg)

![75392d286d42a560f025b9c91fd41aeef83ef3b8b74b974a6fa3dc1a4a804eda.jpg](../icml_results/2966_Bipartite%20Graph%20Attention-based%20Clustering%20for%20Large-scale%20scRNA-seq%20Data/tables/75392d286d42a560f025b9c91fd41aeef83ef3b8b74b974a6fa3dc1a4a804eda.jpg)

![eb696895addc36c39f4be24573d46da4128c64b19d506aaee281a090897d9876.jpg](../icml_results/2966_Bipartite%20Graph%20Attention-based%20Clustering%20for%20Large-scale%20scRNA-seq%20Data/tables/eb696895addc36c39f4be24573d46da4128c64b19d506aaee281a090897d9876.jpg)

<a id="paper-2929"></a>

## Hyperspectral Image Fusion with Spectral-Band and Fusion-Scale Agnosticism


### Images

![043d08116a3143ad97e808fc95a4918af97b1eda18e90768df100ced54d3a04f.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/images/043d08116a3143ad97e808fc95a4918af97b1eda18e90768df100ced54d3a04f.jpg)

![059f8ac8e1918c996305b33c16f7b5950a84afc641ac7bd8a1144143a5ec8436.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/images/059f8ac8e1918c996305b33c16f7b5950a84afc641ac7bd8a1144143a5ec8436.jpg)

![19526dc9b8fabfa8c4ddac57c219aaf40fc78fc3a7b914f3bdb7c3fcddec062e.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/images/19526dc9b8fabfa8c4ddac57c219aaf40fc78fc3a7b914f3bdb7c3fcddec062e.jpg)

![27505a8a6ef338902760ffe1551b1320b1aa1f1dd9b537ad93c564fabb2602eb.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/images/27505a8a6ef338902760ffe1551b1320b1aa1f1dd9b537ad93c564fabb2602eb.jpg)

![38f9203ae25ed49cfecc633b5ce0f5a194564d98f4cd5bbe934e4950fa1ea63b.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/images/38f9203ae25ed49cfecc633b5ce0f5a194564d98f4cd5bbe934e4950fa1ea63b.jpg)

![8142fd40a8e0b2f5c90c4995cf67a48f51058f222d5327bff4a74f6cfa814311.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/images/8142fd40a8e0b2f5c90c4995cf67a48f51058f222d5327bff4a74f6cfa814311.jpg)

![907f94d72e9af7dfae1095217e2be51e7cd4db7c95fa5b8984ec64fdd9058296.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/images/907f94d72e9af7dfae1095217e2be51e7cd4db7c95fa5b8984ec64fdd9058296.jpg)

![93e505419bfd873f25bd0a7bce18d91e36432d9d2b8c6c8af2aecd86ca6c7f86.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/images/93e505419bfd873f25bd0a7bce18d91e36432d9d2b8c6c8af2aecd86ca6c7f86.jpg)

![b9ce2c968809cdcd19229e8e232f62092b25293dc4641b48bb1faa24220ccb31.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/images/b9ce2c968809cdcd19229e8e232f62092b25293dc4641b48bb1faa24220ccb31.jpg)

![c891294f4108d1735bed7ce49eabadf1577aecdaf6f5dfa3690eb7ee19d5ce9e.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/images/c891294f4108d1735bed7ce49eabadf1577aecdaf6f5dfa3690eb7ee19d5ce9e.jpg)

![cf96501a445e1f4b93e165d6e6032835e87ae44dd32267a524a805ec6a611dd4.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/images/cf96501a445e1f4b93e165d6e6032835e87ae44dd32267a524a805ec6a611dd4.jpg)

![ffde6ac52383863254870a3cc0c7d8eff47768e7b8883a453d0f4584be9db644.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/images/ffde6ac52383863254870a3cc0c7d8eff47768e7b8883a453d0f4584be9db644.jpg)

### Tables

![01ff443d4ad7eb61607ebb95376f6635c5536f9809dacca9f57cac0c059234ca.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/tables/01ff443d4ad7eb61607ebb95376f6635c5536f9809dacca9f57cac0c059234ca.jpg)

![1374e402c1afb13b3eccd40ad3c91eba1ec117b3a180fd665ad421dd8347712e.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/tables/1374e402c1afb13b3eccd40ad3c91eba1ec117b3a180fd665ad421dd8347712e.jpg)

![1cea2be97f38413a700e5f7350f90c1849379316426cac171f40183c3ad5a808.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/tables/1cea2be97f38413a700e5f7350f90c1849379316426cac171f40183c3ad5a808.jpg)

![2133198b3872fb0a9e3f8e78d99894aa3d8a21870cbe9e9c6d5164edd060a417.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/tables/2133198b3872fb0a9e3f8e78d99894aa3d8a21870cbe9e9c6d5164edd060a417.jpg)

![26a4baf1e204d4aef4876b1fa314c7ec4ba081bc107c98becc03f1d27cf998e2.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/tables/26a4baf1e204d4aef4876b1fa314c7ec4ba081bc107c98becc03f1d27cf998e2.jpg)

![2d9e6fd8f3e645b527658b9e0741a1d634613409cd793a93b92423385de4bd90.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/tables/2d9e6fd8f3e645b527658b9e0741a1d634613409cd793a93b92423385de4bd90.jpg)

![3873a5347b6f1fb0c47ea4166a1244f3310bb10a0ee3e45d0af6e87e0de828e5.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/tables/3873a5347b6f1fb0c47ea4166a1244f3310bb10a0ee3e45d0af6e87e0de828e5.jpg)

![5a43723ef1afb53e795ffba729d31c7016d58707ff3eb3bc751c3f076daad81d.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/tables/5a43723ef1afb53e795ffba729d31c7016d58707ff3eb3bc751c3f076daad81d.jpg)

![5d8bcd3aa87bbe05b30991c3a17758d712069d63ac6045a7498ccd929d9bc1ae.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/tables/5d8bcd3aa87bbe05b30991c3a17758d712069d63ac6045a7498ccd929d9bc1ae.jpg)

![642786ccf3fbe017a8777e0528bacf92a08c0beddecda964b907503ea5fc91ec.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/tables/642786ccf3fbe017a8777e0528bacf92a08c0beddecda964b907503ea5fc91ec.jpg)

![6b76ae9d6589c72d4c30c99969e959293f3c88aacfb06284db7139a0f7652a3d.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/tables/6b76ae9d6589c72d4c30c99969e959293f3c88aacfb06284db7139a0f7652a3d.jpg)

![72433dd66fab4c4cfd1b6c2a925e41bdbfa4440b2f20c1ae920bdc13142231b1.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/tables/72433dd66fab4c4cfd1b6c2a925e41bdbfa4440b2f20c1ae920bdc13142231b1.jpg)

![88e19ff140148b403a198e1015ab672036ea02ab4a25b357e7a489a9f2cb1ab5.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/tables/88e19ff140148b403a198e1015ab672036ea02ab4a25b357e7a489a9f2cb1ab5.jpg)

![d3a17ba98d059b19c03f6ce94c33a70c1077b796cb1d5a323681aecdd1024fcf.jpg](../icml_results/2967_Hyperspectral%20Image%20Fusion%20with%20Spectral-Band%20and%20Fusion-Scale%20Agnosticism/tables/d3a17ba98d059b19c03f6ce94c33a70c1077b796cb1d5a323681aecdd1024fcf.jpg)

<a id="paper-2930"></a>

## Geometric Decoupling: Diagnosing the Structural Instability of Latent


### Images

![1ece2358718ce7e192ec8a34f0eec1570815ff5a0c58c540e65beff5d32a7fac.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/images/1ece2358718ce7e192ec8a34f0eec1570815ff5a0c58c540e65beff5d32a7fac.jpg)

![2eeb27be6603601f1f75b87d7ba58a5bdbd80b839231323532e23d38740bb664.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/images/2eeb27be6603601f1f75b87d7ba58a5bdbd80b839231323532e23d38740bb664.jpg)

![399786b747176688dd7f180698376a29da0c73a2588a695d0c2636683a382234.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/images/399786b747176688dd7f180698376a29da0c73a2588a695d0c2636683a382234.jpg)

### Tables

![44361239c150944dada2e785ffe730638cc5be0612eb4462660c95733a84dc13.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/44361239c150944dada2e785ffe730638cc5be0612eb4462660c95733a84dc13.jpg)

![4784ac3de527ed5f6a4a8a1f2303f10317fdb101678162027b46dcecac83abbc.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/4784ac3de527ed5f6a4a8a1f2303f10317fdb101678162027b46dcecac83abbc.jpg)

![7ada11c11f0ffed399a3dcdfe0c87411671db3f807fcbf8cf5ec9f9c4b937608.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/7ada11c11f0ffed399a3dcdfe0c87411671db3f807fcbf8cf5ec9f9c4b937608.jpg)

![7ede2e0872e7ec85fa4b67bf14181f749720ba9a89448fb26a0dea78a3d9dcd4.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/7ede2e0872e7ec85fa4b67bf14181f749720ba9a89448fb26a0dea78a3d9dcd4.jpg)

![837f62bc8c5ae1cfce862f5203f8bfb8baa2dfe748d5617285ac5743ee84e44f.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/837f62bc8c5ae1cfce862f5203f8bfb8baa2dfe748d5617285ac5743ee84e44f.jpg)

![956707ec08ac625960549f5b6e2929cf74d2d3c10a73052ce202ad51bb80f167.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/956707ec08ac625960549f5b6e2929cf74d2d3c10a73052ce202ad51bb80f167.jpg)

![a916ad7bf16c5e2add2ac51234bb64c70a9d27cc3a4f35c6d5798eaeef4766c2.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/a916ad7bf16c5e2add2ac51234bb64c70a9d27cc3a4f35c6d5798eaeef4766c2.jpg)

![b569639b428879835a968763fd53591398907f694a92b980d5bc40220e593e51.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/b569639b428879835a968763fd53591398907f694a92b980d5bc40220e593e51.jpg)

![b7b51b445d95db5ec00b52258d0eee2eb863a7be59cd13bd6fdde190430c984e.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/b7b51b445d95db5ec00b52258d0eee2eb863a7be59cd13bd6fdde190430c984e.jpg)

![c112045ede3777fd5b7f58fcd5d69c42929d77ce8264529bff975b76fa63db90.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/c112045ede3777fd5b7f58fcd5d69c42929d77ce8264529bff975b76fa63db90.jpg)

![c2aaccfbc8a60228740af69b4d9c08041ba6be42d8e47f660f01b8503fc82dff.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/c2aaccfbc8a60228740af69b4d9c08041ba6be42d8e47f660f01b8503fc82dff.jpg)

![c37fff1c92a8068ef0e713face7131cd84b42bb96fcdf6325fba66778fcd31b7.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/c37fff1c92a8068ef0e713face7131cd84b42bb96fcdf6325fba66778fcd31b7.jpg)

![cd16c60aead391cbd60f0d1dd8fafb2e34aeae42280572b325918909df8bb53f.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/cd16c60aead391cbd60f0d1dd8fafb2e34aeae42280572b325918909df8bb53f.jpg)

![d0e8790c08fc44223f95cd7207cb779c5a478a710628967251f13fb87543b054.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/d0e8790c08fc44223f95cd7207cb779c5a478a710628967251f13fb87543b054.jpg)

![d12341249dc254272b6dc57ded901c707b103b7af9a10c470f96430743287c46.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/d12341249dc254272b6dc57ded901c707b103b7af9a10c470f96430743287c46.jpg)

![d62bb23a429c80795b1f4609fd2dd00f742719f153b58117854aee9d5a60b0f9.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/d62bb23a429c80795b1f4609fd2dd00f742719f153b58117854aee9d5a60b0f9.jpg)

![dc0db553e0e009f1ceccc4cc8fae709272db3303dfea7823244abce1ebdc8c4a.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/dc0db553e0e009f1ceccc4cc8fae709272db3303dfea7823244abce1ebdc8c4a.jpg)

![eaf93049f83eb4cd7ff81d21ca34d316387987f2e6ba20087b2ac03c38b5c89a.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/eaf93049f83eb4cd7ff81d21ca34d316387987f2e6ba20087b2ac03c38b5c89a.jpg)

![eedf24144317b6a8ac6f48d205d54b99066cc872bbe949143142d4530b0e8b8b.jpg](../icml_results/2968_Geometric%20Decoupling_%20Diagnosing%20the%20Structural%20Instability%20of%20Latent/tables/eedf24144317b6a8ac6f48d205d54b99066cc872bbe949143142d4530b0e8b8b.jpg)

<a id="paper-2931"></a>

## An Evidential Route to Asymptotic Bayes Optimality under Sparsity


### Images

![433b639f126409fd0fee925ccb465ab391f3d2a141c0d1db2bb06844590afc34.jpg](../icml_results/2969_An%20Evidential%20Route%20to%20Asymptotic%20Bayes%20Optimality%20under%20Sparsity/images/433b639f126409fd0fee925ccb465ab391f3d2a141c0d1db2bb06844590afc34.jpg)

![4a3990f7fb45d56f68d2f08802b8e5911506d085f3a9c8d6de206d8dd076e705.jpg](../icml_results/2969_An%20Evidential%20Route%20to%20Asymptotic%20Bayes%20Optimality%20under%20Sparsity/images/4a3990f7fb45d56f68d2f08802b8e5911506d085f3a9c8d6de206d8dd076e705.jpg)

![4bd94dca103928a23569544bd1fdb2c3f54fbc30da2ac6b460fe31f222eb78cc.jpg](../icml_results/2969_An%20Evidential%20Route%20to%20Asymptotic%20Bayes%20Optimality%20under%20Sparsity/images/4bd94dca103928a23569544bd1fdb2c3f54fbc30da2ac6b460fe31f222eb78cc.jpg)

<a id="paper-2932"></a>

## Causal Representation Learning with Optimal Compression under Complex Treatments


### Images

![0acc8a80ec657e2b48c6acc74c36861e3d0b574c62a422a6fdcccd85ceeb2c22.jpg](../icml_results/2970_Causal%20Representation%20Learning%20with%20Optimal%20Compression%20under%20Complex%20Treatments/images/0acc8a80ec657e2b48c6acc74c36861e3d0b574c62a422a6fdcccd85ceeb2c22.jpg)

![10e15c435af1942a81543bb68569e775902c6111d0551d017602852a5066ea13.jpg](../icml_results/2970_Causal%20Representation%20Learning%20with%20Optimal%20Compression%20under%20Complex%20Treatments/images/10e15c435af1942a81543bb68569e775902c6111d0551d017602852a5066ea13.jpg)

![43650f0fd22a5dd6450770f4509d03dc0c23865451a0bcf7e7f947562a475cea.jpg](../icml_results/2970_Causal%20Representation%20Learning%20with%20Optimal%20Compression%20under%20Complex%20Treatments/images/43650f0fd22a5dd6450770f4509d03dc0c23865451a0bcf7e7f947562a475cea.jpg)

![60fb6cca938c1dd3ade1e2ce6a3bdb8c65ee7d7b7589fd1a60e58b46722c01d1.jpg](../icml_results/2970_Causal%20Representation%20Learning%20with%20Optimal%20Compression%20under%20Complex%20Treatments/images/60fb6cca938c1dd3ade1e2ce6a3bdb8c65ee7d7b7589fd1a60e58b46722c01d1.jpg)

![6afa85f62f0c71a3e94b1b9e5038e2a5972c23b64d04a67cca8620699401e297.jpg](../icml_results/2970_Causal%20Representation%20Learning%20with%20Optimal%20Compression%20under%20Complex%20Treatments/images/6afa85f62f0c71a3e94b1b9e5038e2a5972c23b64d04a67cca8620699401e297.jpg)

![6cf7bbfa12c3ddcb6ea4ee0051b7a4e79c125a724fed18d6a427d370930d0234.jpg](../icml_results/2970_Causal%20Representation%20Learning%20with%20Optimal%20Compression%20under%20Complex%20Treatments/images/6cf7bbfa12c3ddcb6ea4ee0051b7a4e79c125a724fed18d6a427d370930d0234.jpg)

![710959d0063441b8e84e56a9a7895ddb1da247ae6dfb7ae53b4d916ed4799452.jpg](../icml_results/2970_Causal%20Representation%20Learning%20with%20Optimal%20Compression%20under%20Complex%20Treatments/images/710959d0063441b8e84e56a9a7895ddb1da247ae6dfb7ae53b4d916ed4799452.jpg)

![766a6c39d7c7e84d13af523a6374c64a8a1dfa4ee25ed0e7f8781414a2e0bf50.jpg](../icml_results/2970_Causal%20Representation%20Learning%20with%20Optimal%20Compression%20under%20Complex%20Treatments/images/766a6c39d7c7e84d13af523a6374c64a8a1dfa4ee25ed0e7f8781414a2e0bf50.jpg)

![78eb831d3b83a255a2e5d9489d781f49d904a53a300eae9183009d22e5709fd6.jpg](../icml_results/2970_Causal%20Representation%20Learning%20with%20Optimal%20Compression%20under%20Complex%20Treatments/images/78eb831d3b83a255a2e5d9489d781f49d904a53a300eae9183009d22e5709fd6.jpg)

![7954833aa7935788296cd9e9c9a31a0f16acca8843f4c9a1ef41b259d6b02a51.jpg](../icml_results/2970_Causal%20Representation%20Learning%20with%20Optimal%20Compression%20under%20Complex%20Treatments/images/7954833aa7935788296cd9e9c9a31a0f16acca8843f4c9a1ef41b259d6b02a51.jpg)

![a5279b35f555c4ac7b02cd7b0db43582a17134d5dd0e56ef39febdd147c19d0b.jpg](../icml_results/2970_Causal%20Representation%20Learning%20with%20Optimal%20Compression%20under%20Complex%20Treatments/images/a5279b35f555c4ac7b02cd7b0db43582a17134d5dd0e56ef39febdd147c19d0b.jpg)

![bb9b162e2f72aed76c7e0d5b901604173b454f9ea8ae2dd202c6ae6402161bf4.jpg](../icml_results/2970_Causal%20Representation%20Learning%20with%20Optimal%20Compression%20under%20Complex%20Treatments/images/bb9b162e2f72aed76c7e0d5b901604173b454f9ea8ae2dd202c6ae6402161bf4.jpg)

![fa927622fa7d97daa7a0b13ad2151dd7b98d87b09322541848d494bda5608ca6.jpg](../icml_results/2970_Causal%20Representation%20Learning%20with%20Optimal%20Compression%20under%20Complex%20Treatments/images/fa927622fa7d97daa7a0b13ad2151dd7b98d87b09322541848d494bda5608ca6.jpg)

### Tables

![1e50ea7e300ed6d2eadd484e2efe8318fcbed7342364b907fa38f175ea7f32f2.jpg](../icml_results/2970_Causal%20Representation%20Learning%20with%20Optimal%20Compression%20under%20Complex%20Treatments/tables/1e50ea7e300ed6d2eadd484e2efe8318fcbed7342364b907fa38f175ea7f32f2.jpg)

![57c11b9b62ccdf667fbbbd198c3f8c8af3ec6f39ec59340768bb9560cb6910c3.jpg](../icml_results/2970_Causal%20Representation%20Learning%20with%20Optimal%20Compression%20under%20Complex%20Treatments/tables/57c11b9b62ccdf667fbbbd198c3f8c8af3ec6f39ec59340768bb9560cb6910c3.jpg)

<a id="paper-2933"></a>

## Seg-ReSearch: Segmentation with Interleaved Reasoning and External Search


### Images

![06c0a47586610dfc15e1f74ec36f51ad6ff44e186221a674b0d377232d056015.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/06c0a47586610dfc15e1f74ec36f51ad6ff44e186221a674b0d377232d056015.jpg)

![1c03c56a54722a4d3cf8a1745a507f734dc92ad64e1155b8573c610b5827c8fe.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/1c03c56a54722a4d3cf8a1745a507f734dc92ad64e1155b8573c610b5827c8fe.jpg)

![2140eb688daf0f6c90e2ec682bc1bbceeb20421dd707a2feb974c82aa61c193c.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/2140eb688daf0f6c90e2ec682bc1bbceeb20421dd707a2feb974c82aa61c193c.jpg)

![25503322278477e391ed44ee9e6268f6a89c283d720a57eecd7189680eabfe20.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/25503322278477e391ed44ee9e6268f6a89c283d720a57eecd7189680eabfe20.jpg)

![2cd4363d602329c211682b3f56084459f930ff144affb07bc5daf5bb33da0100.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/2cd4363d602329c211682b3f56084459f930ff144affb07bc5daf5bb33da0100.jpg)

![3a64feb678b0a4d571ec2346dbd1074f96d9b5b9ce115f9623eea0e715bb7d47.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/3a64feb678b0a4d571ec2346dbd1074f96d9b5b9ce115f9623eea0e715bb7d47.jpg)

![3bf74774f8f670d1cdbb635d7ab6c7909aa5c99a5ea9b37a429e551b2161c4db.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/3bf74774f8f670d1cdbb635d7ab6c7909aa5c99a5ea9b37a429e551b2161c4db.jpg)

![4503b5137d0377917f445e645ad775885122916ef229122a19234b07c68a908a.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/4503b5137d0377917f445e645ad775885122916ef229122a19234b07c68a908a.jpg)

![4ca8fb94cf41abd14d6f257b3ae0f50772178d45fa58f9ad906cdf532d878d29.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/4ca8fb94cf41abd14d6f257b3ae0f50772178d45fa58f9ad906cdf532d878d29.jpg)

![4e5ed0a3242eee8537a692afdb668b39ca2a250cafe1c252e5b2e3d735578fe0.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/4e5ed0a3242eee8537a692afdb668b39ca2a250cafe1c252e5b2e3d735578fe0.jpg)

![52d370486779a39e2d89deb27b2180bb58e033ee33d8de12d89f8acfced47b1c.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/52d370486779a39e2d89deb27b2180bb58e033ee33d8de12d89f8acfced47b1c.jpg)

![5992b7ea42fbdb6f2c265f00ea3e6c65455ecacec145e1497deabaca2af5c27e.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/5992b7ea42fbdb6f2c265f00ea3e6c65455ecacec145e1497deabaca2af5c27e.jpg)

![62cf40105a14918ba5877d10cac53218d4e671eba16bd1407561ed97b57730bd.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/62cf40105a14918ba5877d10cac53218d4e671eba16bd1407561ed97b57730bd.jpg)

![7251e84d16caf5ef96039c3f2753b98a4087b472cedb621cdaa6f76badc6ec59.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/7251e84d16caf5ef96039c3f2753b98a4087b472cedb621cdaa6f76badc6ec59.jpg)

![7d7f7e33099a015a6fa431447f63e78206d979e2de1906cbfa37a380dc900063.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/7d7f7e33099a015a6fa431447f63e78206d979e2de1906cbfa37a380dc900063.jpg)

![858ab158f9aa91f1e15a5fb9dd29a55551fdbf3aff4b972df3e83e0f4c8f0898.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/858ab158f9aa91f1e15a5fb9dd29a55551fdbf3aff4b972df3e83e0f4c8f0898.jpg)

![8f2c2d6e06db2df6e0b64fbf75bc2bde8639669e083bc40416e18be9c0b9f71c.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/8f2c2d6e06db2df6e0b64fbf75bc2bde8639669e083bc40416e18be9c0b9f71c.jpg)

![98f5c1ddcdf8202259ad83523895560fac8520e264722f4dcdf72610c5605f8d.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/98f5c1ddcdf8202259ad83523895560fac8520e264722f4dcdf72610c5605f8d.jpg)

![9c4c97450c74ee73875f120098533f91b178dafba47512e47e8b3890a3cb8030.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/9c4c97450c74ee73875f120098533f91b178dafba47512e47e8b3890a3cb8030.jpg)

![aadc329063eaa090ecbc3cea6ea390aef58505421b2214c628d7ac07d2cd1565.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/aadc329063eaa090ecbc3cea6ea390aef58505421b2214c628d7ac07d2cd1565.jpg)

![aae6f556784a81ecf98679f410fb8a13521c2fd652f82121f9deff3d29be381f.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/aae6f556784a81ecf98679f410fb8a13521c2fd652f82121f9deff3d29be381f.jpg)

![b9cff0058b208ca81d9eb054d01676abc93a7ffeaa14274aa978b1431e2fbd0f.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/b9cff0058b208ca81d9eb054d01676abc93a7ffeaa14274aa978b1431e2fbd0f.jpg)

![d120cb61d2f5bd178016dbf2b41c06d0a29d13820c4eaa2b09833356aada38a2.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/d120cb61d2f5bd178016dbf2b41c06d0a29d13820c4eaa2b09833356aada38a2.jpg)

![d9e4ae26af67d309bf1f199a8680bff27b6760d62c23811cf179a71439f0afa5.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/d9e4ae26af67d309bf1f199a8680bff27b6760d62c23811cf179a71439f0afa5.jpg)

![f3aec11f3fb771416dc3424571b1ffa4e6000d7e5872033f6ac74b18808834aa.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/f3aec11f3fb771416dc3424571b1ffa4e6000d7e5872033f6ac74b18808834aa.jpg)

![fbf43adf3a0611b44dc1d436cbaef44335381cc3875484611abe07c9d0ca02d5.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/images/fbf43adf3a0611b44dc1d436cbaef44335381cc3875484611abe07c9d0ca02d5.jpg)

### Tables

![1c2453153cd0be3ef0f34d1156673648a082dd2b5effdc53387887af72c2058e.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/tables/1c2453153cd0be3ef0f34d1156673648a082dd2b5effdc53387887af72c2058e.jpg)

![1fca386046f5d671b0812394cdc7ef93c92b3a30fb5b9e8f4d9b67b712f24b7e.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/tables/1fca386046f5d671b0812394cdc7ef93c92b3a30fb5b9e8f4d9b67b712f24b7e.jpg)

![2393594b843520bef5e1de4d162938643b1a5d18e170ab67999f347b8adda5d9.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/tables/2393594b843520bef5e1de4d162938643b1a5d18e170ab67999f347b8adda5d9.jpg)

![30f495620015be26fc31ac6e3e1004e7be3df78d3d4a25062326d73aa036903d.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/tables/30f495620015be26fc31ac6e3e1004e7be3df78d3d4a25062326d73aa036903d.jpg)

![678050052b8812506c3d80eb70dca6f5576b2e8992e3ce021bd242a42cbee9bb.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/tables/678050052b8812506c3d80eb70dca6f5576b2e8992e3ce021bd242a42cbee9bb.jpg)

![6e4f82acddb5ba7b6958600d0bff4ae811325a56826b839d8d72c2e1441df530.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/tables/6e4f82acddb5ba7b6958600d0bff4ae811325a56826b839d8d72c2e1441df530.jpg)

![9e58e676392016e7cbc08616b4b47e26bebecb45e5115b228501e54c4397fd16.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/tables/9e58e676392016e7cbc08616b4b47e26bebecb45e5115b228501e54c4397fd16.jpg)

![bedb34ea3f39f52fc06894367c2b380889ba89648ec27b00c0ba35d8b6bcac76.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/tables/bedb34ea3f39f52fc06894367c2b380889ba89648ec27b00c0ba35d8b6bcac76.jpg)

![d5b23dff6a613ae55e508b2874bb5e4555daf0c00ad6362aef58618b8432b99b.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/tables/d5b23dff6a613ae55e508b2874bb5e4555daf0c00ad6362aef58618b8432b99b.jpg)

![d8699175e3caaca3c9edb40eeb260069ff1ada374be72c14e78a91205678db15.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/tables/d8699175e3caaca3c9edb40eeb260069ff1ada374be72c14e78a91205678db15.jpg)

![e36ec55ab2212c634b14137b0b1691013b3d3988054843becabe5d0e0867be93.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/tables/e36ec55ab2212c634b14137b0b1691013b3d3988054843becabe5d0e0867be93.jpg)

![ebb2e9f36406918f28cffa0afedd9ea316ef4f88420979e384fd9a62daee86f6.jpg](../icml_results/2971_Seg-ReSearch_%20Segmentation%20with%20Interleaved%20Reasoning%20and%20External%20Search/tables/ebb2e9f36406918f28cffa0afedd9ea316ef4f88420979e384fd9a62daee86f6.jpg)

<a id="paper-2934"></a>

## CURVE: Learning Causality-Inspired Invariant Representations for Robust Scene Understanding via Uncertainty-Guided Regularization


### Images

![1a38a090341ece44b57643c4d3cfdadaf8fcb9a4d8b6c386892aa77ede1793e6.jpg](../icml_results/2972_CURVE_%20Learning%20Causality-Inspired%20Invariant%20Representations%20for%20Robust%20Scene%20Understanding%20via%20Unce/images/1a38a090341ece44b57643c4d3cfdadaf8fcb9a4d8b6c386892aa77ede1793e6.jpg)

![34596316f579e88b002c7fcf3db611ec8a4674a0d52893fb00c89994e0fb1030.jpg](../icml_results/2972_CURVE_%20Learning%20Causality-Inspired%20Invariant%20Representations%20for%20Robust%20Scene%20Understanding%20via%20Unce/images/34596316f579e88b002c7fcf3db611ec8a4674a0d52893fb00c89994e0fb1030.jpg)

![91b120762b9eb260e1908623ab80d9b14a38ed06c64106af33c09e0934acc041.jpg](../icml_results/2972_CURVE_%20Learning%20Causality-Inspired%20Invariant%20Representations%20for%20Robust%20Scene%20Understanding%20via%20Unce/images/91b120762b9eb260e1908623ab80d9b14a38ed06c64106af33c09e0934acc041.jpg)

![b0051af513294fefe2ad7b84390327173a9a605cc7982dc66a846508819b7ac6.jpg](../icml_results/2972_CURVE_%20Learning%20Causality-Inspired%20Invariant%20Representations%20for%20Robust%20Scene%20Understanding%20via%20Unce/images/b0051af513294fefe2ad7b84390327173a9a605cc7982dc66a846508819b7ac6.jpg)

![e26ec9034a934fd805881517e11db0f7e8e5472406751afbc99244dfa5b6dbd2.jpg](../icml_results/2972_CURVE_%20Learning%20Causality-Inspired%20Invariant%20Representations%20for%20Robust%20Scene%20Understanding%20via%20Unce/images/e26ec9034a934fd805881517e11db0f7e8e5472406751afbc99244dfa5b6dbd2.jpg)

![f28a1962b04d2e82e7b834704e0cf5a7537a01ca6002cd96382d2ca3dfe3d08f.jpg](../icml_results/2972_CURVE_%20Learning%20Causality-Inspired%20Invariant%20Representations%20for%20Robust%20Scene%20Understanding%20via%20Unce/images/f28a1962b04d2e82e7b834704e0cf5a7537a01ca6002cd96382d2ca3dfe3d08f.jpg)

### Tables

![06b05c8ce8692e86f36300084a14a1948637475cd9ebabd1ae5b4a9efe079ff2.jpg](../icml_results/2972_CURVE_%20Learning%20Causality-Inspired%20Invariant%20Representations%20for%20Robust%20Scene%20Understanding%20via%20Unce/tables/06b05c8ce8692e86f36300084a14a1948637475cd9ebabd1ae5b4a9efe079ff2.jpg)

![310c112545dec3567d8aafa784bc5fb2fb67876ccd28584ee52a8fe06e364663.jpg](../icml_results/2972_CURVE_%20Learning%20Causality-Inspired%20Invariant%20Representations%20for%20Robust%20Scene%20Understanding%20via%20Unce/tables/310c112545dec3567d8aafa784bc5fb2fb67876ccd28584ee52a8fe06e364663.jpg)

![32b9bd1ceff121b124022837c8a722bcf834a3c98e662a2f5eaa4ab17849db72.jpg](../icml_results/2972_CURVE_%20Learning%20Causality-Inspired%20Invariant%20Representations%20for%20Robust%20Scene%20Understanding%20via%20Unce/tables/32b9bd1ceff121b124022837c8a722bcf834a3c98e662a2f5eaa4ab17849db72.jpg)

![3934f8200086d73fc29c3cd830fb22a673d9043013736a609d12c357bae5d87a.jpg](../icml_results/2972_CURVE_%20Learning%20Causality-Inspired%20Invariant%20Representations%20for%20Robust%20Scene%20Understanding%20via%20Unce/tables/3934f8200086d73fc29c3cd830fb22a673d9043013736a609d12c357bae5d87a.jpg)

![526f74c56d84bb18ad830d43397812c3ef2b010b82806517d88aa54f6c6b15b1.jpg](../icml_results/2972_CURVE_%20Learning%20Causality-Inspired%20Invariant%20Representations%20for%20Robust%20Scene%20Understanding%20via%20Unce/tables/526f74c56d84bb18ad830d43397812c3ef2b010b82806517d88aa54f6c6b15b1.jpg)

![58ad25f5dad2259188b98b87e19056cbbafa7f933ee9797d233e5a9e82a53539.jpg](../icml_results/2972_CURVE_%20Learning%20Causality-Inspired%20Invariant%20Representations%20for%20Robust%20Scene%20Understanding%20via%20Unce/tables/58ad25f5dad2259188b98b87e19056cbbafa7f933ee9797d233e5a9e82a53539.jpg)

![9fe17402e30d2c59e2c388c2b27f0072b64e93856de0574690847d32bd89d8c7.jpg](../icml_results/2972_CURVE_%20Learning%20Causality-Inspired%20Invariant%20Representations%20for%20Robust%20Scene%20Understanding%20via%20Unce/tables/9fe17402e30d2c59e2c388c2b27f0072b64e93856de0574690847d32bd89d8c7.jpg)

![b1a4221c72134ba4081377e6d0b79d3b7f452433677b595cacf9dff393bd746e.jpg](../icml_results/2972_CURVE_%20Learning%20Causality-Inspired%20Invariant%20Representations%20for%20Robust%20Scene%20Understanding%20via%20Unce/tables/b1a4221c72134ba4081377e6d0b79d3b7f452433677b595cacf9dff393bd746e.jpg)

![e44eb885b36ae68ff696d010e5894be01f34644490f0a42ce237dfc42e4801bb.jpg](../icml_results/2972_CURVE_%20Learning%20Causality-Inspired%20Invariant%20Representations%20for%20Robust%20Scene%20Understanding%20via%20Unce/tables/e44eb885b36ae68ff696d010e5894be01f34644490f0a42ce237dfc42e4801bb.jpg)

![ebc9cfe1200e5e75afb2350371d75c7b94e5e623a53576a2cfafeceae8a99d8e.jpg](../icml_results/2972_CURVE_%20Learning%20Causality-Inspired%20Invariant%20Representations%20for%20Robust%20Scene%20Understanding%20via%20Unce/tables/ebc9cfe1200e5e75afb2350371d75c7b94e5e623a53576a2cfafeceae8a99d8e.jpg)

![f42634c3cebb2986fc366da48d32b0806c0be322d88fde6c850a2e19d1558d75.jpg](../icml_results/2972_CURVE_%20Learning%20Causality-Inspired%20Invariant%20Representations%20for%20Robust%20Scene%20Understanding%20via%20Unce/tables/f42634c3cebb2986fc366da48d32b0806c0be322d88fde6c850a2e19d1558d75.jpg)

<a id="paper-2935"></a>

## SMM Transformer: Leveraging Spiking Neural Networks for Multimodal Tasks


### Images

![17b4dfa751cf67e6beab00b81ad42909571e45fb5ac27702516e145551165c45.jpg](../icml_results/2973_SMM%20Transformer_%20Leveraging%20Spiking%20Neural%20Networks%20for%20Multimodal%20Tasks/images/17b4dfa751cf67e6beab00b81ad42909571e45fb5ac27702516e145551165c45.jpg)

![352f453cc9a03d7b76cedca2585179c13176c82ea817223d21361e223bf7cee6.jpg](../icml_results/2973_SMM%20Transformer_%20Leveraging%20Spiking%20Neural%20Networks%20for%20Multimodal%20Tasks/images/352f453cc9a03d7b76cedca2585179c13176c82ea817223d21361e223bf7cee6.jpg)

![67607c06d69e1cc73ef37e7bd0e96cc9bcadd2694af2ed73dd797ac946ad56c9.jpg](../icml_results/2973_SMM%20Transformer_%20Leveraging%20Spiking%20Neural%20Networks%20for%20Multimodal%20Tasks/images/67607c06d69e1cc73ef37e7bd0e96cc9bcadd2694af2ed73dd797ac946ad56c9.jpg)

![ae8f60ad5e5e1dd3f69d67db4f2d935033fad8b7af5af91c5a341b20c0b4d056.jpg](../icml_results/2973_SMM%20Transformer_%20Leveraging%20Spiking%20Neural%20Networks%20for%20Multimodal%20Tasks/images/ae8f60ad5e5e1dd3f69d67db4f2d935033fad8b7af5af91c5a341b20c0b4d056.jpg)

### Tables

![044b7f557e059d91866f55187afe74e3eb72692dcd48c070934d71c49dc90705.jpg](../icml_results/2973_SMM%20Transformer_%20Leveraging%20Spiking%20Neural%20Networks%20for%20Multimodal%20Tasks/tables/044b7f557e059d91866f55187afe74e3eb72692dcd48c070934d71c49dc90705.jpg)

![0726e2b87349e43dfe9c9ff337b095a1c575abf5b9d4731ae228c90b71aadb2f.jpg](../icml_results/2973_SMM%20Transformer_%20Leveraging%20Spiking%20Neural%20Networks%20for%20Multimodal%20Tasks/tables/0726e2b87349e43dfe9c9ff337b095a1c575abf5b9d4731ae228c90b71aadb2f.jpg)

![4546c4b2c4413d8b6974f9dc6d5ec0c7bc3a1fac783e94fce9ef0442421df9b3.jpg](../icml_results/2973_SMM%20Transformer_%20Leveraging%20Spiking%20Neural%20Networks%20for%20Multimodal%20Tasks/tables/4546c4b2c4413d8b6974f9dc6d5ec0c7bc3a1fac783e94fce9ef0442421df9b3.jpg)

![498e22aebb1b52e2b5ab757d5a04cc4c88ce390c8868c14b21527afb940bf626.jpg](../icml_results/2973_SMM%20Transformer_%20Leveraging%20Spiking%20Neural%20Networks%20for%20Multimodal%20Tasks/tables/498e22aebb1b52e2b5ab757d5a04cc4c88ce390c8868c14b21527afb940bf626.jpg)

![5ef73f56377b6c71cfb69e30eabca1e31e7527d4332eed7e2793f4c05ea9c9b1.jpg](../icml_results/2973_SMM%20Transformer_%20Leveraging%20Spiking%20Neural%20Networks%20for%20Multimodal%20Tasks/tables/5ef73f56377b6c71cfb69e30eabca1e31e7527d4332eed7e2793f4c05ea9c9b1.jpg)

![6b766118d2d8652e55cdcc621caa6412cd4b17c51538fa3e9c0ad8a255721ed3.jpg](../icml_results/2973_SMM%20Transformer_%20Leveraging%20Spiking%20Neural%20Networks%20for%20Multimodal%20Tasks/tables/6b766118d2d8652e55cdcc621caa6412cd4b17c51538fa3e9c0ad8a255721ed3.jpg)

![95b94aeb9de5198c0708841e56e837bc2ed7458057af0ed997151a4943cac6c9.jpg](../icml_results/2973_SMM%20Transformer_%20Leveraging%20Spiking%20Neural%20Networks%20for%20Multimodal%20Tasks/tables/95b94aeb9de5198c0708841e56e837bc2ed7458057af0ed997151a4943cac6c9.jpg)

![c709971593de55d37f790bdf0f31ad0f678bad445c78bc6074a7425f31db0962.jpg](../icml_results/2973_SMM%20Transformer_%20Leveraging%20Spiking%20Neural%20Networks%20for%20Multimodal%20Tasks/tables/c709971593de55d37f790bdf0f31ad0f678bad445c78bc6074a7425f31db0962.jpg)

![d5043d15db5ff3c394cf755b7fb9688b78c5e28fa641511bd137a21f6b8b8919.jpg](../icml_results/2973_SMM%20Transformer_%20Leveraging%20Spiking%20Neural%20Networks%20for%20Multimodal%20Tasks/tables/d5043d15db5ff3c394cf755b7fb9688b78c5e28fa641511bd137a21f6b8b8919.jpg)

![db650ff93cb5cd1d350088afa534214b5354daca4212de0ea3be11d1ef857504.jpg](../icml_results/2973_SMM%20Transformer_%20Leveraging%20Spiking%20Neural%20Networks%20for%20Multimodal%20Tasks/tables/db650ff93cb5cd1d350088afa534214b5354daca4212de0ea3be11d1ef857504.jpg)

![dc547593c1692de781ec12d381a20d293167834600460785a05ca8a20be46d7b.jpg](../icml_results/2973_SMM%20Transformer_%20Leveraging%20Spiking%20Neural%20Networks%20for%20Multimodal%20Tasks/tables/dc547593c1692de781ec12d381a20d293167834600460785a05ca8a20be46d7b.jpg)

![ee09d86a85ae6ef727214ed41296a837f660cc936f2fd0d0f6b5a95b8d60e3bd.jpg](../icml_results/2973_SMM%20Transformer_%20Leveraging%20Spiking%20Neural%20Networks%20for%20Multimodal%20Tasks/tables/ee09d86a85ae6ef727214ed41296a837f660cc936f2fd0d0f6b5a95b8d60e3bd.jpg)

<a id="paper-2936"></a>

## Spike-HTR: Spiking Neural Transformer for Handwritten Text Recognition


### Images

![13ade7c5634c717873b2fc16a183064cf1971781c705241797f4f875a597750e.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/images/13ade7c5634c717873b2fc16a183064cf1971781c705241797f4f875a597750e.jpg)

![3bfa560befa507db740456c5d2860d4f3abd5adccb06e17f63f0cd4154af078b.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/images/3bfa560befa507db740456c5d2860d4f3abd5adccb06e17f63f0cd4154af078b.jpg)

![43091e8348d0629c37fd0949ecdbf9e8a15decec75c69751e80ccdebc7e91fa6.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/images/43091e8348d0629c37fd0949ecdbf9e8a15decec75c69751e80ccdebc7e91fa6.jpg)

![504b6e4710afb234c5c5dc0c36a736cfa029172aa665b3ed7d5ff76c8de69cf9.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/images/504b6e4710afb234c5c5dc0c36a736cfa029172aa665b3ed7d5ff76c8de69cf9.jpg)

![935e2c5e7ddea00c9601bb04843eb1e54077e3a17cd61593a7fb17e5ad7c3946.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/images/935e2c5e7ddea00c9601bb04843eb1e54077e3a17cd61593a7fb17e5ad7c3946.jpg)

![cac5920dcf77e915f3e6c9d307cda1badd458cc957308ce8a89fc9c73a79f604.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/images/cac5920dcf77e915f3e6c9d307cda1badd458cc957308ce8a89fc9c73a79f604.jpg)

![eec89155b87c813d7a7a4178917ccb0e2f7499afdf697d41654572af62c23d96.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/images/eec89155b87c813d7a7a4178917ccb0e2f7499afdf697d41654572af62c23d96.jpg)

### Tables

![01cd0dea9f25e00ac7f69991f370033648c6822b932f94d14f71846c8259ccfb.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/01cd0dea9f25e00ac7f69991f370033648c6822b932f94d14f71846c8259ccfb.jpg)

![223d050cf1dc215d8201bf208728390967940deb3d31bdb4905c40a0ad750001.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/223d050cf1dc215d8201bf208728390967940deb3d31bdb4905c40a0ad750001.jpg)

![2289403c6d40ab3bedc5246a5f98f7a988ff77bd5d6bd71cbadfab9acacc49e6.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/2289403c6d40ab3bedc5246a5f98f7a988ff77bd5d6bd71cbadfab9acacc49e6.jpg)

![3ec715795432b70d00bc355561e0662b8684aefb8cde7f20a749642787d0ebc0.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/3ec715795432b70d00bc355561e0662b8684aefb8cde7f20a749642787d0ebc0.jpg)

![46ce9848ffa9c84286e701576add18b8de34672a7b5f235879e36739f1600a55.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/46ce9848ffa9c84286e701576add18b8de34672a7b5f235879e36739f1600a55.jpg)

![7df77774c0f344164adafa91ccdfc230d8402d681f9d0a42142e2c4516cf3abb.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/7df77774c0f344164adafa91ccdfc230d8402d681f9d0a42142e2c4516cf3abb.jpg)

![86c2c073ed76175b09f4ba53fb856e7ca0f109ad9b02a36e246b911e237f9e66.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/86c2c073ed76175b09f4ba53fb856e7ca0f109ad9b02a36e246b911e237f9e66.jpg)

![87c9944748bba5918a68a028711e7e47706f3f56151110b164c2bd63a968d38d.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/87c9944748bba5918a68a028711e7e47706f3f56151110b164c2bd63a968d38d.jpg)

![9a6501b4f7c919fc941e5e947907bb896846a09f0570fa28e31c02aadd52e013.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/9a6501b4f7c919fc941e5e947907bb896846a09f0570fa28e31c02aadd52e013.jpg)

![9c2e83de0fe651a4cd4d90fa8103c4ce1d83c4dca8012396f12df47e18e2e17a.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/9c2e83de0fe651a4cd4d90fa8103c4ce1d83c4dca8012396f12df47e18e2e17a.jpg)

![9e2b8716d64f913d2dc9235dee6268c2f79bf25d2d3389629c54220f2af4f67e.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/9e2b8716d64f913d2dc9235dee6268c2f79bf25d2d3389629c54220f2af4f67e.jpg)

![ba4868cb6e1b6a50d93a41e3e4f23500bed211fffa56549047a56d5f42755ef9.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/ba4868cb6e1b6a50d93a41e3e4f23500bed211fffa56549047a56d5f42755ef9.jpg)

![bd2acbd1e01ce218541235b134c655e5f8b0f6cc7e984597415f1bb28d36a04a.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/bd2acbd1e01ce218541235b134c655e5f8b0f6cc7e984597415f1bb28d36a04a.jpg)

![bd2d61c130b1ea441a7147e8ec2091004bc25ebdd34c2d940cc5dcac89964068.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/bd2d61c130b1ea441a7147e8ec2091004bc25ebdd34c2d940cc5dcac89964068.jpg)

![cad021c8884e3c3d3033ec152d8f9e6708905e0c0e186f416d92898a8b8f4a34.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/cad021c8884e3c3d3033ec152d8f9e6708905e0c0e186f416d92898a8b8f4a34.jpg)

![eb44b70aca76305210bafc7bb07096ee53edfbcc96b59b7cae16c86eb83e63b9.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/eb44b70aca76305210bafc7bb07096ee53edfbcc96b59b7cae16c86eb83e63b9.jpg)

![fa506fd7a43a364cba15b8cdae373f9d9b36f8f508bc9c69b7ce0de48b5d5a7a.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/fa506fd7a43a364cba15b8cdae373f9d9b36f8f508bc9c69b7ce0de48b5d5a7a.jpg)

![fd6c6ec6ada92cd1b27cace9a7df4f74d7be1cfc8cdc2af14cd285c4015041c5.jpg](../icml_results/2974_Spike-HTR_%20Spiking%20Neural%20Transformer%20for%20Handwritten%20Text%20Recognition/tables/fd6c6ec6ada92cd1b27cace9a7df4f74d7be1cfc8cdc2af14cd285c4015041c5.jpg)

<a id="paper-2937"></a>

## AnyMod-LLVE: Low-Light Video Enhancement with Modality-Agnostic Inference

### Images

![18a89e847eee25f47d9edf05dc99aedb3552614c2157d0cd512ea94bf851cf4d.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/18a89e847eee25f47d9edf05dc99aedb3552614c2157d0cd512ea94bf851cf4d.jpg)

![4554f035eb72fab14e37d3fbb96ea00a2059771cd52e1f812d5440d70098453d.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/4554f035eb72fab14e37d3fbb96ea00a2059771cd52e1f812d5440d70098453d.jpg)

![4a9caec5637a286f6f5889b5fb70799632ec590f5dbd19e7423001c279247fcb.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/4a9caec5637a286f6f5889b5fb70799632ec590f5dbd19e7423001c279247fcb.jpg)

![7ef46a1d85687f120380fc1b238adafb6c73dc664c89339b03cc2b02de818d73.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/7ef46a1d85687f120380fc1b238adafb6c73dc664c89339b03cc2b02de818d73.jpg)

![a4f1b6d00ecf025805a4a70764aad78a7a56c6663fddd92ad78119dec7586da5.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/a4f1b6d00ecf025805a4a70764aad78a7a56c6663fddd92ad78119dec7586da5.jpg)

![b6a5ac96ecff647baf8d3c24ba776186d93777582c450140b60799b436fd22b2.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/b6a5ac96ecff647baf8d3c24ba776186d93777582c450140b60799b436fd22b2.jpg)

![d1fb85295bfe7ceba4087d3363a48faa45e960c31b172cf5231274d4d57fc836.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/d1fb85295bfe7ceba4087d3363a48faa45e960c31b172cf5231274d4d57fc836.jpg)

![dd10d7de0aaf0d17b8865284a6ffc234eb7240d482c13f956b3f30abcb040ba2.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/dd10d7de0aaf0d17b8865284a6ffc234eb7240d482c13f956b3f30abcb040ba2.jpg)

![dd5ac3f96848a194716325efd15ca839b90487647f095d51ece70cc74e75bfc7.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/dd5ac3f96848a194716325efd15ca839b90487647f095d51ece70cc74e75bfc7.jpg)

![e89f874d57517b602ab34244ec78370b386304a89ce44f97cb2ccc4214912e65.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/e89f874d57517b602ab34244ec78370b386304a89ce44f97cb2ccc4214912e65.jpg)

### Tables

![040b0055468a5a08408050d4774ef4245a6dc398bd7ae49b81905003745244cc.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/tables/040b0055468a5a08408050d4774ef4245a6dc398bd7ae49b81905003745244cc.jpg)

![1840e5a40698e215532b7619f522bd8246c0e94a8c97aef6a6df439b7021efb3.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/tables/1840e5a40698e215532b7619f522bd8246c0e94a8c97aef6a6df439b7021efb3.jpg)

![29c69633a41dd722a17d5fbb19de837117f7465c36a6ae49b60960242127a5fb.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/tables/29c69633a41dd722a17d5fbb19de837117f7465c36a6ae49b60960242127a5fb.jpg)

![33c713ed517bdbf64f1fe932ff04769198d2d8e48f786facd48faf8fe174706e.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/tables/33c713ed517bdbf64f1fe932ff04769198d2d8e48f786facd48faf8fe174706e.jpg)

![3b088a239cad15cf990cb1e6f9751092cd215186f9303c02abf78814b2306a64.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/tables/3b088a239cad15cf990cb1e6f9751092cd215186f9303c02abf78814b2306a64.jpg)

![4ee34f024d77786d6c700f89baeef7109837d233da01d816476bf78d551b482a.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/tables/4ee34f024d77786d6c700f89baeef7109837d233da01d816476bf78d551b482a.jpg)

![5e37eab0086a0708cb64f1ff09f5727807f4babf82849667c0a17253f56be5af.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/tables/5e37eab0086a0708cb64f1ff09f5727807f4babf82849667c0a17253f56be5af.jpg)

![63d670bae28d115dfef68d20d53f13bb8b3d47e935e6e4af3287afc614175599.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/tables/63d670bae28d115dfef68d20d53f13bb8b3d47e935e6e4af3287afc614175599.jpg)
