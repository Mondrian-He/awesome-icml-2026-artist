# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 33 of 197

## 目录 (Table of Contents)

1. [Steering Beyond the Support: Adversarial Training on Unsupervised Jailbroken Activation Simulation](#paper-1057)
2. [OrchJail: Jailbreaking Tool-Calling Text-to-Image Agents by Orchestration-Guided Fuzzing](#paper-1058)
3. [Continuity-Regularized Flow Matching for Offline Reinforcement Learning](#paper-1059)
4. [Escaping the Subspace Trap: The Role of Optimizer Geometry in Model Width Expansion](#paper-1060)
5. [Certain Head, Uncertain Tail: Expert-Sample for Test-Time Scaling in Fine-Grained MoE](#paper-1061)
6. [Learning to Self-Verify Makes Language Models Better Reasoners](#paper-1062)
7. [From Extraction to Deduction: Resolving Functional Misalignment in RAG via a Collaborative Critic-Reasoner Framework](#paper-1063)
8. [Context Forcing: Consistent Autoregressive Video Generation with Long Context](#paper-1064)
9. [Stability and Generalization of Nonconvex Optimization with Heavy-Tailed Noise](#paper-1065)
10. [SCHUR-A*: Layer-wise Optimal Expert Pruning for MoEs via Schur-Complement Guided A* Search](#paper-1066)
11. [QiMeng-PerceptOS: Semantic-Aware Kernel Optimization for OS-Intensive Workloads via Hardware-Software Alignment](#paper-1067)
12. [SemanticNVS: Improving Semantic Scene Understanding in Generative Novel View Synthesis](#paper-1068)
13. [Mitigating Reward Hacking in LLM-based Recommendation: A Preference Optimization Approach](#paper-1069)
14. [INT vs. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats](#paper-1070)
15. [Riemannian Networks over Full-Rank Correlation Matrices](#paper-1071)
16. [ScalingAR: Scaling Confidence for Autoregressive Image Generation](#paper-1072)
17. [HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation](#paper-1073)
18. [Towards Disentangled Preference Optimization Dynamics: Suppress the Loser, Preserve the Winner](#paper-1074)
19. [AMDP: Asynchronous Multi-Directional Pipeline Parallelism for Large-Scale Models Training](#paper-1075)
20. [General Quantification of Covariate and Concept Shifts](#paper-1076)
21. [TrustworthyQENN: A Quantum Evidential Neural Network Based on Complex-Valued Contrastive Learning for Uncertainty Pattern Classification](#paper-1077)
22. [OnePO: Direct One-stage Policy Optimization for SFT-free Domain Adaptation](#paper-1078)
23. [BabyVision: Visual Reasoning Beyond Language](#paper-1079)
24. [Self-Supervised Dynamical System Representations for Physiological Time-Series](#paper-1080)
25. [Policy-Driven World Model Adaptation for Robust Offline Model-based Reinforcement Learning](#paper-1081)
26. [NanoSpec: Accelerating Speculative Decoding using Minimalist In-Context Vocabularies](#paper-1082)
27. [VIPO: Value Function Inconsistency Penalized Offline Reinforcement Learning](#paper-1083)
28. [MetaStreet: Semi-Supervised Multimodal Learning for Street-Level Socioeconomic Prediction](#paper-1084)
29. [GuidedBridge: Training-freely Improving Bridge Models with Prior Guidance](#paper-1085)
30. [PCRNet: Phase-aware Complex Refinement Network for EEG-based Auditory Attention Decoding](#paper-1086)
31. [CONCUR: High-Throughput Agentic Batch Inference of LLM via Congestion-Based Concurrency Control](#paper-1087)
32. [Online Linear Programming for Multi-Objective Routing in LLM Serving](#paper-1088)
33. [Boundary Embedding Shaping with Adaptive Contrastive Learning for Graph Structural Disentanglement](#paper-1089)

---


<a id="paper-1057"></a>

## Steering Beyond the Support: Adversarial Training on Unsupervised Jailbroken Activation Simulation

### Images

![1360150f8b49c6f0310a982010e791c5566c61de9d6bdac7afab81ddd0b9bb77.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/1360150f8b49c6f0310a982010e791c5566c61de9d6bdac7afab81ddd0b9bb77.jpg)

![2ce2b27d679868de87c620597488044c45c77d618070f4df39ec85b977ad0558.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/2ce2b27d679868de87c620597488044c45c77d618070f4df39ec85b977ad0558.jpg)

![3cb9cd8d9f3b8c0810c32dafa630fa0b67bd0b9ed80e39d36502ee33f651fe41.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/3cb9cd8d9f3b8c0810c32dafa630fa0b67bd0b9ed80e39d36502ee33f651fe41.jpg)

![450694d581cac33fafc72c0354096335ea0a9c6e7337b1472ec47f8b35ff535a.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/450694d581cac33fafc72c0354096335ea0a9c6e7337b1472ec47f8b35ff535a.jpg)

![5ed0ed22188bc20b3718c0dec3830c0a4ddc514ab0176350f0223f232248cc3c.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/5ed0ed22188bc20b3718c0dec3830c0a4ddc514ab0176350f0223f232248cc3c.jpg)

![720f5dc12913b6cc8862e9fee0efe2d1be54ab0a37e54a92a1d2b6f07bcc4678.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/720f5dc12913b6cc8862e9fee0efe2d1be54ab0a37e54a92a1d2b6f07bcc4678.jpg)

![8957991c06b81b0a495f788a39a985c016988c208d57b360c10588297a1add93.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/8957991c06b81b0a495f788a39a985c016988c208d57b360c10588297a1add93.jpg)

![8fcbab49e3135efb2a621a1624e2c3ede92a64cbf81256e0e835dd9cdddf110b.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/8fcbab49e3135efb2a621a1624e2c3ede92a64cbf81256e0e835dd9cdddf110b.jpg)

![954a71b15430f9e48e44398f715eaac653e464c9d4b6525f523a7056546cd90f.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/954a71b15430f9e48e44398f715eaac653e464c9d4b6525f523a7056546cd90f.jpg)

![9f892716b30d7f569e900c3060f24b1e51fb8bcd38437516b20eb1cf5cce4fa9.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/9f892716b30d7f569e900c3060f24b1e51fb8bcd38437516b20eb1cf5cce4fa9.jpg)

![a947fb48f6901ed54cc38659cd7b1cacff26560ba99b800984d2cdebf2db9910.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/a947fb48f6901ed54cc38659cd7b1cacff26560ba99b800984d2cdebf2db9910.jpg)

![a951147397fc3ecf37727f37d6e3ea78d755831b07794525548908b82c724afc.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/a951147397fc3ecf37727f37d6e3ea78d755831b07794525548908b82c724afc.jpg)

![ea7e937ada241ceb0b843ee27ef9b1ec2704cc15c5f7c2c671c604699e391cfc.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/ea7e937ada241ceb0b843ee27ef9b1ec2704cc15c5f7c2c671c604699e391cfc.jpg)

### Tables

![1c227160d6b39917c958ae62d8a5655ab5976caa366618a593ac6447f9d6fe1c.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/tables/1c227160d6b39917c958ae62d8a5655ab5976caa366618a593ac6447f9d6fe1c.jpg)

## Steering Beyond the Support: Adversarial Training on Unsupervised Jailbroken Activation Simulation


### Images

![017c1473fa5450423879005fcad656995f6614991253f5dbeb71be106c7e3b4f.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/images/017c1473fa5450423879005fcad656995f6614991253f5dbeb71be106c7e3b4f.jpg)

![29311cd44e4497a8d9c543effdd1720ad0d1784f9ac69da78b3556315de7f5fa.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/images/29311cd44e4497a8d9c543effdd1720ad0d1784f9ac69da78b3556315de7f5fa.jpg)

![43e906da42451d9a520b672e2480f48dea30878a2aa8fcbaf4a862ca343a901a.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/images/43e906da42451d9a520b672e2480f48dea30878a2aa8fcbaf4a862ca343a901a.jpg)

![4ea68dec00165959b6785b8c5687c144854d0132494b36bbbf487c3886918ee0.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/images/4ea68dec00165959b6785b8c5687c144854d0132494b36bbbf487c3886918ee0.jpg)

![60e9f98732d57deca15770a292baca77133e4be34ac9948214d24781cb0dc39d.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/images/60e9f98732d57deca15770a292baca77133e4be34ac9948214d24781cb0dc39d.jpg)

![8594104841fad17c99ed045d790b2b9ebfd92cfcea1a64648835161dda364215.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/images/8594104841fad17c99ed045d790b2b9ebfd92cfcea1a64648835161dda364215.jpg)

![9f42b1b9f838b92c5741d8cfbbb9051af915a70dc7f0e49d69387fd0440df29c.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/images/9f42b1b9f838b92c5741d8cfbbb9051af915a70dc7f0e49d69387fd0440df29c.jpg)

![b1ef27675a52e4c43db61f6694438556e6c3b7e2894c751f2d2cdf799dbf7c01.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/images/b1ef27675a52e4c43db61f6694438556e6c3b7e2894c751f2d2cdf799dbf7c01.jpg)

![f11f00de40aa09c57d3e8c2e248cd4a468d51f5c441480fb518649cb4a711971.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/images/f11f00de40aa09c57d3e8c2e248cd4a468d51f5c441480fb518649cb4a711971.jpg)

![f543d1e98ead96621ff4c0ea9b9066be133bdf5f65076bbdb5f406366e09f0ba.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/images/f543d1e98ead96621ff4c0ea9b9066be133bdf5f65076bbdb5f406366e09f0ba.jpg)

### Tables

![233cd6294ba4b9f48436ecfd8053b5ba8ab8c9a069d8ddc2a3a534eea9f20f23.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/tables/233cd6294ba4b9f48436ecfd8053b5ba8ab8c9a069d8ddc2a3a534eea9f20f23.jpg)

![2afadb5a76dcd2c6a4a80b487b511b8358c3d7ab60680c337da686e9409ded08.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/tables/2afadb5a76dcd2c6a4a80b487b511b8358c3d7ab60680c337da686e9409ded08.jpg)

![443dc523d911be6c9b0d748003ea6a3030897e1dd95778623ac0adba1f8ef48d.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/tables/443dc523d911be6c9b0d748003ea6a3030897e1dd95778623ac0adba1f8ef48d.jpg)

![46bbb4be876184486c7fef8055b50bb905ee0caf80034694e0c9ebf2a648171a.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/tables/46bbb4be876184486c7fef8055b50bb905ee0caf80034694e0c9ebf2a648171a.jpg)

![4957364653dd86a505fee56184450378f6f1a2c7c27b8853b5208db4e9d776c2.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/tables/4957364653dd86a505fee56184450378f6f1a2c7c27b8853b5208db4e9d776c2.jpg)

![591ea47c8479b25b8f1d3d8a4fb87b75127febcfd4cde60d3c2b819cafc62f0a.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/tables/591ea47c8479b25b8f1d3d8a4fb87b75127febcfd4cde60d3c2b819cafc62f0a.jpg)

![849f820a8198f4744ca8f8524d9d9b71b430e2b0c6586cefe6969e1707b24e16.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/tables/849f820a8198f4744ca8f8524d9d9b71b430e2b0c6586cefe6969e1707b24e16.jpg)

![99664ab555c6882d31dc6f64968dbc139bad88560b1f3658d4137156fcdc89e0.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/tables/99664ab555c6882d31dc6f64968dbc139bad88560b1f3658d4137156fcdc89e0.jpg)

![b84d839c7623880d6f67a9ac128597efcc3b43bc37fbd015e5d63a9a51a54290.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/tables/b84d839c7623880d6f67a9ac128597efcc3b43bc37fbd015e5d63a9a51a54290.jpg)

![be8638ff5fd434b3e5afcd63b1a462a7bba2a49a1fa2cbb5b478cb8af6e7698a.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/tables/be8638ff5fd434b3e5afcd63b1a462a7bba2a49a1fa2cbb5b478cb8af6e7698a.jpg)

![c697f763b8863dbbbe7d5b3a94246b18af5ddd7ec2c04ebcb32bed9e98e7d680.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/tables/c697f763b8863dbbbe7d5b3a94246b18af5ddd7ec2c04ebcb32bed9e98e7d680.jpg)

![ea1c59687d36d5a97344a8c8661b38b991395fde290d24cd78a258c7bfb223bc.jpg](../icml_results/1075_Steering%20Beyond%20the%20Support_%20Adversarial%20Training%20on%20Unsupervised%20Jailbroken%20Activation%20Simulation/tables/ea1c59687d36d5a97344a8c8661b38b991395fde290d24cd78a258c7bfb223bc.jpg)

<a id="paper-1058"></a>

## OrchJail: Jailbreaking Tool-Calling Text-to-Image Agents by Orchestration-Guided Fuzzing


### Images

![13e8c1197b18d437c2059ece376f56bf9d013724bb2aa594afd7368392206df7.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/images/13e8c1197b18d437c2059ece376f56bf9d013724bb2aa594afd7368392206df7.jpg)

![208ad448d71e7f7b0da71f7bd403e0db969f002f7ae3c1e21b09194b17571035.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/images/208ad448d71e7f7b0da71f7bd403e0db969f002f7ae3c1e21b09194b17571035.jpg)

![2595b627f0d7d438d5098029001c9c83a92b717382ce0d029fb37e7c732c3d00.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/images/2595b627f0d7d438d5098029001c9c83a92b717382ce0d029fb37e7c732c3d00.jpg)

![6ccd60d9849098d34d5a02f927adaac8fba1e6ae78faf4233cbb08957b8b97a0.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/images/6ccd60d9849098d34d5a02f927adaac8fba1e6ae78faf4233cbb08957b8b97a0.jpg)

![73d75a1a07ce243bbd6ab21ed015018a9686f48c1a857ae0adade28be8e8f9de.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/images/73d75a1a07ce243bbd6ab21ed015018a9686f48c1a857ae0adade28be8e8f9de.jpg)

![7eee43363742d94be80cc882b7d35c91b421954079bc23ec4189866e1b771272.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/images/7eee43363742d94be80cc882b7d35c91b421954079bc23ec4189866e1b771272.jpg)

![a03e33d0324e0df45e4d139f00cc2a50b7e94597bd9e58e28e5a29cc8ea73911.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/images/a03e33d0324e0df45e4d139f00cc2a50b7e94597bd9e58e28e5a29cc8ea73911.jpg)

![a051e77a265d5f6176e6a5e530f7d66634542d6c074bc7e98125c4114eb6cad4.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/images/a051e77a265d5f6176e6a5e530f7d66634542d6c074bc7e98125c4114eb6cad4.jpg)

![b29bbcceb729537463b849cd52ff5f981a457fae7cc56bee6417d270b4b6de30.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/images/b29bbcceb729537463b849cd52ff5f981a457fae7cc56bee6417d270b4b6de30.jpg)

![d016ac5bbc6cb49ee9e2293a41a78f739a44ce96b5dafb7d905b5596479f200c.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/images/d016ac5bbc6cb49ee9e2293a41a78f739a44ce96b5dafb7d905b5596479f200c.jpg)

![e82c8b7c113d767973f1523a64772589b5d05c947743ae79023b5c5ae90c58ae.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/images/e82c8b7c113d767973f1523a64772589b5d05c947743ae79023b5c5ae90c58ae.jpg)

![fc02167004153a68966f465757e91cfc61f98c704f16301a00921f27b4c745ec.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/images/fc02167004153a68966f465757e91cfc61f98c704f16301a00921f27b4c745ec.jpg)

### Tables

![051232d49b7342449d623a596f1c2964e7ca35ae6eb672ebc679132d0cc67234.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/tables/051232d49b7342449d623a596f1c2964e7ca35ae6eb672ebc679132d0cc67234.jpg)

![13c84e9695e9af7af4bb9ff34405e658e37f89bc9ca08148bae1c60beab765cc.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/tables/13c84e9695e9af7af4bb9ff34405e658e37f89bc9ca08148bae1c60beab765cc.jpg)

![279fce69f6d0f175fba546529b2e8cc60cb2e00652907ff146cfb105772f0974.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/tables/279fce69f6d0f175fba546529b2e8cc60cb2e00652907ff146cfb105772f0974.jpg)

![36599a7b251d5814af6b3ade4792616316919adc68a3b2b8c2a8b76cba0519e1.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/tables/36599a7b251d5814af6b3ade4792616316919adc68a3b2b8c2a8b76cba0519e1.jpg)

![3835ec504848862ac743b3ed697b5896f6e62b4bf624d3306f60b43b0de0e395.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/tables/3835ec504848862ac743b3ed697b5896f6e62b4bf624d3306f60b43b0de0e395.jpg)

![3e9f1851ddd27b8f6b0826aa4ca7c18972d1b78a461d368341eb0699922f76b8.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/tables/3e9f1851ddd27b8f6b0826aa4ca7c18972d1b78a461d368341eb0699922f76b8.jpg)

![43b028d89e7a019f27c163d3bab79d6c1340528232ed841051ea2f007ddddf7a.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/tables/43b028d89e7a019f27c163d3bab79d6c1340528232ed841051ea2f007ddddf7a.jpg)

![7984180b7138e610e100977da30c5c7c5088c36576276d3c58127b12e5b8fb15.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/tables/7984180b7138e610e100977da30c5c7c5088c36576276d3c58127b12e5b8fb15.jpg)

![a0e91d0f411e50616231c33947019f2229fe586c5db333abc8e1e922e13efb1d.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/tables/a0e91d0f411e50616231c33947019f2229fe586c5db333abc8e1e922e13efb1d.jpg)

![ca1e0b00bbf1ff917371d6ebc4becb51c08f31c1ea4c5a11e2fdc3f94369340f.jpg](../icml_results/1076_OrchJail_%20Jailbreaking%20Tool-Calling%20Text-to-Image%20Agents%20by%20Orchestration-Guided%20Fuzzing/tables/ca1e0b00bbf1ff917371d6ebc4becb51c08f31c1ea4c5a11e2fdc3f94369340f.jpg)

<a id="paper-1059"></a>

## Continuity-Regularized Flow Matching for Offline Reinforcement Learning


### Images

![007383a1b74b97da2169e0317af546fad0d3c1bd7c652f86f8dcfd961e6d5bf4.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/images/007383a1b74b97da2169e0317af546fad0d3c1bd7c652f86f8dcfd961e6d5bf4.jpg)

![0f0ff519e221683db35757b469cdc9bca9b2642dca0cad36d5c1ebfe12d794a2.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/images/0f0ff519e221683db35757b469cdc9bca9b2642dca0cad36d5c1ebfe12d794a2.jpg)

![207913d9d012536580b8da9f57e2411c3d1803e8a538927223594efd100b852d.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/images/207913d9d012536580b8da9f57e2411c3d1803e8a538927223594efd100b852d.jpg)

![2632b03bccbf7a34212ce0594be95ec7d1f16b141c0ab22d68359a38935270a2.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/images/2632b03bccbf7a34212ce0594be95ec7d1f16b141c0ab22d68359a38935270a2.jpg)

![2f09b032df47f25d81f1482d6530feddef3a2dbb861df643f9367c78ce7b3d6d.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/images/2f09b032df47f25d81f1482d6530feddef3a2dbb861df643f9367c78ce7b3d6d.jpg)

![37c01aab9b36110df8b858b9d4cab49b6743c6c2e111b743b4f5a989efad96d2.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/images/37c01aab9b36110df8b858b9d4cab49b6743c6c2e111b743b4f5a989efad96d2.jpg)

![40d820766c40c9752feb25800d55c9cafec9ae21718eb1441849b563d29d1cd2.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/images/40d820766c40c9752feb25800d55c9cafec9ae21718eb1441849b563d29d1cd2.jpg)

![7ad1c19e9924711f5f4eec5955264a9bf862cbf156d6c338a2888313185fec09.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/images/7ad1c19e9924711f5f4eec5955264a9bf862cbf156d6c338a2888313185fec09.jpg)

![96a29667b06fe5cac5f0d07a6c55126991a055389d8bcb1a1498fff77b1d8fb1.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/images/96a29667b06fe5cac5f0d07a6c55126991a055389d8bcb1a1498fff77b1d8fb1.jpg)

![d2117c31e46627f87d4a19a4006eecead01d925d628288eac8733fb9a615e3ef.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/images/d2117c31e46627f87d4a19a4006eecead01d925d628288eac8733fb9a615e3ef.jpg)

![d320f1bedcf6fdf1e04ffad8405989acab8e243d3dc37c201a00baa3a0795b8f.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/images/d320f1bedcf6fdf1e04ffad8405989acab8e243d3dc37c201a00baa3a0795b8f.jpg)

![d96abc03727bc56bdd2f6dc30dcd0f84135e285e17661239ed198e38324383e3.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/images/d96abc03727bc56bdd2f6dc30dcd0f84135e285e17661239ed198e38324383e3.jpg)

![e5d3732cae2b8d91edeba017e42e39abddc08214d107f43b8ed93c2d473d702c.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/images/e5d3732cae2b8d91edeba017e42e39abddc08214d107f43b8ed93c2d473d702c.jpg)

### Tables

![11e30f62217e72d63c30a0fe4c96903475e2aecbf1c662b98d210ee1873d7471.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/tables/11e30f62217e72d63c30a0fe4c96903475e2aecbf1c662b98d210ee1873d7471.jpg)

![1bce7fcc6f0f91051b0544ddde1d1d1da83f39048b7c181794c0ecaaef7fed20.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/tables/1bce7fcc6f0f91051b0544ddde1d1d1da83f39048b7c181794c0ecaaef7fed20.jpg)

![33ed3ca41f6d80b6bbc2ab065b8210dfc7f3e0a466aa78d98f8d338ca999e378.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/tables/33ed3ca41f6d80b6bbc2ab065b8210dfc7f3e0a466aa78d98f8d338ca999e378.jpg)

![459ac2609c365735d819e4077aa5f5590948e592f6bf0e5180039d4077fb9beb.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/tables/459ac2609c365735d819e4077aa5f5590948e592f6bf0e5180039d4077fb9beb.jpg)

![79fc54936791822e3958aeec3c0a6ecb993281720eab34cd62eb74c1ba1a31e1.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/tables/79fc54936791822e3958aeec3c0a6ecb993281720eab34cd62eb74c1ba1a31e1.jpg)

![92844fa5c42b2d17b553445e863068b17f008fd2f1b76b56f97bfaa1c770ab65.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/tables/92844fa5c42b2d17b553445e863068b17f008fd2f1b76b56f97bfaa1c770ab65.jpg)

![94b843db39e637a2c105070dc7f3f33330e119ed1f163bcb5707ce9e88795d3f.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/tables/94b843db39e637a2c105070dc7f3f33330e119ed1f163bcb5707ce9e88795d3f.jpg)

![9acc9a8705d6f8287d6f3202de539e0ddf3ad9c1200c563ff9b30f7e5890d94a.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/tables/9acc9a8705d6f8287d6f3202de539e0ddf3ad9c1200c563ff9b30f7e5890d94a.jpg)

![c505c4bf47107bbd6133f100f8a7810f66b6a3d72812eedd3753c3b2be9e132c.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/tables/c505c4bf47107bbd6133f100f8a7810f66b6a3d72812eedd3753c3b2be9e132c.jpg)

![e673fea1f6f9330fdbb111386c0707bfd77ad81a9a190be420cf494f628551eb.jpg](../icml_results/1077_Continuity-Regularized%20Flow%20Matching%20for%20Offline%20Reinforcement%20Learning/tables/e673fea1f6f9330fdbb111386c0707bfd77ad81a9a190be420cf494f628551eb.jpg)

<a id="paper-1060"></a>

## Escaping the Subspace Trap: The Role of Optimizer Geometry in Model Width Expansion


### Images

![10306843ae4ef21e3c69177a5a96c4e3b78d983264837ea2111a224ed0c990f7.jpg](../icml_results/1078_Escaping%20the%20Subspace%20Trap_%20The%20Role%20of%20Optimizer%20Geometry%20in%20Model%20Width%20Expansion/images/10306843ae4ef21e3c69177a5a96c4e3b78d983264837ea2111a224ed0c990f7.jpg)

![124567237b2b3ef31660dcaf03bd25826d37f63683e5e1370a7e8441206d80d3.jpg](../icml_results/1078_Escaping%20the%20Subspace%20Trap_%20The%20Role%20of%20Optimizer%20Geometry%20in%20Model%20Width%20Expansion/images/124567237b2b3ef31660dcaf03bd25826d37f63683e5e1370a7e8441206d80d3.jpg)

![39bb3dd72432b231f247f14b52a3bf8082202081db96699de471e81c17bee0b8.jpg](../icml_results/1078_Escaping%20the%20Subspace%20Trap_%20The%20Role%20of%20Optimizer%20Geometry%20in%20Model%20Width%20Expansion/images/39bb3dd72432b231f247f14b52a3bf8082202081db96699de471e81c17bee0b8.jpg)

![ba855ecdee806bd9507edd850458408787dbb92760e77b3e0002c8245688c438.jpg](../icml_results/1078_Escaping%20the%20Subspace%20Trap_%20The%20Role%20of%20Optimizer%20Geometry%20in%20Model%20Width%20Expansion/images/ba855ecdee806bd9507edd850458408787dbb92760e77b3e0002c8245688c438.jpg)

![c26645cc40c176c5f942c93560e10693af71fe8a98e3b19eb6f74203becb76bb.jpg](../icml_results/1078_Escaping%20the%20Subspace%20Trap_%20The%20Role%20of%20Optimizer%20Geometry%20in%20Model%20Width%20Expansion/images/c26645cc40c176c5f942c93560e10693af71fe8a98e3b19eb6f74203becb76bb.jpg)

![fed206183b4c8324bdd8c2054659643969b23152ebab2a2f445e950b0815e5f2.jpg](../icml_results/1078_Escaping%20the%20Subspace%20Trap_%20The%20Role%20of%20Optimizer%20Geometry%20in%20Model%20Width%20Expansion/images/fed206183b4c8324bdd8c2054659643969b23152ebab2a2f445e950b0815e5f2.jpg)

### Tables

![66f6f587a529998b60ecf765c4265dfa97dc04ce1118d6f2e687efe4f4b34aa7.jpg](../icml_results/1078_Escaping%20the%20Subspace%20Trap_%20The%20Role%20of%20Optimizer%20Geometry%20in%20Model%20Width%20Expansion/tables/66f6f587a529998b60ecf765c4265dfa97dc04ce1118d6f2e687efe4f4b34aa7.jpg)

![89746bc644bbb6939557644bcdd3805fa3bd3946deb93b727e901ae2950d0e0c.jpg](../icml_results/1078_Escaping%20the%20Subspace%20Trap_%20The%20Role%20of%20Optimizer%20Geometry%20in%20Model%20Width%20Expansion/tables/89746bc644bbb6939557644bcdd3805fa3bd3946deb93b727e901ae2950d0e0c.jpg)

![b5e0aa9168231bd08ba09a8582a2167ba1fba345d101f5cef07d7c4e826491fa.jpg](../icml_results/1078_Escaping%20the%20Subspace%20Trap_%20The%20Role%20of%20Optimizer%20Geometry%20in%20Model%20Width%20Expansion/tables/b5e0aa9168231bd08ba09a8582a2167ba1fba345d101f5cef07d7c4e826491fa.jpg)

![d3e936a8204c278eafe59213cf88fc4af5d4030c34a0b6c4e6501cdf2f238fe2.jpg](../icml_results/1078_Escaping%20the%20Subspace%20Trap_%20The%20Role%20of%20Optimizer%20Geometry%20in%20Model%20Width%20Expansion/tables/d3e936a8204c278eafe59213cf88fc4af5d4030c34a0b6c4e6501cdf2f238fe2.jpg)

<a id="paper-1061"></a>

## Certain Head, Uncertain Tail: Expert-Sample for Test-Time Scaling in Fine-Grained MoE


### Images

![0e74deca670abdcd266ab128a9a119c3ba444fbfe16721d712cdcf424e6db098.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/0e74deca670abdcd266ab128a9a119c3ba444fbfe16721d712cdcf424e6db098.jpg)

![19023c8e4f1cc91580d97cf3f91e11c90eb927af736aafac0c57258d8d842f60.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/19023c8e4f1cc91580d97cf3f91e11c90eb927af736aafac0c57258d8d842f60.jpg)

![1c790d6a26129f194dc1bc5c1ee38509711eb51c52fbdf9ced26faab72599d54.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/1c790d6a26129f194dc1bc5c1ee38509711eb51c52fbdf9ced26faab72599d54.jpg)

![1fe4d90a9dc67cd7fa67e5a097cc8f049a93ec3302ff8941623172e862ec3b80.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/1fe4d90a9dc67cd7fa67e5a097cc8f049a93ec3302ff8941623172e862ec3b80.jpg)

![2044e328598d0fe64fab275f4bad5a014f12e3b4166d8a0b5866fc77c2823515.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/2044e328598d0fe64fab275f4bad5a014f12e3b4166d8a0b5866fc77c2823515.jpg)

![2cc477df0f1f8b908a23f12fe7e441a3c687a589e51e45cc6167e950427ea5e4.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/2cc477df0f1f8b908a23f12fe7e441a3c687a589e51e45cc6167e950427ea5e4.jpg)

![343134ff037dd7f827e30588b8b521acd29bf68bea320e4fc2abaee92d460770.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/343134ff037dd7f827e30588b8b521acd29bf68bea320e4fc2abaee92d460770.jpg)

![478fd6c4de352e978f2ded74e909801c3e6d92ace7b5e5351c11353578b4f1a4.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/478fd6c4de352e978f2ded74e909801c3e6d92ace7b5e5351c11353578b4f1a4.jpg)

![62a5b17e341743e2d5c4bd40f4272bdafc3a0038cc016c11a2c421d9cfe09fc7.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/62a5b17e341743e2d5c4bd40f4272bdafc3a0038cc016c11a2c421d9cfe09fc7.jpg)

![8b4ae08908e4d2b63e61ef96620f3f5bd9a8d4fa4c7e8300982c4fa6bbd1a314.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/8b4ae08908e4d2b63e61ef96620f3f5bd9a8d4fa4c7e8300982c4fa6bbd1a314.jpg)

![8ebb10b4363809dee1b1fb72d57919aa148b68b0b502c196ffd99cd4a5c6eff3.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/8ebb10b4363809dee1b1fb72d57919aa148b68b0b502c196ffd99cd4a5c6eff3.jpg)

![abe269d9cfa981f92f5b6ee70fb698460ed6e888fbf634fa9ff3de0fa99ed09b.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/abe269d9cfa981f92f5b6ee70fb698460ed6e888fbf634fa9ff3de0fa99ed09b.jpg)

![ad83426090a918e20d9f2852968db228504b282fa2c22ee79d5bf2c421d7017a.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/ad83426090a918e20d9f2852968db228504b282fa2c22ee79d5bf2c421d7017a.jpg)

![b97fcc1579782898a8b5cc0e0d459c6bf9fa2c4f7caf81627f9263deab0a1899.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/b97fcc1579782898a8b5cc0e0d459c6bf9fa2c4f7caf81627f9263deab0a1899.jpg)

![c6a5c22bd3569ca64ef026df17e0a9254e13abfdfb3e3de072fcf2b4c24e799a.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/c6a5c22bd3569ca64ef026df17e0a9254e13abfdfb3e3de072fcf2b4c24e799a.jpg)

![ddb19a1d8bd0543d08380c074c44144246410db2d314c236130a7a1939ebcb9f.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/ddb19a1d8bd0543d08380c074c44144246410db2d314c236130a7a1939ebcb9f.jpg)

![ddbd18a1d94270f86254a5920ac2d16335f411254834beeab4a59b9b8054285a.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/ddbd18a1d94270f86254a5920ac2d16335f411254834beeab4a59b9b8054285a.jpg)

![fbf4f034e467fdff52fd114a5952d4655d825a0da0524fc5c659df5bd9de26ff.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/images/fbf4f034e467fdff52fd114a5952d4655d825a0da0524fc5c659df5bd9de26ff.jpg)

### Tables

![25cd3409f79f8db5e5ba1350b61e771a85a1d452d54b8564acd938a67e7de4d5.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/tables/25cd3409f79f8db5e5ba1350b61e771a85a1d452d54b8564acd938a67e7de4d5.jpg)

![35d9f612739d12515b0cd8a99ca6cf8f6f8db73bd892f3b2543eab758c3faac2.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/tables/35d9f612739d12515b0cd8a99ca6cf8f6f8db73bd892f3b2543eab758c3faac2.jpg)

![58415e35192c7933ecc1cd6ee15616a08d5c0903a5fc700da3f013d68eed1f16.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/tables/58415e35192c7933ecc1cd6ee15616a08d5c0903a5fc700da3f013d68eed1f16.jpg)

![590628e230d37993f2230d4e7874318ef986da742a365a1b12568575570f1549.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/tables/590628e230d37993f2230d4e7874318ef986da742a365a1b12568575570f1549.jpg)

![878e90f5370fb5d788af56af6c7da6895f8472ed9df09473cb799120f35788a2.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/tables/878e90f5370fb5d788af56af6c7da6895f8472ed9df09473cb799120f35788a2.jpg)

![96a3d9789bc2875efb206dcd9e36507b590222683df33fb87e0a9817a6558a67.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/tables/96a3d9789bc2875efb206dcd9e36507b590222683df33fb87e0a9817a6558a67.jpg)

![a0537d4580241f4ac96f9524a2a2e301a912acb43dc242feb10a8be5db761d23.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/tables/a0537d4580241f4ac96f9524a2a2e301a912acb43dc242feb10a8be5db761d23.jpg)

![aaf89727316ee70be5a2c253a25dfffd7f704c47837d02d5ba58bcc1028beb70.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/tables/aaf89727316ee70be5a2c253a25dfffd7f704c47837d02d5ba58bcc1028beb70.jpg)

![fb5c632f70746a32d74d27a27a4ba9c56fb5914e8b1f30d6ad6572544e7c2b1b.jpg](../icml_results/1079_Certain%20Head%2C%20Uncertain%20Tail_%20Expert-Sample%20for%20Test-Time%20Scaling%20in%20Fine-Grained%20MoE/tables/fb5c632f70746a32d74d27a27a4ba9c56fb5914e8b1f30d6ad6572544e7c2b1b.jpg)

<a id="paper-1062"></a>

## Learning to Self-Verify Makes Language Models Better Reasoners


### Images

![5fc6b227428e3ed85ea83044176aca7d4ffbfb78bc18bc4194574a036b2785d4.jpg](../icml_results/1080_Learning%20to%20Self-Verify%20Makes%20Language%20Models%20Better%20Reasoners/images/5fc6b227428e3ed85ea83044176aca7d4ffbfb78bc18bc4194574a036b2785d4.jpg)

![96b113aac3bdc6838a0781dce164c50903c2530784df9716a8688337c59f09a6.jpg](../icml_results/1080_Learning%20to%20Self-Verify%20Makes%20Language%20Models%20Better%20Reasoners/images/96b113aac3bdc6838a0781dce164c50903c2530784df9716a8688337c59f09a6.jpg)

![a25e4f516d43b790f37978f54f7fde757ced96b11b540ca0d2ad34ef71be5dbb.jpg](../icml_results/1080_Learning%20to%20Self-Verify%20Makes%20Language%20Models%20Better%20Reasoners/images/a25e4f516d43b790f37978f54f7fde757ced96b11b540ca0d2ad34ef71be5dbb.jpg)

![ef603d2f2fee7aa731ee845a649899df566cacd31e052a9384cb99b89f3a08a6.jpg](../icml_results/1080_Learning%20to%20Self-Verify%20Makes%20Language%20Models%20Better%20Reasoners/images/ef603d2f2fee7aa731ee845a649899df566cacd31e052a9384cb99b89f3a08a6.jpg)

### Tables

![00d4b3994597fc2924cb9aab8c3525ddb07302e5550ba9fd13277ba2f53da9cd.jpg](../icml_results/1080_Learning%20to%20Self-Verify%20Makes%20Language%20Models%20Better%20Reasoners/tables/00d4b3994597fc2924cb9aab8c3525ddb07302e5550ba9fd13277ba2f53da9cd.jpg)

![023fe7ae1199b430023ec5d1c519cf32382d081557cdaa6d4bb42905ff099725.jpg](../icml_results/1080_Learning%20to%20Self-Verify%20Makes%20Language%20Models%20Better%20Reasoners/tables/023fe7ae1199b430023ec5d1c519cf32382d081557cdaa6d4bb42905ff099725.jpg)

![2ecdb4718ffe24c68c4215dd82d36adc055930fdc98b049a9c7732eadb2ad632.jpg](../icml_results/1080_Learning%20to%20Self-Verify%20Makes%20Language%20Models%20Better%20Reasoners/tables/2ecdb4718ffe24c68c4215dd82d36adc055930fdc98b049a9c7732eadb2ad632.jpg)

![410b64bb57a91efd3961387f07419c06a469201136f46a83302dc2a9947d1096.jpg](../icml_results/1080_Learning%20to%20Self-Verify%20Makes%20Language%20Models%20Better%20Reasoners/tables/410b64bb57a91efd3961387f07419c06a469201136f46a83302dc2a9947d1096.jpg)

![726ac82d9fc1bd11a431c71cb65209cd34ffae982ab4ea147be6aec6c20134b8.jpg](../icml_results/1080_Learning%20to%20Self-Verify%20Makes%20Language%20Models%20Better%20Reasoners/tables/726ac82d9fc1bd11a431c71cb65209cd34ffae982ab4ea147be6aec6c20134b8.jpg)

![81c5dc6deee1db947a4fa65852ab5082a0b2222be97bad5aff4e310d3cbb3a04.jpg](../icml_results/1080_Learning%20to%20Self-Verify%20Makes%20Language%20Models%20Better%20Reasoners/tables/81c5dc6deee1db947a4fa65852ab5082a0b2222be97bad5aff4e310d3cbb3a04.jpg)

![94debaf83dbc85046313707575ca5496664099399a3e1ded823ac3bd39124c46.jpg](../icml_results/1080_Learning%20to%20Self-Verify%20Makes%20Language%20Models%20Better%20Reasoners/tables/94debaf83dbc85046313707575ca5496664099399a3e1ded823ac3bd39124c46.jpg)

![b54e269613f1de7b8df8ef05b2dd05f1b8441e7a179801fe8a8b19c0b4ab6d70.jpg](../icml_results/1080_Learning%20to%20Self-Verify%20Makes%20Language%20Models%20Better%20Reasoners/tables/b54e269613f1de7b8df8ef05b2dd05f1b8441e7a179801fe8a8b19c0b4ab6d70.jpg)

![c5c09c4e74cd1339836fe7976d2d1657671066cfee172879c3fcf3edf7e7e240.jpg](../icml_results/1080_Learning%20to%20Self-Verify%20Makes%20Language%20Models%20Better%20Reasoners/tables/c5c09c4e74cd1339836fe7976d2d1657671066cfee172879c3fcf3edf7e7e240.jpg)

![e79a3a8c4bf865291664821b9cbe6ebc192a5988a88ed6157422bf881a075278.jpg](../icml_results/1080_Learning%20to%20Self-Verify%20Makes%20Language%20Models%20Better%20Reasoners/tables/e79a3a8c4bf865291664821b9cbe6ebc192a5988a88ed6157422bf881a075278.jpg)

![f2f0426cf7bcf1176027a41df40904ea082773a842c0a580f3ea189383768ce9.jpg](../icml_results/1080_Learning%20to%20Self-Verify%20Makes%20Language%20Models%20Better%20Reasoners/tables/f2f0426cf7bcf1176027a41df40904ea082773a842c0a580f3ea189383768ce9.jpg)

![fe0beb24bee5306127c3a9b50888e468357dbb0b086d1191e4927adc9e89d896.jpg](../icml_results/1080_Learning%20to%20Self-Verify%20Makes%20Language%20Models%20Better%20Reasoners/tables/fe0beb24bee5306127c3a9b50888e468357dbb0b086d1191e4927adc9e89d896.jpg)

<a id="paper-1063"></a>

## From Extraction to Deduction: Resolving Functional Misalignment in RAG via a Collaborative Critic-Reasoner Framework


### Images

![00501254e34c11574f0163a3d271f0374efc0632ff70b0043e4e9c2b7eddedad.jpg](../icml_results/1081_From%20Extraction%20to%20Deduction_%20Resolving%20Functional%20Misalignment%20in%20RAG%20via%20a%20Collaborative%20Critic-Re/images/00501254e34c11574f0163a3d271f0374efc0632ff70b0043e4e9c2b7eddedad.jpg)

![13be4155c4c330ce60fc8a1e843018c6262d7bf42bfc29c784a3c076fa21a871.jpg](../icml_results/1081_From%20Extraction%20to%20Deduction_%20Resolving%20Functional%20Misalignment%20in%20RAG%20via%20a%20Collaborative%20Critic-Re/images/13be4155c4c330ce60fc8a1e843018c6262d7bf42bfc29c784a3c076fa21a871.jpg)

![2230377811552c59a497437eab9dcc75c0206ea346a0762b0a794df26a9121e2.jpg](../icml_results/1081_From%20Extraction%20to%20Deduction_%20Resolving%20Functional%20Misalignment%20in%20RAG%20via%20a%20Collaborative%20Critic-Re/images/2230377811552c59a497437eab9dcc75c0206ea346a0762b0a794df26a9121e2.jpg)

![69c74ae978e35b2139e8cb2b4f7b9f70e0f2d4ef41f82ff11300c06f639c8281.jpg](../icml_results/1081_From%20Extraction%20to%20Deduction_%20Resolving%20Functional%20Misalignment%20in%20RAG%20via%20a%20Collaborative%20Critic-Re/images/69c74ae978e35b2139e8cb2b4f7b9f70e0f2d4ef41f82ff11300c06f639c8281.jpg)

### Tables

![0e5ffdf131947eec99b2681f0da63fd755303f87fd9a4d11df865538cd36754e.jpg](../icml_results/1081_From%20Extraction%20to%20Deduction_%20Resolving%20Functional%20Misalignment%20in%20RAG%20via%20a%20Collaborative%20Critic-Re/tables/0e5ffdf131947eec99b2681f0da63fd755303f87fd9a4d11df865538cd36754e.jpg)

![129b77c22741448d1ec02ea786ebbd5b63a4ab0c845965683b381709edd28b0f.jpg](../icml_results/1081_From%20Extraction%20to%20Deduction_%20Resolving%20Functional%20Misalignment%20in%20RAG%20via%20a%20Collaborative%20Critic-Re/tables/129b77c22741448d1ec02ea786ebbd5b63a4ab0c845965683b381709edd28b0f.jpg)

![12a4fdc753ca6aedce7caa75b29fd92bf3e1109dbce01a052724549940a8d8e0.jpg](../icml_results/1081_From%20Extraction%20to%20Deduction_%20Resolving%20Functional%20Misalignment%20in%20RAG%20via%20a%20Collaborative%20Critic-Re/tables/12a4fdc753ca6aedce7caa75b29fd92bf3e1109dbce01a052724549940a8d8e0.jpg)

![21943eb1e0b67873d876813ce90a6346fc0667360cbb7cd17cec07f98e4ae9bb.jpg](../icml_results/1081_From%20Extraction%20to%20Deduction_%20Resolving%20Functional%20Misalignment%20in%20RAG%20via%20a%20Collaborative%20Critic-Re/tables/21943eb1e0b67873d876813ce90a6346fc0667360cbb7cd17cec07f98e4ae9bb.jpg)

![2474cbaf5b419a8fd65a2a2901a4762468d9c1256bb6a000357e1e44e93f7efe.jpg](../icml_results/1081_From%20Extraction%20to%20Deduction_%20Resolving%20Functional%20Misalignment%20in%20RAG%20via%20a%20Collaborative%20Critic-Re/tables/2474cbaf5b419a8fd65a2a2901a4762468d9c1256bb6a000357e1e44e93f7efe.jpg)

![2d5f4605fc3b69223afbf20f9d72e8cf3eda9e957b3ef34ed4611624909d3c4c.jpg](../icml_results/1081_From%20Extraction%20to%20Deduction_%20Resolving%20Functional%20Misalignment%20in%20RAG%20via%20a%20Collaborative%20Critic-Re/tables/2d5f4605fc3b69223afbf20f9d72e8cf3eda9e957b3ef34ed4611624909d3c4c.jpg)

![556fe728cf6bfae9eecb55dacf84237268e9e92b4d1d329cedabdb26a4e9b6ad.jpg](../icml_results/1081_From%20Extraction%20to%20Deduction_%20Resolving%20Functional%20Misalignment%20in%20RAG%20via%20a%20Collaborative%20Critic-Re/tables/556fe728cf6bfae9eecb55dacf84237268e9e92b4d1d329cedabdb26a4e9b6ad.jpg)

![5c326c7e0ff8550cd554d2a95c219cc87da7f59a6dc2d6e9e964b5e0f6872cc2.jpg](../icml_results/1081_From%20Extraction%20to%20Deduction_%20Resolving%20Functional%20Misalignment%20in%20RAG%20via%20a%20Collaborative%20Critic-Re/tables/5c326c7e0ff8550cd554d2a95c219cc87da7f59a6dc2d6e9e964b5e0f6872cc2.jpg)

![63e883351dc7e6760b4d1d89a49017e1a92c429917082ecc475c7faf2adc4846.jpg](../icml_results/1081_From%20Extraction%20to%20Deduction_%20Resolving%20Functional%20Misalignment%20in%20RAG%20via%20a%20Collaborative%20Critic-Re/tables/63e883351dc7e6760b4d1d89a49017e1a92c429917082ecc475c7faf2adc4846.jpg)

![a3b062282bde268505a9b7437b70e35e683746ac168c923f8ae07d5e4215503c.jpg](../icml_results/1081_From%20Extraction%20to%20Deduction_%20Resolving%20Functional%20Misalignment%20in%20RAG%20via%20a%20Collaborative%20Critic-Re/tables/a3b062282bde268505a9b7437b70e35e683746ac168c923f8ae07d5e4215503c.jpg)

![d0ce93867f6a5ad75baf1135eebca75a77b28a35e8804e7f47247a40c5c83fb6.jpg](../icml_results/1081_From%20Extraction%20to%20Deduction_%20Resolving%20Functional%20Misalignment%20in%20RAG%20via%20a%20Collaborative%20Critic-Re/tables/d0ce93867f6a5ad75baf1135eebca75a77b28a35e8804e7f47247a40c5c83fb6.jpg)

![de97cb9803aa73b73bb79b08206a03f89aeae96353a7cdbe6303a8b7eb925326.jpg](../icml_results/1081_From%20Extraction%20to%20Deduction_%20Resolving%20Functional%20Misalignment%20in%20RAG%20via%20a%20Collaborative%20Critic-Re/tables/de97cb9803aa73b73bb79b08206a03f89aeae96353a7cdbe6303a8b7eb925326.jpg)

![f98bcd89fd4f11fadcea86862d32ce35f9ef89ad855eee68d0f114e1bf07d75a.jpg](../icml_results/1081_From%20Extraction%20to%20Deduction_%20Resolving%20Functional%20Misalignment%20in%20RAG%20via%20a%20Collaborative%20Critic-Re/tables/f98bcd89fd4f11fadcea86862d32ce35f9ef89ad855eee68d0f114e1bf07d75a.jpg)

<a id="paper-1064"></a>

## Context Forcing: Consistent Autoregressive Video Generation with Long Context


### Images

![17a249d477124569c9c675819c870f4d3999ab231d9d6c167df1bf43cff473cc.jpg](../icml_results/1082_Context%20Forcing_%20Consistent%20Autoregressive%20Video%20Generation%20with%20Long%20Context/images/17a249d477124569c9c675819c870f4d3999ab231d9d6c167df1bf43cff473cc.jpg)

![24710d84d304743e4e7dc5022aa0b3cd354b5e101888cc52eb9185e995c554d2.jpg](../icml_results/1082_Context%20Forcing_%20Consistent%20Autoregressive%20Video%20Generation%20with%20Long%20Context/images/24710d84d304743e4e7dc5022aa0b3cd354b5e101888cc52eb9185e995c554d2.jpg)

![24c6ef8e15e3692969f1271ac62006ec535eb8e7c14070510c9e905ffc9993a4.jpg](../icml_results/1082_Context%20Forcing_%20Consistent%20Autoregressive%20Video%20Generation%20with%20Long%20Context/images/24c6ef8e15e3692969f1271ac62006ec535eb8e7c14070510c9e905ffc9993a4.jpg)

![3a223faaf2706490ea62e510c3c3207e9f92f67f753b281f051e61fe4e761ee1.jpg](../icml_results/1082_Context%20Forcing_%20Consistent%20Autoregressive%20Video%20Generation%20with%20Long%20Context/images/3a223faaf2706490ea62e510c3c3207e9f92f67f753b281f051e61fe4e761ee1.jpg)

![3bfe342f977ec21801a5c0e12576c60863e213a37cc031bec241c50a9e501eaf.jpg](../icml_results/1082_Context%20Forcing_%20Consistent%20Autoregressive%20Video%20Generation%20with%20Long%20Context/images/3bfe342f977ec21801a5c0e12576c60863e213a37cc031bec241c50a9e501eaf.jpg)

![402d9fcb73959f668d2e413cade066952ead83cba8319b5f6706ed4411b881e4.jpg](../icml_results/1082_Context%20Forcing_%20Consistent%20Autoregressive%20Video%20Generation%20with%20Long%20Context/images/402d9fcb73959f668d2e413cade066952ead83cba8319b5f6706ed4411b881e4.jpg)

![7031a36459661e51fbd6ff2e763cef40d0b99909deb30335bb217bbe3f6a2562.jpg](../icml_results/1082_Context%20Forcing_%20Consistent%20Autoregressive%20Video%20Generation%20with%20Long%20Context/images/7031a36459661e51fbd6ff2e763cef40d0b99909deb30335bb217bbe3f6a2562.jpg)

![91bb055b84606ca07424960fadb2a3c575c10cad3c0d40ba98a6508ede904506.jpg](../icml_results/1082_Context%20Forcing_%20Consistent%20Autoregressive%20Video%20Generation%20with%20Long%20Context/images/91bb055b84606ca07424960fadb2a3c575c10cad3c0d40ba98a6508ede904506.jpg)

![bcf33be79a9e1edc3f91fc265c341354af698cf6c1d6ae24e99f9bc88963dca6.jpg](../icml_results/1082_Context%20Forcing_%20Consistent%20Autoregressive%20Video%20Generation%20with%20Long%20Context/images/bcf33be79a9e1edc3f91fc265c341354af698cf6c1d6ae24e99f9bc88963dca6.jpg)

![cfe7d4bd95c31ebaa2a67e7004ad413be8e12f8bf02a03ec8954d5dd584ed4ed.jpg](../icml_results/1082_Context%20Forcing_%20Consistent%20Autoregressive%20Video%20Generation%20with%20Long%20Context/images/cfe7d4bd95c31ebaa2a67e7004ad413be8e12f8bf02a03ec8954d5dd584ed4ed.jpg)

![da0fe9363a010d6301fe9c0b80ac102c1b89e6ba44492cd6fb0e5120dbad6fde.jpg](../icml_results/1082_Context%20Forcing_%20Consistent%20Autoregressive%20Video%20Generation%20with%20Long%20Context/images/da0fe9363a010d6301fe9c0b80ac102c1b89e6ba44492cd6fb0e5120dbad6fde.jpg)

### Tables

![07fe899deed899cfe694150969dbf0f7fbdf676b2e21bc72cd62d541f96d3ed4.jpg](../icml_results/1082_Context%20Forcing_%20Consistent%20Autoregressive%20Video%20Generation%20with%20Long%20Context/tables/07fe899deed899cfe694150969dbf0f7fbdf676b2e21bc72cd62d541f96d3ed4.jpg)

![28a13fbfc6dd204f003497f673488a3f8d083c803213f6682f0f8afba0fcfed8.jpg](../icml_results/1082_Context%20Forcing_%20Consistent%20Autoregressive%20Video%20Generation%20with%20Long%20Context/tables/28a13fbfc6dd204f003497f673488a3f8d083c803213f6682f0f8afba0fcfed8.jpg)

![a24fcd4fa199fd99d6fde8851fa5d8aa6c890382c036fb657e9dc850f43032e6.jpg](../icml_results/1082_Context%20Forcing_%20Consistent%20Autoregressive%20Video%20Generation%20with%20Long%20Context/tables/a24fcd4fa199fd99d6fde8851fa5d8aa6c890382c036fb657e9dc850f43032e6.jpg)

![b1e7bfcd584c1acf7b6c4d3093a5d7872a6d36e975f3c18d11c5d020023bd8d2.jpg](../icml_results/1082_Context%20Forcing_%20Consistent%20Autoregressive%20Video%20Generation%20with%20Long%20Context/tables/b1e7bfcd584c1acf7b6c4d3093a5d7872a6d36e975f3c18d11c5d020023bd8d2.jpg)

![bf9cd21407424323bfe287ee4e6a4c2ad1618111cabf2274e98e25fff412d3a2.jpg](../icml_results/1082_Context%20Forcing_%20Consistent%20Autoregressive%20Video%20Generation%20with%20Long%20Context/tables/bf9cd21407424323bfe287ee4e6a4c2ad1618111cabf2274e98e25fff412d3a2.jpg)

![de27c60f06cd334be55a3cef91a3de5f7cd4eff381caf1dbea02cd8c87bbf9a2.jpg](../icml_results/1082_Context%20Forcing_%20Consistent%20Autoregressive%20Video%20Generation%20with%20Long%20Context/tables/de27c60f06cd334be55a3cef91a3de5f7cd4eff381caf1dbea02cd8c87bbf9a2.jpg)

<a id="paper-1065"></a>

## Stability and Generalization of Nonconvex Optimization with Heavy-Tailed Noise


### Tables

![40c7bc675fe101f9bcebce43d737434f02532bb44719fe0971907503a30d776f.jpg](../icml_results/1083_Stability%20and%20Generalization%20of%20Nonconvex%20Optimization%20with%20Heavy-Tailed%20Noise/tables/40c7bc675fe101f9bcebce43d737434f02532bb44719fe0971907503a30d776f.jpg)

![64a9e6b85254594c1a0cb58ba55e2b5f9e0161fe0c81195d7ecc611c2705a01a.jpg](../icml_results/1083_Stability%20and%20Generalization%20of%20Nonconvex%20Optimization%20with%20Heavy-Tailed%20Noise/tables/64a9e6b85254594c1a0cb58ba55e2b5f9e0161fe0c81195d7ecc611c2705a01a.jpg)

![f2c1680d69e84bc439101c9641d0416e20212f19cb66c45d274f4a0acc81a3a0.jpg](../icml_results/1083_Stability%20and%20Generalization%20of%20Nonconvex%20Optimization%20with%20Heavy-Tailed%20Noise/tables/f2c1680d69e84bc439101c9641d0416e20212f19cb66c45d274f4a0acc81a3a0.jpg)

<a id="paper-1066"></a>

## SCHUR-A*: Layer-wise Optimal Expert Pruning for MoEs via Schur-Complement Guided A* Search


### Images

![174f638ef61da991126203caa07bb97058900ce0a00e9d50c506069d03394a23.jpg](../icml_results/1084_SCHUR-A__%20Layer-wise%20Optimal%20Expert%20Pruning%20for%20MoEs%20via%20Schur-Complement%20Guided%20A_%20Search/images/174f638ef61da991126203caa07bb97058900ce0a00e9d50c506069d03394a23.jpg)

![28a6020cae81fa32a7e281ab30304fbd523848d30c2fdb25053c11bb1d65ae09.jpg](../icml_results/1084_SCHUR-A__%20Layer-wise%20Optimal%20Expert%20Pruning%20for%20MoEs%20via%20Schur-Complement%20Guided%20A_%20Search/images/28a6020cae81fa32a7e281ab30304fbd523848d30c2fdb25053c11bb1d65ae09.jpg)

![79867b23399bbe7a8bccd98ecf0026cea560c25a3388eb93f4fd0686b96451f0.jpg](../icml_results/1084_SCHUR-A__%20Layer-wise%20Optimal%20Expert%20Pruning%20for%20MoEs%20via%20Schur-Complement%20Guided%20A_%20Search/images/79867b23399bbe7a8bccd98ecf0026cea560c25a3388eb93f4fd0686b96451f0.jpg)

![7ab144b3a5fcaa1f2205c9f9dc4575a3ada70f343a838093436b35b9ee4f557e.jpg](../icml_results/1084_SCHUR-A__%20Layer-wise%20Optimal%20Expert%20Pruning%20for%20MoEs%20via%20Schur-Complement%20Guided%20A_%20Search/images/7ab144b3a5fcaa1f2205c9f9dc4575a3ada70f343a838093436b35b9ee4f557e.jpg)

![7ac162febd9c32299005f88ac3e8f14fad2b3ba39ba624bae1d23179d07869b7.jpg](../icml_results/1084_SCHUR-A__%20Layer-wise%20Optimal%20Expert%20Pruning%20for%20MoEs%20via%20Schur-Complement%20Guided%20A_%20Search/images/7ac162febd9c32299005f88ac3e8f14fad2b3ba39ba624bae1d23179d07869b7.jpg)

![90834bc0bb5a91b47013f6479348ffeb25ed94d5cc70b770fd686fcd3c3abb57.jpg](../icml_results/1084_SCHUR-A__%20Layer-wise%20Optimal%20Expert%20Pruning%20for%20MoEs%20via%20Schur-Complement%20Guided%20A_%20Search/images/90834bc0bb5a91b47013f6479348ffeb25ed94d5cc70b770fd686fcd3c3abb57.jpg)

![9d83bfb13c33589ab67f623a172d7dcaa1c4a86266d37f5ebee7da3ce2106e02.jpg](../icml_results/1084_SCHUR-A__%20Layer-wise%20Optimal%20Expert%20Pruning%20for%20MoEs%20via%20Schur-Complement%20Guided%20A_%20Search/images/9d83bfb13c33589ab67f623a172d7dcaa1c4a86266d37f5ebee7da3ce2106e02.jpg)

![ce59f1fe8fcd836a66cf1b9c29d4d95aa1e103c4820d4975c7556ef0a32feae2.jpg](../icml_results/1084_SCHUR-A__%20Layer-wise%20Optimal%20Expert%20Pruning%20for%20MoEs%20via%20Schur-Complement%20Guided%20A_%20Search/images/ce59f1fe8fcd836a66cf1b9c29d4d95aa1e103c4820d4975c7556ef0a32feae2.jpg)

![eb46c817aff34de4eef26083641181dbf99e529a903b5fecaefc63d190d2a454.jpg](../icml_results/1084_SCHUR-A__%20Layer-wise%20Optimal%20Expert%20Pruning%20for%20MoEs%20via%20Schur-Complement%20Guided%20A_%20Search/images/eb46c817aff34de4eef26083641181dbf99e529a903b5fecaefc63d190d2a454.jpg)

### Tables

![0a6fe4c6595a73f923bb29378abdf410969bb47ec2d48b11e2816aad5eeeaec4.jpg](../icml_results/1084_SCHUR-A__%20Layer-wise%20Optimal%20Expert%20Pruning%20for%20MoEs%20via%20Schur-Complement%20Guided%20A_%20Search/tables/0a6fe4c6595a73f923bb29378abdf410969bb47ec2d48b11e2816aad5eeeaec4.jpg)

![2acba7d76f1b5217f9539092e7407de442efa1e5eaedcdf2c1f6bb0ffc5f701e.jpg](../icml_results/1084_SCHUR-A__%20Layer-wise%20Optimal%20Expert%20Pruning%20for%20MoEs%20via%20Schur-Complement%20Guided%20A_%20Search/tables/2acba7d76f1b5217f9539092e7407de442efa1e5eaedcdf2c1f6bb0ffc5f701e.jpg)

![4abbfd74b94866aea549edc83ee149fb065ff1b26fd33183fd0491a574c85a9c.jpg](../icml_results/1084_SCHUR-A__%20Layer-wise%20Optimal%20Expert%20Pruning%20for%20MoEs%20via%20Schur-Complement%20Guided%20A_%20Search/tables/4abbfd74b94866aea549edc83ee149fb065ff1b26fd33183fd0491a574c85a9c.jpg)

![6499c65287c4244f74b042dd81754171ad7ba674c018e8a66f65611c65550c1d.jpg](../icml_results/1084_SCHUR-A__%20Layer-wise%20Optimal%20Expert%20Pruning%20for%20MoEs%20via%20Schur-Complement%20Guided%20A_%20Search/tables/6499c65287c4244f74b042dd81754171ad7ba674c018e8a66f65611c65550c1d.jpg)

![78c1048aed3c6dd083d72c5531a66052a3574477a52ab94d8ebbe9ce94165e9f.jpg](../icml_results/1084_SCHUR-A__%20Layer-wise%20Optimal%20Expert%20Pruning%20for%20MoEs%20via%20Schur-Complement%20Guided%20A_%20Search/tables/78c1048aed3c6dd083d72c5531a66052a3574477a52ab94d8ebbe9ce94165e9f.jpg)

![9ed79b0859fc2d57aad188f8441110f19fb2956521348a0ba5688e8fe677ef62.jpg](../icml_results/1084_SCHUR-A__%20Layer-wise%20Optimal%20Expert%20Pruning%20for%20MoEs%20via%20Schur-Complement%20Guided%20A_%20Search/tables/9ed79b0859fc2d57aad188f8441110f19fb2956521348a0ba5688e8fe677ef62.jpg)

![ca1a83bad96096ae9ce063d5693d62333402d3d02d7512fcd5883bc253c95a34.jpg](../icml_results/1084_SCHUR-A__%20Layer-wise%20Optimal%20Expert%20Pruning%20for%20MoEs%20via%20Schur-Complement%20Guided%20A_%20Search/tables/ca1a83bad96096ae9ce063d5693d62333402d3d02d7512fcd5883bc253c95a34.jpg)

![e386640982e1af366816494cc37757167b5f93107c7b21ec95fc290af08d2a90.jpg](../icml_results/1084_SCHUR-A__%20Layer-wise%20Optimal%20Expert%20Pruning%20for%20MoEs%20via%20Schur-Complement%20Guided%20A_%20Search/tables/e386640982e1af366816494cc37757167b5f93107c7b21ec95fc290af08d2a90.jpg)

<a id="paper-1067"></a>

## QiMeng-PerceptOS: Semantic-Aware Kernel Optimization for OS-Intensive Workloads via Hardware-Software Alignment


### Images

![01f8bd8464e28bd4f50dd7435afb9696cac6c2b8471ac32b7f2c4b9007987212.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/images/01f8bd8464e28bd4f50dd7435afb9696cac6c2b8471ac32b7f2c4b9007987212.jpg)

![20396dcedec97d59eccabf65b0353fbbfc09cfe4b7e78080f7dfa1342be201a4.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/images/20396dcedec97d59eccabf65b0353fbbfc09cfe4b7e78080f7dfa1342be201a4.jpg)

![84a203a5c3d0654d205319c66d67098660018b9009a586985d5947094bce4697.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/images/84a203a5c3d0654d205319c66d67098660018b9009a586985d5947094bce4697.jpg)

![c3162afdfefb27280b31c092466ca50bdc9783c1da0550dec794b5e9fa958f2c.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/images/c3162afdfefb27280b31c092466ca50bdc9783c1da0550dec794b5e9fa958f2c.jpg)

![dc505b36f200cbb36d98cadc6c98f8b95e40409fa951857d9cba3ce4135a4ef6.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/images/dc505b36f200cbb36d98cadc6c98f8b95e40409fa951857d9cba3ce4135a4ef6.jpg)

![e28c26072d43cb11f83a5dee29d54f55faaff427cf34993230f340820e24612e.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/images/e28c26072d43cb11f83a5dee29d54f55faaff427cf34993230f340820e24612e.jpg)

### Tables

![0af51d28eec5be56e6d5c0cbdb38f3a4e2e1587250056cb47422199a02ac3f84.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/tables/0af51d28eec5be56e6d5c0cbdb38f3a4e2e1587250056cb47422199a02ac3f84.jpg)

![12216c58f42ab1545c7fc245b6427656b9018ba38a85054333dfb3f3dadcd041.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/tables/12216c58f42ab1545c7fc245b6427656b9018ba38a85054333dfb3f3dadcd041.jpg)

![14695b0c028ae3d437a35f2c4d4ccc87e026ed7e718e23b8babc05de27d6762a.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/tables/14695b0c028ae3d437a35f2c4d4ccc87e026ed7e718e23b8babc05de27d6762a.jpg)

![16e33cfcf7b22614f0f1f3569e47e474c7452aa0393e1d1b89cfea13d164cb14.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/tables/16e33cfcf7b22614f0f1f3569e47e474c7452aa0393e1d1b89cfea13d164cb14.jpg)

![34370b71cad2e4c501ca238226fb14260dd329ae86bae0d3c6bc9ec3880e5739.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/tables/34370b71cad2e4c501ca238226fb14260dd329ae86bae0d3c6bc9ec3880e5739.jpg)

![4b5422c3516fdcdcc8a819a895138e8ff8edd87eb20a17e4034dde8be59d86f5.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/tables/4b5422c3516fdcdcc8a819a895138e8ff8edd87eb20a17e4034dde8be59d86f5.jpg)

![62bf6eafabd495378b502228fb6059e6e9cd4238e33070417beca2aca12a1cff.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/tables/62bf6eafabd495378b502228fb6059e6e9cd4238e33070417beca2aca12a1cff.jpg)

![6449884e08282ca448bae415452ad26f34411facd6d1568ebb545ba0dc99a48a.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/tables/6449884e08282ca448bae415452ad26f34411facd6d1568ebb545ba0dc99a48a.jpg)

![75e006c9faf76137e4e16ae14e57d8a33bf939dbc18ffb77c77d55da95493f80.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/tables/75e006c9faf76137e4e16ae14e57d8a33bf939dbc18ffb77c77d55da95493f80.jpg)

![a7d1c25d25519b01ee33e33bd784f3222a2cb9c0cf35968a6f71ed7731f0e96a.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/tables/a7d1c25d25519b01ee33e33bd784f3222a2cb9c0cf35968a6f71ed7731f0e96a.jpg)

![aea114986dce6d0ad925a9f280a000c638985df62c8a9c4761fa8de93c19b296.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/tables/aea114986dce6d0ad925a9f280a000c638985df62c8a9c4761fa8de93c19b296.jpg)

![afcee8250b39ce5c28751628212bc0cb594000640343855e3426d0ec8f573cb4.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/tables/afcee8250b39ce5c28751628212bc0cb594000640343855e3426d0ec8f573cb4.jpg)

![b5815602f2d49db74de37e857a60a29a79e75d9448a43d477d59a336ec11752e.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/tables/b5815602f2d49db74de37e857a60a29a79e75d9448a43d477d59a336ec11752e.jpg)

![b76cf7f3e54ff42e55b2ce4b4ef80e4332eb4ce302090f9822a89cbf0cbeadbd.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/tables/b76cf7f3e54ff42e55b2ce4b4ef80e4332eb4ce302090f9822a89cbf0cbeadbd.jpg)

![f9d4c00d9752225852904b9ce05d072471b8c99601f33936859d9d8f8b3c4ee6.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/tables/f9d4c00d9752225852904b9ce05d072471b8c99601f33936859d9d8f8b3c4ee6.jpg)

![feeb9a5c91bc46f81e0f5d516eb68fb7faa5ab855f7cb7e730a12b983cdbbf7e.jpg](../icml_results/1085_QiMeng-PerceptOS_%20Semantic-Aware%20Kernel%20Optimization%20for%20OS-Intensive%20Workloads%20via%20Hardware-Softwar/tables/feeb9a5c91bc46f81e0f5d516eb68fb7faa5ab855f7cb7e730a12b983cdbbf7e.jpg)

<a id="paper-1068"></a>

## SemanticNVS: Improving Semantic Scene Understanding in Generative Novel View Synthesis


### Images

![1c965f20ac9f1113c18566fb92af752dbf70f55d878fa0d9c5ce8c7c075ca4b0.jpg](../icml_results/1086_SemanticNVS_%20Improving%20Semantic%20Scene%20Understanding%20in%20Generative%20Novel%20View%20Synthesis/images/1c965f20ac9f1113c18566fb92af752dbf70f55d878fa0d9c5ce8c7c075ca4b0.jpg)

![21a187092db8ff5a655835bc7503781f5f4d903030f1fd42d31e73695664354b.jpg](../icml_results/1086_SemanticNVS_%20Improving%20Semantic%20Scene%20Understanding%20in%20Generative%20Novel%20View%20Synthesis/images/21a187092db8ff5a655835bc7503781f5f4d903030f1fd42d31e73695664354b.jpg)

![5c8d6d24eaaa275945b6fce36cc020d598ff45ae7194974c7af069f6227cc33b.jpg](../icml_results/1086_SemanticNVS_%20Improving%20Semantic%20Scene%20Understanding%20in%20Generative%20Novel%20View%20Synthesis/images/5c8d6d24eaaa275945b6fce36cc020d598ff45ae7194974c7af069f6227cc33b.jpg)

![88d347371b67b4aa0b7bf6bdcdef63e6ad8f7f5896d0fb84bc15d77dd801a760.jpg](../icml_results/1086_SemanticNVS_%20Improving%20Semantic%20Scene%20Understanding%20in%20Generative%20Novel%20View%20Synthesis/images/88d347371b67b4aa0b7bf6bdcdef63e6ad8f7f5896d0fb84bc15d77dd801a760.jpg)

![a101f9b601cfbd591b7f44438c48376be9e788689f38a85247ffe60c23644c56.jpg](../icml_results/1086_SemanticNVS_%20Improving%20Semantic%20Scene%20Understanding%20in%20Generative%20Novel%20View%20Synthesis/images/a101f9b601cfbd591b7f44438c48376be9e788689f38a85247ffe60c23644c56.jpg)

![de1c3c9c0e32aaff54d3f6c6b44ed1ac381d6df4c1ffeab74355ea9a646392e9.jpg](../icml_results/1086_SemanticNVS_%20Improving%20Semantic%20Scene%20Understanding%20in%20Generative%20Novel%20View%20Synthesis/images/de1c3c9c0e32aaff54d3f6c6b44ed1ac381d6df4c1ffeab74355ea9a646392e9.jpg)

### Tables

![25156230694c8ac06e0dd873a5872666094b3828f37056370f3cb0062af296ed.jpg](../icml_results/1086_SemanticNVS_%20Improving%20Semantic%20Scene%20Understanding%20in%20Generative%20Novel%20View%20Synthesis/tables/25156230694c8ac06e0dd873a5872666094b3828f37056370f3cb0062af296ed.jpg)

![3e4d5c9ed6b8da4fb785b6063a8f1662595d983e174bbc4b7786ab5488561419.jpg](../icml_results/1086_SemanticNVS_%20Improving%20Semantic%20Scene%20Understanding%20in%20Generative%20Novel%20View%20Synthesis/tables/3e4d5c9ed6b8da4fb785b6063a8f1662595d983e174bbc4b7786ab5488561419.jpg)

![9f8bef774cb8a70a9e0d2ffa332d36dc7599f22b5c3a85d03c1f0e14f765d04c.jpg](../icml_results/1086_SemanticNVS_%20Improving%20Semantic%20Scene%20Understanding%20in%20Generative%20Novel%20View%20Synthesis/tables/9f8bef774cb8a70a9e0d2ffa332d36dc7599f22b5c3a85d03c1f0e14f765d04c.jpg)

![a6269f88799710e133c0c9ce45a8c4596f2fda341c365bf6b0bf5aec90ecfa79.jpg](../icml_results/1086_SemanticNVS_%20Improving%20Semantic%20Scene%20Understanding%20in%20Generative%20Novel%20View%20Synthesis/tables/a6269f88799710e133c0c9ce45a8c4596f2fda341c365bf6b0bf5aec90ecfa79.jpg)

![c2ddf6e2ae86747b0137c594610dad4abc12a18aad348b31221bdfc2195a72ff.jpg](../icml_results/1086_SemanticNVS_%20Improving%20Semantic%20Scene%20Understanding%20in%20Generative%20Novel%20View%20Synthesis/tables/c2ddf6e2ae86747b0137c594610dad4abc12a18aad348b31221bdfc2195a72ff.jpg)

![c347b3a1049530b2b989f38dbab2cab10fc9ce037885e7e637580c23ced12233.jpg](../icml_results/1086_SemanticNVS_%20Improving%20Semantic%20Scene%20Understanding%20in%20Generative%20Novel%20View%20Synthesis/tables/c347b3a1049530b2b989f38dbab2cab10fc9ce037885e7e637580c23ced12233.jpg)

![c71014b3df9ee0dfbe332bc3142f40afdad55bc1732044c7ca7252737d7155c3.jpg](../icml_results/1086_SemanticNVS_%20Improving%20Semantic%20Scene%20Understanding%20in%20Generative%20Novel%20View%20Synthesis/tables/c71014b3df9ee0dfbe332bc3142f40afdad55bc1732044c7ca7252737d7155c3.jpg)

![e87c4a37dd172d25fe2c8c5b0fc6a83b49f8259eec9ea53ad7591f82f65ffef4.jpg](../icml_results/1086_SemanticNVS_%20Improving%20Semantic%20Scene%20Understanding%20in%20Generative%20Novel%20View%20Synthesis/tables/e87c4a37dd172d25fe2c8c5b0fc6a83b49f8259eec9ea53ad7591f82f65ffef4.jpg)

![ee3ec1688c73e6da601f4924e01af913a2e8aa882260a066c1f8f775b768b166.jpg](../icml_results/1086_SemanticNVS_%20Improving%20Semantic%20Scene%20Understanding%20in%20Generative%20Novel%20View%20Synthesis/tables/ee3ec1688c73e6da601f4924e01af913a2e8aa882260a066c1f8f775b768b166.jpg)

![f63f183c5a58654f239cafc4af8bc4a53591dc4cf6cca4c7830d2b9958bf8cef.jpg](../icml_results/1086_SemanticNVS_%20Improving%20Semantic%20Scene%20Understanding%20in%20Generative%20Novel%20View%20Synthesis/tables/f63f183c5a58654f239cafc4af8bc4a53591dc4cf6cca4c7830d2b9958bf8cef.jpg)

<a id="paper-1069"></a>

## Mitigating Reward Hacking in LLM-based Recommendation: A Preference Optimization Approach


### Images

![3c3a15dd2a7014ef502900f04e3dfcc6527f93c2d8840b73f10cb9c83755149d.jpg](../icml_results/1087_Mitigating%20Reward%20Hacking%20in%20LLM-based%20Recommendation_%20A%20Preference%20Optimization%20Approach/images/3c3a15dd2a7014ef502900f04e3dfcc6527f93c2d8840b73f10cb9c83755149d.jpg)

![59a4c59cb12c55c7712b7c588cb2f59e75175cee185fd2c52e2e7ebae65cdd8e.jpg](../icml_results/1087_Mitigating%20Reward%20Hacking%20in%20LLM-based%20Recommendation_%20A%20Preference%20Optimization%20Approach/images/59a4c59cb12c55c7712b7c588cb2f59e75175cee185fd2c52e2e7ebae65cdd8e.jpg)

![8e362c409c2fdc816a45b09d37bf0fc68204f84dbcfd704de2960b740c852d46.jpg](../icml_results/1087_Mitigating%20Reward%20Hacking%20in%20LLM-based%20Recommendation_%20A%20Preference%20Optimization%20Approach/images/8e362c409c2fdc816a45b09d37bf0fc68204f84dbcfd704de2960b740c852d46.jpg)

![b7245195038ef940727a01d6f750f0a84ee102f48696c9f92c5afcd28e06cdaa.jpg](../icml_results/1087_Mitigating%20Reward%20Hacking%20in%20LLM-based%20Recommendation_%20A%20Preference%20Optimization%20Approach/images/b7245195038ef940727a01d6f750f0a84ee102f48696c9f92c5afcd28e06cdaa.jpg)

![c458b220a326b102d2eff56d1d557c1308a936021cb96a72e36ee42120c29de3.jpg](../icml_results/1087_Mitigating%20Reward%20Hacking%20in%20LLM-based%20Recommendation_%20A%20Preference%20Optimization%20Approach/images/c458b220a326b102d2eff56d1d557c1308a936021cb96a72e36ee42120c29de3.jpg)

![cd1ac428a65081462139782ef2f83aea6143f56978246a67d78aa43d22261755.jpg](../icml_results/1087_Mitigating%20Reward%20Hacking%20in%20LLM-based%20Recommendation_%20A%20Preference%20Optimization%20Approach/images/cd1ac428a65081462139782ef2f83aea6143f56978246a67d78aa43d22261755.jpg)

### Tables

![16222ccf78c91828d47ecff0821062bcc1793cd03243a54574bca181f4705432.jpg](../icml_results/1087_Mitigating%20Reward%20Hacking%20in%20LLM-based%20Recommendation_%20A%20Preference%20Optimization%20Approach/tables/16222ccf78c91828d47ecff0821062bcc1793cd03243a54574bca181f4705432.jpg)

![95a44e65db6aaa2b93470a7434ec7794e9c8f05c31109345d2f8dd840e016acc.jpg](../icml_results/1087_Mitigating%20Reward%20Hacking%20in%20LLM-based%20Recommendation_%20A%20Preference%20Optimization%20Approach/tables/95a44e65db6aaa2b93470a7434ec7794e9c8f05c31109345d2f8dd840e016acc.jpg)

![986b6eb4e3f2bb05c946cb18bff1bd6eee4393078ed8e2e16b5b59dd4664c006.jpg](../icml_results/1087_Mitigating%20Reward%20Hacking%20in%20LLM-based%20Recommendation_%20A%20Preference%20Optimization%20Approach/tables/986b6eb4e3f2bb05c946cb18bff1bd6eee4393078ed8e2e16b5b59dd4664c006.jpg)

![b3818f367aefeef8b1283012f0f3cd744de6d7889bac65982237f8adc403757b.jpg](../icml_results/1087_Mitigating%20Reward%20Hacking%20in%20LLM-based%20Recommendation_%20A%20Preference%20Optimization%20Approach/tables/b3818f367aefeef8b1283012f0f3cd744de6d7889bac65982237f8adc403757b.jpg)

![b43ce35d56d61a7ee1bd3a01cdfa67cd488f2410a8dd8b99fe05f551aae442c2.jpg](../icml_results/1087_Mitigating%20Reward%20Hacking%20in%20LLM-based%20Recommendation_%20A%20Preference%20Optimization%20Approach/tables/b43ce35d56d61a7ee1bd3a01cdfa67cd488f2410a8dd8b99fe05f551aae442c2.jpg)

![dff4837e9c251f7e3c9f58e5ef1cf0d8e60d7c716c09730019e541cb292c21a4.jpg](../icml_results/1087_Mitigating%20Reward%20Hacking%20in%20LLM-based%20Recommendation_%20A%20Preference%20Optimization%20Approach/tables/dff4837e9c251f7e3c9f58e5ef1cf0d8e60d7c716c09730019e541cb292c21a4.jpg)

<a id="paper-1070"></a>

## INT vs. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats


### Images

![080925034f7d0978a729d1a5ea65c833e639f1c89060ff445fd5f40a0ec28522.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/images/080925034f7d0978a729d1a5ea65c833e639f1c89060ff445fd5f40a0ec28522.jpg)

![13f7d07763079727f67a0d53f8731905c30d1e5655438ff862d364a18c036341.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/images/13f7d07763079727f67a0d53f8731905c30d1e5655438ff862d364a18c036341.jpg)

![a12693911bde39251c62793c62e6c746372b1ab4b0eace97c4e516c8695ee8fc.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/images/a12693911bde39251c62793c62e6c746372b1ab4b0eace97c4e516c8695ee8fc.jpg)

![be996a4d04b6c5aad962fca34c8c46f7276587533e2429cfe0326b0d0e1063b5.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/images/be996a4d04b6c5aad962fca34c8c46f7276587533e2429cfe0326b0d0e1063b5.jpg)

![e8bbd3060000238f45726075a1f62784c83f360ba346308421d200c488664c7a.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/images/e8bbd3060000238f45726075a1f62784c83f360ba346308421d200c488664c7a.jpg)

### Tables

![02c06e1f0b18c493598e4f18151abc5383ace938399225b2b43ecad2f06206fa.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/tables/02c06e1f0b18c493598e4f18151abc5383ace938399225b2b43ecad2f06206fa.jpg)

![135d68370c128fe9f29502d754c9f8b62ca9380ab4a583a0b7bd6f565204361d.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/tables/135d68370c128fe9f29502d754c9f8b62ca9380ab4a583a0b7bd6f565204361d.jpg)

![1a2391c3d24109d5751c5df0dc69653ab5518759e15ca4d6ec6e47a6eb787ed0.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/tables/1a2391c3d24109d5751c5df0dc69653ab5518759e15ca4d6ec6e47a6eb787ed0.jpg)

![37495f5440fcedc458ce510e9d09283e93b8f969c5a687cfb1c2be93e3ebbe6f.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/tables/37495f5440fcedc458ce510e9d09283e93b8f969c5a687cfb1c2be93e3ebbe6f.jpg)

![3f952b1a005b68f117603a468b460906add176c97cffc2ae3b66fbe381a77b25.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/tables/3f952b1a005b68f117603a468b460906add176c97cffc2ae3b66fbe381a77b25.jpg)

![58c2a1d8b654e5e9572f3b0775703dfd22fddae6b3d2ab573daf7da520bc3575.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/tables/58c2a1d8b654e5e9572f3b0775703dfd22fddae6b3d2ab573daf7da520bc3575.jpg)

![5c82aff932dfcf830dd3a6b6aae1b15d7cdcfb4081faeb1339f43b5a28d528d1.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/tables/5c82aff932dfcf830dd3a6b6aae1b15d7cdcfb4081faeb1339f43b5a28d528d1.jpg)

![637d0827e2744e0aab96c3fe90abb4b610d18ef3e6d0fd963e07de536de0e28b.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/tables/637d0827e2744e0aab96c3fe90abb4b610d18ef3e6d0fd963e07de536de0e28b.jpg)

![76644c8e2e7336125fec76b6a95c2ef6167e75d5040c69480b48464ac70b6601.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/tables/76644c8e2e7336125fec76b6a95c2ef6167e75d5040c69480b48464ac70b6601.jpg)

![c7e7b3907ff3a9726c2828b77b5cf9b9be309f892c167e873eccfd0bb7be9e07.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/tables/c7e7b3907ff3a9726c2828b77b5cf9b9be309f892c167e873eccfd0bb7be9e07.jpg)

![c8352227b4fdbe126b9c5042bdf1f72388fa3624502cbf9b4a09ae78928687e4.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/tables/c8352227b4fdbe126b9c5042bdf1f72388fa3624502cbf9b4a09ae78928687e4.jpg)

![cbaf47ce3ae22c49eed1e97f78406b169d5fcd2b0c0bea556b2d82835b95ea06.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/tables/cbaf47ce3ae22c49eed1e97f78406b169d5fcd2b0c0bea556b2d82835b95ea06.jpg)

![dfb61ef4d2d00ec9170863115bc0aac7a18d9f044a0e8c730527dbed2ab6f40a.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/tables/dfb61ef4d2d00ec9170863115bc0aac7a18d9f044a0e8c730527dbed2ab6f40a.jpg)

![ec55bd392075fa1def7cd4d9f6f0befbfa3f0c012fc43cbc0253aa0e1ab742d6.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/tables/ec55bd392075fa1def7cd4d9f6f0befbfa3f0c012fc43cbc0253aa0e1ab742d6.jpg)

![fc4f531c904e35c1c17d453fe827f843d3a777d82545b18f112c1e502cbeaf72.jpg](../icml_results/1088_INT%20vs.%20FP_%20A%20Comprehensive%20Study%20of%20Fine-Grained%20Low-bit%20Quantization%20Formats/tables/fc4f531c904e35c1c17d453fe827f843d3a777d82545b18f112c1e502cbeaf72.jpg)

<a id="paper-1071"></a>

## Riemannian Networks over Full-Rank Correlation Matrices


### Images

![48a994f6cc3e7f935d81224658a6026cc12f711343bfcffe382def78b4e1b187.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/images/48a994f6cc3e7f935d81224658a6026cc12f711343bfcffe382def78b4e1b187.jpg)

![59aee5906b57536652edd585ef91cffc5327090c29d9b6362873ac28dbbb04de.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/images/59aee5906b57536652edd585ef91cffc5327090c29d9b6362873ac28dbbb04de.jpg)

![5b130c0374b6dc8c7e4e058d9cf31ecb5bed59f433c47d1b07c3833079426820.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/images/5b130c0374b6dc8c7e4e058d9cf31ecb5bed59f433c47d1b07c3833079426820.jpg)

![80147fd203fbc836148374a9ca22620f1361e4bc943e46218530a0165b322989.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/images/80147fd203fbc836148374a9ca22620f1361e4bc943e46218530a0165b322989.jpg)

![95f2f25c791a552d18ac9ef82b84caac5ba7bd60da8ba0dfbc5a54cf07d4739e.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/images/95f2f25c791a552d18ac9ef82b84caac5ba7bd60da8ba0dfbc5a54cf07d4739e.jpg)

![ac3d545e16b42eac623dfca864794c3b5b3b2c05408c04826b810724e6cc6d65.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/images/ac3d545e16b42eac623dfca864794c3b5b3b2c05408c04826b810724e6cc6d65.jpg)

![b594f745b5c4daeb1f6048b6e99c5f7fe5cee874a406ed0d1a5cbfb25e814e27.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/images/b594f745b5c4daeb1f6048b6e99c5f7fe5cee874a406ed0d1a5cbfb25e814e27.jpg)

![c3ecdb77ee3141ff9a33d779191efb4c6d6c2b7f6fb8c04024d0c02ac0088114.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/images/c3ecdb77ee3141ff9a33d779191efb4c6d6c2b7f6fb8c04024d0c02ac0088114.jpg)

![d3c277a080000bc780d031434017206cf692d283aa36e7eb22447eab5c725082.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/images/d3c277a080000bc780d031434017206cf692d283aa36e7eb22447eab5c725082.jpg)

### Tables

![0012a507e641f204f739847354be1faf64fd0e27cfd686ae673b7c0a896ca8d0.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/0012a507e641f204f739847354be1faf64fd0e27cfd686ae673b7c0a896ca8d0.jpg)

![00e865856dded877d0be4181478e8d1ecd15d9383a75305690981e67a247259f.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/00e865856dded877d0be4181478e8d1ecd15d9383a75305690981e67a247259f.jpg)

![086a4397471f795c96fc2d69e6088fa6581dcfdba6eaec7d3aa0ce1197ef474d.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/086a4397471f795c96fc2d69e6088fa6581dcfdba6eaec7d3aa0ce1197ef474d.jpg)

![2245da3a675395337081414097e602e926da377487fb8436e292f0e09119aa77.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/2245da3a675395337081414097e602e926da377487fb8436e292f0e09119aa77.jpg)

![2fc34a09d7c6a3fbaf3e04dabc09ab7ccf809a8a74f6c960b4b3a2a7a2dbe35a.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/2fc34a09d7c6a3fbaf3e04dabc09ab7ccf809a8a74f6c960b4b3a2a7a2dbe35a.jpg)

![3eef5a302bcf7e8b9edd3be66e283d10265e0741cd1fd34ddd34d12fe1a8b5ac.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/3eef5a302bcf7e8b9edd3be66e283d10265e0741cd1fd34ddd34d12fe1a8b5ac.jpg)

![4c82660199fdc6cf835734f58b90f4b4f45520b73a3bb80db51c09645cf6970b.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/4c82660199fdc6cf835734f58b90f4b4f45520b73a3bb80db51c09645cf6970b.jpg)

![555e8895fb354404841af79aa70b5e6e0e19b71b2854908ebacda1e1e9d5fe54.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/555e8895fb354404841af79aa70b5e6e0e19b71b2854908ebacda1e1e9d5fe54.jpg)

![59bbd63d00381b3999fb63fcd4647ab7725e3abb09788ff854f53a010abb7dc4.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/59bbd63d00381b3999fb63fcd4647ab7725e3abb09788ff854f53a010abb7dc4.jpg)

![648f68b5f674163210fe0034897aafcaa187802a64ce8f4bddab1a880afcef96.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/648f68b5f674163210fe0034897aafcaa187802a64ce8f4bddab1a880afcef96.jpg)

![79cb4d63ad0b67bac9a22e59911daf124bc47427c7ee6ae31d17538a9f301d99.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/79cb4d63ad0b67bac9a22e59911daf124bc47427c7ee6ae31d17538a9f301d99.jpg)

![960bf2e550fb460864738feafa7b2868f31f237df8e608600ef94eb8ead6676a.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/960bf2e550fb460864738feafa7b2868f31f237df8e608600ef94eb8ead6676a.jpg)

![c14d4d6364a0e6d2af4875f485543545851499a8657f6a62d53140d483eb625f.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/c14d4d6364a0e6d2af4875f485543545851499a8657f6a62d53140d483eb625f.jpg)

![d3f5a67c1af2c56c4fbf3c172cabf8b0e5b5c6fcbc1908ecc53fb886fd949a12.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/d3f5a67c1af2c56c4fbf3c172cabf8b0e5b5c6fcbc1908ecc53fb886fd949a12.jpg)

![da214d8e3935626aba76c3893221984d0db62b3f95949535aeeccf2f0f47d02f.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/da214d8e3935626aba76c3893221984d0db62b3f95949535aeeccf2f0f47d02f.jpg)

![e558ca48b6d9b0433b2332226d64768397eab0bcabf64661d358e1b578213e82.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/e558ca48b6d9b0433b2332226d64768397eab0bcabf64661d358e1b578213e82.jpg)

![e5cc44aa81c9c67887530447f1ed81cec9b6e33c74c995a0540742d9f104e561.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/e5cc44aa81c9c67887530447f1ed81cec9b6e33c74c995a0540742d9f104e561.jpg)

![e8064aa3679708385d8694c5dc8ae0c6a05a3138a2be1724ddcbf18524f35457.jpg](../icml_results/1089_Riemannian%20Networks%20over%20Full-Rank%20Correlation%20Matrices/tables/e8064aa3679708385d8694c5dc8ae0c6a05a3138a2be1724ddcbf18524f35457.jpg)

<a id="paper-1072"></a>

## ScalingAR: Scaling Confidence for Autoregressive Image Generation


### Images

![1db3f96d52968989a9bc39a6106d70cbca9721c57e8d1ab9e7123a57264b5729.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/images/1db3f96d52968989a9bc39a6106d70cbca9721c57e8d1ab9e7123a57264b5729.jpg)

![42238fe7c77c07a2657c369ee1435fa721963b68321909ea6c61798cd8128b11.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/images/42238fe7c77c07a2657c369ee1435fa721963b68321909ea6c61798cd8128b11.jpg)

![5b3c4930a288598efda515a79fc4957825bfe14946bf5a133ab1c2509617e03f.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/images/5b3c4930a288598efda515a79fc4957825bfe14946bf5a133ab1c2509617e03f.jpg)

![6177d40b4eab357b382f6b7b949ec9fbadc551fdf105e8cec4f7d26f27ee0467.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/images/6177d40b4eab357b382f6b7b949ec9fbadc551fdf105e8cec4f7d26f27ee0467.jpg)

![63bf06f190ac1c537ed02e7217db4e2a7057980b9a22302f3fcb3baccae91a27.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/images/63bf06f190ac1c537ed02e7217db4e2a7057980b9a22302f3fcb3baccae91a27.jpg)

![6b1ccc6d8c9e69e14ee0e815f01102d7b07f8dd863e7cc3e41915fd2c25ad2b0.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/images/6b1ccc6d8c9e69e14ee0e815f01102d7b07f8dd863e7cc3e41915fd2c25ad2b0.jpg)

![7803767071fd60c770b4fb964748add6f7ba91aa6e1b4035a91512b8a6235008.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/images/7803767071fd60c770b4fb964748add6f7ba91aa6e1b4035a91512b8a6235008.jpg)

![8c36b3dbdbf1b6a0b45cae30c03e58d012c3fdb84ded031a241173c17dd514e3.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/images/8c36b3dbdbf1b6a0b45cae30c03e58d012c3fdb84ded031a241173c17dd514e3.jpg)

![8db4f27bc26c6678977754112b7e254b62c593a5c9c0b93541843af87eb03493.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/images/8db4f27bc26c6678977754112b7e254b62c593a5c9c0b93541843af87eb03493.jpg)

![c5ea0fbe20d9b1c8375d3e447ed166b5227bed05c12252c52c4e009248a914bc.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/images/c5ea0fbe20d9b1c8375d3e447ed166b5227bed05c12252c52c4e009248a914bc.jpg)

![dc986391bbf61b5d2925a07067d5ac5cd6f4cd57ae5e9966bd3b6eb177821aff.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/images/dc986391bbf61b5d2925a07067d5ac5cd6f4cd57ae5e9966bd3b6eb177821aff.jpg)

![f6bfe981d9b85239736cd5be605e3817f63cf8d9a1211bbf5d75dc3e5e5743d9.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/images/f6bfe981d9b85239736cd5be605e3817f63cf8d9a1211bbf5d75dc3e5e5743d9.jpg)

![f7c200e9e1c808378f6572f551dc7d5791523f8ba5f58e323db2bc7d15dd5fd1.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/images/f7c200e9e1c808378f6572f551dc7d5791523f8ba5f58e323db2bc7d15dd5fd1.jpg)

### Tables

![3daa4af09066665bbf9b25537d0fcdc4cca752687721e33c2bf16ef5b490c988.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/tables/3daa4af09066665bbf9b25537d0fcdc4cca752687721e33c2bf16ef5b490c988.jpg)

![5d4b5d9a1752bf6f3b62ac9e013b2483d23f05e79a80714c92c2727b2f2758e9.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/tables/5d4b5d9a1752bf6f3b62ac9e013b2483d23f05e79a80714c92c2727b2f2758e9.jpg)

![7bd4dd11f8cc36551dbe1750d8a012672e73a2e2231cf03fad12bfb9d9ea8d30.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/tables/7bd4dd11f8cc36551dbe1750d8a012672e73a2e2231cf03fad12bfb9d9ea8d30.jpg)

![7cb9fdde9465521389cdcdc6944f0c28d7563c4fba66defabf4a68d9fbbcde90.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/tables/7cb9fdde9465521389cdcdc6944f0c28d7563c4fba66defabf4a68d9fbbcde90.jpg)

![9bce4964c48894eb2eca53b3d5f43fbfb5547d7e72de2a93b384a3cc71abce0b.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/tables/9bce4964c48894eb2eca53b3d5f43fbfb5547d7e72de2a93b384a3cc71abce0b.jpg)

![eb126c94c7c2855bd4c9496f3dcb988b067305671ffeb255c4289760e0521b95.jpg](../icml_results/1090_ScalingAR_%20Scaling%20Confidence%20for%20Autoregressive%20Image%20Generation/tables/eb126c94c7c2855bd4c9496f3dcb988b067305671ffeb255c4289760e0521b95.jpg)

<a id="paper-1073"></a>

## HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation


### Images

![22f82bc71c7ef9909627f4e7014e3e2359efdfdb16b3869abdd4dac88b16955b.jpg](../icml_results/1091_HO-SFL_%20Hybrid-Order%20Split%20Federated%20Learning%20with%20Backprop-Free%20Clients%20and%20Dimension-Free%20Aggregat/images/22f82bc71c7ef9909627f4e7014e3e2359efdfdb16b3869abdd4dac88b16955b.jpg)

![3d23b2f0eb779bc981d05c5ae789cf74d7325b069ed126a5bf2f55fbfbbbe0fc.jpg](../icml_results/1091_HO-SFL_%20Hybrid-Order%20Split%20Federated%20Learning%20with%20Backprop-Free%20Clients%20and%20Dimension-Free%20Aggregat/images/3d23b2f0eb779bc981d05c5ae789cf74d7325b069ed126a5bf2f55fbfbbbe0fc.jpg)

![50aa1d51de72993117fe7101e99eeb2e738af11229afc3507730ea9ef538c3fc.jpg](../icml_results/1091_HO-SFL_%20Hybrid-Order%20Split%20Federated%20Learning%20with%20Backprop-Free%20Clients%20and%20Dimension-Free%20Aggregat/images/50aa1d51de72993117fe7101e99eeb2e738af11229afc3507730ea9ef538c3fc.jpg)

![5c2929d1452a3b020717781369613080fcf27f2066608bef383b01dbe13ae8c1.jpg](../icml_results/1091_HO-SFL_%20Hybrid-Order%20Split%20Federated%20Learning%20with%20Backprop-Free%20Clients%20and%20Dimension-Free%20Aggregat/images/5c2929d1452a3b020717781369613080fcf27f2066608bef383b01dbe13ae8c1.jpg)

![6f623710c5c94f8866945b4e8ce2a342ee761930ca10ed8fd02382f6b98db9fd.jpg](../icml_results/1091_HO-SFL_%20Hybrid-Order%20Split%20Federated%20Learning%20with%20Backprop-Free%20Clients%20and%20Dimension-Free%20Aggregat/images/6f623710c5c94f8866945b4e8ce2a342ee761930ca10ed8fd02382f6b98db9fd.jpg)

![8010aa285fc7bffeee2d6605a3349d8f4c300f2b437951ec6cbc0988340219d5.jpg](../icml_results/1091_HO-SFL_%20Hybrid-Order%20Split%20Federated%20Learning%20with%20Backprop-Free%20Clients%20and%20Dimension-Free%20Aggregat/images/8010aa285fc7bffeee2d6605a3349d8f4c300f2b437951ec6cbc0988340219d5.jpg)

![ac359696b2ebd62ce20f78eecf16a33c37fdfe4808e50b76cc20d853f0c2a0be.jpg](../icml_results/1091_HO-SFL_%20Hybrid-Order%20Split%20Federated%20Learning%20with%20Backprop-Free%20Clients%20and%20Dimension-Free%20Aggregat/images/ac359696b2ebd62ce20f78eecf16a33c37fdfe4808e50b76cc20d853f0c2a0be.jpg)

![bebaf2aef7e82718267ad17d8258613bc30a3fd9d3a8387278dd6f1a05ae27c9.jpg](../icml_results/1091_HO-SFL_%20Hybrid-Order%20Split%20Federated%20Learning%20with%20Backprop-Free%20Clients%20and%20Dimension-Free%20Aggregat/images/bebaf2aef7e82718267ad17d8258613bc30a3fd9d3a8387278dd6f1a05ae27c9.jpg)

### Tables

![0184fd2b7aa5e0bae472ea593eed49694d3fe07efc58ea8cca320baa03c2057e.jpg](../icml_results/1091_HO-SFL_%20Hybrid-Order%20Split%20Federated%20Learning%20with%20Backprop-Free%20Clients%20and%20Dimension-Free%20Aggregat/tables/0184fd2b7aa5e0bae472ea593eed49694d3fe07efc58ea8cca320baa03c2057e.jpg)

![1c32380d6ab4a3de8ffa12daef9522260c45acd15f56cb73bf9c21e8e795fd8a.jpg](../icml_results/1091_HO-SFL_%20Hybrid-Order%20Split%20Federated%20Learning%20with%20Backprop-Free%20Clients%20and%20Dimension-Free%20Aggregat/tables/1c32380d6ab4a3de8ffa12daef9522260c45acd15f56cb73bf9c21e8e795fd8a.jpg)

![1f74f0d325f0f4007fda24726295e2a870bd71ab69dde94e565ba2320e22b88c.jpg](../icml_results/1091_HO-SFL_%20Hybrid-Order%20Split%20Federated%20Learning%20with%20Backprop-Free%20Clients%20and%20Dimension-Free%20Aggregat/tables/1f74f0d325f0f4007fda24726295e2a870bd71ab69dde94e565ba2320e22b88c.jpg)

![422b122c4e47901348c25a55c18ec0999df6a2cbef9702af8b2fe05f7bdf8dfc.jpg](../icml_results/1091_HO-SFL_%20Hybrid-Order%20Split%20Federated%20Learning%20with%20Backprop-Free%20Clients%20and%20Dimension-Free%20Aggregat/tables/422b122c4e47901348c25a55c18ec0999df6a2cbef9702af8b2fe05f7bdf8dfc.jpg)

<a id="paper-1074"></a>

## Towards Disentangled Preference Optimization Dynamics: Suppress the Loser, Preserve the Winner


### Images

![03f56267500b95130d0197d499ec9b0629f4abfc13daacf5169aba862faf402d.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/images/03f56267500b95130d0197d499ec9b0629f4abfc13daacf5169aba862faf402d.jpg)

![0e14ee0a7e34d364f70d6122877fd02e28553d852050e2be0947f4feed1f7630.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/images/0e14ee0a7e34d364f70d6122877fd02e28553d852050e2be0947f4feed1f7630.jpg)

![1420bf1e95bcdac2d7919ebfe163fc3b02677a971f1c2fb99608e31a74ac16c9.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/images/1420bf1e95bcdac2d7919ebfe163fc3b02677a971f1c2fb99608e31a74ac16c9.jpg)

![291c04546f4d4b7328d4b0c53f45ee4451b411e25089e33a916c1ae52b97b52b.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/images/291c04546f4d4b7328d4b0c53f45ee4451b411e25089e33a916c1ae52b97b52b.jpg)

![3c6a953cdead48871e83860e4fe50d46cd67e162c25c7295602710d30e204d1c.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/images/3c6a953cdead48871e83860e4fe50d46cd67e162c25c7295602710d30e204d1c.jpg)

![3cc444dd89ba968aeec6668fb4d469a768bb673299188fdbb735db0334e25a5a.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/images/3cc444dd89ba968aeec6668fb4d469a768bb673299188fdbb735db0334e25a5a.jpg)

![3d5c365c46a281a6a1e4cc160f905eb678a1bcb5716c77df38ec6e3780b33c35.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/images/3d5c365c46a281a6a1e4cc160f905eb678a1bcb5716c77df38ec6e3780b33c35.jpg)

![5df73bbfd6de6cac34777fa540803b9e2d22801b556d33bc1e957f0cd5d547e5.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/images/5df73bbfd6de6cac34777fa540803b9e2d22801b556d33bc1e957f0cd5d547e5.jpg)

![5e8bf4a9df35548695cb45eb592d6414d3d1c12d5858f30b65b5aa2b160ef3cd.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/images/5e8bf4a9df35548695cb45eb592d6414d3d1c12d5858f30b65b5aa2b160ef3cd.jpg)

![8879cc415b20048c779df67eae1b6ef3bc656feac2971ea8f06db6fe25697410.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/images/8879cc415b20048c779df67eae1b6ef3bc656feac2971ea8f06db6fe25697410.jpg)

![cdd2b556c62950483616ae3733b9843063d89d37305d95cbd2701e452e814428.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/images/cdd2b556c62950483616ae3733b9843063d89d37305d95cbd2701e452e814428.jpg)

![d3fd620ab04937745710fa7d8332cedc19aa318ea1f8985df21bd1c0e44f6663.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/images/d3fd620ab04937745710fa7d8332cedc19aa318ea1f8985df21bd1c0e44f6663.jpg)

![e6b1e220c1e485d9b04624cc70de06bb876b47989ac560aadba436d94a44430b.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/images/e6b1e220c1e485d9b04624cc70de06bb876b47989ac560aadba436d94a44430b.jpg)

![f636bdd901124197697d6ecda56c2b1bd8aa8c2f69b51d44516fcdb57b49d241.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/images/f636bdd901124197697d6ecda56c2b1bd8aa8c2f69b51d44516fcdb57b49d241.jpg)

### Tables

![08c6abdbb58d460bac9520fad9ab58de6d0c1c0331c8ce5e8ea5210321eb0808.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/tables/08c6abdbb58d460bac9520fad9ab58de6d0c1c0331c8ce5e8ea5210321eb0808.jpg)

![2b3c5c0da308fbd74c42f438125f0ba589663515c2b24ba0fb5275e409378a41.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/tables/2b3c5c0da308fbd74c42f438125f0ba589663515c2b24ba0fb5275e409378a41.jpg)

![37562d4505fe4b5ea0012cf73c2da174ab943d1878a3754aaea43f91d70f9e04.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/tables/37562d4505fe4b5ea0012cf73c2da174ab943d1878a3754aaea43f91d70f9e04.jpg)

![54cdd19864bce43e131798479a66e8f90ee93d97d8c66472c1c21cb1f231eb3e.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/tables/54cdd19864bce43e131798479a66e8f90ee93d97d8c66472c1c21cb1f231eb3e.jpg)

![603dab628deaf9d19aa08314ae2880b6563696804d2700e322e6065ad1d1e20b.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/tables/603dab628deaf9d19aa08314ae2880b6563696804d2700e322e6065ad1d1e20b.jpg)

![98eb7c6215c7b42d102370b881b140eb51d6afbb240b82cdc1d49aa00b76b2e8.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/tables/98eb7c6215c7b42d102370b881b140eb51d6afbb240b82cdc1d49aa00b76b2e8.jpg)

![da9e5402a0feb25d56ba3dbb0b17d203b8ba57dc92f947175f01addfba991842.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/tables/da9e5402a0feb25d56ba3dbb0b17d203b8ba57dc92f947175f01addfba991842.jpg)

![e9d8602e793f440ecd90128ca225a6ab2f95e9c1338eee79cf72a5b4f1795f01.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/tables/e9d8602e793f440ecd90128ca225a6ab2f95e9c1338eee79cf72a5b4f1795f01.jpg)

![f9b7470dcee9fb04c63777e7990d55ddf65a2ea6a941c2780401d30390648d69.jpg](../icml_results/1092_Towards%20Disentangled%20Preference%20Optimization%20Dynamics_%20Suppress%20the%20Loser%2C%20Preserve%20the%20Winner/tables/f9b7470dcee9fb04c63777e7990d55ddf65a2ea6a941c2780401d30390648d69.jpg)

<a id="paper-1075"></a>

## AMDP: Asynchronous Multi-Directional Pipeline Parallelism for Large-Scale Models Training


### Images

![1be47ba87c836df6105aac60771cc4367936ccbcbb7ee8e7e34ba0a3327a4462.jpg](../icml_results/1093_AMDP_%20Asynchronous%20Multi-Directional%20Pipeline%20Parallelism%20for%20Large-Scale%20Models%20Training/images/1be47ba87c836df6105aac60771cc4367936ccbcbb7ee8e7e34ba0a3327a4462.jpg)

![26db7f1269d2ad15aa068f7d2a2ff3961465a5e8bdf119e241b15b71bf98fc4e.jpg](../icml_results/1093_AMDP_%20Asynchronous%20Multi-Directional%20Pipeline%20Parallelism%20for%20Large-Scale%20Models%20Training/images/26db7f1269d2ad15aa068f7d2a2ff3961465a5e8bdf119e241b15b71bf98fc4e.jpg)

![325aee0c5550b3cb79abfce8be88c72a03151f790b7d9a63103d7c3fbebb584d.jpg](../icml_results/1093_AMDP_%20Asynchronous%20Multi-Directional%20Pipeline%20Parallelism%20for%20Large-Scale%20Models%20Training/images/325aee0c5550b3cb79abfce8be88c72a03151f790b7d9a63103d7c3fbebb584d.jpg)

![98b54a49fe2e544e1964043142a4f0d7aeb17af68eac68d7f49c6261e67332f8.jpg](../icml_results/1093_AMDP_%20Asynchronous%20Multi-Directional%20Pipeline%20Parallelism%20for%20Large-Scale%20Models%20Training/images/98b54a49fe2e544e1964043142a4f0d7aeb17af68eac68d7f49c6261e67332f8.jpg)

![c54658b144d2c574850e42032b2120b829abe46a99315d5b4bc980b36dc43af5.jpg](../icml_results/1093_AMDP_%20Asynchronous%20Multi-Directional%20Pipeline%20Parallelism%20for%20Large-Scale%20Models%20Training/images/c54658b144d2c574850e42032b2120b829abe46a99315d5b4bc980b36dc43af5.jpg)

![ccf6847df3f3e108e57d5140e8f1c49e04448aed7430ba987f6854d1e566e983.jpg](../icml_results/1093_AMDP_%20Asynchronous%20Multi-Directional%20Pipeline%20Parallelism%20for%20Large-Scale%20Models%20Training/images/ccf6847df3f3e108e57d5140e8f1c49e04448aed7430ba987f6854d1e566e983.jpg)

![d9ab277872cc7dfa598e4d94f8124998c7dbf522b3ea875f1f1e84109c7335cf.jpg](../icml_results/1093_AMDP_%20Asynchronous%20Multi-Directional%20Pipeline%20Parallelism%20for%20Large-Scale%20Models%20Training/images/d9ab277872cc7dfa598e4d94f8124998c7dbf522b3ea875f1f1e84109c7335cf.jpg)

![e1219bc569b4a76bc2819dc05a48ac06d578bd3da731d8ce9516a76f90760872.jpg](../icml_results/1093_AMDP_%20Asynchronous%20Multi-Directional%20Pipeline%20Parallelism%20for%20Large-Scale%20Models%20Training/images/e1219bc569b4a76bc2819dc05a48ac06d578bd3da731d8ce9516a76f90760872.jpg)

![fa4d9e817dcb22e9adaa2b0c58a94744e1207a0b9da31364f8c455a3dfb9bcf9.jpg](../icml_results/1093_AMDP_%20Asynchronous%20Multi-Directional%20Pipeline%20Parallelism%20for%20Large-Scale%20Models%20Training/images/fa4d9e817dcb22e9adaa2b0c58a94744e1207a0b9da31364f8c455a3dfb9bcf9.jpg)

### Tables

![0015d9deda91e38a368205a491ca8cc57a481bee47143fa75555baa1c3720a83.jpg](../icml_results/1093_AMDP_%20Asynchronous%20Multi-Directional%20Pipeline%20Parallelism%20for%20Large-Scale%20Models%20Training/tables/0015d9deda91e38a368205a491ca8cc57a481bee47143fa75555baa1c3720a83.jpg)

![0af503c592eec89bbd2119ce6f5298e87b3e25cc5034661cb96a59fc04141bef.jpg](../icml_results/1093_AMDP_%20Asynchronous%20Multi-Directional%20Pipeline%20Parallelism%20for%20Large-Scale%20Models%20Training/tables/0af503c592eec89bbd2119ce6f5298e87b3e25cc5034661cb96a59fc04141bef.jpg)

![9ef1bdaa823d998f37fea671c91a8fa4eedaffdd6e470258b05bf7fd1271d036.jpg](../icml_results/1093_AMDP_%20Asynchronous%20Multi-Directional%20Pipeline%20Parallelism%20for%20Large-Scale%20Models%20Training/tables/9ef1bdaa823d998f37fea671c91a8fa4eedaffdd6e470258b05bf7fd1271d036.jpg)

![a21bf245ffea04b8cc76f47ec13de61045aab9b4d8c91a4d1c1d591a35d63bb4.jpg](../icml_results/1093_AMDP_%20Asynchronous%20Multi-Directional%20Pipeline%20Parallelism%20for%20Large-Scale%20Models%20Training/tables/a21bf245ffea04b8cc76f47ec13de61045aab9b4d8c91a4d1c1d591a35d63bb4.jpg)

![b94a6fd03d11274388a1117e8363eaad5cf271810646aabd5f5c9abb474a0259.jpg](../icml_results/1093_AMDP_%20Asynchronous%20Multi-Directional%20Pipeline%20Parallelism%20for%20Large-Scale%20Models%20Training/tables/b94a6fd03d11274388a1117e8363eaad5cf271810646aabd5f5c9abb474a0259.jpg)

![c6aa63c9889feafe429ebd1d7758cf2bed60847d1815cc52bbf02c5b7f0d2206.jpg](../icml_results/1093_AMDP_%20Asynchronous%20Multi-Directional%20Pipeline%20Parallelism%20for%20Large-Scale%20Models%20Training/tables/c6aa63c9889feafe429ebd1d7758cf2bed60847d1815cc52bbf02c5b7f0d2206.jpg)

![e901110c5721a723720c1d7bcc56cc20601fc18c8bf6eafe2a00ce69eb5e4c12.jpg](../icml_results/1093_AMDP_%20Asynchronous%20Multi-Directional%20Pipeline%20Parallelism%20for%20Large-Scale%20Models%20Training/tables/e901110c5721a723720c1d7bcc56cc20601fc18c8bf6eafe2a00ce69eb5e4c12.jpg)

![eba8dd382f79a498eb29ab03365957badc834602857f7212487c62a0b9444196.jpg](../icml_results/1093_AMDP_%20Asynchronous%20Multi-Directional%20Pipeline%20Parallelism%20for%20Large-Scale%20Models%20Training/tables/eba8dd382f79a498eb29ab03365957badc834602857f7212487c62a0b9444196.jpg)

<a id="paper-1076"></a>

## General Quantification of Covariate and Concept Shifts


### Images

![16ad06b8e9324dc4839aa226f541a868f9b754ed7edd4deb195169d2aeee4af3.jpg](../icml_results/1094_General%20Quantification%20of%20Covariate%20and%20Concept%20Shifts/images/16ad06b8e9324dc4839aa226f541a868f9b754ed7edd4deb195169d2aeee4af3.jpg)

![30a3f8a0d2981ceb9856ebeab43424fbd52e07d182276e25ce995bf905e0855e.jpg](../icml_results/1094_General%20Quantification%20of%20Covariate%20and%20Concept%20Shifts/images/30a3f8a0d2981ceb9856ebeab43424fbd52e07d182276e25ce995bf905e0855e.jpg)

![cd6e8a552b133345e52234e7aa4873659b7c99262f402bf09c6f301fe8b26db3.jpg](../icml_results/1094_General%20Quantification%20of%20Covariate%20and%20Concept%20Shifts/images/cd6e8a552b133345e52234e7aa4873659b7c99262f402bf09c6f301fe8b26db3.jpg)

### Tables

![eb55b863067fd55eb93980d3a75bdfac47bb32565057571bae081cfb697a6c6b.jpg](../icml_results/1094_General%20Quantification%20of%20Covariate%20and%20Concept%20Shifts/tables/eb55b863067fd55eb93980d3a75bdfac47bb32565057571bae081cfb697a6c6b.jpg)

<a id="paper-1077"></a>

## TrustworthyQENN: A Quantum Evidential Neural Network Based on Complex-Valued Contrastive Learning for Uncertainty Pattern Classification


### Images

![299814f2d020cf90d53a4f0d8c959f9142cc913fe90c58c8caa0e14c58b80950.jpg](../icml_results/1095_TrustworthyQENN_%20A%20Quantum%20Evidential%20Neural%20Network%20Based%20on%20Complex-Valued%20Contrastive%20Learning%20fo/images/299814f2d020cf90d53a4f0d8c959f9142cc913fe90c58c8caa0e14c58b80950.jpg)

![3c6a0d0f3b0b50ec9fcb28035e43b20cf7ece5b0f3d7486860fc3d3adefe4b93.jpg](../icml_results/1095_TrustworthyQENN_%20A%20Quantum%20Evidential%20Neural%20Network%20Based%20on%20Complex-Valued%20Contrastive%20Learning%20fo/images/3c6a0d0f3b0b50ec9fcb28035e43b20cf7ece5b0f3d7486860fc3d3adefe4b93.jpg)

![45e9bb120b58b7120467b8e8b217c0bce7e79e95297684cb3030fb1d11f35f69.jpg](../icml_results/1095_TrustworthyQENN_%20A%20Quantum%20Evidential%20Neural%20Network%20Based%20on%20Complex-Valued%20Contrastive%20Learning%20fo/images/45e9bb120b58b7120467b8e8b217c0bce7e79e95297684cb3030fb1d11f35f69.jpg)

![5a98401eb9b5cf267dd322b28b6ef26b76450da2a403079c9a05a6576a6ca91c.jpg](../icml_results/1095_TrustworthyQENN_%20A%20Quantum%20Evidential%20Neural%20Network%20Based%20on%20Complex-Valued%20Contrastive%20Learning%20fo/images/5a98401eb9b5cf267dd322b28b6ef26b76450da2a403079c9a05a6576a6ca91c.jpg)

![9cb03490156a106525bb1773bd7470d2993406f021a6ba57cd388f11bea37356.jpg](../icml_results/1095_TrustworthyQENN_%20A%20Quantum%20Evidential%20Neural%20Network%20Based%20on%20Complex-Valued%20Contrastive%20Learning%20fo/images/9cb03490156a106525bb1773bd7470d2993406f021a6ba57cd388f11bea37356.jpg)

![b0ea7b1d29b0071c61d8cf36a8358e28a1ff5c2aaefbae01b1d44cfde9d9003a.jpg](../icml_results/1095_TrustworthyQENN_%20A%20Quantum%20Evidential%20Neural%20Network%20Based%20on%20Complex-Valued%20Contrastive%20Learning%20fo/images/b0ea7b1d29b0071c61d8cf36a8358e28a1ff5c2aaefbae01b1d44cfde9d9003a.jpg)

![c115c86d9510e60c04405b5dd4d38993c3010c425f6d18d13e73c12ba704158a.jpg](../icml_results/1095_TrustworthyQENN_%20A%20Quantum%20Evidential%20Neural%20Network%20Based%20on%20Complex-Valued%20Contrastive%20Learning%20fo/images/c115c86d9510e60c04405b5dd4d38993c3010c425f6d18d13e73c12ba704158a.jpg)

![d5ad0b7ba267a9fa6ba8b0243bf1d292673e9173f3648acc5ab9b1b3778da360.jpg](../icml_results/1095_TrustworthyQENN_%20A%20Quantum%20Evidential%20Neural%20Network%20Based%20on%20Complex-Valued%20Contrastive%20Learning%20fo/images/d5ad0b7ba267a9fa6ba8b0243bf1d292673e9173f3648acc5ab9b1b3778da360.jpg)

### Tables

![156a571f571a4711e52c52f16ab5377f6ded25bcf4553c356360c48346ab2f28.jpg](../icml_results/1095_TrustworthyQENN_%20A%20Quantum%20Evidential%20Neural%20Network%20Based%20on%20Complex-Valued%20Contrastive%20Learning%20fo/tables/156a571f571a4711e52c52f16ab5377f6ded25bcf4553c356360c48346ab2f28.jpg)

![5e67042a41d3bba9d7c3519b5c68d60ca07307e9c36476d17c482c210e5558b3.jpg](../icml_results/1095_TrustworthyQENN_%20A%20Quantum%20Evidential%20Neural%20Network%20Based%20on%20Complex-Valued%20Contrastive%20Learning%20fo/tables/5e67042a41d3bba9d7c3519b5c68d60ca07307e9c36476d17c482c210e5558b3.jpg)

![7b0c4d81fa2e572542fdf0ce0e2966272436d767b65fe47104dcf202686d5c02.jpg](../icml_results/1095_TrustworthyQENN_%20A%20Quantum%20Evidential%20Neural%20Network%20Based%20on%20Complex-Valued%20Contrastive%20Learning%20fo/tables/7b0c4d81fa2e572542fdf0ce0e2966272436d767b65fe47104dcf202686d5c02.jpg)

![8fb7ea7f6a61e3ec77bea5eb5e1faea1e6f25579f01b10be9a5e131f54cd9e03.jpg](../icml_results/1095_TrustworthyQENN_%20A%20Quantum%20Evidential%20Neural%20Network%20Based%20on%20Complex-Valued%20Contrastive%20Learning%20fo/tables/8fb7ea7f6a61e3ec77bea5eb5e1faea1e6f25579f01b10be9a5e131f54cd9e03.jpg)

![9de1bcfad639e70ee8ac4600bbe616d49b80d07ad71e016cd58bfcf89182f4dc.jpg](../icml_results/1095_TrustworthyQENN_%20A%20Quantum%20Evidential%20Neural%20Network%20Based%20on%20Complex-Valued%20Contrastive%20Learning%20fo/tables/9de1bcfad639e70ee8ac4600bbe616d49b80d07ad71e016cd58bfcf89182f4dc.jpg)

![a31c9c232bd74a175f20b5071bbd78432b5c03d5aacb235b87eda299378a9025.jpg](../icml_results/1095_TrustworthyQENN_%20A%20Quantum%20Evidential%20Neural%20Network%20Based%20on%20Complex-Valued%20Contrastive%20Learning%20fo/tables/a31c9c232bd74a175f20b5071bbd78432b5c03d5aacb235b87eda299378a9025.jpg)

![aefbf90d3cdb7f9b19a0805ed96e65b5cb5281d9e50dc50de5a549d2c818f192.jpg](../icml_results/1095_TrustworthyQENN_%20A%20Quantum%20Evidential%20Neural%20Network%20Based%20on%20Complex-Valued%20Contrastive%20Learning%20fo/tables/aefbf90d3cdb7f9b19a0805ed96e65b5cb5281d9e50dc50de5a549d2c818f192.jpg)

![ea78cdb22c084c70c32313e338df6c6c21a62b7c879c80e79ced2dcdb755b2b0.jpg](../icml_results/1095_TrustworthyQENN_%20A%20Quantum%20Evidential%20Neural%20Network%20Based%20on%20Complex-Valued%20Contrastive%20Learning%20fo/tables/ea78cdb22c084c70c32313e338df6c6c21a62b7c879c80e79ced2dcdb755b2b0.jpg)

<a id="paper-1078"></a>

## OnePO: Direct One-stage Policy Optimization for SFT-free Domain Adaptation


### Images

![2f2ba7768c7208c174a7737adb89645711e300b2814181afff5495bdc18fc2ef.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/images/2f2ba7768c7208c174a7737adb89645711e300b2814181afff5495bdc18fc2ef.jpg)

![5f30b25e3c34725b88e0a66d255996a0df1efe9a131f5c4f9cc832e582245e8e.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/images/5f30b25e3c34725b88e0a66d255996a0df1efe9a131f5c4f9cc832e582245e8e.jpg)

![6d1a2f6b81516cc487fd9cb215a06b4a314855b5b695e149368aa062188ab4ae.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/images/6d1a2f6b81516cc487fd9cb215a06b4a314855b5b695e149368aa062188ab4ae.jpg)

![a9f49f974a08f55e782b598094301b69d3ea983d7a7ce772261a89cebb11a9c8.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/images/a9f49f974a08f55e782b598094301b69d3ea983d7a7ce772261a89cebb11a9c8.jpg)

![b69cf7bbb886e12d6b4db633eb50c461e7c1588ea16e405bcfa1ab15473c6f89.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/images/b69cf7bbb886e12d6b4db633eb50c461e7c1588ea16e405bcfa1ab15473c6f89.jpg)

![bd3d6db05020058b031155b98589e4036046755409412581abbabfadc721f410.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/images/bd3d6db05020058b031155b98589e4036046755409412581abbabfadc721f410.jpg)

![cd6715d941341e9bee5cd85f373609d526a6899a4850b1a36b53e664757d5cd1.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/images/cd6715d941341e9bee5cd85f373609d526a6899a4850b1a36b53e664757d5cd1.jpg)

![d833dc82f166cb9e66aa6e23ab400590cfbbbc78585dbf7d922b89680081f530.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/images/d833dc82f166cb9e66aa6e23ab400590cfbbbc78585dbf7d922b89680081f530.jpg)

### Tables

![033d99e004e9839c3829d4cb728ecba0d6fecb1d221c825950792f86073391ab.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/tables/033d99e004e9839c3829d4cb728ecba0d6fecb1d221c825950792f86073391ab.jpg)

![0c526574097e91924353456bfcd61abd440ec43537d5dd6314ec62e1d9e1edb1.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/tables/0c526574097e91924353456bfcd61abd440ec43537d5dd6314ec62e1d9e1edb1.jpg)

![2c55df05175ead32fd8b6aa4120b73aad477e064de872c2f4c4a3fd78c40a332.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/tables/2c55df05175ead32fd8b6aa4120b73aad477e064de872c2f4c4a3fd78c40a332.jpg)

![3be7c0626d2628c045430cfaa1706ccf9f11a314387c4e966db4f68cb75824f2.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/tables/3be7c0626d2628c045430cfaa1706ccf9f11a314387c4e966db4f68cb75824f2.jpg)

![46d51d046a9f59a487068a563a00b788975b420eb9fb9bd1310ce173fc259bae.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/tables/46d51d046a9f59a487068a563a00b788975b420eb9fb9bd1310ce173fc259bae.jpg)

![4d069f45a15751533e1f7db4a175815fee4fc38070a819e2b6b425a1602bab0b.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/tables/4d069f45a15751533e1f7db4a175815fee4fc38070a819e2b6b425a1602bab0b.jpg)

![5b75d8e8d18a8ce73b298f7b9fb739a91704050b7a40ad2aab1ae4278e48526d.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/tables/5b75d8e8d18a8ce73b298f7b9fb739a91704050b7a40ad2aab1ae4278e48526d.jpg)

![5fc78a0983548cffeaf4992ff928544a3a40fc276765e46e631c330e7dcf4233.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/tables/5fc78a0983548cffeaf4992ff928544a3a40fc276765e46e631c330e7dcf4233.jpg)

![70a5bb7611a5e93ddd79cc42624534f100bae7e8c4b9a78997e5ecf7d468903d.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/tables/70a5bb7611a5e93ddd79cc42624534f100bae7e8c4b9a78997e5ecf7d468903d.jpg)

![77e55fc449cfde81cd75f0e3a1930de78082fb31d3d7725a946c2233bb6a11a0.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/tables/77e55fc449cfde81cd75f0e3a1930de78082fb31d3d7725a946c2233bb6a11a0.jpg)

![8472b3fc8f305747ba6c62eab42b519df56b0a9e86642451a59c26915e44cac0.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/tables/8472b3fc8f305747ba6c62eab42b519df56b0a9e86642451a59c26915e44cac0.jpg)

![856e446177f682b1fc0750f3370b6da8ba389e18958bb2a477e01961c91e5c46.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/tables/856e446177f682b1fc0750f3370b6da8ba389e18958bb2a477e01961c91e5c46.jpg)

![99a95ffc2d657f1624f8df09d5695a285f7ae8fe8b073fefe047877c6f9a5328.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/tables/99a95ffc2d657f1624f8df09d5695a285f7ae8fe8b073fefe047877c6f9a5328.jpg)

![b147791c375bffb1edc2c09fe10e1919fc61b38696e099bb4288da3f205f76d6.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/tables/b147791c375bffb1edc2c09fe10e1919fc61b38696e099bb4288da3f205f76d6.jpg)

![cfb1b0d356315c915595e8ab2f97bf2700b1ced07e1c87bd9416c49d5c0e73ff.jpg](../icml_results/1096_OnePO_%20Direct%20One-stage%20Policy%20Optimization%20for%20SFT-free%20Domain%20Adaptation/tables/cfb1b0d356315c915595e8ab2f97bf2700b1ced07e1c87bd9416c49d5c0e73ff.jpg)

<a id="paper-1079"></a>

## BabyVision: Visual Reasoning Beyond Language


### Images

![008223effee570f5782fc5341db89a488f4cf3c4891b4fec118e4bb39ad4d817.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/images/008223effee570f5782fc5341db89a488f4cf3c4891b4fec118e4bb39ad4d817.jpg)

![267b386e1f5af3b58b9b66c102dbefc00f46a97f244078b0316ca4f17d8cc523.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/images/267b386e1f5af3b58b9b66c102dbefc00f46a97f244078b0316ca4f17d8cc523.jpg)

![3ef6520cb44fe84c4cce640d95640dc6ef0a3100ba3b74811c7719745e0d6752.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/images/3ef6520cb44fe84c4cce640d95640dc6ef0a3100ba3b74811c7719745e0d6752.jpg)

![4cf1d6a0cf94603e72cbfff4735edde06c451b9966a38acd7bae5945560e3629.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/images/4cf1d6a0cf94603e72cbfff4735edde06c451b9966a38acd7bae5945560e3629.jpg)

![50c1817b3b00e40351931f319ef0b6ac728313fd3d76bb5250ba6a430116eb63.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/images/50c1817b3b00e40351931f319ef0b6ac728313fd3d76bb5250ba6a430116eb63.jpg)

![77b0e548a2a169d399a3253811015dabc37376b404c5aea59e2c99ff35c01afe.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/images/77b0e548a2a169d399a3253811015dabc37376b404c5aea59e2c99ff35c01afe.jpg)

![b49311e4b12d1ff3d1c343c2edc6b292499ce7c0ee65484efe3f502885978c84.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/images/b49311e4b12d1ff3d1c343c2edc6b292499ce7c0ee65484efe3f502885978c84.jpg)

![bfa430774c4d77fe4523015908367ed9d2c737acaed7312684b1e830f2fa63bb.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/images/bfa430774c4d77fe4523015908367ed9d2c737acaed7312684b1e830f2fa63bb.jpg)

![c32783e8236294db2a6385e61cb2622280d80896070710ccb1b6892cb21a1b3c.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/images/c32783e8236294db2a6385e61cb2622280d80896070710ccb1b6892cb21a1b3c.jpg)

![dac6d5057f858c893df2b46a6b240d2671771b1a8bfa16a95de1e0c15fd21a45.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/images/dac6d5057f858c893df2b46a6b240d2671771b1a8bfa16a95de1e0c15fd21a45.jpg)

![e694ff8df4cc459d0face2fee55c5dbde9c5a0702e63a0c7cdca292b9d9c9db3.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/images/e694ff8df4cc459d0face2fee55c5dbde9c5a0702e63a0c7cdca292b9d9c9db3.jpg)

![ec792143798a24ada69825e32e1e7f41ea6212b87edfc31a20fb61cded97855d.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/images/ec792143798a24ada69825e32e1e7f41ea6212b87edfc31a20fb61cded97855d.jpg)

### Tables

![19a6d097a3c10aa10c64542ebcb1eb7723074211d2fba4d8cedf89894086eb8c.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/tables/19a6d097a3c10aa10c64542ebcb1eb7723074211d2fba4d8cedf89894086eb8c.jpg)

![272e0b0b55c3fbde9333dcd4cebd8bfc4cd03bbbae6a96bdc0a22ac85e4810bb.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/tables/272e0b0b55c3fbde9333dcd4cebd8bfc4cd03bbbae6a96bdc0a22ac85e4810bb.jpg)

![383300409e182fce9c80cbce2a55da1125f506d40bb24afa160d94d33f46e532.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/tables/383300409e182fce9c80cbce2a55da1125f506d40bb24afa160d94d33f46e532.jpg)

![3cd87a385a1013f570ec4d1df9d681cd03dd2c32f98f36e3861f7f3427a8bf12.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/tables/3cd87a385a1013f570ec4d1df9d681cd03dd2c32f98f36e3861f7f3427a8bf12.jpg)

![5f5b9f5e84e82bfa06915c6498c6b4a307bf90ba162b07d70ffcac397540b977.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/tables/5f5b9f5e84e82bfa06915c6498c6b4a307bf90ba162b07d70ffcac397540b977.jpg)

![a4bf5541f7305028b8fdad4ec42d370eb2dea686102e5099ed7cea0721d25dc4.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/tables/a4bf5541f7305028b8fdad4ec42d370eb2dea686102e5099ed7cea0721d25dc4.jpg)

![b7b8261a1dc1ff343f32b907dded64fffde76c6e65156e78e7a3210bac6f046d.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/tables/b7b8261a1dc1ff343f32b907dded64fffde76c6e65156e78e7a3210bac6f046d.jpg)

![cdb8602479fa2f7a91d6a467c033a08f7d07768ec2ffa3193e073c1d63b8b7a7.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/tables/cdb8602479fa2f7a91d6a467c033a08f7d07768ec2ffa3193e073c1d63b8b7a7.jpg)

![d5bbadee5cc21be7b7871c9f147ca63f88ffb824ade3138f01e768f89c7bc617.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/tables/d5bbadee5cc21be7b7871c9f147ca63f88ffb824ade3138f01e768f89c7bc617.jpg)

![f4ca56603af03a459c7d49d068eb56ac79d3ea4e73eddab6fa8af9422dce7fd0.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/tables/f4ca56603af03a459c7d49d068eb56ac79d3ea4e73eddab6fa8af9422dce7fd0.jpg)

![fde250367a126effa94d5a1298dd8f031b40841857402060df68205489cbc99f.jpg](../icml_results/1097_BabyVision_%20Visual%20Reasoning%20Beyond%20Language/tables/fde250367a126effa94d5a1298dd8f031b40841857402060df68205489cbc99f.jpg)

<a id="paper-1080"></a>

## Self-Supervised Dynamical System Representations for Physiological Time-Series


### Images

![3aa99f2ec946c701889f8e110bd40cfed305bdecd1b19793106c1b427106e88c.jpg](../icml_results/1098_Self-Supervised%20Dynamical%20System%20Representations%20for%20Physiological%20Time-Series/images/3aa99f2ec946c701889f8e110bd40cfed305bdecd1b19793106c1b427106e88c.jpg)

![600f2d59dc73b9a7c2a816f47766ce3e182b3fca824e8ffc31fb5209387bd4f7.jpg](../icml_results/1098_Self-Supervised%20Dynamical%20System%20Representations%20for%20Physiological%20Time-Series/images/600f2d59dc73b9a7c2a816f47766ce3e182b3fca824e8ffc31fb5209387bd4f7.jpg)

![72777ee168651f1666a739fe05c0a0c18fec2c8d38fb2252a356af0decdda020.jpg](../icml_results/1098_Self-Supervised%20Dynamical%20System%20Representations%20for%20Physiological%20Time-Series/images/72777ee168651f1666a739fe05c0a0c18fec2c8d38fb2252a356af0decdda020.jpg)

![83bbe154b7558c9f2882ad69d7efae73ed86def1ef5fa349d3c38bf5d24daeb7.jpg](../icml_results/1098_Self-Supervised%20Dynamical%20System%20Representations%20for%20Physiological%20Time-Series/images/83bbe154b7558c9f2882ad69d7efae73ed86def1ef5fa349d3c38bf5d24daeb7.jpg)

![85cdc689cde3a0325408fb29c5a434dc75aeadec653b9625101e8d8d71ea8f7c.jpg](../icml_results/1098_Self-Supervised%20Dynamical%20System%20Representations%20for%20Physiological%20Time-Series/images/85cdc689cde3a0325408fb29c5a434dc75aeadec653b9625101e8d8d71ea8f7c.jpg)

![a96127de5f7e41477f4cccfaad951a7e7b14387e77810e791e4a124297c86bb3.jpg](../icml_results/1098_Self-Supervised%20Dynamical%20System%20Representations%20for%20Physiological%20Time-Series/images/a96127de5f7e41477f4cccfaad951a7e7b14387e77810e791e4a124297c86bb3.jpg)

![be3ab46f313037fe0b51a5b1fe97ae1c534e26d8b23bb0ca5c1e178ba279d670.jpg](../icml_results/1098_Self-Supervised%20Dynamical%20System%20Representations%20for%20Physiological%20Time-Series/images/be3ab46f313037fe0b51a5b1fe97ae1c534e26d8b23bb0ca5c1e178ba279d670.jpg)

### Tables

![1260c804f721c65e011f289eae7948c07a823b2ed273e042e0d03f73b6fc130c.jpg](../icml_results/1098_Self-Supervised%20Dynamical%20System%20Representations%20for%20Physiological%20Time-Series/tables/1260c804f721c65e011f289eae7948c07a823b2ed273e042e0d03f73b6fc130c.jpg)

![249ee23359dbd02257c653c3e75ab4d326ccadaf13d577453e54a46e1701afcb.jpg](../icml_results/1098_Self-Supervised%20Dynamical%20System%20Representations%20for%20Physiological%20Time-Series/tables/249ee23359dbd02257c653c3e75ab4d326ccadaf13d577453e54a46e1701afcb.jpg)

![769632ea83f968206c449535c90990612492d04d4989ed627ebca1f305f3f665.jpg](../icml_results/1098_Self-Supervised%20Dynamical%20System%20Representations%20for%20Physiological%20Time-Series/tables/769632ea83f968206c449535c90990612492d04d4989ed627ebca1f305f3f665.jpg)

![a033c5cabbc05026fdd93b1a8a9990c1d7c9706095c10c84fdddc3aa5f74efe0.jpg](../icml_results/1098_Self-Supervised%20Dynamical%20System%20Representations%20for%20Physiological%20Time-Series/tables/a033c5cabbc05026fdd93b1a8a9990c1d7c9706095c10c84fdddc3aa5f74efe0.jpg)

![be51413583acf4ba7a56e17c93e0dedcb796a13a6b0a0328198a9e94b4e67bf7.jpg](../icml_results/1098_Self-Supervised%20Dynamical%20System%20Representations%20for%20Physiological%20Time-Series/tables/be51413583acf4ba7a56e17c93e0dedcb796a13a6b0a0328198a9e94b4e67bf7.jpg)

![cae801b7634995f151413e2c142b8c2591fc53b402a53d5b86d93422b0e0fa74.jpg](../icml_results/1098_Self-Supervised%20Dynamical%20System%20Representations%20for%20Physiological%20Time-Series/tables/cae801b7634995f151413e2c142b8c2591fc53b402a53d5b86d93422b0e0fa74.jpg)

![d7a2450ae17e8920075ae4a194d82b3392ab4978cb7479169e1b88629b28d5ac.jpg](../icml_results/1098_Self-Supervised%20Dynamical%20System%20Representations%20for%20Physiological%20Time-Series/tables/d7a2450ae17e8920075ae4a194d82b3392ab4978cb7479169e1b88629b28d5ac.jpg)

<a id="paper-1081"></a>

## Policy-Driven World Model Adaptation for Robust Offline Model-based Reinforcement Learning


### Images

![5545586002186272071b2e3a3b13a7fa3d166980bcd53f6060e10f9b0fd258c4.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/images/5545586002186272071b2e3a3b13a7fa3d166980bcd53f6060e10f9b0fd258c4.jpg)

![5d21412e891d00faaa59ec4bc123dba3d15e1b455cbd2d7ac4fdca6eb161e013.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/images/5d21412e891d00faaa59ec4bc123dba3d15e1b455cbd2d7ac4fdca6eb161e013.jpg)

![8536b493fe04794a53a77fb8e755497fe7fd6e1d4de144dd429acb51734acba4.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/images/8536b493fe04794a53a77fb8e755497fe7fd6e1d4de144dd429acb51734acba4.jpg)

![8be0540012d301f55451aa5fe82247dff534f29857e84ee9fb7bde8cc1f90adf.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/images/8be0540012d301f55451aa5fe82247dff534f29857e84ee9fb7bde8cc1f90adf.jpg)

![d93a399aef5cda5da873fcdf9652c2a46d02ad29e070239655e785fd7d8040ab.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/images/d93a399aef5cda5da873fcdf9652c2a46d02ad29e070239655e785fd7d8040ab.jpg)

![e48ce52154839be3e7443448089b45208da4ba0e43b0c5cb86e11d175fa28c02.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/images/e48ce52154839be3e7443448089b45208da4ba0e43b0c5cb86e11d175fa28c02.jpg)

![e4fd603ebbbd87bfed492d90eea4312fa1d50f6faa7464f95f0964aa652c41e6.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/images/e4fd603ebbbd87bfed492d90eea4312fa1d50f6faa7464f95f0964aa652c41e6.jpg)

### Tables

![0c18699d4f092df2186abf37c4f3f561121e74655b9b9e61e26166bff7387c33.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/tables/0c18699d4f092df2186abf37c4f3f561121e74655b9b9e61e26166bff7387c33.jpg)

![2cab374b5961776dd0e2da688222161a0e579df7a9aa63ea14db2b0b1fe231d9.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/tables/2cab374b5961776dd0e2da688222161a0e579df7a9aa63ea14db2b0b1fe231d9.jpg)

![2f885ebbeb1ae118c1ebe92497f963e8400edac10c304f77c3e4e319d302a7d7.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/tables/2f885ebbeb1ae118c1ebe92497f963e8400edac10c304f77c3e4e319d302a7d7.jpg)

![3c78e9b8214e725c61f6b6ce024698d1ed324fac13ad17a85cd99866490802d9.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/tables/3c78e9b8214e725c61f6b6ce024698d1ed324fac13ad17a85cd99866490802d9.jpg)

![44c9ca4d3d3b389b77d385fc2a7add5384937cb9ab25495318277835eaa57b17.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/tables/44c9ca4d3d3b389b77d385fc2a7add5384937cb9ab25495318277835eaa57b17.jpg)

![6205bf8de18dcd5a2ac4c20028922a693e05af7869be8b165842c984a407a5b5.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/tables/6205bf8de18dcd5a2ac4c20028922a693e05af7869be8b165842c984a407a5b5.jpg)

![78b69a8207a8d8e83aeb514f676a060e9406c08fdffbbbcac0f9309a49a60d1e.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/tables/78b69a8207a8d8e83aeb514f676a060e9406c08fdffbbbcac0f9309a49a60d1e.jpg)

![880b720946ec518c47da9b128d87772510ddb908f26c07e7dfeae9a6e571f3c4.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/tables/880b720946ec518c47da9b128d87772510ddb908f26c07e7dfeae9a6e571f3c4.jpg)

![a17ceff895a6d6d3b4ae384c422d6bb7829b11c5f29eeecceb198b387ca096e0.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/tables/a17ceff895a6d6d3b4ae384c422d6bb7829b11c5f29eeecceb198b387ca096e0.jpg)

![b48514380ec389f6d7f90cab35693bc11445d483a77f253bbf4efff5cb096241.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/tables/b48514380ec389f6d7f90cab35693bc11445d483a77f253bbf4efff5cb096241.jpg)

![c02427e6bdfa27b3e1542a5ddc05046f71f926de5ac01bd2407fe67162bbc709.jpg](../icml_results/1099_Policy-Driven%20World%20Model%20Adaptation%20for%20Robust%20Offline%20Model-based%20Reinforcement%20Learning/tables/c02427e6bdfa27b3e1542a5ddc05046f71f926de5ac01bd2407fe67162bbc709.jpg)

<a id="paper-1082"></a>

## NanoSpec: Accelerating Speculative Decoding using Minimalist In-Context Vocabularies


### Images

![12b0748164b628ded34c096980859369d62490ed97d454ab9dc4ec0c9e002462.jpg](../icml_results/1100_NanoSpec_%20Accelerating%20Speculative%20Decoding%20using%20Minimalist%20In-Context%20Vocabularies/images/12b0748164b628ded34c096980859369d62490ed97d454ab9dc4ec0c9e002462.jpg)

![2eb8fa2cee1f9c50a771ba6d650142dd7590e0469d7781caaeced4909f49d10f.jpg](../icml_results/1100_NanoSpec_%20Accelerating%20Speculative%20Decoding%20using%20Minimalist%20In-Context%20Vocabularies/images/2eb8fa2cee1f9c50a771ba6d650142dd7590e0469d7781caaeced4909f49d10f.jpg)

![5bb2383174a128b25c472a4565a37e58d26c713dd8157ba7b8c9eb1d56126483.jpg](../icml_results/1100_NanoSpec_%20Accelerating%20Speculative%20Decoding%20using%20Minimalist%20In-Context%20Vocabularies/images/5bb2383174a128b25c472a4565a37e58d26c713dd8157ba7b8c9eb1d56126483.jpg)

![6c8d5c1d6532c71d66838c727f4dcc950450cf07dd03a27cec76a87a3eb1b7ba.jpg](../icml_results/1100_NanoSpec_%20Accelerating%20Speculative%20Decoding%20using%20Minimalist%20In-Context%20Vocabularies/images/6c8d5c1d6532c71d66838c727f4dcc950450cf07dd03a27cec76a87a3eb1b7ba.jpg)

![877f01317e9ef6aaf788496ef52217d79b7a93aa1c614afb3848d184cb176c41.jpg](../icml_results/1100_NanoSpec_%20Accelerating%20Speculative%20Decoding%20using%20Minimalist%20In-Context%20Vocabularies/images/877f01317e9ef6aaf788496ef52217d79b7a93aa1c614afb3848d184cb176c41.jpg)

![ae5f6dd26c459824d42ed3baca1875e2babbe0aea5676eb56529eed1007eabcf.jpg](../icml_results/1100_NanoSpec_%20Accelerating%20Speculative%20Decoding%20using%20Minimalist%20In-Context%20Vocabularies/images/ae5f6dd26c459824d42ed3baca1875e2babbe0aea5676eb56529eed1007eabcf.jpg)

### Tables

![21eb9c02fd2cd680d8acb62a5064fb5e0ed619910652ec0c4faae3c56b7ee69b.jpg](../icml_results/1100_NanoSpec_%20Accelerating%20Speculative%20Decoding%20using%20Minimalist%20In-Context%20Vocabularies/tables/21eb9c02fd2cd680d8acb62a5064fb5e0ed619910652ec0c4faae3c56b7ee69b.jpg)

![325d0cc9591ba992ad15a8804843b579a7b06fc8692135b2809c7a4734d10c4c.jpg](../icml_results/1100_NanoSpec_%20Accelerating%20Speculative%20Decoding%20using%20Minimalist%20In-Context%20Vocabularies/tables/325d0cc9591ba992ad15a8804843b579a7b06fc8692135b2809c7a4734d10c4c.jpg)

![5df4354e44cd729290b9240bb7163328bff3a1f922fd92effdac4e5c90a21cd0.jpg](../icml_results/1100_NanoSpec_%20Accelerating%20Speculative%20Decoding%20using%20Minimalist%20In-Context%20Vocabularies/tables/5df4354e44cd729290b9240bb7163328bff3a1f922fd92effdac4e5c90a21cd0.jpg)

![7dc9210e1b7cef8afbab00632de1b03d68a2a4a184004fe7d10c71c2629be1b5.jpg](../icml_results/1100_NanoSpec_%20Accelerating%20Speculative%20Decoding%20using%20Minimalist%20In-Context%20Vocabularies/tables/7dc9210e1b7cef8afbab00632de1b03d68a2a4a184004fe7d10c71c2629be1b5.jpg)

![8f377abf239282a5eb317c5f424f1338cfce802a09cc5c552c8e2b62c3c527eb.jpg](../icml_results/1100_NanoSpec_%20Accelerating%20Speculative%20Decoding%20using%20Minimalist%20In-Context%20Vocabularies/tables/8f377abf239282a5eb317c5f424f1338cfce802a09cc5c552c8e2b62c3c527eb.jpg)

![a9e0f2664e0898685353eee5aa50ccef2351a683c48075382fab31915fe790c6.jpg](../icml_results/1100_NanoSpec_%20Accelerating%20Speculative%20Decoding%20using%20Minimalist%20In-Context%20Vocabularies/tables/a9e0f2664e0898685353eee5aa50ccef2351a683c48075382fab31915fe790c6.jpg)

![e37763fcd621d98f10d04efbb7bcebc5f10f1535c0302a02c8b0d705d170da60.jpg](../icml_results/1100_NanoSpec_%20Accelerating%20Speculative%20Decoding%20using%20Minimalist%20In-Context%20Vocabularies/tables/e37763fcd621d98f10d04efbb7bcebc5f10f1535c0302a02c8b0d705d170da60.jpg)

![f35f332bcf3a79a8ebcfe34385259a87e4b60abd53f11dbd647043f3816da9c0.jpg](../icml_results/1100_NanoSpec_%20Accelerating%20Speculative%20Decoding%20using%20Minimalist%20In-Context%20Vocabularies/tables/f35f332bcf3a79a8ebcfe34385259a87e4b60abd53f11dbd647043f3816da9c0.jpg)

<a id="paper-1083"></a>

## VIPO: Value Function Inconsistency Penalized Offline Reinforcement Learning


### Images

![404a0e6a1700a44342a76817c6052da8c968759b8b81d850ec8eb282c53be73f.jpg](../icml_results/1101_VIPO_%20Value%20Function%20Inconsistency%20Penalized%20Offline%20Reinforcement%20Learning/images/404a0e6a1700a44342a76817c6052da8c968759b8b81d850ec8eb282c53be73f.jpg)

![9959e850f514b5339750808b0cab2ab030c082d36a0fd1324075b3b9e31c9605.jpg](../icml_results/1101_VIPO_%20Value%20Function%20Inconsistency%20Penalized%20Offline%20Reinforcement%20Learning/images/9959e850f514b5339750808b0cab2ab030c082d36a0fd1324075b3b9e31c9605.jpg)

### Tables

![2be8754760150ce21fedabf4fcb8fb8505f5366be192dc9e870d099f2568ffd7.jpg](../icml_results/1101_VIPO_%20Value%20Function%20Inconsistency%20Penalized%20Offline%20Reinforcement%20Learning/tables/2be8754760150ce21fedabf4fcb8fb8505f5366be192dc9e870d099f2568ffd7.jpg)

![4d5029ae8a039cadee35bc2b8a3843391e8f1462fe48244c84d7898406cde38b.jpg](../icml_results/1101_VIPO_%20Value%20Function%20Inconsistency%20Penalized%20Offline%20Reinforcement%20Learning/tables/4d5029ae8a039cadee35bc2b8a3843391e8f1462fe48244c84d7898406cde38b.jpg)

![5a397ec76b570f967d008016b6ce2d29ec58e25074786cc2c688fe5edc6f5a5a.jpg](../icml_results/1101_VIPO_%20Value%20Function%20Inconsistency%20Penalized%20Offline%20Reinforcement%20Learning/tables/5a397ec76b570f967d008016b6ce2d29ec58e25074786cc2c688fe5edc6f5a5a.jpg)

![6040e12bef7eb7ab38ea601ff695ac47998cc1b2ba822a8bba34d4b2842f0b1c.jpg](../icml_results/1101_VIPO_%20Value%20Function%20Inconsistency%20Penalized%20Offline%20Reinforcement%20Learning/tables/6040e12bef7eb7ab38ea601ff695ac47998cc1b2ba822a8bba34d4b2842f0b1c.jpg)

![644599b97ac2bd417d3be7ffee6eb302deb5c8d24698747d46e41a156698e16f.jpg](../icml_results/1101_VIPO_%20Value%20Function%20Inconsistency%20Penalized%20Offline%20Reinforcement%20Learning/tables/644599b97ac2bd417d3be7ffee6eb302deb5c8d24698747d46e41a156698e16f.jpg)

![66433238663d6e1805eb7237b02117c8b8049eafc29f4d3d265d36f74b3cba31.jpg](../icml_results/1101_VIPO_%20Value%20Function%20Inconsistency%20Penalized%20Offline%20Reinforcement%20Learning/tables/66433238663d6e1805eb7237b02117c8b8049eafc29f4d3d265d36f74b3cba31.jpg)

![84391bc005bb947411262054129ded1828a3861b3e126098fe012548923e22f4.jpg](../icml_results/1101_VIPO_%20Value%20Function%20Inconsistency%20Penalized%20Offline%20Reinforcement%20Learning/tables/84391bc005bb947411262054129ded1828a3861b3e126098fe012548923e22f4.jpg)

<a id="paper-1084"></a>

## MetaStreet: Semi-Supervised Multimodal Learning for Street-Level Socioeconomic Prediction


### Images

![05effbc27953baed3d9a2c0459f56e8b639873a6b7df2a834a31c3b8813fa71b.jpg](../icml_results/1102_MetaStreet_%20Semi-Supervised%20Multimodal%20Learning%20for%20Street-Level%20Socioeconomic%20Prediction/images/05effbc27953baed3d9a2c0459f56e8b639873a6b7df2a834a31c3b8813fa71b.jpg)

![0956eaecdec356560a1a89eda15520d1c82a5b2c6c3d7b725f0ddc415962744e.jpg](../icml_results/1102_MetaStreet_%20Semi-Supervised%20Multimodal%20Learning%20for%20Street-Level%20Socioeconomic%20Prediction/images/0956eaecdec356560a1a89eda15520d1c82a5b2c6c3d7b725f0ddc415962744e.jpg)

![5e457e4c3a6372356c38d4c40f9e9466967427c79aa91c116518b3d8eb499a90.jpg](../icml_results/1102_MetaStreet_%20Semi-Supervised%20Multimodal%20Learning%20for%20Street-Level%20Socioeconomic%20Prediction/images/5e457e4c3a6372356c38d4c40f9e9466967427c79aa91c116518b3d8eb499a90.jpg)

![881bd8814c24b3db719c076aebf74306c5877fe0a2cdf1a40eba2d7693e5e45f.jpg](../icml_results/1102_MetaStreet_%20Semi-Supervised%20Multimodal%20Learning%20for%20Street-Level%20Socioeconomic%20Prediction/images/881bd8814c24b3db719c076aebf74306c5877fe0a2cdf1a40eba2d7693e5e45f.jpg)

![aa47480bd18da7323d057e00912cee5356b243d0795bb3ec3be076ca536402fc.jpg](../icml_results/1102_MetaStreet_%20Semi-Supervised%20Multimodal%20Learning%20for%20Street-Level%20Socioeconomic%20Prediction/images/aa47480bd18da7323d057e00912cee5356b243d0795bb3ec3be076ca536402fc.jpg)

![f25f99452b0bad1ebc62129cc7cc7bf3695c37cb05215a8a3cb006fcc3b53b29.jpg](../icml_results/1102_MetaStreet_%20Semi-Supervised%20Multimodal%20Learning%20for%20Street-Level%20Socioeconomic%20Prediction/images/f25f99452b0bad1ebc62129cc7cc7bf3695c37cb05215a8a3cb006fcc3b53b29.jpg)

![fac51847e0c20204f9bd6141a994c5ba575a4584753a3a44cf5ec82c71781322.jpg](../icml_results/1102_MetaStreet_%20Semi-Supervised%20Multimodal%20Learning%20for%20Street-Level%20Socioeconomic%20Prediction/images/fac51847e0c20204f9bd6141a994c5ba575a4584753a3a44cf5ec82c71781322.jpg)

### Tables

![02ec4a9fb70cb2278b5035c92fc2937ad289a03183db3cc2f2f34956e343e709.jpg](../icml_results/1102_MetaStreet_%20Semi-Supervised%20Multimodal%20Learning%20for%20Street-Level%20Socioeconomic%20Prediction/tables/02ec4a9fb70cb2278b5035c92fc2937ad289a03183db3cc2f2f34956e343e709.jpg)

![4b3367f4af8ce62e9fd592b8482acb8d490bb4e4467253976eb2fa14ab421429.jpg](../icml_results/1102_MetaStreet_%20Semi-Supervised%20Multimodal%20Learning%20for%20Street-Level%20Socioeconomic%20Prediction/tables/4b3367f4af8ce62e9fd592b8482acb8d490bb4e4467253976eb2fa14ab421429.jpg)

![809ebb13566313a9693ee0ca98276f745e6086105a41358e4f71e2eb23d5e440.jpg](../icml_results/1102_MetaStreet_%20Semi-Supervised%20Multimodal%20Learning%20for%20Street-Level%20Socioeconomic%20Prediction/tables/809ebb13566313a9693ee0ca98276f745e6086105a41358e4f71e2eb23d5e440.jpg)

![8343f6a434a4c3c821585fc1132da5b3b381968f5ebbafa62a78629404882ae5.jpg](../icml_results/1102_MetaStreet_%20Semi-Supervised%20Multimodal%20Learning%20for%20Street-Level%20Socioeconomic%20Prediction/tables/8343f6a434a4c3c821585fc1132da5b3b381968f5ebbafa62a78629404882ae5.jpg)

![a10152957877bdf9a3e55d09837072db27166bf0ce197d6b97a0fc91f46c6f39.jpg](../icml_results/1102_MetaStreet_%20Semi-Supervised%20Multimodal%20Learning%20for%20Street-Level%20Socioeconomic%20Prediction/tables/a10152957877bdf9a3e55d09837072db27166bf0ce197d6b97a0fc91f46c6f39.jpg)

<a id="paper-1085"></a>

## GuidedBridge: Training-freely Improving Bridge Models with Prior Guidance


### Images

![0c79c49b01c9e37e03fb1ceaa1500bf1da1cb6afb7e2f89faa62bc84cf8c4143.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/0c79c49b01c9e37e03fb1ceaa1500bf1da1cb6afb7e2f89faa62bc84cf8c4143.jpg)

![262e2a04c84d04763e137ae13a7c002fc4a924ffd8eaca0774efa30bfe8cd6a9.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/262e2a04c84d04763e137ae13a7c002fc4a924ffd8eaca0774efa30bfe8cd6a9.jpg)

![2bbb2102c59bd5691f530d8d0240514d803eeb7ac42bf801dadaa0b74c181375.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/2bbb2102c59bd5691f530d8d0240514d803eeb7ac42bf801dadaa0b74c181375.jpg)

![308966b2db759e9d88be9a4f619b553a0ac01d301e0f2b42a6868c907335f17c.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/308966b2db759e9d88be9a4f619b553a0ac01d301e0f2b42a6868c907335f17c.jpg)

![3822133e19d86a6cb6a14694c8c112c90bd1e61611b7e522d90129e8e331c236.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/3822133e19d86a6cb6a14694c8c112c90bd1e61611b7e522d90129e8e331c236.jpg)

![384486e0a925f752c25964a871f0db1b767abdb190285f7e55f5543e09529b51.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/384486e0a925f752c25964a871f0db1b767abdb190285f7e55f5543e09529b51.jpg)

![384ebc08ab7deee83d725790a6351ddc7f02efaa590c23748de5c4e24b37bb8e.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/384ebc08ab7deee83d725790a6351ddc7f02efaa590c23748de5c4e24b37bb8e.jpg)

![41aa3d42dbe14b139e3c5af000dc5c41a40435e97031a1fbc09b1322ea6e6330.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/41aa3d42dbe14b139e3c5af000dc5c41a40435e97031a1fbc09b1322ea6e6330.jpg)

![4d4cf1c18ca059c0ac2c0477020bd2b0de4f443e696704b59921e1a092f64340.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/4d4cf1c18ca059c0ac2c0477020bd2b0de4f443e696704b59921e1a092f64340.jpg)

![538733e4c076fa67ebbbd29fefc081b478c83feae44d766b99a5459a1c0d6ee2.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/538733e4c076fa67ebbbd29fefc081b478c83feae44d766b99a5459a1c0d6ee2.jpg)

![636398756a771dcd8133accf8ded98abc22d8258aaf04e05978847bcf01bffb5.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/636398756a771dcd8133accf8ded98abc22d8258aaf04e05978847bcf01bffb5.jpg)

![6a14f04e0d7a4513479cc58939101feb2be069de93b53a4bc0197de4d4204f88.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/6a14f04e0d7a4513479cc58939101feb2be069de93b53a4bc0197de4d4204f88.jpg)

![7303caa3c603786c47d7fb912a0995c4b78fac27044b01bb53efa3b967bcef7c.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/7303caa3c603786c47d7fb912a0995c4b78fac27044b01bb53efa3b967bcef7c.jpg)

![839a5ef293ea1524ae911aa5e2c067d0bd29d6eae2f1ebf3b78cad30aa79bd5b.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/839a5ef293ea1524ae911aa5e2c067d0bd29d6eae2f1ebf3b78cad30aa79bd5b.jpg)

![8c7a7d1241eae335b8edf2cf22d15bb5757e0b9919aa672cf0c47b9df35b1735.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/8c7a7d1241eae335b8edf2cf22d15bb5757e0b9919aa672cf0c47b9df35b1735.jpg)

![8f16a1347d34f729eac159493f179be1a4a2cb3d151faba8b937ff6f5c815e1d.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/8f16a1347d34f729eac159493f179be1a4a2cb3d151faba8b937ff6f5c815e1d.jpg)

![99df253875151af867523594bb703ab6ecb45b22a0bced9c1cedb286f12787c0.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/99df253875151af867523594bb703ab6ecb45b22a0bced9c1cedb286f12787c0.jpg)

![9a4852f8ad2f07847e10c1714a8f55194a57bced47587fd86a792ce26cf412d7.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/9a4852f8ad2f07847e10c1714a8f55194a57bced47587fd86a792ce26cf412d7.jpg)

![a81949d49e17fd80f10e7b0b751e6a6ebf5f2a1a7367d7b246e77b8198e279d7.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/a81949d49e17fd80f10e7b0b751e6a6ebf5f2a1a7367d7b246e77b8198e279d7.jpg)

![db030a2ad40620a9ffc34fb0d97f212e16e1c34f7716b5d6067f92a7c3c38486.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/db030a2ad40620a9ffc34fb0d97f212e16e1c34f7716b5d6067f92a7c3c38486.jpg)

![ed40a11edece5a69d9f9911597e84be327012a820106ca9275ae6ac13d2bfbf9.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/ed40a11edece5a69d9f9911597e84be327012a820106ca9275ae6ac13d2bfbf9.jpg)

![fa1860a03ba53d8cd64fe78ad7cc0777cf6b812199d68599c6a5c1c418b702a2.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/images/fa1860a03ba53d8cd64fe78ad7cc0777cf6b812199d68599c6a5c1c418b702a2.jpg)

### Tables

![19abbbd5db4f3e09141f28a4e7d7227462b982c0a6273612cf72c61e77375f1e.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/19abbbd5db4f3e09141f28a4e7d7227462b982c0a6273612cf72c61e77375f1e.jpg)

![2b225947ce55dec690ef859f4d6ecfbd5a30ef491f57d83a935016ac90138596.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/2b225947ce55dec690ef859f4d6ecfbd5a30ef491f57d83a935016ac90138596.jpg)

![2d8dca0d035957ef19bef719ef5b2f92e752132f87a0c500f4ecce72172a9416.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/2d8dca0d035957ef19bef719ef5b2f92e752132f87a0c500f4ecce72172a9416.jpg)

![349bd1a250de4c1792e1a77137fa6827ef5ff94f029bad66c0bcae54c46668fb.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/349bd1a250de4c1792e1a77137fa6827ef5ff94f029bad66c0bcae54c46668fb.jpg)

![3c146663d07d162d66c75423ea500633034f5af1a65fbe801a6da51a971e3ec7.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/3c146663d07d162d66c75423ea500633034f5af1a65fbe801a6da51a971e3ec7.jpg)

![48011452f74a9b7b683ffaa5ce7709522453193f1a769b557523bcfe32556151.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/48011452f74a9b7b683ffaa5ce7709522453193f1a769b557523bcfe32556151.jpg)

![5d686ba4c29b95c976424e68ee16465951bc99ae083caf3b7da9c669f0c94752.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/5d686ba4c29b95c976424e68ee16465951bc99ae083caf3b7da9c669f0c94752.jpg)

![5f34574369df365c61342a684b493b452d21917b5158901ddd901f93605968ed.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/5f34574369df365c61342a684b493b452d21917b5158901ddd901f93605968ed.jpg)

![6b872103c1ed36956a75ffed48742049c6c5117eb692b561d86d496dd5de9441.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/6b872103c1ed36956a75ffed48742049c6c5117eb692b561d86d496dd5de9441.jpg)

![71e82fa96523e6cd3dd56ae1048ebc2656486d38f7cf261a1c371b2cfadb300d.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/71e82fa96523e6cd3dd56ae1048ebc2656486d38f7cf261a1c371b2cfadb300d.jpg)

![7813d90747fdebde57a3711fa206a93f1656b0d7008e8156d879989ea76b7360.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/7813d90747fdebde57a3711fa206a93f1656b0d7008e8156d879989ea76b7360.jpg)

![8178dbe48a9935fbc2d21feb818ba71a6b2bc7e0e5e474bcd2db227ad1998bcf.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/8178dbe48a9935fbc2d21feb818ba71a6b2bc7e0e5e474bcd2db227ad1998bcf.jpg)

![a23c0b7638cde17f5cdf9f00eee0c36054999c1e669c6d7eb7c46e944b46c0e8.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/a23c0b7638cde17f5cdf9f00eee0c36054999c1e669c6d7eb7c46e944b46c0e8.jpg)

![a9e134e6b55ecb671f45a739b8d4163cde660a237baa10c7df1cd7fdb83ed732.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/a9e134e6b55ecb671f45a739b8d4163cde660a237baa10c7df1cd7fdb83ed732.jpg)

![abee62a0a477d49adbfe6fd632ac6971f3312cfad21799b3bcba9987e7968e53.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/abee62a0a477d49adbfe6fd632ac6971f3312cfad21799b3bcba9987e7968e53.jpg)

![bbe275294caa2480980918ac288f44ee9edf805143e0cfbba3aac352a67eff65.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/bbe275294caa2480980918ac288f44ee9edf805143e0cfbba3aac352a67eff65.jpg)

![c4885ae238833d10a0a0f885a20f781893f8eec7d70cb76f5f8df3583f7c517d.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/c4885ae238833d10a0a0f885a20f781893f8eec7d70cb76f5f8df3583f7c517d.jpg)

![d22a87114fa895fd6590c70840f54d374eaee6db8fa8237aadd772bd9e0aecbb.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/d22a87114fa895fd6590c70840f54d374eaee6db8fa8237aadd772bd9e0aecbb.jpg)

![d874e4f6930efff6fa74ec0a7fd4ce96922b689e012b8bb392dc1413cec8e967.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/d874e4f6930efff6fa74ec0a7fd4ce96922b689e012b8bb392dc1413cec8e967.jpg)

![db82c0e7fb8f92bfc7ab1063660c11bf5c9b00b452908bf8de25cbdc490ea928.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/db82c0e7fb8f92bfc7ab1063660c11bf5c9b00b452908bf8de25cbdc490ea928.jpg)

![e226442ad42861fe79962951882706215d33b925897fd47a120557283e71cd2d.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/e226442ad42861fe79962951882706215d33b925897fd47a120557283e71cd2d.jpg)

![e9664fca8a44b46678406d33d479a18dd7c9912ff2ae306d754354e45b365d50.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/e9664fca8a44b46678406d33d479a18dd7c9912ff2ae306d754354e45b365d50.jpg)

![f203325e279a18b088285ce2d607fcb974ec6de970de3756b5125e21ac759c48.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/f203325e279a18b088285ce2d607fcb974ec6de970de3756b5125e21ac759c48.jpg)

![f763c7e91263c2535af5694f744609f56789a6c0f13fec5538e2ceeda411780e.jpg](../icml_results/1103_GuidedBridge_%20Training-freely%20Improving%20Bridge%20Models%20with%20Prior%20Guidance/tables/f763c7e91263c2535af5694f744609f56789a6c0f13fec5538e2ceeda411780e.jpg)

<a id="paper-1086"></a>

## PCRNet: Phase-aware Complex Refinement Network for EEG-based Auditory Attention Decoding


### Images

![1ddc5d447ee8ddad30c340a4ff2257470d792ac11c745bc02efcf7a5e6356cd3.jpg](../icml_results/1104_PCRNet_%20Phase-aware%20Complex%20Refinement%20Network%20for%20EEG-based%20Auditory%20Attention%20Decoding/images/1ddc5d447ee8ddad30c340a4ff2257470d792ac11c745bc02efcf7a5e6356cd3.jpg)

![52e98c0d8604090cb4829b2a63832266b9b72f773dd904e7723508931f1b7927.jpg](../icml_results/1104_PCRNet_%20Phase-aware%20Complex%20Refinement%20Network%20for%20EEG-based%20Auditory%20Attention%20Decoding/images/52e98c0d8604090cb4829b2a63832266b9b72f773dd904e7723508931f1b7927.jpg)

![6bdc8f8fe35b0e940a037f4424c7719537dc17773ae94e7a0b520dda5c6f6689.jpg](../icml_results/1104_PCRNet_%20Phase-aware%20Complex%20Refinement%20Network%20for%20EEG-based%20Auditory%20Attention%20Decoding/images/6bdc8f8fe35b0e940a037f4424c7719537dc17773ae94e7a0b520dda5c6f6689.jpg)

![93b117b26298cb1b56a3352917bc002cb410b0470c3ffb3f0d3d42ed47627e18.jpg](../icml_results/1104_PCRNet_%20Phase-aware%20Complex%20Refinement%20Network%20for%20EEG-based%20Auditory%20Attention%20Decoding/images/93b117b26298cb1b56a3352917bc002cb410b0470c3ffb3f0d3d42ed47627e18.jpg)

![bd0dffb3307b1f863eaab9ed240bfd3ccfe7d1fa5f90adc60cb49e9c5de7c6c9.jpg](../icml_results/1104_PCRNet_%20Phase-aware%20Complex%20Refinement%20Network%20for%20EEG-based%20Auditory%20Attention%20Decoding/images/bd0dffb3307b1f863eaab9ed240bfd3ccfe7d1fa5f90adc60cb49e9c5de7c6c9.jpg)

![f55951af5ac0f477c5ef0b1ede98ad6af25c0bc17c68fb1f0f5a3dfa21162dd2.jpg](../icml_results/1104_PCRNet_%20Phase-aware%20Complex%20Refinement%20Network%20for%20EEG-based%20Auditory%20Attention%20Decoding/images/f55951af5ac0f477c5ef0b1ede98ad6af25c0bc17c68fb1f0f5a3dfa21162dd2.jpg)

### Tables

![0df77207349f115a22b6c4167b932b7684a822fec44ba611948a83e4a1408227.jpg](../icml_results/1104_PCRNet_%20Phase-aware%20Complex%20Refinement%20Network%20for%20EEG-based%20Auditory%20Attention%20Decoding/tables/0df77207349f115a22b6c4167b932b7684a822fec44ba611948a83e4a1408227.jpg)

![2951609002b1f1f143653531fedd6cd5be9521905b336c82cbf603ef85531850.jpg](../icml_results/1104_PCRNet_%20Phase-aware%20Complex%20Refinement%20Network%20for%20EEG-based%20Auditory%20Attention%20Decoding/tables/2951609002b1f1f143653531fedd6cd5be9521905b336c82cbf603ef85531850.jpg)

![460acd709ef3481506aa012960795f860b2323e97aed06e61db2fdbd6aa89aa1.jpg](../icml_results/1104_PCRNet_%20Phase-aware%20Complex%20Refinement%20Network%20for%20EEG-based%20Auditory%20Attention%20Decoding/tables/460acd709ef3481506aa012960795f860b2323e97aed06e61db2fdbd6aa89aa1.jpg)

![7f89df82635313a3a43026efae37898d9e31986e66526c818016e22e761bd211.jpg](../icml_results/1104_PCRNet_%20Phase-aware%20Complex%20Refinement%20Network%20for%20EEG-based%20Auditory%20Attention%20Decoding/tables/7f89df82635313a3a43026efae37898d9e31986e66526c818016e22e761bd211.jpg)

![841ee6ad268682d2c2c36f65d538023bd481ae226265c5c1e3e192deab384b4d.jpg](../icml_results/1104_PCRNet_%20Phase-aware%20Complex%20Refinement%20Network%20for%20EEG-based%20Auditory%20Attention%20Decoding/tables/841ee6ad268682d2c2c36f65d538023bd481ae226265c5c1e3e192deab384b4d.jpg)

<a id="paper-1087"></a>

## CONCUR: High-Throughput Agentic Batch Inference of LLM via Congestion-Based Concurrency Control


### Images

![09a65e0838fab3fbad12cc1531a3103b40a621f3410e7fb361d810c360ea1faf.jpg](../icml_results/1105_CONCUR_%20High-Throughput%20Agentic%20Batch%20Inference%20of%20LLM%20via%20Congestion-Based%20Concurrency%20Control/images/09a65e0838fab3fbad12cc1531a3103b40a621f3410e7fb361d810c360ea1faf.jpg)

![486d2d95b88e135ba689746855c9d2580c4508bc30af36bf3eac5e96ea6bcad5.jpg](../icml_results/1105_CONCUR_%20High-Throughput%20Agentic%20Batch%20Inference%20of%20LLM%20via%20Congestion-Based%20Concurrency%20Control/images/486d2d95b88e135ba689746855c9d2580c4508bc30af36bf3eac5e96ea6bcad5.jpg)

![4e09b749edd25d265ce37d45f5f625267f133f1c79044b2dd450447db282e0c8.jpg](../icml_results/1105_CONCUR_%20High-Throughput%20Agentic%20Batch%20Inference%20of%20LLM%20via%20Congestion-Based%20Concurrency%20Control/images/4e09b749edd25d265ce37d45f5f625267f133f1c79044b2dd450447db282e0c8.jpg)

![85567b9cd809526daa4d6cefc00c7fde30bebb89d6edff3f2ec91b7327ac0e7c.jpg](../icml_results/1105_CONCUR_%20High-Throughput%20Agentic%20Batch%20Inference%20of%20LLM%20via%20Congestion-Based%20Concurrency%20Control/images/85567b9cd809526daa4d6cefc00c7fde30bebb89d6edff3f2ec91b7327ac0e7c.jpg)

![96566a4c1e37fd934e75d3f9348b4b54a7cfb5e9452ecf7a360c39c4e31d09a1.jpg](../icml_results/1105_CONCUR_%20High-Throughput%20Agentic%20Batch%20Inference%20of%20LLM%20via%20Congestion-Based%20Concurrency%20Control/images/96566a4c1e37fd934e75d3f9348b4b54a7cfb5e9452ecf7a360c39c4e31d09a1.jpg)

![efd1e2267e9ed455dd947583e5022505663179540b18fc32364e65f44b42a0f7.jpg](../icml_results/1105_CONCUR_%20High-Throughput%20Agentic%20Batch%20Inference%20of%20LLM%20via%20Congestion-Based%20Concurrency%20Control/images/efd1e2267e9ed455dd947583e5022505663179540b18fc32364e65f44b42a0f7.jpg)

### Tables

![2cea2a97cfe2fb9acc8a73140503fcc0fe99f119d4817d3b913a86dfcc7e0746.jpg](../icml_results/1105_CONCUR_%20High-Throughput%20Agentic%20Batch%20Inference%20of%20LLM%20via%20Congestion-Based%20Concurrency%20Control/tables/2cea2a97cfe2fb9acc8a73140503fcc0fe99f119d4817d3b913a86dfcc7e0746.jpg)

![304a2df619743ce7a74cdbeb4cad78310826209f17e5870ea0d0de63ccbc9637.jpg](../icml_results/1105_CONCUR_%20High-Throughput%20Agentic%20Batch%20Inference%20of%20LLM%20via%20Congestion-Based%20Concurrency%20Control/tables/304a2df619743ce7a74cdbeb4cad78310826209f17e5870ea0d0de63ccbc9637.jpg)

![a1f549ae5a18bf2019ba5363d4f5b433a4ab663eae22d8e5b0a9d8cb1b43aac2.jpg](../icml_results/1105_CONCUR_%20High-Throughput%20Agentic%20Batch%20Inference%20of%20LLM%20via%20Congestion-Based%20Concurrency%20Control/tables/a1f549ae5a18bf2019ba5363d4f5b433a4ab663eae22d8e5b0a9d8cb1b43aac2.jpg)

<a id="paper-1088"></a>

## Online Linear Programming for Multi-Objective Routing in LLM Serving


### Images

![07950a5502cf6319722e16800a190c60b7d25b4344cbf754eceddc293125a1c6.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/07950a5502cf6319722e16800a190c60b7d25b4344cbf754eceddc293125a1c6.jpg)

![0cd0ef0917446362424f60ca5816a0efabe91bd35bf7fdc22f048dde1cddc079.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/0cd0ef0917446362424f60ca5816a0efabe91bd35bf7fdc22f048dde1cddc079.jpg)

![2bcb9cf7406b73d4bbd1d79343c495ebd7d223fb8e14a2448b8d62cc73c44f92.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/2bcb9cf7406b73d4bbd1d79343c495ebd7d223fb8e14a2448b8d62cc73c44f92.jpg)

![37a4536e28e6541712d97feca85660b24b48620ac0ed8b5e9f61e99d42047d8a.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/37a4536e28e6541712d97feca85660b24b48620ac0ed8b5e9f61e99d42047d8a.jpg)

![41974cca8a7832fb0742a221210451c0fa68668b3159a6017d3e9ddf165036e6.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/41974cca8a7832fb0742a221210451c0fa68668b3159a6017d3e9ddf165036e6.jpg)

![45a604bff329bb18457e955e8c54451a6859fd3f8cc96d34dba0c684492218a1.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/45a604bff329bb18457e955e8c54451a6859fd3f8cc96d34dba0c684492218a1.jpg)

![4db1685c0a71abf1c4e1ec38bb495e93641a0bbdcad680a922d0f5e320ae76b8.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/4db1685c0a71abf1c4e1ec38bb495e93641a0bbdcad680a922d0f5e320ae76b8.jpg)

![4f5fd84994acae9422c837f7bee81f7d66c9358bac625f7e04c65e80e3d07a1e.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/4f5fd84994acae9422c837f7bee81f7d66c9358bac625f7e04c65e80e3d07a1e.jpg)

![504cb3f091789d2c37a83024dd50c88c47ff365135b6f8cd7974fc1254fc0012.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/504cb3f091789d2c37a83024dd50c88c47ff365135b6f8cd7974fc1254fc0012.jpg)

![564ad722fd7e2613cfa64d3efd1f3d2a5d29dfefbe9f3383884bb1623d1b0ef3.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/564ad722fd7e2613cfa64d3efd1f3d2a5d29dfefbe9f3383884bb1623d1b0ef3.jpg)

![5aca514d86fd174abf1fd738643202e07edc4b6b9b1133e5726bf850700855ec.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/5aca514d86fd174abf1fd738643202e07edc4b6b9b1133e5726bf850700855ec.jpg)

![68b75e602c724ee6bdc422cb0d4b27b09d7ac254dd0365e20fd867e8129a1f6f.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/68b75e602c724ee6bdc422cb0d4b27b09d7ac254dd0365e20fd867e8129a1f6f.jpg)

![6b84eaf898a534114955a94b5690e4650b299508278c472c1f564b960128e8e9.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/6b84eaf898a534114955a94b5690e4650b299508278c472c1f564b960128e8e9.jpg)

![712f8e2b3922fbf083237f36cc812f2bf170f6681f4b62e7ec9c17697edc70e2.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/712f8e2b3922fbf083237f36cc812f2bf170f6681f4b62e7ec9c17697edc70e2.jpg)

![7e215b1de03a88672ace19722f61a187cda4956617b13a1c07a32bf906078adf.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/7e215b1de03a88672ace19722f61a187cda4956617b13a1c07a32bf906078adf.jpg)

![91fe8ba63251e001fbe0893b96d6fe9d724ae943596d3a7577394f040440fd2c.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/91fe8ba63251e001fbe0893b96d6fe9d724ae943596d3a7577394f040440fd2c.jpg)

![94635d75f9783373da7dd73bca6536904245407d66c132b84e4f090a6cbf2e5f.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/94635d75f9783373da7dd73bca6536904245407d66c132b84e4f090a6cbf2e5f.jpg)

![96221c1f3270e2674e84391ec3360ff588875789410fdf23a30ffb2d52418bf6.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/96221c1f3270e2674e84391ec3360ff588875789410fdf23a30ffb2d52418bf6.jpg)

![9c7342a47df7bb1c96a837afbce5970fd1bc9a2823e318deab7188a3617990e4.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/9c7342a47df7bb1c96a837afbce5970fd1bc9a2823e318deab7188a3617990e4.jpg)

![a9f83881d25adc906e6aa7b697a7d55c375526abc2993fa826b714ab3cded016.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/a9f83881d25adc906e6aa7b697a7d55c375526abc2993fa826b714ab3cded016.jpg)

![b248bad1347256a58709c21c77e3e7c1fc127f083e7ab2559de5c72479340c5b.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/b248bad1347256a58709c21c77e3e7c1fc127f083e7ab2559de5c72479340c5b.jpg)

![b4d05ecf4af4776758b2d58c664a61fc2c606bf5ce4bd68f8629d949e48da157.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/b4d05ecf4af4776758b2d58c664a61fc2c606bf5ce4bd68f8629d949e48da157.jpg)

![b5b5d1258c21b53bb614a62674b7bcfc687b21fa11d92efd063e91904a0267ab.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/b5b5d1258c21b53bb614a62674b7bcfc687b21fa11d92efd063e91904a0267ab.jpg)

![b75957637b3b0b6adeb3317cbef62de0fb18c3e4c80dad63c07c8a98e4101c02.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/b75957637b3b0b6adeb3317cbef62de0fb18c3e4c80dad63c07c8a98e4101c02.jpg)

![c3b96fc81e6cd7ed85cf015cc6062944ab64b7bb75d884091703f87341901f89.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/c3b96fc81e6cd7ed85cf015cc6062944ab64b7bb75d884091703f87341901f89.jpg)

![ca2f9447d6fe76f03ac79168df92729b5621a7e80254170dbe83bf3c9f8f0467.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/ca2f9447d6fe76f03ac79168df92729b5621a7e80254170dbe83bf3c9f8f0467.jpg)

![d11109a8ccc583e5943436062f2a89a5469bf12950fb3aae8e8867116c6a9860.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/d11109a8ccc583e5943436062f2a89a5469bf12950fb3aae8e8867116c6a9860.jpg)

![d11553ca2d5e039a2823e41a1550c116d093dbd1f54ecd5d350e828ecb7e2ecb.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/d11553ca2d5e039a2823e41a1550c116d093dbd1f54ecd5d350e828ecb7e2ecb.jpg)

![e3172006352b98b42abd15fe5fa390058fa9c994091ef669fea6fd0ce604ad96.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/e3172006352b98b42abd15fe5fa390058fa9c994091ef669fea6fd0ce604ad96.jpg)

![e735f1a5c3014b66930d9d7734ce85037c0b8535f7603acd5578fb74e80a394d.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/e735f1a5c3014b66930d9d7734ce85037c0b8535f7603acd5578fb74e80a394d.jpg)

![f6a4d0196336ac45657fda8d7d925711e8b8a8c8fd36d7712c13f2f372d48427.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/images/f6a4d0196336ac45657fda8d7d925711e8b8a8c8fd36d7712c13f2f372d48427.jpg)

### Tables

![8bfd7eee9610bfa2ef3499286ef16b390e21954b8a458c586001d0992492ae7a.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/tables/8bfd7eee9610bfa2ef3499286ef16b390e21954b8a458c586001d0992492ae7a.jpg)

![90667fce26963906c5a717da400cd5020b243ab4ac847beef0525e3a9c22092a.jpg](../icml_results/1106_Online%20Linear%20Programming%20for%20Multi-Objective%20Routing%20in%20LLM%20Serving/tables/90667fce26963906c5a717da400cd5020b243ab4ac847beef0525e3a9c22092a.jpg)

<a id="paper-1089"></a>

## Boundary Embedding Shaping with Adaptive Contrastive Learning for Graph Structural Disentanglement

### Images

![07b3a8ed71c9499e1d42c60123404b50ee881cd1c3e0c6f6832896ea408f89a6.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/07b3a8ed71c9499e1d42c60123404b50ee881cd1c3e0c6f6832896ea408f89a6.jpg)

![1b7f17e7426fd0106c3be3810476bbf1ae789b8309ffaf55324fe90e55bf3ed7.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/1b7f17e7426fd0106c3be3810476bbf1ae789b8309ffaf55324fe90e55bf3ed7.jpg)

![2a4c8c89985bf2d3433971414e4e89866934f968a54ff6bbf20e354149e9a47c.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/2a4c8c89985bf2d3433971414e4e89866934f968a54ff6bbf20e354149e9a47c.jpg)

![45b33fcf4effd12d26a4228fd453350fd4206555f74d4dc2ecb2cb04a2afdef7.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/45b33fcf4effd12d26a4228fd453350fd4206555f74d4dc2ecb2cb04a2afdef7.jpg)

![904441bab8d20172450bf148c67f1982ecc21e4159dda7f645746a207985b96a.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/904441bab8d20172450bf148c67f1982ecc21e4159dda7f645746a207985b96a.jpg)

![a8e3fbfa21157948b58d35cee70fbf1db1d1e5370fa002792927c38d9f6955f1.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/a8e3fbfa21157948b58d35cee70fbf1db1d1e5370fa002792927c38d9f6955f1.jpg)

![b13c2be15725a7097a9b1e3c9eac7c5d859797ba89b7920d0f51e632d817febd.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/b13c2be15725a7097a9b1e3c9eac7c5d859797ba89b7920d0f51e632d817febd.jpg)

![d7368a70612214ff8e699aa591dbc9f9cdae824a4ba0b5ee3f3bd3d5a37a748a.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/d7368a70612214ff8e699aa591dbc9f9cdae824a4ba0b5ee3f3bd3d5a37a748a.jpg)

![ff6f8026c9ceeb565b2b9fd409376c7082701f05cfd07eb69e315f90b1c10b12.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/images/ff6f8026c9ceeb565b2b9fd409376c7082701f05cfd07eb69e315f90b1c10b12.jpg)

### Tables

![13bcaebfa0a4211f75c0e47f49944cb9797415a2465fc316a8b62fd04e965452.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/13bcaebfa0a4211f75c0e47f49944cb9797415a2465fc316a8b62fd04e965452.jpg)

![27cecf789872bc7c4a13daaaf46f917c009aba2c002ba201a56ecdb0719788eb.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/27cecf789872bc7c4a13daaaf46f917c009aba2c002ba201a56ecdb0719788eb.jpg)

![2d139ab3d9065f8c62fe930b1b4873bbb1ac72a5fa89e05ab2b506518124bff7.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/2d139ab3d9065f8c62fe930b1b4873bbb1ac72a5fa89e05ab2b506518124bff7.jpg)

![3c09445bb3aed24b8af9c20b0cb62573b62db69e1660a99dbf60f260c5191459.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/3c09445bb3aed24b8af9c20b0cb62573b62db69e1660a99dbf60f260c5191459.jpg)

![4f82ec39673e620bcb6ec67d17a6cf26b590071eca8cfc1681fe45c019a0bfb5.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/4f82ec39673e620bcb6ec67d17a6cf26b590071eca8cfc1681fe45c019a0bfb5.jpg)

![4f9795d3a4139e0d6919b7fe0560ac0f0c75c4919d1e7ac6da03bd14227abb82.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/4f9795d3a4139e0d6919b7fe0560ac0f0c75c4919d1e7ac6da03bd14227abb82.jpg)

![5d8181f0ea9b8562b947adb6d8bccf492f013bb8db50d5cb035c1fd95675a93e.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/5d8181f0ea9b8562b947adb6d8bccf492f013bb8db50d5cb035c1fd95675a93e.jpg)

![6b28b34170f9821b45d5a1e7c0b3c84c684e997e6276e168dacad28aa52bd9a0.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/6b28b34170f9821b45d5a1e7c0b3c84c684e997e6276e168dacad28aa52bd9a0.jpg)

![7adc443c4ddcd577a8d5217d7f6dffe6286afec5eb072bc5bcf75830578254bb.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/7adc443c4ddcd577a8d5217d7f6dffe6286afec5eb072bc5bcf75830578254bb.jpg)

![811aa3569fb99ffcf3105613eb26c356fdb8ff6c5c077084d07dd1e6b2f3b456.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/811aa3569fb99ffcf3105613eb26c356fdb8ff6c5c077084d07dd1e6b2f3b456.jpg)

![9462ffc5b73326cd69a589157ac366786fe13c7a4681815b4110ae49bb48c5b6.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/9462ffc5b73326cd69a589157ac366786fe13c7a4681815b4110ae49bb48c5b6.jpg)

![9e9486f9c14d625d09505be940a8a1b97575c767d7b56490f0ac85e1e63beced.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/9e9486f9c14d625d09505be940a8a1b97575c767d7b56490f0ac85e1e63beced.jpg)

![bff31d29fbc049d09796bd1ab80256cc9b8e4635fd5cb58cb88aacd857fe1a4b.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/bff31d29fbc049d09796bd1ab80256cc9b8e4635fd5cb58cb88aacd857fe1a4b.jpg)

![c1c59da82ba66bf13fc3495b4e0f6511916c34b06da022795dd11959f6f0b4d2.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/c1c59da82ba66bf13fc3495b4e0f6511916c34b06da022795dd11959f6f0b4d2.jpg)

![c345c89a2e4bd9c1026c9820b5aa906e2aba7954b26a031ed51b6651beafe18b.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/c345c89a2e4bd9c1026c9820b5aa906e2aba7954b26a031ed51b6651beafe18b.jpg)

![c7d3ed16564d0a9dce04e3252a3be9a01f76af500493b615900f2c22d2f466e0.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/c7d3ed16564d0a9dce04e3252a3be9a01f76af500493b615900f2c22d2f466e0.jpg)

![ebfb57f7103d0747d4edbcdc4f0ea4e74aeb31e1ffa44dd331e17e3654843f83.jpg](../icml_results/1107_Boundary%20Embedding%20Shaping%20with%20Adaptive%20Contrastive%20Learning%20for%20Graph%20Structural%20Disentanglement/tables/ebfb57f7103d0747d4edbcdc4f0ea4e74aeb31e1ffa44dd331e17e3654843f83.jpg)
