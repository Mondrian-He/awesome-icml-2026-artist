# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 58 of 197

## 目录 (Table of Contents)

1. [Sequential Kernel-based Conditional Independence Testing via Adaptive Betting](#paper-1882)
2. [Learning Taxonomic Trees with Hierarchical Representation Regularization for Large Multimodal Models](#paper-1883)
3. [Reasoning-preserved Efficient Distillation of Large Language Models via Activation-aware Initialization](#paper-1884)
4. [One LR Doesn’t Fit All: Heavy-Tail Guided Layerwise Learning Rates for LLMs](#paper-1885)
5. [Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks](#paper-1886)
6. [Bend the Basics: Degradation-Aware Deformable Tokenization for All-in-One Image Restoration](#paper-1887)
7. [Motion Dynamics Learning for Few-Shot Embodied Adaptation](#paper-1888)
8. [RLSF-V: Mitigating Hallucinations in MLLMs via Fuzzy Semantic Self-Feedback](#paper-1889)
9. [FocalPolicy: Frequency-Optimized Chunking and Locally Anchored Flow Matching for Coherent Visuomotor Policy](#paper-1890)
10. [On Stable Long-Form Generation: Benchmarking and Mitigating Length Volatility](#paper-1891)
11. [A Graph Foundation Model with Cross-Modal Alignment and Modality-Aware Expert Fusion for Multi-Modal Graphs](#paper-1892)
12. [From Inpainting to Editing: Unlocking Robust Mask-Free Visual Dubbing via Generative Bootstrapping](#paper-1893)
13. [RSAgent: Learning to Reason and Act via Multi-Turn Tool Invocations for Text-Guided Segmentation](#paper-1894)
14. [Refining Context-Entangled Content Segmentation via Curriculum Selection and Anti-Curriculum Promotion](#paper-1895)
15. [Learning Permutation Distributions via Reflected Diffusion on Ranks](#paper-1896)
16. [MMPD-Bench: Bridging Multimodal Fission with Multi-Polarimetric Modalities Decomposition](#paper-1897)
17. [A Flat Vocabulary or a Rich Hierarchy? Re-introducing Intrinsic Structure Transforms the Autoregressive Image Generation](#paper-1898)
18. [IAPO: Information-Aware Policy Optimization for Token-Efficient Reasoning](#paper-1899)
19. [Optimizing Network Simulation: Enhancing Performance Prediction Accuracy via Neural Architecture Search](#paper-1900)
20. [SE(n)-Invariant Flow Matching: A General Framework with Application to Object Reassembly](#paper-1901)
21. [Asymmetric Prompt Weighting for Reinforcement Learning with Verifiable Rewards](#paper-1902)
22. [TN-SHAP-G: Graph-Structured Tensor Network Surrogates for Shapley Values and Interactions](#paper-1903)
23. [From Rashomon Theory to PRAXIS: Efficient Decision Tree Rashomon Sets](#paper-1904)
24. [RADIO1D: Elastic Representations for Condensed Vision Modeling](#paper-1905)
25. [Query-efficient model evaluation using cached responses](#paper-1906)
26. [Score Based Error Correcting Code Decoder](#paper-1907)
27. [What Linear Probes Miss: Multi-View Probing for Weight-Space Learning](#paper-1908)
28. [The Expert Strikes Back: Interpreting Mixture-of-Experts Language Models at Expert Level](#paper-1909)
29. [Locally Coherent Parallel Decoding in Diffusion Language Models](#paper-1910)
30. [What is Missing? Explaining Neurons Activated by Absent Concepts](#paper-1911)
31. [Multi-View Causal Discovery without Non-Gaussianity: Identifiability and Algorithms](#paper-1912)
32. [Factored Gossip DiLoCo: Reducing Blocking Communication in DiLoCo](#paper-1913)
33. [SONAR: Spectral-Contrastive Audio Residuals for Generalizable Deepfake Detection](#paper-1914)

---


<a id="paper-1882"></a>

## Sequential Kernel-based Conditional Independence Testing via Adaptive Betting

### Images

![10597b89bd425a33024682333aa136d3ca7dd8280a4ba600fd99f79ce3d5cdca.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/10597b89bd425a33024682333aa136d3ca7dd8280a4ba600fd99f79ce3d5cdca.jpg)

![24894a21f02777e0819c57eb734456968cbf84b6f872e94733c1ea115ce88761.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/24894a21f02777e0819c57eb734456968cbf84b6f872e94733c1ea115ce88761.jpg)

![37345ed9bd24ecd43329f02c338e910018b7a318f5d131f887b29475b10022ab.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/37345ed9bd24ecd43329f02c338e910018b7a318f5d131f887b29475b10022ab.jpg)

![3a1ea49a861b937c4e264224fb1403776ff8d4faeca8bd9462b6cf5f750e8a86.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/3a1ea49a861b937c4e264224fb1403776ff8d4faeca8bd9462b6cf5f750e8a86.jpg)

![4f1ae7dd4c3ae1169a8a54f7c147f7591d561c180fcbe195079c33a6d9444fbb.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/4f1ae7dd4c3ae1169a8a54f7c147f7591d561c180fcbe195079c33a6d9444fbb.jpg)

![58245cea913a558318f1bf5a9dddbb10cea1fb801fa84458a9a161e27a019bc2.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/58245cea913a558318f1bf5a9dddbb10cea1fb801fa84458a9a161e27a019bc2.jpg)

![77f308037adbed1a68ef2ed354aa066a4e3c09186612b2c1c67dce574187a4fb.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/77f308037adbed1a68ef2ed354aa066a4e3c09186612b2c1c67dce574187a4fb.jpg)

![795968f637089e9792450aaa3be02c5c71b6e6fa6338eac53b8138f9604620f0.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/795968f637089e9792450aaa3be02c5c71b6e6fa6338eac53b8138f9604620f0.jpg)

![7e48b1a6b7e287dbba01a344c3bfac752bb10dbcf63dd3364d25aedead25e7de.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/7e48b1a6b7e287dbba01a344c3bfac752bb10dbcf63dd3364d25aedead25e7de.jpg)

![8577b4d09956da68f7d85890e612729e73de51ae1e4dd1d18944f1a34a2991b4.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/8577b4d09956da68f7d85890e612729e73de51ae1e4dd1d18944f1a34a2991b4.jpg)

![8eebab15400b833d3b34f42c3c2f8d1227032ece70eb4916c93c8710fb3c5163.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/8eebab15400b833d3b34f42c3c2f8d1227032ece70eb4916c93c8710fb3c5163.jpg)

![960e32108288c1fb00cbf770f521e79beb065fe74581d363283e62d953c6fb37.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/960e32108288c1fb00cbf770f521e79beb065fe74581d363283e62d953c6fb37.jpg)

![97ade9ae5421e5d3f4930f339a8ce1ebeaf65d96b6752bf5d7ec5b69d35950c0.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/97ade9ae5421e5d3f4930f339a8ce1ebeaf65d96b6752bf5d7ec5b69d35950c0.jpg)

![9b866f736ffbb0ad7a5e277ccc123eaf7c11df15dfc95cd35bee23860d2e4bc1.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/9b866f736ffbb0ad7a5e277ccc123eaf7c11df15dfc95cd35bee23860d2e4bc1.jpg)

![9ddb5d75798eeb9c71a26bee5a7419f959e36fa90d0cefb5b744265f3d76708e.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/9ddb5d75798eeb9c71a26bee5a7419f959e36fa90d0cefb5b744265f3d76708e.jpg)

![baad57398fe04af1fc77a0eab941febcb569357ab47d6b590355509f4b29b104.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/baad57398fe04af1fc77a0eab941febcb569357ab47d6b590355509f4b29b104.jpg)

![c7672f1a87867cbfa0d7ce7e23026e6cf7f6ac2400a4776beeea014ccbcb04e2.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/c7672f1a87867cbfa0d7ce7e23026e6cf7f6ac2400a4776beeea014ccbcb04e2.jpg)

![d043b2652304875d8cd98dc65d979ad9c2a7bbaada7db58be59723bb74245681.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/d043b2652304875d8cd98dc65d979ad9c2a7bbaada7db58be59723bb74245681.jpg)

![d4eea369c5a71fb23949b6611ee20b0839a05079593afb5d226656c17af2e41c.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/d4eea369c5a71fb23949b6611ee20b0839a05079593afb5d226656c17af2e41c.jpg)

![df364304a5eaacce0957cf7673053a6ab3a44d45fdabdffb8d2a5b260d525e0c.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/df364304a5eaacce0957cf7673053a6ab3a44d45fdabdffb8d2a5b260d525e0c.jpg)

![ee30ad07e19cffb1d500a8084510ab8447f18a08858868c7896efe011280ac4b.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/ee30ad07e19cffb1d500a8084510ab8447f18a08858868c7896efe011280ac4b.jpg)

![f5eb272e9a04c710d152b2509c59d75fb3df8900ea35a16b3a29d3e1ffe4af29.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/images/f5eb272e9a04c710d152b2509c59d75fb3df8900ea35a16b3a29d3e1ffe4af29.jpg)

### Tables

![2cd78a5b6dd269a82ac4da323b20613714fa6c05639099ac31841eb9ec616ad7.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/tables/2cd78a5b6dd269a82ac4da323b20613714fa6c05639099ac31841eb9ec616ad7.jpg)

![50fc64d2b78f538d76fca07ff6f1fc95a0c1b709fea41e5cd7a4e51f8a2d5ed2.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/tables/50fc64d2b78f538d76fca07ff6f1fc95a0c1b709fea41e5cd7a4e51f8a2d5ed2.jpg)

![aaf171d12055031929613c86f7097c25f9efc7d7d7f4020ce143b9a89713dbd9.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/tables/aaf171d12055031929613c86f7097c25f9efc7d7d7f4020ce143b9a89713dbd9.jpg)

![b87e2f05a099d4a6c8311aacc951bdc82bc03ecc2a37fcd68673d748ce47b81e.jpg](../icml_results/1906_Demystifying%20When%20Pruning%20Works%20via%20Representation%20Hierarchies/tables/b87e2f05a099d4a6c8311aacc951bdc82bc03ecc2a37fcd68673d748ce47b81e.jpg)

## Sequential Kernel-based Conditional Independence Testing via Adaptive Betting


### Images

![1638db432431620a8dd09f46f76053a412e406c3996b44b9cb4af6f07845a86c.jpg](../icml_results/1907_Sequential%20Kernel-based%20Conditional%20Independence%20Testing%20via%20Adaptive%20Betting/images/1638db432431620a8dd09f46f76053a412e406c3996b44b9cb4af6f07845a86c.jpg)

![1f76bc6af3e2ad3f06b48f60d0cc08427ebdbd6c692156de651518f36a057c7a.jpg](../icml_results/1907_Sequential%20Kernel-based%20Conditional%20Independence%20Testing%20via%20Adaptive%20Betting/images/1f76bc6af3e2ad3f06b48f60d0cc08427ebdbd6c692156de651518f36a057c7a.jpg)

![66e3c65b4192361331fe143dcdbec7372fca4a8b6ff1cdace5114671237cb1fb.jpg](../icml_results/1907_Sequential%20Kernel-based%20Conditional%20Independence%20Testing%20via%20Adaptive%20Betting/images/66e3c65b4192361331fe143dcdbec7372fca4a8b6ff1cdace5114671237cb1fb.jpg)

![84fff6fa0d3dcf18aae11171f4e84e8cc326bbce97460d485e1a8025edc75241.jpg](../icml_results/1907_Sequential%20Kernel-based%20Conditional%20Independence%20Testing%20via%20Adaptive%20Betting/images/84fff6fa0d3dcf18aae11171f4e84e8cc326bbce97460d485e1a8025edc75241.jpg)

![9e489f6790c113a2ced170f82e1af343be6f87c1bde6052990072bbde3ed8e00.jpg](../icml_results/1907_Sequential%20Kernel-based%20Conditional%20Independence%20Testing%20via%20Adaptive%20Betting/images/9e489f6790c113a2ced170f82e1af343be6f87c1bde6052990072bbde3ed8e00.jpg)

![aa348ebaadcc01032db3f70ebf46e4df01a26c8be4e29e432c5272afdb485185.jpg](../icml_results/1907_Sequential%20Kernel-based%20Conditional%20Independence%20Testing%20via%20Adaptive%20Betting/images/aa348ebaadcc01032db3f70ebf46e4df01a26c8be4e29e432c5272afdb485185.jpg)

![ad0ce0daf5bbe6fd694b455fc88f493b8089126a06f4d2300f0760f3cf9f998f.jpg](../icml_results/1907_Sequential%20Kernel-based%20Conditional%20Independence%20Testing%20via%20Adaptive%20Betting/images/ad0ce0daf5bbe6fd694b455fc88f493b8089126a06f4d2300f0760f3cf9f998f.jpg)

![c37cf76c502c40ca539a021e7b301879a3bb21c548467c8a6f1afce4e251602a.jpg](../icml_results/1907_Sequential%20Kernel-based%20Conditional%20Independence%20Testing%20via%20Adaptive%20Betting/images/c37cf76c502c40ca539a021e7b301879a3bb21c548467c8a6f1afce4e251602a.jpg)

![ec9f4c92008a6ce357f4cfa48796703f6c3fc3923c0d4ee3895f6ec078575ddf.jpg](../icml_results/1907_Sequential%20Kernel-based%20Conditional%20Independence%20Testing%20via%20Adaptive%20Betting/images/ec9f4c92008a6ce357f4cfa48796703f6c3fc3923c0d4ee3895f6ec078575ddf.jpg)

![ecc56f02849ec0257fb4069ea0855c7d4b27cbaff181205ad78031038e7e6e70.jpg](../icml_results/1907_Sequential%20Kernel-based%20Conditional%20Independence%20Testing%20via%20Adaptive%20Betting/images/ecc56f02849ec0257fb4069ea0855c7d4b27cbaff181205ad78031038e7e6e70.jpg)

![efdb1c5ff97afc55b45382253defeaac0f95fe70601d85443df22e13132e1f44.jpg](../icml_results/1907_Sequential%20Kernel-based%20Conditional%20Independence%20Testing%20via%20Adaptive%20Betting/images/efdb1c5ff97afc55b45382253defeaac0f95fe70601d85443df22e13132e1f44.jpg)

### Tables

![1d79dc2dc88bfcf1ef8821e40da9a486a9ccf6deecb45389e900a797587ec214.jpg](../icml_results/1907_Sequential%20Kernel-based%20Conditional%20Independence%20Testing%20via%20Adaptive%20Betting/tables/1d79dc2dc88bfcf1ef8821e40da9a486a9ccf6deecb45389e900a797587ec214.jpg)

![248ee53dd5437a414707e44ca47d527c82707293df59c7bfbed7ae0082fce234.jpg](../icml_results/1907_Sequential%20Kernel-based%20Conditional%20Independence%20Testing%20via%20Adaptive%20Betting/tables/248ee53dd5437a414707e44ca47d527c82707293df59c7bfbed7ae0082fce234.jpg)

![3a2da355189958d34dc21b4cd7b0e85bd54bb9675a065648203b7a0725afb481.jpg](../icml_results/1907_Sequential%20Kernel-based%20Conditional%20Independence%20Testing%20via%20Adaptive%20Betting/tables/3a2da355189958d34dc21b4cd7b0e85bd54bb9675a065648203b7a0725afb481.jpg)

![921e927d4bf1b850544265bd2498eecd015dd2c54f87b8859a1b406f4b15fb34.jpg](../icml_results/1907_Sequential%20Kernel-based%20Conditional%20Independence%20Testing%20via%20Adaptive%20Betting/tables/921e927d4bf1b850544265bd2498eecd015dd2c54f87b8859a1b406f4b15fb34.jpg)

<a id="paper-1883"></a>

## Learning Taxonomic Trees with Hierarchical Representation Regularization for Large Multimodal Models


### Images

![4fd3d23cc97a150c440552371af179b605ea1459ff319468bfa9759dc3f19789.jpg](../icml_results/1908_Learning%20Taxonomic%20Trees%20with%20Hierarchical%20Representation%20Regularization%20for%20Large%20Multimodal%20Models/images/4fd3d23cc97a150c440552371af179b605ea1459ff319468bfa9759dc3f19789.jpg)

![9c972dee7c35a922d8853a8fb0faecb373a86d4174907e4fdb7088a6f3ce5105.jpg](../icml_results/1908_Learning%20Taxonomic%20Trees%20with%20Hierarchical%20Representation%20Regularization%20for%20Large%20Multimodal%20Models/images/9c972dee7c35a922d8853a8fb0faecb373a86d4174907e4fdb7088a6f3ce5105.jpg)

![ad400e80164c8b334d6000078786cbd4d740ea051056a538715011bf69db8414.jpg](../icml_results/1908_Learning%20Taxonomic%20Trees%20with%20Hierarchical%20Representation%20Regularization%20for%20Large%20Multimodal%20Models/images/ad400e80164c8b334d6000078786cbd4d740ea051056a538715011bf69db8414.jpg)

### Tables

![1b52a3e0bdfa4b25108960a4c18aa6fcc1964605a30db4cfe9c6bdaf511ae925.jpg](../icml_results/1908_Learning%20Taxonomic%20Trees%20with%20Hierarchical%20Representation%20Regularization%20for%20Large%20Multimodal%20Models/tables/1b52a3e0bdfa4b25108960a4c18aa6fcc1964605a30db4cfe9c6bdaf511ae925.jpg)

![329db2db56d5c6fc776f708264d678c1b318261159cb024a083088f1e6110dda.jpg](../icml_results/1908_Learning%20Taxonomic%20Trees%20with%20Hierarchical%20Representation%20Regularization%20for%20Large%20Multimodal%20Models/tables/329db2db56d5c6fc776f708264d678c1b318261159cb024a083088f1e6110dda.jpg)

![32fff236893ea94cbe33d90fe4158d9c34ef3a1e5bede162dac9e812ed8d1fc1.jpg](../icml_results/1908_Learning%20Taxonomic%20Trees%20with%20Hierarchical%20Representation%20Regularization%20for%20Large%20Multimodal%20Models/tables/32fff236893ea94cbe33d90fe4158d9c34ef3a1e5bede162dac9e812ed8d1fc1.jpg)

![4e40a442ae15a0fd0ad58b931f11eb81b0376e7cbd9079cdc87b18f054792552.jpg](../icml_results/1908_Learning%20Taxonomic%20Trees%20with%20Hierarchical%20Representation%20Regularization%20for%20Large%20Multimodal%20Models/tables/4e40a442ae15a0fd0ad58b931f11eb81b0376e7cbd9079cdc87b18f054792552.jpg)

![a51830e9b2237a8e6f63a3368e48d3a011b843945bde1ab71c0084f1c4ad08cc.jpg](../icml_results/1908_Learning%20Taxonomic%20Trees%20with%20Hierarchical%20Representation%20Regularization%20for%20Large%20Multimodal%20Models/tables/a51830e9b2237a8e6f63a3368e48d3a011b843945bde1ab71c0084f1c4ad08cc.jpg)

![d17f3d8aacec770149f0a8b4b4f32b623804e6a8fbdab1af47254fd0084e0471.jpg](../icml_results/1908_Learning%20Taxonomic%20Trees%20with%20Hierarchical%20Representation%20Regularization%20for%20Large%20Multimodal%20Models/tables/d17f3d8aacec770149f0a8b4b4f32b623804e6a8fbdab1af47254fd0084e0471.jpg)

![f501a7b42b3c5217529a713cfef2b3a6722faa79556dce0e81954553c3cd2cea.jpg](../icml_results/1908_Learning%20Taxonomic%20Trees%20with%20Hierarchical%20Representation%20Regularization%20for%20Large%20Multimodal%20Models/tables/f501a7b42b3c5217529a713cfef2b3a6722faa79556dce0e81954553c3cd2cea.jpg)

<a id="paper-1884"></a>

## Reasoning-preserved Efficient Distillation of Large Language Models via Activation-aware Initialization


### Images

![1f8892551cf8d4866766de3ad7f07e8f494b47fc54caf1cb1c2f4cac8860504c.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/images/1f8892551cf8d4866766de3ad7f07e8f494b47fc54caf1cb1c2f4cac8860504c.jpg)

![2a5b3bc917db86e2cc5b680185e44651387e62fd63a500411cf39e7b006570d1.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/images/2a5b3bc917db86e2cc5b680185e44651387e62fd63a500411cf39e7b006570d1.jpg)

![320311c70d5bc2536e05655af8d0ba913dbaad01f859d5a53979193afc65e060.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/images/320311c70d5bc2536e05655af8d0ba913dbaad01f859d5a53979193afc65e060.jpg)

![329d69a77d3837cfb3cfb4f900e0831c789ee2ba5054f854c4991f1472cbbd07.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/images/329d69a77d3837cfb3cfb4f900e0831c789ee2ba5054f854c4991f1472cbbd07.jpg)

![985756000d0ee9887a3616c0b76b6d4c59cfc45607b9ab1394b512e8f3f70aa6.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/images/985756000d0ee9887a3616c0b76b6d4c59cfc45607b9ab1394b512e8f3f70aa6.jpg)

![e48cef346ba8eefa72897c101cde6a33f9c3ef4137f6ea37bdfeaa8ff0e32235.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/images/e48cef346ba8eefa72897c101cde6a33f9c3ef4137f6ea37bdfeaa8ff0e32235.jpg)

![f046ca36b34e169d24f0a6af27dcc867d81a2265627009a2db4ddfb227dd5143.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/images/f046ca36b34e169d24f0a6af27dcc867d81a2265627009a2db4ddfb227dd5143.jpg)

### Tables

![02fd28287f08705ec8be252677e1d0cee7e02dfe0f43a79b9a41ad568ad59206.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/tables/02fd28287f08705ec8be252677e1d0cee7e02dfe0f43a79b9a41ad568ad59206.jpg)

![0642d240489398d3863fc4329ef2d2dad71ebcd49fcbfbf4360bfeb81ff9f0ef.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/tables/0642d240489398d3863fc4329ef2d2dad71ebcd49fcbfbf4360bfeb81ff9f0ef.jpg)

![0e5df61be806bb69646b08eebb31bcb639c9caf28e5a53a02db106fb8176f728.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/tables/0e5df61be806bb69646b08eebb31bcb639c9caf28e5a53a02db106fb8176f728.jpg)

![1e6dcda2687019a0cfc1011dde992e9d7b47dec04f640660f54e6f445edd9309.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/tables/1e6dcda2687019a0cfc1011dde992e9d7b47dec04f640660f54e6f445edd9309.jpg)

![201f7b399a85ebb851af61b58a11d8d341d1e80105bc48cd29411d2bc81716e1.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/tables/201f7b399a85ebb851af61b58a11d8d341d1e80105bc48cd29411d2bc81716e1.jpg)

![3c83644b65ee03fdc05c5aebcc1da855dd9a03d3aeaba6813ef54f2f2f3fef96.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/tables/3c83644b65ee03fdc05c5aebcc1da855dd9a03d3aeaba6813ef54f2f2f3fef96.jpg)

![4f111a97d97006a1654e408ac4059861551b37b022eea35ee3680f0da4884e8a.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/tables/4f111a97d97006a1654e408ac4059861551b37b022eea35ee3680f0da4884e8a.jpg)

![612241b67eb0a2e05f5649c613bc28d47b705d6695c628b12f03615f0da39f99.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/tables/612241b67eb0a2e05f5649c613bc28d47b705d6695c628b12f03615f0da39f99.jpg)

![63d97012bf1b02c97988617aa93a9affc43807f52fc35a0a91a8ce9ad0a36b2c.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/tables/63d97012bf1b02c97988617aa93a9affc43807f52fc35a0a91a8ce9ad0a36b2c.jpg)

![65d47c64da340a8cd194b92733b0dc5641d71c67e15fd1ead7d01570b2abb379.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/tables/65d47c64da340a8cd194b92733b0dc5641d71c67e15fd1ead7d01570b2abb379.jpg)

![95ba8b4e01a5a724d5bc3ff785ddf8b26bbbdbd566c38fbabd44960de251c484.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/tables/95ba8b4e01a5a724d5bc3ff785ddf8b26bbbdbd566c38fbabd44960de251c484.jpg)

![9c7f8d8ef30490e1a40a5f3ca0f0b9adcbc512b309005923cacce0f51bf7b4cf.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/tables/9c7f8d8ef30490e1a40a5f3ca0f0b9adcbc512b309005923cacce0f51bf7b4cf.jpg)

![c0f340dc0f7fd4b6244fad9c1708929d50805f9278ea877fe985689eaab89f50.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/tables/c0f340dc0f7fd4b6244fad9c1708929d50805f9278ea877fe985689eaab89f50.jpg)

![cace6e0dfcbcf1c05d90da2d7b0a449ba382896ce6f90645c9fe290397b30850.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/tables/cace6e0dfcbcf1c05d90da2d7b0a449ba382896ce6f90645c9fe290397b30850.jpg)

![d80795ff9c4b201c51d932542a4e2f4c47e43563dd7b752f9624300f94324895.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/tables/d80795ff9c4b201c51d932542a4e2f4c47e43563dd7b752f9624300f94324895.jpg)

![f6be71c2d9889755f78470c75a85e75951edd3e330cc0923e242636ea800dae6.jpg](../icml_results/1909_Reasoning-preserved%20Efficient%20Distillation%20of%20Large%20Language%20Models%20via%20Activation-aware%20Initializat/tables/f6be71c2d9889755f78470c75a85e75951edd3e330cc0923e242636ea800dae6.jpg)

<a id="paper-1885"></a>

## One LR Doesn’t Fit All: Heavy-Tail Guided Layerwise Learning Rates for LLMs


### Images

![0ee10b9b3642d18f578b0ec2a2a41a40dd35396c828ec20a78fdde548c9940d5.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/images/0ee10b9b3642d18f578b0ec2a2a41a40dd35396c828ec20a78fdde548c9940d5.jpg)

![18f9fa8136060cd8219594e0fb667ca48941a6c24e977a08261d9b731dd338b7.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/images/18f9fa8136060cd8219594e0fb667ca48941a6c24e977a08261d9b731dd338b7.jpg)

![3a6d2b795f6c013addc3610c34d386a03b82510b8aace2ba28a8f86e1924a669.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/images/3a6d2b795f6c013addc3610c34d386a03b82510b8aace2ba28a8f86e1924a669.jpg)

![3d19a5066002bdeff290e5ab7c2c376fdaca94b370c76be7aba69f640b7695c9.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/images/3d19a5066002bdeff290e5ab7c2c376fdaca94b370c76be7aba69f640b7695c9.jpg)

![486b00c56f8ca8ff1529b2be64ad4a4a1cfffe7a7664fe1c88d9f8ab2b5a872a.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/images/486b00c56f8ca8ff1529b2be64ad4a4a1cfffe7a7664fe1c88d9f8ab2b5a872a.jpg)

![4ffbb07c05287138b799cbaddc2eae0b1e49f9a2a9ec39b631012eb6212f7643.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/images/4ffbb07c05287138b799cbaddc2eae0b1e49f9a2a9ec39b631012eb6212f7643.jpg)

![5ca33bc6ecc15f2c8b05ed25fe291eb89346cd8713d990b9e0a79673a7dce307.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/images/5ca33bc6ecc15f2c8b05ed25fe291eb89346cd8713d990b9e0a79673a7dce307.jpg)

![7ec074bb2348681b63a51bb34e9c4b871d5c840cf64037dc4bddf374777b71d4.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/images/7ec074bb2348681b63a51bb34e9c4b871d5c840cf64037dc4bddf374777b71d4.jpg)

![901427e87c3f5817ff06121f82f7df58583c7fba0e86adc79c5df02146e6a2b7.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/images/901427e87c3f5817ff06121f82f7df58583c7fba0e86adc79c5df02146e6a2b7.jpg)

![9cfc1404a48d683ebeb0ab2d9771d1c3823f5e7750b6b3fe7b7a1a8d51543438.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/images/9cfc1404a48d683ebeb0ab2d9771d1c3823f5e7750b6b3fe7b7a1a8d51543438.jpg)

![c836acb5619461bf78ca468a0be76fe242a0b2f36382cac7bd1c1c92b4c0e18e.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/images/c836acb5619461bf78ca468a0be76fe242a0b2f36382cac7bd1c1c92b4c0e18e.jpg)

![c8e52516f7b3d3d5338984f609bfbd5aba9968426aa4b1240024bed63c2a0fbc.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/images/c8e52516f7b3d3d5338984f609bfbd5aba9968426aa4b1240024bed63c2a0fbc.jpg)

![cfc46b74170d7ec11296ad3b199338f2a1597e7c97c390e34a6fecc67c71e978.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/images/cfc46b74170d7ec11296ad3b199338f2a1597e7c97c390e34a6fecc67c71e978.jpg)

![fee863eaf54cbc2d22d7d8771fb63f802823f705d768e347b333b354d665b7de.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/images/fee863eaf54cbc2d22d7d8771fb63f802823f705d768e347b333b354d665b7de.jpg)

### Tables

![31d5cffdbe77189a11108e2fe5d1ad428071fee9545ed07295f23199c8e8f42b.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/31d5cffdbe77189a11108e2fe5d1ad428071fee9545ed07295f23199c8e8f42b.jpg)

![37a0edd91bd370e5655da3b2f28269a2bb2b7259e8fc2d95ad3072f6685c9a3c.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/37a0edd91bd370e5655da3b2f28269a2bb2b7259e8fc2d95ad3072f6685c9a3c.jpg)

![3bf063f721b28e120324d28acfa5577626385781422673b1b280b5f4561418c4.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/3bf063f721b28e120324d28acfa5577626385781422673b1b280b5f4561418c4.jpg)

![44995584d506202d3bd177b9c58440453318963765fb2d1f4adc58c67825570e.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/44995584d506202d3bd177b9c58440453318963765fb2d1f4adc58c67825570e.jpg)

![6863a9c22a7e099a9ed6eaed5570dafd14fa411262d3056d43431aff0525d44b.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/6863a9c22a7e099a9ed6eaed5570dafd14fa411262d3056d43431aff0525d44b.jpg)

![6a32f0896b21aa27e1749175d44e6cd2cabc056930b29a2841348c299907bf5b.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/6a32f0896b21aa27e1749175d44e6cd2cabc056930b29a2841348c299907bf5b.jpg)

![81b89686ac1e3f1ce0a610a767006dcc92a3f767d22e064c95e622a5f1b2cef2.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/81b89686ac1e3f1ce0a610a767006dcc92a3f767d22e064c95e622a5f1b2cef2.jpg)

![81e91f84ea15f79679822d1039da542a7723500e48780aad799f040523be9de9.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/81e91f84ea15f79679822d1039da542a7723500e48780aad799f040523be9de9.jpg)

![95f5ddec7285b5126e7129bbc9e8d8144b48f8e1ff39bf4bb73a1f3eaeb241f3.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/95f5ddec7285b5126e7129bbc9e8d8144b48f8e1ff39bf4bb73a1f3eaeb241f3.jpg)

![aa4d2f841f688aa18ed7602c6338063160e72f062fc2d6d2b8e6278b96b19055.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/aa4d2f841f688aa18ed7602c6338063160e72f062fc2d6d2b8e6278b96b19055.jpg)

![b94925db3b0740ea06ef5ef96ab31bf9fcb91c251c1828a0c631fc9124dc3e71.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/b94925db3b0740ea06ef5ef96ab31bf9fcb91c251c1828a0c631fc9124dc3e71.jpg)

![c1325e648ccf0165c36508bf6341efdce62a214dbfdbc844fb390bdf593efe8d.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/c1325e648ccf0165c36508bf6341efdce62a214dbfdbc844fb390bdf593efe8d.jpg)

![c23b054f38435b3fc3e1ae16e676fe93df756e74bbffa35f8ef89b027ebc1997.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/c23b054f38435b3fc3e1ae16e676fe93df756e74bbffa35f8ef89b027ebc1997.jpg)

![d726aa1057d282df2c63afa26d4c2ce089978f4b088eed2a944a1530563ff5f0.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/d726aa1057d282df2c63afa26d4c2ce089978f4b088eed2a944a1530563ff5f0.jpg)

![d7439742907e5d99a42a2e01ed14ea901c3aa772e9d8034a50d3b8912f6c3531.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/d7439742907e5d99a42a2e01ed14ea901c3aa772e9d8034a50d3b8912f6c3531.jpg)

![fa85af2426aa3b4892b297ddc89d292ccdb48ff0186126650a5d8db11d225226.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/fa85af2426aa3b4892b297ddc89d292ccdb48ff0186126650a5d8db11d225226.jpg)

![fba5387c4ee5d70ae33d461a3848b19f228156e115e6c82c6025e80b014c1c6d.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/fba5387c4ee5d70ae33d461a3848b19f228156e115e6c82c6025e80b014c1c6d.jpg)

![ff927065205f02d89132cfb3b0fcc9ec0c8cf278b33cbb98212c7399acfa9077.jpg](../icml_results/1910_One%20LR%20Doesn%E2%80%99t%20Fit%20All_%20Heavy-Tail%20Guided%20Layerwise%20Learning%20Rates%20for%20LLMs/tables/ff927065205f02d89132cfb3b0fcc9ec0c8cf278b33cbb98212c7399acfa9077.jpg)

<a id="paper-1886"></a>

## Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks


### Images

![593519e6507658a42fe810593155616abb25e146ef0d655fd2f4f928f7e32b49.jpg](../icml_results/1911_Benchmarking%20Agent%20Memory%20in%20Interdependent%20Multi-Session%20Agentic%20Tasks/images/593519e6507658a42fe810593155616abb25e146ef0d655fd2f4f928f7e32b49.jpg)

![7491f27068b07acdae2e3bd24f3d6d7355ab1e6ed0e6c7ab0090b97701fc21b7.jpg](../icml_results/1911_Benchmarking%20Agent%20Memory%20in%20Interdependent%20Multi-Session%20Agentic%20Tasks/images/7491f27068b07acdae2e3bd24f3d6d7355ab1e6ed0e6c7ab0090b97701fc21b7.jpg)

![f16b2a0685cdaddc222d180ce7a38d8ce42ae4e4b69993799ccf17062f07bb33.jpg](../icml_results/1911_Benchmarking%20Agent%20Memory%20in%20Interdependent%20Multi-Session%20Agentic%20Tasks/images/f16b2a0685cdaddc222d180ce7a38d8ce42ae4e4b69993799ccf17062f07bb33.jpg)

### Tables

![096ea0d2492ffe4d2a46d3c87a8000645418beffb6187550ad4f52a72bbf9fb5.jpg](../icml_results/1911_Benchmarking%20Agent%20Memory%20in%20Interdependent%20Multi-Session%20Agentic%20Tasks/tables/096ea0d2492ffe4d2a46d3c87a8000645418beffb6187550ad4f52a72bbf9fb5.jpg)

![0f7732947b47691e5126b220ee05e03aed25c8ae4517edbb2ab629c1173b149a.jpg](../icml_results/1911_Benchmarking%20Agent%20Memory%20in%20Interdependent%20Multi-Session%20Agentic%20Tasks/tables/0f7732947b47691e5126b220ee05e03aed25c8ae4517edbb2ab629c1173b149a.jpg)

![16b91f0aed1bd3b38a4a2ef08a68266a31d95e976e2e4031ffd3063404d083fe.jpg](../icml_results/1911_Benchmarking%20Agent%20Memory%20in%20Interdependent%20Multi-Session%20Agentic%20Tasks/tables/16b91f0aed1bd3b38a4a2ef08a68266a31d95e976e2e4031ffd3063404d083fe.jpg)

![20ac298ec7dbbe95e4314eb0737473d4dcf358354d7ce845c770080d3600cbd7.jpg](../icml_results/1911_Benchmarking%20Agent%20Memory%20in%20Interdependent%20Multi-Session%20Agentic%20Tasks/tables/20ac298ec7dbbe95e4314eb0737473d4dcf358354d7ce845c770080d3600cbd7.jpg)

![351888965ddf5fdeb3202ddd9b93d9ae0b9a7c351506d7d3ec650f7714d9f7c1.jpg](../icml_results/1911_Benchmarking%20Agent%20Memory%20in%20Interdependent%20Multi-Session%20Agentic%20Tasks/tables/351888965ddf5fdeb3202ddd9b93d9ae0b9a7c351506d7d3ec650f7714d9f7c1.jpg)

![7476a21b2b1a286576ae060c8782270704ee9cada1e8e6cd60c96cbd020ed84c.jpg](../icml_results/1911_Benchmarking%20Agent%20Memory%20in%20Interdependent%20Multi-Session%20Agentic%20Tasks/tables/7476a21b2b1a286576ae060c8782270704ee9cada1e8e6cd60c96cbd020ed84c.jpg)

![892cec106872768b330c7a9fad38ea3d0aa1dec187958146dc332d75f55834ad.jpg](../icml_results/1911_Benchmarking%20Agent%20Memory%20in%20Interdependent%20Multi-Session%20Agentic%20Tasks/tables/892cec106872768b330c7a9fad38ea3d0aa1dec187958146dc332d75f55834ad.jpg)

![9c9c23b031b5b247f3f83991b39c20b5fad545cdea5a00ed399841414d4653f7.jpg](../icml_results/1911_Benchmarking%20Agent%20Memory%20in%20Interdependent%20Multi-Session%20Agentic%20Tasks/tables/9c9c23b031b5b247f3f83991b39c20b5fad545cdea5a00ed399841414d4653f7.jpg)

![b5b280bc350bb7cefcb30f30da759709fe5044557208d1d324f8018b2576d45b.jpg](../icml_results/1911_Benchmarking%20Agent%20Memory%20in%20Interdependent%20Multi-Session%20Agentic%20Tasks/tables/b5b280bc350bb7cefcb30f30da759709fe5044557208d1d324f8018b2576d45b.jpg)

![ccf66f4df5fb8ccef0fc33691f5361f32b692b6e6b9323daffca1136e504b992.jpg](../icml_results/1911_Benchmarking%20Agent%20Memory%20in%20Interdependent%20Multi-Session%20Agentic%20Tasks/tables/ccf66f4df5fb8ccef0fc33691f5361f32b692b6e6b9323daffca1136e504b992.jpg)

![e74388790405034955c21001075c00f114359657ada708e1fe1668ce2a05475b.jpg](../icml_results/1911_Benchmarking%20Agent%20Memory%20in%20Interdependent%20Multi-Session%20Agentic%20Tasks/tables/e74388790405034955c21001075c00f114359657ada708e1fe1668ce2a05475b.jpg)

![f3769babd56980533f7c091e6cb534be80a7eea1079694d73d6657c17778e1e2.jpg](../icml_results/1911_Benchmarking%20Agent%20Memory%20in%20Interdependent%20Multi-Session%20Agentic%20Tasks/tables/f3769babd56980533f7c091e6cb534be80a7eea1079694d73d6657c17778e1e2.jpg)

![f3bc9a82226db290ecd4ba7d0023dfa00e2bcedb5c00b5e725111356994acd62.jpg](../icml_results/1911_Benchmarking%20Agent%20Memory%20in%20Interdependent%20Multi-Session%20Agentic%20Tasks/tables/f3bc9a82226db290ecd4ba7d0023dfa00e2bcedb5c00b5e725111356994acd62.jpg)

<a id="paper-1887"></a>

## Bend the Basics: Degradation-Aware Deformable Tokenization for All-in-One Image Restoration


### Images

![081dbb4d32b8ef2f9853c679b1f5d6dcc38da30b10b60521b104f2225d5b033f.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/images/081dbb4d32b8ef2f9853c679b1f5d6dcc38da30b10b60521b104f2225d5b033f.jpg)

![0b408694284da64c769e53b16a012c748132977e2d789923be19d3943201409c.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/images/0b408694284da64c769e53b16a012c748132977e2d789923be19d3943201409c.jpg)

![0d4971569d6d5a9f2739db812de6726a5827d9ea8495e9391b881773ba10b207.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/images/0d4971569d6d5a9f2739db812de6726a5827d9ea8495e9391b881773ba10b207.jpg)

![127fc09f4158e792c4a06b79227fc86e547c040d2e599c7eef2debc9d64fa251.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/images/127fc09f4158e792c4a06b79227fc86e547c040d2e599c7eef2debc9d64fa251.jpg)

![35b51eeddbb319618344a0e0aeef47e50576837a1d0de64a41a0eb68158677e5.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/images/35b51eeddbb319618344a0e0aeef47e50576837a1d0de64a41a0eb68158677e5.jpg)

![35d1c6f830f5cccfaa2501cefb2558bf7251615c530b4738094a1d8b2fb2e2e6.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/images/35d1c6f830f5cccfaa2501cefb2558bf7251615c530b4738094a1d8b2fb2e2e6.jpg)

![3ecfbb5dbe48050e543bde8813386f88fc5128714d3f512b06ac8e15686f4f62.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/images/3ecfbb5dbe48050e543bde8813386f88fc5128714d3f512b06ac8e15686f4f62.jpg)

![50800f765e195b5fed8586b5398f72e892037aa41c70f78e09b56a3fdb1c5921.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/images/50800f765e195b5fed8586b5398f72e892037aa41c70f78e09b56a3fdb1c5921.jpg)

![5b9493ef866085b213c0128e0752ecdd70e16a45acc6cbcd423dcd7b4535b973.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/images/5b9493ef866085b213c0128e0752ecdd70e16a45acc6cbcd423dcd7b4535b973.jpg)

![98f3d2961fb0fb4d86913012debf009c591f49b48ba2c55c6c572353ce98a63a.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/images/98f3d2961fb0fb4d86913012debf009c591f49b48ba2c55c6c572353ce98a63a.jpg)

![ad11a84247451df2b67dd8d70f296deb93b1db098b0e3500331a3d6a3b6ed5fb.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/images/ad11a84247451df2b67dd8d70f296deb93b1db098b0e3500331a3d6a3b6ed5fb.jpg)

![f539ae5202ad597924edc2d8f03b2f1a664b0fd1e85924cee2fd208b26e8f547.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/images/f539ae5202ad597924edc2d8f03b2f1a664b0fd1e85924cee2fd208b26e8f547.jpg)

### Tables

![17d8b9965f86d7a5fce7d69ee4ee046735811cf9cc921945c5f15c9f63289f23.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/tables/17d8b9965f86d7a5fce7d69ee4ee046735811cf9cc921945c5f15c9f63289f23.jpg)

![595fe77d5a0474cf0fbb54d2dab71f74d23de4b90d70ff209faf5bee73f4b62e.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/tables/595fe77d5a0474cf0fbb54d2dab71f74d23de4b90d70ff209faf5bee73f4b62e.jpg)

![59f2590980b5117a8147afd5494b4a47b8d34d8fa5d617053a56709328b7e27b.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/tables/59f2590980b5117a8147afd5494b4a47b8d34d8fa5d617053a56709328b7e27b.jpg)

![6bb58935bddf9b06c316bc15de3ef3fb73f7e24df637a4416c8ce8c70ba1771d.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/tables/6bb58935bddf9b06c316bc15de3ef3fb73f7e24df637a4416c8ce8c70ba1771d.jpg)

![86b39eda3f8a688237577251e9e3b63dd90145a3c964dcebff0cbd8aefb732a9.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/tables/86b39eda3f8a688237577251e9e3b63dd90145a3c964dcebff0cbd8aefb732a9.jpg)

![99adcf994db65172715588505f0a364594874886f5795e712afa966af3f7cc8c.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/tables/99adcf994db65172715588505f0a364594874886f5795e712afa966af3f7cc8c.jpg)

![a05c5d0f713ddc4290f69540a8531e0a3496ccd2723d8e803fef8ae99563a2ce.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/tables/a05c5d0f713ddc4290f69540a8531e0a3496ccd2723d8e803fef8ae99563a2ce.jpg)

![a85129121e118f2ad305a027a282ffceb2eefa41f2ac63dc809748654e473317.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/tables/a85129121e118f2ad305a027a282ffceb2eefa41f2ac63dc809748654e473317.jpg)

![e5fdba722be9050abb61f8f2b64d3ec32ae52817956f53615854b76d8293be60.jpg](../icml_results/1912_Bend%20the%20Basics_%20Degradation-Aware%20Deformable%20Tokenization%20for%20All-in-One%20Image%20Restoration/tables/e5fdba722be9050abb61f8f2b64d3ec32ae52817956f53615854b76d8293be60.jpg)

<a id="paper-1888"></a>

## Motion Dynamics Learning for Few-Shot Embodied Adaptation


### Images

![0d50f18656406572a9ca9b50f12db8e16bae6f44a2d2cfdd3feeaee70130cd37.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/0d50f18656406572a9ca9b50f12db8e16bae6f44a2d2cfdd3feeaee70130cd37.jpg)

![17b2545015e4da15d0934e527b7d821ae4ff4ee044909ab6f4840e083ab63a5c.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/17b2545015e4da15d0934e527b7d821ae4ff4ee044909ab6f4840e083ab63a5c.jpg)

![1a2931eec3b37079bb85d73bad2a085a0c3216450451ea802eeeacfab04ef923.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/1a2931eec3b37079bb85d73bad2a085a0c3216450451ea802eeeacfab04ef923.jpg)

![24e2347bfc10d3138b0a874b3faec376cfce777dec9b7f6feb3cb22e50cf0563.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/24e2347bfc10d3138b0a874b3faec376cfce777dec9b7f6feb3cb22e50cf0563.jpg)

![2d7575d2086a15f867c05fadc1487b08cdd2ea888fc4b35c4802bd2ff63e34e8.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/2d7575d2086a15f867c05fadc1487b08cdd2ea888fc4b35c4802bd2ff63e34e8.jpg)

![32f1a083aa80e8a2457f19e23fb9850bfd435923d728a60a9d31d9eb98fbe846.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/32f1a083aa80e8a2457f19e23fb9850bfd435923d728a60a9d31d9eb98fbe846.jpg)

![361e45ee0770a577563ac82945d43b8c82a28e3aca4fb4d4d5dc560c31aeaddd.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/361e45ee0770a577563ac82945d43b8c82a28e3aca4fb4d4d5dc560c31aeaddd.jpg)

![405b48946bf6ebf2bb8401abdf7b4d2aa85bcb3735c6328fdb905626cb420023.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/405b48946bf6ebf2bb8401abdf7b4d2aa85bcb3735c6328fdb905626cb420023.jpg)

![4ef188b951d595238e6a091f6b56bdcd83838b561c95fd7a1dc994d8c44196ef.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/4ef188b951d595238e6a091f6b56bdcd83838b561c95fd7a1dc994d8c44196ef.jpg)

![58fd03da1bdf4848f9b10404218d3706b75d6a354d1aa7a8acdf442afe345868.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/58fd03da1bdf4848f9b10404218d3706b75d6a354d1aa7a8acdf442afe345868.jpg)

![5e29933615b2d4518aaaab7b3e87eb5ab64803534b74c64b3202f3f622e838a2.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/5e29933615b2d4518aaaab7b3e87eb5ab64803534b74c64b3202f3f622e838a2.jpg)

![626c07e98230638c55c5bb84f825b9b8d005db85315423e0d9f949bfcaf4de83.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/626c07e98230638c55c5bb84f825b9b8d005db85315423e0d9f949bfcaf4de83.jpg)

![6353d04e77eb0594d02da0c82924b4b80040eee3c91c26bd06a98b441bbb391a.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/6353d04e77eb0594d02da0c82924b4b80040eee3c91c26bd06a98b441bbb391a.jpg)

![67dbcd18d5c6324321ab20d6a6d371329a7401ad5077ee8e3d4ff9943693525d.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/67dbcd18d5c6324321ab20d6a6d371329a7401ad5077ee8e3d4ff9943693525d.jpg)

![69b916b4d3ee4a8f8b8605ef8e06a18ed3c911b1ac744866ab40d1e25ca68501.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/69b916b4d3ee4a8f8b8605ef8e06a18ed3c911b1ac744866ab40d1e25ca68501.jpg)

![6fa79e20487ae72e83dbaf2be977b2d837c2f1a1f0ed4afc50b6aa991a338d67.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/6fa79e20487ae72e83dbaf2be977b2d837c2f1a1f0ed4afc50b6aa991a338d67.jpg)

![83f0e7b155cad147dcfeb637dc6ff264ec2cad9778ebd773527cccd1a50e03fd.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/83f0e7b155cad147dcfeb637dc6ff264ec2cad9778ebd773527cccd1a50e03fd.jpg)

![897dc9a69c991281664e8293bb0632011a72e23c9a7b212a7f5e5ef94ebac9a6.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/897dc9a69c991281664e8293bb0632011a72e23c9a7b212a7f5e5ef94ebac9a6.jpg)

![9a00c1c81c48e63f7f74d1dd11040afebfe5d96300837b953bbf3597ae97f300.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/9a00c1c81c48e63f7f74d1dd11040afebfe5d96300837b953bbf3597ae97f300.jpg)

![9c098c2a421b62c4f1317e7aaa4667139b5a4d4faf645fce44e523ec5a98f947.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/9c098c2a421b62c4f1317e7aaa4667139b5a4d4faf645fce44e523ec5a98f947.jpg)

![b3bd96454973210d2063c26c1bce0fea8bca0600e1baff16e40501e9abb81962.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/b3bd96454973210d2063c26c1bce0fea8bca0600e1baff16e40501e9abb81962.jpg)

![bc17f9437e1ffdcfd4988602c5d31a61939a949dbbb7074d27064218f498cdde.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/bc17f9437e1ffdcfd4988602c5d31a61939a949dbbb7074d27064218f498cdde.jpg)

![c1b7d42c2b92ddff589265c1b4f3271baae389592c87b5c820209ef21534b2b5.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/c1b7d42c2b92ddff589265c1b4f3271baae389592c87b5c820209ef21534b2b5.jpg)

![e654e43b5ca69bceb6546d2dea0b547dd15d6fcdfb5765b03d6346ecb9aa2689.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/e654e43b5ca69bceb6546d2dea0b547dd15d6fcdfb5765b03d6346ecb9aa2689.jpg)

![e68295ca0018660905384f0d11fbcb2c7f18bb071cde0418fb0125d4a0cfb65e.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/e68295ca0018660905384f0d11fbcb2c7f18bb071cde0418fb0125d4a0cfb65e.jpg)

![fda2c645ecd54b61252897056bdb3408eb9134e92db3f51933f380744e819d98.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/images/fda2c645ecd54b61252897056bdb3408eb9134e92db3f51933f380744e819d98.jpg)

### Tables

![03cc19b3114f30c090cd30a78ae164d1074501f740051b121a3fc2b3b62c239d.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/tables/03cc19b3114f30c090cd30a78ae164d1074501f740051b121a3fc2b3b62c239d.jpg)

![0dacba09a007ffca23559d151a2f967b87cbf3ce57b30191137d4369849943f3.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/tables/0dacba09a007ffca23559d151a2f967b87cbf3ce57b30191137d4369849943f3.jpg)

![1098095c334ecd35a52bd109d05bff70e68c73960b6bd0411fe7c02b177a709e.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/tables/1098095c334ecd35a52bd109d05bff70e68c73960b6bd0411fe7c02b177a709e.jpg)

![4a5fe3706a1b33881deee55361f9fc3c79ca5081a51ff2fc9ec8073db9b96360.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/tables/4a5fe3706a1b33881deee55361f9fc3c79ca5081a51ff2fc9ec8073db9b96360.jpg)

![4c2643049fba6a0524ae0a818a68ed9f09733965002464c593fe2ece7d938535.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/tables/4c2643049fba6a0524ae0a818a68ed9f09733965002464c593fe2ece7d938535.jpg)

![74bb55339062c950dee8a882662a4f57333c119956d89243f4861d14b1c36317.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/tables/74bb55339062c950dee8a882662a4f57333c119956d89243f4861d14b1c36317.jpg)

![ab379a9d9848225ffeebce10821f6157f82b40c8f60a8df92485f745622a122f.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/tables/ab379a9d9848225ffeebce10821f6157f82b40c8f60a8df92485f745622a122f.jpg)

![b40ac00e693a06e7e064ed080827e912f88fb6e3cdc5e6857dc115adcb6853f4.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/tables/b40ac00e693a06e7e064ed080827e912f88fb6e3cdc5e6857dc115adcb6853f4.jpg)

![f3e2bc0578ec3486481ebd2818975e7af23aee41a46a17c6c8f6e2ea2820dfa0.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/tables/f3e2bc0578ec3486481ebd2818975e7af23aee41a46a17c6c8f6e2ea2820dfa0.jpg)

![ffad56b6980fbccffdb077ddd132769a92ffda81d46a64c6dbbeeca2b63bb226.jpg](../icml_results/1913_Motion%20Dynamics%20Learning%20for%20Few-Shot%20Embodied%20Adaptation/tables/ffad56b6980fbccffdb077ddd132769a92ffda81d46a64c6dbbeeca2b63bb226.jpg)

<a id="paper-1889"></a>

## RLSF-V: Mitigating Hallucinations in MLLMs via Fuzzy Semantic Self-Feedback


### Images

![0fbb425d0a1b65b6b892f40c4bb7ae2d067e3ab1ff1f9547c181e7d9cb0b0bdd.jpg](../icml_results/1914_RLSF-V_%20Mitigating%20Hallucinations%20in%20MLLMs%20via%20Fuzzy%20Semantic%20Self-Feedback/images/0fbb425d0a1b65b6b892f40c4bb7ae2d067e3ab1ff1f9547c181e7d9cb0b0bdd.jpg)

![24e188d41df0aa84f521f71dddbbceaf47e736a655fa13fbedced94dba669899.jpg](../icml_results/1914_RLSF-V_%20Mitigating%20Hallucinations%20in%20MLLMs%20via%20Fuzzy%20Semantic%20Self-Feedback/images/24e188d41df0aa84f521f71dddbbceaf47e736a655fa13fbedced94dba669899.jpg)

![5d1f3bdefc3e283dc0cef522fb15cf4ea05ab60f011a6d8be020d682a4a400b8.jpg](../icml_results/1914_RLSF-V_%20Mitigating%20Hallucinations%20in%20MLLMs%20via%20Fuzzy%20Semantic%20Self-Feedback/images/5d1f3bdefc3e283dc0cef522fb15cf4ea05ab60f011a6d8be020d682a4a400b8.jpg)

![626077e214320b39761139f8f7cf8baf7de693224b987d3de7d872e0b2c99846.jpg](../icml_results/1914_RLSF-V_%20Mitigating%20Hallucinations%20in%20MLLMs%20via%20Fuzzy%20Semantic%20Self-Feedback/images/626077e214320b39761139f8f7cf8baf7de693224b987d3de7d872e0b2c99846.jpg)

![7f3bde267bd1762817acc11bf11d39cd57aab5bd3a19a1f4b8afebc9ef7786cc.jpg](../icml_results/1914_RLSF-V_%20Mitigating%20Hallucinations%20in%20MLLMs%20via%20Fuzzy%20Semantic%20Self-Feedback/images/7f3bde267bd1762817acc11bf11d39cd57aab5bd3a19a1f4b8afebc9ef7786cc.jpg)

![800fd38670d436850badce6aed46ff0b7b3c39ba2c78e46e2e594332cfb4fc97.jpg](../icml_results/1914_RLSF-V_%20Mitigating%20Hallucinations%20in%20MLLMs%20via%20Fuzzy%20Semantic%20Self-Feedback/images/800fd38670d436850badce6aed46ff0b7b3c39ba2c78e46e2e594332cfb4fc97.jpg)

![888786d1329693f106fa28d79bca3e974d7f6026daffb2b042bdfe437fdb30c5.jpg](../icml_results/1914_RLSF-V_%20Mitigating%20Hallucinations%20in%20MLLMs%20via%20Fuzzy%20Semantic%20Self-Feedback/images/888786d1329693f106fa28d79bca3e974d7f6026daffb2b042bdfe437fdb30c5.jpg)

![97831e6b87cec41a490e88b5e23f0f45cffe03cbc77fd33d888d13dfe1be8cb8.jpg](../icml_results/1914_RLSF-V_%20Mitigating%20Hallucinations%20in%20MLLMs%20via%20Fuzzy%20Semantic%20Self-Feedback/images/97831e6b87cec41a490e88b5e23f0f45cffe03cbc77fd33d888d13dfe1be8cb8.jpg)

![98b34080d28bda167413d3eeb6f6b4eb0e5126ae6f8cc3a6d817f1cfde6476cf.jpg](../icml_results/1914_RLSF-V_%20Mitigating%20Hallucinations%20in%20MLLMs%20via%20Fuzzy%20Semantic%20Self-Feedback/images/98b34080d28bda167413d3eeb6f6b4eb0e5126ae6f8cc3a6d817f1cfde6476cf.jpg)

![d5356e4f0aab78255d838d450db760584a56d26f68390ab86978bf31ce70fb87.jpg](../icml_results/1914_RLSF-V_%20Mitigating%20Hallucinations%20in%20MLLMs%20via%20Fuzzy%20Semantic%20Self-Feedback/images/d5356e4f0aab78255d838d450db760584a56d26f68390ab86978bf31ce70fb87.jpg)

![dfda3b0b0924a9e91d713020b56e29d9ab5b1e167aafc435ee4385d0eb17094a.jpg](../icml_results/1914_RLSF-V_%20Mitigating%20Hallucinations%20in%20MLLMs%20via%20Fuzzy%20Semantic%20Self-Feedback/images/dfda3b0b0924a9e91d713020b56e29d9ab5b1e167aafc435ee4385d0eb17094a.jpg)

![fe012764e3021fbd5086815194b64bcdc5aa09f49559cd105001175afe3bec31.jpg](../icml_results/1914_RLSF-V_%20Mitigating%20Hallucinations%20in%20MLLMs%20via%20Fuzzy%20Semantic%20Self-Feedback/images/fe012764e3021fbd5086815194b64bcdc5aa09f49559cd105001175afe3bec31.jpg)

### Tables

![128f627e413ab6b5fc1743b52e973066f6c70fa0d7e69e66d8ecb4ca6f6f259e.jpg](../icml_results/1914_RLSF-V_%20Mitigating%20Hallucinations%20in%20MLLMs%20via%20Fuzzy%20Semantic%20Self-Feedback/tables/128f627e413ab6b5fc1743b52e973066f6c70fa0d7e69e66d8ecb4ca6f6f259e.jpg)

![32abdc20100ddf8c42fd11b2bb8bd393f59e94a18adf4a975a14c68ee1a7edd6.jpg](../icml_results/1914_RLSF-V_%20Mitigating%20Hallucinations%20in%20MLLMs%20via%20Fuzzy%20Semantic%20Self-Feedback/tables/32abdc20100ddf8c42fd11b2bb8bd393f59e94a18adf4a975a14c68ee1a7edd6.jpg)

![3c3aa76489b92d176623f907663b2e0c27a0362dd5317479a0c035d73532bb8a.jpg](../icml_results/1914_RLSF-V_%20Mitigating%20Hallucinations%20in%20MLLMs%20via%20Fuzzy%20Semantic%20Self-Feedback/tables/3c3aa76489b92d176623f907663b2e0c27a0362dd5317479a0c035d73532bb8a.jpg)

![5f8be0fe2a0a36f51f1b6508775e251141c8c2326b6e05d93d42a276924a204d.jpg](../icml_results/1914_RLSF-V_%20Mitigating%20Hallucinations%20in%20MLLMs%20via%20Fuzzy%20Semantic%20Self-Feedback/tables/5f8be0fe2a0a36f51f1b6508775e251141c8c2326b6e05d93d42a276924a204d.jpg)

![c5349c2987f8da620cf0351465df20f47e019e6472b9635e7027cb80c050ea54.jpg](../icml_results/1914_RLSF-V_%20Mitigating%20Hallucinations%20in%20MLLMs%20via%20Fuzzy%20Semantic%20Self-Feedback/tables/c5349c2987f8da620cf0351465df20f47e019e6472b9635e7027cb80c050ea54.jpg)

<a id="paper-1890"></a>

## FocalPolicy: Frequency-Optimized Chunking and Locally Anchored Flow Matching for Coherent Visuomotor Policy


### Images

![2053578e0fed8b80ca1fe38c3019768fefc789f1fd05d2a542bf2b846919d885.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/images/2053578e0fed8b80ca1fe38c3019768fefc789f1fd05d2a542bf2b846919d885.jpg)

![227b586a732801eb73a80fd32d5ca4164991da3df07cb2f612331982cbea841e.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/images/227b586a732801eb73a80fd32d5ca4164991da3df07cb2f612331982cbea841e.jpg)

![2cd9410bacd7e9ff855b7dbc9d6a85fcb4e071e4ca61a2566c825cfa87dc9287.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/images/2cd9410bacd7e9ff855b7dbc9d6a85fcb4e071e4ca61a2566c825cfa87dc9287.jpg)

![3170f8b7b5536d4b88e4be5a2a00d31d3eb22166cd7c7f8398739789bc702d7e.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/images/3170f8b7b5536d4b88e4be5a2a00d31d3eb22166cd7c7f8398739789bc702d7e.jpg)

![53659bc299491c89cd72af8365b59196775262e709012ba333a70831ff6de397.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/images/53659bc299491c89cd72af8365b59196775262e709012ba333a70831ff6de397.jpg)

![550468802efb801ae046b09b5740786cf6bcd3458982af301c7932308c09684e.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/images/550468802efb801ae046b09b5740786cf6bcd3458982af301c7932308c09684e.jpg)

![6a9d2e9d11176894ac1f7a43fd822f02527db38084f198f90928cc18a9f471ed.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/images/6a9d2e9d11176894ac1f7a43fd822f02527db38084f198f90928cc18a9f471ed.jpg)

![8e1bea3ced9876dd94880db5bb2558fc5a0780eb2c8e9bc81d6bae8f279ba100.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/images/8e1bea3ced9876dd94880db5bb2558fc5a0780eb2c8e9bc81d6bae8f279ba100.jpg)

![9b800a515e60a67eb405a7e3a74d1d0886102dbe9033129920e1a591a54528c1.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/images/9b800a515e60a67eb405a7e3a74d1d0886102dbe9033129920e1a591a54528c1.jpg)

![fefeb838dd574e48ed16587c3e1ec93d4d5528501f6c2a135bec57db0bdb8499.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/images/fefeb838dd574e48ed16587c3e1ec93d4d5528501f6c2a135bec57db0bdb8499.jpg)

### Tables

![12ab3f28b862e986e5500522bc549fb43af7f4cfee0b501bfdcd4117b231bbc9.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/tables/12ab3f28b862e986e5500522bc549fb43af7f4cfee0b501bfdcd4117b231bbc9.jpg)

![14cd2a9f4b1df330487e55a5af8ce161908f2ea60850eb7f43bf9c7226ea0cc5.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/tables/14cd2a9f4b1df330487e55a5af8ce161908f2ea60850eb7f43bf9c7226ea0cc5.jpg)

![1608b0eeb1ad726c2d4a3d88454d4e0605033e4572e786921f57b22b62b0c2e2.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/tables/1608b0eeb1ad726c2d4a3d88454d4e0605033e4572e786921f57b22b62b0c2e2.jpg)

![27885f34cb5909c7f4e9e87259cc0fcf3f51947ccce013c7a8fc70a3ca9699d4.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/tables/27885f34cb5909c7f4e9e87259cc0fcf3f51947ccce013c7a8fc70a3ca9699d4.jpg)

![292280a71b1daa56109797b9fede2248277e6772bc4f3d6bd1833a0de43546b1.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/tables/292280a71b1daa56109797b9fede2248277e6772bc4f3d6bd1833a0de43546b1.jpg)

![2ed80c9188f37f7a34672e9e0b15f8422966577a210441235a698a4e69620f9b.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/tables/2ed80c9188f37f7a34672e9e0b15f8422966577a210441235a698a4e69620f9b.jpg)

![4fbc9196c3607e99b197c69819b8c9702dd23ab19e0853ca0df705ab213a6b81.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/tables/4fbc9196c3607e99b197c69819b8c9702dd23ab19e0853ca0df705ab213a6b81.jpg)

![5850e24754ee52e2d1c1bc8282261ddb316e46301b6673ecaf240323f01de2c8.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/tables/5850e24754ee52e2d1c1bc8282261ddb316e46301b6673ecaf240323f01de2c8.jpg)

![5e36791b8b01ff7238732943afd2fad8072dd70d4b5e4fd7aeef3e4e953413b7.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/tables/5e36791b8b01ff7238732943afd2fad8072dd70d4b5e4fd7aeef3e4e953413b7.jpg)

![7047b24c845a10a27716e9d7d25295b7200370a56c0b3a8719875efa532411ca.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/tables/7047b24c845a10a27716e9d7d25295b7200370a56c0b3a8719875efa532411ca.jpg)

![9bafcb0c709921db590f081ae77ee93256d877fadba8ad088624d2209e296beb.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/tables/9bafcb0c709921db590f081ae77ee93256d877fadba8ad088624d2209e296beb.jpg)

![b1a84087088e453e80cf3f7c7e420e0f109edf636742dac7b1062b7bca5ce53b.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/tables/b1a84087088e453e80cf3f7c7e420e0f109edf636742dac7b1062b7bca5ce53b.jpg)

![c39a370cf6f302a682f1c0ace7dde31b3f8d57b2137f7281ce1326ff81e40c69.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/tables/c39a370cf6f302a682f1c0ace7dde31b3f8d57b2137f7281ce1326ff81e40c69.jpg)

![cf436f116f7a69f3f4c368ec1349f83c3879e6527a6fe8b9df56d8a398553cc2.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/tables/cf436f116f7a69f3f4c368ec1349f83c3879e6527a6fe8b9df56d8a398553cc2.jpg)

![d3d1e6b85eb8e14940c2491701c9dd943e9a5d9f190efeabbb8dd042e8e42306.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/tables/d3d1e6b85eb8e14940c2491701c9dd943e9a5d9f190efeabbb8dd042e8e42306.jpg)

![ed083d248268ef94303370915ab57a8f64cbfea6bcc730b0bac9b500e3eedbec.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/tables/ed083d248268ef94303370915ab57a8f64cbfea6bcc730b0bac9b500e3eedbec.jpg)

![f2bff35dc54321b4738de0b94e0e6f21d15a7328e235bb93f333d0b278374461.jpg](../icml_results/1915_FocalPolicy_%20Frequency-Optimized%20Chunking%20and%20Locally%20Anchored%20Flow%20Matching%20for%20Coherent%20Visuomotor/tables/f2bff35dc54321b4738de0b94e0e6f21d15a7328e235bb93f333d0b278374461.jpg)

<a id="paper-1891"></a>

## On Stable Long-Form Generation: Benchmarking and Mitigating Length Volatility


### Images

![05989aeaa0b96fb0acfc4f93beb8e9ce9de99457d2e9c71ee228ed470200f86f.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/05989aeaa0b96fb0acfc4f93beb8e9ce9de99457d2e9c71ee228ed470200f86f.jpg)

![1410e9be7608c8e927d518f5426343baa24bf4906dcaf1cdc9768a74d7bf6cb3.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/1410e9be7608c8e927d518f5426343baa24bf4906dcaf1cdc9768a74d7bf6cb3.jpg)

![186cc5f1e3fce607fc504e5da06ce8822ef1ae7d85fa08b48dc3fe0f2c5ce6b7.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/186cc5f1e3fce607fc504e5da06ce8822ef1ae7d85fa08b48dc3fe0f2c5ce6b7.jpg)

![1cd1e6efc9d17a8f016fd4299beea313dbb360e991222a42100c9e6ee5199777.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/1cd1e6efc9d17a8f016fd4299beea313dbb360e991222a42100c9e6ee5199777.jpg)

![1dc815ab7a4d29b12c2fff7ff59839ac7071cac8e185403daf3adde6907e9b45.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/1dc815ab7a4d29b12c2fff7ff59839ac7071cac8e185403daf3adde6907e9b45.jpg)

![25c078e950a4d52b1dfd3304ad5695b882cff38b3a2d60ed5cb46f4fe12aa084.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/25c078e950a4d52b1dfd3304ad5695b882cff38b3a2d60ed5cb46f4fe12aa084.jpg)

![2631da2a2dd7b235f1c8b3648dfc76eaf45f36582959e5acf02fc12307165c9d.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/2631da2a2dd7b235f1c8b3648dfc76eaf45f36582959e5acf02fc12307165c9d.jpg)

![42d0c89c574555ccef6455356488e8b847f1de1139e661c492022f1b9ca5ce36.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/42d0c89c574555ccef6455356488e8b847f1de1139e661c492022f1b9ca5ce36.jpg)

![455a4cfe2589ec2a9ec9fa2a0bcda6f787fb73a73f0795a4aa6da5919f76014b.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/455a4cfe2589ec2a9ec9fa2a0bcda6f787fb73a73f0795a4aa6da5919f76014b.jpg)

![4687b5532d7d63ca66858dd7b11c2fb2cc0583a72e7d54fa9dee1ef94ece6035.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/4687b5532d7d63ca66858dd7b11c2fb2cc0583a72e7d54fa9dee1ef94ece6035.jpg)

![5a57930cf27dd7463ff3b91edc3623f1242679da314f7830721f80cf3f49d8b3.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/5a57930cf27dd7463ff3b91edc3623f1242679da314f7830721f80cf3f49d8b3.jpg)

![6b0b55a90eab5070df974d1751a51461728f953a84599ebc8f40324b156cd288.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/6b0b55a90eab5070df974d1751a51461728f953a84599ebc8f40324b156cd288.jpg)

![7115dd70cb61d5aae7db6d3855819c99e6453ef802d56b92655fb6162d85039c.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/7115dd70cb61d5aae7db6d3855819c99e6453ef802d56b92655fb6162d85039c.jpg)

![737caad8acd7aa0c147a979421cf154022b19ec733c26677c384160196e74d27.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/737caad8acd7aa0c147a979421cf154022b19ec733c26677c384160196e74d27.jpg)

![80814bf736d907969224c1ad77a606ccc05ee306f04322e95957db5ba733f621.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/80814bf736d907969224c1ad77a606ccc05ee306f04322e95957db5ba733f621.jpg)

![881c257c9a7011fa6c81b33e98c19cfdaedcdbd89a81468d6e01e8788874ad7e.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/881c257c9a7011fa6c81b33e98c19cfdaedcdbd89a81468d6e01e8788874ad7e.jpg)

![9794e7688d0ef6d437bf3c4146489f4b8585beaacd7a199ddb572690766afef3.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/9794e7688d0ef6d437bf3c4146489f4b8585beaacd7a199ddb572690766afef3.jpg)

![9dfefe8dfcfb4b227ed28d62c09eeff306cb40f95e24d3873016763d5ae53c87.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/9dfefe8dfcfb4b227ed28d62c09eeff306cb40f95e24d3873016763d5ae53c87.jpg)

![a7821de07dea39492978d0f183853a88d5ae6011b8183867f2c588e2c5f32e01.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/a7821de07dea39492978d0f183853a88d5ae6011b8183867f2c588e2c5f32e01.jpg)

![b321c320d66c6f9247eb3e2ad626a15d8651d957662c33fa780750fd714e3350.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/b321c320d66c6f9247eb3e2ad626a15d8651d957662c33fa780750fd714e3350.jpg)

![b755386d06bddfd0b294f455a73b5c03d573f8dc50f8d788395b16f76250f066.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/b755386d06bddfd0b294f455a73b5c03d573f8dc50f8d788395b16f76250f066.jpg)

![c21eda51d7d47d14ac099dd280a926895a8de890052199db8d9b3920a7f6f658.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/c21eda51d7d47d14ac099dd280a926895a8de890052199db8d9b3920a7f6f658.jpg)

![cf2f9be744e2648c8f037dd4bb7c16b68338fd4027676f353c576fec96e784d3.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/cf2f9be744e2648c8f037dd4bb7c16b68338fd4027676f353c576fec96e784d3.jpg)

![e1b9818f68a6b50bce7032f1151d91ba0b7ea7766501c3b6461eb679d4118292.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/e1b9818f68a6b50bce7032f1151d91ba0b7ea7766501c3b6461eb679d4118292.jpg)

![e8744f7253bd3691cd2d2b84ba32b32f99abef91fb6e547a47c62f1fbeaf23f4.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/e8744f7253bd3691cd2d2b84ba32b32f99abef91fb6e547a47c62f1fbeaf23f4.jpg)

![f28821fcd48d0e77a1854e5e87be43c5f6295d3b5c94bbac6b2e75282e9d887d.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/images/f28821fcd48d0e77a1854e5e87be43c5f6295d3b5c94bbac6b2e75282e9d887d.jpg)

### Tables

![0185d34c4bf92ecd183f09590cf88a5f6ee62132c44bc8e33355dbdaff9b642e.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/0185d34c4bf92ecd183f09590cf88a5f6ee62132c44bc8e33355dbdaff9b642e.jpg)

![208ceabef2b62ccbd0a775d7084405d1bb33a003e8e93e9a0483849c13c66032.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/208ceabef2b62ccbd0a775d7084405d1bb33a003e8e93e9a0483849c13c66032.jpg)

![34014e8233474042dfb71da0eb8e9cc96271c3e90e6881d202574d02ad1de65b.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/34014e8233474042dfb71da0eb8e9cc96271c3e90e6881d202574d02ad1de65b.jpg)

![4a4f3c1f25fcbcdd530f43a5ced83dfd265c592479c4e9d87ff92b322db2a65f.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/4a4f3c1f25fcbcdd530f43a5ced83dfd265c592479c4e9d87ff92b322db2a65f.jpg)

![78ba45ffb5850d781077170c86a0c5df8cb2ffb910589f6b69e30b3a5ab328a0.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/78ba45ffb5850d781077170c86a0c5df8cb2ffb910589f6b69e30b3a5ab328a0.jpg)

![824154291ab70a55cb06bc814bc4eae30e6cfe68bbdaa7f9a7419bcb19fce255.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/824154291ab70a55cb06bc814bc4eae30e6cfe68bbdaa7f9a7419bcb19fce255.jpg)

![84766d5bb8293e25702bdca6f5e5654521e6b054858444726a3795eda29b7a28.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/84766d5bb8293e25702bdca6f5e5654521e6b054858444726a3795eda29b7a28.jpg)

![9ce07466d1dbe4845b5dc193dccd67d8d73ee38969e20dd327adc4387d363f95.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/9ce07466d1dbe4845b5dc193dccd67d8d73ee38969e20dd327adc4387d363f95.jpg)

![9d4f0626605f7c48047176a29295a4c738f7771ec35254f329575375bf2e47d7.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/9d4f0626605f7c48047176a29295a4c738f7771ec35254f329575375bf2e47d7.jpg)

![a35cbf2abbf60ee49258b7e4e0f9653eb6d98ab41d08f85e6834b4fbedffa36f.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/a35cbf2abbf60ee49258b7e4e0f9653eb6d98ab41d08f85e6834b4fbedffa36f.jpg)

![aeaf87379ed79ac6fb7e921950aeaa25351466acd0639dff534f2e1cf0606439.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/aeaf87379ed79ac6fb7e921950aeaa25351466acd0639dff534f2e1cf0606439.jpg)

![c56b10f3d607415bf547c9d84822bf987ab7e94e4fb71bfdd8d08f6edfe19635.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/c56b10f3d607415bf547c9d84822bf987ab7e94e4fb71bfdd8d08f6edfe19635.jpg)

![c7d825f6e20730789bff61e7456d79ecb4d4e40804b79aa830027e06133a1d42.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/c7d825f6e20730789bff61e7456d79ecb4d4e40804b79aa830027e06133a1d42.jpg)

![c850d0d970e9c19b5748957c0b8100557fbcfed36a9cfd93f285ee0e6e05af8c.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/c850d0d970e9c19b5748957c0b8100557fbcfed36a9cfd93f285ee0e6e05af8c.jpg)

![c8ee89a20fef7f97c5158ec18d3223debd7614463a81bacc1f7367a6e54cefb9.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/c8ee89a20fef7f97c5158ec18d3223debd7614463a81bacc1f7367a6e54cefb9.jpg)

![ccbfca186181a1f17d76bc376a6889ebd1fc3a00acf404d535cd69e14beb3025.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/ccbfca186181a1f17d76bc376a6889ebd1fc3a00acf404d535cd69e14beb3025.jpg)

![cd3c69defc82235c66a8d785d0a83581f5d9a0933dfc1f1044e380e692463599.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/cd3c69defc82235c66a8d785d0a83581f5d9a0933dfc1f1044e380e692463599.jpg)

![cda42b59f518e18bd1826dc65f4054b2c11f0ddebe996c563bc82d202e51cb05.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/cda42b59f518e18bd1826dc65f4054b2c11f0ddebe996c563bc82d202e51cb05.jpg)

![d254be09f0ae31732302f708083d721bc8a51f2d47adf270f509296d73b14b6a.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/d254be09f0ae31732302f708083d721bc8a51f2d47adf270f509296d73b14b6a.jpg)

![f05c44f21d5828334816e4d97699948d2c2f1f089f14b26a11c83be90ac95442.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/f05c44f21d5828334816e4d97699948d2c2f1f089f14b26a11c83be90ac95442.jpg)

![f0a2bd04ce674933abad92ea7bb5ab09853284b292ccaa3dab6916b870ca0791.jpg](../icml_results/1916_On%20Stable%20Long-Form%20Generation_%20Benchmarking%20and%20Mitigating%20Length%20Volatility/tables/f0a2bd04ce674933abad92ea7bb5ab09853284b292ccaa3dab6916b870ca0791.jpg)

<a id="paper-1892"></a>

## A Graph Foundation Model with Cross-Modal Alignment and Modality-Aware Expert Fusion for Multi-Modal Graphs


### Images

![04571cf8fe1f9b538a255bc1d6766670daa3587bc71be2d7b289b1cc8b48eac3.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/images/04571cf8fe1f9b538a255bc1d6766670daa3587bc71be2d7b289b1cc8b48eac3.jpg)

![0904b0c8bec1439e7030c682fbefca1d374192a5fbc0ed3c141571996c2b4402.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/images/0904b0c8bec1439e7030c682fbefca1d374192a5fbc0ed3c141571996c2b4402.jpg)

![2b18c55928406907f01dc55ce683fa90e74743d3a3fa2955db9c6f6143224842.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/images/2b18c55928406907f01dc55ce683fa90e74743d3a3fa2955db9c6f6143224842.jpg)

![2bfd8877be40171dfa3088c448c681c6740a755ca9032dbdaad03f0bd64bb4de.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/images/2bfd8877be40171dfa3088c448c681c6740a755ca9032dbdaad03f0bd64bb4de.jpg)

![4fd0eaaa91337e4fc72bd6ab198cf8c25953dd45e302faa403fc69787915e264.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/images/4fd0eaaa91337e4fc72bd6ab198cf8c25953dd45e302faa403fc69787915e264.jpg)

![aa65d057fb9f85c5a897fc94ee4e6a1bb233cd57bfd116b72503f9bc1d1ece83.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/images/aa65d057fb9f85c5a897fc94ee4e6a1bb233cd57bfd116b72503f9bc1d1ece83.jpg)

![acc9ba1ee8fc223ca36d66e9fa1e72f5199bc6431e5f917ac65d56af5eebab93.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/images/acc9ba1ee8fc223ca36d66e9fa1e72f5199bc6431e5f917ac65d56af5eebab93.jpg)

![afbff6938d4643d3703723ed382d8b4c096e8abcd27cb12a0f7adaf0511e400e.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/images/afbff6938d4643d3703723ed382d8b4c096e8abcd27cb12a0f7adaf0511e400e.jpg)

![c286c50948db143753244188922d4e79ebe96cc2bef7cc538123c4c1db04ab9a.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/images/c286c50948db143753244188922d4e79ebe96cc2bef7cc538123c4c1db04ab9a.jpg)

![c4c0504b1f5bae885d6c910a615f9fc03e8b2864273c27df8135500d67c4e94b.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/images/c4c0504b1f5bae885d6c910a615f9fc03e8b2864273c27df8135500d67c4e94b.jpg)

![c5a8ae0c3a5995b46c0c5b12a9ebbe4746f4a3b0e445a11bb4ee2b3faa6e8392.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/images/c5a8ae0c3a5995b46c0c5b12a9ebbe4746f4a3b0e445a11bb4ee2b3faa6e8392.jpg)

![ceeaab878bf3937192c3918a180d7cca9299e694a84a0638f0564ea975ba854f.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/images/ceeaab878bf3937192c3918a180d7cca9299e694a84a0638f0564ea975ba854f.jpg)

![d5b37a62a999494eef9e163c049338f98ed4390598327b09ed8a132410c5e72e.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/images/d5b37a62a999494eef9e163c049338f98ed4390598327b09ed8a132410c5e72e.jpg)

![da92323b22bc0365118f79370e96d419070a0cfabbac17454138cf6511740511.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/images/da92323b22bc0365118f79370e96d419070a0cfabbac17454138cf6511740511.jpg)

![df2b91237994ed5ae9ee33e92f892b00d72642b0d0364578bf56cc62ed039b44.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/images/df2b91237994ed5ae9ee33e92f892b00d72642b0d0364578bf56cc62ed039b44.jpg)

![e5068e4dc2bdb1f945f504ecbbd84782cb672ac8e81b6541700381ecded39851.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/images/e5068e4dc2bdb1f945f504ecbbd84782cb672ac8e81b6541700381ecded39851.jpg)

### Tables

![18e3512f34ec2d0a2433fc0f796694f9dfb5e2ef587979098c43f3284677d0d2.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/tables/18e3512f34ec2d0a2433fc0f796694f9dfb5e2ef587979098c43f3284677d0d2.jpg)

![28d18fde7f0d35014fc259d08d7fe80a556013c632b553cc69343cd12a9af8bc.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/tables/28d18fde7f0d35014fc259d08d7fe80a556013c632b553cc69343cd12a9af8bc.jpg)

![430077500ace4cecf3e08c6e999ff1fd86833e14ac042fa55dc7a9cb575dd68a.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/tables/430077500ace4cecf3e08c6e999ff1fd86833e14ac042fa55dc7a9cb575dd68a.jpg)

![5c173078990da6bdab7f15014a14b4c52682299e09ccf45223f3c13d4619390f.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/tables/5c173078990da6bdab7f15014a14b4c52682299e09ccf45223f3c13d4619390f.jpg)

![af14b07d7e5728cc1d0bfe20a5ee877e89ad92459e8864f350eb340a44a9ef58.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/tables/af14b07d7e5728cc1d0bfe20a5ee877e89ad92459e8864f350eb340a44a9ef58.jpg)

![c0f41cc58158fe37adfb18b92ad846caa438708fa453459240cf63844600a0f6.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/tables/c0f41cc58158fe37adfb18b92ad846caa438708fa453459240cf63844600a0f6.jpg)

![c5f5d6f7c422a62e183832cfbfce849bac0d895aac23b6e71d1c2ce1e292d554.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/tables/c5f5d6f7c422a62e183832cfbfce849bac0d895aac23b6e71d1c2ce1e292d554.jpg)

![d197961259ab415653f16eb7420fbc10e4823fc7fa86417477b8ba83727e2d70.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/tables/d197961259ab415653f16eb7420fbc10e4823fc7fa86417477b8ba83727e2d70.jpg)

![d781d63bb87e186915eb7d5b9bf1fa7302ec2a9dc55639070105eff1f829713b.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/tables/d781d63bb87e186915eb7d5b9bf1fa7302ec2a9dc55639070105eff1f829713b.jpg)

![fb7fcd2cce88759eb65708cfbd944fd451a7633b188b4a7beef5a779d689b8f3.jpg](../icml_results/1917_A%20Graph%20Foundation%20Model%20with%20Cross-Modal%20Alignment%20and%20Modality-Aware%20Expert%20Fusion%20for%20Multi-Modal/tables/fb7fcd2cce88759eb65708cfbd944fd451a7633b188b4a7beef5a779d689b8f3.jpg)

<a id="paper-1893"></a>

## From Inpainting to Editing: Unlocking Robust Mask-Free Visual Dubbing via Generative Bootstrapping


### Images

![0eee8b0d1bc46f79e7c407da0c60c485febc803a815e63e1d76463d2df4654a9.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/0eee8b0d1bc46f79e7c407da0c60c485febc803a815e63e1d76463d2df4654a9.jpg)

![12b7399145a4cbad2aca6ea221c4ee4d97bad6c11d030208778d357cd03d9f2a.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/12b7399145a4cbad2aca6ea221c4ee4d97bad6c11d030208778d357cd03d9f2a.jpg)

![32476c30f704b10939e649b8d012c7797613bdfd825cfd6e165e3d4467faaabe.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/32476c30f704b10939e649b8d012c7797613bdfd825cfd6e165e3d4467faaabe.jpg)

![4843f581756987afe89c74b5d653c5215f119cc87f68b5ad54694b40e11cca6d.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/4843f581756987afe89c74b5d653c5215f119cc87f68b5ad54694b40e11cca6d.jpg)

![546e6b59928c4448575e9f65733c597b1deb7f744f65ce478cfc4f6d4f039f5c.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/546e6b59928c4448575e9f65733c597b1deb7f744f65ce478cfc4f6d4f039f5c.jpg)

![63904c97609a8a689d0b1e18a1f6373d8cf3aa490818190cf59f1ac7d1fc4251.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/63904c97609a8a689d0b1e18a1f6373d8cf3aa490818190cf59f1ac7d1fc4251.jpg)

![63a4ddf8d797b4a2d51e53ac8e2df2c46d39f09b08f2c0f6457d2844ad0e09a1.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/63a4ddf8d797b4a2d51e53ac8e2df2c46d39f09b08f2c0f6457d2844ad0e09a1.jpg)

![84c5dd54ab513c4cb1adbd5aa0b6af7c39b442436bd3e07d4b3ddc1461cdb9f8.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/84c5dd54ab513c4cb1adbd5aa0b6af7c39b442436bd3e07d4b3ddc1461cdb9f8.jpg)

![8df034de408577a82afd755448c059836225e0e8378cd8847e43194870687a72.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/8df034de408577a82afd755448c059836225e0e8378cd8847e43194870687a72.jpg)

![951c5fad3e182012700c8695107af8e205d8880bd761589a079dc77c0a46319c.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/951c5fad3e182012700c8695107af8e205d8880bd761589a079dc77c0a46319c.jpg)

![9f0cc6cb649073fe26533bc08003e239049f4ece97db71d04a4e68a99a983ff3.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/9f0cc6cb649073fe26533bc08003e239049f4ece97db71d04a4e68a99a983ff3.jpg)

![a44bb67384f5f744c820665e598ccb881afbf2a8094413741a93bd8e3afe7bad.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/a44bb67384f5f744c820665e598ccb881afbf2a8094413741a93bd8e3afe7bad.jpg)

![a6a0c2ce33f9b2449384cd21d92dbb9100bbb1edc336c6791606d68a6d73fa98.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/a6a0c2ce33f9b2449384cd21d92dbb9100bbb1edc336c6791606d68a6d73fa98.jpg)

![c9a7ad7c6d861db0f3cfc319d0e296cb6a8536fbb9b3c5c86024b038266d42ee.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/c9a7ad7c6d861db0f3cfc319d0e296cb6a8536fbb9b3c5c86024b038266d42ee.jpg)

![d1e81a4fd3aaaa3ffc0bbdd70df31bec56c85eb669f1fc78f2a0af201065755d.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/d1e81a4fd3aaaa3ffc0bbdd70df31bec56c85eb669f1fc78f2a0af201065755d.jpg)

![d547ddc9806b444ec8a4f97714cebed0f5b0b766c72ac078305638c9542dcb0b.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/d547ddc9806b444ec8a4f97714cebed0f5b0b766c72ac078305638c9542dcb0b.jpg)

![e7beeb9a88dd211dbf8c602ff4e6d74fdf484af89b85d8e6ef1b2964466c5e3c.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/e7beeb9a88dd211dbf8c602ff4e6d74fdf484af89b85d8e6ef1b2964466c5e3c.jpg)

![e9bb1d224dc62f2544c2dc4ef773d2b440834cc96164e4ef5b9f6ec2d9bcfd84.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/images/e9bb1d224dc62f2544c2dc4ef773d2b440834cc96164e4ef5b9f6ec2d9bcfd84.jpg)

### Tables

![12545e933092bc51827832d0b78814699b43d63f70bc85d1942e883aac45e0e1.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/tables/12545e933092bc51827832d0b78814699b43d63f70bc85d1942e883aac45e0e1.jpg)

![28ab304b102cb78cadeecb936ecf21ee253a06d2d6ef22fa23891eb994aff259.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/tables/28ab304b102cb78cadeecb936ecf21ee253a06d2d6ef22fa23891eb994aff259.jpg)

![29da00b676246cae42b7dd6112693df4ba28d8506602e2d7b15af2086f3f3771.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/tables/29da00b676246cae42b7dd6112693df4ba28d8506602e2d7b15af2086f3f3771.jpg)

![3a4588533ef388eb59cd23bc334d80696e4945e9dcada9cc47850fbe15db2f34.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/tables/3a4588533ef388eb59cd23bc334d80696e4945e9dcada9cc47850fbe15db2f34.jpg)

![461fae6f9a4ed80ac4af5f8730b1fa9919cb4b49e04c797afcac0dfd4e65a93a.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/tables/461fae6f9a4ed80ac4af5f8730b1fa9919cb4b49e04c797afcac0dfd4e65a93a.jpg)

![6db2eb41eca43b94efe49fdbc1c1684e122769ce38cfb387f2a7c7323581f422.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/tables/6db2eb41eca43b94efe49fdbc1c1684e122769ce38cfb387f2a7c7323581f422.jpg)

![7404d76a71f19e7872b7243900957ef34254272951205711b9c4b2644a65bc59.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/tables/7404d76a71f19e7872b7243900957ef34254272951205711b9c4b2644a65bc59.jpg)

![8cfaa6bb6aa1bd5a40f703d029cf4a1d6da7dc5cc424b422ec9c051bc17244fe.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/tables/8cfaa6bb6aa1bd5a40f703d029cf4a1d6da7dc5cc424b422ec9c051bc17244fe.jpg)

![962df0287e4b11c08d1c7bb8c7273e7ceebeead36a4d726667c1fe3664032184.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/tables/962df0287e4b11c08d1c7bb8c7273e7ceebeead36a4d726667c1fe3664032184.jpg)

![9b9485c6756dfce7c9fe029d5dffca17e467a76b169bc2601382e32d2a72d8c0.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/tables/9b9485c6756dfce7c9fe029d5dffca17e467a76b169bc2601382e32d2a72d8c0.jpg)

![d593aecad4351c6da01c71809bbd7fdb3a7f0e04a400519a37ae9d01c5865187.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/tables/d593aecad4351c6da01c71809bbd7fdb3a7f0e04a400519a37ae9d01c5865187.jpg)

![d7614f97bbcb095ae3be07c1de5317663307c99abf11d1c5d3b8a025826d5596.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/tables/d7614f97bbcb095ae3be07c1de5317663307c99abf11d1c5d3b8a025826d5596.jpg)

![ed8a20198835161f5a46d6feccb4934ae4f718b5b917e8512a349fa43b4f72e2.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/tables/ed8a20198835161f5a46d6feccb4934ae4f718b5b917e8512a349fa43b4f72e2.jpg)

![f16d8302e92c53603f18999c5edd6f0b65f5e80e31cda5e0f3e60357a00fbeb4.jpg](../icml_results/1918_From%20Inpainting%20to%20Editing_%20Unlocking%20Robust%20Mask-Free%20Visual%20Dubbing%20via%20Generative%20Bootstrapping/tables/f16d8302e92c53603f18999c5edd6f0b65f5e80e31cda5e0f3e60357a00fbeb4.jpg)

<a id="paper-1894"></a>

## RSAgent: Learning to Reason and Act via Multi-Turn Tool Invocations for Text-Guided Segmentation


### Images

![1527b41421a9a4baaf5f4ec949e4ae322d943b9777b22a3abefbf47edb8b4d21.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/1527b41421a9a4baaf5f4ec949e4ae322d943b9777b22a3abefbf47edb8b4d21.jpg)

![174fea412b487302e60023c563a921b5295cb1039b78a5b85263c408ff893ce1.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/174fea412b487302e60023c563a921b5295cb1039b78a5b85263c408ff893ce1.jpg)

![1a9bbe82bc9875fd0043f78e53ce9e676c12dc212b632b8e73ed5dc5314eae9f.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/1a9bbe82bc9875fd0043f78e53ce9e676c12dc212b632b8e73ed5dc5314eae9f.jpg)

![1cb50cb8948e4590b4047368e2b04ba91b0f497fbc2f28df02d7b17c8d683a7a.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/1cb50cb8948e4590b4047368e2b04ba91b0f497fbc2f28df02d7b17c8d683a7a.jpg)

![1fe62e6a4a472f3b29d50277529926c71fab914a7714f2a507eaeb454989372f.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/1fe62e6a4a472f3b29d50277529926c71fab914a7714f2a507eaeb454989372f.jpg)

![320343df7a51c7aaee6b349ed6b02b9cd708e1e3ed6f59dc1ec1387dd6f738f2.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/320343df7a51c7aaee6b349ed6b02b9cd708e1e3ed6f59dc1ec1387dd6f738f2.jpg)

![33476718e7f0e822ed52850bd0bb6a09cfc519e60067e0b3b3b9a828b92492ac.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/33476718e7f0e822ed52850bd0bb6a09cfc519e60067e0b3b3b9a828b92492ac.jpg)

![3534a2acade52297abda5cda149084f154e9e6d65039e7f386ec4be24ec700df.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/3534a2acade52297abda5cda149084f154e9e6d65039e7f386ec4be24ec700df.jpg)

![39872acdf743db3ee0bd5ee1bdcae414cae1b43fbb8cd46a90303ce5290a7358.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/39872acdf743db3ee0bd5ee1bdcae414cae1b43fbb8cd46a90303ce5290a7358.jpg)

![3d901441e9822a2a2858cd31957e14e69f0de199ef0d48eb004200bcc556a1cc.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/3d901441e9822a2a2858cd31957e14e69f0de199ef0d48eb004200bcc556a1cc.jpg)

![54e166bc7a38c3812c88b19ae99f99eb0c5a2907ec05940ec0055a2154775483.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/54e166bc7a38c3812c88b19ae99f99eb0c5a2907ec05940ec0055a2154775483.jpg)

![5974604de1dead5de73209af38da288f069826d406e2c4de200ade65ca0767ff.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/5974604de1dead5de73209af38da288f069826d406e2c4de200ade65ca0767ff.jpg)

![5d49c9e8a939967e5bd1acb8042ded1a2a50b1c67bfbe6b3e26b02c16c584c72.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/5d49c9e8a939967e5bd1acb8042ded1a2a50b1c67bfbe6b3e26b02c16c584c72.jpg)

![5e6ca56c95c80b49d9bfb2da9538248ecac85e35074a239043b623eecd2e21c9.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/5e6ca56c95c80b49d9bfb2da9538248ecac85e35074a239043b623eecd2e21c9.jpg)

![5ec563964b83b3f5039a79601b408b48b8a237701ca63a47f6dc030f846d7897.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/5ec563964b83b3f5039a79601b408b48b8a237701ca63a47f6dc030f846d7897.jpg)

![6e18b49e8990b0849a956052ba2213976e5502dff481b11214a9e7e10d23ae1a.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/6e18b49e8990b0849a956052ba2213976e5502dff481b11214a9e7e10d23ae1a.jpg)

![7b66c909671f643defbd3714bb093a9df2bf4227863f450d7bf2c89b642c3b70.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/7b66c909671f643defbd3714bb093a9df2bf4227863f450d7bf2c89b642c3b70.jpg)

![8424960ccc3d608060649441f0fa87d91b6f812242e4357c2269d8e3658eacaa.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/8424960ccc3d608060649441f0fa87d91b6f812242e4357c2269d8e3658eacaa.jpg)

![9096ade93fc422603fd3893472a67192c81e2da718c6fd376f5cedf8b44cf033.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/9096ade93fc422603fd3893472a67192c81e2da718c6fd376f5cedf8b44cf033.jpg)

![90c0fb311cd71980b71b3a8ccf0d7e937d1664f1a3332e15d6630dceb67934a2.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/90c0fb311cd71980b71b3a8ccf0d7e937d1664f1a3332e15d6630dceb67934a2.jpg)

![984da8478fa35b1f044e6676bf3d00eb5db4031daad1329f500e843851ffffc8.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/984da8478fa35b1f044e6676bf3d00eb5db4031daad1329f500e843851ffffc8.jpg)

![9ea584b64f2f30a2d6232afde02fa9fd806c55d0db9034b9cc313a2bfda701ce.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/9ea584b64f2f30a2d6232afde02fa9fd806c55d0db9034b9cc313a2bfda701ce.jpg)

![b33d725312ac4679155b4fa4bf586000471ebcbb1cf852388171ade0b6c3f279.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/b33d725312ac4679155b4fa4bf586000471ebcbb1cf852388171ade0b6c3f279.jpg)

![dd96d99e4bc5ca314f369b1a01e6030bec3951aeec3b1eff3d49ee5af4f91a4f.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/dd96d99e4bc5ca314f369b1a01e6030bec3951aeec3b1eff3d49ee5af4f91a4f.jpg)

![de8636f4708bef93523fe89f11a5c5159ac146286f9e5851da4c8a2b75ea2cef.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/de8636f4708bef93523fe89f11a5c5159ac146286f9e5851da4c8a2b75ea2cef.jpg)

![dfc33c7824a1ceb09f73bc7f376c3479ede0a87346318802cf3730bdb1942792.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/dfc33c7824a1ceb09f73bc7f376c3479ede0a87346318802cf3730bdb1942792.jpg)

![e14318dd560fe884a4aca0ea751fe81a63a5ef7ae98763f6ee8af2a2cb77bf19.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/e14318dd560fe884a4aca0ea751fe81a63a5ef7ae98763f6ee8af2a2cb77bf19.jpg)

![e3e494030f90df8a27a51c538d55ab6e6098b2e3066605bef5f068792d5faaa9.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/e3e494030f90df8a27a51c538d55ab6e6098b2e3066605bef5f068792d5faaa9.jpg)

![e9aab4becb9e9274c4603b7093bb2d0e0f75c84287cb9b7961d30f29ac63caf6.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/e9aab4becb9e9274c4603b7093bb2d0e0f75c84287cb9b7961d30f29ac63caf6.jpg)

![f90106b6b3090072e14b4ccf4cf6cd91df4bc5da25973cc9db2b89ab3806c1d5.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/f90106b6b3090072e14b4ccf4cf6cd91df4bc5da25973cc9db2b89ab3806c1d5.jpg)

![fc47e7b15b1d20533461ebe24f966ceeb53a33f48c28d53c56467d2ea21ca21f.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/fc47e7b15b1d20533461ebe24f966ceeb53a33f48c28d53c56467d2ea21ca21f.jpg)

![ff179930bc4920a6e5839781c3ebfe37daee9bccd6fcbf8bb4de7802c95f5e4e.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/images/ff179930bc4920a6e5839781c3ebfe37daee9bccd6fcbf8bb4de7802c95f5e4e.jpg)

### Tables

![01acb0f0cc7821516c2ea98eba0ccb616af0556384e0e48634c2571ec1f1dffc.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/tables/01acb0f0cc7821516c2ea98eba0ccb616af0556384e0e48634c2571ec1f1dffc.jpg)

![235e633f3b4ff3e1c83438928b990eeb49bc8562068fd1c939954bae6dde3d47.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/tables/235e633f3b4ff3e1c83438928b990eeb49bc8562068fd1c939954bae6dde3d47.jpg)

![4082f7dbb5c86e96033e8725eabb0e6c21ffb065a2635c7441e29b9b36f6f391.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/tables/4082f7dbb5c86e96033e8725eabb0e6c21ffb065a2635c7441e29b9b36f6f391.jpg)

![42981bb436a52abe3654a371e228e38ad3925a64c8dcae5f5f8d8593c782afe5.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/tables/42981bb436a52abe3654a371e228e38ad3925a64c8dcae5f5f8d8593c782afe5.jpg)

![6559c9bf0a501a49be3630f864e76080ab25c40ebd7df54c5659c24dfde8c53e.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/tables/6559c9bf0a501a49be3630f864e76080ab25c40ebd7df54c5659c24dfde8c53e.jpg)

![8e19126f11e4555886e0c094115c5166fd6f3c2cd546f5c9575c807906cb8d79.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/tables/8e19126f11e4555886e0c094115c5166fd6f3c2cd546f5c9575c807906cb8d79.jpg)

![b2ac2a75db24d45612cb26034c920ecebc1ae8ead7b18d5c816b333ca25ee992.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/tables/b2ac2a75db24d45612cb26034c920ecebc1ae8ead7b18d5c816b333ca25ee992.jpg)

![b7443d5bfd15c17de939091cdeb87566660eff6b5d11df8f7e7059481bc8553f.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/tables/b7443d5bfd15c17de939091cdeb87566660eff6b5d11df8f7e7059481bc8553f.jpg)

![b77e14a0191ad3e1b53be278e5435330504e10d7a25288e6ce71529ed2bed4b4.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/tables/b77e14a0191ad3e1b53be278e5435330504e10d7a25288e6ce71529ed2bed4b4.jpg)

![e6038748cf74cf7e33dc0146a679889fd3c420fe80d8995095753b28ec27fa4d.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/tables/e6038748cf74cf7e33dc0146a679889fd3c420fe80d8995095753b28ec27fa4d.jpg)

![f91accd96bd6cc5cf2a4c566d3f9dca4a51d42c9df4ac0e10aef4599c6a6736e.jpg](../icml_results/1919_RSAgent_%20Learning%20to%20Reason%20and%20Act%20via%20Multi-Turn%20Tool%20Invocations%20for%20Text-Guided%20Segmentation/tables/f91accd96bd6cc5cf2a4c566d3f9dca4a51d42c9df4ac0e10aef4599c6a6736e.jpg)

<a id="paper-1895"></a>

## Refining Context-Entangled Content Segmentation via Curriculum Selection and Anti-Curriculum Promotion


### Images

![038f108388d154a8e922ea49cae5a936a05fa7335b6d5cfcc88076891d564d84.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/images/038f108388d154a8e922ea49cae5a936a05fa7335b6d5cfcc88076891d564d84.jpg)

![1a127d158503abb54a5bfc746bd04d34d72362d1ec77e610047cbc6ce19b9fef.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/images/1a127d158503abb54a5bfc746bd04d34d72362d1ec77e610047cbc6ce19b9fef.jpg)

![28bc0f98a97e0bb53ee8ef7e1896a0e6524642b2513f6c599c6868b42ec33808.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/images/28bc0f98a97e0bb53ee8ef7e1896a0e6524642b2513f6c599c6868b42ec33808.jpg)

![61722cff587cde7cf324632bb7ef5e41f7d2bc35a61b491e2dea7dfd9806b141.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/images/61722cff587cde7cf324632bb7ef5e41f7d2bc35a61b491e2dea7dfd9806b141.jpg)

![653a3a1530c27f58db191342e1fec6f754068b99e34bb0f18c0711293d78f94b.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/images/653a3a1530c27f58db191342e1fec6f754068b99e34bb0f18c0711293d78f94b.jpg)

![c55887e274a2d10aaecac88c2c7b2d6d03feb21ad4ee326b02c35cd039baf1cc.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/images/c55887e274a2d10aaecac88c2c7b2d6d03feb21ad4ee326b02c35cd039baf1cc.jpg)

![f80944efd139cb79a19422bd2d177b078bae436414fb589762c3ceeecabaed80.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/images/f80944efd139cb79a19422bd2d177b078bae436414fb589762c3ceeecabaed80.jpg)

### Tables

![06ab7490bf12dc43d83277496b7801ffe71a9d0385e4f34c126f041c50cdd1dd.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/tables/06ab7490bf12dc43d83277496b7801ffe71a9d0385e4f34c126f041c50cdd1dd.jpg)

![13c62a37a2010971d201e8f6c93d87e9598a829a9cdc99b35850b6edbc34d975.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/tables/13c62a37a2010971d201e8f6c93d87e9598a829a9cdc99b35850b6edbc34d975.jpg)

![5c22d541a20e6e99881fa97c5cc14036aefe72c41c9110fe1eef855be61d1bbc.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/tables/5c22d541a20e6e99881fa97c5cc14036aefe72c41c9110fe1eef855be61d1bbc.jpg)

![608796025a626ceceee2300824c81dde94579cf83f461fded4ed2039b4585567.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/tables/608796025a626ceceee2300824c81dde94579cf83f461fded4ed2039b4585567.jpg)

![7cab4b1da06df93cc7e6bdff020a337007dc5befeeba2b8b79c1c0992e7e8c46.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/tables/7cab4b1da06df93cc7e6bdff020a337007dc5befeeba2b8b79c1c0992e7e8c46.jpg)

![9e895ad672811cb267e1fbaeef9750e4a8e2b27f68ebd2dc6152a9915a5c82a8.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/tables/9e895ad672811cb267e1fbaeef9750e4a8e2b27f68ebd2dc6152a9915a5c82a8.jpg)

![ad7dc666cb5190977156dd0a6c0b9174b9f464fab77932700ec749a93ce700e5.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/tables/ad7dc666cb5190977156dd0a6c0b9174b9f464fab77932700ec749a93ce700e5.jpg)

![b5851dea03c547695857928da05f750d9466743c5d5665e05d81c921ca16ef9c.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/tables/b5851dea03c547695857928da05f750d9466743c5d5665e05d81c921ca16ef9c.jpg)

![b69abe899142c14b1981e98d5f23901fdb0f09bf7b7eab0eda53429a2bfab2fa.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/tables/b69abe899142c14b1981e98d5f23901fdb0f09bf7b7eab0eda53429a2bfab2fa.jpg)

![d164b9f59bfb98e40425d462ab7f9e759353c9aad07a9c6548e019177f5af0f9.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/tables/d164b9f59bfb98e40425d462ab7f9e759353c9aad07a9c6548e019177f5af0f9.jpg)

![d87d57377ce382226903c08e3e8fc73cd70601ac39a38d4e51649e8cdde08a63.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/tables/d87d57377ce382226903c08e3e8fc73cd70601ac39a38d4e51649e8cdde08a63.jpg)

![d8a33fbfa6b82c70fee61151b1838702884b7b3376d4eaaaca829f2873877216.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/tables/d8a33fbfa6b82c70fee61151b1838702884b7b3376d4eaaaca829f2873877216.jpg)

![db367d03ed746a1166c188a2e1d32c7bb8b65a3b858d794a59d47d88383861ec.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/tables/db367d03ed746a1166c188a2e1d32c7bb8b65a3b858d794a59d47d88383861ec.jpg)

![ed2ffaf495a44e15b8314f9574263fd84cdb5603937cb95d19719c14e987a195.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/tables/ed2ffaf495a44e15b8314f9574263fd84cdb5603937cb95d19719c14e987a195.jpg)

![fcfe79a266b5740a6c2ef1ba0ea7c2af7ed8a28e4c04137eebd97cebd092563c.jpg](../icml_results/1920_Refining%20Context-Entangled%20Content%20Segmentation%20via%20Curriculum%20Selection%20and%20Anti-Curriculum%20Promoti/tables/fcfe79a266b5740a6c2ef1ba0ea7c2af7ed8a28e4c04137eebd97cebd092563c.jpg)

<a id="paper-1896"></a>

## Learning Permutation Distributions via Reflected Diffusion on Ranks


### Images

![05ea104654968f2c5603916f42f9b31a99f2328dccfe7a49489b7311c9e39665.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/images/05ea104654968f2c5603916f42f9b31a99f2328dccfe7a49489b7311c9e39665.jpg)

![1a00cfbfd08a1507afb8aca367ce9b4969a8ca2c7f04be953f6f41a8b1355216.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/images/1a00cfbfd08a1507afb8aca367ce9b4969a8ca2c7f04be953f6f41a8b1355216.jpg)

![4515ad51bd7a9f0cf31c439bef1e7d85324cffd3d5a3e6b4710424c35ce7e7ec.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/images/4515ad51bd7a9f0cf31c439bef1e7d85324cffd3d5a3e6b4710424c35ce7e7ec.jpg)

![4ffce992d42937a264a04ec94aa9d8cdaef3600611cbea4b5c4d6f6ad3148abe.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/images/4ffce992d42937a264a04ec94aa9d8cdaef3600611cbea4b5c4d6f6ad3148abe.jpg)

![660ca992eedef3c4a041c874837a84aba4ed56ada40dd5c5ee2b7949ea126e1c.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/images/660ca992eedef3c4a041c874837a84aba4ed56ada40dd5c5ee2b7949ea126e1c.jpg)

![68fa4ab77492d20a99fbfd2880ff060ca1c3b6e9336d1b8920ea9e593f9edbef.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/images/68fa4ab77492d20a99fbfd2880ff060ca1c3b6e9336d1b8920ea9e593f9edbef.jpg)

![a3e4d4513faaab2d21b4c88b65cd0ef9772fd31d6ac650b7d0ae7025c2b6d7bd.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/images/a3e4d4513faaab2d21b4c88b65cd0ef9772fd31d6ac650b7d0ae7025c2b6d7bd.jpg)

![a445a59ff957eaff49ae1599d6ff2c20162eff5bd59f28d927218d3c9da94e70.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/images/a445a59ff957eaff49ae1599d6ff2c20162eff5bd59f28d927218d3c9da94e70.jpg)

![cbfb320023933f88c28aebcfc1eefe955879c4126485f8d300a30067ad7cfba0.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/images/cbfb320023933f88c28aebcfc1eefe955879c4126485f8d300a30067ad7cfba0.jpg)

### Tables

![0a20a17033149a9deafa0328d8f135ef227f4644e7ae6e77c29bf3d49a49d7b7.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/tables/0a20a17033149a9deafa0328d8f135ef227f4644e7ae6e77c29bf3d49a49d7b7.jpg)

![2812b27c734912c60bb2867f5b57eaf6e1ed8af15c886d000c96844d777ea203.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/tables/2812b27c734912c60bb2867f5b57eaf6e1ed8af15c886d000c96844d777ea203.jpg)

![4a9d220e38f6f71bf5c2309c7436490a83b9e372a6784242166bdbff446dc8b3.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/tables/4a9d220e38f6f71bf5c2309c7436490a83b9e372a6784242166bdbff446dc8b3.jpg)

![539629c447118421a066dea009ee40da82f8171707d72b847a7b893a587befeb.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/tables/539629c447118421a066dea009ee40da82f8171707d72b847a7b893a587befeb.jpg)

![5cd5ea73c3fff0f1f35d02c57539f6552a980fd6caedc5388ab3df32ede9124c.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/tables/5cd5ea73c3fff0f1f35d02c57539f6552a980fd6caedc5388ab3df32ede9124c.jpg)

![7225caf790457f0230993e0ee4f1d52bbd96b5122b998cf819995d5f898e2233.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/tables/7225caf790457f0230993e0ee4f1d52bbd96b5122b998cf819995d5f898e2233.jpg)

![8b2baf984761822db36f3fce299b289052f18830bbc2875d03dfa67766c11ec5.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/tables/8b2baf984761822db36f3fce299b289052f18830bbc2875d03dfa67766c11ec5.jpg)

![b56ef7c035f6b11ee4d1409b74f01f4927f492d9986be2c8a3d8738893afe879.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/tables/b56ef7c035f6b11ee4d1409b74f01f4927f492d9986be2c8a3d8738893afe879.jpg)

![e8be47f5ef5061d21f821eb25ed2dd03b6663d2e913bcac7fc86ca6fff4b9368.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/tables/e8be47f5ef5061d21f821eb25ed2dd03b6663d2e913bcac7fc86ca6fff4b9368.jpg)

![f5f3feae3760e4795cd3c67fcfac8ab735cbd9b61c4c385b5d6a1f05d3dcdb36.jpg](../icml_results/1921_Learning%20Permutation%20Distributions%20via%20Reflected%20Diffusion%20on%20Ranks/tables/f5f3feae3760e4795cd3c67fcfac8ab735cbd9b61c4c385b5d6a1f05d3dcdb36.jpg)

<a id="paper-1897"></a>

## MMPD-Bench: Bridging Multimodal Fission with Multi-Polarimetric Modalities Decomposition


### Images

![048a2d953686fb2ef150fabff27d2c84e335e81c5b66991af1ccd358043a9380.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/048a2d953686fb2ef150fabff27d2c84e335e81c5b66991af1ccd358043a9380.jpg)

![0c0a14323046c92e1b1519d91f6d63abfbc5c7fc638b83fff1aa2550f4f0db2b.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/0c0a14323046c92e1b1519d91f6d63abfbc5c7fc638b83fff1aa2550f4f0db2b.jpg)

![1af6c807abf141be849caacca8f44b02b4da26276bf960a2f489c243704c3321.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/1af6c807abf141be849caacca8f44b02b4da26276bf960a2f489c243704c3321.jpg)

![1af6f9fa0852a908f883655df6944799c75818c13f0d257eb7c7e7c0dd230c87.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/1af6f9fa0852a908f883655df6944799c75818c13f0d257eb7c7e7c0dd230c87.jpg)

![1d67c28de6b4de48deaae06fba455d331fe68395b95bdb7c7fcf99d9ad91c715.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/1d67c28de6b4de48deaae06fba455d331fe68395b95bdb7c7fcf99d9ad91c715.jpg)

![32adf5caa3a183c9fbae8e386b763450bb01d727e967be623eab12c3e87f4f47.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/32adf5caa3a183c9fbae8e386b763450bb01d727e967be623eab12c3e87f4f47.jpg)

![3a6f961ad60a49faab5900b82c624d71a28c181c29fd91c8088150209cc89c53.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/3a6f961ad60a49faab5900b82c624d71a28c181c29fd91c8088150209cc89c53.jpg)

![3bd4f200271e5b135455f26227a05ad594e2536dbff2f0ffbc7af835df89251b.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/3bd4f200271e5b135455f26227a05ad594e2536dbff2f0ffbc7af835df89251b.jpg)

![48a7f02a0ed323d4ab17ca18ca5105b8237d18b49a71ab0ce05282ccf2ad1db0.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/48a7f02a0ed323d4ab17ca18ca5105b8237d18b49a71ab0ce05282ccf2ad1db0.jpg)

![4a04f26e3c1fa43af572e647f6f8b2ffdcb41c39393c73eabfca0481f2584ac3.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/4a04f26e3c1fa43af572e647f6f8b2ffdcb41c39393c73eabfca0481f2584ac3.jpg)

![4ade2d63b8f2c9b12f3e5192e759364bf802abf93ea7682fee13fed36d1b9fa8.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/4ade2d63b8f2c9b12f3e5192e759364bf802abf93ea7682fee13fed36d1b9fa8.jpg)

![551adae8a69adedafec818880b8d08635982fb2d8854f63407bd1ff29c32c14a.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/551adae8a69adedafec818880b8d08635982fb2d8854f63407bd1ff29c32c14a.jpg)

![5a04dfd399135a3e328bd306253f528542302bcc8c9283c448181070cefa58d5.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/5a04dfd399135a3e328bd306253f528542302bcc8c9283c448181070cefa58d5.jpg)

![5a75e35d90b9afac205d22a857fc05f1a74d51426b1a5beba3a9641326f9badf.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/5a75e35d90b9afac205d22a857fc05f1a74d51426b1a5beba3a9641326f9badf.jpg)

![661c735b22bae9e2dfcc8fa6eedb3344627e25fe3fe843677f2cb29b6bd3b920.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/661c735b22bae9e2dfcc8fa6eedb3344627e25fe3fe843677f2cb29b6bd3b920.jpg)

![682f4bb70429ad5c1a646d49d5bb35e05bd9f220932c4978dd565114d22ecbf0.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/682f4bb70429ad5c1a646d49d5bb35e05bd9f220932c4978dd565114d22ecbf0.jpg)

![6ed42fae6b3e14f17a3430345e845302cfdb995514adaf693c339132b4df1035.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/6ed42fae6b3e14f17a3430345e845302cfdb995514adaf693c339132b4df1035.jpg)

![6f417d01f2ba1e6c1e4ef0c6f7f72f0e592a1e9d4c223a67ddb068acf5908d10.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/6f417d01f2ba1e6c1e4ef0c6f7f72f0e592a1e9d4c223a67ddb068acf5908d10.jpg)

![6f898e99d46950f8c4a50f4aa87d24c3e3919bbb548613c90170e35c5a6f20e7.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/6f898e99d46950f8c4a50f4aa87d24c3e3919bbb548613c90170e35c5a6f20e7.jpg)

![704b9801f0b4fad419ded34cee552fa5b5eeb2dfc76b162b3109858d15957d43.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/704b9801f0b4fad419ded34cee552fa5b5eeb2dfc76b162b3109858d15957d43.jpg)

![72dff095cf0ab7c8601214b14b3ce7ea7f269c5c95a41dd255a5846dd18c16f2.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/72dff095cf0ab7c8601214b14b3ce7ea7f269c5c95a41dd255a5846dd18c16f2.jpg)

![7831bd9727e8b7bb2ebf15e963f554cdac5255edec2b060f1228b18be74e7899.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/7831bd9727e8b7bb2ebf15e963f554cdac5255edec2b060f1228b18be74e7899.jpg)

![83ceb51e1ecb890fa277fcfc201918a2595ef19f32374fae25951a5dc6b7fd0d.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/83ceb51e1ecb890fa277fcfc201918a2595ef19f32374fae25951a5dc6b7fd0d.jpg)

![8b8a696c1965dbea919d80252bfac4c362a6106053393d5ff03f0a21d1756028.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/8b8a696c1965dbea919d80252bfac4c362a6106053393d5ff03f0a21d1756028.jpg)

![9b822144aba58e3f216ac3fd2738109a20a3b157e9cd69e08b14a2899367af02.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/9b822144aba58e3f216ac3fd2738109a20a3b157e9cd69e08b14a2899367af02.jpg)

![aa1255a190f38befc915d9b852de6bac02103f169849a7ff459246fe90a03e76.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/aa1255a190f38befc915d9b852de6bac02103f169849a7ff459246fe90a03e76.jpg)

![afdcf4960c7923be93df4700c8ab6a1e0ffe6e86b6d87290b1e58e90943fc832.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/afdcf4960c7923be93df4700c8ab6a1e0ffe6e86b6d87290b1e58e90943fc832.jpg)

![bc6c1196408433f687a95e34895e336e5231821ee70e70ffc46d1c2f197b2666.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/bc6c1196408433f687a95e34895e336e5231821ee70e70ffc46d1c2f197b2666.jpg)

![bdcd450943172d574076dfe88f4263c0160306310ad1706093f66bfde0dc92a4.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/bdcd450943172d574076dfe88f4263c0160306310ad1706093f66bfde0dc92a4.jpg)

![be168c7b994784ad1adc6d2b0179adf834b12ea7a49c0f9d22e860a4412d31d6.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/be168c7b994784ad1adc6d2b0179adf834b12ea7a49c0f9d22e860a4412d31d6.jpg)

![be535d0a408cda3681d3a046899683cac6078bc2d1d9ecc64e5058038d28fb42.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/be535d0a408cda3681d3a046899683cac6078bc2d1d9ecc64e5058038d28fb42.jpg)

![c3722520ebccc2c1294352eb84b53c43620bccf3a4a75598557b991b1da8f15f.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/c3722520ebccc2c1294352eb84b53c43620bccf3a4a75598557b991b1da8f15f.jpg)

![d6b4c0e6837ea743a8b79e7ff7cdd3e4934f95a4cbfe86b24e3085a6cf2d1670.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/d6b4c0e6837ea743a8b79e7ff7cdd3e4934f95a4cbfe86b24e3085a6cf2d1670.jpg)

![dc95623ffa2f8424ff6fa00f7ecdd3ad3dcaf777c3b710a85cfc618263fccf22.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/dc95623ffa2f8424ff6fa00f7ecdd3ad3dcaf777c3b710a85cfc618263fccf22.jpg)

![dcb4e8dd8d9eb0b00ff6042931fa95d0a3b2cd9f347be74455d8c818d8797327.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/dcb4e8dd8d9eb0b00ff6042931fa95d0a3b2cd9f347be74455d8c818d8797327.jpg)

![e3a7a450580261137c4bf5e792957a4784ba9f68b4adec60d370c43ae4e5bce6.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/e3a7a450580261137c4bf5e792957a4784ba9f68b4adec60d370c43ae4e5bce6.jpg)

![e460925a09c618236a99b59f85ca25fac3648321e937b54a1c50b77c643fbf43.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/e460925a09c618236a99b59f85ca25fac3648321e937b54a1c50b77c643fbf43.jpg)

![ee6c35d5fcec1f7eed10de21cdac2f6c2f56013d3b88ecb979881a270fdae6ea.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/ee6c35d5fcec1f7eed10de21cdac2f6c2f56013d3b88ecb979881a270fdae6ea.jpg)

![effaddb83c5d607056e598d80c410d75f3042be615bdfc8d9ec0b324a3900a9e.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/effaddb83c5d607056e598d80c410d75f3042be615bdfc8d9ec0b324a3900a9e.jpg)

![f73684117357f8fc0e8c6292e98e1124062228056eb9975f7acf45313219a43a.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/f73684117357f8fc0e8c6292e98e1124062228056eb9975f7acf45313219a43a.jpg)

![fde8764bf7e70529cc2bbe423702337dda21e7e2dbb9b65e5bf7200232f2c585.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/images/fde8764bf7e70529cc2bbe423702337dda21e7e2dbb9b65e5bf7200232f2c585.jpg)

### Tables

![469520f39ee8acf1dd7374b36401429d369f2add9dfc3866b1427d5fe2a80ff1.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/tables/469520f39ee8acf1dd7374b36401429d369f2add9dfc3866b1427d5fe2a80ff1.jpg)

![55c2dda9d38eb515a617b0a92b5a92afc61b0fdebb30e8ac9b7941fdf15d3013.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/tables/55c2dda9d38eb515a617b0a92b5a92afc61b0fdebb30e8ac9b7941fdf15d3013.jpg)

![5a1bda40cda2e5ceb29c6f7901e563401a5108057e5ba47cdd7704639d121d70.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/tables/5a1bda40cda2e5ceb29c6f7901e563401a5108057e5ba47cdd7704639d121d70.jpg)

![6572ac22d2b530517f34c63621ad6d5f65e9a0941a32a7427d3c3084c842a57c.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/tables/6572ac22d2b530517f34c63621ad6d5f65e9a0941a32a7427d3c3084c842a57c.jpg)

![65fc3f47a1a544431e739c99c59889239d23d86ddb8b52d7f096bbc7235c6ab1.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/tables/65fc3f47a1a544431e739c99c59889239d23d86ddb8b52d7f096bbc7235c6ab1.jpg)

![9136d49625b20ce259fd116df2264f395bc4648dda30e4d0aa759b125b3195ab.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/tables/9136d49625b20ce259fd116df2264f395bc4648dda30e4d0aa759b125b3195ab.jpg)

![9616c7a83125dd82a43ae582177799c10c885030cceaae84283037a87f2c32d3.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/tables/9616c7a83125dd82a43ae582177799c10c885030cceaae84283037a87f2c32d3.jpg)

![9877aa529efccc7132f0a41a13d4f9d0e27ed9c9a16afc812bdbdf1c66f8344d.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/tables/9877aa529efccc7132f0a41a13d4f9d0e27ed9c9a16afc812bdbdf1c66f8344d.jpg)

![b660573e993656337f7b653006357c190a0eb0730fa66d1e3e440577ed695049.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/tables/b660573e993656337f7b653006357c190a0eb0730fa66d1e3e440577ed695049.jpg)

![b734ac926e019157fda4d733af522019e22dcd1cb25009ba823f8926295321d9.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/tables/b734ac926e019157fda4d733af522019e22dcd1cb25009ba823f8926295321d9.jpg)

![f0b9fa3910f9dc02aff048485ae9ca175dbc3f3532ca53ab83d0919d5d4f76c5.jpg](../icml_results/1922_MMPD-Bench_%20Bridging%20Multimodal%20Fission%20with%20Multi-Polarimetric%20Modalities%20Decomposition/tables/f0b9fa3910f9dc02aff048485ae9ca175dbc3f3532ca53ab83d0919d5d4f76c5.jpg)

<a id="paper-1898"></a>

## A Flat Vocabulary or a Rich Hierarchy? Re-introducing Intrinsic Structure Transforms the Autoregressive Image Generation


### Images

![4126f1c08af27f7433aff87b57def172e9ce22faabbd41a768cd737e5383cf9e.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/images/4126f1c08af27f7433aff87b57def172e9ce22faabbd41a768cd737e5383cf9e.jpg)

![4ea936e3faae6c75ad45849704a6704cf96b42803481a7d00bb1cea3930c9206.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/images/4ea936e3faae6c75ad45849704a6704cf96b42803481a7d00bb1cea3930c9206.jpg)

![65bce168d5e20ddef04e03d632734e25bd49a7b1edd99baed4bc17f4a6f4c10e.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/images/65bce168d5e20ddef04e03d632734e25bd49a7b1edd99baed4bc17f4a6f4c10e.jpg)

![98c510e91955f5c3b2c2fa351cf3a7d0901782c6ae80c494f29b8442261e88aa.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/images/98c510e91955f5c3b2c2fa351cf3a7d0901782c6ae80c494f29b8442261e88aa.jpg)

![a2c0fb5345e90acde6217d0e8d7788ac24c8955c04c6a979051f8c27a5900229.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/images/a2c0fb5345e90acde6217d0e8d7788ac24c8955c04c6a979051f8c27a5900229.jpg)

![bd86f780dd71c69af720d0c61373dbca062d0045e8abd0452ca0a0ed06b7227a.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/images/bd86f780dd71c69af720d0c61373dbca062d0045e8abd0452ca0a0ed06b7227a.jpg)

![e2261c8498911a7c8a5981743c776a4f9f120563fddb699305d593d509f7cf95.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/images/e2261c8498911a7c8a5981743c776a4f9f120563fddb699305d593d509f7cf95.jpg)

![edd58e9a18e62c9d3eb011d9af5471a6d701651c1e77039d4426c523e1f66738.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/images/edd58e9a18e62c9d3eb011d9af5471a6d701651c1e77039d4426c523e1f66738.jpg)

![f3219dded495692839d42b5e23856ba0e8e4e4cf6bf237c154e98e61b905a134.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/images/f3219dded495692839d42b5e23856ba0e8e4e4cf6bf237c154e98e61b905a134.jpg)

### Tables

![1d6e324b969941b041cc91088179c78c600bac0b2fcff22f97910907c7296229.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/tables/1d6e324b969941b041cc91088179c78c600bac0b2fcff22f97910907c7296229.jpg)

![475980f1f40c0023d9dd35d06c951e3ae6dc38fbbe4ded0e3b863a67630e1fcb.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/tables/475980f1f40c0023d9dd35d06c951e3ae6dc38fbbe4ded0e3b863a67630e1fcb.jpg)

![8e6b521782ad5e7100019775a9bdf2e72fe52bdac03c38be4d60f785c8da62b5.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/tables/8e6b521782ad5e7100019775a9bdf2e72fe52bdac03c38be4d60f785c8da62b5.jpg)

![a91d28ffd3b0daa086b53a5e1dd0a4fb2b8fa1355038ca8f87b61f9476fb4338.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/tables/a91d28ffd3b0daa086b53a5e1dd0a4fb2b8fa1355038ca8f87b61f9476fb4338.jpg)

![ac87cee12b7ea7b5029464d237e46100aa5c3574b0650786965f6f921fe26653.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/tables/ac87cee12b7ea7b5029464d237e46100aa5c3574b0650786965f6f921fe26653.jpg)

![aef4ce502954b1d9394927d122d6990d3562770e8da60577f54d6ce8a56765ca.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/tables/aef4ce502954b1d9394927d122d6990d3562770e8da60577f54d6ce8a56765ca.jpg)

![b0f66c0e14d9b0bc64074070b9bdb55bb7393ee68031d7ab8b007537a5556071.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/tables/b0f66c0e14d9b0bc64074070b9bdb55bb7393ee68031d7ab8b007537a5556071.jpg)

![d96602fa353727b3d8cf114a442101032a41cf08ef8429e9cbe780e6a2014a0c.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/tables/d96602fa353727b3d8cf114a442101032a41cf08ef8429e9cbe780e6a2014a0c.jpg)

![e86d8b876ccccdb4736d377fcc54b2cfe9eadc3568e68542828bb15d4cff65a9.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/tables/e86d8b876ccccdb4736d377fcc54b2cfe9eadc3568e68542828bb15d4cff65a9.jpg)

![ebf038f0671a932c55ec1993d904fe1e51caa12d0e05716eaf3e2fe2a8fc0ca6.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/tables/ebf038f0671a932c55ec1993d904fe1e51caa12d0e05716eaf3e2fe2a8fc0ca6.jpg)

![fdfe870226c6d036fc544329e39d07c71cbbb29c8da805fc2ae418c14f5ad576.jpg](../icml_results/1923_A%20Flat%20Vocabulary%20or%20a%20Rich%20Hierarchy_%20Re-introducing%20Intrinsic%20Structure%20Transforms%20the%20Autoregress/tables/fdfe870226c6d036fc544329e39d07c71cbbb29c8da805fc2ae418c14f5ad576.jpg)

<a id="paper-1899"></a>

## IAPO: Information-Aware Policy Optimization for Token-Efficient Reasoning


### Images

![006c1a43dc1cd6b0a74d8e7bc3d8cc5bb4eae4fabbd982f68e0b9286bb279f26.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/006c1a43dc1cd6b0a74d8e7bc3d8cc5bb4eae4fabbd982f68e0b9286bb279f26.jpg)

![133b396167eda8ff802c08e5aba1b0f4b60a56f3fa191a5e7f9f37082092bb1e.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/133b396167eda8ff802c08e5aba1b0f4b60a56f3fa191a5e7f9f37082092bb1e.jpg)

![1beec76917d2120cb0aa3bf5344e24d64e0d4aef90b160d72d1d667f9bb67c85.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/1beec76917d2120cb0aa3bf5344e24d64e0d4aef90b160d72d1d667f9bb67c85.jpg)

![1c8080b1a55481ba85a022eebbe4d69c594e58331cc06e47f0cc4cb4ab8a09a4.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/1c8080b1a55481ba85a022eebbe4d69c594e58331cc06e47f0cc4cb4ab8a09a4.jpg)

![2487aa99d8593015ed2193a94b97dff60b946a7e1675783da82a864869bc488e.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/2487aa99d8593015ed2193a94b97dff60b946a7e1675783da82a864869bc488e.jpg)

![283401577d94ea58d3c9828b55a6bd96e9dd5d19d91162a7d01cca0ae6d67b57.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/283401577d94ea58d3c9828b55a6bd96e9dd5d19d91162a7d01cca0ae6d67b57.jpg)

![34e73b99425a44be65de5e00d3d4deeecf26fed3d40e87fc6cc9c0547b7e477d.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/34e73b99425a44be65de5e00d3d4deeecf26fed3d40e87fc6cc9c0547b7e477d.jpg)

![3ae822cd2a063c738bb6bb2b959f7af0ce422a4368f326d4eaea67d1ae9d13fa.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/3ae822cd2a063c738bb6bb2b959f7af0ce422a4368f326d4eaea67d1ae9d13fa.jpg)

![4dca8c7933472088d624c50733f2ba132fc0a835d00a7e7221cf336934a5a4e9.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/4dca8c7933472088d624c50733f2ba132fc0a835d00a7e7221cf336934a5a4e9.jpg)

![500110729f21a4d80423ed383f6c7eac61f69e6606ea9899284b4f12d2a8cfd3.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/500110729f21a4d80423ed383f6c7eac61f69e6606ea9899284b4f12d2a8cfd3.jpg)

![62f3ddb61d71942cabaaf6df58520c8b817a878bc385f278646412f37f8f43a5.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/62f3ddb61d71942cabaaf6df58520c8b817a878bc385f278646412f37f8f43a5.jpg)

![6a58baa809e6a82375597d528277bbebc45b1397643f05dc7908b19cd97f838c.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/6a58baa809e6a82375597d528277bbebc45b1397643f05dc7908b19cd97f838c.jpg)

![9defa6ea3cc6eb863826b26e890ad57afb3c9fdce03ceb66a92571fb20ebebea.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/9defa6ea3cc6eb863826b26e890ad57afb3c9fdce03ceb66a92571fb20ebebea.jpg)

![9e2d6fd5fb07ffd6be2f130f0f8a3ef8acbcbeda355413a5f5664afc942ddc5e.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/9e2d6fd5fb07ffd6be2f130f0f8a3ef8acbcbeda355413a5f5664afc942ddc5e.jpg)

![a06870ffcfe736ac18a31ce50531595511acfb129065c70bfab791c273333e7f.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/a06870ffcfe736ac18a31ce50531595511acfb129065c70bfab791c273333e7f.jpg)

![adf4ffb33cfaa1057e2b888b39d0ea9b6943010d7ba2fce4fbbb0be5d069b2d6.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/adf4ffb33cfaa1057e2b888b39d0ea9b6943010d7ba2fce4fbbb0be5d069b2d6.jpg)

![ba5ede22135931ff939bbbcd9bc9166c415f7add90386a223d9fce2366cb1611.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/ba5ede22135931ff939bbbcd9bc9166c415f7add90386a223d9fce2366cb1611.jpg)

![c3cbb9bcba568cae5b9ce4689af08f97383f7246c076df4b5164c984f14bb1ca.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/c3cbb9bcba568cae5b9ce4689af08f97383f7246c076df4b5164c984f14bb1ca.jpg)

![c6275382ba5d67594274f79fc833c244294a0501747aa329e4d991609ebff22a.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/c6275382ba5d67594274f79fc833c244294a0501747aa329e4d991609ebff22a.jpg)

![c78e89c4361b4912b047c8f1b7ebd16694c1d14382777c9e9ce9cdc91a0bd659.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/c78e89c4361b4912b047c8f1b7ebd16694c1d14382777c9e9ce9cdc91a0bd659.jpg)

![cd420c15a3cecd8846ef02bcae712621e410a2f77ed9b91133fd4135f0cdca18.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/cd420c15a3cecd8846ef02bcae712621e410a2f77ed9b91133fd4135f0cdca18.jpg)

![e0e97b4c1ecb8abdb2ca0e82be351176caa75b54bc82ada01e786f995375cb48.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/e0e97b4c1ecb8abdb2ca0e82be351176caa75b54bc82ada01e786f995375cb48.jpg)

![f14dec249683d16db084ebac0ed20431af209f0c0980746e36d3acad49230854.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/f14dec249683d16db084ebac0ed20431af209f0c0980746e36d3acad49230854.jpg)

![f5bae15b07d5f6a4460ab57709a6e99b21c2ef71e903d93c8372238029f624ea.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/f5bae15b07d5f6a4460ab57709a6e99b21c2ef71e903d93c8372238029f624ea.jpg)

![f63afc84eecfe7357a0fcf356c1d537fd9e26a233d3b9ff40dc816dbe813e03a.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/f63afc84eecfe7357a0fcf356c1d537fd9e26a233d3b9ff40dc816dbe813e03a.jpg)

![f915c51094c158d4c69e40d3ee89b7f8726520d9c95f966552d503afec6005e9.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/images/f915c51094c158d4c69e40d3ee89b7f8726520d9c95f966552d503afec6005e9.jpg)

### Tables

![0052823f89bac88c245a5f50685cc094047c84f4ba93a4c715f0c32703f65e3d.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/tables/0052823f89bac88c245a5f50685cc094047c84f4ba93a4c715f0c32703f65e3d.jpg)

![048326db3560e9c32d81ab6f78bb73c31c7420e3ff0fcfe081b40407d269f00f.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/tables/048326db3560e9c32d81ab6f78bb73c31c7420e3ff0fcfe081b40407d269f00f.jpg)

![139af0100a653225f12816ee3a64a6afc6d7a0e8543c1d80308be537aab308ea.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/tables/139af0100a653225f12816ee3a64a6afc6d7a0e8543c1d80308be537aab308ea.jpg)

![2969f1b86ed9a271810e820250a70c6c5b8c7780b2b0dda16040b17010f2606d.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/tables/2969f1b86ed9a271810e820250a70c6c5b8c7780b2b0dda16040b17010f2606d.jpg)

![38e6d3b2d9b968bd63f8bdb2168c672f646da669e32e70fa61fb677603503ee6.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/tables/38e6d3b2d9b968bd63f8bdb2168c672f646da669e32e70fa61fb677603503ee6.jpg)

![47b9920d5b1c89c7db18f89e0a629a3a2a4756ebdf3ccca9ee3a75cab8e14a92.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/tables/47b9920d5b1c89c7db18f89e0a629a3a2a4756ebdf3ccca9ee3a75cab8e14a92.jpg)

![549f3702046eea97c5973be104aa95908a4cd8f5dd008a3cf952eef1051fc451.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/tables/549f3702046eea97c5973be104aa95908a4cd8f5dd008a3cf952eef1051fc451.jpg)

![59263862a06dd4f80684f68c7e1be86e28b813c881e1af8f5155449891ab450d.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/tables/59263862a06dd4f80684f68c7e1be86e28b813c881e1af8f5155449891ab450d.jpg)

![59cad18d3fa9a19f594a91c91cefeae6f334f5d62e210cdf0074e43857f5d997.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/tables/59cad18d3fa9a19f594a91c91cefeae6f334f5d62e210cdf0074e43857f5d997.jpg)

![76f1ba7f4bceecba1b604b41fe637ee83f0f9f469af6dace591733f62fdb98b1.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/tables/76f1ba7f4bceecba1b604b41fe637ee83f0f9f469af6dace591733f62fdb98b1.jpg)

![8501d81f0d87f80440baa519268e9127423316906cc2ad3c465ebe827c5a515f.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/tables/8501d81f0d87f80440baa519268e9127423316906cc2ad3c465ebe827c5a515f.jpg)

![9d97ddd126df23924a7aec6b19d43e769d4d9e4c7ed1acd86e0b5aea9131db77.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/tables/9d97ddd126df23924a7aec6b19d43e769d4d9e4c7ed1acd86e0b5aea9131db77.jpg)

![b36c0996c6d810bf0d693dcdd6061767948c48aad98c533da76243efbceb6139.jpg](../icml_results/1924_IAPO_%20Information-Aware%20Policy%20Optimization%20for%20Token-Efficient%20Reasoning/tables/b36c0996c6d810bf0d693dcdd6061767948c48aad98c533da76243efbceb6139.jpg)

<a id="paper-1900"></a>

## Optimizing Network Simulation: Enhancing Performance Prediction Accuracy via Neural Architecture Search


### Images

![0502f828d459779b2fe4acb2fc6008dd359c76c28bf8c2e7e1581c8de6cc7811.jpg](../icml_results/1925_Optimizing%20Network%20Simulation_%20Enhancing%20Performance%20Prediction%20Accuracy%20via%20Neural%20Architecture%20Sea/images/0502f828d459779b2fe4acb2fc6008dd359c76c28bf8c2e7e1581c8de6cc7811.jpg)

![278827ff50bfa7e93dcadaf043d6597df7c8af82878757a3e4847d426feec83e.jpg](../icml_results/1925_Optimizing%20Network%20Simulation_%20Enhancing%20Performance%20Prediction%20Accuracy%20via%20Neural%20Architecture%20Sea/images/278827ff50bfa7e93dcadaf043d6597df7c8af82878757a3e4847d426feec83e.jpg)

![31f2b3111f1cc703b4e1f3494c2b4492aab3e8a102cc69f85e5e7b7b183788fb.jpg](../icml_results/1925_Optimizing%20Network%20Simulation_%20Enhancing%20Performance%20Prediction%20Accuracy%20via%20Neural%20Architecture%20Sea/images/31f2b3111f1cc703b4e1f3494c2b4492aab3e8a102cc69f85e5e7b7b183788fb.jpg)

![44a94cc57ed5a483b2b803a1f11adbd607fa09d21537785f4572fac72491ab19.jpg](../icml_results/1925_Optimizing%20Network%20Simulation_%20Enhancing%20Performance%20Prediction%20Accuracy%20via%20Neural%20Architecture%20Sea/images/44a94cc57ed5a483b2b803a1f11adbd607fa09d21537785f4572fac72491ab19.jpg)

![7f42d575fc8c80bada9e1212be2309e80ff29ce592a1d8ba8a477b52d080d171.jpg](../icml_results/1925_Optimizing%20Network%20Simulation_%20Enhancing%20Performance%20Prediction%20Accuracy%20via%20Neural%20Architecture%20Sea/images/7f42d575fc8c80bada9e1212be2309e80ff29ce592a1d8ba8a477b52d080d171.jpg)

![f0e781d2434df6e7a4163472ecc8d1be57e0eb1c514842f8c82d98d1f8b1ed48.jpg](../icml_results/1925_Optimizing%20Network%20Simulation_%20Enhancing%20Performance%20Prediction%20Accuracy%20via%20Neural%20Architecture%20Sea/images/f0e781d2434df6e7a4163472ecc8d1be57e0eb1c514842f8c82d98d1f8b1ed48.jpg)

### Tables

![1241109139bcba3aeba77a3d5cb265227fa30e1c971248066d99db594e2799a5.jpg](../icml_results/1925_Optimizing%20Network%20Simulation_%20Enhancing%20Performance%20Prediction%20Accuracy%20via%20Neural%20Architecture%20Sea/tables/1241109139bcba3aeba77a3d5cb265227fa30e1c971248066d99db594e2799a5.jpg)

![126de56d0eb7bd3fb783488445606eb4a8be971e88289b4a33cc000b2ac46827.jpg](../icml_results/1925_Optimizing%20Network%20Simulation_%20Enhancing%20Performance%20Prediction%20Accuracy%20via%20Neural%20Architecture%20Sea/tables/126de56d0eb7bd3fb783488445606eb4a8be971e88289b4a33cc000b2ac46827.jpg)

![15cc44d8c9518a7cba458c23cbb07e0e4ae93499edc1a87a103ad1381a0d5653.jpg](../icml_results/1925_Optimizing%20Network%20Simulation_%20Enhancing%20Performance%20Prediction%20Accuracy%20via%20Neural%20Architecture%20Sea/tables/15cc44d8c9518a7cba458c23cbb07e0e4ae93499edc1a87a103ad1381a0d5653.jpg)

![282ed551f12a1ab8bfe10672884271cf5dd587ec081bdf70bcada8003287f7b3.jpg](../icml_results/1925_Optimizing%20Network%20Simulation_%20Enhancing%20Performance%20Prediction%20Accuracy%20via%20Neural%20Architecture%20Sea/tables/282ed551f12a1ab8bfe10672884271cf5dd587ec081bdf70bcada8003287f7b3.jpg)

![4cfaa50f2524e6184ed3d1458414455f5bbfa904494e70cfdc12b4e9fd76ac00.jpg](../icml_results/1925_Optimizing%20Network%20Simulation_%20Enhancing%20Performance%20Prediction%20Accuracy%20via%20Neural%20Architecture%20Sea/tables/4cfaa50f2524e6184ed3d1458414455f5bbfa904494e70cfdc12b4e9fd76ac00.jpg)

![5c0311bd179cd729c84d07099f86d6ff4161edf2a8a82a9c7479e6852f1ba55b.jpg](../icml_results/1925_Optimizing%20Network%20Simulation_%20Enhancing%20Performance%20Prediction%20Accuracy%20via%20Neural%20Architecture%20Sea/tables/5c0311bd179cd729c84d07099f86d6ff4161edf2a8a82a9c7479e6852f1ba55b.jpg)

![70730c84c233bd7ebe92687b0f41710b9c18c10002799c34cee4f92b24018ae7.jpg](../icml_results/1925_Optimizing%20Network%20Simulation_%20Enhancing%20Performance%20Prediction%20Accuracy%20via%20Neural%20Architecture%20Sea/tables/70730c84c233bd7ebe92687b0f41710b9c18c10002799c34cee4f92b24018ae7.jpg)

![aaf7f4f55ba6c5a076618c30383f69beec4c23b486809554ab058732ebab8e0d.jpg](../icml_results/1925_Optimizing%20Network%20Simulation_%20Enhancing%20Performance%20Prediction%20Accuracy%20via%20Neural%20Architecture%20Sea/tables/aaf7f4f55ba6c5a076618c30383f69beec4c23b486809554ab058732ebab8e0d.jpg)

![cc876e5674db6579ae60a40fd1f5490c4bc8d4592c461eb4e6697190f7a404f0.jpg](../icml_results/1925_Optimizing%20Network%20Simulation_%20Enhancing%20Performance%20Prediction%20Accuracy%20via%20Neural%20Architecture%20Sea/tables/cc876e5674db6579ae60a40fd1f5490c4bc8d4592c461eb4e6697190f7a404f0.jpg)

![ecef22499acd012aaea0c5053c2082b2f5bc783e14c2c284e21fba84dd3e3a6f.jpg](../icml_results/1925_Optimizing%20Network%20Simulation_%20Enhancing%20Performance%20Prediction%20Accuracy%20via%20Neural%20Architecture%20Sea/tables/ecef22499acd012aaea0c5053c2082b2f5bc783e14c2c284e21fba84dd3e3a6f.jpg)

![ffe02e2545b7d8ecafda5c928c7ecfb6b7ede8249aa27b9cef71ecf70e2b46d3.jpg](../icml_results/1925_Optimizing%20Network%20Simulation_%20Enhancing%20Performance%20Prediction%20Accuracy%20via%20Neural%20Architecture%20Sea/tables/ffe02e2545b7d8ecafda5c928c7ecfb6b7ede8249aa27b9cef71ecf70e2b46d3.jpg)

<a id="paper-1901"></a>

## SE(n)-Invariant Flow Matching: A General Framework with Application to Object Reassembly


### Images

![144b0e3d8548e93c011f0cbd3f06295d23be3817b2454ae28b5684507c67ebda.jpg](../icml_results/1926_SE%28n%29-Invariant%20Flow%20Matching_%20A%20General%20Framework%20with%20Application%20to%20Object%20Reassembly/images/144b0e3d8548e93c011f0cbd3f06295d23be3817b2454ae28b5684507c67ebda.jpg)

![2f1c80516307d6e78679c1a3f1218148f5c752bb3de472834b2ab91133c8c265.jpg](../icml_results/1926_SE%28n%29-Invariant%20Flow%20Matching_%20A%20General%20Framework%20with%20Application%20to%20Object%20Reassembly/images/2f1c80516307d6e78679c1a3f1218148f5c752bb3de472834b2ab91133c8c265.jpg)

### Tables

![411dc5252da7380fbc15c4666d5691235b995d8bf703e81981592a150f50da5a.jpg](../icml_results/1926_SE%28n%29-Invariant%20Flow%20Matching_%20A%20General%20Framework%20with%20Application%20to%20Object%20Reassembly/tables/411dc5252da7380fbc15c4666d5691235b995d8bf703e81981592a150f50da5a.jpg)

![57c580c218f121054ad8a399b444b0e178cc13b1de25fa75c74d2a608649080e.jpg](../icml_results/1926_SE%28n%29-Invariant%20Flow%20Matching_%20A%20General%20Framework%20with%20Application%20to%20Object%20Reassembly/tables/57c580c218f121054ad8a399b444b0e178cc13b1de25fa75c74d2a608649080e.jpg)

![5b05ef497a2c7d3faa5cdfeeb7416e08625c59fa07e3c86c3085b5c5d94520d4.jpg](../icml_results/1926_SE%28n%29-Invariant%20Flow%20Matching_%20A%20General%20Framework%20with%20Application%20to%20Object%20Reassembly/tables/5b05ef497a2c7d3faa5cdfeeb7416e08625c59fa07e3c86c3085b5c5d94520d4.jpg)

![72b999513d8ffc18fef62a133ec262eab9e15d1c407ddbd53524e6dc2d87c1e9.jpg](../icml_results/1926_SE%28n%29-Invariant%20Flow%20Matching_%20A%20General%20Framework%20with%20Application%20to%20Object%20Reassembly/tables/72b999513d8ffc18fef62a133ec262eab9e15d1c407ddbd53524e6dc2d87c1e9.jpg)

![860af8bb5d617c61c5ac367b11547d14424b9fa6046d7ef3c4fd3f5956ce14b4.jpg](../icml_results/1926_SE%28n%29-Invariant%20Flow%20Matching_%20A%20General%20Framework%20with%20Application%20to%20Object%20Reassembly/tables/860af8bb5d617c61c5ac367b11547d14424b9fa6046d7ef3c4fd3f5956ce14b4.jpg)

![8d4554e9566f62ec486746a23fa3db26b3672229841c4973e3d7031188598c99.jpg](../icml_results/1926_SE%28n%29-Invariant%20Flow%20Matching_%20A%20General%20Framework%20with%20Application%20to%20Object%20Reassembly/tables/8d4554e9566f62ec486746a23fa3db26b3672229841c4973e3d7031188598c99.jpg)

![aba25d5473074a3b6c65e95df3633a6c0f44c98393792cd97bdfc8e398a35658.jpg](../icml_results/1926_SE%28n%29-Invariant%20Flow%20Matching_%20A%20General%20Framework%20with%20Application%20to%20Object%20Reassembly/tables/aba25d5473074a3b6c65e95df3633a6c0f44c98393792cd97bdfc8e398a35658.jpg)

![b50abbf8bd1bab37c7240823f1ed1b33a5bbc2773326f7f0181516b0b8ee081f.jpg](../icml_results/1926_SE%28n%29-Invariant%20Flow%20Matching_%20A%20General%20Framework%20with%20Application%20to%20Object%20Reassembly/tables/b50abbf8bd1bab37c7240823f1ed1b33a5bbc2773326f7f0181516b0b8ee081f.jpg)

![c09f77e82780063635da4ab0f9235e548dbab022524aa26c552a90268a0a1ef4.jpg](../icml_results/1926_SE%28n%29-Invariant%20Flow%20Matching_%20A%20General%20Framework%20with%20Application%20to%20Object%20Reassembly/tables/c09f77e82780063635da4ab0f9235e548dbab022524aa26c552a90268a0a1ef4.jpg)

![cb5b222cbf394ed402f05024a15a14d4eb5924bdc347ee4d098f053afb2cac60.jpg](../icml_results/1926_SE%28n%29-Invariant%20Flow%20Matching_%20A%20General%20Framework%20with%20Application%20to%20Object%20Reassembly/tables/cb5b222cbf394ed402f05024a15a14d4eb5924bdc347ee4d098f053afb2cac60.jpg)

![db7acad2703d30c69477b7e17d0892339ebd91679217621f4a7250ee7dc6f479.jpg](../icml_results/1926_SE%28n%29-Invariant%20Flow%20Matching_%20A%20General%20Framework%20with%20Application%20to%20Object%20Reassembly/tables/db7acad2703d30c69477b7e17d0892339ebd91679217621f4a7250ee7dc6f479.jpg)

![f9802c94c68b7b71c21346c686b1a004d52c75970d97e32f8ddd8cb45433e737.jpg](../icml_results/1926_SE%28n%29-Invariant%20Flow%20Matching_%20A%20General%20Framework%20with%20Application%20to%20Object%20Reassembly/tables/f9802c94c68b7b71c21346c686b1a004d52c75970d97e32f8ddd8cb45433e737.jpg)

![fe245a28fef54e2950f0cfafcdfd963904a4d7d74a53bffd0475a755c536f8a9.jpg](../icml_results/1926_SE%28n%29-Invariant%20Flow%20Matching_%20A%20General%20Framework%20with%20Application%20to%20Object%20Reassembly/tables/fe245a28fef54e2950f0cfafcdfd963904a4d7d74a53bffd0475a755c536f8a9.jpg)

![fe5331de60aba73949eab4cd100641c9cd28bb5ac7fbe0dfb03c0149d6bc4f08.jpg](../icml_results/1926_SE%28n%29-Invariant%20Flow%20Matching_%20A%20General%20Framework%20with%20Application%20to%20Object%20Reassembly/tables/fe5331de60aba73949eab4cd100641c9cd28bb5ac7fbe0dfb03c0149d6bc4f08.jpg)

<a id="paper-1902"></a>

## Asymmetric Prompt Weighting for Reinforcement Learning with Verifiable Rewards


### Images

![2672107c9c2a34c470fc0b63f42dba1b9eb7a541a403421352e49a6d699dee61.jpg](../icml_results/1927_Asymmetric%20Prompt%20Weighting%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/2672107c9c2a34c470fc0b63f42dba1b9eb7a541a403421352e49a6d699dee61.jpg)

![2b2db2c0559b0f7d69a7ca8271b8604b727323c79dab23e8def7db454be55fa5.jpg](../icml_results/1927_Asymmetric%20Prompt%20Weighting%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/2b2db2c0559b0f7d69a7ca8271b8604b727323c79dab23e8def7db454be55fa5.jpg)

![2f360dfe053b52ec8a7cb2df149940fff227f81e75e708ac4fcb08654a052deb.jpg](../icml_results/1927_Asymmetric%20Prompt%20Weighting%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/2f360dfe053b52ec8a7cb2df149940fff227f81e75e708ac4fcb08654a052deb.jpg)

![8c5d4a0db418c6abe132d231f35538bb35cc53816cf8591c2b7405e9934a14fa.jpg](../icml_results/1927_Asymmetric%20Prompt%20Weighting%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/8c5d4a0db418c6abe132d231f35538bb35cc53816cf8591c2b7405e9934a14fa.jpg)

![98d610223e08800d22e071fb4229dbd159714feb6155abe6363828d9366657a6.jpg](../icml_results/1927_Asymmetric%20Prompt%20Weighting%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/98d610223e08800d22e071fb4229dbd159714feb6155abe6363828d9366657a6.jpg)

![a1bbc18caad97bf8ccc21d68ec7e0804fa51981d82622daffb61d2d5a8765390.jpg](../icml_results/1927_Asymmetric%20Prompt%20Weighting%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/a1bbc18caad97bf8ccc21d68ec7e0804fa51981d82622daffb61d2d5a8765390.jpg)

![a8cdd7326bd3dcef2538d683d0358c138b6fc7afe20c2ad6917b0e650957b615.jpg](../icml_results/1927_Asymmetric%20Prompt%20Weighting%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/a8cdd7326bd3dcef2538d683d0358c138b6fc7afe20c2ad6917b0e650957b615.jpg)

![aea86b200b37963bd2b3776be17fe0fc7acddc7ff4c6c1519a10eb64326208d2.jpg](../icml_results/1927_Asymmetric%20Prompt%20Weighting%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/aea86b200b37963bd2b3776be17fe0fc7acddc7ff4c6c1519a10eb64326208d2.jpg)

![cd6ab07aeb72ae13df6df9f53cd0a5d568242209a7559ea653ecef362afe9c62.jpg](../icml_results/1927_Asymmetric%20Prompt%20Weighting%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/cd6ab07aeb72ae13df6df9f53cd0a5d568242209a7559ea653ecef362afe9c62.jpg)

![db45925cb9788876c8ec009985066ce104ab059b9010e165c4f67e3db78b1c89.jpg](../icml_results/1927_Asymmetric%20Prompt%20Weighting%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/db45925cb9788876c8ec009985066ce104ab059b9010e165c4f67e3db78b1c89.jpg)

![e43e861bd680ec40b7339e8e43d83d3535f0a2eb7e75973e4dd47845739fa4c3.jpg](../icml_results/1927_Asymmetric%20Prompt%20Weighting%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/e43e861bd680ec40b7339e8e43d83d3535f0a2eb7e75973e4dd47845739fa4c3.jpg)

### Tables

![d8bbeee5cb1c9227efd8f37a7fee22b2f7ec7e6472d00ee5396dddfa9e0d2bcc.jpg](../icml_results/1927_Asymmetric%20Prompt%20Weighting%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/d8bbeee5cb1c9227efd8f37a7fee22b2f7ec7e6472d00ee5396dddfa9e0d2bcc.jpg)

![f361504d85c9269886ad21e7dcef04b74f417b852085d4f4af1a5c3ef6c0ac24.jpg](../icml_results/1927_Asymmetric%20Prompt%20Weighting%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/f361504d85c9269886ad21e7dcef04b74f417b852085d4f4af1a5c3ef6c0ac24.jpg)

<a id="paper-1903"></a>

## TN-SHAP-G: Graph-Structured Tensor Network Surrogates for Shapley Values and Interactions


### Images

![01b968bb3d3db6f781b4cb5c7456abbcd9cc1e41324c5815b1a4b0416b44cb04.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/images/01b968bb3d3db6f781b4cb5c7456abbcd9cc1e41324c5815b1a4b0416b44cb04.jpg)

![13e5c55bed275fd55b94647247eeb901c27d7f1740a4586c5c91f05211b3bfba.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/images/13e5c55bed275fd55b94647247eeb901c27d7f1740a4586c5c91f05211b3bfba.jpg)

![149047075f7b654e034b97398291eb00afde66d333e14b39bd7ecff9bc877b7d.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/images/149047075f7b654e034b97398291eb00afde66d333e14b39bd7ecff9bc877b7d.jpg)

![27fc30a35d188780e6cb9b8ec66f9d39d7f81fa40dbe05b38961d9416d53a08c.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/images/27fc30a35d188780e6cb9b8ec66f9d39d7f81fa40dbe05b38961d9416d53a08c.jpg)

![3c78119bf8f7b014d1b69ada87c4ae7e0e41efb81085ab2e207020551071350d.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/images/3c78119bf8f7b014d1b69ada87c4ae7e0e41efb81085ab2e207020551071350d.jpg)

![4235d6aaabfe55f2dbcc71560a8b5da08442e2e5e12743bc6e7203259ba8c99d.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/images/4235d6aaabfe55f2dbcc71560a8b5da08442e2e5e12743bc6e7203259ba8c99d.jpg)

![53226e6f7a1d29ee6ec7913d9ed578bd31a32b7a3e66b25a9edb4a6a8399ba30.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/images/53226e6f7a1d29ee6ec7913d9ed578bd31a32b7a3e66b25a9edb4a6a8399ba30.jpg)

![8fa3beae40ab5fa88f2d024523fd13c21821946b48b25bfe7f0304266ae02c52.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/images/8fa3beae40ab5fa88f2d024523fd13c21821946b48b25bfe7f0304266ae02c52.jpg)

![92e47d2643370eeb3d4c929c333f216a5a9e0a69d0adb3a278b1698e1117176d.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/images/92e47d2643370eeb3d4c929c333f216a5a9e0a69d0adb3a278b1698e1117176d.jpg)

![b0d0e0acb04583545281f3a2c7ebe982d1afbea18e153817f74c095934043d42.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/images/b0d0e0acb04583545281f3a2c7ebe982d1afbea18e153817f74c095934043d42.jpg)

![d8bc34fac825a938a3b9be4bb0e7ead738bfa74b44ed2eda10f4b58a70847b09.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/images/d8bc34fac825a938a3b9be4bb0e7ead738bfa74b44ed2eda10f4b58a70847b09.jpg)

![e6c2a27658f699c6cc13517e4834cfde886a2ee7c821dccfe04bcc5df44fa2ee.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/images/e6c2a27658f699c6cc13517e4834cfde886a2ee7c821dccfe04bcc5df44fa2ee.jpg)

![f89668f0f7fb4eeda53fcbcf8cdda8ce489c64346f89b65f2efe00bedd4d292e.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/images/f89668f0f7fb4eeda53fcbcf8cdda8ce489c64346f89b65f2efe00bedd4d292e.jpg)

### Tables

![0f00fdacde49ce95e5375b7fea7faf5f472ebbb9fd32f345a82c22ad517fa875.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/tables/0f00fdacde49ce95e5375b7fea7faf5f472ebbb9fd32f345a82c22ad517fa875.jpg)

![19c93f50edca191494e7f19a82e5174a6900940aa17b44c5e0eb8344255e7128.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/tables/19c93f50edca191494e7f19a82e5174a6900940aa17b44c5e0eb8344255e7128.jpg)

![30233a2fe3a00fb9ea17046114f143c88580338e1886bcd21241125b59c785a7.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/tables/30233a2fe3a00fb9ea17046114f143c88580338e1886bcd21241125b59c785a7.jpg)

![541a588783ac191aad88dd2a31ecb6c40183c16ca3e7e816af3be96d6f09342f.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/tables/541a588783ac191aad88dd2a31ecb6c40183c16ca3e7e816af3be96d6f09342f.jpg)

![7c183ce14143459d5f61d3c8cb982d115bcd7141b5f8fdefa04a77955c6901b1.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/tables/7c183ce14143459d5f61d3c8cb982d115bcd7141b5f8fdefa04a77955c6901b1.jpg)

![80a3fcc978eaf2f970f94fd3f4c2df4ae78542f57594f9e4c4cda190acfac00c.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/tables/80a3fcc978eaf2f970f94fd3f4c2df4ae78542f57594f9e4c4cda190acfac00c.jpg)

![b32a6a853e7075f4d37ba7bb02ffd38d6a4a830970e7d343f8754c401a07ce52.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/tables/b32a6a853e7075f4d37ba7bb02ffd38d6a4a830970e7d343f8754c401a07ce52.jpg)

![b3d33f833b02f5415835df8df1a6f5dda15da1525c378ee7645412848ccf9c3f.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/tables/b3d33f833b02f5415835df8df1a6f5dda15da1525c378ee7645412848ccf9c3f.jpg)

![b4ec3df5bc9b1812554a50b834a3fcb9c175e66deff61acf3a84a0560505f5c8.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/tables/b4ec3df5bc9b1812554a50b834a3fcb9c175e66deff61acf3a84a0560505f5c8.jpg)

![c41ef6f9dc660a25fb0e89365bc5e4ca9052bff03db6765e09f1431cece89aa3.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/tables/c41ef6f9dc660a25fb0e89365bc5e4ca9052bff03db6765e09f1431cece89aa3.jpg)

![c939efc3ce0d23edbd3c344ea3141557ef1afe15753305c1403a947b04a50e6c.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/tables/c939efc3ce0d23edbd3c344ea3141557ef1afe15753305c1403a947b04a50e6c.jpg)

![e196e98952ffee2b0f794f88143c904368765be77148eed161a31a1aa5623bd7.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/tables/e196e98952ffee2b0f794f88143c904368765be77148eed161a31a1aa5623bd7.jpg)

![f130f2fa219a185b4a7d2479f6dd9fd50df588865a9797be58b79849b591fd2f.jpg](../icml_results/1928_TN-SHAP-G_%20Graph-Structured%20Tensor%20Network%20Surrogates%20for%20Shapley%20Values%20and%20Interactions/tables/f130f2fa219a185b4a7d2479f6dd9fd50df588865a9797be58b79849b591fd2f.jpg)

<a id="paper-1904"></a>

## From Rashomon Theory to PRAXIS: Efficient Decision Tree Rashomon Sets


### Images

![00bb5db94b6b3a873d837626a62bc6faa43ccd96f9c3cbc01fc4461082d588cd.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/00bb5db94b6b3a873d837626a62bc6faa43ccd96f9c3cbc01fc4461082d588cd.jpg)

![0b918401c49eeeafdb79f8577043e3a3c50432540f6b96710024a1e06ef458a9.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/0b918401c49eeeafdb79f8577043e3a3c50432540f6b96710024a1e06ef458a9.jpg)

![0e896f6c913324171a160aa72115d9ceabe9100aa36c5cc56c641898967aa02e.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/0e896f6c913324171a160aa72115d9ceabe9100aa36c5cc56c641898967aa02e.jpg)

![121d33aa775131292c1f79515298d4a3b8f241b5927aa98d511e49c152f9c28b.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/121d33aa775131292c1f79515298d4a3b8f241b5927aa98d511e49c152f9c28b.jpg)

![124f7b1752daee13a512817251a238a6a14c0739451c1ff83ae0e5ef6af1824d.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/124f7b1752daee13a512817251a238a6a14c0739451c1ff83ae0e5ef6af1824d.jpg)

![137a55a297ea62c90a4aa86316a7d63ce7fbeed20404f2a3cfcd3e1bacf8dc61.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/137a55a297ea62c90a4aa86316a7d63ce7fbeed20404f2a3cfcd3e1bacf8dc61.jpg)

![1aef48aaf120ad8e21e86ae24f73de5f3421d8894da2a997a93bee1fb2493dfa.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/1aef48aaf120ad8e21e86ae24f73de5f3421d8894da2a997a93bee1fb2493dfa.jpg)

![1eaa68c9170c43bee5e4c1be9af005ce6f925fed012d407743c87ff083aee336.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/1eaa68c9170c43bee5e4c1be9af005ce6f925fed012d407743c87ff083aee336.jpg)

![1ec9fe94bfc0905fc532f29a2c0c461f1df991a2f65227fc19bb10fc88d8587d.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/1ec9fe94bfc0905fc532f29a2c0c461f1df991a2f65227fc19bb10fc88d8587d.jpg)

![1f3348438067ea6dd76c0c23b439b1120d75aacae7b0e5a1a8e0b226dd042087.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/1f3348438067ea6dd76c0c23b439b1120d75aacae7b0e5a1a8e0b226dd042087.jpg)

![207792f3d08808177ff644523a6b883b1c5d22df80e602319bac5d78d49a4ef5.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/207792f3d08808177ff644523a6b883b1c5d22df80e602319bac5d78d49a4ef5.jpg)

![251918c1d91e6abd8c477e22fad159169c2931f3ec92210f713743127876289e.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/251918c1d91e6abd8c477e22fad159169c2931f3ec92210f713743127876289e.jpg)

![2a9822d5330296ff2e5055d3804a0f9edaf50c77a8117372628e36c9bec932e9.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/2a9822d5330296ff2e5055d3804a0f9edaf50c77a8117372628e36c9bec932e9.jpg)

![347024832cde555ce9c160ccadc76cb0ac0eede42db055756c7f798067fe8f23.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/347024832cde555ce9c160ccadc76cb0ac0eede42db055756c7f798067fe8f23.jpg)

![36dfdf0deff8b86ba886928f8ba7be83578a42437e45cf0804d8d9b2ffddfdd6.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/36dfdf0deff8b86ba886928f8ba7be83578a42437e45cf0804d8d9b2ffddfdd6.jpg)

![3a41e4c9a92aa11610d5836ddc5e0515d381eb74a64bdec0f76e765152dfa1d0.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/3a41e4c9a92aa11610d5836ddc5e0515d381eb74a64bdec0f76e765152dfa1d0.jpg)

![41dc7fbd86918245abce5fa05f8808c3e8eb54a520e8414842975f9ce98369f6.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/41dc7fbd86918245abce5fa05f8808c3e8eb54a520e8414842975f9ce98369f6.jpg)

![43ad221aba85ff156f0d483f875bc90110bb41f0ec196f04a7464a926750fe11.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/43ad221aba85ff156f0d483f875bc90110bb41f0ec196f04a7464a926750fe11.jpg)

![469c9016eff02789d9c2b466004478fc52a535b4cca1220ddcd08705ac7bf8e1.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/469c9016eff02789d9c2b466004478fc52a535b4cca1220ddcd08705ac7bf8e1.jpg)

![47f2e134cd792953e620bf87c457f984405b4bd0b5eb81c7893af62c8be80099.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/47f2e134cd792953e620bf87c457f984405b4bd0b5eb81c7893af62c8be80099.jpg)

![48e22865ff2f30a432f3943bc562f2a6c4c6399f7113b00dadcdedb647959bf0.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/48e22865ff2f30a432f3943bc562f2a6c4c6399f7113b00dadcdedb647959bf0.jpg)

![497476fc7046fba71b0dbcc29d2e24ef393a576c52cf47ee5f51c2be213865f2.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/497476fc7046fba71b0dbcc29d2e24ef393a576c52cf47ee5f51c2be213865f2.jpg)

![4c1dab835c9b92be83e2b0d8608fec1936ae1ea44211fede320377e28099dc99.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/4c1dab835c9b92be83e2b0d8608fec1936ae1ea44211fede320377e28099dc99.jpg)

![51ac993f45ef69e6de5c985171d71d719fbb18db1672709b786345457641bafe.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/51ac993f45ef69e6de5c985171d71d719fbb18db1672709b786345457641bafe.jpg)

![55621be189b019a506858a9dd2a85f84244fcd1a1cc4b837c95e7aba12f6f1e7.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/55621be189b019a506858a9dd2a85f84244fcd1a1cc4b837c95e7aba12f6f1e7.jpg)

![5564cf1552c1b673baa1be368900c27833b5fbf876391fd3d3017ac5ce0625ea.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/5564cf1552c1b673baa1be368900c27833b5fbf876391fd3d3017ac5ce0625ea.jpg)

![57aa78dd82ee0dcc5ab053f12d90661495974288e4cc3e787d5df9c99d5dc05c.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/57aa78dd82ee0dcc5ab053f12d90661495974288e4cc3e787d5df9c99d5dc05c.jpg)

![584e54f748726e7aa73cf785a629689f09f151326df2ab01601624520b6a93cc.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/584e54f748726e7aa73cf785a629689f09f151326df2ab01601624520b6a93cc.jpg)

![588228bae1ad20d70a0fb6071de9e2466c736356d15386e0744dc424c5b8f71a.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/588228bae1ad20d70a0fb6071de9e2466c736356d15386e0744dc424c5b8f71a.jpg)

![58ca2cefb461d4614b89d4890d97e345895e4521c9c2494638d5c9b0c2a14219.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/58ca2cefb461d4614b89d4890d97e345895e4521c9c2494638d5c9b0c2a14219.jpg)

![5bc18c8851e1098a002b46e11dc46e192f0d820cc147f4f11564f04994b7ebf3.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/5bc18c8851e1098a002b46e11dc46e192f0d820cc147f4f11564f04994b7ebf3.jpg)

![5fec6eea52c7b26ae62f2d51a111702a6c06a68c66236dbe4d98c6b4340fc0a6.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/5fec6eea52c7b26ae62f2d51a111702a6c06a68c66236dbe4d98c6b4340fc0a6.jpg)

![60a5b3c34f4451dcd3a9dd703cd079204b731eaba8710863d5b873188397fe19.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/60a5b3c34f4451dcd3a9dd703cd079204b731eaba8710863d5b873188397fe19.jpg)

![65b1e9036f4436294cd2a95fd4ede09140d4bcbd3cd86aa60a4060f6407d108c.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/65b1e9036f4436294cd2a95fd4ede09140d4bcbd3cd86aa60a4060f6407d108c.jpg)

![65b52ae6c1495da7bb7779694be934e9ce338d8a563f39b97a535ee9f7764068.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/65b52ae6c1495da7bb7779694be934e9ce338d8a563f39b97a535ee9f7764068.jpg)

![6604bfe2a6c36caabf5bc959e67aed0a717197d66fa790d79a24be44e2b90ac7.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/6604bfe2a6c36caabf5bc959e67aed0a717197d66fa790d79a24be44e2b90ac7.jpg)

![6ab95c35e638c606d3a3e70dba456161ec771ca82d4233bc0e4323ed6ef2d982.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/6ab95c35e638c606d3a3e70dba456161ec771ca82d4233bc0e4323ed6ef2d982.jpg)

![6d6dae3b64770051c5b55cbc591ab73ef0a0111f3a88ebdf83270ccc7240e51c.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/6d6dae3b64770051c5b55cbc591ab73ef0a0111f3a88ebdf83270ccc7240e51c.jpg)

![6e07b63f8d2a0d33e8e1b87aaab045d63601913166dd026dfd01da89008f1e4d.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/6e07b63f8d2a0d33e8e1b87aaab045d63601913166dd026dfd01da89008f1e4d.jpg)

![6f286b5cb7f579a8fe759769a67a609cebc358e6fd1a44c49abaa0ee69a6b783.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/6f286b5cb7f579a8fe759769a67a609cebc358e6fd1a44c49abaa0ee69a6b783.jpg)

![6f9c5a42283d346fcad4315e50be81100848961cebf68e32c9a4e2240ff3bed6.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/6f9c5a42283d346fcad4315e50be81100848961cebf68e32c9a4e2240ff3bed6.jpg)

![71524503eabf1016a2c3b199c743becf1f35dcfab8f61c9a4966925d55473028.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/71524503eabf1016a2c3b199c743becf1f35dcfab8f61c9a4966925d55473028.jpg)

![743497b7959ee597f48a11764302fc7d8894e96cf92f8712ab90610b4b287864.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/743497b7959ee597f48a11764302fc7d8894e96cf92f8712ab90610b4b287864.jpg)

![7766dbff5eba47290d0a1edfbc0dde4e79527403eedaa4227f0791217342eb34.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/7766dbff5eba47290d0a1edfbc0dde4e79527403eedaa4227f0791217342eb34.jpg)

![78c8e3e33d8438e499802afcf84b050b8ed919fc40b6eef86f8573d8f34e1d90.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/78c8e3e33d8438e499802afcf84b050b8ed919fc40b6eef86f8573d8f34e1d90.jpg)

![7db894384e2cd96125ff4dbd696a2f0fb091b8d32855bbfa9647ea326e0137da.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/7db894384e2cd96125ff4dbd696a2f0fb091b8d32855bbfa9647ea326e0137da.jpg)

![7ec9ca34b26f6e21ac8804398774094a03ff45c7c419913f62e0e2f461f5cf2d.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/7ec9ca34b26f6e21ac8804398774094a03ff45c7c419913f62e0e2f461f5cf2d.jpg)

![807eee6889cd88d65841403b6bcdbe71ba658d71350c09f4b5276a412fdf52fc.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/807eee6889cd88d65841403b6bcdbe71ba658d71350c09f4b5276a412fdf52fc.jpg)

![84216188f2f53f8ad6d9ea85cbd98ae7df2f4b661ea412ac82b3bd1193d02023.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/84216188f2f53f8ad6d9ea85cbd98ae7df2f4b661ea412ac82b3bd1193d02023.jpg)

![86685528840ea8a81dbe5486e2b0d7f51932ce090ff3d4c2bdea36c96d09006c.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/86685528840ea8a81dbe5486e2b0d7f51932ce090ff3d4c2bdea36c96d09006c.jpg)

![8d4a8c70df0a9bf75fa158a19ff915de982b98dc43b29541a0f168f3f644f7e7.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/8d4a8c70df0a9bf75fa158a19ff915de982b98dc43b29541a0f168f3f644f7e7.jpg)

![8e7628994a5f2d2cd5c81433222cd3cc6ecd70305344ec316da2e281c989ff44.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/8e7628994a5f2d2cd5c81433222cd3cc6ecd70305344ec316da2e281c989ff44.jpg)

![8e9bd692dc603542ac17db4213ae3866f5afce1c791460e35c6f4cc3fa75f129.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/8e9bd692dc603542ac17db4213ae3866f5afce1c791460e35c6f4cc3fa75f129.jpg)

![912d675d8b5032565b4794d7705562a9dffec1985ebde943273ec88121c65ca6.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/912d675d8b5032565b4794d7705562a9dffec1985ebde943273ec88121c65ca6.jpg)

![95fc8bb515048254c776ae57ebda96545653e6e2577030a92cda4cd15dcde40b.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/95fc8bb515048254c776ae57ebda96545653e6e2577030a92cda4cd15dcde40b.jpg)

![9702319416e44499b2aee6e4786d251557443dfb0669dbdbc20320c7ca32ca2a.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/9702319416e44499b2aee6e4786d251557443dfb0669dbdbc20320c7ca32ca2a.jpg)

![a38a1dd2ff2b1860b9af962d9e88d982c19d42dbbd73048fcf95130be677f28f.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/a38a1dd2ff2b1860b9af962d9e88d982c19d42dbbd73048fcf95130be677f28f.jpg)

![a3aed34ec43335af7dc295a35d93f2b40e711cac669db4eea66e667430e1eef0.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/a3aed34ec43335af7dc295a35d93f2b40e711cac669db4eea66e667430e1eef0.jpg)

![a6a4bd4a32504607e297cd32079975cc9e180650dc585c567da488fe28b46e7e.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/a6a4bd4a32504607e297cd32079975cc9e180650dc585c567da488fe28b46e7e.jpg)

![a8aa7df9ec5f3ca46fa9b81c6f21e06d195df127744d44b84ca5addf4e80d146.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/a8aa7df9ec5f3ca46fa9b81c6f21e06d195df127744d44b84ca5addf4e80d146.jpg)

![a9df2f0d50ab589eb0c160203e7995122d227f6f834789e28cd04ed5c952666b.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/a9df2f0d50ab589eb0c160203e7995122d227f6f834789e28cd04ed5c952666b.jpg)

![aa20f3396ba9d5d56cdebc52b6ab63f01e5de114748af7f4e2b3dc9364229bcb.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/aa20f3396ba9d5d56cdebc52b6ab63f01e5de114748af7f4e2b3dc9364229bcb.jpg)

![af9e803884e21798917a463754daad87555172b83d5440a141f3dbd96f7516d3.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/af9e803884e21798917a463754daad87555172b83d5440a141f3dbd96f7516d3.jpg)

![b10b462e7a9d93ed635ab6533dd3b17567edb99d238d86525adc82c723d4c3bd.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/b10b462e7a9d93ed635ab6533dd3b17567edb99d238d86525adc82c723d4c3bd.jpg)

![b509bc8d1bf1d361e31fbb23e105e202b7851a8af34c08aa956822d715518a49.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/b509bc8d1bf1d361e31fbb23e105e202b7851a8af34c08aa956822d715518a49.jpg)

![bc62fa3fd0a6a8ea714a7151fd7ff2433d088ae3a1adca7d813c94ac76afe5a8.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/bc62fa3fd0a6a8ea714a7151fd7ff2433d088ae3a1adca7d813c94ac76afe5a8.jpg)

![bc66eac95f04204cb0a04163d430d138ad00ad7d31e1be7fc9eb58d2a79c57b7.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/bc66eac95f04204cb0a04163d430d138ad00ad7d31e1be7fc9eb58d2a79c57b7.jpg)

![bf3c590bf7bdc6bbf2ca52818e7c9574dc938a728e5dbdaff882de9166e464f8.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/bf3c590bf7bdc6bbf2ca52818e7c9574dc938a728e5dbdaff882de9166e464f8.jpg)

![bf904693fa0dcbe2a30b0c1e55408f2a5c7e570ac38c723c1358736c4da57a3f.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/bf904693fa0dcbe2a30b0c1e55408f2a5c7e570ac38c723c1358736c4da57a3f.jpg)

![c2060afa4e45ab3fc3b130b6dfdbc7a6b2b7980830599055d8e9998b1d4c640b.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/c2060afa4e45ab3fc3b130b6dfdbc7a6b2b7980830599055d8e9998b1d4c640b.jpg)

![c2e1b1c741b00166ba21e7873fc6fd685b80aa06309ba431a8a748b8738f1a58.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/c2e1b1c741b00166ba21e7873fc6fd685b80aa06309ba431a8a748b8738f1a58.jpg)

![cab8913280cc9e2a2c0f0be71eba06ad04bee0481ac0780bcc56c35642736e09.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/cab8913280cc9e2a2c0f0be71eba06ad04bee0481ac0780bcc56c35642736e09.jpg)

![ce4abe0811e7de79dbfc372537c13e336f3afbb64d1e068a9d9fb3f6d9cb7662.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/ce4abe0811e7de79dbfc372537c13e336f3afbb64d1e068a9d9fb3f6d9cb7662.jpg)

![d0418ca6805ca822ff3e4730360446d3a65dab1349c14cb0f6c983f531e17ec3.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/d0418ca6805ca822ff3e4730360446d3a65dab1349c14cb0f6c983f531e17ec3.jpg)

![d69591a1b7477a20b6696fcc91832e4dc3ec7d132fcc1e434b99463167b52c5e.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/d69591a1b7477a20b6696fcc91832e4dc3ec7d132fcc1e434b99463167b52c5e.jpg)

![d9349244e4813c740e4736bbec95242a5d3dd94a805cd592f561f6b5f06e7e4a.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/d9349244e4813c740e4736bbec95242a5d3dd94a805cd592f561f6b5f06e7e4a.jpg)

![d9424fabd628ab12daf80c0d383298fc7ac2ab2560e4d20c3ccbbe5c2b8b4056.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/d9424fabd628ab12daf80c0d383298fc7ac2ab2560e4d20c3ccbbe5c2b8b4056.jpg)

![db83d4bec7755f6acedeaa629f1ff51fcc226ecfbc7d07ccae14afdb149699bc.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/db83d4bec7755f6acedeaa629f1ff51fcc226ecfbc7d07ccae14afdb149699bc.jpg)

![dba02ed313759403e9b017f67e0a8cddcf0e0abb2e8c4ac274056b2b4969b8f3.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/dba02ed313759403e9b017f67e0a8cddcf0e0abb2e8c4ac274056b2b4969b8f3.jpg)

![e349d49b023340a93dc28b225b38e6e5ba096193e6807ac6f4f85be75c206055.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/e349d49b023340a93dc28b225b38e6e5ba096193e6807ac6f4f85be75c206055.jpg)

![e4095635532da9f33aa065421dce5f5a2803a64147e475300c07acff127bb630.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/e4095635532da9f33aa065421dce5f5a2803a64147e475300c07acff127bb630.jpg)

![e74e0b273cd2093fd17df2045e34e53fad1e5de03811773e07db0db852e3578d.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/e74e0b273cd2093fd17df2045e34e53fad1e5de03811773e07db0db852e3578d.jpg)

![eb0c10bc5f6afd3befd13f4d215ccc94f836f2e0c4842656502b25aa6dfed56b.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/eb0c10bc5f6afd3befd13f4d215ccc94f836f2e0c4842656502b25aa6dfed56b.jpg)

![eb36b8de15e549c9be6e55506f00156bb31f3cedf4f8f5ab77d51f8a8d92c9f7.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/eb36b8de15e549c9be6e55506f00156bb31f3cedf4f8f5ab77d51f8a8d92c9f7.jpg)

![eb49446cd086cc9c7cac96e85cb6d66550968038f814fbd07e8238e2cc923fc5.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/eb49446cd086cc9c7cac96e85cb6d66550968038f814fbd07e8238e2cc923fc5.jpg)

![ecb9be529de89c12e8272b7dce7d35fcd203a2149013f7452515fa5d5bcf216f.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/ecb9be529de89c12e8272b7dce7d35fcd203a2149013f7452515fa5d5bcf216f.jpg)

![eed2f5f19ff77b8f8508434946ba8c0642f12eba796bb7c59ae6f786c306ebfa.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/eed2f5f19ff77b8f8508434946ba8c0642f12eba796bb7c59ae6f786c306ebfa.jpg)

![ef8076a92c4f9aba078e59fa2f4e16555e137600c6c500f487cc3eaf11a16dae.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/ef8076a92c4f9aba078e59fa2f4e16555e137600c6c500f487cc3eaf11a16dae.jpg)

![f40e4eb1c6387d3c9cede78e1b8400eb9bec1ac1ad2a72d3a807f212f9bd74f6.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/f40e4eb1c6387d3c9cede78e1b8400eb9bec1ac1ad2a72d3a807f212f9bd74f6.jpg)

![f5192d8f24ca692951b5eb571f56fb841b691d49503b04a0084b11c1adb64bc4.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/f5192d8f24ca692951b5eb571f56fb841b691d49503b04a0084b11c1adb64bc4.jpg)

![fe4aa990defa5de3e006f3c24dad175af0ee1a9abab3988313332ecb2d4ef2da.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/images/fe4aa990defa5de3e006f3c24dad175af0ee1a9abab3988313332ecb2d4ef2da.jpg)

### Tables

![135fb19f4fb2002405eb7271dd20c4da3dcbdec1cbf93f0b2a0f5768e1e73ed7.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/135fb19f4fb2002405eb7271dd20c4da3dcbdec1cbf93f0b2a0f5768e1e73ed7.jpg)

![22b4dde6b0cf13ce993a1aa6c66c8d73b280b2ba613ac41e8edf0e953a564fff.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/22b4dde6b0cf13ce993a1aa6c66c8d73b280b2ba613ac41e8edf0e953a564fff.jpg)

![24e2fd411821e3bfc32dc9f14394552d5b3e0be40915a925f8f3b5413ddcc070.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/24e2fd411821e3bfc32dc9f14394552d5b3e0be40915a925f8f3b5413ddcc070.jpg)

![2622025cce7ae2970e11cbce039307f9eaf2e0196a10f5b4b985d7c20e789567.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/2622025cce7ae2970e11cbce039307f9eaf2e0196a10f5b4b985d7c20e789567.jpg)

![3283484c39adfad59aa30f548790d37b97c3c0e43729f465a98a1c4b2b9c9ab7.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/3283484c39adfad59aa30f548790d37b97c3c0e43729f465a98a1c4b2b9c9ab7.jpg)

![347c69aed0f0bdde6782c47edbcd1c0f88496db1483d484c52f59386aebc1387.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/347c69aed0f0bdde6782c47edbcd1c0f88496db1483d484c52f59386aebc1387.jpg)

![428e57287b93c014d5097ef5546bbab63954f69fde2b0d8580f622e1836ec4dc.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/428e57287b93c014d5097ef5546bbab63954f69fde2b0d8580f622e1836ec4dc.jpg)

![4a417b8170e932c93267a56e652e0552397df82ebe44c976a048d78769a02fb5.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/4a417b8170e932c93267a56e652e0552397df82ebe44c976a048d78769a02fb5.jpg)

![591f69155d9fb15880107b9598a261ab8b7efb952194ae0619228d07f459d51e.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/591f69155d9fb15880107b9598a261ab8b7efb952194ae0619228d07f459d51e.jpg)

![5d184299bc4ab75c9152b8830ce3e3d2fd98144dc832d0982b39292ce00cb262.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/5d184299bc4ab75c9152b8830ce3e3d2fd98144dc832d0982b39292ce00cb262.jpg)

![61f864df2e2c29d00ddc68c19468e940e28d5720cbe8839fe894e0c5107107d8.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/61f864df2e2c29d00ddc68c19468e940e28d5720cbe8839fe894e0c5107107d8.jpg)

![63671c6cd5163b30c0ef345c2a2bb80a8499a62f1f2a17e71227b97ceebc60d1.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/63671c6cd5163b30c0ef345c2a2bb80a8499a62f1f2a17e71227b97ceebc60d1.jpg)

![657d13ab20bd289f6990f5f034fe261082e3382a37b675bf6c529ec158e0aed3.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/657d13ab20bd289f6990f5f034fe261082e3382a37b675bf6c529ec158e0aed3.jpg)

![6971182ca42919755b3f7914557735e97ebfa6a8c90ec86585f300a165278b7d.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/6971182ca42919755b3f7914557735e97ebfa6a8c90ec86585f300a165278b7d.jpg)

![8e6e5145b4a24287ae4ec4e86ea5f0f6ae44613254e38cd98592f554c410bc1f.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/8e6e5145b4a24287ae4ec4e86ea5f0f6ae44613254e38cd98592f554c410bc1f.jpg)

![924cf9aa72774aa5024c4cf3bbedc5c88e77a5a84dd422822e31722f92004832.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/924cf9aa72774aa5024c4cf3bbedc5c88e77a5a84dd422822e31722f92004832.jpg)

![981531c5dabe41cb50a215fc12b0b34f4a1566ba9353b033cd6825dcb389003d.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/981531c5dabe41cb50a215fc12b0b34f4a1566ba9353b033cd6825dcb389003d.jpg)

![9c0ba2316f24113ac454e70b6316fd0c90fb81b0b664e7d2e7d89679180c9b91.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/9c0ba2316f24113ac454e70b6316fd0c90fb81b0b664e7d2e7d89679180c9b91.jpg)

![9de3856be57dab7b8bd5e250e6adbdc2a507e3bcb59e28d1f055eaf68cca1314.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/9de3856be57dab7b8bd5e250e6adbdc2a507e3bcb59e28d1f055eaf68cca1314.jpg)

![a132a149310111e43782f56c0bb878da1d300248525faffbdc66bc796855dbac.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/a132a149310111e43782f56c0bb878da1d300248525faffbdc66bc796855dbac.jpg)

![a15e2a29d743377be5a2939ae68e6f11d576700060a8ec04c826151b6fb86043.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/a15e2a29d743377be5a2939ae68e6f11d576700060a8ec04c826151b6fb86043.jpg)

![ad722dc32fb2dd2d4a56febb2f47552c7848a543414aebc47e4010dcdeeca9fb.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/ad722dc32fb2dd2d4a56febb2f47552c7848a543414aebc47e4010dcdeeca9fb.jpg)

![b4732138982ebf806afdcc94113c1f38e816279b484f3f2348fdc24d895e2303.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/b4732138982ebf806afdcc94113c1f38e816279b484f3f2348fdc24d895e2303.jpg)

![b904f77b52965295cab3f01522f3789ff224d7ec85256daef8d824490b003431.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/b904f77b52965295cab3f01522f3789ff224d7ec85256daef8d824490b003431.jpg)

![b95dce164413acb17ea1e0de1f1e5082efada1d2ff99280bd924a66daf4d5974.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/b95dce164413acb17ea1e0de1f1e5082efada1d2ff99280bd924a66daf4d5974.jpg)

![bb917b4ee2f74e9ec3864890f1c02d30bb18ae95c4f2cda50b33a4d271411417.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/bb917b4ee2f74e9ec3864890f1c02d30bb18ae95c4f2cda50b33a4d271411417.jpg)

![bd526397654a9f0fbaa205a1c929069b6707b12c7ba759d0adb0c1897980c230.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/bd526397654a9f0fbaa205a1c929069b6707b12c7ba759d0adb0c1897980c230.jpg)

![c4157c37aaa8b62e2b400836abd20a5567c59fe622603dff3766615f3fc459bc.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/c4157c37aaa8b62e2b400836abd20a5567c59fe622603dff3766615f3fc459bc.jpg)

![c446555475ae2ad1f5c26ce5c5d117d06cd39ad40a246831943860ed32edae62.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/c446555475ae2ad1f5c26ce5c5d117d06cd39ad40a246831943860ed32edae62.jpg)

![d814cbacc0c61472efe1f95acaf529d899362b5166df69c76dc89cd80db01f7d.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/d814cbacc0c61472efe1f95acaf529d899362b5166df69c76dc89cd80db01f7d.jpg)

![d8f5e41711e4be0171af50eaf8d03ac23cd5ecc61d7561ccf37212879ec0c0d6.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/d8f5e41711e4be0171af50eaf8d03ac23cd5ecc61d7561ccf37212879ec0c0d6.jpg)

![da9e0e79c6ed49ce9a3041358e83d182561e5e6405e82f596706dd02d0187a58.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/da9e0e79c6ed49ce9a3041358e83d182561e5e6405e82f596706dd02d0187a58.jpg)

![db7247abe708e6846c5a9a969b73b4af4f0a9f60f5556e079ea1ec22b49ebd82.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/db7247abe708e6846c5a9a969b73b4af4f0a9f60f5556e079ea1ec22b49ebd82.jpg)

![e8f1b0d5b47026bb74e13c654fdb3be097ca859fd37c105b140e7b7124074c8b.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/e8f1b0d5b47026bb74e13c654fdb3be097ca859fd37c105b140e7b7124074c8b.jpg)

![f805f35f1a92a22b9f34805fed6597837d5e8e65080e880163595d447df062f4.jpg](../icml_results/1929_From%20Rashomon%20Theory%20to%20PRAXIS_%20Efficient%20Decision%20Tree%20Rashomon%20Sets/tables/f805f35f1a92a22b9f34805fed6597837d5e8e65080e880163595d447df062f4.jpg)

<a id="paper-1905"></a>

## RADIO1D: Elastic Representations for Condensed Vision Modeling


### Images

![0689b9e780ab1d0748e441c278c226df333c12bbaaebc286a67b54f44f2981f7.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/0689b9e780ab1d0748e441c278c226df333c12bbaaebc286a67b54f44f2981f7.jpg)

![0f4a9133abebbd072e97a1fbef39609858dccb03d4589e78f5d863ab5ac1ec2b.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/0f4a9133abebbd072e97a1fbef39609858dccb03d4589e78f5d863ab5ac1ec2b.jpg)

![113ec5368f0bfdc9fa57a881ac10860aa497cf4f51b4ab695c14efeadb9c7016.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/113ec5368f0bfdc9fa57a881ac10860aa497cf4f51b4ab695c14efeadb9c7016.jpg)

![15375c1be1f70cf7f7594fb450f09cf420f94ace1978cb39ef3831da549b481d.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/15375c1be1f70cf7f7594fb450f09cf420f94ace1978cb39ef3831da549b481d.jpg)

![16486d4e50b8431f30b11de930aa7f26a434acc72a14b4cd1270021458f24603.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/16486d4e50b8431f30b11de930aa7f26a434acc72a14b4cd1270021458f24603.jpg)

![408f037d86048cf78a10741fc4f0df5acb560912e1e6260c200f739185e5d197.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/408f037d86048cf78a10741fc4f0df5acb560912e1e6260c200f739185e5d197.jpg)

![4ef0a4f600b5c0a33fc049d2aee0ac26a9f0ccbe906a193c8dc4f564dd659491.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/4ef0a4f600b5c0a33fc049d2aee0ac26a9f0ccbe906a193c8dc4f564dd659491.jpg)

![6684173a78e0845c201ef31c4d8dd48caab7f8b75ecd4f5e1426b9517df9905d.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/6684173a78e0845c201ef31c4d8dd48caab7f8b75ecd4f5e1426b9517df9905d.jpg)

![6d5ef5a2436942e9e819a6e5eea11266b235adf3195749adfe169d316ba1116d.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/6d5ef5a2436942e9e819a6e5eea11266b235adf3195749adfe169d316ba1116d.jpg)

![7596d39a4f9a22ca669930ca53e0f8f17b21d810bb2bd9f09d81fea34a9d68b3.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/7596d39a4f9a22ca669930ca53e0f8f17b21d810bb2bd9f09d81fea34a9d68b3.jpg)

![831922c0e0a769380ca0d1a0b993f56617e90a4e56db6fea558ed4a9754c4258.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/831922c0e0a769380ca0d1a0b993f56617e90a4e56db6fea558ed4a9754c4258.jpg)

![904992023a5cafc1932f37abc8b7ded89bcf63db00fd32e531e9e05bf2f42363.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/904992023a5cafc1932f37abc8b7ded89bcf63db00fd32e531e9e05bf2f42363.jpg)

![a9badd7d8343d09f430d98e4019e1a5f52c2266021db47e46332ed5bc7ccb305.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/a9badd7d8343d09f430d98e4019e1a5f52c2266021db47e46332ed5bc7ccb305.jpg)

![add8094c810062c161a8445f023d4a8761d0f51a2ebb9f78d8fdea68e5357ffe.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/add8094c810062c161a8445f023d4a8761d0f51a2ebb9f78d8fdea68e5357ffe.jpg)

![ae9f7f5b334cc56b89c1ccba8757d21eb959d7d68803a53e8886afad72a7f3fe.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/ae9f7f5b334cc56b89c1ccba8757d21eb959d7d68803a53e8886afad72a7f3fe.jpg)

![c29bfd134e6015827bf04a246e69eadbec672024afc96851eade3a536ec136d4.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/c29bfd134e6015827bf04a246e69eadbec672024afc96851eade3a536ec136d4.jpg)

![c4b4142a09634e6668505776692f879c7d8ba4e0258a96bd9693116b25b9a3d5.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/c4b4142a09634e6668505776692f879c7d8ba4e0258a96bd9693116b25b9a3d5.jpg)

![edb60785ed096c3c878d1c5644b12cf6bca205763b9db8631abe2dedfda040c2.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/edb60785ed096c3c878d1c5644b12cf6bca205763b9db8631abe2dedfda040c2.jpg)

![f787dbc390f2a5f84484200fc626a3a7781d8c982114abbbccae6bc94a737f23.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/images/f787dbc390f2a5f84484200fc626a3a7781d8c982114abbbccae6bc94a737f23.jpg)

### Tables

![326b2e83cab9f57650d02aeab2ed267d38df9d4571f66db65fcb9a1b01021b56.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/tables/326b2e83cab9f57650d02aeab2ed267d38df9d4571f66db65fcb9a1b01021b56.jpg)

![35583c826e0fa72da3b4ee0769d57ed23763bdb49371788de737efbdd7273d65.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/tables/35583c826e0fa72da3b4ee0769d57ed23763bdb49371788de737efbdd7273d65.jpg)

![37c7dfc701ab77a2735a1d579fbd4b6b4c5af7d9850c358891171adb369ee610.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/tables/37c7dfc701ab77a2735a1d579fbd4b6b4c5af7d9850c358891171adb369ee610.jpg)

![39b96e64d792fae26a8665ff61e6b9f8a6f342f5b125cfd41a1c6d07ce319b8b.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/tables/39b96e64d792fae26a8665ff61e6b9f8a6f342f5b125cfd41a1c6d07ce319b8b.jpg)

![63443a456410aea6e9359bb18f68daed1c981efd91dd9dc28dcf82a2f2e73adb.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/tables/63443a456410aea6e9359bb18f68daed1c981efd91dd9dc28dcf82a2f2e73adb.jpg)

![a09357a8698aaa1ecd901c6106bfb8b9579e79da508dda650515d61631d7804c.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/tables/a09357a8698aaa1ecd901c6106bfb8b9579e79da508dda650515d61631d7804c.jpg)

![b9c134ee3598bfcd119eb5adea8690d979d8f84e223fa74be340170e74a28875.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/tables/b9c134ee3598bfcd119eb5adea8690d979d8f84e223fa74be340170e74a28875.jpg)

![d75c50ccbbfe52056a7987c6939a19e60f03e8a0cc9dae9a0ca4d2a0f645d710.jpg](../icml_results/1930_RADIO1D_%20Elastic%20Representations%20for%20Condensed%20Vision%20Modeling/tables/d75c50ccbbfe52056a7987c6939a19e60f03e8a0cc9dae9a0ca4d2a0f645d710.jpg)

<a id="paper-1906"></a>

## Query-efficient model evaluation using cached responses


### Images

![0d21084de50e1d1873b093bf6fca5f2efca4ccbc954292065230a2384198299f.jpg](../icml_results/1931_Query-efficient%20model%20evaluation%20using%20cached%20responses/images/0d21084de50e1d1873b093bf6fca5f2efca4ccbc954292065230a2384198299f.jpg)

![3f92306633f0cb8a4f82c62b9d2574ae467243ec77c129ff4298c5b4d98107ff.jpg](../icml_results/1931_Query-efficient%20model%20evaluation%20using%20cached%20responses/images/3f92306633f0cb8a4f82c62b9d2574ae467243ec77c129ff4298c5b4d98107ff.jpg)

![41de959e1eb56bfcbe578f6d292f244964306097e26c3a0ae3e79a716ea643e5.jpg](../icml_results/1931_Query-efficient%20model%20evaluation%20using%20cached%20responses/images/41de959e1eb56bfcbe578f6d292f244964306097e26c3a0ae3e79a716ea643e5.jpg)

![76b4f1515237112fc862b249416cebabdc2c5ad43d613dfe5bf5db45b27e356b.jpg](../icml_results/1931_Query-efficient%20model%20evaluation%20using%20cached%20responses/images/76b4f1515237112fc862b249416cebabdc2c5ad43d613dfe5bf5db45b27e356b.jpg)

![bcdd7e9828b300b945781af6505109c54c040cc41852b07a1bdaf5fe11d23271.jpg](../icml_results/1931_Query-efficient%20model%20evaluation%20using%20cached%20responses/images/bcdd7e9828b300b945781af6505109c54c040cc41852b07a1bdaf5fe11d23271.jpg)

### Tables

![1c3300d797e0e4ecd296cd8fbda529c821bca779d9732864b6596eceb46abbda.jpg](../icml_results/1931_Query-efficient%20model%20evaluation%20using%20cached%20responses/tables/1c3300d797e0e4ecd296cd8fbda529c821bca779d9732864b6596eceb46abbda.jpg)

![3327e851c7903750ff9eb2fae611ef45c7a2d4bdc2cc856448ecaa1bdbcaa78a.jpg](../icml_results/1931_Query-efficient%20model%20evaluation%20using%20cached%20responses/tables/3327e851c7903750ff9eb2fae611ef45c7a2d4bdc2cc856448ecaa1bdbcaa78a.jpg)

![a263e544cf171db5db7f78671d2a433e7bea6d2618229c9ee2d5cb8dc25fa33a.jpg](../icml_results/1931_Query-efficient%20model%20evaluation%20using%20cached%20responses/tables/a263e544cf171db5db7f78671d2a433e7bea6d2618229c9ee2d5cb8dc25fa33a.jpg)

![a3a11c024d731cfb38b297a5ab70547a72bd97bf0f790d87569583047801e8c0.jpg](../icml_results/1931_Query-efficient%20model%20evaluation%20using%20cached%20responses/tables/a3a11c024d731cfb38b297a5ab70547a72bd97bf0f790d87569583047801e8c0.jpg)

![b604018e7889a4bc3d9c7c7cfb98f837e188ccbab7e43da7cf3016d51cf300f8.jpg](../icml_results/1931_Query-efficient%20model%20evaluation%20using%20cached%20responses/tables/b604018e7889a4bc3d9c7c7cfb98f837e188ccbab7e43da7cf3016d51cf300f8.jpg)

![c435712d6b5c3973d2b3f029c22f040df65ff471653f1e3975cf8a805c570720.jpg](../icml_results/1931_Query-efficient%20model%20evaluation%20using%20cached%20responses/tables/c435712d6b5c3973d2b3f029c22f040df65ff471653f1e3975cf8a805c570720.jpg)

![c7837209bcb8913adf945170c2cd198b36c7fa1644ef3af523554cadf9f99a1c.jpg](../icml_results/1931_Query-efficient%20model%20evaluation%20using%20cached%20responses/tables/c7837209bcb8913adf945170c2cd198b36c7fa1644ef3af523554cadf9f99a1c.jpg)

![e9df44271ebddd069874e8cfeccc8f37251a0560556935d491771050c3530e52.jpg](../icml_results/1931_Query-efficient%20model%20evaluation%20using%20cached%20responses/tables/e9df44271ebddd069874e8cfeccc8f37251a0560556935d491771050c3530e52.jpg)

<a id="paper-1907"></a>

## Score Based Error Correcting Code Decoder


### Images

![02b50d9ffbb8ad400551d107f9a185f17c7d384b9d0d7f10137abf675ba09e55.jpg](../icml_results/1932_Score%20Based%20Error%20Correcting%20Code%20Decoder/images/02b50d9ffbb8ad400551d107f9a185f17c7d384b9d0d7f10137abf675ba09e55.jpg)

![2da7c33588c38a929cad5a9db99cf9269b20ee8a4365b9bd1cc53d9f187d6f03.jpg](../icml_results/1932_Score%20Based%20Error%20Correcting%20Code%20Decoder/images/2da7c33588c38a929cad5a9db99cf9269b20ee8a4365b9bd1cc53d9f187d6f03.jpg)

![61d223b1f6ac1844dabdd98e2f03fa7b0b5e2dd6f95d65333e9e65ca8e8b0a9b.jpg](../icml_results/1932_Score%20Based%20Error%20Correcting%20Code%20Decoder/images/61d223b1f6ac1844dabdd98e2f03fa7b0b5e2dd6f95d65333e9e65ca8e8b0a9b.jpg)

![be0aa3bd81e9df02245e843fd1bbe2d4448a162675989db479fecd0e1af34a21.jpg](../icml_results/1932_Score%20Based%20Error%20Correcting%20Code%20Decoder/images/be0aa3bd81e9df02245e843fd1bbe2d4448a162675989db479fecd0e1af34a21.jpg)

### Tables

![02a917b29d3cbffa982fd18a21b69fa9e9865885fbf098c2a6bf034dc6731fab.jpg](../icml_results/1932_Score%20Based%20Error%20Correcting%20Code%20Decoder/tables/02a917b29d3cbffa982fd18a21b69fa9e9865885fbf098c2a6bf034dc6731fab.jpg)

![24975aa08932c4ebb52306090ff0947f0e1a08c63c75ea74b6d33a988469cfc2.jpg](../icml_results/1932_Score%20Based%20Error%20Correcting%20Code%20Decoder/tables/24975aa08932c4ebb52306090ff0947f0e1a08c63c75ea74b6d33a988469cfc2.jpg)

![41dd5b1ce5f0e3b659a8d361a8fef4665cd0f235852f950b92e0df94b3582094.jpg](../icml_results/1932_Score%20Based%20Error%20Correcting%20Code%20Decoder/tables/41dd5b1ce5f0e3b659a8d361a8fef4665cd0f235852f950b92e0df94b3582094.jpg)

![62535c4560532b58e7bfdf8f913271041bbef0dc68e4417515954ac6a89f2309.jpg](../icml_results/1932_Score%20Based%20Error%20Correcting%20Code%20Decoder/tables/62535c4560532b58e7bfdf8f913271041bbef0dc68e4417515954ac6a89f2309.jpg)

![67f1ecdabebbb04acc34b64b294898840bb043a3ddd5f6fc5ad3e1ff84b732c3.jpg](../icml_results/1932_Score%20Based%20Error%20Correcting%20Code%20Decoder/tables/67f1ecdabebbb04acc34b64b294898840bb043a3ddd5f6fc5ad3e1ff84b732c3.jpg)

![8d8646ea946f6a280c42f577e2823b102fcd61108f39ae27c98ebf5c2de12d3e.jpg](../icml_results/1932_Score%20Based%20Error%20Correcting%20Code%20Decoder/tables/8d8646ea946f6a280c42f577e2823b102fcd61108f39ae27c98ebf5c2de12d3e.jpg)

![8fd0bbca6fb4474e5144008c31a61872166e3dcf9cde233b9c36edb8503313eb.jpg](../icml_results/1932_Score%20Based%20Error%20Correcting%20Code%20Decoder/tables/8fd0bbca6fb4474e5144008c31a61872166e3dcf9cde233b9c36edb8503313eb.jpg)

![c5d2a35e80ce13bdc3b1cfef817fcd8c95dd50c6306a5eb5dfc0f939db5fbc81.jpg](../icml_results/1932_Score%20Based%20Error%20Correcting%20Code%20Decoder/tables/c5d2a35e80ce13bdc3b1cfef817fcd8c95dd50c6306a5eb5dfc0f939db5fbc81.jpg)

![d3ffd6afb2790621fa411721b263d75b5641ba2e1f4747c6a416be48f546033d.jpg](../icml_results/1932_Score%20Based%20Error%20Correcting%20Code%20Decoder/tables/d3ffd6afb2790621fa411721b263d75b5641ba2e1f4747c6a416be48f546033d.jpg)

<a id="paper-1908"></a>

## What Linear Probes Miss: Multi-View Probing for Weight-Space Learning


### Images

![2bc54c74b43080c55e3dc91157f88da88d8f7fa31cc8801f58da40c62c2d7f99.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/images/2bc54c74b43080c55e3dc91157f88da88d8f7fa31cc8801f58da40c62c2d7f99.jpg)

![304e839bff19268f82aabb91cc13ab7e2266c2118b82b788cf93abc2ac7f6560.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/images/304e839bff19268f82aabb91cc13ab7e2266c2118b82b788cf93abc2ac7f6560.jpg)

![37b1f8273d884dddd5f97bf03c211c304bccd38db729ba5918d4bf8b26740ffd.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/images/37b1f8273d884dddd5f97bf03c211c304bccd38db729ba5918d4bf8b26740ffd.jpg)

![9c6fa775f24400cb75774976bd1a3e992d0fa0537474859a13dc94a2383ea7be.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/images/9c6fa775f24400cb75774976bd1a3e992d0fa0537474859a13dc94a2383ea7be.jpg)

![cbe01812655639b87673b7b856562b2fbbe0948401e5cf2617c0e135420054ec.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/images/cbe01812655639b87673b7b856562b2fbbe0948401e5cf2617c0e135420054ec.jpg)

### Tables

![09391c34a113b37b02d48308dd754b087dabe1de3e4aec1842bdc4ba3776dcd2.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/09391c34a113b37b02d48308dd754b087dabe1de3e4aec1842bdc4ba3776dcd2.jpg)

![0c644357c66f13ac79321969a4bbf494d19fbe6b7ca46229f43694fd9b6306ef.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/0c644357c66f13ac79321969a4bbf494d19fbe6b7ca46229f43694fd9b6306ef.jpg)

![1bd59bb8b20b606ac1a76901cd8a187b3d53ab2d1fe7a3ec83547f72b5daa1cd.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/1bd59bb8b20b606ac1a76901cd8a187b3d53ab2d1fe7a3ec83547f72b5daa1cd.jpg)

![1f6c37356c5b8cd0ffd3722316cc926c9264f9a6618515ef51055dbebfa51e39.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/1f6c37356c5b8cd0ffd3722316cc926c9264f9a6618515ef51055dbebfa51e39.jpg)

![31c7c26ff0494f6cf2444f1c0d0f3754c5cd3e5127e2d257cea8eedb115a311e.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/31c7c26ff0494f6cf2444f1c0d0f3754c5cd3e5127e2d257cea8eedb115a311e.jpg)

![45d8b50ecdb014ee2a92a16080314306757c86734acb7eff3bb014dc6e39a080.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/45d8b50ecdb014ee2a92a16080314306757c86734acb7eff3bb014dc6e39a080.jpg)

![4c5c1ffe07a1d4ac85d36924c9f117d7169655ed6d55b8e31afecd41b7280d4a.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/4c5c1ffe07a1d4ac85d36924c9f117d7169655ed6d55b8e31afecd41b7280d4a.jpg)

![5e624428c068e4aa48064407c123c48b074ca838cc247f84ebc3b81307972642.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/5e624428c068e4aa48064407c123c48b074ca838cc247f84ebc3b81307972642.jpg)

![6c1850deed62968057c3d2b03778a8c6b9ebf79b57b19dfef0d79964d2db131e.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/6c1850deed62968057c3d2b03778a8c6b9ebf79b57b19dfef0d79964d2db131e.jpg)

![7c1c46a0c46a46fd3848ad8046c0aba9f78a3fa8bf26b75b08c2eecf307178bb.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/7c1c46a0c46a46fd3848ad8046c0aba9f78a3fa8bf26b75b08c2eecf307178bb.jpg)

![80686f01e71972e137faa2adabba8763d5055c35069e26296e30f84bd1a25506.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/80686f01e71972e137faa2adabba8763d5055c35069e26296e30f84bd1a25506.jpg)

![82da804efb7582d6af078746aa1bd19ce6c8bb86ee99f70ebafff46f9f0855e3.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/82da804efb7582d6af078746aa1bd19ce6c8bb86ee99f70ebafff46f9f0855e3.jpg)

![9bb5f60610ee703f0ef95690682d84303654cf3a8519b38af4d83b323bc21b91.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/9bb5f60610ee703f0ef95690682d84303654cf3a8519b38af4d83b323bc21b91.jpg)

![aec7a3ba6b7c7cf5218d052935240bef25383e393a35341ecb5da3b4e2d7ecbd.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/aec7a3ba6b7c7cf5218d052935240bef25383e393a35341ecb5da3b4e2d7ecbd.jpg)

![b07e274d1613251bdfa12875a14cb9cc6b0dd0eabdd5a7a9645c480269468788.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/b07e274d1613251bdfa12875a14cb9cc6b0dd0eabdd5a7a9645c480269468788.jpg)

![bb2333ce626cc306724cb18b88fa085d89698e22a8038b125e7c3186eed7a73a.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/bb2333ce626cc306724cb18b88fa085d89698e22a8038b125e7c3186eed7a73a.jpg)

![c1b1b247215d9455d7ea379dacc7f9c5fe1ddbc82d53c7a20a9e9434deff1b8a.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/c1b1b247215d9455d7ea379dacc7f9c5fe1ddbc82d53c7a20a9e9434deff1b8a.jpg)

![c99f2527b9f08f40ab5126602b9eedc34faaa80597989972d9c9c13497c6a69c.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/c99f2527b9f08f40ab5126602b9eedc34faaa80597989972d9c9c13497c6a69c.jpg)

![d120983b8e7a37cae62a9975e46fe6e79406d13713f9df5227543f53005c84a7.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/d120983b8e7a37cae62a9975e46fe6e79406d13713f9df5227543f53005c84a7.jpg)

![dbef6db07581cd697605e8dc50d2cb92f1dc22d0567e60a57450cc2f203bdaf4.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/dbef6db07581cd697605e8dc50d2cb92f1dc22d0567e60a57450cc2f203bdaf4.jpg)

![e46cbcdd506c6dcf15f905a36da5c3fdcaaf630ebdb687de00e4c7888f9ab764.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/e46cbcdd506c6dcf15f905a36da5c3fdcaaf630ebdb687de00e4c7888f9ab764.jpg)

![e5321b2a8813459612cc0fae5aa213f4abd80765fced6f0e5625838969fa61da.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/e5321b2a8813459612cc0fae5aa213f4abd80765fced6f0e5625838969fa61da.jpg)

![ebff3642764e01261a2d8a016f807e03f721b7c7dda3500a8c9a4ed83a070696.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/ebff3642764e01261a2d8a016f807e03f721b7c7dda3500a8c9a4ed83a070696.jpg)

![f2277da32d7b9eeb18487a127c028d18466fe14361385594d6c071a28f368a39.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/f2277da32d7b9eeb18487a127c028d18466fe14361385594d6c071a28f368a39.jpg)

![f32522e258bbc788426c2dd88ff6a1bc9b953cee46f0d06b48c655054bac1348.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/f32522e258bbc788426c2dd88ff6a1bc9b953cee46f0d06b48c655054bac1348.jpg)

![fd5ba4b6bef06e00644e027a12e94b50948599204269cab0aec7af685276e365.jpg](../icml_results/1933_What%20Linear%20Probes%20Miss_%20Multi-View%20Probing%20for%20Weight-Space%20Learning/tables/fd5ba4b6bef06e00644e027a12e94b50948599204269cab0aec7af685276e365.jpg)

<a id="paper-1909"></a>

## The Expert Strikes Back: Interpreting Mixture-of-Experts Language Models at Expert Level


### Images

![1685a4c39515668d876145aceff2c2aaf3a376bd084cf22b5498bc8474592c03.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/images/1685a4c39515668d876145aceff2c2aaf3a376bd084cf22b5498bc8474592c03.jpg)

![1d7e1c1a87045b7cabf71e114977e04d171556b1d269f6c0db5103f1ce2cfa87.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/images/1d7e1c1a87045b7cabf71e114977e04d171556b1d269f6c0db5103f1ce2cfa87.jpg)

![261eed2d69dd12bbba4e497b9f11fe1fe5c957c5bf4a24aeb77b62662ce5843b.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/images/261eed2d69dd12bbba4e497b9f11fe1fe5c957c5bf4a24aeb77b62662ce5843b.jpg)

![4f05e96402fd25ac004a5eee56012f49a5827dd5d9211fe82a864617fbca4f1d.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/images/4f05e96402fd25ac004a5eee56012f49a5827dd5d9211fe82a864617fbca4f1d.jpg)

![ac91239eea75f1092ff3ec6f7f0cddb46c8d402a2f05490f666a4a3475d22dcb.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/images/ac91239eea75f1092ff3ec6f7f0cddb46c8d402a2f05490f666a4a3475d22dcb.jpg)

![b518c212732ee8a1842880db3c2bbf00ac92679bf0c947d2e7697399db00c0aa.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/images/b518c212732ee8a1842880db3c2bbf00ac92679bf0c947d2e7697399db00c0aa.jpg)

![cc7ebe22da20cdf19b483fa75f006598a3cfd29ff8eebe35cbd81b7a87dcc17d.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/images/cc7ebe22da20cdf19b483fa75f006598a3cfd29ff8eebe35cbd81b7a87dcc17d.jpg)

![e3dc0984de2c7bff2c1cb6a5356c86f50d83820f2b4de227c6aa8f383879db53.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/images/e3dc0984de2c7bff2c1cb6a5356c86f50d83820f2b4de227c6aa8f383879db53.jpg)

![f258af01fdd7e1c213afb780e089730126ff15de5b50cecfb7cdbe5b47872ebf.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/images/f258af01fdd7e1c213afb780e089730126ff15de5b50cecfb7cdbe5b47872ebf.jpg)

![fd0b6f2e48218f3e629c893ee5003ec77b7df429779c3eadc15921082a3be58b.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/images/fd0b6f2e48218f3e629c893ee5003ec77b7df429779c3eadc15921082a3be58b.jpg)

### Tables

![11e3803a83d6058a14ebc6cc888b0b3c6e8391fd611e67d14bfcae9576f14963.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/11e3803a83d6058a14ebc6cc888b0b3c6e8391fd611e67d14bfcae9576f14963.jpg)

![164897a3ea4fc22e6353c702f2327b61c085adca9ae40774067ac6d032af39f4.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/164897a3ea4fc22e6353c702f2327b61c085adca9ae40774067ac6d032af39f4.jpg)

![1af5b8b7468fbc12f6a14119ed73eb5d158de46a48e7514c8e67a36f0161f66a.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/1af5b8b7468fbc12f6a14119ed73eb5d158de46a48e7514c8e67a36f0161f66a.jpg)

![1e6868cc6bc09c04af6c08a99af46a3dc423c8ba8cda0e862eb92957e50a5da7.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/1e6868cc6bc09c04af6c08a99af46a3dc423c8ba8cda0e862eb92957e50a5da7.jpg)

![256c5be136c847f5bce1c503637309b256a23d7c4efa2bed53a859145f432f4c.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/256c5be136c847f5bce1c503637309b256a23d7c4efa2bed53a859145f432f4c.jpg)

![316bd7622a8e89767b3e82132a5f5738c8df5664749cf1cb188ac5207872f73b.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/316bd7622a8e89767b3e82132a5f5738c8df5664749cf1cb188ac5207872f73b.jpg)

![4418dcecf1e6f2ddd14d0de7c6fb65324f803fd15fd3f55b0cc92513bbc9e98f.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/4418dcecf1e6f2ddd14d0de7c6fb65324f803fd15fd3f55b0cc92513bbc9e98f.jpg)

![5db771dc0f2893c2e6acce33d32717a5cce45e75cb6255b3a39d233ac56b2764.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/5db771dc0f2893c2e6acce33d32717a5cce45e75cb6255b3a39d233ac56b2764.jpg)

![6fe9e3cd11630034283bb7408fa3d586b4d824970e053da0e7073c21ab91a1f2.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/6fe9e3cd11630034283bb7408fa3d586b4d824970e053da0e7073c21ab91a1f2.jpg)

![78ce1035f58fc0a3257d3c7d59c5715f2439822c1cbf4db04f563c28ed13f5ce.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/78ce1035f58fc0a3257d3c7d59c5715f2439822c1cbf4db04f563c28ed13f5ce.jpg)

![79f0ed52b2813a0ea41797983f925167cf5d57504631d0b0d16d8fda2d699b00.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/79f0ed52b2813a0ea41797983f925167cf5d57504631d0b0d16d8fda2d699b00.jpg)

![7ea21c4a01c286070d8914bda929158e0a9c5912236bcd6a053504ccead9034c.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/7ea21c4a01c286070d8914bda929158e0a9c5912236bcd6a053504ccead9034c.jpg)

![915bee80248d7fb15a5afd528b9be936232b7697dacefcd4559e8e2354108bba.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/915bee80248d7fb15a5afd528b9be936232b7697dacefcd4559e8e2354108bba.jpg)

![9673fdb98e0f634b38febedf812afa964304f4c462a81a72fa1d7963cb4afa62.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/9673fdb98e0f634b38febedf812afa964304f4c462a81a72fa1d7963cb4afa62.jpg)

![b88317e5a8f29fa95b11fea27abe4accfb25196b8fc4b4332e13388f26c88524.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/b88317e5a8f29fa95b11fea27abe4accfb25196b8fc4b4332e13388f26c88524.jpg)

![bc110296d780098a1c39cf0eb33ab5f7d218bde152fb9a5fba63d69301727bdc.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/bc110296d780098a1c39cf0eb33ab5f7d218bde152fb9a5fba63d69301727bdc.jpg)

![c4ee5142419858cc4365018c26af826e34cda003022e0fba83c99faf6fa5a274.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/c4ee5142419858cc4365018c26af826e34cda003022e0fba83c99faf6fa5a274.jpg)

![d25966b74e05a2ef24387af6b65521e446f767f1135b8775039fcf674c33d9d2.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/d25966b74e05a2ef24387af6b65521e446f767f1135b8775039fcf674c33d9d2.jpg)

![d5662a230916d880e224d0b8985e13330e9454508b1773509d978796a9d4dc7a.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/d5662a230916d880e224d0b8985e13330e9454508b1773509d978796a9d4dc7a.jpg)

![e735d2409e630bfbd35ef786430a2456de158119f40aed76e17def01ecc66735.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/e735d2409e630bfbd35ef786430a2456de158119f40aed76e17def01ecc66735.jpg)

![f9a97aa152cf7dcb0698604eabb74fc9f70fafb8995a49271b3b239fff10d426.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/f9a97aa152cf7dcb0698604eabb74fc9f70fafb8995a49271b3b239fff10d426.jpg)

![fcff3ea4b954a8805c2f1719600410301073bc4d3074ab44cc33ceefb16bf302.jpg](../icml_results/1934_The%20Expert%20Strikes%20Back_%20Interpreting%20Mixture-of-Experts%20Language%20Models%20at%20Expert%20Level/tables/fcff3ea4b954a8805c2f1719600410301073bc4d3074ab44cc33ceefb16bf302.jpg)

<a id="paper-1910"></a>

## Locally Coherent Parallel Decoding in Diffusion Language Models


### Images

![8328fd0bf5f8a1289ec69348c963d0049a6582a9bb58059c4013592092859547.jpg](../icml_results/1935_Locally%20Coherent%20Parallel%20Decoding%20in%20Diffusion%20Language%20Models/images/8328fd0bf5f8a1289ec69348c963d0049a6582a9bb58059c4013592092859547.jpg)

![84383cd3dcd076380bd1c5f12bc4a0e779b2829d94a7ff43ec8063e11e0f3c48.jpg](../icml_results/1935_Locally%20Coherent%20Parallel%20Decoding%20in%20Diffusion%20Language%20Models/images/84383cd3dcd076380bd1c5f12bc4a0e779b2829d94a7ff43ec8063e11e0f3c48.jpg)

![89dd7869903d933fb8101fa382e9af725dda45ec71e3bbd4c8239bf6dfb497a7.jpg](../icml_results/1935_Locally%20Coherent%20Parallel%20Decoding%20in%20Diffusion%20Language%20Models/images/89dd7869903d933fb8101fa382e9af725dda45ec71e3bbd4c8239bf6dfb497a7.jpg)

![8cb549f34c400c4e425bd48154704ead002c6d607ace8924634fac6b89066b05.jpg](../icml_results/1935_Locally%20Coherent%20Parallel%20Decoding%20in%20Diffusion%20Language%20Models/images/8cb549f34c400c4e425bd48154704ead002c6d607ace8924634fac6b89066b05.jpg)

![b5cc43c67a08781685f8b42980365d1c35aa714f903d4e5cb5296825e146f4f4.jpg](../icml_results/1935_Locally%20Coherent%20Parallel%20Decoding%20in%20Diffusion%20Language%20Models/images/b5cc43c67a08781685f8b42980365d1c35aa714f903d4e5cb5296825e146f4f4.jpg)

![d1f2f11a6eadf528b439da55ccb67deb43796147b365c34cd97c50f5f4d23ea6.jpg](../icml_results/1935_Locally%20Coherent%20Parallel%20Decoding%20in%20Diffusion%20Language%20Models/images/d1f2f11a6eadf528b439da55ccb67deb43796147b365c34cd97c50f5f4d23ea6.jpg)

![ee1d85be341d7edf30d173d6c56a5b09a9f6e94cfe7482d0979c67d341baca87.jpg](../icml_results/1935_Locally%20Coherent%20Parallel%20Decoding%20in%20Diffusion%20Language%20Models/images/ee1d85be341d7edf30d173d6c56a5b09a9f6e94cfe7482d0979c67d341baca87.jpg)

### Tables

![1c18168f6b88d8414831c2768c6c3b1925d44cdae4ad991be21f15252fa4627c.jpg](../icml_results/1935_Locally%20Coherent%20Parallel%20Decoding%20in%20Diffusion%20Language%20Models/tables/1c18168f6b88d8414831c2768c6c3b1925d44cdae4ad991be21f15252fa4627c.jpg)

![50158fc53aca0cb47434eb11d6039154d6a2896e1844e584e48698aa00af7e21.jpg](../icml_results/1935_Locally%20Coherent%20Parallel%20Decoding%20in%20Diffusion%20Language%20Models/tables/50158fc53aca0cb47434eb11d6039154d6a2896e1844e584e48698aa00af7e21.jpg)

![61f3a8e9075fe6ffa608791172a7e2bfa0c7eadff8bf81b1e8ba5793f8cc96bd.jpg](../icml_results/1935_Locally%20Coherent%20Parallel%20Decoding%20in%20Diffusion%20Language%20Models/tables/61f3a8e9075fe6ffa608791172a7e2bfa0c7eadff8bf81b1e8ba5793f8cc96bd.jpg)

![726efb7d57b95af9b207de0c2536d55ce986a9a6b23ef0dae5910a6fe6af4b3e.jpg](../icml_results/1935_Locally%20Coherent%20Parallel%20Decoding%20in%20Diffusion%20Language%20Models/tables/726efb7d57b95af9b207de0c2536d55ce986a9a6b23ef0dae5910a6fe6af4b3e.jpg)

![87fac0c46289272b14386fa414696aeaff90f9174c39e29af842b82040fbb6a1.jpg](../icml_results/1935_Locally%20Coherent%20Parallel%20Decoding%20in%20Diffusion%20Language%20Models/tables/87fac0c46289272b14386fa414696aeaff90f9174c39e29af842b82040fbb6a1.jpg)

![9986c0a07e28da8da258385b9e4b1adb5a8fb692495614b3d40dc92082c43f01.jpg](../icml_results/1935_Locally%20Coherent%20Parallel%20Decoding%20in%20Diffusion%20Language%20Models/tables/9986c0a07e28da8da258385b9e4b1adb5a8fb692495614b3d40dc92082c43f01.jpg)

![a899deed8553333c07af0d118a106493a5e1b0d0b47bba7e62192feb0826ac9a.jpg](../icml_results/1935_Locally%20Coherent%20Parallel%20Decoding%20in%20Diffusion%20Language%20Models/tables/a899deed8553333c07af0d118a106493a5e1b0d0b47bba7e62192feb0826ac9a.jpg)

![c4f4aa6bad2b2ae673b86ee8d92efa29b0c9fdea6da1a1fd9a86143f34d90618.jpg](../icml_results/1935_Locally%20Coherent%20Parallel%20Decoding%20in%20Diffusion%20Language%20Models/tables/c4f4aa6bad2b2ae673b86ee8d92efa29b0c9fdea6da1a1fd9a86143f34d90618.jpg)

![cbca30017c6c3fddc2e3bc7ab7a2f37a08324a30ca376d53528f3eb422d87ae0.jpg](../icml_results/1935_Locally%20Coherent%20Parallel%20Decoding%20in%20Diffusion%20Language%20Models/tables/cbca30017c6c3fddc2e3bc7ab7a2f37a08324a30ca376d53528f3eb422d87ae0.jpg)

![eb7b70c0e5da61aa9dffa195664d63a64bd70afc1392246fd16ad167844c5e62.jpg](../icml_results/1935_Locally%20Coherent%20Parallel%20Decoding%20in%20Diffusion%20Language%20Models/tables/eb7b70c0e5da61aa9dffa195664d63a64bd70afc1392246fd16ad167844c5e62.jpg)

<a id="paper-1911"></a>

## What is Missing? Explaining Neurons Activated by Absent Concepts


### Images

![05da1ba40c0834831de25ed21e674ba2a07510b091a6f21946b30cf139f47693.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/images/05da1ba40c0834831de25ed21e674ba2a07510b091a6f21946b30cf139f47693.jpg)

![3cdcbb57f22b91791016747b5218a456df92d4e6ebc2944f290e0dcb390bc162.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/images/3cdcbb57f22b91791016747b5218a456df92d4e6ebc2944f290e0dcb390bc162.jpg)

![3eb5b28dc8580ca6abf1cb585ba0ad75409d1fd36e50749843be2d88c4ff8618.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/images/3eb5b28dc8580ca6abf1cb585ba0ad75409d1fd36e50749843be2d88c4ff8618.jpg)

![4c0f36e751b583a9ffebccaedc2a6a309530d7e9aa0faf83971d29e835fa0dd9.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/images/4c0f36e751b583a9ffebccaedc2a6a309530d7e9aa0faf83971d29e835fa0dd9.jpg)

![56693a9f6ff92f610281e634ca7b2762b97b0c756404d2e235b60f745954c8ae.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/images/56693a9f6ff92f610281e634ca7b2762b97b0c756404d2e235b60f745954c8ae.jpg)

![64e169acb92e70bdc37f732e7fec3cf8ba5d8a088543f2c15c87caa70f03c213.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/images/64e169acb92e70bdc37f732e7fec3cf8ba5d8a088543f2c15c87caa70f03c213.jpg)

![6d8906f3eda6bacfa7c245add80ec18b6c202e47393bc0ddff099502512cd01c.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/images/6d8906f3eda6bacfa7c245add80ec18b6c202e47393bc0ddff099502512cd01c.jpg)

![7ef86761fd399bbdbf943e83b9d5030aa729b55ef495342c252ca311cea4769d.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/images/7ef86761fd399bbdbf943e83b9d5030aa729b55ef495342c252ca311cea4769d.jpg)

![816b999be6468efb9032359124c0f9845e629ad72cc6c5322c19e61d8d711a56.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/images/816b999be6468efb9032359124c0f9845e629ad72cc6c5322c19e61d8d711a56.jpg)

![a9382a80099253f1a94c91aa606ff6d01fa2dd92e18b1f253eec1daeb72eb9a1.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/images/a9382a80099253f1a94c91aa606ff6d01fa2dd92e18b1f253eec1daeb72eb9a1.jpg)

![ac86653661da2889bc75b85dc9ce64f19ab245c5125d8b2796a59ba2bddb84b6.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/images/ac86653661da2889bc75b85dc9ce64f19ab245c5125d8b2796a59ba2bddb84b6.jpg)

![b1a0b756403fe7c20f0668141a3ba787d2d7035dedd9cdefa141cfdb05fd9ccf.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/images/b1a0b756403fe7c20f0668141a3ba787d2d7035dedd9cdefa141cfdb05fd9ccf.jpg)

![cb0dcb0d5030681fda15a505da39f0c247b25ce4947ff72ff8fb4fd318b259db.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/images/cb0dcb0d5030681fda15a505da39f0c247b25ce4947ff72ff8fb4fd318b259db.jpg)

![d077decf10dea5157b5ceaa1cf5e3dcdcabbb5e8fd0ea47b7d9837c85c7eab78.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/images/d077decf10dea5157b5ceaa1cf5e3dcdcabbb5e8fd0ea47b7d9837c85c7eab78.jpg)

![eb045e33869b7bd9325ebd7bef7a5a7e7cf260fef03535d51d526a7e025b6eb1.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/images/eb045e33869b7bd9325ebd7bef7a5a7e7cf260fef03535d51d526a7e025b6eb1.jpg)

### Tables

![4680dfb99d8b58010fb903b24bfd16d0291fef83a2f71fe237d32ccab1061df8.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/tables/4680dfb99d8b58010fb903b24bfd16d0291fef83a2f71fe237d32ccab1061df8.jpg)

![4a0ad493eb836e62fd2dcc2e7e3525f61b173f06f9871198b4505eb9a1492d1a.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/tables/4a0ad493eb836e62fd2dcc2e7e3525f61b173f06f9871198b4505eb9a1492d1a.jpg)

![59ff5c24f3065f2c1473e4372e6948f315e93a5ed03c41e2b8aad5f6bb5cfcfc.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/tables/59ff5c24f3065f2c1473e4372e6948f315e93a5ed03c41e2b8aad5f6bb5cfcfc.jpg)

![a791631cec35686e5256ae51cd8aeb4a151c5977eb92a848b4aa481eb3512070.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/tables/a791631cec35686e5256ae51cd8aeb4a151c5977eb92a848b4aa481eb3512070.jpg)

![ad3a0ca303dfdfc1bb896e7c127034a4dfede229d20cd80489230251505d8757.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/tables/ad3a0ca303dfdfc1bb896e7c127034a4dfede229d20cd80489230251505d8757.jpg)

![c84dbaf74b0f5f686b30485800367ae2d95e441a42e3a7a15e871a0dd43bca59.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/tables/c84dbaf74b0f5f686b30485800367ae2d95e441a42e3a7a15e871a0dd43bca59.jpg)

![e2aa3e0e0962187e4daaf31fac7f5b7c547bd24cab1675653957bff17add8c5c.jpg](../icml_results/1936_What%20is%20Missing_%20Explaining%20Neurons%20Activated%20by%20Absent%20Concepts/tables/e2aa3e0e0962187e4daaf31fac7f5b7c547bd24cab1675653957bff17add8c5c.jpg)

<a id="paper-1912"></a>

## Multi-View Causal Discovery without Non-Gaussianity: Identifiability and Algorithms


### Images

![208ac1058dedcc7d445a9f492fd8e261db9d08bedae49703a6c67afd2175865b.jpg](../icml_results/1937_Multi-View%20Causal%20Discovery%20without%20Non-Gaussianity_%20Identifiability%20and%20Algorithms/images/208ac1058dedcc7d445a9f492fd8e261db9d08bedae49703a6c67afd2175865b.jpg)

![5067895977f24ebd482e3da51d81619d435326e376c328cd3efa499c81bb41a0.jpg](../icml_results/1937_Multi-View%20Causal%20Discovery%20without%20Non-Gaussianity_%20Identifiability%20and%20Algorithms/images/5067895977f24ebd482e3da51d81619d435326e376c328cd3efa499c81bb41a0.jpg)

![57e747bd9e9da050751da182f86ee338d1b79f16ab5a111e1d7f9adc9abebf01.jpg](../icml_results/1937_Multi-View%20Causal%20Discovery%20without%20Non-Gaussianity_%20Identifiability%20and%20Algorithms/images/57e747bd9e9da050751da182f86ee338d1b79f16ab5a111e1d7f9adc9abebf01.jpg)

![6db89d91df75238374cc9a7b4ab669594e7f2adfe97f8122f25ddecf03793c9f.jpg](../icml_results/1937_Multi-View%20Causal%20Discovery%20without%20Non-Gaussianity_%20Identifiability%20and%20Algorithms/images/6db89d91df75238374cc9a7b4ab669594e7f2adfe97f8122f25ddecf03793c9f.jpg)

![9fe1b06bb07423cc428ff753de92de2919ec4b7e0cc221a6440b6cc671377610.jpg](../icml_results/1937_Multi-View%20Causal%20Discovery%20without%20Non-Gaussianity_%20Identifiability%20and%20Algorithms/images/9fe1b06bb07423cc428ff753de92de2919ec4b7e0cc221a6440b6cc671377610.jpg)

![b1bf99cda5c67cd08a942bbde4795b43f7cc1ff107a6b18a4a9ac531b94b2c01.jpg](../icml_results/1937_Multi-View%20Causal%20Discovery%20without%20Non-Gaussianity_%20Identifiability%20and%20Algorithms/images/b1bf99cda5c67cd08a942bbde4795b43f7cc1ff107a6b18a4a9ac531b94b2c01.jpg)

![d2c2a675b976ce555d01f936df02bb03a6682312e44c334cd5c1d0d05075013d.jpg](../icml_results/1937_Multi-View%20Causal%20Discovery%20without%20Non-Gaussianity_%20Identifiability%20and%20Algorithms/images/d2c2a675b976ce555d01f936df02bb03a6682312e44c334cd5c1d0d05075013d.jpg)

![d504a4bd3c86b9fdb100278f251964dd8d8f310a7de31aae172c62a649fe3cb2.jpg](../icml_results/1937_Multi-View%20Causal%20Discovery%20without%20Non-Gaussianity_%20Identifiability%20and%20Algorithms/images/d504a4bd3c86b9fdb100278f251964dd8d8f310a7de31aae172c62a649fe3cb2.jpg)

![dd383c69903e170272fae51098dba070f909eca3b69b2fc6c397e21d5fdb7f36.jpg](../icml_results/1937_Multi-View%20Causal%20Discovery%20without%20Non-Gaussianity_%20Identifiability%20and%20Algorithms/images/dd383c69903e170272fae51098dba070f909eca3b69b2fc6c397e21d5fdb7f36.jpg)

![df52e82c1b829fc1ccdc3ed3e976b0283130acbcd7a4656c60d08ba0ba99fc87.jpg](../icml_results/1937_Multi-View%20Causal%20Discovery%20without%20Non-Gaussianity_%20Identifiability%20and%20Algorithms/images/df52e82c1b829fc1ccdc3ed3e976b0283130acbcd7a4656c60d08ba0ba99fc87.jpg)

![e1b9bb2b23f64cab99607dcb71fb61ce39645974cea9d46c94883b43b7db592b.jpg](../icml_results/1937_Multi-View%20Causal%20Discovery%20without%20Non-Gaussianity_%20Identifiability%20and%20Algorithms/images/e1b9bb2b23f64cab99607dcb71fb61ce39645974cea9d46c94883b43b7db592b.jpg)

![e61f3ac74c5e8b70e2c0c02ff64ef8e7215b73718227b1f9380c3628526207ee.jpg](../icml_results/1937_Multi-View%20Causal%20Discovery%20without%20Non-Gaussianity_%20Identifiability%20and%20Algorithms/images/e61f3ac74c5e8b70e2c0c02ff64ef8e7215b73718227b1f9380c3628526207ee.jpg)

![e75749ed5ef4ef803e9fd48637ba1f109344b6aa081a493b691be12302109c29.jpg](../icml_results/1937_Multi-View%20Causal%20Discovery%20without%20Non-Gaussianity_%20Identifiability%20and%20Algorithms/images/e75749ed5ef4ef803e9fd48637ba1f109344b6aa081a493b691be12302109c29.jpg)

![e79fc7a56e5621fba7f047de1984a4c540b619f83a684666a8cd9082a344b2da.jpg](../icml_results/1937_Multi-View%20Causal%20Discovery%20without%20Non-Gaussianity_%20Identifiability%20and%20Algorithms/images/e79fc7a56e5621fba7f047de1984a4c540b619f83a684666a8cd9082a344b2da.jpg)

![f602df6c43f295af7a555cdea6feefc1eb4c41283e10d00928364759a636d827.jpg](../icml_results/1937_Multi-View%20Causal%20Discovery%20without%20Non-Gaussianity_%20Identifiability%20and%20Algorithms/images/f602df6c43f295af7a555cdea6feefc1eb4c41283e10d00928364759a636d827.jpg)

![ff3fa9becc41f8f32d814e1a5f1c726045ad4583af6efa3ac7ce8464b42822e9.jpg](../icml_results/1937_Multi-View%20Causal%20Discovery%20without%20Non-Gaussianity_%20Identifiability%20and%20Algorithms/images/ff3fa9becc41f8f32d814e1a5f1c726045ad4583af6efa3ac7ce8464b42822e9.jpg)

<a id="paper-1913"></a>

## Factored Gossip DiLoCo: Reducing Blocking Communication in DiLoCo


### Images

![092190064544288f652ca90730f31262523844a3468bc80065f675c72073b260.jpg](../icml_results/1938_Factored%20Gossip%20DiLoCo_%20Reducing%20Blocking%20Communication%20in%20DiLoCo/images/092190064544288f652ca90730f31262523844a3468bc80065f675c72073b260.jpg)

![0c0992e8d4492154d18c29de867e8303daf09ce174f500e4dee95ee34d4f8f3c.jpg](../icml_results/1938_Factored%20Gossip%20DiLoCo_%20Reducing%20Blocking%20Communication%20in%20DiLoCo/images/0c0992e8d4492154d18c29de867e8303daf09ce174f500e4dee95ee34d4f8f3c.jpg)

![7d0e1163073f8d93dd1ae520a72c8b0f8ae7fab4d7a2540978422a4107b8c3bc.jpg](../icml_results/1938_Factored%20Gossip%20DiLoCo_%20Reducing%20Blocking%20Communication%20in%20DiLoCo/images/7d0e1163073f8d93dd1ae520a72c8b0f8ae7fab4d7a2540978422a4107b8c3bc.jpg)

![9e0e5cfac08f728f7c8854fbccc7d56de35051c82e162e7c5f67d1a613a75e8b.jpg](../icml_results/1938_Factored%20Gossip%20DiLoCo_%20Reducing%20Blocking%20Communication%20in%20DiLoCo/images/9e0e5cfac08f728f7c8854fbccc7d56de35051c82e162e7c5f67d1a613a75e8b.jpg)

![baee466fc1eef69ddc7e3123b00e12fc33233c46fbf69f35d561a37d0a4db568.jpg](../icml_results/1938_Factored%20Gossip%20DiLoCo_%20Reducing%20Blocking%20Communication%20in%20DiLoCo/images/baee466fc1eef69ddc7e3123b00e12fc33233c46fbf69f35d561a37d0a4db568.jpg)

![cb1d68bd5d2e0fabb75ff1c808f1b69c6d7c05ba23b868ce7d1d89a909f05620.jpg](../icml_results/1938_Factored%20Gossip%20DiLoCo_%20Reducing%20Blocking%20Communication%20in%20DiLoCo/images/cb1d68bd5d2e0fabb75ff1c808f1b69c6d7c05ba23b868ce7d1d89a909f05620.jpg)

![e29c2e53bfafaf4640b58e4a9de9785893ba09e9f4fad87ea8e193d58c2c5e26.jpg](../icml_results/1938_Factored%20Gossip%20DiLoCo_%20Reducing%20Blocking%20Communication%20in%20DiLoCo/images/e29c2e53bfafaf4640b58e4a9de9785893ba09e9f4fad87ea8e193d58c2c5e26.jpg)

### Tables

![02ad3f09eb3da692aea63d308b5955ff19cab9262f49ddd0506baeff34c61217.jpg](../icml_results/1938_Factored%20Gossip%20DiLoCo_%20Reducing%20Blocking%20Communication%20in%20DiLoCo/tables/02ad3f09eb3da692aea63d308b5955ff19cab9262f49ddd0506baeff34c61217.jpg)

![2bad2770e628f65ca9f0a76a6639e24e8a6bfa7d850f4ca2e8390085a6fd6451.jpg](../icml_results/1938_Factored%20Gossip%20DiLoCo_%20Reducing%20Blocking%20Communication%20in%20DiLoCo/tables/2bad2770e628f65ca9f0a76a6639e24e8a6bfa7d850f4ca2e8390085a6fd6451.jpg)

![a700300fc71c6f9f5dd53640f1606d1c3cf6ef6de96c7dbc549e07b2c2d82676.jpg](../icml_results/1938_Factored%20Gossip%20DiLoCo_%20Reducing%20Blocking%20Communication%20in%20DiLoCo/tables/a700300fc71c6f9f5dd53640f1606d1c3cf6ef6de96c7dbc549e07b2c2d82676.jpg)

![b5f2b8a74c26e37c2f4c09dea1253f97a0ed813f64d43d17a921510a30ce770e.jpg](../icml_results/1938_Factored%20Gossip%20DiLoCo_%20Reducing%20Blocking%20Communication%20in%20DiLoCo/tables/b5f2b8a74c26e37c2f4c09dea1253f97a0ed813f64d43d17a921510a30ce770e.jpg)

![be09322a35012880dae0f588e889101b592f4b98067ffbeec759bc9cecbf224b.jpg](../icml_results/1938_Factored%20Gossip%20DiLoCo_%20Reducing%20Blocking%20Communication%20in%20DiLoCo/tables/be09322a35012880dae0f588e889101b592f4b98067ffbeec759bc9cecbf224b.jpg)

![d996b9c4e72c2b24efc3fb5e0c34496bbf20e5623ce4ff322c08e9c1fe57176b.jpg](../icml_results/1938_Factored%20Gossip%20DiLoCo_%20Reducing%20Blocking%20Communication%20in%20DiLoCo/tables/d996b9c4e72c2b24efc3fb5e0c34496bbf20e5623ce4ff322c08e9c1fe57176b.jpg)

<a id="paper-1914"></a>

## SONAR: Spectral-Contrastive Audio Residuals for Generalizable Deepfake Detection

### Images

![0c67f01e348b49453086070e6019d4a4d0a2079a58f269def347eef2d4da9588.jpg](../icml_results/1939_SONAR_%20Spectral-Contrastive%20Audio%20Residuals%20for%20Generalizable%20Deepfake%20Detection/images/0c67f01e348b49453086070e6019d4a4d0a2079a58f269def347eef2d4da9588.jpg)

![24612fc1f7c36fa868c313c04641ee3ec4c91479491aea4fdf7ec1e7bbd2944d.jpg](../icml_results/1939_SONAR_%20Spectral-Contrastive%20Audio%20Residuals%20for%20Generalizable%20Deepfake%20Detection/images/24612fc1f7c36fa868c313c04641ee3ec4c91479491aea4fdf7ec1e7bbd2944d.jpg)

![34d3c9fc2c111ab20fa947f55c71b8e49294638fa12b0f936adee9b575a285ea.jpg](../icml_results/1939_SONAR_%20Spectral-Contrastive%20Audio%20Residuals%20for%20Generalizable%20Deepfake%20Detection/images/34d3c9fc2c111ab20fa947f55c71b8e49294638fa12b0f936adee9b575a285ea.jpg)

![395e5d0f49a213766f273b2f7e9558b9e9e4b4cc2976065dd2414522d0162a11.jpg](../icml_results/1939_SONAR_%20Spectral-Contrastive%20Audio%20Residuals%20for%20Generalizable%20Deepfake%20Detection/images/395e5d0f49a213766f273b2f7e9558b9e9e4b4cc2976065dd2414522d0162a11.jpg)

![3ba13541e55b823e95be04f430579b64ce69fe8bd63ea08ff1f65feb0dae9463.jpg](../icml_results/1939_SONAR_%20Spectral-Contrastive%20Audio%20Residuals%20for%20Generalizable%20Deepfake%20Detection/images/3ba13541e55b823e95be04f430579b64ce69fe8bd63ea08ff1f65feb0dae9463.jpg)

![5c10fc30b4be9adf513c7dee4663530217acdaa431cef5d32a284328a3afb299.jpg](../icml_results/1939_SONAR_%20Spectral-Contrastive%20Audio%20Residuals%20for%20Generalizable%20Deepfake%20Detection/images/5c10fc30b4be9adf513c7dee4663530217acdaa431cef5d32a284328a3afb299.jpg)

![e3852b10ffe3074530a8d9a92696ae98e175a41156b317b3486fd0743657c0e0.jpg](../icml_results/1939_SONAR_%20Spectral-Contrastive%20Audio%20Residuals%20for%20Generalizable%20Deepfake%20Detection/images/e3852b10ffe3074530a8d9a92696ae98e175a41156b317b3486fd0743657c0e0.jpg)

![f152d5ebadeb1a74e24a4a8f8d076fcb8f705abc2270d778effb3298f4e1e8fe.jpg](../icml_results/1939_SONAR_%20Spectral-Contrastive%20Audio%20Residuals%20for%20Generalizable%20Deepfake%20Detection/images/f152d5ebadeb1a74e24a4a8f8d076fcb8f705abc2270d778effb3298f4e1e8fe.jpg)

### Tables

![0ee4c4000de330820d43260974e42701dae6cb4ea8c98a644e1f1e9e04608c00.jpg](../icml_results/1939_SONAR_%20Spectral-Contrastive%20Audio%20Residuals%20for%20Generalizable%20Deepfake%20Detection/tables/0ee4c4000de330820d43260974e42701dae6cb4ea8c98a644e1f1e9e04608c00.jpg)

![7f10b2cab8e1fe90597ac8b9832be0e0524b21ed4cbf6caf326e851b8f149e9a.jpg](../icml_results/1939_SONAR_%20Spectral-Contrastive%20Audio%20Residuals%20for%20Generalizable%20Deepfake%20Detection/tables/7f10b2cab8e1fe90597ac8b9832be0e0524b21ed4cbf6caf326e851b8f149e9a.jpg)

![8410ef2ee47307cd80920c544f4bbcd40181d7037b60e4367cb099c159714ead.jpg](../icml_results/1939_SONAR_%20Spectral-Contrastive%20Audio%20Residuals%20for%20Generalizable%20Deepfake%20Detection/tables/8410ef2ee47307cd80920c544f4bbcd40181d7037b60e4367cb099c159714ead.jpg)

![94505989623fd4b298397316b73b959d826b0477a012a5f284a72752c2267ca8.jpg](../icml_results/1939_SONAR_%20Spectral-Contrastive%20Audio%20Residuals%20for%20Generalizable%20Deepfake%20Detection/tables/94505989623fd4b298397316b73b959d826b0477a012a5f284a72752c2267ca8.jpg)

![a017c89ed72ff84b03fa7878d5e14233bcd3c9b75c23577329bc179530d8b855.jpg](../icml_results/1939_SONAR_%20Spectral-Contrastive%20Audio%20Residuals%20for%20Generalizable%20Deepfake%20Detection/tables/a017c89ed72ff84b03fa7878d5e14233bcd3c9b75c23577329bc179530d8b855.jpg)

![ab9ba4ca196a91243b946d103a33b60964ee4160c409540bc0236da6016205f0.jpg](../icml_results/1939_SONAR_%20Spectral-Contrastive%20Audio%20Residuals%20for%20Generalizable%20Deepfake%20Detection/tables/ab9ba4ca196a91243b946d103a33b60964ee4160c409540bc0236da6016205f0.jpg)
