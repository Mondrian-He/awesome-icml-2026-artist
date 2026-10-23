# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 36 of 197

## 目录 (Table of Contents)

1. [Improving Visual Token Reduction via Rectifying Distortions for Efficient Multimodal LLM Inference](#paper-1156)
2. [GAAVI: Global Asymptotic Anytime Valid Inference for the Conditional Mean Function](#paper-1157)
3. [Balancing Fidelity and Diversity in Diffusion Models via Symmetric Attention Decomposition: Hopfield Perspective](#paper-1158)
4. [Preserve-Then-Quantize: Balancing Rank Budgets for Quantization Error Reconstruction in LLMs](#paper-1159)
5. [Bilinear Bandits with Partially Observable Features](#paper-1160)
6. [CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](#paper-1161)
7. [Expert-guided Clinical Text Augmentation via Query-Based Model Collaboration](#paper-1162)
8. [PinTok: Tokenizers Deserve Dedicated Pinned CPU-Compute and Memory](#paper-1163)
9. [Efficient Skill Grounding via Code Refactoring with Small Language Models](#paper-1164)
10. [Gauge-Equivariant Graph Networks via Self-Interference Cancellation](#paper-1165)
11. [The Truth Stays in the Family: Enhancing Contextual Truthfulness via Inherited Heads in Model Lineages](#paper-1166)
12. [Anchored Self-Play for Code Repair](#paper-1167)
13. [Decentralized Instruction Tuning: Conflict-Aware Splitting and Weight Merging](#paper-1168)
14. [Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts](#paper-1169)
15. [EMBGuard: Constructing Hazard-Aware Guardrails for Safe Planning in Embodied Agents](#paper-1170)
16. [Parameter Decorrelation via Transition-Variance Alignment for Multivariate Time-series Forecasting](#paper-1171)
17. [Chain-of-Goals Hierarchical Policy for Long-Horizon Offline Goal-Conditioned RL](#paper-1172)
18. [Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory](#paper-1173)
19. [Rethinking the Design Space of Reinforcement Learning for Diffusion Models: On the Importance of Likelihood Estimation Beyond Loss Design](#paper-1174)
20. [NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](#paper-1175)
21. [Towards Feedback-to-Plan Decisions for Self-Evolving LLM Agents in CUDA Kernel Generation](#paper-1176)
22. [Beyond Extrapolation: Knowledge Utilization Paradigm with Bidirectional Inspiration for Time Series Forecasting](#paper-1177)
23. [Approximate Proportionality in Online Fair Division](#paper-1178)
24. [Learning Unanimously Acceptable Lotteries via Queries](#paper-1179)
25. [Computationally-efficient Graph Modeling with Refined Graph Random Features](#paper-1180)
26. [Learning When to Attend: Conditional Memory Access for Long-Context LLMs](#paper-1181)
27. [Through the Stealth Lens: Attention-Aware Defenses Against Poisoning in RAG](#paper-1182)
28. [LAGEA: Language Guided Embodied Agents for Robotic Manipulation](#paper-1183)
29. [S2M-Net: Spectral-Spatial Mixing with Morphology-Aware Adaptive Loss for Medical Image Segmentation](#paper-1184)
30. [SpaCeFormer: Fast Proposal-Free Open-Vocabulary 3D Instance Segmentation](#paper-1185)
31. [Efficient LLM Moderation with Multi-Layer Latent Prototypes](#paper-1186)
32. [Value Aggregation with Uncertainty in Online Decentralized MARL](#paper-1187)
33. [Balancing Plasticity and Stability with Fast and Slow Successor Features](#paper-1188)

---


<a id="paper-1156"></a>

## Improving Visual Token Reduction via Rectifying Distortions for Efficient Multimodal LLM Inference

### Images

![06199a0b62f9306c3f50d14613e2660eddcfd61d7262998f8acadb46867c6c76.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/06199a0b62f9306c3f50d14613e2660eddcfd61d7262998f8acadb46867c6c76.jpg)

![2bf65b31d55c4fc795dcead203ae149987eb2cef63b1bc28c4718e60210303fd.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/2bf65b31d55c4fc795dcead203ae149987eb2cef63b1bc28c4718e60210303fd.jpg)

![36e2e32144eb0efb2f6cfe2257ae9926eb86772b4b9825ccb7f0a9e3d33a4ad8.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/36e2e32144eb0efb2f6cfe2257ae9926eb86772b4b9825ccb7f0a9e3d33a4ad8.jpg)

![3751b4de1cafe9763fb64dfa4f88c79c1c0ec13a03b6c3061559e549d9f39d9e.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/3751b4de1cafe9763fb64dfa4f88c79c1c0ec13a03b6c3061559e549d9f39d9e.jpg)

![3d8c1e60b706ebe826e9bcbcbfd232c460e57e8fd91e3aa6e95e26fa59a55589.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/3d8c1e60b706ebe826e9bcbcbfd232c460e57e8fd91e3aa6e95e26fa59a55589.jpg)

![4b260ee72e2ea690a789d5ab82d22b0d189cb33c3d46fddca210e9095bdc9eef.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/4b260ee72e2ea690a789d5ab82d22b0d189cb33c3d46fddca210e9095bdc9eef.jpg)

![4e556b412e696694f30c63a85bf32d2b0027780773b0dc30111e474648f897ed.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/4e556b412e696694f30c63a85bf32d2b0027780773b0dc30111e474648f897ed.jpg)

![558eafea4a72c95282b721f24ecfaff9dd17f6a7eacd18c5ed759060eb804394.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/558eafea4a72c95282b721f24ecfaff9dd17f6a7eacd18c5ed759060eb804394.jpg)

![5bc832e24fc7b97063a142e3c25cfff60b9dcb4fed57b76987c4ab8f5b18828a.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/5bc832e24fc7b97063a142e3c25cfff60b9dcb4fed57b76987c4ab8f5b18828a.jpg)

![64ff0f8a4a682e1dd4a99c062a4aeec98ef0cb40fbf6da16331b14eeedbfef00.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/64ff0f8a4a682e1dd4a99c062a4aeec98ef0cb40fbf6da16331b14eeedbfef00.jpg)

![66b0540e93bcd600c411f4fc62eff436aa2053f17415a42d3ccfd26e7b9b0e48.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/66b0540e93bcd600c411f4fc62eff436aa2053f17415a42d3ccfd26e7b9b0e48.jpg)

![72359509c360eb6f2f44de411f6f198c55218a76a06ef6bee5eea3d5678cefe9.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/72359509c360eb6f2f44de411f6f198c55218a76a06ef6bee5eea3d5678cefe9.jpg)

![726ac5bfcb49aa3f321728cbe06dcb48859033bc535b51c2214774ac62539ef9.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/726ac5bfcb49aa3f321728cbe06dcb48859033bc535b51c2214774ac62539ef9.jpg)

![909cf0e05af606b28283eb565cdf2808efa57e9658ba118a101a0e4b7b5b67cd.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/909cf0e05af606b28283eb565cdf2808efa57e9658ba118a101a0e4b7b5b67cd.jpg)

![a14958b3c618f958bba5fa92924b4a5b59c1221ef432f2fbf6b7148ab23810df.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/a14958b3c618f958bba5fa92924b4a5b59c1221ef432f2fbf6b7148ab23810df.jpg)

![a2f0699c098802bed778d4e8efd35ccb878db75422ca1890b7830a703a078d55.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/a2f0699c098802bed778d4e8efd35ccb878db75422ca1890b7830a703a078d55.jpg)

![bfc2da9502bc41864efabc65ea007d1739cdfe02f1aeeaaf9034e2cbf8169748.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/bfc2da9502bc41864efabc65ea007d1739cdfe02f1aeeaaf9034e2cbf8169748.jpg)

![da26356250af8e97ad04ea20839746e9f81182b2f78723c6ce1bf09b5bc1bd96.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/da26356250af8e97ad04ea20839746e9f81182b2f78723c6ce1bf09b5bc1bd96.jpg)

![e371307dd30392fef2f703e3b95daec6ffa3f8bb6ded71e40f5c1229645bce2a.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/e371307dd30392fef2f703e3b95daec6ffa3f8bb6ded71e40f5c1229645bce2a.jpg)

![ff3d1c83d2f0fc728fc882873091d82aff429dcffeba8ee4347fce5c511b9017.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/images/ff3d1c83d2f0fc728fc882873091d82aff429dcffeba8ee4347fce5c511b9017.jpg)

### Tables

![14628a7239fec1ef47cdafb4981bcb8609d2cbf04ae0d67bede786a335e49348.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/14628a7239fec1ef47cdafb4981bcb8609d2cbf04ae0d67bede786a335e49348.jpg)

![1849c57d0ebf8b191208ae975e9874c1f338ee86ee6d5768675cbea28b9d9978.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/1849c57d0ebf8b191208ae975e9874c1f338ee86ee6d5768675cbea28b9d9978.jpg)

![323b34f2b28e80f324e8e22e3500495e365aff4e79a352bb45cc796d29d829fb.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/323b34f2b28e80f324e8e22e3500495e365aff4e79a352bb45cc796d29d829fb.jpg)

![516335a2c07df721699b7465552c84d19ede4a79eb963111a3d4616222ba72da.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/516335a2c07df721699b7465552c84d19ede4a79eb963111a3d4616222ba72da.jpg)

![5681a1929f3a5014a1e24b0dba01edcc706f7458bb03c032412591ea26d1e171.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/5681a1929f3a5014a1e24b0dba01edcc706f7458bb03c032412591ea26d1e171.jpg)

![58d9885a283bea39763f760e94c363d5540eae8a3a4c88b0a07f89223b35c056.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/58d9885a283bea39763f760e94c363d5540eae8a3a4c88b0a07f89223b35c056.jpg)

![5905eeb90f4cd9d5eabc958c52d97b4d5b44866405651a70d1ac530213bb662a.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/5905eeb90f4cd9d5eabc958c52d97b4d5b44866405651a70d1ac530213bb662a.jpg)

![6892e5945eca67a50374dffe79f578f5a09bad3a1f5a90f29663095abcdad849.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/6892e5945eca67a50374dffe79f578f5a09bad3a1f5a90f29663095abcdad849.jpg)

![6b17b26bd14732b3a4ea7fa8c4e24c24fa618a3655455f9119043d6a8fb0557a.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/6b17b26bd14732b3a4ea7fa8c4e24c24fa618a3655455f9119043d6a8fb0557a.jpg)

![6c1ba52236c24a54d981e9f6484c33cc3b332aaa171b2ca9153c897326c60d96.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/6c1ba52236c24a54d981e9f6484c33cc3b332aaa171b2ca9153c897326c60d96.jpg)

![9737f442777033364f1cfa43f4973ec6ffb14e55fde3a720ab14d76f4fbaba29.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/9737f442777033364f1cfa43f4973ec6ffb14e55fde3a720ab14d76f4fbaba29.jpg)

![af203afca41e72f08226dcc0b2b690aa749222b2bdbe60526a8ce06ee6f15f1c.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/af203afca41e72f08226dcc0b2b690aa749222b2bdbe60526a8ce06ee6f15f1c.jpg)

![af5545b69efc5f7cf1f2a13b043c6f13764b3fd188b1e632ba6d374ae3a63288.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/af5545b69efc5f7cf1f2a13b043c6f13764b3fd188b1e632ba6d374ae3a63288.jpg)

![df268116d0fdd700214945f9450920f3985d38ae1e36db4a54f86e947cdc8ed6.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/df268116d0fdd700214945f9450920f3985d38ae1e36db4a54f86e947cdc8ed6.jpg)

![e1b230424e67da05017a6697de50bac02fcad31155e54c2b6fd0a5f636cb4bdd.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/e1b230424e67da05017a6697de50bac02fcad31155e54c2b6fd0a5f636cb4bdd.jpg)

![e6ec9c515cb68295dc73d7fb1a69e242ff15654facd9e9858d8f6998d014f156.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/e6ec9c515cb68295dc73d7fb1a69e242ff15654facd9e9858d8f6998d014f156.jpg)

![efad7083aa377b735aa92b645eb8242e839bb322c0e3d96ee9403fd90122475c.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/efad7083aa377b735aa92b645eb8242e839bb322c0e3d96ee9403fd90122475c.jpg)

![ff1a3971aa8ca6bafed6e0523a06e58c706135a1b6df4d676dddefdfcbc9160d.jpg](../icml_results/1173_TAG_%20Tangential%20Amplifying%20Guidance%20for%20Hallucination-Resistant%20Sampling/tables/ff1a3971aa8ca6bafed6e0523a06e58c706135a1b6df4d676dddefdfcbc9160d.jpg)

## Improving Visual Token Reduction via Rectifying Distortions for Efficient Multimodal LLM Inference


### Images

![3c143092a5403bb59f9d5819fddbb38ba54042b62ca65c5c59994c8dbde46580.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/images/3c143092a5403bb59f9d5819fddbb38ba54042b62ca65c5c59994c8dbde46580.jpg)

![461bcd1d48915b2eee5f4468c6700373205c3e1d7d48e396df7631232bb08411.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/images/461bcd1d48915b2eee5f4468c6700373205c3e1d7d48e396df7631232bb08411.jpg)

![aa84df84f3779826c1b36acb7ecc06ce91ea13e0d6d7578dfaaa4ab169794958.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/images/aa84df84f3779826c1b36acb7ecc06ce91ea13e0d6d7578dfaaa4ab169794958.jpg)

![b209a91c365224b04795d0c1a1dd31e05c9c3ef4f7358b7831905747cbc834bf.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/images/b209a91c365224b04795d0c1a1dd31e05c9c3ef4f7358b7831905747cbc834bf.jpg)

![e9b161cd7d7fa990584623be6545e77523f60ca4bf1f5fb705668600dd3de394.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/images/e9b161cd7d7fa990584623be6545e77523f60ca4bf1f5fb705668600dd3de394.jpg)

### Tables

![0e44c1d5016fda694d77cd6e929df9fbd197d1ccf8dedf7a0a5ef14bc28addf6.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/0e44c1d5016fda694d77cd6e929df9fbd197d1ccf8dedf7a0a5ef14bc28addf6.jpg)

![11dee3df4d4ce732e07541837f0eed05fcf5edeb3c6d92d8ece70b3b046c47cf.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/11dee3df4d4ce732e07541837f0eed05fcf5edeb3c6d92d8ece70b3b046c47cf.jpg)

![200ff2d5f2badf0f3ce9be0f926a4e8cb39078ab5b19cc52f27e7f01531961d1.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/200ff2d5f2badf0f3ce9be0f926a4e8cb39078ab5b19cc52f27e7f01531961d1.jpg)

![35f021e126c5a59676f861107e09e2740b0b5e2b51f9ecaeda8f5ae94fb969d7.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/35f021e126c5a59676f861107e09e2740b0b5e2b51f9ecaeda8f5ae94fb969d7.jpg)

![3c8cbb253c5138847a24d04b73a8006d04cf6236348fb8353918d6baedc32bdf.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/3c8cbb253c5138847a24d04b73a8006d04cf6236348fb8353918d6baedc32bdf.jpg)

![41a92ebc349a81304859e2645f4e7212210e4a8e5499da7651638651bdaa967e.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/41a92ebc349a81304859e2645f4e7212210e4a8e5499da7651638651bdaa967e.jpg)

![45bed51cd9a8d5d9991200a23c11dc8a5fc6b15000132d641d8d7921afd79b8d.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/45bed51cd9a8d5d9991200a23c11dc8a5fc6b15000132d641d8d7921afd79b8d.jpg)

![45ecb8cddefd8beee39a9c5aafb6d4fc5720d81bd9732e832cf1074bcd019960.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/45ecb8cddefd8beee39a9c5aafb6d4fc5720d81bd9732e832cf1074bcd019960.jpg)

![4d86c0203ef74a1abc34ba8b8ca7ba084cf0fc179dc2a17ab59dc96fd7b4bc34.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/4d86c0203ef74a1abc34ba8b8ca7ba084cf0fc179dc2a17ab59dc96fd7b4bc34.jpg)

![4ee10bf98196cc3c81682127f7c7e5fb5c02e7a995d7074c169c1f58818c16b0.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/4ee10bf98196cc3c81682127f7c7e5fb5c02e7a995d7074c169c1f58818c16b0.jpg)

![69ee7b0c119a7d19ffb524629fad0e262ce812956427f4f273d28edcda139d3f.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/69ee7b0c119a7d19ffb524629fad0e262ce812956427f4f273d28edcda139d3f.jpg)

![7f39228b90341dcb2bc4d2e73bf2a4330169d1c4a2f73f5c586403373215f586.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/7f39228b90341dcb2bc4d2e73bf2a4330169d1c4a2f73f5c586403373215f586.jpg)

![8e8bfe6665377fd61a4509e843992b2db50adc3e420d120f124101e53b30f5c9.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/8e8bfe6665377fd61a4509e843992b2db50adc3e420d120f124101e53b30f5c9.jpg)

![9b39a81eec850462aabb44185876f9d35f22858e31abfa68caf9183ee625651f.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/9b39a81eec850462aabb44185876f9d35f22858e31abfa68caf9183ee625651f.jpg)

![9ecc551fc076968734bb40d9f4e26881b4c5686899c82fa4f191e8486f72b827.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/9ecc551fc076968734bb40d9f4e26881b4c5686899c82fa4f191e8486f72b827.jpg)

![bcd9ffbc5baa514f539d7edd3cbfc0b1429a68285b8587f5c366aa2e3f529a93.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/bcd9ffbc5baa514f539d7edd3cbfc0b1429a68285b8587f5c366aa2e3f529a93.jpg)

![bec59651d3fe2cb527b19a56acd77d5a2f00d4f0ab1a8a5e913296a45f427f0a.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/bec59651d3fe2cb527b19a56acd77d5a2f00d4f0ab1a8a5e913296a45f427f0a.jpg)

![db76e14575e6a02a560f98d98b4c113b44326abdd3f5472eb4900ee630b1a615.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/db76e14575e6a02a560f98d98b4c113b44326abdd3f5472eb4900ee630b1a615.jpg)

![e2dbf7b8113ba688a5d327f3922999188b8b336094668112f39caf2d9aab9ce1.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/e2dbf7b8113ba688a5d327f3922999188b8b336094668112f39caf2d9aab9ce1.jpg)

![ef68eb7457dd63c3db1e200125b2b3a7f3b0e467eb59cb2559cb0eae4fb63afa.jpg](../icml_results/1174_Improving%20Visual%20Token%20Reduction%20via%20Rectifying%20Distortions%20for%20Efficient%20Multimodal%20LLM%20Inference/tables/ef68eb7457dd63c3db1e200125b2b3a7f3b0e467eb59cb2559cb0eae4fb63afa.jpg)

<a id="paper-1157"></a>

## GAAVI: Global Asymptotic Anytime Valid Inference for the Conditional Mean Function


### Images

![0e182c32e6fe8d6aba25a4b730e2dd47821ca16c8e2fc7635ddc1b6f4fc303fd.jpg](../icml_results/1175_GAAVI_%20Global%20Asymptotic%20Anytime%20Valid%20Inference%20for%20the%20Conditional%20Mean%20Function/images/0e182c32e6fe8d6aba25a4b730e2dd47821ca16c8e2fc7635ddc1b6f4fc303fd.jpg)

![2890a5ebd0353d45a1dacd7380e3d0ce48182a2da35aff7ff614bf7810bcc3a5.jpg](../icml_results/1175_GAAVI_%20Global%20Asymptotic%20Anytime%20Valid%20Inference%20for%20the%20Conditional%20Mean%20Function/images/2890a5ebd0353d45a1dacd7380e3d0ce48182a2da35aff7ff614bf7810bcc3a5.jpg)

![6c38d8377f38578446f83f7ec1c42ebd212aa02c1454b4e9fb916e88d6b723b5.jpg](../icml_results/1175_GAAVI_%20Global%20Asymptotic%20Anytime%20Valid%20Inference%20for%20the%20Conditional%20Mean%20Function/images/6c38d8377f38578446f83f7ec1c42ebd212aa02c1454b4e9fb916e88d6b723b5.jpg)

![71ef1a7efaaa792980b9113eb107a2182233a5aac9b6c84f24f63a2dd08bee99.jpg](../icml_results/1175_GAAVI_%20Global%20Asymptotic%20Anytime%20Valid%20Inference%20for%20the%20Conditional%20Mean%20Function/images/71ef1a7efaaa792980b9113eb107a2182233a5aac9b6c84f24f63a2dd08bee99.jpg)

![7f5329e564a10354d7da0eb6c68aeadc0af46a9490c02bbc046d9ed08d34a1c6.jpg](../icml_results/1175_GAAVI_%20Global%20Asymptotic%20Anytime%20Valid%20Inference%20for%20the%20Conditional%20Mean%20Function/images/7f5329e564a10354d7da0eb6c68aeadc0af46a9490c02bbc046d9ed08d34a1c6.jpg)

![c2d2db4e292d26b2400cddd962b629f8a61449689759d18137cd226c31478f64.jpg](../icml_results/1175_GAAVI_%20Global%20Asymptotic%20Anytime%20Valid%20Inference%20for%20the%20Conditional%20Mean%20Function/images/c2d2db4e292d26b2400cddd962b629f8a61449689759d18137cd226c31478f64.jpg)

![f353bfd02309a80ad8e05b927700eebf80833672a2ecf9ce3d7c5a6c29ec8e47.jpg](../icml_results/1175_GAAVI_%20Global%20Asymptotic%20Anytime%20Valid%20Inference%20for%20the%20Conditional%20Mean%20Function/images/f353bfd02309a80ad8e05b927700eebf80833672a2ecf9ce3d7c5a6c29ec8e47.jpg)

### Tables

![aff4b39cf05a59282cd3fd4dff03123ce11e9b89aa978080127522d82a02614e.jpg](../icml_results/1175_GAAVI_%20Global%20Asymptotic%20Anytime%20Valid%20Inference%20for%20the%20Conditional%20Mean%20Function/tables/aff4b39cf05a59282cd3fd4dff03123ce11e9b89aa978080127522d82a02614e.jpg)

<a id="paper-1158"></a>

## Balancing Fidelity and Diversity in Diffusion Models via Symmetric Attention Decomposition: Hopfield Perspective


### Images

![0b3c2be552add9d2517a912625168dd7527ff9cf6281373f0509cfa4f4d0dbd4.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/0b3c2be552add9d2517a912625168dd7527ff9cf6281373f0509cfa4f4d0dbd4.jpg)

![2306340e4b39f2a010ee284adf88145cb31632fec69b37d52adab9e7bcf762ae.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/2306340e4b39f2a010ee284adf88145cb31632fec69b37d52adab9e7bcf762ae.jpg)

![3844f9eabff78293c24b650ff1d5086a7b944d52aa581dcbff85d92d6f4fdeb2.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/3844f9eabff78293c24b650ff1d5086a7b944d52aa581dcbff85d92d6f4fdeb2.jpg)

![39ca58d52de9be41c1ec7ed81e62a483ebf624def212bd8a8b7cec5960955ef5.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/39ca58d52de9be41c1ec7ed81e62a483ebf624def212bd8a8b7cec5960955ef5.jpg)

![462240e0a0754e332d388876f5a35020d5a375df6d0abbe0a74dd74456c86ac9.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/462240e0a0754e332d388876f5a35020d5a375df6d0abbe0a74dd74456c86ac9.jpg)

![4c9d1ca5e4e05f09f3683e2c528be906a7c5788c0f32458ae34706cf70d7eb3f.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/4c9d1ca5e4e05f09f3683e2c528be906a7c5788c0f32458ae34706cf70d7eb3f.jpg)

![58e87706c13f2ebe65f730df8180d4053c5e170ebd08f1942b2f356fbcfc91b6.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/58e87706c13f2ebe65f730df8180d4053c5e170ebd08f1942b2f356fbcfc91b6.jpg)

![652e42a4e7ce184065b4b886706f9b370eb46525c51a47d1695dd5d64451a26a.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/652e42a4e7ce184065b4b886706f9b370eb46525c51a47d1695dd5d64451a26a.jpg)

![65d6769c22b040c92552335a2eedb3cfd863256fc1af876f6b502e0d7bec094c.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/65d6769c22b040c92552335a2eedb3cfd863256fc1af876f6b502e0d7bec094c.jpg)

![79a3620b38580b845f338ba872846f0f77dca6e14d925ed3d355a03ef0fb5383.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/79a3620b38580b845f338ba872846f0f77dca6e14d925ed3d355a03ef0fb5383.jpg)

![acbfa0f1a8568f2b2287449f830643a924e5814cd314dbefce58e12c731f598d.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/acbfa0f1a8568f2b2287449f830643a924e5814cd314dbefce58e12c731f598d.jpg)

![b3cc15346dd932a9a98a4c6efa4cb7c06747f7837d4e18b21adfcc117afcd96b.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/b3cc15346dd932a9a98a4c6efa4cb7c06747f7837d4e18b21adfcc117afcd96b.jpg)

![ca6e6592c530210fbce715e1929f7726f0db274f09196c9fa21039b94ecc71ab.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/ca6e6592c530210fbce715e1929f7726f0db274f09196c9fa21039b94ecc71ab.jpg)

![d007468f942abb739f5e5f6ce8697b68b843d1a9958357c117d7b3c93be2a92b.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/d007468f942abb739f5e5f6ce8697b68b843d1a9958357c117d7b3c93be2a92b.jpg)

![d652921ca90412b1e6feec7bcf1ecb9dfb4c8d19bf43772a30ef65b979ca5a36.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/d652921ca90412b1e6feec7bcf1ecb9dfb4c8d19bf43772a30ef65b979ca5a36.jpg)

![d73c49b98ac20692ce5d78b1cea66878e41899162b5bf90bfbfe36e5a9071333.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/d73c49b98ac20692ce5d78b1cea66878e41899162b5bf90bfbfe36e5a9071333.jpg)

![e66c68effab8112d4dfe0883774d65fa724c9c3bb76f550569b1b0b1025cb54b.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/e66c68effab8112d4dfe0883774d65fa724c9c3bb76f550569b1b0b1025cb54b.jpg)

![fb8677533048c7a9c7af1366dcad3ba975b15857ac10c8b2595d6f2c4c71b534.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/images/fb8677533048c7a9c7af1366dcad3ba975b15857ac10c8b2595d6f2c4c71b534.jpg)

### Tables

![01f6e9c88a4cb9ece57b4d0f98529cbc7fce7c27c7489cee59392c25a06e889b.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/tables/01f6e9c88a4cb9ece57b4d0f98529cbc7fce7c27c7489cee59392c25a06e889b.jpg)

![0465891a76d17b67e3c9e04b7f30d6e8368a9432bafef9a75ecf040e4be455e1.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/tables/0465891a76d17b67e3c9e04b7f30d6e8368a9432bafef9a75ecf040e4be455e1.jpg)

![0daac6b040ba9b26f793241482211420b2cb895c8fa4f634a25c036cc8f6adbf.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/tables/0daac6b040ba9b26f793241482211420b2cb895c8fa4f634a25c036cc8f6adbf.jpg)

![3d3e04a85c6542efb12d153ff70d7674da9340b21914c29516c3afec285edc01.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/tables/3d3e04a85c6542efb12d153ff70d7674da9340b21914c29516c3afec285edc01.jpg)

![6fa846e8734e8d43b57e2346becaf9ee97f0d4c5f25b5e66faadcfb0574fd7c7.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/tables/6fa846e8734e8d43b57e2346becaf9ee97f0d4c5f25b5e66faadcfb0574fd7c7.jpg)

![89cd8cef9f7bf700564f1ad5d6ee47221984c2688db0f5dcea0b258d1869d127.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/tables/89cd8cef9f7bf700564f1ad5d6ee47221984c2688db0f5dcea0b258d1869d127.jpg)

![c78b4cf5860fdd9359cd8be939863a3aa421d20398d981ba9d504948511294e5.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/tables/c78b4cf5860fdd9359cd8be939863a3aa421d20398d981ba9d504948511294e5.jpg)

![c925c2b3afc84c39f5523c84ad812c525678d88a1c3f934a8e3299dea3081494.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/tables/c925c2b3afc84c39f5523c84ad812c525678d88a1c3f934a8e3299dea3081494.jpg)

![dde79eaec937b5f4aa4f457f2b3156b2e2fe5c9b014700f2dd469364083f3d25.jpg](../icml_results/1176_Balancing%20Fidelity%20and%20Diversity%20in%20Diffusion%20Models%20via%20Symmetric%20Attention%20Decomposition_%20Hopfield/tables/dde79eaec937b5f4aa4f457f2b3156b2e2fe5c9b014700f2dd469364083f3d25.jpg)

<a id="paper-1159"></a>

## Preserve-Then-Quantize: Balancing Rank Budgets for Quantization Error Reconstruction in LLMs


### Images

![204fa19661f4f95259b1d1a962fb8abc757860f0af281babbfc4637a62422d1e.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/images/204fa19661f4f95259b1d1a962fb8abc757860f0af281babbfc4637a62422d1e.jpg)

![57746dff5a6f1636df03beedb555fcc91644cf06e6ea2910e69d420b92cce387.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/images/57746dff5a6f1636df03beedb555fcc91644cf06e6ea2910e69d420b92cce387.jpg)

![5ba2e46b74158faac59910e5598665439e706d9eed25d9e2d73c0f558ab63991.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/images/5ba2e46b74158faac59910e5598665439e706d9eed25d9e2d73c0f558ab63991.jpg)

![6497c17db3096929c343ebe4096038d0d4d488b9e2e156e23b719af713692fe1.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/images/6497c17db3096929c343ebe4096038d0d4d488b9e2e156e23b719af713692fe1.jpg)

![870b4f7e7d7965eed2b5019327282f75cdf31b32d396d9d690eda08f79209f18.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/images/870b4f7e7d7965eed2b5019327282f75cdf31b32d396d9d690eda08f79209f18.jpg)

![a9c2fe3a7f0ee07a537dba97f69b51a2c0b0c7a2fb63c78d06fec90cce2ece2c.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/images/a9c2fe3a7f0ee07a537dba97f69b51a2c0b0c7a2fb63c78d06fec90cce2ece2c.jpg)

![b986b6987443a4fabe7b67705562af8ebe41d027b0bd4f3657bb56d133d074d7.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/images/b986b6987443a4fabe7b67705562af8ebe41d027b0bd4f3657bb56d133d074d7.jpg)

![be219b6d4d64032f19e1a9fbb013ff280aacc74ef093bf76e0668f2545d1d439.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/images/be219b6d4d64032f19e1a9fbb013ff280aacc74ef093bf76e0668f2545d1d439.jpg)

![f78a174e75e266673c265264d6f173c61531ee71554f29f08f1fdf3e7f093cf0.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/images/f78a174e75e266673c265264d6f173c61531ee71554f29f08f1fdf3e7f093cf0.jpg)

### Tables

![0d2e7553a293307874df653bd7e01156610af27bd89b73cc2ddb982b5bca831f.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/0d2e7553a293307874df653bd7e01156610af27bd89b73cc2ddb982b5bca831f.jpg)

![25625c54ff981b547cb0e1d4d4ebfffc7c0304352b70e04e8da7749aa1748a21.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/25625c54ff981b547cb0e1d4d4ebfffc7c0304352b70e04e8da7749aa1748a21.jpg)

![299c18b3666ae257f446e1ba0fe2cc529a1819cd60a1c7cbcb4ceee1ad58925c.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/299c18b3666ae257f446e1ba0fe2cc529a1819cd60a1c7cbcb4ceee1ad58925c.jpg)

![2feddcc0bca812639363b7ff9f4c75135826db543566976ce6cd3f1bb08c93ca.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/2feddcc0bca812639363b7ff9f4c75135826db543566976ce6cd3f1bb08c93ca.jpg)

![3032469e384681696f18a308aaecc7707c6fafc4d43c392d4e59ce2bd06b7cbd.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/3032469e384681696f18a308aaecc7707c6fafc4d43c392d4e59ce2bd06b7cbd.jpg)

![325ad077d15e1c6b654dd8e7ec1a0cb5a410f14cafc8b5f54b1d309f2266f63d.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/325ad077d15e1c6b654dd8e7ec1a0cb5a410f14cafc8b5f54b1d309f2266f63d.jpg)

![358817ef26858b6ac7e2eed1430060b3f2720fac8508ec4f5a9da6146ed497ec.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/358817ef26858b6ac7e2eed1430060b3f2720fac8508ec4f5a9da6146ed497ec.jpg)

![3ad9724b6bcf0bb82a606dc548780cbf1669ce19473ce97066216b9925e794d8.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/3ad9724b6bcf0bb82a606dc548780cbf1669ce19473ce97066216b9925e794d8.jpg)

![4049a16d8109bde6439ef68a5beffd80c93e5582531fb23584efaf1840c32018.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/4049a16d8109bde6439ef68a5beffd80c93e5582531fb23584efaf1840c32018.jpg)

![4547782d8b9533f740e9fa7b75c1ccc2e22b1b1f0821625dc6fd8838f1a55ddf.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/4547782d8b9533f740e9fa7b75c1ccc2e22b1b1f0821625dc6fd8838f1a55ddf.jpg)

![48442ee94aa6b3036c68c810ae980e28b8b10cf128c76b2d7c93b8ade3a8793a.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/48442ee94aa6b3036c68c810ae980e28b8b10cf128c76b2d7c93b8ade3a8793a.jpg)

![4caec090a54835350fbadb1fb0a24ee536bb99b4d5d1cfbd820f38da8ca5a49c.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/4caec090a54835350fbadb1fb0a24ee536bb99b4d5d1cfbd820f38da8ca5a49c.jpg)

![6a0b15606065139964739a82dff14a29cc479428d658d53de5604e79d13d7667.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/6a0b15606065139964739a82dff14a29cc479428d658d53de5604e79d13d7667.jpg)

![8664b7109a9c9fc7829a552addb921eac75b4efb9e5ade6caa78ded51639c795.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/8664b7109a9c9fc7829a552addb921eac75b4efb9e5ade6caa78ded51639c795.jpg)

![93f429596c6483983db5b6d4955085b125c47ca796dee50de8468d96f16d042d.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/93f429596c6483983db5b6d4955085b125c47ca796dee50de8468d96f16d042d.jpg)

![94bcfa67920254217a85d5b5164504a5823dbe50afc17857dcbcfe9d4764a86f.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/94bcfa67920254217a85d5b5164504a5823dbe50afc17857dcbcfe9d4764a86f.jpg)

![961b3c60c632e6493f0f6fb8434bd8949a143da6d6547679345dbf22caac2a55.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/961b3c60c632e6493f0f6fb8434bd8949a143da6d6547679345dbf22caac2a55.jpg)

![a230516012d4dfb936831c9e2b5af8492f62e1789269b71c3c8d95aec1c6e7c4.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/a230516012d4dfb936831c9e2b5af8492f62e1789269b71c3c8d95aec1c6e7c4.jpg)

![aebe0602620ce857b52b3ca72a1a760031256c70ccf802bb4267a3d713cab201.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/aebe0602620ce857b52b3ca72a1a760031256c70ccf802bb4267a3d713cab201.jpg)

![b0a6d6e318d5ba0c4fb7672ff98e31441277484482c99564935d907a83a7b628.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/b0a6d6e318d5ba0c4fb7672ff98e31441277484482c99564935d907a83a7b628.jpg)

![e2cf67b70a5859ee297799c4dfdef60fbbcd1b96672ab7a767833033565d8756.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/e2cf67b70a5859ee297799c4dfdef60fbbcd1b96672ab7a767833033565d8756.jpg)

![fd1a6a58121b60016b53149f301fd90a7cc9083cce9fd2cc05afbcebb75adda0.jpg](../icml_results/1177_Preserve-Then-Quantize_%20Balancing%20Rank%20Budgets%20for%20Quantization%20Error%20Reconstruction%20in%20LLMs/tables/fd1a6a58121b60016b53149f301fd90a7cc9083cce9fd2cc05afbcebb75adda0.jpg)

<a id="paper-1160"></a>

## Bilinear Bandits with Partially Observable Features


### Images

![1c93fca4c750e78efc3ce729abea476ed027c7d808e1018cf44c600d360c72ea.jpg](../icml_results/1178_Bilinear%20Bandits%20with%20Partially%20Observable%20Features/images/1c93fca4c750e78efc3ce729abea476ed027c7d808e1018cf44c600d360c72ea.jpg)

![2277ef1ff1fb6b51169467404837f8aaa06f81efa21b5649b252cb611b3f69f1.jpg](../icml_results/1178_Bilinear%20Bandits%20with%20Partially%20Observable%20Features/images/2277ef1ff1fb6b51169467404837f8aaa06f81efa21b5649b252cb611b3f69f1.jpg)

![301ca31bace827a0b93b161e47184078b7dea5c327fbdf0c75f01789210e52c8.jpg](../icml_results/1178_Bilinear%20Bandits%20with%20Partially%20Observable%20Features/images/301ca31bace827a0b93b161e47184078b7dea5c327fbdf0c75f01789210e52c8.jpg)

![33e50c45ace99748322f6ec31efcdcdd87919963e5afae5efe03707035f7c2f4.jpg](../icml_results/1178_Bilinear%20Bandits%20with%20Partially%20Observable%20Features/images/33e50c45ace99748322f6ec31efcdcdd87919963e5afae5efe03707035f7c2f4.jpg)

![4cecc113f630ba4acbd72849b88e6ae460b802236d39357bab595ddd0d612470.jpg](../icml_results/1178_Bilinear%20Bandits%20with%20Partially%20Observable%20Features/images/4cecc113f630ba4acbd72849b88e6ae460b802236d39357bab595ddd0d612470.jpg)

![a242f759e554b8aa396007efd3b7a2955b9e20cc9a2363f467041a3e944713c8.jpg](../icml_results/1178_Bilinear%20Bandits%20with%20Partially%20Observable%20Features/images/a242f759e554b8aa396007efd3b7a2955b9e20cc9a2363f467041a3e944713c8.jpg)

### Tables

![1d7abc9172f0f8df2018b43415e84861ad5333dd71746e189b9f5ea880687e4e.jpg](../icml_results/1178_Bilinear%20Bandits%20with%20Partially%20Observable%20Features/tables/1d7abc9172f0f8df2018b43415e84861ad5333dd71746e189b9f5ea880687e4e.jpg)

![228bd73f908020971bb458b686ea6cbe73782a378f83762011daad0151df2762.jpg](../icml_results/1178_Bilinear%20Bandits%20with%20Partially%20Observable%20Features/tables/228bd73f908020971bb458b686ea6cbe73782a378f83762011daad0151df2762.jpg)

![6c389e74be0dc22f399b34f1f7515cb42ce492b746cd3cbae4d5d01700d2c860.jpg](../icml_results/1178_Bilinear%20Bandits%20with%20Partially%20Observable%20Features/tables/6c389e74be0dc22f399b34f1f7515cb42ce492b746cd3cbae4d5d01700d2c860.jpg)

<a id="paper-1161"></a>

## CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features


### Images

![007eb2ecc0967ef8565a332779da31623a4a1c6736c63e7d09d963210fc7d969.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/007eb2ecc0967ef8565a332779da31623a4a1c6736c63e7d09d963210fc7d969.jpg)

![046e1ae3a174dd84daa40db031b8eeee5cfbc4900519b8d0ce1c01c8c59d601e.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/046e1ae3a174dd84daa40db031b8eeee5cfbc4900519b8d0ce1c01c8c59d601e.jpg)

![05b0e31f2f9ca754915dd3e36bce89615fd4e743cc6d4980ec1ac0ddd1c89b9e.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/05b0e31f2f9ca754915dd3e36bce89615fd4e743cc6d4980ec1ac0ddd1c89b9e.jpg)

![0d27956753513ea0c1bfee9926bf43b0d5a4620649cdf4bb0106e390a086a67b.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/0d27956753513ea0c1bfee9926bf43b0d5a4620649cdf4bb0106e390a086a67b.jpg)

![100b3857f1aa1f8ac015ae96675e1ccae421659b116436aa38274dfa9bccc765.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/100b3857f1aa1f8ac015ae96675e1ccae421659b116436aa38274dfa9bccc765.jpg)

![145c5c023504c435fd6ff1258f5bb682d74d3b2737e240d7acaee258ee07c6f9.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/145c5c023504c435fd6ff1258f5bb682d74d3b2737e240d7acaee258ee07c6f9.jpg)

![195b2ed247acb0a8798e03c239aeb32da28d6b96d7de81f4ebd94e68c0f2ed33.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/195b2ed247acb0a8798e03c239aeb32da28d6b96d7de81f4ebd94e68c0f2ed33.jpg)

![209a7436ed15e4d6d0f18d7c5716078e0985f996c0bbb172fad47089b5110fb4.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/209a7436ed15e4d6d0f18d7c5716078e0985f996c0bbb172fad47089b5110fb4.jpg)

![3726c119eb8a6f426120ef96ade8141490bc45dc78ea572e7a920a8bb559566f.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/3726c119eb8a6f426120ef96ade8141490bc45dc78ea572e7a920a8bb559566f.jpg)

![3cb889556eb6b47df3e160060675399b8d57741b5a0e4aad5e6c080cfcbd4105.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/3cb889556eb6b47df3e160060675399b8d57741b5a0e4aad5e6c080cfcbd4105.jpg)

![523c4d73481721cead09f19885278c5af736ab2bce9958e8768af9b55fd89f03.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/523c4d73481721cead09f19885278c5af736ab2bce9958e8768af9b55fd89f03.jpg)

![5b2a6ea9dfdfed6a4fc4830b7fbbae3455bf89912da86d9acb91999e6da9e860.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/5b2a6ea9dfdfed6a4fc4830b7fbbae3455bf89912da86d9acb91999e6da9e860.jpg)

![5b80b7ac4b5c8906efdaf35fac8a516f83f3023666e31ae9717c6c1dae26ab05.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/5b80b7ac4b5c8906efdaf35fac8a516f83f3023666e31ae9717c6c1dae26ab05.jpg)

![6bf3a9f725d7b855e457c99cc8c47c3f578b0ad10f2366107330e48cd341b4ac.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/6bf3a9f725d7b855e457c99cc8c47c3f578b0ad10f2366107330e48cd341b4ac.jpg)

![6f48ff0d27d73004904d930c25114582735281853bf1de68727d75d98e15e541.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/6f48ff0d27d73004904d930c25114582735281853bf1de68727d75d98e15e541.jpg)

![8925ebfbe3a4e288033c9a63b7cb3453ae725a043044e67985ea7df7c7f2324b.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/8925ebfbe3a4e288033c9a63b7cb3453ae725a043044e67985ea7df7c7f2324b.jpg)

![92816dd5fff028d749fbc6f5a5861c7c50be48024fe091cce7e81780d812e389.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/92816dd5fff028d749fbc6f5a5861c7c50be48024fe091cce7e81780d812e389.jpg)

![98cd0cd7622faa87ec4926e130726c365892aa9d03d532ad5a264abb3d3d43f4.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/98cd0cd7622faa87ec4926e130726c365892aa9d03d532ad5a264abb3d3d43f4.jpg)

![9c1768b5865b104d85b649cc336beca7a9c2e0a174c9ea83ec854db40d3169d7.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/9c1768b5865b104d85b649cc336beca7a9c2e0a174c9ea83ec854db40d3169d7.jpg)

![a9fa91c79b4c31b7b1e39104c71498c0925c7ae2ee9cf46149f14f11a682b7c9.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/a9fa91c79b4c31b7b1e39104c71498c0925c7ae2ee9cf46149f14f11a682b7c9.jpg)

![b4dd9941a6356962e1fc7bde0ac878a9612063110c04ccd56033fc46e3996a24.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/b4dd9941a6356962e1fc7bde0ac878a9612063110c04ccd56033fc46e3996a24.jpg)

![b5c412c73082731f5b44684c8f45272c43f06672b877f9b6ad42d9e7d9a35d8d.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/b5c412c73082731f5b44684c8f45272c43f06672b877f9b6ad42d9e7d9a35d8d.jpg)

![b6288896cd31179400360a8ce5ab8263210f932661cf411ac0d508ede0b6b8e2.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/b6288896cd31179400360a8ce5ab8263210f932661cf411ac0d508ede0b6b8e2.jpg)

![bb30f016e1221d24ad367092a7749d0df1d8920c50150251f89523657919b7a3.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/bb30f016e1221d24ad367092a7749d0df1d8920c50150251f89523657919b7a3.jpg)

![c384919a31571e61268c7850e5a945c7713297ae9120ac9bea32cca60fb3281e.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/c384919a31571e61268c7850e5a945c7713297ae9120ac9bea32cca60fb3281e.jpg)

![c48c687b4fa3eee4c20e39c77969f83c0aaa7d500aaf082d8fedd4dd0fdc92e7.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/c48c687b4fa3eee4c20e39c77969f83c0aaa7d500aaf082d8fedd4dd0fdc92e7.jpg)

![c7086826164214c79b2ab72ecc20dc22dd0a76f554f53494946f957111768bf4.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/c7086826164214c79b2ab72ecc20dc22dd0a76f554f53494946f957111768bf4.jpg)

![cb1b3624b70aa6054cdc736f123111f611a35c04b1b4d5f3948a1aefc9ed57d7.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/cb1b3624b70aa6054cdc736f123111f611a35c04b1b4d5f3948a1aefc9ed57d7.jpg)

![cd66f9b695e98f4c1a78ba790d5d25a397605f44bcca0de2c525115a2c9756d9.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/cd66f9b695e98f4c1a78ba790d5d25a397605f44bcca0de2c525115a2c9756d9.jpg)

![d015ca73e1d57aed15609d4e6f92d33db3c4d5bd2c55f5afdbc23237fe9bce99.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/d015ca73e1d57aed15609d4e6f92d33db3c4d5bd2c55f5afdbc23237fe9bce99.jpg)

![e05dd991af054761950bc8eaf18375c5ed496cf4bd9c862fd2c6adbc5d111515.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/e05dd991af054761950bc8eaf18375c5ed496cf4bd9c862fd2c6adbc5d111515.jpg)

![e32d2220393059dace7b011528c60da0bb9c278a5aaa774f20f3361678339885.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/e32d2220393059dace7b011528c60da0bb9c278a5aaa774f20f3361678339885.jpg)

![e7e3658c16736eb8c98e584ad3c317212d3a698f98c77158d4ce0fd3ec7f5e96.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/e7e3658c16736eb8c98e584ad3c317212d3a698f98c77158d4ce0fd3ec7f5e96.jpg)

![f52ea4663e0bad1b1cbad2e401bb41e6b69058f2d2dbc887ea06f01337a749cb.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/f52ea4663e0bad1b1cbad2e401bb41e6b69058f2d2dbc887ea06f01337a749cb.jpg)

![ff835c99e2d3e47a197e76a271a69fd9102f7d5f676a8b902fb6a514a6d3eaba.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/images/ff835c99e2d3e47a197e76a271a69fd9102f7d5f676a8b902fb6a514a6d3eaba.jpg)

### Tables

![1e26a977c18b38d636defd961c8aabfc1741986ed7ce6eb29f97295596be90fa.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/tables/1e26a977c18b38d636defd961c8aabfc1741986ed7ce6eb29f97295596be90fa.jpg)

![26d5b5e8c10c9f1b6052b9df0e9ed5adc8a1b935789c064824e54ee5870545b2.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/tables/26d5b5e8c10c9f1b6052b9df0e9ed5adc8a1b935789c064824e54ee5870545b2.jpg)

![281fc868e1c67387ab2a5fe9f620ad83abe97330cf62df422d5b9ca39501bf15.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/tables/281fc868e1c67387ab2a5fe9f620ad83abe97330cf62df422d5b9ca39501bf15.jpg)

![3394e60745d7259632c41addd5fb734437136a011dd1df8bcaa7913356754912.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/tables/3394e60745d7259632c41addd5fb734437136a011dd1df8bcaa7913356754912.jpg)

![3bfa3c4df54c5fea6831a4219e1dc0c2e67315164c418bc596008eee7f772e47.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/tables/3bfa3c4df54c5fea6831a4219e1dc0c2e67315164c418bc596008eee7f772e47.jpg)

![511df95c96524a1ceb8a3d987dca1f78e1229c71d9f65c99991678379a63e8f5.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/tables/511df95c96524a1ceb8a3d987dca1f78e1229c71d9f65c99991678379a63e8f5.jpg)

![80286e0ac1a4d89a393f3a6d42d9330df58e7ad17fef3e310ee4484d4dabe540.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/tables/80286e0ac1a4d89a393f3a6d42d9330df58e7ad17fef3e310ee4484d4dabe540.jpg)

![9d023098a2b4419c0384247bcb2c129808137994458c2e921c6be75d76e1d575.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/tables/9d023098a2b4419c0384247bcb2c129808137994458c2e921c6be75d76e1d575.jpg)

![a71059feaf9c6c60547535545ed20d87dfbac3c6b08f35b165ef501fb6b4396c.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/tables/a71059feaf9c6c60547535545ed20d87dfbac3c6b08f35b165ef501fb6b4396c.jpg)

![b5c8bf48bb2797d0eb1e2c057c0443197f80c0b9be82372322672442ab49548e.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/tables/b5c8bf48bb2797d0eb1e2c057c0443197f80c0b9be82372322672442ab49548e.jpg)

![c87e0102cd4c046fe53e1862deec93ca23aa1be17c21711e67eed91366de1d1a.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/tables/c87e0102cd4c046fe53e1862deec93ca23aa1be17c21711e67eed91366de1d1a.jpg)

![d3c56672d15d5b334ffbd4e9843662f1a313324189a904ce4256561fe35f87f3.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/tables/d3c56672d15d5b334ffbd4e9843662f1a313324189a904ce4256561fe35f87f3.jpg)

![db789b06d8881717b33d0b850c7b37ecfbd6b58b2e1709c31398e0ab8cf1bfc5.jpg](../icml_results/1179_CorrSteer_%20Generation-Time%20LLM%20Steering%20via%20Correlated%20Sparse%20Autoencoder%20Features/tables/db789b06d8881717b33d0b850c7b37ecfbd6b58b2e1709c31398e0ab8cf1bfc5.jpg)

<a id="paper-1162"></a>

## Expert-guided Clinical Text Augmentation via Query-Based Model Collaboration


### Images

![964b53af892aa3d43d14cc5538a99faf8bca996b8df4bd790912bc7b33684b5d.jpg](../icml_results/1180_Expert-guided%20Clinical%20Text%20Augmentation%20via%20Query-Based%20Model%20Collaboration/images/964b53af892aa3d43d14cc5538a99faf8bca996b8df4bd790912bc7b33684b5d.jpg)

![a9db9652fb5378fdd341df0eb6b9c25145bf8546ca7b93aa353c217ea47e745c.jpg](../icml_results/1180_Expert-guided%20Clinical%20Text%20Augmentation%20via%20Query-Based%20Model%20Collaboration/images/a9db9652fb5378fdd341df0eb6b9c25145bf8546ca7b93aa353c217ea47e745c.jpg)

![c0a28e0f62305886cd7d522a52eaed7f240f7c94e0bc2962a95a69f74e1b6587.jpg](../icml_results/1180_Expert-guided%20Clinical%20Text%20Augmentation%20via%20Query-Based%20Model%20Collaboration/images/c0a28e0f62305886cd7d522a52eaed7f240f7c94e0bc2962a95a69f74e1b6587.jpg)

![f4fc6a9156c8e9f1b2f286a825b5f3f4097ce8df8f177a3fdd5577c1524ae182.jpg](../icml_results/1180_Expert-guided%20Clinical%20Text%20Augmentation%20via%20Query-Based%20Model%20Collaboration/images/f4fc6a9156c8e9f1b2f286a825b5f3f4097ce8df8f177a3fdd5577c1524ae182.jpg)

![f9dcddf2dd200463732641892ed771d75b99c6849275ad36f6760993eceafddf.jpg](../icml_results/1180_Expert-guided%20Clinical%20Text%20Augmentation%20via%20Query-Based%20Model%20Collaboration/images/f9dcddf2dd200463732641892ed771d75b99c6849275ad36f6760993eceafddf.jpg)

### Tables

![0f97486667d17112c693cf1900a25ede323826957e43bbbc54b93fa1b86b7295.jpg](../icml_results/1180_Expert-guided%20Clinical%20Text%20Augmentation%20via%20Query-Based%20Model%20Collaboration/tables/0f97486667d17112c693cf1900a25ede323826957e43bbbc54b93fa1b86b7295.jpg)

![11d2f4106d7b99883d163dbe3224f699123c592862078562bba92618aec277bf.jpg](../icml_results/1180_Expert-guided%20Clinical%20Text%20Augmentation%20via%20Query-Based%20Model%20Collaboration/tables/11d2f4106d7b99883d163dbe3224f699123c592862078562bba92618aec277bf.jpg)

![286276fc7c34b7f214ee21c42d7ac70a0ce0b7b9ba94a0b3144138b721cea816.jpg](../icml_results/1180_Expert-guided%20Clinical%20Text%20Augmentation%20via%20Query-Based%20Model%20Collaboration/tables/286276fc7c34b7f214ee21c42d7ac70a0ce0b7b9ba94a0b3144138b721cea816.jpg)

![44645a420b35ff83fa715b313ce4849fd79566534727fa4e4311786b9e46ff7b.jpg](../icml_results/1180_Expert-guided%20Clinical%20Text%20Augmentation%20via%20Query-Based%20Model%20Collaboration/tables/44645a420b35ff83fa715b313ce4849fd79566534727fa4e4311786b9e46ff7b.jpg)

![4ef859017e82d97e1b908f0e8187cbc796cfc02a94508148dc9fb79e6ee7ec81.jpg](../icml_results/1180_Expert-guided%20Clinical%20Text%20Augmentation%20via%20Query-Based%20Model%20Collaboration/tables/4ef859017e82d97e1b908f0e8187cbc796cfc02a94508148dc9fb79e6ee7ec81.jpg)

![5ddafceb8fc8edfcc7c7d390767c80c121030db8b37ba4a52ecbd64bff422040.jpg](../icml_results/1180_Expert-guided%20Clinical%20Text%20Augmentation%20via%20Query-Based%20Model%20Collaboration/tables/5ddafceb8fc8edfcc7c7d390767c80c121030db8b37ba4a52ecbd64bff422040.jpg)

![804670e6dee6cc1fb89636ccd23414f66328b444568125a23a7f215b764b85d2.jpg](../icml_results/1180_Expert-guided%20Clinical%20Text%20Augmentation%20via%20Query-Based%20Model%20Collaboration/tables/804670e6dee6cc1fb89636ccd23414f66328b444568125a23a7f215b764b85d2.jpg)

![9649e0fde2f1925f0a2fe68cbf818a4b805bdfd04cc338c6d8fa31518564bfd9.jpg](../icml_results/1180_Expert-guided%20Clinical%20Text%20Augmentation%20via%20Query-Based%20Model%20Collaboration/tables/9649e0fde2f1925f0a2fe68cbf818a4b805bdfd04cc338c6d8fa31518564bfd9.jpg)

![a96b25cc857ae89f27e3e384045292fc4ddca303d86d34e2d86203b74c0f77fd.jpg](../icml_results/1180_Expert-guided%20Clinical%20Text%20Augmentation%20via%20Query-Based%20Model%20Collaboration/tables/a96b25cc857ae89f27e3e384045292fc4ddca303d86d34e2d86203b74c0f77fd.jpg)

![cb791e36caab5b00a82c078dab6a714ff54a1d074a9103f5aacafaa0afa28e93.jpg](../icml_results/1180_Expert-guided%20Clinical%20Text%20Augmentation%20via%20Query-Based%20Model%20Collaboration/tables/cb791e36caab5b00a82c078dab6a714ff54a1d074a9103f5aacafaa0afa28e93.jpg)

![d0870f589601ea7cd3d467f65bf515b170c3ff95b2f2e61efa0aab87257e956e.jpg](../icml_results/1180_Expert-guided%20Clinical%20Text%20Augmentation%20via%20Query-Based%20Model%20Collaboration/tables/d0870f589601ea7cd3d467f65bf515b170c3ff95b2f2e61efa0aab87257e956e.jpg)

![ddb9333a65e98a69bf5048bb400e3730cbdbb105c5af82115497936e0efd802b.jpg](../icml_results/1180_Expert-guided%20Clinical%20Text%20Augmentation%20via%20Query-Based%20Model%20Collaboration/tables/ddb9333a65e98a69bf5048bb400e3730cbdbb105c5af82115497936e0efd802b.jpg)

<a id="paper-1163"></a>

## PinTok: Tokenizers Deserve Dedicated Pinned CPU-Compute and Memory


### Images

![3f1d1ace44aaf4c8aa2e8dec5de33ef7d4a27fba358b0fd518ebb8e7879f2bc8.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/images/3f1d1ace44aaf4c8aa2e8dec5de33ef7d4a27fba358b0fd518ebb8e7879f2bc8.jpg)

![4f4c467759aa4cacaf5da47729587c3de7cb29d403779206a58211cf616f491e.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/images/4f4c467759aa4cacaf5da47729587c3de7cb29d403779206a58211cf616f491e.jpg)

![5b9c5deecd8e0226bfc0205506fce742e3a7468d1a8090050787703bf12d5d1e.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/images/5b9c5deecd8e0226bfc0205506fce742e3a7468d1a8090050787703bf12d5d1e.jpg)

![5ce186f1eb897e1c9a40b19d9aa4a214f7f41bc490994a266feba4aeb0c7c737.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/images/5ce186f1eb897e1c9a40b19d9aa4a214f7f41bc490994a266feba4aeb0c7c737.jpg)

![653bbc44aedfe92dc94018464863be357beb606f1f2e8437d83387725b840a6f.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/images/653bbc44aedfe92dc94018464863be357beb606f1f2e8437d83387725b840a6f.jpg)

![a54d46d0fb55314df718a6d6d2c563adf716fa93eb58064dcf90c2b5a76b98c9.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/images/a54d46d0fb55314df718a6d6d2c563adf716fa93eb58064dcf90c2b5a76b98c9.jpg)

![b745b00789ad78dd8181e8e7316ed9a38e46b12ef0be355403b4c8c04635a1ff.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/images/b745b00789ad78dd8181e8e7316ed9a38e46b12ef0be355403b4c8c04635a1ff.jpg)

![df11e7f1bd30981f2d1061613feab730e275178734f239c21289031f14b21fd2.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/images/df11e7f1bd30981f2d1061613feab730e275178734f239c21289031f14b21fd2.jpg)

![f88cc0e646f973e68e09aa8835ec7bf7971121fc95a6c7f0ed48bafbb0f9a835.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/images/f88cc0e646f973e68e09aa8835ec7bf7971121fc95a6c7f0ed48bafbb0f9a835.jpg)

### Tables

![003f082df8b08f7c81fb99d7197b85ea0adda20f335007f14fa5a5d5c6985399.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/tables/003f082df8b08f7c81fb99d7197b85ea0adda20f335007f14fa5a5d5c6985399.jpg)

![008073b2d41dcb14ffdc7af2800f41327c5dd404e343c705f49b7a9c29644a23.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/tables/008073b2d41dcb14ffdc7af2800f41327c5dd404e343c705f49b7a9c29644a23.jpg)

![0e36788608e49e32838b6dbc42aa8fc2cdca4f1a11c61e4c9ccab28efd99b74f.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/tables/0e36788608e49e32838b6dbc42aa8fc2cdca4f1a11c61e4c9ccab28efd99b74f.jpg)

![1b99f8e781921209a613daa8692d9655cfd322c2c881a2e4196dfbeb63ace9e4.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/tables/1b99f8e781921209a613daa8692d9655cfd322c2c881a2e4196dfbeb63ace9e4.jpg)

![1c036512f56a024d285d9ecd773ac07951cf503b4a73b0ce91a248e5f7654816.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/tables/1c036512f56a024d285d9ecd773ac07951cf503b4a73b0ce91a248e5f7654816.jpg)

![2b517e006a9c46141cbde4398c6e3b1ffe38e94b1bdd3be44b22b82793d2dc41.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/tables/2b517e006a9c46141cbde4398c6e3b1ffe38e94b1bdd3be44b22b82793d2dc41.jpg)

![522241e0bab013151425e01489a7d4f590adb8ef8b670c499e46abba2a7d1d62.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/tables/522241e0bab013151425e01489a7d4f590adb8ef8b670c499e46abba2a7d1d62.jpg)

![5c6e33fe213fbb1d0441d956c24f38b7aad49772993f61a88e5c86b4ba84bbd0.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/tables/5c6e33fe213fbb1d0441d956c24f38b7aad49772993f61a88e5c86b4ba84bbd0.jpg)

![67c021a3e93454a67fe1b4be5b30bd69f65aa1ee354326fe31893a807553aa77.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/tables/67c021a3e93454a67fe1b4be5b30bd69f65aa1ee354326fe31893a807553aa77.jpg)

![81460daffb386a71d4a0b8728347e349ddee76b3416e2542634184165f96b66b.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/tables/81460daffb386a71d4a0b8728347e349ddee76b3416e2542634184165f96b66b.jpg)

![d037d798cf0f368fe80590ba36c5400cc8ded949608d275350dc47bff841c3d0.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/tables/d037d798cf0f368fe80590ba36c5400cc8ded949608d275350dc47bff841c3d0.jpg)

![d93aa9f59c22641b8c340e86510f4c134ede3afdd11d1a2d1da9eea6cacc0424.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/tables/d93aa9f59c22641b8c340e86510f4c134ede3afdd11d1a2d1da9eea6cacc0424.jpg)

![eb03b243ac94e2d8ac5434fe299d0e107984e58c13db6ec904fc91fda5e78f53.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/tables/eb03b243ac94e2d8ac5434fe299d0e107984e58c13db6ec904fc91fda5e78f53.jpg)

![f218417039f51405506d40ed3524a034a7e075e25c150a262fb59df055a6b82b.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/tables/f218417039f51405506d40ed3524a034a7e075e25c150a262fb59df055a6b82b.jpg)

![f590b571ac64e60158248ca63f8ccb0d1fde75b2c38a694e22fa997d3c46437d.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/tables/f590b571ac64e60158248ca63f8ccb0d1fde75b2c38a694e22fa997d3c46437d.jpg)

![fd784c41f3f205d505c9373967d79fe10883142d4e7687b3d598247e8863927c.jpg](../icml_results/1181_PinTok_%20Tokenizers%20Deserve%20Dedicated%20Pinned%20CPU-Compute%20and%20Memory/tables/fd784c41f3f205d505c9373967d79fe10883142d4e7687b3d598247e8863927c.jpg)

<a id="paper-1164"></a>

## Efficient Skill Grounding via Code Refactoring with Small Language Models


### Images

![0395d9bd18f042efb80c5ecc10665329257c1bfedcf03772f83fbe4fe881268e.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/images/0395d9bd18f042efb80c5ecc10665329257c1bfedcf03772f83fbe4fe881268e.jpg)

![12b73a9337cb9fa2ec98e5b32d38358b829f87b2e0b25ef3ba031b8d8b8bedc7.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/images/12b73a9337cb9fa2ec98e5b32d38358b829f87b2e0b25ef3ba031b8d8b8bedc7.jpg)

![1d008593184c5144f3719c8a2f2881adbd9c99a7e1b26247880c015e955fcd02.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/images/1d008593184c5144f3719c8a2f2881adbd9c99a7e1b26247880c015e955fcd02.jpg)

![4f166ee68ee8a261d046c7b1a89150fc5c2e6d0858c367385dbf0b6a2b650f76.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/images/4f166ee68ee8a261d046c7b1a89150fc5c2e6d0858c367385dbf0b6a2b650f76.jpg)

![6a6fd35bf741bc3291557781508cd7d402e277d44a5ae7040ccf04bef7ae80ee.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/images/6a6fd35bf741bc3291557781508cd7d402e277d44a5ae7040ccf04bef7ae80ee.jpg)

![8facd178e0dffbbcc68f47aafe25b470c9b3f20dac96f2eaed7d5a89998c6e88.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/images/8facd178e0dffbbcc68f47aafe25b470c9b3f20dac96f2eaed7d5a89998c6e88.jpg)

![9678c70067e2e23dfdcb133e568019da6e08ca6e0201264a655d7d874738e91e.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/images/9678c70067e2e23dfdcb133e568019da6e08ca6e0201264a655d7d874738e91e.jpg)

![d410e23cf027dfee1dc047cd304c0b09c8589a39cc9c171c76f5a9e4cb6070bd.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/images/d410e23cf027dfee1dc047cd304c0b09c8589a39cc9c171c76f5a9e4cb6070bd.jpg)

![dfa6aee9d35c593fe55e85479edc10fab9093e9de2528211ce652c6e7c2b0d45.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/images/dfa6aee9d35c593fe55e85479edc10fab9093e9de2528211ce652c6e7c2b0d45.jpg)

![e92b62256d4bda953cb41c6b84479f22c7866f15004bef2f40e676686651240f.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/images/e92b62256d4bda953cb41c6b84479f22c7866f15004bef2f40e676686651240f.jpg)

![f20e14ff0e1ad18fedf81dc32279d1020aafab6e3af6fc4269849be28b1d942b.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/images/f20e14ff0e1ad18fedf81dc32279d1020aafab6e3af6fc4269849be28b1d942b.jpg)

### Tables

![031ce7f4dc3cc35ce4c74cc4ae1015bac48b0d20cc03dba572a8d4bba1a8c6b4.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/031ce7f4dc3cc35ce4c74cc4ae1015bac48b0d20cc03dba572a8d4bba1a8c6b4.jpg)

![0f59951383c5256f6dc105d293fe5f7c5b91a17006dfb4f6db10fd070b18067b.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/0f59951383c5256f6dc105d293fe5f7c5b91a17006dfb4f6db10fd070b18067b.jpg)

![0f5f86618947ed1a314b815170b02df67de4642a87aa7d37878a99b17006540c.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/0f5f86618947ed1a314b815170b02df67de4642a87aa7d37878a99b17006540c.jpg)

![2dbf106e817c683fcdeae4bc3531edf2ab13974483d4d35fd1fb21cae71c4375.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/2dbf106e817c683fcdeae4bc3531edf2ab13974483d4d35fd1fb21cae71c4375.jpg)

![39060a0855869a238469796a9bdeaf35f011abf68d441b71047417df9c28bd29.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/39060a0855869a238469796a9bdeaf35f011abf68d441b71047417df9c28bd29.jpg)

![39e0c1d5f0587db944ac3556afa5f00f6c84caebdbdb9db8585df2575ed22b92.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/39e0c1d5f0587db944ac3556afa5f00f6c84caebdbdb9db8585df2575ed22b92.jpg)

![3b27568ea88b642c52a5aca38066929f918de97e436370172f7c07946ffb88f0.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/3b27568ea88b642c52a5aca38066929f918de97e436370172f7c07946ffb88f0.jpg)

![421140de6b21b36d246afc8b6b8b99a4484a06477d5d0471e2ac3f8616a99ee0.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/421140de6b21b36d246afc8b6b8b99a4484a06477d5d0471e2ac3f8616a99ee0.jpg)

![5758a117d10df468d7caf65e93fd30e5087017cd300d7adf2f3b133483873dec.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/5758a117d10df468d7caf65e93fd30e5087017cd300d7adf2f3b133483873dec.jpg)

![66b93b004091676a1bfc2f6ec0d7e5f9d3d308caec03ab02f8e27e2bb478194a.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/66b93b004091676a1bfc2f6ec0d7e5f9d3d308caec03ab02f8e27e2bb478194a.jpg)

![803b443b3d3695fe07d6968250c54219fd2898cd0364749e31e89812e7ae81d2.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/803b443b3d3695fe07d6968250c54219fd2898cd0364749e31e89812e7ae81d2.jpg)

![814a1bc195c2cfc7d901ba0c43270affdf41053af05d9223cb7e3b8ec43a7d2e.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/814a1bc195c2cfc7d901ba0c43270affdf41053af05d9223cb7e3b8ec43a7d2e.jpg)

![82cb1ebfaa473568c466a611976ad0d922e90cc2155db203b21437aaeb1f8948.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/82cb1ebfaa473568c466a611976ad0d922e90cc2155db203b21437aaeb1f8948.jpg)

![8b982d7e5fc31d1b7aa122e1fbdcffc12b36b3e2a02715a6b15a2560ddc73256.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/8b982d7e5fc31d1b7aa122e1fbdcffc12b36b3e2a02715a6b15a2560ddc73256.jpg)

![8f1753f4946544a88989378437bcfe17f93abbf99e40190e4b56695b80cf9047.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/8f1753f4946544a88989378437bcfe17f93abbf99e40190e4b56695b80cf9047.jpg)

![9c0d60e2c0a40505be74cf826cb9c847794a33b1a3ce973edf6a41d772150242.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/9c0d60e2c0a40505be74cf826cb9c847794a33b1a3ce973edf6a41d772150242.jpg)

![a41c7612c345d29a03e0d8266e449b7125f1daa972f784139250bcce74700cf6.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/a41c7612c345d29a03e0d8266e449b7125f1daa972f784139250bcce74700cf6.jpg)

![a99a568fd84620fda5b5048728afc6d091ce2f305ef18412da0d004a79579b05.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/a99a568fd84620fda5b5048728afc6d091ce2f305ef18412da0d004a79579b05.jpg)

![ac3c3d7e68bad13b7349f279f9360e51ae5da274fddf20f5718d55ab3b53b709.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/ac3c3d7e68bad13b7349f279f9360e51ae5da274fddf20f5718d55ab3b53b709.jpg)

![c631d8f045a700447df477b19888c007b8a12030441780b88fa8086e00df5d05.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/c631d8f045a700447df477b19888c007b8a12030441780b88fa8086e00df5d05.jpg)

![d3d282b7d8c26297af255f3385fa0b9d1bccb1c5cb2c81d699ebd29bc44deb69.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/d3d282b7d8c26297af255f3385fa0b9d1bccb1c5cb2c81d699ebd29bc44deb69.jpg)

![e9b07b717e01b07532b3a3162acdc2c0e47b98169d4957aaff521c9ff9dc243b.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/e9b07b717e01b07532b3a3162acdc2c0e47b98169d4957aaff521c9ff9dc243b.jpg)

![ed79743d483a8fd28dd78036fe0adc170714065e7d55839f7daf685bd086ab05.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/ed79743d483a8fd28dd78036fe0adc170714065e7d55839f7daf685bd086ab05.jpg)

![f9928255cbe368e0476b12a32c0a11b7fbbe908d8864ff3ddf4e4eda6b88e6f3.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/f9928255cbe368e0476b12a32c0a11b7fbbe908d8864ff3ddf4e4eda6b88e6f3.jpg)

![ffc9a2f3e97b83ecaa187b0988562c79e34b6e29bfc6afd07df13c6680996f96.jpg](../icml_results/1182_Efficient%20Skill%20Grounding%20via%20Code%20Refactoring%20with%20Small%20Language%20Models/tables/ffc9a2f3e97b83ecaa187b0988562c79e34b6e29bfc6afd07df13c6680996f96.jpg)

<a id="paper-1165"></a>

## Gauge-Equivariant Graph Networks via Self-Interference Cancellation


### Images

![08cd9b42644dc829c30f436d666d2c5970287479546e80a2f46c02654a1604ea.jpg](../icml_results/1183_Gauge-Equivariant%20Graph%20Networks%20via%20Self-Interference%20Cancellation/images/08cd9b42644dc829c30f436d666d2c5970287479546e80a2f46c02654a1604ea.jpg)

![0d7528cd096c0af84895086d6f94dd0a50f8ad2f5f3a769dd53d7a8b34420546.jpg](../icml_results/1183_Gauge-Equivariant%20Graph%20Networks%20via%20Self-Interference%20Cancellation/images/0d7528cd096c0af84895086d6f94dd0a50f8ad2f5f3a769dd53d7a8b34420546.jpg)

![fd59687ea671e988cd0e6a9c50365296b66abb31892f99bcf7080842d960740c.jpg](../icml_results/1183_Gauge-Equivariant%20Graph%20Networks%20via%20Self-Interference%20Cancellation/images/fd59687ea671e988cd0e6a9c50365296b66abb31892f99bcf7080842d960740c.jpg)

### Tables

![016c957e204e9948e6589932bf51ecca93693c6edb5713c2396873d4526259e6.jpg](../icml_results/1183_Gauge-Equivariant%20Graph%20Networks%20via%20Self-Interference%20Cancellation/tables/016c957e204e9948e6589932bf51ecca93693c6edb5713c2396873d4526259e6.jpg)

![74116c07264ea2fa3bd94b77b745a335efaace40a310706e4a4f419661904fb8.jpg](../icml_results/1183_Gauge-Equivariant%20Graph%20Networks%20via%20Self-Interference%20Cancellation/tables/74116c07264ea2fa3bd94b77b745a335efaace40a310706e4a4f419661904fb8.jpg)

![8b48e705d57a056e2c524c872816b469872f3c06afcda10b7c79590aef5c0e36.jpg](../icml_results/1183_Gauge-Equivariant%20Graph%20Networks%20via%20Self-Interference%20Cancellation/tables/8b48e705d57a056e2c524c872816b469872f3c06afcda10b7c79590aef5c0e36.jpg)

![943cbd626583d479f1c9157d6e2c16d82ed26a65d43ebbe72c6d62477c272315.jpg](../icml_results/1183_Gauge-Equivariant%20Graph%20Networks%20via%20Self-Interference%20Cancellation/tables/943cbd626583d479f1c9157d6e2c16d82ed26a65d43ebbe72c6d62477c272315.jpg)

<a id="paper-1166"></a>

## The Truth Stays in the Family: Enhancing Contextual Truthfulness via Inherited Heads in Model Lineages


### Images

![262e2e6fbd87b87405bfdcba92622747587a632a551e36ecbc7d40ddb928558e.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/images/262e2e6fbd87b87405bfdcba92622747587a632a551e36ecbc7d40ddb928558e.jpg)

![4388961933b988dd1682bc795619bcab7bcf342a6ffab384df50d53abb0f0c11.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/images/4388961933b988dd1682bc795619bcab7bcf342a6ffab384df50d53abb0f0c11.jpg)

![6e66913394f5ce689e085fd90beade67265b7e963596dd3f4144cd9dee660216.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/images/6e66913394f5ce689e085fd90beade67265b7e963596dd3f4144cd9dee660216.jpg)

![6f2df271a709e0f4b90b6411f41ef72785219bb32665acdced44a42dccebffdd.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/images/6f2df271a709e0f4b90b6411f41ef72785219bb32665acdced44a42dccebffdd.jpg)

![9e7e238ab874a6a8ba72ff37d51d1ec27a49f11998436b324c385f7e199aeccf.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/images/9e7e238ab874a6a8ba72ff37d51d1ec27a49f11998436b324c385f7e199aeccf.jpg)

![ad52fa06eca841b16fa82ea87308332113674ff6fadac0c1956f23e53bceeaf7.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/images/ad52fa06eca841b16fa82ea87308332113674ff6fadac0c1956f23e53bceeaf7.jpg)

![c4584201bc8906a0d2b5b517a8906a7b4b88d5de866d2978728420e6781c66ca.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/images/c4584201bc8906a0d2b5b517a8906a7b4b88d5de866d2978728420e6781c66ca.jpg)

![c5decdd953d5bbad90c01db9214373faf9790be18e774ac79436255c9e65a09d.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/images/c5decdd953d5bbad90c01db9214373faf9790be18e774ac79436255c9e65a09d.jpg)

![c70da7a5d05024ee34f23b0e5f91d0547f4edb5f39e836dd5f9e81f3689ad38e.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/images/c70da7a5d05024ee34f23b0e5f91d0547f4edb5f39e836dd5f9e81f3689ad38e.jpg)

![d98baa0e49a7e108babf5674441203e447820adcab1680d3279f757a61d774d9.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/images/d98baa0e49a7e108babf5674441203e447820adcab1680d3279f757a61d774d9.jpg)

![d9d6a52528403b67410e614b4a860a7c3e2c8f2c965d033ef4e3ca3161837c8e.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/images/d9d6a52528403b67410e614b4a860a7c3e2c8f2c965d033ef4e3ca3161837c8e.jpg)

### Tables

![083fc12173843c7ad6de354770abb12229316096be0c0cb097a4e57059b45685.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/083fc12173843c7ad6de354770abb12229316096be0c0cb097a4e57059b45685.jpg)

![16819f874722d7bbdcd456fac0b82c72775d93dff24a5a8306ea06ae06f8fc24.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/16819f874722d7bbdcd456fac0b82c72775d93dff24a5a8306ea06ae06f8fc24.jpg)

![17885281403318e28484114f76d1b593f4e2dab37fbea4bc5e7abdadb9d0d760.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/17885281403318e28484114f76d1b593f4e2dab37fbea4bc5e7abdadb9d0d760.jpg)

![18d168f30d731afeccf1db3a9b391598e57e98ce66fe63504fb8d00b5aa497c9.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/18d168f30d731afeccf1db3a9b391598e57e98ce66fe63504fb8d00b5aa497c9.jpg)

![1ba79dadb36bbe1cc1950d9b5fcac58bfa005f7c39f2fbd7ee23a899e5332f7e.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/1ba79dadb36bbe1cc1950d9b5fcac58bfa005f7c39f2fbd7ee23a899e5332f7e.jpg)

![43adbf514a5ac014ae2ca52671cca453a59cedaf14dac88bc72c46241f385378.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/43adbf514a5ac014ae2ca52671cca453a59cedaf14dac88bc72c46241f385378.jpg)

![51e762d6168b7f1e1d9c949781f0f4863e585810857a3149d7f0b915dd840496.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/51e762d6168b7f1e1d9c949781f0f4863e585810857a3149d7f0b915dd840496.jpg)

![5972f4d9e7bd29e1ed6770889c5823639fbbea6b63f4be8cb46b55967ca08269.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/5972f4d9e7bd29e1ed6770889c5823639fbbea6b63f4be8cb46b55967ca08269.jpg)

![666d146e930b435f376e8776e186192c50f2d9820ce90b12c7ca7a1dea755ad6.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/666d146e930b435f376e8776e186192c50f2d9820ce90b12c7ca7a1dea755ad6.jpg)

![6fbb5220a99cf12f25fb687286108d1080cc318d667716b54ac4d18a0bdd99b7.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/6fbb5220a99cf12f25fb687286108d1080cc318d667716b54ac4d18a0bdd99b7.jpg)

![70fbd30233e392ff408a758de5340482663c16e38abaa057916cdc29f3d81711.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/70fbd30233e392ff408a758de5340482663c16e38abaa057916cdc29f3d81711.jpg)

![73deb4acae76c2e2c82abce7ced2c00e84a751076219bc4d7f2932dce8f2f122.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/73deb4acae76c2e2c82abce7ced2c00e84a751076219bc4d7f2932dce8f2f122.jpg)

![770ec5649dc356cd682af6ff240657fc0497e1620f4af8dced48c630378e76b2.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/770ec5649dc356cd682af6ff240657fc0497e1620f4af8dced48c630378e76b2.jpg)

![8667aa241cb60fb2111dfb414461b3acc2f1b7af193e49d8810e86073342cad7.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/8667aa241cb60fb2111dfb414461b3acc2f1b7af193e49d8810e86073342cad7.jpg)

![8d2f7f3d8af2d46624026860d82f86c7acfc573a642353ea19a8f75108672d35.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/8d2f7f3d8af2d46624026860d82f86c7acfc573a642353ea19a8f75108672d35.jpg)

![8ef57d00b95ab8655a1ed2b6407ed9445657d63a3186f5ec9380d243db1ae9e2.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/8ef57d00b95ab8655a1ed2b6407ed9445657d63a3186f5ec9380d243db1ae9e2.jpg)

![9125d52a6f6654b7e564c857c41eb2d60cbb102402bdb148c1f787cc1afc54f2.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/9125d52a6f6654b7e564c857c41eb2d60cbb102402bdb148c1f787cc1afc54f2.jpg)

![9ea53b76bbddbeb6cea54ffec5ff1b0b73b846cda9e8ae8b992b51fb9b57c749.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/9ea53b76bbddbeb6cea54ffec5ff1b0b73b846cda9e8ae8b992b51fb9b57c749.jpg)

![b494988830ee94cf9e81b5cb1f10f63ad23c858adcef7e92bb61af9ab2d77d65.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/b494988830ee94cf9e81b5cb1f10f63ad23c858adcef7e92bb61af9ab2d77d65.jpg)

![d3c64eeb8bb447753ef0c4d63f3b87ca026d02255d7683698e3fcb2b62ce00d0.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/d3c64eeb8bb447753ef0c4d63f3b87ca026d02255d7683698e3fcb2b62ce00d0.jpg)

![de0000c6845b8695202348fddaf799037e2c105934ea462b42dff09fcf641864.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/de0000c6845b8695202348fddaf799037e2c105934ea462b42dff09fcf641864.jpg)

![fd6f3e8015159bcca0e29ac5ce688808a91ecc8906088d04687b90cfcce443cc.jpg](../icml_results/1184_The%20Truth%20Stays%20in%20the%20Family_%20Enhancing%20Contextual%20Truthfulness%20via%20Inherited%20Heads%20in%20Model%20Lineag/tables/fd6f3e8015159bcca0e29ac5ce688808a91ecc8906088d04687b90cfcce443cc.jpg)

<a id="paper-1167"></a>

## Anchored Self-Play for Code Repair


### Images

![569d48dfa4d6b20afe5e665dd87379c2d24ae13fd5f5817039b00698cc905d74.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/images/569d48dfa4d6b20afe5e665dd87379c2d24ae13fd5f5817039b00698cc905d74.jpg)

![59d4f046859808fce7c59783071d9e204dbd77f50a6688121593c87eeedf024b.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/images/59d4f046859808fce7c59783071d9e204dbd77f50a6688121593c87eeedf024b.jpg)

![6215d13acad3247872cc998f9e2e37d18f159b468e70d8e229ac0b3e20e45bd2.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/images/6215d13acad3247872cc998f9e2e37d18f159b468e70d8e229ac0b3e20e45bd2.jpg)

![72107c68d416899a7b09e9b13c3bd1f24bd80c855f967a30f73571180f513032.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/images/72107c68d416899a7b09e9b13c3bd1f24bd80c855f967a30f73571180f513032.jpg)

![8fbdead9e7778da1c90228d8f7f67c18dcee8bb4fddd9d13e1fad2c5af70ef7e.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/images/8fbdead9e7778da1c90228d8f7f67c18dcee8bb4fddd9d13e1fad2c5af70ef7e.jpg)

![c5f33710177082ac6d056e280a751e4be5e7ae9f0f6c04f6cca634eec834d850.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/images/c5f33710177082ac6d056e280a751e4be5e7ae9f0f6c04f6cca634eec834d850.jpg)

![c97ed5066e81c56627e2eec899816ae3c828d136c6a939718f535e70cc4fcf85.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/images/c97ed5066e81c56627e2eec899816ae3c828d136c6a939718f535e70cc4fcf85.jpg)

![cd68d397ad82c5bb7e23eab13f2f4fee65600449134dbdb952dcc2ce273c7813.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/images/cd68d397ad82c5bb7e23eab13f2f4fee65600449134dbdb952dcc2ce273c7813.jpg)

![e4ae91234ae773dfcdcf22b1cf047a272465bc41aa3170668dcc1fb1933ea5e7.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/images/e4ae91234ae773dfcdcf22b1cf047a272465bc41aa3170668dcc1fb1933ea5e7.jpg)

![f483e4eab03bda2692c3627d9268b50fe5b0eaf597b4ecae109ff64dc4859919.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/images/f483e4eab03bda2692c3627d9268b50fe5b0eaf597b4ecae109ff64dc4859919.jpg)

![f9c76f1a4382e4ed5d74f1b33d6d1a89467fcc854760032fab42936c6039ab55.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/images/f9c76f1a4382e4ed5d74f1b33d6d1a89467fcc854760032fab42936c6039ab55.jpg)

### Tables

![000788a2a114cf1caebc1d53a22da616869d0bfd96165c62c19a8951120d0bfb.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/tables/000788a2a114cf1caebc1d53a22da616869d0bfd96165c62c19a8951120d0bfb.jpg)

![0a1226b870e795313f046c763ba87fbc6ffca47469007349f5f0281f1e4562d2.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/tables/0a1226b870e795313f046c763ba87fbc6ffca47469007349f5f0281f1e4562d2.jpg)

![189533d4f7ef9676d66f8e47f2411f74f55d7ec46ed42d51aa535d1c77a50397.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/tables/189533d4f7ef9676d66f8e47f2411f74f55d7ec46ed42d51aa535d1c77a50397.jpg)

![18c200cd15d776139818fc664e9a5a4205fc9597735b552e7e01b756c185a536.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/tables/18c200cd15d776139818fc664e9a5a4205fc9597735b552e7e01b756c185a536.jpg)

![1b87bd625ae284020155eee63b501cc7036608f0f7c36df1428a476a7dd551c4.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/tables/1b87bd625ae284020155eee63b501cc7036608f0f7c36df1428a476a7dd551c4.jpg)

![293859fb9c46c6429bee4966ca929913745942960e0500016184056732dc5e29.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/tables/293859fb9c46c6429bee4966ca929913745942960e0500016184056732dc5e29.jpg)

![2f127a984fd62aca329dba0bc6603709680745cfd307066c60c97b630db0631d.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/tables/2f127a984fd62aca329dba0bc6603709680745cfd307066c60c97b630db0631d.jpg)

![9200ba5664c76b6eb5d9d965947168a6b63e2828be27d7ee7832130f5f793e3e.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/tables/9200ba5664c76b6eb5d9d965947168a6b63e2828be27d7ee7832130f5f793e3e.jpg)

![c3d84a87433a946094539b4b46c23e6c66baf958e6ce6832d4cd7b8f27ec9a34.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/tables/c3d84a87433a946094539b4b46c23e6c66baf958e6ce6832d4cd7b8f27ec9a34.jpg)

![ce3f6abea3d75e67c1e16b460a6b92dd579b4741a9a715641bb99f6bd2fcd03d.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/tables/ce3f6abea3d75e67c1e16b460a6b92dd579b4741a9a715641bb99f6bd2fcd03d.jpg)

![cf647851cd456e8989ae1b871dd7f4921b0d025b81e201b0bfeea053193c63ab.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/tables/cf647851cd456e8989ae1b871dd7f4921b0d025b81e201b0bfeea053193c63ab.jpg)

![cf86eb9d26bad71fbfff7a5654ea9c1dc4e971426a6217149cd503fd22ed44c4.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/tables/cf86eb9d26bad71fbfff7a5654ea9c1dc4e971426a6217149cd503fd22ed44c4.jpg)

![d58f67da850502e1657906aceb5f1d8c2a67ff45c97dcecdeeb74a36301c473e.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/tables/d58f67da850502e1657906aceb5f1d8c2a67ff45c97dcecdeeb74a36301c473e.jpg)

![e7a25bab11ffafc6a8922008ade302ae4fac8ca614cb2944485228bb097a123d.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/tables/e7a25bab11ffafc6a8922008ade302ae4fac8ca614cb2944485228bb097a123d.jpg)

![f3f2f7a4357c4f9fdce1d8f4ba2e6276693e5ec46f73b93a64de39e87d5a0740.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/tables/f3f2f7a4357c4f9fdce1d8f4ba2e6276693e5ec46f73b93a64de39e87d5a0740.jpg)

![fe8dbe4ab986426a46afc90809b39ebd3b92e5fee3efcf14ee11d2086d353bc2.jpg](../icml_results/1185_Anchored%20Self-Play%20for%20Code%20Repair/tables/fe8dbe4ab986426a46afc90809b39ebd3b92e5fee3efcf14ee11d2086d353bc2.jpg)

<a id="paper-1168"></a>

## Decentralized Instruction Tuning: Conflict-Aware Splitting and Weight Merging


### Images

![1bda3ee259325409903d2006d1a6fc049fe8d6c7a3a011d1846b841cb8d5bdc0.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/images/1bda3ee259325409903d2006d1a6fc049fe8d6c7a3a011d1846b841cb8d5bdc0.jpg)

![2d0dfda8f88b7561f3f30944433c0a935e8c85da5768e2a141c174a864f0f414.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/images/2d0dfda8f88b7561f3f30944433c0a935e8c85da5768e2a141c174a864f0f414.jpg)

![44c33444282ad20de5de9bea4b25ce93e1187d56a7c55d485d593561f7ecd3c2.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/images/44c33444282ad20de5de9bea4b25ce93e1187d56a7c55d485d593561f7ecd3c2.jpg)

![47873266257d2485bd801c364036769a152e6285524e8b477ff5f8b94fcc1ae0.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/images/47873266257d2485bd801c364036769a152e6285524e8b477ff5f8b94fcc1ae0.jpg)

![4ebb4f028ffe47ef4a7f5d2b9bf5830d3a0da24198c7396f31914032d63d6dcd.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/images/4ebb4f028ffe47ef4a7f5d2b9bf5830d3a0da24198c7396f31914032d63d6dcd.jpg)

![84e07f89f3e3a2ae317ec7ace9ee1d54eef8e6156ee2bc21786923889f0e6d47.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/images/84e07f89f3e3a2ae317ec7ace9ee1d54eef8e6156ee2bc21786923889f0e6d47.jpg)

![8bf324d2987e7b0c36ae55496c6e8532e0e73a5ffdf3041199404d41a9666064.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/images/8bf324d2987e7b0c36ae55496c6e8532e0e73a5ffdf3041199404d41a9666064.jpg)

![d865900c8a164a8bda6fdc42e27545ef73832d7e77111ef12fe9f6c0214fd461.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/images/d865900c8a164a8bda6fdc42e27545ef73832d7e77111ef12fe9f6c0214fd461.jpg)

![e4ab6c3163b93674afdd941f995b620e44e816e10fc87b3418580f5a02d0c01f.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/images/e4ab6c3163b93674afdd941f995b620e44e816e10fc87b3418580f5a02d0c01f.jpg)

![eb95ceedfff89022931e153023563fd75083b44fde1e4bd9f565a2c3817a9c57.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/images/eb95ceedfff89022931e153023563fd75083b44fde1e4bd9f565a2c3817a9c57.jpg)

### Tables

![037ac54bd7ce1bc358a553e091ad31c48df8d3b6c243ac785a38f7ce6d24197a.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/037ac54bd7ce1bc358a553e091ad31c48df8d3b6c243ac785a38f7ce6d24197a.jpg)

![04334b367a833a57cb048e993020c97b1cdb81f47042bf45937ffa463b579c43.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/04334b367a833a57cb048e993020c97b1cdb81f47042bf45937ffa463b579c43.jpg)

![15c2c1fb0571ea12834bbee8a2cb024e8609a3b17475c62ed850a26d1e8f1022.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/15c2c1fb0571ea12834bbee8a2cb024e8609a3b17475c62ed850a26d1e8f1022.jpg)

![1bbb7ad944f9b62419dd14b406ac1fbb0c2996d776bbd568f39061e4456c3780.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/1bbb7ad944f9b62419dd14b406ac1fbb0c2996d776bbd568f39061e4456c3780.jpg)

![1f3d85a905aa473857dcfa07ae4394b51c754260bde80d763f21a5fec90e3301.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/1f3d85a905aa473857dcfa07ae4394b51c754260bde80d763f21a5fec90e3301.jpg)

![2a206b1b81662cb5927c2e43720e9670e5a52de39f38dbe844fa35accab71866.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/2a206b1b81662cb5927c2e43720e9670e5a52de39f38dbe844fa35accab71866.jpg)

![2ebcd4c01f3e17bf30354e14af3d5aeac02575786c4b31730fbd8a6b8f9fd284.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/2ebcd4c01f3e17bf30354e14af3d5aeac02575786c4b31730fbd8a6b8f9fd284.jpg)

![2efeb0988146b8a0e4dd145dd5093b70e04eaaea66637ac1af679236144fcfec.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/2efeb0988146b8a0e4dd145dd5093b70e04eaaea66637ac1af679236144fcfec.jpg)

![3b2b8246edd2e7d38336f726a71b6f3a022d843ccbe25d327831049338147577.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/3b2b8246edd2e7d38336f726a71b6f3a022d843ccbe25d327831049338147577.jpg)

![43e36c313d8c733acd3aeb0a74ff7eac0567d6641109e11a197b8033902e2532.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/43e36c313d8c733acd3aeb0a74ff7eac0567d6641109e11a197b8033902e2532.jpg)

![638e5f77bf315944805fa5915a61effd5c58e97b485ed73a80e8157caf231c97.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/638e5f77bf315944805fa5915a61effd5c58e97b485ed73a80e8157caf231c97.jpg)

![963e64abd329e2f0be1f9b43119eaab6502baa941151ceab8b1f02f8554e8a02.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/963e64abd329e2f0be1f9b43119eaab6502baa941151ceab8b1f02f8554e8a02.jpg)

![b88a8d28ea8c8d050dca7c20ff96ebb4f46a8209855218ebc58e4795d46b2772.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/b88a8d28ea8c8d050dca7c20ff96ebb4f46a8209855218ebc58e4795d46b2772.jpg)

![be2ca8633556b134aac8e5f87c4928e1abad037a9dd62125f0ddc2751d834c94.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/be2ca8633556b134aac8e5f87c4928e1abad037a9dd62125f0ddc2751d834c94.jpg)

![cc03f19fc7e9f063b935c3dccc95b05c546888ee10ae3c2b9be0da3f0607f30c.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/cc03f19fc7e9f063b935c3dccc95b05c546888ee10ae3c2b9be0da3f0607f30c.jpg)

![e0884e8fa8779643fd012ffa12fa21d9d6d65cee84c4a03a7f7288d592826a3c.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/e0884e8fa8779643fd012ffa12fa21d9d6d65cee84c4a03a7f7288d592826a3c.jpg)

![e9dea3c054b1d274caee05ee50498bd6c5812e3123baa35120f57e9e9616d543.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/e9dea3c054b1d274caee05ee50498bd6c5812e3123baa35120f57e9e9616d543.jpg)

![efb25bb9bbbf325fbf197348cfcb0251dd836f6435a086e62d929b2098d416cb.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/efb25bb9bbbf325fbf197348cfcb0251dd836f6435a086e62d929b2098d416cb.jpg)

![efb48315306585570cc6f8b4d117b08aaf3ab3016ed0a83be59510f26a1c2354.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/efb48315306585570cc6f8b4d117b08aaf3ab3016ed0a83be59510f26a1c2354.jpg)

![f59cce74007c78c1198ca98c3596d6c29b7c3b9d5ee232abc300c680f2e28e0c.jpg](../icml_results/1186_Decentralized%20Instruction%20Tuning_%20Conflict-Aware%20Splitting%20and%20Weight%20Merging/tables/f59cce74007c78c1198ca98c3596d6c29b7c3b9d5ee232abc300c680f2e28e0c.jpg)

<a id="paper-1169"></a>

## Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts


### Images

![25fe63d6ce52695981a9e697b658c6283048f3d5b47f86a41cd1b2143a2fad7d.jpg](../icml_results/1187_Toward%20Structural%20Multimodal%20Representations_%20Specialization%2C%20Selection%2C%20and%20Sparsification%20via%20Mixt/images/25fe63d6ce52695981a9e697b658c6283048f3d5b47f86a41cd1b2143a2fad7d.jpg)

![3da778e713bc98ec801cbf4a7ed07105231bf73f58f61f87ede5ac543d7770a6.jpg](../icml_results/1187_Toward%20Structural%20Multimodal%20Representations_%20Specialization%2C%20Selection%2C%20and%20Sparsification%20via%20Mixt/images/3da778e713bc98ec801cbf4a7ed07105231bf73f58f61f87ede5ac543d7770a6.jpg)

![49a918d9011260c991df813a75990a2d1c5dd45f71da560c4bb8e65eba8ebbf2.jpg](../icml_results/1187_Toward%20Structural%20Multimodal%20Representations_%20Specialization%2C%20Selection%2C%20and%20Sparsification%20via%20Mixt/images/49a918d9011260c991df813a75990a2d1c5dd45f71da560c4bb8e65eba8ebbf2.jpg)

![5e24d1f2b254a72d51b4e772837243243f8eeecf810b52c96f390b27e47b9aca.jpg](../icml_results/1187_Toward%20Structural%20Multimodal%20Representations_%20Specialization%2C%20Selection%2C%20and%20Sparsification%20via%20Mixt/images/5e24d1f2b254a72d51b4e772837243243f8eeecf810b52c96f390b27e47b9aca.jpg)

![78600aa8548907bc1f9da8aba07c8e4f8b56a7dd0f07956232f1b1259a5d45aa.jpg](../icml_results/1187_Toward%20Structural%20Multimodal%20Representations_%20Specialization%2C%20Selection%2C%20and%20Sparsification%20via%20Mixt/images/78600aa8548907bc1f9da8aba07c8e4f8b56a7dd0f07956232f1b1259a5d45aa.jpg)

![97801cefc2776efeee8789e635ded51b1137b6de6e21086c3480e47b9878d905.jpg](../icml_results/1187_Toward%20Structural%20Multimodal%20Representations_%20Specialization%2C%20Selection%2C%20and%20Sparsification%20via%20Mixt/images/97801cefc2776efeee8789e635ded51b1137b6de6e21086c3480e47b9878d905.jpg)

![a1a957b6ac81864664d280b8fe04cea80b1eaf406dc10da9e3b4d52460b2d040.jpg](../icml_results/1187_Toward%20Structural%20Multimodal%20Representations_%20Specialization%2C%20Selection%2C%20and%20Sparsification%20via%20Mixt/images/a1a957b6ac81864664d280b8fe04cea80b1eaf406dc10da9e3b4d52460b2d040.jpg)

![a9843ffa486fc8597a39d012ca61629bd64c8f2b6e17a0ea9b0c953552b19a90.jpg](../icml_results/1187_Toward%20Structural%20Multimodal%20Representations_%20Specialization%2C%20Selection%2C%20and%20Sparsification%20via%20Mixt/images/a9843ffa486fc8597a39d012ca61629bd64c8f2b6e17a0ea9b0c953552b19a90.jpg)

![bed92e47fc6cafa673698eac23122ffe9209a57f01fb434f15c7c7aff1d950cb.jpg](../icml_results/1187_Toward%20Structural%20Multimodal%20Representations_%20Specialization%2C%20Selection%2C%20and%20Sparsification%20via%20Mixt/images/bed92e47fc6cafa673698eac23122ffe9209a57f01fb434f15c7c7aff1d950cb.jpg)

![f7304eb8dea758127af4d0713636019ffbb4ca44934f439477a2dcb3a95b7314.jpg](../icml_results/1187_Toward%20Structural%20Multimodal%20Representations_%20Specialization%2C%20Selection%2C%20and%20Sparsification%20via%20Mixt/images/f7304eb8dea758127af4d0713636019ffbb4ca44934f439477a2dcb3a95b7314.jpg)

### Tables

![4e66cedeeb848a08847e0ed854fcf06e415fdbf40e31763ed265b21bd975c6d5.jpg](../icml_results/1187_Toward%20Structural%20Multimodal%20Representations_%20Specialization%2C%20Selection%2C%20and%20Sparsification%20via%20Mixt/tables/4e66cedeeb848a08847e0ed854fcf06e415fdbf40e31763ed265b21bd975c6d5.jpg)

![53a16a5089f97574d713332b5f6ff8cf0fbae79a59bebec42f0f20110e0ac256.jpg](../icml_results/1187_Toward%20Structural%20Multimodal%20Representations_%20Specialization%2C%20Selection%2C%20and%20Sparsification%20via%20Mixt/tables/53a16a5089f97574d713332b5f6ff8cf0fbae79a59bebec42f0f20110e0ac256.jpg)

![6a76b64f4061bc5990623d307ac528651b1fe9130722350c5987236a2b548db5.jpg](../icml_results/1187_Toward%20Structural%20Multimodal%20Representations_%20Specialization%2C%20Selection%2C%20and%20Sparsification%20via%20Mixt/tables/6a76b64f4061bc5990623d307ac528651b1fe9130722350c5987236a2b548db5.jpg)

![6c2a59223a01b6959870febeac931c17c700e9fe0fd5f81cb41bd3493c6ba012.jpg](../icml_results/1187_Toward%20Structural%20Multimodal%20Representations_%20Specialization%2C%20Selection%2C%20and%20Sparsification%20via%20Mixt/tables/6c2a59223a01b6959870febeac931c17c700e9fe0fd5f81cb41bd3493c6ba012.jpg)

![ae6ad418d539766bda3569c605387fa68fe73f4ca092cc4abad26e04eb9ae2e0.jpg](../icml_results/1187_Toward%20Structural%20Multimodal%20Representations_%20Specialization%2C%20Selection%2C%20and%20Sparsification%20via%20Mixt/tables/ae6ad418d539766bda3569c605387fa68fe73f4ca092cc4abad26e04eb9ae2e0.jpg)

<a id="paper-1170"></a>

## EMBGuard: Constructing Hazard-Aware Guardrails for Safe Planning in Embodied Agents


### Images

![05c63bfc33623f6f8998ec58b3ea4adcf3b3def01a48f38bbca02ed079661fe4.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/05c63bfc33623f6f8998ec58b3ea4adcf3b3def01a48f38bbca02ed079661fe4.jpg)

![0f4547ce7ae7dbb83386f52f8f36a3cd65ec20fbc6ad4d4f3f71e044d0550427.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/0f4547ce7ae7dbb83386f52f8f36a3cd65ec20fbc6ad4d4f3f71e044d0550427.jpg)

![1dcbfd073068db30a12a21d559fcd28d31de7791219bc5b0c65b4c70d9b346f0.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/1dcbfd073068db30a12a21d559fcd28d31de7791219bc5b0c65b4c70d9b346f0.jpg)

![30e44528909b4acc3f0f035d71e259405880f490a413131ac4b7a94214e4ae65.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/30e44528909b4acc3f0f035d71e259405880f490a413131ac4b7a94214e4ae65.jpg)

![48cd96e57bdbbb619aa691af32a15f19735e8d57f261a19cbbf77a2a7001d54b.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/48cd96e57bdbbb619aa691af32a15f19735e8d57f261a19cbbf77a2a7001d54b.jpg)

![522901a35642263f42cf698ca6cf1b812bcc7d8a4cfbd0e2a2b9a1ec6c46cda5.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/522901a35642263f42cf698ca6cf1b812bcc7d8a4cfbd0e2a2b9a1ec6c46cda5.jpg)

![611b26ed318e14c17d3d8bafc9ef60cb2259e2ee4dc861d24b1e32f43aefacce.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/611b26ed318e14c17d3d8bafc9ef60cb2259e2ee4dc861d24b1e32f43aefacce.jpg)

![62bb64f4f4c8050489def8ac4826b7b68d1d520ca5a200efa1205015a2922908.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/62bb64f4f4c8050489def8ac4826b7b68d1d520ca5a200efa1205015a2922908.jpg)

![6513c3eb977fa2a960e7fb2e59bdcd1f5c542baf8590b2ded38ae01499960eda.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/6513c3eb977fa2a960e7fb2e59bdcd1f5c542baf8590b2ded38ae01499960eda.jpg)

![7b25ffed6296fa8b81500e1b6bf750d32e1cde6f4dc464e2711ab86e8210377e.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/7b25ffed6296fa8b81500e1b6bf750d32e1cde6f4dc464e2711ab86e8210377e.jpg)

![99d8e5e519d009bea20f7f447d88d1811877841efdf686c39a39cc0b73a275c3.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/99d8e5e519d009bea20f7f447d88d1811877841efdf686c39a39cc0b73a275c3.jpg)

![a3d565858084bef9f27d10b655ee5a3583a85f572347124b86733bbd01d45b73.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/a3d565858084bef9f27d10b655ee5a3583a85f572347124b86733bbd01d45b73.jpg)

![a7974058f88f67c1926bfa07a6818934c0efb0260f795e90cd9b9e7f3dcd8615.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/a7974058f88f67c1926bfa07a6818934c0efb0260f795e90cd9b9e7f3dcd8615.jpg)

![a97d7620f83935826268f3fcf4e1e6197e57f6669b011a1a124e468704e8c6f0.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/a97d7620f83935826268f3fcf4e1e6197e57f6669b011a1a124e468704e8c6f0.jpg)

![b6049806c38e4a9005f32046221a72481c80ad452038a5372637850c2f4c07b0.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/b6049806c38e4a9005f32046221a72481c80ad452038a5372637850c2f4c07b0.jpg)

![bcc4b793b51f3c83bbaa7ba650d77717e94df31719388266b6144391d936b642.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/bcc4b793b51f3c83bbaa7ba650d77717e94df31719388266b6144391d936b642.jpg)

![d5cc783bb4ab2de917f44fd659528dcec2923e28c646fc22721c68de40255349.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/d5cc783bb4ab2de917f44fd659528dcec2923e28c646fc22721c68de40255349.jpg)

![ea94610efc84cfc0bc099db3b340137e78e321b99d87a0a44a71443072faa951.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/ea94610efc84cfc0bc099db3b340137e78e321b99d87a0a44a71443072faa951.jpg)

![efd159a5122dbb3b0aa8d6f013167362c262daa8e09bd810fb2cc99e864ca3c3.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/images/efd159a5122dbb3b0aa8d6f013167362c262daa8e09bd810fb2cc99e864ca3c3.jpg)

### Tables

![04860d45f3dd274fbcd8f3336972055759b52461961286694b422dffb63d2a44.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/04860d45f3dd274fbcd8f3336972055759b52461961286694b422dffb63d2a44.jpg)

![09af8146f7c9b23f5fb0352bf1f41d40c41b9d6613d86e1dac1768a73e80d5ff.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/09af8146f7c9b23f5fb0352bf1f41d40c41b9d6613d86e1dac1768a73e80d5ff.jpg)

![0d0b0e8d1af4fc2b2b419bf5b4b24263bdd821b50f285fd35d70c152e3741bd4.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/0d0b0e8d1af4fc2b2b419bf5b4b24263bdd821b50f285fd35d70c152e3741bd4.jpg)

![0f53f6a28e567637097d263095c3b6a8ea6d0a9e49b9488d4d7079b0f48a78a8.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/0f53f6a28e567637097d263095c3b6a8ea6d0a9e49b9488d4d7079b0f48a78a8.jpg)

![15f1a78e23428a0e1b13eb7543339131e0f70ee4e9533ac08dd8dda8e46e9fcd.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/15f1a78e23428a0e1b13eb7543339131e0f70ee4e9533ac08dd8dda8e46e9fcd.jpg)

![3c41e1fbfd340f61156b7768962a30034088c89b63fd2fe0d5f280c02e662ee0.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/3c41e1fbfd340f61156b7768962a30034088c89b63fd2fe0d5f280c02e662ee0.jpg)

![547e7f10e2436a72b8c7b8c3ca223fd95b52affffd313845b06af41dc584a129.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/547e7f10e2436a72b8c7b8c3ca223fd95b52affffd313845b06af41dc584a129.jpg)

![5d81655759006663d695693ee437d79be5600586d54ea1122e8d7b2867fabc3c.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/5d81655759006663d695693ee437d79be5600586d54ea1122e8d7b2867fabc3c.jpg)

![6a742718e6d23d12707cfbc414a066da60c6dbc38f0ecb155d41da9df2e66688.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/6a742718e6d23d12707cfbc414a066da60c6dbc38f0ecb155d41da9df2e66688.jpg)

![6affc2da8488d2045fd02118fba6c52b6f257560da73ca93675ff814532ae88b.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/6affc2da8488d2045fd02118fba6c52b6f257560da73ca93675ff814532ae88b.jpg)

![765d0483cc3c8979f753a5d4fcb5d6159dbce21e115ca458c376e471b592528e.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/765d0483cc3c8979f753a5d4fcb5d6159dbce21e115ca458c376e471b592528e.jpg)

![7bd05c5149bfa025de65ed085465592966f3f806cf3d8b90d90dbbd924978210.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/7bd05c5149bfa025de65ed085465592966f3f806cf3d8b90d90dbbd924978210.jpg)

![7c3c2eb365d67d8ad26e20466b2119bdf08b3b2c705bf2e0755129c5e96031cd.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/7c3c2eb365d67d8ad26e20466b2119bdf08b3b2c705bf2e0755129c5e96031cd.jpg)

![7cde1b8a267b7f41f45885d38401e42388e62347ef4218d2f13422b8cc92d35d.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/7cde1b8a267b7f41f45885d38401e42388e62347ef4218d2f13422b8cc92d35d.jpg)

![8cbcca653651c19df6ab7c65d23fefc2e38e2d39efd85d3a139bb8cd9b8a9103.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/8cbcca653651c19df6ab7c65d23fefc2e38e2d39efd85d3a139bb8cd9b8a9103.jpg)

![9d223894ab0bc9a68ab8b1667987eae2a451e332fcf9f0dc5fbccabf19a177ad.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/9d223894ab0bc9a68ab8b1667987eae2a451e332fcf9f0dc5fbccabf19a177ad.jpg)

![a5d21f92d1a85def512c96f7efc9d79db30d1b36731079031504594ef4daa794.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/a5d21f92d1a85def512c96f7efc9d79db30d1b36731079031504594ef4daa794.jpg)

![ab62558abe1ad25db213e3e6d636ceebcd127e129ea3cb309fa2322c82fd1bfd.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/ab62558abe1ad25db213e3e6d636ceebcd127e129ea3cb309fa2322c82fd1bfd.jpg)

![b0be91bbdf5e7219454372f799b0f305351dc16e91bb6af3c396939e38b5affd.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/b0be91bbdf5e7219454372f799b0f305351dc16e91bb6af3c396939e38b5affd.jpg)

![c17f04830f06d598446222ef1d8fe2ff3d2c3d5b20a6b8c929d5b921c5881c9a.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/c17f04830f06d598446222ef1d8fe2ff3d2c3d5b20a6b8c929d5b921c5881c9a.jpg)

![c3276ec21de9aa0c871ab3cc41aede6e82b44959f504e0c8fe888272b0116bb9.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/c3276ec21de9aa0c871ab3cc41aede6e82b44959f504e0c8fe888272b0116bb9.jpg)

![cab90920a1d76e15a387337d30a702ec48230e48abc2532127cad4b0049245d9.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/cab90920a1d76e15a387337d30a702ec48230e48abc2532127cad4b0049245d9.jpg)

![d2fcd603ed08858aba1e838f0341db5ec5095e9fb94c8dc25d478524cc8021ec.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/d2fcd603ed08858aba1e838f0341db5ec5095e9fb94c8dc25d478524cc8021ec.jpg)

![dbeaa7275dd3e18beffb80ad97eba6d81efb275b7a0275ee7615bd9d3370e212.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/dbeaa7275dd3e18beffb80ad97eba6d81efb275b7a0275ee7615bd9d3370e212.jpg)

![f607701b123358f778fa1fc846d93683b97947a0080b9350d4416819945a43ed.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/f607701b123358f778fa1fc846d93683b97947a0080b9350d4416819945a43ed.jpg)

![fc69accbb0e8d068221438634ece2e92127d0ab09e1b71224770856e0e2fa0b0.jpg](../icml_results/1188_EMBGuard_%20Constructing%20Hazard-Aware%20Guardrails%20for%20Safe%20Planning%20in%20Embodied%20Agents/tables/fc69accbb0e8d068221438634ece2e92127d0ab09e1b71224770856e0e2fa0b0.jpg)

<a id="paper-1171"></a>

## Parameter Decorrelation via Transition-Variance Alignment for Multivariate Time-series Forecasting


### Images

![2dc05df84f39fa16f6988604828a3616576a2b2f4dd3fac459d468f6d91fc771.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/images/2dc05df84f39fa16f6988604828a3616576a2b2f4dd3fac459d468f6d91fc771.jpg)

![3d0345c53278547ef3359c8269cf0519d38a7e54942ffdd3005273b2ffc1f969.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/images/3d0345c53278547ef3359c8269cf0519d38a7e54942ffdd3005273b2ffc1f969.jpg)

![47c2f92f904304b28dc7e1ece9381c68378b2a46672b71d9a0a8beeed88f74e0.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/images/47c2f92f904304b28dc7e1ece9381c68378b2a46672b71d9a0a8beeed88f74e0.jpg)

![4ba8d4f342248219dff835afe99e9e8f48b3062e5c83f39032da48ec8008f9de.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/images/4ba8d4f342248219dff835afe99e9e8f48b3062e5c83f39032da48ec8008f9de.jpg)

![7e6611e39c9e143c4297fa722c26622c275a71ee0faad3eb3c4eb77b5a276038.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/images/7e6611e39c9e143c4297fa722c26622c275a71ee0faad3eb3c4eb77b5a276038.jpg)

![a740fc879524732fff54080fbf423779d3e748b6ab12efee37287c9c8b556a19.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/images/a740fc879524732fff54080fbf423779d3e748b6ab12efee37287c9c8b556a19.jpg)

![af1b4566a9ee8701e2dd314aa4580ed4c3451ebcf5010c9bce46fe53ef8ee99e.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/images/af1b4566a9ee8701e2dd314aa4580ed4c3451ebcf5010c9bce46fe53ef8ee99e.jpg)

![c12ddce3e5d3bad77d9e0815944b9109d6f3bcaf2f3011a0638e6902367aa417.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/images/c12ddce3e5d3bad77d9e0815944b9109d6f3bcaf2f3011a0638e6902367aa417.jpg)

![e82ad0e463da24c40c226a17cfaf877608106e53ad2e12255f0da22e3122e2e5.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/images/e82ad0e463da24c40c226a17cfaf877608106e53ad2e12255f0da22e3122e2e5.jpg)

### Tables

![018c465e4668ab0616ab70554bdd3c1161d8a0b138da1e039993ddd431d40812.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/tables/018c465e4668ab0616ab70554bdd3c1161d8a0b138da1e039993ddd431d40812.jpg)

![186d5cfc518490f650e27b37701e6d279257339f70c8ca141adb91e70e5df848.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/tables/186d5cfc518490f650e27b37701e6d279257339f70c8ca141adb91e70e5df848.jpg)

![28d5730d31887d1aa331a23bc96260bbe9c3d91c4d456bcab737db23c157332a.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/tables/28d5730d31887d1aa331a23bc96260bbe9c3d91c4d456bcab737db23c157332a.jpg)

![43e7835adfa02283a512fef1f9fcf6bcc89154fdb5a748fd64f3190f0eea8454.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/tables/43e7835adfa02283a512fef1f9fcf6bcc89154fdb5a748fd64f3190f0eea8454.jpg)

![4bf903fa5986b3a5017ae3694a018af6e58f682a9588931dca090f81e12e3b0a.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/tables/4bf903fa5986b3a5017ae3694a018af6e58f682a9588931dca090f81e12e3b0a.jpg)

![544834298a006a263799a4fc009f2faa943b3d3c7db767cc9eb7fa633c2ae744.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/tables/544834298a006a263799a4fc009f2faa943b3d3c7db767cc9eb7fa633c2ae744.jpg)

![5ef0b93eda962c77fe39b6451e51aefac6bab5acf2e9dfc490addacf43dcf2a6.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/tables/5ef0b93eda962c77fe39b6451e51aefac6bab5acf2e9dfc490addacf43dcf2a6.jpg)

![61f4482b778a15ffbde649912f34fcd1151ac5814f6ad67955cc3159a398cf80.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/tables/61f4482b778a15ffbde649912f34fcd1151ac5814f6ad67955cc3159a398cf80.jpg)

![67a05b022c062934ec48b95617583449eb94201c99a39e04576af42345197189.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/tables/67a05b022c062934ec48b95617583449eb94201c99a39e04576af42345197189.jpg)

![84e78b8a28c01c81124792594391ecadb5dedada85be08eeb64ca7a602befea6.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/tables/84e78b8a28c01c81124792594391ecadb5dedada85be08eeb64ca7a602befea6.jpg)

![a08a3648172d7c2c69e49f60a5ee61443107d00043114a62ce54c749e6356cb5.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/tables/a08a3648172d7c2c69e49f60a5ee61443107d00043114a62ce54c749e6356cb5.jpg)

![a9e00cd8911d8f5d23ab40f158197f8ed697445c870dfd0af1b4fcaa7d993bed.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/tables/a9e00cd8911d8f5d23ab40f158197f8ed697445c870dfd0af1b4fcaa7d993bed.jpg)

![f7aa25166d11e46cfb5d84433f29708f83cf8d3a58f8d55ebe70c0881b503402.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/tables/f7aa25166d11e46cfb5d84433f29708f83cf8d3a58f8d55ebe70c0881b503402.jpg)

![fad3b367745fe53409aa5b6d0a6a17cfa9000e41ecb7a71d81b822dfac044a62.jpg](../icml_results/1189_Parameter%20Decorrelation%20via%20Transition-Variance%20Alignment%20for%20Multivariate%20Time-series%20Forecasting/tables/fad3b367745fe53409aa5b6d0a6a17cfa9000e41ecb7a71d81b822dfac044a62.jpg)

<a id="paper-1172"></a>

## Chain-of-Goals Hierarchical Policy for Long-Horizon Offline Goal-Conditioned RL


### Images

![10150160af7e60210a9d5ade64702995131a09d491559aea3278743dec930614.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/images/10150160af7e60210a9d5ade64702995131a09d491559aea3278743dec930614.jpg)

![2135e3084c38ddc6071455c4c47f808a58043a923f5a9a66a67488bb3cd35b3a.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/images/2135e3084c38ddc6071455c4c47f808a58043a923f5a9a66a67488bb3cd35b3a.jpg)

![34a90979c0b465198e9903972b609cc0808b8ac038ddf23e8c6f6c9deb6e194e.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/images/34a90979c0b465198e9903972b609cc0808b8ac038ddf23e8c6f6c9deb6e194e.jpg)

![3d987834ae9fe372d98fa20833ea9f43e6111f93350f1908626efc5ac003ae4a.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/images/3d987834ae9fe372d98fa20833ea9f43e6111f93350f1908626efc5ac003ae4a.jpg)

![615f3783386063a2e88a8a11f37700e7c0ff6037c774eac168ce6e4b4871c8d4.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/images/615f3783386063a2e88a8a11f37700e7c0ff6037c774eac168ce6e4b4871c8d4.jpg)

![7d411373e04d7b010d077ba7a49128f0a978c6606c90d9fce63989684daae87f.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/images/7d411373e04d7b010d077ba7a49128f0a978c6606c90d9fce63989684daae87f.jpg)

![fc17d1ef8df569fc97d8fe436f0e522957904d7e062d3779073bfdd2fd90f5a5.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/images/fc17d1ef8df569fc97d8fe436f0e522957904d7e062d3779073bfdd2fd90f5a5.jpg)

![fdc198b24fe7eed68b58a68032cb0ded00f0a74fee5ee4936254c4bb1a047a77.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/images/fdc198b24fe7eed68b58a68032cb0ded00f0a74fee5ee4936254c4bb1a047a77.jpg)

### Tables

![3f4b4b4caa348c7a7cb49d65618fbdd14c28afc6bc9e53b44340699c160bc793.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/tables/3f4b4b4caa348c7a7cb49d65618fbdd14c28afc6bc9e53b44340699c160bc793.jpg)

![71e820b074ca958e2c80b21c568b9106d04a350a348f8f9befd865b80c83f68a.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/tables/71e820b074ca958e2c80b21c568b9106d04a350a348f8f9befd865b80c83f68a.jpg)

![75057be15eba952625a0a9c64639f6e5104ba0237c8b4ca67a62204ae0a8c8ec.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/tables/75057be15eba952625a0a9c64639f6e5104ba0237c8b4ca67a62204ae0a8c8ec.jpg)

![75d511499932b9d8e91a0b539a0aa0f935536d5a37b2c9df87b5b169fd229156.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/tables/75d511499932b9d8e91a0b539a0aa0f935536d5a37b2c9df87b5b169fd229156.jpg)

![b85781334c988bf74966c17c4094e03ebc6dfa15a328676ca9e0f14cb212358c.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/tables/b85781334c988bf74966c17c4094e03ebc6dfa15a328676ca9e0f14cb212358c.jpg)

![db78bdf0237d817c59ea33361423ccb263ae6e864b9da1f744ac1146f8d07cb7.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/tables/db78bdf0237d817c59ea33361423ccb263ae6e864b9da1f744ac1146f8d07cb7.jpg)

![dd46972a3b9af9008eb7f45e5229eb900fbc7e98b264b3272c70408136a35cd9.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/tables/dd46972a3b9af9008eb7f45e5229eb900fbc7e98b264b3272c70408136a35cd9.jpg)

![de5e7cfede63970dd318c4eb64762815d4e8e4ad0b544ebe6a39309e727fbc51.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/tables/de5e7cfede63970dd318c4eb64762815d4e8e4ad0b544ebe6a39309e727fbc51.jpg)

![e2a964bf90c43668be686e06510ad5634f96a463f30b38c27285d6c21af43f3d.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/tables/e2a964bf90c43668be686e06510ad5634f96a463f30b38c27285d6c21af43f3d.jpg)

![e340955c6fadec776e5bdae43121acf1c18da916bd4896c06565997bcc8014ef.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/tables/e340955c6fadec776e5bdae43121acf1c18da916bd4896c06565997bcc8014ef.jpg)

![fa5ff620b63a8a1df0325445a6adfbd11237632c9a7a11071f79df1cb157721d.jpg](../icml_results/1190_Chain-of-Goals%20Hierarchical%20Policy%20for%20Long-Horizon%20Offline%20Goal-Conditioned%20RL/tables/fa5ff620b63a8a1df0325445a6adfbd11237632c9a7a11071f79df1cb157721d.jpg)

<a id="paper-1173"></a>

## Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory


### Images

![7d27867adfea08f0af2123db368a886715c66a28d07bb7f22df3c440685f3484.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/images/7d27867adfea08f0af2123db368a886715c66a28d07bb7f22df3c440685f3484.jpg)

![97ccec676d4c5fe5418228867f7461dfe6c4ba5544a1c5bc152576413e832b82.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/images/97ccec676d4c5fe5418228867f7461dfe6c4ba5544a1c5bc152576413e832b82.jpg)

![9a3c4b4cf7913c98866d5ff5a411cc4d047dc7d07820e91495039bc35d3ea94c.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/images/9a3c4b4cf7913c98866d5ff5a411cc4d047dc7d07820e91495039bc35d3ea94c.jpg)

![a26eb2f9ce379f2f59297a96042f9b324f7aac55d0b8771b7ff5c1daf3e7d4f3.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/images/a26eb2f9ce379f2f59297a96042f9b324f7aac55d0b8771b7ff5c1daf3e7d4f3.jpg)

![a57fe21ad10dea011d27d19dce3ebe34b7cc2455e5b7810dbc1cbe7482a4651d.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/images/a57fe21ad10dea011d27d19dce3ebe34b7cc2455e5b7810dbc1cbe7482a4651d.jpg)

![b8460601e96667b13ec098c62aee8e503cdfdf212ccac49856d4fe0d5bfc6cb1.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/images/b8460601e96667b13ec098c62aee8e503cdfdf212ccac49856d4fe0d5bfc6cb1.jpg)

![d9fd47eb8e7b864e2fdee321503867ad257666e20f4ef596b65471abd8d820fa.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/images/d9fd47eb8e7b864e2fdee321503867ad257666e20f4ef596b65471abd8d820fa.jpg)

### Tables

![401fa1ccc91a732dd1e19965319873175469f7ef1d767e40195a0d49d6445972.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/tables/401fa1ccc91a732dd1e19965319873175469f7ef1d767e40195a0d49d6445972.jpg)

![501e325d2f3aca246601d6116d42f67f168158fd1a3c4040537857f65f78879c.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/tables/501e325d2f3aca246601d6116d42f67f168158fd1a3c4040537857f65f78879c.jpg)

![69f7ce7cc4b568769b9dea054bafda9f8ec5a09bd5504b8d044059934b7dc492.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/tables/69f7ce7cc4b568769b9dea054bafda9f8ec5a09bd5504b8d044059934b7dc492.jpg)

![74ddddcdbfab1e7f8a3420bd8fc82f2081b19c5e3593931943eb3db752671270.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/tables/74ddddcdbfab1e7f8a3420bd8fc82f2081b19c5e3593931943eb3db752671270.jpg)

![760bd3750cca3e4dfb20416040d62771342e9a0e1d934cc647fce17956ad445c.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/tables/760bd3750cca3e4dfb20416040d62771342e9a0e1d934cc647fce17956ad445c.jpg)

![83b575692f8793dc44d849a06813733ef74e2ecd36107439d0f5c124c1e273f2.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/tables/83b575692f8793dc44d849a06813733ef74e2ecd36107439d0f5c124c1e273f2.jpg)

![955476e8a1ed7af81cb21755494cb9e43fd0487e953c52cb0663b5db00098ae9.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/tables/955476e8a1ed7af81cb21755494cb9e43fd0487e953c52cb0663b5db00098ae9.jpg)

![9788e6dabb1311dac22081bd4702264757683057c1bbe2a888fa8422f8790a92.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/tables/9788e6dabb1311dac22081bd4702264757683057c1bbe2a888fa8422f8790a92.jpg)

![b9190c443d76e78d43aa4706dcc936824d2b850e788c783dad1f58d6c783e734.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/tables/b9190c443d76e78d43aa4706dcc936824d2b850e788c783dad1f58d6c783e734.jpg)

![cced20ba2de6a9f6eb4ac20c7414035d7905953e0943b5ca7945daecf0b3a880.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/tables/cced20ba2de6a9f6eb4ac20c7414035d7905953e0943b5ca7945daecf0b3a880.jpg)

![d34273b1ea3456f52fe65bb672fa53a811eb02129083daf35ab58a24b1ac5030.jpg](../icml_results/1191_Diagnosing%20the%20Reliability%20of%20LLM-as-a-Judge%20via%20Item%20Response%20Theory/tables/d34273b1ea3456f52fe65bb672fa53a811eb02129083daf35ab58a24b1ac5030.jpg)

<a id="paper-1174"></a>

## Rethinking the Design Space of Reinforcement Learning for Diffusion Models: On the Importance of Likelihood Estimation Beyond Loss Design


### Images

![05d7c3bb0ff103868be0438d7a01bc7bde81e610414b9311ebbc777653ba4ba9.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/05d7c3bb0ff103868be0438d7a01bc7bde81e610414b9311ebbc777653ba4ba9.jpg)

![0e1057eafb6d34fad769e4b4590470b6653abfc9f8c475dbd620d4a1c53f79cc.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/0e1057eafb6d34fad769e4b4590470b6653abfc9f8c475dbd620d4a1c53f79cc.jpg)

![134f00005ac19ab602d264a4a6fb6bcc97f6d844e244627480156f66b965ae1c.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/134f00005ac19ab602d264a4a6fb6bcc97f6d844e244627480156f66b965ae1c.jpg)

![3489dc3eff52ba44b131f8478d2173169ec0cb31e7590e0ca9bc802bf43c4e64.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/3489dc3eff52ba44b131f8478d2173169ec0cb31e7590e0ca9bc802bf43c4e64.jpg)

![3b1bf8f6357b342ca7a29e3f78d0eef535932076172a8faf6be94855b25f8153.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/3b1bf8f6357b342ca7a29e3f78d0eef535932076172a8faf6be94855b25f8153.jpg)

![3c7b45c017500585541f67b7d1a88193c5e88121d303c97eb27a1cc62846d555.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/3c7b45c017500585541f67b7d1a88193c5e88121d303c97eb27a1cc62846d555.jpg)

![46cb8d7bfd242787f969044a82158892917d07923eef33d55b1b165b8dbb6e4b.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/46cb8d7bfd242787f969044a82158892917d07923eef33d55b1b165b8dbb6e4b.jpg)

![7ced389e81fd00de9410536ee935751478cea44a18aa832490d2695005375221.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/7ced389e81fd00de9410536ee935751478cea44a18aa832490d2695005375221.jpg)

![7dfefebc3fcf1a5980d48f135cb8e4f0b11fb99c54444b49ebea98082340cc06.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/7dfefebc3fcf1a5980d48f135cb8e4f0b11fb99c54444b49ebea98082340cc06.jpg)

![91ae4253a2de8bfc84df7dd4ea7c3290bce1e62a1c1fa0c2b6050362be5ffd75.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/91ae4253a2de8bfc84df7dd4ea7c3290bce1e62a1c1fa0c2b6050362be5ffd75.jpg)

![9377e5bd79f452916021d2934b50e06cc0f748512ee8c3b6531f4d2559359cad.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/9377e5bd79f452916021d2934b50e06cc0f748512ee8c3b6531f4d2559359cad.jpg)

![95604840ea3c2131f2447eb845962cd4a7063cf555b601a31ed65aa12ae68d4e.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/95604840ea3c2131f2447eb845962cd4a7063cf555b601a31ed65aa12ae68d4e.jpg)

![99f61373e6e043d12e000feda4c40020fe0ccb8bbffc9bd2f716f45fa90917e2.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/99f61373e6e043d12e000feda4c40020fe0ccb8bbffc9bd2f716f45fa90917e2.jpg)

![9c68eb2a24be28961791d237e962871f97cb6c886efb6fb8518ba6557097b720.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/9c68eb2a24be28961791d237e962871f97cb6c886efb6fb8518ba6557097b720.jpg)

![9da46d5a9600b04109fbf3fee6f52c425549244bf1c0647e3ff547831cd814c7.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/9da46d5a9600b04109fbf3fee6f52c425549244bf1c0647e3ff547831cd814c7.jpg)

![c0fbc66f7fd51a3da0e72e441d83d10456ebb872e2898c6a214416dd4d44bee1.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/c0fbc66f7fd51a3da0e72e441d83d10456ebb872e2898c6a214416dd4d44bee1.jpg)

![c16c98bcd688d38e0a4fa8323942de511e1909c708a32fe42f23128734023a7d.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/c16c98bcd688d38e0a4fa8323942de511e1909c708a32fe42f23128734023a7d.jpg)

![c68c1cc127c926b0efc4129984eb569db5a31dfb031dafb9c7186919126c8555.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/c68c1cc127c926b0efc4129984eb569db5a31dfb031dafb9c7186919126c8555.jpg)

![c7bab8afe1e4ed6f6058a36d941b63716842a2c9511766c1eb718fb9ca42cea1.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/c7bab8afe1e4ed6f6058a36d941b63716842a2c9511766c1eb718fb9ca42cea1.jpg)

![cda6e2e6a9fa494d4139a1bcc579b5f53e27c15053d9672e0701cc5936828401.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/cda6e2e6a9fa494d4139a1bcc579b5f53e27c15053d9672e0701cc5936828401.jpg)

![d1b38dd0d5ddb9eba40ec36d0efc40ca17211630cfb5b27018bb40e868c0481c.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/d1b38dd0d5ddb9eba40ec36d0efc40ca17211630cfb5b27018bb40e868c0481c.jpg)

![d43dd58516b2ab50e33de2ebb9d3e3a855f37ae798fbcbb70d1a3040b006ee8b.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/d43dd58516b2ab50e33de2ebb9d3e3a855f37ae798fbcbb70d1a3040b006ee8b.jpg)

![e4b5ba8276b0c642370737deffaf5f39d4cb1673745d54696d1c6be5bc804c3b.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/e4b5ba8276b0c642370737deffaf5f39d4cb1673745d54696d1c6be5bc804c3b.jpg)

![fc21d0e72631fc9fe47ca9a87ab643a1a4b80f6eff659d7f6601b7cd6f924a19.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/images/fc21d0e72631fc9fe47ca9a87ab643a1a4b80f6eff659d7f6601b7cd6f924a19.jpg)

### Tables

![3b4e8f5e5810c809333256ddc4cf4b5d6625b2ab99b71020b74f4c4f73ea26b1.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/tables/3b4e8f5e5810c809333256ddc4cf4b5d6625b2ab99b71020b74f4c4f73ea26b1.jpg)

![5be7bc651eb73c5c1e30878f9f068f16ecd6f18b7cfc92e05bb3fba9908afd1d.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/tables/5be7bc651eb73c5c1e30878f9f068f16ecd6f18b7cfc92e05bb3fba9908afd1d.jpg)

![60d2cd5296c01dc04a02fa2e62cd77b2d942acc6d63144456fbe8b84f5531b51.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/tables/60d2cd5296c01dc04a02fa2e62cd77b2d942acc6d63144456fbe8b84f5531b51.jpg)

![65d192f17bc7c841acb443d808d319fd29af83f70038e8d2d6a875c0a0ebddbf.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/tables/65d192f17bc7c841acb443d808d319fd29af83f70038e8d2d6a875c0a0ebddbf.jpg)

![6e37413ef55ce87f1cad0fc71d4dd6bf79f61ade04890124c5505eaa23625669.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/tables/6e37413ef55ce87f1cad0fc71d4dd6bf79f61ade04890124c5505eaa23625669.jpg)

![7a4319e0a207b4eabe6d0555a63f7bff3ad76b25975da7a263c34168fc9d3bee.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/tables/7a4319e0a207b4eabe6d0555a63f7bff3ad76b25975da7a263c34168fc9d3bee.jpg)

![8006de8a8c70b54de61a2d1941744bfbbe09aaa9ff0a1df2a18c1428930f3776.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/tables/8006de8a8c70b54de61a2d1941744bfbbe09aaa9ff0a1df2a18c1428930f3776.jpg)

![ad24b9a05aeeefb96b770c643e2913115e35fb955da084250977fbb52035930c.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/tables/ad24b9a05aeeefb96b770c643e2913115e35fb955da084250977fbb52035930c.jpg)

![eabdb79a8eeb9fd6f67916516a197a6ad6cb9d4b7b2d9ab36fe064522210db5e.jpg](../icml_results/1192_Rethinking%20the%20Design%20Space%20of%20Reinforcement%20Learning%20for%20Diffusion%20Models_%20On%20the%20Importance%20of%20Lik/tables/eabdb79a8eeb9fd6f67916516a197a6ad6cb9d4b7b2d9ab36fe064522210db5e.jpg)

<a id="paper-1175"></a>

## NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models


### Images

![00cbccb997f736048364d398cbe2b8c657677ec030a10ce29501b92ca65ed63a.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/images/00cbccb997f736048364d398cbe2b8c657677ec030a10ce29501b92ca65ed63a.jpg)

![05ca25556fd616e3e97c1301102330a1088472547b3bcf68c7f665cc54a5d2ba.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/images/05ca25556fd616e3e97c1301102330a1088472547b3bcf68c7f665cc54a5d2ba.jpg)

![07c65fb7aeb1746152eed931413925ffc7749058121f41a2b86d8b3d30caeba5.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/images/07c65fb7aeb1746152eed931413925ffc7749058121f41a2b86d8b3d30caeba5.jpg)

![0dcbf1c44e3f7a6099e97726e7af2128b8439e7bde04d89a5ba196e9400f7257.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/images/0dcbf1c44e3f7a6099e97726e7af2128b8439e7bde04d89a5ba196e9400f7257.jpg)

![11413d4a5d65b761db46cdba37723bfda07a0afceecb96d4fd4a193eadc0b14d.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/images/11413d4a5d65b761db46cdba37723bfda07a0afceecb96d4fd4a193eadc0b14d.jpg)

![37b297542b856778339f7b640b3004e1bb0722193099b0670932add6ead348f7.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/images/37b297542b856778339f7b640b3004e1bb0722193099b0670932add6ead348f7.jpg)

![48399d480ca9db30759d355c202666cea5688c20102da329ce0e859abec182dd.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/images/48399d480ca9db30759d355c202666cea5688c20102da329ce0e859abec182dd.jpg)

![519ab654de601f2539132b13a363db498eff7eeda174739c654323dff5ac8f2c.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/images/519ab654de601f2539132b13a363db498eff7eeda174739c654323dff5ac8f2c.jpg)

![73420e37ffe8901ca5bdb278add62197eb7fa1f12973f2dc0630b9cc6d6de15c.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/images/73420e37ffe8901ca5bdb278add62197eb7fa1f12973f2dc0630b9cc6d6de15c.jpg)

![d14a75c4493cc94cd6b830f41dc943f4acf65fac6793811779de3fed6fdd58de.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/images/d14a75c4493cc94cd6b830f41dc943f4acf65fac6793811779de3fed6fdd58de.jpg)

![d7431f0032f409a3e92b59fd68bc81d59bebb8693b477d5fb244716296a7493d.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/images/d7431f0032f409a3e92b59fd68bc81d59bebb8693b477d5fb244716296a7493d.jpg)

![f182d4e15f2f602ef0f6d9e762f19c509d95c155ffd16f323e11fe1a4de2e289.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/images/f182d4e15f2f602ef0f6d9e762f19c509d95c155ffd16f323e11fe1a4de2e289.jpg)

![f2f8e722a267abeb758f595283e16e27ef8464fb7d1f8c69cb198c62aee5266d.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/images/f2f8e722a267abeb758f595283e16e27ef8464fb7d1f8c69cb198c62aee5266d.jpg)

![fd13e1ed3cb94df50e97a9cc6f877b0da3a279342671c7d8c50fead887663711.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/images/fd13e1ed3cb94df50e97a9cc6f877b0da3a279342671c7d8c50fead887663711.jpg)

### Tables

![16423834eb86b49a6f6607643ebb63cebc06e0ade1293abdd70aceec6390e577.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/tables/16423834eb86b49a6f6607643ebb63cebc06e0ade1293abdd70aceec6390e577.jpg)

![2c8387846f589e5e30f8b551885efbed447293eff6084927a4200c3e7f2ea65c.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/tables/2c8387846f589e5e30f8b551885efbed447293eff6084927a4200c3e7f2ea65c.jpg)

![59775bed236bd63160032748dbedb42e9134b196187eba78ffee7b8b19604700.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/tables/59775bed236bd63160032748dbedb42e9134b196187eba78ffee7b8b19604700.jpg)

![6c8298effa0abbb2a822bc825718c07e7c45f6f679e0c4f4e07bc6c10b8957ee.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/tables/6c8298effa0abbb2a822bc825718c07e7c45f6f679e0c4f4e07bc6c10b8957ee.jpg)

![78f7f545745750478b543b5209924180616532fe6ca7658c6df7780bfd63ed4a.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/tables/78f7f545745750478b543b5209924180616532fe6ca7658c6df7780bfd63ed4a.jpg)

![7b3ee1e7d928654a0a1db620280e1ab454ca0893ab7c83faa46770065a27f0b4.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/tables/7b3ee1e7d928654a0a1db620280e1ab454ca0893ab7c83faa46770065a27f0b4.jpg)

![85bcb067b34b8f468f44dbbb06a146b70fd8a499f0cb81d862dceb9885b1de23.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/tables/85bcb067b34b8f468f44dbbb06a146b70fd8a499f0cb81d862dceb9885b1de23.jpg)

![8dd8375309e2c7df25b19e746f8b69bccffa6e579ed316a6cdcb327dfef5fea7.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/tables/8dd8375309e2c7df25b19e746f8b69bccffa6e579ed316a6cdcb327dfef5fea7.jpg)

![8edfce159528dba0c168d9bafd967c7dc9fe4011d26529dcbdee52768e85bf94.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/tables/8edfce159528dba0c168d9bafd967c7dc9fe4011d26529dcbdee52768e85bf94.jpg)

![9be86a4c6afa25e1dc8904348a5ce59849f5b59b5ac469cee8e049a2d15e80d4.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/tables/9be86a4c6afa25e1dc8904348a5ce59849f5b59b5ac469cee8e049a2d15e80d4.jpg)

![a188af19a84b259a9fc5b234693f33bda19e8c7d70fbb99bdeb914908843cbf3.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/tables/a188af19a84b259a9fc5b234693f33bda19e8c7d70fbb99bdeb914908843cbf3.jpg)

![b39845788eb2a57934ac9066c05e733edabf44d20605f4422ac95e0c3413d9ce.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/tables/b39845788eb2a57934ac9066c05e733edabf44d20605f4422ac95e0c3413d9ce.jpg)

![b4648009bd1ee7ac6d979182a27b8452ac0688a23b9f9a5bc642ae97137fc863.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/tables/b4648009bd1ee7ac6d979182a27b8452ac0688a23b9f9a5bc642ae97137fc863.jpg)

![b4e863e6ebe604226389cee221eb33237000fa09e9144a49255d71487d36ac13.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/tables/b4e863e6ebe604226389cee221eb33237000fa09e9144a49255d71487d36ac13.jpg)

![e04ff646304c6ba69c259752c20f96bbdf973de9d65c690a493a66271039f670.jpg](../icml_results/1193_NanoQuant_%20Efficient%20Sub-1-Bit%20Quantization%20of%20Large%20Language%20Models/tables/e04ff646304c6ba69c259752c20f96bbdf973de9d65c690a493a66271039f670.jpg)

<a id="paper-1176"></a>

## Towards Feedback-to-Plan Decisions for Self-Evolving LLM Agents in CUDA Kernel Generation


### Images

![08d7ca7fdb8613b511017ab52c52359c009323da55719661772eab31331a2d53.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/08d7ca7fdb8613b511017ab52c52359c009323da55719661772eab31331a2d53.jpg)

![0ab39d57f2969e1b7669eb1bdf0f27a7222a60b98826e924c583e6a34c849793.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/0ab39d57f2969e1b7669eb1bdf0f27a7222a60b98826e924c583e6a34c849793.jpg)

![0ebb68579a12eb973d0aae37b894aa6e43b20e31775ba3c560f850321885f674.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/0ebb68579a12eb973d0aae37b894aa6e43b20e31775ba3c560f850321885f674.jpg)

![17e419b90abab45bbd4a6cf56d3578dc87632fdb2d586596d0a01be7e4297294.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/17e419b90abab45bbd4a6cf56d3578dc87632fdb2d586596d0a01be7e4297294.jpg)

![1e586635e8088171b0ef75f6e5385757a26c5a148c8e9b88ba7c883e77db8623.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/1e586635e8088171b0ef75f6e5385757a26c5a148c8e9b88ba7c883e77db8623.jpg)

![27fd1bbd268e4b9f35c26afd69ddfe94097c5d178c94431646e8bedfe3a27737.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/27fd1bbd268e4b9f35c26afd69ddfe94097c5d178c94431646e8bedfe3a27737.jpg)

![32745fc03521752db049f1c357246afeb3ccab272290f6987fb150454fcc587b.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/32745fc03521752db049f1c357246afeb3ccab272290f6987fb150454fcc587b.jpg)

![36c0cff351ea3bcec293fcbcd55d2e0bdfd0d8da0b63dc4fd80d0e7bf742c026.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/36c0cff351ea3bcec293fcbcd55d2e0bdfd0d8da0b63dc4fd80d0e7bf742c026.jpg)

![3d7e1966d81d1bcf3975a68c6b01988b87771eafcb06c178feda4af69ff23e33.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/3d7e1966d81d1bcf3975a68c6b01988b87771eafcb06c178feda4af69ff23e33.jpg)

![624acca88993b0457573e753784668ecc6a674448e7214d2808758648da72fd4.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/624acca88993b0457573e753784668ecc6a674448e7214d2808758648da72fd4.jpg)

![6c1c944385f2d3ae7bc7b529216706f4a4874a2d361464bb035be5ed5024154f.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/6c1c944385f2d3ae7bc7b529216706f4a4874a2d361464bb035be5ed5024154f.jpg)

![70c251c752c04443b58f998301b329ba0785a9d991cdf95f27858ee1e7a4bf1c.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/70c251c752c04443b58f998301b329ba0785a9d991cdf95f27858ee1e7a4bf1c.jpg)

![7b633989013343a9e6768e0e90773d469837d813149f4bda861d436a111041ca.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/7b633989013343a9e6768e0e90773d469837d813149f4bda861d436a111041ca.jpg)

![806fee69b544f58b8147ab9771ca998acf6b2887fd4172cd04f2954da1730d60.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/806fee69b544f58b8147ab9771ca998acf6b2887fd4172cd04f2954da1730d60.jpg)

![97e2ee7e30ce481504b3ad37766570a9cd1d50aef76e7cae833addee8f0df4dc.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/97e2ee7e30ce481504b3ad37766570a9cd1d50aef76e7cae833addee8f0df4dc.jpg)

![ab7585aa017abfa27a503878455ce3f22890d0eb6b41d1083eab39cab5660aff.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/ab7585aa017abfa27a503878455ce3f22890d0eb6b41d1083eab39cab5660aff.jpg)

![af9aec60b5719719bb0610da8d50ed0f7aa50edfd9f4ec2f3184b82faabb5a56.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/af9aec60b5719719bb0610da8d50ed0f7aa50edfd9f4ec2f3184b82faabb5a56.jpg)

![afed04d76414ebcd9b790f3a7d335ea43b4a2cfc4228539c6b0f30d8ff0c02a5.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/afed04d76414ebcd9b790f3a7d335ea43b4a2cfc4228539c6b0f30d8ff0c02a5.jpg)

![b047912f3a81dd8d0cb689469f606e8503e727f0992bd105dd58ffa003bbd79b.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/b047912f3a81dd8d0cb689469f606e8503e727f0992bd105dd58ffa003bbd79b.jpg)

![b8dabdd64517a92ef6b1544f7ee0e151fb8965fe41256145633c0461f1321c6d.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/b8dabdd64517a92ef6b1544f7ee0e151fb8965fe41256145633c0461f1321c6d.jpg)

![bc0bb0e495760c36ca5ad6c9af81c5be2fc03ed1859f0b735b2dac8a2da03097.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/bc0bb0e495760c36ca5ad6c9af81c5be2fc03ed1859f0b735b2dac8a2da03097.jpg)

![bf645e176e526bd30d12ac7887afe189024f315847e08e4d7383d6885246c3d0.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/bf645e176e526bd30d12ac7887afe189024f315847e08e4d7383d6885246c3d0.jpg)

![c2567edd052034e0ccf32872e41d8c11b220131c254b8efdf20784e4bbbca150.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/c2567edd052034e0ccf32872e41d8c11b220131c254b8efdf20784e4bbbca150.jpg)

![cdf72a5c0ff4184162ba39e0188ca15686431d46db50a519b38e45ea58a200a0.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/cdf72a5c0ff4184162ba39e0188ca15686431d46db50a519b38e45ea58a200a0.jpg)

![dde201cf848ed7f7779a4a5fa1975e0b57f9137c60a029e420f16e139a53a17b.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/dde201cf848ed7f7779a4a5fa1975e0b57f9137c60a029e420f16e139a53a17b.jpg)

![e7188d8e1da1050d79957b233844d94ec6267d25b55148ab61749a1fa7033081.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/e7188d8e1da1050d79957b233844d94ec6267d25b55148ab61749a1fa7033081.jpg)

![f749f1c20e5c7f50e11940bd4db23291b445e86b1f03fb9ea75b4cbed6ffb392.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/f749f1c20e5c7f50e11940bd4db23291b445e86b1f03fb9ea75b4cbed6ffb392.jpg)

![f77464e1898132f0123348c40207d0e27c8bd6efa2e9c9076fd609c1037f7eab.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/f77464e1898132f0123348c40207d0e27c8bd6efa2e9c9076fd609c1037f7eab.jpg)

![ffc2968208161ef2593d45d43159f41aa45f7f80726760efc8021e5a22352fe5.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/images/ffc2968208161ef2593d45d43159f41aa45f7f80726760efc8021e5a22352fe5.jpg)

### Tables

![0812fffcb1a8b468cab57d8dc8bce2d81b1dc2f358e798f9dc4bd915a77e0997.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/0812fffcb1a8b468cab57d8dc8bce2d81b1dc2f358e798f9dc4bd915a77e0997.jpg)

![2e150289ea601039a488264d2bfe3e70dde2c29f2814204a568fdddb384a6f8b.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/2e150289ea601039a488264d2bfe3e70dde2c29f2814204a568fdddb384a6f8b.jpg)

![345ccaa035a69b386b4608c7105475356c2a54929eb2bd1f704649de2391bc0f.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/345ccaa035a69b386b4608c7105475356c2a54929eb2bd1f704649de2391bc0f.jpg)

![34a1ad57288c4ec3e88fa922910a52c1945d3e73b9f129770b3099889252a46f.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/34a1ad57288c4ec3e88fa922910a52c1945d3e73b9f129770b3099889252a46f.jpg)

![37fcf2d5ec4acf83d1a94b6118d91f2288661bc40e6a661b4215de26f118a589.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/37fcf2d5ec4acf83d1a94b6118d91f2288661bc40e6a661b4215de26f118a589.jpg)

![561dc90dcc8cb0d938ae1443cb62c7721d99c15b27ddc05b30323948da777faf.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/561dc90dcc8cb0d938ae1443cb62c7721d99c15b27ddc05b30323948da777faf.jpg)

![5e8d8190c7e38fc621b4a3dd76c957c8ae2d2d039afe114df20db30349e6f64a.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/5e8d8190c7e38fc621b4a3dd76c957c8ae2d2d039afe114df20db30349e6f64a.jpg)

![689f68613a2d7cbefe1b40cb966003dd45e2dc3e34e338630d37de8429ee75ae.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/689f68613a2d7cbefe1b40cb966003dd45e2dc3e34e338630d37de8429ee75ae.jpg)

![790d52b2c4dcef93c35674560c334b02cbb098bada7fbfdc678447e13d47dcef.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/790d52b2c4dcef93c35674560c334b02cbb098bada7fbfdc678447e13d47dcef.jpg)

![7ec0769199422c8ea815498cf6a3e714bdd588f850b69af264ca15de3a8c6e49.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/7ec0769199422c8ea815498cf6a3e714bdd588f850b69af264ca15de3a8c6e49.jpg)

![85fb6d886358da1caea001b4820e357e557c2e91710e9b8b5c1f49781f0b9433.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/85fb6d886358da1caea001b4820e357e557c2e91710e9b8b5c1f49781f0b9433.jpg)

![884894c1b23def95bcc082ca080873fc1827c164b5bffc9c11ffa5e7f55794cd.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/884894c1b23def95bcc082ca080873fc1827c164b5bffc9c11ffa5e7f55794cd.jpg)

![8a40ed459ce73ab1857ba7704a0846bf5b7ffd6d9241a891389aa77712bba4da.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/8a40ed459ce73ab1857ba7704a0846bf5b7ffd6d9241a891389aa77712bba4da.jpg)

![9156b44200cf414e9cd904cde02fb5af125e4463393612f770a7f75454f91dc9.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/9156b44200cf414e9cd904cde02fb5af125e4463393612f770a7f75454f91dc9.jpg)

![a761df5a034fb8037cb39a4bac65bbdabf529bef4e8e3a3ef425b2851173a090.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/a761df5a034fb8037cb39a4bac65bbdabf529bef4e8e3a3ef425b2851173a090.jpg)

![aaf1795186de6b729ffaf98354dfff03db78627728fb9ca90946f7a35ec9f6a6.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/aaf1795186de6b729ffaf98354dfff03db78627728fb9ca90946f7a35ec9f6a6.jpg)

![ac79ec706904b7f36597fc0bfe8e1d7de83232aa9f75aae03eaa745c143d42cf.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/ac79ec706904b7f36597fc0bfe8e1d7de83232aa9f75aae03eaa745c143d42cf.jpg)

![b6672a744506e9db717f4da372aeb06b582061357ec7e8a4a402df9405b3e670.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/b6672a744506e9db717f4da372aeb06b582061357ec7e8a4a402df9405b3e670.jpg)

![bd4a0f6c66c66b38d9e1fc70baec9825cfce91d8faa2e0d5354c1dfa73add6c8.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/bd4a0f6c66c66b38d9e1fc70baec9825cfce91d8faa2e0d5354c1dfa73add6c8.jpg)

![d4dd97657a730cc65ae98b971c465768209beb4709fafd708e6cb07bbc9c1c6f.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/d4dd97657a730cc65ae98b971c465768209beb4709fafd708e6cb07bbc9c1c6f.jpg)

![daf4e813a94756b661a54652d26b2b1aad15fb89b4cd8b18896d8a044fdec178.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/daf4e813a94756b661a54652d26b2b1aad15fb89b4cd8b18896d8a044fdec178.jpg)

![ed21b0b5cff6d453c8f2e181337e8d77585ee9da4544a8a8a36785c2f6f2f6be.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/ed21b0b5cff6d453c8f2e181337e8d77585ee9da4544a8a8a36785c2f6f2f6be.jpg)

![fb5f04f38378e2c7ee1a423d7a94eac40f5db89469c6fe328c964ea717d55aa9.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/fb5f04f38378e2c7ee1a423d7a94eac40f5db89469c6fe328c964ea717d55aa9.jpg)

![fe18866efb3e71fb21ee6d9ea1d98f5de613d2dd4f7f4d1e682dd91863a268b1.jpg](../icml_results/1194_Towards%20Feedback-to-Plan%20Decisions%20for%20Self-Evolving%20LLM%20Agents%20in%20CUDA%20Kernel%20Generation/tables/fe18866efb3e71fb21ee6d9ea1d98f5de613d2dd4f7f4d1e682dd91863a268b1.jpg)

<a id="paper-1177"></a>

## Beyond Extrapolation: Knowledge Utilization Paradigm with Bidirectional Inspiration for Time Series Forecasting


### Images

![08f4f5161c2f1acc76ae637dc20a2ccb04da4a6e591814a7e222dabbb057919e.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/images/08f4f5161c2f1acc76ae637dc20a2ccb04da4a6e591814a7e222dabbb057919e.jpg)

![1460e4beffe75587c0e8fb87880926cff2f387fc802ba6ab05311b5af8b6043e.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/images/1460e4beffe75587c0e8fb87880926cff2f387fc802ba6ab05311b5af8b6043e.jpg)

![a56376c503f8128141242ddf459e9d2ae267f0d035da8647160ddd3d0f206556.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/images/a56376c503f8128141242ddf459e9d2ae267f0d035da8647160ddd3d0f206556.jpg)

![b65ccd01e560619fcc00a8a6936330a571156ad92e2e5586d17cf0c5de038975.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/images/b65ccd01e560619fcc00a8a6936330a571156ad92e2e5586d17cf0c5de038975.jpg)

![bc3a6248cc63dc1f8b08f2fe1193fdc0d01a57be0d029758e3b0beca079a9c35.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/images/bc3a6248cc63dc1f8b08f2fe1193fdc0d01a57be0d029758e3b0beca079a9c35.jpg)

![f11a16ff2b00ee8d98a28143988383e4877b274ed1870adb29cb34b6b9ef0d2d.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/images/f11a16ff2b00ee8d98a28143988383e4877b274ed1870adb29cb34b6b9ef0d2d.jpg)

### Tables

![2f12fd8ea87cb077351172e7f1bac19f83e9256ba4e98fe4313d1c6f4a60bfe2.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/tables/2f12fd8ea87cb077351172e7f1bac19f83e9256ba4e98fe4313d1c6f4a60bfe2.jpg)

![32447a826ad7617442fa2475f9a68bb84cd3f411c36e40d71103c59cd686f167.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/tables/32447a826ad7617442fa2475f9a68bb84cd3f411c36e40d71103c59cd686f167.jpg)

![4df0f71a1d0af0619cf2ffa89d6fd2f5fabc3e1f708690710c4915ceba851a39.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/tables/4df0f71a1d0af0619cf2ffa89d6fd2f5fabc3e1f708690710c4915ceba851a39.jpg)

![5383399be3d54928a2e0ab74c531b2b6367559a7df8f507a34c804bebd68712b.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/tables/5383399be3d54928a2e0ab74c531b2b6367559a7df8f507a34c804bebd68712b.jpg)

![5ad6d40a2c6450661e7cf1aae51905210907c5bfd5e1d8a1cefd3eed6be07ded.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/tables/5ad6d40a2c6450661e7cf1aae51905210907c5bfd5e1d8a1cefd3eed6be07ded.jpg)

![5f054848fd55bfafbed36797bb76337cabe9688158c397382a454d2e6a2b00ee.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/tables/5f054848fd55bfafbed36797bb76337cabe9688158c397382a454d2e6a2b00ee.jpg)

![66f7ab18ee7e907dbad17a6c012182884bab0ac97d40268c6e288fbf6d699811.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/tables/66f7ab18ee7e907dbad17a6c012182884bab0ac97d40268c6e288fbf6d699811.jpg)

![a321fa804098966ba89d55466ef80b20daef9cacee0cce7f9f06b70d5d92355e.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/tables/a321fa804098966ba89d55466ef80b20daef9cacee0cce7f9f06b70d5d92355e.jpg)

![ae51a9d2967be09f809eafdfc10c2507f9e500b7f46df7b96ae9919954779095.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/tables/ae51a9d2967be09f809eafdfc10c2507f9e500b7f46df7b96ae9919954779095.jpg)

![bedfb9cef3282b1e02ce88119fa85eed1101dfba98e252ba346604854185e1a3.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/tables/bedfb9cef3282b1e02ce88119fa85eed1101dfba98e252ba346604854185e1a3.jpg)

![cd87762abc1946521064db6be31b7ddd3a2145aa43d83f5e5496bfc9ada39810.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/tables/cd87762abc1946521064db6be31b7ddd3a2145aa43d83f5e5496bfc9ada39810.jpg)

![d7fdf57525354eb950363e8e04527dbe7b829b855db8468f41a23bcbea0246d9.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/tables/d7fdf57525354eb950363e8e04527dbe7b829b855db8468f41a23bcbea0246d9.jpg)

![e5c4b4faada6d14a360d6dfcbf9f364e5f02e27214fd0232bc3619d40a1f5562.jpg](../icml_results/1195_Beyond%20Extrapolation_%20Knowledge%20Utilization%20Paradigm%20with%20Bidirectional%20Inspiration%20for%20Time%20Series%20/tables/e5c4b4faada6d14a360d6dfcbf9f364e5f02e27214fd0232bc3619d40a1f5562.jpg)

<a id="paper-1178"></a>

## Approximate Proportionality in Online Fair Division


### Images

![a334c8931d1d8fc3759072f3d09defdef3bd1dc3d1725128394ef68512a03c16.jpg](../icml_results/1196_Approximate%20Proportionality%20in%20Online%20Fair%20Division/images/a334c8931d1d8fc3759072f3d09defdef3bd1dc3d1725128394ef68512a03c16.jpg)

![f45c46cdad334d66d279796b76ccc9bfe12c66e14995112838e2492b5fe9a345.jpg](../icml_results/1196_Approximate%20Proportionality%20in%20Online%20Fair%20Division/images/f45c46cdad334d66d279796b76ccc9bfe12c66e14995112838e2492b5fe9a345.jpg)

### Tables

![108653b6f418f8f958bc6b92abc3a93a6187c06543a2380c4c21c0d3a482e862.jpg](../icml_results/1196_Approximate%20Proportionality%20in%20Online%20Fair%20Division/tables/108653b6f418f8f958bc6b92abc3a93a6187c06543a2380c4c21c0d3a482e862.jpg)

![477799a4472f3f3c656439e0f0d17d0eb775423ecf36afcd650e15c8ce74a5b1.jpg](../icml_results/1196_Approximate%20Proportionality%20in%20Online%20Fair%20Division/tables/477799a4472f3f3c656439e0f0d17d0eb775423ecf36afcd650e15c8ce74a5b1.jpg)

<a id="paper-1179"></a>

## Learning Unanimously Acceptable Lotteries via Queries


### Images

![77d34cda9c1c30f9f315b98f36a51c4a0303a6972603762498aa3a1f7573b45b.jpg](../icml_results/1197_Learning%20Unanimously%20Acceptable%20Lotteries%20via%20Queries/images/77d34cda9c1c30f9f315b98f36a51c4a0303a6972603762498aa3a1f7573b45b.jpg)

![bed31cd84b5b168c2d1bce2a083f6b94f4c8ce231345f2950915d3e913ec28b4.jpg](../icml_results/1197_Learning%20Unanimously%20Acceptable%20Lotteries%20via%20Queries/images/bed31cd84b5b168c2d1bce2a083f6b94f4c8ce231345f2950915d3e913ec28b4.jpg)

### Tables

![1cceff14b5f11cd1f3c08cd40d9ab24b453fe8a3ffa255d3021d9f50c908952f.jpg](../icml_results/1197_Learning%20Unanimously%20Acceptable%20Lotteries%20via%20Queries/tables/1cceff14b5f11cd1f3c08cd40d9ab24b453fe8a3ffa255d3021d9f50c908952f.jpg)

<a id="paper-1180"></a>

## Computationally-efficient Graph Modeling with Refined Graph Random Features


### Images

![0dcb282737fde196a22272414f68391c52c01a9b4b6bdc8a3298f08ac8fd1ec2.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/images/0dcb282737fde196a22272414f68391c52c01a9b4b6bdc8a3298f08ac8fd1ec2.jpg)

![18aebeab35496e7fdf36e1e7b43cfead684eccae928126d0552ddd59b1ee829d.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/images/18aebeab35496e7fdf36e1e7b43cfead684eccae928126d0552ddd59b1ee829d.jpg)

![25cb249accc0fb11d50dc9ae3c78e6a72b09927854e0c0e0ebdb86aab10a3cf6.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/images/25cb249accc0fb11d50dc9ae3c78e6a72b09927854e0c0e0ebdb86aab10a3cf6.jpg)

![459bef807362376fba62f4121d6a532c2d9c5cbfa675a9a02eb0ed1f9675062f.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/images/459bef807362376fba62f4121d6a532c2d9c5cbfa675a9a02eb0ed1f9675062f.jpg)

![61f08f8a7ea6d67c4399242822dd68895be0a6253f29ad8ede5ec456a5b29ef4.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/images/61f08f8a7ea6d67c4399242822dd68895be0a6253f29ad8ede5ec456a5b29ef4.jpg)

![852fecf455544cbae0230626a1954e14876096fe35fade8be58147f8caddc220.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/images/852fecf455544cbae0230626a1954e14876096fe35fade8be58147f8caddc220.jpg)

![a385de385cce74dd4ef0a063b8fe2d9d0b41f19fe157176e5a6d732052de820d.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/images/a385de385cce74dd4ef0a063b8fe2d9d0b41f19fe157176e5a6d732052de820d.jpg)

![b5646c004a5a9229418fd02bb9c01770ab080c1abf6866acb48d49e341a2eba2.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/images/b5646c004a5a9229418fd02bb9c01770ab080c1abf6866acb48d49e341a2eba2.jpg)

![c8369b6b2662e1399c4e8d9300416eb063713f79ae825a253577d2a582b523be.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/images/c8369b6b2662e1399c4e8d9300416eb063713f79ae825a253577d2a582b523be.jpg)

![cd7dcadd42d94a393c47513371da2d125d9a9ddb9514145507bef2c50388d7cf.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/images/cd7dcadd42d94a393c47513371da2d125d9a9ddb9514145507bef2c50388d7cf.jpg)

![d6ba45ae03d45705530521c62c500568930ae50df91e0e2e8a9f027356336de5.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/images/d6ba45ae03d45705530521c62c500568930ae50df91e0e2e8a9f027356336de5.jpg)

### Tables

![01faadcbc5ee8eb2ffab939153fa4ae5e2c690ca7e5c49af0c6f300925bd7b19.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/tables/01faadcbc5ee8eb2ffab939153fa4ae5e2c690ca7e5c49af0c6f300925bd7b19.jpg)

![409e14d811dd29d7159ef0a5a0cbc46225a1e5c256a24aacd578ef1c928857f3.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/tables/409e14d811dd29d7159ef0a5a0cbc46225a1e5c256a24aacd578ef1c928857f3.jpg)

![520c250cd622d9de5410ee63321d07c858ac23705c8112c6bb511a1982055e32.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/tables/520c250cd622d9de5410ee63321d07c858ac23705c8112c6bb511a1982055e32.jpg)

![651b62143c41286092de705dd0d105a72f3bc0d55fce4c805159fbe1b87c3c7a.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/tables/651b62143c41286092de705dd0d105a72f3bc0d55fce4c805159fbe1b87c3c7a.jpg)

![66a73c2cae481c86a9feb6a0d6df16685de9114e19d0906ba24e4392a85bad57.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/tables/66a73c2cae481c86a9feb6a0d6df16685de9114e19d0906ba24e4392a85bad57.jpg)

![6dc08ae6513df411a91acca57dda1a3d9d242c95258e99d97d7a7246221f4047.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/tables/6dc08ae6513df411a91acca57dda1a3d9d242c95258e99d97d7a7246221f4047.jpg)

![6e34da7be18742af2b8aeb5dec482f328f46efe5d3fe45272cbfd38df7ac0867.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/tables/6e34da7be18742af2b8aeb5dec482f328f46efe5d3fe45272cbfd38df7ac0867.jpg)

![c30dd2b5cb86ed6d75e96572edbbc2ab160bc8a7c89320ed3ed7512ce8f074b1.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/tables/c30dd2b5cb86ed6d75e96572edbbc2ab160bc8a7c89320ed3ed7512ce8f074b1.jpg)

![ca0138675c7881b80ee3d942d033d1339c70fdf220825ea4bd0ef788ccb913a1.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/tables/ca0138675c7881b80ee3d942d033d1339c70fdf220825ea4bd0ef788ccb913a1.jpg)

![d5534044c0a92591aea7553bd8046877cad2560f39428758c22c6f74004935d0.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/tables/d5534044c0a92591aea7553bd8046877cad2560f39428758c22c6f74004935d0.jpg)

![edee49a118e701af864f8125db2e8b99ba6d8f247db250a2d76c3c67d8c103c7.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/tables/edee49a118e701af864f8125db2e8b99ba6d8f247db250a2d76c3c67d8c103c7.jpg)

![fc1243fda4afd4db051062a5628aba30402af4e650e59968307be16adad35834.jpg](../icml_results/1198_Computationally-efficient%20Graph%20Modeling%20with%20Refined%20Graph%20Random%20Features/tables/fc1243fda4afd4db051062a5628aba30402af4e650e59968307be16adad35834.jpg)

<a id="paper-1181"></a>

## Learning When to Attend: Conditional Memory Access for Long-Context LLMs


### Images

![0f762121dd5478615875e606212aca5bf509d7ca17c7a3a963630e8c37db49d1.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/0f762121dd5478615875e606212aca5bf509d7ca17c7a3a963630e8c37db49d1.jpg)

![3ea6133462977f6af122b9be45a339ce451b65651321d4c5e13c5a72c56f2ca5.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/3ea6133462977f6af122b9be45a339ce451b65651321d4c5e13c5a72c56f2ca5.jpg)

![4768599b75c2d94d2a3bfad8ebb49d23a3c1cc71c53efc23cbc6f2a7f63352c0.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/4768599b75c2d94d2a3bfad8ebb49d23a3c1cc71c53efc23cbc6f2a7f63352c0.jpg)

![5d15642f791c24471fc8c2f57a23f1b4bada1993536e5533dc23356885d5e2e9.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/5d15642f791c24471fc8c2f57a23f1b4bada1993536e5533dc23356885d5e2e9.jpg)

![5ddceede50297600577bf707026d0872eea3c318e1239624364e5cb4f59ea742.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/5ddceede50297600577bf707026d0872eea3c318e1239624364e5cb4f59ea742.jpg)

![6a2ca067ec8ec79fd07f078d376758a56e46b1bbd3254f2407646ebe4ca7a4d9.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/6a2ca067ec8ec79fd07f078d376758a56e46b1bbd3254f2407646ebe4ca7a4d9.jpg)

![6ed0d2b03cb9b7e51463d4be9771fcd5953bfc3bb5590dcac7f3a91ecc9dd1c5.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/6ed0d2b03cb9b7e51463d4be9771fcd5953bfc3bb5590dcac7f3a91ecc9dd1c5.jpg)

![71f67a74e859081bb7728a299b060be888b7bd0b6951a499cfa062a0b4e4d0ca.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/71f67a74e859081bb7728a299b060be888b7bd0b6951a499cfa062a0b4e4d0ca.jpg)

![7b2854399537d3fb85e375d2aea3da9cee3e4c7b893aa207b235e04041d0d1f4.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/7b2854399537d3fb85e375d2aea3da9cee3e4c7b893aa207b235e04041d0d1f4.jpg)

![9733a622581b2503b58f75f7bc984497d5d03f87aff3c3995cc421c901c43f05.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/9733a622581b2503b58f75f7bc984497d5d03f87aff3c3995cc421c901c43f05.jpg)

![a79a5d328a8a111cf68d79c9b12e0ee978e425544902f237230d3a9dd0dc38d2.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/a79a5d328a8a111cf68d79c9b12e0ee978e425544902f237230d3a9dd0dc38d2.jpg)

![c0ed1aca0925e7bdb8d646e5943ddfde587bc107d9e481666c271d4984371ba7.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/c0ed1aca0925e7bdb8d646e5943ddfde587bc107d9e481666c271d4984371ba7.jpg)

![da07f933e084e22890b46d057e94426521b2c3f96fca2618e13d052cf6580f39.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/da07f933e084e22890b46d057e94426521b2c3f96fca2618e13d052cf6580f39.jpg)

![da340b77ab617a549d8f458e483567428e915c91e018a1a9a12242373d626811.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/da340b77ab617a549d8f458e483567428e915c91e018a1a9a12242373d626811.jpg)

![e7deb6b16d1cff5924a22d55e453cdb216f1a7e9440fe52e5dbc7b80a63541ec.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/e7deb6b16d1cff5924a22d55e453cdb216f1a7e9440fe52e5dbc7b80a63541ec.jpg)

![eb654117f82795384738f68ac7e9798c83b1f198086ba07a09459cdabb8fc560.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/eb654117f82795384738f68ac7e9798c83b1f198086ba07a09459cdabb8fc560.jpg)

![ebf5ca8d3a753ac89eebea15b7dbe1672faff530ce1a05820140f02e5f0ea6a6.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/ebf5ca8d3a753ac89eebea15b7dbe1672faff530ce1a05820140f02e5f0ea6a6.jpg)

![edc88d97a3997f26cc89349386ae442acb45984163c012cd9edc71ea9980999f.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/edc88d97a3997f26cc89349386ae442acb45984163c012cd9edc71ea9980999f.jpg)

![f6e01602946a8934e273fc0da0803f03b851db6bfc63eb68f88978c9d1b3dd17.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/images/f6e01602946a8934e273fc0da0803f03b851db6bfc63eb68f88978c9d1b3dd17.jpg)

### Tables

![32493fb0fbc0c8aaeb62f974aebab8f7f9f97315c20f7e1fc3c904d98127d45f.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/tables/32493fb0fbc0c8aaeb62f974aebab8f7f9f97315c20f7e1fc3c904d98127d45f.jpg)

![4230e71d77cebbcc988a23ae65a3001c6fe812ebe2b1466a1a50d3091cbeb0ee.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/tables/4230e71d77cebbcc988a23ae65a3001c6fe812ebe2b1466a1a50d3091cbeb0ee.jpg)

![8017b798b3bcc505a742dd78faba42c1b947f5571721bebc33ff980d7a828c0d.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/tables/8017b798b3bcc505a742dd78faba42c1b947f5571721bebc33ff980d7a828c0d.jpg)

![87a62ec048f62ffb528b876895e2b7e85872e0297d2b955814202b48719f70b7.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/tables/87a62ec048f62ffb528b876895e2b7e85872e0297d2b955814202b48719f70b7.jpg)

![910e4142fa01836785708f2f16e0e059771dccfe01e17fb30e6a70f0ea98fc81.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/tables/910e4142fa01836785708f2f16e0e059771dccfe01e17fb30e6a70f0ea98fc81.jpg)

![c63c21acd87381abbf3999905f74a26afdbaffff6bcbfc6f4a9cd5316bdb5cf2.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/tables/c63c21acd87381abbf3999905f74a26afdbaffff6bcbfc6f4a9cd5316bdb5cf2.jpg)

![d9d4f83956d42d78bd9914591ff91d0bb905d7f61c4343b4e3a96c88bb6d919f.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/tables/d9d4f83956d42d78bd9914591ff91d0bb905d7f61c4343b4e3a96c88bb6d919f.jpg)

![e9dbc23937612b1a084dc5a3b40211efd7dfc825330fe0ad4b06c5eb585070bf.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/tables/e9dbc23937612b1a084dc5a3b40211efd7dfc825330fe0ad4b06c5eb585070bf.jpg)

![eb5a361a266bf8603047e84cfd47ee4b758a36053a2af11efcadf9e5dd857dae.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/tables/eb5a361a266bf8603047e84cfd47ee4b758a36053a2af11efcadf9e5dd857dae.jpg)

![f3bc734793ed56a8512e62c3eb6dfd3a3e15d99d09baaeb320b4461113705322.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/tables/f3bc734793ed56a8512e62c3eb6dfd3a3e15d99d09baaeb320b4461113705322.jpg)

![f95d7652c3040716780779d51703a9bf90d9d017e436bf7f6d64863de0351b18.jpg](../icml_results/1199_Learning%20When%20to%20Attend_%20Conditional%20Memory%20Access%20for%20Long-Context%20LLMs/tables/f95d7652c3040716780779d51703a9bf90d9d017e436bf7f6d64863de0351b18.jpg)

<a id="paper-1182"></a>

## Through the Stealth Lens: Attention-Aware Defenses Against Poisoning in RAG


### Images

![194ec5c4291a3203230ba49e8a150e9adacaf1bf97c08ed0c58c897242f0a75a.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/images/194ec5c4291a3203230ba49e8a150e9adacaf1bf97c08ed0c58c897242f0a75a.jpg)

![507123548b236ecbd9e123068a10774a0a516ab0c1fc74bbbf79e0c97e071169.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/images/507123548b236ecbd9e123068a10774a0a516ab0c1fc74bbbf79e0c97e071169.jpg)

![65ef834a2ac8cb26a4a9e857d048889a6625ef31700cf838fdb8b899cdc6171d.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/images/65ef834a2ac8cb26a4a9e857d048889a6625ef31700cf838fdb8b899cdc6171d.jpg)

![df97d5da13bf3b4ed8a48b07f9d4ab6d9e6d08ae48068c06649032c0701ceb5b.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/images/df97d5da13bf3b4ed8a48b07f9d4ab6d9e6d08ae48068c06649032c0701ceb5b.jpg)

![f6a2b112cabc984c76283729ad42f6788b044d86d4e5ec3f5a6835651bb37bc3.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/images/f6a2b112cabc984c76283729ad42f6788b044d86d4e5ec3f5a6835651bb37bc3.jpg)

### Tables

![1285577721c8f632740eff2e08e018c9218c9e53c006d95f8e6c13721bb919ac.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/tables/1285577721c8f632740eff2e08e018c9218c9e53c006d95f8e6c13721bb919ac.jpg)

![15a1409246a708de27f41ab5262bc55458a27b1009fabce8374d02406d518922.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/tables/15a1409246a708de27f41ab5262bc55458a27b1009fabce8374d02406d518922.jpg)

![1e60ecb1caff12fcfc2d35fa844b9c0d526ad57d478467670ca246c0b34bd509.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/tables/1e60ecb1caff12fcfc2d35fa844b9c0d526ad57d478467670ca246c0b34bd509.jpg)

![3265d07a1e4d9f6eb9e709ab0dccad77aafb60127a01f6baee23398e4f0bab06.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/tables/3265d07a1e4d9f6eb9e709ab0dccad77aafb60127a01f6baee23398e4f0bab06.jpg)

![33ae96457ccff25205e111db0825e5c030aed43c1a3201759a122007354676ad.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/tables/33ae96457ccff25205e111db0825e5c030aed43c1a3201759a122007354676ad.jpg)

![3708a2d79d460e3464f6ce9bf5a99d07e27ecf7c72185f4c2c7f5c82c51b3cc4.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/tables/3708a2d79d460e3464f6ce9bf5a99d07e27ecf7c72185f4c2c7f5c82c51b3cc4.jpg)

![502cc224ace5aedabf8e90911ae7790589221af53e48d78d5cdbad21e17b9f19.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/tables/502cc224ace5aedabf8e90911ae7790589221af53e48d78d5cdbad21e17b9f19.jpg)

![5f258e4aa8d60d07c827b1bc25db43e91c77214007322d87e7cc71b0587d6772.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/tables/5f258e4aa8d60d07c827b1bc25db43e91c77214007322d87e7cc71b0587d6772.jpg)

![6bd56e2ab610aec83bc490d340d0a16a30d3a77b6f8bba0cf61003d6beedca60.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/tables/6bd56e2ab610aec83bc490d340d0a16a30d3a77b6f8bba0cf61003d6beedca60.jpg)

![76f85f4dc283721f42389878322fa5f7275fa6fbe343fbd75ad25418f29171fb.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/tables/76f85f4dc283721f42389878322fa5f7275fa6fbe343fbd75ad25418f29171fb.jpg)

![87b4c2ced5441bf0c4b890af73b11c7626edd8cc3d2a1f2b150b3353a8718f5a.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/tables/87b4c2ced5441bf0c4b890af73b11c7626edd8cc3d2a1f2b150b3353a8718f5a.jpg)

![95c6a0cdced71882303fb04d8ed6c67567ba1ea21cd907dc4528b1206dc2bc2c.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/tables/95c6a0cdced71882303fb04d8ed6c67567ba1ea21cd907dc4528b1206dc2bc2c.jpg)

![ab0c973ab08464011c2e28473705249bdfbd909550d91dafce00b947d3b1bf50.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/tables/ab0c973ab08464011c2e28473705249bdfbd909550d91dafce00b947d3b1bf50.jpg)

![bbb1864b047760571af3b9a252c7d84af7f3db26b67e79e1b93f92685ee828ec.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/tables/bbb1864b047760571af3b9a252c7d84af7f3db26b67e79e1b93f92685ee828ec.jpg)

![fc167e87c792944c232b6ce618c1ffc00353794ac5b9944b4ddadabd02c5690f.jpg](../icml_results/1200_Through%20the%20Stealth%20Lens_%20Attention-Aware%20Defenses%20Against%20Poisoning%20in%20RAG/tables/fc167e87c792944c232b6ce618c1ffc00353794ac5b9944b4ddadabd02c5690f.jpg)

<a id="paper-1183"></a>

## LAGEA: Language Guided Embodied Agents for Robotic Manipulation


### Images

![07dc88d73c54f986343e395460e1ee74224d47a15ea02df859e825d495e9f1e9.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/images/07dc88d73c54f986343e395460e1ee74224d47a15ea02df859e825d495e9f1e9.jpg)

![1c3ec755ad6d2f4dbba387a526298f4597e404add6b7835c65864176d6a316ee.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/images/1c3ec755ad6d2f4dbba387a526298f4597e404add6b7835c65864176d6a316ee.jpg)

![495e109a2cffa68033083a60814400b6db1e152aaa54f39d2edc17b991c1de5f.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/images/495e109a2cffa68033083a60814400b6db1e152aaa54f39d2edc17b991c1de5f.jpg)

![73c2e67df2792f36471cb8506b291ee3144c647867fd7a4ec70126a8518dd399.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/images/73c2e67df2792f36471cb8506b291ee3144c647867fd7a4ec70126a8518dd399.jpg)

![965d69e6c6a1227ef7e0a72331d1826788b59fc38f04bf5c20c117b90aee0bc1.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/images/965d69e6c6a1227ef7e0a72331d1826788b59fc38f04bf5c20c117b90aee0bc1.jpg)

![9ffc98e658eee171cd329dad182d6e47912c2d9d081c7b58706e1c125dfe9499.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/images/9ffc98e658eee171cd329dad182d6e47912c2d9d081c7b58706e1c125dfe9499.jpg)

![b953f416419b2491ce1c4912afe16c3428f63bb07f439ae4b75bad2d9c3ccbe2.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/images/b953f416419b2491ce1c4912afe16c3428f63bb07f439ae4b75bad2d9c3ccbe2.jpg)

![d8e62ca3d19a71ad91d99e1f0337696f2717b7f2216e1fecfd542e624b00ade2.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/images/d8e62ca3d19a71ad91d99e1f0337696f2717b7f2216e1fecfd542e624b00ade2.jpg)

![dbda235d2f3381e243b1a9aea3ee0e0784462bc704296fd1aa02c81524e997ed.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/images/dbda235d2f3381e243b1a9aea3ee0e0784462bc704296fd1aa02c81524e997ed.jpg)

![f0b91d8cf2711dbdb03e836929b924eb134def6d8e2a776e76d527ec67fe5b06.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/images/f0b91d8cf2711dbdb03e836929b924eb134def6d8e2a776e76d527ec67fe5b06.jpg)

### Tables

![6d5c30e3c61a9e7d5d7995fc6dd9564fd08e8a7aeb82daaf182915a3142b6b97.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/tables/6d5c30e3c61a9e7d5d7995fc6dd9564fd08e8a7aeb82daaf182915a3142b6b97.jpg)

![783d1b3e5ffe0a3279166f531f7056033c000d004a4d95c5cd3aedcca6b52585.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/tables/783d1b3e5ffe0a3279166f531f7056033c000d004a4d95c5cd3aedcca6b52585.jpg)

![829d3a9a7e2a01d58b04ec2a53bd08587c35a44cd4f7c71b6413cb290ea9cc78.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/tables/829d3a9a7e2a01d58b04ec2a53bd08587c35a44cd4f7c71b6413cb290ea9cc78.jpg)

![85166abbac217c88bf21971bf29a44aab3066646a77fc6287f65cadd6a297725.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/tables/85166abbac217c88bf21971bf29a44aab3066646a77fc6287f65cadd6a297725.jpg)

![96f8691496fe9d0264701acbb1f4ed2e7b3014c18dfba226a86b40a4b6124f0a.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/tables/96f8691496fe9d0264701acbb1f4ed2e7b3014c18dfba226a86b40a4b6124f0a.jpg)

![97589165b271053d650e29237abecd0b04fdb1bd7f6149e4572defb7cd97fae5.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/tables/97589165b271053d650e29237abecd0b04fdb1bd7f6149e4572defb7cd97fae5.jpg)

![9d3c03004f79eddfbfd42ac5d6fa3a996d43ddd5fa3846875f5e3f1c50ebab38.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/tables/9d3c03004f79eddfbfd42ac5d6fa3a996d43ddd5fa3846875f5e3f1c50ebab38.jpg)

![d6588438b9e639a790cbfaf8aa860de241daafcda743667238fc5d03b4ce6446.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/tables/d6588438b9e639a790cbfaf8aa860de241daafcda743667238fc5d03b4ce6446.jpg)

![da1d48ea2fc8c5e42d2d9ce1ae7adbdf8463d08e47f0726bf08768ab1519693c.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/tables/da1d48ea2fc8c5e42d2d9ce1ae7adbdf8463d08e47f0726bf08768ab1519693c.jpg)

![e8f5c633c4746887910d1857cc58a71baad3ed778bec06ca25c445da6f9bf5c2.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/tables/e8f5c633c4746887910d1857cc58a71baad3ed778bec06ca25c445da6f9bf5c2.jpg)

![ee3c6f9b62761afce5dc53dc390a1e19d06f36a13749af4a118dd1a56f07721a.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/tables/ee3c6f9b62761afce5dc53dc390a1e19d06f36a13749af4a118dd1a56f07721a.jpg)

![f04a4c2c4ee108736c690215b552ab477995d68de10fbd54618b769a8183ea37.jpg](../icml_results/1201_LAGEA_%20Language%20Guided%20Embodied%20Agents%20for%20Robotic%20Manipulation/tables/f04a4c2c4ee108736c690215b552ab477995d68de10fbd54618b769a8183ea37.jpg)

<a id="paper-1184"></a>

## S2M-Net: Spectral-Spatial Mixing with Morphology-Aware Adaptive Loss for Medical Image Segmentation


### Images

![2b7c60986c10c475376fb6c92c5f2b9360b737e690547087ff1e5c3295cfb98a.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/images/2b7c60986c10c475376fb6c92c5f2b9360b737e690547087ff1e5c3295cfb98a.jpg)

![4ee94071a5528deabbf93d4e4f88eb32d8f66bb336f939048c843a5467dfbc19.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/images/4ee94071a5528deabbf93d4e4f88eb32d8f66bb336f939048c843a5467dfbc19.jpg)

![afd1a7f3bef1410acc65d273b852d27f9988c5fa22e6680a7cbc7dec48f052cd.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/images/afd1a7f3bef1410acc65d273b852d27f9988c5fa22e6680a7cbc7dec48f052cd.jpg)

![b3801e6c411b9eaa364ce1340f9172054504476a50dc76f66dc40bdb9a521057.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/images/b3801e6c411b9eaa364ce1340f9172054504476a50dc76f66dc40bdb9a521057.jpg)

![c20a4f5864e4bc0c373295e8761c7465faf6351a61532c3569efbd2cb6206cc3.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/images/c20a4f5864e4bc0c373295e8761c7465faf6351a61532c3569efbd2cb6206cc3.jpg)

![c76330f5d0b5c93c0c1a81957f37daa026d65be7c73d0387eecdba070540a58f.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/images/c76330f5d0b5c93c0c1a81957f37daa026d65be7c73d0387eecdba070540a58f.jpg)

![c7ec7ef9ce32f9e6eab7f4f93d499e21e56cba7b6218cbc03450ec5c13963887.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/images/c7ec7ef9ce32f9e6eab7f4f93d499e21e56cba7b6218cbc03450ec5c13963887.jpg)

![cd3cce68057250a9332e5f00440827fdcdd7d92ee70de097adf8f03f06c5e50c.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/images/cd3cce68057250a9332e5f00440827fdcdd7d92ee70de097adf8f03f06c5e50c.jpg)

![f4f9b4e9e0c1d10380b03dc8fc23d2564514841427ef243505c2395a2b2bb1c2.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/images/f4f9b4e9e0c1d10380b03dc8fc23d2564514841427ef243505c2395a2b2bb1c2.jpg)

### Tables

![03cb293b6360b5a988b315167ff364baa0ebb5428c0dac8bfad8332693269076.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/03cb293b6360b5a988b315167ff364baa0ebb5428c0dac8bfad8332693269076.jpg)

![103bc277b45d55b3825a35e16beb6c50351b9008bcb06ff4937ff2ec72bb1707.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/103bc277b45d55b3825a35e16beb6c50351b9008bcb06ff4937ff2ec72bb1707.jpg)

![13fc1ce70decae952bb8b7468d51145cb8c217400ff09d28d8c50332b669bfbf.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/13fc1ce70decae952bb8b7468d51145cb8c217400ff09d28d8c50332b669bfbf.jpg)

![1bb4f20f8abff7be4b82dfdf3f000c9ebc778204f5cd2e784d15d62bedc59235.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/1bb4f20f8abff7be4b82dfdf3f000c9ebc778204f5cd2e784d15d62bedc59235.jpg)

![2350e33228306b6792a9bb79abf9b492a1e5d6fc6718e775ce4ffcb06f2ca315.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/2350e33228306b6792a9bb79abf9b492a1e5d6fc6718e775ce4ffcb06f2ca315.jpg)

![2a78096d1f624e80d4e28eb8d9bf39b0a5a64aa7639574f1da8ba4c26d310733.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/2a78096d1f624e80d4e28eb8d9bf39b0a5a64aa7639574f1da8ba4c26d310733.jpg)

![316468d23ffaaec9b886de9e276404139bbb17aa0263d40797d6e567d769c5dc.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/316468d23ffaaec9b886de9e276404139bbb17aa0263d40797d6e567d769c5dc.jpg)

![32cfd721c7c00de2f6c8a51c34ac0de15e70ee2b8a13ce041ea90f0eca882e9e.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/32cfd721c7c00de2f6c8a51c34ac0de15e70ee2b8a13ce041ea90f0eca882e9e.jpg)

![356f3a943335f79349293fd539b9ef27f9acd6a9b78e02a2adcaef4568295ce8.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/356f3a943335f79349293fd539b9ef27f9acd6a9b78e02a2adcaef4568295ce8.jpg)

![48dafb9ff9b2f9d8fb6ef6ab2acad9ccd3e923ec591edf39a30830f41969f50b.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/48dafb9ff9b2f9d8fb6ef6ab2acad9ccd3e923ec591edf39a30830f41969f50b.jpg)

![4a2fc8c7df6e3d627bff673ab44804e92e7135576a85c3d10a57fd636a8749d8.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/4a2fc8c7df6e3d627bff673ab44804e92e7135576a85c3d10a57fd636a8749d8.jpg)

![771173e77a66792a45230b7a092ba86a48848b859d913ab739b75b69c08dc1de.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/771173e77a66792a45230b7a092ba86a48848b859d913ab739b75b69c08dc1de.jpg)

![86c882ef075c22b90970b31f4b0604f53be4a49c9e03fdd388e6ad4188f93369.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/86c882ef075c22b90970b31f4b0604f53be4a49c9e03fdd388e6ad4188f93369.jpg)

![8d9a472c206b96107dc9518782593ec6a72e629466abd6b771d0877e7d563f06.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/8d9a472c206b96107dc9518782593ec6a72e629466abd6b771d0877e7d563f06.jpg)

![96519de4a32f0170a76e7871fcd22922a5bbbc0c29caf99f0e34c6e06d27a0a5.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/96519de4a32f0170a76e7871fcd22922a5bbbc0c29caf99f0e34c6e06d27a0a5.jpg)

![98df6fcc592168a04c3e123bbee3285ed944b4a0e01ceeb2ef2437c5a44c889c.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/98df6fcc592168a04c3e123bbee3285ed944b4a0e01ceeb2ef2437c5a44c889c.jpg)

![bac412b4059f18fca8d283a667da5ff612bc32c24fda98a34c7cb7c03086b17e.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/bac412b4059f18fca8d283a667da5ff612bc32c24fda98a34c7cb7c03086b17e.jpg)

![bc5f63d5f2d1912292a24b142549f8036cbd9f1187ab22d5cc24bf767cdf23c9.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/bc5f63d5f2d1912292a24b142549f8036cbd9f1187ab22d5cc24bf767cdf23c9.jpg)

![cc021d2058f9c1d8578e9084a730889f5e388222f515592676af150869d46067.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/cc021d2058f9c1d8578e9084a730889f5e388222f515592676af150869d46067.jpg)

![e9d5fee20f064dfd2180df6ce481eb44d0d0d93baab8446489202145f2b8c9b6.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/e9d5fee20f064dfd2180df6ce481eb44d0d0d93baab8446489202145f2b8c9b6.jpg)

![eea31d22b410f565770f7282feb52aa81a5589c8fb818036a230c3c56bf97fbd.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/eea31d22b410f565770f7282feb52aa81a5589c8fb818036a230c3c56bf97fbd.jpg)

![eebe19956ba027c44f35440982c515b0e93bd788eb12152520407b2f7e9494fe.jpg](../icml_results/1202_S2M-Net_%20Spectral-Spatial%20Mixing%20with%20Morphology-Aware%20Adaptive%20Loss%20for%20Medical%20Image%20Segmentation/tables/eebe19956ba027c44f35440982c515b0e93bd788eb12152520407b2f7e9494fe.jpg)

<a id="paper-1185"></a>

## SpaCeFormer: Fast Proposal-Free Open-Vocabulary 3D Instance Segmentation


### Images

![0033564c45127018c656c3b4510faa441f646df7e2d15be37a7be582512775ae.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/0033564c45127018c656c3b4510faa441f646df7e2d15be37a7be582512775ae.jpg)

![02b2789399161460b30c7cbe17b5bfdaa2b4474ad610946143870ead7e5e9671.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/02b2789399161460b30c7cbe17b5bfdaa2b4474ad610946143870ead7e5e9671.jpg)

![05e9d7f2642702ecbd68e255fda0c5c1ef7819b6496d85dc8ea50cd18a1849f8.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/05e9d7f2642702ecbd68e255fda0c5c1ef7819b6496d85dc8ea50cd18a1849f8.jpg)

![150ab24a1ff25395c82bfee970021775e80a592ddddb9c1ecb345b148c2ac2b1.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/150ab24a1ff25395c82bfee970021775e80a592ddddb9c1ecb345b148c2ac2b1.jpg)

![1d2548f2c82e37207cc50f62a18049c5f0bd739c3b19dbe7906de21d2f325f24.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/1d2548f2c82e37207cc50f62a18049c5f0bd739c3b19dbe7906de21d2f325f24.jpg)

![2a7f83809f763f6b80ddafb87e03aa7241ea343fbaed8ec554830c043d118c87.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/2a7f83809f763f6b80ddafb87e03aa7241ea343fbaed8ec554830c043d118c87.jpg)

![34d16851e56fbccb39c63e6ed086b2983668630c25b034e8732628763c8dde6b.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/34d16851e56fbccb39c63e6ed086b2983668630c25b034e8732628763c8dde6b.jpg)

![377d714b81a45342f5b78b417fa49edaedd18d37135f38bc1d16f9289af2c270.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/377d714b81a45342f5b78b417fa49edaedd18d37135f38bc1d16f9289af2c270.jpg)

![3d61bf35e413104e3e19673104e608da77a0df1d1654f52c260fbb80d9de8b40.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/3d61bf35e413104e3e19673104e608da77a0df1d1654f52c260fbb80d9de8b40.jpg)

![4b1a5bfcc6e45bc9d8353587c813a9aefe140bd33135fb96421e69d270779827.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/4b1a5bfcc6e45bc9d8353587c813a9aefe140bd33135fb96421e69d270779827.jpg)

![686b59ae8eefe99c81aaac0bae7f4bac72eb8a31ffd0e6fbb1bfcf901422eff8.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/686b59ae8eefe99c81aaac0bae7f4bac72eb8a31ffd0e6fbb1bfcf901422eff8.jpg)

![68b8fba57b3b5e5e2f5e7302448dbf4c3f772e0efcba9fde9861693935ae8a03.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/68b8fba57b3b5e5e2f5e7302448dbf4c3f772e0efcba9fde9861693935ae8a03.jpg)

![6fa3d8db6443ab91ad20a4f6d5437479ed1ec526b224a8fa3f0ff974f4a638b2.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/6fa3d8db6443ab91ad20a4f6d5437479ed1ec526b224a8fa3f0ff974f4a638b2.jpg)

![8b789025e1e73b8669050072558295c2bf6c3040e7e8c08b06305a99ae5245dd.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/8b789025e1e73b8669050072558295c2bf6c3040e7e8c08b06305a99ae5245dd.jpg)

![8d294c2adec0866dde64b56018d0f4186ff6e1bb5e8bfd3ad6ac040faaa42511.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/8d294c2adec0866dde64b56018d0f4186ff6e1bb5e8bfd3ad6ac040faaa42511.jpg)

![9cc9c5f8afe06042f1c4b675839ab52a996aa91d631a0f9d9a81faa4da12546a.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/9cc9c5f8afe06042f1c4b675839ab52a996aa91d631a0f9d9a81faa4da12546a.jpg)

![b27c4037a216e87f19faf81a453c4ae1cf7695a308cb807d5a6728d51314f1b5.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/b27c4037a216e87f19faf81a453c4ae1cf7695a308cb807d5a6728d51314f1b5.jpg)

![b33281178d83e6711d1f353ce4ecadf7e1d369a477394abecf8bf16c0fed4cc6.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/b33281178d83e6711d1f353ce4ecadf7e1d369a477394abecf8bf16c0fed4cc6.jpg)

![b8f1601f23729cf2896ed67d4471844be236e0c50d8e3adce4859618d54d14e5.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/b8f1601f23729cf2896ed67d4471844be236e0c50d8e3adce4859618d54d14e5.jpg)

![b9d1560bb4f77e5900a96e1a8cffc75b3ec346a8c3dedc96501c205f349c74c7.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/b9d1560bb4f77e5900a96e1a8cffc75b3ec346a8c3dedc96501c205f349c74c7.jpg)

![c377cc6d9ce96358abb735e35813812a9223d143a44b044dc91670c6272f08a5.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/c377cc6d9ce96358abb735e35813812a9223d143a44b044dc91670c6272f08a5.jpg)

![c8fe46cd2c77f0932ca2e64facf221f4c5dd0e4d278e3a95758c5e18bd0e8500.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/c8fe46cd2c77f0932ca2e64facf221f4c5dd0e4d278e3a95758c5e18bd0e8500.jpg)

![cf886e1a4835b34fa0ee2e4d1e11d191d389dfbcbdeacca458d78d5e4e2b5392.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/cf886e1a4835b34fa0ee2e4d1e11d191d389dfbcbdeacca458d78d5e4e2b5392.jpg)

![d40e3308d28ae203a56cee01fa2b1aff05edd8120e7eab3201a969e929c67054.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/d40e3308d28ae203a56cee01fa2b1aff05edd8120e7eab3201a969e929c67054.jpg)

![d81bb1f7a29dce58f25b4ab5afd4e91b499716e08d19ea6078ac40b141758d57.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/d81bb1f7a29dce58f25b4ab5afd4e91b499716e08d19ea6078ac40b141758d57.jpg)

![dc02ea6b9abef3fede751f32b1e5e18c526d6e92cefc1fb7bc0d03d67d4aeb10.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/dc02ea6b9abef3fede751f32b1e5e18c526d6e92cefc1fb7bc0d03d67d4aeb10.jpg)

![e6172f687e2f95c3538b9ec47a0af1c64dad26a97c1ab369523242de86823a17.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/e6172f687e2f95c3538b9ec47a0af1c64dad26a97c1ab369523242de86823a17.jpg)

![eee0b30a9b2b0a59a51aedfa653b9ecf1c6b610f7c75e8516f8ee1d629f3dd1c.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/eee0b30a9b2b0a59a51aedfa653b9ecf1c6b610f7c75e8516f8ee1d629f3dd1c.jpg)

![f488ef08ebb2dfaabde26652ab5a2eab036c26e582473deeabb2493150c75242.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/images/f488ef08ebb2dfaabde26652ab5a2eab036c26e582473deeabb2493150c75242.jpg)

### Tables

![0a8cdd3b78f3f5ace13cedc1e8a8bfdf73a77accc6c7c304b2546e0a43603a05.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/tables/0a8cdd3b78f3f5ace13cedc1e8a8bfdf73a77accc6c7c304b2546e0a43603a05.jpg)

![15b18b2c01d90c4d303cf490aa092fba6bce1916ce7f15ffbc97f2084273241f.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/tables/15b18b2c01d90c4d303cf490aa092fba6bce1916ce7f15ffbc97f2084273241f.jpg)

![1b5046cd3248f0d5d39777ac36b0b6486295e14a98aa116389146e9e01426bed.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/tables/1b5046cd3248f0d5d39777ac36b0b6486295e14a98aa116389146e9e01426bed.jpg)

![279ec4196876d9d6782fa23379f10355316ec65522c7211f951e85f4514d0596.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/tables/279ec4196876d9d6782fa23379f10355316ec65522c7211f951e85f4514d0596.jpg)

![2ef11fc623b161a71712c8e19d233f4b846abfa716eb96bfa844ef38eaa7b743.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/tables/2ef11fc623b161a71712c8e19d233f4b846abfa716eb96bfa844ef38eaa7b743.jpg)

![4210ee9c692bc9e3fb1bdbaea339b833876326a709af9307ead995c576111760.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/tables/4210ee9c692bc9e3fb1bdbaea339b833876326a709af9307ead995c576111760.jpg)

![6334bebb80cd181cf2d225ca7b3a6269f188bdcfff33420d1f51e4893efff661.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/tables/6334bebb80cd181cf2d225ca7b3a6269f188bdcfff33420d1f51e4893efff661.jpg)

![6430cc1f532176d2d9075e350aeadd7ae734bfdc6e5f63b8467428567c0e3b04.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/tables/6430cc1f532176d2d9075e350aeadd7ae734bfdc6e5f63b8467428567c0e3b04.jpg)

![6dc349e711af0544563535b2009b9458412df33d18f369bc9f8a97407f0ffce8.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/tables/6dc349e711af0544563535b2009b9458412df33d18f369bc9f8a97407f0ffce8.jpg)

![7683aa0807e6f38aa24d03da53b4b29324c8a3a1a4a6b4baea417456a31d115a.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/tables/7683aa0807e6f38aa24d03da53b4b29324c8a3a1a4a6b4baea417456a31d115a.jpg)

![819087a1c2186593667296540dba0f2c06857627123050cc994078911f3d1266.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/tables/819087a1c2186593667296540dba0f2c06857627123050cc994078911f3d1266.jpg)

![8ad837520dba1fca6aaa393795ca428b8123e53564effd4f58b7e2caa6ddc862.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/tables/8ad837520dba1fca6aaa393795ca428b8123e53564effd4f58b7e2caa6ddc862.jpg)

![b079be6f0eb059b6221ad19bfc74dcd4642322c011c180727f1297c0c761c48a.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/tables/b079be6f0eb059b6221ad19bfc74dcd4642322c011c180727f1297c0c761c48a.jpg)

![d554d38be5cd87cba08363ce07b4c6b53bb85763daa6dbbb804c739d76075c4d.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/tables/d554d38be5cd87cba08363ce07b4c6b53bb85763daa6dbbb804c739d76075c4d.jpg)

![de0e21bbba9e31990cffd1c4eb5e14d4d5435217d382b92a81e62214c3aed64c.jpg](../icml_results/1203_SpaCeFormer_%20Fast%20Proposal-Free%20Open-Vocabulary%203D%20Instance%20Segmentation/tables/de0e21bbba9e31990cffd1c4eb5e14d4d5435217d382b92a81e62214c3aed64c.jpg)

<a id="paper-1186"></a>

## Efficient LLM Moderation with Multi-Layer Latent Prototypes


### Images

![0695419708bfb2fa33f628ea3d6dc373b829fad28e6c6fb2ebc9814e688a0a6b.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/images/0695419708bfb2fa33f628ea3d6dc373b829fad28e6c6fb2ebc9814e688a0a6b.jpg)

![0f6f144bc011754f0cb52f41cc30fe17d19dfe5b965407970a3c25073836408d.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/images/0f6f144bc011754f0cb52f41cc30fe17d19dfe5b965407970a3c25073836408d.jpg)

![11c99154773af38646d65265a4e0ba0ae028af90a27e87073973b2a638bbab6d.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/images/11c99154773af38646d65265a4e0ba0ae028af90a27e87073973b2a638bbab6d.jpg)

![16d9a7a3157065e3a0864c0a67f1321970fb9e52e4401894b17a6c51605ecd98.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/images/16d9a7a3157065e3a0864c0a67f1321970fb9e52e4401894b17a6c51605ecd98.jpg)

![1d3cca68105f0a8cdb3f354e1340bcdbdc19a67f18f3d3ffbda02b9ea1c9b2ae.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/images/1d3cca68105f0a8cdb3f354e1340bcdbdc19a67f18f3d3ffbda02b9ea1c9b2ae.jpg)

![3cbe8bf8110ae6c02fdf5d94ac1a816e3aa225bfe63d91d1e1d0ebeb6a02b608.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/images/3cbe8bf8110ae6c02fdf5d94ac1a816e3aa225bfe63d91d1e1d0ebeb6a02b608.jpg)

![3fb353c97f4cab20e2222c97d9514445297b6f4dae7e9a4b89420919095d8426.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/images/3fb353c97f4cab20e2222c97d9514445297b6f4dae7e9a4b89420919095d8426.jpg)

![5812555f625c07472c2ed3e194a46dbc4a80c7f4d27f469318da56f7aa6dfa63.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/images/5812555f625c07472c2ed3e194a46dbc4a80c7f4d27f469318da56f7aa6dfa63.jpg)

![65790315f3e3857a0b4ca9476f741c578c31621236b1bad8a2513f60a263b215.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/images/65790315f3e3857a0b4ca9476f741c578c31621236b1bad8a2513f60a263b215.jpg)

![7591c00f7c3acc146fa54aa1738349f205b36e8ad802e82437f03f237c2190be.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/images/7591c00f7c3acc146fa54aa1738349f205b36e8ad802e82437f03f237c2190be.jpg)

### Tables

![0cb92f4952929c243156957d07b4135f223ccccf4668000eb1842c277707ea18.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/0cb92f4952929c243156957d07b4135f223ccccf4668000eb1842c277707ea18.jpg)

![1a5afd4b24a9f6e173e7d34bf5d11698073b23c37e706a90a7841115be7daa86.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/1a5afd4b24a9f6e173e7d34bf5d11698073b23c37e706a90a7841115be7daa86.jpg)

![2d20a2a8872d1011c56d98b029c007534ba15c7113cf2c1048e8cbed60cc072a.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/2d20a2a8872d1011c56d98b029c007534ba15c7113cf2c1048e8cbed60cc072a.jpg)

![3fa71f566c4445d9447ee8c105327fda5def0196dd6d2cd8d84e786188ee2948.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/3fa71f566c4445d9447ee8c105327fda5def0196dd6d2cd8d84e786188ee2948.jpg)

![490307fec97eaa480f8296d8ff7192b204075fb889473c343c7f104033c3d555.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/490307fec97eaa480f8296d8ff7192b204075fb889473c343c7f104033c3d555.jpg)

![49d722dcfedac6b0ecd1838ae29c1394a1f11b29b19559e9426be15a0702076f.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/49d722dcfedac6b0ecd1838ae29c1394a1f11b29b19559e9426be15a0702076f.jpg)

![53317b04de5235347172cef1fdb1f6ae73e5045ea5442a8484ac53429cf054bf.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/53317b04de5235347172cef1fdb1f6ae73e5045ea5442a8484ac53429cf054bf.jpg)

![61149cbdf5cfc2dc54df913406ce3102c03f72a59a8534b26ae503d72731fb12.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/61149cbdf5cfc2dc54df913406ce3102c03f72a59a8534b26ae503d72731fb12.jpg)

![67032cf1c803aa995a34577b0e11c1d77b947b9ad259fe5ee4c5e6a6ffeb0b87.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/67032cf1c803aa995a34577b0e11c1d77b947b9ad259fe5ee4c5e6a6ffeb0b87.jpg)

![6fe31058237c497cb744f8b3819c1e92d0555f40bad3f632208e56f9bb98a560.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/6fe31058237c497cb744f8b3819c1e92d0555f40bad3f632208e56f9bb98a560.jpg)

![82f1a60e123bdfc3c70b7c30cf235cc3b382918355036ddf173f616438e7131d.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/82f1a60e123bdfc3c70b7c30cf235cc3b382918355036ddf173f616438e7131d.jpg)

![837294e1763126aa64adb3c35f6be8f4ab8b70eddbaafa7fa571c3064627e0c2.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/837294e1763126aa64adb3c35f6be8f4ab8b70eddbaafa7fa571c3064627e0c2.jpg)

![9b9965960195f3012ee4be86186746487fad042f654ada9bf5f9e7e45dea85bf.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/9b9965960195f3012ee4be86186746487fad042f654ada9bf5f9e7e45dea85bf.jpg)

![a4d90bd0e6da30dab53cab8be36d78afc37d51b281805afdcee0574723e94c04.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/a4d90bd0e6da30dab53cab8be36d78afc37d51b281805afdcee0574723e94c04.jpg)

![ab04589c9e8e736936b67fbc78369428caceb9e564c6bda27653b8e62afc1efe.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/ab04589c9e8e736936b67fbc78369428caceb9e564c6bda27653b8e62afc1efe.jpg)

![b9b5d6f543708d51a581a2d8c8d6d6a39c6fd10c8dc93fa36e5b92dcef25574f.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/b9b5d6f543708d51a581a2d8c8d6d6a39c6fd10c8dc93fa36e5b92dcef25574f.jpg)

![bb5560fd64a989e9118f0a309c0969b2813af1c42500e52635e564f8848215a5.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/bb5560fd64a989e9118f0a309c0969b2813af1c42500e52635e564f8848215a5.jpg)

![c3ac66f0dd7d3a66965945828a2891938cdb5a06c297bef841e343350300d00e.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/c3ac66f0dd7d3a66965945828a2891938cdb5a06c297bef841e343350300d00e.jpg)

![d0ad4a4eeb3f8193208273fbdecf482bfb923526d1e57541d3d803f58bc9f631.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/d0ad4a4eeb3f8193208273fbdecf482bfb923526d1e57541d3d803f58bc9f631.jpg)

![d6b2b16d1bc50bb109ad62a9881e5d79c159cef3b0203fa540355a8d8f0acd76.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/d6b2b16d1bc50bb109ad62a9881e5d79c159cef3b0203fa540355a8d8f0acd76.jpg)

![e779a3e0544d077faf04afbfa03cd0cd6324d4723827daf3551006bcd80bb6a0.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/e779a3e0544d077faf04afbfa03cd0cd6324d4723827daf3551006bcd80bb6a0.jpg)

![fad53cb5aac6eec56642788920f0eb3ab5de6e8fbc35774f7e4628f410de15a9.jpg](../icml_results/1204_Efficient%20LLM%20Moderation%20with%20Multi-Layer%20Latent%20Prototypes/tables/fad53cb5aac6eec56642788920f0eb3ab5de6e8fbc35774f7e4628f410de15a9.jpg)

<a id="paper-1187"></a>

## Value Aggregation with Uncertainty in Online Decentralized MARL


### Images

![7eb372824382f6e02dc3a48b32bcb46a1992833612bc4cfbf45d84b311a64d2f.jpg](../icml_results/1205_Value%20Aggregation%20with%20Uncertainty%20in%20Online%20Decentralized%20MARL/images/7eb372824382f6e02dc3a48b32bcb46a1992833612bc4cfbf45d84b311a64d2f.jpg)

![8517098bae2e3cea34ce0db822184963a7fd18785415a310791f20089b6b8bfd.jpg](../icml_results/1205_Value%20Aggregation%20with%20Uncertainty%20in%20Online%20Decentralized%20MARL/images/8517098bae2e3cea34ce0db822184963a7fd18785415a310791f20089b6b8bfd.jpg)

![864e1df96869f9e5fec75dce54066642d2ea3c54afe1a17c2a719497d08661c4.jpg](../icml_results/1205_Value%20Aggregation%20with%20Uncertainty%20in%20Online%20Decentralized%20MARL/images/864e1df96869f9e5fec75dce54066642d2ea3c54afe1a17c2a719497d08661c4.jpg)

![955fc6076df6651adffadc05f9d9866ab451af5d1ceeacad9ab23a438aa9a498.jpg](../icml_results/1205_Value%20Aggregation%20with%20Uncertainty%20in%20Online%20Decentralized%20MARL/images/955fc6076df6651adffadc05f9d9866ab451af5d1ceeacad9ab23a438aa9a498.jpg)

![a0c31c9165ef9330547a9cbbc1fea0ae456d03bbc404c3077ef8bb7617f83f29.jpg](../icml_results/1205_Value%20Aggregation%20with%20Uncertainty%20in%20Online%20Decentralized%20MARL/images/a0c31c9165ef9330547a9cbbc1fea0ae456d03bbc404c3077ef8bb7617f83f29.jpg)

![a445f1790835aa86a7706d87baa228c85cea310a25921dbb5a75980d373d67e9.jpg](../icml_results/1205_Value%20Aggregation%20with%20Uncertainty%20in%20Online%20Decentralized%20MARL/images/a445f1790835aa86a7706d87baa228c85cea310a25921dbb5a75980d373d67e9.jpg)

![a93f47c15462b474e315130567ccec9192095163e6787559c60013cb8c211af5.jpg](../icml_results/1205_Value%20Aggregation%20with%20Uncertainty%20in%20Online%20Decentralized%20MARL/images/a93f47c15462b474e315130567ccec9192095163e6787559c60013cb8c211af5.jpg)

![bb9f7cdf475b2fb10df39ef08151bebae4fe2df46bdd23aa16738ccad9289dbc.jpg](../icml_results/1205_Value%20Aggregation%20with%20Uncertainty%20in%20Online%20Decentralized%20MARL/images/bb9f7cdf475b2fb10df39ef08151bebae4fe2df46bdd23aa16738ccad9289dbc.jpg)

![ccbdcb850ee9c0ed121366f2372e5b2801419c981dd84e78eaf1b2cf71878d60.jpg](../icml_results/1205_Value%20Aggregation%20with%20Uncertainty%20in%20Online%20Decentralized%20MARL/images/ccbdcb850ee9c0ed121366f2372e5b2801419c981dd84e78eaf1b2cf71878d60.jpg)

![dca2d77f854f98e7006d616c122fabedb996c91233da311305788b09d620b386.jpg](../icml_results/1205_Value%20Aggregation%20with%20Uncertainty%20in%20Online%20Decentralized%20MARL/images/dca2d77f854f98e7006d616c122fabedb996c91233da311305788b09d620b386.jpg)

![ee93410db54bf30022db760e4a6acb33c473f0e6f66038d3845389887360fe62.jpg](../icml_results/1205_Value%20Aggregation%20with%20Uncertainty%20in%20Online%20Decentralized%20MARL/images/ee93410db54bf30022db760e4a6acb33c473f0e6f66038d3845389887360fe62.jpg)

### Tables

![e0bc2d160f91c48dfbf11ab89c2b87c12b1ff9a3389c2d574a96c6d9eca8208f.jpg](../icml_results/1205_Value%20Aggregation%20with%20Uncertainty%20in%20Online%20Decentralized%20MARL/tables/e0bc2d160f91c48dfbf11ab89c2b87c12b1ff9a3389c2d574a96c6d9eca8208f.jpg)

![ec5a39fd0c07419c90c972bb1f53edd0da455fa18eb05fb2e8fe32825f49b296.jpg](../icml_results/1205_Value%20Aggregation%20with%20Uncertainty%20in%20Online%20Decentralized%20MARL/tables/ec5a39fd0c07419c90c972bb1f53edd0da455fa18eb05fb2e8fe32825f49b296.jpg)

<a id="paper-1188"></a>

## Balancing Plasticity and Stability with Fast and Slow Successor Features

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
