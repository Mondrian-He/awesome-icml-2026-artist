# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 139 of 197

## 目录 (Table of Contents)

1. [MultiLoReFT: Decoupling Shared and Modality-Specific Subspaces in Multimodal Learning via Low-Rank Representation Fine-Tuning](#paper-4555)
2. [CoF-T2I: Video Models as Pure Visual Reasoners for Text-to-Image Generation](#paper-4556)
3. [Heterogeneous Customizable Personalized Federated Fine-Tuning Approach for Large Language Models](#paper-4557)
4. [Is Code Better Than Language for Algorithmic Reasoning?](#paper-4558)
5. [CoT is Not the Chain of Truth: An Empirical Internal Analysis of Reasoning LLMs for Fake News Generation](#paper-4559)
6. [Diversity Matters: Revisiting Test-Time Compute in Vision-Language Models](#paper-4560)
7. [Causal-Adapter: Taming Text-to-Image Diffusion for Faithful Counterfactual Generation](#paper-4561)
8. [Alleviating Sparse Rewards by Modeling Step-Wise and Long-Term Sampling Effects in Flow-Based GRPO](#paper-4562)
9. [VideoMAETok: Boosting Video Diffusion Models via Masked Autoencoders as Tokenizers](#paper-4563)
10. [Neural Control: Adjoint Learning Through Equilibrium Constraints](#paper-4564)
11. [Entropy-Aware Dynamic KV Cache Sparsification for Autoregressive Image Generation and Editing](#paper-4565)
12. [AuTAgent: A Reinforcement Learning Framework for Tool-Augmented Audio Reasoning](#paper-4566)
13. [RefChess: Training-Free Contextual Search for Zero-Shot Referring Image Segmentation](#paper-4567)
14. [ProactiveLLM: Learning Active Interaction for Streaming Large Language Models](#paper-4568)
15. [Magnitude Distance: A Geometric Measure of Dataset Similarity](#paper-4569)
16. [A Theory of Contrastive Learning with Natural Images](#paper-4570)
17. [Singular Bayesian Neural Networks](#paper-4571)
18. [Evaluating Newtonian Mechanics in Video Generative Models with Real Physical Systems](#paper-4572)
19. [Differentially Private Synthetic Data via APIs 4: Tabular Data](#paper-4573)
20. [Poison with Style: A Practical Poisoning Attack on Code Large Language Models](#paper-4574)
21. [Functional Equivalence in Attention: A Comprehensive Study with Applications to Linear Mode Connectivity](#paper-4575)
22. [Cross-Modal Knowledge Distillation without Paired Data: Theoretical Foundation and Algorithm](#paper-4576)
23. [UB-SMoE: Universally Balanced Sparse Mixture-of-Experts for Resource-adaptive Federated Fine-tuning of Foundation Models](#paper-4577)
24. [SEMA: a Scalable and Efficient Mamba like Attention via Token Localization and Averaging](#paper-4578)
25. [Unified Time Series Explanations via Amortized Optimization and Instance-level Multi-Expert Knowledge Distillation](#paper-4579)
26. [Looking Locally: Object-Centric Vision Transformers as Foundation Models for Efficient Segmentation](#paper-4580)
27. [Differentiable Optimization Layers for Guaranteed Fairness in Deep Learning](#paper-4581)
28. [Plain Transformers are Surprisingly Powerful Link Predictors](#paper-4582)
29. [Item Response Scaling Laws: A Measurement Theory Approach for Efficient and Generalizable Neural Scaling Estimation](#paper-4583)
30. [Creat3r: Confidence Reaggregation for Exploration-aware Active 3D Reconstruction](#paper-4584)
31. [$L^3$: Large Lookup Layers](#paper-4585)
32. [Model-Preserving Adaptive Rounding](#paper-4586)
33. [Protein Circuit Tracing via Cross-layer Transcoders](#paper-4587)

---


<a id="paper-4555"></a>

## MultiLoReFT: Decoupling Shared and Modality-Specific Subspaces in Multimodal Learning via Low-Rank Representation Fine-Tuning

### Images

![1d5288be9716876ef78bdb1f45001ac133e39ccf25640c33b0cb305da681c9f0.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/1d5288be9716876ef78bdb1f45001ac133e39ccf25640c33b0cb305da681c9f0.jpg)

![2192d0e44fecbcee77592aa20791234ba6137c6f1ab855c4435660c020640b4f.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/2192d0e44fecbcee77592aa20791234ba6137c6f1ab855c4435660c020640b4f.jpg)

![3a7f04b9b594a0a5f147ac01396ccb13b13ab98e5a2662d5ef4d0c344d84a5a8.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/3a7f04b9b594a0a5f147ac01396ccb13b13ab98e5a2662d5ef4d0c344d84a5a8.jpg)

![42988d47cd7ad466a13f8fbb5d72fd458c28febbab628eeeb81972c4004cdd90.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/42988d47cd7ad466a13f8fbb5d72fd458c28febbab628eeeb81972c4004cdd90.jpg)

![4e384bcda06a4557ff2e4624eedeb6e1b2ed7791f9ead5628e5ce33e3c10fc8c.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/4e384bcda06a4557ff2e4624eedeb6e1b2ed7791f9ead5628e5ce33e3c10fc8c.jpg)

![6bf7db60e6106f98489d2a2ba4afd1edd62e3364341002e443fc14470377ec5f.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/6bf7db60e6106f98489d2a2ba4afd1edd62e3364341002e443fc14470377ec5f.jpg)

![7618acb89a3c6f3c0b38d5aa6e4ced1db14ea6730f3602d526d156f88021661b.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/7618acb89a3c6f3c0b38d5aa6e4ced1db14ea6730f3602d526d156f88021661b.jpg)

![a4cd017f9a3d2d39bed142d5244cbfb6d099ff43e9757c29c14a6ab350287e0f.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/a4cd017f9a3d2d39bed142d5244cbfb6d099ff43e9757c29c14a6ab350287e0f.jpg)

![a566aa7426b4bfda698a2e905bac3bb134c48e48bbc35e39f9601d8ee283f515.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/a566aa7426b4bfda698a2e905bac3bb134c48e48bbc35e39f9601d8ee283f515.jpg)

![c6576a0167e76bad2f36b0f8db3723693fa9571aacd4c2d7e52575d841561fcf.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/c6576a0167e76bad2f36b0f8db3723693fa9571aacd4c2d7e52575d841561fcf.jpg)

![f72717ac833c8b7aabfd83322d272dc035b9f9421aa28cbdbec50181f5276410.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/images/f72717ac833c8b7aabfd83322d272dc035b9f9421aa28cbdbec50181f5276410.jpg)

### Tables

![1ed22e483b83961456b5aea9a925c66fc6245c4f01e3f8c2c7da995126da4025.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/1ed22e483b83961456b5aea9a925c66fc6245c4f01e3f8c2c7da995126da4025.jpg)

![2b4c921a4c7fd757b18ed7d360b110ed3edbb17b05ebc815128cca8c6c5f3e80.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/2b4c921a4c7fd757b18ed7d360b110ed3edbb17b05ebc815128cca8c6c5f3e80.jpg)

![330d4aae7436b47386b0c8403459effc34902cbc21470d1114b43e91ecf7761c.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/330d4aae7436b47386b0c8403459effc34902cbc21470d1114b43e91ecf7761c.jpg)

![3b6588245d968dc6404be7313a12226e0164df3adaa60d5aa6fb7e9834f8495a.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/3b6588245d968dc6404be7313a12226e0164df3adaa60d5aa6fb7e9834f8495a.jpg)

![4b5ced5db5e1a21510179854fa91b75aa4f61fac64bd7be7084c1980dd212420.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/4b5ced5db5e1a21510179854fa91b75aa4f61fac64bd7be7084c1980dd212420.jpg)

![5818746e2f7cb94d6c10dbd364c37b65616cba8539e1948362f67dc5b06fa3a4.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/5818746e2f7cb94d6c10dbd364c37b65616cba8539e1948362f67dc5b06fa3a4.jpg)

![93e37471bc109b6b799d3a88a7bc853fe042a17c922372ad435531af02a1d077.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/93e37471bc109b6b799d3a88a7bc853fe042a17c922372ad435531af02a1d077.jpg)

![a4fc261512652fb07e30f0f8b427d5a9d35002ad2ef0db272defaf578919b9fa.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/a4fc261512652fb07e30f0f8b427d5a9d35002ad2ef0db272defaf578919b9fa.jpg)

![ad6a6bd054697c4ca1121baacaedd39057cce57d40e4218dd13713111ac02b0e.jpg](../icml_results/4601_Task-Awareness%20Improves%20LLM%20Generations%20and%20Uncertainty/tables/ad6a6bd054697c4ca1121baacaedd39057cce57d40e4218dd13713111ac02b0e.jpg)

## MultiLoReFT: Decoupling Shared and Modality-Specific Subspaces in Multimodal Learning via Low-Rank Representation Fine-Tuning


### Images

![8a2d6e546730c1f5351f45da52924f8f8639b2fe69e4b2e46c48313975fdcd5d.jpg](../icml_results/4602_MultiLoReFT_%20Decoupling%20Shared%20and%20Modality-Specific%20Subspaces%20in%20Multimodal%20Learning%20via%20Low-Rank%20R/images/8a2d6e546730c1f5351f45da52924f8f8639b2fe69e4b2e46c48313975fdcd5d.jpg)

![a44916eb6cf7727c3930a2251c5e768960502f708a828158b25119c405bda010.jpg](../icml_results/4602_MultiLoReFT_%20Decoupling%20Shared%20and%20Modality-Specific%20Subspaces%20in%20Multimodal%20Learning%20via%20Low-Rank%20R/images/a44916eb6cf7727c3930a2251c5e768960502f708a828158b25119c405bda010.jpg)

![a61caa5e33b042fdc30f56bf1d8519212348b3d7179e391ff4c60bc2eeec114c.jpg](../icml_results/4602_MultiLoReFT_%20Decoupling%20Shared%20and%20Modality-Specific%20Subspaces%20in%20Multimodal%20Learning%20via%20Low-Rank%20R/images/a61caa5e33b042fdc30f56bf1d8519212348b3d7179e391ff4c60bc2eeec114c.jpg)

![c1ab64d66ab00662b7f93eacfcd019d222de4fe063c856c325674955a1ae1f5f.jpg](../icml_results/4602_MultiLoReFT_%20Decoupling%20Shared%20and%20Modality-Specific%20Subspaces%20in%20Multimodal%20Learning%20via%20Low-Rank%20R/images/c1ab64d66ab00662b7f93eacfcd019d222de4fe063c856c325674955a1ae1f5f.jpg)

### Tables

![3643a0952dcaafa12872cdc10617f33935f9bc0a1225f10afd117ed0cdd539ad.jpg](../icml_results/4602_MultiLoReFT_%20Decoupling%20Shared%20and%20Modality-Specific%20Subspaces%20in%20Multimodal%20Learning%20via%20Low-Rank%20R/tables/3643a0952dcaafa12872cdc10617f33935f9bc0a1225f10afd117ed0cdd539ad.jpg)

![515ca6f35d8985ea28ff80e55f821a11298685dc5ae10e065c51014f1c858d7f.jpg](../icml_results/4602_MultiLoReFT_%20Decoupling%20Shared%20and%20Modality-Specific%20Subspaces%20in%20Multimodal%20Learning%20via%20Low-Rank%20R/tables/515ca6f35d8985ea28ff80e55f821a11298685dc5ae10e065c51014f1c858d7f.jpg)

![88e3d0f59510b33ecc8e1609a5f29a642adad9da9af0c8fb70cb9f3595ee7570.jpg](../icml_results/4602_MultiLoReFT_%20Decoupling%20Shared%20and%20Modality-Specific%20Subspaces%20in%20Multimodal%20Learning%20via%20Low-Rank%20R/tables/88e3d0f59510b33ecc8e1609a5f29a642adad9da9af0c8fb70cb9f3595ee7570.jpg)

![8c63150d418736cb933fd36660cb139f4413d1baea2e110b1a2a4a7e4c5deb29.jpg](../icml_results/4602_MultiLoReFT_%20Decoupling%20Shared%20and%20Modality-Specific%20Subspaces%20in%20Multimodal%20Learning%20via%20Low-Rank%20R/tables/8c63150d418736cb933fd36660cb139f4413d1baea2e110b1a2a4a7e4c5deb29.jpg)

![ad51d683b322b2dcd2f2e9113d4fedb887882ef8726ead24b79ebacf3ab19382.jpg](../icml_results/4602_MultiLoReFT_%20Decoupling%20Shared%20and%20Modality-Specific%20Subspaces%20in%20Multimodal%20Learning%20via%20Low-Rank%20R/tables/ad51d683b322b2dcd2f2e9113d4fedb887882ef8726ead24b79ebacf3ab19382.jpg)

![bbe873087b3a29190c536b3d1550a6468a93573b8496701c7da849b4379ee0a6.jpg](../icml_results/4602_MultiLoReFT_%20Decoupling%20Shared%20and%20Modality-Specific%20Subspaces%20in%20Multimodal%20Learning%20via%20Low-Rank%20R/tables/bbe873087b3a29190c536b3d1550a6468a93573b8496701c7da849b4379ee0a6.jpg)

<a id="paper-4556"></a>

## CoF-T2I: Video Models as Pure Visual Reasoners for Text-to-Image Generation


### Images

![1385a84b419cfda2517fe327fe00e694a19403e7fb0e03b29444f0a8c462f9e1.jpg](../icml_results/4603_CoF-T2I_%20Video%20Models%20as%20Pure%20Visual%20Reasoners%20for%20Text-to-Image%20Generation/images/1385a84b419cfda2517fe327fe00e694a19403e7fb0e03b29444f0a8c462f9e1.jpg)

![2272fd6a5656f4a050a153f0bdbc3a2e05be792c003e002bb2d589f8575ac62f.jpg](../icml_results/4603_CoF-T2I_%20Video%20Models%20as%20Pure%20Visual%20Reasoners%20for%20Text-to-Image%20Generation/images/2272fd6a5656f4a050a153f0bdbc3a2e05be792c003e002bb2d589f8575ac62f.jpg)

![6272abcb5685a7acb5a3799de3ffba8249d86735a270aed2e7884e9daa1e85cc.jpg](../icml_results/4603_CoF-T2I_%20Video%20Models%20as%20Pure%20Visual%20Reasoners%20for%20Text-to-Image%20Generation/images/6272abcb5685a7acb5a3799de3ffba8249d86735a270aed2e7884e9daa1e85cc.jpg)

![70d10ce54c3989e8675f3e99d312d509d84bf23cf442fc580a728ca2223174e3.jpg](../icml_results/4603_CoF-T2I_%20Video%20Models%20as%20Pure%20Visual%20Reasoners%20for%20Text-to-Image%20Generation/images/70d10ce54c3989e8675f3e99d312d509d84bf23cf442fc580a728ca2223174e3.jpg)

![bec384ae858edb8c2ac921ebcb5e4e8e507f65914a94fc9b77860781d2a56da7.jpg](../icml_results/4603_CoF-T2I_%20Video%20Models%20as%20Pure%20Visual%20Reasoners%20for%20Text-to-Image%20Generation/images/bec384ae858edb8c2ac921ebcb5e4e8e507f65914a94fc9b77860781d2a56da7.jpg)

![e589282cf72835034cbd183a1815190a0aaab5c9656ba103d817b04d05dfd29f.jpg](../icml_results/4603_CoF-T2I_%20Video%20Models%20as%20Pure%20Visual%20Reasoners%20for%20Text-to-Image%20Generation/images/e589282cf72835034cbd183a1815190a0aaab5c9656ba103d817b04d05dfd29f.jpg)

![fcfe82b7f7b16b8baa9f79861628fb126af4d4a7781490610727ccd36e02eeb2.jpg](../icml_results/4603_CoF-T2I_%20Video%20Models%20as%20Pure%20Visual%20Reasoners%20for%20Text-to-Image%20Generation/images/fcfe82b7f7b16b8baa9f79861628fb126af4d4a7781490610727ccd36e02eeb2.jpg)

![fe4655a96ace13e673ec3bc4cdf42c2f2e25905b8a50fa856dcfd9e74416d2fe.jpg](../icml_results/4603_CoF-T2I_%20Video%20Models%20as%20Pure%20Visual%20Reasoners%20for%20Text-to-Image%20Generation/images/fe4655a96ace13e673ec3bc4cdf42c2f2e25905b8a50fa856dcfd9e74416d2fe.jpg)

### Tables

![232e1027ca376fca817b6e911cf2410cc86ae57f6f60ffacc1a6b7bf40d53153.jpg](../icml_results/4603_CoF-T2I_%20Video%20Models%20as%20Pure%20Visual%20Reasoners%20for%20Text-to-Image%20Generation/tables/232e1027ca376fca817b6e911cf2410cc86ae57f6f60ffacc1a6b7bf40d53153.jpg)

![7018d6b70b8fc02d63974c8196252b36b2f761731db23098aeb100a2343ba1c2.jpg](../icml_results/4603_CoF-T2I_%20Video%20Models%20as%20Pure%20Visual%20Reasoners%20for%20Text-to-Image%20Generation/tables/7018d6b70b8fc02d63974c8196252b36b2f761731db23098aeb100a2343ba1c2.jpg)

![8ffeee7696871b8fd2f084cc4e9afbf10b84e8a5e1facac3a3eb2877b980abd0.jpg](../icml_results/4603_CoF-T2I_%20Video%20Models%20as%20Pure%20Visual%20Reasoners%20for%20Text-to-Image%20Generation/tables/8ffeee7696871b8fd2f084cc4e9afbf10b84e8a5e1facac3a3eb2877b980abd0.jpg)

![9792f09d31f5cafed515322efdd415527fc703d79710ea7563cf1db23c424338.jpg](../icml_results/4603_CoF-T2I_%20Video%20Models%20as%20Pure%20Visual%20Reasoners%20for%20Text-to-Image%20Generation/tables/9792f09d31f5cafed515322efdd415527fc703d79710ea7563cf1db23c424338.jpg)

![a836a9318892306b641e0dc2b188aa567c5e45354eb4368ffbdf4768aaecbe39.jpg](../icml_results/4603_CoF-T2I_%20Video%20Models%20as%20Pure%20Visual%20Reasoners%20for%20Text-to-Image%20Generation/tables/a836a9318892306b641e0dc2b188aa567c5e45354eb4368ffbdf4768aaecbe39.jpg)

![b79b95fd17e797cfbb71c3b0b62c1e4a8422937ff8679685f988a05501b894a4.jpg](../icml_results/4603_CoF-T2I_%20Video%20Models%20as%20Pure%20Visual%20Reasoners%20for%20Text-to-Image%20Generation/tables/b79b95fd17e797cfbb71c3b0b62c1e4a8422937ff8679685f988a05501b894a4.jpg)

![be53f7750bd141f6ff207235704c2bfb5a4c9797af58e30845c12333fad72a96.jpg](../icml_results/4603_CoF-T2I_%20Video%20Models%20as%20Pure%20Visual%20Reasoners%20for%20Text-to-Image%20Generation/tables/be53f7750bd141f6ff207235704c2bfb5a4c9797af58e30845c12333fad72a96.jpg)

<a id="paper-4557"></a>

## Heterogeneous Customizable Personalized Federated Fine-Tuning Approach for Large Language Models


### Images

![2e809ef13a7eff5b52a808237a132de24b706239d5455efa8f397784a563254e.jpg](../icml_results/4604_Heterogeneous%20Customizable%20Personalized%20Federated%20Fine-Tuning%20Approach%20for%20Large%20Language%20Models/images/2e809ef13a7eff5b52a808237a132de24b706239d5455efa8f397784a563254e.jpg)

![4936f7fff19be088823f9993a51c329a1f9950198eaf4220463996a45e3b85f2.jpg](../icml_results/4604_Heterogeneous%20Customizable%20Personalized%20Federated%20Fine-Tuning%20Approach%20for%20Large%20Language%20Models/images/4936f7fff19be088823f9993a51c329a1f9950198eaf4220463996a45e3b85f2.jpg)

![58df00bffb563a3cec9db4ed8c47120bac75941ba2c72bebb3d145a569b6b126.jpg](../icml_results/4604_Heterogeneous%20Customizable%20Personalized%20Federated%20Fine-Tuning%20Approach%20for%20Large%20Language%20Models/images/58df00bffb563a3cec9db4ed8c47120bac75941ba2c72bebb3d145a569b6b126.jpg)

![5eaec8de34ef0b2399cffce51c970b4ea822fdb2f1090be7f8f503d1a48d02f2.jpg](../icml_results/4604_Heterogeneous%20Customizable%20Personalized%20Federated%20Fine-Tuning%20Approach%20for%20Large%20Language%20Models/images/5eaec8de34ef0b2399cffce51c970b4ea822fdb2f1090be7f8f503d1a48d02f2.jpg)

![71b8acf8fd0ba3611512cd7f6b216efd60a79db5bff71e9636fac10991843f6d.jpg](../icml_results/4604_Heterogeneous%20Customizable%20Personalized%20Federated%20Fine-Tuning%20Approach%20for%20Large%20Language%20Models/images/71b8acf8fd0ba3611512cd7f6b216efd60a79db5bff71e9636fac10991843f6d.jpg)

![c571527058733b58db7f194199ad403f90e7ee2c67c8fffbba1056dfc0799676.jpg](../icml_results/4604_Heterogeneous%20Customizable%20Personalized%20Federated%20Fine-Tuning%20Approach%20for%20Large%20Language%20Models/images/c571527058733b58db7f194199ad403f90e7ee2c67c8fffbba1056dfc0799676.jpg)

![ff1853adf6a5486a1c3b4084c3fde8fe8dcb7870d68790fc0ac30cdb0f6b6e58.jpg](../icml_results/4604_Heterogeneous%20Customizable%20Personalized%20Federated%20Fine-Tuning%20Approach%20for%20Large%20Language%20Models/images/ff1853adf6a5486a1c3b4084c3fde8fe8dcb7870d68790fc0ac30cdb0f6b6e58.jpg)

### Tables

![4d3cd0bc8d8451d151d22ac25814bd291400b6a1edd3f11da7881c19c6c0fbca.jpg](../icml_results/4604_Heterogeneous%20Customizable%20Personalized%20Federated%20Fine-Tuning%20Approach%20for%20Large%20Language%20Models/tables/4d3cd0bc8d8451d151d22ac25814bd291400b6a1edd3f11da7881c19c6c0fbca.jpg)

![538a18d7d037a0d624679c9f284a49f28196c202196a86868849c6768584c670.jpg](../icml_results/4604_Heterogeneous%20Customizable%20Personalized%20Federated%20Fine-Tuning%20Approach%20for%20Large%20Language%20Models/tables/538a18d7d037a0d624679c9f284a49f28196c202196a86868849c6768584c670.jpg)

![70a8adad98b1025c346ab9da69dc1942cf39852a760e20b2afe4f038754c3d1c.jpg](../icml_results/4604_Heterogeneous%20Customizable%20Personalized%20Federated%20Fine-Tuning%20Approach%20for%20Large%20Language%20Models/tables/70a8adad98b1025c346ab9da69dc1942cf39852a760e20b2afe4f038754c3d1c.jpg)

![b758d7d7fc40feedb1dc0ab1da884f35e2e5d9bfd05a143a380c671403cba477.jpg](../icml_results/4604_Heterogeneous%20Customizable%20Personalized%20Federated%20Fine-Tuning%20Approach%20for%20Large%20Language%20Models/tables/b758d7d7fc40feedb1dc0ab1da884f35e2e5d9bfd05a143a380c671403cba477.jpg)

![c1084b5d11dbd72b84924ab18daf5c8d1406053ac2e1da0afaf56d6b13afe806.jpg](../icml_results/4604_Heterogeneous%20Customizable%20Personalized%20Federated%20Fine-Tuning%20Approach%20for%20Large%20Language%20Models/tables/c1084b5d11dbd72b84924ab18daf5c8d1406053ac2e1da0afaf56d6b13afe806.jpg)

![dabc074a61ba17c0f303c351710df27ab2f2bb0bfdd761ce9c4295e103b97f50.jpg](../icml_results/4604_Heterogeneous%20Customizable%20Personalized%20Federated%20Fine-Tuning%20Approach%20for%20Large%20Language%20Models/tables/dabc074a61ba17c0f303c351710df27ab2f2bb0bfdd761ce9c4295e103b97f50.jpg)

![e0452a1ddf10612735098f8e186394c77f1cf043277e3697a5dec859aab919e7.jpg](../icml_results/4604_Heterogeneous%20Customizable%20Personalized%20Federated%20Fine-Tuning%20Approach%20for%20Large%20Language%20Models/tables/e0452a1ddf10612735098f8e186394c77f1cf043277e3697a5dec859aab919e7.jpg)

![e33d35229efb5ff9f0adaae8f3805216745fb16041becd7105c6cac3741f7555.jpg](../icml_results/4604_Heterogeneous%20Customizable%20Personalized%20Federated%20Fine-Tuning%20Approach%20for%20Large%20Language%20Models/tables/e33d35229efb5ff9f0adaae8f3805216745fb16041becd7105c6cac3741f7555.jpg)

<a id="paper-4558"></a>

## Is Code Better Than Language for Algorithmic Reasoning?


### Images

![06cca88cd145d5268eba87147f0bc9b3e8c8195ec7acc135fbfa4dd953dd9dc0.jpg](../icml_results/4605_Is%20Code%20Better%20Than%20Language%20for%20Algorithmic%20Reasoning_/images/06cca88cd145d5268eba87147f0bc9b3e8c8195ec7acc135fbfa4dd953dd9dc0.jpg)

![40aa91a80987a943bd60f624c4b88919fe90d961729cc8251f59b436b6744ea4.jpg](../icml_results/4605_Is%20Code%20Better%20Than%20Language%20for%20Algorithmic%20Reasoning_/images/40aa91a80987a943bd60f624c4b88919fe90d961729cc8251f59b436b6744ea4.jpg)

![43b0f95765128894cbef0d22ff2a6cbb21dc660e66d21764795acfa0f79a8412.jpg](../icml_results/4605_Is%20Code%20Better%20Than%20Language%20for%20Algorithmic%20Reasoning_/images/43b0f95765128894cbef0d22ff2a6cbb21dc660e66d21764795acfa0f79a8412.jpg)

![73e905987e1b8f956a6b5720a0f2a0908fab82ac9e682a436d124a0331ac6da8.jpg](../icml_results/4605_Is%20Code%20Better%20Than%20Language%20for%20Algorithmic%20Reasoning_/images/73e905987e1b8f956a6b5720a0f2a0908fab82ac9e682a436d124a0331ac6da8.jpg)

![c28ef0e98b0495af2ebfd4b42f74c591a916f508587d69b6dd3460b6b3f87de4.jpg](../icml_results/4605_Is%20Code%20Better%20Than%20Language%20for%20Algorithmic%20Reasoning_/images/c28ef0e98b0495af2ebfd4b42f74c591a916f508587d69b6dd3460b6b3f87de4.jpg)

![dc0485211045dc803d4036b43ecdad7003fdf7c0461139ec3908614f02977c95.jpg](../icml_results/4605_Is%20Code%20Better%20Than%20Language%20for%20Algorithmic%20Reasoning_/images/dc0485211045dc803d4036b43ecdad7003fdf7c0461139ec3908614f02977c95.jpg)

![e3e813c0979bf480f2d6e99a505c1f1ed1d1b07008148043c18f8e97472c6a89.jpg](../icml_results/4605_Is%20Code%20Better%20Than%20Language%20for%20Algorithmic%20Reasoning_/images/e3e813c0979bf480f2d6e99a505c1f1ed1d1b07008148043c18f8e97472c6a89.jpg)

### Tables

![10c927b78abf0fd21aecf310ed24d918e7a7140a326ab68b743d66607e5b376e.jpg](../icml_results/4605_Is%20Code%20Better%20Than%20Language%20for%20Algorithmic%20Reasoning_/tables/10c927b78abf0fd21aecf310ed24d918e7a7140a326ab68b743d66607e5b376e.jpg)

![37ef2f06ce18b5f3a5d9fb13d8e0c1742975b1dd72d8d78d808767a2c8637afc.jpg](../icml_results/4605_Is%20Code%20Better%20Than%20Language%20for%20Algorithmic%20Reasoning_/tables/37ef2f06ce18b5f3a5d9fb13d8e0c1742975b1dd72d8d78d808767a2c8637afc.jpg)

![612fb59aef1d1bb172f324048c405080e6b2b7585b6c48f01350ceb7a8b1ec13.jpg](../icml_results/4605_Is%20Code%20Better%20Than%20Language%20for%20Algorithmic%20Reasoning_/tables/612fb59aef1d1bb172f324048c405080e6b2b7585b6c48f01350ceb7a8b1ec13.jpg)

![949234b7706c5d012c57d915650e211f13b0c26ac493baeb9a5e7277d2b67272.jpg](../icml_results/4605_Is%20Code%20Better%20Than%20Language%20for%20Algorithmic%20Reasoning_/tables/949234b7706c5d012c57d915650e211f13b0c26ac493baeb9a5e7277d2b67272.jpg)

![95bcc80d8c0dc1f52fd85e266e55745bc532a5d313afaaba2dcde70cbf83df8f.jpg](../icml_results/4605_Is%20Code%20Better%20Than%20Language%20for%20Algorithmic%20Reasoning_/tables/95bcc80d8c0dc1f52fd85e266e55745bc532a5d313afaaba2dcde70cbf83df8f.jpg)

![b3be77673ec9451a6a0c3b3d1adb32ec8ba09a850b1b5f344542ab803d762a31.jpg](../icml_results/4605_Is%20Code%20Better%20Than%20Language%20for%20Algorithmic%20Reasoning_/tables/b3be77673ec9451a6a0c3b3d1adb32ec8ba09a850b1b5f344542ab803d762a31.jpg)

![f830e58ddbcfb8f798cf968fbfe7a0613d42d7fff2c86ac842e397fbccee5265.jpg](../icml_results/4605_Is%20Code%20Better%20Than%20Language%20for%20Algorithmic%20Reasoning_/tables/f830e58ddbcfb8f798cf968fbfe7a0613d42d7fff2c86ac842e397fbccee5265.jpg)

<a id="paper-4559"></a>

## CoT is Not the Chain of Truth: An Empirical Internal Analysis of Reasoning LLMs for Fake News Generation


### Images

![0efb3bd498c4761636713a7bff2236fe3121d41c7725a884ec60092661a064d4.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/0efb3bd498c4761636713a7bff2236fe3121d41c7725a884ec60092661a064d4.jpg)

![144889d91a0da36149fb34bda285a699e6c03714bc5a3ecd10646ff93769cb14.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/144889d91a0da36149fb34bda285a699e6c03714bc5a3ecd10646ff93769cb14.jpg)

![14bd79404f31203cb868947ce2a194e327f73fdfa7f46da3f1986b49f16ab2e8.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/14bd79404f31203cb868947ce2a194e327f73fdfa7f46da3f1986b49f16ab2e8.jpg)

![1ae05c23fce8b92a765138727df269f9e817415b2d54bf69b44d44a799198671.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/1ae05c23fce8b92a765138727df269f9e817415b2d54bf69b44d44a799198671.jpg)

![1d07f34ee89273eb50ce395763042810d65f94a12f94b96ce93d4cec5fb29823.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/1d07f34ee89273eb50ce395763042810d65f94a12f94b96ce93d4cec5fb29823.jpg)

![2d6d4faf2a9d7f139c7e9107aa4abaeed3b69d2f468589dfed633d23b273f844.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/2d6d4faf2a9d7f139c7e9107aa4abaeed3b69d2f468589dfed633d23b273f844.jpg)

![46d6583437515a09753a1b203bc456881ab8115eb0847237584951942d270282.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/46d6583437515a09753a1b203bc456881ab8115eb0847237584951942d270282.jpg)

![4987d4c172a9930d80db3749150011fe402244e08af4044a7be373b8394d1f55.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/4987d4c172a9930d80db3749150011fe402244e08af4044a7be373b8394d1f55.jpg)

![4a9480ef0f0398a9d4e8d0e4fc51205c38e924c1cc08fdffb451841e9466fc5f.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/4a9480ef0f0398a9d4e8d0e4fc51205c38e924c1cc08fdffb451841e9466fc5f.jpg)

![5319bf53060209d8b788f90775f84146f978f6e7e7636e70baef9ceb0330d1cb.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/5319bf53060209d8b788f90775f84146f978f6e7e7636e70baef9ceb0330d1cb.jpg)

![5401d75b655c2867e68f53146ff806173f7f6b01ed8d6294220213cdfe68139f.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/5401d75b655c2867e68f53146ff806173f7f6b01ed8d6294220213cdfe68139f.jpg)

![5424f02adb70e4d0b5e9a7f040fe38b5ef4ad4d837d4f1bc8fef1a51b74341a4.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/5424f02adb70e4d0b5e9a7f040fe38b5ef4ad4d837d4f1bc8fef1a51b74341a4.jpg)

![5bd57432eb2a454a228f53a0adf4b718afbdabe438624e320608dff7afe3a42f.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/5bd57432eb2a454a228f53a0adf4b718afbdabe438624e320608dff7afe3a42f.jpg)

![627abc0653be7d34f3a2e3a93fc8b537b31ef07fb8170ee572960732adaefbda.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/627abc0653be7d34f3a2e3a93fc8b537b31ef07fb8170ee572960732adaefbda.jpg)

![723df5653442c06ba3903b7066c1967733faed3154417bacb0514dee2d87debf.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/723df5653442c06ba3903b7066c1967733faed3154417bacb0514dee2d87debf.jpg)

![77444f61699464e6c873ba41c32981ab8fca0aa6f321f18092b7c0dbc1fa0f2a.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/77444f61699464e6c873ba41c32981ab8fca0aa6f321f18092b7c0dbc1fa0f2a.jpg)

![79fca706231db3b9f23ccb803186ce3d4e26169f3a5372929b963d4194f4d5a9.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/79fca706231db3b9f23ccb803186ce3d4e26169f3a5372929b963d4194f4d5a9.jpg)

![86b5031b9ffa9630959b8d74ca6d645f9d8df17a9ab53a3736f5927cdbdb30ea.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/86b5031b9ffa9630959b8d74ca6d645f9d8df17a9ab53a3736f5927cdbdb30ea.jpg)

![9031166c3f9aae5bdbbfa1fe5152a2bfbe3cdcc33d204f7401675c91148a2fa8.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/9031166c3f9aae5bdbbfa1fe5152a2bfbe3cdcc33d204f7401675c91148a2fa8.jpg)

![9f2fbb048b61c6a8a12c7960565df6b48bcfe798ee5e4976466d6be465c8bf33.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/9f2fbb048b61c6a8a12c7960565df6b48bcfe798ee5e4976466d6be465c8bf33.jpg)

![bed0b4f2098b5fa0a6e9ba00f8b32e06f6b72c5196fc4f3d867e0e333dbcf460.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/bed0b4f2098b5fa0a6e9ba00f8b32e06f6b72c5196fc4f3d867e0e333dbcf460.jpg)

![cd7b815ef577fae0d0817224788544953fbc3094a102c7906d9bf5b3077d96a5.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/cd7b815ef577fae0d0817224788544953fbc3094a102c7906d9bf5b3077d96a5.jpg)

![d67124da6ee05aea5815a2f677f30a5f28c5ee5dc4aef6b2c0df82d029398967.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/d67124da6ee05aea5815a2f677f30a5f28c5ee5dc4aef6b2c0df82d029398967.jpg)

![d762ebe17e948ee177bdcfe9ec6bffc460ae055a76a2601cdc810e7c517c1caf.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/d762ebe17e948ee177bdcfe9ec6bffc460ae055a76a2601cdc810e7c517c1caf.jpg)

![dfe08b8b4511a7ecb0d411e94f05e03b1267fccdbfc4be6624a69e1dfedd31fa.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/dfe08b8b4511a7ecb0d411e94f05e03b1267fccdbfc4be6624a69e1dfedd31fa.jpg)

![e42ad4a08a5f1e8e5374d8bf385c24f339713d703045fddd4f675d84ce43c9dd.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/e42ad4a08a5f1e8e5374d8bf385c24f339713d703045fddd4f675d84ce43c9dd.jpg)

![ea599cd174f3ef34f396f7bce72638a949c211ead110b35276b3b00d20c95ce9.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/ea599cd174f3ef34f396f7bce72638a949c211ead110b35276b3b00d20c95ce9.jpg)

![ec2cd27d12edd8acaf4cfb374f4801fdcd9c2c90eb6b900fd410f54a8202f012.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/images/ec2cd27d12edd8acaf4cfb374f4801fdcd9c2c90eb6b900fd410f54a8202f012.jpg)

### Tables

![35ce6de62bf9f563e0690377ce7dd4f137ec642b8cb5c587fd0bebc6ce87004f.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/tables/35ce6de62bf9f563e0690377ce7dd4f137ec642b8cb5c587fd0bebc6ce87004f.jpg)

![52874aa64087ace63fd3885767bd7974294ee90d93c9e0fdd80aedff50a08412.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/tables/52874aa64087ace63fd3885767bd7974294ee90d93c9e0fdd80aedff50a08412.jpg)

![6ddfc4acacaa8858865012d2ed12af5057b61fcfefb17723da16753512f792a9.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/tables/6ddfc4acacaa8858865012d2ed12af5057b61fcfefb17723da16753512f792a9.jpg)

![8c715b69c03285bf9d7ccf21758801e1bedd26a3e14a3dde72b876130e03c370.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/tables/8c715b69c03285bf9d7ccf21758801e1bedd26a3e14a3dde72b876130e03c370.jpg)

![9f13d0288f77388ea8750dbb8e5a17578cbc322ab923c39ba5d5cff84d95a161.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/tables/9f13d0288f77388ea8750dbb8e5a17578cbc322ab923c39ba5d5cff84d95a161.jpg)

![a6326be031551cb767c93a44d2ece5acb882c15c75811b6951b989da374c59b0.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/tables/a6326be031551cb767c93a44d2ece5acb882c15c75811b6951b989da374c59b0.jpg)

![c06d28c94d26ab59dae620aaa69e65cec22f7a887ff6677b86b1d84bc1c44702.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/tables/c06d28c94d26ab59dae620aaa69e65cec22f7a887ff6677b86b1d84bc1c44702.jpg)

![c40299f96ed150db64cd7678f69ff2021a8fd11a34196c4d258734341bb8008f.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/tables/c40299f96ed150db64cd7678f69ff2021a8fd11a34196c4d258734341bb8008f.jpg)

![d0c8a080ef072ed2aa219cf5199c384fb20ecaa2fbcbfdd29978ef42d8154d29.jpg](../icml_results/4606_CoT%20is%20Not%20the%20Chain%20of%20Truth_%20An%20Empirical%20Internal%20Analysis%20of%20Reasoning%20LLMs%20for%20Fake%20News%20Genera/tables/d0c8a080ef072ed2aa219cf5199c384fb20ecaa2fbcbfdd29978ef42d8154d29.jpg)

<a id="paper-4560"></a>

## Diversity Matters: Revisiting Test-Time Compute in Vision-Language Models


### Images

![23078dab22d71549c564433c6297ca49433a8f68d03d86967b99f982d7dc0c39.jpg](../icml_results/4607_Diversity%20Matters_%20Revisiting%20Test-Time%20Compute%20in%20Vision-Language%20Models/images/23078dab22d71549c564433c6297ca49433a8f68d03d86967b99f982d7dc0c39.jpg)

![23562abcc1732a40555c4f0d404e98a3c1fb470f5fa82d6adfcd02c668c1957e.jpg](../icml_results/4607_Diversity%20Matters_%20Revisiting%20Test-Time%20Compute%20in%20Vision-Language%20Models/images/23562abcc1732a40555c4f0d404e98a3c1fb470f5fa82d6adfcd02c668c1957e.jpg)

![267d459135f7957ec220c828a4b185dabe346eeca7177a5847c5f5a8f2b01186.jpg](../icml_results/4607_Diversity%20Matters_%20Revisiting%20Test-Time%20Compute%20in%20Vision-Language%20Models/images/267d459135f7957ec220c828a4b185dabe346eeca7177a5847c5f5a8f2b01186.jpg)

![33514c5c777e8499ddd55ceb92d165514010df29b71aebd7b421f8cd524228e9.jpg](../icml_results/4607_Diversity%20Matters_%20Revisiting%20Test-Time%20Compute%20in%20Vision-Language%20Models/images/33514c5c777e8499ddd55ceb92d165514010df29b71aebd7b421f8cd524228e9.jpg)

![495c5d3fd02893bfb2f9fd6e45df471212819efe94e814cd9e563a44e7432051.jpg](../icml_results/4607_Diversity%20Matters_%20Revisiting%20Test-Time%20Compute%20in%20Vision-Language%20Models/images/495c5d3fd02893bfb2f9fd6e45df471212819efe94e814cd9e563a44e7432051.jpg)

![6a165f5674e561e84a1d6b37d08e035bc57b2465cdd44d0032772e37d7e52904.jpg](../icml_results/4607_Diversity%20Matters_%20Revisiting%20Test-Time%20Compute%20in%20Vision-Language%20Models/images/6a165f5674e561e84a1d6b37d08e035bc57b2465cdd44d0032772e37d7e52904.jpg)

![aa247c8df9c0cd8d9fb21e532465493b2da9bd72a2296f2add2cfe49cff0a017.jpg](../icml_results/4607_Diversity%20Matters_%20Revisiting%20Test-Time%20Compute%20in%20Vision-Language%20Models/images/aa247c8df9c0cd8d9fb21e532465493b2da9bd72a2296f2add2cfe49cff0a017.jpg)

![aa5155305e9297d64c5a922af3a3239f9305bacc3fd21d23d4caa54c7876c55d.jpg](../icml_results/4607_Diversity%20Matters_%20Revisiting%20Test-Time%20Compute%20in%20Vision-Language%20Models/images/aa5155305e9297d64c5a922af3a3239f9305bacc3fd21d23d4caa54c7876c55d.jpg)

### Tables

![2b7fba602aee79ea0fe318e19d209f1609b3d0c14aa81697642090b8638d5ee8.jpg](../icml_results/4607_Diversity%20Matters_%20Revisiting%20Test-Time%20Compute%20in%20Vision-Language%20Models/tables/2b7fba602aee79ea0fe318e19d209f1609b3d0c14aa81697642090b8638d5ee8.jpg)

![401c24bef1832e1a86adcc1c7f68b83270a8b3a970eaaaead781b129018f9830.jpg](../icml_results/4607_Diversity%20Matters_%20Revisiting%20Test-Time%20Compute%20in%20Vision-Language%20Models/tables/401c24bef1832e1a86adcc1c7f68b83270a8b3a970eaaaead781b129018f9830.jpg)

![4395a394229b55eb4ae53a35d2215a9d50c2e3bbd436afbfa741d0eecb3fde53.jpg](../icml_results/4607_Diversity%20Matters_%20Revisiting%20Test-Time%20Compute%20in%20Vision-Language%20Models/tables/4395a394229b55eb4ae53a35d2215a9d50c2e3bbd436afbfa741d0eecb3fde53.jpg)

![514049d90dbac81b1384de8a5f78e6c88a6f3152cb5c4314ffdf9dac05ace8c5.jpg](../icml_results/4607_Diversity%20Matters_%20Revisiting%20Test-Time%20Compute%20in%20Vision-Language%20Models/tables/514049d90dbac81b1384de8a5f78e6c88a6f3152cb5c4314ffdf9dac05ace8c5.jpg)

![7a7538cb0c58dc3d822a05c25bffcfe0c85ce2a6647ae0648c82db39a852fd11.jpg](../icml_results/4607_Diversity%20Matters_%20Revisiting%20Test-Time%20Compute%20in%20Vision-Language%20Models/tables/7a7538cb0c58dc3d822a05c25bffcfe0c85ce2a6647ae0648c82db39a852fd11.jpg)

![9d73b8ffe35a650823c91fad7ec3993b8b71881729356ffaa94d04af33c4a466.jpg](../icml_results/4607_Diversity%20Matters_%20Revisiting%20Test-Time%20Compute%20in%20Vision-Language%20Models/tables/9d73b8ffe35a650823c91fad7ec3993b8b71881729356ffaa94d04af33c4a466.jpg)

![aff6986da74e0b1ded1c296de0fa2e9fe78dab67dcc60ab42cd8140ef5f8a588.jpg](../icml_results/4607_Diversity%20Matters_%20Revisiting%20Test-Time%20Compute%20in%20Vision-Language%20Models/tables/aff6986da74e0b1ded1c296de0fa2e9fe78dab67dcc60ab42cd8140ef5f8a588.jpg)

![bd430177564e5c0c97a08bf9a712c93259d16cb738ba0d3ccad34a90b72a63b3.jpg](../icml_results/4607_Diversity%20Matters_%20Revisiting%20Test-Time%20Compute%20in%20Vision-Language%20Models/tables/bd430177564e5c0c97a08bf9a712c93259d16cb738ba0d3ccad34a90b72a63b3.jpg)

![f9660b221cf43d699f12d7ec4e79946fc8f54f685b5d5f19a5208fd3591d5370.jpg](../icml_results/4607_Diversity%20Matters_%20Revisiting%20Test-Time%20Compute%20in%20Vision-Language%20Models/tables/f9660b221cf43d699f12d7ec4e79946fc8f54f685b5d5f19a5208fd3591d5370.jpg)

<a id="paper-4561"></a>

## Causal-Adapter: Taming Text-to-Image Diffusion for Faithful Counterfactual Generation


### Images

![00dd5e63e7b5301ae5325768a0816f942888525f02e121ba7a0b7c48432c102b.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/00dd5e63e7b5301ae5325768a0816f942888525f02e121ba7a0b7c48432c102b.jpg)

![016fbcb28af9f137e2460f6056b62003c5f10c3451a37988c5ab92036edd841b.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/016fbcb28af9f137e2460f6056b62003c5f10c3451a37988c5ab92036edd841b.jpg)

![17a9a81ba794e9fd8858eb92f73e84782e5be19fb936ed236526c8b6a401bd94.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/17a9a81ba794e9fd8858eb92f73e84782e5be19fb936ed236526c8b6a401bd94.jpg)

![18e394371161d3c95c7984f65894d374236649d9c41580dce1b0339c3f31179d.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/18e394371161d3c95c7984f65894d374236649d9c41580dce1b0339c3f31179d.jpg)

![1ac96cfcdb943cb3ab32f9e9a6cd4b01310affdf9105fe7c60e9623287cf6254.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/1ac96cfcdb943cb3ab32f9e9a6cd4b01310affdf9105fe7c60e9623287cf6254.jpg)

![1ca1e1fe379e0bac1e44a854d885027489ee3d21cd3e0d24a11708b7d38956a7.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/1ca1e1fe379e0bac1e44a854d885027489ee3d21cd3e0d24a11708b7d38956a7.jpg)

![1fab04b6d063631f956fb7281305d4a08ba28267b2356420199d0557dedb3943.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/1fab04b6d063631f956fb7281305d4a08ba28267b2356420199d0557dedb3943.jpg)

![200f5f3c37fe53a221d429908c87dfdc730fff31a679eabd5dc2b7ec9619413d.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/200f5f3c37fe53a221d429908c87dfdc730fff31a679eabd5dc2b7ec9619413d.jpg)

![23b5bf215cbf1ef2810831fe90f4e0ce85d366791e92e00f7f0f7cfeb22cb547.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/23b5bf215cbf1ef2810831fe90f4e0ce85d366791e92e00f7f0f7cfeb22cb547.jpg)

![252b0dfd66fa6ead90fc3a43579e64082abd2691ef96138bb81eeb154ce3620e.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/252b0dfd66fa6ead90fc3a43579e64082abd2691ef96138bb81eeb154ce3620e.jpg)

![2b90640f01e031b361d5279de1a2d013c325e48aef28d332f964be19880515e0.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/2b90640f01e031b361d5279de1a2d013c325e48aef28d332f964be19880515e0.jpg)

![2f73ef1c0161ac538f18e445091de1150166248491636aef109e696357927adc.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/2f73ef1c0161ac538f18e445091de1150166248491636aef109e696357927adc.jpg)

![32a42adf03c5af294eef4d774b5a783f6f99cdafaab023e222dc73a415b79103.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/32a42adf03c5af294eef4d774b5a783f6f99cdafaab023e222dc73a415b79103.jpg)

![35a68b26f359cb878c7695181cabb74ebb22d3ccd61eee6cedd039e354064d26.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/35a68b26f359cb878c7695181cabb74ebb22d3ccd61eee6cedd039e354064d26.jpg)

![43d15a2fc230263046258714a9543568a20393a85c58a88cf0ead4dfb6700ccc.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/43d15a2fc230263046258714a9543568a20393a85c58a88cf0ead4dfb6700ccc.jpg)

![4fa93bf20e550bef8d39aba0200c2309639b78efef518a094d6940f7f8c82581.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/4fa93bf20e550bef8d39aba0200c2309639b78efef518a094d6940f7f8c82581.jpg)

![567e9728390223aece7f6b96895175b40bf750e80d7de65cd00c2648aaabdd60.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/567e9728390223aece7f6b96895175b40bf750e80d7de65cd00c2648aaabdd60.jpg)

![56da55a24d9539e522e82bc2cb19aa211be6df05e82e0de2a4c3825f3e8551da.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/56da55a24d9539e522e82bc2cb19aa211be6df05e82e0de2a4c3825f3e8551da.jpg)

![57353c2781d7e43da3bbbf74fa98b170accbad2f23111236562d4b85e6af1e60.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/57353c2781d7e43da3bbbf74fa98b170accbad2f23111236562d4b85e6af1e60.jpg)

![5b572ad35d5aa8fcd19d9eea1ffd946e1317995260f8046889f650ac0ab2bdba.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/5b572ad35d5aa8fcd19d9eea1ffd946e1317995260f8046889f650ac0ab2bdba.jpg)

![621f22ab0eb0f3a1c19e7904cd7c28d1643942f4faa7c641092c6bd7da975e64.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/621f22ab0eb0f3a1c19e7904cd7c28d1643942f4faa7c641092c6bd7da975e64.jpg)

![66ed80f622e9b86f120874531482fbab3a04a374288c9439f5d9081ac8adfa21.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/66ed80f622e9b86f120874531482fbab3a04a374288c9439f5d9081ac8adfa21.jpg)

![6989a250aec770e424f48c752affdfa0973dd62b51207c86e0a038989d78325c.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/6989a250aec770e424f48c752affdfa0973dd62b51207c86e0a038989d78325c.jpg)

![6dd79badbc16f8f20660af1d14d1f3c8e7dde6e7932ae1752f53c80a753f3901.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/6dd79badbc16f8f20660af1d14d1f3c8e7dde6e7932ae1752f53c80a753f3901.jpg)

![72e57986bcb6b576033b46857a955d7f5a5fc6d69baec25c9fd5ae7b53207365.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/72e57986bcb6b576033b46857a955d7f5a5fc6d69baec25c9fd5ae7b53207365.jpg)

![7d27900d18f4031a91713140de4bd85f661668dfc3d78a63060b2dd8fc10d1b9.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/7d27900d18f4031a91713140de4bd85f661668dfc3d78a63060b2dd8fc10d1b9.jpg)

![83601ca8f6be10a7b7e4813fcb353d0092bbb860cc26e87d638ab0981407d508.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/83601ca8f6be10a7b7e4813fcb353d0092bbb860cc26e87d638ab0981407d508.jpg)

![89f73a8dfac0dce45fc1793f6fd3f838871447ed0d511744f2de147ccee022d1.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/89f73a8dfac0dce45fc1793f6fd3f838871447ed0d511744f2de147ccee022d1.jpg)

![93dfd1cac5bac81caff1aeb4ec713a991638d946d84bfdd288bcda2b042c301e.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/93dfd1cac5bac81caff1aeb4ec713a991638d946d84bfdd288bcda2b042c301e.jpg)

![96a68837638e86538027cc88dadb79f74353b1a44ec1d6e08b1f22780b88d18d.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/96a68837638e86538027cc88dadb79f74353b1a44ec1d6e08b1f22780b88d18d.jpg)

![99ee2941efdffb305af4d916cbfa5432cd1d28c19c886a104c84f95bb3febdc7.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/99ee2941efdffb305af4d916cbfa5432cd1d28c19c886a104c84f95bb3febdc7.jpg)

![acaa899c9901a9ead0a952caab966e71787287aef2cc090e9ee4f7949aff588c.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/acaa899c9901a9ead0a952caab966e71787287aef2cc090e9ee4f7949aff588c.jpg)

![affcf1e927e9bb6b4961af7f4fb8bba1793726802a4dae744fab49ea5da28d57.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/affcf1e927e9bb6b4961af7f4fb8bba1793726802a4dae744fab49ea5da28d57.jpg)

![b08d8c02937e05059e47ceaa75ec4cbaa5da2dd6c5e2947dae1011db116e2ed1.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/b08d8c02937e05059e47ceaa75ec4cbaa5da2dd6c5e2947dae1011db116e2ed1.jpg)

![bcbfad1db16c1bac7c1f4bb5455a5913f276d16247f139bdb8d845ab0f149d0a.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/bcbfad1db16c1bac7c1f4bb5455a5913f276d16247f139bdb8d845ab0f149d0a.jpg)

![bd2ef5a1999328dd508c64fea7f2b36a7c87070eaf07647c40c191f362aa7dfb.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/bd2ef5a1999328dd508c64fea7f2b36a7c87070eaf07647c40c191f362aa7dfb.jpg)

![bf5a69c737e3e21f39a71de0b478baf081a7f74d7d3813a0d2890c7ac8143834.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/bf5a69c737e3e21f39a71de0b478baf081a7f74d7d3813a0d2890c7ac8143834.jpg)

![cae4a3c39d085386a8ed6a0272e10b8fe076737b0237b1346552774d87d4adbf.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/cae4a3c39d085386a8ed6a0272e10b8fe076737b0237b1346552774d87d4adbf.jpg)

![d15bea1466f95c51f55f7217832b123433a9e31835364a7cefc828a2404c463f.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/d15bea1466f95c51f55f7217832b123433a9e31835364a7cefc828a2404c463f.jpg)

![d3d9bf8d546c777ba5bc141c208b488968531a1a422f9c1f2168a1190045d5bd.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/d3d9bf8d546c777ba5bc141c208b488968531a1a422f9c1f2168a1190045d5bd.jpg)

![d703c3ade8d7fb0b9fa994d060d2998410ff77ead00afc202e3f41f604dab79b.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/d703c3ade8d7fb0b9fa994d060d2998410ff77ead00afc202e3f41f604dab79b.jpg)

![dd84a3fd6a88592d34125a79eda2fc5359bc6da04d550006088710c82caeebb0.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/dd84a3fd6a88592d34125a79eda2fc5359bc6da04d550006088710c82caeebb0.jpg)

![de0c0a089c38b3a54bc3d02e2656be0a1e474c757b9e1db45a37a7f17adb0755.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/de0c0a089c38b3a54bc3d02e2656be0a1e474c757b9e1db45a37a7f17adb0755.jpg)

![de3ff259067f51cec08241c395f423f096490c719e2b491a4280f1b941ad8210.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/de3ff259067f51cec08241c395f423f096490c719e2b491a4280f1b941ad8210.jpg)

![e3c7ca602d16e05dcb9fd566059b2d7eb7b303ac6a60cb940cf5c7164854100f.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/e3c7ca602d16e05dcb9fd566059b2d7eb7b303ac6a60cb940cf5c7164854100f.jpg)

![e3de732759d9e2d571b02b74ce392d40815b3e7b366d62f5f025c00818eebc70.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/e3de732759d9e2d571b02b74ce392d40815b3e7b366d62f5f025c00818eebc70.jpg)

![e572d5cefc271bd34daae097a41606ae809baeeb33c90ff58738693a0574f8e8.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/e572d5cefc271bd34daae097a41606ae809baeeb33c90ff58738693a0574f8e8.jpg)

![e8d5a9d95d3f1227fa2101ec0acf8bd39515a430742807568113d7cb000dced4.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/e8d5a9d95d3f1227fa2101ec0acf8bd39515a430742807568113d7cb000dced4.jpg)

![ec7435361cf7c6b17769bc0141d68b240a7eb66f9689c589b7e13397187d864b.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/ec7435361cf7c6b17769bc0141d68b240a7eb66f9689c589b7e13397187d864b.jpg)

![f136512013fe8a48cedb6f6babc4bd0a13decfa051ea3604eef4698f01b86e5c.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/f136512013fe8a48cedb6f6babc4bd0a13decfa051ea3604eef4698f01b86e5c.jpg)

![f50ef0ee8dcca0d8d01c5f6e48d6638f8a9b4e136f5bc7c7d5106c413c69c86e.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/f50ef0ee8dcca0d8d01c5f6e48d6638f8a9b4e136f5bc7c7d5106c413c69c86e.jpg)

![f58f7e5b76a98387b9f73b2a377020c0ff2344a49c82d99b8c6376a33a890e37.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/images/f58f7e5b76a98387b9f73b2a377020c0ff2344a49c82d99b8c6376a33a890e37.jpg)

### Tables

![128432466d358e696a0b34adefc62067e02a61ef43a550760afdac54fe4b7570.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/128432466d358e696a0b34adefc62067e02a61ef43a550760afdac54fe4b7570.jpg)

![12f13e7e588a42b64710aa424b81e1cbac8e5cf2a4f7e3f7fa329aacd0c43c4a.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/12f13e7e588a42b64710aa424b81e1cbac8e5cf2a4f7e3f7fa329aacd0c43c4a.jpg)

![227f19d55ce939cdea395569e1f506b9cccb55cc975a423b7b7d68b90ac10ae6.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/227f19d55ce939cdea395569e1f506b9cccb55cc975a423b7b7d68b90ac10ae6.jpg)

![3cd571a5167a59c5541a567f8ac72db0b896b30231dc3f255ae828954afe9bd0.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/3cd571a5167a59c5541a567f8ac72db0b896b30231dc3f255ae828954afe9bd0.jpg)

![59e78e36d158692abf669deda76a3f6906a77bb570364d3108d242313ef0d876.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/59e78e36d158692abf669deda76a3f6906a77bb570364d3108d242313ef0d876.jpg)

![6001512a01de179a3ff91252a8826e41dc5c668649b344fd3b1760e124a87ab1.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/6001512a01de179a3ff91252a8826e41dc5c668649b344fd3b1760e124a87ab1.jpg)

![6c375e6e729fe58d3b5e858422ccd32a8cfb1382e6e8d2101f156a9dc80cebf9.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/6c375e6e729fe58d3b5e858422ccd32a8cfb1382e6e8d2101f156a9dc80cebf9.jpg)

![6c8fe5ef1c6be82c52bd3b4715e1ecf58376a94b559c5c2e649acda828d68818.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/6c8fe5ef1c6be82c52bd3b4715e1ecf58376a94b559c5c2e649acda828d68818.jpg)

![89d9f8380c5eac55d27fd6477e029382adb7caa5e56c4aa4583cf7e4f6f19cd7.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/89d9f8380c5eac55d27fd6477e029382adb7caa5e56c4aa4583cf7e4f6f19cd7.jpg)

![8c431bb5c0585da227a995c3195531c6b9d1f453b80064666537082ab0f5894e.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/8c431bb5c0585da227a995c3195531c6b9d1f453b80064666537082ab0f5894e.jpg)

![9e9ec5dd6c68a984ddaf551af840a12a6117b51efd531e7f2e469bcd791a12a8.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/9e9ec5dd6c68a984ddaf551af840a12a6117b51efd531e7f2e469bcd791a12a8.jpg)

![a2cdb29323c2a12ba1310e52b305752a5e9bd722a1f4cd2ace688180653d2f68.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/a2cdb29323c2a12ba1310e52b305752a5e9bd722a1f4cd2ace688180653d2f68.jpg)

![a5702c8d03f31ebc9a3fb359a0ff0554ac7a2bddf95d978c1b414cd1a73784bc.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/a5702c8d03f31ebc9a3fb359a0ff0554ac7a2bddf95d978c1b414cd1a73784bc.jpg)

![af780d7dd04a07b86ebe7e6e3e62563b248060961a08683e21175f76af914b44.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/af780d7dd04a07b86ebe7e6e3e62563b248060961a08683e21175f76af914b44.jpg)

![c9b75086442f32f074f1aaaf86241379ba6c1f5a054368b2937597c9c83ecf9a.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/c9b75086442f32f074f1aaaf86241379ba6c1f5a054368b2937597c9c83ecf9a.jpg)

![dcdc62b60a3a635c91231ac30f3365ac365ffee74728825e7a066286ac81a8d4.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/dcdc62b60a3a635c91231ac30f3365ac365ffee74728825e7a066286ac81a8d4.jpg)

![f1978fc183ea4e7d1b1e0261325dfb1847bdd79158d17d2870584b3969979eef.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/f1978fc183ea4e7d1b1e0261325dfb1847bdd79158d17d2870584b3969979eef.jpg)

![f51e1f70e09ff4c0ba4b73fd67e3a1c089e298186b3e10df0043adc6684c6f09.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/f51e1f70e09ff4c0ba4b73fd67e3a1c089e298186b3e10df0043adc6684c6f09.jpg)

![f6f00be9ca23d3506fc8476f59ac73ff8d8b867defb5986e90b20f29ed6f16ca.jpg](../icml_results/4608_Causal-Adapter_%20Taming%20Text-to-Image%20Diffusion%20for%20Faithful%20Counterfactual%20Generation/tables/f6f00be9ca23d3506fc8476f59ac73ff8d8b867defb5986e90b20f29ed6f16ca.jpg)

<a id="paper-4562"></a>

## Alleviating Sparse Rewards by Modeling Step-Wise and Long-Term Sampling Effects in Flow-Based GRPO


### Images

![02a20c87cafba2443514990a23046eaaf745c2e70e2d3e5d375e51466adb28a5.jpg](../icml_results/4609_Alleviating%20Sparse%20Rewards%20by%20Modeling%20Step-Wise%20and%20Long-Term%20Sampling%20Effects%20in%20Flow-Based%20GRPO/images/02a20c87cafba2443514990a23046eaaf745c2e70e2d3e5d375e51466adb28a5.jpg)

![06b07dc2ffe630552d2431e90b1cfc5278ac30e5e07fe998a2babed7cd489724.jpg](../icml_results/4609_Alleviating%20Sparse%20Rewards%20by%20Modeling%20Step-Wise%20and%20Long-Term%20Sampling%20Effects%20in%20Flow-Based%20GRPO/images/06b07dc2ffe630552d2431e90b1cfc5278ac30e5e07fe998a2babed7cd489724.jpg)

![06b2c6a9aaac3679a97a12db57c64d2bf325823f7b5ce50ffc72804dc561092e.jpg](../icml_results/4609_Alleviating%20Sparse%20Rewards%20by%20Modeling%20Step-Wise%20and%20Long-Term%20Sampling%20Effects%20in%20Flow-Based%20GRPO/images/06b2c6a9aaac3679a97a12db57c64d2bf325823f7b5ce50ffc72804dc561092e.jpg)

![06fb8f49b21fc2e77b842946865284ee97059fd6dcaa6a0995f526d3e3186189.jpg](../icml_results/4609_Alleviating%20Sparse%20Rewards%20by%20Modeling%20Step-Wise%20and%20Long-Term%20Sampling%20Effects%20in%20Flow-Based%20GRPO/images/06fb8f49b21fc2e77b842946865284ee97059fd6dcaa6a0995f526d3e3186189.jpg)

![12b68e8b9f08018aafa7858d1b8b915b4e8aad0a6288a07303c0358cd0338b01.jpg](../icml_results/4609_Alleviating%20Sparse%20Rewards%20by%20Modeling%20Step-Wise%20and%20Long-Term%20Sampling%20Effects%20in%20Flow-Based%20GRPO/images/12b68e8b9f08018aafa7858d1b8b915b4e8aad0a6288a07303c0358cd0338b01.jpg)

![2350f6d3b35d415d9560c8303237b6a0aa812e2975ccc74711cdb14fdf60dfba.jpg](../icml_results/4609_Alleviating%20Sparse%20Rewards%20by%20Modeling%20Step-Wise%20and%20Long-Term%20Sampling%20Effects%20in%20Flow-Based%20GRPO/images/2350f6d3b35d415d9560c8303237b6a0aa812e2975ccc74711cdb14fdf60dfba.jpg)

![2c72ae048c40d95239dba11c0b95f2c285b1ad6ab61e869ef9eb3a4b4b4695c6.jpg](../icml_results/4609_Alleviating%20Sparse%20Rewards%20by%20Modeling%20Step-Wise%20and%20Long-Term%20Sampling%20Effects%20in%20Flow-Based%20GRPO/images/2c72ae048c40d95239dba11c0b95f2c285b1ad6ab61e869ef9eb3a4b4b4695c6.jpg)

![30e1f0543ed4d4b606b449df1bbf8251852df9ac819560895c6478d0e43f83b5.jpg](../icml_results/4609_Alleviating%20Sparse%20Rewards%20by%20Modeling%20Step-Wise%20and%20Long-Term%20Sampling%20Effects%20in%20Flow-Based%20GRPO/images/30e1f0543ed4d4b606b449df1bbf8251852df9ac819560895c6478d0e43f83b5.jpg)

![6eb55d55d4e6ccb46d60b4b01318381c5e8f1212587bf87c3ec8d3846c09b368.jpg](../icml_results/4609_Alleviating%20Sparse%20Rewards%20by%20Modeling%20Step-Wise%20and%20Long-Term%20Sampling%20Effects%20in%20Flow-Based%20GRPO/images/6eb55d55d4e6ccb46d60b4b01318381c5e8f1212587bf87c3ec8d3846c09b368.jpg)

![78c23e8e3ec2a759b9751f29f39f13c7395d028c0fbe00c5073c705c74e39422.jpg](../icml_results/4609_Alleviating%20Sparse%20Rewards%20by%20Modeling%20Step-Wise%20and%20Long-Term%20Sampling%20Effects%20in%20Flow-Based%20GRPO/images/78c23e8e3ec2a759b9751f29f39f13c7395d028c0fbe00c5073c705c74e39422.jpg)

![cd15262dc96e84e8e6db37e004223c6f8f8b55f4efa025ea5da797e4c02c3533.jpg](../icml_results/4609_Alleviating%20Sparse%20Rewards%20by%20Modeling%20Step-Wise%20and%20Long-Term%20Sampling%20Effects%20in%20Flow-Based%20GRPO/images/cd15262dc96e84e8e6db37e004223c6f8f8b55f4efa025ea5da797e4c02c3533.jpg)

![db3a7b1a14ef39e45e1ca7e0b314467c1c8b0a7ede2db5a8a67621e56464baae.jpg](../icml_results/4609_Alleviating%20Sparse%20Rewards%20by%20Modeling%20Step-Wise%20and%20Long-Term%20Sampling%20Effects%20in%20Flow-Based%20GRPO/images/db3a7b1a14ef39e45e1ca7e0b314467c1c8b0a7ede2db5a8a67621e56464baae.jpg)

![f4777f8cb41e571e9981569bb9c0dd8beda59c90046c93995fdb20cf2fe7ef37.jpg](../icml_results/4609_Alleviating%20Sparse%20Rewards%20by%20Modeling%20Step-Wise%20and%20Long-Term%20Sampling%20Effects%20in%20Flow-Based%20GRPO/images/f4777f8cb41e571e9981569bb9c0dd8beda59c90046c93995fdb20cf2fe7ef37.jpg)

![f88d5ef889a1a31f1f30be296c6626a010b3bf666edf5f8906cf84d2c909abf5.jpg](../icml_results/4609_Alleviating%20Sparse%20Rewards%20by%20Modeling%20Step-Wise%20and%20Long-Term%20Sampling%20Effects%20in%20Flow-Based%20GRPO/images/f88d5ef889a1a31f1f30be296c6626a010b3bf666edf5f8906cf84d2c909abf5.jpg)

![fadde201dd22bf6a6e22fc0f1805381401df143c23d2328e42d79cb70a503c84.jpg](../icml_results/4609_Alleviating%20Sparse%20Rewards%20by%20Modeling%20Step-Wise%20and%20Long-Term%20Sampling%20Effects%20in%20Flow-Based%20GRPO/images/fadde201dd22bf6a6e22fc0f1805381401df143c23d2328e42d79cb70a503c84.jpg)

### Tables

![112dc605fd9d4f79b59562606af940622d490e57db0a6108e00c2c3bd254a3a8.jpg](../icml_results/4609_Alleviating%20Sparse%20Rewards%20by%20Modeling%20Step-Wise%20and%20Long-Term%20Sampling%20Effects%20in%20Flow-Based%20GRPO/tables/112dc605fd9d4f79b59562606af940622d490e57db0a6108e00c2c3bd254a3a8.jpg)

![eee71d60ce48076456383dfaa768d2f785ac063efc1bd10396f779ea18d39d99.jpg](../icml_results/4609_Alleviating%20Sparse%20Rewards%20by%20Modeling%20Step-Wise%20and%20Long-Term%20Sampling%20Effects%20in%20Flow-Based%20GRPO/tables/eee71d60ce48076456383dfaa768d2f785ac063efc1bd10396f779ea18d39d99.jpg)

<a id="paper-4563"></a>

## VideoMAETok: Boosting Video Diffusion Models via Masked Autoencoders as Tokenizers


### Images

![05ec92d6d6772856b76bf5418893fe819570cb6b3592489a2633a46dd2224556.jpg](../icml_results/4610_VideoMAETok_%20Boosting%20Video%20Diffusion%20Models%20via%20Masked%20Autoencoders%20as%20Tokenizers/images/05ec92d6d6772856b76bf5418893fe819570cb6b3592489a2633a46dd2224556.jpg)

![1a818bba4edae8c99133ef69a88b75824b258e60756f6b2deacf9c9ba930e161.jpg](../icml_results/4610_VideoMAETok_%20Boosting%20Video%20Diffusion%20Models%20via%20Masked%20Autoencoders%20as%20Tokenizers/images/1a818bba4edae8c99133ef69a88b75824b258e60756f6b2deacf9c9ba930e161.jpg)

![70f97544390d5b3ec60d627ef5f2e52b08e53f70b9a8176d2ad3a441aaea9c2d.jpg](../icml_results/4610_VideoMAETok_%20Boosting%20Video%20Diffusion%20Models%20via%20Masked%20Autoencoders%20as%20Tokenizers/images/70f97544390d5b3ec60d627ef5f2e52b08e53f70b9a8176d2ad3a441aaea9c2d.jpg)

![8cc233fcf191421d114fa45700c69672b052c5abe7c603b1af44ba049c1fbd21.jpg](../icml_results/4610_VideoMAETok_%20Boosting%20Video%20Diffusion%20Models%20via%20Masked%20Autoencoders%20as%20Tokenizers/images/8cc233fcf191421d114fa45700c69672b052c5abe7c603b1af44ba049c1fbd21.jpg)

![baf9729160040a6ff45348e06ad128a5b5ae6c5d12e3a39211849462ed5b1c63.jpg](../icml_results/4610_VideoMAETok_%20Boosting%20Video%20Diffusion%20Models%20via%20Masked%20Autoencoders%20as%20Tokenizers/images/baf9729160040a6ff45348e06ad128a5b5ae6c5d12e3a39211849462ed5b1c63.jpg)

![ca49beaabcf59af1796c0208979c817a98272f265d6d6d28f1f6bee54520f83a.jpg](../icml_results/4610_VideoMAETok_%20Boosting%20Video%20Diffusion%20Models%20via%20Masked%20Autoencoders%20as%20Tokenizers/images/ca49beaabcf59af1796c0208979c817a98272f265d6d6d28f1f6bee54520f83a.jpg)

![fc86d5083159f04a3096aaba802e4ac33a1a126384c59f9cbc2a2425dc11b69a.jpg](../icml_results/4610_VideoMAETok_%20Boosting%20Video%20Diffusion%20Models%20via%20Masked%20Autoencoders%20as%20Tokenizers/images/fc86d5083159f04a3096aaba802e4ac33a1a126384c59f9cbc2a2425dc11b69a.jpg)

### Tables

![0fbb5df9a499079439f66a76e218d4bb3b639cf64709edd39e342d159b2e2243.jpg](../icml_results/4610_VideoMAETok_%20Boosting%20Video%20Diffusion%20Models%20via%20Masked%20Autoencoders%20as%20Tokenizers/tables/0fbb5df9a499079439f66a76e218d4bb3b639cf64709edd39e342d159b2e2243.jpg)

![17b5ff05a59dd6497afa4b688b5fe36b1f412930fcc97de54fb95bbca02a8a40.jpg](../icml_results/4610_VideoMAETok_%20Boosting%20Video%20Diffusion%20Models%20via%20Masked%20Autoencoders%20as%20Tokenizers/tables/17b5ff05a59dd6497afa4b688b5fe36b1f412930fcc97de54fb95bbca02a8a40.jpg)

![364dbccaf398d88daf081e40fbb927d760ae8b0cb1e184f78ad21e5ef42e6b69.jpg](../icml_results/4610_VideoMAETok_%20Boosting%20Video%20Diffusion%20Models%20via%20Masked%20Autoencoders%20as%20Tokenizers/tables/364dbccaf398d88daf081e40fbb927d760ae8b0cb1e184f78ad21e5ef42e6b69.jpg)

![75e3023084f74d6eca160ef1180fd983d8a844732d1814629324ddcfdcb56f1e.jpg](../icml_results/4610_VideoMAETok_%20Boosting%20Video%20Diffusion%20Models%20via%20Masked%20Autoencoders%20as%20Tokenizers/tables/75e3023084f74d6eca160ef1180fd983d8a844732d1814629324ddcfdcb56f1e.jpg)

![9fc8b73ac08918b5b31e1d6769d80c5c0d5d97d0a1d7cd0a724e3df877079909.jpg](../icml_results/4610_VideoMAETok_%20Boosting%20Video%20Diffusion%20Models%20via%20Masked%20Autoencoders%20as%20Tokenizers/tables/9fc8b73ac08918b5b31e1d6769d80c5c0d5d97d0a1d7cd0a724e3df877079909.jpg)

![de736ee36f680ce6e618ec989d76be91b623d9e49d8492350ee45bf54875cfa3.jpg](../icml_results/4610_VideoMAETok_%20Boosting%20Video%20Diffusion%20Models%20via%20Masked%20Autoencoders%20as%20Tokenizers/tables/de736ee36f680ce6e618ec989d76be91b623d9e49d8492350ee45bf54875cfa3.jpg)

![dff37dc878d3073eb52a7da370b97f41fd946e653d25f5efe5dfa5c826c406c4.jpg](../icml_results/4610_VideoMAETok_%20Boosting%20Video%20Diffusion%20Models%20via%20Masked%20Autoencoders%20as%20Tokenizers/tables/dff37dc878d3073eb52a7da370b97f41fd946e653d25f5efe5dfa5c826c406c4.jpg)

![f2a240511306470463a093259320abf4c743634c90e81069088f380ef52b089d.jpg](../icml_results/4610_VideoMAETok_%20Boosting%20Video%20Diffusion%20Models%20via%20Masked%20Autoencoders%20as%20Tokenizers/tables/f2a240511306470463a093259320abf4c743634c90e81069088f380ef52b089d.jpg)

<a id="paper-4564"></a>

## Neural Control: Adjoint Learning Through Equilibrium Constraints


### Images

![1bd5b2bd33554d2ab4088afba89ca3eea350887d9c86991ae5946b1fd1f9107d.jpg](../icml_results/4611_Neural%20Control_%20Adjoint%20Learning%20Through%20Equilibrium%20Constraints/images/1bd5b2bd33554d2ab4088afba89ca3eea350887d9c86991ae5946b1fd1f9107d.jpg)

![562cb353a4b56a6e659e31da8fe426c84baaeba4f6f20638994adcf178f2fa63.jpg](../icml_results/4611_Neural%20Control_%20Adjoint%20Learning%20Through%20Equilibrium%20Constraints/images/562cb353a4b56a6e659e31da8fe426c84baaeba4f6f20638994adcf178f2fa63.jpg)

![70b0c86c290ba92b921db2832c749015b4666cc9b898cb78b62ea032805c8c64.jpg](../icml_results/4611_Neural%20Control_%20Adjoint%20Learning%20Through%20Equilibrium%20Constraints/images/70b0c86c290ba92b921db2832c749015b4666cc9b898cb78b62ea032805c8c64.jpg)

![7e2e9183e653aec4ce87cbf20391d92586c9831cf126b7448d187d5dc15ef126.jpg](../icml_results/4611_Neural%20Control_%20Adjoint%20Learning%20Through%20Equilibrium%20Constraints/images/7e2e9183e653aec4ce87cbf20391d92586c9831cf126b7448d187d5dc15ef126.jpg)

![ac506e078704cf54cda9775dea2be6843828cb53a9c036c5526d0e8e377471b4.jpg](../icml_results/4611_Neural%20Control_%20Adjoint%20Learning%20Through%20Equilibrium%20Constraints/images/ac506e078704cf54cda9775dea2be6843828cb53a9c036c5526d0e8e377471b4.jpg)

![f0443902763a1fb1c96df08914b16b0f3c6525d252315e25a8f652bf714b28cd.jpg](../icml_results/4611_Neural%20Control_%20Adjoint%20Learning%20Through%20Equilibrium%20Constraints/images/f0443902763a1fb1c96df08914b16b0f3c6525d252315e25a8f652bf714b28cd.jpg)

### Tables

![00b0cd85925266abca0a51f01ade9fe6a188ec7cf5ede482c3368e526c662eb5.jpg](../icml_results/4611_Neural%20Control_%20Adjoint%20Learning%20Through%20Equilibrium%20Constraints/tables/00b0cd85925266abca0a51f01ade9fe6a188ec7cf5ede482c3368e526c662eb5.jpg)

![ca9e5d84264ca25f70a850a40e381ddb92621d0ec68312f57518abc54c9db60d.jpg](../icml_results/4611_Neural%20Control_%20Adjoint%20Learning%20Through%20Equilibrium%20Constraints/tables/ca9e5d84264ca25f70a850a40e381ddb92621d0ec68312f57518abc54c9db60d.jpg)

![fef71d80808285cac2d2b68dec64242b6798bb65e4a466f5124808ef3cdd07eb.jpg](../icml_results/4611_Neural%20Control_%20Adjoint%20Learning%20Through%20Equilibrium%20Constraints/tables/fef71d80808285cac2d2b68dec64242b6798bb65e4a466f5124808ef3cdd07eb.jpg)

<a id="paper-4565"></a>

## Entropy-Aware Dynamic KV Cache Sparsification for Autoregressive Image Generation and Editing


### Images

![0d11e4c2946826f0bf490219c3bdb0122c4e2c667d33fa674e56fcfd950b658a.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/images/0d11e4c2946826f0bf490219c3bdb0122c4e2c667d33fa674e56fcfd950b658a.jpg)

![2762e2b42394e224cef41364e8315f9c41bb5e6fa2aa6707ecfc236ed9257a61.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/images/2762e2b42394e224cef41364e8315f9c41bb5e6fa2aa6707ecfc236ed9257a61.jpg)

![2a6b4d999836673e5b68674df1cabd1de94092650e5ecd8df2d1426a319bced6.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/images/2a6b4d999836673e5b68674df1cabd1de94092650e5ecd8df2d1426a319bced6.jpg)

![3696df3aa37eea961c820adc705f44ff229ab6dbaaabec419119df574b52411b.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/images/3696df3aa37eea961c820adc705f44ff229ab6dbaaabec419119df574b52411b.jpg)

![46b4611b42a8bff1924d33280e6a3216544f0c176e419c98b78a554599f95618.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/images/46b4611b42a8bff1924d33280e6a3216544f0c176e419c98b78a554599f95618.jpg)

![46d52930f3dd0dc8d1f9aae4fff0f8846ef86493c627073de9ff27199791a556.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/images/46d52930f3dd0dc8d1f9aae4fff0f8846ef86493c627073de9ff27199791a556.jpg)

![4b0cad6ff789fb5af17dc1f05f58734d1360feafe8ee3bdb0906edaa2daf4e45.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/images/4b0cad6ff789fb5af17dc1f05f58734d1360feafe8ee3bdb0906edaa2daf4e45.jpg)

![64691a2221a94b60213cb501f6854178b3cb3d13db4dec2e9a606fd20e956c17.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/images/64691a2221a94b60213cb501f6854178b3cb3d13db4dec2e9a606fd20e956c17.jpg)

![7818d1d69b3eb9fb456da1273ca4ef5ac995d0ccfb09ec365706e28f0d9db9ff.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/images/7818d1d69b3eb9fb456da1273ca4ef5ac995d0ccfb09ec365706e28f0d9db9ff.jpg)

![8e0a4cb361de78e249faa82846adce1149a981957e98a412efad6f0ef933b946.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/images/8e0a4cb361de78e249faa82846adce1149a981957e98a412efad6f0ef933b946.jpg)

![978a206e4368fda0c51301a07eb482091760b35ff1478a8a6cef6114ddcc4b79.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/images/978a206e4368fda0c51301a07eb482091760b35ff1478a8a6cef6114ddcc4b79.jpg)

![a4fe8e92e56ae3c1b47960198d774d65dfaa13026221950df76c6c03f43f0a4f.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/images/a4fe8e92e56ae3c1b47960198d774d65dfaa13026221950df76c6c03f43f0a4f.jpg)

![a74632c9a5ca556ab4528fd4a2e5293a6135c990445019f71049714245b2bda6.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/images/a74632c9a5ca556ab4528fd4a2e5293a6135c990445019f71049714245b2bda6.jpg)

![c32c8e6aff69ba81ca4221dfe065a1ec32c072d02dd7cb04ef942d1c7ed0792d.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/images/c32c8e6aff69ba81ca4221dfe065a1ec32c072d02dd7cb04ef942d1c7ed0792d.jpg)

![e8e1dd08b555501cb8cae81d03aa764a006e10b7524924f2e63662900983c117.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/images/e8e1dd08b555501cb8cae81d03aa764a006e10b7524924f2e63662900983c117.jpg)

### Tables

![343816f9d42f7cb41158d284e6c94de239081a1edf1808b391d4e90ca818eb3f.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/tables/343816f9d42f7cb41158d284e6c94de239081a1edf1808b391d4e90ca818eb3f.jpg)

![3722f34a5c5efbbbdd7926c516250395885167f00cf035246c94800bf076740f.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/tables/3722f34a5c5efbbbdd7926c516250395885167f00cf035246c94800bf076740f.jpg)

![3c5451dea735b429cc889ac02ccb283207b7ee6cedca7558fcb78dec5da7a5d5.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/tables/3c5451dea735b429cc889ac02ccb283207b7ee6cedca7558fcb78dec5da7a5d5.jpg)

![64f24bba4aff3c59a639f3fa1b66c70fcf9a93654301577c82b9bc469c4fb2c3.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/tables/64f24bba4aff3c59a639f3fa1b66c70fcf9a93654301577c82b9bc469c4fb2c3.jpg)

![f1a60c039f44a5310556e9ecd1d8d20f6e9669e21983ef7f87138149b864c37b.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/tables/f1a60c039f44a5310556e9ecd1d8d20f6e9669e21983ef7f87138149b864c37b.jpg)

![f99eddfc0dddbeda44b76c98e62ef1d79d14de3bfd499d30e6600b36b0d812d8.jpg](../icml_results/4612_Entropy-Aware%20Dynamic%20KV%20Cache%20Sparsification%20for%20Autoregressive%20Image%20Generation%20and%20Editing/tables/f99eddfc0dddbeda44b76c98e62ef1d79d14de3bfd499d30e6600b36b0d812d8.jpg)

<a id="paper-4566"></a>

## AuTAgent: A Reinforcement Learning Framework for Tool-Augmented Audio Reasoning


### Images

![61e6646bcd70585371b08b6453988c68d7cc704fc317ad009dfba3c2c8db1292.jpg](../icml_results/4613_AuTAgent_%20A%20Reinforcement%20Learning%20Framework%20for%20Tool-Augmented%20Audio%20Reasoning/images/61e6646bcd70585371b08b6453988c68d7cc704fc317ad009dfba3c2c8db1292.jpg)

![62b3cad8b06fa5c91bf207a58a3c0cc4322c4bc74035b1f9cdd2b4a0097893fe.jpg](../icml_results/4613_AuTAgent_%20A%20Reinforcement%20Learning%20Framework%20for%20Tool-Augmented%20Audio%20Reasoning/images/62b3cad8b06fa5c91bf207a58a3c0cc4322c4bc74035b1f9cdd2b4a0097893fe.jpg)

![c71699cf4ffc1ef029a344760b7c08873f2e23ab8ace06eadb6d4c3dd8bd89c3.jpg](../icml_results/4613_AuTAgent_%20A%20Reinforcement%20Learning%20Framework%20for%20Tool-Augmented%20Audio%20Reasoning/images/c71699cf4ffc1ef029a344760b7c08873f2e23ab8ace06eadb6d4c3dd8bd89c3.jpg)

![dd8e1902a8ad313aaad02beaab94266c93615cc76d761dc1bc2c49bcc6600a7b.jpg](../icml_results/4613_AuTAgent_%20A%20Reinforcement%20Learning%20Framework%20for%20Tool-Augmented%20Audio%20Reasoning/images/dd8e1902a8ad313aaad02beaab94266c93615cc76d761dc1bc2c49bcc6600a7b.jpg)

### Tables

![2a731810c5c5c431025427625d2e51ea3ee107cab486e01fd1b8ff47039825a7.jpg](../icml_results/4613_AuTAgent_%20A%20Reinforcement%20Learning%20Framework%20for%20Tool-Augmented%20Audio%20Reasoning/tables/2a731810c5c5c431025427625d2e51ea3ee107cab486e01fd1b8ff47039825a7.jpg)

![8876bf7fef08b30b90d64b755f4e5db093e547b431d131f8c72aa8259473e89d.jpg](../icml_results/4613_AuTAgent_%20A%20Reinforcement%20Learning%20Framework%20for%20Tool-Augmented%20Audio%20Reasoning/tables/8876bf7fef08b30b90d64b755f4e5db093e547b431d131f8c72aa8259473e89d.jpg)

![9c3bc8c4244a50de9fbba6463323ab0d31b14d0b6196874ac78c9135fd51c667.jpg](../icml_results/4613_AuTAgent_%20A%20Reinforcement%20Learning%20Framework%20for%20Tool-Augmented%20Audio%20Reasoning/tables/9c3bc8c4244a50de9fbba6463323ab0d31b14d0b6196874ac78c9135fd51c667.jpg)

![e03cfc8d32d439c1792a9a2690bc3cfcb6f972ed5ec467227b00e4f0036997cd.jpg](../icml_results/4613_AuTAgent_%20A%20Reinforcement%20Learning%20Framework%20for%20Tool-Augmented%20Audio%20Reasoning/tables/e03cfc8d32d439c1792a9a2690bc3cfcb6f972ed5ec467227b00e4f0036997cd.jpg)

![fb40d620b39e49c2b30924f3e34d46ded415d479412e28c0b2928b8d79479128.jpg](../icml_results/4613_AuTAgent_%20A%20Reinforcement%20Learning%20Framework%20for%20Tool-Augmented%20Audio%20Reasoning/tables/fb40d620b39e49c2b30924f3e34d46ded415d479412e28c0b2928b8d79479128.jpg)

<a id="paper-4567"></a>

## RefChess: Training-Free Contextual Search for Zero-Shot Referring Image Segmentation


### Images

![3ca8a90d5a75003cf61e0828088d0e86d67381dec2e60e85b46ba579453f069a.jpg](../icml_results/4614_RefChess_%20Training-Free%20Contextual%20Search%20for%20Zero-Shot%20Referring%20Image%20Segmentation/images/3ca8a90d5a75003cf61e0828088d0e86d67381dec2e60e85b46ba579453f069a.jpg)

![43f638b07d07740eec7641ff73625a1d87d6d27b011619b9da786554eb85df55.jpg](../icml_results/4614_RefChess_%20Training-Free%20Contextual%20Search%20for%20Zero-Shot%20Referring%20Image%20Segmentation/images/43f638b07d07740eec7641ff73625a1d87d6d27b011619b9da786554eb85df55.jpg)

![5d2a6024c845e2792d229da4b20fe2907db7722356a64b4d3957f87977564f6f.jpg](../icml_results/4614_RefChess_%20Training-Free%20Contextual%20Search%20for%20Zero-Shot%20Referring%20Image%20Segmentation/images/5d2a6024c845e2792d229da4b20fe2907db7722356a64b4d3957f87977564f6f.jpg)

![7a3e0320a80ebd5f2132d2e52517857ddb9a1e680b5a047b9201b66ca8b072d6.jpg](../icml_results/4614_RefChess_%20Training-Free%20Contextual%20Search%20for%20Zero-Shot%20Referring%20Image%20Segmentation/images/7a3e0320a80ebd5f2132d2e52517857ddb9a1e680b5a047b9201b66ca8b072d6.jpg)

![930964f2fa2422ff8101b97542f3be4c1da011f6b428a25655b2eccac2d7bc51.jpg](../icml_results/4614_RefChess_%20Training-Free%20Contextual%20Search%20for%20Zero-Shot%20Referring%20Image%20Segmentation/images/930964f2fa2422ff8101b97542f3be4c1da011f6b428a25655b2eccac2d7bc51.jpg)

![b9485b88bfbba645eb95f91d1c1f7eeb92de70919d5925bf4a2006d0b87c0fd2.jpg](../icml_results/4614_RefChess_%20Training-Free%20Contextual%20Search%20for%20Zero-Shot%20Referring%20Image%20Segmentation/images/b9485b88bfbba645eb95f91d1c1f7eeb92de70919d5925bf4a2006d0b87c0fd2.jpg)

![c115537e743b21cc3922fb51a64aac31ed41cb3bfaab8c98200d5de136595b1e.jpg](../icml_results/4614_RefChess_%20Training-Free%20Contextual%20Search%20for%20Zero-Shot%20Referring%20Image%20Segmentation/images/c115537e743b21cc3922fb51a64aac31ed41cb3bfaab8c98200d5de136595b1e.jpg)

![faa63a22bcea043e8912c3fa40dad9e77a334ba97c9344d026d2c42b24dea6eb.jpg](../icml_results/4614_RefChess_%20Training-Free%20Contextual%20Search%20for%20Zero-Shot%20Referring%20Image%20Segmentation/images/faa63a22bcea043e8912c3fa40dad9e77a334ba97c9344d026d2c42b24dea6eb.jpg)

### Tables

![2eb276d5f8f0d3289d4f5deddb5845c68f74fd8f21d7bda61d1413efbb2e9555.jpg](../icml_results/4614_RefChess_%20Training-Free%20Contextual%20Search%20for%20Zero-Shot%20Referring%20Image%20Segmentation/tables/2eb276d5f8f0d3289d4f5deddb5845c68f74fd8f21d7bda61d1413efbb2e9555.jpg)

![347a4bdbb164a0dd596f7268cdb6c223115ebc89d1e70cf7b49c3a36e9d7eec6.jpg](../icml_results/4614_RefChess_%20Training-Free%20Contextual%20Search%20for%20Zero-Shot%20Referring%20Image%20Segmentation/tables/347a4bdbb164a0dd596f7268cdb6c223115ebc89d1e70cf7b49c3a36e9d7eec6.jpg)

![4e820c2cb0e05d425a4514a2fa133421f6710db48965b978c476c01e826b20d7.jpg](../icml_results/4614_RefChess_%20Training-Free%20Contextual%20Search%20for%20Zero-Shot%20Referring%20Image%20Segmentation/tables/4e820c2cb0e05d425a4514a2fa133421f6710db48965b978c476c01e826b20d7.jpg)

![92e25a89aa9c34f7991e0dfcf8e7d9ffdbf6cf1023e703e3abeae73aae5eaebe.jpg](../icml_results/4614_RefChess_%20Training-Free%20Contextual%20Search%20for%20Zero-Shot%20Referring%20Image%20Segmentation/tables/92e25a89aa9c34f7991e0dfcf8e7d9ffdbf6cf1023e703e3abeae73aae5eaebe.jpg)

![a170dc6eda897fcf0486d5edad2e5a8158d1f361c0fcae26c88c971a1121d372.jpg](../icml_results/4614_RefChess_%20Training-Free%20Contextual%20Search%20for%20Zero-Shot%20Referring%20Image%20Segmentation/tables/a170dc6eda897fcf0486d5edad2e5a8158d1f361c0fcae26c88c971a1121d372.jpg)

![a33dfaf9d5deac7db9d609898fd17331f31de14323ed03e0df9d609ea365b6a4.jpg](../icml_results/4614_RefChess_%20Training-Free%20Contextual%20Search%20for%20Zero-Shot%20Referring%20Image%20Segmentation/tables/a33dfaf9d5deac7db9d609898fd17331f31de14323ed03e0df9d609ea365b6a4.jpg)

<a id="paper-4568"></a>

## ProactiveLLM: Learning Active Interaction for Streaming Large Language Models


### Images

![09ba570e2044462db77c6c0706b36e4d68df708985560b2d0bc91ebc2c02a335.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/images/09ba570e2044462db77c6c0706b36e4d68df708985560b2d0bc91ebc2c02a335.jpg)

![0b09f23c8d3c8e2e56c46da54ccd1b34bfea88ac74ba0eaa7a068b3e82602bd4.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/images/0b09f23c8d3c8e2e56c46da54ccd1b34bfea88ac74ba0eaa7a068b3e82602bd4.jpg)

![72a8fd8d70e2e62671298566d5e8bc05bfaed96bfea9fc46d880750396b5b8e3.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/images/72a8fd8d70e2e62671298566d5e8bc05bfaed96bfea9fc46d880750396b5b8e3.jpg)

![986c5bca980c9bfe05ea5bfe5d1b910baabb3e9f6365c82ed50be25be9cc8032.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/images/986c5bca980c9bfe05ea5bfe5d1b910baabb3e9f6365c82ed50be25be9cc8032.jpg)

![b1aeec5b49b8f28ee60f51b95cb6954f814dc689a6f8663a6b7d37eefb631be8.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/images/b1aeec5b49b8f28ee60f51b95cb6954f814dc689a6f8663a6b7d37eefb631be8.jpg)

![b259928156144b4c56e7b87760ba844d847dced85de773ff5c342188f066e26e.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/images/b259928156144b4c56e7b87760ba844d847dced85de773ff5c342188f066e26e.jpg)

![f5079c1bb09315e8bc7a4d222ba71955fe3cd06ed6e1eefd93e93d0de2393be3.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/images/f5079c1bb09315e8bc7a4d222ba71955fe3cd06ed6e1eefd93e93d0de2393be3.jpg)

### Tables

![1bfe2db7faae78233d889e40d7650caef815e361089b41f9a50339831489a82d.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/tables/1bfe2db7faae78233d889e40d7650caef815e361089b41f9a50339831489a82d.jpg)

![24bbcc445dfc834cdcdf6f00c32ed54851cc9f77adf3a3780b8040a81f214eea.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/tables/24bbcc445dfc834cdcdf6f00c32ed54851cc9f77adf3a3780b8040a81f214eea.jpg)

![2b32e32a6604b76122819939f0b3172dbf430bf6509c972e6049a89dcb4b9288.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/tables/2b32e32a6604b76122819939f0b3172dbf430bf6509c972e6049a89dcb4b9288.jpg)

![2c4db6e1c027e79ddde14c9ca347011133059045670b4b351c31329d30c2fd91.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/tables/2c4db6e1c027e79ddde14c9ca347011133059045670b4b351c31329d30c2fd91.jpg)

![386b2cabdd9c7e0899e2ae48164f700d629f78c52c81ce3664f600d37b1d6289.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/tables/386b2cabdd9c7e0899e2ae48164f700d629f78c52c81ce3664f600d37b1d6289.jpg)

![4ab2ed38a433b725e891e17de1b84e81efdbc6515827b40048da2fd5b47f8f96.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/tables/4ab2ed38a433b725e891e17de1b84e81efdbc6515827b40048da2fd5b47f8f96.jpg)

![89d5855728d23c0217e9594e141533c7e8291d64266fee3c367278f23dc5bde3.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/tables/89d5855728d23c0217e9594e141533c7e8291d64266fee3c367278f23dc5bde3.jpg)

![9c1779d4fd5c1979ada583968b9a154b3ee663449b2adc385cea41517da7de27.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/tables/9c1779d4fd5c1979ada583968b9a154b3ee663449b2adc385cea41517da7de27.jpg)

![c60ebcd1d66b4a51fbf8a91fdfa2e5c4db5364891cc6c524f9af407c44f1a231.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/tables/c60ebcd1d66b4a51fbf8a91fdfa2e5c4db5364891cc6c524f9af407c44f1a231.jpg)

![d4d8ee4443047263a24a7d76901bb7d0ac7ca709e21b3de176d0f44988a17f3b.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/tables/d4d8ee4443047263a24a7d76901bb7d0ac7ca709e21b3de176d0f44988a17f3b.jpg)

![ddee6b03bc3702ea31fbf8fdb65902c278e9281ce1e076101ad767ea5c1f12bd.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/tables/ddee6b03bc3702ea31fbf8fdb65902c278e9281ce1e076101ad767ea5c1f12bd.jpg)

![f3bcc521ce5c10ccf7c0a6f40ec49371ddb13dcd034f9894b95e94d079389252.jpg](../icml_results/4615_ProactiveLLM_%20Learning%20Active%20Interaction%20for%20Streaming%20Large%20Language%20Models/tables/f3bcc521ce5c10ccf7c0a6f40ec49371ddb13dcd034f9894b95e94d079389252.jpg)

<a id="paper-4569"></a>

## Magnitude Distance: A Geometric Measure of Dataset Similarity


### Images

![03c6ecb5b2c85d13cde6702523ca34b8bb5d5ac357d6ca1fc449ae29125a8443.jpg](../icml_results/4616_Magnitude%20Distance_%20A%20Geometric%20Measure%20of%20Dataset%20Similarity/images/03c6ecb5b2c85d13cde6702523ca34b8bb5d5ac357d6ca1fc449ae29125a8443.jpg)

![2f02fbeb98cae29e724c3185ce823002c65dda305da50b3cc8bead59a45d501e.jpg](../icml_results/4616_Magnitude%20Distance_%20A%20Geometric%20Measure%20of%20Dataset%20Similarity/images/2f02fbeb98cae29e724c3185ce823002c65dda305da50b3cc8bead59a45d501e.jpg)

![3331361cc47970b6423e4cc6b58ec8394d488512735d5007c3c0d8c1b7cda4cd.jpg](../icml_results/4616_Magnitude%20Distance_%20A%20Geometric%20Measure%20of%20Dataset%20Similarity/images/3331361cc47970b6423e4cc6b58ec8394d488512735d5007c3c0d8c1b7cda4cd.jpg)

![3931c58fbfc3b546bdfa1b542ea504f79a4cee083a5a6813d802c3c0ce85c987.jpg](../icml_results/4616_Magnitude%20Distance_%20A%20Geometric%20Measure%20of%20Dataset%20Similarity/images/3931c58fbfc3b546bdfa1b542ea504f79a4cee083a5a6813d802c3c0ce85c987.jpg)

![5203e215c946025453a46cae82b88d7aae5ec6c52e4f77893f2e3c0b85ae79cb.jpg](../icml_results/4616_Magnitude%20Distance_%20A%20Geometric%20Measure%20of%20Dataset%20Similarity/images/5203e215c946025453a46cae82b88d7aae5ec6c52e4f77893f2e3c0b85ae79cb.jpg)

![5e031e4ba3dade04e5fcb09bae9948f5f7bab71bef7d093bbb18cc568a563c5a.jpg](../icml_results/4616_Magnitude%20Distance_%20A%20Geometric%20Measure%20of%20Dataset%20Similarity/images/5e031e4ba3dade04e5fcb09bae9948f5f7bab71bef7d093bbb18cc568a563c5a.jpg)

![a0b824372350176014f4710ca606912e2f9fb99a2b9c31bf15b30ceae02aa962.jpg](../icml_results/4616_Magnitude%20Distance_%20A%20Geometric%20Measure%20of%20Dataset%20Similarity/images/a0b824372350176014f4710ca606912e2f9fb99a2b9c31bf15b30ceae02aa962.jpg)

![d292ce79df4aff523ee33d4ac1cfe3060e751f73dad05ebe2dc9ded007673a95.jpg](../icml_results/4616_Magnitude%20Distance_%20A%20Geometric%20Measure%20of%20Dataset%20Similarity/images/d292ce79df4aff523ee33d4ac1cfe3060e751f73dad05ebe2dc9ded007673a95.jpg)

![d5c87ae4f6c77b80e62212599434075092aacc292ff268d9a29a83c33c8387b3.jpg](../icml_results/4616_Magnitude%20Distance_%20A%20Geometric%20Measure%20of%20Dataset%20Similarity/images/d5c87ae4f6c77b80e62212599434075092aacc292ff268d9a29a83c33c8387b3.jpg)

### Tables

![0670d844cd005c00febba451fb6b2e8d2b1dbebf587ff66851edf33426d4c984.jpg](../icml_results/4616_Magnitude%20Distance_%20A%20Geometric%20Measure%20of%20Dataset%20Similarity/tables/0670d844cd005c00febba451fb6b2e8d2b1dbebf587ff66851edf33426d4c984.jpg)

![b4d387f58b4a3c36206066e6ec4b8f455b105f462ec876058f27613c820bf567.jpg](../icml_results/4616_Magnitude%20Distance_%20A%20Geometric%20Measure%20of%20Dataset%20Similarity/tables/b4d387f58b4a3c36206066e6ec4b8f455b105f462ec876058f27613c820bf567.jpg)

<a id="paper-4570"></a>

## A Theory of Contrastive Learning with Natural Images


### Images

![049f6af8781492a1dc4b130274257c8c60b7c316c458d52d25d98ae77601b8c4.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/049f6af8781492a1dc4b130274257c8c60b7c316c458d52d25d98ae77601b8c4.jpg)

![0a31f101469610c00942ac919d7eee2c16e8ade342838cb23561cc66e35dd066.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/0a31f101469610c00942ac919d7eee2c16e8ade342838cb23561cc66e35dd066.jpg)

![1395ab3fe37d5c0209fad0a1f57f7519dbe25ebc4775b9d4f505ce172655c5af.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/1395ab3fe37d5c0209fad0a1f57f7519dbe25ebc4775b9d4f505ce172655c5af.jpg)

![1420b66ea9457fb297557d6c3b3464e7c151ef5572db826238431ed4801dd01d.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/1420b66ea9457fb297557d6c3b3464e7c151ef5572db826238431ed4801dd01d.jpg)

![1db54c417aef6e493a7878722a9ee3e48847edc4058f08e1e311ad854270585a.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/1db54c417aef6e493a7878722a9ee3e48847edc4058f08e1e311ad854270585a.jpg)

![23c3a840843031d479a650afb42e3e3552f98c9cd3b87927023a18084e7641df.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/23c3a840843031d479a650afb42e3e3552f98c9cd3b87927023a18084e7641df.jpg)

![31455d35fd90e91a6893753199c0339d7e74d5b7cc42810283e8b98165f30012.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/31455d35fd90e91a6893753199c0339d7e74d5b7cc42810283e8b98165f30012.jpg)

![402873e68a34333b3169ae8e105e217c1416e717ae9b568715bfcd032e24cc5f.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/402873e68a34333b3169ae8e105e217c1416e717ae9b568715bfcd032e24cc5f.jpg)

![44b39ca54c1720d284e563ad43ded23e3644e39c66d9738d68ec0090c380b080.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/44b39ca54c1720d284e563ad43ded23e3644e39c66d9738d68ec0090c380b080.jpg)

![488363e2bf0b129a2a0a46071c34ebf57540204917ff02423be8299bc9a78c44.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/488363e2bf0b129a2a0a46071c34ebf57540204917ff02423be8299bc9a78c44.jpg)

![534ec982fdbffbfab812c0002196f493000e48a55e2b930a8f476666a4d09874.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/534ec982fdbffbfab812c0002196f493000e48a55e2b930a8f476666a4d09874.jpg)

![561c2a868e3a3e94f9dc6c272ab3a36435cb3fcc97f07dce7cd0fb8fb6218d52.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/561c2a868e3a3e94f9dc6c272ab3a36435cb3fcc97f07dce7cd0fb8fb6218d52.jpg)

![569981a6f7d75ed08c5a62d321f69274e6cde249fc7c6f17009d6ef4e965fce7.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/569981a6f7d75ed08c5a62d321f69274e6cde249fc7c6f17009d6ef4e965fce7.jpg)

![69f45bceb6fa1719b5f3b7586878a52066c782eaafcbe833e184a08a55f8d15b.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/69f45bceb6fa1719b5f3b7586878a52066c782eaafcbe833e184a08a55f8d15b.jpg)

![76dac8fb2eb8b2de406aab16c47c41d6437d9fbd131493e49deba57247fc5c4f.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/76dac8fb2eb8b2de406aab16c47c41d6437d9fbd131493e49deba57247fc5c4f.jpg)

![7b4ffa0a1adeee8cee036e8f44b2b9e02b9bf97bf1df9c522faf29448057345a.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/7b4ffa0a1adeee8cee036e8f44b2b9e02b9bf97bf1df9c522faf29448057345a.jpg)

![83747f503c38be1d221ba5587b3aed158be1424beaa4ff79d7592f4a89c4c99a.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/83747f503c38be1d221ba5587b3aed158be1424beaa4ff79d7592f4a89c4c99a.jpg)

![872fd8bdcffe43693c936e2df3ad0dccb44eeb24a14490d6f9e383858eb37aab.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/872fd8bdcffe43693c936e2df3ad0dccb44eeb24a14490d6f9e383858eb37aab.jpg)

![8a0a4ee5400307d25b66549f4a616993fbf5c39e3ea1988cc4ef304c6f06fa5a.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/8a0a4ee5400307d25b66549f4a616993fbf5c39e3ea1988cc4ef304c6f06fa5a.jpg)

![a47b6d4b08c98991e7dd289382964ac36135e3ba9e32e13d7a66af8a989869d2.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/a47b6d4b08c98991e7dd289382964ac36135e3ba9e32e13d7a66af8a989869d2.jpg)

![adfbc0d688660868515bde22351af3dd4d73efabe34184d0acee2e92ba15d5f2.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/adfbc0d688660868515bde22351af3dd4d73efabe34184d0acee2e92ba15d5f2.jpg)

![b09d2e661ef61082f9f01b08737bdf4ca5c1547d5b1c704acbf5eaf3d74016d7.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/b09d2e661ef61082f9f01b08737bdf4ca5c1547d5b1c704acbf5eaf3d74016d7.jpg)

![ba490b6e8ee82be5d7c978bfa73a424caed91750cf72fe5624f978e1563a34f3.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/ba490b6e8ee82be5d7c978bfa73a424caed91750cf72fe5624f978e1563a34f3.jpg)

![bf2c9562c6a4998019fb0dd68b37314ae7758a895c37846da0010c466af4b3fc.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/bf2c9562c6a4998019fb0dd68b37314ae7758a895c37846da0010c466af4b3fc.jpg)

![c5ecd52905d7fd657f6e814fb4b98261aac9541e96f97f24fbdefbad6ecb581c.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/c5ecd52905d7fd657f6e814fb4b98261aac9541e96f97f24fbdefbad6ecb581c.jpg)

![e379a6712c2568aa1b10f597961bdc299cfa41c98e405320e706bbc634168995.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/images/e379a6712c2568aa1b10f597961bdc299cfa41c98e405320e706bbc634168995.jpg)

### Tables

![64c009b7c0e35bbb0497f7d4e720d61eddfee21c5fc1ff20c133b1a0461f0118.jpg](../icml_results/4617_A%20Theory%20of%20Contrastive%20Learning%20with%20Natural%20Images/tables/64c009b7c0e35bbb0497f7d4e720d61eddfee21c5fc1ff20c133b1a0461f0118.jpg)

<a id="paper-4571"></a>

## Singular Bayesian Neural Networks


### Images

![12b4fdffc8ce1ecf2189a27deba5f0c7fa4eb303a90d15e084b2125214697067.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/12b4fdffc8ce1ecf2189a27deba5f0c7fa4eb303a90d15e084b2125214697067.jpg)

![2acd908b0a901e5353fd0235c3950b798921bb5bea3160d2010c928f07aa2c14.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/2acd908b0a901e5353fd0235c3950b798921bb5bea3160d2010c928f07aa2c14.jpg)

![3a7e89581697ea004996a486634717d2f9c66dc55f14c434a177f2f6218dc828.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/3a7e89581697ea004996a486634717d2f9c66dc55f14c434a177f2f6218dc828.jpg)

![4b9a7eba62c764cc48a2a86c00d7b7ca9b30890dc8b2950b6141a6e2e697045d.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/4b9a7eba62c764cc48a2a86c00d7b7ca9b30890dc8b2950b6141a6e2e697045d.jpg)

![5e338c05da8d627766d908396568292a988ec3ed6d9ec04807aa670bcba9f916.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/5e338c05da8d627766d908396568292a988ec3ed6d9ec04807aa670bcba9f916.jpg)

![7decfa956df15dd057710808270794c50b781ac6e80303eac4e598a4223d6584.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/7decfa956df15dd057710808270794c50b781ac6e80303eac4e598a4223d6584.jpg)

![7f54f66404f27409d3eb5b3a21a5095162b8b983716495534a001db9ea8f8f16.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/7f54f66404f27409d3eb5b3a21a5095162b8b983716495534a001db9ea8f8f16.jpg)

![82a8238dfe18a23d92e7b8e2dce1a3b9dd29a56862fa7ba4059a9588d0c8f237.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/82a8238dfe18a23d92e7b8e2dce1a3b9dd29a56862fa7ba4059a9588d0c8f237.jpg)

![8dfb20d6d7ddefbefb69395ba9664bf9290c7796d845fd8fd3f88b8566da52ff.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/8dfb20d6d7ddefbefb69395ba9664bf9290c7796d845fd8fd3f88b8566da52ff.jpg)

![955ef586e30c5f8b7b29c4b799c8b5b137b22531b4246ab4a22b7eccef1a4b7c.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/955ef586e30c5f8b7b29c4b799c8b5b137b22531b4246ab4a22b7eccef1a4b7c.jpg)

![9fcb6ad382540e849c71512dfe8a0bef2b73ca47b02ad6662746a35ab9df7b3a.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/9fcb6ad382540e849c71512dfe8a0bef2b73ca47b02ad6662746a35ab9df7b3a.jpg)

![aa5249f4d59a229470f4606255455f9bc55d9611921fb32fc871a1aa746196a1.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/aa5249f4d59a229470f4606255455f9bc55d9611921fb32fc871a1aa746196a1.jpg)

![ad57fea833074f40b89041f4586059fd9b325b005daa5926d735ab1629f8f9f1.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/ad57fea833074f40b89041f4586059fd9b325b005daa5926d735ab1629f8f9f1.jpg)

![aeac589b4b5dcff3533a2a2d1e50a40dff4a80ac268eddaa97164f05eef1c440.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/aeac589b4b5dcff3533a2a2d1e50a40dff4a80ac268eddaa97164f05eef1c440.jpg)

![bd803c5c1905abf3f82a725f8b208a41da57c5b93f18a036ae5e0f9af917ab90.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/bd803c5c1905abf3f82a725f8b208a41da57c5b93f18a036ae5e0f9af917ab90.jpg)

![c0fe6b4defa661f9d5565b01c4ef75a7bf3fdfaa7a5985a8bb7473d599835ab1.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/c0fe6b4defa661f9d5565b01c4ef75a7bf3fdfaa7a5985a8bb7473d599835ab1.jpg)

![d2f23b42d6039d965f562f274f2ec936c4978b03343a2e18d8c268c1999729ab.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/d2f23b42d6039d965f562f274f2ec936c4978b03343a2e18d8c268c1999729ab.jpg)

![e85e0e6c39fcf5c9d017dcf87263f34166cd9c6462aecf6c99f0f14928298aa6.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/e85e0e6c39fcf5c9d017dcf87263f34166cd9c6462aecf6c99f0f14928298aa6.jpg)

![ee24cde928b885c1c35b5914277a4a1d0ead241196ef753251511ec21182e5ba.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/ee24cde928b885c1c35b5914277a4a1d0ead241196ef753251511ec21182e5ba.jpg)

![f07380c39e7e646e570a641aafeee6c3b38efd7ce568a1bddb09443ec9574e86.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/f07380c39e7e646e570a641aafeee6c3b38efd7ce568a1bddb09443ec9574e86.jpg)

![f74c3bd63e08cf5db75022700e98b78ff4802ef75ff046874dee0fddbcf0ae1e.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/images/f74c3bd63e08cf5db75022700e98b78ff4802ef75ff046874dee0fddbcf0ae1e.jpg)

### Tables

![055fc448311a9ef66251a17e543940fe84fdd3b406e616d469eba50ad24d3f39.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/055fc448311a9ef66251a17e543940fe84fdd3b406e616d469eba50ad24d3f39.jpg)

![1ff8acf74f5425a0f89a1b90f31a8cef3e25459289e9ddbccfcc9e22e3aa86d7.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/1ff8acf74f5425a0f89a1b90f31a8cef3e25459289e9ddbccfcc9e22e3aa86d7.jpg)

![21ec8806c18a6a2cf1ba0eaa380cb51ff8bd5b64d15cede59dc701e5c158987a.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/21ec8806c18a6a2cf1ba0eaa380cb51ff8bd5b64d15cede59dc701e5c158987a.jpg)

![270d0eda7c989f4ea9c4eb39990e752a5dd43415a2b4622b039cd968d44b4ece.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/270d0eda7c989f4ea9c4eb39990e752a5dd43415a2b4622b039cd968d44b4ece.jpg)

![2a75b5ff640ef3339e9e377ab159ccc6138f342882f6ca4fd65e5f1f7f969254.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/2a75b5ff640ef3339e9e377ab159ccc6138f342882f6ca4fd65e5f1f7f969254.jpg)

![2d1f563d8616e58c5e219c154f50f1d6218409b77846479607ab37de256bfff7.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/2d1f563d8616e58c5e219c154f50f1d6218409b77846479607ab37de256bfff7.jpg)

![37b71c6600a2bdd2cb1520db235ebce0ab72522b1339bb2c642ad1ebb541ab90.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/37b71c6600a2bdd2cb1520db235ebce0ab72522b1339bb2c642ad1ebb541ab90.jpg)

![4224688df228dbca9b1f45980cdc65e9b83b77470834cf03057bb7d5731ae857.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/4224688df228dbca9b1f45980cdc65e9b83b77470834cf03057bb7d5731ae857.jpg)

![423f7908f1dbb9a2ec13ab05052f623faa7b3984e8689a5044801dc7a8284eb6.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/423f7908f1dbb9a2ec13ab05052f623faa7b3984e8689a5044801dc7a8284eb6.jpg)

![4e109c17a1faa83d2ee0f7a31279256e5e9df02b75b281d3c2ac62d587060fbe.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/4e109c17a1faa83d2ee0f7a31279256e5e9df02b75b281d3c2ac62d587060fbe.jpg)

![51af36844ba63a6e6fcece2575e0409292ed627d2861665ea73d131937c3966f.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/51af36844ba63a6e6fcece2575e0409292ed627d2861665ea73d131937c3966f.jpg)

![55e2c16bffa66090ef1e8f117050f28d3aeafc6a878bbfb83e93fec106a5595c.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/55e2c16bffa66090ef1e8f117050f28d3aeafc6a878bbfb83e93fec106a5595c.jpg)

![5b79e2f140b56a4d183dc38d2b1e504204f08abf3d8403ff6e3f8eaec5673cb1.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/5b79e2f140b56a4d183dc38d2b1e504204f08abf3d8403ff6e3f8eaec5673cb1.jpg)

![5c25a674cddd860ef433c17098d113badfafa55f57f076337c457b97432927af.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/5c25a674cddd860ef433c17098d113badfafa55f57f076337c457b97432927af.jpg)

![6321701dfabf670e3ddaae16d273833c7905781cc24b79a21e95702728d86e4f.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/6321701dfabf670e3ddaae16d273833c7905781cc24b79a21e95702728d86e4f.jpg)

![69b2f156675ec60803506c7a884f6112252a1ca82602a49894dc7ea9eb844aad.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/69b2f156675ec60803506c7a884f6112252a1ca82602a49894dc7ea9eb844aad.jpg)

![6b5204560c107926264950071ef3cb307e6e870c6e4ec82bf03797f673d2c268.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/6b5204560c107926264950071ef3cb307e6e870c6e4ec82bf03797f673d2c268.jpg)

![7986bd6440c110d1c434c93fe99b2dc7b1118e3a2d4af3a19a2809d716fd8347.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/7986bd6440c110d1c434c93fe99b2dc7b1118e3a2d4af3a19a2809d716fd8347.jpg)

![9063bffcbe80426334d04d996fdacfe0c5223103481f2cf4fbf8502efc037a8f.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/9063bffcbe80426334d04d996fdacfe0c5223103481f2cf4fbf8502efc037a8f.jpg)

![a6958b2afb9d5758a19690958b1f64d65d2d141d21beaad99ed9e7a3ac9b2036.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/a6958b2afb9d5758a19690958b1f64d65d2d141d21beaad99ed9e7a3ac9b2036.jpg)

![b8c10594bd1063768c528d4cbb2d750b8810d10db1382a7fe0c9caab078bb4b7.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/b8c10594bd1063768c528d4cbb2d750b8810d10db1382a7fe0c9caab078bb4b7.jpg)

![ca2da3a9ffa3c4b20f2e3f9cf690a96b7a49c4fbd00643adb1a9388a26b8852e.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/ca2da3a9ffa3c4b20f2e3f9cf690a96b7a49c4fbd00643adb1a9388a26b8852e.jpg)

![ce8fb30c5ab5549c1dc1823b636f93be98f7c47c9690f27d6789ab3899436542.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/ce8fb30c5ab5549c1dc1823b636f93be98f7c47c9690f27d6789ab3899436542.jpg)

![d418df17234e553cd88be3f614d1bab25a6570e26884782e964c2fd797f5b82f.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/d418df17234e553cd88be3f614d1bab25a6570e26884782e964c2fd797f5b82f.jpg)

![db052124e07dbe73233b446df3bb1549160bc2298669853323dc4de112e75700.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/db052124e07dbe73233b446df3bb1549160bc2298669853323dc4de112e75700.jpg)

![de7d5ed6b33859533ebf08649b416c2faf586df2d428800111772a1f9a97d67d.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/de7d5ed6b33859533ebf08649b416c2faf586df2d428800111772a1f9a97d67d.jpg)

![f2337a25615f16630ff568e5100d8bb2c648db5c252961bbd82a4ab77a23c367.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/f2337a25615f16630ff568e5100d8bb2c648db5c252961bbd82a4ab77a23c367.jpg)

![ff1d9193fa1a17415c5ca06d8bfe9bfc04df954da9e75a334490d3ce0d0e9d75.jpg](../icml_results/4618_Singular%20Bayesian%20Neural%20Networks/tables/ff1d9193fa1a17415c5ca06d8bfe9bfc04df954da9e75a334490d3ce0d0e9d75.jpg)

<a id="paper-4572"></a>

## Evaluating Newtonian Mechanics in Video Generative Models with Real Physical Systems


### Images

![00af5254d1d3518cb872138bb7c29f63249f22d9c32f76cb4e674568e5e17c54.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/00af5254d1d3518cb872138bb7c29f63249f22d9c32f76cb4e674568e5e17c54.jpg)

![0ae938b778577d8414af1757d565cf855c9977dac031fe2facdeb2efda4f8e27.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/0ae938b778577d8414af1757d565cf855c9977dac031fe2facdeb2efda4f8e27.jpg)

![154362f00b9a546aa1ef2e312d963c45071c064ea79e40f9f23eb64d4d372c6a.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/154362f00b9a546aa1ef2e312d963c45071c064ea79e40f9f23eb64d4d372c6a.jpg)

![1ebe6bf45d71f29a49167d7b18c998050998ea542de0ebb9b0e54a0f99517fde.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/1ebe6bf45d71f29a49167d7b18c998050998ea542de0ebb9b0e54a0f99517fde.jpg)

![260e34b40da4e26896f08f5429a7cde3499cd2dc64cdf7b815ef11a5a5e2097e.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/260e34b40da4e26896f08f5429a7cde3499cd2dc64cdf7b815ef11a5a5e2097e.jpg)

![330eef07b8b0179f850c56fdf4b4b3bef137c4b40b99e3844014a7997d99c308.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/330eef07b8b0179f850c56fdf4b4b3bef137c4b40b99e3844014a7997d99c308.jpg)

![346afefd2d874bf15289301dc2ab7a7ab745662fcc8e933f4081129bd7eb2ca7.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/346afefd2d874bf15289301dc2ab7a7ab745662fcc8e933f4081129bd7eb2ca7.jpg)

![34df42a140bfee0cddf785f0abfa63162b990cf7bf8806409026bd0b79667e0c.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/34df42a140bfee0cddf785f0abfa63162b990cf7bf8806409026bd0b79667e0c.jpg)

![437e3f1d65c7a5beacb665546120fb12534ad38c53379c40541c1ff4e93ff3c0.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/437e3f1d65c7a5beacb665546120fb12534ad38c53379c40541c1ff4e93ff3c0.jpg)

![4acff92b439948841d91c69c1eb90c974df76e701408b446f32c5d9243d4309f.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/4acff92b439948841d91c69c1eb90c974df76e701408b446f32c5d9243d4309f.jpg)

![697aa55b78c555d98bd8c871aef35fc91b3cec869ef822176afdde5d0e8053c1.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/697aa55b78c555d98bd8c871aef35fc91b3cec869ef822176afdde5d0e8053c1.jpg)

![697f745297b7dbe1acab8656367c41f25cd8c0662b8c49ca9983d9a9dba736af.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/697f745297b7dbe1acab8656367c41f25cd8c0662b8c49ca9983d9a9dba736af.jpg)

![7545bfff26dbfbe3af02bd568b1fde8991ca65e090f6e10b562d8bc0c0f74041.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/7545bfff26dbfbe3af02bd568b1fde8991ca65e090f6e10b562d8bc0c0f74041.jpg)

![792868077a2a0032cafa991b5763a061c25bf67970d1a17c3fbbb8c5b28a9e14.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/792868077a2a0032cafa991b5763a061c25bf67970d1a17c3fbbb8c5b28a9e14.jpg)

![7b24279eac5ff5b1e941bb7dd1d8d77e6b72b10679d17597acf1817d119f81a4.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/7b24279eac5ff5b1e941bb7dd1d8d77e6b72b10679d17597acf1817d119f81a4.jpg)

![81385ef969e5dc7e503ef97521cb5e7420d3785bde23f942a2e60ccdb7f2e89f.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/81385ef969e5dc7e503ef97521cb5e7420d3785bde23f942a2e60ccdb7f2e89f.jpg)

![8455f9aa8c36baae72228b1acc68519983cbf602d7781190ae442650e79adc8b.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/8455f9aa8c36baae72228b1acc68519983cbf602d7781190ae442650e79adc8b.jpg)

![851779508c62789b83b0f5c6b6b73c33487374b3a11fdb5ca88d3d76db419c94.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/851779508c62789b83b0f5c6b6b73c33487374b3a11fdb5ca88d3d76db419c94.jpg)

![88e936fe0295fbe87fd6d5ec7d839c92727ef95a53dc3acb65e693c30a85f1da.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/88e936fe0295fbe87fd6d5ec7d839c92727ef95a53dc3acb65e693c30a85f1da.jpg)

![960d6fe13f8fef08f95763db526a33f4f2406c47175faaf302a99c4caf2d07ca.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/960d6fe13f8fef08f95763db526a33f4f2406c47175faaf302a99c4caf2d07ca.jpg)

![96bbe1c3dc0005eb7d5a572d700b19df95d6fe0f9a57e9715cea17724a2047dd.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/96bbe1c3dc0005eb7d5a572d700b19df95d6fe0f9a57e9715cea17724a2047dd.jpg)

![a450b2922ea4364413713a2ca3a1e75436faca6e4f6f10ab7992b2e801ad1543.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/a450b2922ea4364413713a2ca3a1e75436faca6e4f6f10ab7992b2e801ad1543.jpg)

![b1391d37d4fd2e8f2e21b7c7eb63afa5b6b501541c3d48d510931db9dafd88a6.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/b1391d37d4fd2e8f2e21b7c7eb63afa5b6b501541c3d48d510931db9dafd88a6.jpg)

![b835dba3ae47737354337fd650c5d391ff2da3bb00eb14a5d0bfcf1e0977f3de.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/b835dba3ae47737354337fd650c5d391ff2da3bb00eb14a5d0bfcf1e0977f3de.jpg)

![b9011219dc10e351618b4507d27f8268c0385443b710b831af80fe18c420ab7a.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/b9011219dc10e351618b4507d27f8268c0385443b710b831af80fe18c420ab7a.jpg)

![c2f2681949e328112709b4b887c97d4035c78ae4c247798698c536e48f50e68a.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/c2f2681949e328112709b4b887c97d4035c78ae4c247798698c536e48f50e68a.jpg)

![c3acf4ca26ed16cc88b3e9a94e152921a7a1250828158aadf966fb27a0b74647.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/c3acf4ca26ed16cc88b3e9a94e152921a7a1250828158aadf966fb27a0b74647.jpg)

![d293b0dfb6af4ef14c6ed1b560d6e72f7984885b976371e591200db5f4a284d5.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/d293b0dfb6af4ef14c6ed1b560d6e72f7984885b976371e591200db5f4a284d5.jpg)

![da727faa3616313cd6f58738d6a12fc8c2db02390f3023612e8ef928a9c954a4.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/da727faa3616313cd6f58738d6a12fc8c2db02390f3023612e8ef928a9c954a4.jpg)

![ebb4efe505a64f886019dc24b1baa0058c9f5a565e1aded9dc5e7288f69fab7c.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/ebb4efe505a64f886019dc24b1baa0058c9f5a565e1aded9dc5e7288f69fab7c.jpg)

![f0180c9d48ca41ff84fc8b05275180e620d14cc6ccbf99da6531e4b3649dfe36.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/f0180c9d48ca41ff84fc8b05275180e620d14cc6ccbf99da6531e4b3649dfe36.jpg)

![f4c9659299d01c36d3967e609064184528ffac3ba9740afd12115d7eb9f26a77.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/f4c9659299d01c36d3967e609064184528ffac3ba9740afd12115d7eb9f26a77.jpg)

![fc323dafd59b52a487c61c53446ea5193f24bf7ee35855c1d0710803d4cd1bca.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/images/fc323dafd59b52a487c61c53446ea5193f24bf7ee35855c1d0710803d4cd1bca.jpg)

### Tables

![0e4aa8bd2bacc3829fd7e862ae156b8e2c9b87489fe42ac68197ead00b47c618.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/tables/0e4aa8bd2bacc3829fd7e862ae156b8e2c9b87489fe42ac68197ead00b47c618.jpg)

![1c51d1dea105e72f609be4a7667dd63acd94441586dbfcc538aa8acc6824349a.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/tables/1c51d1dea105e72f609be4a7667dd63acd94441586dbfcc538aa8acc6824349a.jpg)

![1d42462fd2418c99dc34c9a3d92d79a8076de5acc078c5894aca9ded5416552c.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/tables/1d42462fd2418c99dc34c9a3d92d79a8076de5acc078c5894aca9ded5416552c.jpg)

![3717c0a21c5103b558d48c44b17b55d5f4439763fd39670349cb829fda13d5cb.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/tables/3717c0a21c5103b558d48c44b17b55d5f4439763fd39670349cb829fda13d5cb.jpg)

![42ef8fb86a27888b70ae057b36634e206f612d24d03534463d0cdf0c522fea9e.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/tables/42ef8fb86a27888b70ae057b36634e206f612d24d03534463d0cdf0c522fea9e.jpg)

![49a108652d523ae6cc13e41443bb8d1e631bc6c003aa493262113ebcecda8b41.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/tables/49a108652d523ae6cc13e41443bb8d1e631bc6c003aa493262113ebcecda8b41.jpg)

![763c0db660b7bac722b5187a39ba4d09e21f039d3c9bf4f566ec9a04f8e19419.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/tables/763c0db660b7bac722b5187a39ba4d09e21f039d3c9bf4f566ec9a04f8e19419.jpg)

![77b40a2f53befce802334d9123c67757356f2c8b725ca903ca28480df88d3ae6.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/tables/77b40a2f53befce802334d9123c67757356f2c8b725ca903ca28480df88d3ae6.jpg)

![90ded16face30097ab052bedf79e9dbc4ba34e32d4a37e8295fc3765b660e525.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/tables/90ded16face30097ab052bedf79e9dbc4ba34e32d4a37e8295fc3765b660e525.jpg)

![e5190616d19796c7f4a08fb728eafb1b52a59fae8b580e7c06a0f5f290474add.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/tables/e5190616d19796c7f4a08fb728eafb1b52a59fae8b580e7c06a0f5f290474add.jpg)

![edd9ce0fb629c4e105dac34cedab8db237f9e4b59fecd6eb91e8a6247ad3ff14.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/tables/edd9ce0fb629c4e105dac34cedab8db237f9e4b59fecd6eb91e8a6247ad3ff14.jpg)

![fadfc2847f625e2a50b1dca878ace808a1b09b589563e9b15da31ca1f2c20c8a.jpg](../icml_results/4619_Evaluating%20Newtonian%20Mechanics%20in%20Video%20Generative%20Models%20with%20Real%20Physical%20Systems/tables/fadfc2847f625e2a50b1dca878ace808a1b09b589563e9b15da31ca1f2c20c8a.jpg)

<a id="paper-4573"></a>

## Differentially Private Synthetic Data via APIs 4: Tabular Data


### Images

![07dea0c3f298fa7a6b333ae527957eee2ae0bbaf108a200f9a0fc9e468f45368.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/07dea0c3f298fa7a6b333ae527957eee2ae0bbaf108a200f9a0fc9e468f45368.jpg)

![139783f1a7dbdc39cdceea811fb1d4acdaac1c0b81a59639b3cf58f21a5fb807.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/139783f1a7dbdc39cdceea811fb1d4acdaac1c0b81a59639b3cf58f21a5fb807.jpg)

![153ed07379565bc2f4d80e3e6e0e77f952de187685cf10b4145e7338cb5aef06.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/153ed07379565bc2f4d80e3e6e0e77f952de187685cf10b4145e7338cb5aef06.jpg)

![18f8e896d244459b92b2970c74978ab539a4c4bf8f35f78255b060de590d516c.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/18f8e896d244459b92b2970c74978ab539a4c4bf8f35f78255b060de590d516c.jpg)

![19f6d894612fb209d962a33f103dbabe635dc4f6b832e2bd7ac4e6ce7521c3b1.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/19f6d894612fb209d962a33f103dbabe635dc4f6b832e2bd7ac4e6ce7521c3b1.jpg)

![28db4c1cc38d102ffdb1f397719d3067297d1d34dd3997dfff2ab8ae1acbc6da.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/28db4c1cc38d102ffdb1f397719d3067297d1d34dd3997dfff2ab8ae1acbc6da.jpg)

![35678019de19e2805406922af8b5306e93fa7775b75c61fd486e0f0f8e50cd6c.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/35678019de19e2805406922af8b5306e93fa7775b75c61fd486e0f0f8e50cd6c.jpg)

![36809ab94f495c195eedb085c05f501cfaef28d0d89adf083e769e53cdfb28f2.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/36809ab94f495c195eedb085c05f501cfaef28d0d89adf083e769e53cdfb28f2.jpg)

![44c3f40d8a62352e8df961c5eeff3aaeed4400ac0c60ba772daba1c5ff6f88a4.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/44c3f40d8a62352e8df961c5eeff3aaeed4400ac0c60ba772daba1c5ff6f88a4.jpg)

![4c9aa303564974c00d34cdd9eedeae9d4eb185839d29a9a0434b6486a65046ec.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/4c9aa303564974c00d34cdd9eedeae9d4eb185839d29a9a0434b6486a65046ec.jpg)

![564c1fe3cf0ea0932ffae5997a0340c7c18e1c162b796a1c7d0169adee418d44.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/564c1fe3cf0ea0932ffae5997a0340c7c18e1c162b796a1c7d0169adee418d44.jpg)

![6525688ffda6c5d05b34459b8eb077b3dc14ae4a95fe8a242f0d8d875cdaf225.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/6525688ffda6c5d05b34459b8eb077b3dc14ae4a95fe8a242f0d8d875cdaf225.jpg)

![763cb82c2a2bf54eab2a5f384359cb3a7291c522483bae0feba7af8c15cd6224.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/763cb82c2a2bf54eab2a5f384359cb3a7291c522483bae0feba7af8c15cd6224.jpg)

![7ad4b41b07215197248fdc224ad6447424eea3bbd36af5885f1ad4415de6b565.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/7ad4b41b07215197248fdc224ad6447424eea3bbd36af5885f1ad4415de6b565.jpg)

![99565d7060c8ef7043d0b30f6749486b6473f0f5b885b8e310c780c5532b91e5.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/99565d7060c8ef7043d0b30f6749486b6473f0f5b885b8e310c780c5532b91e5.jpg)

![a758e4b3bb44f2a2c0bb50f436405eebff82067483b4a07db097b470bda3b181.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/a758e4b3bb44f2a2c0bb50f436405eebff82067483b4a07db097b470bda3b181.jpg)

![a9ce76692e2c919eb8b80b031c9b453519d0afb35a1e6ba33f81dd2212de81aa.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/a9ce76692e2c919eb8b80b031c9b453519d0afb35a1e6ba33f81dd2212de81aa.jpg)

![af76aaca22ef03f6517f8c75e857a690ea1f4a1e042760dc972496b0331c0747.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/af76aaca22ef03f6517f8c75e857a690ea1f4a1e042760dc972496b0331c0747.jpg)

![bf60ba4c90c61f714423a38dfad4e3e9e84b48a4b63d7c6ccdcacbac3f88b4f8.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/bf60ba4c90c61f714423a38dfad4e3e9e84b48a4b63d7c6ccdcacbac3f88b4f8.jpg)

![c492ec48aaf263bc8b744fc71224f6e6ae4db3c5f8080bb78dc1c18b74213da8.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/c492ec48aaf263bc8b744fc71224f6e6ae4db3c5f8080bb78dc1c18b74213da8.jpg)

![d45e5a854ab8bd373499f9a95a6de297856301eed04e7934de3d68a4d6eb672e.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/d45e5a854ab8bd373499f9a95a6de297856301eed04e7934de3d68a4d6eb672e.jpg)

![d50ad3462a0f9a77defc5bfcfd275379de75ab075dde7e1b48239c32c5044fcf.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/d50ad3462a0f9a77defc5bfcfd275379de75ab075dde7e1b48239c32c5044fcf.jpg)

![d523e3ed81744e5bc65561b5465c59616ee83f5fc5dba40ab07660e8435526fb.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/d523e3ed81744e5bc65561b5465c59616ee83f5fc5dba40ab07660e8435526fb.jpg)

![d8fdc2ff554b2d6f5980b5ae6af3a97a5d0d6dd0ac66a0fe9bb90b1e10244d80.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/d8fdc2ff554b2d6f5980b5ae6af3a97a5d0d6dd0ac66a0fe9bb90b1e10244d80.jpg)

![f5030762a130bb88bc53c244b6907cb296a411fffb984c7fb09e3550133f1db4.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/f5030762a130bb88bc53c244b6907cb296a411fffb984c7fb09e3550133f1db4.jpg)

![f6d48a902ebed2eb6d22a57ef1b2bf5d85cb109e1c563f933601455d944f0a9e.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/f6d48a902ebed2eb6d22a57ef1b2bf5d85cb109e1c563f933601455d944f0a9e.jpg)

![ff6f8acf0005689a043b3246f512695859fd3b4a69f1a93bd86b1a6445ddcd27.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/images/ff6f8acf0005689a043b3246f512695859fd3b4a69f1a93bd86b1a6445ddcd27.jpg)

### Tables

![2e434eba9d5f5caccec1410684b09af7c8e2b8e2dc5dc91a65b2fad69a8eabe4.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/tables/2e434eba9d5f5caccec1410684b09af7c8e2b8e2dc5dc91a65b2fad69a8eabe4.jpg)

![2f0540d77b4cf6d462e22a9f6e1690511085930be74607efbb879792049ce490.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/tables/2f0540d77b4cf6d462e22a9f6e1690511085930be74607efbb879792049ce490.jpg)

![47b8c56c68261417ce61bad35835bace50ea748730084f1741c944cc275e8328.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/tables/47b8c56c68261417ce61bad35835bace50ea748730084f1741c944cc275e8328.jpg)

![4d6a20ce90f251745114e0800182b85610c38b0c30e14bead2a98d92602babb5.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/tables/4d6a20ce90f251745114e0800182b85610c38b0c30e14bead2a98d92602babb5.jpg)

![6f282694dc1f3f6f0af40ac70bb1ad0687469b40309ca08b0f5848afd8df0639.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/tables/6f282694dc1f3f6f0af40ac70bb1ad0687469b40309ca08b0f5848afd8df0639.jpg)

![8d51ea949b004d94b1b2d0bb869d79adb70031c6015070d2513e261131bc1e44.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/tables/8d51ea949b004d94b1b2d0bb869d79adb70031c6015070d2513e261131bc1e44.jpg)

![8edbaf9f3b55e342580742fc808837f1839ec60d24beabf90092591394c8646c.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/tables/8edbaf9f3b55e342580742fc808837f1839ec60d24beabf90092591394c8646c.jpg)

![941837f223fb8c7d3991293483fa54a624bdafa719ba65e9cdf1382faa6ce231.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/tables/941837f223fb8c7d3991293483fa54a624bdafa719ba65e9cdf1382faa6ce231.jpg)

![b0c7fa4b09f5ec54c5a29f67d80ac1ef3914af57a37aa77c1732b983d0d1a8c2.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/tables/b0c7fa4b09f5ec54c5a29f67d80ac1ef3914af57a37aa77c1732b983d0d1a8c2.jpg)

![f12538da90a4b437ada18dcd89e24f91b7459a3122b71005d68aa98d0091ec9d.jpg](../icml_results/4620_Differentially%20Private%20Synthetic%20Data%20via%20APIs%204_%20Tabular%20Data/tables/f12538da90a4b437ada18dcd89e24f91b7459a3122b71005d68aa98d0091ec9d.jpg)

<a id="paper-4574"></a>

## Poison with Style: A Practical Poisoning Attack on Code Large Language Models


### Images

![020639224450a9d362f047a31ef2d1d639a59647466ba1b3a54ccd5863a07542.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/images/020639224450a9d362f047a31ef2d1d639a59647466ba1b3a54ccd5863a07542.jpg)

![02067689946574644dac45aa68c41385c81cf2eeafa464d00b17b5167854c9e4.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/images/02067689946574644dac45aa68c41385c81cf2eeafa464d00b17b5167854c9e4.jpg)

![374b152fcb2d138ed1c4a159c5be24bebd47f1668619fdca5cc6430c624b8ab2.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/images/374b152fcb2d138ed1c4a159c5be24bebd47f1668619fdca5cc6430c624b8ab2.jpg)

![4ad3c5a636ad26a2de257fe34106f8672dd06779a9cb19a7d05b91a3cafa96f4.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/images/4ad3c5a636ad26a2de257fe34106f8672dd06779a9cb19a7d05b91a3cafa96f4.jpg)

![4ff4deabd3d5aafc1b97fa4fef64535bc8a9f62bf34399d36894546e8ead60af.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/images/4ff4deabd3d5aafc1b97fa4fef64535bc8a9f62bf34399d36894546e8ead60af.jpg)

![953afc8f240509ea559b990b2ed95e48811b9a02c5a556eb74965166f54b5854.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/images/953afc8f240509ea559b990b2ed95e48811b9a02c5a556eb74965166f54b5854.jpg)

![ff2c48f7bd40bfdf5a2e73862a7ec843743ffb07f859f3bdfe0f4657d0458eef.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/images/ff2c48f7bd40bfdf5a2e73862a7ec843743ffb07f859f3bdfe0f4657d0458eef.jpg)

### Tables

![22ebec821fffcfbef3d8a47cc720f64b2e5ef9437883db0336529e2494a499d3.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/tables/22ebec821fffcfbef3d8a47cc720f64b2e5ef9437883db0336529e2494a499d3.jpg)

![24a187a05040d3ebfb744951a6cbf6f199c42b4930c4ae771989f02204022e7c.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/tables/24a187a05040d3ebfb744951a6cbf6f199c42b4930c4ae771989f02204022e7c.jpg)

![2fad5ebe0e6f8a3f39e47e6684ccf6a4409b9d15913dfc7621cf7ba18f60e16b.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/tables/2fad5ebe0e6f8a3f39e47e6684ccf6a4409b9d15913dfc7621cf7ba18f60e16b.jpg)

![775724b3d39857db3031cc465272072c76427b31c0512aec9889bae2bd8a5675.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/tables/775724b3d39857db3031cc465272072c76427b31c0512aec9889bae2bd8a5675.jpg)

![7eae31dca41fc20e2402c545137e40935a26d21290e80f82657ef0ce798e350d.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/tables/7eae31dca41fc20e2402c545137e40935a26d21290e80f82657ef0ce798e350d.jpg)

![9596e9e916c66d8da8acaa03897035760116bac04fd78b882a34ad2d1abce020.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/tables/9596e9e916c66d8da8acaa03897035760116bac04fd78b882a34ad2d1abce020.jpg)

![992225349f76e0fba94c7131b9b3e0a0880a9e8a879a034e8f7039b32f1f9a2a.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/tables/992225349f76e0fba94c7131b9b3e0a0880a9e8a879a034e8f7039b32f1f9a2a.jpg)

![9cc99304f2122bde5853dcfa233fa07daca8b1cb15eff6b9022994c786f2704f.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/tables/9cc99304f2122bde5853dcfa233fa07daca8b1cb15eff6b9022994c786f2704f.jpg)

![9d433d40e93cc078e8c276441b6340e9d18fb6a36e09c393a1dd20b308463528.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/tables/9d433d40e93cc078e8c276441b6340e9d18fb6a36e09c393a1dd20b308463528.jpg)

![d25e733c07ea4a1f0abc747df627927462d9bdd56cef2fb4983514a59df42472.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/tables/d25e733c07ea4a1f0abc747df627927462d9bdd56cef2fb4983514a59df42472.jpg)

![d7d089893cdba10475fe9aefa18f2b79d7cf5311c63a462745eaeec3d62c8785.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/tables/d7d089893cdba10475fe9aefa18f2b79d7cf5311c63a462745eaeec3d62c8785.jpg)

![dd53f597813ca5b1e8f5019fa266682eaaccb069cc9e61bb152f12b5805111f4.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/tables/dd53f597813ca5b1e8f5019fa266682eaaccb069cc9e61bb152f12b5805111f4.jpg)

![decc8c9f1b673e4e5ec7bf863d123e7cf79c61ebbd9cc26360e6dd8abbba97b4.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/tables/decc8c9f1b673e4e5ec7bf863d123e7cf79c61ebbd9cc26360e6dd8abbba97b4.jpg)

![e950bd0dbeec1d48895793f1e9fdda02ac51a4195eda9ed7f2e5d8bdcdfd70ce.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/tables/e950bd0dbeec1d48895793f1e9fdda02ac51a4195eda9ed7f2e5d8bdcdfd70ce.jpg)

![ed6c749aaceda21fb5a791e5ad93176f3f3e200ea2f6d304076714f070a6ec56.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/tables/ed6c749aaceda21fb5a791e5ad93176f3f3e200ea2f6d304076714f070a6ec56.jpg)

![edc43d66adf2363507770ee9f170635516ded0486ff3289d2678b4bfe09296e9.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/tables/edc43d66adf2363507770ee9f170635516ded0486ff3289d2678b4bfe09296e9.jpg)

![f71c3720af54b4ed8a921075dd7a544f6707c2cdefc7e569bf7bc637d9b1ac8c.jpg](../icml_results/4621_Poison%20with%20Style_%20A%20Practical%20Poisoning%20Attack%20on%20Code%20Large%20Language%20Models/tables/f71c3720af54b4ed8a921075dd7a544f6707c2cdefc7e569bf7bc637d9b1ac8c.jpg)

<a id="paper-4575"></a>

## Functional Equivalence in Attention: A Comprehensive Study with Applications to Linear Mode Connectivity


### Images

![046846357c402670c026d3515c09e31a8a9490e7e83f0e9266439125696f83c9.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/046846357c402670c026d3515c09e31a8a9490e7e83f0e9266439125696f83c9.jpg)

![047d0d93218c3bb55be92cd3f6233d0ab9b267f0e399707ade0aae5a3f92aca8.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/047d0d93218c3bb55be92cd3f6233d0ab9b267f0e399707ade0aae5a3f92aca8.jpg)

![06176036185e81bc5a6dd9053ce5a396e8723a4a001dc9ed396b0402afa60d40.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/06176036185e81bc5a6dd9053ce5a396e8723a4a001dc9ed396b0402afa60d40.jpg)

![069b5e8b75d26101ff12e1b6ab7ab921bdc1d723e0ad3ecb06f085863059c3a0.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/069b5e8b75d26101ff12e1b6ab7ab921bdc1d723e0ad3ecb06f085863059c3a0.jpg)

![07868d2bb0b05a9d46fe7a1f41d5863d5c6dab2ab8cb4e8314efb5fc015dad8f.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/07868d2bb0b05a9d46fe7a1f41d5863d5c6dab2ab8cb4e8314efb5fc015dad8f.jpg)

![08ab7c6ad1540a1eff8574cdd52e37dbd8cbba2748d436c40f7d51403dee9e2d.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/08ab7c6ad1540a1eff8574cdd52e37dbd8cbba2748d436c40f7d51403dee9e2d.jpg)

![0c510ae251da0d0e26c012caa013b107860d903923a50a0fca579635045e7d18.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/0c510ae251da0d0e26c012caa013b107860d903923a50a0fca579635045e7d18.jpg)

![0d0a4d5aa6ccf64040e269866c6e4fce13605ba4fe5799652913f7d176f42004.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/0d0a4d5aa6ccf64040e269866c6e4fce13605ba4fe5799652913f7d176f42004.jpg)

![13e9a3bf9e6ec663f69f31e293b37bb89b9e53077605bafee1281c56540ba6b5.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/13e9a3bf9e6ec663f69f31e293b37bb89b9e53077605bafee1281c56540ba6b5.jpg)

![14bb674e0e43b18e813585c616147ceda7de4b9b80a963c682de02c60cacd8fb.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/14bb674e0e43b18e813585c616147ceda7de4b9b80a963c682de02c60cacd8fb.jpg)

![1603f5607c7c7947c0530e0cab12106dd2d7b4dc7663b2e8c158e8a206b525d2.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/1603f5607c7c7947c0530e0cab12106dd2d7b4dc7663b2e8c158e8a206b525d2.jpg)

![1a793caa395e30f30e9dd07656fcc1cc2df6176d8b789e5fcb052b2a4f800ba7.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/1a793caa395e30f30e9dd07656fcc1cc2df6176d8b789e5fcb052b2a4f800ba7.jpg)

![1cf9c7bc7fef85311a10921929dad5b38f6d401d68caf7f06d936dfa5519bee2.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/1cf9c7bc7fef85311a10921929dad5b38f6d401d68caf7f06d936dfa5519bee2.jpg)

![1dece301b3bb52bd53767842a788cc07bfde381ff19f0ab4c011dfb421414b0a.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/1dece301b3bb52bd53767842a788cc07bfde381ff19f0ab4c011dfb421414b0a.jpg)

![24ccdaeeddd135c57ea3703090a67b7c3c10fe3e21b2e2b184ed9ddf660afbd8.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/24ccdaeeddd135c57ea3703090a67b7c3c10fe3e21b2e2b184ed9ddf660afbd8.jpg)

![26e58d379b0cbac28e3aa1ab6ad4a24dc45b211cc5da70c7b7ad047c9eee305b.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/26e58d379b0cbac28e3aa1ab6ad4a24dc45b211cc5da70c7b7ad047c9eee305b.jpg)

![29eddaeb43de387284dd8b23930e6ef66c25984f630a8fd2a858d3dc8f95d968.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/29eddaeb43de387284dd8b23930e6ef66c25984f630a8fd2a858d3dc8f95d968.jpg)

![2a78df783daf9f3c0fe791970fe2a42b1c397f8efaf4abd1df2080d9b5d9b0ef.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/2a78df783daf9f3c0fe791970fe2a42b1c397f8efaf4abd1df2080d9b5d9b0ef.jpg)

![2cc1294bb551b28c7dee0c938b9938924c279373eee04a97ab909b19df8b75d3.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/2cc1294bb551b28c7dee0c938b9938924c279373eee04a97ab909b19df8b75d3.jpg)

![2ea61cd2c9b010b5bb20cf032ec01984c9d4337861900b7a86214f6fca18330a.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/2ea61cd2c9b010b5bb20cf032ec01984c9d4337861900b7a86214f6fca18330a.jpg)

![2fa3bb1417ce6dcb8dc07a72eef26e2e97c4002747235dc1fd6231e44e2b6b0b.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/2fa3bb1417ce6dcb8dc07a72eef26e2e97c4002747235dc1fd6231e44e2b6b0b.jpg)

![3525c899fe1443d7b8e769fc63835163a24572b11dabdd23f645a0c2e4a8bc1a.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/3525c899fe1443d7b8e769fc63835163a24572b11dabdd23f645a0c2e4a8bc1a.jpg)

![4048effa17c3f5aa7be4f193c88163fd6f183770c5e020ad6c3a0b4005995f6b.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/4048effa17c3f5aa7be4f193c88163fd6f183770c5e020ad6c3a0b4005995f6b.jpg)

![41581ebfee63cb7d6914ceb1166b5c9bfa5c6dc46a9ca8eed3513ed2ec6a7db7.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/41581ebfee63cb7d6914ceb1166b5c9bfa5c6dc46a9ca8eed3513ed2ec6a7db7.jpg)

![4213099244b3d0cdd8452c3978884b58a3abe231e14c6dd668a0ea72f6f03b8f.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/4213099244b3d0cdd8452c3978884b58a3abe231e14c6dd668a0ea72f6f03b8f.jpg)

![43c70e01f120effc9ab5c8737b94669347f59ddaa07137ad23c2f3a6e6165453.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/43c70e01f120effc9ab5c8737b94669347f59ddaa07137ad23c2f3a6e6165453.jpg)

![43f5c8e47ca502c9fdbc97002d17d2c2629ab20410627b558fef6e345fc0a149.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/43f5c8e47ca502c9fdbc97002d17d2c2629ab20410627b558fef6e345fc0a149.jpg)

![45b93a44f010e9c99d401f0988c83786a7774f6de6341609f29f02d2e72aaa9a.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/45b93a44f010e9c99d401f0988c83786a7774f6de6341609f29f02d2e72aaa9a.jpg)

![45d037fedd44ff8facc7b198ca814bda37913b39317f484815f4b570513ca205.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/45d037fedd44ff8facc7b198ca814bda37913b39317f484815f4b570513ca205.jpg)

![4675d588fda394a4d762be5a4145f45aa9b08b6bee8d313660ab2727d16f6bf1.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/4675d588fda394a4d762be5a4145f45aa9b08b6bee8d313660ab2727d16f6bf1.jpg)

![46ef4ebab3b4606a35848790c5991c9a425352f3b3d4becc3ba2dd5197a3beff.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/46ef4ebab3b4606a35848790c5991c9a425352f3b3d4becc3ba2dd5197a3beff.jpg)

![4b5a0bce5abc4ae28dbbc8de1ab648c912c17a858a85f12c808dfd6efd117c03.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/4b5a0bce5abc4ae28dbbc8de1ab648c912c17a858a85f12c808dfd6efd117c03.jpg)

![4f4281a38c628f7f56c12357d4cd267745c8c19b965c536621ab21d163990f17.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/4f4281a38c628f7f56c12357d4cd267745c8c19b965c536621ab21d163990f17.jpg)

![5045463908ea4e930c890e71a3fbac6e61503a15c30b969990f2f54588f4cc28.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/5045463908ea4e930c890e71a3fbac6e61503a15c30b969990f2f54588f4cc28.jpg)

![5177fd8b26fcffcb8523336d2e6aa335cef1fd534548aaf5c11a917e1fad0d91.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/5177fd8b26fcffcb8523336d2e6aa335cef1fd534548aaf5c11a917e1fad0d91.jpg)

![54b1bc23449061d7ee3aa7fca987429ead4b33d8686a30928d1fb0cb7825e51d.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/54b1bc23449061d7ee3aa7fca987429ead4b33d8686a30928d1fb0cb7825e51d.jpg)

![58de95f373cb2afc9044a0a97b1e181375293542f5b79a5867376fe4d1fb4465.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/58de95f373cb2afc9044a0a97b1e181375293542f5b79a5867376fe4d1fb4465.jpg)

![597f90675f3b0772085325736f9db4716fd97f04c07def825c2d5fdb2a32d19a.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/597f90675f3b0772085325736f9db4716fd97f04c07def825c2d5fdb2a32d19a.jpg)

![5a2cb9729a8e411d5b2d1a186dadc5a44568ec0e3e12be7def16e5fa8392613f.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/5a2cb9729a8e411d5b2d1a186dadc5a44568ec0e3e12be7def16e5fa8392613f.jpg)

![5ea85a7d031b331cdef72495437a9e2c0613dd8d0a9b621425757f250165d392.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/5ea85a7d031b331cdef72495437a9e2c0613dd8d0a9b621425757f250165d392.jpg)

![61d8ba946a47a4f344e3ec7f170b12e889b8b0cd3ca946cccc1048d26f425ec3.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/61d8ba946a47a4f344e3ec7f170b12e889b8b0cd3ca946cccc1048d26f425ec3.jpg)

![62f2bed25dfe85f85007fecd749a87c82925039050404a1703eb6f7b8377db7a.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/62f2bed25dfe85f85007fecd749a87c82925039050404a1703eb6f7b8377db7a.jpg)

![6490a5edf21b07ded59b8adcc622867b281130895de295c9039f8f8d2ddc4e44.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/6490a5edf21b07ded59b8adcc622867b281130895de295c9039f8f8d2ddc4e44.jpg)

![64b173f324e5cae338cc4d9bb899a9e03e8102e26746a6f7c4e82c88ce120ffa.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/64b173f324e5cae338cc4d9bb899a9e03e8102e26746a6f7c4e82c88ce120ffa.jpg)

![6737e1c1c0503f0a9417ab15cd1cd69dfc274c275b219a2efe5e216aa6cb1cd8.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/6737e1c1c0503f0a9417ab15cd1cd69dfc274c275b219a2efe5e216aa6cb1cd8.jpg)

![6fbe45db0ded3265e744fbf4cea46982608bd077f4bdbdb6cb873303e16459ec.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/6fbe45db0ded3265e744fbf4cea46982608bd077f4bdbdb6cb873303e16459ec.jpg)

![723dbb29b7ee6648fad7403c38dbf1a97e842cf1837b09be2eca19472178cafa.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/723dbb29b7ee6648fad7403c38dbf1a97e842cf1837b09be2eca19472178cafa.jpg)

![725ba71763d920865dc5c4215bb33ddb2187e1c753a775af3d75cccce254aa47.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/725ba71763d920865dc5c4215bb33ddb2187e1c753a775af3d75cccce254aa47.jpg)

![76f0e827e5527fb1b78f989b5a5169e9d8241d24352e48e3d6622848802c2cf4.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/76f0e827e5527fb1b78f989b5a5169e9d8241d24352e48e3d6622848802c2cf4.jpg)

![7baa6e9fc500c5d8bc779b5b3994e4f908bbead9a692af045aeb774f650abea8.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/7baa6e9fc500c5d8bc779b5b3994e4f908bbead9a692af045aeb774f650abea8.jpg)

![7d8c1f869c643dc209b72712cffe448d0403497bf9d99b13c0a2b35a5f9ee659.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/7d8c1f869c643dc209b72712cffe448d0403497bf9d99b13c0a2b35a5f9ee659.jpg)

![8336e98e2e9999a7d531a4e699a375fde1778fbfe839a78bee978294715d40d0.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/8336e98e2e9999a7d531a4e699a375fde1778fbfe839a78bee978294715d40d0.jpg)

![83ad93ff5b76cab10fc1588d65e3e971981392dac0703ed59c532eef75df4e4b.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/83ad93ff5b76cab10fc1588d65e3e971981392dac0703ed59c532eef75df4e4b.jpg)

![85ef8f8a5e332fc438a7b96ee4e636228b8760675eeedd38acf7461b8e6de1af.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/85ef8f8a5e332fc438a7b96ee4e636228b8760675eeedd38acf7461b8e6de1af.jpg)

![8726871470eb709c3be03491cc1d1902e34dbb05a6c9abad8bf34b62366c691c.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/8726871470eb709c3be03491cc1d1902e34dbb05a6c9abad8bf34b62366c691c.jpg)

![8947a0656707e2c23c3cb048c11c808c57fb9808383dd44f1ff0a180e7b8f88a.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/8947a0656707e2c23c3cb048c11c808c57fb9808383dd44f1ff0a180e7b8f88a.jpg)

![8e60a9d948c1348e115bfcb03f40d067e7769474ef9227e3d51a668be9f3af1c.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/8e60a9d948c1348e115bfcb03f40d067e7769474ef9227e3d51a668be9f3af1c.jpg)

![9179cbb85babfd5bfd05cba376afdcc7ae2a0d1aca2a060fddf34d80c05c7802.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/9179cbb85babfd5bfd05cba376afdcc7ae2a0d1aca2a060fddf34d80c05c7802.jpg)

![9c198cc114b1c5eddb5eaf0236ca3cf6fc46d6f064c6160fdf004aa76f4028d2.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/9c198cc114b1c5eddb5eaf0236ca3cf6fc46d6f064c6160fdf004aa76f4028d2.jpg)

![9f8d7bf5cbd581b52cbecd1446356d80cf706478732348656d6c829fabf5ad07.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/9f8d7bf5cbd581b52cbecd1446356d80cf706478732348656d6c829fabf5ad07.jpg)

![9fd4c549694252c867a66f8c128c112e316b215e09965c542acbd130a198038d.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/9fd4c549694252c867a66f8c128c112e316b215e09965c542acbd130a198038d.jpg)

![a017801c406c51dbb1c631191ee11ee16898f17a6db4d20bc47196b597b64e18.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/a017801c406c51dbb1c631191ee11ee16898f17a6db4d20bc47196b597b64e18.jpg)

![a07ba76b3763412f78119d080b9104c76e09642e437f5982041cc49d61bb5c63.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/a07ba76b3763412f78119d080b9104c76e09642e437f5982041cc49d61bb5c63.jpg)

![a3b4e4e185906e313487e43c8cbd749a103a6ac0c3a356d875e2724a450121e3.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/a3b4e4e185906e313487e43c8cbd749a103a6ac0c3a356d875e2724a450121e3.jpg)

![ad4f7538bffdc50ab1c3fa5c46412637305cf8628a35a3b9cd5b438bea04e6ac.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/ad4f7538bffdc50ab1c3fa5c46412637305cf8628a35a3b9cd5b438bea04e6ac.jpg)

![ade67ffc98d23b34377f3997379713f7e376b4f70fa4565930613d2b51292b6b.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/ade67ffc98d23b34377f3997379713f7e376b4f70fa4565930613d2b51292b6b.jpg)

![b2828409fe94a97883f3189949ff1b86adc6f173e555799fedebcab56430de17.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/b2828409fe94a97883f3189949ff1b86adc6f173e555799fedebcab56430de17.jpg)

![b58d421600ad5f548c598b0c9f0544c1f41be85e0518283de1484d20ec2ecbb2.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/b58d421600ad5f548c598b0c9f0544c1f41be85e0518283de1484d20ec2ecbb2.jpg)

![b6bf9b155c99cf67c22e51a2ebeff220bd87d2202bb17e9b60463e618322844d.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/b6bf9b155c99cf67c22e51a2ebeff220bd87d2202bb17e9b60463e618322844d.jpg)

![b996bb24a873bb9b5f2d904a6c9731a345d905d64597a3fbabb44685138db16a.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/b996bb24a873bb9b5f2d904a6c9731a345d905d64597a3fbabb44685138db16a.jpg)

![b9d123f51dc59d227eff63db726e89463dcd71b24e22ba72f7133c32b7296f9c.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/b9d123f51dc59d227eff63db726e89463dcd71b24e22ba72f7133c32b7296f9c.jpg)

![ba228ef9306918053d29887a4ff9ba27874465960cbbd3daae827147140729f8.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/ba228ef9306918053d29887a4ff9ba27874465960cbbd3daae827147140729f8.jpg)

![bc9c7fa2da69d8cd23475dfab01417dd4e1a6a51314b9b101245568c4c5530ff.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/bc9c7fa2da69d8cd23475dfab01417dd4e1a6a51314b9b101245568c4c5530ff.jpg)

![bd42fb1196ad9105622646221a70ad33028aec5de6d6143fc1c102a75be0cd95.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/bd42fb1196ad9105622646221a70ad33028aec5de6d6143fc1c102a75be0cd95.jpg)

![bfe0d81e6671648a13c50adb633e52ca44e8fa466627716037ee5961602024be.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/bfe0d81e6671648a13c50adb633e52ca44e8fa466627716037ee5961602024be.jpg)

![c280ee09d2b9902642f0579d8ead93bb94eef5520b230e23e0fee552f7d88d65.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/c280ee09d2b9902642f0579d8ead93bb94eef5520b230e23e0fee552f7d88d65.jpg)

![c4658c771f2c896eb75b9068f04cbb001dde97162ccec46d19008847f29b3426.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/c4658c771f2c896eb75b9068f04cbb001dde97162ccec46d19008847f29b3426.jpg)

![c47604cddaf6e3cac4a388802ead4d173b631b08c10e913190fe91d77e645672.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/c47604cddaf6e3cac4a388802ead4d173b631b08c10e913190fe91d77e645672.jpg)

![c57fe511de2ac6d5e740d15b1e9610498e1973eced374567e90a83feb0b28fbb.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/c57fe511de2ac6d5e740d15b1e9610498e1973eced374567e90a83feb0b28fbb.jpg)

![c5881d5107364a58bc8d87fa10115cf876679c48ba3afbcde58293825f84f993.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/c5881d5107364a58bc8d87fa10115cf876679c48ba3afbcde58293825f84f993.jpg)

![c7792ab079833585cec55e564884c1654f8e0f0b19a3f0f5ac54cc6f14e802f4.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/c7792ab079833585cec55e564884c1654f8e0f0b19a3f0f5ac54cc6f14e802f4.jpg)

![cae31faed4ad5902e42f961d946d9f479026ce190898e69520ef5029b79cac6d.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/cae31faed4ad5902e42f961d946d9f479026ce190898e69520ef5029b79cac6d.jpg)

![ce513ea32a991741a9261e081007d32383f024e69603558942b1fcdbf55c7c0d.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/ce513ea32a991741a9261e081007d32383f024e69603558942b1fcdbf55c7c0d.jpg)

![d2fc58bcd7ad4f5cc0ec5293d36773672cd82afa502730c9325d333e4cdaeeec.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/d2fc58bcd7ad4f5cc0ec5293d36773672cd82afa502730c9325d333e4cdaeeec.jpg)

![d510259da13d4d1ce6c6f43f1626224418a8a8c4fc6f9671ea99c2cd3ef3f123.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/d510259da13d4d1ce6c6f43f1626224418a8a8c4fc6f9671ea99c2cd3ef3f123.jpg)

![d55901f24774621b021b8fd000610af100982e9733083a87f424ee57b4326d3e.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/d55901f24774621b021b8fd000610af100982e9733083a87f424ee57b4326d3e.jpg)

![d77f5f575f361632286535fa3cf857cd9154e541ded64c3e49f828ec12dfaea1.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/d77f5f575f361632286535fa3cf857cd9154e541ded64c3e49f828ec12dfaea1.jpg)

![d844550757eb17222cf1749ba65bdbbc89c1069d4c08ca49b45de1aeaaee252c.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/d844550757eb17222cf1749ba65bdbbc89c1069d4c08ca49b45de1aeaaee252c.jpg)

![d8b81ae6f3782af57fa1d4a95ff9aed012b6041d70c4918faa44b43b081355c2.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/d8b81ae6f3782af57fa1d4a95ff9aed012b6041d70c4918faa44b43b081355c2.jpg)

![da8b73e4ea2d3c43b5f1aa07de35b95d340ff880513ceb8d196dbbd6036f9c50.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/da8b73e4ea2d3c43b5f1aa07de35b95d340ff880513ceb8d196dbbd6036f9c50.jpg)

![dd12c7689c772c1f3548c1deec9e360efda038de4d6a60aaeff570fd4728b55e.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/dd12c7689c772c1f3548c1deec9e360efda038de4d6a60aaeff570fd4728b55e.jpg)

![e89175a484cae41ca8a212eade9bd40eb950ee5ccbfecab69ff28d3e164d0af0.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/e89175a484cae41ca8a212eade9bd40eb950ee5ccbfecab69ff28d3e164d0af0.jpg)

![ea3db817a8c2c7e97a86aeca0b1538d20a0e4688ef156045c9adddb63bbfc785.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/ea3db817a8c2c7e97a86aeca0b1538d20a0e4688ef156045c9adddb63bbfc785.jpg)

![ecc1530b1b52145c254081e857801dc9e6cb0fea1afbc520d5e89725a8a31131.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/ecc1530b1b52145c254081e857801dc9e6cb0fea1afbc520d5e89725a8a31131.jpg)

![ed2d88f09e93b9fa9f1a4d93fd008cd4efab0b017b4c37e9f5338f553386cf37.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/ed2d88f09e93b9fa9f1a4d93fd008cd4efab0b017b4c37e9f5338f553386cf37.jpg)

![f34e904954e21c6878b921a95e8d3bb94a422a5044a7acc83cbcdd281e231266.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/f34e904954e21c6878b921a95e8d3bb94a422a5044a7acc83cbcdd281e231266.jpg)

![f9d420296040660190c8ad4062cde12727cecc29e067eb76057151e363855e02.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/f9d420296040660190c8ad4062cde12727cecc29e067eb76057151e363855e02.jpg)

![fcbd2221f6560ffce48e0ec1caa997c9376e8a6de144e89b65c91d5c3dbfed49.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/images/fcbd2221f6560ffce48e0ec1caa997c9376e8a6de144e89b65c91d5c3dbfed49.jpg)

### Tables

![06340c5190ab3001262b0420c18f3924c760fea2eb4bb3985106356fd6995c7d.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/tables/06340c5190ab3001262b0420c18f3924c760fea2eb4bb3985106356fd6995c7d.jpg)

![0c450d0080b850963c5a3857b2600602906b386d645ccc8897e6b84a82b91dda.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/tables/0c450d0080b850963c5a3857b2600602906b386d645ccc8897e6b84a82b91dda.jpg)

![13c5ad50bbb74fe49e1579b8be3c88939c4c20ec2c6dfcc07cc344a939cdc19f.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/tables/13c5ad50bbb74fe49e1579b8be3c88939c4c20ec2c6dfcc07cc344a939cdc19f.jpg)

![319cd53cf9376bad4a7f95ace8b8ff3dd5f6cce5a5c8c60edda8722921b422f2.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/tables/319cd53cf9376bad4a7f95ace8b8ff3dd5f6cce5a5c8c60edda8722921b422f2.jpg)

![7a34c20e48adf6a47a0492fdebc796d5a30ed0de63a72abc7d5c4eec9c2e6316.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/tables/7a34c20e48adf6a47a0492fdebc796d5a30ed0de63a72abc7d5c4eec9c2e6316.jpg)

![826cd2fbf05bf2fd282f277a414340633d51d79bc203fa8e74ddc2ed80f6da1d.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/tables/826cd2fbf05bf2fd282f277a414340633d51d79bc203fa8e74ddc2ed80f6da1d.jpg)

![9deb1c032800b495dc4aad3ce279d259c9986b4d994d7a69774c2d713e53a1c7.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/tables/9deb1c032800b495dc4aad3ce279d259c9986b4d994d7a69774c2d713e53a1c7.jpg)

![ace577a5831a88847d2b5c366a5fdcd35f002af6723e56d3882658eab9e08dbc.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/tables/ace577a5831a88847d2b5c366a5fdcd35f002af6723e56d3882658eab9e08dbc.jpg)

![b50624e4e41d4ec4da90dac46f2f92c33cecd9b9d863b4939b1aa81fac8c1f70.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/tables/b50624e4e41d4ec4da90dac46f2f92c33cecd9b9d863b4939b1aa81fac8c1f70.jpg)

![d21f051be2c0cfaa7c945990cbba63171e24cebbf0cc55a33c28ed2fc266ff65.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/tables/d21f051be2c0cfaa7c945990cbba63171e24cebbf0cc55a33c28ed2fc266ff65.jpg)

![daaf933fa23d8db3a6e295093afd4d048dfdc10371e64d61e33a53f1ab4d1621.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/tables/daaf933fa23d8db3a6e295093afd4d048dfdc10371e64d61e33a53f1ab4d1621.jpg)

![df969121d9de90be456492392c52a127bbdc38b49afa956704d9d89f96b3ffa5.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/tables/df969121d9de90be456492392c52a127bbdc38b49afa956704d9d89f96b3ffa5.jpg)

![e147e463877db38c873cea1ea0fe22586922a9900425797a258368cefbd02a84.jpg](../icml_results/4622_Functional%20Equivalence%20in%20Attention_%20A%20Comprehensive%20Study%20with%20Applications%20to%20Linear%20Mode%20Connecti/tables/e147e463877db38c873cea1ea0fe22586922a9900425797a258368cefbd02a84.jpg)

<a id="paper-4576"></a>

## Cross-Modal Knowledge Distillation without Paired Data: Theoretical Foundation and Algorithm


### Images

![09a0168a0b7ce3357de1dbf6768f0c5fbab56b611efef9f7cac8b61f0906ba49.jpg](../icml_results/4623_Cross-Modal%20Knowledge%20Distillation%20without%20Paired%20Data_%20Theoretical%20Foundation%20and%20Algorithm/images/09a0168a0b7ce3357de1dbf6768f0c5fbab56b611efef9f7cac8b61f0906ba49.jpg)

![1b2295c02381e685374055b648d0bee4440dd6bac79d32d9cdc2e296b3596d39.jpg](../icml_results/4623_Cross-Modal%20Knowledge%20Distillation%20without%20Paired%20Data_%20Theoretical%20Foundation%20and%20Algorithm/images/1b2295c02381e685374055b648d0bee4440dd6bac79d32d9cdc2e296b3596d39.jpg)

![ae1f03385fe2dd9b4a57d6df8f026fd19993ca3e85808f4e5075598a74d3f22b.jpg](../icml_results/4623_Cross-Modal%20Knowledge%20Distillation%20without%20Paired%20Data_%20Theoretical%20Foundation%20and%20Algorithm/images/ae1f03385fe2dd9b4a57d6df8f026fd19993ca3e85808f4e5075598a74d3f22b.jpg)

### Tables

![133fd1cd169dbfadf40843fa442d173c0629efa42f937497ead121567b26e4f5.jpg](../icml_results/4623_Cross-Modal%20Knowledge%20Distillation%20without%20Paired%20Data_%20Theoretical%20Foundation%20and%20Algorithm/tables/133fd1cd169dbfadf40843fa442d173c0629efa42f937497ead121567b26e4f5.jpg)

![1d13a6cc41b80a30f87da5bef4de35e33ead9459d6bf563dea3070aec830532b.jpg](../icml_results/4623_Cross-Modal%20Knowledge%20Distillation%20without%20Paired%20Data_%20Theoretical%20Foundation%20and%20Algorithm/tables/1d13a6cc41b80a30f87da5bef4de35e33ead9459d6bf563dea3070aec830532b.jpg)

![1d7c899b200fa04817660f1380cea617c34b9a26f755d9090c20c2467be57574.jpg](../icml_results/4623_Cross-Modal%20Knowledge%20Distillation%20without%20Paired%20Data_%20Theoretical%20Foundation%20and%20Algorithm/tables/1d7c899b200fa04817660f1380cea617c34b9a26f755d9090c20c2467be57574.jpg)

![29e80b38db009f7ac57acc2e37d1d9b551954430e80c7f59d951dffb2ad8f680.jpg](../icml_results/4623_Cross-Modal%20Knowledge%20Distillation%20without%20Paired%20Data_%20Theoretical%20Foundation%20and%20Algorithm/tables/29e80b38db009f7ac57acc2e37d1d9b551954430e80c7f59d951dffb2ad8f680.jpg)

![4a69e8e64272625c3210c8ed2abf062fc0db7efb4ee9a4acb70bedd37e25b279.jpg](../icml_results/4623_Cross-Modal%20Knowledge%20Distillation%20without%20Paired%20Data_%20Theoretical%20Foundation%20and%20Algorithm/tables/4a69e8e64272625c3210c8ed2abf062fc0db7efb4ee9a4acb70bedd37e25b279.jpg)

![4bcc6b1ee855c038e0c73fc348cdc20afdf652a0e54f51d478baab00d3967af9.jpg](../icml_results/4623_Cross-Modal%20Knowledge%20Distillation%20without%20Paired%20Data_%20Theoretical%20Foundation%20and%20Algorithm/tables/4bcc6b1ee855c038e0c73fc348cdc20afdf652a0e54f51d478baab00d3967af9.jpg)

![4eda13b032900424ec3625ddb9260dba71b834b3eeefe3e6697063a00a461929.jpg](../icml_results/4623_Cross-Modal%20Knowledge%20Distillation%20without%20Paired%20Data_%20Theoretical%20Foundation%20and%20Algorithm/tables/4eda13b032900424ec3625ddb9260dba71b834b3eeefe3e6697063a00a461929.jpg)

![5bd61d2aad38f5d3387e366c44faece5af633aabbab86b809da0a7c9c46df0bf.jpg](../icml_results/4623_Cross-Modal%20Knowledge%20Distillation%20without%20Paired%20Data_%20Theoretical%20Foundation%20and%20Algorithm/tables/5bd61d2aad38f5d3387e366c44faece5af633aabbab86b809da0a7c9c46df0bf.jpg)

![6e7e8233ba6402ce8179cd75fcfc15359365201dd299d8021b2700a386cc56e7.jpg](../icml_results/4623_Cross-Modal%20Knowledge%20Distillation%20without%20Paired%20Data_%20Theoretical%20Foundation%20and%20Algorithm/tables/6e7e8233ba6402ce8179cd75fcfc15359365201dd299d8021b2700a386cc56e7.jpg)

![84ae1ada1fe63908c7416ab2ce3ac46dc64cb4102626753f6d08fdbd5a1f58c1.jpg](../icml_results/4623_Cross-Modal%20Knowledge%20Distillation%20without%20Paired%20Data_%20Theoretical%20Foundation%20and%20Algorithm/tables/84ae1ada1fe63908c7416ab2ce3ac46dc64cb4102626753f6d08fdbd5a1f58c1.jpg)

![a3d58a6035da1751c97b0cfa5b66a602e0ea61a688551b752fa20c394adbbeba.jpg](../icml_results/4623_Cross-Modal%20Knowledge%20Distillation%20without%20Paired%20Data_%20Theoretical%20Foundation%20and%20Algorithm/tables/a3d58a6035da1751c97b0cfa5b66a602e0ea61a688551b752fa20c394adbbeba.jpg)

![beef5a9494b3c5a7b997b809be65fa68756b26556b09e267cca062635f3350fe.jpg](../icml_results/4623_Cross-Modal%20Knowledge%20Distillation%20without%20Paired%20Data_%20Theoretical%20Foundation%20and%20Algorithm/tables/beef5a9494b3c5a7b997b809be65fa68756b26556b09e267cca062635f3350fe.jpg)

![e059c9bb3a5336672396da39cde2b279573e328407ce0c32c0ba9f62f8a8721f.jpg](../icml_results/4623_Cross-Modal%20Knowledge%20Distillation%20without%20Paired%20Data_%20Theoretical%20Foundation%20and%20Algorithm/tables/e059c9bb3a5336672396da39cde2b279573e328407ce0c32c0ba9f62f8a8721f.jpg)

![e9777d6e4eba4de4421cd010ad8f1f3fee43807a46e5caf8a47850a115d0a2cb.jpg](../icml_results/4623_Cross-Modal%20Knowledge%20Distillation%20without%20Paired%20Data_%20Theoretical%20Foundation%20and%20Algorithm/tables/e9777d6e4eba4de4421cd010ad8f1f3fee43807a46e5caf8a47850a115d0a2cb.jpg)

<a id="paper-4577"></a>

## UB-SMoE: Universally Balanced Sparse Mixture-of-Experts for Resource-adaptive Federated Fine-tuning of Foundation Models


### Images

![4669bf7f6f68fcc7262368d2b0b2b4c58920176f8b4933ee0ee4c5c58e05cb83.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/images/4669bf7f6f68fcc7262368d2b0b2b4c58920176f8b4933ee0ee4c5c58e05cb83.jpg)

![6f1b615ced381a318baad3c4521915ad1680ef763a71ec4d89dab6c3ad284450.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/images/6f1b615ced381a318baad3c4521915ad1680ef763a71ec4d89dab6c3ad284450.jpg)

![9491da86398e363fcb5acb1425cc095fc9471770a9f5e4bba1ac2e9d6aa2551a.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/images/9491da86398e363fcb5acb1425cc095fc9471770a9f5e4bba1ac2e9d6aa2551a.jpg)

![a4236c4847f7df146181de19a0fc961c1f9f31b7a950b0be19a39c361802f15a.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/images/a4236c4847f7df146181de19a0fc961c1f9f31b7a950b0be19a39c361802f15a.jpg)

![a5982ab62311a3f016cb55d0e472bdcd4f4b60f6f55fb0f61945404c32decd38.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/images/a5982ab62311a3f016cb55d0e472bdcd4f4b60f6f55fb0f61945404c32decd38.jpg)

![acf8f3ab4fcd3c1a788701897a7abc514714fb2b37758a3eb6d938336c645a09.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/images/acf8f3ab4fcd3c1a788701897a7abc514714fb2b37758a3eb6d938336c645a09.jpg)

![adb8fd59a62a0f11ef671c2890015026009202b022273b8510bfe9d896f999e6.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/images/adb8fd59a62a0f11ef671c2890015026009202b022273b8510bfe9d896f999e6.jpg)

![f7e264b7fabd72e0773301350e68b0b51bf89aed233d9576d23d906cb8c516e2.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/images/f7e264b7fabd72e0773301350e68b0b51bf89aed233d9576d23d906cb8c516e2.jpg)

### Tables

![0ad376fc622cd7a861b5e5c0ae941f8545d196dea8e78f5f353f9e7dc62d38af.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/0ad376fc622cd7a861b5e5c0ae941f8545d196dea8e78f5f353f9e7dc62d38af.jpg)

![0d5fd296240be0affd387b22fc44e33dd2e1b311bde04f8031eab6f9167ccc14.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/0d5fd296240be0affd387b22fc44e33dd2e1b311bde04f8031eab6f9167ccc14.jpg)

![2217cedfb6c341d160d72d0dd92f9fa52383af3da0cfa6f8648e5ea33a2df9fb.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/2217cedfb6c341d160d72d0dd92f9fa52383af3da0cfa6f8648e5ea33a2df9fb.jpg)

![31c51411a7fd5327ed0f450feed7b96e31c6128b87a7cb9a81c3e23fe6a759ea.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/31c51411a7fd5327ed0f450feed7b96e31c6128b87a7cb9a81c3e23fe6a759ea.jpg)

![3201a8e9abcc3f8741e7218ed4ef47bc88351407961fd0a139fa90a990ee2bed.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/3201a8e9abcc3f8741e7218ed4ef47bc88351407961fd0a139fa90a990ee2bed.jpg)

![4c4b8b0a67d233484255e27dc48ed55d37c9945ec760c3c13d81ba46f07f8bce.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/4c4b8b0a67d233484255e27dc48ed55d37c9945ec760c3c13d81ba46f07f8bce.jpg)

![6fb011c5cc7cc5af3a017e6578373c73d0fa2651bf7ba9ac51fff2d8ba4d82dc.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/6fb011c5cc7cc5af3a017e6578373c73d0fa2651bf7ba9ac51fff2d8ba4d82dc.jpg)

![7aade2bd480e2b9e13be648ec73a9897e71616121154b288dab687847c53349c.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/7aade2bd480e2b9e13be648ec73a9897e71616121154b288dab687847c53349c.jpg)

![7e102e44a8bd85a288b4db608aa5f633d2e008531d78d8133b82483170dbfe9c.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/7e102e44a8bd85a288b4db608aa5f633d2e008531d78d8133b82483170dbfe9c.jpg)

![9fa530bdeba116d33fc5d79c8eb4d3c836ac8a81451475aa4d048d1dc3842c64.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/9fa530bdeba116d33fc5d79c8eb4d3c836ac8a81451475aa4d048d1dc3842c64.jpg)

![a5e705f289ba9ca9544c167b2f47cf49401a268fbb244c4f9b2ac5c1ba98ccdd.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/a5e705f289ba9ca9544c167b2f47cf49401a268fbb244c4f9b2ac5c1ba98ccdd.jpg)

![a6d89afc97799fb4b1c6d87e5696edce926660fbb59bafa9b0319f78c86a6c4f.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/a6d89afc97799fb4b1c6d87e5696edce926660fbb59bafa9b0319f78c86a6c4f.jpg)

![b4536a96ed3f295f24caf179d921646d3f70491c9e795b20a6608c43204b353d.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/b4536a96ed3f295f24caf179d921646d3f70491c9e795b20a6608c43204b353d.jpg)

![b71be11a56e1406e378c5c69a4911d5ee2c5ed4be2ea82b45639be6d571c13bd.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/b71be11a56e1406e378c5c69a4911d5ee2c5ed4be2ea82b45639be6d571c13bd.jpg)

![c2723dba644eadec89634c5647fe25669488b7f2bdaada79ccda0a0546a05cf6.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/c2723dba644eadec89634c5647fe25669488b7f2bdaada79ccda0a0546a05cf6.jpg)

![c50105875dbada2c864a5d4d12751a0b056311055c1bd3f9a2755acbf4e2afc0.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/c50105875dbada2c864a5d4d12751a0b056311055c1bd3f9a2755acbf4e2afc0.jpg)

![c94428434ebb158789577a9ee4fbc247d0b7f243fafa7901dcc961f2e8fc4aaa.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/c94428434ebb158789577a9ee4fbc247d0b7f243fafa7901dcc961f2e8fc4aaa.jpg)

![cd8184fa11ab2d747f24778b780befdb1519852f04b7191157157212a6d94c5b.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/cd8184fa11ab2d747f24778b780befdb1519852f04b7191157157212a6d94c5b.jpg)

![d3d6053c698f1f22df9eaa0c2929cc5296774cc4eae1974eb060d4cd7f4298b7.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/d3d6053c698f1f22df9eaa0c2929cc5296774cc4eae1974eb060d4cd7f4298b7.jpg)

![f1390cab05e7106fc9fc45e2ad4d46f6339353e84bbd4ca31b0fdd285f630885.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/f1390cab05e7106fc9fc45e2ad4d46f6339353e84bbd4ca31b0fdd285f630885.jpg)

![f5601ae009e1f695d625f15b28fb6b057d69197a0ba4734e89fc7ff582e6515a.jpg](../icml_results/4624_UB-SMoE_%20Universally%20Balanced%20Sparse%20Mixture-of-Experts%20for%20Resource-adaptive%20Federated%20Fine-tuning%20/tables/f5601ae009e1f695d625f15b28fb6b057d69197a0ba4734e89fc7ff582e6515a.jpg)

<a id="paper-4578"></a>

## SEMA: a Scalable and Efficient Mamba like Attention via Token Localization and Averaging


### Images

![1970c6c0c2149b01802883698c7943b416352044832d0d6725ec395502e65e19.jpg](../icml_results/4625_SEMA_%20a%20Scalable%20and%20Efficient%20Mamba%20like%20Attention%20via%20Token%20Localization%20and%20Averaging/images/1970c6c0c2149b01802883698c7943b416352044832d0d6725ec395502e65e19.jpg)

![1ae8df31f1ed3d27c14fb11eda348e90720f2bc6ebc3fe052fbacebcc7c1c832.jpg](../icml_results/4625_SEMA_%20a%20Scalable%20and%20Efficient%20Mamba%20like%20Attention%20via%20Token%20Localization%20and%20Averaging/images/1ae8df31f1ed3d27c14fb11eda348e90720f2bc6ebc3fe052fbacebcc7c1c832.jpg)

![d92259f0a847d84ccc3a75258ad0174ac32b6d8610ff754cea4ba006622f5afd.jpg](../icml_results/4625_SEMA_%20a%20Scalable%20and%20Efficient%20Mamba%20like%20Attention%20via%20Token%20Localization%20and%20Averaging/images/d92259f0a847d84ccc3a75258ad0174ac32b6d8610ff754cea4ba006622f5afd.jpg)

### Tables

![01f14c09198112c1ee21048a4bddf9c0e74eb266c1f044901f76a1c04e0ac664.jpg](../icml_results/4625_SEMA_%20a%20Scalable%20and%20Efficient%20Mamba%20like%20Attention%20via%20Token%20Localization%20and%20Averaging/tables/01f14c09198112c1ee21048a4bddf9c0e74eb266c1f044901f76a1c04e0ac664.jpg)

![1c26e888f7791976a5bb7437c063ae5defec0d9fbbf4bf10e07d521c6972b081.jpg](../icml_results/4625_SEMA_%20a%20Scalable%20and%20Efficient%20Mamba%20like%20Attention%20via%20Token%20Localization%20and%20Averaging/tables/1c26e888f7791976a5bb7437c063ae5defec0d9fbbf4bf10e07d521c6972b081.jpg)

![3d056bc364318660b34e8757deb48a6fadf443bbb5b9c1ab99f6191eda2c01cd.jpg](../icml_results/4625_SEMA_%20a%20Scalable%20and%20Efficient%20Mamba%20like%20Attention%20via%20Token%20Localization%20and%20Averaging/tables/3d056bc364318660b34e8757deb48a6fadf443bbb5b9c1ab99f6191eda2c01cd.jpg)

![4d588c2cf74d548a3254f8a94e1dc433217441317cfd3acf0eb95f520941c280.jpg](../icml_results/4625_SEMA_%20a%20Scalable%20and%20Efficient%20Mamba%20like%20Attention%20via%20Token%20Localization%20and%20Averaging/tables/4d588c2cf74d548a3254f8a94e1dc433217441317cfd3acf0eb95f520941c280.jpg)

![527e9e714710387d6b72b9b7be258713a37f333f72fed8608db4119cc31167e1.jpg](../icml_results/4625_SEMA_%20a%20Scalable%20and%20Efficient%20Mamba%20like%20Attention%20via%20Token%20Localization%20and%20Averaging/tables/527e9e714710387d6b72b9b7be258713a37f333f72fed8608db4119cc31167e1.jpg)

![5cff7d36b683366bdd8bdd5292a3ec6320ea0c94cd9017484c21fe7c59e16a9b.jpg](../icml_results/4625_SEMA_%20a%20Scalable%20and%20Efficient%20Mamba%20like%20Attention%20via%20Token%20Localization%20and%20Averaging/tables/5cff7d36b683366bdd8bdd5292a3ec6320ea0c94cd9017484c21fe7c59e16a9b.jpg)

![60ef93dd0e0b2b7ae2b6a97af5065abf96e918ea9667560b33aa889605f38d5c.jpg](../icml_results/4625_SEMA_%20a%20Scalable%20and%20Efficient%20Mamba%20like%20Attention%20via%20Token%20Localization%20and%20Averaging/tables/60ef93dd0e0b2b7ae2b6a97af5065abf96e918ea9667560b33aa889605f38d5c.jpg)

![7fa400c4cd2de371025d171c062948b5a7176dd9b2397d292359b959d749e2f6.jpg](../icml_results/4625_SEMA_%20a%20Scalable%20and%20Efficient%20Mamba%20like%20Attention%20via%20Token%20Localization%20and%20Averaging/tables/7fa400c4cd2de371025d171c062948b5a7176dd9b2397d292359b959d749e2f6.jpg)

![82ea6dfbf54e5fcd35a7b2b70a1dbdf604486978780e9d26f1ce995439a7c777.jpg](../icml_results/4625_SEMA_%20a%20Scalable%20and%20Efficient%20Mamba%20like%20Attention%20via%20Token%20Localization%20and%20Averaging/tables/82ea6dfbf54e5fcd35a7b2b70a1dbdf604486978780e9d26f1ce995439a7c777.jpg)

![d85fd388de452435d76a27d352fb8fd2e7c5ef2a7dd3c70cb2a0fd6529dba9a9.jpg](../icml_results/4625_SEMA_%20a%20Scalable%20and%20Efficient%20Mamba%20like%20Attention%20via%20Token%20Localization%20and%20Averaging/tables/d85fd388de452435d76a27d352fb8fd2e7c5ef2a7dd3c70cb2a0fd6529dba9a9.jpg)

<a id="paper-4579"></a>

## Unified Time Series Explanations via Amortized Optimization and Instance-level Multi-Expert Knowledge Distillation


### Images

![2d52415d86aec91598ce9c6bd915dc8ecf3bd976523e1a4d43b950ac9b620d8b.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/images/2d52415d86aec91598ce9c6bd915dc8ecf3bd976523e1a4d43b950ac9b620d8b.jpg)

![5b30265cddc3561c7a9256231cf44087a3666d3bff9749fcffe4a634be563476.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/images/5b30265cddc3561c7a9256231cf44087a3666d3bff9749fcffe4a634be563476.jpg)

![a53e99df2023e5613fff54b4e9409038da510a3a8df18fad3f6b26ea3e9ccfa4.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/images/a53e99df2023e5613fff54b4e9409038da510a3a8df18fad3f6b26ea3e9ccfa4.jpg)

![b9225fdf48fef596f7fc96c9dab26090958d6da7cbc35942aef7bafa18df648e.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/images/b9225fdf48fef596f7fc96c9dab26090958d6da7cbc35942aef7bafa18df648e.jpg)

![bbe80d6a43561d7d2e7803c85ed98e54435e62d036e39d59e44ea9b96d102998.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/images/bbe80d6a43561d7d2e7803c85ed98e54435e62d036e39d59e44ea9b96d102998.jpg)

![d2bd366571c24ca2426f92011bb8bc1cf405686ece9b4d3a7aabeb908e6a701c.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/images/d2bd366571c24ca2426f92011bb8bc1cf405686ece9b4d3a7aabeb908e6a701c.jpg)

![d4224cbbd29720fae3225fc98896be8b8dda898ab2059ef923056356c9feaf2a.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/images/d4224cbbd29720fae3225fc98896be8b8dda898ab2059ef923056356c9feaf2a.jpg)

![e6a7e36a5aed501236f31d1017a4bbfcce92bfbfd40a1b639b6b1a0d6f5de52a.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/images/e6a7e36a5aed501236f31d1017a4bbfcce92bfbfd40a1b639b6b1a0d6f5de52a.jpg)

![f72429f1ad2b39d67d9318aff66f2dd3d4dedd804cf43ce81cb0da1726b283a2.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/images/f72429f1ad2b39d67d9318aff66f2dd3d4dedd804cf43ce81cb0da1726b283a2.jpg)

### Tables

![00ee41ef8de27896ae66e5ce7d5722da1f8f9ed8d2593f9d2254f707cd895c6b.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/00ee41ef8de27896ae66e5ce7d5722da1f8f9ed8d2593f9d2254f707cd895c6b.jpg)

![0eddc3c46aaf8d943579d13ffc466b5b8525eb2a3b56f9ccb16e23cf8c432871.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/0eddc3c46aaf8d943579d13ffc466b5b8525eb2a3b56f9ccb16e23cf8c432871.jpg)

![115420fe428eb20a976f55fe72630af0d99bc09936dd42d709b5ea22a2feb7fd.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/115420fe428eb20a976f55fe72630af0d99bc09936dd42d709b5ea22a2feb7fd.jpg)

![16d77e763a10637d77b6f5625cb4a41827d3dba0ca9bd1c73d909fa362dfcda7.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/16d77e763a10637d77b6f5625cb4a41827d3dba0ca9bd1c73d909fa362dfcda7.jpg)

![2aeebb85408fba92022fb0b00813f066b95fbb5ec37d7b5349b6d4d42ff1307c.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/2aeebb85408fba92022fb0b00813f066b95fbb5ec37d7b5349b6d4d42ff1307c.jpg)

![310ac4fbcc7b99a7e4c173bb4b6f040b93118245c0db073e5a0e3343a274efe2.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/310ac4fbcc7b99a7e4c173bb4b6f040b93118245c0db073e5a0e3343a274efe2.jpg)

![52689473ee0d803a5b0ba4964b7431aca83249222ce0ba0b21337d04bc783284.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/52689473ee0d803a5b0ba4964b7431aca83249222ce0ba0b21337d04bc783284.jpg)

![5b1443897f093b0751233de3a085446dc05dc39ae2e6aaa99c0b040e2625f181.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/5b1443897f093b0751233de3a085446dc05dc39ae2e6aaa99c0b040e2625f181.jpg)

![68a42592fd4bfbdfd5ab9c4ecec221e86111feb5f3fc2f1ff1f7c74695f6381f.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/68a42592fd4bfbdfd5ab9c4ecec221e86111feb5f3fc2f1ff1f7c74695f6381f.jpg)

![6a541aea527ab2c68e7fa232ae2d2afb46dd663e1e7b04a39bd5e080ff81300b.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/6a541aea527ab2c68e7fa232ae2d2afb46dd663e1e7b04a39bd5e080ff81300b.jpg)

![7e63b6e5c35d974291c1e963e0dafdbd7a2dff2ca6a70649a6778f519a5bc3c3.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/7e63b6e5c35d974291c1e963e0dafdbd7a2dff2ca6a70649a6778f519a5bc3c3.jpg)

![8a3f5c328edcfcc05578dfe6b60b77d99ca6dd412c4fbea25b5461b1faee711d.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/8a3f5c328edcfcc05578dfe6b60b77d99ca6dd412c4fbea25b5461b1faee711d.jpg)

![8bdbfbd251a8721e6d9baf1505f7fc6640eaf92ddb006e750ceea9846acbe1bc.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/8bdbfbd251a8721e6d9baf1505f7fc6640eaf92ddb006e750ceea9846acbe1bc.jpg)

![9136f16bad9bfce2e7347c6b415049a4c9bf25348a067e88a3f8ceea04d240e3.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/9136f16bad9bfce2e7347c6b415049a4c9bf25348a067e88a3f8ceea04d240e3.jpg)

![9143bbfa5eeeca354990c183dbf45ffaeef97d9418f80b52b59bdfbf1eb4e9f5.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/9143bbfa5eeeca354990c183dbf45ffaeef97d9418f80b52b59bdfbf1eb4e9f5.jpg)

![92e95e0d87e7e49705e60f89a9f0a6aacf1f0ef817240f2026bee911895767cf.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/92e95e0d87e7e49705e60f89a9f0a6aacf1f0ef817240f2026bee911895767cf.jpg)

![9ad132bda21ff765df25fd5ddf98e982f176594a1a51624613b0541ba262a1da.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/9ad132bda21ff765df25fd5ddf98e982f176594a1a51624613b0541ba262a1da.jpg)

![a0927fb79d2f5c2bdceaf4d1b232e72465f716a4c774b8ed4202f47c8aa2b12e.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/a0927fb79d2f5c2bdceaf4d1b232e72465f716a4c774b8ed4202f47c8aa2b12e.jpg)

![b3090149f3b50ad5e726790aa6aca53bb359af36659976463fa34c497eadeea2.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/b3090149f3b50ad5e726790aa6aca53bb359af36659976463fa34c497eadeea2.jpg)

![be7a5ae01a750204c335622068e8bceeb113a6e36b01ae4d0b414d52db659ca6.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/be7a5ae01a750204c335622068e8bceeb113a6e36b01ae4d0b414d52db659ca6.jpg)

![cfb62df796178013af07b55ae00bd1886980602fb8d5fb1a7170f5e5f77702ae.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/cfb62df796178013af07b55ae00bd1886980602fb8d5fb1a7170f5e5f77702ae.jpg)

![db0b068985225324bd2da56fbf446b65a6879071074af458947823c0d2dc3b6b.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/db0b068985225324bd2da56fbf446b65a6879071074af458947823c0d2dc3b6b.jpg)

![ee1a4d834b6670e5a87c9435504dde53eccaa61d06835ac9fe63474263a0013d.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/ee1a4d834b6670e5a87c9435504dde53eccaa61d06835ac9fe63474263a0013d.jpg)

![f1c0ecd6d7e0009bc84b729276c965cc6d9d944ce199e10674ed17aa37eb5d88.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/f1c0ecd6d7e0009bc84b729276c965cc6d9d944ce199e10674ed17aa37eb5d88.jpg)

![f243025be722fcb9b5a1257fc69b629c321c3f202fdb76b5a7be6dad190adcca.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/f243025be722fcb9b5a1257fc69b629c321c3f202fdb76b5a7be6dad190adcca.jpg)

![fa5e51d4dc9dcfa86b307acc8fe04f20aa39ddd0a12cb7274a4a470949a7dcf9.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/fa5e51d4dc9dcfa86b307acc8fe04f20aa39ddd0a12cb7274a4a470949a7dcf9.jpg)

![fdf61d44a368cd113689e105684fb3d922492a9d32135a5a6dae2a4c2993824a.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/fdf61d44a368cd113689e105684fb3d922492a9d32135a5a6dae2a4c2993824a.jpg)

![ffb11132833da76e541862102ac09b988d95b6c821c319afe8c9a89e83ca0b59.jpg](../icml_results/4626_Unified%20Time%20Series%20Explanations%20via%20Amortized%20Optimization%20and%20Instance-level%20Multi-Expert%20Knowledg/tables/ffb11132833da76e541862102ac09b988d95b6c821c319afe8c9a89e83ca0b59.jpg)

<a id="paper-4580"></a>

## Looking Locally: Object-Centric Vision Transformers as Foundation Models for Efficient Segmentation


### Images

![4f59e787a04c215dac89868f972c6d7a5bbb69c214c4a4f763392f0717b601ed.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/images/4f59e787a04c215dac89868f972c6d7a5bbb69c214c4a4f763392f0717b601ed.jpg)

![7287738ddaafe065fb3a6ba46e5aba7f7ec5500ac2d6d39d4b466071355ee239.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/images/7287738ddaafe065fb3a6ba46e5aba7f7ec5500ac2d6d39d4b466071355ee239.jpg)

![757ed9afeae58dfc5950b75113dbdb5ae84844b900aa7d904160c425d3b3de69.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/images/757ed9afeae58dfc5950b75113dbdb5ae84844b900aa7d904160c425d3b3de69.jpg)

![82d919153e8f0b644bbe500d0cd6daebec0c178c178fc6aab52e1edf212ab5ae.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/images/82d919153e8f0b644bbe500d0cd6daebec0c178c178fc6aab52e1edf212ab5ae.jpg)

![8558ffa0650cbb60386978688d56ebce72c221bbc3d356e2db60ff00de85585a.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/images/8558ffa0650cbb60386978688d56ebce72c221bbc3d356e2db60ff00de85585a.jpg)

![bf15dc56b1e11d1d37fc4c64bca74df8a9a0bf9652bbaa3a84b6c16c3844967c.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/images/bf15dc56b1e11d1d37fc4c64bca74df8a9a0bf9652bbaa3a84b6c16c3844967c.jpg)

![c1edf555ef0538e766937514203da25f41f8dcdcac494cf4b99746e9c40b619e.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/images/c1edf555ef0538e766937514203da25f41f8dcdcac494cf4b99746e9c40b619e.jpg)

![c68595981a0ccee2d0103bb6b6c880dc36ba6724a9e7a08106b0cade9d645759.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/images/c68595981a0ccee2d0103bb6b6c880dc36ba6724a9e7a08106b0cade9d645759.jpg)

![f8979107412eb3eb72da6093ded4b83c0dc117870f93b52d90ad2c1ba37678f0.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/images/f8979107412eb3eb72da6093ded4b83c0dc117870f93b52d90ad2c1ba37678f0.jpg)

![fbaa2cd7613c25ac154e7715947c0be8390fd72ce1fd285b0aa25992b64720c9.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/images/fbaa2cd7613c25ac154e7715947c0be8390fd72ce1fd285b0aa25992b64720c9.jpg)

### Tables

![105e83e1901417b45a3514659871e35e02e3e37df2de1daf9f16c793c6081da8.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/tables/105e83e1901417b45a3514659871e35e02e3e37df2de1daf9f16c793c6081da8.jpg)

![3396811567716336d7ed1a172425b0e2462a5b770db550415c8c644c5b322f83.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/tables/3396811567716336d7ed1a172425b0e2462a5b770db550415c8c644c5b322f83.jpg)

![358e19ea235ce81bff785f1e627fabd894dc6a2dfffcf929ec9af56a71642665.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/tables/358e19ea235ce81bff785f1e627fabd894dc6a2dfffcf929ec9af56a71642665.jpg)

![5437429f41c57d5070d5f8c23c36711991f4a4d16491a8fd5e1630cfc2a90c75.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/tables/5437429f41c57d5070d5f8c23c36711991f4a4d16491a8fd5e1630cfc2a90c75.jpg)

![5f882e663d285e407829a1e2442e3970062dc47ef1722a829aa4cde0b3adfaa4.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/tables/5f882e663d285e407829a1e2442e3970062dc47ef1722a829aa4cde0b3adfaa4.jpg)

![8ace55d843dd3814a39118aa0e909ec398199fa5f861ccc0128114085d99e4fa.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/tables/8ace55d843dd3814a39118aa0e909ec398199fa5f861ccc0128114085d99e4fa.jpg)

![9dc089e31a5ab5a44b88f45f65b5308c3a56dd1853fd7f880a2181aee97a39cb.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/tables/9dc089e31a5ab5a44b88f45f65b5308c3a56dd1853fd7f880a2181aee97a39cb.jpg)

![ab7ebf039aa4ea084b9bb5036bd7be5c4f3ee91fd98b0297c1d09b92a6bee312.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/tables/ab7ebf039aa4ea084b9bb5036bd7be5c4f3ee91fd98b0297c1d09b92a6bee312.jpg)

![cef2435da24117832d456313069bb823897326c29942c1d02e37e96256791c31.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/tables/cef2435da24117832d456313069bb823897326c29942c1d02e37e96256791c31.jpg)

![d8d2ab0f786b3e6c5daf3a333d85efb62db4f7adb89886d7aa41f820269ea2d4.jpg](../icml_results/4627_Looking%20Locally_%20Object-Centric%20Vision%20Transformers%20as%20Foundation%20Models%20for%20Efficient%20Segmentation/tables/d8d2ab0f786b3e6c5daf3a333d85efb62db4f7adb89886d7aa41f820269ea2d4.jpg)

<a id="paper-4581"></a>

## Differentiable Optimization Layers for Guaranteed Fairness in Deep Learning


### Images

![63bd4951d6e39f829c51b500172e1f16ea7be3f4a5c679e458dc8b0a2df7dd0d.jpg](../icml_results/4628_Differentiable%20Optimization%20Layers%20for%20Guaranteed%20Fairness%20in%20Deep%20Learning/images/63bd4951d6e39f829c51b500172e1f16ea7be3f4a5c679e458dc8b0a2df7dd0d.jpg)

![7d41445890092faaef540e2874881e0f0a3e1c97f4f68611957b7f1b3de5bd5b.jpg](../icml_results/4628_Differentiable%20Optimization%20Layers%20for%20Guaranteed%20Fairness%20in%20Deep%20Learning/images/7d41445890092faaef540e2874881e0f0a3e1c97f4f68611957b7f1b3de5bd5b.jpg)

![8de1da767baa2714796f4cd23c98dfc55fa178094c701521dabedbb50f8b0c39.jpg](../icml_results/4628_Differentiable%20Optimization%20Layers%20for%20Guaranteed%20Fairness%20in%20Deep%20Learning/images/8de1da767baa2714796f4cd23c98dfc55fa178094c701521dabedbb50f8b0c39.jpg)

![ceb5882e1b0a3c1a57c824a85c2b512951aac4b756db522fa54b631cc7d9ccbd.jpg](../icml_results/4628_Differentiable%20Optimization%20Layers%20for%20Guaranteed%20Fairness%20in%20Deep%20Learning/images/ceb5882e1b0a3c1a57c824a85c2b512951aac4b756db522fa54b631cc7d9ccbd.jpg)

### Tables

![14401b34dee8495f6089b65862336d3f0a8040f14c09d322ad160809de1a27ee.jpg](../icml_results/4628_Differentiable%20Optimization%20Layers%20for%20Guaranteed%20Fairness%20in%20Deep%20Learning/tables/14401b34dee8495f6089b65862336d3f0a8040f14c09d322ad160809de1a27ee.jpg)

![2832b25be188d695ecba7aa14db76f04e686e3d3d20d2f9cbdbe401b7ac9d0ac.jpg](../icml_results/4628_Differentiable%20Optimization%20Layers%20for%20Guaranteed%20Fairness%20in%20Deep%20Learning/tables/2832b25be188d695ecba7aa14db76f04e686e3d3d20d2f9cbdbe401b7ac9d0ac.jpg)

![6007c62cde2a9e7a7d6a6af2d99d665cbfd949d0757fd5f1a114d14ba56e822a.jpg](../icml_results/4628_Differentiable%20Optimization%20Layers%20for%20Guaranteed%20Fairness%20in%20Deep%20Learning/tables/6007c62cde2a9e7a7d6a6af2d99d665cbfd949d0757fd5f1a114d14ba56e822a.jpg)

![7db2ef7989fbb7fe1f45e7d15076d6465f12cb37e2be499ae2e932a6fa0855ba.jpg](../icml_results/4628_Differentiable%20Optimization%20Layers%20for%20Guaranteed%20Fairness%20in%20Deep%20Learning/tables/7db2ef7989fbb7fe1f45e7d15076d6465f12cb37e2be499ae2e932a6fa0855ba.jpg)

![b8a991517fe8417d5100f522116671d0fcd3f324153afacd962e6243e3d4cbcb.jpg](../icml_results/4628_Differentiable%20Optimization%20Layers%20for%20Guaranteed%20Fairness%20in%20Deep%20Learning/tables/b8a991517fe8417d5100f522116671d0fcd3f324153afacd962e6243e3d4cbcb.jpg)

![dd950a828449cba4a8d54257b771d6a7ecc453071cb62d5cbc6be0b0e087e73a.jpg](../icml_results/4628_Differentiable%20Optimization%20Layers%20for%20Guaranteed%20Fairness%20in%20Deep%20Learning/tables/dd950a828449cba4a8d54257b771d6a7ecc453071cb62d5cbc6be0b0e087e73a.jpg)

![ffb35e6752252cec88d614371d59960924ecdc62bdbf9e2bf1d78b21ce9ab263.jpg](../icml_results/4628_Differentiable%20Optimization%20Layers%20for%20Guaranteed%20Fairness%20in%20Deep%20Learning/tables/ffb35e6752252cec88d614371d59960924ecdc62bdbf9e2bf1d78b21ce9ab263.jpg)

<a id="paper-4582"></a>

## Plain Transformers are Surprisingly Powerful Link Predictors


### Images

![1099b346c97aeee39f9abb4976d090c4e389aa28ca98a84d69e0d18c4cd6031c.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/images/1099b346c97aeee39f9abb4976d090c4e389aa28ca98a84d69e0d18c4cd6031c.jpg)

![1440e8e4d4ac5997075e7ba6b9290a8d792ae1f437bb68dfd0d9da3631cfe60a.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/images/1440e8e4d4ac5997075e7ba6b9290a8d792ae1f437bb68dfd0d9da3631cfe60a.jpg)

![1b54910e66aa9757860a1ccae9251ad4bbe5eb7b07c1f7057985c506c5722be7.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/images/1b54910e66aa9757860a1ccae9251ad4bbe5eb7b07c1f7057985c506c5722be7.jpg)

![331ae6f748581879457da923a675aa2dce53248f322701459a25e74b50a36806.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/images/331ae6f748581879457da923a675aa2dce53248f322701459a25e74b50a36806.jpg)

![511111cc3dacd411c2b94c48480edcd36cf3decdc682782d14dc1fcc0400f0e0.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/images/511111cc3dacd411c2b94c48480edcd36cf3decdc682782d14dc1fcc0400f0e0.jpg)

![6e231773d63e28a2e334deabe6dd7a440b4eba33a0b308f9fd2f600d95416c4f.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/images/6e231773d63e28a2e334deabe6dd7a440b4eba33a0b308f9fd2f600d95416c4f.jpg)

![85b02d6e5a16a48b184dbaa435dd1228bf4b515862a93731191e28ae524b2648.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/images/85b02d6e5a16a48b184dbaa435dd1228bf4b515862a93731191e28ae524b2648.jpg)

![8f10e50f30538dbe71d9741d06b3d23f6bd1692fbc97cd165581954109c96d2b.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/images/8f10e50f30538dbe71d9741d06b3d23f6bd1692fbc97cd165581954109c96d2b.jpg)

![971c9a2953e244ec3be306c943d63d63d8400db4da128ef706da4c4f2d1e8d27.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/images/971c9a2953e244ec3be306c943d63d63d8400db4da128ef706da4c4f2d1e8d27.jpg)

![dacd00c835e1356a91ab153f22fced37257ec8d8cfdb79ff7df963f19efab65c.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/images/dacd00c835e1356a91ab153f22fced37257ec8d8cfdb79ff7df963f19efab65c.jpg)

### Tables

![17d03c66802f536f7a44867681a89bc883cdc53b1742b834f58267c5a331f253.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/tables/17d03c66802f536f7a44867681a89bc883cdc53b1742b834f58267c5a331f253.jpg)

![4c3062ca16dff2fcc57de671f9e9bd95a7592c59dfdacfd7f8294b702914deb7.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/tables/4c3062ca16dff2fcc57de671f9e9bd95a7592c59dfdacfd7f8294b702914deb7.jpg)

![4fc04f14877a775712088d402cd730d8fbada63359d6c650c7b7b3de94fbb233.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/tables/4fc04f14877a775712088d402cd730d8fbada63359d6c650c7b7b3de94fbb233.jpg)

![82caffc0b9c9e04b9377ddfb629ed503777edda061170ee185621d9f4a7637a7.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/tables/82caffc0b9c9e04b9377ddfb629ed503777edda061170ee185621d9f4a7637a7.jpg)

![96d97e5f088602e946df57fc606b40d96123ab8360f911f2a6c69653ab1421ae.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/tables/96d97e5f088602e946df57fc606b40d96123ab8360f911f2a6c69653ab1421ae.jpg)

![9999f8a0831ceb725ec88c236051b6af495b0b41dc37eedcfbf208df1d9450cb.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/tables/9999f8a0831ceb725ec88c236051b6af495b0b41dc37eedcfbf208df1d9450cb.jpg)

![a11043868d234d4ab2f05aadb027c3df07d3b821e39909c976c73738ed19b4c5.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/tables/a11043868d234d4ab2f05aadb027c3df07d3b821e39909c976c73738ed19b4c5.jpg)

![a231f94fda2f75059777708455e5f9e135ce87f07e2b98cc58358ccd54a9812a.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/tables/a231f94fda2f75059777708455e5f9e135ce87f07e2b98cc58358ccd54a9812a.jpg)

![aca7eb0d7167400f389135412f691b66fdb5b9c4388df34d222e558348495cbd.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/tables/aca7eb0d7167400f389135412f691b66fdb5b9c4388df34d222e558348495cbd.jpg)

![ad1963a1933fc095754c72d8fe43831bc73efd7d515a7d9cf20a055eb5139ebe.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/tables/ad1963a1933fc095754c72d8fe43831bc73efd7d515a7d9cf20a055eb5139ebe.jpg)

![bae7bcf9034ce8ef8c67c46e0abceceba4ef52b1354659f812507e83b8be4eae.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/tables/bae7bcf9034ce8ef8c67c46e0abceceba4ef52b1354659f812507e83b8be4eae.jpg)

![cc99cc042cc2823617c7c1c4b7c17005be340e39ccf8e2ca8a652fa0882b4060.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/tables/cc99cc042cc2823617c7c1c4b7c17005be340e39ccf8e2ca8a652fa0882b4060.jpg)

![e1fab1752416334dd743ee3eae50acb8b8ab65ad2aa1c2a4d0e9934999823372.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/tables/e1fab1752416334dd743ee3eae50acb8b8ab65ad2aa1c2a4d0e9934999823372.jpg)

![f40fc343291e1462b1d0d3926bfb85b8066f42cd9ac37454032e222907823d0e.jpg](../icml_results/4629_Plain%20Transformers%20are%20Surprisingly%20Powerful%20Link%20Predictors/tables/f40fc343291e1462b1d0d3926bfb85b8066f42cd9ac37454032e222907823d0e.jpg)

<a id="paper-4583"></a>

## Item Response Scaling Laws: A Measurement Theory Approach for Efficient and Generalizable Neural Scaling Estimation


### Images

![022d01dcbd9c0d3efb2876b46e05bd8c755ad4aaf230d48882d4391fb00e2e0e.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/022d01dcbd9c0d3efb2876b46e05bd8c755ad4aaf230d48882d4391fb00e2e0e.jpg)

![036f8c28884686661d7070dd238a2f1635ac0cdebeedd0a130aca2deb478c743.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/036f8c28884686661d7070dd238a2f1635ac0cdebeedd0a130aca2deb478c743.jpg)

![08465402f4e2d1eeb0e17057ba98aaca72ec10ad5fa7f7654a73aaf02f0d4a63.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/08465402f4e2d1eeb0e17057ba98aaca72ec10ad5fa7f7654a73aaf02f0d4a63.jpg)

![097dfac0faae9cafcee175d7f04310aea5a0be99694f64170e9cc5367f9a1a72.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/097dfac0faae9cafcee175d7f04310aea5a0be99694f64170e9cc5367f9a1a72.jpg)

![11248a1479246cdd73823d1fbb1b0cec58b298997f8450d061f6cbaba260944e.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/11248a1479246cdd73823d1fbb1b0cec58b298997f8450d061f6cbaba260944e.jpg)

![4a24e88b103c0ccdf12231836c7884805f97153e3cfea9b8af2267effb4eaf46.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/4a24e88b103c0ccdf12231836c7884805f97153e3cfea9b8af2267effb4eaf46.jpg)

![4aa3c50abb78670191bf114a94273bf921eecdab7dfee1c4fda32fd11cadc76b.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/4aa3c50abb78670191bf114a94273bf921eecdab7dfee1c4fda32fd11cadc76b.jpg)

![52c05078f16fe5875dbd6ead8a80a6ebbdcd3d9204f462e3ed24b0490ec3f279.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/52c05078f16fe5875dbd6ead8a80a6ebbdcd3d9204f462e3ed24b0490ec3f279.jpg)

![562442218859191d9e370470e10b14be91b9d39caf0df2af6e141b4ecdff4532.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/562442218859191d9e370470e10b14be91b9d39caf0df2af6e141b4ecdff4532.jpg)

![60f1cf92613ccdcf1d476eec3006b668732825698d58906c834fdfaafd6d1da2.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/60f1cf92613ccdcf1d476eec3006b668732825698d58906c834fdfaafd6d1da2.jpg)

![66c35f38e2e4b9dccb73415a272fd1d70f1e1e4d790ddf20a1b4cf57b5db86a3.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/66c35f38e2e4b9dccb73415a272fd1d70f1e1e4d790ddf20a1b4cf57b5db86a3.jpg)

![6cac744b9068db539f5e86e3acb4cb960ce143658edd39afae82c2619c786598.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/6cac744b9068db539f5e86e3acb4cb960ce143658edd39afae82c2619c786598.jpg)

![719acfed87bb8dc04e809d1df1f381925866ccf6bfc0358b0336ed8f5f2c284e.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/719acfed87bb8dc04e809d1df1f381925866ccf6bfc0358b0336ed8f5f2c284e.jpg)

![7302beba4d40fe61614b7056f8443e937d132a3edd88c7a2b2e4a92650071eba.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/7302beba4d40fe61614b7056f8443e937d132a3edd88c7a2b2e4a92650071eba.jpg)

![860ebe965d31356f3b59cde2b98e64bc5cc44d96e30688a6708f06454ac812a8.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/860ebe965d31356f3b59cde2b98e64bc5cc44d96e30688a6708f06454ac812a8.jpg)

![9af51952239864d150a3d84e1018f31f1909e4c2efe716cfa3ceffa3c76f06f9.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/9af51952239864d150a3d84e1018f31f1909e4c2efe716cfa3ceffa3c76f06f9.jpg)

![a300c861d688dadaf9dd6ed72c7ffa348dab43673c9774142b007c8379c93dcf.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/a300c861d688dadaf9dd6ed72c7ffa348dab43673c9774142b007c8379c93dcf.jpg)

![a3d9ae21c648819972fad77710eeab8c15d0e6f27786db8c3711dda090925ef4.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/a3d9ae21c648819972fad77710eeab8c15d0e6f27786db8c3711dda090925ef4.jpg)

![a554825162f0ad5d5e69a8fe39e67721abb02d4992b65208fb52bb1fbed770b5.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/a554825162f0ad5d5e69a8fe39e67721abb02d4992b65208fb52bb1fbed770b5.jpg)

![af13608a4c98ec5b69a4dc782d5c0bd0ba28f56334f6fd8a474757179c0ce421.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/af13608a4c98ec5b69a4dc782d5c0bd0ba28f56334f6fd8a474757179c0ce421.jpg)

![b1e19926fe8d18e6081ab2036c5ff4ab057d75b5c26a225ac224137c3e2d67fd.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/b1e19926fe8d18e6081ab2036c5ff4ab057d75b5c26a225ac224137c3e2d67fd.jpg)

![c7b4831463935b562d43214d23bddf3e3f2686be1ae05243c8193dd4632feee3.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/c7b4831463935b562d43214d23bddf3e3f2686be1ae05243c8193dd4632feee3.jpg)

![e7fef226ff69a8eaadf537505101df6c36b148b8ca03db9b70a2b16e2f6cfa57.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/e7fef226ff69a8eaadf537505101df6c36b148b8ca03db9b70a2b16e2f6cfa57.jpg)

![f65fabf3e0dd6a7d56dd3add959111d016d73fedb1e458328724c8e21f3b7f68.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/f65fabf3e0dd6a7d56dd3add959111d016d73fedb1e458328724c8e21f3b7f68.jpg)

![f67eba5b428ca80c09dc6c21ce001ba8bc11ad667413a2b1ab85379cedc69e77.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/f67eba5b428ca80c09dc6c21ce001ba8bc11ad667413a2b1ab85379cedc69e77.jpg)

![f70a62bc5093b5f808f836f1f42ef6590dd60ac9c0ba02d8b86665b5b287e7cb.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/images/f70a62bc5093b5f808f836f1f42ef6590dd60ac9c0ba02d8b86665b5b287e7cb.jpg)

### Tables

![e9b25f80456d46041b094a3cca9256f737846a7a353a8f3e76e390c4d2e5fbf5.jpg](../icml_results/4630_Item%20Response%20Scaling%20Laws_%20A%20Measurement%20Theory%20Approach%20for%20Efficient%20and%20Generalizable%20Neural%20Sca/tables/e9b25f80456d46041b094a3cca9256f737846a7a353a8f3e76e390c4d2e5fbf5.jpg)

<a id="paper-4584"></a>

## Creat3r: Confidence Reaggregation for Exploration-aware Active 3D Reconstruction


### Images

![1370f6ec9b2a2532f5f13e0323da22eb42e5e44aa141a83ddf7815f9760f27cd.jpg](../icml_results/4631_Creat3r_%20Confidence%20Reaggregation%20for%20Exploration-aware%20Active%203D%20Reconstruction/images/1370f6ec9b2a2532f5f13e0323da22eb42e5e44aa141a83ddf7815f9760f27cd.jpg)

![343b0b9080e9afdb2f1b40c7ec27da0fdf42f483211c41ce4fa5560a3b73672d.jpg](../icml_results/4631_Creat3r_%20Confidence%20Reaggregation%20for%20Exploration-aware%20Active%203D%20Reconstruction/images/343b0b9080e9afdb2f1b40c7ec27da0fdf42f483211c41ce4fa5560a3b73672d.jpg)

![436eabcf4f8786deab817957adb9e6ed2deb18b742ea20e1081e1c6d9d5457f2.jpg](../icml_results/4631_Creat3r_%20Confidence%20Reaggregation%20for%20Exploration-aware%20Active%203D%20Reconstruction/images/436eabcf4f8786deab817957adb9e6ed2deb18b742ea20e1081e1c6d9d5457f2.jpg)

![b6a075e3afe804e51cd32bdf535c199da56ccc248506c0018b0bbf7ce820e48a.jpg](../icml_results/4631_Creat3r_%20Confidence%20Reaggregation%20for%20Exploration-aware%20Active%203D%20Reconstruction/images/b6a075e3afe804e51cd32bdf535c199da56ccc248506c0018b0bbf7ce820e48a.jpg)

![c7300a98a26183b6f9bd37491f0783feaa73f065b90256105728257e70a96650.jpg](../icml_results/4631_Creat3r_%20Confidence%20Reaggregation%20for%20Exploration-aware%20Active%203D%20Reconstruction/images/c7300a98a26183b6f9bd37491f0783feaa73f065b90256105728257e70a96650.jpg)

![ce4f00da92d75956bc0f470f41262d64dd6d535bab3eda86242dbe0366b57f9c.jpg](../icml_results/4631_Creat3r_%20Confidence%20Reaggregation%20for%20Exploration-aware%20Active%203D%20Reconstruction/images/ce4f00da92d75956bc0f470f41262d64dd6d535bab3eda86242dbe0366b57f9c.jpg)

![fc0eb280605baafd504e4b1053e32b7ffaab3c465317fccc23d1aadddc7234e7.jpg](../icml_results/4631_Creat3r_%20Confidence%20Reaggregation%20for%20Exploration-aware%20Active%203D%20Reconstruction/images/fc0eb280605baafd504e4b1053e32b7ffaab3c465317fccc23d1aadddc7234e7.jpg)

### Tables

![06b5b17dcc48789485fb3e2d72fdb4909010240c23dc3aa39d129f3311564164.jpg](../icml_results/4631_Creat3r_%20Confidence%20Reaggregation%20for%20Exploration-aware%20Active%203D%20Reconstruction/tables/06b5b17dcc48789485fb3e2d72fdb4909010240c23dc3aa39d129f3311564164.jpg)

![2db852eb763e790ea9c0b8fd18b1f45b7323cbb58e606a857c81edb70ab1fd54.jpg](../icml_results/4631_Creat3r_%20Confidence%20Reaggregation%20for%20Exploration-aware%20Active%203D%20Reconstruction/tables/2db852eb763e790ea9c0b8fd18b1f45b7323cbb58e606a857c81edb70ab1fd54.jpg)

![47e0e78722a936e2c4ac2b2d98f7e0bf8e37eac5efa71afc6c421c5e5ac36e73.jpg](../icml_results/4631_Creat3r_%20Confidence%20Reaggregation%20for%20Exploration-aware%20Active%203D%20Reconstruction/tables/47e0e78722a936e2c4ac2b2d98f7e0bf8e37eac5efa71afc6c421c5e5ac36e73.jpg)

![765e6908486e901f5caa06932f04cb68c92807cafc72c162826eb8ca62b11140.jpg](../icml_results/4631_Creat3r_%20Confidence%20Reaggregation%20for%20Exploration-aware%20Active%203D%20Reconstruction/tables/765e6908486e901f5caa06932f04cb68c92807cafc72c162826eb8ca62b11140.jpg)

![84e785915b7bdddf6beecc570259710d48eb887f7dd975c11ccd849263db303e.jpg](../icml_results/4631_Creat3r_%20Confidence%20Reaggregation%20for%20Exploration-aware%20Active%203D%20Reconstruction/tables/84e785915b7bdddf6beecc570259710d48eb887f7dd975c11ccd849263db303e.jpg)

![995a0a2a63fdbe11e418e3ec2e0665c35c01c15867e48886b3bf89d629868b95.jpg](../icml_results/4631_Creat3r_%20Confidence%20Reaggregation%20for%20Exploration-aware%20Active%203D%20Reconstruction/tables/995a0a2a63fdbe11e418e3ec2e0665c35c01c15867e48886b3bf89d629868b95.jpg)

![a1766c17cf37fb089857ea4c755c9d09dc0b1438bf35669df6142609f4c4001a.jpg](../icml_results/4631_Creat3r_%20Confidence%20Reaggregation%20for%20Exploration-aware%20Active%203D%20Reconstruction/tables/a1766c17cf37fb089857ea4c755c9d09dc0b1438bf35669df6142609f4c4001a.jpg)

![cf2eac4ad20ebadffcee3093be430b0af873eabffc2f024a88a3e3dd29b386ee.jpg](../icml_results/4631_Creat3r_%20Confidence%20Reaggregation%20for%20Exploration-aware%20Active%203D%20Reconstruction/tables/cf2eac4ad20ebadffcee3093be430b0af873eabffc2f024a88a3e3dd29b386ee.jpg)

![f004a5407241be047930a068bdc6b65a0c831d32d13e5b9b21c9ec7305136a70.jpg](../icml_results/4631_Creat3r_%20Confidence%20Reaggregation%20for%20Exploration-aware%20Active%203D%20Reconstruction/tables/f004a5407241be047930a068bdc6b65a0c831d32d13e5b9b21c9ec7305136a70.jpg)

<a id="paper-4585"></a>

## $L^3$: Large Lookup Layers


### Images

![0fe7650b124a40d4fb435834068a46551750bdf7b1993cca6b09c4f978bc98ef.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/images/0fe7650b124a40d4fb435834068a46551750bdf7b1993cca6b09c4f978bc98ef.jpg)

![41bb1c491f1a6066bed386870383ea3f2c97e356a03dbeb35f8e9758087f25a4.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/images/41bb1c491f1a6066bed386870383ea3f2c97e356a03dbeb35f8e9758087f25a4.jpg)

![4723b5af9e92d783300f45613d69287f44c6f0e65e3c9614655d81ed5b3561cf.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/images/4723b5af9e92d783300f45613d69287f44c6f0e65e3c9614655d81ed5b3561cf.jpg)

![5dcab27473e851d388c2ca5e8a1fbf723f6b1425fb833ceffe3c1cc7edc3935b.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/images/5dcab27473e851d388c2ca5e8a1fbf723f6b1425fb833ceffe3c1cc7edc3935b.jpg)

![7dfa29141d973fd898221e248b4b9ca5d0a19b5714ab8460cfd75def6d27b893.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/images/7dfa29141d973fd898221e248b4b9ca5d0a19b5714ab8460cfd75def6d27b893.jpg)

![8e6654a4180a06d4d34a81e9a1bb65dcba2b36b83d344966d8b5bb7318b0c57f.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/images/8e6654a4180a06d4d34a81e9a1bb65dcba2b36b83d344966d8b5bb7318b0c57f.jpg)

![9c4292b270fe0d6f65e9f2f50f6d6714ae33948c1e83c3fc28028aa4b9f81b56.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/images/9c4292b270fe0d6f65e9f2f50f6d6714ae33948c1e83c3fc28028aa4b9f81b56.jpg)

![a37ca3f47fd36e2d72a9f156059bf94d94a83a653b0d4dc6b96c90fc02bce828.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/images/a37ca3f47fd36e2d72a9f156059bf94d94a83a653b0d4dc6b96c90fc02bce828.jpg)

![b11ecef57eff5bf190aac11c61c40b95e18f0176d2fdfe29480d6e39d471a948.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/images/b11ecef57eff5bf190aac11c61c40b95e18f0176d2fdfe29480d6e39d471a948.jpg)

![b7375ad637181dbcea85f14eb622cc53fa8de1d7882e8cdfe17f3a0769499713.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/images/b7375ad637181dbcea85f14eb622cc53fa8de1d7882e8cdfe17f3a0769499713.jpg)

![bf77e3d15a18fd7717704218a41c097850c4ba3a762b368d6dfba91dcdfb4446.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/images/bf77e3d15a18fd7717704218a41c097850c4ba3a762b368d6dfba91dcdfb4446.jpg)

![ff8047d00586c01cecf33f29ab7f94e22e2b0ff0b788b2efa04ad97aa4845289.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/images/ff8047d00586c01cecf33f29ab7f94e22e2b0ff0b788b2efa04ad97aa4845289.jpg)

### Tables

![0dab6657078e1f230c5efe85ff6565bf1c7dc923519f9c050aeedc4eea243c14.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/tables/0dab6657078e1f230c5efe85ff6565bf1c7dc923519f9c050aeedc4eea243c14.jpg)

![19d8f1513d8300c100f88b5845d471ee459c4a2a0acc57215d9caf1dd7290e06.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/tables/19d8f1513d8300c100f88b5845d471ee459c4a2a0acc57215d9caf1dd7290e06.jpg)

![4841d731f00fd1f91cf769e881426e85957541ca0be5bf31fd71634a2b83af7c.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/tables/4841d731f00fd1f91cf769e881426e85957541ca0be5bf31fd71634a2b83af7c.jpg)

![722e3d36e05faffcb33928ed9b38edb26223e67fa57d13118bbe6b417a9747ab.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/tables/722e3d36e05faffcb33928ed9b38edb26223e67fa57d13118bbe6b417a9747ab.jpg)

![acd3d2d7bfadba98b199cddb159948f080f6c77eda39d8c9b9fdad391a0cfb2c.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/tables/acd3d2d7bfadba98b199cddb159948f080f6c77eda39d8c9b9fdad391a0cfb2c.jpg)

![c6e8064970783bf2500b0282b83c37edda40ffa8dd383154c8661b8e83cde748.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/tables/c6e8064970783bf2500b0282b83c37edda40ffa8dd383154c8661b8e83cde748.jpg)

![dce23b3be190751908bfa5fd0258e4d492bf4359844a0820ee84ac82e037c916.jpg](../icml_results/4632_%24L%5E3%24_%20Large%20Lookup%20Layers/tables/dce23b3be190751908bfa5fd0258e4d492bf4359844a0820ee84ac82e037c916.jpg)

<a id="paper-4586"></a>

## Model-Preserving Adaptive Rounding


### Images

![9041c8b1e9dce56175c7b216fd64fc25b5d54db51d1d425a33c694fc38fb51ee.jpg](../icml_results/4633_Model-Preserving%20Adaptive%20Rounding/images/9041c8b1e9dce56175c7b216fd64fc25b5d54db51d1d425a33c694fc38fb51ee.jpg)

![9fa1618199561267d78160b76f3d024409174e99b7f88568586bd1c3fe35d699.jpg](../icml_results/4633_Model-Preserving%20Adaptive%20Rounding/images/9fa1618199561267d78160b76f3d024409174e99b7f88568586bd1c3fe35d699.jpg)

![d68c4987ae4e5564578cef9314148459fad3e68ca98633cdca852c04f15eeab5.jpg](../icml_results/4633_Model-Preserving%20Adaptive%20Rounding/images/d68c4987ae4e5564578cef9314148459fad3e68ca98633cdca852c04f15eeab5.jpg)

![ef9cb06d420fc2be2fb1a021134ab7146fe9ebc57c0a8996e4d32caa7fccd67f.jpg](../icml_results/4633_Model-Preserving%20Adaptive%20Rounding/images/ef9cb06d420fc2be2fb1a021134ab7146fe9ebc57c0a8996e4d32caa7fccd67f.jpg)

### Tables

![04564c978b489df0408f073413a334825864225b7aa9c9b186926bbb4d653ace.jpg](../icml_results/4633_Model-Preserving%20Adaptive%20Rounding/tables/04564c978b489df0408f073413a334825864225b7aa9c9b186926bbb4d653ace.jpg)

![1198d13a0721adff50c32daf8edf1c2c4d25f5f7c5c9487627d8c93984c52d3b.jpg](../icml_results/4633_Model-Preserving%20Adaptive%20Rounding/tables/1198d13a0721adff50c32daf8edf1c2c4d25f5f7c5c9487627d8c93984c52d3b.jpg)

![5e223c223d43a7431e6060ee4de3ae63fd97b7b137815023d64dcaf0f6822b5f.jpg](../icml_results/4633_Model-Preserving%20Adaptive%20Rounding/tables/5e223c223d43a7431e6060ee4de3ae63fd97b7b137815023d64dcaf0f6822b5f.jpg)

![923be5b02853518a8b8c8da24a2de22aa963dee370e2b97379fb4f6a97830b59.jpg](../icml_results/4633_Model-Preserving%20Adaptive%20Rounding/tables/923be5b02853518a8b8c8da24a2de22aa963dee370e2b97379fb4f6a97830b59.jpg)

![abd4d1dd6bfa0ee6a9b99a7fe1c18e90031ea8ed9198aede62ba0888aba43137.jpg](../icml_results/4633_Model-Preserving%20Adaptive%20Rounding/tables/abd4d1dd6bfa0ee6a9b99a7fe1c18e90031ea8ed9198aede62ba0888aba43137.jpg)

![ad13f96ae385b4d3669e832db17b3c1bb36a32a20c4f7c2c926b16b2ef6ac372.jpg](../icml_results/4633_Model-Preserving%20Adaptive%20Rounding/tables/ad13f96ae385b4d3669e832db17b3c1bb36a32a20c4f7c2c926b16b2ef6ac372.jpg)

![b6b4b2d23aaff9805ffe781bde08a3e56176936c62c6a198ccac1142e4b243df.jpg](../icml_results/4633_Model-Preserving%20Adaptive%20Rounding/tables/b6b4b2d23aaff9805ffe781bde08a3e56176936c62c6a198ccac1142e4b243df.jpg)

![cbb1c400c0132c19e7e5934ba310fc6218e62b8eaf6ed80bfa1afaad4c26695e.jpg](../icml_results/4633_Model-Preserving%20Adaptive%20Rounding/tables/cbb1c400c0132c19e7e5934ba310fc6218e62b8eaf6ed80bfa1afaad4c26695e.jpg)

![cd49e9de6f7f043063076fd01febe14e78bdda54bd084fd74f80f4be0ddcfd7e.jpg](../icml_results/4633_Model-Preserving%20Adaptive%20Rounding/tables/cd49e9de6f7f043063076fd01febe14e78bdda54bd084fd74f80f4be0ddcfd7e.jpg)

<a id="paper-4587"></a>

## Protein Circuit Tracing via Cross-layer Transcoders

### Images

![0670715f5a510f0b94e26b7b0daf23ea6d46d0c6b7c95bef57b99860fd4ef9c1.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/0670715f5a510f0b94e26b7b0daf23ea6d46d0c6b7c95bef57b99860fd4ef9c1.jpg)

![0bc23acdaf3bffe21758dc1c9a8245bc87588123b060a556776af7c5828a992a.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/0bc23acdaf3bffe21758dc1c9a8245bc87588123b060a556776af7c5828a992a.jpg)

![0ed5560589b0ddd08863ef6ffae8d081f606bb3c7935942293a41f55b2b78db4.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/0ed5560589b0ddd08863ef6ffae8d081f606bb3c7935942293a41f55b2b78db4.jpg)

![142c3b22f6630bb9a231897cd7e395f0ac7f5f007832f17f7c915636ecec3286.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/142c3b22f6630bb9a231897cd7e395f0ac7f5f007832f17f7c915636ecec3286.jpg)

![231744735515e97230d185d0ac544e5e5fb520142fb545569f00ea4526ecf913.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/231744735515e97230d185d0ac544e5e5fb520142fb545569f00ea4526ecf913.jpg)

![26ce346038ea9ead68d9ff63941511e7b548c6e32fbf71eccd6969d2cd9e09b0.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/26ce346038ea9ead68d9ff63941511e7b548c6e32fbf71eccd6969d2cd9e09b0.jpg)

![2fa5342da47b5b731e7d90b3de1923c649b9f3a01f10ad5393ac6e6579bc11b2.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/2fa5342da47b5b731e7d90b3de1923c649b9f3a01f10ad5393ac6e6579bc11b2.jpg)

![3c0fbb8dad746e2f1ad171d9f7e09a2ea6d648f712d5a61e50b64f8659bd5fa3.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/3c0fbb8dad746e2f1ad171d9f7e09a2ea6d648f712d5a61e50b64f8659bd5fa3.jpg)

![42af4ad6aa9fdc9f2a1c0fc7a30e785510b831d70bb46b687766b0d5a35eb956.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/42af4ad6aa9fdc9f2a1c0fc7a30e785510b831d70bb46b687766b0d5a35eb956.jpg)

![44553a95bfa4d8233a3405ef4d29985398513bbaa0f5ef723ea8c1fe4e11c90a.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/44553a95bfa4d8233a3405ef4d29985398513bbaa0f5ef723ea8c1fe4e11c90a.jpg)

![6f6f84e201e2381d4dbf6423174e7eea5495ec08f5c68f0d14a185fa11e250e3.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/6f6f84e201e2381d4dbf6423174e7eea5495ec08f5c68f0d14a185fa11e250e3.jpg)

![728816faf85836625d584b959ea901a673a95bb5ed378109f7f091a73d82ac1c.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/728816faf85836625d584b959ea901a673a95bb5ed378109f7f091a73d82ac1c.jpg)

![888ecb8f72a86524382c311771f50ed87bed1b4046f089e1298f00a083feaa90.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/888ecb8f72a86524382c311771f50ed87bed1b4046f089e1298f00a083feaa90.jpg)

![976b7dd7db454a56d8519ad8a97b7ed500b14efbfe41f336d5a50e217a1bf3c4.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/976b7dd7db454a56d8519ad8a97b7ed500b14efbfe41f336d5a50e217a1bf3c4.jpg)

![a321ba489245a642ca59076e53ee1e61ca9b1f354bab697d3cd2a4186887ee3c.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/a321ba489245a642ca59076e53ee1e61ca9b1f354bab697d3cd2a4186887ee3c.jpg)

![cdb6065df10e77fe44f560f7d481dbc3f7af65fb51e0b4f8e32865362847f000.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/cdb6065df10e77fe44f560f7d481dbc3f7af65fb51e0b4f8e32865362847f000.jpg)

![cf402613c958d03d23dfc4d8d23fa85ad4c23375f8c789b7aa27f302613b0834.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/cf402613c958d03d23dfc4d8d23fa85ad4c23375f8c789b7aa27f302613b0834.jpg)

![d7e95d70e780c7d201c0962d1bb139965568dafd81e4386cb633fc82e4b0bd98.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/d7e95d70e780c7d201c0962d1bb139965568dafd81e4386cb633fc82e4b0bd98.jpg)

### Tables

![15f8ccd38c7c18b7d8895de196a130864c9baf12722715e627fe39722e8045e3.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/15f8ccd38c7c18b7d8895de196a130864c9baf12722715e627fe39722e8045e3.jpg)

![2d7b06918c3ab3d31e0e61901114b93459458a5a18dc3ba536f8f05b9d0f1c29.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/2d7b06918c3ab3d31e0e61901114b93459458a5a18dc3ba536f8f05b9d0f1c29.jpg)

![3eca65b7a92975729bc824c7e0b743be84628059c6d605041383c19103516ba5.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/3eca65b7a92975729bc824c7e0b743be84628059c6d605041383c19103516ba5.jpg)

![802ceaad152fba5b089eacd48e46d5580e26d27cf8a1c416ffaaf346a95bbb88.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/802ceaad152fba5b089eacd48e46d5580e26d27cf8a1c416ffaaf346a95bbb88.jpg)

![83cb89d419a369a462fba9b62cfadac55d5de4588920be5e94e573aa3315daea.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/83cb89d419a369a462fba9b62cfadac55d5de4588920be5e94e573aa3315daea.jpg)

![8777fb9fd0fe863f73fbfa8a3180bd3fb44eef885bc14618c553e657429d1f1e.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/8777fb9fd0fe863f73fbfa8a3180bd3fb44eef885bc14618c553e657429d1f1e.jpg)

![9a5c22af11b813c1055baa452913e8e095b5181981ef34d8f1ba06a8025a4e84.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/9a5c22af11b813c1055baa452913e8e095b5181981ef34d8f1ba06a8025a4e84.jpg)

![a839af2b474953ed50a654251a7fee786c5328fc68222047061487d1bec92c11.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/a839af2b474953ed50a654251a7fee786c5328fc68222047061487d1bec92c11.jpg)

![ac8733cd1d641fe4f432b6d22faaa7b2f883c56185dd263e5314c7375e79eddb.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/ac8733cd1d641fe4f432b6d22faaa7b2f883c56185dd263e5314c7375e79eddb.jpg)

![e3eb3d17ddac136d0f8b3ec9c13a03c435b952af4feab9ddd432b48dd0b925f8.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/e3eb3d17ddac136d0f8b3ec9c13a03c435b952af4feab9ddd432b48dd0b925f8.jpg)
