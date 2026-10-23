# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 158 of 197

## 目录 (Table of Contents)

1. [Offline Multi-agent Continual Cooperation via Skill Partition and Reuse](#paper-5182)
2. [MACD: Model-Aware Contrastive Decoding via Counterfactual Data for Video-LLMs](#paper-5183)
3. [When Distance Distracts: Representation Distance Bias in BT-Loss for Reward Models](#paper-5184)
4. [Sobolev Regularized Score Difference Estimation in Diffusion Models](#paper-5185)
5. [Cost-aware Stopping for Bayesian Optimization](#paper-5186)
6. [A Narrowing Geometry in Contaminated Reasoning](#paper-5187)
7. [Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding](#paper-5188)
8. [Dissecting the Safety Circuit: Neuronal Intervention for Transferable Adversarial Attacks on VLMs](#paper-5189)
9. [SHERPA: Fine-tuning Segment Anything Models with Task-relevant Guidance](#paper-5190)
10. [Asymptotically Fast Clebsch-Gordan Tensor Products with Vector Spherical Harmonics](#paper-5191)
11. [Self-Supervised Weight Templates for Scalable Vision Model Initialization](#paper-5192)
12. [Rethinking Code Complexity Through the Lens of Large Language Models](#paper-5193)
13. [Clustering as Reasoning: A $k$-Means Interpretation of Chain-of-Thought Graph Learning](#paper-5194)
14. [FedUSD: Unbiased Synthetic Data for Federated Learning](#paper-5195)
15. [Finding the Correct Visual Evidence Without Forgetting: Mitigating Hallucination in LVLMs via Inter-Layer Visual Attention Discrepancy](#paper-5196)
16. [Enhanced Latent-Space Adversarial Training for Super-Resolution](#paper-5197)
17. [TeamTR: Trust-Region Fine-Tuning for Multi-Agent LLM Coordination](#paper-5198)
18. [Hybrid-Gym: Training Coding Agents to Generalize Across Tasks](#paper-5199)
19. [Where Concept Erasure Should Occur: Concept–Layer Alignment in Text-to-Video Diffusion Models](#paper-5200)
20. [BESplit: Bias-Compensated Split Federated Learning with Evidential Aggregation](#paper-5201)
21. [Training-Free Multimodal Large Language Model Orchestration](#paper-5202)
22. [AREA: Attribute Extraction and Aggregation for CLIP-Based Class-Incremental Learning](#paper-5203)
23. [SAME: Stabilized Mixture-of-Experts for Multimodal Continual Instruction Tuning](#paper-5204)
24. [Can LLMs Reason Like Automated Theorem Provers for Rust Verification? VCoT-Bench: Evaluating via Verification Chain of Thought](#paper-5205)
25. [MEMO: Memory-Augmented Model Context Optimization for Robust Multi-Turn Multi-Agent LLM Games](#paper-5206)
26. [FG-CLIP 2: A Bilingual Fine-grained Vision-Language Alignment Model](#paper-5207)
27. [Statistical Early Stopping for Reasoning Models](#paper-5208)
28. [AnyCanvas: Potential Field Guidance for Training-Free Spatial Control in Text-to-Image Diffusion](#paper-5209)
29. [SkillNet: Hierarchical Skill Modeling for Compositional Generalization in Vision-Language Action Models](#paper-5210)
30. [Dynamic Regret via Discounted-to-Dynamic Reduction with Applications to Curved Losses and Adam Optimizer](#paper-5211)
31. [Probing How Scalable Table Data Enhances General Long-Context Reasoning](#paper-5212)
32. [Instruction Decomposition and Action Alignment for Vision-and-Language Navigation](#paper-5213)
33. [FedVeer: Self-Adaptive Skew Estimation for Robust Federated Learning](#paper-5214)

---


<a id="paper-5182"></a>

## Offline Multi-agent Continual Cooperation via Skill Partition and Reuse

### Images

![2647b59cbf7a1623a7e80a254feedd8257fd2c8afc840cd6d5daa720a535ca9d.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/images/2647b59cbf7a1623a7e80a254feedd8257fd2c8afc840cd6d5daa720a535ca9d.jpg)

![29295f0d4e451e7f2c1b2ad95c796a88f6ceb8a78dac63fbb2425bf2872d8ac0.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/images/29295f0d4e451e7f2c1b2ad95c796a88f6ceb8a78dac63fbb2425bf2872d8ac0.jpg)

![308083b435975649cda25e7c421f29f7d91cc4b9763693d013a575de37dfe344.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/images/308083b435975649cda25e7c421f29f7d91cc4b9763693d013a575de37dfe344.jpg)

![3e592019df1b2eb83dd0ca1edcf9a302a2010ff1d56c6e8494030e13a539db6e.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/images/3e592019df1b2eb83dd0ca1edcf9a302a2010ff1d56c6e8494030e13a539db6e.jpg)

![570065f0c13b3a43e0bc79efb316df83739853ed1c2fd30fadc3a2020f679d82.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/images/570065f0c13b3a43e0bc79efb316df83739853ed1c2fd30fadc3a2020f679d82.jpg)

![b4aa0f621c44eb481a872379512cdc325b4f5c75e362387f0583efee0947fe7e.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/images/b4aa0f621c44eb481a872379512cdc325b4f5c75e362387f0583efee0947fe7e.jpg)

![eeb2e3d76158e2f7becc215a4145c750ce6758eb4696788b4353a43a40aaa04f.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/images/eeb2e3d76158e2f7becc215a4145c750ce6758eb4696788b4353a43a40aaa04f.jpg)

### Tables

![2913f8ec719414fab3c533c44f1ccd774d82ec78d74ebbe28c94f25f6b3dfd22.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/tables/2913f8ec719414fab3c533c44f1ccd774d82ec78d74ebbe28c94f25f6b3dfd22.jpg)

![29aa4f3572f91bc8ea522a00f58d8b01b99a63f188088bdde23626cd410f85ce.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/tables/29aa4f3572f91bc8ea522a00f58d8b01b99a63f188088bdde23626cd410f85ce.jpg)

![2e138a9a17e070f07dc537e4440fa2d59696c4e9fd81ddc341f2c968992ad4bd.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/tables/2e138a9a17e070f07dc537e4440fa2d59696c4e9fd81ddc341f2c968992ad4bd.jpg)

![61c61c262ca281f40982ba1acf36d2c5842f040f33b63cf66d6eb48df459230a.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/tables/61c61c262ca281f40982ba1acf36d2c5842f040f33b63cf66d6eb48df459230a.jpg)

![62230c131a060be8188375eb0a10eb5a7d669a11675eedc85d09dacac52c89d4.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/tables/62230c131a060be8188375eb0a10eb5a7d669a11675eedc85d09dacac52c89d4.jpg)

![700339d17e20f1de90e1b1b3b98ce9f15bc643ba91620d53032342b690ce4e61.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/tables/700339d17e20f1de90e1b1b3b98ce9f15bc643ba91620d53032342b690ce4e61.jpg)

![7f53be45eabe268e317ca4d3620a4997ce3236c466450f87fdf2cd2ab135d88c.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/tables/7f53be45eabe268e317ca4d3620a4997ce3236c466450f87fdf2cd2ab135d88c.jpg)

![c40953857b468408a0ea89a9691768f04abf121e542ba62df7bb6ed971f73a50.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/tables/c40953857b468408a0ea89a9691768f04abf121e542ba62df7bb6ed971f73a50.jpg)

## Offline Multi-agent Continual Cooperation via Skill Partition and Reuse


### Images

![37b9f591a63e5566076f7c28da3e5e68a88dd4a95a4251db293a4559cd790d0b.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/images/37b9f591a63e5566076f7c28da3e5e68a88dd4a95a4251db293a4559cd790d0b.jpg)

![398d1b23da938538e2a49d080de076462ce3dbd9ddb6ec5aceda5837a43dcf5e.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/images/398d1b23da938538e2a49d080de076462ce3dbd9ddb6ec5aceda5837a43dcf5e.jpg)

![647128d0a2ea1c43860a35bda89961f57056205c096931baab3b617582e4757c.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/images/647128d0a2ea1c43860a35bda89961f57056205c096931baab3b617582e4757c.jpg)

![71bef8d2d580aaa7934ad5c1c0b9e9c598d63d6bc2a45023e6b566bfaccc4cbd.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/images/71bef8d2d580aaa7934ad5c1c0b9e9c598d63d6bc2a45023e6b566bfaccc4cbd.jpg)

![779c5835b35bf38d820f803e3ebee4bf982cadc9d8539eb8efbb2bed80bc64ca.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/images/779c5835b35bf38d820f803e3ebee4bf982cadc9d8539eb8efbb2bed80bc64ca.jpg)

![7b61024b26aa6e174e953169ab2b557f9dc9ecc8f47548cb9042cedceb8e5004.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/images/7b61024b26aa6e174e953169ab2b557f9dc9ecc8f47548cb9042cedceb8e5004.jpg)

![a1e9de602ad532b877281d6c998b6f17f2ef31e8bce7606fbb3b3567127984ff.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/images/a1e9de602ad532b877281d6c998b6f17f2ef31e8bce7606fbb3b3567127984ff.jpg)

![e3edb94ceeb0bbe21446ae20908d58f8843c0dc9449e253e07c02a6fa84a55c5.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/images/e3edb94ceeb0bbe21446ae20908d58f8843c0dc9449e253e07c02a6fa84a55c5.jpg)

![e5e3c141257bf467215f9ddff0e885e68a5730aeb5e3f103ae2278488a3219db.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/images/e5e3c141257bf467215f9ddff0e885e68a5730aeb5e3f103ae2278488a3219db.jpg)

![ee5922d44189977b136ae760fd9db559391abf692c93b3a2048cb53c0684953f.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/images/ee5922d44189977b136ae760fd9db559391abf692c93b3a2048cb53c0684953f.jpg)

![f1df0f24b00d1fed0fafa620299485db540f60561018662ca51de1d4048238ef.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/images/f1df0f24b00d1fed0fafa620299485db540f60561018662ca51de1d4048238ef.jpg)

![fe92d3d06fc4811d39bf46f18581b86a9030b62cfc8031ef96f19c9c7ac6eb08.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/images/fe92d3d06fc4811d39bf46f18581b86a9030b62cfc8031ef96f19c9c7ac6eb08.jpg)

### Tables

![014ee045a353356e70eba3be6f9974f10aaa9705b72f8cfa3f396a784d3254f0.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/tables/014ee045a353356e70eba3be6f9974f10aaa9705b72f8cfa3f396a784d3254f0.jpg)

![01ce097b760e043e4d1dc5adb13759882ed1241e5edae15a39bb17b2e96dead2.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/tables/01ce097b760e043e4d1dc5adb13759882ed1241e5edae15a39bb17b2e96dead2.jpg)

![0968248c915727471c50024ed54e90aa0c04a5dbe45802b86a2386ed4e114c7a.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/tables/0968248c915727471c50024ed54e90aa0c04a5dbe45802b86a2386ed4e114c7a.jpg)

![169acbfd36bf6095fabb68159c6d20c098cee753ab2d22f7c37fde045fb19826.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/tables/169acbfd36bf6095fabb68159c6d20c098cee753ab2d22f7c37fde045fb19826.jpg)

![336c965698f593e35147fb3f23f6fa1d94c260b62888497c46b39d8b92c391c5.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/tables/336c965698f593e35147fb3f23f6fa1d94c260b62888497c46b39d8b92c391c5.jpg)

![4c572462750dccf912e201d3f9358d957be3e0c6aa3d037f7e7c0359c26c9086.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/tables/4c572462750dccf912e201d3f9358d957be3e0c6aa3d037f7e7c0359c26c9086.jpg)

![7f1ff53ba02bc9ab10b72cfd79aeacce1daf8758597354637acbfb117b1c89cd.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/tables/7f1ff53ba02bc9ab10b72cfd79aeacce1daf8758597354637acbfb117b1c89cd.jpg)

![97cd120621ce5068ac3110507e20c5aca67f30132f5b788cc1c53faefd2bd0ab.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/tables/97cd120621ce5068ac3110507e20c5aca67f30132f5b788cc1c53faefd2bd0ab.jpg)

![b41dc2a156dfaaff2f5b7e63d91bbc8986337ae99e6ca5be8861918228f0dbba.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/tables/b41dc2a156dfaaff2f5b7e63d91bbc8986337ae99e6ca5be8861918228f0dbba.jpg)

![b4eea19563422240df2916f652408d2879b704eaa8173f0578b79ac5f91d8ac1.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/tables/b4eea19563422240df2916f652408d2879b704eaa8173f0578b79ac5f91d8ac1.jpg)

![ccca885cea4c43f007fde48c5168ab81ea4f6e3de666daa137e57a092cbecf2c.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/tables/ccca885cea4c43f007fde48c5168ab81ea4f6e3de666daa137e57a092cbecf2c.jpg)

![e47ee285a20bc69d8402c1d1741204348ee7fed6fcd2814f96af8c614bb1f1ab.jpg](../icml_results/5230_Offline%20Multi-agent%20Continual%20Cooperation%20via%20Skill%20Partition%20and%20Reuse/tables/e47ee285a20bc69d8402c1d1741204348ee7fed6fcd2814f96af8c614bb1f1ab.jpg)

<a id="paper-5183"></a>

## MACD: Model-Aware Contrastive Decoding via Counterfactual Data for Video-LLMs


### Images

![152b74838028f30b9969abbf8c3cd64b884fddd87ea6ea79faa2ddf57f46ed3d.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/images/152b74838028f30b9969abbf8c3cd64b884fddd87ea6ea79faa2ddf57f46ed3d.jpg)

### Tables

![1cc32b23672009f84961b9b2080abeac33fbf5891c0431afa8fe61f50e8659ca.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/tables/1cc32b23672009f84961b9b2080abeac33fbf5891c0431afa8fe61f50e8659ca.jpg)

![2a8bc7596eeabec0726820e42e69772d525e7d4bb35043b32679f577bf92cdc3.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/tables/2a8bc7596eeabec0726820e42e69772d525e7d4bb35043b32679f577bf92cdc3.jpg)

![37c9888da403fedafae2c8af4f0ab5ba7c7737441b0a62c209ae8252b3133f07.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/tables/37c9888da403fedafae2c8af4f0ab5ba7c7737441b0a62c209ae8252b3133f07.jpg)

![4e02bf47730c6213e5072ebba5688bb8ab597bbf69785e04b8106db00005b613.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/tables/4e02bf47730c6213e5072ebba5688bb8ab597bbf69785e04b8106db00005b613.jpg)

![5db7e4a29ba9e555e55e95bea61ba8960369523497b7d4812a0766d94eb724af.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/tables/5db7e4a29ba9e555e55e95bea61ba8960369523497b7d4812a0766d94eb724af.jpg)

![6f6ef6dca20036c3e962f4b045221e788051874261fef9276498b0f354682a7e.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/tables/6f6ef6dca20036c3e962f4b045221e788051874261fef9276498b0f354682a7e.jpg)

![747a3bf87b008303698a319f1b56a45a4692d0682d9a3af7052f33d0eeba385d.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/tables/747a3bf87b008303698a319f1b56a45a4692d0682d9a3af7052f33d0eeba385d.jpg)

![87d34a340d83471cf7f8a03b8e8d75e25c6a1577ef7628f2bef4f293e6eecde4.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/tables/87d34a340d83471cf7f8a03b8e8d75e25c6a1577ef7628f2bef4f293e6eecde4.jpg)

![a3782330f4080a6cab8892c00696c9ef5aed35ff742f6df916fe8794061aa81f.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/tables/a3782330f4080a6cab8892c00696c9ef5aed35ff742f6df916fe8794061aa81f.jpg)

![a9c927950d2effcc193ce23b8cba0088a51f4d709c88d45ce005b60520113dc9.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/tables/a9c927950d2effcc193ce23b8cba0088a51f4d709c88d45ce005b60520113dc9.jpg)

![aa173a96bba3c0c8ca4615ae564b4e18dd7e859d2b8a69132cd38f02f0431c6b.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/tables/aa173a96bba3c0c8ca4615ae564b4e18dd7e859d2b8a69132cd38f02f0431c6b.jpg)

![af4d367a3d360f7dce70b11a3bb8c42c97bf16f94bfc0924dcbf4eaed316a175.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/tables/af4d367a3d360f7dce70b11a3bb8c42c97bf16f94bfc0924dcbf4eaed316a175.jpg)

![d1cf36ec2521e9f350c89752109704eeffa6ebf02b4fad241ea09c35f98b43b9.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/tables/d1cf36ec2521e9f350c89752109704eeffa6ebf02b4fad241ea09c35f98b43b9.jpg)

![d22d357d881e0441e80a3ea4364f4806ba9305ba9a35dcafad1ebe1cafc4d20c.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/tables/d22d357d881e0441e80a3ea4364f4806ba9305ba9a35dcafad1ebe1cafc4d20c.jpg)

![dd2648722beba29bd67cd74c63ba8ca843f6802e1dd008ba37d9db7c1fcb9592.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/tables/dd2648722beba29bd67cd74c63ba8ca843f6802e1dd008ba37d9db7c1fcb9592.jpg)

![e218eff45410fd07f32941ee3292c86f7b35d776b39cee6c6b854c049c654d2a.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/tables/e218eff45410fd07f32941ee3292c86f7b35d776b39cee6c6b854c049c654d2a.jpg)

![ea5f4e08ea200ba54a9129a4066db65572d313b4ef949842c847172c9d37c5fa.jpg](../icml_results/5231_MACD_%20Model-Aware%20Contrastive%20Decoding%20via%20Counterfactual%20Data%20for%20Video-LLMs/tables/ea5f4e08ea200ba54a9129a4066db65572d313b4ef949842c847172c9d37c5fa.jpg)

<a id="paper-5184"></a>

## When Distance Distracts: Representation Distance Bias in BT-Loss for Reward Models


### Images

![08238233ee77f11f81832c79dbac95d402fc48bcf56ff2c30bb3c35ee1196ed3.jpg](../icml_results/5232_When%20Distance%20Distracts_%20Representation%20Distance%20Bias%20in%20BT-Loss%20for%20Reward%20Models/images/08238233ee77f11f81832c79dbac95d402fc48bcf56ff2c30bb3c35ee1196ed3.jpg)

![1568f78d6b4f4a79859272014c1162d6412a8ee90db327a06e69f627a310dff9.jpg](../icml_results/5232_When%20Distance%20Distracts_%20Representation%20Distance%20Bias%20in%20BT-Loss%20for%20Reward%20Models/images/1568f78d6b4f4a79859272014c1162d6412a8ee90db327a06e69f627a310dff9.jpg)

![1f310de4fb2e06246a7bea0b8481dff1e8c0182fa2b82fd3f7676f5897795ff4.jpg](../icml_results/5232_When%20Distance%20Distracts_%20Representation%20Distance%20Bias%20in%20BT-Loss%20for%20Reward%20Models/images/1f310de4fb2e06246a7bea0b8481dff1e8c0182fa2b82fd3f7676f5897795ff4.jpg)

![7742892981849309a46878a985fdf9538a896509d921c6815e3d35c63e4ff96e.jpg](../icml_results/5232_When%20Distance%20Distracts_%20Representation%20Distance%20Bias%20in%20BT-Loss%20for%20Reward%20Models/images/7742892981849309a46878a985fdf9538a896509d921c6815e3d35c63e4ff96e.jpg)

![94d9361d113232e4040bdd6480d8cb264c5a76fe8419c34cf24ac2bf2b9eb83a.jpg](../icml_results/5232_When%20Distance%20Distracts_%20Representation%20Distance%20Bias%20in%20BT-Loss%20for%20Reward%20Models/images/94d9361d113232e4040bdd6480d8cb264c5a76fe8419c34cf24ac2bf2b9eb83a.jpg)

![97f56128b31f6b8de087d087b03d76664cab5afeff9fa925805a717cde9b696d.jpg](../icml_results/5232_When%20Distance%20Distracts_%20Representation%20Distance%20Bias%20in%20BT-Loss%20for%20Reward%20Models/images/97f56128b31f6b8de087d087b03d76664cab5afeff9fa925805a717cde9b696d.jpg)

![c91714ddfd04725fea3c84834752e6d96c558e69e37d9e968388728a33b244a8.jpg](../icml_results/5232_When%20Distance%20Distracts_%20Representation%20Distance%20Bias%20in%20BT-Loss%20for%20Reward%20Models/images/c91714ddfd04725fea3c84834752e6d96c558e69e37d9e968388728a33b244a8.jpg)

![efaea1717b727d89d38ff6a1927c09972ce50ccaf11ba7f843ea206204562f05.jpg](../icml_results/5232_When%20Distance%20Distracts_%20Representation%20Distance%20Bias%20in%20BT-Loss%20for%20Reward%20Models/images/efaea1717b727d89d38ff6a1927c09972ce50ccaf11ba7f843ea206204562f05.jpg)

### Tables

![01a9fb4e633673e4e2a11554f7142d5cbd7e65a08fcfa675d93dfba7f210622f.jpg](../icml_results/5232_When%20Distance%20Distracts_%20Representation%20Distance%20Bias%20in%20BT-Loss%20for%20Reward%20Models/tables/01a9fb4e633673e4e2a11554f7142d5cbd7e65a08fcfa675d93dfba7f210622f.jpg)

![651ab2c441673f33a8464e69d6a3b172168437be488824874f3f6ea96ead0fa1.jpg](../icml_results/5232_When%20Distance%20Distracts_%20Representation%20Distance%20Bias%20in%20BT-Loss%20for%20Reward%20Models/tables/651ab2c441673f33a8464e69d6a3b172168437be488824874f3f6ea96ead0fa1.jpg)

![6ff003407a6aa289c2f1060f52e1f86c18afb0c6ba52f3f4459f55df220a5c63.jpg](../icml_results/5232_When%20Distance%20Distracts_%20Representation%20Distance%20Bias%20in%20BT-Loss%20for%20Reward%20Models/tables/6ff003407a6aa289c2f1060f52e1f86c18afb0c6ba52f3f4459f55df220a5c63.jpg)

![95f6ffc9e2f0666a0cf913c3e1e4cd23d52143e53d55ea9d95debc08356463f5.jpg](../icml_results/5232_When%20Distance%20Distracts_%20Representation%20Distance%20Bias%20in%20BT-Loss%20for%20Reward%20Models/tables/95f6ffc9e2f0666a0cf913c3e1e4cd23d52143e53d55ea9d95debc08356463f5.jpg)

![bf4d39f564f0e0de5d4c4bdd8303a152a4242039567970646566d15cf7ca91de.jpg](../icml_results/5232_When%20Distance%20Distracts_%20Representation%20Distance%20Bias%20in%20BT-Loss%20for%20Reward%20Models/tables/bf4d39f564f0e0de5d4c4bdd8303a152a4242039567970646566d15cf7ca91de.jpg)

![e55ff794fc36cbee6d5bef7f1eecad3d45689c8152d7196dd3e29eaf62b08b6d.jpg](../icml_results/5232_When%20Distance%20Distracts_%20Representation%20Distance%20Bias%20in%20BT-Loss%20for%20Reward%20Models/tables/e55ff794fc36cbee6d5bef7f1eecad3d45689c8152d7196dd3e29eaf62b08b6d.jpg)

![e77a38777c4bf2bbb07da0763195a1e81092e61a19be6ba420225356eb998adc.jpg](../icml_results/5232_When%20Distance%20Distracts_%20Representation%20Distance%20Bias%20in%20BT-Loss%20for%20Reward%20Models/tables/e77a38777c4bf2bbb07da0763195a1e81092e61a19be6ba420225356eb998adc.jpg)

<a id="paper-5185"></a>

## Sobolev Regularized Score Difference Estimation in Diffusion Models


### Images

![04568b447a1cdb59a37ed165f4a3069b7cfee34d71f8add8acc9d65675c0b2c7.jpg](../icml_results/5233_Sobolev%20Regularized%20Score%20Difference%20Estimation%20in%20Diffusion%20Models/images/04568b447a1cdb59a37ed165f4a3069b7cfee34d71f8add8acc9d65675c0b2c7.jpg)

![2e3d5eda336305993babcd13dfb72f8e292103f06c7beeadd840b4abfee29944.jpg](../icml_results/5233_Sobolev%20Regularized%20Score%20Difference%20Estimation%20in%20Diffusion%20Models/images/2e3d5eda336305993babcd13dfb72f8e292103f06c7beeadd840b4abfee29944.jpg)

![5237d9d349fbe967f56def5efe92b1bf703834a05edf5035ba5b7387d321ec53.jpg](../icml_results/5233_Sobolev%20Regularized%20Score%20Difference%20Estimation%20in%20Diffusion%20Models/images/5237d9d349fbe967f56def5efe92b1bf703834a05edf5035ba5b7387d321ec53.jpg)

![ebb0227926a2521e6785ef2bd2f1e15c8507f029de49229ed1919bcba9e05e93.jpg](../icml_results/5233_Sobolev%20Regularized%20Score%20Difference%20Estimation%20in%20Diffusion%20Models/images/ebb0227926a2521e6785ef2bd2f1e15c8507f029de49229ed1919bcba9e05e93.jpg)

### Tables

![05603428f91d56eb59d5694d35754d8be758c99610e2884c57a2013b5c04020d.jpg](../icml_results/5233_Sobolev%20Regularized%20Score%20Difference%20Estimation%20in%20Diffusion%20Models/tables/05603428f91d56eb59d5694d35754d8be758c99610e2884c57a2013b5c04020d.jpg)

![219781543e64113cf6b40cedeca988fc502b65273a60ecf222d92e409dba0895.jpg](../icml_results/5233_Sobolev%20Regularized%20Score%20Difference%20Estimation%20in%20Diffusion%20Models/tables/219781543e64113cf6b40cedeca988fc502b65273a60ecf222d92e409dba0895.jpg)

![3790e1ef64b7de869bf51e1002c77333aeccc7fae44e0c3a47780d70e0cbed3b.jpg](../icml_results/5233_Sobolev%20Regularized%20Score%20Difference%20Estimation%20in%20Diffusion%20Models/tables/3790e1ef64b7de869bf51e1002c77333aeccc7fae44e0c3a47780d70e0cbed3b.jpg)

![481052f4b00681a7d9c97176997782631bf85922e8a7a821f45607ffe32cc9f5.jpg](../icml_results/5233_Sobolev%20Regularized%20Score%20Difference%20Estimation%20in%20Diffusion%20Models/tables/481052f4b00681a7d9c97176997782631bf85922e8a7a821f45607ffe32cc9f5.jpg)

![9d6cae173f99dd2e4520cb15c431679f8b3521ae2e641a2997a7e6fbeb4272f7.jpg](../icml_results/5233_Sobolev%20Regularized%20Score%20Difference%20Estimation%20in%20Diffusion%20Models/tables/9d6cae173f99dd2e4520cb15c431679f8b3521ae2e641a2997a7e6fbeb4272f7.jpg)

![eeded7817bc44f1cc512b3c11ddc00d5ff47cd0b4f818520fd796bab5a25de4c.jpg](../icml_results/5233_Sobolev%20Regularized%20Score%20Difference%20Estimation%20in%20Diffusion%20Models/tables/eeded7817bc44f1cc512b3c11ddc00d5ff47cd0b4f818520fd796bab5a25de4c.jpg)

<a id="paper-5186"></a>

## Cost-aware Stopping for Bayesian Optimization


### Images

![000033d5e9ee0fb5653acab6e6e68a758584671d9053b8567176335d7e55e662.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/000033d5e9ee0fb5653acab6e6e68a758584671d9053b8567176335d7e55e662.jpg)

![0b5ae1e87b31f5178e01eff8d11af73b246037e43e6d0fa225db4f7c68c120c7.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/0b5ae1e87b31f5178e01eff8d11af73b246037e43e6d0fa225db4f7c68c120c7.jpg)

![0cab0b87168a756b04a0e7cbc21cc2c79b887efac912514d0a135d78447371c1.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/0cab0b87168a756b04a0e7cbc21cc2c79b887efac912514d0a135d78447371c1.jpg)

![0d1b3eea337901d626503234c2d2fe220f3a46ce0154278bffcd1662ae4f7257.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/0d1b3eea337901d626503234c2d2fe220f3a46ce0154278bffcd1662ae4f7257.jpg)

![13dfaf98f93ac75c2eae131d994f162bb14cbe496ac80c7bf2dcd51afdc55a2a.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/13dfaf98f93ac75c2eae131d994f162bb14cbe496ac80c7bf2dcd51afdc55a2a.jpg)

![1dfb3236b1c648f1950c4e177eaf43ebcf464de65ec3d6b242f0ff993a07df82.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/1dfb3236b1c648f1950c4e177eaf43ebcf464de65ec3d6b242f0ff993a07df82.jpg)

![1fa442b8506f745e244011768c714cab773996344e94f75466583a2f9d8ad0c5.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/1fa442b8506f745e244011768c714cab773996344e94f75466583a2f9d8ad0c5.jpg)

![21f6d4af71d7eebd7b27a590827fb448655fa29c5f684c1c8dab1d88cb23f0b4.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/21f6d4af71d7eebd7b27a590827fb448655fa29c5f684c1c8dab1d88cb23f0b4.jpg)

![224492f4d9e469c119caf79a5320c48c9e5559c00ee2afe34d7a63264b464f6b.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/224492f4d9e469c119caf79a5320c48c9e5559c00ee2afe34d7a63264b464f6b.jpg)

![512a3c3a8fae941151a4b96003eefd35f8ab6401dacc82303df5a675085ea43e.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/512a3c3a8fae941151a4b96003eefd35f8ab6401dacc82303df5a675085ea43e.jpg)

![56e565fce1985c3d48369098b2c024fcd4d1581d56fa4ba4446791d3bb88633e.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/56e565fce1985c3d48369098b2c024fcd4d1581d56fa4ba4446791d3bb88633e.jpg)

![57a94e4939aca9925cc4a27274e75606e652d6fdd3d0322cc101e61c199c224c.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/57a94e4939aca9925cc4a27274e75606e652d6fdd3d0322cc101e61c199c224c.jpg)

![5a060eb239c51714ace49634dd62d2c2069973e7305c7efcb799a93ce647b8e5.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/5a060eb239c51714ace49634dd62d2c2069973e7305c7efcb799a93ce647b8e5.jpg)

![5b510d44d23fdcc84aef53e65f606aaf0891d0ccf8ccb9fd1c36e2807c31f4c1.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/5b510d44d23fdcc84aef53e65f606aaf0891d0ccf8ccb9fd1c36e2807c31f4c1.jpg)

![63d323bbc5981e8f1ee2a7f9ad8548242e4573322d7ed4ffc9ff1ce08e27f749.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/63d323bbc5981e8f1ee2a7f9ad8548242e4573322d7ed4ffc9ff1ce08e27f749.jpg)

![6c0c9413bffa80e8ed10064b37f30049dc6c82b3c23f5c033ff1a7f591d96a9f.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/6c0c9413bffa80e8ed10064b37f30049dc6c82b3c23f5c033ff1a7f591d96a9f.jpg)

![6d45818fad0ef400f649bff0dac28eb0049c8ad2522904d2146e386217ce5dcc.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/6d45818fad0ef400f649bff0dac28eb0049c8ad2522904d2146e386217ce5dcc.jpg)

![71dab3744804c0804f15f592eb06ad292946a3262f97865dfdef375a192bd42b.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/71dab3744804c0804f15f592eb06ad292946a3262f97865dfdef375a192bd42b.jpg)

![71f5541d4203c42172b184c5a9b08e0e27b2669c0f0dee4ba60f9d2b9a7859e9.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/71f5541d4203c42172b184c5a9b08e0e27b2669c0f0dee4ba60f9d2b9a7859e9.jpg)

![7db828268776844ca532a8317d3cc512465cb21f83273824efe02180b23d71f3.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/7db828268776844ca532a8317d3cc512465cb21f83273824efe02180b23d71f3.jpg)

![87214859b2a56073e9021c6301c8afecde750789fee41a018a0193b315f2b343.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/87214859b2a56073e9021c6301c8afecde750789fee41a018a0193b315f2b343.jpg)

![88ee019f7f8cec902df77871578e3fa73eb57d238bbfec575b77bab9f30296d3.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/88ee019f7f8cec902df77871578e3fa73eb57d238bbfec575b77bab9f30296d3.jpg)

![8ca9736f6747a84399baeaf7c848d0714116f664991ba2ce77cfe5fb6b50f663.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/8ca9736f6747a84399baeaf7c848d0714116f664991ba2ce77cfe5fb6b50f663.jpg)

![8cf00076aaabdfacd2b4e91c4086c9eb34fb2ac9d4643f122fdb6ed573b3ae33.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/8cf00076aaabdfacd2b4e91c4086c9eb34fb2ac9d4643f122fdb6ed573b3ae33.jpg)

![9741d3165771467c92bf5490ab41850cfc5e1a780ad463585718254d139e2465.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/9741d3165771467c92bf5490ab41850cfc5e1a780ad463585718254d139e2465.jpg)

![acc37b2a547de6072ebe6d6a409d7604fe5ba00cc06858046d66c1f84dd34b27.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/acc37b2a547de6072ebe6d6a409d7604fe5ba00cc06858046d66c1f84dd34b27.jpg)

![afb1e4b721a05b16b3b35dcddaa3169ea10f13eb130bc4f0dc6930e4aa68e1a8.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/afb1e4b721a05b16b3b35dcddaa3169ea10f13eb130bc4f0dc6930e4aa68e1a8.jpg)

![b8dd6be6aef6fec3eb0c9433a723516295e037e1fc88bf41ec39807011b39afb.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/b8dd6be6aef6fec3eb0c9433a723516295e037e1fc88bf41ec39807011b39afb.jpg)

![c7ddec2ba9f74405ba844a29f4b36d44ca9bfcb6f850b759e17e5c54b548d875.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/c7ddec2ba9f74405ba844a29f4b36d44ca9bfcb6f850b759e17e5c54b548d875.jpg)

![d5c492985895118cb2c8a93b7a3d5f53897f646e62619c08c29eca1ec3396cd9.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/d5c492985895118cb2c8a93b7a3d5f53897f646e62619c08c29eca1ec3396cd9.jpg)

![d7a67a52249567d324e42599f0245715eaeb722788309c73316c528f8b148283.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/d7a67a52249567d324e42599f0245715eaeb722788309c73316c528f8b148283.jpg)

![d924502a5bc33253975c8351805ae95a2af7baf09b5ec765a508ce4029d57b92.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/d924502a5bc33253975c8351805ae95a2af7baf09b5ec765a508ce4029d57b92.jpg)

![e8e1988902123050b4f12d99756483eec73d36192e943a18c89ee155046203a8.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/e8e1988902123050b4f12d99756483eec73d36192e943a18c89ee155046203a8.jpg)

![eb6d76ecfd190955ccec8a41a722a2e22f5ae893c6b379e08415ce39d9919d2d.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/eb6d76ecfd190955ccec8a41a722a2e22f5ae893c6b379e08415ce39d9919d2d.jpg)

![f9b4eeecbec2368ec93757f9bd01c17a8d3b6c9db6d9e8bca9d1b3c1a017d23a.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/images/f9b4eeecbec2368ec93757f9bd01c17a8d3b6c9db6d9e8bca9d1b3c1a017d23a.jpg)

### Tables

![909015b695f8b56336de4c462060728f66de7be26b0fbe56e49cd78b59226412.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/tables/909015b695f8b56336de4c462060728f66de7be26b0fbe56e49cd78b59226412.jpg)

![d9e15be7d2b219adf4cddf450222196f8333564368d1bf2ad384a1dd0077cd27.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/tables/d9e15be7d2b219adf4cddf450222196f8333564368d1bf2ad384a1dd0077cd27.jpg)

![f91afc30ae4114e1c37def73b026291a8111ca1de93a5e25b5ff6247ff9ac98c.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/tables/f91afc30ae4114e1c37def73b026291a8111ca1de93a5e25b5ff6247ff9ac98c.jpg)

![faee3f86ac27f888404e3db65ddc9208fd12a244e09439aea869af1c91b52aa0.jpg](../icml_results/5234_Cost-aware%20Stopping%20for%20Bayesian%20Optimization/tables/faee3f86ac27f888404e3db65ddc9208fd12a244e09439aea869af1c91b52aa0.jpg)

<a id="paper-5187"></a>

## A Narrowing Geometry in Contaminated Reasoning


### Images

![03f6c143ba6b64c2fbebdfee9b1de0f6bd3172409c646f33795db39d70cf6889.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/03f6c143ba6b64c2fbebdfee9b1de0f6bd3172409c646f33795db39d70cf6889.jpg)

![0939d37eb1a480cee6b331b6ab0f123e1fd8e57fdf6bc62ab4d2039e8be6a55b.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/0939d37eb1a480cee6b331b6ab0f123e1fd8e57fdf6bc62ab4d2039e8be6a55b.jpg)

![0b8bafac767b80ac461f2b31f7540441bd09baccf5d1b150034344f5e49037a9.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/0b8bafac767b80ac461f2b31f7540441bd09baccf5d1b150034344f5e49037a9.jpg)

![0ea8a630f5428e3d0fd1ac750a4249e440f57947b99e32bec078f8febda90f2d.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/0ea8a630f5428e3d0fd1ac750a4249e440f57947b99e32bec078f8febda90f2d.jpg)

![1896440374a90cc33ad17b3c650f0a0ca610b0b61a3e87cf85a23f040b5fa1bb.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/1896440374a90cc33ad17b3c650f0a0ca610b0b61a3e87cf85a23f040b5fa1bb.jpg)

![1a5ae18ab6165128233d7dc52103b23c002c1aa001786b211e7c3f3b8f0a7faf.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/1a5ae18ab6165128233d7dc52103b23c002c1aa001786b211e7c3f3b8f0a7faf.jpg)

![1bc2a20ffb9ad8eee9fd9f1138712d2f974d11b317f6087247b827615718b821.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/1bc2a20ffb9ad8eee9fd9f1138712d2f974d11b317f6087247b827615718b821.jpg)

![1d0bd2d2ec34d07eb4eea555fe7e3113d0a902f4cf810c158faed4ccd039b210.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/1d0bd2d2ec34d07eb4eea555fe7e3113d0a902f4cf810c158faed4ccd039b210.jpg)

![1e640edc923188ab49459494c29c57ddeb4b8e267107d90be814845f8e988456.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/1e640edc923188ab49459494c29c57ddeb4b8e267107d90be814845f8e988456.jpg)

![2210200975694a6ec5c6fb895f5bdf80aaded2b48dfef71b6caffd30a34e2e0c.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/2210200975694a6ec5c6fb895f5bdf80aaded2b48dfef71b6caffd30a34e2e0c.jpg)

![231a4f99527949267d13599754ba795940be2ececce554ea0766feae9fd29154.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/231a4f99527949267d13599754ba795940be2ececce554ea0766feae9fd29154.jpg)

![239df5833dcf27addc01f66060e6a94d37dc357b1d779b6a2f9f3150bccd992c.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/239df5833dcf27addc01f66060e6a94d37dc357b1d779b6a2f9f3150bccd992c.jpg)

![2403cfe7e5cd5a08e8b4e208cb600cd94314f71a8db2494e93411a921455b6dc.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/2403cfe7e5cd5a08e8b4e208cb600cd94314f71a8db2494e93411a921455b6dc.jpg)

![2647ce058eb378c249eb4e859da6f343d3d0ab32a95cbac76a7dbaec5936f64c.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/2647ce058eb378c249eb4e859da6f343d3d0ab32a95cbac76a7dbaec5936f64c.jpg)

![2749489caa712d23d191814a9eb9d3b3c0d6d5030b05b3073b6324b8b49851ea.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/2749489caa712d23d191814a9eb9d3b3c0d6d5030b05b3073b6324b8b49851ea.jpg)

![287a36bbd528756c1a66f410ee1b85fe496af1382c04ce3632ce05a3acf855f1.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/287a36bbd528756c1a66f410ee1b85fe496af1382c04ce3632ce05a3acf855f1.jpg)

![2a7247619420f458971946c79f42948f91447115b8aba4b2251bad2142f112bc.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/2a7247619420f458971946c79f42948f91447115b8aba4b2251bad2142f112bc.jpg)

![3132cfd09773a21674b630a365c9a2941e08db2f38d03663663985fd9488e58d.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/3132cfd09773a21674b630a365c9a2941e08db2f38d03663663985fd9488e58d.jpg)

![33fee888181d1182ec81d63e941e6648547e391c092c1d4c6f934897ba96e421.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/33fee888181d1182ec81d63e941e6648547e391c092c1d4c6f934897ba96e421.jpg)

![3ec04d75b0fb8f5a284018d0a2f80dbfd0072e3779711a1afec641bf8d21b3b0.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/3ec04d75b0fb8f5a284018d0a2f80dbfd0072e3779711a1afec641bf8d21b3b0.jpg)

![3fec5aa834ff3cf5ff757f4a7018733fb3e1bc89b76673a2a6907b8c5d5dd72a.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/3fec5aa834ff3cf5ff757f4a7018733fb3e1bc89b76673a2a6907b8c5d5dd72a.jpg)

![406e8459e99e8bfcdd40e17506045f35b80e9fc4d56a5b26bbf0abcf5c2d992a.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/406e8459e99e8bfcdd40e17506045f35b80e9fc4d56a5b26bbf0abcf5c2d992a.jpg)

![43238bb7488f981fc47d12481dc69717333455213ac4c299435e0feee0ceec2f.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/43238bb7488f981fc47d12481dc69717333455213ac4c299435e0feee0ceec2f.jpg)

![438a208662222783a4a2fbf361b7bd15a8b196249783f790ad0f34c35d5d14c6.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/438a208662222783a4a2fbf361b7bd15a8b196249783f790ad0f34c35d5d14c6.jpg)

![44bf081bbbcce6a54672c4607e022630082bee5429cab5c21703360ffd3a601c.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/44bf081bbbcce6a54672c4607e022630082bee5429cab5c21703360ffd3a601c.jpg)

![457d0fc72fc14e78046232f47c9bf36585ee35704cb3aa858a6c601561736731.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/457d0fc72fc14e78046232f47c9bf36585ee35704cb3aa858a6c601561736731.jpg)

![45960243d5adee4cf4d542dad7b65d39dd4c4ada1032218275b7bbe50c92ef77.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/45960243d5adee4cf4d542dad7b65d39dd4c4ada1032218275b7bbe50c92ef77.jpg)

![46825976ae460e2db63b3452d22b9fa7f212e066ffe624bbdcf3f1ff439ae574.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/46825976ae460e2db63b3452d22b9fa7f212e066ffe624bbdcf3f1ff439ae574.jpg)

![46a99655599afdcb0a669960ac43ac82b886f81ede77949e1ab9bd034e434080.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/46a99655599afdcb0a669960ac43ac82b886f81ede77949e1ab9bd034e434080.jpg)

![48a3cfcb7fb097b65971bb8d9bbf5c21def5466182441132531aefc0d258968d.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/48a3cfcb7fb097b65971bb8d9bbf5c21def5466182441132531aefc0d258968d.jpg)

![49e22dd36508b9e78f64908d210d6e7563ab0ad8708adb316be724f8ff55ce7c.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/49e22dd36508b9e78f64908d210d6e7563ab0ad8708adb316be724f8ff55ce7c.jpg)

![4a156fd176364d6e98541bcb9b6407b20f70f89d6d1cebed994a807682ca2dda.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/4a156fd176364d6e98541bcb9b6407b20f70f89d6d1cebed994a807682ca2dda.jpg)

![4b40a5846d56143af3019a656954177147648cd2b280e93eb1ca72e1219546d9.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/4b40a5846d56143af3019a656954177147648cd2b280e93eb1ca72e1219546d9.jpg)

![4d9faee03ef3277656d28bbff0274467a158a87f92ffe5ce71f7be23db580088.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/4d9faee03ef3277656d28bbff0274467a158a87f92ffe5ce71f7be23db580088.jpg)

![50b7479330afe43235800aa112f61e08f2264e63a11ec70f6960ce665d7bcf2b.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/50b7479330afe43235800aa112f61e08f2264e63a11ec70f6960ce665d7bcf2b.jpg)

![537236346dd4355ada1322de7481ee16bc4061fee1843cc272cbeeb2f35aafbb.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/537236346dd4355ada1322de7481ee16bc4061fee1843cc272cbeeb2f35aafbb.jpg)

![5513a8ed808e9d18e8d6d9466d3b862c2307ab364f43e89a333e0cea620ebd4a.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/5513a8ed808e9d18e8d6d9466d3b862c2307ab364f43e89a333e0cea620ebd4a.jpg)

![5a76253493b04e8636d5f4b9935ee345814518cad7562eb4affbe62bbabe9094.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/5a76253493b04e8636d5f4b9935ee345814518cad7562eb4affbe62bbabe9094.jpg)

![5e47db38601649929418f3ad467f208cf5adbd5539a705974322cdabfc22df05.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/5e47db38601649929418f3ad467f208cf5adbd5539a705974322cdabfc22df05.jpg)

![61df4ebc784db527c928eaa5f01f7bcf32de07b208a47b6aebce3e782a815e88.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/61df4ebc784db527c928eaa5f01f7bcf32de07b208a47b6aebce3e782a815e88.jpg)

![657cb25d692028b384e8e50f8fe1c988e1c6c13797b47732e45d8582eb7840e3.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/657cb25d692028b384e8e50f8fe1c988e1c6c13797b47732e45d8582eb7840e3.jpg)

![65cadd7b6e1bfad38d76b5a44c1a01176cd6c3d1d603bf02e98da16264eaaafc.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/65cadd7b6e1bfad38d76b5a44c1a01176cd6c3d1d603bf02e98da16264eaaafc.jpg)

![661b2ab3c1b210862d96ea2d0f99de0f2f91079abef0529b6a5e2877ec343a3e.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/661b2ab3c1b210862d96ea2d0f99de0f2f91079abef0529b6a5e2877ec343a3e.jpg)

![69c89d1fb20147fb59bfc8be41dde6569edf4ccba39ae167ecf0c739e3ccc4d9.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/69c89d1fb20147fb59bfc8be41dde6569edf4ccba39ae167ecf0c739e3ccc4d9.jpg)

![6a3f8f67523e5775ce5c992be3da0a3d949b4a1402a9d1bd0416aaf25d0a68f9.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/6a3f8f67523e5775ce5c992be3da0a3d949b4a1402a9d1bd0416aaf25d0a68f9.jpg)

![6aa16889662cc52d4aa748a6542e48727f011a53bda07ba4f4368799bc6cf87b.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/6aa16889662cc52d4aa748a6542e48727f011a53bda07ba4f4368799bc6cf87b.jpg)

![6d338f6d83b945756f85d6fc69a69852987bf9419b509bf17095b6de1478bc22.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/6d338f6d83b945756f85d6fc69a69852987bf9419b509bf17095b6de1478bc22.jpg)

![6e0cdd61d7296714793346bbff7e5f8cfd9da3ed69a7ba7942119996845f4fce.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/6e0cdd61d7296714793346bbff7e5f8cfd9da3ed69a7ba7942119996845f4fce.jpg)

![71c6f5561c40bc7eae49d4442d359cc5e858c21d3f126861bde3426ca8a61b55.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/71c6f5561c40bc7eae49d4442d359cc5e858c21d3f126861bde3426ca8a61b55.jpg)

![76fadfc4410787260dee136c4e0ff6fd629c5934e8572cf487d7e35fd782dcae.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/76fadfc4410787260dee136c4e0ff6fd629c5934e8572cf487d7e35fd782dcae.jpg)

![7f9f57855cd083aa09c412db60b659e231ff8b35bc6802309368afdc8caadd79.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/7f9f57855cd083aa09c412db60b659e231ff8b35bc6802309368afdc8caadd79.jpg)

![7faf2fc80d61e9d95409b2c9827b3cbbb7612f4c7bf002f4faff679e117bfe6c.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/7faf2fc80d61e9d95409b2c9827b3cbbb7612f4c7bf002f4faff679e117bfe6c.jpg)

![8463095cec39db4af834a15c298dbef9152db3c8819c0751407a9051e9d3ad6c.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/8463095cec39db4af834a15c298dbef9152db3c8819c0751407a9051e9d3ad6c.jpg)

![8628cba885b1261ae168fcfb7944c58e33d0d46cf4c85b38d2aa6a7eb9f44167.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/8628cba885b1261ae168fcfb7944c58e33d0d46cf4c85b38d2aa6a7eb9f44167.jpg)

![86b533064691d5fa260dc7afa8e04a3b1e857459331f19c104a1eda5d1de6b05.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/86b533064691d5fa260dc7afa8e04a3b1e857459331f19c104a1eda5d1de6b05.jpg)

![88e55b754e8ee46a5de79fe41fad4928edbaf771945f967f653763199f2181d4.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/88e55b754e8ee46a5de79fe41fad4928edbaf771945f967f653763199f2181d4.jpg)

![88f6cd25ec4db5855eec0db24c28eb6abeefb89e30a4ebddeeb32830a0a12f39.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/88f6cd25ec4db5855eec0db24c28eb6abeefb89e30a4ebddeeb32830a0a12f39.jpg)

![8d892991ad3d5550387ad065dc958eb182f6ccfc2db00c1f48f43aa6727256c0.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/8d892991ad3d5550387ad065dc958eb182f6ccfc2db00c1f48f43aa6727256c0.jpg)

![8dadb93e24b36d04a6931015eacface42cfc54abde68117c3dbb5e7c7403a559.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/8dadb93e24b36d04a6931015eacface42cfc54abde68117c3dbb5e7c7403a559.jpg)

![8dec312edfe553a7b4ee397ef7c94e478d0fc153d66b9996255c3a39c4fb3889.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/8dec312edfe553a7b4ee397ef7c94e478d0fc153d66b9996255c3a39c4fb3889.jpg)

![92ab7399926b006d6962ca05848c930480fff50613de3bb690d0d423c559762f.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/92ab7399926b006d6962ca05848c930480fff50613de3bb690d0d423c559762f.jpg)

![940038bb0879078caf4b71d18e7872c41a254960b213efa2debac7d0dbbe51c0.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/940038bb0879078caf4b71d18e7872c41a254960b213efa2debac7d0dbbe51c0.jpg)

![974c07d914b2de3c72beb6ec62e87003b782dcfeb14f86be4e8508aa98d265e1.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/974c07d914b2de3c72beb6ec62e87003b782dcfeb14f86be4e8508aa98d265e1.jpg)

![97ade91bed8857d3c180e05494edcfeffe93cadd10bba4b755cb8e504d7b437e.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/97ade91bed8857d3c180e05494edcfeffe93cadd10bba4b755cb8e504d7b437e.jpg)

![9889973b18c28317c5ecafaed49732fcda00ba3ffa943f1071a25c40db318538.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/9889973b18c28317c5ecafaed49732fcda00ba3ffa943f1071a25c40db318538.jpg)

![9ccaa14f64a56e2fc3813034f28b2e1b1634ca762b6c3f79baed98d27c07ead3.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/9ccaa14f64a56e2fc3813034f28b2e1b1634ca762b6c3f79baed98d27c07ead3.jpg)

![9d4f3782c0169f22a4db4b4d4c6d07b8eb4f7326080177042f8e6600f6bb85b8.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/9d4f3782c0169f22a4db4b4d4c6d07b8eb4f7326080177042f8e6600f6bb85b8.jpg)

![9e516dab74cd45536a55a2b99075272af480777809d037154797d0c903dc08a0.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/9e516dab74cd45536a55a2b99075272af480777809d037154797d0c903dc08a0.jpg)

![9f7179edd71fc342a2d33d57037d4be909c94d1ef7d622d5f8a3e353c4a866b6.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/9f7179edd71fc342a2d33d57037d4be909c94d1ef7d622d5f8a3e353c4a866b6.jpg)

![a0d0e77a618328dc1f49331caa003662ac29d677db54d8005e93eb27aff907f7.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/a0d0e77a618328dc1f49331caa003662ac29d677db54d8005e93eb27aff907f7.jpg)

![a13a2a804f05dd99d5359435632f6dd847fa421a5e486789bea8dcf40095ddd8.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/a13a2a804f05dd99d5359435632f6dd847fa421a5e486789bea8dcf40095ddd8.jpg)

![a621182854067044beb53326538770cc39096c5479f75b16a298f40200e8c1a4.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/a621182854067044beb53326538770cc39096c5479f75b16a298f40200e8c1a4.jpg)

![a7915790879a446c65a56e19787ff9a7abdd291a7962d6ab457a57c9ab94b8ec.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/a7915790879a446c65a56e19787ff9a7abdd291a7962d6ab457a57c9ab94b8ec.jpg)

![a8bfa17ca22fc99ad1031347cef1d3cf9ad71a92fd30ba213bbdfb6e4ed5da9a.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/a8bfa17ca22fc99ad1031347cef1d3cf9ad71a92fd30ba213bbdfb6e4ed5da9a.jpg)

![aa7ee72dbdaf0b825c1b0952b2213431546d4f2d0af579fcae1ab33a0bba547d.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/aa7ee72dbdaf0b825c1b0952b2213431546d4f2d0af579fcae1ab33a0bba547d.jpg)

![af7d247372ccd8c37dd2e6ec221071f22456ea9bf598b74d0a4c9d7d29f769df.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/af7d247372ccd8c37dd2e6ec221071f22456ea9bf598b74d0a4c9d7d29f769df.jpg)

![afd21cdbf5a38bfc36baca0285c1070aa8b43c76a98bd02c387b77b81f66b522.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/afd21cdbf5a38bfc36baca0285c1070aa8b43c76a98bd02c387b77b81f66b522.jpg)

![b2612c9253b8a59ead2d34c7211bad63dbca005c68844408b17a3f5f0764c896.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/b2612c9253b8a59ead2d34c7211bad63dbca005c68844408b17a3f5f0764c896.jpg)

![b553159f81334d3b2afc3482c61ef09b900e3f32f425164ac044ddb69821875f.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/b553159f81334d3b2afc3482c61ef09b900e3f32f425164ac044ddb69821875f.jpg)

![b86a8f15f100c8126fdabd1f1b724b8d3c656804522f893e4209fdfa271feb33.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/b86a8f15f100c8126fdabd1f1b724b8d3c656804522f893e4209fdfa271feb33.jpg)

![b8835ab69e2312c61bbc2e456b824ee9268c12830f285dee0f306d7f748b6307.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/b8835ab69e2312c61bbc2e456b824ee9268c12830f285dee0f306d7f748b6307.jpg)

![bee04ac4d637d0e4e1a751009f5d231464358ef3c6d107353546d0a415ed939a.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/bee04ac4d637d0e4e1a751009f5d231464358ef3c6d107353546d0a415ed939a.jpg)

![c017d93f814fc0ed5cf7b648076fc4ee8f6f88754f205db8bb1d474cdd257688.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/c017d93f814fc0ed5cf7b648076fc4ee8f6f88754f205db8bb1d474cdd257688.jpg)

![c16b3fac0a91fe9728d531b2cb744d7501c0198c5329a6eaa66aa332f99b29d3.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/c16b3fac0a91fe9728d531b2cb744d7501c0198c5329a6eaa66aa332f99b29d3.jpg)

![c8cc0634629ed31fd9ae3d7ec314f99a821e9fcd79008391e83248b14fc48b5c.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/c8cc0634629ed31fd9ae3d7ec314f99a821e9fcd79008391e83248b14fc48b5c.jpg)

![ca01fbd0c1d89d4e9644a510534abfd94df0f98bd28d80015c728c921984a0f6.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/ca01fbd0c1d89d4e9644a510534abfd94df0f98bd28d80015c728c921984a0f6.jpg)

![cb6c975738c31d8ec3b3e0519ef51f9b163eb962e8db70091545ac74f634ccc5.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/cb6c975738c31d8ec3b3e0519ef51f9b163eb962e8db70091545ac74f634ccc5.jpg)

![ce9ed4339a0123f3b2fb9ddcd459fffa9c9489109b69caed09267bebc2586b80.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/ce9ed4339a0123f3b2fb9ddcd459fffa9c9489109b69caed09267bebc2586b80.jpg)

![d624fc5b1e2f1455ff5d3bdf81650308ae55bf0dd66af7dee459e47b6cfc370d.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/d624fc5b1e2f1455ff5d3bdf81650308ae55bf0dd66af7dee459e47b6cfc370d.jpg)

![dd82b137f60024e414f779971409d78350df855d371f7cdf941e0532ab6db1ab.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/dd82b137f60024e414f779971409d78350df855d371f7cdf941e0532ab6db1ab.jpg)

![e1ccc23bf0108d9492250dc4f140d326e5b993a10da2d23294fbf721f6fcaa45.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/e1ccc23bf0108d9492250dc4f140d326e5b993a10da2d23294fbf721f6fcaa45.jpg)

![e3a12d947d9dbd9fb1b393d4b29fe21667aa1453de73b867d00e51348a327b8e.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/e3a12d947d9dbd9fb1b393d4b29fe21667aa1453de73b867d00e51348a327b8e.jpg)

![e4e31601181779e062c2f42ac7480f077df07c0f2d95f9421884c7cb7e1c10a1.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/e4e31601181779e062c2f42ac7480f077df07c0f2d95f9421884c7cb7e1c10a1.jpg)

![e7302bd0dd53e382dd9855c845a6171101285290454513c1aad6b4b5b1636963.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/e7302bd0dd53e382dd9855c845a6171101285290454513c1aad6b4b5b1636963.jpg)

![edc91cadbe7da695f1abbbe319ca42100f89f3ab08e7e1d884f4f57ef7e65d65.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/edc91cadbe7da695f1abbbe319ca42100f89f3ab08e7e1d884f4f57ef7e65d65.jpg)

![efc4d53136e55f15c71e5f5594abe302a73056e22bc08a99f9a58822b962159d.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/efc4d53136e55f15c71e5f5594abe302a73056e22bc08a99f9a58822b962159d.jpg)

![f58fb65e8215f03ddcf7020bcbb6ed887e86fed7cc9b0c6ba0cd7000a09c4e20.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/f58fb65e8215f03ddcf7020bcbb6ed887e86fed7cc9b0c6ba0cd7000a09c4e20.jpg)

![f7384c83b73ff3b9290831be90ad72172e97732ef0f1b184009a1493bc8c16ae.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/f7384c83b73ff3b9290831be90ad72172e97732ef0f1b184009a1493bc8c16ae.jpg)

![fbb6732d474704048c96546e1197992c77be08e2f7486237a2aa3258bcd9c83b.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/fbb6732d474704048c96546e1197992c77be08e2f7486237a2aa3258bcd9c83b.jpg)

![fd59e9ac19a9b65d2e716176d0daa14564bbe127fadaf05dc99e72a04f2601fe.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/images/fd59e9ac19a9b65d2e716176d0daa14564bbe127fadaf05dc99e72a04f2601fe.jpg)

### Tables

![4304f4b393a0853ca06c9ec1ae41957d882f1c42a53a112dad64765209988592.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/tables/4304f4b393a0853ca06c9ec1ae41957d882f1c42a53a112dad64765209988592.jpg)

![4a0e11ed5dd21ea2877855b1613a96ba35990f677052fc1f104dd11f30c1b5a9.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/tables/4a0e11ed5dd21ea2877855b1613a96ba35990f677052fc1f104dd11f30c1b5a9.jpg)

![4e69b85ddb343b9d36c9b8be39e406bf0a8a86272f2cca7df1fd1f140d5fabe9.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/tables/4e69b85ddb343b9d36c9b8be39e406bf0a8a86272f2cca7df1fd1f140d5fabe9.jpg)

![582a841eea988abc7e06834e4e33142816548c6b75195b860f2284370be7ea99.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/tables/582a841eea988abc7e06834e4e33142816548c6b75195b860f2284370be7ea99.jpg)

![71620fca02a5d91466b9bb29c650d76da4bb5cf738fcfdb4732c11a0ba35ca39.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/tables/71620fca02a5d91466b9bb29c650d76da4bb5cf738fcfdb4732c11a0ba35ca39.jpg)

![b3978baaffcf5196be966817e0cd1f3f7ab3588280d89c8cee805430e273b116.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/tables/b3978baaffcf5196be966817e0cd1f3f7ab3588280d89c8cee805430e273b116.jpg)

![c11ffbfdd330986ffbfbb99a5fbcdec64d2892ab8537fef2673da018f59b45c9.jpg](../icml_results/5235_A%20Narrowing%20Geometry%20in%20Contaminated%20Reasoning/tables/c11ffbfdd330986ffbfbb99a5fbcdec64d2892ab8537fef2673da018f59b45c9.jpg)

<a id="paper-5188"></a>

## Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding


### Images

![0c7d758830e1247c957a86d34da1729d87d0c8e75a6fc50a4a78370b020e3539.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/images/0c7d758830e1247c957a86d34da1729d87d0c8e75a6fc50a4a78370b020e3539.jpg)

![2bafba736b057dfaf28e6c089545b9f16e1d96534568aa0669c1896815623c88.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/images/2bafba736b057dfaf28e6c089545b9f16e1d96534568aa0669c1896815623c88.jpg)

![2c52f67bd7df4dcec06fefbd29bfaebe76fb95967d8b2d2a63fccd8c6a2f5583.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/images/2c52f67bd7df4dcec06fefbd29bfaebe76fb95967d8b2d2a63fccd8c6a2f5583.jpg)

![532f08d97ded61672ae79631e2a4d9e0f21b63ae3e9a12a1c4ca1206aa0f6c63.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/images/532f08d97ded61672ae79631e2a4d9e0f21b63ae3e9a12a1c4ca1206aa0f6c63.jpg)

![6a1371f4d0d6b05a77c4ce61f81a1221c15e865b76caed95e847cc1ac2c844bf.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/images/6a1371f4d0d6b05a77c4ce61f81a1221c15e865b76caed95e847cc1ac2c844bf.jpg)

![6a7b624165e4e5b9b99cdfae55b3ce45e4f6a057a0c0d763af6e12326ff128a4.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/images/6a7b624165e4e5b9b99cdfae55b3ce45e4f6a057a0c0d763af6e12326ff128a4.jpg)

![7b26c83e5c3a518923cc2d94ccafe6e608e81bd669d4d00686b232e9c52eca19.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/images/7b26c83e5c3a518923cc2d94ccafe6e608e81bd669d4d00686b232e9c52eca19.jpg)

![9198d67ee726f97299555df270cc006cc08c171e69aa0708aa369b63c78cd387.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/images/9198d67ee726f97299555df270cc006cc08c171e69aa0708aa369b63c78cd387.jpg)

![d15f9cc57a79e7698d2e853b5d1a886755caaa3a4fe411324f5487e0ebbd2b75.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/images/d15f9cc57a79e7698d2e853b5d1a886755caaa3a4fe411324f5487e0ebbd2b75.jpg)

![d2e0456a0493bbd0c95d4736467aa16bc937be2a2baa34431b874ce5ecc5788a.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/images/d2e0456a0493bbd0c95d4736467aa16bc937be2a2baa34431b874ce5ecc5788a.jpg)

![d740ad9c3bf2ead05becfdb0749a68af868f57206672953a40e9b9662e8eb1a1.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/images/d740ad9c3bf2ead05becfdb0749a68af868f57206672953a40e9b9662e8eb1a1.jpg)

![de65028392193d3e828e8bcadf345c0fbd2a9d560b6410e4b051eb2217c80882.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/images/de65028392193d3e828e8bcadf345c0fbd2a9d560b6410e4b051eb2217c80882.jpg)

![ea4f612fb57655d6c00a08ace830dcdf23eeaa54787631eb92638e3fa764c319.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/images/ea4f612fb57655d6c00a08ace830dcdf23eeaa54787631eb92638e3fa764c319.jpg)

![f754226216d7d4e78ae3ed110ab30d0805e07143179e014b3ba025193f7995af.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/images/f754226216d7d4e78ae3ed110ab30d0805e07143179e014b3ba025193f7995af.jpg)

### Tables

![0c70e4ba07ba4b0645e2c06a6989d6af429e8a8450e90daee47ac2db2acf4206.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/tables/0c70e4ba07ba4b0645e2c06a6989d6af429e8a8450e90daee47ac2db2acf4206.jpg)

![53bb660afb7760f77e097ba01b70c1496721c2c422d33ace8e9ffe8382310b53.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/tables/53bb660afb7760f77e097ba01b70c1496721c2c422d33ace8e9ffe8382310b53.jpg)

![6cff2f5d46039f81d265cc819fed1635ae333ed7e0f55534d49fef549a4e8d03.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/tables/6cff2f5d46039f81d265cc819fed1635ae333ed7e0f55534d49fef549a4e8d03.jpg)

![897254b74ffa06aee6634fb8eef6bab88370abaa96e6d03fd491a19d187ff469.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/tables/897254b74ffa06aee6634fb8eef6bab88370abaa96e6d03fd491a19d187ff469.jpg)

![8ab9aa281d8862c790bf68f6cc4be10f5ab1db938c09b876e7a5cb778502edb7.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/tables/8ab9aa281d8862c790bf68f6cc4be10f5ab1db938c09b876e7a5cb778502edb7.jpg)

![b030a85e8e31b7d593a2ab48d927eb17b52c78f0a623f3fbc88c63e04fec3e1e.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/tables/b030a85e8e31b7d593a2ab48d927eb17b52c78f0a623f3fbc88c63e04fec3e1e.jpg)

![f00a75358cea6ca560c604ee573c497be8037d449f1dc685722e24fea867baf6.jpg](../icml_results/5236_Video-MTR_%20Reinforced%20Multi-Turn%20Reasoning%20for%20Long%20Video%20Understanding/tables/f00a75358cea6ca560c604ee573c497be8037d449f1dc685722e24fea867baf6.jpg)

<a id="paper-5189"></a>

## Dissecting the Safety Circuit: Neuronal Intervention for Transferable Adversarial Attacks on VLMs


### Images

![2bd8f0a40bf87d56c30709f1bcdd06189aac61c07d3e2d2ebbe4e62a8268c65a.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/images/2bd8f0a40bf87d56c30709f1bcdd06189aac61c07d3e2d2ebbe4e62a8268c65a.jpg)

![2e55f197f7bb21cd064b95e9200727ebc933d1eaf929d9009b08b9a9ef0f777c.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/images/2e55f197f7bb21cd064b95e9200727ebc933d1eaf929d9009b08b9a9ef0f777c.jpg)

![41e3812f012949cbd087979e198a31274f97ef7d83c36c75ede468d2eeaace97.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/images/41e3812f012949cbd087979e198a31274f97ef7d83c36c75ede468d2eeaace97.jpg)

![4fea4835e121db0e64826ec3b09b61032a8f8a66b15bfaeba535e274cfdacfbe.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/images/4fea4835e121db0e64826ec3b09b61032a8f8a66b15bfaeba535e274cfdacfbe.jpg)

![5ff0e16cc9936c76d927df999a7da1527f4ed19131d3d3584c10edaeef9f02e5.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/images/5ff0e16cc9936c76d927df999a7da1527f4ed19131d3d3584c10edaeef9f02e5.jpg)

![8e685ae21ba2127eb925c397d5653d3dca807ec43afaed59e23790bd2b77f162.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/images/8e685ae21ba2127eb925c397d5653d3dca807ec43afaed59e23790bd2b77f162.jpg)

![95aff62978b33fb469f96536fb20c95c074e95ed9134c0c291a51d2852ad202d.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/images/95aff62978b33fb469f96536fb20c95c074e95ed9134c0c291a51d2852ad202d.jpg)

![a0569ddb9ac133e29cf5941aeb67046ea1d5078271a66563fbb4928dadae6c10.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/images/a0569ddb9ac133e29cf5941aeb67046ea1d5078271a66563fbb4928dadae6c10.jpg)

### Tables

![12756f34d9e6ead9ab84ec09c4829791e711a3f3bf483df3a870b8c9b69f1468.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/tables/12756f34d9e6ead9ab84ec09c4829791e711a3f3bf483df3a870b8c9b69f1468.jpg)

![1dd5a1630cfc797a98964ea242ce2d62ab110e16f4859afe287af544ba282fbb.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/tables/1dd5a1630cfc797a98964ea242ce2d62ab110e16f4859afe287af544ba282fbb.jpg)

![2c8320a553b0309139f4210de7c9f98a6baffdb17260984c599e806a9c530bdd.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/tables/2c8320a553b0309139f4210de7c9f98a6baffdb17260984c599e806a9c530bdd.jpg)

![4ee38891f7ba83cafe5ff5c3ca5eadf18d8c1e035e210d8b61f6ecd0f9ab90ee.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/tables/4ee38891f7ba83cafe5ff5c3ca5eadf18d8c1e035e210d8b61f6ecd0f9ab90ee.jpg)

![688eeb3d9600a34d2ad8b6b2aa05276fd9ee13b3645ba15513bb7e150c671da0.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/tables/688eeb3d9600a34d2ad8b6b2aa05276fd9ee13b3645ba15513bb7e150c671da0.jpg)

![6f3bc89b1b18297492aa2b8e32c4e677357515d015d690a17955d3eead1d06ed.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/tables/6f3bc89b1b18297492aa2b8e32c4e677357515d015d690a17955d3eead1d06ed.jpg)

![74b1f8ebd266e817ec33a44d81b1f11c6ae5e13ae5cb441568b275672b500a83.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/tables/74b1f8ebd266e817ec33a44d81b1f11c6ae5e13ae5cb441568b275672b500a83.jpg)

![855a4d69025c3e38adde983113db704ce832e678889e8bf2874afaa942b7ddff.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/tables/855a4d69025c3e38adde983113db704ce832e678889e8bf2874afaa942b7ddff.jpg)

![87fd08c85230b98d967d9792011c35c135eaec6f359736d36065eb808f287fb3.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/tables/87fd08c85230b98d967d9792011c35c135eaec6f359736d36065eb808f287fb3.jpg)

![9296fe057e98cbbfb249735c95ab2fa0985d9f7807b127387a8d526a5ec1bee9.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/tables/9296fe057e98cbbfb249735c95ab2fa0985d9f7807b127387a8d526a5ec1bee9.jpg)

![ad399249e8575ed9ac16ac01e2859de4251a6fa77863a71e3067f9562c69833b.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/tables/ad399249e8575ed9ac16ac01e2859de4251a6fa77863a71e3067f9562c69833b.jpg)

![ad5744137d447ee3a4d30bba40500dee577f0af6c9432a15f64260251a00ba1e.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/tables/ad5744137d447ee3a4d30bba40500dee577f0af6c9432a15f64260251a00ba1e.jpg)

![c70c94dc9de1536cd0aeae016def065020643b358cae346c1c16bffefe6409ec.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/tables/c70c94dc9de1536cd0aeae016def065020643b358cae346c1c16bffefe6409ec.jpg)

![cf67ca47b28330e6aee1e954d52440ecaab145d1779972a0665617cc68e2f778.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/tables/cf67ca47b28330e6aee1e954d52440ecaab145d1779972a0665617cc68e2f778.jpg)

![df3f6623b41f4229765e117fa571bf85b74bc34c62a5d1995c46e0b2cc24f9a5.jpg](../icml_results/5237_Dissecting%20the%20Safety%20Circuit_%20Neuronal%20Intervention%20for%20Transferable%20Adversarial%20Attacks%20on%20VLMs/tables/df3f6623b41f4229765e117fa571bf85b74bc34c62a5d1995c46e0b2cc24f9a5.jpg)

<a id="paper-5190"></a>

## SHERPA: Fine-tuning Segment Anything Models with Task-relevant Guidance


### Images

![2932fdd7c9084d27002841ce92ec089c9bc9b86ee7f0a806b341f46986bc8eb4.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/images/2932fdd7c9084d27002841ce92ec089c9bc9b86ee7f0a806b341f46986bc8eb4.jpg)

![2de419fb506bdf2ec3469d1af6c34e098bd6a3ad43809a9a337aad7be017fb15.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/images/2de419fb506bdf2ec3469d1af6c34e098bd6a3ad43809a9a337aad7be017fb15.jpg)

![634630ad292918a4dad59a743d6981b1f181c496acbe855748909679135438b6.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/images/634630ad292918a4dad59a743d6981b1f181c496acbe855748909679135438b6.jpg)

![7adc7c6353c86fa15ca216292733e3fc8cb06390cd481338d10e50d8231497e5.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/images/7adc7c6353c86fa15ca216292733e3fc8cb06390cd481338d10e50d8231497e5.jpg)

![8b61d0093271568d3a723165e441a0a49e61bc39f890d7349473903d6f45620d.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/images/8b61d0093271568d3a723165e441a0a49e61bc39f890d7349473903d6f45620d.jpg)

![920bd0e9a5352c4c031dfcfd0e3780c5f1ac0fb48a033c0968841d9733cbc7f5.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/images/920bd0e9a5352c4c031dfcfd0e3780c5f1ac0fb48a033c0968841d9733cbc7f5.jpg)

![996b8319d0667a71965deb02bc2ab64e824c1efaa109d0e57d6e195f4f5e542b.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/images/996b8319d0667a71965deb02bc2ab64e824c1efaa109d0e57d6e195f4f5e542b.jpg)

![9ad2a825aa302b7ecf084dec77779b7f3542b02f999912e236f4f5f840e40ff2.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/images/9ad2a825aa302b7ecf084dec77779b7f3542b02f999912e236f4f5f840e40ff2.jpg)

![d1e77fd30d7280cd0d0d7c833a4a7b68819863d988172e3e68e73d6764df837d.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/images/d1e77fd30d7280cd0d0d7c833a4a7b68819863d988172e3e68e73d6764df837d.jpg)

### Tables

![0f104073ed40e281722a91ea8e92812852f0c8358895d9316209d9e8d7c8675b.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/0f104073ed40e281722a91ea8e92812852f0c8358895d9316209d9e8d7c8675b.jpg)

![15f66dcff7c0a800f2600b9e7001b8083fcdd3fcf20ed04ea97e451ffda0dd9b.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/15f66dcff7c0a800f2600b9e7001b8083fcdd3fcf20ed04ea97e451ffda0dd9b.jpg)

![19e17e2489ed5e90b3bf72479a8f584eecc1cddbe7fe293027280459d06b0509.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/19e17e2489ed5e90b3bf72479a8f584eecc1cddbe7fe293027280459d06b0509.jpg)

![2049db2db980e53264abc365608a17927d4c35db9f9c9fde7f8f77b9178706ec.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/2049db2db980e53264abc365608a17927d4c35db9f9c9fde7f8f77b9178706ec.jpg)

![3c76685c3483c4ddafc54fa6bbdea3f8a1cbeb870b6b6b339c2452ecada4dbaa.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/3c76685c3483c4ddafc54fa6bbdea3f8a1cbeb870b6b6b339c2452ecada4dbaa.jpg)

![45fc1475fb66660160365cff1a6e5d96719ae0691ab209f6aca369b09e2336d4.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/45fc1475fb66660160365cff1a6e5d96719ae0691ab209f6aca369b09e2336d4.jpg)

![4f8c3fd5b0605bd5707b61bc1669b392400285017fc14d7e65ae10ba69bdd391.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/4f8c3fd5b0605bd5707b61bc1669b392400285017fc14d7e65ae10ba69bdd391.jpg)

![56db932997732fa6bbed708837ddf61aa74935875d8974514b3e386ad8f0f858.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/56db932997732fa6bbed708837ddf61aa74935875d8974514b3e386ad8f0f858.jpg)

![5cacf27abd4744ae9af23434e768635b54e8f625b3afb52cedd79d20a750bc85.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/5cacf27abd4744ae9af23434e768635b54e8f625b3afb52cedd79d20a750bc85.jpg)

![65617281b3fbe5d824f1d6ee753d2a4e11788550f254aa0f4ed2070fa62f4422.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/65617281b3fbe5d824f1d6ee753d2a4e11788550f254aa0f4ed2070fa62f4422.jpg)

![6b49ef97f6936a07d70fbcba6c8dc88067c961ef3caf47346b6fec10c05557ac.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/6b49ef97f6936a07d70fbcba6c8dc88067c961ef3caf47346b6fec10c05557ac.jpg)

![6c2bff107c82433f6921c55094be58060c658f74922f5b12db851c2feca1624a.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/6c2bff107c82433f6921c55094be58060c658f74922f5b12db851c2feca1624a.jpg)

![6c37e80f293cccb06dd6a5c686dec8f83de1f8a395d5ad691fb9ebaeda6ee6e9.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/6c37e80f293cccb06dd6a5c686dec8f83de1f8a395d5ad691fb9ebaeda6ee6e9.jpg)

![7c7a2dbcb03b1d022f1ed7ea7efd5354c9cea67578068614b93c55285612ed75.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/7c7a2dbcb03b1d022f1ed7ea7efd5354c9cea67578068614b93c55285612ed75.jpg)

![85ef2e6ae9401705d52ce7f72e363f2a63261650af12bf26464a5a7bc87c49ba.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/85ef2e6ae9401705d52ce7f72e363f2a63261650af12bf26464a5a7bc87c49ba.jpg)

![86dd335e9d260eed68d248070e9268fd185307c98c42db0ba9420f5b1e62c9d1.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/86dd335e9d260eed68d248070e9268fd185307c98c42db0ba9420f5b1e62c9d1.jpg)

![97207fb930e54c92e84087173196b206821aa8c371b874c1d942fb66832d081a.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/97207fb930e54c92e84087173196b206821aa8c371b874c1d942fb66832d081a.jpg)

![9e21996a48baa0712c1f51f10ff85efd2fe832499e0ddb5d18c246e5ce9e4fbd.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/9e21996a48baa0712c1f51f10ff85efd2fe832499e0ddb5d18c246e5ce9e4fbd.jpg)

![a9e260f7639d70055a868cf59a97d0ab40e7f04de6c78eb9ed6598f7787d03b7.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/a9e260f7639d70055a868cf59a97d0ab40e7f04de6c78eb9ed6598f7787d03b7.jpg)

![adf00656dbe07fd137615e1d4fd54867f11a392e2597ac7b740b5a84404a864a.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/adf00656dbe07fd137615e1d4fd54867f11a392e2597ac7b740b5a84404a864a.jpg)

![b6601dcf36602bffbd8d224dd42f2871624e446180ca89f5285d7f9a97fc3cf5.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/b6601dcf36602bffbd8d224dd42f2871624e446180ca89f5285d7f9a97fc3cf5.jpg)

![c0aa7fd4f9490d1a3e6e68118d25df44e4a87f4758f32f7e649808689d212a61.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/c0aa7fd4f9490d1a3e6e68118d25df44e4a87f4758f32f7e649808689d212a61.jpg)

![ee5753403ea0c0cc02b4a57460f4b77e78b1d5fa4e003dc551bda2cc6cd306d8.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/ee5753403ea0c0cc02b4a57460f4b77e78b1d5fa4e003dc551bda2cc6cd306d8.jpg)

![f295e924fb8dacd3f889a1d4678010f7a4d5a26fce11aa771e3f4182b7629cd7.jpg](../icml_results/5238_SHERPA_%20Fine-tuning%20Segment%20Anything%20Models%20with%20Task-relevant%20Guidance/tables/f295e924fb8dacd3f889a1d4678010f7a4d5a26fce11aa771e3f4182b7629cd7.jpg)

<a id="paper-5191"></a>

## Asymptotically Fast Clebsch-Gordan Tensor Products with Vector Spherical Harmonics


### Images

![3ba836bf3183954f015ca7512fb13ccba9688873c48bf9a376c5966a0cd36d9d.jpg](../icml_results/5239_Asymptotically%20Fast%20Clebsch-Gordan%20Tensor%20Products%20with%20Vector%20Spherical%20Harmonics/images/3ba836bf3183954f015ca7512fb13ccba9688873c48bf9a376c5966a0cd36d9d.jpg)

![52a5905349d6b45ee02f758968904a18bfceb82575edd1595648686e178260d5.jpg](../icml_results/5239_Asymptotically%20Fast%20Clebsch-Gordan%20Tensor%20Products%20with%20Vector%20Spherical%20Harmonics/images/52a5905349d6b45ee02f758968904a18bfceb82575edd1595648686e178260d5.jpg)

![787d90651bbd7ba556b7723a2b8a6d7757c8c49ce7a058825e72ce7beacd870d.jpg](../icml_results/5239_Asymptotically%20Fast%20Clebsch-Gordan%20Tensor%20Products%20with%20Vector%20Spherical%20Harmonics/images/787d90651bbd7ba556b7723a2b8a6d7757c8c49ce7a058825e72ce7beacd870d.jpg)

![d3e938f03e301f6e5d1c86c850dd490c84c0de4a526a497c55308d06c457fbcf.jpg](../icml_results/5239_Asymptotically%20Fast%20Clebsch-Gordan%20Tensor%20Products%20with%20Vector%20Spherical%20Harmonics/images/d3e938f03e301f6e5d1c86c850dd490c84c0de4a526a497c55308d06c457fbcf.jpg)

![dc8b7c622e5f23cfc30d83080c8093ac7de167c8ee3c5f5768323c025b41bd33.jpg](../icml_results/5239_Asymptotically%20Fast%20Clebsch-Gordan%20Tensor%20Products%20with%20Vector%20Spherical%20Harmonics/images/dc8b7c622e5f23cfc30d83080c8093ac7de167c8ee3c5f5768323c025b41bd33.jpg)

![e4cac7ed7b5c7614803024717ebf7b2827c472079edec1ec3bc0b6309f983444.jpg](../icml_results/5239_Asymptotically%20Fast%20Clebsch-Gordan%20Tensor%20Products%20with%20Vector%20Spherical%20Harmonics/images/e4cac7ed7b5c7614803024717ebf7b2827c472079edec1ec3bc0b6309f983444.jpg)

### Tables

![7af7c64eec2b12d3fc19b83d82e5b89d84d864ae3bd3f564bccfa72b50eed610.jpg](../icml_results/5239_Asymptotically%20Fast%20Clebsch-Gordan%20Tensor%20Products%20with%20Vector%20Spherical%20Harmonics/tables/7af7c64eec2b12d3fc19b83d82e5b89d84d864ae3bd3f564bccfa72b50eed610.jpg)

![7f3a380203635215528246bd5a0aa55f363592207be0bb0c6c00c838d4b106fe.jpg](../icml_results/5239_Asymptotically%20Fast%20Clebsch-Gordan%20Tensor%20Products%20with%20Vector%20Spherical%20Harmonics/tables/7f3a380203635215528246bd5a0aa55f363592207be0bb0c6c00c838d4b106fe.jpg)

![86ea693d54d4bc977e69dd8e24ea5e149a1330c65cc61b9afcb4d95ab0f7f513.jpg](../icml_results/5239_Asymptotically%20Fast%20Clebsch-Gordan%20Tensor%20Products%20with%20Vector%20Spherical%20Harmonics/tables/86ea693d54d4bc977e69dd8e24ea5e149a1330c65cc61b9afcb4d95ab0f7f513.jpg)

![a08a09b85224be74ebfcbe89cf15cc181b14d0ea906572ac8aa94dfbc3f4da91.jpg](../icml_results/5239_Asymptotically%20Fast%20Clebsch-Gordan%20Tensor%20Products%20with%20Vector%20Spherical%20Harmonics/tables/a08a09b85224be74ebfcbe89cf15cc181b14d0ea906572ac8aa94dfbc3f4da91.jpg)

![baedf3d258f22619e894f306695473776f53838b835a047b6a2d95fb9e57865a.jpg](../icml_results/5239_Asymptotically%20Fast%20Clebsch-Gordan%20Tensor%20Products%20with%20Vector%20Spherical%20Harmonics/tables/baedf3d258f22619e894f306695473776f53838b835a047b6a2d95fb9e57865a.jpg)

![fa89aaf227b02d3fb39b22927b3eac93ba853c14a708b08be97cf937d40d455c.jpg](../icml_results/5239_Asymptotically%20Fast%20Clebsch-Gordan%20Tensor%20Products%20with%20Vector%20Spherical%20Harmonics/tables/fa89aaf227b02d3fb39b22927b3eac93ba853c14a708b08be97cf937d40d455c.jpg)

![fecbb87bfa3aad33c31c64415fd55dd0ed80e91669a3081c9c09eb2a605997e0.jpg](../icml_results/5239_Asymptotically%20Fast%20Clebsch-Gordan%20Tensor%20Products%20with%20Vector%20Spherical%20Harmonics/tables/fecbb87bfa3aad33c31c64415fd55dd0ed80e91669a3081c9c09eb2a605997e0.jpg)

<a id="paper-5192"></a>

## Self-Supervised Weight Templates for Scalable Vision Model Initialization


### Images

![1958d0c1b2eedc2bd19e53ac029340723e7067506ad404f54f8f8161cf3c4c97.jpg](../icml_results/5240_Self-Supervised%20Weight%20Templates%20for%20Scalable%20Vision%20Model%20Initialization/images/1958d0c1b2eedc2bd19e53ac029340723e7067506ad404f54f8f8161cf3c4c97.jpg)

![3287fae034b88d2da75f8833ac112a9e904cfaed216bb91bf92862a09c00b6ed.jpg](../icml_results/5240_Self-Supervised%20Weight%20Templates%20for%20Scalable%20Vision%20Model%20Initialization/images/3287fae034b88d2da75f8833ac112a9e904cfaed216bb91bf92862a09c00b6ed.jpg)

![58e8a4da101b126132eb8b3fd646ba23e8024cd0fe03958b65e6f1344d051fc8.jpg](../icml_results/5240_Self-Supervised%20Weight%20Templates%20for%20Scalable%20Vision%20Model%20Initialization/images/58e8a4da101b126132eb8b3fd646ba23e8024cd0fe03958b65e6f1344d051fc8.jpg)

![7e4ed1cdc07ded19e03ac5fdd22b7c58a4936e52133300d06d5682edf5c9e6f1.jpg](../icml_results/5240_Self-Supervised%20Weight%20Templates%20for%20Scalable%20Vision%20Model%20Initialization/images/7e4ed1cdc07ded19e03ac5fdd22b7c58a4936e52133300d06d5682edf5c9e6f1.jpg)

![a416f93075d20cd54ac831fc7d6164408580ac84b9561e9a69cb38792990f264.jpg](../icml_results/5240_Self-Supervised%20Weight%20Templates%20for%20Scalable%20Vision%20Model%20Initialization/images/a416f93075d20cd54ac831fc7d6164408580ac84b9561e9a69cb38792990f264.jpg)

### Tables

![0abd07b5d6958120d326ff4375fb44ee5542f719ed7d946272d127fb0ef1ca2f.jpg](../icml_results/5240_Self-Supervised%20Weight%20Templates%20for%20Scalable%20Vision%20Model%20Initialization/tables/0abd07b5d6958120d326ff4375fb44ee5542f719ed7d946272d127fb0ef1ca2f.jpg)

![21c4923d01764059ebfe6fd44992a3c683ef5a12633e737d5c7dcf5557cb139e.jpg](../icml_results/5240_Self-Supervised%20Weight%20Templates%20for%20Scalable%20Vision%20Model%20Initialization/tables/21c4923d01764059ebfe6fd44992a3c683ef5a12633e737d5c7dcf5557cb139e.jpg)

![433b67aa31c81e27e0aae6cba9351fd8d004d60fde2e36c5c3db404ab1638dc0.jpg](../icml_results/5240_Self-Supervised%20Weight%20Templates%20for%20Scalable%20Vision%20Model%20Initialization/tables/433b67aa31c81e27e0aae6cba9351fd8d004d60fde2e36c5c3db404ab1638dc0.jpg)

![4e0103d47152b007dfcb38a9457d8cfada71abafc84a06e368caa031e0166d95.jpg](../icml_results/5240_Self-Supervised%20Weight%20Templates%20for%20Scalable%20Vision%20Model%20Initialization/tables/4e0103d47152b007dfcb38a9457d8cfada71abafc84a06e368caa031e0166d95.jpg)

![55bf0dc59d5a04728d4fd0c561ac15cefb7f5739edea6846104ca860b4e82363.jpg](../icml_results/5240_Self-Supervised%20Weight%20Templates%20for%20Scalable%20Vision%20Model%20Initialization/tables/55bf0dc59d5a04728d4fd0c561ac15cefb7f5739edea6846104ca860b4e82363.jpg)

![6416f56d7c57fbba3542e91bd77bf54725758451aa29ae0f51a0f0906d4be1be.jpg](../icml_results/5240_Self-Supervised%20Weight%20Templates%20for%20Scalable%20Vision%20Model%20Initialization/tables/6416f56d7c57fbba3542e91bd77bf54725758451aa29ae0f51a0f0906d4be1be.jpg)

![8cdeb35d36661a229d4f249a51ceb2bde202036407de730067b446cdf80c233f.jpg](../icml_results/5240_Self-Supervised%20Weight%20Templates%20for%20Scalable%20Vision%20Model%20Initialization/tables/8cdeb35d36661a229d4f249a51ceb2bde202036407de730067b446cdf80c233f.jpg)

![c7f14d5127a159f6b8caf19f8d7897d6f95661033b8fbf4ae9996ac9407e8c48.jpg](../icml_results/5240_Self-Supervised%20Weight%20Templates%20for%20Scalable%20Vision%20Model%20Initialization/tables/c7f14d5127a159f6b8caf19f8d7897d6f95661033b8fbf4ae9996ac9407e8c48.jpg)

![e1232d7f634498d0b00105f3ed5e80b74e9b34ec4db5b3a4dcbba6e03e636eec.jpg](../icml_results/5240_Self-Supervised%20Weight%20Templates%20for%20Scalable%20Vision%20Model%20Initialization/tables/e1232d7f634498d0b00105f3ed5e80b74e9b34ec4db5b3a4dcbba6e03e636eec.jpg)

<a id="paper-5193"></a>

## Rethinking Code Complexity Through the Lens of Large Language Models


### Images

![11b5df85ce99a6ae9b41c5727f37650d59e0cbbe398b338690d1ce2fc3628344.jpg](../icml_results/5241_Rethinking%20Code%20Complexity%20Through%20the%20Lens%20of%20Large%20Language%20Models/images/11b5df85ce99a6ae9b41c5727f37650d59e0cbbe398b338690d1ce2fc3628344.jpg)

![9e8973442b82808670d6c4091d949f06641bce537c63307908df345b0e3c5f0a.jpg](../icml_results/5241_Rethinking%20Code%20Complexity%20Through%20the%20Lens%20of%20Large%20Language%20Models/images/9e8973442b82808670d6c4091d949f06641bce537c63307908df345b0e3c5f0a.jpg)

![a9f0a92751b9cc8c83c22ee02a86051f2b3aed26526e537b6235a3ac1eb90280.jpg](../icml_results/5241_Rethinking%20Code%20Complexity%20Through%20the%20Lens%20of%20Large%20Language%20Models/images/a9f0a92751b9cc8c83c22ee02a86051f2b3aed26526e537b6235a3ac1eb90280.jpg)

### Tables

![08cd5b743fecaf1d2a1b1beb52af5b2ad11babb4e8eb21640bd23eabb63b628f.jpg](../icml_results/5241_Rethinking%20Code%20Complexity%20Through%20the%20Lens%20of%20Large%20Language%20Models/tables/08cd5b743fecaf1d2a1b1beb52af5b2ad11babb4e8eb21640bd23eabb63b628f.jpg)

![2cfd15463ae24cba14a1e66296c1724f68ac0720e49c09c860d710f15ad81eb3.jpg](../icml_results/5241_Rethinking%20Code%20Complexity%20Through%20the%20Lens%20of%20Large%20Language%20Models/tables/2cfd15463ae24cba14a1e66296c1724f68ac0720e49c09c860d710f15ad81eb3.jpg)

![4e16e014f9c1ad351e039d218f87117df67a34f5ae27cd920fb2e3f07d7588a6.jpg](../icml_results/5241_Rethinking%20Code%20Complexity%20Through%20the%20Lens%20of%20Large%20Language%20Models/tables/4e16e014f9c1ad351e039d218f87117df67a34f5ae27cd920fb2e3f07d7588a6.jpg)

![7130de77d92cd7cd8024ee6d098be4f4891ee510d33351a65d3cd05e31950b2d.jpg](../icml_results/5241_Rethinking%20Code%20Complexity%20Through%20the%20Lens%20of%20Large%20Language%20Models/tables/7130de77d92cd7cd8024ee6d098be4f4891ee510d33351a65d3cd05e31950b2d.jpg)

![862801ecabf398b4e1388c8931c14b93949c51d174b303089a890f90e1340803.jpg](../icml_results/5241_Rethinking%20Code%20Complexity%20Through%20the%20Lens%20of%20Large%20Language%20Models/tables/862801ecabf398b4e1388c8931c14b93949c51d174b303089a890f90e1340803.jpg)

![b9801b5af47d26d8be10f5fef6201a7270ca5bb24f22017beb6f4914eb5c4ab8.jpg](../icml_results/5241_Rethinking%20Code%20Complexity%20Through%20the%20Lens%20of%20Large%20Language%20Models/tables/b9801b5af47d26d8be10f5fef6201a7270ca5bb24f22017beb6f4914eb5c4ab8.jpg)

![c6300d7a14c05e7e28d59d321f04249d2b413dedb4f33b3ca7115740d5e57112.jpg](../icml_results/5241_Rethinking%20Code%20Complexity%20Through%20the%20Lens%20of%20Large%20Language%20Models/tables/c6300d7a14c05e7e28d59d321f04249d2b413dedb4f33b3ca7115740d5e57112.jpg)

<a id="paper-5194"></a>

## Clustering as Reasoning: A $k$-Means Interpretation of Chain-of-Thought Graph Learning


### Images

![1e6acbd7e227c52bf00473d2a11fef0e226cc218c82011702d525c39328704db.jpg](../icml_results/5242_Clustering%20as%20Reasoning_%20A%20%24k%24-Means%20Interpretation%20of%20Chain-of-Thought%20Graph%20Learning/images/1e6acbd7e227c52bf00473d2a11fef0e226cc218c82011702d525c39328704db.jpg)

![70ba672494f94f3f877721864c7ec9ec2c19c91a0d420cdd072b1b3b40cb3297.jpg](../icml_results/5242_Clustering%20as%20Reasoning_%20A%20%24k%24-Means%20Interpretation%20of%20Chain-of-Thought%20Graph%20Learning/images/70ba672494f94f3f877721864c7ec9ec2c19c91a0d420cdd072b1b3b40cb3297.jpg)

![7711ffcee0eed8d714a7d8d9f96dc8c581f6ba113fc1e8c0c3d392b9ba7aa533.jpg](../icml_results/5242_Clustering%20as%20Reasoning_%20A%20%24k%24-Means%20Interpretation%20of%20Chain-of-Thought%20Graph%20Learning/images/7711ffcee0eed8d714a7d8d9f96dc8c581f6ba113fc1e8c0c3d392b9ba7aa533.jpg)

![a9d303ecbe5eb3053b1793f96071f888e72e462e6e79e49f75a5928a13827b82.jpg](../icml_results/5242_Clustering%20as%20Reasoning_%20A%20%24k%24-Means%20Interpretation%20of%20Chain-of-Thought%20Graph%20Learning/images/a9d303ecbe5eb3053b1793f96071f888e72e462e6e79e49f75a5928a13827b82.jpg)

![d0b45d18caf8e420203662d29db0843523dde56ba66979f871b1318a2ff065ae.jpg](../icml_results/5242_Clustering%20as%20Reasoning_%20A%20%24k%24-Means%20Interpretation%20of%20Chain-of-Thought%20Graph%20Learning/images/d0b45d18caf8e420203662d29db0843523dde56ba66979f871b1318a2ff065ae.jpg)

![f95e88fd99aed4990786074ee9b2be84d99a27172c0a939c9cb8674b1196be2a.jpg](../icml_results/5242_Clustering%20as%20Reasoning_%20A%20%24k%24-Means%20Interpretation%20of%20Chain-of-Thought%20Graph%20Learning/images/f95e88fd99aed4990786074ee9b2be84d99a27172c0a939c9cb8674b1196be2a.jpg)

### Tables

![027a653a76d06aa7361862e6491893710df93ef313b0393de8e66d5e5e34c049.jpg](../icml_results/5242_Clustering%20as%20Reasoning_%20A%20%24k%24-Means%20Interpretation%20of%20Chain-of-Thought%20Graph%20Learning/tables/027a653a76d06aa7361862e6491893710df93ef313b0393de8e66d5e5e34c049.jpg)

![0651ae40c43064e5a0393e88263ea095df8ac3b851e39f97f4f2e2a665ace7b4.jpg](../icml_results/5242_Clustering%20as%20Reasoning_%20A%20%24k%24-Means%20Interpretation%20of%20Chain-of-Thought%20Graph%20Learning/tables/0651ae40c43064e5a0393e88263ea095df8ac3b851e39f97f4f2e2a665ace7b4.jpg)

![5abe88d1caff859f53c4e95209439321c1dcc0c93fe0b04ff6fa4105d8e3f35d.jpg](../icml_results/5242_Clustering%20as%20Reasoning_%20A%20%24k%24-Means%20Interpretation%20of%20Chain-of-Thought%20Graph%20Learning/tables/5abe88d1caff859f53c4e95209439321c1dcc0c93fe0b04ff6fa4105d8e3f35d.jpg)

![aee7b082c2a4260fa67f79ded5d9ce7a13825cd5511cd9e6a798f379cbaaee64.jpg](../icml_results/5242_Clustering%20as%20Reasoning_%20A%20%24k%24-Means%20Interpretation%20of%20Chain-of-Thought%20Graph%20Learning/tables/aee7b082c2a4260fa67f79ded5d9ce7a13825cd5511cd9e6a798f379cbaaee64.jpg)

<a id="paper-5195"></a>

## FedUSD: Unbiased Synthetic Data for Federated Learning


### Images

![0047372c8662ac036cd35abb1ae0628d283ff19ec499a8a54eb8ace4317e223b.jpg](../icml_results/5243_FedUSD_%20Unbiased%20Synthetic%20Data%20for%20Federated%20Learning/images/0047372c8662ac036cd35abb1ae0628d283ff19ec499a8a54eb8ace4317e223b.jpg)

![77c145c316512c09904409ea8cec8e794a9b7a29faf5292498c1471503fd1293.jpg](../icml_results/5243_FedUSD_%20Unbiased%20Synthetic%20Data%20for%20Federated%20Learning/images/77c145c316512c09904409ea8cec8e794a9b7a29faf5292498c1471503fd1293.jpg)

![81a0115a549475614be95627eae84d1fb1194fcb9b943365737cfe09fccc7f94.jpg](../icml_results/5243_FedUSD_%20Unbiased%20Synthetic%20Data%20for%20Federated%20Learning/images/81a0115a549475614be95627eae84d1fb1194fcb9b943365737cfe09fccc7f94.jpg)

![85254c06905e9da2f7991ade16302ff39faf8b4ee084b3d055239bc4deb24676.jpg](../icml_results/5243_FedUSD_%20Unbiased%20Synthetic%20Data%20for%20Federated%20Learning/images/85254c06905e9da2f7991ade16302ff39faf8b4ee084b3d055239bc4deb24676.jpg)

![be6c43d8d955fbb504521553fcbd634cc0c570518f56cfc2fe0edccf9c727a2a.jpg](../icml_results/5243_FedUSD_%20Unbiased%20Synthetic%20Data%20for%20Federated%20Learning/images/be6c43d8d955fbb504521553fcbd634cc0c570518f56cfc2fe0edccf9c727a2a.jpg)

![c7e3a185d8458c3c8415020d301899ac92222e319b0558aaae4bcaed739cd246.jpg](../icml_results/5243_FedUSD_%20Unbiased%20Synthetic%20Data%20for%20Federated%20Learning/images/c7e3a185d8458c3c8415020d301899ac92222e319b0558aaae4bcaed739cd246.jpg)

### Tables

![26bedc239ab8e20a0549f48a968dfcd5e370597685cd58afe4826a50f8d2ddf7.jpg](../icml_results/5243_FedUSD_%20Unbiased%20Synthetic%20Data%20for%20Federated%20Learning/tables/26bedc239ab8e20a0549f48a968dfcd5e370597685cd58afe4826a50f8d2ddf7.jpg)

![5473ce7aaa99abe95356ebd85e293804b3c0c72f5c2cd34eee471a0e4ade1b37.jpg](../icml_results/5243_FedUSD_%20Unbiased%20Synthetic%20Data%20for%20Federated%20Learning/tables/5473ce7aaa99abe95356ebd85e293804b3c0c72f5c2cd34eee471a0e4ade1b37.jpg)

![8a805caf5fbb3b321877ef73e7d0f6249ea3ede6deeaf58d790f6bddc6659b92.jpg](../icml_results/5243_FedUSD_%20Unbiased%20Synthetic%20Data%20for%20Federated%20Learning/tables/8a805caf5fbb3b321877ef73e7d0f6249ea3ede6deeaf58d790f6bddc6659b92.jpg)

![aace831b5520b07f3b11942cef1f0a838873b0a1541ca0e609c97482f71a6be9.jpg](../icml_results/5243_FedUSD_%20Unbiased%20Synthetic%20Data%20for%20Federated%20Learning/tables/aace831b5520b07f3b11942cef1f0a838873b0a1541ca0e609c97482f71a6be9.jpg)

![b08d939b5a4f7e99b6bf52c8c6b161f3870cf18c596886c95e60f52b5bef775a.jpg](../icml_results/5243_FedUSD_%20Unbiased%20Synthetic%20Data%20for%20Federated%20Learning/tables/b08d939b5a4f7e99b6bf52c8c6b161f3870cf18c596886c95e60f52b5bef775a.jpg)

![b7be486441989562f19597e4ebdeb2a218f24b5ac63d7f6c6964be0cef8c2462.jpg](../icml_results/5243_FedUSD_%20Unbiased%20Synthetic%20Data%20for%20Federated%20Learning/tables/b7be486441989562f19597e4ebdeb2a218f24b5ac63d7f6c6964be0cef8c2462.jpg)

<a id="paper-5196"></a>

## Finding the Correct Visual Evidence Without Forgetting: Mitigating Hallucination in LVLMs via Inter-Layer Visual Attention Discrepancy


### Images

![0e9c982a7e47761337eb0a0683e9897593b6e447deb807a55a2fb9783966dde4.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/images/0e9c982a7e47761337eb0a0683e9897593b6e447deb807a55a2fb9783966dde4.jpg)

![1a1e30204e406f7097c34b743ef40849ae60253851b68ceaf838e9b5f5fe15dc.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/images/1a1e30204e406f7097c34b743ef40849ae60253851b68ceaf838e9b5f5fe15dc.jpg)

![4d3e59a1b1f41bcf5fa6f33950457c86a069b89cf0592890da99f74f8c2c7853.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/images/4d3e59a1b1f41bcf5fa6f33950457c86a069b89cf0592890da99f74f8c2c7853.jpg)

![5a0265588193ec184b643d4674be92de77e75fe1923c47ecbda01d8dd7ea597b.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/images/5a0265588193ec184b643d4674be92de77e75fe1923c47ecbda01d8dd7ea597b.jpg)

![5bb6a08c6904b25395eeb89e5c9a000b801e9c88402bb0f48f62d5d54a067091.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/images/5bb6a08c6904b25395eeb89e5c9a000b801e9c88402bb0f48f62d5d54a067091.jpg)

![61c52100f737a579814ec5ccaa2b04a5b67fc92ff01d02f298df39703a8fe0b4.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/images/61c52100f737a579814ec5ccaa2b04a5b67fc92ff01d02f298df39703a8fe0b4.jpg)

![7d9cc300fd2cab28388163b17cde19b6718cea5879a37cf5998d00d4749972fe.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/images/7d9cc300fd2cab28388163b17cde19b6718cea5879a37cf5998d00d4749972fe.jpg)

![a35b957c34f27c2df71f89ee7b0903f81fb6d0a7330cb4e95049a20849fa3643.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/images/a35b957c34f27c2df71f89ee7b0903f81fb6d0a7330cb4e95049a20849fa3643.jpg)

![ab99c0a58469fedc4f8d3fb51066b57a56eb6c5c7976019d8e46c57b67990616.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/images/ab99c0a58469fedc4f8d3fb51066b57a56eb6c5c7976019d8e46c57b67990616.jpg)

### Tables

![181ce1dbd1a8d2a3d65adb377bff5a15b446c71d59ccad1539fd92d9925ab118.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/tables/181ce1dbd1a8d2a3d65adb377bff5a15b446c71d59ccad1539fd92d9925ab118.jpg)

![39e8a697af4d24b6db58b37c19f14bff0d896bfb38d1b87c4c2359112093a665.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/tables/39e8a697af4d24b6db58b37c19f14bff0d896bfb38d1b87c4c2359112093a665.jpg)

![63b81ee2e5ee323c48133908d7107210d5a146cd68a49fb2d80ad72fefa8b279.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/tables/63b81ee2e5ee323c48133908d7107210d5a146cd68a49fb2d80ad72fefa8b279.jpg)

![8eb7e66f230b1bdfc877fcbbadb6f3a95022f37a465152fb3782083d64de5a7f.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/tables/8eb7e66f230b1bdfc877fcbbadb6f3a95022f37a465152fb3782083d64de5a7f.jpg)

![91e83c836f7d16a818c232e1e8f805cf2f4cbb59bdc8ca9ec0f31fac6ca4d133.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/tables/91e83c836f7d16a818c232e1e8f805cf2f4cbb59bdc8ca9ec0f31fac6ca4d133.jpg)

![92e80579698edda612a19e625382a81cd8d67136f358d292b633fa48a10857bf.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/tables/92e80579698edda612a19e625382a81cd8d67136f358d292b633fa48a10857bf.jpg)

![ab8a4bd27b8eec6514de41bfa5488fdf5a5f73e44e89b77a17589f1312f34953.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/tables/ab8a4bd27b8eec6514de41bfa5488fdf5a5f73e44e89b77a17589f1312f34953.jpg)

![e410345f0247a49e66505862b3b304037a3bed0e9a2bd4701d8bac3ae6ba06db.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/tables/e410345f0247a49e66505862b3b304037a3bed0e9a2bd4701d8bac3ae6ba06db.jpg)

![fc6baa9ac3c8721dfd401f0f8575edd08fad9a3df96a3eb836189d5e12078c6e.jpg](../icml_results/5244_Finding%20the%20Correct%20Visual%20Evidence%20Without%20Forgetting_%20Mitigating%20Hallucination%20in%20LVLMs%20via%20Inter-/tables/fc6baa9ac3c8721dfd401f0f8575edd08fad9a3df96a3eb836189d5e12078c6e.jpg)

<a id="paper-5197"></a>

## Enhanced Latent-Space Adversarial Training for Super-Resolution


### Images

![00b7783c9dc454dde496d73822551a8de74d71fbbad37a5e4823014655747af9.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/images/00b7783c9dc454dde496d73822551a8de74d71fbbad37a5e4823014655747af9.jpg)

![0354572c1caa820df63cfb0cd1848281ee03a353af8d9f2d10e100feaaa5b6a3.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/images/0354572c1caa820df63cfb0cd1848281ee03a353af8d9f2d10e100feaaa5b6a3.jpg)

![057454bb73a4001c9b45e4f1818bff617de28e88c45f62900ed577ae5662301e.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/images/057454bb73a4001c9b45e4f1818bff617de28e88c45f62900ed577ae5662301e.jpg)

![1f8d2713b352e945d1184ed20aa214600f881906da082d959b779a636aba804e.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/images/1f8d2713b352e945d1184ed20aa214600f881906da082d959b779a636aba804e.jpg)

![2c247040ba0caaf9a654fb794f5f491d5e18f92f4d38d2fa02e326b02efb9010.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/images/2c247040ba0caaf9a654fb794f5f491d5e18f92f4d38d2fa02e326b02efb9010.jpg)

![2f52afcbe860ebaa4d8fa10e0e770e75735f1ff1ed9adbd02cda6581a95ff99b.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/images/2f52afcbe860ebaa4d8fa10e0e770e75735f1ff1ed9adbd02cda6581a95ff99b.jpg)

![3992bca52dc1555322e5daf40824b084aec72484a4537426eda8d1b30b603a13.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/images/3992bca52dc1555322e5daf40824b084aec72484a4537426eda8d1b30b603a13.jpg)

![4dfbc4b7cb9b4b0bc1fb743416ebb99b0a552018c68cec121f8c6cbaa467c6d2.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/images/4dfbc4b7cb9b4b0bc1fb743416ebb99b0a552018c68cec121f8c6cbaa467c6d2.jpg)

![674281318c3aaeb020d05841108cd0f6cbb4606138ca4ffae250ed1fb86ede6c.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/images/674281318c3aaeb020d05841108cd0f6cbb4606138ca4ffae250ed1fb86ede6c.jpg)

![6927e55e85e40fc345fefc64713a0a7eeaefc0f25ec1de1d083a53dfd29a0201.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/images/6927e55e85e40fc345fefc64713a0a7eeaefc0f25ec1de1d083a53dfd29a0201.jpg)

![7c80d8042a5065961856acffebee25a0595670ea1cad0c806c791ab370c368c2.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/images/7c80d8042a5065961856acffebee25a0595670ea1cad0c806c791ab370c368c2.jpg)

![829c64ad7b6aa3a63a5fad890a11a911c6785a502e587e699faa70cb153ff687.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/images/829c64ad7b6aa3a63a5fad890a11a911c6785a502e587e699faa70cb153ff687.jpg)

![b54088bf734ab3703b1644a88892962738c78e6f473712bbd7f9a2f94bf8c914.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/images/b54088bf734ab3703b1644a88892962738c78e6f473712bbd7f9a2f94bf8c914.jpg)

![b8e95c4c3e20d42baefdaa55666ddf4557d72de4d457b3710153720e81bad09b.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/images/b8e95c4c3e20d42baefdaa55666ddf4557d72de4d457b3710153720e81bad09b.jpg)

![b9436e76f2088f0477bb13e5329eeae5011b8bdc8b46bad969ed35a2facdce90.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/images/b9436e76f2088f0477bb13e5329eeae5011b8bdc8b46bad969ed35a2facdce90.jpg)

![cb2a87a73aa0b2b44172ff47c8822bc73a95f0417e113e26c85cc167d31930ce.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/images/cb2a87a73aa0b2b44172ff47c8822bc73a95f0417e113e26c85cc167d31930ce.jpg)

### Tables

![1674a3df916e3f0e438140ae18e75186df6f8479808f1bae593dee3e53930939.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/tables/1674a3df916e3f0e438140ae18e75186df6f8479808f1bae593dee3e53930939.jpg)

![66856a821fd0a6f149264b30b41aa2d76449cf6219047c9865c4a98b35d62c00.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/tables/66856a821fd0a6f149264b30b41aa2d76449cf6219047c9865c4a98b35d62c00.jpg)

![727301996ae150e90fd7654633537cc67c33a7d35da2cca153186db138c22303.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/tables/727301996ae150e90fd7654633537cc67c33a7d35da2cca153186db138c22303.jpg)

![a001d297836729580fb7f175e126a2c989f1180b3abbc0b582b0fe06e0790d51.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/tables/a001d297836729580fb7f175e126a2c989f1180b3abbc0b582b0fe06e0790d51.jpg)

![bdf3e3728df2a51ce360bb8bd959110a7f156869b3450d379256cff2bd7401bf.jpg](../icml_results/5245_Enhanced%20Latent-Space%20Adversarial%20Training%20for%20Super-Resolution/tables/bdf3e3728df2a51ce360bb8bd959110a7f156869b3450d379256cff2bd7401bf.jpg)

<a id="paper-5198"></a>

## TeamTR: Trust-Region Fine-Tuning for Multi-Agent LLM Coordination


### Images

![153fb63628bd505dcb99cc335ffdfed7edfac16ba611be861b975367c597a8f6.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/images/153fb63628bd505dcb99cc335ffdfed7edfac16ba611be861b975367c597a8f6.jpg)

![56fade5e5c08b0a5bdf619b0448d9c6c28b015c10771ddf3d0705d7562ab3c07.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/images/56fade5e5c08b0a5bdf619b0448d9c6c28b015c10771ddf3d0705d7562ab3c07.jpg)

![7984d8851964a872f7e04db26c4c7110296602b21011c41c25d7bc0bd37e4da2.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/images/7984d8851964a872f7e04db26c4c7110296602b21011c41c25d7bc0bd37e4da2.jpg)

![af6f89d67d053f1aed0a749a99ff4e4dc698c2ac7ca2c66fd081dd01840f695c.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/images/af6f89d67d053f1aed0a749a99ff4e4dc698c2ac7ca2c66fd081dd01840f695c.jpg)

![affc991dc2b5b72bb113bb5cc005ba5001ae9a3680a58fcbfa8faece07f40d1b.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/images/affc991dc2b5b72bb113bb5cc005ba5001ae9a3680a58fcbfa8faece07f40d1b.jpg)

![d21c36452a472a11c58c1b593ec2961bd695a37e825919f6ce6165fa826e69a9.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/images/d21c36452a472a11c58c1b593ec2961bd695a37e825919f6ce6165fa826e69a9.jpg)

![f3a8d3d4c43bcda0d91d024109d871f624cbe541376ec9adca03df17ec1cb4cc.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/images/f3a8d3d4c43bcda0d91d024109d871f624cbe541376ec9adca03df17ec1cb4cc.jpg)

### Tables

![16b615bad6a0adac337c9bfd5acc12b7161dd525adf361212b18a9c4121e31cd.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/tables/16b615bad6a0adac337c9bfd5acc12b7161dd525adf361212b18a9c4121e31cd.jpg)

![3148a3bcb394f2b0e1e351420fb4266a446a04ce8ca58c649a88187bbc0df9b5.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/tables/3148a3bcb394f2b0e1e351420fb4266a446a04ce8ca58c649a88187bbc0df9b5.jpg)

![3d5b1a1ec9721de6461d76d6bef90963274012821ccbf4e18db0f3e7b2deafa8.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/tables/3d5b1a1ec9721de6461d76d6bef90963274012821ccbf4e18db0f3e7b2deafa8.jpg)

![42e09d7a2518d37f7cb1f05a2764248ff1fa847bb0e5d7dca8e6e635feed03ff.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/tables/42e09d7a2518d37f7cb1f05a2764248ff1fa847bb0e5d7dca8e6e635feed03ff.jpg)

![4be8899d68450035a5cbf93fa2441e3a1f65587374d9a2003bcecd2b701846c1.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/tables/4be8899d68450035a5cbf93fa2441e3a1f65587374d9a2003bcecd2b701846c1.jpg)

![657ae9aab8727cc05ba583efc4b4a1c69cd6616165314e33cef774c8828b0a21.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/tables/657ae9aab8727cc05ba583efc4b4a1c69cd6616165314e33cef774c8828b0a21.jpg)

![667aa269dd01c8c09e77ab3c232523e7aa80cb22f6e8d309449025f1a67af421.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/tables/667aa269dd01c8c09e77ab3c232523e7aa80cb22f6e8d309449025f1a67af421.jpg)

![905fd6e404de4d0b3e74cac2da16d54198029654d8a268c6a0b67c2d7edd8630.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/tables/905fd6e404de4d0b3e74cac2da16d54198029654d8a268c6a0b67c2d7edd8630.jpg)

![b4b97f592766a19829b0bbc4274627bbb9b7b93581ec8204cb94206e2131eda0.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/tables/b4b97f592766a19829b0bbc4274627bbb9b7b93581ec8204cb94206e2131eda0.jpg)

![c182387023c17c829453374824aca0d76310160a3ed0e580044ad961ebfc40ba.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/tables/c182387023c17c829453374824aca0d76310160a3ed0e580044ad961ebfc40ba.jpg)

![e7de1f07d4fa7f07808b214e7b00ea6827617ab612cffca7a0e3969780de0d21.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/tables/e7de1f07d4fa7f07808b214e7b00ea6827617ab612cffca7a0e3969780de0d21.jpg)

![eae841897fba27e5406fea1261c2bf56f0af98b50af3e6f6c76302131d2e47d2.jpg](../icml_results/5246_TeamTR_%20Trust-Region%20Fine-Tuning%20for%20Multi-Agent%20LLM%20Coordination/tables/eae841897fba27e5406fea1261c2bf56f0af98b50af3e6f6c76302131d2e47d2.jpg)

<a id="paper-5199"></a>

## Hybrid-Gym: Training Coding Agents to Generalize Across Tasks


### Images

![03a7273884667b12a9139dfd859a6172b0ca51f22fca8884a6d32932afac37ff.jpg](../icml_results/5247_Hybrid-Gym_%20Training%20Coding%20Agents%20to%20Generalize%20Across%20Tasks/images/03a7273884667b12a9139dfd859a6172b0ca51f22fca8884a6d32932afac37ff.jpg)

![1805b93d51bdf2164f4faee83227245cba4d407ddc4e15cbaa8dcad8b3a304f0.jpg](../icml_results/5247_Hybrid-Gym_%20Training%20Coding%20Agents%20to%20Generalize%20Across%20Tasks/images/1805b93d51bdf2164f4faee83227245cba4d407ddc4e15cbaa8dcad8b3a304f0.jpg)

![646cf3759035a902654e890dadfa8c1b40fe796b44a7b1ee5da7eba6b1b00c86.jpg](../icml_results/5247_Hybrid-Gym_%20Training%20Coding%20Agents%20to%20Generalize%20Across%20Tasks/images/646cf3759035a902654e890dadfa8c1b40fe796b44a7b1ee5da7eba6b1b00c86.jpg)

![75b54677d8966e042a5841b3db681d0acb4df9b2db41cff9845b04debe2cd881.jpg](../icml_results/5247_Hybrid-Gym_%20Training%20Coding%20Agents%20to%20Generalize%20Across%20Tasks/images/75b54677d8966e042a5841b3db681d0acb4df9b2db41cff9845b04debe2cd881.jpg)

![bc7ed7cc35bc2dc3e3543da9ca9fc1fe993dfbb859ea3bf1b8a15687c87c4942.jpg](../icml_results/5247_Hybrid-Gym_%20Training%20Coding%20Agents%20to%20Generalize%20Across%20Tasks/images/bc7ed7cc35bc2dc3e3543da9ca9fc1fe993dfbb859ea3bf1b8a15687c87c4942.jpg)

### Tables

![178b9229d313145d6435c8ffeff1e31a4a4c26ec8072a4fa4863c8ec150cbed6.jpg](../icml_results/5247_Hybrid-Gym_%20Training%20Coding%20Agents%20to%20Generalize%20Across%20Tasks/tables/178b9229d313145d6435c8ffeff1e31a4a4c26ec8072a4fa4863c8ec150cbed6.jpg)

![27d9bb6848661cf9b9fcd5a14d9ff5850952720ede68ee2e99417bc6cd7d99f3.jpg](../icml_results/5247_Hybrid-Gym_%20Training%20Coding%20Agents%20to%20Generalize%20Across%20Tasks/tables/27d9bb6848661cf9b9fcd5a14d9ff5850952720ede68ee2e99417bc6cd7d99f3.jpg)

![61492ad6bc7ca7e6b892cc243551de6ac1095108ff4b9bdb07bb8c0e5a948984.jpg](../icml_results/5247_Hybrid-Gym_%20Training%20Coding%20Agents%20to%20Generalize%20Across%20Tasks/tables/61492ad6bc7ca7e6b892cc243551de6ac1095108ff4b9bdb07bb8c0e5a948984.jpg)

![81ec569c72d97e9bb93d9934252af705c1a37233faaaedddf432bd24829b3eea.jpg](../icml_results/5247_Hybrid-Gym_%20Training%20Coding%20Agents%20to%20Generalize%20Across%20Tasks/tables/81ec569c72d97e9bb93d9934252af705c1a37233faaaedddf432bd24829b3eea.jpg)

![b9acb2dc0ae750032c33586404d308358e17352a74e4628115f3a496c4e724cf.jpg](../icml_results/5247_Hybrid-Gym_%20Training%20Coding%20Agents%20to%20Generalize%20Across%20Tasks/tables/b9acb2dc0ae750032c33586404d308358e17352a74e4628115f3a496c4e724cf.jpg)

![d0b2580122708e466d71653b14137f8c27d52f6276bc1ea8a854b9914210fe0e.jpg](../icml_results/5247_Hybrid-Gym_%20Training%20Coding%20Agents%20to%20Generalize%20Across%20Tasks/tables/d0b2580122708e466d71653b14137f8c27d52f6276bc1ea8a854b9914210fe0e.jpg)

![e8bd2366cedf66b4e0674477d261124dcdbfca8443568cd5622bfe4af07eddfc.jpg](../icml_results/5247_Hybrid-Gym_%20Training%20Coding%20Agents%20to%20Generalize%20Across%20Tasks/tables/e8bd2366cedf66b4e0674477d261124dcdbfca8443568cd5622bfe4af07eddfc.jpg)

<a id="paper-5200"></a>

## Where Concept Erasure Should Occur: Concept–Layer Alignment in Text-to-Video Diffusion Models


### Images

![013a4d966522d7bb95bd71ec002e74ecf037b809b523917063ecce32a64324c9.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/images/013a4d966522d7bb95bd71ec002e74ecf037b809b523917063ecce32a64324c9.jpg)

![39eb852a60da9fc6d4ea109b4900d095384795d35f5a730e910b6735b188ab31.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/images/39eb852a60da9fc6d4ea109b4900d095384795d35f5a730e910b6735b188ab31.jpg)

![4203af26f79edd4e59634b201f141c56fd033b4cf56067e7acdc2c6fb668f9e9.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/images/4203af26f79edd4e59634b201f141c56fd033b4cf56067e7acdc2c6fb668f9e9.jpg)

![483b31b909bfa86bcb67ed05f966e7b3446694cacc7f8a09862220368bd26e82.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/images/483b31b909bfa86bcb67ed05f966e7b3446694cacc7f8a09862220368bd26e82.jpg)

![548a78f47411772be12131bcabefaa1932cac6137131e1d58952cd27db943ca8.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/images/548a78f47411772be12131bcabefaa1932cac6137131e1d58952cd27db943ca8.jpg)

![8f8382cc63c7840db723ace2fd0ab6c47dab27505495439d61ed34a544c80ef6.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/images/8f8382cc63c7840db723ace2fd0ab6c47dab27505495439d61ed34a544c80ef6.jpg)

![ac638345a7524f50b6fe3f78727e01bd43cfecbd742fbd90f6581b441ab81679.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/images/ac638345a7524f50b6fe3f78727e01bd43cfecbd742fbd90f6581b441ab81679.jpg)

![ad80e945c1395f21882da9fa88a4238f47cbf545eedf09e25edf1f854161a54e.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/images/ad80e945c1395f21882da9fa88a4238f47cbf545eedf09e25edf1f854161a54e.jpg)

![ee9cdcdb9e3542a367b34b297a6c54bef248303c6341508b181ae51dce5868a9.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/images/ee9cdcdb9e3542a367b34b297a6c54bef248303c6341508b181ae51dce5868a9.jpg)

### Tables

![01f5bf33ac7886cec58032abb1f451860055d0735d295d7273e559eebac8ba2f.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/01f5bf33ac7886cec58032abb1f451860055d0735d295d7273e559eebac8ba2f.jpg)

![02af083b318d895b88890a4d3e70ad4c2ef9de332cb57647dccec0afd1fc1324.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/02af083b318d895b88890a4d3e70ad4c2ef9de332cb57647dccec0afd1fc1324.jpg)

![06c926c8f4b02024ea1fa614303cebe5958278007cd3be964edaf5aa7a67a8bd.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/06c926c8f4b02024ea1fa614303cebe5958278007cd3be964edaf5aa7a67a8bd.jpg)

![096d054b5e549c23bce780a34cc6fbc1601fb51361b1e923ebf31fedf97f5514.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/096d054b5e549c23bce780a34cc6fbc1601fb51361b1e923ebf31fedf97f5514.jpg)

![0bccb543e1cc43a15f2efd361056506095301935fba636c99642fbb323d79d8f.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/0bccb543e1cc43a15f2efd361056506095301935fba636c99642fbb323d79d8f.jpg)

![0d3db376532faa8be585eb18889df56e6f97d8d806a619a0ce70826b746ddef2.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/0d3db376532faa8be585eb18889df56e6f97d8d806a619a0ce70826b746ddef2.jpg)

![162a9254dfa0aeebfc067c38cfaf06930eff2a965de80dc7e8f3107fd33fe37f.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/162a9254dfa0aeebfc067c38cfaf06930eff2a965de80dc7e8f3107fd33fe37f.jpg)

![17fa1de63e8a40e703db8e4eaf1df35a83945a5c3e68076dee43b9ddde0d01b4.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/17fa1de63e8a40e703db8e4eaf1df35a83945a5c3e68076dee43b9ddde0d01b4.jpg)

![3ad504b122bc249ec6c8f17558963d2d58fe70f56f9f4ec64b93ee99b3a12b03.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/3ad504b122bc249ec6c8f17558963d2d58fe70f56f9f4ec64b93ee99b3a12b03.jpg)

![57e0b2715c0afd0ab4aa97619ad2c6dcf661b917a989ff0bf65fcf51714de1cc.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/57e0b2715c0afd0ab4aa97619ad2c6dcf661b917a989ff0bf65fcf51714de1cc.jpg)

![607e5a7fcedd6dc4f0d9bea7a6e32b846f27889f009c1ffc7858b828f96f0f2e.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/607e5a7fcedd6dc4f0d9bea7a6e32b846f27889f009c1ffc7858b828f96f0f2e.jpg)

![67f0c570f2833d78ef292926d626d3face6466699861b1064d2e84adaa8e841a.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/67f0c570f2833d78ef292926d626d3face6466699861b1064d2e84adaa8e841a.jpg)

![7787bc14033ba052938672d725076a29e6e15103fe8641667a80237146212bdb.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/7787bc14033ba052938672d725076a29e6e15103fe8641667a80237146212bdb.jpg)

![78b2fa0a5e39249f3333b25a163d770ccd364155a2be3de3f2d27518a3ec8ff2.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/78b2fa0a5e39249f3333b25a163d770ccd364155a2be3de3f2d27518a3ec8ff2.jpg)

![79ec6446dc68314bffba118c974f66a6e0e68c9ab14612a79f8969c3e9179a63.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/79ec6446dc68314bffba118c974f66a6e0e68c9ab14612a79f8969c3e9179a63.jpg)

![83c1fa575471fd281d5c847c13ce617099051a7a70aee23041d251de2a21b48b.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/83c1fa575471fd281d5c847c13ce617099051a7a70aee23041d251de2a21b48b.jpg)

![8a4324c083cf7787260404126ff7d509f6692cb01be1328a3fdf809865198dc2.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/8a4324c083cf7787260404126ff7d509f6692cb01be1328a3fdf809865198dc2.jpg)

![8a4d91b8ec85259c9645398197d3a710411b43a3e57aea4406e8ffae953a7ad4.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/8a4d91b8ec85259c9645398197d3a710411b43a3e57aea4406e8ffae953a7ad4.jpg)

![998406fd8c0837de9c99a4b6564bdc09a3a2750dc095d41dcaa8e58564da9bee.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/998406fd8c0837de9c99a4b6564bdc09a3a2750dc095d41dcaa8e58564da9bee.jpg)

![9abc8ec9dc6d8dd4ffa394c6e50199337b02cca43a781f00dd9784e7ab392bc8.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/9abc8ec9dc6d8dd4ffa394c6e50199337b02cca43a781f00dd9784e7ab392bc8.jpg)

![a000afc00c80124e05ad11d4eb7fa00d44dc8f692cead7aa6f05838147db1344.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/a000afc00c80124e05ad11d4eb7fa00d44dc8f692cead7aa6f05838147db1344.jpg)

![a20b8cda5848cab63739c51fa634b0d5602b66a00aa7498f10f45264bc8c9122.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/a20b8cda5848cab63739c51fa634b0d5602b66a00aa7498f10f45264bc8c9122.jpg)

![a36c563b954bf4b854b6f911ab3f220a8171c693dfe54a6f3102222d1aae6170.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/a36c563b954bf4b854b6f911ab3f220a8171c693dfe54a6f3102222d1aae6170.jpg)

![b55d7dc01ac7c32341eb09b11fc06588ecc279d2d320b4d1c8771f3d7bc2981f.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/b55d7dc01ac7c32341eb09b11fc06588ecc279d2d320b4d1c8771f3d7bc2981f.jpg)

![b9a9d343b8ad210373a7724fc68b9598d33bd03e67965446e88a1a50a4d7c65f.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/b9a9d343b8ad210373a7724fc68b9598d33bd03e67965446e88a1a50a4d7c65f.jpg)

![df01445bfdc0d77a0892d59002f9e45026c72953efdbc7f01d6bc0ec592b6048.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/df01445bfdc0d77a0892d59002f9e45026c72953efdbc7f01d6bc0ec592b6048.jpg)

![f51b948f75e78237896caf30464d2a5e72dc8a976219a89460d3452bcce8dea0.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/f51b948f75e78237896caf30464d2a5e72dc8a976219a89460d3452bcce8dea0.jpg)

![fb060f352bcfae888f4533d4f4368740e478dc717ec5d54d7244cba2f26ec709.jpg](../icml_results/5248_Where%20Concept%20Erasure%20Should%20Occur_%20Concept%E2%80%93Layer%20Alignment%20in%20Text-to-Video%20Diffusion%20Models/tables/fb060f352bcfae888f4533d4f4368740e478dc717ec5d54d7244cba2f26ec709.jpg)

<a id="paper-5201"></a>

## BESplit: Bias-Compensated Split Federated Learning with Evidential Aggregation


### Images

![03e2673c9a357d89265b2b0a3d7409247693106aedb829a46efa99a9feeb7e33.jpg](../icml_results/5249_BESplit_%20Bias-Compensated%20Split%20Federated%20Learning%20with%20Evidential%20Aggregation/images/03e2673c9a357d89265b2b0a3d7409247693106aedb829a46efa99a9feeb7e33.jpg)

![0bf3a8502970a0f32d1d028bb0ad7d4c8da4dfb91e693acbf12409775ae7aa09.jpg](../icml_results/5249_BESplit_%20Bias-Compensated%20Split%20Federated%20Learning%20with%20Evidential%20Aggregation/images/0bf3a8502970a0f32d1d028bb0ad7d4c8da4dfb91e693acbf12409775ae7aa09.jpg)

![18b93a27b8fd77d9e10d2574464732106e844f18075ee7dc28581c0bc93156e1.jpg](../icml_results/5249_BESplit_%20Bias-Compensated%20Split%20Federated%20Learning%20with%20Evidential%20Aggregation/images/18b93a27b8fd77d9e10d2574464732106e844f18075ee7dc28581c0bc93156e1.jpg)

![b28f3732a36078db5e3600349b2842ddf21e46550a368d622f4c6f7ab148e67b.jpg](../icml_results/5249_BESplit_%20Bias-Compensated%20Split%20Federated%20Learning%20with%20Evidential%20Aggregation/images/b28f3732a36078db5e3600349b2842ddf21e46550a368d622f4c6f7ab148e67b.jpg)

![ba8ba30bb76b7801f070fcead64aef75628d03947adae533e15171b16ca90978.jpg](../icml_results/5249_BESplit_%20Bias-Compensated%20Split%20Federated%20Learning%20with%20Evidential%20Aggregation/images/ba8ba30bb76b7801f070fcead64aef75628d03947adae533e15171b16ca90978.jpg)

![e46bcaaa421078df87f25e6968b66bf74a022535c630d54b35b12f1dc7e781dd.jpg](../icml_results/5249_BESplit_%20Bias-Compensated%20Split%20Federated%20Learning%20with%20Evidential%20Aggregation/images/e46bcaaa421078df87f25e6968b66bf74a022535c630d54b35b12f1dc7e781dd.jpg)

![e88b605fcf50089925599c192bcc82c8bc7e62196ccb0a483ce4b08550cf008d.jpg](../icml_results/5249_BESplit_%20Bias-Compensated%20Split%20Federated%20Learning%20with%20Evidential%20Aggregation/images/e88b605fcf50089925599c192bcc82c8bc7e62196ccb0a483ce4b08550cf008d.jpg)

![ed2ee25c389605ba08e944d7e3a948da81b08de07d16f218b5bdf816819ad216.jpg](../icml_results/5249_BESplit_%20Bias-Compensated%20Split%20Federated%20Learning%20with%20Evidential%20Aggregation/images/ed2ee25c389605ba08e944d7e3a948da81b08de07d16f218b5bdf816819ad216.jpg)

### Tables

![2cd1bc3f9a35a3f01082827bb357a3e2d0a5a29374f7aed0e9b836f9297a4f49.jpg](../icml_results/5249_BESplit_%20Bias-Compensated%20Split%20Federated%20Learning%20with%20Evidential%20Aggregation/tables/2cd1bc3f9a35a3f01082827bb357a3e2d0a5a29374f7aed0e9b836f9297a4f49.jpg)

![618d4d499d6d9215281ac9377693b59ae6e1b4bcffbb394c688aaa1b01ca87a9.jpg](../icml_results/5249_BESplit_%20Bias-Compensated%20Split%20Federated%20Learning%20with%20Evidential%20Aggregation/tables/618d4d499d6d9215281ac9377693b59ae6e1b4bcffbb394c688aaa1b01ca87a9.jpg)

![640ebf56090499f0056070ee36d9aec74385f644181ea393965ff4c32ec4e783.jpg](../icml_results/5249_BESplit_%20Bias-Compensated%20Split%20Federated%20Learning%20with%20Evidential%20Aggregation/tables/640ebf56090499f0056070ee36d9aec74385f644181ea393965ff4c32ec4e783.jpg)

![8b745928d5b9b289a657063533e7e80e71aa59a4d119f39f54259c658eae5df8.jpg](../icml_results/5249_BESplit_%20Bias-Compensated%20Split%20Federated%20Learning%20with%20Evidential%20Aggregation/tables/8b745928d5b9b289a657063533e7e80e71aa59a4d119f39f54259c658eae5df8.jpg)

![a6addaea04065af3654c0063df930dc0ec5ae1da65489a685c340ac876103f2a.jpg](../icml_results/5249_BESplit_%20Bias-Compensated%20Split%20Federated%20Learning%20with%20Evidential%20Aggregation/tables/a6addaea04065af3654c0063df930dc0ec5ae1da65489a685c340ac876103f2a.jpg)

![abcd27054c46f446a96ae55fe556485f211279f457582b06433a4889143586d1.jpg](../icml_results/5249_BESplit_%20Bias-Compensated%20Split%20Federated%20Learning%20with%20Evidential%20Aggregation/tables/abcd27054c46f446a96ae55fe556485f211279f457582b06433a4889143586d1.jpg)

![bc7708769c8c60d84bae5ea59f3f8cdd8e837f2ef98ff65d9faf9aec35794802.jpg](../icml_results/5249_BESplit_%20Bias-Compensated%20Split%20Federated%20Learning%20with%20Evidential%20Aggregation/tables/bc7708769c8c60d84bae5ea59f3f8cdd8e837f2ef98ff65d9faf9aec35794802.jpg)

![ca84afcca2d05b8c901bb493b8bde13f71bf70975431ad01d56b49902bd3ab59.jpg](../icml_results/5249_BESplit_%20Bias-Compensated%20Split%20Federated%20Learning%20with%20Evidential%20Aggregation/tables/ca84afcca2d05b8c901bb493b8bde13f71bf70975431ad01d56b49902bd3ab59.jpg)

<a id="paper-5202"></a>

## Training-Free Multimodal Large Language Model Orchestration


### Images

![1a181ea93c8071609c89260a7d46b7f3817df980f869eef4c124d43eabf17224.jpg](../icml_results/5250_Training-Free%20Multimodal%20Large%20Language%20Model%20Orchestration/images/1a181ea93c8071609c89260a7d46b7f3817df980f869eef4c124d43eabf17224.jpg)

![644898f9e157963af57550d0bf14f57d650d753a1cddae35a20a35fedd040e5b.jpg](../icml_results/5250_Training-Free%20Multimodal%20Large%20Language%20Model%20Orchestration/images/644898f9e157963af57550d0bf14f57d650d753a1cddae35a20a35fedd040e5b.jpg)

![a4e48517b5c8bc29833740fd1894a662cf88851ff8fa7194be1d11db52b4cf2c.jpg](../icml_results/5250_Training-Free%20Multimodal%20Large%20Language%20Model%20Orchestration/images/a4e48517b5c8bc29833740fd1894a662cf88851ff8fa7194be1d11db52b4cf2c.jpg)

### Tables

![1cf73314a8b54f5c12724956cb525ad493d87c6d4a59c137046f804170ba356b.jpg](../icml_results/5250_Training-Free%20Multimodal%20Large%20Language%20Model%20Orchestration/tables/1cf73314a8b54f5c12724956cb525ad493d87c6d4a59c137046f804170ba356b.jpg)

![231d9dd9ff1bb36a2c44a00da2e84d65636c1a4b324a9f12aedf5954d6bbaf80.jpg](../icml_results/5250_Training-Free%20Multimodal%20Large%20Language%20Model%20Orchestration/tables/231d9dd9ff1bb36a2c44a00da2e84d65636c1a4b324a9f12aedf5954d6bbaf80.jpg)

![2496e9dce7813c5cb27d33039f161ee7175cfdfb991d6c4809408bed8403e01d.jpg](../icml_results/5250_Training-Free%20Multimodal%20Large%20Language%20Model%20Orchestration/tables/2496e9dce7813c5cb27d33039f161ee7175cfdfb991d6c4809408bed8403e01d.jpg)

![33423e2e842a929e40c9ca4ff8e2512b5dbdf61efedb53d440e7bd13c67508c2.jpg](../icml_results/5250_Training-Free%20Multimodal%20Large%20Language%20Model%20Orchestration/tables/33423e2e842a929e40c9ca4ff8e2512b5dbdf61efedb53d440e7bd13c67508c2.jpg)

![4bdd7b2e79b8d01e92c835e8b0c0dc03a3b4173724d6e65d885d09e39bd12046.jpg](../icml_results/5250_Training-Free%20Multimodal%20Large%20Language%20Model%20Orchestration/tables/4bdd7b2e79b8d01e92c835e8b0c0dc03a3b4173724d6e65d885d09e39bd12046.jpg)

![660a1a86bf2fefc7bea35023068813e966e5c5d810e0a24f68102660e0ea7705.jpg](../icml_results/5250_Training-Free%20Multimodal%20Large%20Language%20Model%20Orchestration/tables/660a1a86bf2fefc7bea35023068813e966e5c5d810e0a24f68102660e0ea7705.jpg)

![b60319e32ea9de1926894d0e5f4e61d84be751ccc12fbd4ce1b9ff64d9eda9f5.jpg](../icml_results/5250_Training-Free%20Multimodal%20Large%20Language%20Model%20Orchestration/tables/b60319e32ea9de1926894d0e5f4e61d84be751ccc12fbd4ce1b9ff64d9eda9f5.jpg)

![d857bb0f6421db67054df3a4e5f70d42fcb57869fe280e7a4c8ee12cd11995e6.jpg](../icml_results/5250_Training-Free%20Multimodal%20Large%20Language%20Model%20Orchestration/tables/d857bb0f6421db67054df3a4e5f70d42fcb57869fe280e7a4c8ee12cd11995e6.jpg)

<a id="paper-5203"></a>

## AREA: Attribute Extraction and Aggregation for CLIP-Based Class-Incremental Learning


### Images

![1545a7aeae2d1aad60b180d4517092ad95a4837045bace6da94845dca7bd8fbd.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/images/1545a7aeae2d1aad60b180d4517092ad95a4837045bace6da94845dca7bd8fbd.jpg)

![355f0d204f2ce837747192a5568d4a8f71161a125203bfef14510305f405de1a.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/images/355f0d204f2ce837747192a5568d4a8f71161a125203bfef14510305f405de1a.jpg)

![3c314e465b733bfb54b89d907caf384208e20deda1f8f5ac941edfaf704ae796.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/images/3c314e465b733bfb54b89d907caf384208e20deda1f8f5ac941edfaf704ae796.jpg)

![42a345b2dc308eba9af68ba3e8443fdf1b4d5e7e8914ad8181735d7674c0fce6.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/images/42a345b2dc308eba9af68ba3e8443fdf1b4d5e7e8914ad8181735d7674c0fce6.jpg)

![514081bb34ee3dd29593d2032c73fb855b66d4e47b61e0c03db6420f3776972f.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/images/514081bb34ee3dd29593d2032c73fb855b66d4e47b61e0c03db6420f3776972f.jpg)

![546d7057333ee5827492c12c5016a130ff428d796aaf4a2f4bb4ba3e29ddf6e6.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/images/546d7057333ee5827492c12c5016a130ff428d796aaf4a2f4bb4ba3e29ddf6e6.jpg)

![570c9ca62eef1d64711766c49d0c0e9e8bb0b1bfe15124652e3ad62e4bccef1d.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/images/570c9ca62eef1d64711766c49d0c0e9e8bb0b1bfe15124652e3ad62e4bccef1d.jpg)

![604e19de4f20cdbcc2ed529501dab6051a6b291a81100e0c3821177b794bc6d6.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/images/604e19de4f20cdbcc2ed529501dab6051a6b291a81100e0c3821177b794bc6d6.jpg)

![708f900334f31042232dcdbad72580c46fbeac47d9f749af383fa80d50d6a7e6.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/images/708f900334f31042232dcdbad72580c46fbeac47d9f749af383fa80d50d6a7e6.jpg)

![7d84929edbaa026720d88875e7ed11ad66838928026c256654ea5877830e649c.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/images/7d84929edbaa026720d88875e7ed11ad66838928026c256654ea5877830e649c.jpg)

![8778631b0bd8444d1701718c2dadf8e65a31d44148380324b622421c5e3ba708.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/images/8778631b0bd8444d1701718c2dadf8e65a31d44148380324b622421c5e3ba708.jpg)

### Tables

![0eeb42a3763adc0b11b6c994576306d50d7f77b0eb0aadb0ac9d4acb6a4b92fc.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/tables/0eeb42a3763adc0b11b6c994576306d50d7f77b0eb0aadb0ac9d4acb6a4b92fc.jpg)

![1beff723bd2c4cebb9cb390242e55715a559b2560f6155666b739614b5097fd6.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/tables/1beff723bd2c4cebb9cb390242e55715a559b2560f6155666b739614b5097fd6.jpg)

![2892f6237e10ee814574d9446fd2258378f4642a03255037c8e10b357268dd31.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/tables/2892f6237e10ee814574d9446fd2258378f4642a03255037c8e10b357268dd31.jpg)

![31ee876f4f73029d102ea725c47af3eff8cfac6341f6d5b0e2cc81641c561a39.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/tables/31ee876f4f73029d102ea725c47af3eff8cfac6341f6d5b0e2cc81641c561a39.jpg)

![868321a96c765732029ffacde172a5afe80c0138248b1b0bff90a0529a1a8294.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/tables/868321a96c765732029ffacde172a5afe80c0138248b1b0bff90a0529a1a8294.jpg)

![8ba76985ba10bb506367f475a7f70629a0b9afdacd856a156f864da422ef01cc.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/tables/8ba76985ba10bb506367f475a7f70629a0b9afdacd856a156f864da422ef01cc.jpg)

![e579ba08925a58b77898f0b099871bb86b13fe0f920a3b3e4071d6723a2eee01.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/tables/e579ba08925a58b77898f0b099871bb86b13fe0f920a3b3e4071d6723a2eee01.jpg)

![e72c722e2b7d53c40cde82e1e630350b2d2fea6c8a588df20cca022b8ea8882e.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/tables/e72c722e2b7d53c40cde82e1e630350b2d2fea6c8a588df20cca022b8ea8882e.jpg)

![f42f4719a2a8a9f1dc00af6f2a1d6f8429ed5e1dfe1a82e755b15744fceb023a.jpg](../icml_results/5251_AREA_%20Attribute%20Extraction%20and%20Aggregation%20for%20CLIP-Based%20Class-Incremental%20Learning/tables/f42f4719a2a8a9f1dc00af6f2a1d6f8429ed5e1dfe1a82e755b15744fceb023a.jpg)

<a id="paper-5204"></a>

## SAME: Stabilized Mixture-of-Experts for Multimodal Continual Instruction Tuning


### Images

![0af109d9aa52eaa9b0d9118ed84172d515c8d13fb5c7d418f381a5ad2f9d765b.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/0af109d9aa52eaa9b0d9118ed84172d515c8d13fb5c7d418f381a5ad2f9d765b.jpg)

![0c06884cc25ca6ebf2f786a99da50d27055e69db018e14f4070a926a1c62dff2.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/0c06884cc25ca6ebf2f786a99da50d27055e69db018e14f4070a926a1c62dff2.jpg)

![171a282b0f24a8e24defdd88ada2eeb7bc69692d99e29e088617d1e188255865.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/171a282b0f24a8e24defdd88ada2eeb7bc69692d99e29e088617d1e188255865.jpg)

![18dbabcc7ded0400edbfcf1b25babf3bae3ffdc9c5b4c275cc32419011539f64.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/18dbabcc7ded0400edbfcf1b25babf3bae3ffdc9c5b4c275cc32419011539f64.jpg)

![48287127b450f850672c7cd1fde8c7ecdcba03347cd423e4ef7c7dd45bf74ef8.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/48287127b450f850672c7cd1fde8c7ecdcba03347cd423e4ef7c7dd45bf74ef8.jpg)

![5d0e39ff66a40af8519da04f47d1dff561e3347123aeb231c2c68f9d1a42e416.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/5d0e39ff66a40af8519da04f47d1dff561e3347123aeb231c2c68f9d1a42e416.jpg)

![66bd4c82a4e4f5e545b569778f35caa47bcf201103ddf967c61120676fb6045f.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/66bd4c82a4e4f5e545b569778f35caa47bcf201103ddf967c61120676fb6045f.jpg)

![79c33002cae5c55d77a70d421856ef033114582f552c45db30470ac1f999fff9.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/79c33002cae5c55d77a70d421856ef033114582f552c45db30470ac1f999fff9.jpg)

![7b22dc6e845f1e05a3dd2592a41fc5727395373db93452e76c52a8c2196339c4.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/7b22dc6e845f1e05a3dd2592a41fc5727395373db93452e76c52a8c2196339c4.jpg)

![8fa970b7138cb77a3cea5033fd106bacbb873acf36cada7bb9908253c9188974.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/8fa970b7138cb77a3cea5033fd106bacbb873acf36cada7bb9908253c9188974.jpg)

![9215426767ba29465ed466192541749764b7dd5c3f8b2e44f51fe5ec3da8dcfb.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/9215426767ba29465ed466192541749764b7dd5c3f8b2e44f51fe5ec3da8dcfb.jpg)

![98e9f456a287be10fa7683621d32de499357ed846386e5f1ce06a83e728c8615.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/98e9f456a287be10fa7683621d32de499357ed846386e5f1ce06a83e728c8615.jpg)

![b08ee13944b900a4cf629e5fff68da6d38c22545ec46ad6300d917668153f218.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/b08ee13944b900a4cf629e5fff68da6d38c22545ec46ad6300d917668153f218.jpg)

![b32a3035cfa64be0391085e49a2700dd1a9cabb18084c70078a184c088169918.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/b32a3035cfa64be0391085e49a2700dd1a9cabb18084c70078a184c088169918.jpg)

![b6fdbc6a73d2a9607c0681581a0cdd63c1dcd5d32b7588e5aa0636efa0d23dce.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/b6fdbc6a73d2a9607c0681581a0cdd63c1dcd5d32b7588e5aa0636efa0d23dce.jpg)

![ccc1cbe1784558f71f490849a16b931ba8a38d45887ce0ba6dd5e85997212f3a.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/ccc1cbe1784558f71f490849a16b931ba8a38d45887ce0ba6dd5e85997212f3a.jpg)

![d624412a5b9e1babf815fc8ae8ff979cfa586ed582ff9d9ed899655d259e7388.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/d624412a5b9e1babf815fc8ae8ff979cfa586ed582ff9d9ed899655d259e7388.jpg)

![de7383f675e5e1c79a0527e4f18eb823f223f926f7afc15da9053ce38f5118e9.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/de7383f675e5e1c79a0527e4f18eb823f223f926f7afc15da9053ce38f5118e9.jpg)

![ee3e4841e862b299ada4880f3c891b8f2a536bde7286873fc605a5f68b73ea1e.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/images/ee3e4841e862b299ada4880f3c891b8f2a536bde7286873fc605a5f68b73ea1e.jpg)

### Tables

![122663405a26f40e38c181bef64035b4ab06e27ca1ab5f75a9bc390d0d0780f4.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/tables/122663405a26f40e38c181bef64035b4ab06e27ca1ab5f75a9bc390d0d0780f4.jpg)

![2bfdcf917623feca9ffb803dcbc3a9a22421f3a52198d8da2f37d00626fe8424.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/tables/2bfdcf917623feca9ffb803dcbc3a9a22421f3a52198d8da2f37d00626fe8424.jpg)

![33e11fd8581e134d1aaba307d6c36071f2cad7d9acf83b709bda056bb0f15175.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/tables/33e11fd8581e134d1aaba307d6c36071f2cad7d9acf83b709bda056bb0f15175.jpg)

![6cc6d59b6b2691f24af3edd1faef13254f53d148ef24a267a7f8f7b94659366a.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/tables/6cc6d59b6b2691f24af3edd1faef13254f53d148ef24a267a7f8f7b94659366a.jpg)

![70a89edda4f46da57f46e74176c111dfe3f2107dfc16579d41c316e39f5fe25e.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/tables/70a89edda4f46da57f46e74176c111dfe3f2107dfc16579d41c316e39f5fe25e.jpg)

![89eb18ca8122dab7c71f2a32927fd34855abb1cc1c2fcac14933cf2646edd502.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/tables/89eb18ca8122dab7c71f2a32927fd34855abb1cc1c2fcac14933cf2646edd502.jpg)

![a13347049cc1ef32e9cd32e8e51401cbeaaaede4095d6b5de55116887e70f759.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/tables/a13347049cc1ef32e9cd32e8e51401cbeaaaede4095d6b5de55116887e70f759.jpg)

![b7f7441bbc31e95e73bc4f415f6e1b929b2dacd17552126443bd10f2097a7a01.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/tables/b7f7441bbc31e95e73bc4f415f6e1b929b2dacd17552126443bd10f2097a7a01.jpg)

![ba77f62f44932ed006f63c1d0ea1536ce8da54880dd89c1d6854d0cf781f3639.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/tables/ba77f62f44932ed006f63c1d0ea1536ce8da54880dd89c1d6854d0cf781f3639.jpg)

![d8586da8a4377bf158166c43d8e813118c9d2d2b46ed85089e7985e415173ad0.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/tables/d8586da8a4377bf158166c43d8e813118c9d2d2b46ed85089e7985e415173ad0.jpg)

![daeedb9d28ee1c70525b616d2fe69a268b8c0ebbcc7be13b19c75d2f679357b6.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/tables/daeedb9d28ee1c70525b616d2fe69a268b8c0ebbcc7be13b19c75d2f679357b6.jpg)

![dca68175532ef8c7fe93982b04d4784f637b0fbada9ca846b61e7512d714fddb.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/tables/dca68175532ef8c7fe93982b04d4784f637b0fbada9ca846b61e7512d714fddb.jpg)

![e76a7990ea35c3a6f6bea905d9a13ef2ebce148f8d6fa7e25f2d66a9766ab408.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/tables/e76a7990ea35c3a6f6bea905d9a13ef2ebce148f8d6fa7e25f2d66a9766ab408.jpg)

![fad125866550a2c385742e2e74de5a8d157ff93e9545408a281f4a4b5579eed9.jpg](../icml_results/5252_SAME_%20Stabilized%20Mixture-of-Experts%20for%20Multimodal%20Continual%20Instruction%20Tuning/tables/fad125866550a2c385742e2e74de5a8d157ff93e9545408a281f4a4b5579eed9.jpg)

<a id="paper-5205"></a>

## Can LLMs Reason Like Automated Theorem Provers for Rust Verification? VCoT-Bench: Evaluating via Verification Chain of Thought


### Images

![01de618fb0d0f1c2eee3082222adc13eba2f4e4288e8bf7c2d294b75879ca0e3.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/images/01de618fb0d0f1c2eee3082222adc13eba2f4e4288e8bf7c2d294b75879ca0e3.jpg)

![15bedc519e66ada629991a67d616df52a900cb2f5483a0a380f18d4655d9f08b.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/images/15bedc519e66ada629991a67d616df52a900cb2f5483a0a380f18d4655d9f08b.jpg)

![1e8ebd74be44cf013bebabffad087278b3719f9eb7d9ea3ef223abf126481788.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/images/1e8ebd74be44cf013bebabffad087278b3719f9eb7d9ea3ef223abf126481788.jpg)

![4a078ecf79d5daaaebb310cc60e4ef68540c820d8c417de56a1f023a973a3152.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/images/4a078ecf79d5daaaebb310cc60e4ef68540c820d8c417de56a1f023a973a3152.jpg)

![50fac6b35db207d74c9639fa8578dcc609cf44c51e7985685269f1177e485b70.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/images/50fac6b35db207d74c9639fa8578dcc609cf44c51e7985685269f1177e485b70.jpg)

![550c13ad9e861e48255fc3397bf2070eb1ee50edbdf0f4b1fcd16bd02998729d.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/images/550c13ad9e861e48255fc3397bf2070eb1ee50edbdf0f4b1fcd16bd02998729d.jpg)

![64610dad11a2d06691a514160a29069bec03a73f4c81e190704ec9ac330bc0d4.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/images/64610dad11a2d06691a514160a29069bec03a73f4c81e190704ec9ac330bc0d4.jpg)

![9bc6ca2f7a4788b550ca93fa477bec333d7de6642c2edbe120aaa4e96f3c3643.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/images/9bc6ca2f7a4788b550ca93fa477bec333d7de6642c2edbe120aaa4e96f3c3643.jpg)

![abbfa943fadddb2137806d76bfa416ab81754a8935a9c2c35cace0430b0d45f0.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/images/abbfa943fadddb2137806d76bfa416ab81754a8935a9c2c35cace0430b0d45f0.jpg)

![c2014abe8f132107c0af463acccec1800ede77a0c2f18f9802adb9ff7830c05c.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/images/c2014abe8f132107c0af463acccec1800ede77a0c2f18f9802adb9ff7830c05c.jpg)

![cc9b761d5483f7038711d1f2ca738b828c4a64bb34bc3f121a9f562886251403.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/images/cc9b761d5483f7038711d1f2ca738b828c4a64bb34bc3f121a9f562886251403.jpg)

![e79b64e6053f474e1871eaa2d8111f024cf86f0c8a36def4773ad887ae46809f.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/images/e79b64e6053f474e1871eaa2d8111f024cf86f0c8a36def4773ad887ae46809f.jpg)

### Tables

![022f4a67d61f8e2655ada99abdc5ffd9ca795d62089aa7b17375d68851abb958.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/tables/022f4a67d61f8e2655ada99abdc5ffd9ca795d62089aa7b17375d68851abb958.jpg)

![16905eb0578b6590d44ca8b33414146c65efc81de6a12c5f7ef6e3feba34aa18.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/tables/16905eb0578b6590d44ca8b33414146c65efc81de6a12c5f7ef6e3feba34aa18.jpg)

![2981f72955f7e434f9d02894542fc16ae77b0dab2ef6cd1169d3a3a78391f520.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/tables/2981f72955f7e434f9d02894542fc16ae77b0dab2ef6cd1169d3a3a78391f520.jpg)

![3c76e57d848da9e70cb7a85f0541a572a8313ff3f9675c2f3ad7f6df7385341f.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/tables/3c76e57d848da9e70cb7a85f0541a572a8313ff3f9675c2f3ad7f6df7385341f.jpg)

![3d2f23c1484fd20b197ef6d70f3ea0e882a5a2e74b41717b88661e60b9800e67.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/tables/3d2f23c1484fd20b197ef6d70f3ea0e882a5a2e74b41717b88661e60b9800e67.jpg)

![6813f268f8065fe1e62a74bd99f85b6b35044e36bf93ce1a9f5515e0a15c53d8.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/tables/6813f268f8065fe1e62a74bd99f85b6b35044e36bf93ce1a9f5515e0a15c53d8.jpg)

![b776a051c8d0e499e0a5f3e97852b66d6b97891c1165c87cc59f72b92337577c.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/tables/b776a051c8d0e499e0a5f3e97852b66d6b97891c1165c87cc59f72b92337577c.jpg)

![ff9b1b807426e956ac6f7c4bb65f28ce3ff9137d0304d88f0743be721b52aa6b.jpg](../icml_results/5253_Can%20LLMs%20Reason%20Like%20Automated%20Theorem%20Provers%20for%20Rust%20Verification_%20VCoT-Bench_%20Evaluating%20via%20Ver/tables/ff9b1b807426e956ac6f7c4bb65f28ce3ff9137d0304d88f0743be721b52aa6b.jpg)

<a id="paper-5206"></a>

## MEMO: Memory-Augmented Model Context Optimization for Robust Multi-Turn Multi-Agent LLM Games


### Images

![41a439645519538cd9c68fd15e90021d87555ece23568e91d17a4f3c43db3b83.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/images/41a439645519538cd9c68fd15e90021d87555ece23568e91d17a4f3c43db3b83.jpg)

![64ddf74bf1b96824bf12de1ff6b4ee1892fe082f13098fbf83a1968addcd7e74.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/images/64ddf74bf1b96824bf12de1ff6b4ee1892fe082f13098fbf83a1968addcd7e74.jpg)

![bf7cf26d95b60b466d44f4cbde371fcd6dd34cde1cdbc7f3181e46e024f54e88.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/images/bf7cf26d95b60b466d44f4cbde371fcd6dd34cde1cdbc7f3181e46e024f54e88.jpg)

![bfdfda7fe645c3ffd54c37332c35f2128b0f004036419cb12872ce026778d339.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/images/bfdfda7fe645c3ffd54c37332c35f2128b0f004036419cb12872ce026778d339.jpg)

![c015946122ac02a6926a1fc846d418007840b6979b7f3a363d95d5ffa709e45e.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/images/c015946122ac02a6926a1fc846d418007840b6979b7f3a363d95d5ffa709e45e.jpg)

![c762bffffea6cd51e696b495a631d1f161c507c361aca1cab569d98128e45d0f.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/images/c762bffffea6cd51e696b495a631d1f161c507c361aca1cab569d98128e45d0f.jpg)

![dd71dfc6c304d77407d4a44b63bfb5df68694938b2e6baaca1fa55f042fa3d3d.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/images/dd71dfc6c304d77407d4a44b63bfb5df68694938b2e6baaca1fa55f042fa3d3d.jpg)

![de0a734961d412a880456f63e783af3dcafefc7452524039c395e669a10dce00.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/images/de0a734961d412a880456f63e783af3dcafefc7452524039c395e669a10dce00.jpg)

### Tables

![0cf40b6c7fadad3ac45ad054421fefdf5e231f4c5270d802e3d1492b54fea8ae.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/0cf40b6c7fadad3ac45ad054421fefdf5e231f4c5270d802e3d1492b54fea8ae.jpg)

![159d002ab3adc3a0fe7c06c939386b734a98357dcedde393f58c1e60f78086f1.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/159d002ab3adc3a0fe7c06c939386b734a98357dcedde393f58c1e60f78086f1.jpg)

![31fe956c750842a686fd4e569756e3c4b4cb01437530381c34233caf2eac9529.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/31fe956c750842a686fd4e569756e3c4b4cb01437530381c34233caf2eac9529.jpg)

![37764fc58f0675ae80f88b3fca364f6a944ebc03256d174468e3007aa4720092.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/37764fc58f0675ae80f88b3fca364f6a944ebc03256d174468e3007aa4720092.jpg)

![455837f70185e65c3fdcd239e1e2fe970171edfbedcaf13b12889ef38bcbdf4f.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/455837f70185e65c3fdcd239e1e2fe970171edfbedcaf13b12889ef38bcbdf4f.jpg)

![48d43f3124c2a611d4d14a18ff772cc7c79df2c6d7cacef3dee12655ee32ba00.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/48d43f3124c2a611d4d14a18ff772cc7c79df2c6d7cacef3dee12655ee32ba00.jpg)

![63fe7f283221be5a1da32ccc5fc5501becea9cb85dbc5f88640110b5deaf50d4.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/63fe7f283221be5a1da32ccc5fc5501becea9cb85dbc5f88640110b5deaf50d4.jpg)

![86a045ef97e71732f71a4632d8a1901a86beb7cc2377b9af0a5d00e0db058a3e.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/86a045ef97e71732f71a4632d8a1901a86beb7cc2377b9af0a5d00e0db058a3e.jpg)

![8b0cc00d4e1d07e89f96472a530f2df8d3700dbc3c7bafdaadbf7e69ceff458a.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/8b0cc00d4e1d07e89f96472a530f2df8d3700dbc3c7bafdaadbf7e69ceff458a.jpg)

![8d66f33492bb153f4b1ad9050640f5891f30b27878c1384e5de43937c85874b2.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/8d66f33492bb153f4b1ad9050640f5891f30b27878c1384e5de43937c85874b2.jpg)

![a5eec6c247481cd1853924426079fca1016aa392a60ccec656febe2330826e76.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/a5eec6c247481cd1853924426079fca1016aa392a60ccec656febe2330826e76.jpg)

![a7754b73ff691a5210e744330b59a782ee4d1e5687c7f71d6654a4a377f2dd71.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/a7754b73ff691a5210e744330b59a782ee4d1e5687c7f71d6654a4a377f2dd71.jpg)

![bacba82dc183931d137f17d9a003cdd91bec6b7572525b9bfdf8653d18041290.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/bacba82dc183931d137f17d9a003cdd91bec6b7572525b9bfdf8653d18041290.jpg)

![c355bdb4471f5885eb816e365be7177e335c7ea30580fe633a7eb40aa812ab4b.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/c355bdb4471f5885eb816e365be7177e335c7ea30580fe633a7eb40aa812ab4b.jpg)

![cc0da5b6d88658e294c544a2a281906db11984be8f6bc90e25da21ed317a86e0.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/cc0da5b6d88658e294c544a2a281906db11984be8f6bc90e25da21ed317a86e0.jpg)

![d4e9d74b49e01f7b1850aa767ace2426f03a04ee43f9187193a587f1010833a5.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/d4e9d74b49e01f7b1850aa767ace2426f03a04ee43f9187193a587f1010833a5.jpg)

![dde14cfa357baba4d90d2e6b3b8428e96cb7e8d21b195fd5f43e37a5cb6ac8db.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/dde14cfa357baba4d90d2e6b3b8428e96cb7e8d21b195fd5f43e37a5cb6ac8db.jpg)

![e4b34078853d4559b5013519c4ddaf9970a38ad7b648a372cc980ad94c004e08.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/e4b34078853d4559b5013519c4ddaf9970a38ad7b648a372cc980ad94c004e08.jpg)

![f566ddbc7f78706b9d88854c2e70966c91268e617e105db66256a75885f6d260.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/f566ddbc7f78706b9d88854c2e70966c91268e617e105db66256a75885f6d260.jpg)

![fe95858053e4a53198c6c90b677133b952077f39e4be7ff765d66034e56aba3c.jpg](../icml_results/5254_MEMO_%20Memory-Augmented%20Model%20Context%20Optimization%20for%20Robust%20Multi-Turn%20Multi-Agent%20LLM%20Games/tables/fe95858053e4a53198c6c90b677133b952077f39e4be7ff765d66034e56aba3c.jpg)

<a id="paper-5207"></a>

## FG-CLIP 2: A Bilingual Fine-grained Vision-Language Alignment Model


### Images

![26c01de9e88d87e5b3aa1af02b3276f4cccb499e5a6343c8636692d1ba7c66ba.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/26c01de9e88d87e5b3aa1af02b3276f4cccb499e5a6343c8636692d1ba7c66ba.jpg)

![412776b3b093cd19b95cb20e1107564d0a08a87ef7df4b8be29f132a7b1f866e.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/412776b3b093cd19b95cb20e1107564d0a08a87ef7df4b8be29f132a7b1f866e.jpg)

![4eb8eb8bda3afe57d5302127a8acf05f9dee119c9c97cf57b953a77ec35de5c1.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/4eb8eb8bda3afe57d5302127a8acf05f9dee119c9c97cf57b953a77ec35de5c1.jpg)

![53c9852f7de191591a00a5324ffab998ec8613a4ce386000d862fe7f6ae3da54.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/53c9852f7de191591a00a5324ffab998ec8613a4ce386000d862fe7f6ae3da54.jpg)

![694738de27e58883ae7bb3b7b2cfa4d6bd5612d6c597b184b2d8f878ef06b8f1.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/694738de27e58883ae7bb3b7b2cfa4d6bd5612d6c597b184b2d8f878ef06b8f1.jpg)

![6b7064042aec8294a337d2f38806b248935e8eb843fd6995b9072bd789fa16f1.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/6b7064042aec8294a337d2f38806b248935e8eb843fd6995b9072bd789fa16f1.jpg)

![6fab66b115715bcb21ee71719f0845b899f3eae98914f1b2f49dfaf2a64dedf4.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/6fab66b115715bcb21ee71719f0845b899f3eae98914f1b2f49dfaf2a64dedf4.jpg)

![8e74aed286f57854365296067475063baeb562b12c6168298c9da737e28e9857.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/8e74aed286f57854365296067475063baeb562b12c6168298c9da737e28e9857.jpg)

![97c686f15ff8469acc573fe638ec1be2ab16c142fe21b00983c7b7935f968dcd.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/97c686f15ff8469acc573fe638ec1be2ab16c142fe21b00983c7b7935f968dcd.jpg)

![b73d493deb9291a48b920d749eb53811aff7b0e23783c2848c9d7913444b0910.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/b73d493deb9291a48b920d749eb53811aff7b0e23783c2848c9d7913444b0910.jpg)

![ba3f57844f97b84c9a10b8f7b5e40b34cdacb764be91458085c69791398e9332.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/ba3f57844f97b84c9a10b8f7b5e40b34cdacb764be91458085c69791398e9332.jpg)

![bd153941b9c50624499b5a3eaa3c491cb77517c6b50a93f8e827678deb382c0b.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/bd153941b9c50624499b5a3eaa3c491cb77517c6b50a93f8e827678deb382c0b.jpg)

![cd94ddab277fa08c50827e43e9f9ac5a50862bb5255b27d2e9f5dc36a353351c.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/cd94ddab277fa08c50827e43e9f9ac5a50862bb5255b27d2e9f5dc36a353351c.jpg)

![db323fa6dabe53f219bc4ed614449f5a5cbdc513b1d06219f23df9ae30081b4b.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/db323fa6dabe53f219bc4ed614449f5a5cbdc513b1d06219f23df9ae30081b4b.jpg)

![e0cb4b9fb333c7476361ff4fbafc609a6f3e95db6dee47e25491e59b873e72d9.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/e0cb4b9fb333c7476361ff4fbafc609a6f3e95db6dee47e25491e59b873e72d9.jpg)

![e6e0756d46d65d4ae4f8fb6ebaae2df8a20a34d176db4c22013ca75c502ab060.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/e6e0756d46d65d4ae4f8fb6ebaae2df8a20a34d176db4c22013ca75c502ab060.jpg)

![f16bcdc8a36a9ac557d48428183da5d75cfe096ca6d86ce2690f46b63dbe8672.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/f16bcdc8a36a9ac557d48428183da5d75cfe096ca6d86ce2690f46b63dbe8672.jpg)

![fca43856a5115b49e6b0bbf72c8fdc7a812e391220a2e5fe249be8685c2c124e.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/images/fca43856a5115b49e6b0bbf72c8fdc7a812e391220a2e5fe249be8685c2c124e.jpg)

### Tables

![09dbb51235ae6a4b7ad9299c706618f37b42f2fb8d33eb04c464ac48f623a0d0.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/tables/09dbb51235ae6a4b7ad9299c706618f37b42f2fb8d33eb04c464ac48f623a0d0.jpg)

![1b9e7336fb95ad593d92dc1fa95c5fcb1697ecfaeec49d91c47f51cd5eab1c44.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/tables/1b9e7336fb95ad593d92dc1fa95c5fcb1697ecfaeec49d91c47f51cd5eab1c44.jpg)

![3f6982919502c10655f3e35b81286c450a563079c30e432df350052ebd647bed.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/tables/3f6982919502c10655f3e35b81286c450a563079c30e432df350052ebd647bed.jpg)

![628ba70d5020b93801aede04b33f643a72003ab46053f88799467e7989112056.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/tables/628ba70d5020b93801aede04b33f643a72003ab46053f88799467e7989112056.jpg)

![6dff888c976bfa3b36bf4555a30d079f418191c845b862ec45ab0839ea19807c.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/tables/6dff888c976bfa3b36bf4555a30d079f418191c845b862ec45ab0839ea19807c.jpg)

![6f15273c5e4319d3c32b1644d0733f8c5b04f1770a719546e8374159dc46be5d.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/tables/6f15273c5e4319d3c32b1644d0733f8c5b04f1770a719546e8374159dc46be5d.jpg)

![76c4ae8b2930a2e1f82079d685380ecda1dc430ffed57456b89d2c3320d1a64f.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/tables/76c4ae8b2930a2e1f82079d685380ecda1dc430ffed57456b89d2c3320d1a64f.jpg)

![7fa8725e8703f36b530fb00e4f0b2acf4e480ade73c11e549d100ef8f74352dd.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/tables/7fa8725e8703f36b530fb00e4f0b2acf4e480ade73c11e549d100ef8f74352dd.jpg)

![88c24b962345f41bec39516d294fb8a30026091fab0283d46d683a26da592eee.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/tables/88c24b962345f41bec39516d294fb8a30026091fab0283d46d683a26da592eee.jpg)

![a1da2391969682d76668319d43145c266492170715104f701c9bc79a75afb24e.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/tables/a1da2391969682d76668319d43145c266492170715104f701c9bc79a75afb24e.jpg)

![ac69227f166aa2c0965c5a1fdb6c5d9695838d233b340e044cfa46e5f1da2bdf.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/tables/ac69227f166aa2c0965c5a1fdb6c5d9695838d233b340e044cfa46e5f1da2bdf.jpg)

![ad6e208e85d600ece35f9588232d42899f914f815cb3bca2934a3f65a6c81c80.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/tables/ad6e208e85d600ece35f9588232d42899f914f815cb3bca2934a3f65a6c81c80.jpg)

![b45825bc4d8ec461615b5af0030000a0d43536de47e921a006ef3510430530ff.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/tables/b45825bc4d8ec461615b5af0030000a0d43536de47e921a006ef3510430530ff.jpg)

![c06689c59c871cb9cefb8781f2263bb3a1f651ddbaf3d6f23ada280e470ee2b6.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/tables/c06689c59c871cb9cefb8781f2263bb3a1f651ddbaf3d6f23ada280e470ee2b6.jpg)

![c41abbe5c7f86afe0b217c13f71fc96791e463716c9142f7593f62b1538203f0.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/tables/c41abbe5c7f86afe0b217c13f71fc96791e463716c9142f7593f62b1538203f0.jpg)

![dc5e2c62269a187ac3835d7f165c913ffa32123c1aee340353fe7b9faa253d1c.jpg](../icml_results/5255_FG-CLIP%202_%20A%20Bilingual%20Fine-grained%20Vision-Language%20Alignment%20Model/tables/dc5e2c62269a187ac3835d7f165c913ffa32123c1aee340353fe7b9faa253d1c.jpg)

<a id="paper-5208"></a>

## Statistical Early Stopping for Reasoning Models


### Images

![0f62629b790c091dc7f51aecea6f7678755048777292cfb464ab7c1ad2dfeee2.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/images/0f62629b790c091dc7f51aecea6f7678755048777292cfb464ab7c1ad2dfeee2.jpg)

![1c527a55ebf5a13b11de9b83ac041d54f19a80a8d3e9daf814cbd1045a510789.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/images/1c527a55ebf5a13b11de9b83ac041d54f19a80a8d3e9daf814cbd1045a510789.jpg)

![502c0751fb31d74691d044eec3e7d817a7b3998be8785d64cc3819250c57de7e.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/images/502c0751fb31d74691d044eec3e7d817a7b3998be8785d64cc3819250c57de7e.jpg)

![5a7492b3cf9c93dcc2bf5d0a98950092f75c53497134bae7def11c9176b7eecd.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/images/5a7492b3cf9c93dcc2bf5d0a98950092f75c53497134bae7def11c9176b7eecd.jpg)

![994d966533062c32ff3a6de072cf0d4160ddba1bd9277d08c78b6a87c6ff4a46.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/images/994d966533062c32ff3a6de072cf0d4160ddba1bd9277d08c78b6a87c6ff4a46.jpg)

![ab08973c40fc89a4f60f1623257f157357cae41a0e12554741f0565ab2798075.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/images/ab08973c40fc89a4f60f1623257f157357cae41a0e12554741f0565ab2798075.jpg)

![ad96eaf690456226a03c6f187286e2c7b7d99bf9a8a09eb55a1116a36d41efc5.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/images/ad96eaf690456226a03c6f187286e2c7b7d99bf9a8a09eb55a1116a36d41efc5.jpg)

![afef9d28b9377c1066c916e1295f94bcc843b29875097623a0b108a344f1113a.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/images/afef9d28b9377c1066c916e1295f94bcc843b29875097623a0b108a344f1113a.jpg)

![d441b5901a9501479a910e9e7baaa7781a80aa30ccf9b0e8a0d01e7a19ed6a81.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/images/d441b5901a9501479a910e9e7baaa7781a80aa30ccf9b0e8a0d01e7a19ed6a81.jpg)

![f106911dd6b6fa0595e9180e027eac17c27c609d103e644715897f2220b2a5a8.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/images/f106911dd6b6fa0595e9180e027eac17c27c609d103e644715897f2220b2a5a8.jpg)

![fc452f07963f7633570a8d0db2379f971b2bb5c8d8a71b462ed75da4549a4179.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/images/fc452f07963f7633570a8d0db2379f971b2bb5c8d8a71b462ed75da4549a4179.jpg)

### Tables

![04586efd6e0a4ce5f65912c09aa896ee0129fa6a65c47c435cbd6e2c12e511aa.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/04586efd6e0a4ce5f65912c09aa896ee0129fa6a65c47c435cbd6e2c12e511aa.jpg)

![0bc39140a2945b2df1e13f4f321d3ee00c99230806129c7412b6be5ab53bf6d3.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/0bc39140a2945b2df1e13f4f321d3ee00c99230806129c7412b6be5ab53bf6d3.jpg)

![20a5243a4c68533bf5515f4345508dcede71b692cd947f5c6d0b0935dde67108.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/20a5243a4c68533bf5515f4345508dcede71b692cd947f5c6d0b0935dde67108.jpg)

![223fa88b29406c778a0370506508aea7bc429fdcf9821d7a85789c5d5a8716f4.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/223fa88b29406c778a0370506508aea7bc429fdcf9821d7a85789c5d5a8716f4.jpg)

![268f08bba39fcb4a9151806e8bbd500ad9c4b1c7f85efee1000651cb8faf23fc.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/268f08bba39fcb4a9151806e8bbd500ad9c4b1c7f85efee1000651cb8faf23fc.jpg)

![679f124aa9fb7361d46acf7709d7e7f16acc046b67984bf6d4af60a0420d9493.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/679f124aa9fb7361d46acf7709d7e7f16acc046b67984bf6d4af60a0420d9493.jpg)

![71cfca8bf2bc375874d32b63c2eae4b00ea29ae15acdd40398abd7d18a94637b.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/71cfca8bf2bc375874d32b63c2eae4b00ea29ae15acdd40398abd7d18a94637b.jpg)

![7394c374299605aa3a528e42f82706c5f31ad271edf3098ca0f9877e5ca78d25.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/7394c374299605aa3a528e42f82706c5f31ad271edf3098ca0f9877e5ca78d25.jpg)

![744dd3adac623dcdecaae2829cb20a1d86deb4a0c4088a8520c669d2b5d691b9.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/744dd3adac623dcdecaae2829cb20a1d86deb4a0c4088a8520c669d2b5d691b9.jpg)

![8304af7b84c5a1d4fc867dd65a46b647ec8d266af11f3f77b1c6731f297712d2.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/8304af7b84c5a1d4fc867dd65a46b647ec8d266af11f3f77b1c6731f297712d2.jpg)

![9d415dee72fc81487584bf82222bc4c1821218393a60e488bdbbe8063fbcb448.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/9d415dee72fc81487584bf82222bc4c1821218393a60e488bdbbe8063fbcb448.jpg)

![c1ca98e14a1c95cd15ac6e4e4c479c810cd5bf8775f5864f868da874c80484cf.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/c1ca98e14a1c95cd15ac6e4e4c479c810cd5bf8775f5864f868da874c80484cf.jpg)

![c724e48584cec9fff20a7f67218bb7fd4e7596b4b010d716fffc2ea0d213960c.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/c724e48584cec9fff20a7f67218bb7fd4e7596b4b010d716fffc2ea0d213960c.jpg)

![ce2357fcc8175f0d9df13f93efd9bdea911413d13fc885a69b3ee44f6484dcb9.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/ce2357fcc8175f0d9df13f93efd9bdea911413d13fc885a69b3ee44f6484dcb9.jpg)

![ce77a113c4288e63cc3a03895c5f5226c551d0be4894e6acff2dda5e1cd37c57.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/ce77a113c4288e63cc3a03895c5f5226c551d0be4894e6acff2dda5e1cd37c57.jpg)

![d5d4ca2ee1321e9c77f4c93fec95587247ba5ccacaddc909d4e0cb479e433b7a.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/d5d4ca2ee1321e9c77f4c93fec95587247ba5ccacaddc909d4e0cb479e433b7a.jpg)

![da1520e14b7f603ae1e57b723e0b35add40e827171afb9b34d2cfa479ff6ddb9.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/da1520e14b7f603ae1e57b723e0b35add40e827171afb9b34d2cfa479ff6ddb9.jpg)

![fc7e2c8325862cb43dd96f0b4f8fee570489407c15a9b2f06c9c4db101eb90e4.jpg](../icml_results/5256_Statistical%20Early%20Stopping%20for%20Reasoning%20Models/tables/fc7e2c8325862cb43dd96f0b4f8fee570489407c15a9b2f06c9c4db101eb90e4.jpg)

<a id="paper-5209"></a>

## AnyCanvas: Potential Field Guidance for Training-Free Spatial Control in Text-to-Image Diffusion


### Images

![4d99ea3f631499d9c4a3126fb949418feadb1e0c604f36177bf81a98d526ccfd.jpg](../icml_results/5257_AnyCanvas_%20Potential%20Field%20Guidance%20for%20Training-Free%20Spatial%20Control%20in%20Text-to-Image%20Diffusion/images/4d99ea3f631499d9c4a3126fb949418feadb1e0c604f36177bf81a98d526ccfd.jpg)

![876fb29b712405ed06806fa641890af7d6e842a7e4e9d3661f393cb54871c2a0.jpg](../icml_results/5257_AnyCanvas_%20Potential%20Field%20Guidance%20for%20Training-Free%20Spatial%20Control%20in%20Text-to-Image%20Diffusion/images/876fb29b712405ed06806fa641890af7d6e842a7e4e9d3661f393cb54871c2a0.jpg)

![a24dea0b6206cdd5eb93802558a3a019d134411e5a19c109b057202c07b39861.jpg](../icml_results/5257_AnyCanvas_%20Potential%20Field%20Guidance%20for%20Training-Free%20Spatial%20Control%20in%20Text-to-Image%20Diffusion/images/a24dea0b6206cdd5eb93802558a3a019d134411e5a19c109b057202c07b39861.jpg)

![a4f33ab215a6ff9bac362e445ba64ec5be6cc96cb2ac6d25abbc55e1bf90418e.jpg](../icml_results/5257_AnyCanvas_%20Potential%20Field%20Guidance%20for%20Training-Free%20Spatial%20Control%20in%20Text-to-Image%20Diffusion/images/a4f33ab215a6ff9bac362e445ba64ec5be6cc96cb2ac6d25abbc55e1bf90418e.jpg)

![b082f96d3bcb4374e082b8d2ac5edecbbbe683953b013645c5592001b679fce7.jpg](../icml_results/5257_AnyCanvas_%20Potential%20Field%20Guidance%20for%20Training-Free%20Spatial%20Control%20in%20Text-to-Image%20Diffusion/images/b082f96d3bcb4374e082b8d2ac5edecbbbe683953b013645c5592001b679fce7.jpg)

![b550d2d47c1cb3122d86632d6290e679ca98ee9a187d55bcd0a104ffe63ac7bf.jpg](../icml_results/5257_AnyCanvas_%20Potential%20Field%20Guidance%20for%20Training-Free%20Spatial%20Control%20in%20Text-to-Image%20Diffusion/images/b550d2d47c1cb3122d86632d6290e679ca98ee9a187d55bcd0a104ffe63ac7bf.jpg)

![c3202da2a52ac08b3aadc90e42e2412c7d6e8251a0f7b265129a196eefa13c07.jpg](../icml_results/5257_AnyCanvas_%20Potential%20Field%20Guidance%20for%20Training-Free%20Spatial%20Control%20in%20Text-to-Image%20Diffusion/images/c3202da2a52ac08b3aadc90e42e2412c7d6e8251a0f7b265129a196eefa13c07.jpg)

### Tables

![1a3c19f8585c5ec2aad8c6a3dbc86388f9cfa99772430a1dbbac960411a6671f.jpg](../icml_results/5257_AnyCanvas_%20Potential%20Field%20Guidance%20for%20Training-Free%20Spatial%20Control%20in%20Text-to-Image%20Diffusion/tables/1a3c19f8585c5ec2aad8c6a3dbc86388f9cfa99772430a1dbbac960411a6671f.jpg)

![26d2de6e20fa524816cc322c8a3630fad0edcd8d30c89c603af763083c7dd65c.jpg](../icml_results/5257_AnyCanvas_%20Potential%20Field%20Guidance%20for%20Training-Free%20Spatial%20Control%20in%20Text-to-Image%20Diffusion/tables/26d2de6e20fa524816cc322c8a3630fad0edcd8d30c89c603af763083c7dd65c.jpg)

![2a95369a250636636138456e78a30f02ed7927729d06fd85429003af26b57ee5.jpg](../icml_results/5257_AnyCanvas_%20Potential%20Field%20Guidance%20for%20Training-Free%20Spatial%20Control%20in%20Text-to-Image%20Diffusion/tables/2a95369a250636636138456e78a30f02ed7927729d06fd85429003af26b57ee5.jpg)

![2b56536cfef065db826e18a520e162ab58bc393b85722dd70f3cda50d3538c44.jpg](../icml_results/5257_AnyCanvas_%20Potential%20Field%20Guidance%20for%20Training-Free%20Spatial%20Control%20in%20Text-to-Image%20Diffusion/tables/2b56536cfef065db826e18a520e162ab58bc393b85722dd70f3cda50d3538c44.jpg)

![c05aac4b0aca9d08a3f1025556359b2b284adf27db0872b17d6c24693675680e.jpg](../icml_results/5257_AnyCanvas_%20Potential%20Field%20Guidance%20for%20Training-Free%20Spatial%20Control%20in%20Text-to-Image%20Diffusion/tables/c05aac4b0aca9d08a3f1025556359b2b284adf27db0872b17d6c24693675680e.jpg)

<a id="paper-5210"></a>

## SkillNet: Hierarchical Skill Modeling for Compositional Generalization in Vision-Language Action Models


### Images

![0f5b1bd7924f0f24ba398a1fd2544cb4f3d63ce704fa10886d040f68ee9bb729.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/images/0f5b1bd7924f0f24ba398a1fd2544cb4f3d63ce704fa10886d040f68ee9bb729.jpg)

![19cae892fae035c48b8e294e61588abcee2cec066b4e79b66300a2f44b98c779.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/images/19cae892fae035c48b8e294e61588abcee2cec066b4e79b66300a2f44b98c779.jpg)

![34b3cef511e46cd88cd470afcdd461409b9d0ed0f1ed55dba79b872643b502b4.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/images/34b3cef511e46cd88cd470afcdd461409b9d0ed0f1ed55dba79b872643b502b4.jpg)

![38525dc1439f9eb94538b0d47f92bb2d119ebed4254ca717cd2391f55b5822cb.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/images/38525dc1439f9eb94538b0d47f92bb2d119ebed4254ca717cd2391f55b5822cb.jpg)

![7fa5f8b9f9d4e538d84ceb23013194d8b02cbfc8d314044ed7102ff14bed261b.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/images/7fa5f8b9f9d4e538d84ceb23013194d8b02cbfc8d314044ed7102ff14bed261b.jpg)

![d62d8ec6d1fb695b9f2c82c30c7dfc26cf71d99d1f9830ce4e5e0b6008a1edc5.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/images/d62d8ec6d1fb695b9f2c82c30c7dfc26cf71d99d1f9830ce4e5e0b6008a1edc5.jpg)

![d82ce6abbbbf0eff491e69c9b3ef2010781c5c3a606a9eef9112f7335094996b.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/images/d82ce6abbbbf0eff491e69c9b3ef2010781c5c3a606a9eef9112f7335094996b.jpg)

![e63ad34d3ac0fb4773e2dc9c3fc1df88b854f00139f3c5eb5bf6a2c208ee3ba4.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/images/e63ad34d3ac0fb4773e2dc9c3fc1df88b854f00139f3c5eb5bf6a2c208ee3ba4.jpg)

![f4470cb8be99dc4a4a349ac30a564e0682ffdaa78b0ca73e8ab4545c2aab5be8.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/images/f4470cb8be99dc4a4a349ac30a564e0682ffdaa78b0ca73e8ab4545c2aab5be8.jpg)

![fe52b920766755def44c82de34110e26e6c6909a4b4809e7c296dc9b402ab17c.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/images/fe52b920766755def44c82de34110e26e6c6909a4b4809e7c296dc9b402ab17c.jpg)

### Tables

![00cd505a929140180787f13e961bdf6e904591622b004961895195c97882199c.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/00cd505a929140180787f13e961bdf6e904591622b004961895195c97882199c.jpg)

![170ffa019d51495f8d5cfd185dfb55d2ae4c58a7f443bc727d5f959f288cce59.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/170ffa019d51495f8d5cfd185dfb55d2ae4c58a7f443bc727d5f959f288cce59.jpg)

![194855f9e4832e7f326198ce4c51787cd97f88b76c25cede9f7465449d93e56f.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/194855f9e4832e7f326198ce4c51787cd97f88b76c25cede9f7465449d93e56f.jpg)

![1d3a00cda13b0b756e51b3465298fe48ccdd18fc23e86fdeb38a5b6e387348e7.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/1d3a00cda13b0b756e51b3465298fe48ccdd18fc23e86fdeb38a5b6e387348e7.jpg)

![2af8fe60c561ed2d51f95b586604995dfda4afc1971d9525550cfb33588f13c0.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/2af8fe60c561ed2d51f95b586604995dfda4afc1971d9525550cfb33588f13c0.jpg)

![367f7fe35267c1b6899b377ffdd121a23aea096417e8c379ab0a124f74877fdd.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/367f7fe35267c1b6899b377ffdd121a23aea096417e8c379ab0a124f74877fdd.jpg)

![388ded20eb6f094d201b09031d557e472093dcc32c7d8aca3c63b6aecb075989.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/388ded20eb6f094d201b09031d557e472093dcc32c7d8aca3c63b6aecb075989.jpg)

![3d8ce7a55c7f087ca2889c21c1445c855d16ca1f5db2acd07591da239e3d9b20.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/3d8ce7a55c7f087ca2889c21c1445c855d16ca1f5db2acd07591da239e3d9b20.jpg)

![3ef27d9406c1d441be98b286835fbae05e1f61181436d1468a9144e4283d6336.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/3ef27d9406c1d441be98b286835fbae05e1f61181436d1468a9144e4283d6336.jpg)

![4df872b465ec858476dde85f45a338b06d8b30d86a273b4ae4f17e75761a1e9e.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/4df872b465ec858476dde85f45a338b06d8b30d86a273b4ae4f17e75761a1e9e.jpg)

![50f364758a0f38e9aeb13da515df0811493343b0ac99645232a667709ca44a65.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/50f364758a0f38e9aeb13da515df0811493343b0ac99645232a667709ca44a65.jpg)

![55227a9bdde27d8601d4dc6ad88f261e03edd7d132e9de2e1631a2d942ce2629.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/55227a9bdde27d8601d4dc6ad88f261e03edd7d132e9de2e1631a2d942ce2629.jpg)

![642456937b9e67a84229b6a7eb5be235bea596a2100c54d70252639ca6bf3a55.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/642456937b9e67a84229b6a7eb5be235bea596a2100c54d70252639ca6bf3a55.jpg)

![652c1ea8fb380a17437463dd33f4efbf1bccfde436970097df7a15d84c16dde3.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/652c1ea8fb380a17437463dd33f4efbf1bccfde436970097df7a15d84c16dde3.jpg)

![6f2cd91f40011f615e3486b6f7a27c5270aaf37c65033cfe7d60a79e22576119.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/6f2cd91f40011f615e3486b6f7a27c5270aaf37c65033cfe7d60a79e22576119.jpg)

![7052b0c7c7437c97ff4bde3a3067d4469568a675e83f5fb88530862c6f23144d.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/7052b0c7c7437c97ff4bde3a3067d4469568a675e83f5fb88530862c6f23144d.jpg)

![80922e404a0d8cd2263f5fe8b49df9c2fbea5c436f32625ec8b7bc876e77fa29.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/80922e404a0d8cd2263f5fe8b49df9c2fbea5c436f32625ec8b7bc876e77fa29.jpg)

![81965b1e7fa1c737b4011bb69d9171ec02f3ed92bb20eaf73366de4bbd2a5f97.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/81965b1e7fa1c737b4011bb69d9171ec02f3ed92bb20eaf73366de4bbd2a5f97.jpg)

![81f5686d4baeea56dad229781cf3fde493bfd8cc48cc3daa720aabe5eca113f1.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/81f5686d4baeea56dad229781cf3fde493bfd8cc48cc3daa720aabe5eca113f1.jpg)

![a2624f2fca414aa0bb68e81aee6baa23bdade741f5a9f0e9698a6094c56c9a7c.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/a2624f2fca414aa0bb68e81aee6baa23bdade741f5a9f0e9698a6094c56c9a7c.jpg)

![a5a4ece0abefc7e4de25c2c0256748f2e9a8c41dd0de064a61807e91158632f8.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/a5a4ece0abefc7e4de25c2c0256748f2e9a8c41dd0de064a61807e91158632f8.jpg)

![b6968724d0766b845415ec17495821f7625237d7acbe11ae621366927d32bdf2.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/b6968724d0766b845415ec17495821f7625237d7acbe11ae621366927d32bdf2.jpg)

![b733ced0bbe35f2d07f120432b1ac0f9226b655d56bfca241f40c00b3d895858.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/b733ced0bbe35f2d07f120432b1ac0f9226b655d56bfca241f40c00b3d895858.jpg)

![c4565492ee4cab37a0359874cf94d972516d869065c97fe390b30306393251ca.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/c4565492ee4cab37a0359874cf94d972516d869065c97fe390b30306393251ca.jpg)

![c5d0cab6d1e05ffd93a12f2354691600883e1fb4ca544666cb788461bcc6b10f.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/c5d0cab6d1e05ffd93a12f2354691600883e1fb4ca544666cb788461bcc6b10f.jpg)

![ce354a6a1b35632bceccfe09fc97be47f697a96ea33eadc750e3c1c0f1c8b502.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/ce354a6a1b35632bceccfe09fc97be47f697a96ea33eadc750e3c1c0f1c8b502.jpg)

![ce65948398e2bea46266e732c5eb99bedef1505eb3e286ca7a7fdd9c1f4f5915.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/ce65948398e2bea46266e732c5eb99bedef1505eb3e286ca7a7fdd9c1f4f5915.jpg)

![ef7f372d5001944a4dbff93178ab8e8c1a64c65ef1b0c43dafdf27740ba1f5be.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/ef7f372d5001944a4dbff93178ab8e8c1a64c65ef1b0c43dafdf27740ba1f5be.jpg)

![f1b27f0a9631c8b69dd3259ebe75135322648e73b9b8755a0ae65facaee032b2.jpg](../icml_results/5258_SkillNet_%20Hierarchical%20Skill%20Modeling%20for%20Compositional%20Generalization%20in%20Vision-Language%20Action%20Mod/tables/f1b27f0a9631c8b69dd3259ebe75135322648e73b9b8755a0ae65facaee032b2.jpg)

<a id="paper-5211"></a>

## Dynamic Regret via Discounted-to-Dynamic Reduction with Applications to Curved Losses and Adam Optimizer


### Images

![e4fc9662c00d1f0854ee0e06e6e39451564d4eca086023e753166147ab6c9278.jpg](../icml_results/5259_Dynamic%20Regret%20via%20Discounted-to-Dynamic%20Reduction%20with%20Applications%20to%20Curved%20Losses%20and%20Adam%20Optim/images/e4fc9662c00d1f0854ee0e06e6e39451564d4eca086023e753166147ab6c9278.jpg)

### Tables

![d86e7affc4e7ff495a8a06f6401c905bfd63282c20dd39722adb0ff527405704.jpg](../icml_results/5259_Dynamic%20Regret%20via%20Discounted-to-Dynamic%20Reduction%20with%20Applications%20to%20Curved%20Losses%20and%20Adam%20Optim/tables/d86e7affc4e7ff495a8a06f6401c905bfd63282c20dd39722adb0ff527405704.jpg)

<a id="paper-5212"></a>

## Probing How Scalable Table Data Enhances General Long-Context Reasoning


### Images

![09c1799c325002c3a2a971c30c6346199c51708b2b764479bfcbb1275f2de39b.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/images/09c1799c325002c3a2a971c30c6346199c51708b2b764479bfcbb1275f2de39b.jpg)

![34ba71fbc45def36a1ca768593e23962670299cbbde80a92ff91abb8149e6e2f.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/images/34ba71fbc45def36a1ca768593e23962670299cbbde80a92ff91abb8149e6e2f.jpg)

![41fca4b53286f12328ed05ba1d1741372bf743a211e438648c380999d87d13d9.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/images/41fca4b53286f12328ed05ba1d1741372bf743a211e438648c380999d87d13d9.jpg)

![83f48e41e9a3d08d007c12ec5491068d3a2bc83c34ae7335c14c8f47208a9f3b.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/images/83f48e41e9a3d08d007c12ec5491068d3a2bc83c34ae7335c14c8f47208a9f3b.jpg)

![85f4ee378db8315150bead9c1f74c0682e468a56557cce65cf9f9e7538537796.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/images/85f4ee378db8315150bead9c1f74c0682e468a56557cce65cf9f9e7538537796.jpg)

![924dab8bc677035923ee9ead0283d813139d556eaf638e16a46d61d83de9f334.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/images/924dab8bc677035923ee9ead0283d813139d556eaf638e16a46d61d83de9f334.jpg)

![a1dd6bb1e9a05430db7f613325b2292169710fa1c5cfc94e1a9a8c51bf68189c.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/images/a1dd6bb1e9a05430db7f613325b2292169710fa1c5cfc94e1a9a8c51bf68189c.jpg)

![b0f092569b929e9a435aec7bae16a9d47f88ebea1f992900ff15f7014c85e084.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/images/b0f092569b929e9a435aec7bae16a9d47f88ebea1f992900ff15f7014c85e084.jpg)

![c81f3ff3e156616868c2cafb8b438e5ad8c40b3e5d317848717485ce50e48cea.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/images/c81f3ff3e156616868c2cafb8b438e5ad8c40b3e5d317848717485ce50e48cea.jpg)

![ed40983669265196d6d316b31e1b7d2c3ecd0338016a71f8f40cdd5213ad7daf.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/images/ed40983669265196d6d316b31e1b7d2c3ecd0338016a71f8f40cdd5213ad7daf.jpg)

![efe7d530d85bd29a4d78c2634c59961a4e502d0c4a427ebc1be7ca23b147e104.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/images/efe7d530d85bd29a4d78c2634c59961a4e502d0c4a427ebc1be7ca23b147e104.jpg)

![fde0cb5f40d41640352b765da540b09f654dfc821c4af77f1f2d7dc3f0ed01ed.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/images/fde0cb5f40d41640352b765da540b09f654dfc821c4af77f1f2d7dc3f0ed01ed.jpg)

### Tables

![0943d8f8709cf7290b969720ab52bb0f6bb2ee048fe5e7b57f71cbe61c6d12af.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/tables/0943d8f8709cf7290b969720ab52bb0f6bb2ee048fe5e7b57f71cbe61c6d12af.jpg)

![0a14e5ed29dd591987597b271c5e09c24a9408fe7e88c49fe82f0aeacd0a4036.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/tables/0a14e5ed29dd591987597b271c5e09c24a9408fe7e88c49fe82f0aeacd0a4036.jpg)

![18d45b3dbf1a4d837fbb5886105ac697c72722edcd8c0b960a3c8b69f508a46f.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/tables/18d45b3dbf1a4d837fbb5886105ac697c72722edcd8c0b960a3c8b69f508a46f.jpg)

![23bbf05dd8566d6d66b77b977022329134626ef0acb9ec8aa60c09ba158d4d1c.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/tables/23bbf05dd8566d6d66b77b977022329134626ef0acb9ec8aa60c09ba158d4d1c.jpg)

![416cf1fa3eb43ab7b550732de3cc2ad4d1d53fcc79f5478572307c7b4939256e.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/tables/416cf1fa3eb43ab7b550732de3cc2ad4d1d53fcc79f5478572307c7b4939256e.jpg)

![4460918bb2d72f9f795299c5d93f990fea5b6044b24b5f874fbae35b344b2df6.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/tables/4460918bb2d72f9f795299c5d93f990fea5b6044b24b5f874fbae35b344b2df6.jpg)

![4c2a32406745bd8e3d194371f7f9a6f3dfdd98150fef05cf655355fbddbc9642.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/tables/4c2a32406745bd8e3d194371f7f9a6f3dfdd98150fef05cf655355fbddbc9642.jpg)

![53633318c69595e0468970fcc6d6dbe4d9dc6fa5bbd2e319dfba01f435d99cc9.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/tables/53633318c69595e0468970fcc6d6dbe4d9dc6fa5bbd2e319dfba01f435d99cc9.jpg)

![54303c97f4491f0b05821a3561eecd7084e49791ead1a3f206acba8e28bbea58.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/tables/54303c97f4491f0b05821a3561eecd7084e49791ead1a3f206acba8e28bbea58.jpg)

![a7af5fcbc9ebbf2abf6f74a63f8f0fec8cebecb932bc4d42ef67dd979b3f1a87.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/tables/a7af5fcbc9ebbf2abf6f74a63f8f0fec8cebecb932bc4d42ef67dd979b3f1a87.jpg)

![aef5489637f77faea05fc35dc4cd516ce28d1ad4d9c8501b2c212bdf157567dc.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/tables/aef5489637f77faea05fc35dc4cd516ce28d1ad4d9c8501b2c212bdf157567dc.jpg)

![b9585708e13a16e8877ca3b9ed611899333faf50c7188a170ce287e324111200.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/tables/b9585708e13a16e8877ca3b9ed611899333faf50c7188a170ce287e324111200.jpg)

![c23b965ffcafe757747b8fe53705b7ea514d494e7a43b22406257b1189973f12.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/tables/c23b965ffcafe757747b8fe53705b7ea514d494e7a43b22406257b1189973f12.jpg)

![dc5f36a86d7a47778cce3513c134fd719db9e361a064956d61d8faf4871eb51c.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/tables/dc5f36a86d7a47778cce3513c134fd719db9e361a064956d61d8faf4871eb51c.jpg)

![e4ee4183c56a88916f0a5c8dc2265dbaf5e6f7dbcf1fea6f6792584168dff4cc.jpg](../icml_results/5260_Probing%20How%20Scalable%20Table%20Data%20Enhances%20General%20Long-Context%20Reasoning/tables/e4ee4183c56a88916f0a5c8dc2265dbaf5e6f7dbcf1fea6f6792584168dff4cc.jpg)

<a id="paper-5213"></a>

## Instruction Decomposition and Action Alignment for Vision-and-Language Navigation


### Images

![10967ce94abb35f650e2d3248fd14b0b79804d5aa2444d04f9c62dce0f655ea1.jpg](../icml_results/5261_Instruction%20Decomposition%20and%20Action%20Alignment%20for%20Vision-and-Language%20Navigation/images/10967ce94abb35f650e2d3248fd14b0b79804d5aa2444d04f9c62dce0f655ea1.jpg)

![195bb35dadbaa510fcc504fe4bd149fcc6174d6c151925476538550774f86f63.jpg](../icml_results/5261_Instruction%20Decomposition%20and%20Action%20Alignment%20for%20Vision-and-Language%20Navigation/images/195bb35dadbaa510fcc504fe4bd149fcc6174d6c151925476538550774f86f63.jpg)

![5c9f5645d95b07d4e7933e0b48b5733b74e02e52b7ac411dbda06c554f14969f.jpg](../icml_results/5261_Instruction%20Decomposition%20and%20Action%20Alignment%20for%20Vision-and-Language%20Navigation/images/5c9f5645d95b07d4e7933e0b48b5733b74e02e52b7ac411dbda06c554f14969f.jpg)

![7e03acbc9df250bc2d81f06ef22a9caf0cc9f41fd97e0f636b148944e759d0d7.jpg](../icml_results/5261_Instruction%20Decomposition%20and%20Action%20Alignment%20for%20Vision-and-Language%20Navigation/images/7e03acbc9df250bc2d81f06ef22a9caf0cc9f41fd97e0f636b148944e759d0d7.jpg)

![7e16b896732a8f3fb64ae3dd905b13410243911be7bd5795a152b0c2a4e658c7.jpg](../icml_results/5261_Instruction%20Decomposition%20and%20Action%20Alignment%20for%20Vision-and-Language%20Navigation/images/7e16b896732a8f3fb64ae3dd905b13410243911be7bd5795a152b0c2a4e658c7.jpg)

![8f23e6cc7efc4b70bffa60b9dee83bb8e2b07f0ec3f94bab3a09c981cebcb087.jpg](../icml_results/5261_Instruction%20Decomposition%20and%20Action%20Alignment%20for%20Vision-and-Language%20Navigation/images/8f23e6cc7efc4b70bffa60b9dee83bb8e2b07f0ec3f94bab3a09c981cebcb087.jpg)

### Tables

![09896bc5a2a6ab94b91600f01057a10731b07c36a32747f764715c1df8220863.jpg](../icml_results/5261_Instruction%20Decomposition%20and%20Action%20Alignment%20for%20Vision-and-Language%20Navigation/tables/09896bc5a2a6ab94b91600f01057a10731b07c36a32747f764715c1df8220863.jpg)

![26accdeff6df42087c68cb4829da71589cf516b7289425688153b54db42cda0d.jpg](../icml_results/5261_Instruction%20Decomposition%20and%20Action%20Alignment%20for%20Vision-and-Language%20Navigation/tables/26accdeff6df42087c68cb4829da71589cf516b7289425688153b54db42cda0d.jpg)

![4ab31194c507e9dac74ce9152dfc5140c374a9851014ba41b317636fdc979272.jpg](../icml_results/5261_Instruction%20Decomposition%20and%20Action%20Alignment%20for%20Vision-and-Language%20Navigation/tables/4ab31194c507e9dac74ce9152dfc5140c374a9851014ba41b317636fdc979272.jpg)

![9d2520c155a2f738f81bfe8f56091d411ccdbcc711b14feda481c84e5877b89b.jpg](../icml_results/5261_Instruction%20Decomposition%20and%20Action%20Alignment%20for%20Vision-and-Language%20Navigation/tables/9d2520c155a2f738f81bfe8f56091d411ccdbcc711b14feda481c84e5877b89b.jpg)

![c20226bc64b927698fb63d4c6d141974242696adba40c62d4bd0c7f67c5bbb64.jpg](../icml_results/5261_Instruction%20Decomposition%20and%20Action%20Alignment%20for%20Vision-and-Language%20Navigation/tables/c20226bc64b927698fb63d4c6d141974242696adba40c62d4bd0c7f67c5bbb64.jpg)

![cd8c70e864814c3bc604b88cdf512e11d08226616fdb0e64196c0ff4d2271e78.jpg](../icml_results/5261_Instruction%20Decomposition%20and%20Action%20Alignment%20for%20Vision-and-Language%20Navigation/tables/cd8c70e864814c3bc604b88cdf512e11d08226616fdb0e64196c0ff4d2271e78.jpg)

![cf64c7585d2ddf572e87114d33255ae60953ca032c60f0d388aa3cfccbf170d5.jpg](../icml_results/5261_Instruction%20Decomposition%20and%20Action%20Alignment%20for%20Vision-and-Language%20Navigation/tables/cf64c7585d2ddf572e87114d33255ae60953ca032c60f0d388aa3cfccbf170d5.jpg)

<a id="paper-5214"></a>

## FedVeer: Self-Adaptive Skew Estimation for Robust Federated Learning

### Images

![32320bc818f81ad5be22ac499df56309e69d7e82b7682b3328fa1754fe7c8970.jpg](../icml_results/5262_FedVeer_%20Self-Adaptive%20Skew%20Estimation%20for%20Robust%20Federated%20Learning/images/32320bc818f81ad5be22ac499df56309e69d7e82b7682b3328fa1754fe7c8970.jpg)

![47111d1d2b6e81ff4eccc8b83e5265964b70ce4810148e2b9d66a7fc456cc086.jpg](../icml_results/5262_FedVeer_%20Self-Adaptive%20Skew%20Estimation%20for%20Robust%20Federated%20Learning/images/47111d1d2b6e81ff4eccc8b83e5265964b70ce4810148e2b9d66a7fc456cc086.jpg)

![5c0d903bd79dc1b0772c0db501cd1287998e631110fd0d3c4542f61f912ab112.jpg](../icml_results/5262_FedVeer_%20Self-Adaptive%20Skew%20Estimation%20for%20Robust%20Federated%20Learning/images/5c0d903bd79dc1b0772c0db501cd1287998e631110fd0d3c4542f61f912ab112.jpg)

![774557d9e671367eb28cbafd33627263719326abcbc7b1deb7b7a75e76a2ade7.jpg](../icml_results/5262_FedVeer_%20Self-Adaptive%20Skew%20Estimation%20for%20Robust%20Federated%20Learning/images/774557d9e671367eb28cbafd33627263719326abcbc7b1deb7b7a75e76a2ade7.jpg)

![81f02edca11b7508a03e6066144f353f5cd1625d1916c618dcd188bea6f480af.jpg](../icml_results/5262_FedVeer_%20Self-Adaptive%20Skew%20Estimation%20for%20Robust%20Federated%20Learning/images/81f02edca11b7508a03e6066144f353f5cd1625d1916c618dcd188bea6f480af.jpg)

![ccaef765d6988167c0111d9e20980956c502c7b3027c7c44c022ee786886506f.jpg](../icml_results/5262_FedVeer_%20Self-Adaptive%20Skew%20Estimation%20for%20Robust%20Federated%20Learning/images/ccaef765d6988167c0111d9e20980956c502c7b3027c7c44c022ee786886506f.jpg)

![e1382e1c2a09673b95cebc2cedb8b297124e3c2334b43d88d415d88acba4b4f4.jpg](../icml_results/5262_FedVeer_%20Self-Adaptive%20Skew%20Estimation%20for%20Robust%20Federated%20Learning/images/e1382e1c2a09673b95cebc2cedb8b297124e3c2334b43d88d415d88acba4b4f4.jpg)

![e66ab4198ea1260f5ad4e8985ecb9bdbf88b70290044438abe8663183a09b52e.jpg](../icml_results/5262_FedVeer_%20Self-Adaptive%20Skew%20Estimation%20for%20Robust%20Federated%20Learning/images/e66ab4198ea1260f5ad4e8985ecb9bdbf88b70290044438abe8663183a09b52e.jpg)

### Tables

![00da2e2519317e5ba4dc1293a3b108896f97f970abb3e717977e3a3cac4549ad.jpg](../icml_results/5262_FedVeer_%20Self-Adaptive%20Skew%20Estimation%20for%20Robust%20Federated%20Learning/tables/00da2e2519317e5ba4dc1293a3b108896f97f970abb3e717977e3a3cac4549ad.jpg)

![343d9962c843ea85c8cdf9d035fc88c9996554a4653ccf3e92e5ad69e9ac8dac.jpg](../icml_results/5262_FedVeer_%20Self-Adaptive%20Skew%20Estimation%20for%20Robust%20Federated%20Learning/tables/343d9962c843ea85c8cdf9d035fc88c9996554a4653ccf3e92e5ad69e9ac8dac.jpg)

![65c9a005194040c37a94e15dfc4bb2d673e0996739642cc0d17f2e6bc8320f33.jpg](../icml_results/5262_FedVeer_%20Self-Adaptive%20Skew%20Estimation%20for%20Robust%20Federated%20Learning/tables/65c9a005194040c37a94e15dfc4bb2d673e0996739642cc0d17f2e6bc8320f33.jpg)

![764df828175c5622704b0493dd9206500d4c300e54a4d0f5e985ad811735a8f8.jpg](../icml_results/5262_FedVeer_%20Self-Adaptive%20Skew%20Estimation%20for%20Robust%20Federated%20Learning/tables/764df828175c5622704b0493dd9206500d4c300e54a4d0f5e985ad811735a8f8.jpg)

![d536a1f62495ee9b0327ff12aa4c189ac10dd698f638763d3f8144a7bf39eec0.jpg](../icml_results/5262_FedVeer_%20Self-Adaptive%20Skew%20Estimation%20for%20Robust%20Federated%20Learning/tables/d536a1f62495ee9b0327ff12aa4c189ac10dd698f638763d3f8144a7bf39eec0.jpg)
