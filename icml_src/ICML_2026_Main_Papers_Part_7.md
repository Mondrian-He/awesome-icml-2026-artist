# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 7 of 197

## 目录 (Table of Contents)

1. [DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation](#paper-0199)
2. [Time series saliency maps: Explaining models across multiple domains](#paper-0200)
3. [Manifold-Aware Perturbations for Constrained Generative Modeling](#paper-0201)
4. [Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges](#paper-0202)
5. [Linguistic Nepotism: Trading-off Quality for Language Preference in Multilingual RAG](#paper-0203)
6. [A Regret Minimization Framework on Preference Learning in Large Language Models](#paper-0204)
7. [On the Sharp Input-Output Analysis of Nonlinear Systems under Adversarial Attacks](#paper-0205)
8. [VALUEFLOW: Toward Pluralistic and Steerable Value-based Alignment in Large Language Models](#paper-0206)
9. [Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models](#paper-0207)
10. [When to Trust the Cheap Check: Weak and Strong Verification for Reasoning](#paper-0208)
11. [Latent Spherical Flow Policy for Reinforcement Learning with Combinatorial Actions](#paper-0209)
12. [The Axiomatic Value of Regularization in AI Alignment from Human Preferences](#paper-0210)
13. [Efficient Numeracy in Language Models through Single-Token Number Encodings](#paper-0211)
14. [Disentangling Geometry, Performance, and Training in Language Models](#paper-0212)
15. [Learning Structured Reasoning via Tractable Trajectory Control](#paper-0213)
16. [Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance](#paper-0214)
17. [AI Engram: In Search of Memory Traces in Artificial Intelligence](#paper-0215)
18. [Guaranteed Optimal Compositional Explanations for Neurons](#paper-0216)
19. [Simultaneous Speech-to-Speech Translation Without Aligned Data](#paper-0217)
20. [Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](#paper-0218)
21. [Minimax Optimal Strategy for Delayed Observations in Online Reinforcement Learning](#paper-0219)
22. [Stabilizing the Q-Gradient Field for Policy Smoothness in Actor-Critic Methods](#paper-0220)
23. [ReQAT: Achieving Full-Precision Reasoning Accuracy with 4-bit Floating-Point Quantization-Aware Training](#paper-0221)
24. [FOCUS & RePAIR: Mitigating Text Degeneration via Token-Level Guidance For Pruned Large Language Models](#paper-0222)
25. [Root Cause Analysis of Failures in Microservices via Bayesian Root Cause Discovery](#paper-0223)
26. [Near-Optimal Private Linear Regression via Iterative Hessian Mixing](#paper-0224)
27. [MetaphorVU: Towards Metaphorical Video Understanding](#paper-0225)
28. [DroneDINO: Towards Heterogeneous Routed Mixture of Experts for Drone-based Unified Object Detection](#paper-0226)
29. [SAW-Bench: Learning Situated Awareness in the Real World](#paper-0227)
30. [OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration](#paper-0228)
31. [What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity](#paper-0229)
32. [A Kinetic Energy Perspective of Flow Matching](#paper-0230)
33. [FIDIA: Function-Informed Sequence Design via Inference-Aligned Policy Optimization](#paper-0231)

---


<a id="paper-0199"></a>

## DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation

### Images

![0468f573244f16e8b55257e3237cd51e91bf40d3ba8728144b6d08b99933038f.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/0468f573244f16e8b55257e3237cd51e91bf40d3ba8728144b6d08b99933038f.jpg)

![06baa71221f73475157be8ea688a01064a82e13dd56b087c85a79c2f58cc660a.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/06baa71221f73475157be8ea688a01064a82e13dd56b087c85a79c2f58cc660a.jpg)

![072af555f2d46ecc13cd3a9ed6cfced2b2f18f65d7a6dc97716ae40c45c5bdb8.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/072af555f2d46ecc13cd3a9ed6cfced2b2f18f65d7a6dc97716ae40c45c5bdb8.jpg)

![2199ec9d3afb240d8ffa49bec9f849e7e72704c9f7a65a68a8eab0c5bd5157db.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/2199ec9d3afb240d8ffa49bec9f849e7e72704c9f7a65a68a8eab0c5bd5157db.jpg)

![423313bd7e9e3e1f8b8e2f4b041d9f70877bcca36965389ef5a68e463b29c92c.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/423313bd7e9e3e1f8b8e2f4b041d9f70877bcca36965389ef5a68e463b29c92c.jpg)

![5226fbe8a5726d8afeeb702ddac7b1086eae69097b0c1b0d421531f3fe1d7b11.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/5226fbe8a5726d8afeeb702ddac7b1086eae69097b0c1b0d421531f3fe1d7b11.jpg)

![680bb4f7492d80af00ccc6e766a30fe528f8a7e4228f748ab6b6f9be018b1c20.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/680bb4f7492d80af00ccc6e766a30fe528f8a7e4228f748ab6b6f9be018b1c20.jpg)

![69c64a150b2603670441c69565b3b93ed85e33fc6ac48fb5e8d816448ce639c1.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/69c64a150b2603670441c69565b3b93ed85e33fc6ac48fb5e8d816448ce639c1.jpg)

![8539e25ab444f985e86ac6f2bf1412f08816bd0afe7ca9f9d381ccba693200a4.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/8539e25ab444f985e86ac6f2bf1412f08816bd0afe7ca9f9d381ccba693200a4.jpg)

![8bc6d8a11dcfba2f1deabf41a5db8bc8ca41ecfc4c852d424c69036f66ab5116.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/8bc6d8a11dcfba2f1deabf41a5db8bc8ca41ecfc4c852d424c69036f66ab5116.jpg)

![8d6b0980188307ba9d960ddc0a419250d9f8f4a272ddae8a2e299bad7b8861e8.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/8d6b0980188307ba9d960ddc0a419250d9f8f4a272ddae8a2e299bad7b8861e8.jpg)

![9521679e41f5347804d95a2593dc86adaefbf29bf319ebf3b566cd61216092b7.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/9521679e41f5347804d95a2593dc86adaefbf29bf319ebf3b566cd61216092b7.jpg)

![9977d119c275cddb13d8c71a996d199137d12f157ac3522236f051dc3f47f0f2.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/9977d119c275cddb13d8c71a996d199137d12f157ac3522236f051dc3f47f0f2.jpg)

![a5a1a7f6c3ef70e3cb0de48a33935758b64c3e24be0d9bea9bbf2ad812eab1ca.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/a5a1a7f6c3ef70e3cb0de48a33935758b64c3e24be0d9bea9bbf2ad812eab1ca.jpg)

![c20a2ae8626a8a90633324f591d5886e310bab2f4bc74d41fd33f5b97868f714.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/c20a2ae8626a8a90633324f591d5886e310bab2f4bc74d41fd33f5b97868f714.jpg)

![cbb4b5a0973a9e78925f502793907ad8a4844ad8b81767317914bbbf86d84f79.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/cbb4b5a0973a9e78925f502793907ad8a4844ad8b81767317914bbbf86d84f79.jpg)

![d074a6c6c98cf691368768ceb61c7dcc9e68ef450b2c40c24f58fc5a570ccc0a.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/d074a6c6c98cf691368768ceb61c7dcc9e68ef450b2c40c24f58fc5a570ccc0a.jpg)

![d9b3847d9d6fe7b570d9aa02b0d498773feec6239e7471e54fcda966a7fb552b.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/d9b3847d9d6fe7b570d9aa02b0d498773feec6239e7471e54fcda966a7fb552b.jpg)

![dbff832e889f59bdde02ce174dba4ceac1663c4903114b219bd7b5af24bf663a.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/dbff832e889f59bdde02ce174dba4ceac1663c4903114b219bd7b5af24bf663a.jpg)

![f51b097d3b2eaacce6782bf3dfdf6fad6762fc907720aa18925a765092e719aa.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/f51b097d3b2eaacce6782bf3dfdf6fad6762fc907720aa18925a765092e719aa.jpg)

![f5eeca36e4c4d10f99393286e86abf501bc5265614afe41d816a95dd3b56f654.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/f5eeca36e4c4d10f99393286e86abf501bc5265614afe41d816a95dd3b56f654.jpg)

![f6f130d06253a8363912a06d95a51ecb1d1c6d636d56e23589469e81f959b7e0.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/images/f6f130d06253a8363912a06d95a51ecb1d1c6d636d56e23589469e81f959b7e0.jpg)

### Tables

![99df50d9f89ae14a37ed92553a62dfef2f67eec22584adf52747a1a499e95053.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/tables/99df50d9f89ae14a37ed92553a62dfef2f67eec22584adf52747a1a499e95053.jpg)

![bba7e4fffb09a6df58cda2409b34a97eadbb977a051810fc04b5b21c432116ba.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/tables/bba7e4fffb09a6df58cda2409b34a97eadbb977a051810fc04b5b21c432116ba.jpg)

![f9cd0f220208815d76f940c4ead0b76d50bcbc73aec775c2ed6c423941fbde67.jpg](../icml_results/201_Activation%20Oracles_%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers/tables/f9cd0f220208815d76f940c4ead0b76d50bcbc73aec775c2ed6c423941fbde67.jpg)

## DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation


### Images

![11812c2d5a6036bd90df0ce4d33a70d4d2c7ea49ff4143bf5b52cb3b76fb534a.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/images/11812c2d5a6036bd90df0ce4d33a70d4d2c7ea49ff4143bf5b52cb3b76fb534a.jpg)

![22f3b8aca6ec229a57312cf9bdb8cf1e575bed78b756a09f825201a7624f1441.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/images/22f3b8aca6ec229a57312cf9bdb8cf1e575bed78b756a09f825201a7624f1441.jpg)

![29319913e70fc1c35ded267eb5a00da84f257a809c23e65f26bc5b5b9d02bc98.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/images/29319913e70fc1c35ded267eb5a00da84f257a809c23e65f26bc5b5b9d02bc98.jpg)

![48ca86e6689347776055b8a254956b8470adb55e120e7293eac4908f04c906f7.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/images/48ca86e6689347776055b8a254956b8470adb55e120e7293eac4908f04c906f7.jpg)

![4a2b5300b886a366e62ca6eaffd5f678599bcab86970d97a7782546df56cc3e5.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/images/4a2b5300b886a366e62ca6eaffd5f678599bcab86970d97a7782546df56cc3e5.jpg)

![b24ec27c30c969ee65ffb8e6c5e48f4082a0b5b56114557e6d0e1a837e7a7e55.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/images/b24ec27c30c969ee65ffb8e6c5e48f4082a0b5b56114557e6d0e1a837e7a7e55.jpg)

![b516564ad009ae07f262cfea52fd563a33b50aa63765f3789e5b808a6b1e02b0.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/images/b516564ad009ae07f262cfea52fd563a33b50aa63765f3789e5b808a6b1e02b0.jpg)

![c8eeab4e24cbe975d8e47f8d0816d59faad56a31e579ff4039244b98f7015424.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/images/c8eeab4e24cbe975d8e47f8d0816d59faad56a31e579ff4039244b98f7015424.jpg)

![d2dba56d7c68a2b47d7e925ed2af5279dacbf4fbd53f9334c4338c18a0ac4e73.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/images/d2dba56d7c68a2b47d7e925ed2af5279dacbf4fbd53f9334c4338c18a0ac4e73.jpg)

![e3e51a51817ac9d0e8845284175158b2045181c43c4ed2cdb8182e4a26610dc3.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/images/e3e51a51817ac9d0e8845284175158b2045181c43c4ed2cdb8182e4a26610dc3.jpg)

![eac8fa723c6e93ed766f8a2905c01418fbec6ea5f4b123bceae4a82c4ce3e818.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/images/eac8fa723c6e93ed766f8a2905c01418fbec6ea5f4b123bceae4a82c4ce3e818.jpg)

### Tables

![1772c41e457bf34174423db48d75329fce4de45767e40369a06b42aa0dc9cffb.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/tables/1772c41e457bf34174423db48d75329fce4de45767e40369a06b42aa0dc9cffb.jpg)

![1b6834da50248bf3198777750e42771ce87f9de94e8a7244b6c3c1d9fd91a6de.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/tables/1b6834da50248bf3198777750e42771ce87f9de94e8a7244b6c3c1d9fd91a6de.jpg)

![34fc4e2d02cd0f9d6c97a9e006be910b4a97c581f76a38a5a20d3033acd95837.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/tables/34fc4e2d02cd0f9d6c97a9e006be910b4a97c581f76a38a5a20d3033acd95837.jpg)

![52f77f8593404ec115494957bbf0e6df8f3ae691eb0ffd316ee7c0d89ccc39c0.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/tables/52f77f8593404ec115494957bbf0e6df8f3ae691eb0ffd316ee7c0d89ccc39c0.jpg)

![853b8bfb15c9efca309e8987718fe86ea4a7f5bf1ef73f4bb1af28120be2affc.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/tables/853b8bfb15c9efca309e8987718fe86ea4a7f5bf1ef73f4bb1af28120be2affc.jpg)

![cc6f6658c1afe3d0a2592d82968ae3d291d00ae6a1cfc1d84e32b4e6437364a8.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/tables/cc6f6658c1afe3d0a2592d82968ae3d291d00ae6a1cfc1d84e32b4e6437364a8.jpg)

![de46f5028a58295e5bcfa8f657b44896c2a02313053282648eb1adb18a3e89cb.jpg](../icml_results/202_DISCO_%20Mitigating%20Bias%20in%20Deep%20Learning%20with%20Conditional%20Distance%20Correlation/tables/de46f5028a58295e5bcfa8f657b44896c2a02313053282648eb1adb18a3e89cb.jpg)

<a id="paper-0200"></a>

## Time series saliency maps: Explaining models across multiple domains


### Images

![115a3638cb25b3dd929392e60ac5cd325e376ded5bf30dda67cff15c9a3ffaad.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/images/115a3638cb25b3dd929392e60ac5cd325e376ded5bf30dda67cff15c9a3ffaad.jpg)

![15190f8a3fc0c0d6aef44cd4113da6075e8860cf383c2215f77631479a7a6eb3.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/images/15190f8a3fc0c0d6aef44cd4113da6075e8860cf383c2215f77631479a7a6eb3.jpg)

![1d7828594a7cffee9061be364bef5f65e6915159f95156bf5b8dc0805f855f51.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/images/1d7828594a7cffee9061be364bef5f65e6915159f95156bf5b8dc0805f855f51.jpg)

![25598bd9ba6fb51c520c0faab3016b79b9b3ce99a2a946a0ea84a4d34b5fd6b0.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/images/25598bd9ba6fb51c520c0faab3016b79b9b3ce99a2a946a0ea84a4d34b5fd6b0.jpg)

![31dcdf742240f532e5071c5f426812e3ff4fe33464bba762e58be23d287a88e9.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/images/31dcdf742240f532e5071c5f426812e3ff4fe33464bba762e58be23d287a88e9.jpg)

![51d178bb6d8eed11817b5cd81c42896d31af201331512bef32aa3b67ff2f56e3.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/images/51d178bb6d8eed11817b5cd81c42896d31af201331512bef32aa3b67ff2f56e3.jpg)

![655fcf39b92afc864d4d41f2931e5a2e75d6982842738e9bbc528b503418a9b2.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/images/655fcf39b92afc864d4d41f2931e5a2e75d6982842738e9bbc528b503418a9b2.jpg)

![6dab73bc4a43fd7bf45ec4e774db534300cb8e3c793131273fd7ff325da785d7.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/images/6dab73bc4a43fd7bf45ec4e774db534300cb8e3c793131273fd7ff325da785d7.jpg)

![7fb3e155b6b2440eb855460127c4fe3e33a793cbad2542a67e298039d20dfc31.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/images/7fb3e155b6b2440eb855460127c4fe3e33a793cbad2542a67e298039d20dfc31.jpg)

![9d093d5790427fba0ec58a9ff6ae82793809ca2b6b99c46abc87a9271873001e.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/images/9d093d5790427fba0ec58a9ff6ae82793809ca2b6b99c46abc87a9271873001e.jpg)

![aae0eb974a70e43ddd471a01014c765c86cd080b3df197ad9c50960262d83a7c.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/images/aae0eb974a70e43ddd471a01014c765c86cd080b3df197ad9c50960262d83a7c.jpg)

![c22f07788e55a0b9a74f6db0b6fb1a4d274dfc73000bdfc1b898391f1e919275.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/images/c22f07788e55a0b9a74f6db0b6fb1a4d274dfc73000bdfc1b898391f1e919275.jpg)

![cf604b7d12d74af71368de4f1337f50d85fe9a889d54b36ff8c69b2e6ca3b6a2.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/images/cf604b7d12d74af71368de4f1337f50d85fe9a889d54b36ff8c69b2e6ca3b6a2.jpg)

![d195f6a0164ae6b18fe68c35a837cc7f8802303ac965b8e162a8a2641f8c9112.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/images/d195f6a0164ae6b18fe68c35a837cc7f8802303ac965b8e162a8a2641f8c9112.jpg)

![def52ce6b7694010e7cb0cc3c4f23e55da3bcbbaf12f3e83ea93c72a7f54e8aa.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/images/def52ce6b7694010e7cb0cc3c4f23e55da3bcbbaf12f3e83ea93c72a7f54e8aa.jpg)

![fe39f0ce62994ee11a081c0ef3ed02879c5de2d3e1efdc61e0d48aba259d70ef.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/images/fe39f0ce62994ee11a081c0ef3ed02879c5de2d3e1efdc61e0d48aba259d70ef.jpg)

### Tables

![19d25fbe51154cc378e8e6baeca6474f4f0bbb5653248add8c98ac58813f57d4.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/tables/19d25fbe51154cc378e8e6baeca6474f4f0bbb5653248add8c98ac58813f57d4.jpg)

![34054743a3e2cd28c6f140e0f88a1e7e2d494c3d64265e5038f0b26f2fac3665.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/tables/34054743a3e2cd28c6f140e0f88a1e7e2d494c3d64265e5038f0b26f2fac3665.jpg)

![6c705786e098beec935f250e5ab856c7b1c29efad518de8577e674a33973c148.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/tables/6c705786e098beec935f250e5ab856c7b1c29efad518de8577e674a33973c148.jpg)

![a56c141ca4b511dfd3c89e3b2fc5772608458af0d748464933c97f8e673ad1dd.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/tables/a56c141ca4b511dfd3c89e3b2fc5772608458af0d748464933c97f8e673ad1dd.jpg)

![a77fbb025754fce97d931677863a59f52e1618c9dc9d535480b36b0ba7feadb9.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/tables/a77fbb025754fce97d931677863a59f52e1618c9dc9d535480b36b0ba7feadb9.jpg)

![c39c19da757392261cc2d419a3493dbc4c562577534f57222d85076424d95c09.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/tables/c39c19da757392261cc2d419a3493dbc4c562577534f57222d85076424d95c09.jpg)

![c7ffc8e20dbacbd7560f21af9f5725a214c5b8b4e664ebac4bbd0a51c6319238.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/tables/c7ffc8e20dbacbd7560f21af9f5725a214c5b8b4e664ebac4bbd0a51c6319238.jpg)

![d5eb4e949e19468704057d3cd19cb3c80c6e0897e2adad70fa438facf070b80e.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/tables/d5eb4e949e19468704057d3cd19cb3c80c6e0897e2adad70fa438facf070b80e.jpg)

![f1053b91eca9c6a7fa7e389e67097a2d1099c104caf9cbf42a372964a6f31a8d.jpg](../icml_results/203_Time%20series%20saliency%20maps_%20Explaining%20models%20across%20multiple%20domains/tables/f1053b91eca9c6a7fa7e389e67097a2d1099c104caf9cbf42a372964a6f31a8d.jpg)

<a id="paper-0201"></a>

## Manifold-Aware Perturbations for Constrained Generative Modeling


### Images

![0e9f483e645a5c40ee23d2df63f6aea491e019d5d574ecf8d5160bb84b92d40a.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/images/0e9f483e645a5c40ee23d2df63f6aea491e019d5d574ecf8d5160bb84b92d40a.jpg)

![1f28d12128c61a088e33e36550f38f7c54f9b6819356e555ab9ec6bbf081013f.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/images/1f28d12128c61a088e33e36550f38f7c54f9b6819356e555ab9ec6bbf081013f.jpg)

![1f54d215255eafbefab25a06651fbff6a5f54d5c9c0b81b2bd1cd38198d6d12c.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/images/1f54d215255eafbefab25a06651fbff6a5f54d5c9c0b81b2bd1cd38198d6d12c.jpg)

![2ca69f4326efd7b7f6b0d396179925d64be9df869aebf0dc6a39ab30ffee1488.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/images/2ca69f4326efd7b7f6b0d396179925d64be9df869aebf0dc6a39ab30ffee1488.jpg)

![313ef30a1025f1b5abc06c6133aff7158f6a46f66261bfa68161862468e71060.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/images/313ef30a1025f1b5abc06c6133aff7158f6a46f66261bfa68161862468e71060.jpg)

![37c6e29514ebc12431d8b910067aa79bc456bf8ec864f5f08a784b99f09fe9f5.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/images/37c6e29514ebc12431d8b910067aa79bc456bf8ec864f5f08a784b99f09fe9f5.jpg)

![556ce94632bc219ddd2b413c7935422de5b2818e62cc36c737a07043e8753368.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/images/556ce94632bc219ddd2b413c7935422de5b2818e62cc36c737a07043e8753368.jpg)

![599d80f3454fa98922df17deaf3e7e92a380619a81abd3a2dcb268806065b45b.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/images/599d80f3454fa98922df17deaf3e7e92a380619a81abd3a2dcb268806065b45b.jpg)

![64425c2a45220c7b481ac35e97bba9d126cd683fd937d557766f3243bcab9060.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/images/64425c2a45220c7b481ac35e97bba9d126cd683fd937d557766f3243bcab9060.jpg)

![80c208e227d753d56a0210b884124183d5104267f67243228a8bc80a69c14080.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/images/80c208e227d753d56a0210b884124183d5104267f67243228a8bc80a69c14080.jpg)

![9685f9a4688d4b0981d062cc12881e6ce884b21a4cab7b2296783adcb58d1b7b.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/images/9685f9a4688d4b0981d062cc12881e6ce884b21a4cab7b2296783adcb58d1b7b.jpg)

![c6b47c657a193cbebed790db94190cda89615bd041ad6796113722b4ab33d633.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/images/c6b47c657a193cbebed790db94190cda89615bd041ad6796113722b4ab33d633.jpg)

![d75ebf52f1d43a5e6a685f2fd1500f199357e85517da4ac71adf526d67177c4a.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/images/d75ebf52f1d43a5e6a685f2fd1500f199357e85517da4ac71adf526d67177c4a.jpg)

![e924eedeabfee857b04483966439cb4b9530903d2d25e0313167cf337ae3ca14.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/images/e924eedeabfee857b04483966439cb4b9530903d2d25e0313167cf337ae3ca14.jpg)

![f5657ffe1792f60697bb51779fde29931285c053ea290ea1107f9e67cc42f141.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/images/f5657ffe1792f60697bb51779fde29931285c053ea290ea1107f9e67cc42f141.jpg)

### Tables

![4f76c83491cb8122c2deaa93a8cb956dbdd44ebaba9efb9d490e2e40f832cab6.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/tables/4f76c83491cb8122c2deaa93a8cb956dbdd44ebaba9efb9d490e2e40f832cab6.jpg)

![571d4d08d50a2c0e1097fa8d9c5a5590d2a7757f513660a73f8a022369210e40.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/tables/571d4d08d50a2c0e1097fa8d9c5a5590d2a7757f513660a73f8a022369210e40.jpg)

![5e9d66a6c98f97e5b7bbfb776ea9ebe95353136c752123a82ddcc636f68c59a7.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/tables/5e9d66a6c98f97e5b7bbfb776ea9ebe95353136c752123a82ddcc636f68c59a7.jpg)

![68f0f69f9685d93e14db206fee368ccd9c7584e3a7ee79ce873ce1a1360c23c5.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/tables/68f0f69f9685d93e14db206fee368ccd9c7584e3a7ee79ce873ce1a1360c23c5.jpg)

![8fac952914cadaaa22f796a49a52a23fb96e6d761678526c6fada7ec28614457.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/tables/8fac952914cadaaa22f796a49a52a23fb96e6d761678526c6fada7ec28614457.jpg)

![a1eefa335b8b2dd5cab2bf01ffcf8d8c52e008b8dae81d5883a176a05c1573a2.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/tables/a1eefa335b8b2dd5cab2bf01ffcf8d8c52e008b8dae81d5883a176a05c1573a2.jpg)

![bd32ae4e3d630dbbbe80b67fbfb3787b9fed4afd49deb95512c4862f2624313a.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/tables/bd32ae4e3d630dbbbe80b67fbfb3787b9fed4afd49deb95512c4862f2624313a.jpg)

![d8e37ebf8d8079ae07ec6b7eada2ba5b2f1864b240a0bf989ecbebc63f6df5ce.jpg](../icml_results/204_Manifold-Aware%20Perturbations%20for%20Constrained%20Generative%20Modeling/tables/d8e37ebf8d8079ae07ec6b7eada2ba5b2f1864b240a0bf989ecbebc63f6df5ce.jpg)

<a id="paper-0202"></a>

## Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges


### Images

![133a49e49c320bafabf14d1381510989e6d16e59ed6b72c03cd95a4de6809075.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/133a49e49c320bafabf14d1381510989e6d16e59ed6b72c03cd95a4de6809075.jpg)

![137cacdd26d07688af07b0f3acf3bd55186559ffada4db3a0eff238c44269d73.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/137cacdd26d07688af07b0f3acf3bd55186559ffada4db3a0eff238c44269d73.jpg)

![1429ba334d99230cdf39c35e2ea81e0acb3866bd37b885c1edc1b78b8a20915e.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/1429ba334d99230cdf39c35e2ea81e0acb3866bd37b885c1edc1b78b8a20915e.jpg)

![153e41351820bff3d5d5b461398cebb7efd4b081cfdedbc8e06bfd612171a568.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/153e41351820bff3d5d5b461398cebb7efd4b081cfdedbc8e06bfd612171a568.jpg)

![1c40c97a57c7841499bd48f49ebc0bf4be5e2aba5c589254ac59a220ae03a28b.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/1c40c97a57c7841499bd48f49ebc0bf4be5e2aba5c589254ac59a220ae03a28b.jpg)

![1da20cf221967d689ff62382c1183c88ac411791bd46223aa8002aeb994ee597.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/1da20cf221967d689ff62382c1183c88ac411791bd46223aa8002aeb994ee597.jpg)

![26137b699df53075c6c9eb5d5bab4cc6f908f6095a5fdab791f47a01d5931fe8.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/26137b699df53075c6c9eb5d5bab4cc6f908f6095a5fdab791f47a01d5931fe8.jpg)

![28bac630d81b5ab0949f36fa5f10d63f090b261a16c236c7349322fff0e19059.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/28bac630d81b5ab0949f36fa5f10d63f090b261a16c236c7349322fff0e19059.jpg)

![37e355a293e1f9a1dc6f3da2497fd05a251341eed5c5a816481aae230dd6a4bd.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/37e355a293e1f9a1dc6f3da2497fd05a251341eed5c5a816481aae230dd6a4bd.jpg)

![394fa8952194eeb679895570ce03ead39664cd5ce5c64064fcb94a00f573ecb0.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/394fa8952194eeb679895570ce03ead39664cd5ce5c64064fcb94a00f573ecb0.jpg)

![496930e654b7b6b5753b5002edee49a3185c0c7ff73b9803b6e46134881d8742.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/496930e654b7b6b5753b5002edee49a3185c0c7ff73b9803b6e46134881d8742.jpg)

![51b55a3cdf5b29580617dbaafbd6473b4792b97329ace61dbc684c24dd02496d.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/51b55a3cdf5b29580617dbaafbd6473b4792b97329ace61dbc684c24dd02496d.jpg)

![68cb5384689e38c1bbc1d46ea6f59bdad4fb7c4c6a6d5fe4ccf1fa803227183c.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/68cb5384689e38c1bbc1d46ea6f59bdad4fb7c4c6a6d5fe4ccf1fa803227183c.jpg)

![6cb2c8b3680d9ccb016dbd6dfee758bdcfa773bf576a821ac8b5287ed7f64681.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/6cb2c8b3680d9ccb016dbd6dfee758bdcfa773bf576a821ac8b5287ed7f64681.jpg)

![6d2a1c47995938790af915cc8f86a94e077f363aa32cbb038a43f08951bf8aab.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/6d2a1c47995938790af915cc8f86a94e077f363aa32cbb038a43f08951bf8aab.jpg)

![775d32608d5153030dfe2a6201e9d502c83707a91eb631a8cb4cbcaa909e4c4d.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/775d32608d5153030dfe2a6201e9d502c83707a91eb631a8cb4cbcaa909e4c4d.jpg)

![7e8074d9327147fd00369070601fdfa2f4d17bb47ed1344807e63d9d9e9c2bde.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/7e8074d9327147fd00369070601fdfa2f4d17bb47ed1344807e63d9d9e9c2bde.jpg)

![883a70c8e14489cb9655fe1476ee6c60e38e837b3ce1c08d370e1c8b12635b1d.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/883a70c8e14489cb9655fe1476ee6c60e38e837b3ce1c08d370e1c8b12635b1d.jpg)

![98ff0ee4ebc5057f50dc7b9059c6c88bccd5fed1ea2d2ea5e59c34ab30831caa.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/98ff0ee4ebc5057f50dc7b9059c6c88bccd5fed1ea2d2ea5e59c34ab30831caa.jpg)

![9d3a68c7d92eac588637ef1fe29f466963782b621e01bb1b9e2ca30858fc9f16.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/9d3a68c7d92eac588637ef1fe29f466963782b621e01bb1b9e2ca30858fc9f16.jpg)

![9d62de70fcc23b98f1841abebf44decba3f5a0bbd5042b5da1100524fc7b28c5.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/9d62de70fcc23b98f1841abebf44decba3f5a0bbd5042b5da1100524fc7b28c5.jpg)

![a58f1fdd20bea010261c3fbbc5ccbdaad86a5000a6704dd4bb9275e05097b015.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/a58f1fdd20bea010261c3fbbc5ccbdaad86a5000a6704dd4bb9275e05097b015.jpg)

![b3e0744452d126e11a2f5c6af22c58f4c95631894ff8a80836ba57162be0c84d.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/b3e0744452d126e11a2f5c6af22c58f4c95631894ff8a80836ba57162be0c84d.jpg)

![c62844a9439c1f8fdf16a0a91b88b4ac0425f0624737d62cc003375a45b0b609.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/c62844a9439c1f8fdf16a0a91b88b4ac0425f0624737d62cc003375a45b0b609.jpg)

![c781ced76380226ee85532ebff8fa895b48f599b61810080f4071aa017a70761.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/c781ced76380226ee85532ebff8fa895b48f599b61810080f4071aa017a70761.jpg)

![d123865a1435c7e51f3b36e0e9e8ba1fd6fcef8fe717058e28b23210b7e83350.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/d123865a1435c7e51f3b36e0e9e8ba1fd6fcef8fe717058e28b23210b7e83350.jpg)

![d302a991d4fc731d4bba8e95e2c03b8b6f10e934ed8fa0b58e58a0dfdcb61697.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/d302a991d4fc731d4bba8e95e2c03b8b6f10e934ed8fa0b58e58a0dfdcb61697.jpg)

![ecf9cd0077868946037bd2910fa789dd3688743a545311802fa801486b5e52f8.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/ecf9cd0077868946037bd2910fa789dd3688743a545311802fa801486b5e52f8.jpg)

![f19429efd8219c26ad819edeefb9fc7249eaf4e368529a0d298d23d51c713920.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/f19429efd8219c26ad819edeefb9fc7249eaf4e368529a0d298d23d51c713920.jpg)

![f1ac5f6802f449634ebac6052c34862359a008cfcb5129f8a4dab2cd4dc8af36.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/f1ac5f6802f449634ebac6052c34862359a008cfcb5129f8a4dab2cd4dc8af36.jpg)

![f6d6d39498933716a0c1c2ad9867e6cbae42d20a3a24b83fd7053fc3fcd365b5.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/f6d6d39498933716a0c1c2ad9867e6cbae42d20a3a24b83fd7053fc3fcd365b5.jpg)

![fac420fd2ee119c63d641ccb9856a84b3f954b6f15d746303dc18f9c9dc72f54.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/fac420fd2ee119c63d641ccb9856a84b3f954b6f15d746303dc18f9c9dc72f54.jpg)

![fd67eec7efc728862b08c1e3a9997cf192e86909d40a19951cfdf28b864cb8d4.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/fd67eec7efc728862b08c1e3a9997cf192e86909d40a19951cfdf28b864cb8d4.jpg)

![fecf9d56c4711a505f46e0ed3dd213e8f9c9b91c0dd7ef63e14e581bfc4bb558.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/images/fecf9d56c4711a505f46e0ed3dd213e8f9c9b91c0dd7ef63e14e581bfc4bb558.jpg)

### Tables

![4d4898203d24f637873124c3673171026d734b75deb60675f86de17a4f0ff9e6.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/tables/4d4898203d24f637873124c3673171026d734b75deb60675f86de17a4f0ff9e6.jpg)

![5ebdc2f674c5c41496d97c69d9ba5999302e4a8e69971ca22470a71105aaba05.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/tables/5ebdc2f674c5c41496d97c69d9ba5999302e4a8e69971ca22470a71105aaba05.jpg)

![78cddc42da886838e5891d1401b8240db56f13cfed680cf34c637a475e7af1ef.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/tables/78cddc42da886838e5891d1401b8240db56f13cfed680cf34c637a475e7af1ef.jpg)

![c320d22ecb64b573c9aaa929d2fbf7569b965a306617c3f45a4eff56585db955.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/tables/c320d22ecb64b573c9aaa929d2fbf7569b965a306617c3f45a4eff56585db955.jpg)

![eca7257c52c088b3f90da124a40ade3b5766fd12c7be63cf9aa3fb265b5da8c7.jpg](../icml_results/205_Optimal%20and%20Scalable%20MAPF%20via%20Multi-Marginal%20Optimal%20Transport%20and%20Schr%C3%B6dinger%20Bridges/tables/eca7257c52c088b3f90da124a40ade3b5766fd12c7be63cf9aa3fb265b5da8c7.jpg)

<a id="paper-0203"></a>

## Linguistic Nepotism: Trading-off Quality for Language Preference in Multilingual RAG


### Images

![0346f2e83925534a16fdb68531590f623a871d057b1212141d6397b6d3106b68.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/0346f2e83925534a16fdb68531590f623a871d057b1212141d6397b6d3106b68.jpg)

![1647faf104ddb097f783292b7e23c59a04a293fa5df65555f70301b0896063b6.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/1647faf104ddb097f783292b7e23c59a04a293fa5df65555f70301b0896063b6.jpg)

![1bde844f9a60885541ced42cee702d8f417245b040fbece55b0f248de41cde39.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/1bde844f9a60885541ced42cee702d8f417245b040fbece55b0f248de41cde39.jpg)

![1f00507fb9178c2b1e083614d735d5601892fd0ec17d5b76fa782f410e8e77e7.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/1f00507fb9178c2b1e083614d735d5601892fd0ec17d5b76fa782f410e8e77e7.jpg)

![303f1f82052c1f19f723024f766c50ef34fec84da141661c76d1ae8b88b215e1.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/303f1f82052c1f19f723024f766c50ef34fec84da141661c76d1ae8b88b215e1.jpg)

![3209eaefc63cc947a4dee3fc4978fe9e2fd7a52350aca3e7d99aa0c908512599.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/3209eaefc63cc947a4dee3fc4978fe9e2fd7a52350aca3e7d99aa0c908512599.jpg)

![43240f1260bccc0f6a23b0e9e35da49a36a86da1f40c3264835f98d465906d3a.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/43240f1260bccc0f6a23b0e9e35da49a36a86da1f40c3264835f98d465906d3a.jpg)

![4633494ae045ac9d4ebc94e8367519003b77c038c5ec7d172712d9f77978e836.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/4633494ae045ac9d4ebc94e8367519003b77c038c5ec7d172712d9f77978e836.jpg)

![641bf83bc49f35d851d6ed738c45f8b09f81330542f88e38b9154af297ed6f1f.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/641bf83bc49f35d851d6ed738c45f8b09f81330542f88e38b9154af297ed6f1f.jpg)

![673d6d1dcb8fe1dd8fdd0cc3877681e1c17709f948120da13fb9b10518bf2f85.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/673d6d1dcb8fe1dd8fdd0cc3877681e1c17709f948120da13fb9b10518bf2f85.jpg)

![71c192fc643842102cb5b34fdf67e603480971a3e58d60e340514ffcfa73b318.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/71c192fc643842102cb5b34fdf67e603480971a3e58d60e340514ffcfa73b318.jpg)

![77f8db3243a860f187ba227f4242e87979dee68a1181568ae97cd4551a261a68.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/77f8db3243a860f187ba227f4242e87979dee68a1181568ae97cd4551a261a68.jpg)

![7fa5e0a1823d5e958aa47dacd5b69893f5314929fa86043bd2e4be879390fd37.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/7fa5e0a1823d5e958aa47dacd5b69893f5314929fa86043bd2e4be879390fd37.jpg)

![87ed71c5e90be3dfa18165b2278d05fe516aa2fd9a116ed3a1d36ad6c4f9ed15.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/87ed71c5e90be3dfa18165b2278d05fe516aa2fd9a116ed3a1d36ad6c4f9ed15.jpg)

![9f476ad9f1c6d3de61434f949076267739075a7f599d8d43770d6e392c736b87.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/9f476ad9f1c6d3de61434f949076267739075a7f599d8d43770d6e392c736b87.jpg)

![a4c3c5e06f662dfc17ebcfde674ff3365679f8e1a65688b5a7650941cd24eb9e.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/a4c3c5e06f662dfc17ebcfde674ff3365679f8e1a65688b5a7650941cd24eb9e.jpg)

![bbe7ccfa2e8e1ae3c95cb3651598a1d31bc138be55f27a14a46617e744fb6fe9.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/bbe7ccfa2e8e1ae3c95cb3651598a1d31bc138be55f27a14a46617e744fb6fe9.jpg)

![f787d3c4424f153119469ff4b1868cdc6a7f4799e6c9f7e2e3b98be21a7452d1.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/images/f787d3c4424f153119469ff4b1868cdc6a7f4799e6c9f7e2e3b98be21a7452d1.jpg)

### Tables

![0acebf7259ea2a91629b5ee0916c2d3f3f65bbfae3f968c5a984a22b1011fab1.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/0acebf7259ea2a91629b5ee0916c2d3f3f65bbfae3f968c5a984a22b1011fab1.jpg)

![1c487268b16d193a65ce053087800c0208419c61a8c9fb4c9b3cad93fcc5db7d.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/1c487268b16d193a65ce053087800c0208419c61a8c9fb4c9b3cad93fcc5db7d.jpg)

![1e315b09751052dd5ecd73419b9fc364fb3b981d5de5bf2c6f36de6b9bfcb7d5.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/1e315b09751052dd5ecd73419b9fc364fb3b981d5de5bf2c6f36de6b9bfcb7d5.jpg)

![2e598dec7b458d87283eccbd6484111037166217ba9b21bc1ea7a76b61906765.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/2e598dec7b458d87283eccbd6484111037166217ba9b21bc1ea7a76b61906765.jpg)

![38e23cafb83d638f6053a467e96491072c73513ee3c4977084b571145674fe26.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/38e23cafb83d638f6053a467e96491072c73513ee3c4977084b571145674fe26.jpg)

![43f191940ff9f54be3f2248aac9fd61d7a26b26d2068a62d1dd0d98ff01bc1b2.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/43f191940ff9f54be3f2248aac9fd61d7a26b26d2068a62d1dd0d98ff01bc1b2.jpg)

![45b7eb7a50490798628a10be66d00588218ec6eb5e82fb46cf6db589fd5e87ee.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/45b7eb7a50490798628a10be66d00588218ec6eb5e82fb46cf6db589fd5e87ee.jpg)

![57046a5c6262149050a048f1bb8e4f6cf70d27ea9b919402ed623f290353942d.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/57046a5c6262149050a048f1bb8e4f6cf70d27ea9b919402ed623f290353942d.jpg)

![59dac9c00b308cdebe3131882d3854a71708ef7d0dfaba5bcc8652ac8b6d5f74.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/59dac9c00b308cdebe3131882d3854a71708ef7d0dfaba5bcc8652ac8b6d5f74.jpg)

![634551a4dd3901afa99c102f9258d725df631a4fd124a658dabbbf0b5b4c1fd5.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/634551a4dd3901afa99c102f9258d725df631a4fd124a658dabbbf0b5b4c1fd5.jpg)

![76e04910d38ab6d2731601d2b5b25d076bc81b57b5ec97e852a2b3d8780cd0a7.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/76e04910d38ab6d2731601d2b5b25d076bc81b57b5ec97e852a2b3d8780cd0a7.jpg)

![8130aaf22ec496d9b7e5572dfbe1d924ce3fdf645b5c7f4f82ddad91b5804837.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/8130aaf22ec496d9b7e5572dfbe1d924ce3fdf645b5c7f4f82ddad91b5804837.jpg)

![8c2440efa36d09db21c47e641ac854fe561b638013413ded169b3ccee0acd686.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/8c2440efa36d09db21c47e641ac854fe561b638013413ded169b3ccee0acd686.jpg)

![9e9ba98115128e67b7721fcd1c764398c2f241775d80d557f04ebf01b3a53a6b.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/9e9ba98115128e67b7721fcd1c764398c2f241775d80d557f04ebf01b3a53a6b.jpg)

![a2feb29bc56f23240781816f6af895318ed96a30abc5151f0becc05498e32cbe.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/a2feb29bc56f23240781816f6af895318ed96a30abc5151f0becc05498e32cbe.jpg)

![a422fff79be7496625df1b66b6ddfcc50c83eff1a39f3032129c0962e0799bca.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/a422fff79be7496625df1b66b6ddfcc50c83eff1a39f3032129c0962e0799bca.jpg)

![b8fc0a01a93125672b98400a74ebf2cd95959ce9725a90b3a044958317eeeb0e.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/b8fc0a01a93125672b98400a74ebf2cd95959ce9725a90b3a044958317eeeb0e.jpg)

![c847ca36f07a649234297ef65bed7edd1f26155d822b6851f648f41560a7ca28.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/c847ca36f07a649234297ef65bed7edd1f26155d822b6851f648f41560a7ca28.jpg)

![d444df2bbfe09ca645480e56880d853a5e6b1efc761d3adf32bfb42bc56f16a7.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/d444df2bbfe09ca645480e56880d853a5e6b1efc761d3adf32bfb42bc56f16a7.jpg)

![d59fbc9e9f9896329911b2872b430f8f0a98c7a6021ccd064e7b414534746e76.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/d59fbc9e9f9896329911b2872b430f8f0a98c7a6021ccd064e7b414534746e76.jpg)

![db488c15d278918582b3daa940d38cb16552d4e0f9a52f189d175412ffacc2a0.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/db488c15d278918582b3daa940d38cb16552d4e0f9a52f189d175412ffacc2a0.jpg)

![ed043a52aff5f6a07fd0f23bf8797b91f9555e69071d71be3c84e46c1fa390db.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/ed043a52aff5f6a07fd0f23bf8797b91f9555e69071d71be3c84e46c1fa390db.jpg)

![f3dd421efed4da440069aeb30a7ad44269dbfeed047db6eb21dcd5cc54556f94.jpg](../icml_results/206_Linguistic%20Nepotism_%20Trading-off%20Quality%20for%20Language%20Preference%20in%20Multilingual%20RAG/tables/f3dd421efed4da440069aeb30a7ad44269dbfeed047db6eb21dcd5cc54556f94.jpg)

<a id="paper-0204"></a>

## A Regret Minimization Framework on Preference Learning in Large Language Models


### Images

![159b46c67ab50787f44b2d690148bdb7079c9d643cbf48f6c01c606d05238ca6.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/images/159b46c67ab50787f44b2d690148bdb7079c9d643cbf48f6c01c606d05238ca6.jpg)

![16e8b11522c5f6203c1b98738a0e814809cd61c12c267e865176913015a8dcfa.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/images/16e8b11522c5f6203c1b98738a0e814809cd61c12c267e865176913015a8dcfa.jpg)

![384a66c4565cae824be6c045f1c9e081e18053eca4418dd21c2b6ea37f990901.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/images/384a66c4565cae824be6c045f1c9e081e18053eca4418dd21c2b6ea37f990901.jpg)

![3e42bdb891a24145e6ccc7925cd1899e836f4805ffd19caaf681f83b0cdf0eb9.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/images/3e42bdb891a24145e6ccc7925cd1899e836f4805ffd19caaf681f83b0cdf0eb9.jpg)

![68eab8ae037bb7e663c346952f8b3ac082314f7619853eb794284247c2857b52.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/images/68eab8ae037bb7e663c346952f8b3ac082314f7619853eb794284247c2857b52.jpg)

![6f7a811e29917843898fbcdc224e11b28f65a241ca88ca4355ccf1bdde335f8b.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/images/6f7a811e29917843898fbcdc224e11b28f65a241ca88ca4355ccf1bdde335f8b.jpg)

![95c67f75dcf67037fd3bab029a2faaffbad9bf53080ac06e2c43d58e335f85b6.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/images/95c67f75dcf67037fd3bab029a2faaffbad9bf53080ac06e2c43d58e335f85b6.jpg)

![9c1365437158add5035aed2446612f1493ff64d9925d265e6a876ccbbcc1a539.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/images/9c1365437158add5035aed2446612f1493ff64d9925d265e6a876ccbbcc1a539.jpg)

![9ee9dcf1e45d813dcd3b1b64b5a5a9ff2c35789d073e8df77bd30736561f24a3.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/images/9ee9dcf1e45d813dcd3b1b64b5a5a9ff2c35789d073e8df77bd30736561f24a3.jpg)

![9fa84064d6a012a07dcc630fdc4fa54d890fbe33f76bae4c332ee6390c18d48a.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/images/9fa84064d6a012a07dcc630fdc4fa54d890fbe33f76bae4c332ee6390c18d48a.jpg)

![c80d5aeb35d4b292c6ba815f09370ccd2174d888a3d41c093475c386af1afabf.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/images/c80d5aeb35d4b292c6ba815f09370ccd2174d888a3d41c093475c386af1afabf.jpg)

### Tables

![049232ad360ddd21bb48a43fe00134e54a0088211100ba75eeb8f9e1daefe06a.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/tables/049232ad360ddd21bb48a43fe00134e54a0088211100ba75eeb8f9e1daefe06a.jpg)

![370ae306ab9631102c90f6edd9fdcf3276b6ab813c8ff01841a88b9e771cb4c2.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/tables/370ae306ab9631102c90f6edd9fdcf3276b6ab813c8ff01841a88b9e771cb4c2.jpg)

![389daa290b3dea0028aab85e324bf41b3368c7a84b503897c5623dde89e5a28a.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/tables/389daa290b3dea0028aab85e324bf41b3368c7a84b503897c5623dde89e5a28a.jpg)

![40d113131af1ef13f4b865c755eb4c5d7bf866a718ccebccc168adb8bd72072f.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/tables/40d113131af1ef13f4b865c755eb4c5d7bf866a718ccebccc168adb8bd72072f.jpg)

![5840ba60c35bef8ab6a0bfa0b32112ee45ac284b94d1d9bcf32d55459d60f80f.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/tables/5840ba60c35bef8ab6a0bfa0b32112ee45ac284b94d1d9bcf32d55459d60f80f.jpg)

![66ec82b50373b47c2e390540172ec44b41fe3712a391970d9672dbb39d305eb0.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/tables/66ec82b50373b47c2e390540172ec44b41fe3712a391970d9672dbb39d305eb0.jpg)

![6a2581ce6d2b1c558c00c891a9dc10340ed691976df51f10ee71488d653e8a3b.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/tables/6a2581ce6d2b1c558c00c891a9dc10340ed691976df51f10ee71488d653e8a3b.jpg)

![9a32882a4913b84d663e2182745a14f19f56741e999049bc31e2cbf060ec6078.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/tables/9a32882a4913b84d663e2182745a14f19f56741e999049bc31e2cbf060ec6078.jpg)

![acbd662919b489e5b7f3dbd17dacd4140b89253152b5112256fef60984a2c830.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/tables/acbd662919b489e5b7f3dbd17dacd4140b89253152b5112256fef60984a2c830.jpg)

![b754679dd82466364b91d1b53d5f376729c3db022ee0aed69bef252e3df63b34.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/tables/b754679dd82466364b91d1b53d5f376729c3db022ee0aed69bef252e3df63b34.jpg)

![d399f72f54e1499e164f68cc87b3f2bf05c896e13eff818d9965a527ac20a113.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/tables/d399f72f54e1499e164f68cc87b3f2bf05c896e13eff818d9965a527ac20a113.jpg)

![eea1ca358c4f0382b49ac0e9d72215eb2aa2a1f2888992696701fa4e1cef4e56.jpg](../icml_results/207_A%20Regret%20Minimization%20Framework%20on%20Preference%20Learning%20in%20Large%20Language%20Models/tables/eea1ca358c4f0382b49ac0e9d72215eb2aa2a1f2888992696701fa4e1cef4e56.jpg)

<a id="paper-0205"></a>

## On the Sharp Input-Output Analysis of Nonlinear Systems under Adversarial Attacks


### Images

![419d7f82ac80567a3bfb8d4ea03a4daeb61cccaf76df02d4a5a5916490307ea1.jpg](../icml_results/208_On%20the%20Sharp%20Input-Output%20Analysis%20of%20Nonlinear%20Systems%20under%20Adversarial%20Attacks/images/419d7f82ac80567a3bfb8d4ea03a4daeb61cccaf76df02d4a5a5916490307ea1.jpg)

![44e99e9a86b85a2767a078ab9e261d20e29210f7bd1b559e7c1947ed5be0bbe8.jpg](../icml_results/208_On%20the%20Sharp%20Input-Output%20Analysis%20of%20Nonlinear%20Systems%20under%20Adversarial%20Attacks/images/44e99e9a86b85a2767a078ab9e261d20e29210f7bd1b559e7c1947ed5be0bbe8.jpg)

![5a202fd477bdbc84e0aec97e1882ec8fbe74ca8b77035b9c250372ea717b2c04.jpg](../icml_results/208_On%20the%20Sharp%20Input-Output%20Analysis%20of%20Nonlinear%20Systems%20under%20Adversarial%20Attacks/images/5a202fd477bdbc84e0aec97e1882ec8fbe74ca8b77035b9c250372ea717b2c04.jpg)

![920f9ce3005f5d0d850a270238b1fb8828bd47801e57108a7ce1167c5add38a0.jpg](../icml_results/208_On%20the%20Sharp%20Input-Output%20Analysis%20of%20Nonlinear%20Systems%20under%20Adversarial%20Attacks/images/920f9ce3005f5d0d850a270238b1fb8828bd47801e57108a7ce1167c5add38a0.jpg)

![a4b35260ee50795c496ffce0cbc8c51f19022aa4715976f2b8a0ac977ce9734c.jpg](../icml_results/208_On%20the%20Sharp%20Input-Output%20Analysis%20of%20Nonlinear%20Systems%20under%20Adversarial%20Attacks/images/a4b35260ee50795c496ffce0cbc8c51f19022aa4715976f2b8a0ac977ce9734c.jpg)

![af28eec2979459cc9a3e21704bb6246f83c9f9a5b54c79b2bb1a786fa882b096.jpg](../icml_results/208_On%20the%20Sharp%20Input-Output%20Analysis%20of%20Nonlinear%20Systems%20under%20Adversarial%20Attacks/images/af28eec2979459cc9a3e21704bb6246f83c9f9a5b54c79b2bb1a786fa882b096.jpg)

![f998260ff5875982e0ab9c9da807ecec670354152b30b0523cd6dd358c5e264d.jpg](../icml_results/208_On%20the%20Sharp%20Input-Output%20Analysis%20of%20Nonlinear%20Systems%20under%20Adversarial%20Attacks/images/f998260ff5875982e0ab9c9da807ecec670354152b30b0523cd6dd358c5e264d.jpg)

![f9d33868f0c48ca1243bfb4534ac484466cea6d78288b5d7fdc9be5c0ef48175.jpg](../icml_results/208_On%20the%20Sharp%20Input-Output%20Analysis%20of%20Nonlinear%20Systems%20under%20Adversarial%20Attacks/images/f9d33868f0c48ca1243bfb4534ac484466cea6d78288b5d7fdc9be5c0ef48175.jpg)

### Tables

![659d1118514d45bae721107b9ee2d8c6337856495dd8df0485f4ccd7b5d9d926.jpg](../icml_results/208_On%20the%20Sharp%20Input-Output%20Analysis%20of%20Nonlinear%20Systems%20under%20Adversarial%20Attacks/tables/659d1118514d45bae721107b9ee2d8c6337856495dd8df0485f4ccd7b5d9d926.jpg)

![7de746fb439f47bc5295966b1b081331bcdadc6bf926edb51ef9a570d283c77e.jpg](../icml_results/208_On%20the%20Sharp%20Input-Output%20Analysis%20of%20Nonlinear%20Systems%20under%20Adversarial%20Attacks/tables/7de746fb439f47bc5295966b1b081331bcdadc6bf926edb51ef9a570d283c77e.jpg)

<a id="paper-0206"></a>

## VALUEFLOW: Toward Pluralistic and Steerable Value-based Alignment in Large Language Models


### Images

![12480a1cad94a65e5e280aa33a7c23ffaf0adc8eafa7d862f1233eb311921137.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/12480a1cad94a65e5e280aa33a7c23ffaf0adc8eafa7d862f1233eb311921137.jpg)

![1a58d41962458f4c18c6d90d7c8ccb2fa9a448e61fe572f74da1b6b24473ce4c.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/1a58d41962458f4c18c6d90d7c8ccb2fa9a448e61fe572f74da1b6b24473ce4c.jpg)

![24e6ba789ef28aab126627f8f2819b8e38046f95b8f3f4c4ea3a580a292aff0a.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/24e6ba789ef28aab126627f8f2819b8e38046f95b8f3f4c4ea3a580a292aff0a.jpg)

![3215f6bef644954a58fc743fd48c787ace33411ceb1d62589e1165e773240eb0.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/3215f6bef644954a58fc743fd48c787ace33411ceb1d62589e1165e773240eb0.jpg)

![327582d80b41190c7e694d78101959421a377dca5b6c399a2b8067d6b30745dd.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/327582d80b41190c7e694d78101959421a377dca5b6c399a2b8067d6b30745dd.jpg)

![32cb28183c0bd98a01fcaf5a124e15efdba3328b4618005476f8ec8a663c85bb.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/32cb28183c0bd98a01fcaf5a124e15efdba3328b4618005476f8ec8a663c85bb.jpg)

![370d3466a5926223f6afea93f0074676948a2f7f3cb3e4171810dc5c9ec601ec.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/370d3466a5926223f6afea93f0074676948a2f7f3cb3e4171810dc5c9ec601ec.jpg)

![3b9e2b0e2bcfcecd3a5e3340a2d982fd9a3c58769a9c7ba8e9867b306d6ec7ca.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/3b9e2b0e2bcfcecd3a5e3340a2d982fd9a3c58769a9c7ba8e9867b306d6ec7ca.jpg)

![3c734cae7821a8867bd61075c64e5ba09a631f9a1f239c960d2a356169ca579a.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/3c734cae7821a8867bd61075c64e5ba09a631f9a1f239c960d2a356169ca579a.jpg)

![3e83ceb720be389b0e2a2c9d17b2a6cd754b5e6f883586faa1627c820f4eb6ad.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/3e83ceb720be389b0e2a2c9d17b2a6cd754b5e6f883586faa1627c820f4eb6ad.jpg)

![3f6e779bb124cbb297a2a69216fac79522c95fba4467c4cba6603b1f2ca738a5.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/3f6e779bb124cbb297a2a69216fac79522c95fba4467c4cba6603b1f2ca738a5.jpg)

![3ff5b5c6224f28f489adcb8d065842328a05dea6faea2de9876140800d907019.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/3ff5b5c6224f28f489adcb8d065842328a05dea6faea2de9876140800d907019.jpg)

![44adc778a223076c3bb610ef1e763ad7a65f6b99e52472c73d6279c5c98e3f43.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/44adc778a223076c3bb610ef1e763ad7a65f6b99e52472c73d6279c5c98e3f43.jpg)

![4782e13c3d9a5d97b0986e58084bf4df5591b11517349c8b93d5e2d2ed170c87.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/4782e13c3d9a5d97b0986e58084bf4df5591b11517349c8b93d5e2d2ed170c87.jpg)

![4e5a156a14bb3e6379530fe3eab579965d4fb850406595a9bb351c33f031bfe6.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/4e5a156a14bb3e6379530fe3eab579965d4fb850406595a9bb351c33f031bfe6.jpg)

![4f014eefa05a194a1a203bc42531c9ad9703d9216ffb93e9cb384e07efdf1e9a.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/4f014eefa05a194a1a203bc42531c9ad9703d9216ffb93e9cb384e07efdf1e9a.jpg)

![58f6abc7998f783a2c3a86075e6f80d004de4f47bd9d1dde97c4a51acba656d7.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/58f6abc7998f783a2c3a86075e6f80d004de4f47bd9d1dde97c4a51acba656d7.jpg)

![5ab010c1bd7bcff387ac915012b306c1781c4e4d413206f70e92804b1d2ac886.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/5ab010c1bd7bcff387ac915012b306c1781c4e4d413206f70e92804b1d2ac886.jpg)

![5c074c9390bae2ee7ffdaa177bf324f7c2b4d4a5be7e09f91bc39b24115ee26e.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/5c074c9390bae2ee7ffdaa177bf324f7c2b4d4a5be7e09f91bc39b24115ee26e.jpg)

![678556329b07cc817ee3d95c6e51a5af8eb4650da2f0d9106216b2c2c98534ac.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/678556329b07cc817ee3d95c6e51a5af8eb4650da2f0d9106216b2c2c98534ac.jpg)

![6877a1e282fe20b2a9a225481c6ed7fa36b1c7cbc746eac77022576022873463.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/6877a1e282fe20b2a9a225481c6ed7fa36b1c7cbc746eac77022576022873463.jpg)

![690e32d463a36a5c98b71117f3419f9acd7893f1ded652f465acec53c650373c.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/690e32d463a36a5c98b71117f3419f9acd7893f1ded652f465acec53c650373c.jpg)

![6b64c0f3e1bd67ac304f3778865d5beeb8b84183cb46a9e9cc2ec5f9fab6f1a4.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/6b64c0f3e1bd67ac304f3778865d5beeb8b84183cb46a9e9cc2ec5f9fab6f1a4.jpg)

![806c0542fcfc45cdb2056fcaf12cd6d2a552cbfd1d4dbebe1517f4e1ea76eccc.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/806c0542fcfc45cdb2056fcaf12cd6d2a552cbfd1d4dbebe1517f4e1ea76eccc.jpg)

![81c83f90d8fcc74a39566f65298d4597a9ae4a7061cf42d2b50674471b937ba3.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/81c83f90d8fcc74a39566f65298d4597a9ae4a7061cf42d2b50674471b937ba3.jpg)

![863feada844ea8ef11d50a62a2cb86e226709bd16f08fd9e659d9130a467c3b2.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/863feada844ea8ef11d50a62a2cb86e226709bd16f08fd9e659d9130a467c3b2.jpg)

![89d6c5528b7e0ec9ddbf4cbefaa672c11a572a5d9011b969aaf00bfe046510e5.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/89d6c5528b7e0ec9ddbf4cbefaa672c11a572a5d9011b969aaf00bfe046510e5.jpg)

![919e7645ebd8ef9fddfc2a4527ebf2e868f56e8978315c7152d7ae3e2c37d960.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/919e7645ebd8ef9fddfc2a4527ebf2e868f56e8978315c7152d7ae3e2c37d960.jpg)

![92146d6af6211e260500a6dedaafc757b2dbeaffdceb91bf41751fb029de46eb.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/92146d6af6211e260500a6dedaafc757b2dbeaffdceb91bf41751fb029de46eb.jpg)

![a43d329edd11b56fc6eae61c5ec757539c7ff4434b8fcc829cd0c32f6a929777.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/a43d329edd11b56fc6eae61c5ec757539c7ff4434b8fcc829cd0c32f6a929777.jpg)

![a4d7c39e81a666df1feb309ba33febfa9ee57fe5baa2757cef6f528d92b45249.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/a4d7c39e81a666df1feb309ba33febfa9ee57fe5baa2757cef6f528d92b45249.jpg)

![b2c34b6780bf4fabd5dc9046261c01d7606d29a63370be1171348b987f948189.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/b2c34b6780bf4fabd5dc9046261c01d7606d29a63370be1171348b987f948189.jpg)

![b6c5321f9c93e44a3e93bd12de96968b2f2d7490a9b9e990f3e04582caca584f.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/b6c5321f9c93e44a3e93bd12de96968b2f2d7490a9b9e990f3e04582caca584f.jpg)

![b9011c5e454143c34f662124eb484929067d3ed833e0a3f8a0ccad4933076214.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/b9011c5e454143c34f662124eb484929067d3ed833e0a3f8a0ccad4933076214.jpg)

![bfa603373d7940c56af6464f75f1bc239e91ae1c91dd8e98b8286c9693b83167.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/bfa603373d7940c56af6464f75f1bc239e91ae1c91dd8e98b8286c9693b83167.jpg)

![c0d42dd0ecb831cd2bab3bb7498c564a4156ca65359ab5917602003671fb53e9.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/c0d42dd0ecb831cd2bab3bb7498c564a4156ca65359ab5917602003671fb53e9.jpg)

![c362376c98d375d03c1e70eaff8b13c04dfb7ebd40e5b28fb4929c64c170ea78.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/c362376c98d375d03c1e70eaff8b13c04dfb7ebd40e5b28fb4929c64c170ea78.jpg)

![cabc5bd9f3f8784a27e2747f72841778bd252c0442b9fdf82304a6093c7bba1c.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/cabc5bd9f3f8784a27e2747f72841778bd252c0442b9fdf82304a6093c7bba1c.jpg)

![cafc51310a4fb5554fbd84b11ebacc4660ca775b1547c197d63d5ca4917eba0d.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/cafc51310a4fb5554fbd84b11ebacc4660ca775b1547c197d63d5ca4917eba0d.jpg)

![cc7cc0d43b9ec8b240366a31c9e2a6326adfed062d6e48abba2213ffcec9b878.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/cc7cc0d43b9ec8b240366a31c9e2a6326adfed062d6e48abba2213ffcec9b878.jpg)

![cc96f354242dd9272368338ee3e6d9a5954fa3c790e6f6f0d6cb54ad5b743b34.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/cc96f354242dd9272368338ee3e6d9a5954fa3c790e6f6f0d6cb54ad5b743b34.jpg)

![d4672dce41587465b5f8ba8ba522c741e23635e850e5ecafedb095fe6322e07c.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/d4672dce41587465b5f8ba8ba522c741e23635e850e5ecafedb095fe6322e07c.jpg)

![e0399da0c9a05db191c1c0052ea9aed99b9fdedffc51452246463bf0ded227df.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/e0399da0c9a05db191c1c0052ea9aed99b9fdedffc51452246463bf0ded227df.jpg)

![e631b72bdfbd517e01f2e3aeea83582d223a2bbadd2d44ca877e30d65b41a688.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/e631b72bdfbd517e01f2e3aeea83582d223a2bbadd2d44ca877e30d65b41a688.jpg)

![fd3d5655c237b6953cde69f8fcf2e2d679a311037a01ab0bbcbe48221a07c727.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/images/fd3d5655c237b6953cde69f8fcf2e2d679a311037a01ab0bbcbe48221a07c727.jpg)

### Tables

![0a3ddf5e679e14eb2d184c52adadbd9361ef2ae5fc0792d2dc92bae41b024788.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/tables/0a3ddf5e679e14eb2d184c52adadbd9361ef2ae5fc0792d2dc92bae41b024788.jpg)

![2a0e2e29a542ffe77e7513a47dc63f3d355fb5ee8e9787e7ab897c59739fab56.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/tables/2a0e2e29a542ffe77e7513a47dc63f3d355fb5ee8e9787e7ab897c59739fab56.jpg)

![2d2e42bc81af1b783a4604e513eb5a536544a68e907ba099790ec8140c1581bf.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/tables/2d2e42bc81af1b783a4604e513eb5a536544a68e907ba099790ec8140c1581bf.jpg)

![2e5031057036ec505db8fc8709fc12f4b18171a90be52f5871848ce54cde0bef.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/tables/2e5031057036ec505db8fc8709fc12f4b18171a90be52f5871848ce54cde0bef.jpg)

![42310fe03bb764b5b7b0f4fca750b45aeff5438508a2ab4e8c74d6db2fac12b4.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/tables/42310fe03bb764b5b7b0f4fca750b45aeff5438508a2ab4e8c74d6db2fac12b4.jpg)

![46c725c7f18ab905ba0965934766d7c38898cefd5b1a13e082b10e739cfa9dc4.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/tables/46c725c7f18ab905ba0965934766d7c38898cefd5b1a13e082b10e739cfa9dc4.jpg)

![53722332a5cf2a69ccc64de5a7f79da4fb784fe0fa86cefffa3550e735e52bec.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/tables/53722332a5cf2a69ccc64de5a7f79da4fb784fe0fa86cefffa3550e735e52bec.jpg)

![59691ab63bcbce632ae0cdefaf003b1201b5fe8dc187b0f29e086d9bc8178ce4.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/tables/59691ab63bcbce632ae0cdefaf003b1201b5fe8dc187b0f29e086d9bc8178ce4.jpg)

![692ce6244871a2b79c0a3625893c31beb62982b63df03ce254fa0a4ed7a039da.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/tables/692ce6244871a2b79c0a3625893c31beb62982b63df03ce254fa0a4ed7a039da.jpg)

![6ffa2b5c8ef882b66ac23b50a0db8bbcb11865327fa02c326c5ed6b60e93f269.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/tables/6ffa2b5c8ef882b66ac23b50a0db8bbcb11865327fa02c326c5ed6b60e93f269.jpg)

![716d9e1fde1c6febb19408142278de7962dc9cc6970ef17d5d954df22c47e802.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/tables/716d9e1fde1c6febb19408142278de7962dc9cc6970ef17d5d954df22c47e802.jpg)

![7fe224fa8800cf23b9ee148d9c24e02763d083d96da84df3dcae027c6b812b5d.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/tables/7fe224fa8800cf23b9ee148d9c24e02763d083d96da84df3dcae027c6b812b5d.jpg)

![8e8785dab90d6c7caab4e5b19c6f81a4db36bb7ae95cade659a545856cb8426e.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/tables/8e8785dab90d6c7caab4e5b19c6f81a4db36bb7ae95cade659a545856cb8426e.jpg)

![b04b139440b34f9fd6d7fa704076ac25c7d75624b6f3033cc133f4bf155d5850.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/tables/b04b139440b34f9fd6d7fa704076ac25c7d75624b6f3033cc133f4bf155d5850.jpg)

![befd7d064149621a6e4071c98d68ec5a99a569e0ab83a4adf915941c7788e36e.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/tables/befd7d064149621a6e4071c98d68ec5a99a569e0ab83a4adf915941c7788e36e.jpg)

![f2ba92727cb6a3e58ca314cad299d7027f2361e8258963a38bf8825c7a3004ad.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/tables/f2ba92727cb6a3e58ca314cad299d7027f2361e8258963a38bf8825c7a3004ad.jpg)

![fa4beee29f1a1fbaf7470e14bf822b16580f2f0e6fb1ffaba33d03532b3d32a2.jpg](../icml_results/209_VALUEFLOW_%20Toward%20Pluralistic%20and%20Steerable%20Value-based%20Alignment%20in%20Large%20Language%20Models/tables/fa4beee29f1a1fbaf7470e14bf822b16580f2f0e6fb1ffaba33d03532b3d32a2.jpg)

<a id="paper-0207"></a>

## Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models


### Images

![04edb5436e6046b7c58fb85408152c49cc55a8615a1b9bcdbc6d061c55171e8a.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/images/04edb5436e6046b7c58fb85408152c49cc55a8615a1b9bcdbc6d061c55171e8a.jpg)

![1717f326e22e085680876d1603cc42674e6721abda7062e23450025ec9d39e2b.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/images/1717f326e22e085680876d1603cc42674e6721abda7062e23450025ec9d39e2b.jpg)

![49d5fb900729c18470b748b752ec63975a57b017395593c8a1f92d0c60414790.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/images/49d5fb900729c18470b748b752ec63975a57b017395593c8a1f92d0c60414790.jpg)

![82723af6a4daa1d427b89f9b86fa6f6b56f9c524a34c428a9925c7e58adbbee0.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/images/82723af6a4daa1d427b89f9b86fa6f6b56f9c524a34c428a9925c7e58adbbee0.jpg)

![91a450bc660c5b3f2a351758a7e4e7d32ff8bc8fa861df75a9ec729ba0048910.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/images/91a450bc660c5b3f2a351758a7e4e7d32ff8bc8fa861df75a9ec729ba0048910.jpg)

![992fb1fd8bc74629d2014dc9a6e4306d5f7d87f49326b0fd879b18f2c957769b.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/images/992fb1fd8bc74629d2014dc9a6e4306d5f7d87f49326b0fd879b18f2c957769b.jpg)

![9d7b28dabe58f7e3489599d7c49ebc2e144c1ba10f960333594b1eb4dcb8ef34.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/images/9d7b28dabe58f7e3489599d7c49ebc2e144c1ba10f960333594b1eb4dcb8ef34.jpg)

![aa49a9d86fb055384daf918a59cc893acc5aaba8a16dea1ca87f5deb3da426d9.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/images/aa49a9d86fb055384daf918a59cc893acc5aaba8a16dea1ca87f5deb3da426d9.jpg)

![bc7f66459c59a857548fed56faeecde2a8e6d1113fe864670cc56eb45b9033cc.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/images/bc7f66459c59a857548fed56faeecde2a8e6d1113fe864670cc56eb45b9033cc.jpg)

![bda4324f383e6f5a9b07c7a46d5554ce11452fefc1b18a61e20ee58a35b46341.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/images/bda4324f383e6f5a9b07c7a46d5554ce11452fefc1b18a61e20ee58a35b46341.jpg)

![be79a14190c993557918137ed6a30b2ae0eb9da89dccb2fc0421db057d5075c5.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/images/be79a14190c993557918137ed6a30b2ae0eb9da89dccb2fc0421db057d5075c5.jpg)

![eafd4c18a351d50a26c7b24dcbbe05e7b8c04580732060d43dc996a64e1c03cf.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/images/eafd4c18a351d50a26c7b24dcbbe05e7b8c04580732060d43dc996a64e1c03cf.jpg)

![f16e2de59118d7c52748dcc88e17ed3b6460ec2287aa07cde353a01193a9315c.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/images/f16e2de59118d7c52748dcc88e17ed3b6460ec2287aa07cde353a01193a9315c.jpg)

![f3b5421b089c6014d74b5d70362cf6fb40c8ad47bce4517c17d75cb4a8ccc371.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/images/f3b5421b089c6014d74b5d70362cf6fb40c8ad47bce4517c17d75cb4a8ccc371.jpg)

![f66be087db6cde162ac6bd3e3d873447df99135347faf4aace523c58d98e7940.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/images/f66be087db6cde162ac6bd3e3d873447df99135347faf4aace523c58d98e7940.jpg)

![f97e9ac1e12d6dec0e6729a6dd456cbfe7423a896e17af29ceb0bdce6b73ee91.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/images/f97e9ac1e12d6dec0e6729a6dd456cbfe7423a896e17af29ceb0bdce6b73ee91.jpg)

### Tables

![0358802f6af828f093d251a7d544b341676753ce4535c5241b2d0e30455eaf5b.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/tables/0358802f6af828f093d251a7d544b341676753ce4535c5241b2d0e30455eaf5b.jpg)

![251add401fe36e6bf72b8e2ee92549d691301843e9cfc0cc032946a600eae131.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/tables/251add401fe36e6bf72b8e2ee92549d691301843e9cfc0cc032946a600eae131.jpg)

![775867ded4f613422710711589624b936c297199043cccc4557cf12a39acdb7f.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/tables/775867ded4f613422710711589624b936c297199043cccc4557cf12a39acdb7f.jpg)

![89b8472948e52bc97a9af66c6b6e275340ec9d386455a27b08d50463c0ee6355.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/tables/89b8472948e52bc97a9af66c6b6e275340ec9d386455a27b08d50463c0ee6355.jpg)

![9950f6d6cb7ebb8608ea4b477d109282554162fe6f93ef9b3fe833b110ce4d07.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/tables/9950f6d6cb7ebb8608ea4b477d109282554162fe6f93ef9b3fe833b110ce4d07.jpg)

![9bde931b9fbdc4056bd0edf31f649bf8db31cec11c76e0f7545bab6237925748.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/tables/9bde931b9fbdc4056bd0edf31f649bf8db31cec11c76e0f7545bab6237925748.jpg)

![b1e24e71568622dccbadaa7e032f39806d425008734c4ec0cf288807ec574686.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/tables/b1e24e71568622dccbadaa7e032f39806d425008734c4ec0cf288807ec574686.jpg)

![bbfed8aff3e387c3fbd8649cd68c07a82e5f9110a1630ef9a62e6fd6cf17d935.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/tables/bbfed8aff3e387c3fbd8649cd68c07a82e5f9110a1630ef9a62e6fd6cf17d935.jpg)

![d5bfc59a26f9474386b955acaa41fcf1c12ab7ff00e6a9a9575d68a698bc4585.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/tables/d5bfc59a26f9474386b955acaa41fcf1c12ab7ff00e6a9a9575d68a698bc4585.jpg)

![e623abf2db798155a75781da8caf88894074c2c75b49b5e6dcb9d75393b2671e.jpg](../icml_results/210_Lookahead%20Sample%20Reward%20Guidance%20for%20Test-Time%20Scaling%20of%20Diffusion%20Models/tables/e623abf2db798155a75781da8caf88894074c2c75b49b5e6dcb9d75393b2671e.jpg)

<a id="paper-0208"></a>

## When to Trust the Cheap Check: Weak and Strong Verification for Reasoning


### Images

![135a047036a766256cd209044d601de9d41efd7c2d00a2db9b5b11ce3f4122a8.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/135a047036a766256cd209044d601de9d41efd7c2d00a2db9b5b11ce3f4122a8.jpg)

![1a4f8fb0067a0b09ad459d43998f7a4f9516b65c66c32839757a2bcccd23b615.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/1a4f8fb0067a0b09ad459d43998f7a4f9516b65c66c32839757a2bcccd23b615.jpg)

![393f7c5844fa8c22bf32a52c2dce3f834f655cea5b698422db424e89ff507e0f.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/393f7c5844fa8c22bf32a52c2dce3f834f655cea5b698422db424e89ff507e0f.jpg)

![450faee083860b99acaa0bdf52024a0ae6caa5aa9c92085cb97e717915d1494e.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/450faee083860b99acaa0bdf52024a0ae6caa5aa9c92085cb97e717915d1494e.jpg)

![451ef07df8135bc90b980bbb0bc9e2b73bbfadf05e91fa8f35434b6e426081a1.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/451ef07df8135bc90b980bbb0bc9e2b73bbfadf05e91fa8f35434b6e426081a1.jpg)

![6418caf2cf20ce78922df7a9fdcf77df6b36f5162b97ff0f9ffbe37d6dd62971.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/6418caf2cf20ce78922df7a9fdcf77df6b36f5162b97ff0f9ffbe37d6dd62971.jpg)

![845781f8a9ae80bea7f8e325252d0ba4aa486c8e5c0e05bc117a664a98df19ab.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/845781f8a9ae80bea7f8e325252d0ba4aa486c8e5c0e05bc117a664a98df19ab.jpg)

![867cf2d36bd9e4b9386ce64a5b7bf599cf8349375273a277f7ce025415b7bfcc.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/867cf2d36bd9e4b9386ce64a5b7bf599cf8349375273a277f7ce025415b7bfcc.jpg)

![9bfd2e6cfa3dcd858fdd104f6141fc4253d52180ca1299a995f67992c87b6760.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/9bfd2e6cfa3dcd858fdd104f6141fc4253d52180ca1299a995f67992c87b6760.jpg)

![adbf2114af2bafdf2c2b1e28c4c04bd0b319801ae2e92956a88da91c8f06c880.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/adbf2114af2bafdf2c2b1e28c4c04bd0b319801ae2e92956a88da91c8f06c880.jpg)

![ae81f5c912d7c442647eeea5269ea3f7e71ee92f748893e6b75123d0d215b0ee.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/ae81f5c912d7c442647eeea5269ea3f7e71ee92f748893e6b75123d0d215b0ee.jpg)

![b97ab544d2aad1e41a2f04d6bc4020f6de75de106d1ba46e0ddcb68718a627af.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/b97ab544d2aad1e41a2f04d6bc4020f6de75de106d1ba46e0ddcb68718a627af.jpg)

![bebf8e53bd9235b362ead6d760bde159cf0f2f70c70ae1a2f7716dd470470c76.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/bebf8e53bd9235b362ead6d760bde159cf0f2f70c70ae1a2f7716dd470470c76.jpg)

![bf0493d97599414967b40d70f488c6c819517148429029763ed0f08854ccd22f.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/bf0493d97599414967b40d70f488c6c819517148429029763ed0f08854ccd22f.jpg)

![c6495a6cc8b0d1303abe89cffbddc99ef1b7045e8108e5638072590ca31fb036.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/c6495a6cc8b0d1303abe89cffbddc99ef1b7045e8108e5638072590ca31fb036.jpg)

![cc3a7a35c57e2e63898a4a70f2843a39cc5b5534ec293fb2383f92bcf1f50d67.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/cc3a7a35c57e2e63898a4a70f2843a39cc5b5534ec293fb2383f92bcf1f50d67.jpg)

![e16664c471e9cbdef242be918e270053487d2712231390c5278ea7f6c8daa9aa.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/e16664c471e9cbdef242be918e270053487d2712231390c5278ea7f6c8daa9aa.jpg)

![eeb23c1e6bf8deb28a2ed26d8b44407fe33fde6cc814dc1fa2d13b28c516fbb8.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/eeb23c1e6bf8deb28a2ed26d8b44407fe33fde6cc814dc1fa2d13b28c516fbb8.jpg)

![fa8291fbbfb3643d49600f0779e05d3a5ed4fa126d553fa1f5aedc3054d4b27f.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/images/fa8291fbbfb3643d49600f0779e05d3a5ed4fa126d553fa1f5aedc3054d4b27f.jpg)

### Tables

![4bdd17fd709f0f2ef39e8bc4262d9d8c6e7fa45f6ddc7a9f7d7ce8ac08f37399.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/tables/4bdd17fd709f0f2ef39e8bc4262d9d8c6e7fa45f6ddc7a9f7d7ce8ac08f37399.jpg)

![4ce12f96b93f31607947e0a018c665684d0a6f78b915a5609ecef79fd3192611.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/tables/4ce12f96b93f31607947e0a018c665684d0a6f78b915a5609ecef79fd3192611.jpg)

![5b3cfc651ffdcdc27892a2d73c3c741cbd8cf1a6ddf3229d5e96b8b4151e83d4.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/tables/5b3cfc651ffdcdc27892a2d73c3c741cbd8cf1a6ddf3229d5e96b8b4151e83d4.jpg)

![a1dcf190faef49cf4aded0cc59b0486104771b6abac6f54992cb2bf64878fb10.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/tables/a1dcf190faef49cf4aded0cc59b0486104771b6abac6f54992cb2bf64878fb10.jpg)

![a71a5dcca72796c4770849be145c2c0112ab4620fc06a1ceb6d3713ed16f363d.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/tables/a71a5dcca72796c4770849be145c2c0112ab4620fc06a1ceb6d3713ed16f363d.jpg)

![df2119bebe2b856f0415cfff9212cde1149038407d42e6239d45fb1f73b68c07.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/tables/df2119bebe2b856f0415cfff9212cde1149038407d42e6239d45fb1f73b68c07.jpg)

![f4cf38ae695a4066e6e138a8e979f38a5d60538f8b5e557fa947809c24dc443b.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/tables/f4cf38ae695a4066e6e138a8e979f38a5d60538f8b5e557fa947809c24dc443b.jpg)

![fb25904c80744840a7a4e9accd6cf7d523f6f9a6f73da5bf54eeefb6a070add6.jpg](../icml_results/211_When%20to%20Trust%20the%20Cheap%20Check_%20Weak%20and%20Strong%20Verification%20for%20Reasoning/tables/fb25904c80744840a7a4e9accd6cf7d523f6f9a6f73da5bf54eeefb6a070add6.jpg)

<a id="paper-0209"></a>

## Latent Spherical Flow Policy for Reinforcement Learning with Combinatorial Actions


### Images

![2fff535b28450d6cc3727dc41ee48d366191bc42b5749ede24d336fba514fbb6.jpg](../icml_results/212_Latent%20Spherical%20Flow%20Policy%20for%20Reinforcement%20Learning%20with%20Combinatorial%20Actions/images/2fff535b28450d6cc3727dc41ee48d366191bc42b5749ede24d336fba514fbb6.jpg)

![6c8de0eba7cf7428b8d307a3f8a58b2162022a1f1fa3ba7e94a32d8905ec13cd.jpg](../icml_results/212_Latent%20Spherical%20Flow%20Policy%20for%20Reinforcement%20Learning%20with%20Combinatorial%20Actions/images/6c8de0eba7cf7428b8d307a3f8a58b2162022a1f1fa3ba7e94a32d8905ec13cd.jpg)

![8c81ad2a95379f61b60b6e2f68fd5fc074bb2613c285d5ea8beaf872711ab2ac.jpg](../icml_results/212_Latent%20Spherical%20Flow%20Policy%20for%20Reinforcement%20Learning%20with%20Combinatorial%20Actions/images/8c81ad2a95379f61b60b6e2f68fd5fc074bb2613c285d5ea8beaf872711ab2ac.jpg)

![99651c8976d4fd3d79c5c9e0ab9a4872150ab4c233c3521bbd08d32a846d6615.jpg](../icml_results/212_Latent%20Spherical%20Flow%20Policy%20for%20Reinforcement%20Learning%20with%20Combinatorial%20Actions/images/99651c8976d4fd3d79c5c9e0ab9a4872150ab4c233c3521bbd08d32a846d6615.jpg)

![9fae7a87bdbcbba1687d223d4019561baa6c9e464c5dd0aeb2cb1f5e0449b9f8.jpg](../icml_results/212_Latent%20Spherical%20Flow%20Policy%20for%20Reinforcement%20Learning%20with%20Combinatorial%20Actions/images/9fae7a87bdbcbba1687d223d4019561baa6c9e464c5dd0aeb2cb1f5e0449b9f8.jpg)

![d7942a55a30647665f6500885cc8117737954872cb2cc5b5e6d6086134c93542.jpg](../icml_results/212_Latent%20Spherical%20Flow%20Policy%20for%20Reinforcement%20Learning%20with%20Combinatorial%20Actions/images/d7942a55a30647665f6500885cc8117737954872cb2cc5b5e6d6086134c93542.jpg)

### Tables

![3d400d77795ced97717112b2c8439d15c86a55c94342b1c9e4bce09a6c8f910d.jpg](../icml_results/212_Latent%20Spherical%20Flow%20Policy%20for%20Reinforcement%20Learning%20with%20Combinatorial%20Actions/tables/3d400d77795ced97717112b2c8439d15c86a55c94342b1c9e4bce09a6c8f910d.jpg)

![4f38c4529b967d2b6814b3b2c48439fbbe3fd90514c1aec8ee0a6fdcf2d113a9.jpg](../icml_results/212_Latent%20Spherical%20Flow%20Policy%20for%20Reinforcement%20Learning%20with%20Combinatorial%20Actions/tables/4f38c4529b967d2b6814b3b2c48439fbbe3fd90514c1aec8ee0a6fdcf2d113a9.jpg)

![7bbba321132205e865b8975d9b8b9b71608bff52b73e427dbcd2b673777382dd.jpg](../icml_results/212_Latent%20Spherical%20Flow%20Policy%20for%20Reinforcement%20Learning%20with%20Combinatorial%20Actions/tables/7bbba321132205e865b8975d9b8b9b71608bff52b73e427dbcd2b673777382dd.jpg)

![c05d9330430cfa22777b1eafb2c1d8da8a3a4f5a6cf232741c9abb0ccd4bc728.jpg](../icml_results/212_Latent%20Spherical%20Flow%20Policy%20for%20Reinforcement%20Learning%20with%20Combinatorial%20Actions/tables/c05d9330430cfa22777b1eafb2c1d8da8a3a4f5a6cf232741c9abb0ccd4bc728.jpg)

![c33ccead787dfb4041881f2a0b2ab3631db8695a0ff7dbd2b2380936a20886b5.jpg](../icml_results/212_Latent%20Spherical%20Flow%20Policy%20for%20Reinforcement%20Learning%20with%20Combinatorial%20Actions/tables/c33ccead787dfb4041881f2a0b2ab3631db8695a0ff7dbd2b2380936a20886b5.jpg)

<a id="paper-0210"></a>

## The Axiomatic Value of Regularization in AI Alignment from Human Preferences


### Images

![fef77b6e6da1d87fd99323c5113cae067ee119a22b702403afa9ec3ef5dfdf21.jpg](../icml_results/213_The%20Axiomatic%20Value%20of%20Regularization%20in%20AI%20Alignment%20from%20Human%20Preferences/images/fef77b6e6da1d87fd99323c5113cae067ee119a22b702403afa9ec3ef5dfdf21.jpg)

<a id="paper-0211"></a>

## Efficient Numeracy in Language Models through Single-Token Number Encodings


### Images

![0174eaf2b07345917dbc7f681f202824f7f01bbadbd7e6f53c896019bae0a416.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/images/0174eaf2b07345917dbc7f681f202824f7f01bbadbd7e6f53c896019bae0a416.jpg)

![09e81fd669deb16753a4c4d8aa92fab5705ce8a5c4a5010bbd11bef580093583.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/images/09e81fd669deb16753a4c4d8aa92fab5705ce8a5c4a5010bbd11bef580093583.jpg)

![226f9803dd91059e021842c88760e0ee3b545251c0ca86f01f0ba876a693013a.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/images/226f9803dd91059e021842c88760e0ee3b545251c0ca86f01f0ba876a693013a.jpg)

![2f45f722e74192b6c96613b36d3e0dcec0e41d469874832ee03e0cf2d5ce28f2.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/images/2f45f722e74192b6c96613b36d3e0dcec0e41d469874832ee03e0cf2d5ce28f2.jpg)

![4fd8ca68f8d0f027326cf1c006d6cde4fd1a810aa40af126e9cf3dee0c2e6073.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/images/4fd8ca68f8d0f027326cf1c006d6cde4fd1a810aa40af126e9cf3dee0c2e6073.jpg)

![6d8ececc46a9b87c962c148d85490efd99f6958a650c201aaca747e85ee705c1.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/images/6d8ececc46a9b87c962c148d85490efd99f6958a650c201aaca747e85ee705c1.jpg)

![944e365d039d64ac1c5577eb29233871f72e24c3a29056cb80672d68a385c1ce.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/images/944e365d039d64ac1c5577eb29233871f72e24c3a29056cb80672d68a385c1ce.jpg)

![a7f5ecf8b1ec259d2da6b6be1f30222fae23c42e6e924166d44925ad647d1b46.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/images/a7f5ecf8b1ec259d2da6b6be1f30222fae23c42e6e924166d44925ad647d1b46.jpg)

![b3a9c144739a56f57217d49031165fbddd47c20e9bb3182bab55283ea5a4093b.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/images/b3a9c144739a56f57217d49031165fbddd47c20e9bb3182bab55283ea5a4093b.jpg)

![f8220f30631a10a4ebddc1114007a519d0eb7173e00ed2f707af7190080bf49f.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/images/f8220f30631a10a4ebddc1114007a519d0eb7173e00ed2f707af7190080bf49f.jpg)

### Tables

![082d0affd3263a4d9c5cfa217025b423a65956e68fac069ad48cdd98b1ed3eb9.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/tables/082d0affd3263a4d9c5cfa217025b423a65956e68fac069ad48cdd98b1ed3eb9.jpg)

![11c4c0d0c5722a585f1e0a0e4dfeb8193338baba791378b3ea9de5f0a12f2513.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/tables/11c4c0d0c5722a585f1e0a0e4dfeb8193338baba791378b3ea9de5f0a12f2513.jpg)

![3708b8f52e351f59251aa9b5424147d52b03c9255cdad0b56d346f5a3932a8e5.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/tables/3708b8f52e351f59251aa9b5424147d52b03c9255cdad0b56d346f5a3932a8e5.jpg)

![58fdde4738f276682410ba25cc270c85282f4620c8136b48499c04d055817837.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/tables/58fdde4738f276682410ba25cc270c85282f4620c8136b48499c04d055817837.jpg)

![65d839c65ed586caf81bfe624dfffb87ab636bb0fc7338e714ebd3387d0ff6a2.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/tables/65d839c65ed586caf81bfe624dfffb87ab636bb0fc7338e714ebd3387d0ff6a2.jpg)

![75dc63bf3670d2ee547c74e3b1b6810b7c361fce88fbef351cf89a78dba61373.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/tables/75dc63bf3670d2ee547c74e3b1b6810b7c361fce88fbef351cf89a78dba61373.jpg)

![7b4bb307bca3c0ce567c4382ef487507716d5a42568007898dfb854180a151b1.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/tables/7b4bb307bca3c0ce567c4382ef487507716d5a42568007898dfb854180a151b1.jpg)

![9b07c8c44c14d00619cb76276ae8370b318222a4d0b13dd6a33e745eecb2687c.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/tables/9b07c8c44c14d00619cb76276ae8370b318222a4d0b13dd6a33e745eecb2687c.jpg)

![b03eccd13f24112115e309187b93201c8f41d8cee5e7c9d315e1a93ef3bfd31d.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/tables/b03eccd13f24112115e309187b93201c8f41d8cee5e7c9d315e1a93ef3bfd31d.jpg)

![cd1989c0d6f74fe8999e3f03b8009e9a43ce7811ab1d2554c52050a81fa2b38b.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/tables/cd1989c0d6f74fe8999e3f03b8009e9a43ce7811ab1d2554c52050a81fa2b38b.jpg)

![dc66277d7d69b16ec1f29cd06437d745966a2545ebdcdcafdffa190932f66618.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/tables/dc66277d7d69b16ec1f29cd06437d745966a2545ebdcdcafdffa190932f66618.jpg)

![ddfbcb0ab2a27e0a8df0a5b70bd26dbddeaa7a7db05f0589178be6abcf7e6c61.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/tables/ddfbcb0ab2a27e0a8df0a5b70bd26dbddeaa7a7db05f0589178be6abcf7e6c61.jpg)

![f037e411a4370482d560c8d42039f1d0679d29e523beef5fa09a19e0c030d5c1.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/tables/f037e411a4370482d560c8d42039f1d0679d29e523beef5fa09a19e0c030d5c1.jpg)

![f4c504ac232bcfc84097d1d32bb45ee446216eb7251d3d2c8e0d050e3aa0bb82.jpg](../icml_results/215_Efficient%20Numeracy%20in%20Language%20Models%20through%20Single-Token%20Number%20Encodings/tables/f4c504ac232bcfc84097d1d32bb45ee446216eb7251d3d2c8e0d050e3aa0bb82.jpg)

<a id="paper-0212"></a>

## Disentangling Geometry, Performance, and Training in Language Models


### Images

![0d34a6b3af5b2047de1a6b8441e4071345b4d4834312eb0fb053e473b7a313e0.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/0d34a6b3af5b2047de1a6b8441e4071345b4d4834312eb0fb053e473b7a313e0.jpg)

![120326ec46e5620ccd0997e061af28df2f4ebbb41746fe94f9a9949de64e3bde.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/120326ec46e5620ccd0997e061af28df2f4ebbb41746fe94f9a9949de64e3bde.jpg)

![183171d9a2f683bea95f3d11648ab4181de78237a5cf02a5061a2a92646249f0.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/183171d9a2f683bea95f3d11648ab4181de78237a5cf02a5061a2a92646249f0.jpg)

![3c81fdba3f84842759f60e226e26d9ed82b90bf4d2fa00007dffd0f54d37b1ff.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/3c81fdba3f84842759f60e226e26d9ed82b90bf4d2fa00007dffd0f54d37b1ff.jpg)

![3ddc7061059a43dcded21108da74c67dffbb50f2d0d47148ade24a0f0074d497.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/3ddc7061059a43dcded21108da74c67dffbb50f2d0d47148ade24a0f0074d497.jpg)

![4327efdca12e192488a5c1234361ced4cd548204a4a185723d103e74e6eac9d4.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/4327efdca12e192488a5c1234361ced4cd548204a4a185723d103e74e6eac9d4.jpg)

![4d078b92e79872eddec8044606c96d19bedc496564ea6056dc8c30b0cca9e7e6.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/4d078b92e79872eddec8044606c96d19bedc496564ea6056dc8c30b0cca9e7e6.jpg)

![55e05cdcbde76a2b1650646c4fbfc49e2d59e01fb6b677851aa021ecd4083855.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/55e05cdcbde76a2b1650646c4fbfc49e2d59e01fb6b677851aa021ecd4083855.jpg)

![65b9488223fae628acf81d0de0cc0e5c53c5df8dae72ceada4c6f3ec0055f89d.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/65b9488223fae628acf81d0de0cc0e5c53c5df8dae72ceada4c6f3ec0055f89d.jpg)

![77d8a28bd86690cd89ceb582cee0d0c999cbfdb99003128a2c9796504fee6f9b.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/77d8a28bd86690cd89ceb582cee0d0c999cbfdb99003128a2c9796504fee6f9b.jpg)

![7b16a81bb2e7c090653f1a972a4bb3aef610acde264e6cde1ae42806013ceaba.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/7b16a81bb2e7c090653f1a972a4bb3aef610acde264e6cde1ae42806013ceaba.jpg)

![7fca9ef56d9e33bbd7281fd26f066da1bb53c4992ca7c85a9f0d33f5d827d950.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/7fca9ef56d9e33bbd7281fd26f066da1bb53c4992ca7c85a9f0d33f5d827d950.jpg)

![8207d5f2c58217fb423d459976beaaaecd8c5ed98f778dd7a0714ddb43e6dbc7.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/8207d5f2c58217fb423d459976beaaaecd8c5ed98f778dd7a0714ddb43e6dbc7.jpg)

![88d22e2f0f1ff05733efdad0d0bccfc3d98b4df87bc610dbe4d956c837d38e21.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/88d22e2f0f1ff05733efdad0d0bccfc3d98b4df87bc610dbe4d956c837d38e21.jpg)

![8df47913b4afd594325e81e99e94c48b94492b688774dc4984a74131b2b947a7.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/8df47913b4afd594325e81e99e94c48b94492b688774dc4984a74131b2b947a7.jpg)

![8ef68eb6535fc69df74eec5e5e7b91febf3f473d754905492fbed15657996e59.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/8ef68eb6535fc69df74eec5e5e7b91febf3f473d754905492fbed15657996e59.jpg)

![9224f7038d3852299e553bcec348c850711ab6916418162b6d8d6b32cd64fb0c.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/9224f7038d3852299e553bcec348c850711ab6916418162b6d8d6b32cd64fb0c.jpg)

![a1c62aa66d0a2327c861e229ee8fa6fd23534c0ca4b71221f62727ffd4051a43.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/a1c62aa66d0a2327c861e229ee8fa6fd23534c0ca4b71221f62727ffd4051a43.jpg)

![ba5af4180da9a303a1da30c74d73cd4b8b5d447dbf95e57458368ef377790024.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/ba5af4180da9a303a1da30c74d73cd4b8b5d447dbf95e57458368ef377790024.jpg)

![cbf27aec870e31a2198f4b49ea05ea244a0989e9148f1e44fb70dfcae051c889.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/cbf27aec870e31a2198f4b49ea05ea244a0989e9148f1e44fb70dfcae051c889.jpg)

![cd6b9f4bc01ae93fb9288167a8c1f0b8b146c410d36511c89e3cc7a42c5cc8b8.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/cd6b9f4bc01ae93fb9288167a8c1f0b8b146c410d36511c89e3cc7a42c5cc8b8.jpg)

![cdf2aa6e0c3dcdf2bf5ca42bad41f3728eb3bd0c615e3110e1cafdf2f9dd2b93.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/cdf2aa6e0c3dcdf2bf5ca42bad41f3728eb3bd0c615e3110e1cafdf2f9dd2b93.jpg)

![d07ef37d6bee7e26711589e2cf3cbecbf5370cc59736be23895bb223032ef611.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/d07ef37d6bee7e26711589e2cf3cbecbf5370cc59736be23895bb223032ef611.jpg)

![d99292240673339bef94fe521eef05e2f0ab89c5d60a07e5b9266b9ffa3ce9f2.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/d99292240673339bef94fe521eef05e2f0ab89c5d60a07e5b9266b9ffa3ce9f2.jpg)

![e73d2fc0e8ba7b8f5f69e9f2d515deb06748710cacd40370ac767ea048301a27.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/e73d2fc0e8ba7b8f5f69e9f2d515deb06748710cacd40370ac767ea048301a27.jpg)

![e7ec74aad131fcaade0e19b010ebf856716e580ad351615d8a37ba1e2bec847c.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/e7ec74aad131fcaade0e19b010ebf856716e580ad351615d8a37ba1e2bec847c.jpg)

![ed33af43056ab33fa68d75f3660a228e831a833e3384285c92508077c0113692.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/ed33af43056ab33fa68d75f3660a228e831a833e3384285c92508077c0113692.jpg)

![ed40cd2c20e515611ee4f9de101b381d17e57c09e1958e8cbc744374f2235d03.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/ed40cd2c20e515611ee4f9de101b381d17e57c09e1958e8cbc744374f2235d03.jpg)

![ee752dada4fe338e04df400e9912386c198e1f869f932b4e24180f77c68fabd1.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/ee752dada4fe338e04df400e9912386c198e1f869f932b4e24180f77c68fabd1.jpg)

![f24be7668c7e75c7a3256ee59e67e52197ad630d0fce001dfa3008c8c052fa3e.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/f24be7668c7e75c7a3256ee59e67e52197ad630d0fce001dfa3008c8c052fa3e.jpg)

![fb9cd84fd1d440b0ed41ab25d6aa5d64705b22c563381d2827b2bce2626288cd.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/images/fb9cd84fd1d440b0ed41ab25d6aa5d64705b22c563381d2827b2bce2626288cd.jpg)

### Tables

![1070b1700ce5e8323488c987f1e035b6990437bf7451f03821d4a49c4f51581f.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/tables/1070b1700ce5e8323488c987f1e035b6990437bf7451f03821d4a49c4f51581f.jpg)

![2c20d077352c4c9008ae01fdd5ef9a52d51c2ce22d4de2f2cabf2c5daef85f63.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/tables/2c20d077352c4c9008ae01fdd5ef9a52d51c2ce22d4de2f2cabf2c5daef85f63.jpg)

![39aaba27c7642c62a55620b8d891bfb93052c56417ce1e2d9fafabcd6a69f80f.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/tables/39aaba27c7642c62a55620b8d891bfb93052c56417ce1e2d9fafabcd6a69f80f.jpg)

![3f209719bee098b2cdd987f222810a073c45a2e980d207fb33a56dce9c9981a5.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/tables/3f209719bee098b2cdd987f222810a073c45a2e980d207fb33a56dce9c9981a5.jpg)

![40169a6b6c1bf4ca7271caa22a2f382f175e1f0af93936855706507d0af2f5af.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/tables/40169a6b6c1bf4ca7271caa22a2f382f175e1f0af93936855706507d0af2f5af.jpg)

![4a590f86ad5daa666faa8a2d445dd48e0548653c8564a529ef89119f36f45d1c.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/tables/4a590f86ad5daa666faa8a2d445dd48e0548653c8564a529ef89119f36f45d1c.jpg)

![4f52c4be38f44feb25bc54dc15023f96b42e405e469a01d83d65da6517f14fcb.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/tables/4f52c4be38f44feb25bc54dc15023f96b42e405e469a01d83d65da6517f14fcb.jpg)

![511f5b7bc5c8b118c0cbf1b4a9ce97d078cad98612a4fbc82c894b54cb3c35f2.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/tables/511f5b7bc5c8b118c0cbf1b4a9ce97d078cad98612a4fbc82c894b54cb3c35f2.jpg)

![57c87dbcffad65059ee0e7c5c08588f84a28009022c94db934c06a367dde7d42.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/tables/57c87dbcffad65059ee0e7c5c08588f84a28009022c94db934c06a367dde7d42.jpg)

![6b4f2fb7ccdc681d68a34480feb3d46593e8a85d55170a8bc6c72783c74f9974.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/tables/6b4f2fb7ccdc681d68a34480feb3d46593e8a85d55170a8bc6c72783c74f9974.jpg)

![7dee152edc06556a2eba5c0239835f6bcb34f98fac98932485761161157dba18.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/tables/7dee152edc06556a2eba5c0239835f6bcb34f98fac98932485761161157dba18.jpg)

![b423cb05af13c600df75d07ed5b2e124dd61267e534e3c9ea92fce24d14963a7.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/tables/b423cb05af13c600df75d07ed5b2e124dd61267e534e3c9ea92fce24d14963a7.jpg)

![c8e9bd9de7fbd9925ef0095dfa29771994dcd9a970c2ffa550426ce26c5a0f9e.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/tables/c8e9bd9de7fbd9925ef0095dfa29771994dcd9a970c2ffa550426ce26c5a0f9e.jpg)

![e1085f4d02c717fd3f7ca5e0bc94c99beac96937267843253598a811167adef1.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/tables/e1085f4d02c717fd3f7ca5e0bc94c99beac96937267843253598a811167adef1.jpg)

![e2d7e707f7759e02026d051b0abd1c5a172d3100e2a7fe0686901570187b4653.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/tables/e2d7e707f7759e02026d051b0abd1c5a172d3100e2a7fe0686901570187b4653.jpg)

![e3fffb2945a4906b9a1cd676d0540efb792c5d4db6ac0155c5f8725e19c8e1fd.jpg](../icml_results/216_Disentangling%20Geometry%2C%20Performance%2C%20and%20Training%20in%20Language%20Models/tables/e3fffb2945a4906b9a1cd676d0540efb792c5d4db6ac0155c5f8725e19c8e1fd.jpg)

<a id="paper-0213"></a>

## Learning Structured Reasoning via Tractable Trajectory Control


### Images

![3d13bbcfcf6590380ee4eeff1dfe4b016b690bcafea93b1d68fe4653c620acd9.jpg](../icml_results/218_Learning%20Structured%20Reasoning%20via%20Tractable%20Trajectory%20Control/images/3d13bbcfcf6590380ee4eeff1dfe4b016b690bcafea93b1d68fe4653c620acd9.jpg)

![440340c4ddd55331ae6fe6b667f5303e3e8ecb463ec4aa169eb8d2f2a78f0fd7.jpg](../icml_results/218_Learning%20Structured%20Reasoning%20via%20Tractable%20Trajectory%20Control/images/440340c4ddd55331ae6fe6b667f5303e3e8ecb463ec4aa169eb8d2f2a78f0fd7.jpg)

![4ee05f6f030ee2dc3923bf40f1376d9d2afae21d17cec7cc079d5af4f06b2fb5.jpg](../icml_results/218_Learning%20Structured%20Reasoning%20via%20Tractable%20Trajectory%20Control/images/4ee05f6f030ee2dc3923bf40f1376d9d2afae21d17cec7cc079d5af4f06b2fb5.jpg)

![71fee1f7bb7aa715c8fc52429ac55a41b61f5e72aeb5f7cda1d117f2714a4426.jpg](../icml_results/218_Learning%20Structured%20Reasoning%20via%20Tractable%20Trajectory%20Control/images/71fee1f7bb7aa715c8fc52429ac55a41b61f5e72aeb5f7cda1d117f2714a4426.jpg)

![7a11ee0ef63b60bd08157902ce8ea608cea27eceda6ea1850d9bfaee45182988.jpg](../icml_results/218_Learning%20Structured%20Reasoning%20via%20Tractable%20Trajectory%20Control/images/7a11ee0ef63b60bd08157902ce8ea608cea27eceda6ea1850d9bfaee45182988.jpg)

![86f80641bff5107c576470c4413fa6f9ff32129ca96845ce838cefdf706b3e62.jpg](../icml_results/218_Learning%20Structured%20Reasoning%20via%20Tractable%20Trajectory%20Control/images/86f80641bff5107c576470c4413fa6f9ff32129ca96845ce838cefdf706b3e62.jpg)

![8bd604c394054fbb71a0fd39cd77f2275aa6b54993fcb05c47395174ef5d1ba9.jpg](../icml_results/218_Learning%20Structured%20Reasoning%20via%20Tractable%20Trajectory%20Control/images/8bd604c394054fbb71a0fd39cd77f2275aa6b54993fcb05c47395174ef5d1ba9.jpg)

![a282bb0252d6211a608ae266a53ccdb9b57251c0b4e111e8f6a9169e33934c8f.jpg](../icml_results/218_Learning%20Structured%20Reasoning%20via%20Tractable%20Trajectory%20Control/images/a282bb0252d6211a608ae266a53ccdb9b57251c0b4e111e8f6a9169e33934c8f.jpg)

![b5d9ec9f0d8c7ab05bbb82e4f9e6dc8760c2ef2f27a7849c30cd5cefe1710607.jpg](../icml_results/218_Learning%20Structured%20Reasoning%20via%20Tractable%20Trajectory%20Control/images/b5d9ec9f0d8c7ab05bbb82e4f9e6dc8760c2ef2f27a7849c30cd5cefe1710607.jpg)

### Tables

![4b6a55cdf104904db67126afc481fe3b82a7ddd615fc0f3f7255689802047050.jpg](../icml_results/218_Learning%20Structured%20Reasoning%20via%20Tractable%20Trajectory%20Control/tables/4b6a55cdf104904db67126afc481fe3b82a7ddd615fc0f3f7255689802047050.jpg)

![5af64757949dc0e4d7e3b963f06f718b97a1415e25bed57bb7653c4131b501bb.jpg](../icml_results/218_Learning%20Structured%20Reasoning%20via%20Tractable%20Trajectory%20Control/tables/5af64757949dc0e4d7e3b963f06f718b97a1415e25bed57bb7653c4131b501bb.jpg)

![c21872b062ca8fd71c5246584cd76fabbb9f7e437c9efd6c525937e3260bf7bd.jpg](../icml_results/218_Learning%20Structured%20Reasoning%20via%20Tractable%20Trajectory%20Control/tables/c21872b062ca8fd71c5246584cd76fabbb9f7e437c9efd6c525937e3260bf7bd.jpg)

![d000b76aebb6a8be9732b949157af3068a7221cba129746690a61bd63d4076ea.jpg](../icml_results/218_Learning%20Structured%20Reasoning%20via%20Tractable%20Trajectory%20Control/tables/d000b76aebb6a8be9732b949157af3068a7221cba129746690a61bd63d4076ea.jpg)

![f837e45814cab918e1c9924285dbdadddbafd1345c574442aa5d66e9fdb65ddb.jpg](../icml_results/218_Learning%20Structured%20Reasoning%20via%20Tractable%20Trajectory%20Control/tables/f837e45814cab918e1c9924285dbdadddbafd1345c574442aa5d66e9fdb65ddb.jpg)

<a id="paper-0214"></a>

## Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance


### Images

![0e805617bc84bdfc6f24fbc56e53c7a32f1c6c0bf84f0ab5a56978fd73553819.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/images/0e805617bc84bdfc6f24fbc56e53c7a32f1c6c0bf84f0ab5a56978fd73553819.jpg)

![1f78b3d84737fd46cffa1943016e926397be25bd6fd04fc3b1ec5d69ba40526e.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/images/1f78b3d84737fd46cffa1943016e926397be25bd6fd04fc3b1ec5d69ba40526e.jpg)

![2729e5ac25b084c8e48dd286e5c5f67d88fcede8b3dfe208698b6c41d0f9ab1e.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/images/2729e5ac25b084c8e48dd286e5c5f67d88fcede8b3dfe208698b6c41d0f9ab1e.jpg)

![4260e0c7e68554652c37bd52e954357adf69cf14f1c36a954487bec5a2e3f6db.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/images/4260e0c7e68554652c37bd52e954357adf69cf14f1c36a954487bec5a2e3f6db.jpg)

![4cd4b8adfdaf4d64305f076abe6cce05dba6f248fed8fcb181b55bcb07dccdce.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/images/4cd4b8adfdaf4d64305f076abe6cce05dba6f248fed8fcb181b55bcb07dccdce.jpg)

![5cc595b2e579f723f3952d3356546e4bf5bf0485043ba05269df9a9332f11794.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/images/5cc595b2e579f723f3952d3356546e4bf5bf0485043ba05269df9a9332f11794.jpg)

![5efbd0e826ed2b4ff585878b1229409895e04ad87060e125765adf341fc6497a.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/images/5efbd0e826ed2b4ff585878b1229409895e04ad87060e125765adf341fc6497a.jpg)

![7eb55ad0038e8b997926ca86cf02011f84f9d84b45453b37c9e5554df302c98d.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/images/7eb55ad0038e8b997926ca86cf02011f84f9d84b45453b37c9e5554df302c98d.jpg)

![81b67ba560177c4eb7f2c9f6956c351a22b44a931e35e85ba2cd1ad5e7da101c.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/images/81b67ba560177c4eb7f2c9f6956c351a22b44a931e35e85ba2cd1ad5e7da101c.jpg)

![94f357fa9fad9ad7f793b9f11bf0a0b1dcc6fb24cd5b7b2ab848a464bebc5015.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/images/94f357fa9fad9ad7f793b9f11bf0a0b1dcc6fb24cd5b7b2ab848a464bebc5015.jpg)

![cf6b1b129645648c59c6a24f66e1654e155f28ed7ecd102bf3187c61f014aeec.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/images/cf6b1b129645648c59c6a24f66e1654e155f28ed7ecd102bf3187c61f014aeec.jpg)

![ed504d91040a1e2ccdbf852e4f398d45a4b8b040208e3513db0d3c6c66c2d849.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/images/ed504d91040a1e2ccdbf852e4f398d45a4b8b040208e3513db0d3c6c66c2d849.jpg)

![eec8d0264e71d6b450d0f97773942fc201144d83ef9982907024c37a94276cfa.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/images/eec8d0264e71d6b450d0f97773942fc201144d83ef9982907024c37a94276cfa.jpg)

### Tables

![07af9fe7817bd8c922625d7b463dc64508f8faeb7b5e7f98403b45167ddbc95b.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/tables/07af9fe7817bd8c922625d7b463dc64508f8faeb7b5e7f98403b45167ddbc95b.jpg)

![0d293faa00ba91a1a8a307f958156180f5e98dfc15825109f5fb17712a968117.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/tables/0d293faa00ba91a1a8a307f958156180f5e98dfc15825109f5fb17712a968117.jpg)

![1b331673b5a8c945ee27c038a9975e0a5827aa1065a878554530f0c6e601cf9f.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/tables/1b331673b5a8c945ee27c038a9975e0a5827aa1065a878554530f0c6e601cf9f.jpg)

![24d9b621e2cd9eef5242e944ad7fd34f197bc41c58adc153526ed783f2a6d3f6.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/tables/24d9b621e2cd9eef5242e944ad7fd34f197bc41c58adc153526ed783f2a6d3f6.jpg)

![27f968654dd5426b306a5b871c2d5fdeb0a9cfcf3d0aab433cc0d5eebb92a32f.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/tables/27f968654dd5426b306a5b871c2d5fdeb0a9cfcf3d0aab433cc0d5eebb92a32f.jpg)

![3a5d55ee461e9f81a15c876f317b7c0ba1f67f3a4331b5034de41c4c0c553281.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/tables/3a5d55ee461e9f81a15c876f317b7c0ba1f67f3a4331b5034de41c4c0c553281.jpg)

![3f70d76dc34449e09a164987ce9a8a8134bc914bdf899acb3f414e7de69d12f5.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/tables/3f70d76dc34449e09a164987ce9a8a8134bc914bdf899acb3f414e7de69d12f5.jpg)

![41682baf64d792c95142e3807c63e70e82eb8660044278413d767c6d9865da3c.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/tables/41682baf64d792c95142e3807c63e70e82eb8660044278413d767c6d9865da3c.jpg)

![45f22ad3716c3bc7b17bcdb1de012efdbac13b2124dd3d729a176849ebafcab1.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/tables/45f22ad3716c3bc7b17bcdb1de012efdbac13b2124dd3d729a176849ebafcab1.jpg)

![b45dc31d5bf3d43b01406d75fa9d3634dd13bfbc48a407c6758a423b928b93b0.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/tables/b45dc31d5bf3d43b01406d75fa9d3634dd13bfbc48a407c6758a423b928b93b0.jpg)

![fe50fd5b4f653a60db916ca317014575c3c0e8d9377f16454c530cccab07ac63.jpg](../icml_results/219_Stable-GFlowNet_%20Toward%20Diverse%20and%20Robust%20LLM%20Red-Teaming%20via%20Contrastive%20Trajectory%20Balance/tables/fe50fd5b4f653a60db916ca317014575c3c0e8d9377f16454c530cccab07ac63.jpg)

<a id="paper-0215"></a>

## AI Engram: In Search of Memory Traces in Artificial Intelligence


### Images

![036bde0d8e8c3002bb05028f6a60d4ef8d17989242f56e7c49b8b95453058968.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/036bde0d8e8c3002bb05028f6a60d4ef8d17989242f56e7c49b8b95453058968.jpg)

![0bbf4806894c872ff3350964aa487a107769d1c452c3c8eb0319d1917657d19d.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/0bbf4806894c872ff3350964aa487a107769d1c452c3c8eb0319d1917657d19d.jpg)

![16516531d1b1dbc3bedcb3d576edd0f4b241f0b5f898ce7097a075615cca66a4.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/16516531d1b1dbc3bedcb3d576edd0f4b241f0b5f898ce7097a075615cca66a4.jpg)

![1dcbcb9cbdf4a91c3d70cb8644f70d7fb7e0b49b9af8bd1bf0d043fba0ab623d.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/1dcbcb9cbdf4a91c3d70cb8644f70d7fb7e0b49b9af8bd1bf0d043fba0ab623d.jpg)

![26321f51df9a133a4e9cc325255054aec48d901cb9309019c3ab930176df8c55.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/26321f51df9a133a4e9cc325255054aec48d901cb9309019c3ab930176df8c55.jpg)

![3824fc3e24141fd68dca89a66a3c95cdea218c17468b63f0096b223725c2b282.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/3824fc3e24141fd68dca89a66a3c95cdea218c17468b63f0096b223725c2b282.jpg)

![495ad3485e66c8a67f321e15ec74a3273a482c13dc1034a4027f2ec8462c603e.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/495ad3485e66c8a67f321e15ec74a3273a482c13dc1034a4027f2ec8462c603e.jpg)

![49f3c32cf4acfcc94897daed64c2aac16ea6494d52c67c13bc16ba6694afc81e.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/49f3c32cf4acfcc94897daed64c2aac16ea6494d52c67c13bc16ba6694afc81e.jpg)

![62d132d65e66eb77cc11ef9eeac2c53fe8796d08a6172ffc9ccbf2ff0eb5cc9c.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/62d132d65e66eb77cc11ef9eeac2c53fe8796d08a6172ffc9ccbf2ff0eb5cc9c.jpg)

![6c0bdad31d70c0634598adfd85972ffde39ac2f80138766abc20fdaea5d77e92.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/6c0bdad31d70c0634598adfd85972ffde39ac2f80138766abc20fdaea5d77e92.jpg)

![7c91aca7bd835f920a3f45c1a817fc9e4eb351705813f13feef35c1b1701ab95.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/7c91aca7bd835f920a3f45c1a817fc9e4eb351705813f13feef35c1b1701ab95.jpg)

![b20a47c91fef8b8a5dc329f8c1f714574a00c468fe68b9512341d972c30042c3.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/b20a47c91fef8b8a5dc329f8c1f714574a00c468fe68b9512341d972c30042c3.jpg)

![b36a334e2ac674bddda94c9fbbeabac055d4536085c576b4985cbf860612f089.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/b36a334e2ac674bddda94c9fbbeabac055d4536085c576b4985cbf860612f089.jpg)

![b4a6e7ce882d7d4a13f657d878fc939d75bf5a5c5e667fade7086fb5b7bb5ee6.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/b4a6e7ce882d7d4a13f657d878fc939d75bf5a5c5e667fade7086fb5b7bb5ee6.jpg)

![c2426547b5fcd691e8b841658d6e36022db2924f43a49d384a0f7c43ab222f4d.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/c2426547b5fcd691e8b841658d6e36022db2924f43a49d384a0f7c43ab222f4d.jpg)

![efe9e5443b2fd742ca4e08e70179394a7489df14f98e5d631feefb4695f4b75f.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/efe9e5443b2fd742ca4e08e70179394a7489df14f98e5d631feefb4695f4b75f.jpg)

![f4933c27a09a00e04135695d2c18f727eb81ca8b3264852657e36fe577bd4337.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/f4933c27a09a00e04135695d2c18f727eb81ca8b3264852657e36fe577bd4337.jpg)

![f68cc05b9575fe6752d7cd5f8b95d1daff30f3ddb02d85ca4d68ecc7aedea5ba.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/images/f68cc05b9575fe6752d7cd5f8b95d1daff30f3ddb02d85ca4d68ecc7aedea5ba.jpg)

### Tables

![3ba417fd188b27361a0be8683dbfeda058900724204b3f3c8449c3e2c86e2420.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/tables/3ba417fd188b27361a0be8683dbfeda058900724204b3f3c8449c3e2c86e2420.jpg)

![48d8327e8c2a673bf55fe9efbc7d1aa1f327f52577105068697dda893bf37eb3.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/tables/48d8327e8c2a673bf55fe9efbc7d1aa1f327f52577105068697dda893bf37eb3.jpg)

![4e53511994be05a9bc3defffe257115e66d4c6d9a0d311c9dd149f48cde00f24.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/tables/4e53511994be05a9bc3defffe257115e66d4c6d9a0d311c9dd149f48cde00f24.jpg)

![6fd21c22cb8020c8e0d7878b7a478cc0164968996fb7cb6fe81722fba0bbabf3.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/tables/6fd21c22cb8020c8e0d7878b7a478cc0164968996fb7cb6fe81722fba0bbabf3.jpg)

![72058a15e870ccfee56f692a9d78e5af7392ce5b28c8d07c291a1a96e4be914c.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/tables/72058a15e870ccfee56f692a9d78e5af7392ce5b28c8d07c291a1a96e4be914c.jpg)

![807bc47f8bbafdadc8d693d7cb09c9954159f4bda017a9c08da27d2e690a914d.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/tables/807bc47f8bbafdadc8d693d7cb09c9954159f4bda017a9c08da27d2e690a914d.jpg)

![9c72a6bacf6b898abe04c822055f52e6d68cc51c6ab6ef6920dcb8b80c19adce.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/tables/9c72a6bacf6b898abe04c822055f52e6d68cc51c6ab6ef6920dcb8b80c19adce.jpg)

![ba2a679d84e138c8fc2b92c5b4b0e0bb0fa23d0e099d71ff3027bdfe9d7a7cdf.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/tables/ba2a679d84e138c8fc2b92c5b4b0e0bb0fa23d0e099d71ff3027bdfe9d7a7cdf.jpg)

![ecf9c27eff43b5ef930f27bee7703c15f472074a52de0d861d1e88759af911c2.jpg](../icml_results/220_AI%20Engram_%20In%20Search%20of%20Memory%20Traces%20in%20Artificial%20Intelligence/tables/ecf9c27eff43b5ef930f27bee7703c15f472074a52de0d861d1e88759af911c2.jpg)

<a id="paper-0216"></a>

## Guaranteed Optimal Compositional Explanations for Neurons


### Images

![03282eb4fcf9c1565a3771e5877fa4a26c574303228c671980bef204c3f1e873.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/03282eb4fcf9c1565a3771e5877fa4a26c574303228c671980bef204c3f1e873.jpg)

![045105a6c751e163b3fddd8195b1d9d97a5c107910cb47de5e160969a25300c2.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/045105a6c751e163b3fddd8195b1d9d97a5c107910cb47de5e160969a25300c2.jpg)

![070f50e677427bc8214b45f796d071907e6214d2bc7acf956ea08f16f1b73f72.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/070f50e677427bc8214b45f796d071907e6214d2bc7acf956ea08f16f1b73f72.jpg)

![0a11dec7edd052847fa9e98f56f3f08f893c74ab8f202efc414f259678761162.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/0a11dec7edd052847fa9e98f56f3f08f893c74ab8f202efc414f259678761162.jpg)

![0bd2d682c1dd363c1b15dc601a732660609714626ab05f4a6e82e32425732631.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/0bd2d682c1dd363c1b15dc601a732660609714626ab05f4a6e82e32425732631.jpg)

![0c3364c1e80a2ab5627a962a6bcfa93825c9de97906c8b7a2483cde555404f20.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/0c3364c1e80a2ab5627a962a6bcfa93825c9de97906c8b7a2483cde555404f20.jpg)

![0f1ca46b76a5693bb55f2b2e9889b5228530e33b098befedd4efc6e7b1176af3.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/0f1ca46b76a5693bb55f2b2e9889b5228530e33b098befedd4efc6e7b1176af3.jpg)

![104677b234dcf96b606287c679c31be667d3444620eb1fdbe2cbf7748afa3983.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/104677b234dcf96b606287c679c31be667d3444620eb1fdbe2cbf7748afa3983.jpg)

![1640702f480e6fdbfeb33603cef5504b674a736138ab62c3f39059d662903112.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/1640702f480e6fdbfeb33603cef5504b674a736138ab62c3f39059d662903112.jpg)

![168229bca92de614a83c5bf328f3efbc26c5604ddc3888a192aa58ba6188e1bb.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/168229bca92de614a83c5bf328f3efbc26c5604ddc3888a192aa58ba6188e1bb.jpg)

![299fc282d128fb57b5649965d9322d6bff113cf43e95411d1fafcbc6614538ae.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/299fc282d128fb57b5649965d9322d6bff113cf43e95411d1fafcbc6614538ae.jpg)

![34982e5796af9360005ce55d5fea8dc25a70355b401ef6af98415ba80cf2a81a.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/34982e5796af9360005ce55d5fea8dc25a70355b401ef6af98415ba80cf2a81a.jpg)

![39f20f88c7c7b72d50a4951d5ead76f0cdca40a216234e8a3613dbd8e438b694.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/39f20f88c7c7b72d50a4951d5ead76f0cdca40a216234e8a3613dbd8e438b694.jpg)

![3c1cd3d7043474186041fa5c486946d1369d9f5c2b4de1c095f9cb0b1054c969.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/3c1cd3d7043474186041fa5c486946d1369d9f5c2b4de1c095f9cb0b1054c969.jpg)

![4771d1902e5a7b1dc7f18cf809183c478c4d553c41244010d26b96e99a16db02.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/4771d1902e5a7b1dc7f18cf809183c478c4d553c41244010d26b96e99a16db02.jpg)

![4a8bcea8dffd088c32993ece97c9fccdcd7a40e59dcd28138e443b4bcddece55.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/4a8bcea8dffd088c32993ece97c9fccdcd7a40e59dcd28138e443b4bcddece55.jpg)

![4bf8d9ea8788cb202742dd8d43e96260d8dcd16c18d985fd2d8cd3414ddaa238.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/4bf8d9ea8788cb202742dd8d43e96260d8dcd16c18d985fd2d8cd3414ddaa238.jpg)

![4d4bf048483573b904ca95a6edb70f595cc75306e1495c8b51bbbd3c47033621.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/4d4bf048483573b904ca95a6edb70f595cc75306e1495c8b51bbbd3c47033621.jpg)

![541353cd0f8a5a201dc13c733cdc6de5130cab30becb11d9511001b569af5966.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/541353cd0f8a5a201dc13c733cdc6de5130cab30becb11d9511001b569af5966.jpg)

![543e22f867913e60fe5e05267b421e444fe66c3870a039c19039e7d4991296cc.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/543e22f867913e60fe5e05267b421e444fe66c3870a039c19039e7d4991296cc.jpg)

![5a9eb5e54eb40cc4cfde0ede744a69cd0e6a2fd19c73183c91dd900044ce2ac8.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/5a9eb5e54eb40cc4cfde0ede744a69cd0e6a2fd19c73183c91dd900044ce2ac8.jpg)

![6237300ae7adf9b3efd8af2ff0df9935a1f3f40c615939ce4b80890dd58c09e3.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/6237300ae7adf9b3efd8af2ff0df9935a1f3f40c615939ce4b80890dd58c09e3.jpg)

![69ee0600adff86fab58471c14c88f785b54ed689d636a1ecedbb0025799c166b.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/69ee0600adff86fab58471c14c88f785b54ed689d636a1ecedbb0025799c166b.jpg)

![6d4f305bed798985c06adc5df838275c9e680707b491a25b5733309fd9f0d59a.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/6d4f305bed798985c06adc5df838275c9e680707b491a25b5733309fd9f0d59a.jpg)

![74e9c1083e77653accd5c7d2ce528e5f303e4b5a4cc08bf4c4063ba01c8f6a6d.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/74e9c1083e77653accd5c7d2ce528e5f303e4b5a4cc08bf4c4063ba01c8f6a6d.jpg)

![77da2722670214b8865856f53cbbc20b3eb72a1ba01852e80ef469ed3d68a22e.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/77da2722670214b8865856f53cbbc20b3eb72a1ba01852e80ef469ed3d68a22e.jpg)

![7e685a82fcc38744033cf2f225ac751cb5f8ecf578d9af399936447dfb8849f6.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/7e685a82fcc38744033cf2f225ac751cb5f8ecf578d9af399936447dfb8849f6.jpg)

![80ce422f06fac7f329ca35509d43576dccd064390b15b80dad98d0a6b33eb5f7.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/80ce422f06fac7f329ca35509d43576dccd064390b15b80dad98d0a6b33eb5f7.jpg)

![8cc22cc7a2dfc1c842f6913c771a5c974285ffe4e7c3f814a9a40132322ad6f2.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/8cc22cc7a2dfc1c842f6913c771a5c974285ffe4e7c3f814a9a40132322ad6f2.jpg)

![929c7d3331825350be28b2e81b41ec0b548ebf98c47c9433ef3a143b5d1b9610.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/929c7d3331825350be28b2e81b41ec0b548ebf98c47c9433ef3a143b5d1b9610.jpg)

![99aefc18e1f6d98c2c41e68f00d237dafbf6b5eff6d8fe07cd52d16f0885c2b3.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/99aefc18e1f6d98c2c41e68f00d237dafbf6b5eff6d8fe07cd52d16f0885c2b3.jpg)

![9a2c8835664c2bac1a1108bee5c1a1d019e9e3b31727dfd2ab2e8d9df6043d63.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/9a2c8835664c2bac1a1108bee5c1a1d019e9e3b31727dfd2ab2e8d9df6043d63.jpg)

![a0bb8ca81597f8a75c346d1245d77fdb757d3444e89ec662c3c799a8c38a322f.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/a0bb8ca81597f8a75c346d1245d77fdb757d3444e89ec662c3c799a8c38a322f.jpg)

![a40c378e1a1057b2402cb9be4afeeef1499868230b7b150e1a7f415acefb2b2d.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/a40c378e1a1057b2402cb9be4afeeef1499868230b7b150e1a7f415acefb2b2d.jpg)

![a599ced0c2bbba5b3bd788e404a0a445ee27db9b472f37d452bc74484b6a075a.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/a599ced0c2bbba5b3bd788e404a0a445ee27db9b472f37d452bc74484b6a075a.jpg)

![aef135b87903ab953a3c260bc7cd54b9e3143effbf8bed4823aa51d2d8bd8db8.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/aef135b87903ab953a3c260bc7cd54b9e3143effbf8bed4823aa51d2d8bd8db8.jpg)

![b036f7829b6939bb9c9fc37942ba4f9a7b080be794a5f35311e8d6c03054143c.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/b036f7829b6939bb9c9fc37942ba4f9a7b080be794a5f35311e8d6c03054143c.jpg)

![b744218f8f192f7cede13485196fd1d12241a43fa4485155b45cd4bdcc21d106.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/b744218f8f192f7cede13485196fd1d12241a43fa4485155b45cd4bdcc21d106.jpg)

![b7f46feaf99d4fa18fd16e5399b5f48fd2e6eea4ce4e5f5bb109c187a199d7ec.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/b7f46feaf99d4fa18fd16e5399b5f48fd2e6eea4ce4e5f5bb109c187a199d7ec.jpg)

![ba71f7b4cc12a1db0cce934d7fdfa817b93c7a28185cc26d8b7c6f67fdf47e44.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/ba71f7b4cc12a1db0cce934d7fdfa817b93c7a28185cc26d8b7c6f67fdf47e44.jpg)

![bf3bb6f25eb003db7af1867fbdcab5183292d507615df0f16fafb0ead1f3e541.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/bf3bb6f25eb003db7af1867fbdcab5183292d507615df0f16fafb0ead1f3e541.jpg)

![c680a07ab920b70f2eb73bef4556740428dd5faffe8a14c26fcc92ba98512618.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/c680a07ab920b70f2eb73bef4556740428dd5faffe8a14c26fcc92ba98512618.jpg)

![c753ae4ec0f3f052b618a5f5ac8f23b46a567da45e82222ac4b7001fa29dd0a9.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/c753ae4ec0f3f052b618a5f5ac8f23b46a567da45e82222ac4b7001fa29dd0a9.jpg)

![cbc6b540a4c0d82009f4fabd69f8dfc88c19dbdc0a56ac6e1440f8e877459f99.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/cbc6b540a4c0d82009f4fabd69f8dfc88c19dbdc0a56ac6e1440f8e877459f99.jpg)

![cbf9740b21768fda3350c7aeff1aaacc7fb9d31f517f4b7791556677c69712c9.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/cbf9740b21768fda3350c7aeff1aaacc7fb9d31f517f4b7791556677c69712c9.jpg)

![d11f026ce41f8887539d66b5a09e4cba0c2f61a5edfe9f0daedef1797a5ee476.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/d11f026ce41f8887539d66b5a09e4cba0c2f61a5edfe9f0daedef1797a5ee476.jpg)

![d8f094089d874f8af3aae72e8dde2a80018d14e8ed3428111ad52749242f9f3f.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/d8f094089d874f8af3aae72e8dde2a80018d14e8ed3428111ad52749242f9f3f.jpg)

![dbc08b13cf6d0d2363331f6fe438d67d4e8b571c16f966ee344a521430aeff0c.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/dbc08b13cf6d0d2363331f6fe438d67d4e8b571c16f966ee344a521430aeff0c.jpg)

![dcb371ff3f7c7983aefcf957891ee8de7d9d819ecdd7084dc7ac886373fd05d7.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/dcb371ff3f7c7983aefcf957891ee8de7d9d819ecdd7084dc7ac886373fd05d7.jpg)

![e8a64347311e82e792ba32de5dd0c07ec2baf39cc952543c754eb059c60885dd.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/e8a64347311e82e792ba32de5dd0c07ec2baf39cc952543c754eb059c60885dd.jpg)

![ee602e41641d24d9ff5ba34f984e0f06bd90df738d6773ef5e0a6f3adaf06f4b.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/ee602e41641d24d9ff5ba34f984e0f06bd90df738d6773ef5e0a6f3adaf06f4b.jpg)

![efd953f61936982cb859411dd41b9f8acaa2b730e416dfb13cead364f7fdf832.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/efd953f61936982cb859411dd41b9f8acaa2b730e416dfb13cead364f7fdf832.jpg)

![f2a9bd59f8979e2ab00c478af0041335a795c3050a96bbc6d80577d73786a2db.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/f2a9bd59f8979e2ab00c478af0041335a795c3050a96bbc6d80577d73786a2db.jpg)

![f9ac62bb2cade3a3b4ba61d127a153d2380eb169c612a43d550a7ef67a819ceb.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/f9ac62bb2cade3a3b4ba61d127a153d2380eb169c612a43d550a7ef67a819ceb.jpg)

![fdf9b5bd030e53112e420484abd5ab3030be73c36dc278ebcbd651f3f49e7310.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/images/fdf9b5bd030e53112e420484abd5ab3030be73c36dc278ebcbd651f3f49e7310.jpg)

### Tables

![0fe627cffbc23b0e8400275b07f1a3628a6955696d95391218f8b9f3066dfc86.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/tables/0fe627cffbc23b0e8400275b07f1a3628a6955696d95391218f8b9f3066dfc86.jpg)

![277fc50606125643916c7a701bb3d9444dcdd3f481ad91d5f0ced348f0159dc9.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/tables/277fc50606125643916c7a701bb3d9444dcdd3f481ad91d5f0ced348f0159dc9.jpg)

![47eedc8e27a2a0554449c2ea27fc6e108b9d50d2a36807ac7772b97ebeea800d.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/tables/47eedc8e27a2a0554449c2ea27fc6e108b9d50d2a36807ac7772b97ebeea800d.jpg)

![6e4f8d14e6f46bd1a1f1afd39f15370832bf0a8caef604749523690a06a40429.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/tables/6e4f8d14e6f46bd1a1f1afd39f15370832bf0a8caef604749523690a06a40429.jpg)

![84dcd60e00ce125b86dca7c4ece30ec232d8d0d61e7778389e2c7b7f924bda27.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/tables/84dcd60e00ce125b86dca7c4ece30ec232d8d0d61e7778389e2c7b7f924bda27.jpg)

![acf7c1f1cb1401c00d2a030796894d4e732f87e055ecb8a57ef017085b9310d4.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/tables/acf7c1f1cb1401c00d2a030796894d4e732f87e055ecb8a57ef017085b9310d4.jpg)

![c967a6f87424e015199dbbd02807614711b2bf448e33bdecd8fc1c0c15de1612.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/tables/c967a6f87424e015199dbbd02807614711b2bf448e33bdecd8fc1c0c15de1612.jpg)

![f820c7010977c6df4bf02583e653c3f20d096eb5cde32d9a555bad43a90e2339.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/tables/f820c7010977c6df4bf02583e653c3f20d096eb5cde32d9a555bad43a90e2339.jpg)

![f8e596d9c8175be039648a78a052d4440fce563a609143a7f073bb6a643fa49e.jpg](../icml_results/221_Guaranteed%20Optimal%20Compositional%20Explanations%20for%20Neurons/tables/f8e596d9c8175be039648a78a052d4440fce563a609143a7f073bb6a643fa49e.jpg)

<a id="paper-0217"></a>

## Simultaneous Speech-to-Speech Translation Without Aligned Data


### Images

![025666151889eec771efe51d6203a12afaa260ad0dcdba28c14b4d340f0cef48.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/images/025666151889eec771efe51d6203a12afaa260ad0dcdba28c14b4d340f0cef48.jpg)

![15c8201a6089db0f4d8159147a2c8e8892b47fed41be3cce2cb2679df2f962d8.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/images/15c8201a6089db0f4d8159147a2c8e8892b47fed41be3cce2cb2679df2f962d8.jpg)

![33f6c11e429ea0638608f59c3bf878a8193abcc4bbe24448b9f9e739805e9a1e.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/images/33f6c11e429ea0638608f59c3bf878a8193abcc4bbe24448b9f9e739805e9a1e.jpg)

![4d5edb4c1bbfcba480e92feb69feb4ba8b1cc6e7d6f3e5136d02e3416ece5fcc.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/images/4d5edb4c1bbfcba480e92feb69feb4ba8b1cc6e7d6f3e5136d02e3416ece5fcc.jpg)

![674edd2ce0c2c8298d88fd7565b767c5dae1f73c52cdd4657f978991c6c12068.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/images/674edd2ce0c2c8298d88fd7565b767c5dae1f73c52cdd4657f978991c6c12068.jpg)

![71ca1d4dc8a02f5559c6dd82ccd3f04030b844f3f3b9f0449b157b2541a9ee2f.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/images/71ca1d4dc8a02f5559c6dd82ccd3f04030b844f3f3b9f0449b157b2541a9ee2f.jpg)

![75a562a5edde6760cd1a0201499635e126957e2a20bdc1bf6cf5f40b6c358932.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/images/75a562a5edde6760cd1a0201499635e126957e2a20bdc1bf6cf5f40b6c358932.jpg)

![7cf51333af3eb48ce3b3dc5e900f0e7e90ad9d990187d68fe3f6e7c0d56c63d9.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/images/7cf51333af3eb48ce3b3dc5e900f0e7e90ad9d990187d68fe3f6e7c0d56c63d9.jpg)

![801fb1ce48da19fca9e1df3b0b71e97c1a03e3114b6fae25442369010bc4c9b4.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/images/801fb1ce48da19fca9e1df3b0b71e97c1a03e3114b6fae25442369010bc4c9b4.jpg)

![921c3ca453abbfa305c9d28ae60b5b2b9f2c8485f074ed6c97fb45e7ac81c576.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/images/921c3ca453abbfa305c9d28ae60b5b2b9f2c8485f074ed6c97fb45e7ac81c576.jpg)

![a84d56994c8a96627737201b1f574d749458debe31e8c1e4b8474293abc72cff.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/images/a84d56994c8a96627737201b1f574d749458debe31e8c1e4b8474293abc72cff.jpg)

![cafa7f8a48f18ed10c30656b3a829d78ecfc7f6e45cefe71edbb82989e071a92.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/images/cafa7f8a48f18ed10c30656b3a829d78ecfc7f6e45cefe71edbb82989e071a92.jpg)

![de5c39b4a5709a7b95727dd552c6ace13e7c50c3af89975d4e1db374d76e51bf.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/images/de5c39b4a5709a7b95727dd552c6ace13e7c50c3af89975d4e1db374d76e51bf.jpg)

![ef656f7f2e04cb0511950cd4a5b3772089d5240a1c5d6118eb9cf1b012856d77.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/images/ef656f7f2e04cb0511950cd4a5b3772089d5240a1c5d6118eb9cf1b012856d77.jpg)

### Tables

![1ed04c2fd1eb7f6178ed0bd62b2ba8dafe95afeb80e16700ef6fe3068bc5c386.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/tables/1ed04c2fd1eb7f6178ed0bd62b2ba8dafe95afeb80e16700ef6fe3068bc5c386.jpg)

![417b244c9762735687f4de10c95fd96f70e3d636d7730b98114053f9f933d081.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/tables/417b244c9762735687f4de10c95fd96f70e3d636d7730b98114053f9f933d081.jpg)

![44f6ee66813751e8d54823e84a49bcd4b06681bd6eca8d4861be24fba15c5ab6.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/tables/44f6ee66813751e8d54823e84a49bcd4b06681bd6eca8d4861be24fba15c5ab6.jpg)

![c2d938b109a5f5c466d8266ac02d7f206ad10392c1aa51229475e7b916fc7781.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/tables/c2d938b109a5f5c466d8266ac02d7f206ad10392c1aa51229475e7b916fc7781.jpg)

![e02777e9ea4073bdfc718af3dc0cef3b3e416ea8df17661868ad55698d980e6e.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/tables/e02777e9ea4073bdfc718af3dc0cef3b3e416ea8df17661868ad55698d980e6e.jpg)

![eac290875aa78d4704820d9be7402913208fd56b06e60333059545a0fa01d63a.jpg](../icml_results/222_Simultaneous%20Speech-to-Speech%20Translation%20Without%20Aligned%20Data/tables/eac290875aa78d4704820d9be7402913208fd56b06e60333059545a0fa01d63a.jpg)

<a id="paper-0218"></a>

## Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models


### Images

![5622ce0021dd6948793cf58df9790f749601dda92c1263c62fbf55e99dd79d47.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/images/5622ce0021dd6948793cf58df9790f749601dda92c1263c62fbf55e99dd79d47.jpg)

![9eadc20e194dbc8132d357c711b9d5121efe9ea7ed8f958b6c04081280743522.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/images/9eadc20e194dbc8132d357c711b9d5121efe9ea7ed8f958b6c04081280743522.jpg)

![a46ac038fc12cb8ff544f54def1abc18f93c6bcc3c7fb1a2e0d4cb66efb90de9.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/images/a46ac038fc12cb8ff544f54def1abc18f93c6bcc3c7fb1a2e0d4cb66efb90de9.jpg)

![c6c10e8f8d1ef94d8cfdf0de291804e64e93250b5939d8cae68cc44a766d20f1.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/images/c6c10e8f8d1ef94d8cfdf0de291804e64e93250b5939d8cae68cc44a766d20f1.jpg)

![d56741b95e22cada252ae376d68792421974d1c897941442808653d2119d33f8.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/images/d56741b95e22cada252ae376d68792421974d1c897941442808653d2119d33f8.jpg)

![d6d1985005413c23b39be3f88975daf328bc122d05bb4c39712f907fe71dca3e.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/images/d6d1985005413c23b39be3f88975daf328bc122d05bb4c39712f907fe71dca3e.jpg)

![d754bce79e340be65fc0f368135a373c74b051aa53cecfa20848cc89f7080f34.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/images/d754bce79e340be65fc0f368135a373c74b051aa53cecfa20848cc89f7080f34.jpg)

![db6c0d34e89106b97e4b10e7e90686e98da3597028ce7a27d7a40f9b85700c20.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/images/db6c0d34e89106b97e4b10e7e90686e98da3597028ce7a27d7a40f9b85700c20.jpg)

![e834f0735b9d2b219e0a8584a6e705d64f1a25e87a081a07568318ad4c08c647.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/images/e834f0735b9d2b219e0a8584a6e705d64f1a25e87a081a07568318ad4c08c647.jpg)

### Tables

![086a39653ff9e68145e460e29670ad554a94eec22df2f3c68f269855b6fb0e95.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/tables/086a39653ff9e68145e460e29670ad554a94eec22df2f3c68f269855b6fb0e95.jpg)

![142067a39351609f3536f334b72776665a533402505ea79c22470f8b248e0318.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/tables/142067a39351609f3536f334b72776665a533402505ea79c22470f8b248e0318.jpg)

![35881430cc56a37cc33dfe2841de428d26ac4bf80f30c48358fd13cf5367676b.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/tables/35881430cc56a37cc33dfe2841de428d26ac4bf80f30c48358fd13cf5367676b.jpg)

![47e10d688b86cb0db04293b467ad196d97c5d1d9d32242966acfbbed847164ce.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/tables/47e10d688b86cb0db04293b467ad196d97c5d1d9d32242966acfbbed847164ce.jpg)

![6ac05af613a7b5e036e9b580f83ba877843ae6980b843d2cbdb5ad3be4e03ed6.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/tables/6ac05af613a7b5e036e9b580f83ba877843ae6980b843d2cbdb5ad3be4e03ed6.jpg)

![6e17a606d9a7d6c9ed237d5b41b635b98d638fc1f6b2f403703fa2cbc40950f2.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/tables/6e17a606d9a7d6c9ed237d5b41b635b98d638fc1f6b2f403703fa2cbc40950f2.jpg)

![704950676f8abb40b846987c85f0eedd9a4abc05968d480524bf75e549d4c425.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/tables/704950676f8abb40b846987c85f0eedd9a4abc05968d480524bf75e549d4c425.jpg)

![81b462c5cf396bbda3ab7749e93b26135e6940584c9ec3c9f417f76ff9bdc4e0.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/tables/81b462c5cf396bbda3ab7749e93b26135e6940584c9ec3c9f417f76ff9bdc4e0.jpg)

![8587283e9bab47b30ed25be334bc39c4e0ad0d2990da9c56f2021b57bf45e003.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/tables/8587283e9bab47b30ed25be334bc39c4e0ad0d2990da9c56f2021b57bf45e003.jpg)

![8dc4e516b36fe498e17985941730c9ebea54ab6e24bc611ed2b6b771a96c8f29.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/tables/8dc4e516b36fe498e17985941730c9ebea54ab6e24bc611ed2b6b771a96c8f29.jpg)

![90bf2105e25b170cfdfa9f6b4f0caf7bb81a6467f7ecd5e4356ec63ac1d22837.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/tables/90bf2105e25b170cfdfa9f6b4f0caf7bb81a6467f7ecd5e4356ec63ac1d22837.jpg)

![913400d5909116203e418b317249e174dbc569cc685791cafc40c05e3813f625.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/tables/913400d5909116203e418b317249e174dbc569cc685791cafc40c05e3813f625.jpg)

![fdab2b03e20e54423f40c6f422c608a9e70a735f3cd0d23d65485d38b7b23d28.jpg](../icml_results/223_Alignment-Guided%20Score%20Matching%20for%20Text-to-Image%20Alignment%20in%20Diffusion%20Models/tables/fdab2b03e20e54423f40c6f422c608a9e70a735f3cd0d23d65485d38b7b23d28.jpg)

<a id="paper-0219"></a>

## Minimax Optimal Strategy for Delayed Observations in Online Reinforcement Learning


### Images

![36c58870d8df619b3e2c45bcf364ec150191b52f6d9ba11adebd1c54cc02d034.jpg](../icml_results/224_Minimax%20Optimal%20Strategy%20for%20Delayed%20Observations%20in%20Online%20Reinforcement%20Learning/images/36c58870d8df619b3e2c45bcf364ec150191b52f6d9ba11adebd1c54cc02d034.jpg)

![3d8620c35c46564f839ab6caeb437a347a053af5f91bad94eb3822c1be65457b.jpg](../icml_results/224_Minimax%20Optimal%20Strategy%20for%20Delayed%20Observations%20in%20Online%20Reinforcement%20Learning/images/3d8620c35c46564f839ab6caeb437a347a053af5f91bad94eb3822c1be65457b.jpg)

![c1c142bc6d8cd505443b91d09fb107febacfdd5db37a6e69569de5d2428b2205.jpg](../icml_results/224_Minimax%20Optimal%20Strategy%20for%20Delayed%20Observations%20in%20Online%20Reinforcement%20Learning/images/c1c142bc6d8cd505443b91d09fb107febacfdd5db37a6e69569de5d2428b2205.jpg)

### Tables

![44fdf3755fdad880ba938be1b24f2080ce93b30b8fe055c3728d3418f81e54ce.jpg](../icml_results/224_Minimax%20Optimal%20Strategy%20for%20Delayed%20Observations%20in%20Online%20Reinforcement%20Learning/tables/44fdf3755fdad880ba938be1b24f2080ce93b30b8fe055c3728d3418f81e54ce.jpg)

![a96ec435f00b715a51342f5f3fd93beef59997073b88ee72d1729fdcc43b5028.jpg](../icml_results/224_Minimax%20Optimal%20Strategy%20for%20Delayed%20Observations%20in%20Online%20Reinforcement%20Learning/tables/a96ec435f00b715a51342f5f3fd93beef59997073b88ee72d1729fdcc43b5028.jpg)

<a id="paper-0220"></a>

## Stabilizing the Q-Gradient Field for Policy Smoothness in Actor-Critic Methods


### Images

![1876b83fca55031b0d372267da1e5c15fdfef86ffaa656f95fcce73c0d057195.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/images/1876b83fca55031b0d372267da1e5c15fdfef86ffaa656f95fcce73c0d057195.jpg)

![85081a5ea848ff229945272e8fb9c10f051b362cdc9e9360f0ab10a7cf3dd871.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/images/85081a5ea848ff229945272e8fb9c10f051b362cdc9e9360f0ab10a7cf3dd871.jpg)

![9da461ab07d667003dbfa97d55c33010f49e1988ca8db040d8079bed10df18dd.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/images/9da461ab07d667003dbfa97d55c33010f49e1988ca8db040d8079bed10df18dd.jpg)

![a17f7f9744d727336c2d5e5a2a3daef874648698b4a501eded645658b3799992.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/images/a17f7f9744d727336c2d5e5a2a3daef874648698b4a501eded645658b3799992.jpg)

![d3368b52459e7df4124e2a9ecadc26cd88ae23e38225f78624d5a7c76dbccedc.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/images/d3368b52459e7df4124e2a9ecadc26cd88ae23e38225f78624d5a7c76dbccedc.jpg)

![d5a1b03be6b2d7c0787e8609664da745094ee295c95324e396d73e671cafbd3c.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/images/d5a1b03be6b2d7c0787e8609664da745094ee295c95324e396d73e671cafbd3c.jpg)

### Tables

![02379fe90a65a9136570fbb92b6243680ff6ba9a9f323cc8701df5856eb96f3d.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/02379fe90a65a9136570fbb92b6243680ff6ba9a9f323cc8701df5856eb96f3d.jpg)

![2c4f695c8b6226a53ca67ee8a4c5011b090c57b359130599ef265b156d539f36.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/2c4f695c8b6226a53ca67ee8a4c5011b090c57b359130599ef265b156d539f36.jpg)

![44fb2d0f85ca1766ac5b482bbbc2be5a439773f43c206842e7b5a0bfa47ab085.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/44fb2d0f85ca1766ac5b482bbbc2be5a439773f43c206842e7b5a0bfa47ab085.jpg)

![48c28eb3aa8ff57aa4aa8d2809886aba0b89f355c2cd765f4499e7885934f725.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/48c28eb3aa8ff57aa4aa8d2809886aba0b89f355c2cd765f4499e7885934f725.jpg)

![49ec7cfd78f31c94e7b130ac54e013fc198581fde92aee009e6298046e177d9a.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/49ec7cfd78f31c94e7b130ac54e013fc198581fde92aee009e6298046e177d9a.jpg)

![4e25e0e1da640a2b576664d02c18e2e397acafa416faf770af45e619855c3fb1.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/4e25e0e1da640a2b576664d02c18e2e397acafa416faf770af45e619855c3fb1.jpg)

![620fb4a7db97e894870c5da34b2e5285528987bdd61a335d951da3173b259549.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/620fb4a7db97e894870c5da34b2e5285528987bdd61a335d951da3173b259549.jpg)

![626d590657cac857fcd4bd7abd5726e571efdb14b985b917a138ae0cc920fdd0.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/626d590657cac857fcd4bd7abd5726e571efdb14b985b917a138ae0cc920fdd0.jpg)

![72cead2daa3b4710d65ee2d15f7137be0d2cc4ab3e6689b70089736de522b952.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/72cead2daa3b4710d65ee2d15f7137be0d2cc4ab3e6689b70089736de522b952.jpg)

![8b998b061b91fa2cdb376c660aa34e4063c0df5cdd3a255a668b2eeb61d7a4b1.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/8b998b061b91fa2cdb376c660aa34e4063c0df5cdd3a255a668b2eeb61d7a4b1.jpg)

![8edad685a2fa63bc7df07b4bbc4448aa8dd180629e222e6736022173fb2210b4.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/8edad685a2fa63bc7df07b4bbc4448aa8dd180629e222e6736022173fb2210b4.jpg)

![a743635c5e016cf1c0effada2ca2c6ee86a12dddd499ce5d9c40188f3c0f320d.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/a743635c5e016cf1c0effada2ca2c6ee86a12dddd499ce5d9c40188f3c0f320d.jpg)

![abdb1e1a56c5ab2de77cd06334decfa30b9fa62d191907a52838ae7c4b7f3c16.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/abdb1e1a56c5ab2de77cd06334decfa30b9fa62d191907a52838ae7c4b7f3c16.jpg)

![b22947859f7bd71aab205e63308cf53313bf6a52682bfca418af709d4c685ba4.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/b22947859f7bd71aab205e63308cf53313bf6a52682bfca418af709d4c685ba4.jpg)

![ba7aad6d8e542fd2caeb39c5a2f19dc7720d392f660274730d10f3415212e70f.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/ba7aad6d8e542fd2caeb39c5a2f19dc7720d392f660274730d10f3415212e70f.jpg)

![c02cfbd9b4920b73a573459c97890af2d65c7c0302e824b780d3e560209deb06.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/c02cfbd9b4920b73a573459c97890af2d65c7c0302e824b780d3e560209deb06.jpg)

![cae81bc41403618d0e22d686e472c1f434022ed0f77a7cc62b84951ae419c8aa.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/cae81bc41403618d0e22d686e472c1f434022ed0f77a7cc62b84951ae419c8aa.jpg)

![cbf882fb01c084da10c47687e5c8e5c0b06b673143e414c797178459f3214d4f.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/cbf882fb01c084da10c47687e5c8e5c0b06b673143e414c797178459f3214d4f.jpg)

![d4340622f5963359e894f1684f2c6fc6308bb2a9b97f1fdc2f3e0c07c802e102.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/d4340622f5963359e894f1684f2c6fc6308bb2a9b97f1fdc2f3e0c07c802e102.jpg)

![e44fc0852a0d88c063059ba15a329298dcc6826a87180444e8e29968bd533026.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/e44fc0852a0d88c063059ba15a329298dcc6826a87180444e8e29968bd533026.jpg)

![f3e387253b2f94464c6f4a179df386a80ac913cd3bfa75dd57aa7e564688aa03.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/f3e387253b2f94464c6f4a179df386a80ac913cd3bfa75dd57aa7e564688aa03.jpg)

![f4a2efe72d615de7d8a4eacfd1bbf8d9d9d3669efe2100e682bf7769af8267ca.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/f4a2efe72d615de7d8a4eacfd1bbf8d9d9d3669efe2100e682bf7769af8267ca.jpg)

![f98dd0041cc7639bbe21f234a62c8f87f51732d84b97c2a753b2caf74a7879e9.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/f98dd0041cc7639bbe21f234a62c8f87f51732d84b97c2a753b2caf74a7879e9.jpg)

![fa59a8d367a144805d53efe855589d37a60245f29e458a67794d3345f6df319a.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/fa59a8d367a144805d53efe855589d37a60245f29e458a67794d3345f6df319a.jpg)

![fb209abf0a1f5722697f2dd9335f4c148018fcda7bb4d5c0431a1734fe69cf56.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/fb209abf0a1f5722697f2dd9335f4c148018fcda7bb4d5c0431a1734fe69cf56.jpg)

![ffc815126df2b803a16a4d1a4fcfa794e172087be59b640c456a7599113bd259.jpg](../icml_results/225_Stabilizing%20the%20Q-Gradient%20Field%20for%20Policy%20Smoothness%20in%20Actor-Critic%20Methods/tables/ffc815126df2b803a16a4d1a4fcfa794e172087be59b640c456a7599113bd259.jpg)

<a id="paper-0221"></a>

## ReQAT: Achieving Full-Precision Reasoning Accuracy with 4-bit Floating-Point Quantization-Aware Training


### Images

![0f4c9883a0da2454edb9d70cb856d0ac866899a7733e6c1af4ea72f84ab26ede.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/images/0f4c9883a0da2454edb9d70cb856d0ac866899a7733e6c1af4ea72f84ab26ede.jpg)

![2ad89128786b087267c39d495726147166fc7d24217774713c28107a55f22a86.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/images/2ad89128786b087267c39d495726147166fc7d24217774713c28107a55f22a86.jpg)

![3353b15ca81428cd80881066601688bfd6e465821bbac46966541e67a15bfec8.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/images/3353b15ca81428cd80881066601688bfd6e465821bbac46966541e67a15bfec8.jpg)

![50aefbb4edf347540a4435c5f4dc842d256c527af90cc3eb8bb9f3becd96d7f9.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/images/50aefbb4edf347540a4435c5f4dc842d256c527af90cc3eb8bb9f3becd96d7f9.jpg)

![650b49842d157ab34449ccf7e9cac7b0d9012e16d40dcd4fa86dcb793ad0a79b.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/images/650b49842d157ab34449ccf7e9cac7b0d9012e16d40dcd4fa86dcb793ad0a79b.jpg)

![7dfd54c198b78f6738ad4606e990bdc42ded64c6d5a00a0c6c0145f3af8ce017.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/images/7dfd54c198b78f6738ad4606e990bdc42ded64c6d5a00a0c6c0145f3af8ce017.jpg)

![9ef179232fa0c854d4dee5e1c88fa2e47685010c0fed7e022e355621e1a724c5.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/images/9ef179232fa0c854d4dee5e1c88fa2e47685010c0fed7e022e355621e1a724c5.jpg)

![b04ece1ffd873559b3e0ca8f40e13385a43c354cd5f01516f383a26f5f47e086.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/images/b04ece1ffd873559b3e0ca8f40e13385a43c354cd5f01516f383a26f5f47e086.jpg)

![b8e46e15f457e8c6eae833652f8ba9b926415c8f6eb3e1fff8b2aa055a867abe.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/images/b8e46e15f457e8c6eae833652f8ba9b926415c8f6eb3e1fff8b2aa055a867abe.jpg)

![dbbf91a85e5007998498c6ec4d6263268fa74585f777d85058ad709248804b46.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/images/dbbf91a85e5007998498c6ec4d6263268fa74585f777d85058ad709248804b46.jpg)

### Tables

![1493648e760fc0e02f605cc083a8804d4227a4903e911204511e4bd02e6cae79.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/1493648e760fc0e02f605cc083a8804d4227a4903e911204511e4bd02e6cae79.jpg)

![251b8013f62f6524dc4e6cabc5a33e4dfec9294c0390a8ac11e5fce469ff5f16.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/251b8013f62f6524dc4e6cabc5a33e4dfec9294c0390a8ac11e5fce469ff5f16.jpg)

![2dcb051c37cb6fc3003bf6c9498a6b44e3e4e4276a39cef953f2b96454c12482.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/2dcb051c37cb6fc3003bf6c9498a6b44e3e4e4276a39cef953f2b96454c12482.jpg)

![3a219e50cb41257064f832dc3ec3ebd47fdf4b0c058fd298ddf98f6e6e973881.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/3a219e50cb41257064f832dc3ec3ebd47fdf4b0c058fd298ddf98f6e6e973881.jpg)

![4d4c613ac64dd2750875ad9c3470d5558a19542b65d3d48ab447a19f0a5aef71.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/4d4c613ac64dd2750875ad9c3470d5558a19542b65d3d48ab447a19f0a5aef71.jpg)

![53215ffe221f14672a301c6037b8ed01304f359e334c120f5a8e2bcefd0195b3.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/53215ffe221f14672a301c6037b8ed01304f359e334c120f5a8e2bcefd0195b3.jpg)

![5f80b084b422be8ceb9c25bc30c9d75a5af887f2c862689d2c94b48617fc85da.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/5f80b084b422be8ceb9c25bc30c9d75a5af887f2c862689d2c94b48617fc85da.jpg)

![6c0632392e2f78557a2b1c8bfdd5981de251726f90250e4a6819de727e2b42ee.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/6c0632392e2f78557a2b1c8bfdd5981de251726f90250e4a6819de727e2b42ee.jpg)

![6c6ab86b4ac57a04b5808c354b021313e64384b1c092942ceb438cc1dd06a942.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/6c6ab86b4ac57a04b5808c354b021313e64384b1c092942ceb438cc1dd06a942.jpg)

![844b5ace21ba8f67b74396033ca8c1586bcf61cba8bd5b6226f14558c00a51cf.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/844b5ace21ba8f67b74396033ca8c1586bcf61cba8bd5b6226f14558c00a51cf.jpg)

![844e9efffb2b1600ec9d2800912328b3299ec60f9106ac52dafb8ae82212a13e.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/844e9efffb2b1600ec9d2800912328b3299ec60f9106ac52dafb8ae82212a13e.jpg)

![9892d66d593ea146184da167511f996fbbaa721169baf9ae65447fe46a249423.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/9892d66d593ea146184da167511f996fbbaa721169baf9ae65447fe46a249423.jpg)

![a424b83027347d046113cbcef5c1a8b7ad6a651d89703872a4642ba4e07a1238.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/a424b83027347d046113cbcef5c1a8b7ad6a651d89703872a4642ba4e07a1238.jpg)

![a53bdd5bff1a42a0d81dd15b451819f383d85649becf3ec052ac71c365143f66.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/a53bdd5bff1a42a0d81dd15b451819f383d85649becf3ec052ac71c365143f66.jpg)

![bcdf80c54583473a44fee3b649796f1d2c966b1358f3357cf1c129a82276aaac.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/bcdf80c54583473a44fee3b649796f1d2c966b1358f3357cf1c129a82276aaac.jpg)

![d537c12e42ec17ebeab0a6329dae834e97a59867ce1feadc161c65ab0038738b.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/d537c12e42ec17ebeab0a6329dae834e97a59867ce1feadc161c65ab0038738b.jpg)

![daab7805ef5606a4a41f150a945bbb74289bf074da12f206e35f1ea096521d9b.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/daab7805ef5606a4a41f150a945bbb74289bf074da12f206e35f1ea096521d9b.jpg)

![e27aa4ec59351fed03d2c4818721bc043d2a5bcf21e9b61ba7afac5c4dc5656e.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/e27aa4ec59351fed03d2c4818721bc043d2a5bcf21e9b61ba7afac5c4dc5656e.jpg)

![f54ea71647e285f21df5c70bf12d21cd5439fc296826799cd398883414d8e381.jpg](../icml_results/226_ReQAT_%20Achieving%20Full-Precision%20Reasoning%20Accuracy%20with%204-bit%20Floating-Point%20Quantization-Aware%20Trai/tables/f54ea71647e285f21df5c70bf12d21cd5439fc296826799cd398883414d8e381.jpg)

<a id="paper-0222"></a>

## FOCUS & RePAIR: Mitigating Text Degeneration via Token-Level Guidance For Pruned Large Language Models


### Images

![063bbef2f573e2fbaf1b7bca0e8795e4a824a9ab0914625fde657659441d6996.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/images/063bbef2f573e2fbaf1b7bca0e8795e4a824a9ab0914625fde657659441d6996.jpg)

![42019b92ca4d07948db2c3574da1458b740b69b3aa09f58e8305c5bddadebdfb.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/images/42019b92ca4d07948db2c3574da1458b740b69b3aa09f58e8305c5bddadebdfb.jpg)

![5c91c112116d4f0e368945312296dc2f8a612bd83d93d9213f8172515424fdf5.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/images/5c91c112116d4f0e368945312296dc2f8a612bd83d93d9213f8172515424fdf5.jpg)

![738baf71755993cd721febc47e18ba05c7017754132c4258bf0f9cb35d53e18e.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/images/738baf71755993cd721febc47e18ba05c7017754132c4258bf0f9cb35d53e18e.jpg)

![a7f4077a3645d09e681756d6710bbecca9496c13933aa1efd37e24391fcd374d.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/images/a7f4077a3645d09e681756d6710bbecca9496c13933aa1efd37e24391fcd374d.jpg)

![b49fc0c858662341c632207fc04aced6b2a733489a1905cfe475db8a7bfbfe39.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/images/b49fc0c858662341c632207fc04aced6b2a733489a1905cfe475db8a7bfbfe39.jpg)

![df6c9e279c107bf95847c985cdf36ea8500414ea918e2dc286aa99595fdd8b65.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/images/df6c9e279c107bf95847c985cdf36ea8500414ea918e2dc286aa99595fdd8b65.jpg)

![ebd1a0948a93a498423bb1cdb9adb4c092baa0bddce4fa0af6d1d4f3bb8604fb.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/images/ebd1a0948a93a498423bb1cdb9adb4c092baa0bddce4fa0af6d1d4f3bb8604fb.jpg)

### Tables

![20f59cf29650ccf41e30cc6522153899242a9da755abdc390cd3c3dcff0b6e36.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/tables/20f59cf29650ccf41e30cc6522153899242a9da755abdc390cd3c3dcff0b6e36.jpg)

![2db5fd9e76e5cd515ad3e86221b37db964c4a20f56960488e8d9b938cac8b2e3.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/tables/2db5fd9e76e5cd515ad3e86221b37db964c4a20f56960488e8d9b938cac8b2e3.jpg)

![34c795daa9e7dc8cce15af78d8f20bdb2a9b4e9cabffb1fb22b33191601eb8c7.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/tables/34c795daa9e7dc8cce15af78d8f20bdb2a9b4e9cabffb1fb22b33191601eb8c7.jpg)

![423756b9e07359c7251a7170515f2c0bacd152822f3a90ddfbc5dcf1f308cdde.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/tables/423756b9e07359c7251a7170515f2c0bacd152822f3a90ddfbc5dcf1f308cdde.jpg)

![69ed11b5c8796b02de5cb8762efbda2edca185be7cd75ea67f51893379065081.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/tables/69ed11b5c8796b02de5cb8762efbda2edca185be7cd75ea67f51893379065081.jpg)

![75ac6abb5904df25412a1db803695053d4f7965670ac70957af7edc7f37c79be.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/tables/75ac6abb5904df25412a1db803695053d4f7965670ac70957af7edc7f37c79be.jpg)

![838003f387e9f66a7d42e28bfd2bd83c0ad4d766a163b002db962239841d394b.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/tables/838003f387e9f66a7d42e28bfd2bd83c0ad4d766a163b002db962239841d394b.jpg)

![8aae9f33f355f2cb988d9794b32eef88876fc5d7407ac0c3b88d7b669597036f.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/tables/8aae9f33f355f2cb988d9794b32eef88876fc5d7407ac0c3b88d7b669597036f.jpg)

![8caf8ab05a4740098ee41fa5c30de9fc21b5b5d323fe1790bba59fdd9d54e9a0.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/tables/8caf8ab05a4740098ee41fa5c30de9fc21b5b5d323fe1790bba59fdd9d54e9a0.jpg)

![8da377c76bffc97760ecf915361942cc2c4ead2c0e7abc3260f5edb67a566231.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/tables/8da377c76bffc97760ecf915361942cc2c4ead2c0e7abc3260f5edb67a566231.jpg)

![aafc8d68d2ddaa2e973cc7bde88c5ec7d7ddfb58fca7580decb9332786f4de5d.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/tables/aafc8d68d2ddaa2e973cc7bde88c5ec7d7ddfb58fca7580decb9332786f4de5d.jpg)

![b22d042943eff639e65d241fc30ab929d5754872c115a581ba2a422aaaf5afda.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/tables/b22d042943eff639e65d241fc30ab929d5754872c115a581ba2a422aaaf5afda.jpg)

![b3674cc05531c18e91278328c5119338597f488d7072978a11b48b738b944e11.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/tables/b3674cc05531c18e91278328c5119338597f488d7072978a11b48b738b944e11.jpg)

![e9fc945a560177ab267d734754ca4feab6c0651503e78a37862f26e706aef371.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/tables/e9fc945a560177ab267d734754ca4feab6c0651503e78a37862f26e706aef371.jpg)

![fddc2fa82478189d66e0e0dbd21d1c47d4b28c080b3583598dfc6055cdd553d9.jpg](../icml_results/227_FOCUS%20%26%20RePAIR_%20Mitigating%20Text%20Degeneration%20via%20Token-Level%20Guidance%20For%20Pruned%20Large%20Language%20Mode/tables/fddc2fa82478189d66e0e0dbd21d1c47d4b28c080b3583598dfc6055cdd553d9.jpg)

<a id="paper-0223"></a>

## Root Cause Analysis of Failures in Microservices via Bayesian Root Cause Discovery


### Images

![35ced6e0ae847aad5024e82e83e43da5e5c7c08da3f14b8dcf204f822e969b21.jpg](../icml_results/228_Root%20Cause%20Analysis%20of%20Failures%20in%20Microservices%20via%20Bayesian%20Root%20Cause%20Discovery/images/35ced6e0ae847aad5024e82e83e43da5e5c7c08da3f14b8dcf204f822e969b21.jpg)

![72a6accdfca4de2d8f0d49419e75327c961023ccc60a21408e9eb500e9586728.jpg](../icml_results/228_Root%20Cause%20Analysis%20of%20Failures%20in%20Microservices%20via%20Bayesian%20Root%20Cause%20Discovery/images/72a6accdfca4de2d8f0d49419e75327c961023ccc60a21408e9eb500e9586728.jpg)

![a0e06c423391a9c79038e7dc54bdc9b3d31272e954bdd5e1e9f5cb8e546b65e6.jpg](../icml_results/228_Root%20Cause%20Analysis%20of%20Failures%20in%20Microservices%20via%20Bayesian%20Root%20Cause%20Discovery/images/a0e06c423391a9c79038e7dc54bdc9b3d31272e954bdd5e1e9f5cb8e546b65e6.jpg)

![a69e07e4d44d422f5442f6236713f512c27ce31100bc2e47ae7b5b0f1eb03d8c.jpg](../icml_results/228_Root%20Cause%20Analysis%20of%20Failures%20in%20Microservices%20via%20Bayesian%20Root%20Cause%20Discovery/images/a69e07e4d44d422f5442f6236713f512c27ce31100bc2e47ae7b5b0f1eb03d8c.jpg)

![b1d4b8a5105d288d4f0cb892967ce041fe5deae587ca31c750ca8c7648158b4b.jpg](../icml_results/228_Root%20Cause%20Analysis%20of%20Failures%20in%20Microservices%20via%20Bayesian%20Root%20Cause%20Discovery/images/b1d4b8a5105d288d4f0cb892967ce041fe5deae587ca31c750ca8c7648158b4b.jpg)

![d5bbd1e20b156fd960170e97a58dbe1c9b6c2fa5e9bcb5697d364d5cc7edab01.jpg](../icml_results/228_Root%20Cause%20Analysis%20of%20Failures%20in%20Microservices%20via%20Bayesian%20Root%20Cause%20Discovery/images/d5bbd1e20b156fd960170e97a58dbe1c9b6c2fa5e9bcb5697d364d5cc7edab01.jpg)

### Tables

![1d426bf1d6ba2086c85cdc68b64b9d1909efc18b34c6494fe83de65b9bd3e3d8.jpg](../icml_results/228_Root%20Cause%20Analysis%20of%20Failures%20in%20Microservices%20via%20Bayesian%20Root%20Cause%20Discovery/tables/1d426bf1d6ba2086c85cdc68b64b9d1909efc18b34c6494fe83de65b9bd3e3d8.jpg)

![3db0c9c0e627ea79bf0a854be9d285cfefcdb67d086d747445a6b42f673f9187.jpg](../icml_results/228_Root%20Cause%20Analysis%20of%20Failures%20in%20Microservices%20via%20Bayesian%20Root%20Cause%20Discovery/tables/3db0c9c0e627ea79bf0a854be9d285cfefcdb67d086d747445a6b42f673f9187.jpg)

![85a38107a24d22b281da39edc6d7f4ed5307d8878254c1015638ac4e5fd65935.jpg](../icml_results/228_Root%20Cause%20Analysis%20of%20Failures%20in%20Microservices%20via%20Bayesian%20Root%20Cause%20Discovery/tables/85a38107a24d22b281da39edc6d7f4ed5307d8878254c1015638ac4e5fd65935.jpg)

![8ffc674e6cc5ebf8ae6160ad0d1756c1b47a683c632e77cf2c0320df0d7b4131.jpg](../icml_results/228_Root%20Cause%20Analysis%20of%20Failures%20in%20Microservices%20via%20Bayesian%20Root%20Cause%20Discovery/tables/8ffc674e6cc5ebf8ae6160ad0d1756c1b47a683c632e77cf2c0320df0d7b4131.jpg)

![99b8f40a9f2750ccc510b32ad7c2d605fd939299a4e409d5bff2fa50630922f4.jpg](../icml_results/228_Root%20Cause%20Analysis%20of%20Failures%20in%20Microservices%20via%20Bayesian%20Root%20Cause%20Discovery/tables/99b8f40a9f2750ccc510b32ad7c2d605fd939299a4e409d5bff2fa50630922f4.jpg)

![ad8ea18c6541b7f8dce1946ce7d8aac1a46a1f327b1b3ba8e8e37abfbd270fa4.jpg](../icml_results/228_Root%20Cause%20Analysis%20of%20Failures%20in%20Microservices%20via%20Bayesian%20Root%20Cause%20Discovery/tables/ad8ea18c6541b7f8dce1946ce7d8aac1a46a1f327b1b3ba8e8e37abfbd270fa4.jpg)

![c6f3207cc579bd6f7ed1fd19d2c24b1f4161d79904e1fa79ea830f453a8875da.jpg](../icml_results/228_Root%20Cause%20Analysis%20of%20Failures%20in%20Microservices%20via%20Bayesian%20Root%20Cause%20Discovery/tables/c6f3207cc579bd6f7ed1fd19d2c24b1f4161d79904e1fa79ea830f453a8875da.jpg)

![f28375c17a9a4d0e12a2748f55dab41d07cc43f9df66c1101a4974ab30794718.jpg](../icml_results/228_Root%20Cause%20Analysis%20of%20Failures%20in%20Microservices%20via%20Bayesian%20Root%20Cause%20Discovery/tables/f28375c17a9a4d0e12a2748f55dab41d07cc43f9df66c1101a4974ab30794718.jpg)

<a id="paper-0224"></a>

## Near-Optimal Private Linear Regression via Iterative Hessian Mixing


### Images

![095cb53b0a33b841df83a1e37b3474d9f8d5b44f6688884dddf255a7e7c0bb8e.jpg](../icml_results/229_Near-Optimal%20Private%20Linear%20Regression%20via%20Iterative%20Hessian%20Mixing/images/095cb53b0a33b841df83a1e37b3474d9f8d5b44f6688884dddf255a7e7c0bb8e.jpg)

![14c6314e6e352b7dfe5f3c84b7bda3499cdb0b3f3db6c67dbdc17008471f4bb7.jpg](../icml_results/229_Near-Optimal%20Private%20Linear%20Regression%20via%20Iterative%20Hessian%20Mixing/images/14c6314e6e352b7dfe5f3c84b7bda3499cdb0b3f3db6c67dbdc17008471f4bb7.jpg)

![1f17229c50093ef98e6d196a4460b092c465c803d62a65db523a298d63c55934.jpg](../icml_results/229_Near-Optimal%20Private%20Linear%20Regression%20via%20Iterative%20Hessian%20Mixing/images/1f17229c50093ef98e6d196a4460b092c465c803d62a65db523a298d63c55934.jpg)

![64524dbbb70330f55f9e63a5cbc227cb810d6614221a3a4edc06d5cff64be9af.jpg](../icml_results/229_Near-Optimal%20Private%20Linear%20Regression%20via%20Iterative%20Hessian%20Mixing/images/64524dbbb70330f55f9e63a5cbc227cb810d6614221a3a4edc06d5cff64be9af.jpg)

![742f63d52521c82571a75d82685e245b92136351705918fd98093b2e53ccc7ea.jpg](../icml_results/229_Near-Optimal%20Private%20Linear%20Regression%20via%20Iterative%20Hessian%20Mixing/images/742f63d52521c82571a75d82685e245b92136351705918fd98093b2e53ccc7ea.jpg)

![ad9c8c17fd22664d12099598553c35845c594925107787253c2601cc27e9919f.jpg](../icml_results/229_Near-Optimal%20Private%20Linear%20Regression%20via%20Iterative%20Hessian%20Mixing/images/ad9c8c17fd22664d12099598553c35845c594925107787253c2601cc27e9919f.jpg)

![cb3009a36bc74744a7bd51185c16ab576f930901b6444ec62ca4d2ca5ccdd497.jpg](../icml_results/229_Near-Optimal%20Private%20Linear%20Regression%20via%20Iterative%20Hessian%20Mixing/images/cb3009a36bc74744a7bd51185c16ab576f930901b6444ec62ca4d2ca5ccdd497.jpg)

![e8350e459156e67df9abec90ae5fc8315451d76976496b3fc0dc64721092c8af.jpg](../icml_results/229_Near-Optimal%20Private%20Linear%20Regression%20via%20Iterative%20Hessian%20Mixing/images/e8350e459156e67df9abec90ae5fc8315451d76976496b3fc0dc64721092c8af.jpg)

![ee751817c2810b8a22cd15587f1eb9e37afc7807d706e1e9851c9edc08f49d81.jpg](../icml_results/229_Near-Optimal%20Private%20Linear%20Regression%20via%20Iterative%20Hessian%20Mixing/images/ee751817c2810b8a22cd15587f1eb9e37afc7807d706e1e9851c9edc08f49d81.jpg)

### Tables

![aba86529b75b04646460588ad0520f9a0a6169f59d929e8423279da9db0cf067.jpg](../icml_results/229_Near-Optimal%20Private%20Linear%20Regression%20via%20Iterative%20Hessian%20Mixing/tables/aba86529b75b04646460588ad0520f9a0a6169f59d929e8423279da9db0cf067.jpg)

![b851247fe2cde5881ed92fa43c020f364c578e48d8a7deeee16dfeaa9bbd23d6.jpg](../icml_results/229_Near-Optimal%20Private%20Linear%20Regression%20via%20Iterative%20Hessian%20Mixing/tables/b851247fe2cde5881ed92fa43c020f364c578e48d8a7deeee16dfeaa9bbd23d6.jpg)

![e010421f7552a881f5c6c24c4f9258e9a1860d2358ab734c0e487ad22257180c.jpg](../icml_results/229_Near-Optimal%20Private%20Linear%20Regression%20via%20Iterative%20Hessian%20Mixing/tables/e010421f7552a881f5c6c24c4f9258e9a1860d2358ab734c0e487ad22257180c.jpg)

<a id="paper-0225"></a>

## MetaphorVU: Towards Metaphorical Video Understanding


### Images

![0811b006b4c304cff43697f50887c6dc8c2be3d64918829eb54f2096a873c589.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/0811b006b4c304cff43697f50887c6dc8c2be3d64918829eb54f2096a873c589.jpg)

![089f0244e5abcd3ea4db1928cc8c5edbcca33f4e8d2ffc6c236194bfc63891b6.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/089f0244e5abcd3ea4db1928cc8c5edbcca33f4e8d2ffc6c236194bfc63891b6.jpg)

![08ddad490ea776a0258d5480bbb18c67ffa385f24bb34ee505f26e0c9addbc91.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/08ddad490ea776a0258d5480bbb18c67ffa385f24bb34ee505f26e0c9addbc91.jpg)

![1027fbbbde8fe3a35bfe505ea05bc10fa11bb8379f758e662fa42fa161fa5c29.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/1027fbbbde8fe3a35bfe505ea05bc10fa11bb8379f758e662fa42fa161fa5c29.jpg)

![21db89c5a0bc7dd819a6a4c1b6def003a4ce6f51208628ae1641267901dc2072.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/21db89c5a0bc7dd819a6a4c1b6def003a4ce6f51208628ae1641267901dc2072.jpg)

![26973d6beb955ce4af702b4310b81bf690f36a888c5c0225c74db966880b023e.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/26973d6beb955ce4af702b4310b81bf690f36a888c5c0225c74db966880b023e.jpg)

![32d899b4b6d4a71aa99d3734affe8b6ebc54ef3cd834d262eb0e671d9a588610.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/32d899b4b6d4a71aa99d3734affe8b6ebc54ef3cd834d262eb0e671d9a588610.jpg)

![3f562a324fe9c223274ecb70d94fc33b0394bf5934531016235701ba22a2942a.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/3f562a324fe9c223274ecb70d94fc33b0394bf5934531016235701ba22a2942a.jpg)

![48f1c71bb12f87e26a68512b79d89c440f098cc4903ddc398cd2289ed09d843b.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/48f1c71bb12f87e26a68512b79d89c440f098cc4903ddc398cd2289ed09d843b.jpg)

![4e24ddafabf15b19cc67972f0771c3d6dc8f6e499b497f7f7c60876d0f97b4ff.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/4e24ddafabf15b19cc67972f0771c3d6dc8f6e499b497f7f7c60876d0f97b4ff.jpg)

![522d9d6a8aa31216b3524d791ba1e185b5bf0655d76a124df81955ee02b506d6.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/522d9d6a8aa31216b3524d791ba1e185b5bf0655d76a124df81955ee02b506d6.jpg)

![59d1e4b2c9d4ada9efe8411bd1d9e81e6481be1dae02a4bd420231cb512df419.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/59d1e4b2c9d4ada9efe8411bd1d9e81e6481be1dae02a4bd420231cb512df419.jpg)

![6239c9ee67fdfcdf0a25ba18c0eb35bdd59d8aae16c6c1d1a66426dc13d9a9bd.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/6239c9ee67fdfcdf0a25ba18c0eb35bdd59d8aae16c6c1d1a66426dc13d9a9bd.jpg)

![62a2dea0f4b4eb38f08c5d721b8baab4daf585564dca1c3ad965f1b4dfe2715b.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/62a2dea0f4b4eb38f08c5d721b8baab4daf585564dca1c3ad965f1b4dfe2715b.jpg)

![6fe986c96c3b1d1148a77f9fe5aebad3c63f429fff1abdf77ede154a09244a5d.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/6fe986c96c3b1d1148a77f9fe5aebad3c63f429fff1abdf77ede154a09244a5d.jpg)

![81be70e89bd62b62f2b43f5d20c309b8e315a06808ead0e1e3bb5880989b329a.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/81be70e89bd62b62f2b43f5d20c309b8e315a06808ead0e1e3bb5880989b329a.jpg)

![8c8973734a5d854eb9163f05867cb94b11fd8d851be9557f976e624048dde33f.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/8c8973734a5d854eb9163f05867cb94b11fd8d851be9557f976e624048dde33f.jpg)

![9661f03c57d8e54357bfc8d4357f9cf3892a3c53df104c94928e0901a2056e5c.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/9661f03c57d8e54357bfc8d4357f9cf3892a3c53df104c94928e0901a2056e5c.jpg)

![984f7654556c11a99cc427c9a31b6eed983b75bfd29bd2b990207536b1306d7f.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/984f7654556c11a99cc427c9a31b6eed983b75bfd29bd2b990207536b1306d7f.jpg)

![9bfafd9708d313d54b468132038fa95478cefc98abe1edbb72b1845c32fde4ea.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/9bfafd9708d313d54b468132038fa95478cefc98abe1edbb72b1845c32fde4ea.jpg)

![a7e5e055c5ed6490ffd19bd572318af21c046b4c91106632758adf54468be727.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/a7e5e055c5ed6490ffd19bd572318af21c046b4c91106632758adf54468be727.jpg)

![bf36f3588aadaf97ab9f117e0d6474bdd2bbed7e879bbfb2fa86e22bea7a3674.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/bf36f3588aadaf97ab9f117e0d6474bdd2bbed7e879bbfb2fa86e22bea7a3674.jpg)

![c688913dffa327edebd7c666bcce9dc0e228656aeda593ce1da5e7d383dad33a.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/c688913dffa327edebd7c666bcce9dc0e228656aeda593ce1da5e7d383dad33a.jpg)

![c69ff20b1adc7086513fe46b9238007d236dbe0ecb5c6b1d359d779a7b6b3078.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/c69ff20b1adc7086513fe46b9238007d236dbe0ecb5c6b1d359d779a7b6b3078.jpg)

![cd8c3a0de15aa67f512e35f5edb2226f0cc0f4f86ce98720ef36faebc2ef894d.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/cd8c3a0de15aa67f512e35f5edb2226f0cc0f4f86ce98720ef36faebc2ef894d.jpg)

![dd9f133cdde7fff320a5da4dcdf1d04fde7afd35802f90d3094fb6912aadc798.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/dd9f133cdde7fff320a5da4dcdf1d04fde7afd35802f90d3094fb6912aadc798.jpg)

![e2a076d93ecbdd7a327174f543f69ffc42591a71606d710347e5a2c003667e98.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/e2a076d93ecbdd7a327174f543f69ffc42591a71606d710347e5a2c003667e98.jpg)

![e7b9ca681a7f0434330c1fe31180b7241aee30bdc40b8e39f0f68d6b0e8a7820.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/e7b9ca681a7f0434330c1fe31180b7241aee30bdc40b8e39f0f68d6b0e8a7820.jpg)

![edc4451a4c13cc5192fe06a301205b5f8f167be0b2d954a386b74cc91688b38f.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/edc4451a4c13cc5192fe06a301205b5f8f167be0b2d954a386b74cc91688b38f.jpg)

![f08ebebdffdafaf495749535b2cd6ddd26cd7d12a94d87d44d00484afc5dddc8.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/f08ebebdffdafaf495749535b2cd6ddd26cd7d12a94d87d44d00484afc5dddc8.jpg)

![f6fb2fc7c06a67a29e31d7516da1e5a12e2822e5bcf14b741b2f8cb7fddedf52.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/images/f6fb2fc7c06a67a29e31d7516da1e5a12e2822e5bcf14b741b2f8cb7fddedf52.jpg)

### Tables

![33b44d34908ff5c2f68406002e32220fea62ca5b592ad22f822fe073d47223ff.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/tables/33b44d34908ff5c2f68406002e32220fea62ca5b592ad22f822fe073d47223ff.jpg)

![4612dd092c31600404a5d4f131dcef7c5dacaa4b7490ec20cd4a2c495bd25553.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/tables/4612dd092c31600404a5d4f131dcef7c5dacaa4b7490ec20cd4a2c495bd25553.jpg)

![69824654b2458a5b4f730f6944ea531df977a77c43c3e71f1ac40364794e729b.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/tables/69824654b2458a5b4f730f6944ea531df977a77c43c3e71f1ac40364794e729b.jpg)

![6ab24d574d2b2904099b635726952ba0f5c0de8b692db2f63e0e747304603147.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/tables/6ab24d574d2b2904099b635726952ba0f5c0de8b692db2f63e0e747304603147.jpg)

![7b962dda36e1346430ee78135e574f9cbcd6b9a934742d309de54667bfd83ef6.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/tables/7b962dda36e1346430ee78135e574f9cbcd6b9a934742d309de54667bfd83ef6.jpg)

![f27ba13f86ddf124cdc3c4bb014945281889de2013ea08861e7668c6684e2794.jpg](../icml_results/230_MetaphorVU_%20Towards%20Metaphorical%20Video%20Understanding/tables/f27ba13f86ddf124cdc3c4bb014945281889de2013ea08861e7668c6684e2794.jpg)

<a id="paper-0226"></a>

## DroneDINO: Towards Heterogeneous Routed Mixture of Experts for Drone-based Unified Object Detection


### Images

![0672155881083ad81a4970a0293e09955697b7672dcf03a0cbd30e1ae5681928.jpg](../icml_results/231_DroneDINO_%20Towards%20Heterogeneous%20Routed%20Mixture%20of%20Experts%20for%20Drone-based%20Unified%20Object%20Detection/images/0672155881083ad81a4970a0293e09955697b7672dcf03a0cbd30e1ae5681928.jpg)

![64707ecdcc3d0a67dc6b7b7e9ceeb9e6aa1f5868a8fc3f5281a8feadf69d4b58.jpg](../icml_results/231_DroneDINO_%20Towards%20Heterogeneous%20Routed%20Mixture%20of%20Experts%20for%20Drone-based%20Unified%20Object%20Detection/images/64707ecdcc3d0a67dc6b7b7e9ceeb9e6aa1f5868a8fc3f5281a8feadf69d4b58.jpg)

![7375c4e9e95fdbf357630f82ed2b4c9f4441ab19bef508df88f1921a28f93dfd.jpg](../icml_results/231_DroneDINO_%20Towards%20Heterogeneous%20Routed%20Mixture%20of%20Experts%20for%20Drone-based%20Unified%20Object%20Detection/images/7375c4e9e95fdbf357630f82ed2b4c9f4441ab19bef508df88f1921a28f93dfd.jpg)

![acf6cb933b7b6f4f1ea8dae44ad942c87514ad0e0942595e3ce36d781f0593ad.jpg](../icml_results/231_DroneDINO_%20Towards%20Heterogeneous%20Routed%20Mixture%20of%20Experts%20for%20Drone-based%20Unified%20Object%20Detection/images/acf6cb933b7b6f4f1ea8dae44ad942c87514ad0e0942595e3ce36d781f0593ad.jpg)

### Tables

![1148427d162adccb341f93064282712d51cb16e116f1887a2f63f48527ada305.jpg](../icml_results/231_DroneDINO_%20Towards%20Heterogeneous%20Routed%20Mixture%20of%20Experts%20for%20Drone-based%20Unified%20Object%20Detection/tables/1148427d162adccb341f93064282712d51cb16e116f1887a2f63f48527ada305.jpg)

![4a08856b3a1cef3b9bc5bbca0657bef5559c183421ff872e9fc47987c109cf97.jpg](../icml_results/231_DroneDINO_%20Towards%20Heterogeneous%20Routed%20Mixture%20of%20Experts%20for%20Drone-based%20Unified%20Object%20Detection/tables/4a08856b3a1cef3b9bc5bbca0657bef5559c183421ff872e9fc47987c109cf97.jpg)

![52294ec2e23d9ad29895467ac759431941389887539d1e3a3e3d543d67bb7937.jpg](../icml_results/231_DroneDINO_%20Towards%20Heterogeneous%20Routed%20Mixture%20of%20Experts%20for%20Drone-based%20Unified%20Object%20Detection/tables/52294ec2e23d9ad29895467ac759431941389887539d1e3a3e3d543d67bb7937.jpg)

![8c205705909f05ed749b7946726c99bc7f9695e9d80a9d269942cb4117463dfd.jpg](../icml_results/231_DroneDINO_%20Towards%20Heterogeneous%20Routed%20Mixture%20of%20Experts%20for%20Drone-based%20Unified%20Object%20Detection/tables/8c205705909f05ed749b7946726c99bc7f9695e9d80a9d269942cb4117463dfd.jpg)

![d3981d7ebd2a8412ab5410d381117bce8fe1858c5fef4bccbc6858d7fb3cbcdc.jpg](../icml_results/231_DroneDINO_%20Towards%20Heterogeneous%20Routed%20Mixture%20of%20Experts%20for%20Drone-based%20Unified%20Object%20Detection/tables/d3981d7ebd2a8412ab5410d381117bce8fe1858c5fef4bccbc6858d7fb3cbcdc.jpg)

![d66aaa5750dfb42d17ca8be9e251a06747a0d319e3f47fd7b28b40ac194003c0.jpg](../icml_results/231_DroneDINO_%20Towards%20Heterogeneous%20Routed%20Mixture%20of%20Experts%20for%20Drone-based%20Unified%20Object%20Detection/tables/d66aaa5750dfb42d17ca8be9e251a06747a0d319e3f47fd7b28b40ac194003c0.jpg)

![da9e49bde156afaf76bb65784c4a989adfe6b06c400f15c7b85a02e6c98881d2.jpg](../icml_results/231_DroneDINO_%20Towards%20Heterogeneous%20Routed%20Mixture%20of%20Experts%20for%20Drone-based%20Unified%20Object%20Detection/tables/da9e49bde156afaf76bb65784c4a989adfe6b06c400f15c7b85a02e6c98881d2.jpg)

<a id="paper-0227"></a>

## SAW-Bench: Learning Situated Awareness in the Real World


### Images

![23f524a43d503b6143fe870078861f288bc5867cf934738f98b0b999f74ee7d0.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/23f524a43d503b6143fe870078861f288bc5867cf934738f98b0b999f74ee7d0.jpg)

![2d6b20b3c063cd5d751dcdeb8dd919fb9b1fea190dc6a9935cb788836430758a.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/2d6b20b3c063cd5d751dcdeb8dd919fb9b1fea190dc6a9935cb788836430758a.jpg)

![3c54a431b4d852eb70ae8cbfb782f3149d75aa3ef4b59f38c74a25d045a0ed36.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/3c54a431b4d852eb70ae8cbfb782f3149d75aa3ef4b59f38c74a25d045a0ed36.jpg)

![4071cdf37af4f1a7f1db8a98aa8705c85fa31a19975bb5554fc1ba8c90a0bfe9.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/4071cdf37af4f1a7f1db8a98aa8705c85fa31a19975bb5554fc1ba8c90a0bfe9.jpg)

![42c9223394870485a9383a863e7f5eef2512abe20d9712b04f63b3bd1e8cb341.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/42c9223394870485a9383a863e7f5eef2512abe20d9712b04f63b3bd1e8cb341.jpg)

![4553256da549d9f0a321fecfe735d168f5592ad0255f5e09384ea6bb07956e93.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/4553256da549d9f0a321fecfe735d168f5592ad0255f5e09384ea6bb07956e93.jpg)

![474e95f0fdde8f05805d192c6877a81c86dae719de14d73e88c98eedadba3a1d.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/474e95f0fdde8f05805d192c6877a81c86dae719de14d73e88c98eedadba3a1d.jpg)

![55671a0923104b6a4149f61eed8e8fb9ac2386b848ef0a4c529367d333749dd8.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/55671a0923104b6a4149f61eed8e8fb9ac2386b848ef0a4c529367d333749dd8.jpg)

![57276c1fa677848438143abc3351737ffb1cd5c601492e04007fccd3346abf75.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/57276c1fa677848438143abc3351737ffb1cd5c601492e04007fccd3346abf75.jpg)

![844c23ed99c99b9ac5dc5d83fa048468fe30eddd063a2c7e5f7360344978d9a3.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/844c23ed99c99b9ac5dc5d83fa048468fe30eddd063a2c7e5f7360344978d9a3.jpg)

![8792d065b199546b9fb1784bad87bfcd12854aea8b747e3519e0381143ba0317.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/8792d065b199546b9fb1784bad87bfcd12854aea8b747e3519e0381143ba0317.jpg)

![87ebddaff743d0a43b8cf1a12985aa503f5ff866faaa440d0ab4b38cd77c9cdb.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/87ebddaff743d0a43b8cf1a12985aa503f5ff866faaa440d0ab4b38cd77c9cdb.jpg)

![8e089cada493664c1082f10fab941f09e8d6c6b05bec85158d0e5d65d910d38c.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/8e089cada493664c1082f10fab941f09e8d6c6b05bec85158d0e5d65d910d38c.jpg)

![a58fb5ff9e2f90cdddfb7003e8aaa649b6f566fa1f35ef18a1ed9be8a74dbcd4.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/a58fb5ff9e2f90cdddfb7003e8aaa649b6f566fa1f35ef18a1ed9be8a74dbcd4.jpg)

![b48eb223915b107432cc964090872825c2eaf812cdcf3a76f0e722b13a4f7946.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/b48eb223915b107432cc964090872825c2eaf812cdcf3a76f0e722b13a4f7946.jpg)

![b7fc2a4931e70ee4a281302f55d7dc98efe35046852453047e7d03748b793970.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/b7fc2a4931e70ee4a281302f55d7dc98efe35046852453047e7d03748b793970.jpg)

![bb7430a96c77b51f10ef1ec5062463a4f27c0e223264e636bb1ca53835e29e8b.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/bb7430a96c77b51f10ef1ec5062463a4f27c0e223264e636bb1ca53835e29e8b.jpg)

![c2e9addbd998bad78f7566042b61c5b15a455656f1062a5001f159cdddd611a0.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/c2e9addbd998bad78f7566042b61c5b15a455656f1062a5001f159cdddd611a0.jpg)

![c3ad01841de1cdb0254a3e94061d1ccf9b7a568d0d567fae408264e0066379dd.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/c3ad01841de1cdb0254a3e94061d1ccf9b7a568d0d567fae408264e0066379dd.jpg)

![c49c7f4938734c3604f93729f516503601b2a1b50d1e44159e187398d5b35a99.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/c49c7f4938734c3604f93729f516503601b2a1b50d1e44159e187398d5b35a99.jpg)

![c71217ae4ca7c1a78a1241c811a0bce580be6717164fc33221415f583cf49fa8.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/c71217ae4ca7c1a78a1241c811a0bce580be6717164fc33221415f583cf49fa8.jpg)

![c8d2365a1dac01d0b0804ca174e80bf96106afa6d92801f14d5fe24c807f3c78.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/c8d2365a1dac01d0b0804ca174e80bf96106afa6d92801f14d5fe24c807f3c78.jpg)

![cb04909be01348bdd7a048bb178ec1ea23ee4836b8931f0cf265fea2bbd60b38.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/cb04909be01348bdd7a048bb178ec1ea23ee4836b8931f0cf265fea2bbd60b38.jpg)

![cbb0a7aa6590166cf9556dee1e76064993d1c25d25f6c5aa72b0c0780919df37.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/cbb0a7aa6590166cf9556dee1e76064993d1c25d25f6c5aa72b0c0780919df37.jpg)

![daeb3f97fb872e5ccd5c4de69a81ea87b59af901dbd4fac013f5ed9a63b2aa26.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/daeb3f97fb872e5ccd5c4de69a81ea87b59af901dbd4fac013f5ed9a63b2aa26.jpg)

![dca8e3be6e620aa9d807e345aea9ba2ab002398532b1127949dba603afc0dcb5.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/dca8e3be6e620aa9d807e345aea9ba2ab002398532b1127949dba603afc0dcb5.jpg)

![eb81215abfa1aa5386448530e96a442e3b15cb024712d5625e2b6cde7bfa3b13.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/eb81215abfa1aa5386448530e96a442e3b15cb024712d5625e2b6cde7bfa3b13.jpg)

![f5b8f0419cea3c1ab00f9367be14b63ab85b9493c85a85ffa07936270a83a9f3.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/f5b8f0419cea3c1ab00f9367be14b63ab85b9493c85a85ffa07936270a83a9f3.jpg)

![f80515a7fcb6906ba4bbcdd4e3f876c974e0a645a4f313a4221fc98cac47339b.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/f80515a7fcb6906ba4bbcdd4e3f876c974e0a645a4f313a4221fc98cac47339b.jpg)

![f991636f1c0952985f9a8ed3165651c299f633a02eda4498e5df355158425e6e.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/f991636f1c0952985f9a8ed3165651c299f633a02eda4498e5df355158425e6e.jpg)

![fa099d5933f9590b3d5a5740de385b603de08cde2649542c1a4d4440490d92c3.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/images/fa099d5933f9590b3d5a5740de385b603de08cde2649542c1a4d4440490d92c3.jpg)

### Tables

![2981bcef1eac4ab9f651654a6000bc704d671124f055925306aeeadb15ae394b.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/tables/2981bcef1eac4ab9f651654a6000bc704d671124f055925306aeeadb15ae394b.jpg)

![3cba301b6c97bd7e33c9402abe347f38f245d0bab2b5c9d025555e0b87be82da.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/tables/3cba301b6c97bd7e33c9402abe347f38f245d0bab2b5c9d025555e0b87be82da.jpg)

![485cc77c785a051982b083e880684ad321f923098c7299564beb672eb7e3eb3a.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/tables/485cc77c785a051982b083e880684ad321f923098c7299564beb672eb7e3eb3a.jpg)

![5159140ed7f8577e389bb75c8f24e97ac0dfd6eabe5732ae7f6b15a2d3ebde06.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/tables/5159140ed7f8577e389bb75c8f24e97ac0dfd6eabe5732ae7f6b15a2d3ebde06.jpg)

![5cfc5e16d9759a6b30aec252ca1e7304f664bfe679447f628d95b41a2767c7a8.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/tables/5cfc5e16d9759a6b30aec252ca1e7304f664bfe679447f628d95b41a2767c7a8.jpg)

![6fc6af1402a06cab83097ea8b8823adca94fc99728a09997675e4760d35325e6.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/tables/6fc6af1402a06cab83097ea8b8823adca94fc99728a09997675e4760d35325e6.jpg)

![89e5dc1efbd257f058d11765618edad7b98bf42486421feec4c6f05d4054fa5e.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/tables/89e5dc1efbd257f058d11765618edad7b98bf42486421feec4c6f05d4054fa5e.jpg)

![8cc6ba188acf72fa87804804d39f368eca982f560e9110781d42ef4fb5a7411b.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/tables/8cc6ba188acf72fa87804804d39f368eca982f560e9110781d42ef4fb5a7411b.jpg)

![a4c5739114ada66eae01a13be24d9f8c0681b7f14a4eccf0423e12176a8c7d1f.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/tables/a4c5739114ada66eae01a13be24d9f8c0681b7f14a4eccf0423e12176a8c7d1f.jpg)

![c4e9ccbbed7b178b68bda3198e135f485501875a6f0263e5fac216d9803d7d92.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/tables/c4e9ccbbed7b178b68bda3198e135f485501875a6f0263e5fac216d9803d7d92.jpg)

![cecea0c117abd677e584f26a9f0fa1e3831b2efae318f1fbc23b42bded49509b.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/tables/cecea0c117abd677e584f26a9f0fa1e3831b2efae318f1fbc23b42bded49509b.jpg)

![dbed1296a8eaa28a6b2cfa0ba86553746e65cf4977cb65c4efdfc1e1bef72f6b.jpg](../icml_results/232_SAW-Bench_%20Learning%20Situated%20Awareness%20in%20the%20Real%20World/tables/dbed1296a8eaa28a6b2cfa0ba86553746e65cf4977cb65c4efdfc1e1bef72f6b.jpg)

<a id="paper-0228"></a>

## OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration


### Images

![0688499d7a5d0c4f9baafbd6382f644a9664de16b0def22e1f5a486a634e7a0f.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/images/0688499d7a5d0c4f9baafbd6382f644a9664de16b0def22e1f5a486a634e7a0f.jpg)

![236b9a383ec3ce1012bc4bcf41489011362882c07f4c29eb12087df5063ca8c1.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/images/236b9a383ec3ce1012bc4bcf41489011362882c07f4c29eb12087df5063ca8c1.jpg)

![39e83c4f974efbd704dcde022d43f6ab1a46832d42969f95aedc311327e787e6.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/images/39e83c4f974efbd704dcde022d43f6ab1a46832d42969f95aedc311327e787e6.jpg)

![65e8058296b75106fb4eb96786849bb1a5c4da1c79157ee10fdf313b47f5f449.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/images/65e8058296b75106fb4eb96786849bb1a5c4da1c79157ee10fdf313b47f5f449.jpg)

![961c1387991fe5b4aea7976af84dae15b5d9bb0453f4eefe1c5b03217128f562.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/images/961c1387991fe5b4aea7976af84dae15b5d9bb0453f4eefe1c5b03217128f562.jpg)

![a14ecb77950ed928396984ddb870315855ad67b385872f95e6037bcadaeeb552.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/images/a14ecb77950ed928396984ddb870315855ad67b385872f95e6037bcadaeeb552.jpg)

![d8e039c92bb74d00f0a43f634c646b767fa300529d9aa3ba0ddd0afae5139a29.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/images/d8e039c92bb74d00f0a43f634c646b767fa300529d9aa3ba0ddd0afae5139a29.jpg)

![e12036fd472cf52e3e21d7710644f647235390aa36a9fc736e654748fbf0db11.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/images/e12036fd472cf52e3e21d7710644f647235390aa36a9fc736e654748fbf0db11.jpg)

### Tables

![10688ec98267032d51356d8d4e908febe4f7580387ee9a9066dee96cfdefcc2f.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/10688ec98267032d51356d8d4e908febe4f7580387ee9a9066dee96cfdefcc2f.jpg)

![16c34deb4805bc585a208b4ba6d2de264493776e75b9139ff1de1b977c274562.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/16c34deb4805bc585a208b4ba6d2de264493776e75b9139ff1de1b977c274562.jpg)

![1a84c9edfa7ec2e2a31082c25177eb191d86f2b331460eede5e1cefd69d52006.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/1a84c9edfa7ec2e2a31082c25177eb191d86f2b331460eede5e1cefd69d52006.jpg)

![21ac73d1f76716f57bede499b2932ed9b5f6452320aa5151990a2915cac94d6d.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/21ac73d1f76716f57bede499b2932ed9b5f6452320aa5151990a2915cac94d6d.jpg)

![2c222a0e100f439bf4e84f2caaaf6c671fa0321a2331b5e6f0e7f1dfd3ebe9c8.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/2c222a0e100f439bf4e84f2caaaf6c671fa0321a2331b5e6f0e7f1dfd3ebe9c8.jpg)

![2d0e053d3ca213cd55e464242339594eeee8f38f9fb9cdd429e6e37c2a09c1d6.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/2d0e053d3ca213cd55e464242339594eeee8f38f9fb9cdd429e6e37c2a09c1d6.jpg)

![38164a4112557cc34b1d526c617b6e9d622c8bd1905265f46e9e781caf2b27dd.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/38164a4112557cc34b1d526c617b6e9d622c8bd1905265f46e9e781caf2b27dd.jpg)

![394298e2f53a03dcb94e6473dcbda5fd2dd3e02820db6bfc06d950987473646d.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/394298e2f53a03dcb94e6473dcbda5fd2dd3e02820db6bfc06d950987473646d.jpg)

![394abb811da8c0df82f87302864d20fb0213f100843c20b3b11d63df31603352.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/394abb811da8c0df82f87302864d20fb0213f100843c20b3b11d63df31603352.jpg)

![43bf4624cdb3ec4e019ab792a18a753a57707d135ed0445bde5f24a8bd6d1f67.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/43bf4624cdb3ec4e019ab792a18a753a57707d135ed0445bde5f24a8bd6d1f67.jpg)

![5e5a7553d045a9b1788d53ad43639d835427bb3f36465520ed10fc739ac83140.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/5e5a7553d045a9b1788d53ad43639d835427bb3f36465520ed10fc739ac83140.jpg)

![749a83a2e2765c3f252f34a824460772fa91d3eb7932a85e2b02d92ffb93bf5a.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/749a83a2e2765c3f252f34a824460772fa91d3eb7932a85e2b02d92ffb93bf5a.jpg)

![7a887b9c07aeae263ebb79b580af505dedccfd447aa720e9f7902211827ec071.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/7a887b9c07aeae263ebb79b580af505dedccfd447aa720e9f7902211827ec071.jpg)

![7a8ca46e2557abf89ca97558ff546653681bc3d93851d8005e4073ca0163ebab.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/7a8ca46e2557abf89ca97558ff546653681bc3d93851d8005e4073ca0163ebab.jpg)

![7b1f2044e6a10c7c6b3beb3594db5736c23e9b5bdbd9609b2d6d413314ae7237.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/7b1f2044e6a10c7c6b3beb3594db5736c23e9b5bdbd9609b2d6d413314ae7237.jpg)

![8db46d3fb450d584983b16f51ae3da9d894e80a251db37f14edd0c6060c39696.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/8db46d3fb450d584983b16f51ae3da9d894e80a251db37f14edd0c6060c39696.jpg)

![915485d12b03658c54d372955f943a263aba850c3f077ad40402ddbd4b310637.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/915485d12b03658c54d372955f943a263aba850c3f077ad40402ddbd4b310637.jpg)

![9da6cce36ae8ab53e2c09c561f29427083709045f448b8bc2750e21e83fc1472.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/9da6cce36ae8ab53e2c09c561f29427083709045f448b8bc2750e21e83fc1472.jpg)

![b4f71e2828a8b7e859297852e7e18bbb923a93baf042d2b41ce783d99d8aaa6b.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/b4f71e2828a8b7e859297852e7e18bbb923a93baf042d2b41ce783d99d8aaa6b.jpg)

![c88da389de8c35e42ef6f1fdbcd25d7903b34160361199d2f4a04c4ea69a98f5.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/c88da389de8c35e42ef6f1fdbcd25d7903b34160361199d2f4a04c4ea69a98f5.jpg)

![cbd6c6af214176d9fb2dcfb5e7cc733d4bca4918f6f5bc94e22d477edec51dd6.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/cbd6c6af214176d9fb2dcfb5e7cc733d4bca4918f6f5bc94e22d477edec51dd6.jpg)

![d3f587d399ba93c4b13e37e9ee6e9d4cb6cc37141ee6e30915a9d42f9aac6fb3.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/d3f587d399ba93c4b13e37e9ee6e9d4cb6cc37141ee6e30915a9d42f9aac6fb3.jpg)

![e23dc32546a4f84904e7df56fe4ce3123c22061a5072ae0621d217c815793e82.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/e23dc32546a4f84904e7df56fe4ce3123c22061a5072ae0621d217c815793e82.jpg)

![f5eaf23b4aaa87432187640629ed3ffcdc71ce2350a78213d752ee50633b8778.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/f5eaf23b4aaa87432187640629ed3ffcdc71ce2350a78213d752ee50633b8778.jpg)

![f7c46e8474207d27957a61210adbbcf882522cfec7ea9db9b357892a77c05692.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/f7c46e8474207d27957a61210adbbcf882522cfec7ea9db9b357892a77c05692.jpg)

![f98e195e77f7ae446bfa7ed55d79335dc5c7c16ee91043503d41b4d94c39986e.jpg](../icml_results/233_OMAC_%20A%20Holistic%20Optimization%20Framework%20for%20LLM-Based%20Multi-Agent%20Collaboration/tables/f98e195e77f7ae446bfa7ed55d79335dc5c7c16ee91043503d41b4d94c39986e.jpg)

<a id="paper-0229"></a>

## What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity


### Images

![17014ac79b8c3eb98a12d3a19e7ab79e9e01532d7229df59539ffe031c359090.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/images/17014ac79b8c3eb98a12d3a19e7ab79e9e01532d7229df59539ffe031c359090.jpg)

![1a5f28d2ad57f2e82d2995fe03bb62e3c7ce819950d4f6f8ed1b04bccace1526.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/images/1a5f28d2ad57f2e82d2995fe03bb62e3c7ce819950d4f6f8ed1b04bccace1526.jpg)

![27821c74c6136c09f569c38e4fd98d8c89a6603793aa9694ab3f8e3c72f871a4.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/images/27821c74c6136c09f569c38e4fd98d8c89a6603793aa9694ab3f8e3c72f871a4.jpg)

![34bcab62f2f3ed97b9833a768e9b39a65efe210fc90594603e3856ac750fc256.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/images/34bcab62f2f3ed97b9833a768e9b39a65efe210fc90594603e3856ac750fc256.jpg)

![537c388ed48bdfc023661c19c4b62f420ca153268d114d723943c3840788bb84.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/images/537c388ed48bdfc023661c19c4b62f420ca153268d114d723943c3840788bb84.jpg)

![a27ee7cc069d175ce9ae442694f2e2d0bb9f11107f4e7f41ef8c2a4cc607b3b3.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/images/a27ee7cc069d175ce9ae442694f2e2d0bb9f11107f4e7f41ef8c2a4cc607b3b3.jpg)

![d68aa6ddeb8d966f933b150c09502973ff297fa37aa9020911bab93ea2b9d37a.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/images/d68aa6ddeb8d966f933b150c09502973ff297fa37aa9020911bab93ea2b9d37a.jpg)

### Tables

![121111ccd45f5ac011ef399cf979daf2706add7e41191ae8a6a566e042ace6ac.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/121111ccd45f5ac011ef399cf979daf2706add7e41191ae8a6a566e042ace6ac.jpg)

![1221bc4052d4c199e83c79989cdb3e81e88c30ea9b5d16a00e4fbe79c6d2a6b7.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/1221bc4052d4c199e83c79989cdb3e81e88c30ea9b5d16a00e4fbe79c6d2a6b7.jpg)

![1e61553df8f86da84acb218909f4549fa300723530220b9977ede1dfb1ba95f0.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/1e61553df8f86da84acb218909f4549fa300723530220b9977ede1dfb1ba95f0.jpg)

![1fc662ecb8ea1743896d4743b8e8261456d702aeaf09ff7f71b888bf0136a955.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/1fc662ecb8ea1743896d4743b8e8261456d702aeaf09ff7f71b888bf0136a955.jpg)

![45f189fe0d4c82e0c40347e13d465ef7f933aa8d49b597485c8b73c901301b04.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/45f189fe0d4c82e0c40347e13d465ef7f933aa8d49b597485c8b73c901301b04.jpg)

![51513f67fc25e21cacde80085c10d880ea5decf446a13560cc8cb1ada7daafd1.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/51513f67fc25e21cacde80085c10d880ea5decf446a13560cc8cb1ada7daafd1.jpg)

![61b2370208e20e65e51de3f9ec4baf6bcf8eaed42c6770fa03d07086dc7cd9c8.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/61b2370208e20e65e51de3f9ec4baf6bcf8eaed42c6770fa03d07086dc7cd9c8.jpg)

![676dc98f5f7890bcdf6eb918e66a26903dbee83a63fe180f419e5a77d3ff8e9f.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/676dc98f5f7890bcdf6eb918e66a26903dbee83a63fe180f419e5a77d3ff8e9f.jpg)

![6ce7be4a0aea4ab2f140a943b505e815c8437b2fce4903016d10cd77a87b7219.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/6ce7be4a0aea4ab2f140a943b505e815c8437b2fce4903016d10cd77a87b7219.jpg)

![84ec4700a7cb514f7d2d1ee608692c0270c28e18e7dd1d9c06f6cfaecf65c823.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/84ec4700a7cb514f7d2d1ee608692c0270c28e18e7dd1d9c06f6cfaecf65c823.jpg)

![8500c9118a404222a325fdbdc2d04d576ee1951010e9aa929b3f422104d16e35.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/8500c9118a404222a325fdbdc2d04d576ee1951010e9aa929b3f422104d16e35.jpg)

![874e31df1b6b860d52c82fc6d267c55379587e92ef492b8a8509287b6f6261df.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/874e31df1b6b860d52c82fc6d267c55379587e92ef492b8a8509287b6f6261df.jpg)

![a59b9c85d1b4abc9087c7ade58e99fa287e2542507f17d643140fbfec23fbd46.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/a59b9c85d1b4abc9087c7ade58e99fa287e2542507f17d643140fbfec23fbd46.jpg)

![b4305aeea4c1040ec093be81cd7ed4b028240881e4816b58ade34cbc37d8307d.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/b4305aeea4c1040ec093be81cd7ed4b028240881e4816b58ade34cbc37d8307d.jpg)

![b566268d83fd413197cc315dc3a1575ac8b156b827c7ae6411a6473e849a89f6.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/b566268d83fd413197cc315dc3a1575ac8b156b827c7ae6411a6473e849a89f6.jpg)

![b82ae82e1847fe9dcffea9b28110bbbc8d6773dfba7747bd7cbdd93b1ded6327.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/b82ae82e1847fe9dcffea9b28110bbbc8d6773dfba7747bd7cbdd93b1ded6327.jpg)

![ca85bcdb5790e1413c9ed0779b707364595e2f38c8330e68aaa0f6ebbfdd0fd8.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/ca85bcdb5790e1413c9ed0779b707364595e2f38c8330e68aaa0f6ebbfdd0fd8.jpg)

![ed617240b41e9df6d0278618f5971e87c0314b2f469dcb2169616bc88260d7f8.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/ed617240b41e9df6d0278618f5971e87c0314b2f469dcb2169616bc88260d7f8.jpg)

![f7069cec45c8e288e2c061f6c8d49e1889e55fcc1ec587ed16d51a21ac06b098.jpg](../icml_results/234_What%20You%20Think%20is%20What%20You%20See_%20Driving%20Exploration%20in%20VLM%20Agents%20via%20Visual-Linguistic%20Curiosity/tables/f7069cec45c8e288e2c061f6c8d49e1889e55fcc1ec587ed16d51a21ac06b098.jpg)

<a id="paper-0230"></a>

## A Kinetic Energy Perspective of Flow Matching


### Images

![15578d0bac115c9d01f8334807565a2651af1f7ff5983371a37f3d19c63a2d5c.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/images/15578d0bac115c9d01f8334807565a2651af1f7ff5983371a37f3d19c63a2d5c.jpg)

![1d79b71c40e7d6a547ca5a9e728ead20b079dec92fbfe825511fc68beebb15c7.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/images/1d79b71c40e7d6a547ca5a9e728ead20b079dec92fbfe825511fc68beebb15c7.jpg)

![500b282b560a05b506f1fcf75cdcf7137f7c784bb0c942c3f65ffe98311bee87.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/images/500b282b560a05b506f1fcf75cdcf7137f7c784bb0c942c3f65ffe98311bee87.jpg)

![550f59bd1d8b3c8843cb6b1ba431217764f4d06dbaaefc0509b0fe9baf46ca14.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/images/550f59bd1d8b3c8843cb6b1ba431217764f4d06dbaaefc0509b0fe9baf46ca14.jpg)

![56f399db062ee3236c5fef2201927e3a9f48901f09755735748606712e4bcbb6.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/images/56f399db062ee3236c5fef2201927e3a9f48901f09755735748606712e4bcbb6.jpg)

![5d6d888075795ba1788212fe9a45f9b80b27c8d1f41a7fe7444a090c7e1d5d31.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/images/5d6d888075795ba1788212fe9a45f9b80b27c8d1f41a7fe7444a090c7e1d5d31.jpg)

![6849065b38d57ded9cdd24467d3ccc6ae9ae4e8fc9857952f0d74fc9216894dd.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/images/6849065b38d57ded9cdd24467d3ccc6ae9ae4e8fc9857952f0d74fc9216894dd.jpg)

![6cdfc5a9fa4c64fe232803de4c1a31d20cf1f53ae3c75606754ce563309396fb.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/images/6cdfc5a9fa4c64fe232803de4c1a31d20cf1f53ae3c75606754ce563309396fb.jpg)

![89a434849e1b2d84a11dfde211613ae37a868dc22b72331784d314bb5b0a4d31.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/images/89a434849e1b2d84a11dfde211613ae37a868dc22b72331784d314bb5b0a4d31.jpg)

![8c5b8dabc99ca998c5d176c330d265994deaacaa32ac65387bef5ff8b961de0c.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/images/8c5b8dabc99ca998c5d176c330d265994deaacaa32ac65387bef5ff8b961de0c.jpg)

![950da32f200026494474b82c36bb14b9b19872e89aca1c594e40b5c71289dc9b.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/images/950da32f200026494474b82c36bb14b9b19872e89aca1c594e40b5c71289dc9b.jpg)

![ae12b80ed4cd00275759f94868e4e05b3de4aec8a16fdbaa02d6ab77ca724ed4.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/images/ae12b80ed4cd00275759f94868e4e05b3de4aec8a16fdbaa02d6ab77ca724ed4.jpg)

![af30eb079cb021bd585664f788496ab66b31125c60d9e523950cea8c3ae28350.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/images/af30eb079cb021bd585664f788496ab66b31125c60d9e523950cea8c3ae28350.jpg)

![ba34b4a53a197c63735516a05d8180512f2c282d92aea0be16b34a1283f00c05.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/images/ba34b4a53a197c63735516a05d8180512f2c282d92aea0be16b34a1283f00c05.jpg)

![da08d7bf109835f220a35c72d17852d9709ddfd5edca5a253fa5de279cd7039e.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/images/da08d7bf109835f220a35c72d17852d9709ddfd5edca5a253fa5de279cd7039e.jpg)

![ec1b7633f72e071937b0196cc6ab5971a8e42cfd9e85bd1d2fd841318ed807a5.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/images/ec1b7633f72e071937b0196cc6ab5971a8e42cfd9e85bd1d2fd841318ed807a5.jpg)

![fec640ddd9e744b746b10e63e8fccf7f62f0a4e1e7022600b192f57a07336567.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/images/fec640ddd9e744b746b10e63e8fccf7f62f0a4e1e7022600b192f57a07336567.jpg)

### Tables

![09de82ced04318242e06d78b064fcdb4821ec6b155b6bd51c05fec73cf37da50.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/tables/09de82ced04318242e06d78b064fcdb4821ec6b155b6bd51c05fec73cf37da50.jpg)

![301d8bdc99f40c48fb6f55c80b185c5ef5e51db4781048704a5c301bd0c7e4bc.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/tables/301d8bdc99f40c48fb6f55c80b185c5ef5e51db4781048704a5c301bd0c7e4bc.jpg)

![50fa40862013ab61eb3bd2f2fe217cba06807c3a4a8e015ff3bf42263fc69723.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/tables/50fa40862013ab61eb3bd2f2fe217cba06807c3a4a8e015ff3bf42263fc69723.jpg)

![56f681f737d828f8c25d3004ca436230e51411d34ea8aaca825218379e2ed7d3.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/tables/56f681f737d828f8c25d3004ca436230e51411d34ea8aaca825218379e2ed7d3.jpg)

![6a474b21ba8de91a72641262cb51f158a1326a85bb684a518a8c8ba99030be52.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/tables/6a474b21ba8de91a72641262cb51f158a1326a85bb684a518a8c8ba99030be52.jpg)

![84158481746bb4e19e995f68e7c4512bc9a22583016efc1699ada9003d3c3455.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/tables/84158481746bb4e19e995f68e7c4512bc9a22583016efc1699ada9003d3c3455.jpg)

![9be1569e4c6e6831a4a6d73298d35920ffc241cbed99e8c8c559e8aab6146fd8.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/tables/9be1569e4c6e6831a4a6d73298d35920ffc241cbed99e8c8c559e8aab6146fd8.jpg)

![b23717753f2cf7b915bbdc0b06cea58ec3638093a64d796add260642dbbbbca6.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/tables/b23717753f2cf7b915bbdc0b06cea58ec3638093a64d796add260642dbbbbca6.jpg)

![bb572a403032b5973c35503e087a838f9dedf4c6045e5c56fd14467d37af4c72.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/tables/bb572a403032b5973c35503e087a838f9dedf4c6045e5c56fd14467d37af4c72.jpg)

![c6ee0abe7458cd6f51f265ad7c9d125d7cca013fa335819b95b83139dc99f0c1.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/tables/c6ee0abe7458cd6f51f265ad7c9d125d7cca013fa335819b95b83139dc99f0c1.jpg)

![d4ca95fff8a44cef6b75834d7d3a6817a75509adaa43c33e876a3661a6e47beb.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/tables/d4ca95fff8a44cef6b75834d7d3a6817a75509adaa43c33e876a3661a6e47beb.jpg)

![f812c16ce50d8770c54d3faef6c4d503320070fe0869f43b6282778d02c59e67.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/tables/f812c16ce50d8770c54d3faef6c4d503320070fe0869f43b6282778d02c59e67.jpg)

![f9a09becb5f1d8d865b6a6a08f0907abe4e91b235908f846e471dc2fe2e8cdd1.jpg](../icml_results/235_A%20Kinetic%20Energy%20Perspective%20of%20Flow%20Matching/tables/f9a09becb5f1d8d865b6a6a08f0907abe4e91b235908f846e471dc2fe2e8cdd1.jpg)

<a id="paper-0231"></a>

## FIDIA: Function-Informed Sequence Design via Inference-Aligned Policy Optimization

### Images

![36631bd34f100947209d9d65a7c8ede0f5a95949bf68196045c13be39ddcfb18.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/images/36631bd34f100947209d9d65a7c8ede0f5a95949bf68196045c13be39ddcfb18.jpg)

![3e9f49a77611d4eb19478399e54f51c599d5f0980ed115a9e8d46f284da11674.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/images/3e9f49a77611d4eb19478399e54f51c599d5f0980ed115a9e8d46f284da11674.jpg)

![721e4812011bd2f1adb1aba2aa6b57ea76e45fb88a4aaa2d410297ee28b78e7a.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/images/721e4812011bd2f1adb1aba2aa6b57ea76e45fb88a4aaa2d410297ee28b78e7a.jpg)

### Tables

![08b32e7e781d6cdc6fb23ae234ae57e9d5d861657ddc30721af991596f41093e.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/tables/08b32e7e781d6cdc6fb23ae234ae57e9d5d861657ddc30721af991596f41093e.jpg)

![243e5d8c520ad15b5d412d485c13df745524d0517bf7bd2537769d22b09c228b.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/tables/243e5d8c520ad15b5d412d485c13df745524d0517bf7bd2537769d22b09c228b.jpg)

![2cc08817404f8b743e9349925813b5059023b5cdc30536f12acfad0c1f3e86fb.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/tables/2cc08817404f8b743e9349925813b5059023b5cdc30536f12acfad0c1f3e86fb.jpg)

![7338817da293acaa8ce57e3deeb9c29eb879714062bb2048e4ee40601069a3e5.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/tables/7338817da293acaa8ce57e3deeb9c29eb879714062bb2048e4ee40601069a3e5.jpg)

![882e040f867452c85393733f7b6f18e2ff1880776e0e243c4f119d3763608c19.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/tables/882e040f867452c85393733f7b6f18e2ff1880776e0e243c4f119d3763608c19.jpg)

![a9b5c9f0a71eee53264d22f7fda4a3d0277034672487ae31c2421688c6ac37cc.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/tables/a9b5c9f0a71eee53264d22f7fda4a3d0277034672487ae31c2421688c6ac37cc.jpg)

![e8ce5e59ae329de9c156ee5999e1e65323f6ae2bbb228cf215a5e995fc35626f.jpg](../icml_results/236_FIDIA_%20Function-Informed%20Sequence%20Design%20via%20Inference-Aligned%20Policy%20Optimization/tables/e8ce5e59ae329de9c156ee5999e1e65323f6ae2bbb228cf215a5e995fc35626f.jpg)
