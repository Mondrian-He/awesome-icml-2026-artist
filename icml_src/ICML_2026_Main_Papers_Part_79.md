# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 79 of 197

## 目录 (Table of Contents)

1. [Scout Before You Attend: Sketch-and-Walk Sparse Attention for Efficient LLM Inference](#paper-2575)
2. [PISCES: Annotation-free Text-to-Video Post-Training via Optimal Transport-Aligned Rewards](#paper-2576)
3. [Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function](#paper-2577)
4. [FAFO: Lossy KV Cache Compression for Lossless Inference Acceleration via Draftless Fumble Decoding](#paper-2578)
5. [Can I Have Your Order? Monte-Carlo Tree Search for Slot Filling Ordering in Diffusion Language Models](#paper-2579)
6. [Path-conditioned training: a principled way to rescale ReLU neural networks](#paper-2580)
7. [When Does Adaptation Win? Scaling Laws for Meta-Learning in Quantum Control](#paper-2581)
8. [Policy Search via Bayesian Optimization with Temporal Difference Gaussian Processes](#paper-2582)
9. [The Catastrophic Failure of *the* k-Means Algorithm in High Dimensions, and How Hartigan’s Algorithm Avoids It](#paper-2583)
10. [LDARNet: DNA Adaptive Representation Network with Learnable Tokenization for Genomic Modeling](#paper-2584)
11. [GENEB: Why Genomic Models Are Hard to Compare](#paper-2585)
12. [Query Lens: Interpreting Sparse Key-Value Features with Indirect Effects](#paper-2586)
13. [Decomposing Query-Key Feature Interactions Using Contrastive Covariances](#paper-2587)
14. [SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant LLM Pretraining Systems with 100k+ GPUs](#paper-2588)
15. [Dustin: Draft-Augmented Sparse Verification for Efficient Long-Context Generation with Speculative Decoding](#paper-2589)
16. [CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large Vision-Language Models](#paper-2590)
17. [ReGen: Hierarchical Multi-Prompt Representation Generation for Efficient Waveform Diffusion Models](#paper-2591)
18. [Toward Understanding Adversarial Distillation: Why Robust Teachers Fail](#paper-2592)
19. [Dynamic Relational Priming Improves Transformer in Multivariate Time Series](#paper-2593)
20. [DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention](#paper-2594)
21. [Stochastic Sparse Attention for Memory-Bound Inference](#paper-2595)
22. [Variable-Length Tokenization via Learnable Global Merging for Diffusion Transformers](#paper-2596)
23. [Focusing Where Vision Matters: Selective Training for Large Vision Language Models via Visual Information Gain](#paper-2597)
24. [On the Collapse of Generative Paths: A Criterion and Correction for Diffusion Steering](#paper-2598)
25. [HInT: Hypergraph Infusion at the Structural Layers Improves Table Understanding](#paper-2599)
26. [Decision-Focused Learning via Tangent-Space Projection of Prediction Error](#paper-2600)
27. [SAGE: Shaping Anchors for Guided Exploration in RLVR of LLMs](#paper-2601)
28. [Interaction-Breaking Adversarial Learning Framework for Robust Multi-Agent Reinforcement Learning](#paper-2602)
29. [Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](#paper-2603)
30. [From Volume to Value: Preference-Aligned Memory Construction for On-Device RAG](#paper-2604)
31. [BIT-LLM: Brain Instruction Tuned LLM with persistent Cross-Attention for fMRI-to-Text Decoding](#paper-2605)
32. [Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy](#paper-2606)
33. [Geometry-Aware Image Flow Matching](#paper-2607)

---


<a id="paper-2575"></a>

## Scout Before You Attend: Sketch-and-Walk Sparse Attention for Efficient LLM Inference

### Images

![195aacc37356b68cb4163262f1d6fcd5c32b0a20d9cf4ea5a9cd3bbf74c3c443.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/images/195aacc37356b68cb4163262f1d6fcd5c32b0a20d9cf4ea5a9cd3bbf74c3c443.jpg)

![9cafeb4da0a1586d04bb94aed447ea9653eb964ad68aa0ee154569334058fcc8.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/images/9cafeb4da0a1586d04bb94aed447ea9653eb964ad68aa0ee154569334058fcc8.jpg)

![d549172c9b28e3defd1b6041383f1d9e7f6f8c97e0b94b197e916af4f3e6f6b3.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/images/d549172c9b28e3defd1b6041383f1d9e7f6f8c97e0b94b197e916af4f3e6f6b3.jpg)

![dcd9b9d583d4e294de5325ab5e2387ee0e4ead0e0e0f46d27f6b15d203f3afc4.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/images/dcd9b9d583d4e294de5325ab5e2387ee0e4ead0e0e0f46d27f6b15d203f3afc4.jpg)

![efc24939a168259c2e7ff60769523be10974afc82add095b048fca017267e6fa.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/images/efc24939a168259c2e7ff60769523be10974afc82add095b048fca017267e6fa.jpg)

### Tables

![05a59777a1231fc5771c9778f8078920e83756577adc09854151e805b8e487ee.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/05a59777a1231fc5771c9778f8078920e83756577adc09854151e805b8e487ee.jpg)

![2bd23d250614c1b37b30725d69b4958a36783a7a51414f495f86cfde47cf4290.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/2bd23d250614c1b37b30725d69b4958a36783a7a51414f495f86cfde47cf4290.jpg)

![37389204f816c189bc5bf9978e93395e0697105c8df5709f1e924deab162ad20.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/37389204f816c189bc5bf9978e93395e0697105c8df5709f1e924deab162ad20.jpg)

![458879bc3a855f148fd4886901a40bd6ddc4fed9fdfef84f41dc9a3b2a023b71.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/458879bc3a855f148fd4886901a40bd6ddc4fed9fdfef84f41dc9a3b2a023b71.jpg)

![4856fed97035a1ca4537c74c9042caf6c951e7ef6c36ffdcf015529f531e7a09.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/4856fed97035a1ca4537c74c9042caf6c951e7ef6c36ffdcf015529f531e7a09.jpg)

![647d20ed80f9f634a09884e4e1c3d3b07faa5f7fd200560b544a212c726600dc.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/647d20ed80f9f634a09884e4e1c3d3b07faa5f7fd200560b544a212c726600dc.jpg)

![6b6c38d6efec4de0437df9c579c535a1af2342e4ea15e763f97dc4c5d4714a0e.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/6b6c38d6efec4de0437df9c579c535a1af2342e4ea15e763f97dc4c5d4714a0e.jpg)

![72c0c64b8ecee1953d41890016c9de44d407f9a9c468b5fed98187e5f619e209.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/72c0c64b8ecee1953d41890016c9de44d407f9a9c468b5fed98187e5f619e209.jpg)

![741fe895769cdd246ad8beaf689350e0d1d8cdc5ab97c41ae1c6d772af695d14.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/741fe895769cdd246ad8beaf689350e0d1d8cdc5ab97c41ae1c6d772af695d14.jpg)

![7e73b76756f18bc3c41b5351da47c94e4715d3042cf9118e9bf21b60eb1f3dfc.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/7e73b76756f18bc3c41b5351da47c94e4715d3042cf9118e9bf21b60eb1f3dfc.jpg)

![81a9859c6d2d6c985a99daeabb96a236c905067b07f9148cb896841d49fa9ecd.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/81a9859c6d2d6c985a99daeabb96a236c905067b07f9148cb896841d49fa9ecd.jpg)

![a7a039059411b40a9761b053d38c146808fab74d9ecdcd40efef9c062d2eb26f.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/a7a039059411b40a9761b053d38c146808fab74d9ecdcd40efef9c062d2eb26f.jpg)

![b9f28a16e00a7d543dba8c703ba3b1e90ff5c69f654854320256904df983f873.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/b9f28a16e00a7d543dba8c703ba3b1e90ff5c69f654854320256904df983f873.jpg)

![bd11f2afda6bf76e0e89f006c7abd9961d357ce9c52415485cdaf98ee37149c5.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/bd11f2afda6bf76e0e89f006c7abd9961d357ce9c52415485cdaf98ee37149c5.jpg)

![c946cc6f246737a28b9ecefabbbb7b7457952221d32de4ca88c28e170fc220f9.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/c946cc6f246737a28b9ecefabbbb7b7457952221d32de4ca88c28e170fc220f9.jpg)

![cc6efc6a0ed5974c03e2ae24d8e7fb2e04d3a6f492e7f7a8a56bf139c84aeb18.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/cc6efc6a0ed5974c03e2ae24d8e7fb2e04d3a6f492e7f7a8a56bf139c84aeb18.jpg)

![d32703f49f0e5e262faf45e21f95904fd5cbe2f7fe5797a6b8e7bbfe69cc77ab.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/d32703f49f0e5e262faf45e21f95904fd5cbe2f7fe5797a6b8e7bbfe69cc77ab.jpg)

![d4d1e8775bcb79470940caaf4600a85f6ae4b3c59c2de12bbbc925a80f5bf5c4.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/d4d1e8775bcb79470940caaf4600a85f6ae4b3c59c2de12bbbc925a80f5bf5c4.jpg)

![dd3f2b78bda220008e9abd4e2666dfa03598bf4bbb93a266c65019a788422c54.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/dd3f2b78bda220008e9abd4e2666dfa03598bf4bbb93a266c65019a788422c54.jpg)

![e9cb417a5fb9f11eac5c9ce116fa1bd54eb4b98e94fee45135ef9dc317d98986.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/e9cb417a5fb9f11eac5c9ce116fa1bd54eb4b98e94fee45135ef9dc317d98986.jpg)

![f600f18b410d3211e9686676bd364c0bc5de8ad43134210f29d35a74d6ccf469.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/f600f18b410d3211e9686676bd364c0bc5de8ad43134210f29d35a74d6ccf469.jpg)

## Scout Before You Attend: Sketch-and-Walk Sparse Attention for Efficient LLM Inference


### Images

![4f7a70b03f3bc22cdeaacf470bf4b016e9f24d269b17212aa19cb4697f688fe0.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/images/4f7a70b03f3bc22cdeaacf470bf4b016e9f24d269b17212aa19cb4697f688fe0.jpg)

![78c9e3b893b2fae8954452e85b37350bb0392823b97b9b28493dbf1caedb7b08.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/images/78c9e3b893b2fae8954452e85b37350bb0392823b97b9b28493dbf1caedb7b08.jpg)

![a40519ab100d060f834807a14485651fb323a01f5319594bd604dcffad98f2df.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/images/a40519ab100d060f834807a14485651fb323a01f5319594bd604dcffad98f2df.jpg)

![b8dcaa839727964554dd3c2707cfee995ef9177b2531138ee7cede574316dde5.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/images/b8dcaa839727964554dd3c2707cfee995ef9177b2531138ee7cede574316dde5.jpg)

![d63fb1bb5278a9d08f71464b2218d0ccb8623dff6b6b30c97d5c5619ca4c9ba7.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/images/d63fb1bb5278a9d08f71464b2218d0ccb8623dff6b6b30c97d5c5619ca4c9ba7.jpg)

![d66f2a0f6f12d2083aec15faabe86e125937d6fefc7da24608b0853506918573.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/images/d66f2a0f6f12d2083aec15faabe86e125937d6fefc7da24608b0853506918573.jpg)

![f95ba1fa345a388d6b17c8cb96b2dde35c5bcba82ef7fa48e969d58dad4c19b7.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/images/f95ba1fa345a388d6b17c8cb96b2dde35c5bcba82ef7fa48e969d58dad4c19b7.jpg)

### Tables

![0d2d47e424c3c874b85701189a12528e0c49f0ae7308b0af48bf93ad3f99f2e2.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/tables/0d2d47e424c3c874b85701189a12528e0c49f0ae7308b0af48bf93ad3f99f2e2.jpg)

![1c4b6f12f704e125ab093b1f88ae84224a70655775b79d93b628c3e85aa8facb.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/tables/1c4b6f12f704e125ab093b1f88ae84224a70655775b79d93b628c3e85aa8facb.jpg)

![36615b92c9636d5c8e4060937d0f6403bcbb1cdb93f654a719bfa011e4d122f4.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/tables/36615b92c9636d5c8e4060937d0f6403bcbb1cdb93f654a719bfa011e4d122f4.jpg)

![4a9757b57c15fa024739c5c283643996949eebd23231fc66fed8bf8073938dfd.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/tables/4a9757b57c15fa024739c5c283643996949eebd23231fc66fed8bf8073938dfd.jpg)

![56c6694e14e79a12b6dd4fa81746e1faa1ce5da703856b82c26847c9a7a1bdc3.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/tables/56c6694e14e79a12b6dd4fa81746e1faa1ce5da703856b82c26847c9a7a1bdc3.jpg)

![82676c440722724e9d04aacfd2432c3fbe0101cfbe9a6a62e051ec67b7eb6bc3.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/tables/82676c440722724e9d04aacfd2432c3fbe0101cfbe9a6a62e051ec67b7eb6bc3.jpg)

![890a4d967c9d262a2062a2a7634c10e1dc0b730ace3ae919f1165a6f4c391321.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/tables/890a4d967c9d262a2062a2a7634c10e1dc0b730ace3ae919f1165a6f4c391321.jpg)

![9556a2caa5c029a58f706af6662437ddec82d8333872a7b766986b40ebac1bdf.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/tables/9556a2caa5c029a58f706af6662437ddec82d8333872a7b766986b40ebac1bdf.jpg)

![aeec36fc708ef1cb6984fed8b9c18cc499fd7bd6433d2d46a9cf3d40e4bb11d3.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/tables/aeec36fc708ef1cb6984fed8b9c18cc499fd7bd6433d2d46a9cf3d40e4bb11d3.jpg)

![ca4e95fede8c7554bb78af6fa00d7c4c0a5d79f5780998dcbeb05fc5095f1879.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/tables/ca4e95fede8c7554bb78af6fa00d7c4c0a5d79f5780998dcbeb05fc5095f1879.jpg)

![ec9adc91483f9fcbc5b3dade0d2c80057a02fe8f795ae0dc5c2c7dc567b1b7de.jpg](../icml_results/2608_Scout%20Before%20You%20Attend_%20Sketch-and-Walk%20Sparse%20Attention%20for%20Efficient%20LLM%20Inference/tables/ec9adc91483f9fcbc5b3dade0d2c80057a02fe8f795ae0dc5c2c7dc567b1b7de.jpg)

<a id="paper-2576"></a>

## PISCES: Annotation-free Text-to-Video Post-Training via Optimal Transport-Aligned Rewards


### Images

![18820bd672c7d68c4858d8289e0a65c260c33b7673d09dd0906c25afee00206e.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/images/18820bd672c7d68c4858d8289e0a65c260c33b7673d09dd0906c25afee00206e.jpg)

![39d256c10d6a1470ff191ed0ce5dee3cc218a5f6ab92376ab8baa92c6bcaeb80.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/images/39d256c10d6a1470ff191ed0ce5dee3cc218a5f6ab92376ab8baa92c6bcaeb80.jpg)

![83d574bc3056b5353356ce2885c5ad4368d3ec66c709002b0eec342090dd6c94.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/images/83d574bc3056b5353356ce2885c5ad4368d3ec66c709002b0eec342090dd6c94.jpg)

![93ce9a2ea43153bb7e6405b66188bafca3df44aa8fa036783d17824311dd3f13.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/images/93ce9a2ea43153bb7e6405b66188bafca3df44aa8fa036783d17824311dd3f13.jpg)

![a1e8dcfb172fecc16bcdbb780fc761ef90c8b39cf0fda91849d8a7380402f09c.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/images/a1e8dcfb172fecc16bcdbb780fc761ef90c8b39cf0fda91849d8a7380402f09c.jpg)

![c1bde6155ec77eb6cc91978ffe7419b2be47739f2f0ca8c28fffc2e2120e1ac7.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/images/c1bde6155ec77eb6cc91978ffe7419b2be47739f2f0ca8c28fffc2e2120e1ac7.jpg)

![ceef5a852394ced7ae9331ab49c3cedd4abaa25feef9668b4375012aae98935a.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/images/ceef5a852394ced7ae9331ab49c3cedd4abaa25feef9668b4375012aae98935a.jpg)

![e6a74140c78f7b1d84fc92421b33b258bfa2a89ec9d2ae6f55bd584182db52aa.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/images/e6a74140c78f7b1d84fc92421b33b258bfa2a89ec9d2ae6f55bd584182db52aa.jpg)

![ed0794190ff321bd08f4e1ea0dbf1fb38b89b0f37388e4fd37b28d097d67900b.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/images/ed0794190ff321bd08f4e1ea0dbf1fb38b89b0f37388e4fd37b28d097d67900b.jpg)

![f342335479cfdf68828b6eec4946e8210da01df1ad586799f091897cb6543f71.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/images/f342335479cfdf68828b6eec4946e8210da01df1ad586799f091897cb6543f71.jpg)

### Tables

![07898d402b186dcdb3c712c1bc03acd9ff465f725371c3d173090cc7c65de484.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/tables/07898d402b186dcdb3c712c1bc03acd9ff465f725371c3d173090cc7c65de484.jpg)

![1b52b9f0f84dacd96c2890e6ebbe56a89e1e5799895dde999f76d6a53ca647be.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/tables/1b52b9f0f84dacd96c2890e6ebbe56a89e1e5799895dde999f76d6a53ca647be.jpg)

![2bd7148a1989fcf66bc4685a27deb7e9083ecedefab3efa0ce399569dc4cbf92.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/tables/2bd7148a1989fcf66bc4685a27deb7e9083ecedefab3efa0ce399569dc4cbf92.jpg)

![2eefc445b9180b63fdde920645de0d1a506f9e65b532e71d3f7403aa4320ca7d.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/tables/2eefc445b9180b63fdde920645de0d1a506f9e65b532e71d3f7403aa4320ca7d.jpg)

![4caade0c49a405c8422bb81cadc185518288742a39d7daa3868e887e736e0c4d.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/tables/4caade0c49a405c8422bb81cadc185518288742a39d7daa3868e887e736e0c4d.jpg)

![a077716645a388843ba404064d891e2e1e37cee40738b6b28146b06efdc74f2f.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/tables/a077716645a388843ba404064d891e2e1e37cee40738b6b28146b06efdc74f2f.jpg)

![aaba055d0e3c90c3d95d30db5f703c869ba540b1454ad67aa6eb25d98edbc453.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/tables/aaba055d0e3c90c3d95d30db5f703c869ba540b1454ad67aa6eb25d98edbc453.jpg)

![cf51682d2321a11f47ec17d6f79e1241a69af820677eb006df1875186a571db8.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/tables/cf51682d2321a11f47ec17d6f79e1241a69af820677eb006df1875186a571db8.jpg)

![d41dad43a9060bf3a6597c66590f4b9eb62b71836c8b744cb92b81fe6e4853f5.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/tables/d41dad43a9060bf3a6597c66590f4b9eb62b71836c8b744cb92b81fe6e4853f5.jpg)

![e326b63ea30777066d4ec387f8040c333c4c232ab37e4dbdb978a7b6d3193b42.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/tables/e326b63ea30777066d4ec387f8040c333c4c232ab37e4dbdb978a7b6d3193b42.jpg)

![e377c4618802dbb921c8c102a8010df96fbbb1de8b630bcafa91a609bdd03aa6.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/tables/e377c4618802dbb921c8c102a8010df96fbbb1de8b630bcafa91a609bdd03aa6.jpg)

![f403736ea1b270c52d7fece0c7977685cdb5458fdf2cdc5f480c1c0000310b34.jpg](../icml_results/2609_PISCES_%20Annotation-free%20Text-to-Video%20Post-Training%20via%20Optimal%20Transport-Aligned%20Rewards/tables/f403736ea1b270c52d7fece0c7977685cdb5458fdf2cdc5f480c1c0000310b34.jpg)

<a id="paper-2577"></a>

## Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function


### Images

![1453d8f889382ce0d3731855a7d61212f198273f7fe8ec34385cee1810538eca.jpg](../icml_results/2611_Provably%20Data-driven%20Multiple%20Hyper-parameter%20Tuning%20with%20Structured%20Loss%20Function/images/1453d8f889382ce0d3731855a7d61212f198273f7fe8ec34385cee1810538eca.jpg)

<a id="paper-2578"></a>

## FAFO: Lossy KV Cache Compression for Lossless Inference Acceleration via Draftless Fumble Decoding


### Images

![29d91b94792c896eefc895ff0187ccfd4b49bac28f93731a13416d0b8967a3bf.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/images/29d91b94792c896eefc895ff0187ccfd4b49bac28f93731a13416d0b8967a3bf.jpg)

![323e374d2112bc2f99e11134efb4ee9ea7c35d5b49a1981bb06d92dda9365006.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/images/323e374d2112bc2f99e11134efb4ee9ea7c35d5b49a1981bb06d92dda9365006.jpg)

![cdbefe11f3f2084865d05acea55e1c64a9cb6a8049718a122c11143cf448eaf6.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/images/cdbefe11f3f2084865d05acea55e1c64a9cb6a8049718a122c11143cf448eaf6.jpg)

![d7bb4804ddbee00f48e0e60be17c080c5a9a89b6079b12aba6891e52ce2d8b53.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/images/d7bb4804ddbee00f48e0e60be17c080c5a9a89b6079b12aba6891e52ce2d8b53.jpg)

![e8e8efa21f9bbb6f0397e3e95ddb557bca48fa2c405d7e48f317d2ba1bfae519.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/images/e8e8efa21f9bbb6f0397e3e95ddb557bca48fa2c405d7e48f317d2ba1bfae519.jpg)

### Tables

![1ae6720a92bb44061f0d85ba0c3a047d2a5d4a4db015ee1f24007e890e5e9b81.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/1ae6720a92bb44061f0d85ba0c3a047d2a5d4a4db015ee1f24007e890e5e9b81.jpg)

![1c3ea68a426420f183773dc206eb287d03abe8ddd328cd568a16f6d65eec9346.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/1c3ea68a426420f183773dc206eb287d03abe8ddd328cd568a16f6d65eec9346.jpg)

![20d31f6fc3b4a8a789f2bb424fbb6f28db4d74ea302ad4ea11c80f3cd3db1b27.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/20d31f6fc3b4a8a789f2bb424fbb6f28db4d74ea302ad4ea11c80f3cd3db1b27.jpg)

![25ecceb383d49c817fb54fc286c37c211a600002844017ac86e3b7bde611f348.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/25ecceb383d49c817fb54fc286c37c211a600002844017ac86e3b7bde611f348.jpg)

![2b043f77e00508ba8777a512b966c067eddd525892b9d7b94cb4998b05e2b56f.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/2b043f77e00508ba8777a512b966c067eddd525892b9d7b94cb4998b05e2b56f.jpg)

![320d7721bd20ceba8e4a65d98989e089b6d427c1494522c6416ce64354e74058.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/320d7721bd20ceba8e4a65d98989e089b6d427c1494522c6416ce64354e74058.jpg)

![425293c3439e907b1b84ec4a3a1ed752fb20855dd578789ef7cd182e309eda84.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/425293c3439e907b1b84ec4a3a1ed752fb20855dd578789ef7cd182e309eda84.jpg)

![43c651407a2e348f8351be45405285875237b5b7039fbdffe7064a2a1d99744b.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/43c651407a2e348f8351be45405285875237b5b7039fbdffe7064a2a1d99744b.jpg)

![44c29d4a96a5d645c56af11c95b880abb721eff38593cfea9b9a35eabbb5d313.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/44c29d4a96a5d645c56af11c95b880abb721eff38593cfea9b9a35eabbb5d313.jpg)

![56049bf87b3d4fe4272b08be3ceea5b06702bb67701c751135ef49957785e700.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/56049bf87b3d4fe4272b08be3ceea5b06702bb67701c751135ef49957785e700.jpg)

![669721b64486a607f40497ce1b214bd38c9fd25afd4dd3e6c560579f723c8599.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/669721b64486a607f40497ce1b214bd38c9fd25afd4dd3e6c560579f723c8599.jpg)

![79e4ca317419f76ac7e3eb21347ebdc0e94ec59529f583972fb1e82ba879d487.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/79e4ca317419f76ac7e3eb21347ebdc0e94ec59529f583972fb1e82ba879d487.jpg)

![841ddf763d9dbca2b7fb0a00bfe977e0d54d6fd2dad8ad20dd89f8c7324734ee.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/841ddf763d9dbca2b7fb0a00bfe977e0d54d6fd2dad8ad20dd89f8c7324734ee.jpg)

![8790f50132d17ce6d6c9d5e02f863aa34e014b27a2037aaa432aceb012f01299.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/8790f50132d17ce6d6c9d5e02f863aa34e014b27a2037aaa432aceb012f01299.jpg)

![96f5e006adb54d7d3cb816f5bdb98c9faa757144ec5cd8effe4bd4473e96e6c9.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/96f5e006adb54d7d3cb816f5bdb98c9faa757144ec5cd8effe4bd4473e96e6c9.jpg)

![97594f2f074242b0f75cd60e197209df657163daaddb694b0cb77dd4197881ac.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/97594f2f074242b0f75cd60e197209df657163daaddb694b0cb77dd4197881ac.jpg)

![9c5a13bf727f007b5c06a84fdc3b5bab6293cf68765a377955a6f8c15c853031.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/9c5a13bf727f007b5c06a84fdc3b5bab6293cf68765a377955a6f8c15c853031.jpg)

![9f515e4c18f73b75088ff8d64469588bbb3039c029b5433d40e21282e34e47de.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/9f515e4c18f73b75088ff8d64469588bbb3039c029b5433d40e21282e34e47de.jpg)

![a0331156e39b4a41b757bd778de01ef3b6b267526a6ee82d34c199130734a0a5.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/a0331156e39b4a41b757bd778de01ef3b6b267526a6ee82d34c199130734a0a5.jpg)

![b11966223a15753f637466e948d3f439d1ca6e6aedda774e508fa07c34ad5200.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/b11966223a15753f637466e948d3f439d1ca6e6aedda774e508fa07c34ad5200.jpg)

![bb8bad34569cf71dc96427578bdc00a974d6f230499953b4d0e4b357de0eb608.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/bb8bad34569cf71dc96427578bdc00a974d6f230499953b4d0e4b357de0eb608.jpg)

![cf831620d8a710da2a55ab44ae6819b96a75cdbd1ccba86de2ea964857b8bb77.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/cf831620d8a710da2a55ab44ae6819b96a75cdbd1ccba86de2ea964857b8bb77.jpg)

![f84312844dfb2ec680a0d40aca603b50860263ac0e8db77f5f17a685c09b79d1.jpg](../icml_results/2612_FAFO_%20Lossy%20KV%20Cache%20Compression%20for%20Lossless%20Inference%20Acceleration%20via%20Draftless%20Fumble%20Decoding/tables/f84312844dfb2ec680a0d40aca603b50860263ac0e8db77f5f17a685c09b79d1.jpg)

<a id="paper-2579"></a>

## Can I Have Your Order? Monte-Carlo Tree Search for Slot Filling Ordering in Diffusion Language Models


### Images

![06a13604629e25b3bf7a8f1f81378c632c473f0624bd681aeadbda05a7a70e62.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/images/06a13604629e25b3bf7a8f1f81378c632c473f0624bd681aeadbda05a7a70e62.jpg)

![166c1eee4769e3752a17ff6614114faca26bd2b764c8b0f02191e186a60a57f8.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/images/166c1eee4769e3752a17ff6614114faca26bd2b764c8b0f02191e186a60a57f8.jpg)

![3a6b150bb47e569c3144f59426cc350146d7ceac0b29b8a9d4abfd993f12327f.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/images/3a6b150bb47e569c3144f59426cc350146d7ceac0b29b8a9d4abfd993f12327f.jpg)

![3af52b9cccd345f1716aaf1a8b7d17e45591462e48d897190af99960ac496af8.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/images/3af52b9cccd345f1716aaf1a8b7d17e45591462e48d897190af99960ac496af8.jpg)

![4d6138bf1cc7639b0daf4763f69e6310ea8c076d3ce83e63634c6e25dffdf71e.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/images/4d6138bf1cc7639b0daf4763f69e6310ea8c076d3ce83e63634c6e25dffdf71e.jpg)

![5864c1b562061aa667289b25efd7587d9777d864c15fdbf8644fcf49005b96a6.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/images/5864c1b562061aa667289b25efd7587d9777d864c15fdbf8644fcf49005b96a6.jpg)

![9c8f5213db155a1924a5f4fb2024f2237c5e41719de2cbbe1bf5d1e4d80647dc.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/images/9c8f5213db155a1924a5f4fb2024f2237c5e41719de2cbbe1bf5d1e4d80647dc.jpg)

![a90126c9899c825fc27648d756ecff20e682b15dc1ddd745df4e526fd6d0cfe9.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/images/a90126c9899c825fc27648d756ecff20e682b15dc1ddd745df4e526fd6d0cfe9.jpg)

![b19aac9bc9e71978af52f45ef76aa9b61639e7bd2362278d631579ebbda07f7b.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/images/b19aac9bc9e71978af52f45ef76aa9b61639e7bd2362278d631579ebbda07f7b.jpg)

### Tables

![14df942e0c53964b0d3003b5abb5fdf130a730b9ecddbad33491253de9e6c9b3.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/tables/14df942e0c53964b0d3003b5abb5fdf130a730b9ecddbad33491253de9e6c9b3.jpg)

![2d3ce927578d01b667ee90735010b5431c667a0442413298a97d4542425eb250.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/tables/2d3ce927578d01b667ee90735010b5431c667a0442413298a97d4542425eb250.jpg)

![317809fb0ce6f6aba3117dee3355322517593a3e1e99ea99f1ac866d78861275.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/tables/317809fb0ce6f6aba3117dee3355322517593a3e1e99ea99f1ac866d78861275.jpg)

![8fdfe8c841862fd4ab59f445d73055c096e2fa7f69545d8d7955922cf2d69305.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/tables/8fdfe8c841862fd4ab59f445d73055c096e2fa7f69545d8d7955922cf2d69305.jpg)

![9d1e3807c5324df0e1fd72383362f5f6bda379e631179cd0e45f8cd958ab2a64.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/tables/9d1e3807c5324df0e1fd72383362f5f6bda379e631179cd0e45f8cd958ab2a64.jpg)

![d26da4b885a1e2f44b6b490ed15d13e654e96628e289769bad51f17b494dec82.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/tables/d26da4b885a1e2f44b6b490ed15d13e654e96628e289769bad51f17b494dec82.jpg)

![e11db414354865e0553c1cfaf66c20b378f5c1d57967d4c10629aa697af4c50d.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/tables/e11db414354865e0553c1cfaf66c20b378f5c1d57967d4c10629aa697af4c50d.jpg)

![ec1f72387fe6dacb3fc89783261091e7bd4854cbaf117e3c5fb78df8a9555482.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/tables/ec1f72387fe6dacb3fc89783261091e7bd4854cbaf117e3c5fb78df8a9555482.jpg)

![ff4d8761aba162580823f39ce09752057875c6828fa1b9bb57bb3db85925f12a.jpg](../icml_results/2613_Can%20I%20Have%20Your%20Order_%20Monte-Carlo%20Tree%20Search%20for%20Slot%20Filling%20Ordering%20in%20Diffusion%20Language%20Model/tables/ff4d8761aba162580823f39ce09752057875c6828fa1b9bb57bb3db85925f12a.jpg)

<a id="paper-2580"></a>

## Path-conditioned training: a principled way to rescale ReLU neural networks


### Images

![0dfa6a1ff2bf9fbe92373990a93657c9c226090c841c713be1eccf81e57e2dd0.jpg](../icml_results/2614_Path-conditioned%20training_%20a%20principled%20way%20to%20rescale%20ReLU%20neural%20networks/images/0dfa6a1ff2bf9fbe92373990a93657c9c226090c841c713be1eccf81e57e2dd0.jpg)

![217c50cf03630404b37c13ab6f25d3628fa53cdce335032a9c38f60f94b63114.jpg](../icml_results/2614_Path-conditioned%20training_%20a%20principled%20way%20to%20rescale%20ReLU%20neural%20networks/images/217c50cf03630404b37c13ab6f25d3628fa53cdce335032a9c38f60f94b63114.jpg)

![42c82b7da0261a4bdb22cb2840ea23f4453a0e85d215bed37f72bf202e4d99df.jpg](../icml_results/2614_Path-conditioned%20training_%20a%20principled%20way%20to%20rescale%20ReLU%20neural%20networks/images/42c82b7da0261a4bdb22cb2840ea23f4453a0e85d215bed37f72bf202e4d99df.jpg)

![4d89b212863815041a6305842ceed0518ef955c261d192c30a90a8666348d315.jpg](../icml_results/2614_Path-conditioned%20training_%20a%20principled%20way%20to%20rescale%20ReLU%20neural%20networks/images/4d89b212863815041a6305842ceed0518ef955c261d192c30a90a8666348d315.jpg)

![6d05024d15eb42aef4e07cdc88ce2bd60877325fee483c409e26d8e137f8d7b0.jpg](../icml_results/2614_Path-conditioned%20training_%20a%20principled%20way%20to%20rescale%20ReLU%20neural%20networks/images/6d05024d15eb42aef4e07cdc88ce2bd60877325fee483c409e26d8e137f8d7b0.jpg)

![85fab07d21663ea81b170a1b64b18751c40fdb3071890fd04db52a759a60f146.jpg](../icml_results/2614_Path-conditioned%20training_%20a%20principled%20way%20to%20rescale%20ReLU%20neural%20networks/images/85fab07d21663ea81b170a1b64b18751c40fdb3071890fd04db52a759a60f146.jpg)

![b827b3118320a945af00f45214afcabac3814447ebb56975262c16820c153601.jpg](../icml_results/2614_Path-conditioned%20training_%20a%20principled%20way%20to%20rescale%20ReLU%20neural%20networks/images/b827b3118320a945af00f45214afcabac3814447ebb56975262c16820c153601.jpg)

![c0d8d6e76ad9e3d1066205d8cfac9d8d2f8c2549a7f20b8c20d9bf12994920db.jpg](../icml_results/2614_Path-conditioned%20training_%20a%20principled%20way%20to%20rescale%20ReLU%20neural%20networks/images/c0d8d6e76ad9e3d1066205d8cfac9d8d2f8c2549a7f20b8c20d9bf12994920db.jpg)

![c1bcebd6be57b272f2c02bd8b3e65e44aa62dbd30057e46c2256d358a946e667.jpg](../icml_results/2614_Path-conditioned%20training_%20a%20principled%20way%20to%20rescale%20ReLU%20neural%20networks/images/c1bcebd6be57b272f2c02bd8b3e65e44aa62dbd30057e46c2256d358a946e667.jpg)

![d325678593ca8fa5cc735b23d805b1a0bb784b0768f806b4cca4785328d9bb20.jpg](../icml_results/2614_Path-conditioned%20training_%20a%20principled%20way%20to%20rescale%20ReLU%20neural%20networks/images/d325678593ca8fa5cc735b23d805b1a0bb784b0768f806b4cca4785328d9bb20.jpg)

![db36ffd8d37811d81076e5c8517fa39b0087dbe736dcb8624db78dc8ee508eb7.jpg](../icml_results/2614_Path-conditioned%20training_%20a%20principled%20way%20to%20rescale%20ReLU%20neural%20networks/images/db36ffd8d37811d81076e5c8517fa39b0087dbe736dcb8624db78dc8ee508eb7.jpg)

![e4f62a83e8b5ad65ccfbfa267117c2975a6f072e98f14f29947c65dc58041de0.jpg](../icml_results/2614_Path-conditioned%20training_%20a%20principled%20way%20to%20rescale%20ReLU%20neural%20networks/images/e4f62a83e8b5ad65ccfbfa267117c2975a6f072e98f14f29947c65dc58041de0.jpg)

![e907d1a81beb61447450767110fdc2ee29bf6cbcbc01e88b090b19f185733c85.jpg](../icml_results/2614_Path-conditioned%20training_%20a%20principled%20way%20to%20rescale%20ReLU%20neural%20networks/images/e907d1a81beb61447450767110fdc2ee29bf6cbcbc01e88b090b19f185733c85.jpg)

### Tables

![05784112c884c5fe53753965cf0381686367d60d6dbe84f2986e0390b692207f.jpg](../icml_results/2614_Path-conditioned%20training_%20a%20principled%20way%20to%20rescale%20ReLU%20neural%20networks/tables/05784112c884c5fe53753965cf0381686367d60d6dbe84f2986e0390b692207f.jpg)

![ae3c55c4b5095adcd17a3b13f6a4b7122b95017f990cda4234eaccb3410c452d.jpg](../icml_results/2614_Path-conditioned%20training_%20a%20principled%20way%20to%20rescale%20ReLU%20neural%20networks/tables/ae3c55c4b5095adcd17a3b13f6a4b7122b95017f990cda4234eaccb3410c452d.jpg)

<a id="paper-2581"></a>

## When Does Adaptation Win? Scaling Laws for Meta-Learning in Quantum Control


### Images

![1a40a968e53494cab64502f1a028d7f00ffa9df69c1becea53211065d244dd36.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/images/1a40a968e53494cab64502f1a028d7f00ffa9df69c1becea53211065d244dd36.jpg)

![1d0ecaf5e828e1cb0fa33d94be92fa62c5ea3df979b813de6da8f279f4ad12c0.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/images/1d0ecaf5e828e1cb0fa33d94be92fa62c5ea3df979b813de6da8f279f4ad12c0.jpg)

![1ea4fc9a95c288c508dadf26c6aa012c7a95d0d57f854a38e078a421f13fdec2.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/images/1ea4fc9a95c288c508dadf26c6aa012c7a95d0d57f854a38e078a421f13fdec2.jpg)

![4cbf2275b8578ed8e9e2f015e335cc1f21773f98a83a29bc2edfdd34b21d16ed.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/images/4cbf2275b8578ed8e9e2f015e335cc1f21773f98a83a29bc2edfdd34b21d16ed.jpg)

![6486e0129e636e3f555c76657ffecf483dbd45cf3f1c7fbeb4297761446835c1.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/images/6486e0129e636e3f555c76657ffecf483dbd45cf3f1c7fbeb4297761446835c1.jpg)

![6a1f493449177dc9f05218ca18effe75eb6b2cf94e760e26aa4c70b4a05e74f9.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/images/6a1f493449177dc9f05218ca18effe75eb6b2cf94e760e26aa4c70b4a05e74f9.jpg)

![6c6c4879085c8a08dd7986b134a2b1361ba798819989888990e7ce78ab275b04.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/images/6c6c4879085c8a08dd7986b134a2b1361ba798819989888990e7ce78ab275b04.jpg)

![72fae5052abb412cd2c84b1a4ba57e4f4e61babe5de9feca5fd245562a1a3133.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/images/72fae5052abb412cd2c84b1a4ba57e4f4e61babe5de9feca5fd245562a1a3133.jpg)

![756872f341ac769d2b5e70675101c3e2803449a166347f77a3a2bfd6bbcb89d0.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/images/756872f341ac769d2b5e70675101c3e2803449a166347f77a3a2bfd6bbcb89d0.jpg)

![7614fb48f3daf90ec7fb305625422c061dc863213a25f6cdfa1ae344a827bb11.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/images/7614fb48f3daf90ec7fb305625422c061dc863213a25f6cdfa1ae344a827bb11.jpg)

![9d54d4562e935a14c2df7dc4d6fc807ad94bcf47757cb48238db6bcf69dddb95.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/images/9d54d4562e935a14c2df7dc4d6fc807ad94bcf47757cb48238db6bcf69dddb95.jpg)

![b6ca53132037a0909a6e8de94bfa9c08b6d5f8f4648f36c09bc511266cb3a2b0.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/images/b6ca53132037a0909a6e8de94bfa9c08b6d5f8f4648f36c09bc511266cb3a2b0.jpg)

![c401ba16dd2cb270adfbb640fc6143321f20f4b58c93cf29b9180851c2288900.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/images/c401ba16dd2cb270adfbb640fc6143321f20f4b58c93cf29b9180851c2288900.jpg)

![cc18c9088681e616ebf812c0d5a7a2c4a5cc4e6b8e36dcbfc4d6af58eb8b6cfa.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/images/cc18c9088681e616ebf812c0d5a7a2c4a5cc4e6b8e36dcbfc4d6af58eb8b6cfa.jpg)

![ffd8b5fdd32452c4023022167bbf335fa308ab49e9f8f881f0b77997050a3049.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/images/ffd8b5fdd32452c4023022167bbf335fa308ab49e9f8f881f0b77997050a3049.jpg)

### Tables

![10f8ceb34a9103b1b21aed488ddb6e3cd50cee09053bdf077da74cdc754c8a3a.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/tables/10f8ceb34a9103b1b21aed488ddb6e3cd50cee09053bdf077da74cdc754c8a3a.jpg)

![4c6826ec42999330c7a72a8db6ee828002a22360e82ea9acf78d5336be254f4a.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/tables/4c6826ec42999330c7a72a8db6ee828002a22360e82ea9acf78d5336be254f4a.jpg)

![527f85554bdbb06dd4fe6a6ba7adf4be3eb2a5f8c2b78053128a97223236de06.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/tables/527f85554bdbb06dd4fe6a6ba7adf4be3eb2a5f8c2b78053128a97223236de06.jpg)

![8a02b1632ded74269c00575835d9117af8f92a3c89452d76157d3af9ad7ec145.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/tables/8a02b1632ded74269c00575835d9117af8f92a3c89452d76157d3af9ad7ec145.jpg)

![a688a3a20e58bb4564efcdd8ceb646659e237e14780116b65d5529df559a0ea2.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/tables/a688a3a20e58bb4564efcdd8ceb646659e237e14780116b65d5529df559a0ea2.jpg)

![b468bd48dcafa332f1c74f6d7dccbeaf5104de2d0dfa9d53920f3a62fdbf15cb.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/tables/b468bd48dcafa332f1c74f6d7dccbeaf5104de2d0dfa9d53920f3a62fdbf15cb.jpg)

![bc725e3d0e7df5e178272419fe9e225a85921ddf7e93629d6658b047fe7c843e.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/tables/bc725e3d0e7df5e178272419fe9e225a85921ddf7e93629d6658b047fe7c843e.jpg)

![c9dbc0abe805c490e80def9ad413847ce331facc3a0fba6d13c6432f1321351f.jpg](../icml_results/2615_When%20Does%20Adaptation%20Win_%20Scaling%20Laws%20for%20Meta-Learning%20in%20Quantum%20Control/tables/c9dbc0abe805c490e80def9ad413847ce331facc3a0fba6d13c6432f1321351f.jpg)

<a id="paper-2582"></a>

## Policy Search via Bayesian Optimization with Temporal Difference Gaussian Processes


### Images

![0664061a9bcac1feb49a5471afdb6d0f9a53098cb5e1522f4e78f6e8b6b265fe.jpg](../icml_results/2616_Policy%20Search%20via%20Bayesian%20Optimization%20with%20Temporal%20Difference%20Gaussian%20Processes/images/0664061a9bcac1feb49a5471afdb6d0f9a53098cb5e1522f4e78f6e8b6b265fe.jpg)

![2d58a04203e9e98da3564c59b9651f395e73e9f33b215339d451cb185d907c15.jpg](../icml_results/2616_Policy%20Search%20via%20Bayesian%20Optimization%20with%20Temporal%20Difference%20Gaussian%20Processes/images/2d58a04203e9e98da3564c59b9651f395e73e9f33b215339d451cb185d907c15.jpg)

![35539e3d021c415702fe5aa401c523d23df782f899dab47aca0e9bb69f0d233a.jpg](../icml_results/2616_Policy%20Search%20via%20Bayesian%20Optimization%20with%20Temporal%20Difference%20Gaussian%20Processes/images/35539e3d021c415702fe5aa401c523d23df782f899dab47aca0e9bb69f0d233a.jpg)

![3b122cbc7eb3b616bb0178828a2b458928a0c3f6fc3e99fef4c3b20d13b6b890.jpg](../icml_results/2616_Policy%20Search%20via%20Bayesian%20Optimization%20with%20Temporal%20Difference%20Gaussian%20Processes/images/3b122cbc7eb3b616bb0178828a2b458928a0c3f6fc3e99fef4c3b20d13b6b890.jpg)

![82521fe28724514e08f6d01fb1e0c5b48a6a4d170316bf3eb5f0574d0e52b1f9.jpg](../icml_results/2616_Policy%20Search%20via%20Bayesian%20Optimization%20with%20Temporal%20Difference%20Gaussian%20Processes/images/82521fe28724514e08f6d01fb1e0c5b48a6a4d170316bf3eb5f0574d0e52b1f9.jpg)

![972e6f0a6351275f7ec8da60e43f3ed23a2c7887949fccbc0b64893660d43111.jpg](../icml_results/2616_Policy%20Search%20via%20Bayesian%20Optimization%20with%20Temporal%20Difference%20Gaussian%20Processes/images/972e6f0a6351275f7ec8da60e43f3ed23a2c7887949fccbc0b64893660d43111.jpg)

### Tables

![1fceb4d98b1a30354c9468831880edb66df914517f1d024f6ddcf2c0fe47d621.jpg](../icml_results/2616_Policy%20Search%20via%20Bayesian%20Optimization%20with%20Temporal%20Difference%20Gaussian%20Processes/tables/1fceb4d98b1a30354c9468831880edb66df914517f1d024f6ddcf2c0fe47d621.jpg)

![dd58bbf009de169ac1962a04e19d977fae0fd7aed1b376bc2a4f885b550e6d0d.jpg](../icml_results/2616_Policy%20Search%20via%20Bayesian%20Optimization%20with%20Temporal%20Difference%20Gaussian%20Processes/tables/dd58bbf009de169ac1962a04e19d977fae0fd7aed1b376bc2a4f885b550e6d0d.jpg)

<a id="paper-2583"></a>

## The Catastrophic Failure of *the* k-Means Algorithm in High Dimensions, and How Hartigan’s Algorithm Avoids It


### Images

![2048a3e9d077924b678bbbe8b0f7bd043850b4dba1524f60f2d3d6e5339c0479.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/images/2048a3e9d077924b678bbbe8b0f7bd043850b4dba1524f60f2d3d6e5339c0479.jpg)

![27661d8d6d5bad1ff2f88a8e532f70911b7a1cb3b76da43b1ab350bd830f4857.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/images/27661d8d6d5bad1ff2f88a8e532f70911b7a1cb3b76da43b1ab350bd830f4857.jpg)

![2a21d6725050c66632f5e4c6b2c38951d127ce81b969fac81902c0ed664aaa35.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/images/2a21d6725050c66632f5e4c6b2c38951d127ce81b969fac81902c0ed664aaa35.jpg)

![309b5bee4a5c9e101f551d0c6a920ef214c14ef28278ccd8053b552ea0cd9ad0.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/images/309b5bee4a5c9e101f551d0c6a920ef214c14ef28278ccd8053b552ea0cd9ad0.jpg)

![323e90e56089fcafdc0b5eeb838f4fe84ebebabf221ca4fc0008e43bc91e8232.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/images/323e90e56089fcafdc0b5eeb838f4fe84ebebabf221ca4fc0008e43bc91e8232.jpg)

![3fc3194c68720b4c2c5527601cfead878fb5612e73e3e132e9bb43d08c194527.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/images/3fc3194c68720b4c2c5527601cfead878fb5612e73e3e132e9bb43d08c194527.jpg)

![458c4b0586241867e2cf9c63cae4f3c60aa2f98e17ac1d3853ae3d44930fce46.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/images/458c4b0586241867e2cf9c63cae4f3c60aa2f98e17ac1d3853ae3d44930fce46.jpg)

![878154ed912d25a810a31172314b731c77364335c370189af9b9ecf0005bc15b.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/images/878154ed912d25a810a31172314b731c77364335c370189af9b9ecf0005bc15b.jpg)

![8ee582e845b5ff21721aaf0ecc6744edcb2b8c7eed1e8b29d34a2c8d24165fd6.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/images/8ee582e845b5ff21721aaf0ecc6744edcb2b8c7eed1e8b29d34a2c8d24165fd6.jpg)

![96c48bdecb07cd2950c76ee2e724bc9ee22bae6707d33daa2f2e97789119bfee.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/images/96c48bdecb07cd2950c76ee2e724bc9ee22bae6707d33daa2f2e97789119bfee.jpg)

![a7156152b4a8d01e98ae827dd36a2952930904c2f9e3bf3181c26f4dfa5b1a16.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/images/a7156152b4a8d01e98ae827dd36a2952930904c2f9e3bf3181c26f4dfa5b1a16.jpg)

![ae4147db6fb653a40476936b27a14e3634a1733e2ca1b8954654eef55077b379.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/images/ae4147db6fb653a40476936b27a14e3634a1733e2ca1b8954654eef55077b379.jpg)

![be8b077a56d6b91af40c2a5322c37d6807e08bb5c8ecf2d7e5c2c372449b3672.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/images/be8b077a56d6b91af40c2a5322c37d6807e08bb5c8ecf2d7e5c2c372449b3672.jpg)

![cfc0c690fa4c6b82b591e1f3a508f06a19fe094f6d066e3be8f8e0bfbbcbf8d2.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/images/cfc0c690fa4c6b82b591e1f3a508f06a19fe094f6d066e3be8f8e0bfbbcbf8d2.jpg)

![e3cfbf9654baa1858c441ed529e2e82edd3683f6cd3046410f1f5f2c299a555d.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/images/e3cfbf9654baa1858c441ed529e2e82edd3683f6cd3046410f1f5f2c299a555d.jpg)

![ef4306b1a48a414bc1b6402df8b26a0f565d1c19fbb18e69de0d300ad3a687f8.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/images/ef4306b1a48a414bc1b6402df8b26a0f565d1c19fbb18e69de0d300ad3a687f8.jpg)

### Tables

![3443c1b55e98f9b4dbc1a83f9934bea6190ba59655e8f3c785062b79a103359b.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/tables/3443c1b55e98f9b4dbc1a83f9934bea6190ba59655e8f3c785062b79a103359b.jpg)

![4de555683cb6622d9590894883f4ee6fa99f8ef96738710744b61fd0df80c14a.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/tables/4de555683cb6622d9590894883f4ee6fa99f8ef96738710744b61fd0df80c14a.jpg)

![cc62bfdd6317786f34b5fee48884e5e3e3a77f5e389f0e67eefc43cc7d550684.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/tables/cc62bfdd6317786f34b5fee48884e5e3e3a77f5e389f0e67eefc43cc7d550684.jpg)

![d20cb65cc2770c95b69e2cb6f4831af83cb8e2582966889387bccb33fe0858f5.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/tables/d20cb65cc2770c95b69e2cb6f4831af83cb8e2582966889387bccb33fe0858f5.jpg)

![fd675415b3f5c6505fce636c1f8cc16279b95c631e7a2a0c5e1fcd68263f7839.jpg](../icml_results/2617_The%20Catastrophic%20Failure%20of%20_the_%20k-Means%20Algorithm%20in%20High%20Dimensions%2C%20and%20How%20Hartigan%E2%80%99s%20Algorithm/tables/fd675415b3f5c6505fce636c1f8cc16279b95c631e7a2a0c5e1fcd68263f7839.jpg)

<a id="paper-2584"></a>

## LDARNet: DNA Adaptive Representation Network with Learnable Tokenization for Genomic Modeling


### Images

![1d0aa739223d92857943f846d7265af6209ad12a9a86302a7543d140c0d7f7a1.jpg](../icml_results/2618_LDARNet_%20DNA%20Adaptive%20Representation%20Network%20with%20Learnable%20Tokenization%20for%20Genomic%20Modeling/images/1d0aa739223d92857943f846d7265af6209ad12a9a86302a7543d140c0d7f7a1.jpg)

![45194a23bb5cdbf42cb730be547e31f6bfd1f827348d91cec9545f2461ee148a.jpg](../icml_results/2618_LDARNet_%20DNA%20Adaptive%20Representation%20Network%20with%20Learnable%20Tokenization%20for%20Genomic%20Modeling/images/45194a23bb5cdbf42cb730be547e31f6bfd1f827348d91cec9545f2461ee148a.jpg)

![bc337b10ffdaa92ffd32686fc5326753bb621102baaa6e4cd24b69742c29e045.jpg](../icml_results/2618_LDARNet_%20DNA%20Adaptive%20Representation%20Network%20with%20Learnable%20Tokenization%20for%20Genomic%20Modeling/images/bc337b10ffdaa92ffd32686fc5326753bb621102baaa6e4cd24b69742c29e045.jpg)

![bdb59907029ac3fef0e593070cfd1fa5744a08b55ff7a5f03d20da060fa19b64.jpg](../icml_results/2618_LDARNet_%20DNA%20Adaptive%20Representation%20Network%20with%20Learnable%20Tokenization%20for%20Genomic%20Modeling/images/bdb59907029ac3fef0e593070cfd1fa5744a08b55ff7a5f03d20da060fa19b64.jpg)

### Tables

![2dec0d5d506ec9b42009ceae2524308d62891d9005b615f6d949482a804f5360.jpg](../icml_results/2618_LDARNet_%20DNA%20Adaptive%20Representation%20Network%20with%20Learnable%20Tokenization%20for%20Genomic%20Modeling/tables/2dec0d5d506ec9b42009ceae2524308d62891d9005b615f6d949482a804f5360.jpg)

![399ec9fc43e1693ed9d37a8fb99dfac282bb1129890dcb40a276aadcf41b3e06.jpg](../icml_results/2618_LDARNet_%20DNA%20Adaptive%20Representation%20Network%20with%20Learnable%20Tokenization%20for%20Genomic%20Modeling/tables/399ec9fc43e1693ed9d37a8fb99dfac282bb1129890dcb40a276aadcf41b3e06.jpg)

![6e9d4097693e781af963a08f863db08149633f9ac8d9113cd8af001b120a7370.jpg](../icml_results/2618_LDARNet_%20DNA%20Adaptive%20Representation%20Network%20with%20Learnable%20Tokenization%20for%20Genomic%20Modeling/tables/6e9d4097693e781af963a08f863db08149633f9ac8d9113cd8af001b120a7370.jpg)

![b16c9749b0129565cb3dd7c20c1e225cd3786038cd9813d99b7b5c4ea9c36d99.jpg](../icml_results/2618_LDARNet_%20DNA%20Adaptive%20Representation%20Network%20with%20Learnable%20Tokenization%20for%20Genomic%20Modeling/tables/b16c9749b0129565cb3dd7c20c1e225cd3786038cd9813d99b7b5c4ea9c36d99.jpg)

![c60643f3e7cb68ec0682d8b70857270bb74076da3c1c5905a03ab701b9e21b5c.jpg](../icml_results/2618_LDARNet_%20DNA%20Adaptive%20Representation%20Network%20with%20Learnable%20Tokenization%20for%20Genomic%20Modeling/tables/c60643f3e7cb68ec0682d8b70857270bb74076da3c1c5905a03ab701b9e21b5c.jpg)

![ebeb22782112254e1064b87a0f3d13d26d7e36cd3e56fd0b4ca27c2fbe2636ae.jpg](../icml_results/2618_LDARNet_%20DNA%20Adaptive%20Representation%20Network%20with%20Learnable%20Tokenization%20for%20Genomic%20Modeling/tables/ebeb22782112254e1064b87a0f3d13d26d7e36cd3e56fd0b4ca27c2fbe2636ae.jpg)

![efebb3375fee24ebd0fcd51d560dd51424c16dbfb7d50b75572cf74582051738.jpg](../icml_results/2618_LDARNet_%20DNA%20Adaptive%20Representation%20Network%20with%20Learnable%20Tokenization%20for%20Genomic%20Modeling/tables/efebb3375fee24ebd0fcd51d560dd51424c16dbfb7d50b75572cf74582051738.jpg)

![f42db0af29152ce110524fab8404c03bdc48ae3d8eed97cdf9c3acb6e31031f6.jpg](../icml_results/2618_LDARNet_%20DNA%20Adaptive%20Representation%20Network%20with%20Learnable%20Tokenization%20for%20Genomic%20Modeling/tables/f42db0af29152ce110524fab8404c03bdc48ae3d8eed97cdf9c3acb6e31031f6.jpg)

<a id="paper-2585"></a>

## GENEB: Why Genomic Models Are Hard to Compare


### Images

![043999667b8f7ec8fba4fe79863240d6675389f410b556101b848016be35f0f5.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/043999667b8f7ec8fba4fe79863240d6675389f410b556101b848016be35f0f5.jpg)

![0ebb1190d4ba7e9ff547fd56e01130667a7c3cbdc844cfe10ccad7aef432271c.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/0ebb1190d4ba7e9ff547fd56e01130667a7c3cbdc844cfe10ccad7aef432271c.jpg)

![14c56aca16f58919b31add40d21fda88256c8c10180a985ddcb73c2a24d2fbd6.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/14c56aca16f58919b31add40d21fda88256c8c10180a985ddcb73c2a24d2fbd6.jpg)

![161d07b00b2222249f349908af9e51bf00e89c29715f9d03f0cb5c2280bdb469.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/161d07b00b2222249f349908af9e51bf00e89c29715f9d03f0cb5c2280bdb469.jpg)

![2508787a47333d70c2dc0369175a6d519dd84c8b411614d7d1f2405ac1a337fe.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/2508787a47333d70c2dc0369175a6d519dd84c8b411614d7d1f2405ac1a337fe.jpg)

![258a69d675643f35572e51041d84e5e61625edc273ccfaa03863e4cc85f270c3.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/258a69d675643f35572e51041d84e5e61625edc273ccfaa03863e4cc85f270c3.jpg)

![259139da120ea2e820a3015cead43a7f02bd50de66218440c8133b923ead4adc.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/259139da120ea2e820a3015cead43a7f02bd50de66218440c8133b923ead4adc.jpg)

![2937d0704d810965a7598609898ede70d0c52e8f1b554750434a35d8f0eddea0.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/2937d0704d810965a7598609898ede70d0c52e8f1b554750434a35d8f0eddea0.jpg)

![2adef974e311f5bd7bccb185e94dc2bcb9ca83935a75cebacc54b62e83dc1db5.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/2adef974e311f5bd7bccb185e94dc2bcb9ca83935a75cebacc54b62e83dc1db5.jpg)

![31527129137453daa7e7d39f46ecd17ab89a7f0adfcc7a11550117d0953ecb32.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/31527129137453daa7e7d39f46ecd17ab89a7f0adfcc7a11550117d0953ecb32.jpg)

![33b832e2ed1d9a242184b247a3bf9e3230664f1f4a93807134af9897080585ee.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/33b832e2ed1d9a242184b247a3bf9e3230664f1f4a93807134af9897080585ee.jpg)

![359ba51aaaa986e6358b0204f5af6c962a2c17c10615308cbc0df35d3ce9c358.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/359ba51aaaa986e6358b0204f5af6c962a2c17c10615308cbc0df35d3ce9c358.jpg)

![3b7a5f9264f3f5753983c4ed09988c7bf2e115b5eb3d1d6c39296fe54a88407e.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/3b7a5f9264f3f5753983c4ed09988c7bf2e115b5eb3d1d6c39296fe54a88407e.jpg)

![3bbf293cfeb35cb67612258f52715ac1f7726150f64a291d785c9c637c6db10d.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/3bbf293cfeb35cb67612258f52715ac1f7726150f64a291d785c9c637c6db10d.jpg)

![3d6dd14ae81ada10938e7306948d9c02d21fb86acd93c0f0d4ef6c79502b1a3a.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/3d6dd14ae81ada10938e7306948d9c02d21fb86acd93c0f0d4ef6c79502b1a3a.jpg)

![3e387e41f9b1984953c1d54ec902f4d7727109af2fb0715b6b52623eb935ca13.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/3e387e41f9b1984953c1d54ec902f4d7727109af2fb0715b6b52623eb935ca13.jpg)

![4ba2c76a5944ecb32e9a908d6c9215179576455e80a9568408fcfd4d453ab71d.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/4ba2c76a5944ecb32e9a908d6c9215179576455e80a9568408fcfd4d453ab71d.jpg)

![4ea741ee8effb51d717898e6722af8450ad34ae4799b61ced2703e9e3b7cad6c.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/4ea741ee8effb51d717898e6722af8450ad34ae4799b61ced2703e9e3b7cad6c.jpg)

![5717ceba5e24eb8f2ef8a9c649af40b00735b09ce780339ade77c4fd7000fe89.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/5717ceba5e24eb8f2ef8a9c649af40b00735b09ce780339ade77c4fd7000fe89.jpg)

![5aa18346633fdad7272589bafb9bb97ca2fe1dfc0f36442fb0aacec3556e8297.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/5aa18346633fdad7272589bafb9bb97ca2fe1dfc0f36442fb0aacec3556e8297.jpg)

![5d192b3598b28d35b341c6ee735f8e61b9bc69b3bbcf63a1fa369a7ca4b5e4c9.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/5d192b3598b28d35b341c6ee735f8e61b9bc69b3bbcf63a1fa369a7ca4b5e4c9.jpg)

![5f289bc1231b959a7f3c10324769a96c21feba413a9300b5f26281c7e83b7d46.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/5f289bc1231b959a7f3c10324769a96c21feba413a9300b5f26281c7e83b7d46.jpg)

![6612abc92172f24b29d1f9392c24b0335a772c07488209ed5e8329b9f597bd4a.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/6612abc92172f24b29d1f9392c24b0335a772c07488209ed5e8329b9f597bd4a.jpg)

![789673b9086f7d6a4e2b81bdb0f6f8378456b6bea2cf82e891b55ae9ceeac5c6.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/789673b9086f7d6a4e2b81bdb0f6f8378456b6bea2cf82e891b55ae9ceeac5c6.jpg)

![79959fb1984bbecc3cd0d381528f8f525c66a534a62669d8b1e7fcd0a1859a26.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/79959fb1984bbecc3cd0d381528f8f525c66a534a62669d8b1e7fcd0a1859a26.jpg)

![7ba75636ce31ae2103fe822e0be0f1af7e0894a4fab7e84f653642c308fddd08.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/7ba75636ce31ae2103fe822e0be0f1af7e0894a4fab7e84f653642c308fddd08.jpg)

![819789c49fab6c2d29e3d6cccefc53b315f2da5cdf109a378bf838a74e33e79c.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/819789c49fab6c2d29e3d6cccefc53b315f2da5cdf109a378bf838a74e33e79c.jpg)

![822d7c4343b782aa6a790800bdcb3d3fc7e50a2f6e71662e7815d0237c0cc0df.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/822d7c4343b782aa6a790800bdcb3d3fc7e50a2f6e71662e7815d0237c0cc0df.jpg)

![84beaea2ef8522cf4bbc66bd3b914ba26dbd5395b3526efe168faa649ca38a3d.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/84beaea2ef8522cf4bbc66bd3b914ba26dbd5395b3526efe168faa649ca38a3d.jpg)

![8deb8936110638ab511d0017e9ef9e1c3b340b8bf4476f31af6d9cf905fb5d19.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/8deb8936110638ab511d0017e9ef9e1c3b340b8bf4476f31af6d9cf905fb5d19.jpg)

![919d22de7b483e64b852ad32617b29ce98d9f5cfbb2e84f61b03da8d719bda3c.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/919d22de7b483e64b852ad32617b29ce98d9f5cfbb2e84f61b03da8d719bda3c.jpg)

![996735ffe3d7a3476eb11ffa03e08853126120a214e1c8844fe63952d5509a92.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/996735ffe3d7a3476eb11ffa03e08853126120a214e1c8844fe63952d5509a92.jpg)

![99efe8f1bf43dc5e5d732d0d565eeb7f5def2d67e547713b166b417613ad4185.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/99efe8f1bf43dc5e5d732d0d565eeb7f5def2d67e547713b166b417613ad4185.jpg)

![9bfcd5d9274ce3ba2f5f1884ed6a7a69a8123456eaa753b9b93dc745b56fefd8.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/9bfcd5d9274ce3ba2f5f1884ed6a7a69a8123456eaa753b9b93dc745b56fefd8.jpg)

![a0f338c2808132adadaabe8d5a020bd38e578029dfdc4064f96b0f8594599fb1.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/a0f338c2808132adadaabe8d5a020bd38e578029dfdc4064f96b0f8594599fb1.jpg)

![a35ba24fab6a82c70a2d10ead9b48043cbe76cdbc471df990d3c7316e3f553c3.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/a35ba24fab6a82c70a2d10ead9b48043cbe76cdbc471df990d3c7316e3f553c3.jpg)

![b727905be91db088fe2107550f3987d484d8c90894236b1132f87901cbcc0739.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/b727905be91db088fe2107550f3987d484d8c90894236b1132f87901cbcc0739.jpg)

![ba63f100e0779635e8ec49f798904466d4cf2ab8aa435e6f08da850fd3b12194.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/ba63f100e0779635e8ec49f798904466d4cf2ab8aa435e6f08da850fd3b12194.jpg)

![bbd800fc4ba85c9555ab834bc91c0994475b0538df42ef771bcf524e726da85f.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/bbd800fc4ba85c9555ab834bc91c0994475b0538df42ef771bcf524e726da85f.jpg)

![c47872214217bdaf880b3fd67f71e64709d655b9392f390d03ede62fa5e3f94c.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/c47872214217bdaf880b3fd67f71e64709d655b9392f390d03ede62fa5e3f94c.jpg)

![c60261d4d6c92d714067c92ef6cd89c99deab520c7bc6f3bdf612b42d9fed1e8.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/c60261d4d6c92d714067c92ef6cd89c99deab520c7bc6f3bdf612b42d9fed1e8.jpg)

![c9e07b7941a9ef09c89739091cdbf05d30a99501a0a3921737a687fb0fc2fd8f.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/c9e07b7941a9ef09c89739091cdbf05d30a99501a0a3921737a687fb0fc2fd8f.jpg)

![cb977b919e106316b0605695b414b37479f97c254a6fafb6da525119746f9fd9.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/cb977b919e106316b0605695b414b37479f97c254a6fafb6da525119746f9fd9.jpg)

![cb9eb3958edca44aacba85b4df8d4fb925c28db69db62934d27b110ecb1b4d71.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/cb9eb3958edca44aacba85b4df8d4fb925c28db69db62934d27b110ecb1b4d71.jpg)

![ce55bc4be3d7279fd9d059b623cdd357a0a5a6d55338c0ce3d74fb41e6fb27c5.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/ce55bc4be3d7279fd9d059b623cdd357a0a5a6d55338c0ce3d74fb41e6fb27c5.jpg)

![d430fa2d1afbb3c029ef5ead872d6b02b601af53c098a1129da6c429f618005f.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/d430fa2d1afbb3c029ef5ead872d6b02b601af53c098a1129da6c429f618005f.jpg)

![e43fbf2dbd57e2be96d1f5f3a7dd14fa4e6af1f55c47815ff37ec04ee5f76fd2.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/e43fbf2dbd57e2be96d1f5f3a7dd14fa4e6af1f55c47815ff37ec04ee5f76fd2.jpg)

![e4bf821203a5c9141bd4e9b30e0711e10b0e42d7a2b2f84607bdee599128ec43.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/e4bf821203a5c9141bd4e9b30e0711e10b0e42d7a2b2f84607bdee599128ec43.jpg)

![eb35103c31a8e0226094a46a6c469139087d5d6ea2515352048830b4c655ef3f.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/eb35103c31a8e0226094a46a6c469139087d5d6ea2515352048830b4c655ef3f.jpg)

![ec804ef0622960409902aa97cd8a75e7b46882abdfbb8fc88472943fd004cb66.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/ec804ef0622960409902aa97cd8a75e7b46882abdfbb8fc88472943fd004cb66.jpg)

![ff7b777ca240799320e7beddc77dbeb944fccbf48992bc1a4d1406aa51b18cc9.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/images/ff7b777ca240799320e7beddc77dbeb944fccbf48992bc1a4d1406aa51b18cc9.jpg)

### Tables

![07e1ab5b9a27833c277b1af5529818722ca58d552c874ed22e462311b8c455f2.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/tables/07e1ab5b9a27833c277b1af5529818722ca58d552c874ed22e462311b8c455f2.jpg)

![08b0b9bdb88a619f8ccb7f11c6b696bf5ff639781962e6bbdf54081dd7e72115.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/tables/08b0b9bdb88a619f8ccb7f11c6b696bf5ff639781962e6bbdf54081dd7e72115.jpg)

![0c2ba79adf67fa4bede58e26d38656d97f91950be2801a8e47d4d568ce5e9db4.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/tables/0c2ba79adf67fa4bede58e26d38656d97f91950be2801a8e47d4d568ce5e9db4.jpg)

![0d9e0ebbb1994eb17229cb775ed072043cea03eac7f9cef708081445e293db92.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/tables/0d9e0ebbb1994eb17229cb775ed072043cea03eac7f9cef708081445e293db92.jpg)

![2f563880543a662fab7a911b5923c092a1b191ebaf5496c9d83763298bfe8668.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/tables/2f563880543a662fab7a911b5923c092a1b191ebaf5496c9d83763298bfe8668.jpg)

![30a5402d337b415be75865d2139923aa875f68265f8b9f16efac636965cdc548.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/tables/30a5402d337b415be75865d2139923aa875f68265f8b9f16efac636965cdc548.jpg)

![4befe0f730e52533e77a813a7a5f6b9cfb7b7eeb967675b0c85e9bea3ebeeddb.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/tables/4befe0f730e52533e77a813a7a5f6b9cfb7b7eeb967675b0c85e9bea3ebeeddb.jpg)

![58699926400055d9c38f0f5426854483ca41569787c2fa48c5911cb2647f325b.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/tables/58699926400055d9c38f0f5426854483ca41569787c2fa48c5911cb2647f325b.jpg)

![5c895fbbd8e0d69075ca9cf84e4f3ad2519761a1e2e0ab3f9f56b68df0f8b848.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/tables/5c895fbbd8e0d69075ca9cf84e4f3ad2519761a1e2e0ab3f9f56b68df0f8b848.jpg)

![a524a3e3b553236dceaa663bfa6a97c6010693f7583e9df03f871c32170f5a7d.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/tables/a524a3e3b553236dceaa663bfa6a97c6010693f7583e9df03f871c32170f5a7d.jpg)

![d29dc6db833641ce29c8be8a5f195e868a9838f4c45c54e89d7fefb884d1960a.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/tables/d29dc6db833641ce29c8be8a5f195e868a9838f4c45c54e89d7fefb884d1960a.jpg)

![d47fa48b13f1db05d886fb82ad93f17b6a63ad12a7af5bbe76aa3ee8aafbf91b.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/tables/d47fa48b13f1db05d886fb82ad93f17b6a63ad12a7af5bbe76aa3ee8aafbf91b.jpg)

![df633abe9ff68d7995d3f9122ddc2407baf5a794aeac1f443674e43ea45ee2fc.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/tables/df633abe9ff68d7995d3f9122ddc2407baf5a794aeac1f443674e43ea45ee2fc.jpg)

![e1f04f15cb86c496edc89b4e54841cf69ba71bc38c0dcc3935ab47f4d61f76ff.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/tables/e1f04f15cb86c496edc89b4e54841cf69ba71bc38c0dcc3935ab47f4d61f76ff.jpg)

![e339f4ab89976ef32c2d21b3188886b276f326142134f96302fb4d8f9636ed8b.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/tables/e339f4ab89976ef32c2d21b3188886b276f326142134f96302fb4d8f9636ed8b.jpg)

![ef2e6fdcda8656b0cb20d47d702217825147039074f402dcdccfdde5408b014f.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/tables/ef2e6fdcda8656b0cb20d47d702217825147039074f402dcdccfdde5408b014f.jpg)

![fdd48555d825086cff724edcb125d3c3a4888a513ad5979247e1822175333f0e.jpg](../icml_results/2619_GENEB_%20Why%20Genomic%20Models%20Are%20Hard%20to%20Compare/tables/fdd48555d825086cff724edcb125d3c3a4888a513ad5979247e1822175333f0e.jpg)

<a id="paper-2586"></a>

## Query Lens: Interpreting Sparse Key-Value Features with Indirect Effects


### Images

![32955a39619dcd628cc749f82331acd239fed7c26db7935c841b64cfd67ec1ea.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/images/32955a39619dcd628cc749f82331acd239fed7c26db7935c841b64cfd67ec1ea.jpg)

![850535b09ef21b3416ad9d491748cf5d6cbfd248ca5ed9c6d16d639a42449b48.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/images/850535b09ef21b3416ad9d491748cf5d6cbfd248ca5ed9c6d16d639a42449b48.jpg)

![948b71f65c6bdb444f5d7ba9c74201690c9042e344a0e5898d8520b7775ff304.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/images/948b71f65c6bdb444f5d7ba9c74201690c9042e344a0e5898d8520b7775ff304.jpg)

![e2690dee3b1a6eb7a0cbddb36f4b67d417793560aa008823b14c03281457af6c.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/images/e2690dee3b1a6eb7a0cbddb36f4b67d417793560aa008823b14c03281457af6c.jpg)

![f6aa25917464604e201f9a1bbb96579f9c6a94712f0befe081f912e101401762.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/images/f6aa25917464604e201f9a1bbb96579f9c6a94712f0befe081f912e101401762.jpg)

![f866d851b775d22dd57a9f0cef706ff74c6c5dd36f5d1b51758d24ad349fd96a.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/images/f866d851b775d22dd57a9f0cef706ff74c6c5dd36f5d1b51758d24ad349fd96a.jpg)

### Tables

![087d855ec9c7bd92cbffb8776962da885b70f82ef1695c04ad16b1d6484e0b80.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/087d855ec9c7bd92cbffb8776962da885b70f82ef1695c04ad16b1d6484e0b80.jpg)

![0c018db40ae29fa447a952cc93419cb9ffff4ccf9597b93651b26e4b60c12699.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/0c018db40ae29fa447a952cc93419cb9ffff4ccf9597b93651b26e4b60c12699.jpg)

![17b80a7a55aaef222bb29757336cdce97a768e7acd196b4373469e4eaed20739.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/17b80a7a55aaef222bb29757336cdce97a768e7acd196b4373469e4eaed20739.jpg)

![19193a4452d90f68de0c2de551bab18034351217d9c5a2d1233078cfa01827ce.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/19193a4452d90f68de0c2de551bab18034351217d9c5a2d1233078cfa01827ce.jpg)

![25fffb02717c2540b4c5d58fe31321f7d43ee017251b8e987b6ebbad85b23416.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/25fffb02717c2540b4c5d58fe31321f7d43ee017251b8e987b6ebbad85b23416.jpg)

![2c98d3a97bd69cffddd6f140ad572b454f29553ba35ff16ccf4ea39c5af62816.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/2c98d3a97bd69cffddd6f140ad572b454f29553ba35ff16ccf4ea39c5af62816.jpg)

![3ff5ab99d8a0ba2066dfb93b9648f0c8418b8737dec3e00b92fd8c03c0ae5217.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/3ff5ab99d8a0ba2066dfb93b9648f0c8418b8737dec3e00b92fd8c03c0ae5217.jpg)

![528ac0a7b553b4634f7efc6ac73228983810f33bccf080084fa276f94e4a22a2.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/528ac0a7b553b4634f7efc6ac73228983810f33bccf080084fa276f94e4a22a2.jpg)

![538d0f2d304c9b58ce3b212f114f8e454d57a38560035e2b173a1a1235a236ad.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/538d0f2d304c9b58ce3b212f114f8e454d57a38560035e2b173a1a1235a236ad.jpg)

![55d60953ef99dd29b6d643c3887f360c4f895fd4925e047eb4276b4e91403e13.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/55d60953ef99dd29b6d643c3887f360c4f895fd4925e047eb4276b4e91403e13.jpg)

![575b8f561c6e309cd60d7985aaafd49e77709ec75202304b40e69692ed0e95a3.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/575b8f561c6e309cd60d7985aaafd49e77709ec75202304b40e69692ed0e95a3.jpg)

![5c3af74631e8c126e6e02ca6a55ffd7e6ae8f395641dc04174f06de1cdc8e03b.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/5c3af74631e8c126e6e02ca6a55ffd7e6ae8f395641dc04174f06de1cdc8e03b.jpg)

![5d1661a302cc37086e78f47c2e89dca4786787247e16906a2649872031551bfa.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/5d1661a302cc37086e78f47c2e89dca4786787247e16906a2649872031551bfa.jpg)

![61658e2a407a240638195439499c9bc6b41955f6b04570c2d1ab821e122775dd.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/61658e2a407a240638195439499c9bc6b41955f6b04570c2d1ab821e122775dd.jpg)

![76d91b64ae731e2add6eaed866119ceca318399d5b4a1f75488006f3f27d146e.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/76d91b64ae731e2add6eaed866119ceca318399d5b4a1f75488006f3f27d146e.jpg)

![77da198b3e6285925881f00e4a699266a68966003d0cb1ab3b9db1fabf10a361.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/77da198b3e6285925881f00e4a699266a68966003d0cb1ab3b9db1fabf10a361.jpg)

![78d32d0c083de51ca4575e9faf4d27f9aac88bf2c5db0d5a13c8059e30bb1d7a.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/78d32d0c083de51ca4575e9faf4d27f9aac88bf2c5db0d5a13c8059e30bb1d7a.jpg)

![8188c9f28c072f3549ee1f6ad51a9b44705753075f4509f9e4d8cd63d71b5703.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/8188c9f28c072f3549ee1f6ad51a9b44705753075f4509f9e4d8cd63d71b5703.jpg)

![8dd830655d2cab41b424859344cdc46fbdf4143d7eac9085fcf38b107599c906.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/8dd830655d2cab41b424859344cdc46fbdf4143d7eac9085fcf38b107599c906.jpg)

![97cd923cf7d35e0aee7008f110e173246750cce4d0c537c4e571bd16d451c8a9.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/97cd923cf7d35e0aee7008f110e173246750cce4d0c537c4e571bd16d451c8a9.jpg)

![9b62c6cf1ca19ed0c80ca3ab4f656de5908c5ea22aa4cddabf0c11d33aa8e477.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/9b62c6cf1ca19ed0c80ca3ab4f656de5908c5ea22aa4cddabf0c11d33aa8e477.jpg)

![9f34d3193d1c5558e2ecc2f49a2919102e3d2f3e846c1abf92153ad877ec5230.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/9f34d3193d1c5558e2ecc2f49a2919102e3d2f3e846c1abf92153ad877ec5230.jpg)

![adb3e4887a9a42a982e0bd10ed34569f70ce8338fa3e5114fdf746e4396e82d7.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/adb3e4887a9a42a982e0bd10ed34569f70ce8338fa3e5114fdf746e4396e82d7.jpg)

![af7b516cc8976088737d40e886c34ace41c9360b19e042fff30b46eda215f56f.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/af7b516cc8976088737d40e886c34ace41c9360b19e042fff30b46eda215f56f.jpg)

![ba5e33cf0b7fd73aa1061fc0cff27467efd033c1c9fed5d9322c51bd2ae27ce2.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/ba5e33cf0b7fd73aa1061fc0cff27467efd033c1c9fed5d9322c51bd2ae27ce2.jpg)

![ccf8ef58be34f4a6315d9b3d35f4171ddba64621f2833145b6c8a82bdf283634.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/ccf8ef58be34f4a6315d9b3d35f4171ddba64621f2833145b6c8a82bdf283634.jpg)

![ce227d98f025b0eca8fde71193bb3de51a8ca5a9caea2e105c8d2001cc71ca80.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/ce227d98f025b0eca8fde71193bb3de51a8ca5a9caea2e105c8d2001cc71ca80.jpg)

![d5c3d224ab5d70ee4f959cf9d0f3ce04f96ca5f2f422420704d3c1709de9975f.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/d5c3d224ab5d70ee4f959cf9d0f3ce04f96ca5f2f422420704d3c1709de9975f.jpg)

![e4f135e296198a7dc7de71abd186de265919c66e593d18a60327aef785e3ed4d.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/e4f135e296198a7dc7de71abd186de265919c66e593d18a60327aef785e3ed4d.jpg)

![e7ca76ffe45e99fe836d4f8dd7c2bd2f3a245ccecde7414277cbc21113d89e96.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/e7ca76ffe45e99fe836d4f8dd7c2bd2f3a245ccecde7414277cbc21113d89e96.jpg)

![f467dda2ef26d5d9110e2db140a1cd596a2868c5051d26e2c3c69cfc4479b826.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/f467dda2ef26d5d9110e2db140a1cd596a2868c5051d26e2c3c69cfc4479b826.jpg)

![ff692df8100c130269d71d6de1cd403407981111bbbb91f781f024214ce7bd75.jpg](../icml_results/2620_Query%20Lens_%20Interpreting%20Sparse%20Key-Value%20Features%20with%20Indirect%20Effects/tables/ff692df8100c130269d71d6de1cd403407981111bbbb91f781f024214ce7bd75.jpg)

<a id="paper-2587"></a>

## Decomposing Query-Key Feature Interactions Using Contrastive Covariances


### Images

![035e87743a975c7397f34cd950b5ecc3144ef7b37e85f3c5e51a0139eb68ee49.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/035e87743a975c7397f34cd950b5ecc3144ef7b37e85f3c5e51a0139eb68ee49.jpg)

![120ccd29ed7d4da0f75dc1d7910939c44f26d7aeb2cc743e141b75711d9b43a7.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/120ccd29ed7d4da0f75dc1d7910939c44f26d7aeb2cc743e141b75711d9b43a7.jpg)

![242eb0e0cfcfb29e3ea6092f204738d9e9c93e7496592edcfca3c5fb7c730b02.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/242eb0e0cfcfb29e3ea6092f204738d9e9c93e7496592edcfca3c5fb7c730b02.jpg)

![37af69f355dbbbeb97a1833a70865c3026dc0aa74a0bb04580cd96f68b30da92.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/37af69f355dbbbeb97a1833a70865c3026dc0aa74a0bb04580cd96f68b30da92.jpg)

![42398d4d96d3767c9a81230031c7fe3167c73002cba6103de8f451cd936c6bcd.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/42398d4d96d3767c9a81230031c7fe3167c73002cba6103de8f451cd936c6bcd.jpg)

![6bffb334f5eb74d7e34becea26a43528e6a9d6d22c93a4ad5c8b601d4e78e70a.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/6bffb334f5eb74d7e34becea26a43528e6a9d6d22c93a4ad5c8b601d4e78e70a.jpg)

![706b32ef0e6aadbaeef71a9292586ff8048742d6a8737848584b533624a4f4fd.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/706b32ef0e6aadbaeef71a9292586ff8048742d6a8737848584b533624a4f4fd.jpg)

![77277cc9c1376443a204e486fc68b53c9d91b292146e9112f3527da57a67c0af.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/77277cc9c1376443a204e486fc68b53c9d91b292146e9112f3527da57a67c0af.jpg)

![829968046dfd8e1ab4b7874cbb37a4936534643e62f307fc314c32cf81133a45.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/829968046dfd8e1ab4b7874cbb37a4936534643e62f307fc314c32cf81133a45.jpg)

![9a229aaef7c597fb9059e2a548488e5e1fe453fab4d9805201b28312f2e2f927.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/9a229aaef7c597fb9059e2a548488e5e1fe453fab4d9805201b28312f2e2f927.jpg)

![9d89f5671f0e5f279ff7fd51829d9489fbd40d144fc64f10cf0e0ce7b83c9375.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/9d89f5671f0e5f279ff7fd51829d9489fbd40d144fc64f10cf0e0ce7b83c9375.jpg)

![aab66212a0a588f6fa53e1e9409cf591c70ec1a1291fb0a4d237f35f54254a97.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/aab66212a0a588f6fa53e1e9409cf591c70ec1a1291fb0a4d237f35f54254a97.jpg)

![ab56e9d0a17af7511f10a330f1d43153a3825d2cfe856690e9fc905aabdafcea.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/ab56e9d0a17af7511f10a330f1d43153a3825d2cfe856690e9fc905aabdafcea.jpg)

![b75e8a6a4f6adef4b9c08044f95c12bf8e2b41ac7b98911c9cbdb8d4d44af2e7.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/b75e8a6a4f6adef4b9c08044f95c12bf8e2b41ac7b98911c9cbdb8d4d44af2e7.jpg)

![c1b293f349839cdcc3427acc322c9252cc5106971046a1215c0ef5411fc48fda.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/c1b293f349839cdcc3427acc322c9252cc5106971046a1215c0ef5411fc48fda.jpg)

![ca2d5b992d26d4a803254f16dd9abc7c657a7f06af3b8bdb27fd17b02016624c.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/ca2d5b992d26d4a803254f16dd9abc7c657a7f06af3b8bdb27fd17b02016624c.jpg)

![d1812ff0c16df4e7686fb60503c11017c2d8259a574389f3ed1630d9ae858c51.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/d1812ff0c16df4e7686fb60503c11017c2d8259a574389f3ed1630d9ae858c51.jpg)

![f32990a5d478084b0c266bac513bb124af6894cd3f908bb4d5253b80a2950157.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/f32990a5d478084b0c266bac513bb124af6894cd3f908bb4d5253b80a2950157.jpg)

![f6913b3ab36d6e8983e29cf445976792980babbb2a5122ce6b5b3070a7a6fc8b.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/f6913b3ab36d6e8983e29cf445976792980babbb2a5122ce6b5b3070a7a6fc8b.jpg)

![fbbb49b79405c0b6d90df79fafd42404da2130a8ff2e49824a619ac20e88fb42.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/images/fbbb49b79405c0b6d90df79fafd42404da2130a8ff2e49824a619ac20e88fb42.jpg)

### Tables

![956f9b6c39956632e34f79b33543603e58817e0f006519ca0e931630fb28db19.jpg](../icml_results/2621_Decomposing%20Query-Key%20Feature%20Interactions%20Using%20Contrastive%20Covariances/tables/956f9b6c39956632e34f79b33543603e58817e0f006519ca0e931630fb28db19.jpg)

<a id="paper-2588"></a>

## SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant LLM Pretraining Systems with 100k+ GPUs


### Images

![03f64249e34f94e5d80bef0c23b1d51b497f089cacc5114d07c1bc07b4fe8a0c.jpg](../icml_results/2622_SPARe_%20Stacked%20Parallelism%20with%20Adaptive%20Reordering%20for%20Fault-Tolerant%20LLM%20Pretraining%20Systems%20with%20/images/03f64249e34f94e5d80bef0c23b1d51b497f089cacc5114d07c1bc07b4fe8a0c.jpg)

![275060e28844a541074cbe65ec7616ac2e58562bb9c1c0939bd063dac08a03b1.jpg](../icml_results/2622_SPARe_%20Stacked%20Parallelism%20with%20Adaptive%20Reordering%20for%20Fault-Tolerant%20LLM%20Pretraining%20Systems%20with%20/images/275060e28844a541074cbe65ec7616ac2e58562bb9c1c0939bd063dac08a03b1.jpg)

![5a2e3e25b3abf86f507674131643d6205127f53c2c27a1d8c9210df4188df47a.jpg](../icml_results/2622_SPARe_%20Stacked%20Parallelism%20with%20Adaptive%20Reordering%20for%20Fault-Tolerant%20LLM%20Pretraining%20Systems%20with%20/images/5a2e3e25b3abf86f507674131643d6205127f53c2c27a1d8c9210df4188df47a.jpg)

![9751ca4040b9c35f5eaddfa4f13d3a22168657ddeb64e78ad01f4ca4ab63c67a.jpg](../icml_results/2622_SPARe_%20Stacked%20Parallelism%20with%20Adaptive%20Reordering%20for%20Fault-Tolerant%20LLM%20Pretraining%20Systems%20with%20/images/9751ca4040b9c35f5eaddfa4f13d3a22168657ddeb64e78ad01f4ca4ab63c67a.jpg)

![a887c6ffea9a55a34526ab0a9f0ca14859b779b02837ec44c5203535bf0d9a7a.jpg](../icml_results/2622_SPARe_%20Stacked%20Parallelism%20with%20Adaptive%20Reordering%20for%20Fault-Tolerant%20LLM%20Pretraining%20Systems%20with%20/images/a887c6ffea9a55a34526ab0a9f0ca14859b779b02837ec44c5203535bf0d9a7a.jpg)

![bc6d425148ae654a3d2619e18acd45b11de06916076cbab5b9f9f0b1afc7ad28.jpg](../icml_results/2622_SPARe_%20Stacked%20Parallelism%20with%20Adaptive%20Reordering%20for%20Fault-Tolerant%20LLM%20Pretraining%20Systems%20with%20/images/bc6d425148ae654a3d2619e18acd45b11de06916076cbab5b9f9f0b1afc7ad28.jpg)

![d4da37d829a9513d4e59ca6a5bc916d275a20b37c1fe91ee7db6e6c537dbd6b3.jpg](../icml_results/2622_SPARe_%20Stacked%20Parallelism%20with%20Adaptive%20Reordering%20for%20Fault-Tolerant%20LLM%20Pretraining%20Systems%20with%20/images/d4da37d829a9513d4e59ca6a5bc916d275a20b37c1fe91ee7db6e6c537dbd6b3.jpg)

![d90757093e8c5a00d6ff6df9eddac325a7548dfbb439d3e5721dfbc4558da8c0.jpg](../icml_results/2622_SPARe_%20Stacked%20Parallelism%20with%20Adaptive%20Reordering%20for%20Fault-Tolerant%20LLM%20Pretraining%20Systems%20with%20/images/d90757093e8c5a00d6ff6df9eddac325a7548dfbb439d3e5721dfbc4558da8c0.jpg)

![f28c4f2b3d69775efec7c02a13b66445f66968e66b05c2682b724d97ba1b6d90.jpg](../icml_results/2622_SPARe_%20Stacked%20Parallelism%20with%20Adaptive%20Reordering%20for%20Fault-Tolerant%20LLM%20Pretraining%20Systems%20with%20/images/f28c4f2b3d69775efec7c02a13b66445f66968e66b05c2682b724d97ba1b6d90.jpg)

![f4416d6b2c3f73738470a17cdfcfdb13c6a77f329cd4a626b411b43082ab78a8.jpg](../icml_results/2622_SPARe_%20Stacked%20Parallelism%20with%20Adaptive%20Reordering%20for%20Fault-Tolerant%20LLM%20Pretraining%20Systems%20with%20/images/f4416d6b2c3f73738470a17cdfcfdb13c6a77f329cd4a626b411b43082ab78a8.jpg)

### Tables

![15c08afd3eb60d284f6a7a9765f31dc0f4b511b52912385149115879a771e5a7.jpg](../icml_results/2622_SPARe_%20Stacked%20Parallelism%20with%20Adaptive%20Reordering%20for%20Fault-Tolerant%20LLM%20Pretraining%20Systems%20with%20/tables/15c08afd3eb60d284f6a7a9765f31dc0f4b511b52912385149115879a771e5a7.jpg)

![3d70e21e515779b4c53a729702171feaa285d8c90a497007f5bf01bbf8d67078.jpg](../icml_results/2622_SPARe_%20Stacked%20Parallelism%20with%20Adaptive%20Reordering%20for%20Fault-Tolerant%20LLM%20Pretraining%20Systems%20with%20/tables/3d70e21e515779b4c53a729702171feaa285d8c90a497007f5bf01bbf8d67078.jpg)

![7d28f2dfac4eab8e4c796f2f82165252615fbcc952197e2bd8ecb1ad4f99ed9c.jpg](../icml_results/2622_SPARe_%20Stacked%20Parallelism%20with%20Adaptive%20Reordering%20for%20Fault-Tolerant%20LLM%20Pretraining%20Systems%20with%20/tables/7d28f2dfac4eab8e4c796f2f82165252615fbcc952197e2bd8ecb1ad4f99ed9c.jpg)

![80ee83ab64771eefee347721bbb9abcceb6fa93d16f8ce1a6c9ab06bb6179486.jpg](../icml_results/2622_SPARe_%20Stacked%20Parallelism%20with%20Adaptive%20Reordering%20for%20Fault-Tolerant%20LLM%20Pretraining%20Systems%20with%20/tables/80ee83ab64771eefee347721bbb9abcceb6fa93d16f8ce1a6c9ab06bb6179486.jpg)

![c05b080d06577f99ca047c027c07b93afbe5b986a440e80ed28aa9bf0fcb7077.jpg](../icml_results/2622_SPARe_%20Stacked%20Parallelism%20with%20Adaptive%20Reordering%20for%20Fault-Tolerant%20LLM%20Pretraining%20Systems%20with%20/tables/c05b080d06577f99ca047c027c07b93afbe5b986a440e80ed28aa9bf0fcb7077.jpg)

![f970f4332527ce682b6945c0915a352b68a4eabb74b28cdfcfa190766b6c5147.jpg](../icml_results/2622_SPARe_%20Stacked%20Parallelism%20with%20Adaptive%20Reordering%20for%20Fault-Tolerant%20LLM%20Pretraining%20Systems%20with%20/tables/f970f4332527ce682b6945c0915a352b68a4eabb74b28cdfcfa190766b6c5147.jpg)

<a id="paper-2589"></a>

## Dustin: Draft-Augmented Sparse Verification for Efficient Long-Context Generation with Speculative Decoding


### Images

![24167ea6fa2436f2705b4d7f8c59bbeab4de275137415f3a03fe1a7dcc2a02ee.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/images/24167ea6fa2436f2705b4d7f8c59bbeab4de275137415f3a03fe1a7dcc2a02ee.jpg)

![2848e382eab6ebaaaf2e63f0aee8c0539be7e327584e104bf3e4298b53de95e6.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/images/2848e382eab6ebaaaf2e63f0aee8c0539be7e327584e104bf3e4298b53de95e6.jpg)

![294e3726c7467aa8314b03f3549e6f6f94755362b634e24e81ac36c1bce954ef.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/images/294e3726c7467aa8314b03f3549e6f6f94755362b634e24e81ac36c1bce954ef.jpg)

![3d610e6443ccedea8eb15cc51134936b85c8ffe06157fe653712225f2c912e2d.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/images/3d610e6443ccedea8eb15cc51134936b85c8ffe06157fe653712225f2c912e2d.jpg)

![4eca61eed91b9667349ed91e98a5a75472e0a22d02eab5c9bf6b26cce82765d8.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/images/4eca61eed91b9667349ed91e98a5a75472e0a22d02eab5c9bf6b26cce82765d8.jpg)

![561694a32d8b375316f5ed0ea80501866d2b922424e78fdb9591b24b43b849df.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/images/561694a32d8b375316f5ed0ea80501866d2b922424e78fdb9591b24b43b849df.jpg)

![5e3aa2358c91705404a678041cfd29560e30932561ce0a50f518eda1b79e87e1.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/images/5e3aa2358c91705404a678041cfd29560e30932561ce0a50f518eda1b79e87e1.jpg)

![9c982c2b125a8f5e2035ce63e3ca8cf190b019cedf9fcc09316cd148c6c08791.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/images/9c982c2b125a8f5e2035ce63e3ca8cf190b019cedf9fcc09316cd148c6c08791.jpg)

![9ca24230d7fff338ffe879f6ec71a24817e0375b913a7d2ec9ea78be4ba1f2ca.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/images/9ca24230d7fff338ffe879f6ec71a24817e0375b913a7d2ec9ea78be4ba1f2ca.jpg)

![aca0b663aceda8a4827821bf6af4364a0ef263e094ed9267d135d2a0a9f12959.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/images/aca0b663aceda8a4827821bf6af4364a0ef263e094ed9267d135d2a0a9f12959.jpg)

![c24ad22bcd727a1bc1c5907d89368ed96f13269b02c4cc69ebf282294d92b4fb.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/images/c24ad22bcd727a1bc1c5907d89368ed96f13269b02c4cc69ebf282294d92b4fb.jpg)

![ca2b802744e2959288563f0fbd542a7861fe3793ed0083f28a467278723d2845.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/images/ca2b802744e2959288563f0fbd542a7861fe3793ed0083f28a467278723d2845.jpg)

![f13747e8da7795288d4b6c43e19bc63d64ff31fa5451005b492f3449bca82448.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/images/f13747e8da7795288d4b6c43e19bc63d64ff31fa5451005b492f3449bca82448.jpg)

### Tables

![0faa5346825219080f555cbd9406cc5c2967e9a0124d16877d344c50e38a2e83.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/tables/0faa5346825219080f555cbd9406cc5c2967e9a0124d16877d344c50e38a2e83.jpg)

![23dd32a68d390f0570d2e89612a8dd33eb538b889af311ac938bf9d1f0c1a376.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/tables/23dd32a68d390f0570d2e89612a8dd33eb538b889af311ac938bf9d1f0c1a376.jpg)

![32fecd494402f927c368676f7128c76ffef43a1132c3e637045010035eb3f53e.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/tables/32fecd494402f927c368676f7128c76ffef43a1132c3e637045010035eb3f53e.jpg)

![3699bc6e1ce7fec416f1b6936942b5e4e864f28e01cce7cd4755f392607ac0f3.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/tables/3699bc6e1ce7fec416f1b6936942b5e4e864f28e01cce7cd4755f392607ac0f3.jpg)

![5057a5b09766951bb2e08f8c099453fc801b422d378dabb048a7e8f64c0217ae.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/tables/5057a5b09766951bb2e08f8c099453fc801b422d378dabb048a7e8f64c0217ae.jpg)

![5db9eda13c5201474fe78b2364794278cad8cc3e0207af914ba21d0be9949aed.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/tables/5db9eda13c5201474fe78b2364794278cad8cc3e0207af914ba21d0be9949aed.jpg)

![65dc27db7eafabd143421540e1c824873ffa5224ac0106eab60002e2cadbdb54.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/tables/65dc27db7eafabd143421540e1c824873ffa5224ac0106eab60002e2cadbdb54.jpg)

![6a6606abf47bdc533cc26268c876a8ec0ddb78a91727dbd07454033f8e0910d9.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/tables/6a6606abf47bdc533cc26268c876a8ec0ddb78a91727dbd07454033f8e0910d9.jpg)

![6ac012417ca1d94a076e949499a57ca62fc4d300afc8f2663bf7667b2992e999.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/tables/6ac012417ca1d94a076e949499a57ca62fc4d300afc8f2663bf7667b2992e999.jpg)

![9cd841f8bf7a8c2efe63a360fc63ce296d73c9c396f0034d3aeab75fdbfa60a1.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/tables/9cd841f8bf7a8c2efe63a360fc63ce296d73c9c396f0034d3aeab75fdbfa60a1.jpg)

![b47b1f21325f24b52f3f45990835b5ef3d4fe143e466bb5c62827327f9db2aed.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/tables/b47b1f21325f24b52f3f45990835b5ef3d4fe143e466bb5c62827327f9db2aed.jpg)

![c6ae7a5621d21d7c049a2ea25f3fa16e67cf737b638f1a73b526655c07abe46d.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/tables/c6ae7a5621d21d7c049a2ea25f3fa16e67cf737b638f1a73b526655c07abe46d.jpg)

![d0fc2b1e3b1e938d24428ca030d9e59459af9a2f492432fecb615edd609780a1.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/tables/d0fc2b1e3b1e938d24428ca030d9e59459af9a2f492432fecb615edd609780a1.jpg)

![eaa52d9ec5ae23c9f4195294cd1bdd72b4f3370cc463d3bd90bb7e8858b0abe5.jpg](../icml_results/2623_Dustin_%20Draft-Augmented%20Sparse%20Verification%20for%20Efficient%20Long-Context%20Generation%20with%20Speculative%20D/tables/eaa52d9ec5ae23c9f4195294cd1bdd72b4f3370cc463d3bd90bb7e8858b0abe5.jpg)

<a id="paper-2590"></a>

## CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large Vision-Language Models


### Images

![5693d526ef27c768231145efdfeded3fbde203762e4d5fcd8c8b01ec7c209429.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/images/5693d526ef27c768231145efdfeded3fbde203762e4d5fcd8c8b01ec7c209429.jpg)

![686f32e5c7f3cb31101c699ecd68823f57c7733b2231324bd8cf6e2a96170f9c.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/images/686f32e5c7f3cb31101c699ecd68823f57c7733b2231324bd8cf6e2a96170f9c.jpg)

![a38b43922cd20ffb98ef3db0d762d5cde4107107ac6c2cb51c7d16bac752d554.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/images/a38b43922cd20ffb98ef3db0d762d5cde4107107ac6c2cb51c7d16bac752d554.jpg)

![b098f59fe784861d9f5ec89cb5ca98d8f26970d5abd66874f7e2b3513af7cc99.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/images/b098f59fe784861d9f5ec89cb5ca98d8f26970d5abd66874f7e2b3513af7cc99.jpg)

![b30df57f382332e12f7f1ad70afd7624bbbd7fabe85576ef92d3d0323238a31a.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/images/b30df57f382332e12f7f1ad70afd7624bbbd7fabe85576ef92d3d0323238a31a.jpg)

![c2dd1321860d5e07c74017a72487a36a32cbb1e17e269217cd3bceb1a4b89c86.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/images/c2dd1321860d5e07c74017a72487a36a32cbb1e17e269217cd3bceb1a4b89c86.jpg)

![e9effb21a17aa8604d8747bc5644000f01b0f3f75f73d3b2d659a77a9c797a54.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/images/e9effb21a17aa8604d8747bc5644000f01b0f3f75f73d3b2d659a77a9c797a54.jpg)

![fcecdec7ef6012b4bbe348706afddf940193a882d390c55f1006152a814719d6.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/images/fcecdec7ef6012b4bbe348706afddf940193a882d390c55f1006152a814719d6.jpg)

### Tables

![0582475b9dff84594e2788d33a6bd1a0057660816d521520b081093518c59bdd.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/tables/0582475b9dff84594e2788d33a6bd1a0057660816d521520b081093518c59bdd.jpg)

![51f2099b3f20478a0d48db8d0ea44444abb87595b2c0f291808852a2a8920283.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/tables/51f2099b3f20478a0d48db8d0ea44444abb87595b2c0f291808852a2a8920283.jpg)

![523d99994fa43abfb91b57ce828b5a8575a2315c18a0b88753e3547a3d8406b0.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/tables/523d99994fa43abfb91b57ce828b5a8575a2315c18a0b88753e3547a3d8406b0.jpg)

![52bbbc5470054672d6d146a4cedadc694ee0e503f2241adb2e145804350d22b0.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/tables/52bbbc5470054672d6d146a4cedadc694ee0e503f2241adb2e145804350d22b0.jpg)

![751726ead84e79f1bb58b9942546d57c38ea5cfec0ad6443839f6d2348c86a53.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/tables/751726ead84e79f1bb58b9942546d57c38ea5cfec0ad6443839f6d2348c86a53.jpg)

![77c8d22e9b0f8cf64a51c9c8b125cc286d9ae286e4ec7b7c895e3a7d52706678.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/tables/77c8d22e9b0f8cf64a51c9c8b125cc286d9ae286e4ec7b7c895e3a7d52706678.jpg)

![939c9f34e6a5a3650232fd0ca27bab7dbfe56f8833e104cd2ec425167cf71685.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/tables/939c9f34e6a5a3650232fd0ca27bab7dbfe56f8833e104cd2ec425167cf71685.jpg)

![b324b09ef0578129c855930cb898901e740d6a1fe52c84b10ffd576427f7d91e.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/tables/b324b09ef0578129c855930cb898901e740d6a1fe52c84b10ffd576427f7d91e.jpg)

![d4fe04aefd122a85f1184c2c35d911ffa20b36f765af32004218bee4395eb583.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/tables/d4fe04aefd122a85f1184c2c35d911ffa20b36f765af32004218bee4395eb583.jpg)

![d71916abd91eaf3e99f257b0701709e233ba264b71c58bd546df427f46fd9a66.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/tables/d71916abd91eaf3e99f257b0701709e233ba264b71c58bd546df427f46fd9a66.jpg)

![f914c7df2dff7604c71e6dea03a35dd910755118669dd3402038a07715be0472.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/tables/f914c7df2dff7604c71e6dea03a35dd910755118669dd3402038a07715be0472.jpg)

![fbee97eed9b9f71d91f6fe74ff4c4d28aa79f388126044b2c9bb341bdabae294.jpg](../icml_results/2624_CLIP%20Tricks%20You_%20Training-free%20Token%20Pruning%20for%20Efficient%20Pixel%20Grounding%20in%20Large%20Vision-Language%20/tables/fbee97eed9b9f71d91f6fe74ff4c4d28aa79f388126044b2c9bb341bdabae294.jpg)

<a id="paper-2591"></a>

## ReGen: Hierarchical Multi-Prompt Representation Generation for Efficient Waveform Diffusion Models


### Images

![6456e71825bf277aba5c97f3f96f022aa3621cb8af9a2e7f3a82c71eccd9e7ef.jpg](../icml_results/2625_ReGen_%20Hierarchical%20Multi-Prompt%20Representation%20Generation%20for%20Efficient%20Waveform%20Diffusion%20Models/images/6456e71825bf277aba5c97f3f96f022aa3621cb8af9a2e7f3a82c71eccd9e7ef.jpg)

### Tables

![0658c43473b52aefda958f487010e148ca54a1307d8f1326bf288f9b15d6cb8d.jpg](../icml_results/2625_ReGen_%20Hierarchical%20Multi-Prompt%20Representation%20Generation%20for%20Efficient%20Waveform%20Diffusion%20Models/tables/0658c43473b52aefda958f487010e148ca54a1307d8f1326bf288f9b15d6cb8d.jpg)

![2212327eae2c38781ced1a7038cfde7fa333efcd9bc43e61e240a7f70b006e02.jpg](../icml_results/2625_ReGen_%20Hierarchical%20Multi-Prompt%20Representation%20Generation%20for%20Efficient%20Waveform%20Diffusion%20Models/tables/2212327eae2c38781ced1a7038cfde7fa333efcd9bc43e61e240a7f70b006e02.jpg)

![297b012e31ad09cbed183a6cff96396b10e7c3845bbf026defa3b8005a3d75f2.jpg](../icml_results/2625_ReGen_%20Hierarchical%20Multi-Prompt%20Representation%20Generation%20for%20Efficient%20Waveform%20Diffusion%20Models/tables/297b012e31ad09cbed183a6cff96396b10e7c3845bbf026defa3b8005a3d75f2.jpg)

![4bee9c00a54569860018671ec09330cf4a58de13d0ced2865a9e0134ee9ed0ed.jpg](../icml_results/2625_ReGen_%20Hierarchical%20Multi-Prompt%20Representation%20Generation%20for%20Efficient%20Waveform%20Diffusion%20Models/tables/4bee9c00a54569860018671ec09330cf4a58de13d0ced2865a9e0134ee9ed0ed.jpg)

![692efee3994f646c548a6f39df399b152f9698f2fa73122cb93e39560feec4ba.jpg](../icml_results/2625_ReGen_%20Hierarchical%20Multi-Prompt%20Representation%20Generation%20for%20Efficient%20Waveform%20Diffusion%20Models/tables/692efee3994f646c548a6f39df399b152f9698f2fa73122cb93e39560feec4ba.jpg)

![8575eff1a5c3025755799d85b27c8b6216bc7be60149412517d537a90813c862.jpg](../icml_results/2625_ReGen_%20Hierarchical%20Multi-Prompt%20Representation%20Generation%20for%20Efficient%20Waveform%20Diffusion%20Models/tables/8575eff1a5c3025755799d85b27c8b6216bc7be60149412517d537a90813c862.jpg)

![92d025ed0e2eb005bac0d7097e1a01311878bcb33a09bb0d6586000732354f4f.jpg](../icml_results/2625_ReGen_%20Hierarchical%20Multi-Prompt%20Representation%20Generation%20for%20Efficient%20Waveform%20Diffusion%20Models/tables/92d025ed0e2eb005bac0d7097e1a01311878bcb33a09bb0d6586000732354f4f.jpg)

![96503db39aef888de64d9625dbe9859eec8a43a5a7c07b98e560a3a2f3d6cdd7.jpg](../icml_results/2625_ReGen_%20Hierarchical%20Multi-Prompt%20Representation%20Generation%20for%20Efficient%20Waveform%20Diffusion%20Models/tables/96503db39aef888de64d9625dbe9859eec8a43a5a7c07b98e560a3a2f3d6cdd7.jpg)

![a5a600b949a6d071a04b055d7aad5a795f5d43420bcc06636160efd3b1f57e16.jpg](../icml_results/2625_ReGen_%20Hierarchical%20Multi-Prompt%20Representation%20Generation%20for%20Efficient%20Waveform%20Diffusion%20Models/tables/a5a600b949a6d071a04b055d7aad5a795f5d43420bcc06636160efd3b1f57e16.jpg)

![ae04b2100ca496d7b0d080bbd549516bfef469ec871b37204508ba7a7ffa0315.jpg](../icml_results/2625_ReGen_%20Hierarchical%20Multi-Prompt%20Representation%20Generation%20for%20Efficient%20Waveform%20Diffusion%20Models/tables/ae04b2100ca496d7b0d080bbd549516bfef469ec871b37204508ba7a7ffa0315.jpg)

![e41a0758f7c8f512cc7240ecbe550d34e54f1ca392f9a33be209d19c1340da40.jpg](../icml_results/2625_ReGen_%20Hierarchical%20Multi-Prompt%20Representation%20Generation%20for%20Efficient%20Waveform%20Diffusion%20Models/tables/e41a0758f7c8f512cc7240ecbe550d34e54f1ca392f9a33be209d19c1340da40.jpg)

<a id="paper-2592"></a>

## Toward Understanding Adversarial Distillation: Why Robust Teachers Fail


### Images

![341c0789edaaeb7e0e0d4576627e4ff2e430ec431d7532064c4ca6a7986344e6.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/images/341c0789edaaeb7e0e0d4576627e4ff2e430ec431d7532064c4ca6a7986344e6.jpg)

![34a831d33cf126b7554e1efbd11d723d43f73f93efbe7a9d786187647e693c19.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/images/34a831d33cf126b7554e1efbd11d723d43f73f93efbe7a9d786187647e693c19.jpg)

![6b25fd0c8dcabe24569e15a72df07d785ca7e5c8d3579a48332c3851152cb38d.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/images/6b25fd0c8dcabe24569e15a72df07d785ca7e5c8d3579a48332c3851152cb38d.jpg)

![8766ae52cb07af617ba6d15ecf5b7b3710484747fc5901faae21fe8600963030.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/images/8766ae52cb07af617ba6d15ecf5b7b3710484747fc5901faae21fe8600963030.jpg)

![9c872e76142d512573fb680072e8e9219fa5d655ab0e943e2ec450578f1e009c.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/images/9c872e76142d512573fb680072e8e9219fa5d655ab0e943e2ec450578f1e009c.jpg)

![bcd8758ab216ce9002912b7afc5ad5d3165dc8a3dd44e39691939f028119b9f2.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/images/bcd8758ab216ce9002912b7afc5ad5d3165dc8a3dd44e39691939f028119b9f2.jpg)

![e19e3713b633da52b4de94023994d75029b8c1d785d40c9ca5eeb5c99f9ea06e.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/images/e19e3713b633da52b4de94023994d75029b8c1d785d40c9ca5eeb5c99f9ea06e.jpg)

![e87e5b451df91e5200bafcca88fc0b6f69339505c8dcfc2d803e9d7428f675c4.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/images/e87e5b451df91e5200bafcca88fc0b6f69339505c8dcfc2d803e9d7428f675c4.jpg)

![f8d44a1d2068f6ea3260c1996a0c7b7c5ed60e62daa241f554cf3ee4c794f8af.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/images/f8d44a1d2068f6ea3260c1996a0c7b7c5ed60e62daa241f554cf3ee4c794f8af.jpg)

### Tables

![0165906fb436e0f84dc5c6c455e9dfcde2669ec2404865584fc5cef97d7dc0cd.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/tables/0165906fb436e0f84dc5c6c455e9dfcde2669ec2404865584fc5cef97d7dc0cd.jpg)

![06f3318ec63d9cfbaf2fb878964919b62adc5c28caa4d2a91c053ff3c841a098.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/tables/06f3318ec63d9cfbaf2fb878964919b62adc5c28caa4d2a91c053ff3c841a098.jpg)

![1dad783e959b069e36cc0f612fb2c68fab636fbcf4d9033a828880f1e7670192.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/tables/1dad783e959b069e36cc0f612fb2c68fab636fbcf4d9033a828880f1e7670192.jpg)

![52513fd35b1cc73de33e7a368ea9368381ef652710c0ed415cd97bcf00ee53ec.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/tables/52513fd35b1cc73de33e7a368ea9368381ef652710c0ed415cd97bcf00ee53ec.jpg)

![6786cfa5c63520d89d1771615142a01f6fd9ba35370d96e4347ea571b9684b93.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/tables/6786cfa5c63520d89d1771615142a01f6fd9ba35370d96e4347ea571b9684b93.jpg)

![978ffe41306304b738ed4a036435689c04c182f0d48e0f6f4d73123ad443b130.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/tables/978ffe41306304b738ed4a036435689c04c182f0d48e0f6f4d73123ad443b130.jpg)

![a1fb2d85789addcdc2c87d0f7dee38169c781b1e472583b737a4a477d6ffd144.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/tables/a1fb2d85789addcdc2c87d0f7dee38169c781b1e472583b737a4a477d6ffd144.jpg)

![b4cd21eb6b787b539b35924005c6a927754e5ba913017cdf5cbde02ac38f17da.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/tables/b4cd21eb6b787b539b35924005c6a927754e5ba913017cdf5cbde02ac38f17da.jpg)

![b83b6a3d950d76c09caaa11ddd458e8b1e08914c1fadad3250a0c44f8071b499.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/tables/b83b6a3d950d76c09caaa11ddd458e8b1e08914c1fadad3250a0c44f8071b499.jpg)

![c75da3bc943227b26d09508c8c6048bdd1efc491a570c3f3e263a81f489598e0.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/tables/c75da3bc943227b26d09508c8c6048bdd1efc491a570c3f3e263a81f489598e0.jpg)

![d632d53a4dff1e9b39b62957dc1b8402f5ebd2cf39d2d381364d1b980fc484b1.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/tables/d632d53a4dff1e9b39b62957dc1b8402f5ebd2cf39d2d381364d1b980fc484b1.jpg)

![ed9682a5c2145bf57455195439b5f0d9d25b00c2df1c099e5034a5f31a5a5fef.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/tables/ed9682a5c2145bf57455195439b5f0d9d25b00c2df1c099e5034a5f31a5a5fef.jpg)

![faf3fb291b5b6877fa5e8e2c4cf732e579889c7633e920bf861e5515ea7bf801.jpg](../icml_results/2626_Toward%20Understanding%20Adversarial%20Distillation_%20Why%20Robust%20Teachers%20Fail/tables/faf3fb291b5b6877fa5e8e2c4cf732e579889c7633e920bf861e5515ea7bf801.jpg)

<a id="paper-2593"></a>

## Dynamic Relational Priming Improves Transformer in Multivariate Time Series


### Images

![059167790a99255b31dc752c1b2d927e6f7bab6b500ba8f3b1fc39025d4be2e8.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/images/059167790a99255b31dc752c1b2d927e6f7bab6b500ba8f3b1fc39025d4be2e8.jpg)

![159482e8bc15b11bfcbcbf63b97c42f728a511420dc2322f672234740960f02c.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/images/159482e8bc15b11bfcbcbf63b97c42f728a511420dc2322f672234740960f02c.jpg)

![23c547c98a334f0907843b0c85908ae2ae181b1dc84f1b5b8151914461397848.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/images/23c547c98a334f0907843b0c85908ae2ae181b1dc84f1b5b8151914461397848.jpg)

![53304edd03f58796ecbcd8b41adb37ffcf1ab4423646ee41ab9bcda1ec564050.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/images/53304edd03f58796ecbcd8b41adb37ffcf1ab4423646ee41ab9bcda1ec564050.jpg)

![620942f216a0afc61e694d1ff2f65b7ff10909f8c1fcd12196c23ab78f703d97.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/images/620942f216a0afc61e694d1ff2f65b7ff10909f8c1fcd12196c23ab78f703d97.jpg)

![665ed8ecd9eb54e8ebfb7c95ab46b0d6992d2b9f7105faac1e8d9c9c55a2f541.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/images/665ed8ecd9eb54e8ebfb7c95ab46b0d6992d2b9f7105faac1e8d9c9c55a2f541.jpg)

![7aac1bfb0b24c429cba62776c69df478c2d4fa2be1f7cd028a48075b61f3e23e.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/images/7aac1bfb0b24c429cba62776c69df478c2d4fa2be1f7cd028a48075b61f3e23e.jpg)

![88977bf409d6f52f5ac2af900a89e9e6ddac0404425bf49493bc5d0825ee99b3.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/images/88977bf409d6f52f5ac2af900a89e9e6ddac0404425bf49493bc5d0825ee99b3.jpg)

![b5c9fcaf97ab23a857ad67c79c57f31da6e35363cb9d0066fd0cc30e8cc4aa8a.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/images/b5c9fcaf97ab23a857ad67c79c57f31da6e35363cb9d0066fd0cc30e8cc4aa8a.jpg)

![cabe7e97ceb6a099123a918597da7fcd552d98c2d9af95220ef886a139ef04f4.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/images/cabe7e97ceb6a099123a918597da7fcd552d98c2d9af95220ef886a139ef04f4.jpg)

![ceee1c5d24a51dafdc2996485785dcc4a8e68a84ebe43c77ef89789370935987.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/images/ceee1c5d24a51dafdc2996485785dcc4a8e68a84ebe43c77ef89789370935987.jpg)

![f7e397cec7505c215c603e0ef06b8a9ee8660d7a404772288de71af606ec3f2f.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/images/f7e397cec7505c215c603e0ef06b8a9ee8660d7a404772288de71af606ec3f2f.jpg)

![febeeb3a8e306467c0d68bce34d42f23ef69911c9f9eb40371ec74287b0029cd.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/images/febeeb3a8e306467c0d68bce34d42f23ef69911c9f9eb40371ec74287b0029cd.jpg)

### Tables

![314655b1e02340e0e1fd6d4167cda979aecf86b822a8fccbb3fd2d5e610448de.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/tables/314655b1e02340e0e1fd6d4167cda979aecf86b822a8fccbb3fd2d5e610448de.jpg)

![41de376965d32b5b7c7d7edbb03990001864870397edd9300ecfb471b9044a43.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/tables/41de376965d32b5b7c7d7edbb03990001864870397edd9300ecfb471b9044a43.jpg)

![5518c57f6f15dc3ebfd6b276e89cef6a32fb9e50a422817a29a71d254ed60f71.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/tables/5518c57f6f15dc3ebfd6b276e89cef6a32fb9e50a422817a29a71d254ed60f71.jpg)

![c2f34916230df6de1a8cc4ff0a119a3b78fbfee6f6768b00ac0292df8a20d374.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/tables/c2f34916230df6de1a8cc4ff0a119a3b78fbfee6f6768b00ac0292df8a20d374.jpg)

![cd5b3344d6ddd00cd3d7e366bc3a248780b9e0c12bf33439429fc28c098b5d46.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/tables/cd5b3344d6ddd00cd3d7e366bc3a248780b9e0c12bf33439429fc28c098b5d46.jpg)

![d0cbf35353de753117419421761db93825252c84d2fd47c849847e9f177af1bb.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/tables/d0cbf35353de753117419421761db93825252c84d2fd47c849847e9f177af1bb.jpg)

![dd427bf52c09baa9a7f7d7531423797518e4a96f0f08a38b309782d7c50c0459.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/tables/dd427bf52c09baa9a7f7d7531423797518e4a96f0f08a38b309782d7c50c0459.jpg)

![dec85b3cfcd018948c9d72e4a0467852f4515944c8118fa66dfb4fa6531756d2.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/tables/dec85b3cfcd018948c9d72e4a0467852f4515944c8118fa66dfb4fa6531756d2.jpg)

![e676a4243507bf9fc25ac846746ff30850f7efd619d4b75b960e17efb7a7d84c.jpg](../icml_results/2627_Dynamic%20Relational%20Priming%20Improves%20Transformer%20in%20Multivariate%20Time%20Series/tables/e676a4243507bf9fc25ac846746ff30850f7efd619d4b75b960e17efb7a7d84c.jpg)

<a id="paper-2594"></a>

## DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention


### Images

![28bf28aceb051bb8c241d93dc96d9c4df527773fb06c9fb7e5b6e1ec250891d4.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/images/28bf28aceb051bb8c241d93dc96d9c4df527773fb06c9fb7e5b6e1ec250891d4.jpg)

![36505a96d08055a34496bc985fb5d6c43d9df6bd49da87fda952612793010a1c.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/images/36505a96d08055a34496bc985fb5d6c43d9df6bd49da87fda952612793010a1c.jpg)

![4628994bd9d0936b940c276e6b844f8f55d2e760c645bcebda8e728e1232ffcf.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/images/4628994bd9d0936b940c276e6b844f8f55d2e760c645bcebda8e728e1232ffcf.jpg)

![61948f40e4c1f1465796a24ba8ce56d5d7ee810ab4d9d4215e2cab226770407c.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/images/61948f40e4c1f1465796a24ba8ce56d5d7ee810ab4d9d4215e2cab226770407c.jpg)

![8a89ce067b98b42043d06db0bfdab460a7b2fc0d086a62b858f51c555bf9eccc.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/images/8a89ce067b98b42043d06db0bfdab460a7b2fc0d086a62b858f51c555bf9eccc.jpg)

![a4b13d4aeddddd1d2f966755d2c416b396fad9a7aaad0e94c70e47ef0b8eef09.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/images/a4b13d4aeddddd1d2f966755d2c416b396fad9a7aaad0e94c70e47ef0b8eef09.jpg)

![b696b2fdd2fad32c9c19dc8a62568d06ae279158d87fe13983fab77c4e2ec0f1.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/images/b696b2fdd2fad32c9c19dc8a62568d06ae279158d87fe13983fab77c4e2ec0f1.jpg)

![c0b66cd9d41d7056baaae63b1148682e979cacf58b47237a2f6f42e7edd332ce.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/images/c0b66cd9d41d7056baaae63b1148682e979cacf58b47237a2f6f42e7edd332ce.jpg)

![e7ec8a2aef94bb2a04fad434cc8893c1ef197e9190afc69e2341c97bf6109283.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/images/e7ec8a2aef94bb2a04fad434cc8893c1ef197e9190afc69e2341c97bf6109283.jpg)

![f47ef78502b12a154ca947bc197fe8c47952a960943e4606df1bcf02cbc812d0.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/images/f47ef78502b12a154ca947bc197fe8c47952a960943e4606df1bcf02cbc812d0.jpg)

### Tables

![368dffe1a2b855fc5b794cc3aa3971922518bf50090ade15303f80f2dbf72067.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/tables/368dffe1a2b855fc5b794cc3aa3971922518bf50090ade15303f80f2dbf72067.jpg)

![49429651ca3e256c16cbfc99a2e1e0a67b3e4bb76c1ccc45fe0b4762c9168230.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/tables/49429651ca3e256c16cbfc99a2e1e0a67b3e4bb76c1ccc45fe0b4762c9168230.jpg)

![4d58cbd90023d4dc71f50470409bdf8f87b88ce5f73a58a83e37efdab0629de6.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/tables/4d58cbd90023d4dc71f50470409bdf8f87b88ce5f73a58a83e37efdab0629de6.jpg)

![64225bbc92be33d04befa3b37fe4e46e12a310b521988c4633d582e8b86eface.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/tables/64225bbc92be33d04befa3b37fe4e46e12a310b521988c4633d582e8b86eface.jpg)

![7146f9d485e0aec5b3f147d5c5c6d43235d3a1658c029416f40867f8cee31144.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/tables/7146f9d485e0aec5b3f147d5c5c6d43235d3a1658c029416f40867f8cee31144.jpg)

![721c2cebe3fc11fc5775fca8b5d97d079d5771d11e223bd971197b6f828f2682.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/tables/721c2cebe3fc11fc5775fca8b5d97d079d5771d11e223bd971197b6f828f2682.jpg)

![a629317729a3594908bc1c4d0f06baa41dc967b573f9723a581a4d6a6f2e78d0.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/tables/a629317729a3594908bc1c4d0f06baa41dc967b573f9723a581a4d6a6f2e78d0.jpg)

![c1e8f6d0b0d23d3df52aff8ed054359374a482ce94c320b3035184ff83415c42.jpg](../icml_results/2628_DyLLM_%20Efficient%20Diffusion%20LLM%20Inference%20via%20Saliency-based%20Token%20Selection%20and%20Partial%20Attention/tables/c1e8f6d0b0d23d3df52aff8ed054359374a482ce94c320b3035184ff83415c42.jpg)

<a id="paper-2595"></a>

## Stochastic Sparse Attention for Memory-Bound Inference


### Images

![04536c6795cfebb395d9c939984476ebf4160d1c315a6de06f20d6547f611739.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/images/04536c6795cfebb395d9c939984476ebf4160d1c315a6de06f20d6547f611739.jpg)

![115c6b68d063f027fbc6a73b6259f47ecb5973734945369dc42a4e2983d04592.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/images/115c6b68d063f027fbc6a73b6259f47ecb5973734945369dc42a4e2983d04592.jpg)

![1608f1e9bf414420bcc231acc760f1e3c78e94646802bce1d0613bd9dbdd921f.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/images/1608f1e9bf414420bcc231acc760f1e3c78e94646802bce1d0613bd9dbdd921f.jpg)

![6f1309e31dd3d69a1d9f192326d5aa625447a2add479c1455b046c9d1bd4af56.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/images/6f1309e31dd3d69a1d9f192326d5aa625447a2add479c1455b046c9d1bd4af56.jpg)

![892635068644e1b7476418c35926aa77a441023542474e32701bcaadf61837a7.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/images/892635068644e1b7476418c35926aa77a441023542474e32701bcaadf61837a7.jpg)

![b3b6e6f9e8f5ffd780b14c7965083c7a204617a7de04c6b04ab910080cd4b056.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/images/b3b6e6f9e8f5ffd780b14c7965083c7a204617a7de04c6b04ab910080cd4b056.jpg)

![b4ca2331ab9425159a73a655cd1ffcb0a4d3112b42bbeb938f9e44451d24a9ea.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/images/b4ca2331ab9425159a73a655cd1ffcb0a4d3112b42bbeb938f9e44451d24a9ea.jpg)

![b4e7b1856fcafd628ea79a6319688e0d7abc53a2bb192c96bf135d6deb3db847.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/images/b4e7b1856fcafd628ea79a6319688e0d7abc53a2bb192c96bf135d6deb3db847.jpg)

![deb4488ab499ef7c64888e14a14020272ac037cc4fe2fe9b8a8d0bd5f1b6f8dc.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/images/deb4488ab499ef7c64888e14a14020272ac037cc4fe2fe9b8a8d0bd5f1b6f8dc.jpg)

![f3ea525b7ce80719e67c03f8ba7703367734db87cefee025b263f480ae264e76.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/images/f3ea525b7ce80719e67c03f8ba7703367734db87cefee025b263f480ae264e76.jpg)

### Tables

![0b375df57b0602b123bc8e84e7da89cb97d3a456fc238d1f72af4923c4c325c3.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/0b375df57b0602b123bc8e84e7da89cb97d3a456fc238d1f72af4923c4c325c3.jpg)

![13d9005141f660594383cd4dedd591f5bb7acac2653b92ad77c44f1441f61191.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/13d9005141f660594383cd4dedd591f5bb7acac2653b92ad77c44f1441f61191.jpg)

![26233a3e8010199917ea4debb22e0732477a28de4127541601ef89aa7f1eb03b.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/26233a3e8010199917ea4debb22e0732477a28de4127541601ef89aa7f1eb03b.jpg)

![2a01ab0d08d558dd892e227b26024565221319e11c98cad4e6ec8a5cd341eb70.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/2a01ab0d08d558dd892e227b26024565221319e11c98cad4e6ec8a5cd341eb70.jpg)

![4014f649495d712b818d50d1faefcb4ee1895e71347a6e96cfe49358db52c5d9.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/4014f649495d712b818d50d1faefcb4ee1895e71347a6e96cfe49358db52c5d9.jpg)

![47f1da7a68b3489f328c4fa99580789897046599ff5b9a70ee98617acc3f7a4e.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/47f1da7a68b3489f328c4fa99580789897046599ff5b9a70ee98617acc3f7a4e.jpg)

![578d00430ca7ce3376290046bc0f9a7086401ce62703fc5bbb119bb80ce1c3b5.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/578d00430ca7ce3376290046bc0f9a7086401ce62703fc5bbb119bb80ce1c3b5.jpg)

![57e3b6e3176f2fae72a1bd8b5a352c11eaa12e3f4471ec789e313fefb6ba95a9.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/57e3b6e3176f2fae72a1bd8b5a352c11eaa12e3f4471ec789e313fefb6ba95a9.jpg)

![5e4d20215fbd72cc577b9f2410cc405ef89c59bdf715adc236ee54837d6b69cb.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/5e4d20215fbd72cc577b9f2410cc405ef89c59bdf715adc236ee54837d6b69cb.jpg)

![5f68db31cb2d110f81f53ea88a313b487959f9e3627b483f57a688935e4edbb9.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/5f68db31cb2d110f81f53ea88a313b487959f9e3627b483f57a688935e4edbb9.jpg)

![736071ee2e199661ec8358283cb5a7b1d3f2923b1b5abaee6d3a4c380cd7eefb.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/736071ee2e199661ec8358283cb5a7b1d3f2923b1b5abaee6d3a4c380cd7eefb.jpg)

![79cf5e4de52e81500dab4a286c3ced7a9aeab12ea8e89935e4cfaf5e5cccbe6f.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/79cf5e4de52e81500dab4a286c3ced7a9aeab12ea8e89935e4cfaf5e5cccbe6f.jpg)

![8032e6bf8b45eeb640b365c90f678067fd40d637d136237c5a2b14939d77afcc.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/8032e6bf8b45eeb640b365c90f678067fd40d637d136237c5a2b14939d77afcc.jpg)

![81435f89183a155252660207b76130ab35ee61a7af3c38d9fa9f9e315522f364.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/81435f89183a155252660207b76130ab35ee61a7af3c38d9fa9f9e315522f364.jpg)

![8305b390b0713faf1853af468fe7e7716030520ccdbe026b7fb810387d5acfb1.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/8305b390b0713faf1853af468fe7e7716030520ccdbe026b7fb810387d5acfb1.jpg)

![8c21ceec5515fca6878eaa2264b3d42b4af2c68e0f864f222c10b9c98ac548b2.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/8c21ceec5515fca6878eaa2264b3d42b4af2c68e0f864f222c10b9c98ac548b2.jpg)

![911d88c3b22e47c7b5cd922cba2a62e0e2dbd814d1aa642663fe84f5e2328555.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/911d88c3b22e47c7b5cd922cba2a62e0e2dbd814d1aa642663fe84f5e2328555.jpg)

![984cea30db4e8490954e1a5ac6bd92da71459a0c6392bded024c211924e3c1b3.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/984cea30db4e8490954e1a5ac6bd92da71459a0c6392bded024c211924e3c1b3.jpg)

![a501e609af6dbf19fba442d28f95a502317c9035a5cbc0eab29ba95dd42daed1.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/a501e609af6dbf19fba442d28f95a502317c9035a5cbc0eab29ba95dd42daed1.jpg)

![abc75bea697c758d10329525a61b7ea5689ee7ed8e23fd054f18cca7dcc18aba.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/abc75bea697c758d10329525a61b7ea5689ee7ed8e23fd054f18cca7dcc18aba.jpg)

![b93a7d12b648b32b114594c53ca5e9f5e14a645efa5c9e570d2bc9884829a886.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/b93a7d12b648b32b114594c53ca5e9f5e14a645efa5c9e570d2bc9884829a886.jpg)

![bb5f18a0a3f00e79d9e7bdb8a3189e6062d88823ab420841390f4c78353b2407.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/bb5f18a0a3f00e79d9e7bdb8a3189e6062d88823ab420841390f4c78353b2407.jpg)

![ca18763ca8d99165231ef61773f91d3df5078c62cb7a2660e0adc69c93b0bab0.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/ca18763ca8d99165231ef61773f91d3df5078c62cb7a2660e0adc69c93b0bab0.jpg)

![df2571955b0537d5b664299cced0536e57ed99f1cd61d26a698dd8eb58fc0c24.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/df2571955b0537d5b664299cced0536e57ed99f1cd61d26a698dd8eb58fc0c24.jpg)

![df84a0877f1922fa1ef90e49c48d9c7f16afcc4defda13df6d9cb446c29d90c0.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/df84a0877f1922fa1ef90e49c48d9c7f16afcc4defda13df6d9cb446c29d90c0.jpg)

![e0d30fbabd297dda246f0d2e7449f06a85d3679e916b5c5e431631465c4dc23d.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/e0d30fbabd297dda246f0d2e7449f06a85d3679e916b5c5e431631465c4dc23d.jpg)

![e2c7183372058a855db38c6e484ab88acb2bc7ed8f55e0d6f14b9ec0df4697ca.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/e2c7183372058a855db38c6e484ab88acb2bc7ed8f55e0d6f14b9ec0df4697ca.jpg)

![ec7e8ea9c998c9c32415fe1b798464b949d61c5778428a64d7166274adb3fd15.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/ec7e8ea9c998c9c32415fe1b798464b949d61c5778428a64d7166274adb3fd15.jpg)

![f6fbbe2b3380d4e5572221118c631b84229625dc107d82ba74c54bc9bcb32a61.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/f6fbbe2b3380d4e5572221118c631b84229625dc107d82ba74c54bc9bcb32a61.jpg)

![fa5c04480c6eb9bf18c3914241665ad6050cbb3813e85d910129e285a9129d5b.jpg](../icml_results/2629_Stochastic%20Sparse%20Attention%20for%20Memory-Bound%20Inference/tables/fa5c04480c6eb9bf18c3914241665ad6050cbb3813e85d910129e285a9129d5b.jpg)

<a id="paper-2596"></a>

## Variable-Length Tokenization via Learnable Global Merging for Diffusion Transformers


### Images

![12ced509d23e682e7288d7f23f14737d4823aec0ba313d07866023390b52d0c2.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/images/12ced509d23e682e7288d7f23f14737d4823aec0ba313d07866023390b52d0c2.jpg)

![157da5042ae10b915500941240a61cbeef8620b81c5ae25316b6316b0a9a5948.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/images/157da5042ae10b915500941240a61cbeef8620b81c5ae25316b6316b0a9a5948.jpg)

![1928554686e96a1d3cbbd7f301ee1b9856ff3d583ccdebbfb3883c16498b4d59.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/images/1928554686e96a1d3cbbd7f301ee1b9856ff3d583ccdebbfb3883c16498b4d59.jpg)

![4483c1c9a8b6215bf28b9d2cc1d2c91398f207904687705263ec1b053c7f3557.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/images/4483c1c9a8b6215bf28b9d2cc1d2c91398f207904687705263ec1b053c7f3557.jpg)

![5fa54f60f36e088a6d5b0ed6564a5cb8c4c3e5f96510bc30d0e6402af3ff9d0a.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/images/5fa54f60f36e088a6d5b0ed6564a5cb8c4c3e5f96510bc30d0e6402af3ff9d0a.jpg)

![61a0a14bd07847bf27d7af1ed720fcfd4421dfe78d91dad45f379587bb6b5832.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/images/61a0a14bd07847bf27d7af1ed720fcfd4421dfe78d91dad45f379587bb6b5832.jpg)

![83774ea8a5aa6be3df2478c9bd5e5aa51a383ecca8f632532f84f08ba4eaabf1.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/images/83774ea8a5aa6be3df2478c9bd5e5aa51a383ecca8f632532f84f08ba4eaabf1.jpg)

![8c937ae1cd5243961fa0375c09d8ed0c067e1ebff7d3aa513218cbe9e3d26d65.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/images/8c937ae1cd5243961fa0375c09d8ed0c067e1ebff7d3aa513218cbe9e3d26d65.jpg)

![9f252a955a0959d541de4c714583d72a5824982aa0faadf00735b3ba8ca4ebcf.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/images/9f252a955a0959d541de4c714583d72a5824982aa0faadf00735b3ba8ca4ebcf.jpg)

![c4609e9421af5246c8c73e6ed8f7405a320b8d346c8c4bcf1138cb2b8657a843.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/images/c4609e9421af5246c8c73e6ed8f7405a320b8d346c8c4bcf1138cb2b8657a843.jpg)

![d4ac25a0b333b11852dd7bfd48a6dd17d533ee1f8d229d279f527ebe96b151df.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/images/d4ac25a0b333b11852dd7bfd48a6dd17d533ee1f8d229d279f527ebe96b151df.jpg)

![d7e6cfb75bdc37484156e66295fb1959445577b3df65cbab9d2bd9d3ff4b8726.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/images/d7e6cfb75bdc37484156e66295fb1959445577b3df65cbab9d2bd9d3ff4b8726.jpg)

![def3650b280b9d516c1189b0f8019fa6c9c67f600d8c410e309ab97562583833.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/images/def3650b280b9d516c1189b0f8019fa6c9c67f600d8c410e309ab97562583833.jpg)

![f7dc04d82332ec9f4b4e5b52958fdfe581d157e88b727d91c0187e917c32a2de.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/images/f7dc04d82332ec9f4b4e5b52958fdfe581d157e88b727d91c0187e917c32a2de.jpg)

### Tables

![2d6d1b7c270b3c74f7e6e90fd2ee77983f08d7f4cc90ea6be95c33506b48ec10.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/tables/2d6d1b7c270b3c74f7e6e90fd2ee77983f08d7f4cc90ea6be95c33506b48ec10.jpg)

![4204fa389ec2db2f3300baabac49194f3326540448bbb6fcb6e8fcc8def823bb.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/tables/4204fa389ec2db2f3300baabac49194f3326540448bbb6fcb6e8fcc8def823bb.jpg)

![5faa6e77eb763e83723253cf5062faa2789f6c34d1daeeb0a055d0414a9d4a8b.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/tables/5faa6e77eb763e83723253cf5062faa2789f6c34d1daeeb0a055d0414a9d4a8b.jpg)

![dbfcffecb16ef3b9c5d5f1c6f6ab2bf20a753970bcbf0572c72cb8c9c4457236.jpg](../icml_results/2630_Variable-Length%20Tokenization%20via%20Learnable%20Global%20Merging%20for%20Diffusion%20Transformers/tables/dbfcffecb16ef3b9c5d5f1c6f6ab2bf20a753970bcbf0572c72cb8c9c4457236.jpg)

<a id="paper-2597"></a>

## Focusing Where Vision Matters: Selective Training for Large Vision Language Models via Visual Information Gain


### Images

![0a849dd2dcee633a2da96abd9f173841ab4f0ec81d53aec024647c0b77bef9e3.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/0a849dd2dcee633a2da96abd9f173841ab4f0ec81d53aec024647c0b77bef9e3.jpg)

![0fcbc00aa4eef7bd20745c67634452e01cb0b809086ae7789fd32c97940dd8d4.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/0fcbc00aa4eef7bd20745c67634452e01cb0b809086ae7789fd32c97940dd8d4.jpg)

![175a5c3f5045b8bdedf1786ab00507f58797b0f9d294b8709d520a353bca1bd4.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/175a5c3f5045b8bdedf1786ab00507f58797b0f9d294b8709d520a353bca1bd4.jpg)

![394ed19543f530dd24e02c4358c26b5a9de9cd9751dff4b4970fef75e5ea6bcc.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/394ed19543f530dd24e02c4358c26b5a9de9cd9751dff4b4970fef75e5ea6bcc.jpg)

![40282e3b159ea82cd81c9c383ba5032b97256fe0a4bec068701e74651863f973.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/40282e3b159ea82cd81c9c383ba5032b97256fe0a4bec068701e74651863f973.jpg)

![4a9edb720b7e12af3c19d74635e7f8df23d006eb49c0ece970a2bf300c264555.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/4a9edb720b7e12af3c19d74635e7f8df23d006eb49c0ece970a2bf300c264555.jpg)

![57a6948dc1f86d7bb3a4246e173e9f5217fb4fa94196d67b8496a0094e6352be.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/57a6948dc1f86d7bb3a4246e173e9f5217fb4fa94196d67b8496a0094e6352be.jpg)

![5a4f01af0139b9e661707d052995cff052f5a0921dbdb5f5a2bf3286a308054a.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/5a4f01af0139b9e661707d052995cff052f5a0921dbdb5f5a2bf3286a308054a.jpg)

![7d6682a21976e77feed8c2b2b97e48985a080d4b16be072d8241148134e46648.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/7d6682a21976e77feed8c2b2b97e48985a080d4b16be072d8241148134e46648.jpg)

![8143b6fb4f03e7ecce964ef2b9894221e8f08cc029d98b523fc7c453fdb4b658.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/8143b6fb4f03e7ecce964ef2b9894221e8f08cc029d98b523fc7c453fdb4b658.jpg)

![8a7d6c8fcc976deb4d92472d622e42c4f57db2e2ff1e53a183149f22166dd0a9.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/8a7d6c8fcc976deb4d92472d622e42c4f57db2e2ff1e53a183149f22166dd0a9.jpg)

![916720c654106b71fb04799c7dce93c5f97e8da63e1669eb81e92635dbdf1150.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/916720c654106b71fb04799c7dce93c5f97e8da63e1669eb81e92635dbdf1150.jpg)

![9b849a9d5d920d028e38458e30cc73563bcaef642fd254bfd2e6465fb37b418d.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/9b849a9d5d920d028e38458e30cc73563bcaef642fd254bfd2e6465fb37b418d.jpg)

![a5ba410a653ea377345523771e76ddf1252f50d0bc25e5bb86c037514501130e.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/a5ba410a653ea377345523771e76ddf1252f50d0bc25e5bb86c037514501130e.jpg)

![a97a77cec963edac99ab165d5a6efad70a0bd121ded01e948cfcc54de25c0517.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/a97a77cec963edac99ab165d5a6efad70a0bd121ded01e948cfcc54de25c0517.jpg)

![aa5c4e11c17e174861d6d48116ee81651b1b5cac03cebe18652784e26b07858b.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/aa5c4e11c17e174861d6d48116ee81651b1b5cac03cebe18652784e26b07858b.jpg)

![b2848fb5a654aa60e5925ab6c36620db7a7685d1d9e3957ae5fea1e7239c825a.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/b2848fb5a654aa60e5925ab6c36620db7a7685d1d9e3957ae5fea1e7239c825a.jpg)

![b3ff85df53fb94e92d4ca90fa1e9982f73cbd434bd66731cdf2a16298f8cafd3.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/b3ff85df53fb94e92d4ca90fa1e9982f73cbd434bd66731cdf2a16298f8cafd3.jpg)

![ba691506b13ecb144d8e70a15a2effef583a86373ecd31ed23293ef724748960.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/ba691506b13ecb144d8e70a15a2effef583a86373ecd31ed23293ef724748960.jpg)

![c158564c1f6ec29aef4edbab338207020362eac3b43c05def38606f2f86745be.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/c158564c1f6ec29aef4edbab338207020362eac3b43c05def38606f2f86745be.jpg)

![d6ba1cd34f55192f997758dc31d3ce61ebd00162be9556a186bfeb1afa7ce6f7.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/d6ba1cd34f55192f997758dc31d3ce61ebd00162be9556a186bfeb1afa7ce6f7.jpg)

![d9c3a17387e8bdb6dd3bc000ee8bfd02938f10792b11f1dd6a009e8916ee8b5a.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/d9c3a17387e8bdb6dd3bc000ee8bfd02938f10792b11f1dd6a009e8916ee8b5a.jpg)

![ef55e2cc0643f73c495aca3fa0a7e24b80c37ad18537deac394c94ad4c77c68e.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/ef55e2cc0643f73c495aca3fa0a7e24b80c37ad18537deac394c94ad4c77c68e.jpg)

![f01f49bd671b632933b558946b5b2998afe2d4401dc12509866cfbb1ef12379a.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/f01f49bd671b632933b558946b5b2998afe2d4401dc12509866cfbb1ef12379a.jpg)

![f42a54774d8f237ea8fba981095fab0f0d50bdfa40b765cc393abc280fd820f2.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/f42a54774d8f237ea8fba981095fab0f0d50bdfa40b765cc393abc280fd820f2.jpg)

![f4b669846ab76f2c6cbe1c772ebe2844fcb098a4b8bd4578f1e4bcd4f3e00ce6.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/f4b669846ab76f2c6cbe1c772ebe2844fcb098a4b8bd4578f1e4bcd4f3e00ce6.jpg)

![f8035b408d2cb7c25ab550d4abb3190ab5e7af5f3a2cad3613dbd606255e4705.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/f8035b408d2cb7c25ab550d4abb3190ab5e7af5f3a2cad3613dbd606255e4705.jpg)

![fb87852ee17de247b24ff2c016530815ed9a16634e3e6c1fe6007bea0e8e31cd.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/images/fb87852ee17de247b24ff2c016530815ed9a16634e3e6c1fe6007bea0e8e31cd.jpg)

### Tables

![01ead2a84b5e261ad0790d5c512cab8d040265a68a3b5315e12e8602a2c189b3.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/tables/01ead2a84b5e261ad0790d5c512cab8d040265a68a3b5315e12e8602a2c189b3.jpg)

![1af148ee7f24d5ee811c8f224c5332bdd71b984df3d57a045889cdd1395970c9.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/tables/1af148ee7f24d5ee811c8f224c5332bdd71b984df3d57a045889cdd1395970c9.jpg)

![1e7b5f8dd8652776b55bf85b1032639bed2c965b3e3aedd8cc84c91bf1e5c572.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/tables/1e7b5f8dd8652776b55bf85b1032639bed2c965b3e3aedd8cc84c91bf1e5c572.jpg)

![43217c51688cee7b7f10f420b2572172ecfada238dc2d6f455dbf8a4d2f638ea.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/tables/43217c51688cee7b7f10f420b2572172ecfada238dc2d6f455dbf8a4d2f638ea.jpg)

![469a47ca1c70dcd13bb57768617d8703a9a5318ced0b115e469ea36bdb913761.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/tables/469a47ca1c70dcd13bb57768617d8703a9a5318ced0b115e469ea36bdb913761.jpg)

![5865f04f75d20faaaaa68f5c55e721f4217311961b54aa16290562d35479726f.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/tables/5865f04f75d20faaaaa68f5c55e721f4217311961b54aa16290562d35479726f.jpg)

![6e00ecb6554a374c0560832fe21ded02433fe988730172fd561776ca43c5d0b1.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/tables/6e00ecb6554a374c0560832fe21ded02433fe988730172fd561776ca43c5d0b1.jpg)

![8c486f9bb896f845d207fd356b9a614df005cfc2a579c9e5a7654d6de2c5cc57.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/tables/8c486f9bb896f845d207fd356b9a614df005cfc2a579c9e5a7654d6de2c5cc57.jpg)

![8ed74733d8a734ad2796100190cb98550ff22c12ed94fa1e6999046883130321.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/tables/8ed74733d8a734ad2796100190cb98550ff22c12ed94fa1e6999046883130321.jpg)

![9010c21eb71d244840c9dade8f86eb5a097699417ce02821223782127db82fd7.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/tables/9010c21eb71d244840c9dade8f86eb5a097699417ce02821223782127db82fd7.jpg)

![ac5b5c6d1969364ed1b79f44cc1e304718c9850e1e8680c1f5d6504e5e819b35.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/tables/ac5b5c6d1969364ed1b79f44cc1e304718c9850e1e8680c1f5d6504e5e819b35.jpg)

![ad05bdead6669a36763d5b25a0f224c1350a86f78c613fb7b170a276a3361edf.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/tables/ad05bdead6669a36763d5b25a0f224c1350a86f78c613fb7b170a276a3361edf.jpg)

![bd90b19df77e4985dd61d01df44ac13189817b2e524062b03b6228b36c733eb9.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/tables/bd90b19df77e4985dd61d01df44ac13189817b2e524062b03b6228b36c733eb9.jpg)

![cf25dfdc5d26698a15e0e79066d29f341e9f6df863cdadf703d4e8892fc79e43.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/tables/cf25dfdc5d26698a15e0e79066d29f341e9f6df863cdadf703d4e8892fc79e43.jpg)

![dd9c296f57d126fba5fc1715ff1d3e01de0fbc77a59f44980046f6d13e438245.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/tables/dd9c296f57d126fba5fc1715ff1d3e01de0fbc77a59f44980046f6d13e438245.jpg)

![f4fe27a10d49dd8d121e92db7d0c1f39a6b4f165232cf66e627c680e3f1036bc.jpg](../icml_results/2631_Focusing%20Where%20Vision%20Matters_%20Selective%20Training%20for%20Large%20Vision%20Language%20Models%20via%20Visual%20Inform/tables/f4fe27a10d49dd8d121e92db7d0c1f39a6b4f165232cf66e627c680e3f1036bc.jpg)

<a id="paper-2598"></a>

## On the Collapse of Generative Paths: A Criterion and Correction for Diffusion Steering


### Images

![04b17cae2cc7684e1bd4947896ca7852abfbe5a80aabf689ac3eaeb1a9fc90af.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/04b17cae2cc7684e1bd4947896ca7852abfbe5a80aabf689ac3eaeb1a9fc90af.jpg)

![1383741ccb68797587dd8fa0831aea399411e5782613587effc38c253417aa6f.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/1383741ccb68797587dd8fa0831aea399411e5782613587effc38c253417aa6f.jpg)

![285909d0e5c3749cf93c0b7638d4ecc85b8b2c994ceccf44b72ac8c02a818bda.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/285909d0e5c3749cf93c0b7638d4ecc85b8b2c994ceccf44b72ac8c02a818bda.jpg)

![37ede486330bfbcdf53cde7f16ea75e09696954e5aec8d0a4dca68b7adadb154.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/37ede486330bfbcdf53cde7f16ea75e09696954e5aec8d0a4dca68b7adadb154.jpg)

![4a2d4495fa070623548e5ec588ff0e09b5b88727451d10bb37fd83f305793e39.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/4a2d4495fa070623548e5ec588ff0e09b5b88727451d10bb37fd83f305793e39.jpg)

![4b41ea4e6652a81c5a71e078c54fcf615f93e2c5707bb31bd1cf9749214928cd.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/4b41ea4e6652a81c5a71e078c54fcf615f93e2c5707bb31bd1cf9749214928cd.jpg)

![4d47b92fab42649538f8fdb1853698c21386c5b4116540b59cbde31178ebae43.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/4d47b92fab42649538f8fdb1853698c21386c5b4116540b59cbde31178ebae43.jpg)

![4f542f74a05f009724396611353c390f6ab659101adb230e33125fe797e6ea6f.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/4f542f74a05f009724396611353c390f6ab659101adb230e33125fe797e6ea6f.jpg)

![511f6400b4d89af1f5aefc77441d1c5d6a4192cf73f2efc439edd2689707f6ff.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/511f6400b4d89af1f5aefc77441d1c5d6a4192cf73f2efc439edd2689707f6ff.jpg)

![6d6fbbecff91a984e247b47470d5d4496205cd0e5d535e85d9664ca81ce8f9a9.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/6d6fbbecff91a984e247b47470d5d4496205cd0e5d535e85d9664ca81ce8f9a9.jpg)

![6e5a2f85b6539e447b342acd34c03d848bc620de140cab6b5440b1329d694f24.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/6e5a2f85b6539e447b342acd34c03d848bc620de140cab6b5440b1329d694f24.jpg)

![76d02545f9883d70c4f9039fb638b0a14aab87cd4c27d2b45184f345a41a86f5.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/76d02545f9883d70c4f9039fb638b0a14aab87cd4c27d2b45184f345a41a86f5.jpg)

![76ea51490bcff108b3b600b95bb89c3c949baa4bbc3f5a6fb7ae287909939139.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/76ea51490bcff108b3b600b95bb89c3c949baa4bbc3f5a6fb7ae287909939139.jpg)

![773b4f894033e6e6924135413fd07d1dce5d414fc389d61f36cae3b00a04ce8a.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/773b4f894033e6e6924135413fd07d1dce5d414fc389d61f36cae3b00a04ce8a.jpg)

![7878f8e17499ee5e44f41a59588169a1fb195255e8d758512a28cecb1e342a86.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/7878f8e17499ee5e44f41a59588169a1fb195255e8d758512a28cecb1e342a86.jpg)

![85a057f8a885d5986deb6979acfb99ebecf7edd3883c081fbf3c0926779b2441.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/85a057f8a885d5986deb6979acfb99ebecf7edd3883c081fbf3c0926779b2441.jpg)

![89504e9cd6369a75e2ef8536a7585ff373ab098b3dce9c9f27cfe73336d71436.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/89504e9cd6369a75e2ef8536a7585ff373ab098b3dce9c9f27cfe73336d71436.jpg)

![89bc949da3d487f47479d20b7a5b1ec85af9ee1a19a267cfdfcdb0f604aefc53.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/89bc949da3d487f47479d20b7a5b1ec85af9ee1a19a267cfdfcdb0f604aefc53.jpg)

![9c9c3217f75ca50b94da0964a13793c163aec65822a17d2cc103dc4fea9c28fb.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/9c9c3217f75ca50b94da0964a13793c163aec65822a17d2cc103dc4fea9c28fb.jpg)

![a04de62ec3d47ea2d5b17f6793a481160b4fcf29b37d1ccae4bf12722406e4cd.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/a04de62ec3d47ea2d5b17f6793a481160b4fcf29b37d1ccae4bf12722406e4cd.jpg)

![a3a1c2a18f10a803a11e06601e17174ad9e689602a57e54b9037f51f46b25dd5.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/a3a1c2a18f10a803a11e06601e17174ad9e689602a57e54b9037f51f46b25dd5.jpg)

![ae3db06d353daaf5f24b1cd216f3eb8b1b53edf029a2a91bfadae6f338aeae7a.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/ae3db06d353daaf5f24b1cd216f3eb8b1b53edf029a2a91bfadae6f338aeae7a.jpg)

![b63768c08fb3b62316e789bb9acff5d4a2a0c4c29f18b1c57f0165c3a3ea7bb2.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/b63768c08fb3b62316e789bb9acff5d4a2a0c4c29f18b1c57f0165c3a3ea7bb2.jpg)

![c0ba50e2ede37ee86f0091eab6d1ef6a3f1798491fac5caa95abf68b2d7e29c3.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/c0ba50e2ede37ee86f0091eab6d1ef6a3f1798491fac5caa95abf68b2d7e29c3.jpg)

![c28159931651abc0f0597f5c19007396f3a2435d8ae3e1664f0344c4bbc1179f.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/c28159931651abc0f0597f5c19007396f3a2435d8ae3e1664f0344c4bbc1179f.jpg)

![dc5bfbef7dc7b91a12eee738baaa9901a1389912989b0648b7f974b507605dce.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/dc5bfbef7dc7b91a12eee738baaa9901a1389912989b0648b7f974b507605dce.jpg)

![dc68bc43abfc8fd0b067e0296261843eb282de52f8de76e0879f30bbe85baa42.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/dc68bc43abfc8fd0b067e0296261843eb282de52f8de76e0879f30bbe85baa42.jpg)

![dcaa084efcad29ca52bc64cd8d300a7530b36375d2bf42f2629ff8f2fcdca9dd.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/dcaa084efcad29ca52bc64cd8d300a7530b36375d2bf42f2629ff8f2fcdca9dd.jpg)

![dcc93591e37e7f46f48082c410e06d0f83d2651fdec8274d781904e12274fcae.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/dcc93591e37e7f46f48082c410e06d0f83d2651fdec8274d781904e12274fcae.jpg)

![dd3197f8562d027460ecdfa4ad5c51f833334e6ed8ebd712a0a467427da70e97.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/dd3197f8562d027460ecdfa4ad5c51f833334e6ed8ebd712a0a467427da70e97.jpg)

![e1add2daa3a99d5c51edd95e5e8c66af00c1020c442cac2ed9791181a19dfcf3.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/e1add2daa3a99d5c51edd95e5e8c66af00c1020c442cac2ed9791181a19dfcf3.jpg)

![f495460412eeed63cf536a66b4ea5dca75e0698bba44fe612a9d32d665b7e28f.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/f495460412eeed63cf536a66b4ea5dca75e0698bba44fe612a9d32d665b7e28f.jpg)

![f7488127f44ac57e474c54996cdc358346e50ad86415871313c68a62ad9ad7b3.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/f7488127f44ac57e474c54996cdc358346e50ad86415871313c68a62ad9ad7b3.jpg)

![f8c7997c8e696772f5629592c47448a0be4f8ac4a06dbb529f15bd63f599b817.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/images/f8c7997c8e696772f5629592c47448a0be4f8ac4a06dbb529f15bd63f599b817.jpg)

### Tables

![01acb10f922d39b54a5cdbbdb3f4c2cc107f29dae055986a9cce680890fcc8ca.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/01acb10f922d39b54a5cdbbdb3f4c2cc107f29dae055986a9cce680890fcc8ca.jpg)

![054c76024e92e5077f1882c49800456406dc3e8566885976d2a0e52692a26dd1.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/054c76024e92e5077f1882c49800456406dc3e8566885976d2a0e52692a26dd1.jpg)

![11d39be1ec716710f0810496a4762056797e128de48685a28856883a0feb0e48.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/11d39be1ec716710f0810496a4762056797e128de48685a28856883a0feb0e48.jpg)

![135f752491e77c1da1961f5fbb28c2922511ce9a3548ed0560c24bed30a85ebf.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/135f752491e77c1da1961f5fbb28c2922511ce9a3548ed0560c24bed30a85ebf.jpg)

![2077f795153a6ef24de89925f9a9da006a6fceb26b9610fff0e28c629a31818a.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/2077f795153a6ef24de89925f9a9da006a6fceb26b9610fff0e28c629a31818a.jpg)

![234ea4e7f2bab772b5c3964df0f3a67a35dd190dbeb157e7a0a9d0422dcd1906.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/234ea4e7f2bab772b5c3964df0f3a67a35dd190dbeb157e7a0a9d0422dcd1906.jpg)

![257d7790ea4935b9946684ea797629b31918a1a8c21d5f40cf957052f04f7b1a.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/257d7790ea4935b9946684ea797629b31918a1a8c21d5f40cf957052f04f7b1a.jpg)

![323eba2cb5e205ca890d99aef3ead6bae2972a2ac812c61a54acf8dcbfd508b3.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/323eba2cb5e205ca890d99aef3ead6bae2972a2ac812c61a54acf8dcbfd508b3.jpg)

![349185720c2179d83fd0a2097175f62074b43e9fae8c049ae7c8f614b6ae6057.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/349185720c2179d83fd0a2097175f62074b43e9fae8c049ae7c8f614b6ae6057.jpg)

![3c957a9203fb61b4b45fdd605cd2508ac250de96b0881ea26413c8091271af8f.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/3c957a9203fb61b4b45fdd605cd2508ac250de96b0881ea26413c8091271af8f.jpg)

![3fc79cea7a2f0a655212b99f4dc0604bc1dc78f3f322a791a2b4e2dda10e83d7.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/3fc79cea7a2f0a655212b99f4dc0604bc1dc78f3f322a791a2b4e2dda10e83d7.jpg)

![46285250ec9dd901d2c6cbb79b564ae1c3d87446a417589b53bc0c9e6f03b9b8.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/46285250ec9dd901d2c6cbb79b564ae1c3d87446a417589b53bc0c9e6f03b9b8.jpg)

![687521d297d1330583a7f167e85ae062cfd7818a53243f7b816f8fa79714ab71.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/687521d297d1330583a7f167e85ae062cfd7818a53243f7b816f8fa79714ab71.jpg)

![6e5fb962d01d6133851c5d443df644226165f42839506204dc9c6c087e991adc.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/6e5fb962d01d6133851c5d443df644226165f42839506204dc9c6c087e991adc.jpg)

![79bf759bd27638e7ec81dcc9dbf9f7a158c58900483727fcf872651154d9450e.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/79bf759bd27638e7ec81dcc9dbf9f7a158c58900483727fcf872651154d9450e.jpg)

![93a4ed4e528f6a2bf320328e3daa7f183ee161bb471656fd712356b61e320d29.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/93a4ed4e528f6a2bf320328e3daa7f183ee161bb471656fd712356b61e320d29.jpg)

![a3e3bafa8e2d4a65a4389e00e4362511eadf2456826fc23bfe77f30945298078.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/a3e3bafa8e2d4a65a4389e00e4362511eadf2456826fc23bfe77f30945298078.jpg)

![ba624aa8afc65d55f4a642f42a2e108104f34298fe3f6d69d816cdf7463ceae9.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/ba624aa8afc65d55f4a642f42a2e108104f34298fe3f6d69d816cdf7463ceae9.jpg)

![c5f1cbb11cc2d0481d6737795e56fda283a99a54f65799b5d86703f81c0c42e1.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/c5f1cbb11cc2d0481d6737795e56fda283a99a54f65799b5d86703f81c0c42e1.jpg)

![d4113f54423b55e2f96663a81f3fb2858889946c26af1156ed6676e8c2746669.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/d4113f54423b55e2f96663a81f3fb2858889946c26af1156ed6676e8c2746669.jpg)

![e6825ee22e9422f0d7edf3df3c58bd148741b8e3cccce9e3d32f5807f000a14f.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/e6825ee22e9422f0d7edf3df3c58bd148741b8e3cccce9e3d32f5807f000a14f.jpg)

![fe690837cfe814d50ce187399d77bc2245d7ab58385a0118269e8db1e1aa2101.jpg](../icml_results/2632_On%20the%20Collapse%20of%20Generative%20Paths_%20A%20Criterion%20and%20Correction%20for%20Diffusion%20Steering/tables/fe690837cfe814d50ce187399d77bc2245d7ab58385a0118269e8db1e1aa2101.jpg)

<a id="paper-2599"></a>

## HInT: Hypergraph Infusion at the Structural Layers Improves Table Understanding


### Images

![2114823316bf224365516fab5bad6146dad2aca29d5f009ccdfc3a932ad0391d.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/images/2114823316bf224365516fab5bad6146dad2aca29d5f009ccdfc3a932ad0391d.jpg)

![25bcadc27d6e17477733af092fb2a1b809e138d78c8635be0de7469e237ea050.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/images/25bcadc27d6e17477733af092fb2a1b809e138d78c8635be0de7469e237ea050.jpg)

![3eab41f20ce375a906b283e33cb6874b92a8327abf2d737e4bda08368fe0b369.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/images/3eab41f20ce375a906b283e33cb6874b92a8327abf2d737e4bda08368fe0b369.jpg)

![7de0f7e308a4b3bb82362f70a5e334726f57f26119a3cf3c8a9080a1eed6c8ef.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/images/7de0f7e308a4b3bb82362f70a5e334726f57f26119a3cf3c8a9080a1eed6c8ef.jpg)

![8c9c6a79f1c6db0f0331787694853babe8c88c4cab5186e9b1e1588520e10b39.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/images/8c9c6a79f1c6db0f0331787694853babe8c88c4cab5186e9b1e1588520e10b39.jpg)

![d83fd6fabb18e6b6101110c787e17833d0b9b24cc75302bd47ba07376e749826.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/images/d83fd6fabb18e6b6101110c787e17833d0b9b24cc75302bd47ba07376e749826.jpg)

![e1be79313b249c745ac139d9d6829a1432b5bc5080cbc4192e410040bbb06010.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/images/e1be79313b249c745ac139d9d6829a1432b5bc5080cbc4192e410040bbb06010.jpg)

![e1c0e53b7f1f05e97a7e27919370b80192e22bc0b7507d5bd46fa4503b2db83f.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/images/e1c0e53b7f1f05e97a7e27919370b80192e22bc0b7507d5bd46fa4503b2db83f.jpg)

![f53bd03061dca3a4cdf11d105c549cfba59e5b25e8ffd3f9f7d6929ae1da516d.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/images/f53bd03061dca3a4cdf11d105c549cfba59e5b25e8ffd3f9f7d6929ae1da516d.jpg)

### Tables

![245bc7f45194f7ba2f8a683b82a0636c81ac4cb66b92a3fd4634df3945e0f1b8.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/245bc7f45194f7ba2f8a683b82a0636c81ac4cb66b92a3fd4634df3945e0f1b8.jpg)

![2cb220c94dfe9a1df31c1e65965fef9571392815698cccd3f56fc3e91aac4c45.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/2cb220c94dfe9a1df31c1e65965fef9571392815698cccd3f56fc3e91aac4c45.jpg)

![3189fae80634faf9275523d342322c748cb12348c6e1fa07b830d86038c3996b.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/3189fae80634faf9275523d342322c748cb12348c6e1fa07b830d86038c3996b.jpg)

![3361fb2fd9e8ba11ecb1b309a48a3b7a6df24a17b0975820027f9e0b067d172a.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/3361fb2fd9e8ba11ecb1b309a48a3b7a6df24a17b0975820027f9e0b067d172a.jpg)

![3da16c8b33508fbffe95e2dad532d26443eafd0222631fffdb6aef2d59d6c61d.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/3da16c8b33508fbffe95e2dad532d26443eafd0222631fffdb6aef2d59d6c61d.jpg)

![5df4001ccd7febb275877eca3f7901d2b6abed9a0fdaea47d3b4da8b75fcf303.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/5df4001ccd7febb275877eca3f7901d2b6abed9a0fdaea47d3b4da8b75fcf303.jpg)

![64f7eaa6c1d8adec4dc977c72612a82bb026fa6f97dfec30111d510e5fef6df4.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/64f7eaa6c1d8adec4dc977c72612a82bb026fa6f97dfec30111d510e5fef6df4.jpg)

![6dfc8505473da9b358939153dd7bacdc107a1cdb42f80a8e29f690dd72186cd7.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/6dfc8505473da9b358939153dd7bacdc107a1cdb42f80a8e29f690dd72186cd7.jpg)

![73ca748bddaa0e84af7f51fefa4b0054e9845888896f501a1405c946d2d614cd.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/73ca748bddaa0e84af7f51fefa4b0054e9845888896f501a1405c946d2d614cd.jpg)

![80a6851c6654f7686d02648bc2303460f76c226a48e0b323edd97f3a14f96e74.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/80a6851c6654f7686d02648bc2303460f76c226a48e0b323edd97f3a14f96e74.jpg)

![92a819f191e593bfd4f919fd4c096b6430289db26856f8036c08154ba9e6f20f.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/92a819f191e593bfd4f919fd4c096b6430289db26856f8036c08154ba9e6f20f.jpg)

![a2450cbce8f1a7cec28f83b968b1dd7157835cf079b4a06ecb919326d0f7ae88.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/a2450cbce8f1a7cec28f83b968b1dd7157835cf079b4a06ecb919326d0f7ae88.jpg)

![a7a60f1683fe448cb5881e311b7691d53ea165dcef38b1f3dbcdfc4d54908ad7.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/a7a60f1683fe448cb5881e311b7691d53ea165dcef38b1f3dbcdfc4d54908ad7.jpg)

![a7e6b2aa7dac65ea75d524b3396370ea88cec4c40a1ecf00055231b76932836e.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/a7e6b2aa7dac65ea75d524b3396370ea88cec4c40a1ecf00055231b76932836e.jpg)

![a89583d42d71ad398df0303309e2084da832588fad4e85cd5f3c0adc22b96fde.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/a89583d42d71ad398df0303309e2084da832588fad4e85cd5f3c0adc22b96fde.jpg)

![ae102d41dbe6fe36f45025d1a0a9ce876a5114bc03cf697a3f4018674fd23c19.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/ae102d41dbe6fe36f45025d1a0a9ce876a5114bc03cf697a3f4018674fd23c19.jpg)

![b45d35ffcd3d0ca1d04ae183cbf3bc4c6eaf75d9400a9eeae9a72bc38be5670f.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/b45d35ffcd3d0ca1d04ae183cbf3bc4c6eaf75d9400a9eeae9a72bc38be5670f.jpg)

![c23011b22a36d924b04cd499d870c5b62763d9ffc966b874ed703b510cb4132f.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/c23011b22a36d924b04cd499d870c5b62763d9ffc966b874ed703b510cb4132f.jpg)

![d0ca5f3982dd6520afaf64a463a96bac4647df1dc225ad1ccfb36c60e6958aaf.jpg](../icml_results/2633_HInT_%20Hypergraph%20Infusion%20at%20the%20Structural%20Layers%20Improves%20Table%20Understanding/tables/d0ca5f3982dd6520afaf64a463a96bac4647df1dc225ad1ccfb36c60e6958aaf.jpg)

<a id="paper-2600"></a>

## Decision-Focused Learning via Tangent-Space Projection of Prediction Error


### Images

![6bf8b2157b0d56b456251a38fddd8602ac91918804a618bfa1e30cae401ab94c.jpg](../icml_results/2634_Decision-Focused%20Learning%20via%20Tangent-Space%20Projection%20of%20Prediction%20Error/images/6bf8b2157b0d56b456251a38fddd8602ac91918804a618bfa1e30cae401ab94c.jpg)

![9d36769c41571e1538a81e87fd40004bb178de2a4a81542094e75b5709446822.jpg](../icml_results/2634_Decision-Focused%20Learning%20via%20Tangent-Space%20Projection%20of%20Prediction%20Error/images/9d36769c41571e1538a81e87fd40004bb178de2a4a81542094e75b5709446822.jpg)

![deff52b7ed2b0719bf2d9362c3d66ada6e34d5db18be297fe8735b53ed01a4ac.jpg](../icml_results/2634_Decision-Focused%20Learning%20via%20Tangent-Space%20Projection%20of%20Prediction%20Error/images/deff52b7ed2b0719bf2d9362c3d66ada6e34d5db18be297fe8735b53ed01a4ac.jpg)

![ec1be0b8ebf87394eb18fda0496c896a8ef79e5f05a77300d977af4e278ccc5f.jpg](../icml_results/2634_Decision-Focused%20Learning%20via%20Tangent-Space%20Projection%20of%20Prediction%20Error/images/ec1be0b8ebf87394eb18fda0496c896a8ef79e5f05a77300d977af4e278ccc5f.jpg)

### Tables

![0bbb00bce63504e422fd8a8a7d0dc782a53acdee7998bfb6becfa420affe1d42.jpg](../icml_results/2634_Decision-Focused%20Learning%20via%20Tangent-Space%20Projection%20of%20Prediction%20Error/tables/0bbb00bce63504e422fd8a8a7d0dc782a53acdee7998bfb6becfa420affe1d42.jpg)

![10e7c25a00afa107bc23ef12fde56d0f05dc0c7303a1f87d9ab55be3c21cd71b.jpg](../icml_results/2634_Decision-Focused%20Learning%20via%20Tangent-Space%20Projection%20of%20Prediction%20Error/tables/10e7c25a00afa107bc23ef12fde56d0f05dc0c7303a1f87d9ab55be3c21cd71b.jpg)

![33aea1ef8905e3570e3ec899112c918b8be848705e2e50c6012aa342f3b1fd55.jpg](../icml_results/2634_Decision-Focused%20Learning%20via%20Tangent-Space%20Projection%20of%20Prediction%20Error/tables/33aea1ef8905e3570e3ec899112c918b8be848705e2e50c6012aa342f3b1fd55.jpg)

![4199d06661653ff7a5a9c45966fdad9d85bc224b32b2426740f7b454b415f88a.jpg](../icml_results/2634_Decision-Focused%20Learning%20via%20Tangent-Space%20Projection%20of%20Prediction%20Error/tables/4199d06661653ff7a5a9c45966fdad9d85bc224b32b2426740f7b454b415f88a.jpg)

![4e4c6068639a105041f4a510909bb8bb3c14df7fbb3ec8fcb9bb10f26bbfeb83.jpg](../icml_results/2634_Decision-Focused%20Learning%20via%20Tangent-Space%20Projection%20of%20Prediction%20Error/tables/4e4c6068639a105041f4a510909bb8bb3c14df7fbb3ec8fcb9bb10f26bbfeb83.jpg)

![50685993f5c79c50b5f610b103d4d5e179963e92db0d84785b546a4ad40674a8.jpg](../icml_results/2634_Decision-Focused%20Learning%20via%20Tangent-Space%20Projection%20of%20Prediction%20Error/tables/50685993f5c79c50b5f610b103d4d5e179963e92db0d84785b546a4ad40674a8.jpg)

![5e91a86218b27b3a34057551c06718da0ecdbeae0925d34862b75619d34ac666.jpg](../icml_results/2634_Decision-Focused%20Learning%20via%20Tangent-Space%20Projection%20of%20Prediction%20Error/tables/5e91a86218b27b3a34057551c06718da0ecdbeae0925d34862b75619d34ac666.jpg)

![667eb7be0ec171cefea9198528fe683c88577e4cc22f55b6dc7039e6b725dc07.jpg](../icml_results/2634_Decision-Focused%20Learning%20via%20Tangent-Space%20Projection%20of%20Prediction%20Error/tables/667eb7be0ec171cefea9198528fe683c88577e4cc22f55b6dc7039e6b725dc07.jpg)

![abd962d5f956972a35520b274348eba28059436f7f5ec774577cf82b9b0b267d.jpg](../icml_results/2634_Decision-Focused%20Learning%20via%20Tangent-Space%20Projection%20of%20Prediction%20Error/tables/abd962d5f956972a35520b274348eba28059436f7f5ec774577cf82b9b0b267d.jpg)

![d4c28c155243349ea28333bf49131788d7a39f23fe0d676d37777a773f700bd3.jpg](../icml_results/2634_Decision-Focused%20Learning%20via%20Tangent-Space%20Projection%20of%20Prediction%20Error/tables/d4c28c155243349ea28333bf49131788d7a39f23fe0d676d37777a773f700bd3.jpg)

![ec088d465332eaf8c37290d054b7487c7f42cc6d1a9f21f5c5f82fbc95a12141.jpg](../icml_results/2634_Decision-Focused%20Learning%20via%20Tangent-Space%20Projection%20of%20Prediction%20Error/tables/ec088d465332eaf8c37290d054b7487c7f42cc6d1a9f21f5c5f82fbc95a12141.jpg)

![f7f3ff40adfa39f741856bf4eeceac4db8f5a5581f58ea0a4ffa5aad19b37ca8.jpg](../icml_results/2634_Decision-Focused%20Learning%20via%20Tangent-Space%20Projection%20of%20Prediction%20Error/tables/f7f3ff40adfa39f741856bf4eeceac4db8f5a5581f58ea0a4ffa5aad19b37ca8.jpg)

<a id="paper-2601"></a>

## SAGE: Shaping Anchors for Guided Exploration in RLVR of LLMs


### Images

![5b91e553fd94af0d1748aa8ce5cb15001875bd122b0883d9044e9aef1a70a4c6.jpg](../icml_results/2635_SAGE_%20Shaping%20Anchors%20for%20Guided%20Exploration%20in%20RLVR%20of%20LLMs/images/5b91e553fd94af0d1748aa8ce5cb15001875bd122b0883d9044e9aef1a70a4c6.jpg)

![7832eb2bc6f2276b5ba979859c260938f44c5380705d59f75f4dd8c68f82d6f3.jpg](../icml_results/2635_SAGE_%20Shaping%20Anchors%20for%20Guided%20Exploration%20in%20RLVR%20of%20LLMs/images/7832eb2bc6f2276b5ba979859c260938f44c5380705d59f75f4dd8c68f82d6f3.jpg)

![82e56545d9a2aa5af94e3288032d66e2341bc97417a60c300a260e9f7928cbbf.jpg](../icml_results/2635_SAGE_%20Shaping%20Anchors%20for%20Guided%20Exploration%20in%20RLVR%20of%20LLMs/images/82e56545d9a2aa5af94e3288032d66e2341bc97417a60c300a260e9f7928cbbf.jpg)

![8e9e48fde4c486084368bbb4a4f990b09ac99f734e8d1a61cb24604dfc2b6009.jpg](../icml_results/2635_SAGE_%20Shaping%20Anchors%20for%20Guided%20Exploration%20in%20RLVR%20of%20LLMs/images/8e9e48fde4c486084368bbb4a4f990b09ac99f734e8d1a61cb24604dfc2b6009.jpg)

![a9d7f7a95246cfdb58244f4bdfc9837f4a0323d7a77ee82646c23aeaef76b7f9.jpg](../icml_results/2635_SAGE_%20Shaping%20Anchors%20for%20Guided%20Exploration%20in%20RLVR%20of%20LLMs/images/a9d7f7a95246cfdb58244f4bdfc9837f4a0323d7a77ee82646c23aeaef76b7f9.jpg)

![c0f13c119739f3994251860b48fab2c146628d3acf13527eeda9f8132f0929e1.jpg](../icml_results/2635_SAGE_%20Shaping%20Anchors%20for%20Guided%20Exploration%20in%20RLVR%20of%20LLMs/images/c0f13c119739f3994251860b48fab2c146628d3acf13527eeda9f8132f0929e1.jpg)

![c0faecc728cb24e3ffe3f0be2dabe5e300a34501e7faaa71c45cb7ee10318eaf.jpg](../icml_results/2635_SAGE_%20Shaping%20Anchors%20for%20Guided%20Exploration%20in%20RLVR%20of%20LLMs/images/c0faecc728cb24e3ffe3f0be2dabe5e300a34501e7faaa71c45cb7ee10318eaf.jpg)

![f8887ab95ea091085aeba1b382bb3b9887a9574b8cec7ad0970974c596a9c642.jpg](../icml_results/2635_SAGE_%20Shaping%20Anchors%20for%20Guided%20Exploration%20in%20RLVR%20of%20LLMs/images/f8887ab95ea091085aeba1b382bb3b9887a9574b8cec7ad0970974c596a9c642.jpg)

### Tables

![3b2568299dca506b32f6496cd40447baeafcb48a9915335db0cf820470d2dcb6.jpg](../icml_results/2635_SAGE_%20Shaping%20Anchors%20for%20Guided%20Exploration%20in%20RLVR%20of%20LLMs/tables/3b2568299dca506b32f6496cd40447baeafcb48a9915335db0cf820470d2dcb6.jpg)

![47627625f3e9111c687e068b79eb5178875cebec8229f8ed6aba7be66beeb758.jpg](../icml_results/2635_SAGE_%20Shaping%20Anchors%20for%20Guided%20Exploration%20in%20RLVR%20of%20LLMs/tables/47627625f3e9111c687e068b79eb5178875cebec8229f8ed6aba7be66beeb758.jpg)

![54caa558f1d850ceeaaa139755c378e3327aed18e6182bceb027c8d6b203b9d2.jpg](../icml_results/2635_SAGE_%20Shaping%20Anchors%20for%20Guided%20Exploration%20in%20RLVR%20of%20LLMs/tables/54caa558f1d850ceeaaa139755c378e3327aed18e6182bceb027c8d6b203b9d2.jpg)

![83f84a80d7842ef814e5e4fc737e35d545711b67b94223e7643b617cd71fc583.jpg](../icml_results/2635_SAGE_%20Shaping%20Anchors%20for%20Guided%20Exploration%20in%20RLVR%20of%20LLMs/tables/83f84a80d7842ef814e5e4fc737e35d545711b67b94223e7643b617cd71fc583.jpg)

![9225292ce53f448a1016abaeb6ff57c7ffe36bf749ed7bbda18205de061924b3.jpg](../icml_results/2635_SAGE_%20Shaping%20Anchors%20for%20Guided%20Exploration%20in%20RLVR%20of%20LLMs/tables/9225292ce53f448a1016abaeb6ff57c7ffe36bf749ed7bbda18205de061924b3.jpg)

![ad228f2993ea538c24196b15f1651ba6faa3fa6c33984d04b07d913475724f0d.jpg](../icml_results/2635_SAGE_%20Shaping%20Anchors%20for%20Guided%20Exploration%20in%20RLVR%20of%20LLMs/tables/ad228f2993ea538c24196b15f1651ba6faa3fa6c33984d04b07d913475724f0d.jpg)

![d3a58e74f5dfa15941fedf61f0a88b95eb89a5a14fd655809589dd66ac2e4930.jpg](../icml_results/2635_SAGE_%20Shaping%20Anchors%20for%20Guided%20Exploration%20in%20RLVR%20of%20LLMs/tables/d3a58e74f5dfa15941fedf61f0a88b95eb89a5a14fd655809589dd66ac2e4930.jpg)

![d4a6ed75681aae571c4a757e51b314704238082cff7863b052ba6085016319fc.jpg](../icml_results/2635_SAGE_%20Shaping%20Anchors%20for%20Guided%20Exploration%20in%20RLVR%20of%20LLMs/tables/d4a6ed75681aae571c4a757e51b314704238082cff7863b052ba6085016319fc.jpg)

![ee651c562fc49536c14f3e56b1db985b97016e2cc2ec3ac17de99ec3983e5bb7.jpg](../icml_results/2635_SAGE_%20Shaping%20Anchors%20for%20Guided%20Exploration%20in%20RLVR%20of%20LLMs/tables/ee651c562fc49536c14f3e56b1db985b97016e2cc2ec3ac17de99ec3983e5bb7.jpg)

<a id="paper-2602"></a>

## Interaction-Breaking Adversarial Learning Framework for Robust Multi-Agent Reinforcement Learning


### Images

![000ddfa9a29046734eeea71137b2fd05f91ec47731e01e35d10dbb678652c4fa.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/images/000ddfa9a29046734eeea71137b2fd05f91ec47731e01e35d10dbb678652c4fa.jpg)

![0778d0c51f8b9ac52db1fe775ace5553c1dae0a656280e6946117dcdabf61ad5.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/images/0778d0c51f8b9ac52db1fe775ace5553c1dae0a656280e6946117dcdabf61ad5.jpg)

![27ca6131476f111cc1a840f2fd96b525cd19f58ef80f84c166da863f3980e5a7.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/images/27ca6131476f111cc1a840f2fd96b525cd19f58ef80f84c166da863f3980e5a7.jpg)

![2a7c4a859ce347e7a86b8aa64590062d060cfb6d32c85da29ae2787e3255023e.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/images/2a7c4a859ce347e7a86b8aa64590062d060cfb6d32c85da29ae2787e3255023e.jpg)

![38636b1dd023865103ab83fbd1f3a969458914741d76eb6d9015e2a40c60d6d8.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/images/38636b1dd023865103ab83fbd1f3a969458914741d76eb6d9015e2a40c60d6d8.jpg)

![40294ac5d4deb80267ec04d36237280fdf091d1b00e0b4a59771aa549b41b734.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/images/40294ac5d4deb80267ec04d36237280fdf091d1b00e0b4a59771aa549b41b734.jpg)

![4d50fd73b655f50795f2484f287eaa2cf19df8d5f61d522cf5ba50102d483ea3.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/images/4d50fd73b655f50795f2484f287eaa2cf19df8d5f61d522cf5ba50102d483ea3.jpg)

![561a71594fd655205ab764c2dd27ad8e12bb0c34f51954b8d64776032c0231da.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/images/561a71594fd655205ab764c2dd27ad8e12bb0c34f51954b8d64776032c0231da.jpg)

![58261ab3f76cf94fee1e25bd659a1c608858402158d1edccdecc638e06c087cd.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/images/58261ab3f76cf94fee1e25bd659a1c608858402158d1edccdecc638e06c087cd.jpg)

![5c99c88343973d81ebe2c5df12cb33c5a4c4df32d0418faac14200460f95409e.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/images/5c99c88343973d81ebe2c5df12cb33c5a4c4df32d0418faac14200460f95409e.jpg)

![8d867a5aab10d9b7536153bf60f69e8e0725d6002a26b4ae66b6796d3f634d7e.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/images/8d867a5aab10d9b7536153bf60f69e8e0725d6002a26b4ae66b6796d3f634d7e.jpg)

![96dea3d8eff3b97ea2d2fa934f6a7d4e69674180a050b4f72ce8303364d0b767.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/images/96dea3d8eff3b97ea2d2fa934f6a7d4e69674180a050b4f72ce8303364d0b767.jpg)

![9f0a29b209b45169cc533a08c1fc23f0739226aa1560858af46aaec958d0df90.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/images/9f0a29b209b45169cc533a08c1fc23f0739226aa1560858af46aaec958d0df90.jpg)

![b307664d9e7b5fa39613e11a2222ce1754b6bee4df918ee8fe0066737db14785.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/images/b307664d9e7b5fa39613e11a2222ce1754b6bee4df918ee8fe0066737db14785.jpg)

![bb12fa2188ff91e6930eab5331c7085988096e0c457f274baa34adb917ce34c9.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/images/bb12fa2188ff91e6930eab5331c7085988096e0c457f274baa34adb917ce34c9.jpg)

![fc00af41770e8332721fc1fc53bfe9df1011c81fb3799b0a8de25d350f862e5d.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/images/fc00af41770e8332721fc1fc53bfe9df1011c81fb3799b0a8de25d350f862e5d.jpg)

![fc5bbd2cfed2eee3685d3634587dd754d9c37341c28b65796b532c17a124920c.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/images/fc5bbd2cfed2eee3685d3634587dd754d9c37341c28b65796b532c17a124920c.jpg)

### Tables

![14182f381f6c81201167274e6354c57f25fd879cba033c5af9cdae3acbe2d35c.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/tables/14182f381f6c81201167274e6354c57f25fd879cba033c5af9cdae3acbe2d35c.jpg)

![3674408c76c29fd10111864467c477d906783db4da273a340713c6e0c431e266.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/tables/3674408c76c29fd10111864467c477d906783db4da273a340713c6e0c431e266.jpg)

![38bc30dc07910c5e07abf4de53858c604dbce419d0260808fbeec6027b554266.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/tables/38bc30dc07910c5e07abf4de53858c604dbce419d0260808fbeec6027b554266.jpg)

![65818f340d4c2d39c0b05a7058c7cda20e50d14fcf9cd601a59d95e8ff21295a.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/tables/65818f340d4c2d39c0b05a7058c7cda20e50d14fcf9cd601a59d95e8ff21295a.jpg)

![66e7b4fa2508de21190570bdf17ed2b2ce926f814bdcd75c766eea953e8a6285.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/tables/66e7b4fa2508de21190570bdf17ed2b2ce926f814bdcd75c766eea953e8a6285.jpg)

![7f7ae7e4f6ad1b9654fcf49944f61102b610f129ba14fabaa3af13aff4a296fe.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/tables/7f7ae7e4f6ad1b9654fcf49944f61102b610f129ba14fabaa3af13aff4a296fe.jpg)

![81951b3d2c5cab6700d82aff8c4d45065a4471dfb5d99eee140bba58a6d8a49e.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/tables/81951b3d2c5cab6700d82aff8c4d45065a4471dfb5d99eee140bba58a6d8a49e.jpg)

![83c07bc02796a2a40d8df13b4dc1cf71f267ced2d92a81696f8963887327b0b1.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/tables/83c07bc02796a2a40d8df13b4dc1cf71f267ced2d92a81696f8963887327b0b1.jpg)

![b5d5c0616178beb83b6ace4b8efc3ce7b6049f74d3d81f49cdac5e54492c3494.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/tables/b5d5c0616178beb83b6ace4b8efc3ce7b6049f74d3d81f49cdac5e54492c3494.jpg)

![c2875e9375a28fd2b0a207f441be22876cf8cb2d448b69755616e6d4a4ab4961.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/tables/c2875e9375a28fd2b0a207f441be22876cf8cb2d448b69755616e6d4a4ab4961.jpg)

![c65f98bc25bae7d6a59cc53fd9674c518a3e6eea762eee9d63ce3f96e5ed3484.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/tables/c65f98bc25bae7d6a59cc53fd9674c518a3e6eea762eee9d63ce3f96e5ed3484.jpg)

![d6f9bc21ebf3253e542ad4137a59b0efa5f91fa977575ea38fd37874c943ca5d.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/tables/d6f9bc21ebf3253e542ad4137a59b0efa5f91fa977575ea38fd37874c943ca5d.jpg)

![e0ef77a887982769b5cabe2a1cfe9643227669fb5ff32c1cec203182479e9c2f.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/tables/e0ef77a887982769b5cabe2a1cfe9643227669fb5ff32c1cec203182479e9c2f.jpg)

![eb9b92c431461237a737bef3023d6d614e64f49aebd1dbe7f624c43f15e4d5b1.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/tables/eb9b92c431461237a737bef3023d6d614e64f49aebd1dbe7f624c43f15e4d5b1.jpg)

![fb726095e20ff192f53ffbee8d208a1b765f7fe04d49a123ae1e73faabd73112.jpg](../icml_results/2636_Interaction-Breaking%20Adversarial%20Learning%20Framework%20for%20Robust%20Multi-Agent%20Reinforcement%20Learning/tables/fb726095e20ff192f53ffbee8d208a1b765f7fe04d49a123ae1e73faabd73112.jpg)

<a id="paper-2603"></a>

## Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning


### Images

![02f0ed72e52952e3d3fb8ae7627d613bc52956cccd60cec70cfb8ef9a4a50833.jpg](../icml_results/2637_Towards%20Efficient%20and%20Expressive%20Offline%20RL%20via%20Flow-Anchored%20Noise-conditioned%20Q-Learning/images/02f0ed72e52952e3d3fb8ae7627d613bc52956cccd60cec70cfb8ef9a4a50833.jpg)

![24d69ae22a874add3e34954a751c48e1f79fe1938b0a7ae1d8a13cd3462f17bc.jpg](../icml_results/2637_Towards%20Efficient%20and%20Expressive%20Offline%20RL%20via%20Flow-Anchored%20Noise-conditioned%20Q-Learning/images/24d69ae22a874add3e34954a751c48e1f79fe1938b0a7ae1d8a13cd3462f17bc.jpg)

![2aad5d1941619e9e07a92aa0a3a84b70122b1d24cf3842b9c777dc3cbc10e5cd.jpg](../icml_results/2637_Towards%20Efficient%20and%20Expressive%20Offline%20RL%20via%20Flow-Anchored%20Noise-conditioned%20Q-Learning/images/2aad5d1941619e9e07a92aa0a3a84b70122b1d24cf3842b9c777dc3cbc10e5cd.jpg)

![38231567d0a3214eca6b3cdd14409f9b537c90c3db4b171b1dc7700ff0f1d031.jpg](../icml_results/2637_Towards%20Efficient%20and%20Expressive%20Offline%20RL%20via%20Flow-Anchored%20Noise-conditioned%20Q-Learning/images/38231567d0a3214eca6b3cdd14409f9b537c90c3db4b171b1dc7700ff0f1d031.jpg)

![bb78e29924f9cd3fdced23e623810346d22bfec64dcf99e4e6f20fbb97ed0fed.jpg](../icml_results/2637_Towards%20Efficient%20and%20Expressive%20Offline%20RL%20via%20Flow-Anchored%20Noise-conditioned%20Q-Learning/images/bb78e29924f9cd3fdced23e623810346d22bfec64dcf99e4e6f20fbb97ed0fed.jpg)

![bea52908f9c7933ee232bbaeac073fd4e9b04ad632e2cfa41cc4a01d442eb8ef.jpg](../icml_results/2637_Towards%20Efficient%20and%20Expressive%20Offline%20RL%20via%20Flow-Anchored%20Noise-conditioned%20Q-Learning/images/bea52908f9c7933ee232bbaeac073fd4e9b04ad632e2cfa41cc4a01d442eb8ef.jpg)

![c0ccd3bc5da575ae3561a491aeb5b7c57f062277406161eb425f458991b07655.jpg](../icml_results/2637_Towards%20Efficient%20and%20Expressive%20Offline%20RL%20via%20Flow-Anchored%20Noise-conditioned%20Q-Learning/images/c0ccd3bc5da575ae3561a491aeb5b7c57f062277406161eb425f458991b07655.jpg)

### Tables

![2ca200e03acc101b5c2855ef0d05c83a27cefa9b363e98573484e8377e37ea23.jpg](../icml_results/2637_Towards%20Efficient%20and%20Expressive%20Offline%20RL%20via%20Flow-Anchored%20Noise-conditioned%20Q-Learning/tables/2ca200e03acc101b5c2855ef0d05c83a27cefa9b363e98573484e8377e37ea23.jpg)

![4a00ce18fa74276cf977299c962fe9ff92609e8f6c3d608c6086b01b75d2c94e.jpg](../icml_results/2637_Towards%20Efficient%20and%20Expressive%20Offline%20RL%20via%20Flow-Anchored%20Noise-conditioned%20Q-Learning/tables/4a00ce18fa74276cf977299c962fe9ff92609e8f6c3d608c6086b01b75d2c94e.jpg)

![830e8980aa3e6ff40095d4909057aab54b961f60331b443f8a1afa48f0495548.jpg](../icml_results/2637_Towards%20Efficient%20and%20Expressive%20Offline%20RL%20via%20Flow-Anchored%20Noise-conditioned%20Q-Learning/tables/830e8980aa3e6ff40095d4909057aab54b961f60331b443f8a1afa48f0495548.jpg)

![d18d614714ce6afefef30ebfb03f31cf74db2ba9bb6daa475c76e5ddfc19ed81.jpg](../icml_results/2637_Towards%20Efficient%20and%20Expressive%20Offline%20RL%20via%20Flow-Anchored%20Noise-conditioned%20Q-Learning/tables/d18d614714ce6afefef30ebfb03f31cf74db2ba9bb6daa475c76e5ddfc19ed81.jpg)

![e6a60fab0c646a0b6fc94fb9d93b46070d1e1eea55230613b003e02251d85e02.jpg](../icml_results/2637_Towards%20Efficient%20and%20Expressive%20Offline%20RL%20via%20Flow-Anchored%20Noise-conditioned%20Q-Learning/tables/e6a60fab0c646a0b6fc94fb9d93b46070d1e1eea55230613b003e02251d85e02.jpg)

![f8151b0a4cd6cb413c3ba70bd03be0b2ffdde893d31a6c0bf44f4fae6a74d062.jpg](../icml_results/2637_Towards%20Efficient%20and%20Expressive%20Offline%20RL%20via%20Flow-Anchored%20Noise-conditioned%20Q-Learning/tables/f8151b0a4cd6cb413c3ba70bd03be0b2ffdde893d31a6c0bf44f4fae6a74d062.jpg)

![fc84d91c31290032dba22ad369d2d1487d21ae8a4085e2da57a974a16ee35ef3.jpg](../icml_results/2637_Towards%20Efficient%20and%20Expressive%20Offline%20RL%20via%20Flow-Anchored%20Noise-conditioned%20Q-Learning/tables/fc84d91c31290032dba22ad369d2d1487d21ae8a4085e2da57a974a16ee35ef3.jpg)

<a id="paper-2604"></a>

## From Volume to Value: Preference-Aligned Memory Construction for On-Device RAG


### Images

![17bd6b8316e484961c834a0c2b4938e4660bc13831678ef75d4eb15f4efe2337.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/images/17bd6b8316e484961c834a0c2b4938e4660bc13831678ef75d4eb15f4efe2337.jpg)

![34f02ed0f1b07dabc892f9e5b0efe19e0590ff21442208667cc2d1a79b2c2c81.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/images/34f02ed0f1b07dabc892f9e5b0efe19e0590ff21442208667cc2d1a79b2c2c81.jpg)

![6de02890144c728eb3c27231b24f86b5040716e1dd95f7a5023015226198b49b.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/images/6de02890144c728eb3c27231b24f86b5040716e1dd95f7a5023015226198b49b.jpg)

![db32fd00fee0e670339a79f560c50bdf95c0682993b6fc9931a29d4fa9151008.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/images/db32fd00fee0e670339a79f560c50bdf95c0682993b6fc9931a29d4fa9151008.jpg)

![fbbd704b50343f2c5f84ee80f85105d445b37815050abf0174c326c050dc1755.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/images/fbbd704b50343f2c5f84ee80f85105d445b37815050abf0174c326c050dc1755.jpg)

### Tables

![038fe6f53eac62a7cbe79f87a496aeb61c687716b3d1c1077c338d92ef8636a6.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/tables/038fe6f53eac62a7cbe79f87a496aeb61c687716b3d1c1077c338d92ef8636a6.jpg)

![107e397778a7a3661939bb3e0ba272044baa975c896bd8ce640d707efa47ad29.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/tables/107e397778a7a3661939bb3e0ba272044baa975c896bd8ce640d707efa47ad29.jpg)

![1d41c635bf44d08f352de98f8ff19587294a539f25fa9561469068ce3fd69f7f.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/tables/1d41c635bf44d08f352de98f8ff19587294a539f25fa9561469068ce3fd69f7f.jpg)

![2e39a2316475467d5a501917ed4fdd26b403ff8acb06d7c4345205aea767effc.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/tables/2e39a2316475467d5a501917ed4fdd26b403ff8acb06d7c4345205aea767effc.jpg)

![2fcbe077915d0132a02d757355978fdaa92a66daccce74cdedea571cc83e0e32.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/tables/2fcbe077915d0132a02d757355978fdaa92a66daccce74cdedea571cc83e0e32.jpg)

![5d84b0e5458f63349ae1231e2fb84a7d142da83cfb437afd0dbbfcc1140dc7bd.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/tables/5d84b0e5458f63349ae1231e2fb84a7d142da83cfb437afd0dbbfcc1140dc7bd.jpg)

![64e3eaaced1ca336e4bf38604823febd74b806a8fe6dd024ed9a90849244b93a.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/tables/64e3eaaced1ca336e4bf38604823febd74b806a8fe6dd024ed9a90849244b93a.jpg)

![85f7b2de14615e427cf319194c4337819331d3a9beacd0a6b2f8f356f443f4c2.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/tables/85f7b2de14615e427cf319194c4337819331d3a9beacd0a6b2f8f356f443f4c2.jpg)

![9235033d8ccf8be067d146d0c7e874a794236cf65517bf6ad24dcdf7b297a316.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/tables/9235033d8ccf8be067d146d0c7e874a794236cf65517bf6ad24dcdf7b297a316.jpg)

![a86667bab3ea381ae39f40b1e535555e9c4e2b3220ecea35b739bafcddfb9da9.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/tables/a86667bab3ea381ae39f40b1e535555e9c4e2b3220ecea35b739bafcddfb9da9.jpg)

![a8854b9cf43c370145638b39fb406594af8daff9b5b6674c3a75b42a48d7ed0e.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/tables/a8854b9cf43c370145638b39fb406594af8daff9b5b6674c3a75b42a48d7ed0e.jpg)

![c7238dc7315d3bd16995b6d9e1dd056591d9e851bf959fc6ec2ee8ad60c89bcd.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/tables/c7238dc7315d3bd16995b6d9e1dd056591d9e851bf959fc6ec2ee8ad60c89bcd.jpg)

![d3257c308a9e6a3caeb5c834ccfd4c9e57449f53f9feb5e96691a831bf686005.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/tables/d3257c308a9e6a3caeb5c834ccfd4c9e57449f53f9feb5e96691a831bf686005.jpg)

![da491938d3e807bc30e930e6e6665529b04024302b73e57ab334776d14a52653.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/tables/da491938d3e807bc30e930e6e6665529b04024302b73e57ab334776d14a52653.jpg)

![e82ad536fc0d4663150d3c260768445a1c7dedbd086dcdff43dacf9d038505c7.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/tables/e82ad536fc0d4663150d3c260768445a1c7dedbd086dcdff43dacf9d038505c7.jpg)

![f146a4169d86d535271ca10f8dea52d98a0a642b5f3bb58e4c1315d93e6ddcae.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/tables/f146a4169d86d535271ca10f8dea52d98a0a642b5f3bb58e4c1315d93e6ddcae.jpg)

![f393d6237bd5fcbc5a75f0cfd8f2fae2d3caf6a97a43db92f704434e1b79b66d.jpg](../icml_results/2638_From%20Volume%20to%20Value_%20Preference-Aligned%20Memory%20Construction%20for%20On-Device%20RAG/tables/f393d6237bd5fcbc5a75f0cfd8f2fae2d3caf6a97a43db92f704434e1b79b66d.jpg)

<a id="paper-2605"></a>

## BIT-LLM: Brain Instruction Tuned LLM with persistent Cross-Attention for fMRI-to-Text Decoding


### Images

![0cffb05727d2ed3ec1801a548576eb830472c242e3efa1f4821ab42ee761af47.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/images/0cffb05727d2ed3ec1801a548576eb830472c242e3efa1f4821ab42ee761af47.jpg)

![4e65ce705181c23f31f264963bce7a851fe5638eb97381e361c7f811ef469b04.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/images/4e65ce705181c23f31f264963bce7a851fe5638eb97381e361c7f811ef469b04.jpg)

![5934c7affceee5f32208e22a0c0631ff91f98ab830f630ea0239730d865ab8da.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/images/5934c7affceee5f32208e22a0c0631ff91f98ab830f630ea0239730d865ab8da.jpg)

![8c2f1b4a41069d04046c1bf51b8f91097f95bb50879942e4d6184c62f5b8accd.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/images/8c2f1b4a41069d04046c1bf51b8f91097f95bb50879942e4d6184c62f5b8accd.jpg)

![9a999744ef6fb1108214800b2d859e72b659456b28404850987bb8ca9dba22db.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/images/9a999744ef6fb1108214800b2d859e72b659456b28404850987bb8ca9dba22db.jpg)

![ea979692a714abf56c19a58ab93bc3f1535b793e578e7ba01b0bae3d3ac78127.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/images/ea979692a714abf56c19a58ab93bc3f1535b793e578e7ba01b0bae3d3ac78127.jpg)

### Tables

![0601c4d99ac363a59f5eeb3d4f65c6e4fcc0328665be5d9f871cd3a8229bebd3.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/0601c4d99ac363a59f5eeb3d4f65c6e4fcc0328665be5d9f871cd3a8229bebd3.jpg)

![0e490c3f40a06ca92af69601dc5bf5f4ed073e4b6db7766ad7295927fb12c38b.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/0e490c3f40a06ca92af69601dc5bf5f4ed073e4b6db7766ad7295927fb12c38b.jpg)

![116ff5bc465c04a37b93bc9bb2d60fe59fee6a7486e6145ee4c0eca510870b09.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/116ff5bc465c04a37b93bc9bb2d60fe59fee6a7486e6145ee4c0eca510870b09.jpg)

![11f087dcd88e6c64e53e4789bed15b1fb2af0723948f56dab015fa90ff2b1fea.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/11f087dcd88e6c64e53e4789bed15b1fb2af0723948f56dab015fa90ff2b1fea.jpg)

![19d57829d5a3deb8f2a2a1d6185850b4ab0d2f821973d780a40d5c07ef5b54c0.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/19d57829d5a3deb8f2a2a1d6185850b4ab0d2f821973d780a40d5c07ef5b54c0.jpg)

![1fcc3cb3304dad7d76feb1862b78bdf626b6df732e22f80d82a5eb17a5bd0c71.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/1fcc3cb3304dad7d76feb1862b78bdf626b6df732e22f80d82a5eb17a5bd0c71.jpg)

![303c0725710f756ad3146ad69f93537551b4469a9efad586f9e1d4952d766e23.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/303c0725710f756ad3146ad69f93537551b4469a9efad586f9e1d4952d766e23.jpg)

![3536027a252f38aba5a2da1a82a1cc95c794755af3ca063cebcd986f73134ceb.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/3536027a252f38aba5a2da1a82a1cc95c794755af3ca063cebcd986f73134ceb.jpg)

![39d0c8054b0a5e1f7d9d8c1cfb6ff75c4119cc8e55e892b5e85c084be5644dd3.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/39d0c8054b0a5e1f7d9d8c1cfb6ff75c4119cc8e55e892b5e85c084be5644dd3.jpg)

![3b526b9a03dd3ba5f17409399390b657e16c1d80c0ccd3d1e26661a58948b08b.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/3b526b9a03dd3ba5f17409399390b657e16c1d80c0ccd3d1e26661a58948b08b.jpg)

![4072cd7c5cc341d9ed0a373dfaa8a776d3587d6a8067b8bde83f1c9c8e2db036.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/4072cd7c5cc341d9ed0a373dfaa8a776d3587d6a8067b8bde83f1c9c8e2db036.jpg)

![62874fdc691df47f34fe49c1386bc4d4ab60394f6d69d4037e8e099de1a45390.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/62874fdc691df47f34fe49c1386bc4d4ab60394f6d69d4037e8e099de1a45390.jpg)

![672c78aed3816aa9766052788da9a77a74911adfa81a5fdedf1ed5ceb463e580.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/672c78aed3816aa9766052788da9a77a74911adfa81a5fdedf1ed5ceb463e580.jpg)

![6fdb05e828532cfef5e94eb0b4c9d4ebd1e2bf14740c866666945efc6be8cdff.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/6fdb05e828532cfef5e94eb0b4c9d4ebd1e2bf14740c866666945efc6be8cdff.jpg)

![70af376799bf31148547a958dd7f55da33c5721907bb57ecc9db193523ea9d88.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/70af376799bf31148547a958dd7f55da33c5721907bb57ecc9db193523ea9d88.jpg)

![75ed8a2ba9a214d9df52390ec7fd401fadd9d9cd4c98a91634020c89850e5de2.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/75ed8a2ba9a214d9df52390ec7fd401fadd9d9cd4c98a91634020c89850e5de2.jpg)

![78795ceb4a942507bccb994dd4bee3b87e96b37ccb7f7328c1599d8c399b5915.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/78795ceb4a942507bccb994dd4bee3b87e96b37ccb7f7328c1599d8c399b5915.jpg)

![97d7c8f857055a28c6be975c1e337962b692fe333c072ae581a6e5da7e4713f8.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/97d7c8f857055a28c6be975c1e337962b692fe333c072ae581a6e5da7e4713f8.jpg)

![aff9e9231e57c9834d71206caf559c45e782dd84414e62a538969b9dd22f21d0.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/aff9e9231e57c9834d71206caf559c45e782dd84414e62a538969b9dd22f21d0.jpg)

![ca4634e9547cb8b77ab143042449dc767564a547a67687a6be96b689273f3e1d.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/ca4634e9547cb8b77ab143042449dc767564a547a67687a6be96b689273f3e1d.jpg)

![caece506a867523912872ae6127324f34a9d111ade436942a28fa48f3c045eba.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/caece506a867523912872ae6127324f34a9d111ade436942a28fa48f3c045eba.jpg)

![e2050083dc0b554de8285a172d4f43481f7d1474360f435fa72b8c058743975c.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/e2050083dc0b554de8285a172d4f43481f7d1474360f435fa72b8c058743975c.jpg)

![ef83a479e98a3b970d6a515e391899fc90630211bdbc2c4cd48a1cf4ab77a7aa.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/ef83a479e98a3b970d6a515e391899fc90630211bdbc2c4cd48a1cf4ab77a7aa.jpg)

![eff1b8d586612c4aeedfaba66932dd76bce3e6833abe8ea8b6b49983e613970a.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/eff1b8d586612c4aeedfaba66932dd76bce3e6833abe8ea8b6b49983e613970a.jpg)

![f49d4d8b266d98b3a74e0e7e2065d0145163fade5b73edcc93be2660e990c533.jpg](../icml_results/2639_BIT-LLM_%20Brain%20Instruction%20Tuned%20LLM%20with%20persistent%20Cross-Attention%20for%20fMRI-to-Text%20Decoding/tables/f49d4d8b266d98b3a74e0e7e2065d0145163fade5b73edcc93be2660e990c533.jpg)

<a id="paper-2606"></a>

## Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy


### Images

![1ab5e98db6d43cebd399518d2c8ce48dba198455d100075f55516dc25a9e0809.jpg](../icml_results/2640_Rethinking%20Contrastive%20Learning%20for%20Graph%20Collaborative%20Filtering_%20Limitations%20and%20a%20Simple%20Remedy/images/1ab5e98db6d43cebd399518d2c8ce48dba198455d100075f55516dc25a9e0809.jpg)

![2918073072c78e280d0a0bbd8d8a2460d61309957706ad6eb8cd22f787f4bc2c.jpg](../icml_results/2640_Rethinking%20Contrastive%20Learning%20for%20Graph%20Collaborative%20Filtering_%20Limitations%20and%20a%20Simple%20Remedy/images/2918073072c78e280d0a0bbd8d8a2460d61309957706ad6eb8cd22f787f4bc2c.jpg)

![32f7d2ee3ed4a4900e276a317895e9417bb232b615feb2bf3d5d341693597c49.jpg](../icml_results/2640_Rethinking%20Contrastive%20Learning%20for%20Graph%20Collaborative%20Filtering_%20Limitations%20and%20a%20Simple%20Remedy/images/32f7d2ee3ed4a4900e276a317895e9417bb232b615feb2bf3d5d341693597c49.jpg)

![99ab0046e5d7c95d4b5a1afde6fa1cfe51e0d784c83779e2a08556e0c23e9314.jpg](../icml_results/2640_Rethinking%20Contrastive%20Learning%20for%20Graph%20Collaborative%20Filtering_%20Limitations%20and%20a%20Simple%20Remedy/images/99ab0046e5d7c95d4b5a1afde6fa1cfe51e0d784c83779e2a08556e0c23e9314.jpg)

![edb44b4d85508f1bbee5b5c691a7385b5d761fc5c01855c5153ed4267e4f4c6d.jpg](../icml_results/2640_Rethinking%20Contrastive%20Learning%20for%20Graph%20Collaborative%20Filtering_%20Limitations%20and%20a%20Simple%20Remedy/images/edb44b4d85508f1bbee5b5c691a7385b5d761fc5c01855c5153ed4267e4f4c6d.jpg)

### Tables

![14dd375b10f2467d348caaac2ef46404fd6a245973cb7e32c9de6d5656099d2a.jpg](../icml_results/2640_Rethinking%20Contrastive%20Learning%20for%20Graph%20Collaborative%20Filtering_%20Limitations%20and%20a%20Simple%20Remedy/tables/14dd375b10f2467d348caaac2ef46404fd6a245973cb7e32c9de6d5656099d2a.jpg)

![3623ce096dcac1547b330aeab0e65d89a4efd1207ddc9289e38045cc1a7deab0.jpg](../icml_results/2640_Rethinking%20Contrastive%20Learning%20for%20Graph%20Collaborative%20Filtering_%20Limitations%20and%20a%20Simple%20Remedy/tables/3623ce096dcac1547b330aeab0e65d89a4efd1207ddc9289e38045cc1a7deab0.jpg)

![61bec260913a5232e7ca01939c3149cb28147dfb8afce2ec109cb761994e4720.jpg](../icml_results/2640_Rethinking%20Contrastive%20Learning%20for%20Graph%20Collaborative%20Filtering_%20Limitations%20and%20a%20Simple%20Remedy/tables/61bec260913a5232e7ca01939c3149cb28147dfb8afce2ec109cb761994e4720.jpg)

![918e36a25d9a75cf089af4588243721dcee8d0baee61f052b8e01a3ddb86751f.jpg](../icml_results/2640_Rethinking%20Contrastive%20Learning%20for%20Graph%20Collaborative%20Filtering_%20Limitations%20and%20a%20Simple%20Remedy/tables/918e36a25d9a75cf089af4588243721dcee8d0baee61f052b8e01a3ddb86751f.jpg)

![b01680629e93ef68cb26199d267b55975c62db346f24b27b6941b154fd26c339.jpg](../icml_results/2640_Rethinking%20Contrastive%20Learning%20for%20Graph%20Collaborative%20Filtering_%20Limitations%20and%20a%20Simple%20Remedy/tables/b01680629e93ef68cb26199d267b55975c62db346f24b27b6941b154fd26c339.jpg)

![b4586c1fed9137131b3adc5428a63bf770a9578f4899ddefc692e3b4baa6e3dc.jpg](../icml_results/2640_Rethinking%20Contrastive%20Learning%20for%20Graph%20Collaborative%20Filtering_%20Limitations%20and%20a%20Simple%20Remedy/tables/b4586c1fed9137131b3adc5428a63bf770a9578f4899ddefc692e3b4baa6e3dc.jpg)

![b72065a58345b9f9caf90904963bccd3dc976ae2154bcb8eb68e08f3df559689.jpg](../icml_results/2640_Rethinking%20Contrastive%20Learning%20for%20Graph%20Collaborative%20Filtering_%20Limitations%20and%20a%20Simple%20Remedy/tables/b72065a58345b9f9caf90904963bccd3dc976ae2154bcb8eb68e08f3df559689.jpg)

<a id="paper-2607"></a>

## Geometry-Aware Image Flow Matching

### Images

![1924cedce0547a0eaeac1582afd4cbed799966c374ad046d19d5b2e8df8c963a.jpg](../icml_results/2641_Geometry-Aware%20Image%20Flow%20Matching/images/1924cedce0547a0eaeac1582afd4cbed799966c374ad046d19d5b2e8df8c963a.jpg)

![325324094f09585f32c9190e11fd7085e03b66b6c9ebb286ceb1fa0fabfef988.jpg](../icml_results/2641_Geometry-Aware%20Image%20Flow%20Matching/images/325324094f09585f32c9190e11fd7085e03b66b6c9ebb286ceb1fa0fabfef988.jpg)

![34cf0fc393efaf76df8d3f08cdcec0b11c29ff75b06188317aeef054bda20097.jpg](../icml_results/2641_Geometry-Aware%20Image%20Flow%20Matching/images/34cf0fc393efaf76df8d3f08cdcec0b11c29ff75b06188317aeef054bda20097.jpg)

![762d660dbeb69203270ec0ffe3059fefcbe0f41b33dc33a71eff3d998cfc5e1a.jpg](../icml_results/2641_Geometry-Aware%20Image%20Flow%20Matching/images/762d660dbeb69203270ec0ffe3059fefcbe0f41b33dc33a71eff3d998cfc5e1a.jpg)

![993e8f6ca79624cd682f277a22644b1bc65ff13679e0a407d6451c607d0686c1.jpg](../icml_results/2641_Geometry-Aware%20Image%20Flow%20Matching/images/993e8f6ca79624cd682f277a22644b1bc65ff13679e0a407d6451c607d0686c1.jpg)

![a1991079660db66143d4806870726a21474780315673f351bbaa4ce94742faa3.jpg](../icml_results/2641_Geometry-Aware%20Image%20Flow%20Matching/images/a1991079660db66143d4806870726a21474780315673f351bbaa4ce94742faa3.jpg)

![a4f1304332c34f15507e3e91acaf91ec4bdc6a4cb61c4c9d6b77055fcc5cb594.jpg](../icml_results/2641_Geometry-Aware%20Image%20Flow%20Matching/images/a4f1304332c34f15507e3e91acaf91ec4bdc6a4cb61c4c9d6b77055fcc5cb594.jpg)

![c60aa1ddf31f2997bbaa5d6fc690444652546348fc4d1d96b33e9676ca6a34d7.jpg](../icml_results/2641_Geometry-Aware%20Image%20Flow%20Matching/images/c60aa1ddf31f2997bbaa5d6fc690444652546348fc4d1d96b33e9676ca6a34d7.jpg)

![e7e39f59391ef857f859e9dafd4375d1d3c8fdf72c68e2a9c78e0d47c6c9f23e.jpg](../icml_results/2641_Geometry-Aware%20Image%20Flow%20Matching/images/e7e39f59391ef857f859e9dafd4375d1d3c8fdf72c68e2a9c78e0d47c6c9f23e.jpg)

### Tables

![13303354aea22d90a9720578eb2990464a6c36e0f57b8b490ff4556cea0336d3.jpg](../icml_results/2641_Geometry-Aware%20Image%20Flow%20Matching/tables/13303354aea22d90a9720578eb2990464a6c36e0f57b8b490ff4556cea0336d3.jpg)

![1bfa0c4574982eb99e901853610349415dd5de832e310cdef6024486b6d40938.jpg](../icml_results/2641_Geometry-Aware%20Image%20Flow%20Matching/tables/1bfa0c4574982eb99e901853610349415dd5de832e310cdef6024486b6d40938.jpg)

![9ac42dc174d2896c70888746ee8c8b56d8847644c5161e97c5fe01ee7cd37a01.jpg](../icml_results/2641_Geometry-Aware%20Image%20Flow%20Matching/tables/9ac42dc174d2896c70888746ee8c8b56d8847644c5161e97c5fe01ee7cd37a01.jpg)

![9bbb8571630348c0bbbff6c3c36a5ef332492d9c7f30c97d867134fdb2e7cc14.jpg](../icml_results/2641_Geometry-Aware%20Image%20Flow%20Matching/tables/9bbb8571630348c0bbbff6c3c36a5ef332492d9c7f30c97d867134fdb2e7cc14.jpg)

![bf186e58ec5915e66e62de0c950530c0a7adc9c48ae77d74d0d1d66cf8d3ecd3.jpg](../icml_results/2641_Geometry-Aware%20Image%20Flow%20Matching/tables/bf186e58ec5915e66e62de0c950530c0a7adc9c48ae77d74d0d1d66cf8d3ecd3.jpg)

![c53e4ab3a8c1273cbc8df447fcefa3deeffdf0ee1a35cdb6773e9d533e0fc906.jpg](../icml_results/2641_Geometry-Aware%20Image%20Flow%20Matching/tables/c53e4ab3a8c1273cbc8df447fcefa3deeffdf0ee1a35cdb6773e9d533e0fc906.jpg)

![c803caff6274c20e3da295d458779bdf3a08a745ed62381c4311b31293f33781.jpg](../icml_results/2641_Geometry-Aware%20Image%20Flow%20Matching/tables/c803caff6274c20e3da295d458779bdf3a08a745ed62381c4311b31293f33781.jpg)
