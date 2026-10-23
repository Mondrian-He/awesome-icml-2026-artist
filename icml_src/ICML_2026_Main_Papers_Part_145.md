# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 145 of 197

## 目录 (Table of Contents)

1. [ConsMSA: Semantic Distribution Consistency Learning for Multimodal Sentiment Analysis](#paper-4753)
2. [Test-Time Guidance for Flow-Based Generative Models via Parallel Tempering on Source Distributions](#paper-4754)
3. [Negative Sampling From the Ground Up: A Redesign for Recommendations](#paper-4755)
4. [Elastic Diffusion Transformer](#paper-4756)
5. [Beyond Single-View Indexing: Structure-Aware Multi-View Retrieval for Knowledge-Based VQA](#paper-4757)
6. [AdaGC: Enhancing LLM Pretraining Stability via Adaptive Gradient Clipping](#paper-4758)
7. [NoiseSDF2NoiseSDF: Learning Clean Neural Fields from Noisy Supervision](#paper-4759)
8. [Test-Time Debiasing with Probabilistic Prompts via Wasserstein Distance in Vision-Language Models](#paper-4760)
9. [Anchored Policy Optimization: Mitigating Exploration Collapse via Support-Constrained Rectification](#paper-4761)
10. [ECo-MoE: Embodiment-Conditioned Mixture of Experts Increases the Evolvability of Robots](#paper-4762)
11. [One Coin Has Two Sides: Single Positive Multi Label Learning from Salient Annotations](#paper-4763)
12. [From Coarse to Fine: Deep Prototype Refinement Network for Few-Shot Point Cloud Semantic Segmentation](#paper-4764)
13. [DeepHA: Scaling Action Chains Elicits Deep Hierarchical Agents](#paper-4765)
14. [OpenHA: A Series of Open-Source Hierarchical Agentic Models in Minecraft](#paper-4766)
15. [Reflect-then-Correct: Rebalancing Task Optimization for Generalizable Meta-Reinforcement Learning via Distributional Value Error Reduction](#paper-4767)
16. [InfraRL: A Benchmark for Constrained Resource Allocation in Large-Scale Infrastructure Asset Management](#paper-4768)
17. [Decomposing the Basic Abilities of Large Language Models: Mitigating Cross-Task Interference in Multi-Task Instruct-Tuning](#paper-4769)
18. [SLIP-RS: Structured-Attribute Language-Image Pre-Training for Remote Sensing Object Detection](#paper-4770)
19. [Lightweight Federated Incremental Learning via Decoupled Replay](#paper-4771)
20. [Universal Skeleton Understanding via Differentiable Rendering and MLLMs](#paper-4772)
21. [CodeChemist: Test-Time Scaling for Low-Resource Code Generation via Functional Knowledge Transfer](#paper-4773)
22. [Muon in Associative Memory Learning: Training Dynamics and Scaling Laws](#paper-4774)
23. [Learning Attribute–Affordance Hierarchies in Hyperbolic Space for Open-Vocabulary 3D Object Affordance Grounding](#paper-4775)
24. [SuCo: Sufficiency-guided Continuous Adaptive Reasoning](#paper-4776)
25. [CausalRAG2: Hierarchical Causal Knowledge Graph Design for RAG](#paper-4777)
26. [Reward Shaping for (Inference-Time) Alignment: A Stackelberg Game Perspective](#paper-4778)
27. [Crisp: A Spectral-Based Interaction Strategy for Multivariate Time Series Forecasting](#paper-4779)
28. [Towards Understanding Massive Activations in Attention Sink Mechanism](#paper-4780)
29. [E-mem: Multi-Agent Based Episodic Context Reconstruction for LLM Agent Memory](#paper-4781)
30. [Preference-Calibrated Optimization with Score-Level Distribution Alignment for Text-to-Image Diffusion Model Unlearning](#paper-4782)
31. [Beyond Static Pipelines: Learning Dynamic Workflows for Text-to-SQL](#paper-4783)
32. [Beyond Model Base Retrieval: Weaving Knowledge to Master Fine-grained Neural Network Design](#paper-4784)
33. [DTKG: Dual-Track Knowledge Graph-Verified Reasoning Framework for Multi-Hop QA](#paper-4785)

---


<a id="paper-4753"></a>

## ConsMSA: Semantic Distribution Consistency Learning for Multimodal Sentiment Analysis

### Images

![13f6125bd0f4d10081419aa1b37f9864fc45722433907616efd238ba6ba114c7.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/13f6125bd0f4d10081419aa1b37f9864fc45722433907616efd238ba6ba114c7.jpg)

![15e7c3363b031be4f679a8f83d18e606e4c5b7e2829f11b01a4301b5ad8bf165.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/15e7c3363b031be4f679a8f83d18e606e4c5b7e2829f11b01a4301b5ad8bf165.jpg)

![3a89c2ddf53ed7c1ff4696dfe19675239f3855a54889857b4408109a24e520d4.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/3a89c2ddf53ed7c1ff4696dfe19675239f3855a54889857b4408109a24e520d4.jpg)

![3c0ddf9ee3017cc0de53ba725b52caba25745a4c596fc8181efff11b7771fd6b.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/3c0ddf9ee3017cc0de53ba725b52caba25745a4c596fc8181efff11b7771fd6b.jpg)

![3c9ff92c67ccfdbd37c952855848cabae949785410839d91a264a423517f210b.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/3c9ff92c67ccfdbd37c952855848cabae949785410839d91a264a423517f210b.jpg)

![3d9096193950663a5559136120b1bec988542b166fc7698ca303080e03cee13b.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/3d9096193950663a5559136120b1bec988542b166fc7698ca303080e03cee13b.jpg)

![4ef856ca4f6a6be75ad3d716a3fe2bb8aeb4a3187a10ebf1607e104a18464349.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/4ef856ca4f6a6be75ad3d716a3fe2bb8aeb4a3187a10ebf1607e104a18464349.jpg)

![5ce30542dfad26de530f945c59cd9813e575e3f69c0cc5099d923c79ece6a748.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/5ce30542dfad26de530f945c59cd9813e575e3f69c0cc5099d923c79ece6a748.jpg)

![7acddc3c0f582bbe087a6b74f2ad56302a3c15ea55b6d1d21a25bbc98eae1a4e.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/7acddc3c0f582bbe087a6b74f2ad56302a3c15ea55b6d1d21a25bbc98eae1a4e.jpg)

![8581c21864af0f8fac0d087ca50a5b2afe0529bfa6da8ea8aada07ea5bfe5c74.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/8581c21864af0f8fac0d087ca50a5b2afe0529bfa6da8ea8aada07ea5bfe5c74.jpg)

![959080bdd69fbdf3275c6e4b469e3f30128d6be5bce634ab8c871f5099aeea66.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/959080bdd69fbdf3275c6e4b469e3f30128d6be5bce634ab8c871f5099aeea66.jpg)

![a79de15b1911ff87920f6388e9ba26899ddccf236f7a1cdb12dab1b35f5b04de.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/a79de15b1911ff87920f6388e9ba26899ddccf236f7a1cdb12dab1b35f5b04de.jpg)

![fc0596a4a6a878a20ca12705da702883848993853c7e46d8666fba7d407dda2a.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/fc0596a4a6a878a20ca12705da702883848993853c7e46d8666fba7d407dda2a.jpg)

### Tables

![281f0c8b439a38eed16f02801b9a3bb93aed8b2cf0b96c94a0d23c3996f9376a.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/tables/281f0c8b439a38eed16f02801b9a3bb93aed8b2cf0b96c94a0d23c3996f9376a.jpg)

![518fdb96b99c45856e66b233f33adde8c8b7616685a25f27197eced3bf9f8539.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/tables/518fdb96b99c45856e66b233f33adde8c8b7616685a25f27197eced3bf9f8539.jpg)

![5ec3275ece175b727eaf0240b610d31f9de58b6986da2f91ed471286879af599.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/tables/5ec3275ece175b727eaf0240b610d31f9de58b6986da2f91ed471286879af599.jpg)

![da8d9b797c5ff756fe38ede140e3c0da57fb718ab00e9ae801c654b917101b45.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/tables/da8d9b797c5ff756fe38ede140e3c0da57fb718ab00e9ae801c654b917101b45.jpg)

![f6535214da12d182c00c58e3ad4a362c04f0d686a744adae03ccccb1184b39b0.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/tables/f6535214da12d182c00c58e3ad4a362c04f0d686a744adae03ccccb1184b39b0.jpg)

## ConsMSA: Semantic Distribution Consistency Learning for Multimodal Sentiment Analysis


### Images

![4a22087c8945c479f380956cb9928e20db1cb698ea089563cf912cc20cf0cae0.jpg](../icml_results/4801_ConsMSA_%20Semantic%20Distribution%20Consistency%20Learning%20for%20Multimodal%20Sentiment%20Analysis/images/4a22087c8945c479f380956cb9928e20db1cb698ea089563cf912cc20cf0cae0.jpg)

![8cb79e1d454715c85690a0d711d0a9b2dc6890b1c34bb220767c7a507a02d623.jpg](../icml_results/4801_ConsMSA_%20Semantic%20Distribution%20Consistency%20Learning%20for%20Multimodal%20Sentiment%20Analysis/images/8cb79e1d454715c85690a0d711d0a9b2dc6890b1c34bb220767c7a507a02d623.jpg)

![9e9e8b2d126f4076331b1f833e22eb9347c5e13fb4ffb2cb6ebed91bb92b2e88.jpg](../icml_results/4801_ConsMSA_%20Semantic%20Distribution%20Consistency%20Learning%20for%20Multimodal%20Sentiment%20Analysis/images/9e9e8b2d126f4076331b1f833e22eb9347c5e13fb4ffb2cb6ebed91bb92b2e88.jpg)

![b7b4d3182dbb56c19e633f26c60a47101b06c26177ecaa2f82dcb8908127371c.jpg](../icml_results/4801_ConsMSA_%20Semantic%20Distribution%20Consistency%20Learning%20for%20Multimodal%20Sentiment%20Analysis/images/b7b4d3182dbb56c19e633f26c60a47101b06c26177ecaa2f82dcb8908127371c.jpg)

![d636ac2a7a00a1ee94156a40e22bd2ab534cf2fc1f9f46173ec71ee0b31d2291.jpg](../icml_results/4801_ConsMSA_%20Semantic%20Distribution%20Consistency%20Learning%20for%20Multimodal%20Sentiment%20Analysis/images/d636ac2a7a00a1ee94156a40e22bd2ab534cf2fc1f9f46173ec71ee0b31d2291.jpg)

### Tables

![25813a744fca0634f8d7a7fd12b9dc10e9f1b1b2ee6c62e12a6793e1610f5597.jpg](../icml_results/4801_ConsMSA_%20Semantic%20Distribution%20Consistency%20Learning%20for%20Multimodal%20Sentiment%20Analysis/tables/25813a744fca0634f8d7a7fd12b9dc10e9f1b1b2ee6c62e12a6793e1610f5597.jpg)

![73462e8fe4055e0c392e112bf2561bd18c349819d4a8222fc9aa5393f52c685d.jpg](../icml_results/4801_ConsMSA_%20Semantic%20Distribution%20Consistency%20Learning%20for%20Multimodal%20Sentiment%20Analysis/tables/73462e8fe4055e0c392e112bf2561bd18c349819d4a8222fc9aa5393f52c685d.jpg)

![79b35ff17bb571118c37ab83f6e14f3cb55e395991514808ad33bc6551cbf1ab.jpg](../icml_results/4801_ConsMSA_%20Semantic%20Distribution%20Consistency%20Learning%20for%20Multimodal%20Sentiment%20Analysis/tables/79b35ff17bb571118c37ab83f6e14f3cb55e395991514808ad33bc6551cbf1ab.jpg)

![a181884a6a58056a03b14b3f797fcf7798a39980f424d1369012720a405bb57a.jpg](../icml_results/4801_ConsMSA_%20Semantic%20Distribution%20Consistency%20Learning%20for%20Multimodal%20Sentiment%20Analysis/tables/a181884a6a58056a03b14b3f797fcf7798a39980f424d1369012720a405bb57a.jpg)

![a9d3409968e720be2e7feca0f656e47968d5a15c994a2ff876c2862650f4bed3.jpg](../icml_results/4801_ConsMSA_%20Semantic%20Distribution%20Consistency%20Learning%20for%20Multimodal%20Sentiment%20Analysis/tables/a9d3409968e720be2e7feca0f656e47968d5a15c994a2ff876c2862650f4bed3.jpg)

![b91387903ab8acfd04029345ae24d308e059aeb3b2a6db5e1423d97eb6f0a073.jpg](../icml_results/4801_ConsMSA_%20Semantic%20Distribution%20Consistency%20Learning%20for%20Multimodal%20Sentiment%20Analysis/tables/b91387903ab8acfd04029345ae24d308e059aeb3b2a6db5e1423d97eb6f0a073.jpg)

![c41cd21ace951471ae29d7b6b908418fe3803c0312027547ab82e1546cfb2b56.jpg](../icml_results/4801_ConsMSA_%20Semantic%20Distribution%20Consistency%20Learning%20for%20Multimodal%20Sentiment%20Analysis/tables/c41cd21ace951471ae29d7b6b908418fe3803c0312027547ab82e1546cfb2b56.jpg)

![c9729d8ffdf47fad8a2c977c8dafbbf692f4a25e4d09ed7039b774ecc1500b44.jpg](../icml_results/4801_ConsMSA_%20Semantic%20Distribution%20Consistency%20Learning%20for%20Multimodal%20Sentiment%20Analysis/tables/c9729d8ffdf47fad8a2c977c8dafbbf692f4a25e4d09ed7039b774ecc1500b44.jpg)

![ca99bf8b67766444dab3fdaa984afc027588727f57fc0af7957807c7eaeaa924.jpg](../icml_results/4801_ConsMSA_%20Semantic%20Distribution%20Consistency%20Learning%20for%20Multimodal%20Sentiment%20Analysis/tables/ca99bf8b67766444dab3fdaa984afc027588727f57fc0af7957807c7eaeaa924.jpg)

![e7db028a55d7eeabd5b96bff97e3a3851de7345a5a0cba2124f77a6483a1a465.jpg](../icml_results/4801_ConsMSA_%20Semantic%20Distribution%20Consistency%20Learning%20for%20Multimodal%20Sentiment%20Analysis/tables/e7db028a55d7eeabd5b96bff97e3a3851de7345a5a0cba2124f77a6483a1a465.jpg)

![f899479fee26878119c7c313a59e669ee9b1b1c814573257b0ace2284db917bb.jpg](../icml_results/4801_ConsMSA_%20Semantic%20Distribution%20Consistency%20Learning%20for%20Multimodal%20Sentiment%20Analysis/tables/f899479fee26878119c7c313a59e669ee9b1b1c814573257b0ace2284db917bb.jpg)

![fb02dd9a53ac076cd3e4c8498e1fc1d94bab1ba3a4b74f4ad552da1126b4ec69.jpg](../icml_results/4801_ConsMSA_%20Semantic%20Distribution%20Consistency%20Learning%20for%20Multimodal%20Sentiment%20Analysis/tables/fb02dd9a53ac076cd3e4c8498e1fc1d94bab1ba3a4b74f4ad552da1126b4ec69.jpg)

<a id="paper-4754"></a>

## Test-Time Guidance for Flow-Based Generative Models via Parallel Tempering on Source Distributions


### Images

![045aa3bf9b4391ae71763c070c7ca770df96d37a985cdaf0f2d70f1c5bd390b4.jpg](../icml_results/4802_Test-Time%20Guidance%20for%20Flow-Based%20Generative%20Models%20via%20Parallel%20Tempering%20on%20Source%20Distributions/images/045aa3bf9b4391ae71763c070c7ca770df96d37a985cdaf0f2d70f1c5bd390b4.jpg)

![0fc024718cb2755fc98b9e45051a481b34a33ea95b57d88997696fd1cf659e7b.jpg](../icml_results/4802_Test-Time%20Guidance%20for%20Flow-Based%20Generative%20Models%20via%20Parallel%20Tempering%20on%20Source%20Distributions/images/0fc024718cb2755fc98b9e45051a481b34a33ea95b57d88997696fd1cf659e7b.jpg)

![1d3e4d66d60337f1ada26f75c2d5866ec0ca34bab1596ea94b89d9859690e9c4.jpg](../icml_results/4802_Test-Time%20Guidance%20for%20Flow-Based%20Generative%20Models%20via%20Parallel%20Tempering%20on%20Source%20Distributions/images/1d3e4d66d60337f1ada26f75c2d5866ec0ca34bab1596ea94b89d9859690e9c4.jpg)

![2225135a686b3747991cdf7055b45537be85db2ab5753e1983694e63c522a38a.jpg](../icml_results/4802_Test-Time%20Guidance%20for%20Flow-Based%20Generative%20Models%20via%20Parallel%20Tempering%20on%20Source%20Distributions/images/2225135a686b3747991cdf7055b45537be85db2ab5753e1983694e63c522a38a.jpg)

![5dd1e8363ea2eb30110c8c63dd2e33e47f99cf039013232a1ec65a43c7a4d7a6.jpg](../icml_results/4802_Test-Time%20Guidance%20for%20Flow-Based%20Generative%20Models%20via%20Parallel%20Tempering%20on%20Source%20Distributions/images/5dd1e8363ea2eb30110c8c63dd2e33e47f99cf039013232a1ec65a43c7a4d7a6.jpg)

![6a5bacab5eeb51a74f4d7aeae591981c9a0fce863532dd3f32e97d705712d38b.jpg](../icml_results/4802_Test-Time%20Guidance%20for%20Flow-Based%20Generative%20Models%20via%20Parallel%20Tempering%20on%20Source%20Distributions/images/6a5bacab5eeb51a74f4d7aeae591981c9a0fce863532dd3f32e97d705712d38b.jpg)

![88e0e423fc06de60ab95f70b23782efde41e5424690bc967a02bb9bdb638fb33.jpg](../icml_results/4802_Test-Time%20Guidance%20for%20Flow-Based%20Generative%20Models%20via%20Parallel%20Tempering%20on%20Source%20Distributions/images/88e0e423fc06de60ab95f70b23782efde41e5424690bc967a02bb9bdb638fb33.jpg)

![92e6057204c284231f891036e1e178241eb894e50c3d45344a43e3fb495736ce.jpg](../icml_results/4802_Test-Time%20Guidance%20for%20Flow-Based%20Generative%20Models%20via%20Parallel%20Tempering%20on%20Source%20Distributions/images/92e6057204c284231f891036e1e178241eb894e50c3d45344a43e3fb495736ce.jpg)

![dfe1df7e101457783f26350abd2088ccc0b32e7a668b903b44d88260273e10de.jpg](../icml_results/4802_Test-Time%20Guidance%20for%20Flow-Based%20Generative%20Models%20via%20Parallel%20Tempering%20on%20Source%20Distributions/images/dfe1df7e101457783f26350abd2088ccc0b32e7a668b903b44d88260273e10de.jpg)

### Tables

![0808428b9e04bc7c0248eaf9903033443ae97878f5e490b124b2c991272b9a2a.jpg](../icml_results/4802_Test-Time%20Guidance%20for%20Flow-Based%20Generative%20Models%20via%20Parallel%20Tempering%20on%20Source%20Distributions/tables/0808428b9e04bc7c0248eaf9903033443ae97878f5e490b124b2c991272b9a2a.jpg)

![3647d558b337b78d0895fe7454459bfcb107ddfa7fa2e1cbc2db2d1f1625c017.jpg](../icml_results/4802_Test-Time%20Guidance%20for%20Flow-Based%20Generative%20Models%20via%20Parallel%20Tempering%20on%20Source%20Distributions/tables/3647d558b337b78d0895fe7454459bfcb107ddfa7fa2e1cbc2db2d1f1625c017.jpg)

![59cec3d5db20fd85eba9ca4535594c265b111e59c1dc520df1cdf1814c1087e4.jpg](../icml_results/4802_Test-Time%20Guidance%20for%20Flow-Based%20Generative%20Models%20via%20Parallel%20Tempering%20on%20Source%20Distributions/tables/59cec3d5db20fd85eba9ca4535594c265b111e59c1dc520df1cdf1814c1087e4.jpg)

![5bf8347080f07209fbb763eb0ef6c69faa0226aa59117e44d6312252183d3ad3.jpg](../icml_results/4802_Test-Time%20Guidance%20for%20Flow-Based%20Generative%20Models%20via%20Parallel%20Tempering%20on%20Source%20Distributions/tables/5bf8347080f07209fbb763eb0ef6c69faa0226aa59117e44d6312252183d3ad3.jpg)

![6067556ef006288cac020902b184a23b05ea161e6e8efddd283ce543cf0bce48.jpg](../icml_results/4802_Test-Time%20Guidance%20for%20Flow-Based%20Generative%20Models%20via%20Parallel%20Tempering%20on%20Source%20Distributions/tables/6067556ef006288cac020902b184a23b05ea161e6e8efddd283ce543cf0bce48.jpg)

![74e126cda90e15eb2acc17ebdcd2af2b8b2042f27f7057d04f042b18c2efb981.jpg](../icml_results/4802_Test-Time%20Guidance%20for%20Flow-Based%20Generative%20Models%20via%20Parallel%20Tempering%20on%20Source%20Distributions/tables/74e126cda90e15eb2acc17ebdcd2af2b8b2042f27f7057d04f042b18c2efb981.jpg)

![a7080fe31f2299d90e2a9295708711bd3bb7333a2d850de182aa4ec3b02487ea.jpg](../icml_results/4802_Test-Time%20Guidance%20for%20Flow-Based%20Generative%20Models%20via%20Parallel%20Tempering%20on%20Source%20Distributions/tables/a7080fe31f2299d90e2a9295708711bd3bb7333a2d850de182aa4ec3b02487ea.jpg)

![cc8d3b78e6c010b51ab296af3ac9de13f48e7555126d80b611c0837e0ff82ba6.jpg](../icml_results/4802_Test-Time%20Guidance%20for%20Flow-Based%20Generative%20Models%20via%20Parallel%20Tempering%20on%20Source%20Distributions/tables/cc8d3b78e6c010b51ab296af3ac9de13f48e7555126d80b611c0837e0ff82ba6.jpg)

<a id="paper-4755"></a>

## Negative Sampling From the Ground Up: A Redesign for Recommendations


### Images

![14f8d73e9e09e08b8c1adfb83ff3034d2710945388cd3f4dae3c3fec5df0f78b.jpg](../icml_results/4803_Negative%20Sampling%20From%20the%20Ground%20Up_%20A%20Redesign%20for%20Recommendations/images/14f8d73e9e09e08b8c1adfb83ff3034d2710945388cd3f4dae3c3fec5df0f78b.jpg)

![6c403926b60a845fc4d9015317cd2c5031ead05adf2ea47e5ff53d4eb42b1939.jpg](../icml_results/4803_Negative%20Sampling%20From%20the%20Ground%20Up_%20A%20Redesign%20for%20Recommendations/images/6c403926b60a845fc4d9015317cd2c5031ead05adf2ea47e5ff53d4eb42b1939.jpg)

![d850e5da989ac9d6a1f4b8c7c7b2c8524bac53493a22966df6e34549f3ca51ec.jpg](../icml_results/4803_Negative%20Sampling%20From%20the%20Ground%20Up_%20A%20Redesign%20for%20Recommendations/images/d850e5da989ac9d6a1f4b8c7c7b2c8524bac53493a22966df6e34549f3ca51ec.jpg)

### Tables

![441cf71465d279fd17ca740a549a875fe904a1e572a34dfd124714963b09f1e9.jpg](../icml_results/4803_Negative%20Sampling%20From%20the%20Ground%20Up_%20A%20Redesign%20for%20Recommendations/tables/441cf71465d279fd17ca740a549a875fe904a1e572a34dfd124714963b09f1e9.jpg)

![477d413f038e78e60c95ce55c0d8e712338cc6bc29935d58962344132afe78e1.jpg](../icml_results/4803_Negative%20Sampling%20From%20the%20Ground%20Up_%20A%20Redesign%20for%20Recommendations/tables/477d413f038e78e60c95ce55c0d8e712338cc6bc29935d58962344132afe78e1.jpg)

![4f7067899efa435685f21dd416f1b65487feacb96db87a39ad26bbe3ff4c1a28.jpg](../icml_results/4803_Negative%20Sampling%20From%20the%20Ground%20Up_%20A%20Redesign%20for%20Recommendations/tables/4f7067899efa435685f21dd416f1b65487feacb96db87a39ad26bbe3ff4c1a28.jpg)

![581436d0385aa20ce270122eef6d65e65f1be962d1f9a9d7b146108a7c0c3175.jpg](../icml_results/4803_Negative%20Sampling%20From%20the%20Ground%20Up_%20A%20Redesign%20for%20Recommendations/tables/581436d0385aa20ce270122eef6d65e65f1be962d1f9a9d7b146108a7c0c3175.jpg)

![6383a86f11e65e52cbaec1d17f6fe466ef7c1af98d6cd28f415bed51249b6c59.jpg](../icml_results/4803_Negative%20Sampling%20From%20the%20Ground%20Up_%20A%20Redesign%20for%20Recommendations/tables/6383a86f11e65e52cbaec1d17f6fe466ef7c1af98d6cd28f415bed51249b6c59.jpg)

![6d507251cbf5b30cdca46055fd116c3f4b058330934786217197c351fa9d3e05.jpg](../icml_results/4803_Negative%20Sampling%20From%20the%20Ground%20Up_%20A%20Redesign%20for%20Recommendations/tables/6d507251cbf5b30cdca46055fd116c3f4b058330934786217197c351fa9d3e05.jpg)

![c3703e6d6d8142a30907d5fd7d52f4a40714d2e588ec85a5a85e3f4dc2892566.jpg](../icml_results/4803_Negative%20Sampling%20From%20the%20Ground%20Up_%20A%20Redesign%20for%20Recommendations/tables/c3703e6d6d8142a30907d5fd7d52f4a40714d2e588ec85a5a85e3f4dc2892566.jpg)

![df14f5fd2948ea858b17794d5235733353d3be504cf7703f3137206d154106fe.jpg](../icml_results/4803_Negative%20Sampling%20From%20the%20Ground%20Up_%20A%20Redesign%20for%20Recommendations/tables/df14f5fd2948ea858b17794d5235733353d3be504cf7703f3137206d154106fe.jpg)

![e2a6359e52e51a2a1d248d116743d1a7fd0388ac8049f4c130b7e5d4db69864f.jpg](../icml_results/4803_Negative%20Sampling%20From%20the%20Ground%20Up_%20A%20Redesign%20for%20Recommendations/tables/e2a6359e52e51a2a1d248d116743d1a7fd0388ac8049f4c130b7e5d4db69864f.jpg)

<a id="paper-4756"></a>

## Elastic Diffusion Transformer


### Images

![0b7818af8899014079b7d0f8dba5b099ae030a65de4f7ffdf0cc0e2b19c3cb9a.jpg](../icml_results/4804_Elastic%20Diffusion%20Transformer/images/0b7818af8899014079b7d0f8dba5b099ae030a65de4f7ffdf0cc0e2b19c3cb9a.jpg)

![0d902694f39d0a432adad5482261aa6fdb0170bd58012d89494c36f16dfddd69.jpg](../icml_results/4804_Elastic%20Diffusion%20Transformer/images/0d902694f39d0a432adad5482261aa6fdb0170bd58012d89494c36f16dfddd69.jpg)

![12cf1a5009906974aa8cfe906a90b5479ad67818e2ee5e75aeb785b5d825f169.jpg](../icml_results/4804_Elastic%20Diffusion%20Transformer/images/12cf1a5009906974aa8cfe906a90b5479ad67818e2ee5e75aeb785b5d825f169.jpg)

![4e7ef6f1f74d498a870a82eb7786c8b6ab4c39b46d46c10663089aaf979eeaa2.jpg](../icml_results/4804_Elastic%20Diffusion%20Transformer/images/4e7ef6f1f74d498a870a82eb7786c8b6ab4c39b46d46c10663089aaf979eeaa2.jpg)

![76ff69c046ee6fa98c2ccf66d70a45ed44f83fdaa2de271e3da4ca9871aef6ba.jpg](../icml_results/4804_Elastic%20Diffusion%20Transformer/images/76ff69c046ee6fa98c2ccf66d70a45ed44f83fdaa2de271e3da4ca9871aef6ba.jpg)

![8dc9c05986018f9c3eeab51a01429997ae7616e363877ccce091d2aeab842de6.jpg](../icml_results/4804_Elastic%20Diffusion%20Transformer/images/8dc9c05986018f9c3eeab51a01429997ae7616e363877ccce091d2aeab842de6.jpg)

![a00f8812211a9a33983d6a5ebd853ddd941b7846cd8404a7800cb36df2f3b057.jpg](../icml_results/4804_Elastic%20Diffusion%20Transformer/images/a00f8812211a9a33983d6a5ebd853ddd941b7846cd8404a7800cb36df2f3b057.jpg)

![e0e83af34f6547c391d0cc353d883dd90c3dadb518f5e42c1b237ef845272128.jpg](../icml_results/4804_Elastic%20Diffusion%20Transformer/images/e0e83af34f6547c391d0cc353d883dd90c3dadb518f5e42c1b237ef845272128.jpg)

![f268ee969d0f67d93d9d9f672e50c18c8ab48e628362f7b4c7ef678af2bdc3ab.jpg](../icml_results/4804_Elastic%20Diffusion%20Transformer/images/f268ee969d0f67d93d9d9f672e50c18c8ab48e628362f7b4c7ef678af2bdc3ab.jpg)

### Tables

![328fd4bd7f97422fe88eecad2c60bf09bd8bcfea7c81017a5328a422fbe32510.jpg](../icml_results/4804_Elastic%20Diffusion%20Transformer/tables/328fd4bd7f97422fe88eecad2c60bf09bd8bcfea7c81017a5328a422fbe32510.jpg)

![4ca1da17aeeef5bb6e33f17e8576cfd27185cc9d9b800d2dd45106fdd7d83cad.jpg](../icml_results/4804_Elastic%20Diffusion%20Transformer/tables/4ca1da17aeeef5bb6e33f17e8576cfd27185cc9d9b800d2dd45106fdd7d83cad.jpg)

![4fbb2b108d249a2cdb640bebae0b03c9b6a536cf00c839b44db08d73bbf76350.jpg](../icml_results/4804_Elastic%20Diffusion%20Transformer/tables/4fbb2b108d249a2cdb640bebae0b03c9b6a536cf00c839b44db08d73bbf76350.jpg)

![5017d6abe4b5572c3dab5e2e62ad01864913f4ed1b16db46fc33168f478ba9a7.jpg](../icml_results/4804_Elastic%20Diffusion%20Transformer/tables/5017d6abe4b5572c3dab5e2e62ad01864913f4ed1b16db46fc33168f478ba9a7.jpg)

![79eff4d25c2d85b94f0efd88801073cd6f9cfb72248bc879e96d8a0171914ea4.jpg](../icml_results/4804_Elastic%20Diffusion%20Transformer/tables/79eff4d25c2d85b94f0efd88801073cd6f9cfb72248bc879e96d8a0171914ea4.jpg)

![84d065ba1e34c0fc4bcc4df3619585cd282eb69a64fd7e3dfab8a24facf6d704.jpg](../icml_results/4804_Elastic%20Diffusion%20Transformer/tables/84d065ba1e34c0fc4bcc4df3619585cd282eb69a64fd7e3dfab8a24facf6d704.jpg)

<a id="paper-4757"></a>

## Beyond Single-View Indexing: Structure-Aware Multi-View Retrieval for Knowledge-Based VQA


### Images

![6b75d8fe61e25a6e04e2a0a8aed77481dd01e85f70e522a0005b0260ce8f4f33.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/images/6b75d8fe61e25a6e04e2a0a8aed77481dd01e85f70e522a0005b0260ce8f4f33.jpg)

![7f8a8eeaf203dc152f93c25a54dea4c89074bca971cbcb290c28480b53a781b5.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/images/7f8a8eeaf203dc152f93c25a54dea4c89074bca971cbcb290c28480b53a781b5.jpg)

![804916c4eaede327f39990c4b6c9ec173fd83d7df3c51439f01c6fd0a808d764.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/images/804916c4eaede327f39990c4b6c9ec173fd83d7df3c51439f01c6fd0a808d764.jpg)

![ad56340a53459925fe5847eefa830a25f0fd48e398dea826951fe01afa6e1bd3.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/images/ad56340a53459925fe5847eefa830a25f0fd48e398dea826951fe01afa6e1bd3.jpg)

![b7624cbc5fbe4468e0ba615aa17c2818f3c28a3dce58e7e7958de86d5938d178.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/images/b7624cbc5fbe4468e0ba615aa17c2818f3c28a3dce58e7e7958de86d5938d178.jpg)

![e9e359a6ac9af362e29f74920c6d0e691c289b6386ff8e0789dd3a39a0d4d9a6.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/images/e9e359a6ac9af362e29f74920c6d0e691c289b6386ff8e0789dd3a39a0d4d9a6.jpg)

### Tables

![23160cf840949fe29e67b7133b2077f38365ed94f80dfde6caf3bf3bc60298b7.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/23160cf840949fe29e67b7133b2077f38365ed94f80dfde6caf3bf3bc60298b7.jpg)

![37942d4ebf646e3ae63346341e949880a54d0f21505650a1e8c4667b3b6d08da.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/37942d4ebf646e3ae63346341e949880a54d0f21505650a1e8c4667b3b6d08da.jpg)

![3caa23d1ffebe56185b327689ca8ad0ee0b4313a99ad105dd6015bfe31ba1f45.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/3caa23d1ffebe56185b327689ca8ad0ee0b4313a99ad105dd6015bfe31ba1f45.jpg)

![5f1d3dee5ec3b384fe8536dec32bc8c117b49dd1b8862872b4695217966aa708.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/5f1d3dee5ec3b384fe8536dec32bc8c117b49dd1b8862872b4695217966aa708.jpg)

![6231bafbb1046990117b3733935169314cfd1fa88bb9854a0d1ad3db190d5ab6.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/6231bafbb1046990117b3733935169314cfd1fa88bb9854a0d1ad3db190d5ab6.jpg)

![639c3c36602dc08a30a44712582ad9c264827c27fba416e2f44e848274667f45.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/639c3c36602dc08a30a44712582ad9c264827c27fba416e2f44e848274667f45.jpg)

![65e99645b7a991f2c868c056a85d44b03440b8c8185129f1e34169b60da0123b.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/65e99645b7a991f2c868c056a85d44b03440b8c8185129f1e34169b60da0123b.jpg)

![7205a3c49a6a1d854c7d13476c1f5fc53f4667f664ea6306b2a4bf90f3e2e565.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/7205a3c49a6a1d854c7d13476c1f5fc53f4667f664ea6306b2a4bf90f3e2e565.jpg)

![785ce7d1de8f2dabc52cb21b6e868c7adecb40dde612332d11127eadc6c0f96d.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/785ce7d1de8f2dabc52cb21b6e868c7adecb40dde612332d11127eadc6c0f96d.jpg)

![86a8ebf10e85a4f6043cc9717849ae6cafbadaeeedfe6f2f449725848471b941.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/86a8ebf10e85a4f6043cc9717849ae6cafbadaeeedfe6f2f449725848471b941.jpg)

![a2de222d727381253829ab574c258b7f39f47c358166d0a36de2ef9875f65c21.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/a2de222d727381253829ab574c258b7f39f47c358166d0a36de2ef9875f65c21.jpg)

![acc1fd5646bc9787a20524136475a2df6922b5255d91ed567ea2b90e7d42b7e9.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/acc1fd5646bc9787a20524136475a2df6922b5255d91ed567ea2b90e7d42b7e9.jpg)

![b90b2867bf99c127a75fca992ba3a61d6096e34c3d610a23304824135721748f.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/b90b2867bf99c127a75fca992ba3a61d6096e34c3d610a23304824135721748f.jpg)

![c4bbaff57948b08d9aaa49dfdd82f4b78caa95b9ae235bc87b79462e547a13fe.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/c4bbaff57948b08d9aaa49dfdd82f4b78caa95b9ae235bc87b79462e547a13fe.jpg)

![cc35a9bc74443f14b5e6dcabe44f6d9acadeb3b1d89e183443e0c25006032bbf.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/cc35a9bc74443f14b5e6dcabe44f6d9acadeb3b1d89e183443e0c25006032bbf.jpg)

![d6bcd14779223d35eddc61f0c67f6639f0d4349849ff120063df97de1387c7b8.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/d6bcd14779223d35eddc61f0c67f6639f0d4349849ff120063df97de1387c7b8.jpg)

![d6d9d4476162e99f33b64f404cf5aa9b4c557f4373ad3df66d259271646dfe16.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/d6d9d4476162e99f33b64f404cf5aa9b4c557f4373ad3df66d259271646dfe16.jpg)

![e61bbb0d6655c77703d1e6407488267520d20d942645bd63e077417f5ab5a3fc.jpg](../icml_results/4805_Beyond%20Single-View%20Indexing_%20Structure-Aware%20Multi-View%20Retrieval%20for%20Knowledge-Based%20VQA/tables/e61bbb0d6655c77703d1e6407488267520d20d942645bd63e077417f5ab5a3fc.jpg)

<a id="paper-4758"></a>

## AdaGC: Enhancing LLM Pretraining Stability via Adaptive Gradient Clipping


### Images

![0ae83fa7d08ffe2ee75cfd3823aa8d55cde4fd47026dcf81c35afc625b21b9e2.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/images/0ae83fa7d08ffe2ee75cfd3823aa8d55cde4fd47026dcf81c35afc625b21b9e2.jpg)

![10738b9301c7eb0195027702748b32fbc09fd5a8bccbb650ede3e25cd819a124.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/images/10738b9301c7eb0195027702748b32fbc09fd5a8bccbb650ede3e25cd819a124.jpg)

![1ad634162f3d34a7ab94008da062f35e4010ac716b444f0605abc6920966a271.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/images/1ad634162f3d34a7ab94008da062f35e4010ac716b444f0605abc6920966a271.jpg)

![212d60e19fad94af98c76db936b021418d93bec6e0aea8ca2fba48ba309a33cc.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/images/212d60e19fad94af98c76db936b021418d93bec6e0aea8ca2fba48ba309a33cc.jpg)

![24623bcf75eeb75e99347e4a8a7ac88051f9fc493eb3378e1a7e172768352bf7.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/images/24623bcf75eeb75e99347e4a8a7ac88051f9fc493eb3378e1a7e172768352bf7.jpg)

![2eef0f35f3d204d11639ca805bc7173ab418fb76c4c37faf0801e4e995e97b6f.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/images/2eef0f35f3d204d11639ca805bc7173ab418fb76c4c37faf0801e4e995e97b6f.jpg)

![398e45534b1fc9799183632156b102891b6f609fab3ab3ea138bd359fe530936.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/images/398e45534b1fc9799183632156b102891b6f609fab3ab3ea138bd359fe530936.jpg)

![3e04f9e46af3b629f75fc09f3ac44949fba1691af88d4592e801d247f421e038.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/images/3e04f9e46af3b629f75fc09f3ac44949fba1691af88d4592e801d247f421e038.jpg)

![53f3653fadbd4840c7983f723c227fe99fd013a6c815314b983836277c629ac7.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/images/53f3653fadbd4840c7983f723c227fe99fd013a6c815314b983836277c629ac7.jpg)

![57d7d1c3a24288c73d6c28b4dedc00e1c0a53900bf2c5c300433ef7f6f0410d5.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/images/57d7d1c3a24288c73d6c28b4dedc00e1c0a53900bf2c5c300433ef7f6f0410d5.jpg)

![9866313f0b23ac87372b796b89a5f4b56a253292f5fec4403f70ce1172484d10.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/images/9866313f0b23ac87372b796b89a5f4b56a253292f5fec4403f70ce1172484d10.jpg)

![9aaf06abb687bf329711d0e282e908a84e2c06e58f089a7dde5fda393bd84a44.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/images/9aaf06abb687bf329711d0e282e908a84e2c06e58f089a7dde5fda393bd84a44.jpg)

![b6f4a395158f324bbeb87179a6bb480e88c01c5f1774773bba24e7b650e15577.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/images/b6f4a395158f324bbeb87179a6bb480e88c01c5f1774773bba24e7b650e15577.jpg)

![cc9d1ebf816bb34e78ea5f3e40a19f8dac391ea1cfa13a0ce943d18ce5797e3f.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/images/cc9d1ebf816bb34e78ea5f3e40a19f8dac391ea1cfa13a0ce943d18ce5797e3f.jpg)

![e329c4fe326f5a84e59610daa4ee9ecce22b9e150322e74aba9caa99a52728ca.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/images/e329c4fe326f5a84e59610daa4ee9ecce22b9e150322e74aba9caa99a52728ca.jpg)

![e6250154f323a218bd6f6d58465183fe19d82d917ae246dacf68a1cd595db22e.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/images/e6250154f323a218bd6f6d58465183fe19d82d917ae246dacf68a1cd595db22e.jpg)

### Tables

![01db5809f5d4b0e4377c31309cc0292f33c55578aafea6d419483a7435fdd05c.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/tables/01db5809f5d4b0e4377c31309cc0292f33c55578aafea6d419483a7435fdd05c.jpg)

![1c2f808781def30e1c34b7bd59f4f622dbff8fb9ff18f0208f9f381e463011cb.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/tables/1c2f808781def30e1c34b7bd59f4f622dbff8fb9ff18f0208f9f381e463011cb.jpg)

![50db9caf799d0f462b762bed52ddb08a8b9e8793b5094065782d3e5e3f298924.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/tables/50db9caf799d0f462b762bed52ddb08a8b9e8793b5094065782d3e5e3f298924.jpg)

![58363bff1c416601e51452b73e56e5c25b7346c285e5d5da3f701fee54dc2e43.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/tables/58363bff1c416601e51452b73e56e5c25b7346c285e5d5da3f701fee54dc2e43.jpg)

![7988ad5a75baba1b63e08f996376ecc402a9dad0094d2978d716203abe5af0bb.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/tables/7988ad5a75baba1b63e08f996376ecc402a9dad0094d2978d716203abe5af0bb.jpg)

![7a6b239321e93b4eabb22143ee86cfddb86e0839a101ed21400f2c7a887f7124.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/tables/7a6b239321e93b4eabb22143ee86cfddb86e0839a101ed21400f2c7a887f7124.jpg)

![89f41667395dea8263c1d1305e89be9a6f1685914b80ddd966c4c1f364bf5ebf.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/tables/89f41667395dea8263c1d1305e89be9a6f1685914b80ddd966c4c1f364bf5ebf.jpg)

![9a20084ad3a66d86b6755053cbcb99398063d069103d0c8b5e975d0161d0c20e.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/tables/9a20084ad3a66d86b6755053cbcb99398063d069103d0c8b5e975d0161d0c20e.jpg)

![a753d7d8068d3d67c2e5eba43166f282fcd83002f5453ed5f2a80869e1e17bd7.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/tables/a753d7d8068d3d67c2e5eba43166f282fcd83002f5453ed5f2a80869e1e17bd7.jpg)

![cd98cf0139aa7e88007ef253f0a1d18bca96678636ed880e10d9d1da8316ee51.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/tables/cd98cf0139aa7e88007ef253f0a1d18bca96678636ed880e10d9d1da8316ee51.jpg)

![d94c492913a992b578254ef565216dc6e0e4d0e3f7e3506635729d1b0992918c.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/tables/d94c492913a992b578254ef565216dc6e0e4d0e3f7e3506635729d1b0992918c.jpg)

![de102c1eaaaef5d0910639834292b202fc0b556240b08ebf0670a037e0bfc795.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/tables/de102c1eaaaef5d0910639834292b202fc0b556240b08ebf0670a037e0bfc795.jpg)

![ed25c84bd43a65d0c3baf4229728c630738fc5517e3dd0cfb05cefc5d4cabbc8.jpg](../icml_results/4806_AdaGC_%20Enhancing%20LLM%20Pretraining%20Stability%20via%20Adaptive%20Gradient%20Clipping/tables/ed25c84bd43a65d0c3baf4229728c630738fc5517e3dd0cfb05cefc5d4cabbc8.jpg)

<a id="paper-4759"></a>

## NoiseSDF2NoiseSDF: Learning Clean Neural Fields from Noisy Supervision


### Images

![1301ff021d4b3f13462e155c03fa8dc664275a6df896b60eac332bcf8890d5e7.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/images/1301ff021d4b3f13462e155c03fa8dc664275a6df896b60eac332bcf8890d5e7.jpg)

![2635b181df9ab66a0de3c650fe137f4b67402a6efd108d4da909427bd58f8457.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/images/2635b181df9ab66a0de3c650fe137f4b67402a6efd108d4da909427bd58f8457.jpg)

![35a5ea46f67057c33ff0fc7ba9b132fd6c0661ce4c3ca2c1aacd926c3b226a06.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/images/35a5ea46f67057c33ff0fc7ba9b132fd6c0661ce4c3ca2c1aacd926c3b226a06.jpg)

![446927324cbe5f35b4b74ab719126e6256e363eeee4c73bad2550cc1491665df.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/images/446927324cbe5f35b4b74ab719126e6256e363eeee4c73bad2550cc1491665df.jpg)

![86a30900f8841182b5a3fc9c7f6799b8994c6c2540c1523aefe455b7b059aaf6.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/images/86a30900f8841182b5a3fc9c7f6799b8994c6c2540c1523aefe455b7b059aaf6.jpg)

![bcbddb9e8d0d49c2d045efe27cb4f939ecf78299087f5c08d2bc4f66306f61f6.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/images/bcbddb9e8d0d49c2d045efe27cb4f939ecf78299087f5c08d2bc4f66306f61f6.jpg)

![d68e1f497ffe1906ff605d100d438a31993369254775e044f9203f5f9cf24e00.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/images/d68e1f497ffe1906ff605d100d438a31993369254775e044f9203f5f9cf24e00.jpg)

### Tables

![0a7d0371ecb09f17f08d10e9af3a02f0e8c77409b455f939f06c76a66056b6ec.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/tables/0a7d0371ecb09f17f08d10e9af3a02f0e8c77409b455f939f06c76a66056b6ec.jpg)

![10cbde9eff758a594d92a6fe6b61e59d56c5b4d012e4315db4ffe41dd0a2d64b.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/tables/10cbde9eff758a594d92a6fe6b61e59d56c5b4d012e4315db4ffe41dd0a2d64b.jpg)

![452e0e123f2fed1e572459647460c69c440d084304cfaf4f141e340d7c2528d6.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/tables/452e0e123f2fed1e572459647460c69c440d084304cfaf4f141e340d7c2528d6.jpg)

![5154a243af3744f4d557ec7ac552344fe917a66c654e159189408df7881adb78.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/tables/5154a243af3744f4d557ec7ac552344fe917a66c654e159189408df7881adb78.jpg)

![65721ba202c89ff39608c330c954d6d18a3b503b2c98d125f0fccc0e75ac5162.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/tables/65721ba202c89ff39608c330c954d6d18a3b503b2c98d125f0fccc0e75ac5162.jpg)

![7b4704a369f938a31088c1a7adb1038cfce562d3105aa67de020d30371a95cf2.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/tables/7b4704a369f938a31088c1a7adb1038cfce562d3105aa67de020d30371a95cf2.jpg)

![7e7af540e6d351bfba35b9b04a853ecbd684680188f4e8fbb685128a2003a15d.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/tables/7e7af540e6d351bfba35b9b04a853ecbd684680188f4e8fbb685128a2003a15d.jpg)

![8d33771e1c2976012082f35ccca63d8960434457c3f18fcdb119e2b5eb7f7669.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/tables/8d33771e1c2976012082f35ccca63d8960434457c3f18fcdb119e2b5eb7f7669.jpg)

![9b26bbc2e9689c74facaf982bcda88ad1a8f214880f4e20c69f0ff9cfaf3c35b.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/tables/9b26bbc2e9689c74facaf982bcda88ad1a8f214880f4e20c69f0ff9cfaf3c35b.jpg)

![a96c275077c9f30e03d90ab73fe45cb0094275e7baed359a35c1d90890f36c42.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/tables/a96c275077c9f30e03d90ab73fe45cb0094275e7baed359a35c1d90890f36c42.jpg)

![aaa2f1aa27fe63dbd65ae703ad563b4d916a9e232173b58aa53db4d5bb32b35d.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/tables/aaa2f1aa27fe63dbd65ae703ad563b4d916a9e232173b58aa53db4d5bb32b35d.jpg)

![ce91e40be05a5e9fd61afbf67935c45ebb94020166d1574ae817be6d8676ce1c.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/tables/ce91e40be05a5e9fd61afbf67935c45ebb94020166d1574ae817be6d8676ce1c.jpg)

![d9dd1fc0124a8a68e6fe0d38f4ba2009c1f3397e81e717cc681af2b35070c527.jpg](../icml_results/4807_NoiseSDF2NoiseSDF_%20Learning%20Clean%20Neural%20Fields%20from%20Noisy%20Supervision/tables/d9dd1fc0124a8a68e6fe0d38f4ba2009c1f3397e81e717cc681af2b35070c527.jpg)

<a id="paper-4760"></a>

## Test-Time Debiasing with Probabilistic Prompts via Wasserstein Distance in Vision-Language Models


### Images

![2fab4d4662ac39b738d921db2cc5fb4bdfd86866fdd892631c1fca84c78579c3.jpg](../icml_results/4808_Test-Time%20Debiasing%20with%20Probabilistic%20Prompts%20via%20Wasserstein%20Distance%20in%20Vision-Language%20Models/images/2fab4d4662ac39b738d921db2cc5fb4bdfd86866fdd892631c1fca84c78579c3.jpg)

![32656a42f10fd09bf2fe072bd5cfd25a2eba6f374695890a18683f728a583723.jpg](../icml_results/4808_Test-Time%20Debiasing%20with%20Probabilistic%20Prompts%20via%20Wasserstein%20Distance%20in%20Vision-Language%20Models/images/32656a42f10fd09bf2fe072bd5cfd25a2eba6f374695890a18683f728a583723.jpg)

![7950360b92f290abe5fa5a8e0d4eb8e7121d289a0c11e68cd3d5a6ccc32265b2.jpg](../icml_results/4808_Test-Time%20Debiasing%20with%20Probabilistic%20Prompts%20via%20Wasserstein%20Distance%20in%20Vision-Language%20Models/images/7950360b92f290abe5fa5a8e0d4eb8e7121d289a0c11e68cd3d5a6ccc32265b2.jpg)

![bd0a017280c94ccdd220266fbef0eca1ba87f2e3de2089cfc62c0192f9a60346.jpg](../icml_results/4808_Test-Time%20Debiasing%20with%20Probabilistic%20Prompts%20via%20Wasserstein%20Distance%20in%20Vision-Language%20Models/images/bd0a017280c94ccdd220266fbef0eca1ba87f2e3de2089cfc62c0192f9a60346.jpg)

![d684e502c58604f5dd652e74aa49ac21b8c43ab51398efc6817d63be4b32c8dd.jpg](../icml_results/4808_Test-Time%20Debiasing%20with%20Probabilistic%20Prompts%20via%20Wasserstein%20Distance%20in%20Vision-Language%20Models/images/d684e502c58604f5dd652e74aa49ac21b8c43ab51398efc6817d63be4b32c8dd.jpg)

![d9c5a6935edf2a82f54a0e857070eb70555d0ac9eaffb8403c115fdce3203f71.jpg](../icml_results/4808_Test-Time%20Debiasing%20with%20Probabilistic%20Prompts%20via%20Wasserstein%20Distance%20in%20Vision-Language%20Models/images/d9c5a6935edf2a82f54a0e857070eb70555d0ac9eaffb8403c115fdce3203f71.jpg)

![ffeb7a91689578f184210c8103b5b858f18dd951601292f803a81e7566120222.jpg](../icml_results/4808_Test-Time%20Debiasing%20with%20Probabilistic%20Prompts%20via%20Wasserstein%20Distance%20in%20Vision-Language%20Models/images/ffeb7a91689578f184210c8103b5b858f18dd951601292f803a81e7566120222.jpg)

### Tables

![0df7cf45911644cc47983c3c8c06981aedab77c9d3a65a4563bc7f9cdbff31b5.jpg](../icml_results/4808_Test-Time%20Debiasing%20with%20Probabilistic%20Prompts%20via%20Wasserstein%20Distance%20in%20Vision-Language%20Models/tables/0df7cf45911644cc47983c3c8c06981aedab77c9d3a65a4563bc7f9cdbff31b5.jpg)

![2957c6df6f834ec1201312f839d0e1dcd9d16c275640972745076d051ecce4b7.jpg](../icml_results/4808_Test-Time%20Debiasing%20with%20Probabilistic%20Prompts%20via%20Wasserstein%20Distance%20in%20Vision-Language%20Models/tables/2957c6df6f834ec1201312f839d0e1dcd9d16c275640972745076d051ecce4b7.jpg)

![3b1dcaef009c286924ae3acff6a0764d8aac375a594e5599e8a9ff26921c2f86.jpg](../icml_results/4808_Test-Time%20Debiasing%20with%20Probabilistic%20Prompts%20via%20Wasserstein%20Distance%20in%20Vision-Language%20Models/tables/3b1dcaef009c286924ae3acff6a0764d8aac375a594e5599e8a9ff26921c2f86.jpg)

![63f1fe0179e9cf55384ac0262c9e389f0b1219d8c48eb677eda09afba0ab1122.jpg](../icml_results/4808_Test-Time%20Debiasing%20with%20Probabilistic%20Prompts%20via%20Wasserstein%20Distance%20in%20Vision-Language%20Models/tables/63f1fe0179e9cf55384ac0262c9e389f0b1219d8c48eb677eda09afba0ab1122.jpg)

![6c645fb67ef4e2b3ae0c2c783ff20d1ac49d94ac50431e24a8fd812c239c75c5.jpg](../icml_results/4808_Test-Time%20Debiasing%20with%20Probabilistic%20Prompts%20via%20Wasserstein%20Distance%20in%20Vision-Language%20Models/tables/6c645fb67ef4e2b3ae0c2c783ff20d1ac49d94ac50431e24a8fd812c239c75c5.jpg)

![8e6581343cd5c669cb376f45100ba92a26266b350b39ff1b85e3a9359f7f3204.jpg](../icml_results/4808_Test-Time%20Debiasing%20with%20Probabilistic%20Prompts%20via%20Wasserstein%20Distance%20in%20Vision-Language%20Models/tables/8e6581343cd5c669cb376f45100ba92a26266b350b39ff1b85e3a9359f7f3204.jpg)

![b08a16726f0dc8913ca0763b5cad09045b67196e022c5e62ec696f765d8bcae5.jpg](../icml_results/4808_Test-Time%20Debiasing%20with%20Probabilistic%20Prompts%20via%20Wasserstein%20Distance%20in%20Vision-Language%20Models/tables/b08a16726f0dc8913ca0763b5cad09045b67196e022c5e62ec696f765d8bcae5.jpg)

![f17601306ad040271232e5783b16f4aee9e8a34ea58f13424754d924ee756d21.jpg](../icml_results/4808_Test-Time%20Debiasing%20with%20Probabilistic%20Prompts%20via%20Wasserstein%20Distance%20in%20Vision-Language%20Models/tables/f17601306ad040271232e5783b16f4aee9e8a34ea58f13424754d924ee756d21.jpg)

<a id="paper-4761"></a>

## Anchored Policy Optimization: Mitigating Exploration Collapse via Support-Constrained Rectification


### Images

![195ccb6a5a8d5de727a9f05bcda7bdaa76c163e61f7fcea827ec96b7c4e6e5c1.jpg](../icml_results/4809_Anchored%20Policy%20Optimization_%20Mitigating%20Exploration%20Collapse%20via%20Support-Constrained%20Rectification/images/195ccb6a5a8d5de727a9f05bcda7bdaa76c163e61f7fcea827ec96b7c4e6e5c1.jpg)

![2e81da0f52335c73def418e06aa9b873fbde35058a095bda360abbc9a553432c.jpg](../icml_results/4809_Anchored%20Policy%20Optimization_%20Mitigating%20Exploration%20Collapse%20via%20Support-Constrained%20Rectification/images/2e81da0f52335c73def418e06aa9b873fbde35058a095bda360abbc9a553432c.jpg)

![33665fa6e02cf8bc48fc4c764bc28957692b16985fa1e3ce36dda5edf0f665dd.jpg](../icml_results/4809_Anchored%20Policy%20Optimization_%20Mitigating%20Exploration%20Collapse%20via%20Support-Constrained%20Rectification/images/33665fa6e02cf8bc48fc4c764bc28957692b16985fa1e3ce36dda5edf0f665dd.jpg)

![5302e814df4eb9b9a8bf6742e3232f8818d202ba8cae0bda2ff9f89ce1b6f0dd.jpg](../icml_results/4809_Anchored%20Policy%20Optimization_%20Mitigating%20Exploration%20Collapse%20via%20Support-Constrained%20Rectification/images/5302e814df4eb9b9a8bf6742e3232f8818d202ba8cae0bda2ff9f89ce1b6f0dd.jpg)

![9cd8e869c6b3f441b4332309a906932b9dcf7f018725343f416acbe1dae3abfd.jpg](../icml_results/4809_Anchored%20Policy%20Optimization_%20Mitigating%20Exploration%20Collapse%20via%20Support-Constrained%20Rectification/images/9cd8e869c6b3f441b4332309a906932b9dcf7f018725343f416acbe1dae3abfd.jpg)

![e610b512237b5bb94b9145cd071beb7ab1a3cf0dc27723f4eab000101dd18aa3.jpg](../icml_results/4809_Anchored%20Policy%20Optimization_%20Mitigating%20Exploration%20Collapse%20via%20Support-Constrained%20Rectification/images/e610b512237b5bb94b9145cd071beb7ab1a3cf0dc27723f4eab000101dd18aa3.jpg)

### Tables

![33d2c9a36ff111d7b1255436af254a18ffb0bd585b0e3259159061873c4a1e45.jpg](../icml_results/4809_Anchored%20Policy%20Optimization_%20Mitigating%20Exploration%20Collapse%20via%20Support-Constrained%20Rectification/tables/33d2c9a36ff111d7b1255436af254a18ffb0bd585b0e3259159061873c4a1e45.jpg)

![6d8c0baedd0430a69855df2a69cd8d16ca66cf6b62362414139641235c11977b.jpg](../icml_results/4809_Anchored%20Policy%20Optimization_%20Mitigating%20Exploration%20Collapse%20via%20Support-Constrained%20Rectification/tables/6d8c0baedd0430a69855df2a69cd8d16ca66cf6b62362414139641235c11977b.jpg)

![b2d47e8c660ca43254d68421cbe8cd298fe69cd87b5f5906fe92175409c40c8f.jpg](../icml_results/4809_Anchored%20Policy%20Optimization_%20Mitigating%20Exploration%20Collapse%20via%20Support-Constrained%20Rectification/tables/b2d47e8c660ca43254d68421cbe8cd298fe69cd87b5f5906fe92175409c40c8f.jpg)

![b764f71a008b566b3171fc112a9a975ee81dd726c48d8cfe54500e534e540162.jpg](../icml_results/4809_Anchored%20Policy%20Optimization_%20Mitigating%20Exploration%20Collapse%20via%20Support-Constrained%20Rectification/tables/b764f71a008b566b3171fc112a9a975ee81dd726c48d8cfe54500e534e540162.jpg)

![e06c248aba028322bd95651beea666fdfe717b5c70574f5d953382a279e1ca26.jpg](../icml_results/4809_Anchored%20Policy%20Optimization_%20Mitigating%20Exploration%20Collapse%20via%20Support-Constrained%20Rectification/tables/e06c248aba028322bd95651beea666fdfe717b5c70574f5d953382a279e1ca26.jpg)

![e07eb081b8f8286ea961bea38451f634791783142982623f9668c9b5f1447b20.jpg](../icml_results/4809_Anchored%20Policy%20Optimization_%20Mitigating%20Exploration%20Collapse%20via%20Support-Constrained%20Rectification/tables/e07eb081b8f8286ea961bea38451f634791783142982623f9668c9b5f1447b20.jpg)

![fb3a989af722f95b642c441f0bf6b7097ae65b2b08c8b770027503f166ab8c16.jpg](../icml_results/4809_Anchored%20Policy%20Optimization_%20Mitigating%20Exploration%20Collapse%20via%20Support-Constrained%20Rectification/tables/fb3a989af722f95b642c441f0bf6b7097ae65b2b08c8b770027503f166ab8c16.jpg)

<a id="paper-4762"></a>

## ECo-MoE: Embodiment-Conditioned Mixture of Experts Increases the Evolvability of Robots


### Images

![271214b50836725352270f6ed6aaac2fe5428090084a2b1fd32a4509f7cc1633.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/images/271214b50836725352270f6ed6aaac2fe5428090084a2b1fd32a4509f7cc1633.jpg)

![3158439b8e7a8bdf1550f33b13123daa39be127d45510a604fa43effcf12315f.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/images/3158439b8e7a8bdf1550f33b13123daa39be127d45510a604fa43effcf12315f.jpg)

![3779a8a06a56e28f385f5b9e8ec5fa8ac6a0d8c51c8a26809c979d6bfcd153c4.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/images/3779a8a06a56e28f385f5b9e8ec5fa8ac6a0d8c51c8a26809c979d6bfcd153c4.jpg)

![48f1256d64a60763fac633c6f5f2d4258557e55408639867212a941974cb6860.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/images/48f1256d64a60763fac633c6f5f2d4258557e55408639867212a941974cb6860.jpg)

![4e7727b9eb3e117430b058dd4d853c58b000ca6f06cd85b07f9500407acd1952.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/images/4e7727b9eb3e117430b058dd4d853c58b000ca6f06cd85b07f9500407acd1952.jpg)

![5dbef6d11be761b312c909d0022b0dab792d64b84766f0ccf2648a2d014863b1.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/images/5dbef6d11be761b312c909d0022b0dab792d64b84766f0ccf2648a2d014863b1.jpg)

![64091988f59acc8b738a9482cb7354a7e749bbfd01475d3f6eef0ac1c08e2c87.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/images/64091988f59acc8b738a9482cb7354a7e749bbfd01475d3f6eef0ac1c08e2c87.jpg)

![8d38454a54dc63809dc829bf8d4ac1f8153101af2ca9fc5cb0199b2eda9618d2.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/images/8d38454a54dc63809dc829bf8d4ac1f8153101af2ca9fc5cb0199b2eda9618d2.jpg)

![9629706f39d9a270fea48c90da4832d54391437901436442c105fdcbb3993582.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/images/9629706f39d9a270fea48c90da4832d54391437901436442c105fdcbb3993582.jpg)

![999480dae16eda50d82f4fefa54e20409b0c76888b2853e857824729ad67bed6.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/images/999480dae16eda50d82f4fefa54e20409b0c76888b2853e857824729ad67bed6.jpg)

![a04327d200ef592bfc650db73106d38fd017cfa7e30a8bb2cca74034f5e8efed.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/images/a04327d200ef592bfc650db73106d38fd017cfa7e30a8bb2cca74034f5e8efed.jpg)

![a9945265442c90f4b2c1c2fa72adb7161457ec35b7bf75636328af0446175575.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/images/a9945265442c90f4b2c1c2fa72adb7161457ec35b7bf75636328af0446175575.jpg)

![b56e6303f40fbd17b2e3d6b5dd601413f3f6a629b02d4f3b936e0f3b079d1411.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/images/b56e6303f40fbd17b2e3d6b5dd601413f3f6a629b02d4f3b936e0f3b079d1411.jpg)

![f6a855de84cc8d7b97a327b3169f04533759aab08d6da9c7366d747a3ba3122c.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/images/f6a855de84cc8d7b97a327b3169f04533759aab08d6da9c7366d747a3ba3122c.jpg)

### Tables

![0a11c5b45e36985fc67f5c65975fd49789c2849f5386f6013d7284c2d2df0aec.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/tables/0a11c5b45e36985fc67f5c65975fd49789c2849f5386f6013d7284c2d2df0aec.jpg)

![1c8768bf172724d5d5e6d63e65509ca115e496a5b77d09984855bfd1a8c60320.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/tables/1c8768bf172724d5d5e6d63e65509ca115e496a5b77d09984855bfd1a8c60320.jpg)

![77e1bef4c0887202f3b909a17056a2b69b545a79befd9e49e5dce40320a6599b.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/tables/77e1bef4c0887202f3b909a17056a2b69b545a79befd9e49e5dce40320a6599b.jpg)

![8826a242940b89d9553e81e24e8cc3f1657166e0e88d9cf69a28889ff9893565.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/tables/8826a242940b89d9553e81e24e8cc3f1657166e0e88d9cf69a28889ff9893565.jpg)

![ad694b346195c084742af582b7824491eaa5518f2407213e05174397cd901a1a.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/tables/ad694b346195c084742af582b7824491eaa5518f2407213e05174397cd901a1a.jpg)

![c83f39244e78a24134c3e5dadaa95a7bda22a3452af9939bb9a4e5109eabd99b.jpg](../icml_results/4810_ECo-MoE_%20Embodiment-Conditioned%20Mixture%20of%20Experts%20Increases%20the%20Evolvability%20of%20Robots/tables/c83f39244e78a24134c3e5dadaa95a7bda22a3452af9939bb9a4e5109eabd99b.jpg)

<a id="paper-4763"></a>

## One Coin Has Two Sides: Single Positive Multi Label Learning from Salient Annotations


### Images

![04c0327e9c4d679cb049da0a45aec50005aaffc6c79e02412b0eec185055fa4d.jpg](../icml_results/4811_One%20Coin%20Has%20Two%20Sides_%20Single%20Positive%20Multi%20Label%20Learning%20from%20Salient%20Annotations/images/04c0327e9c4d679cb049da0a45aec50005aaffc6c79e02412b0eec185055fa4d.jpg)

![05f6b6d3f31f0996a1e3883d0ae743cdb102f97806750a212164c2e3ae847615.jpg](../icml_results/4811_One%20Coin%20Has%20Two%20Sides_%20Single%20Positive%20Multi%20Label%20Learning%20from%20Salient%20Annotations/images/05f6b6d3f31f0996a1e3883d0ae743cdb102f97806750a212164c2e3ae847615.jpg)

![4b1688637957f85f46720c5532c68be0bade74802705431452deac15a1f409c0.jpg](../icml_results/4811_One%20Coin%20Has%20Two%20Sides_%20Single%20Positive%20Multi%20Label%20Learning%20from%20Salient%20Annotations/images/4b1688637957f85f46720c5532c68be0bade74802705431452deac15a1f409c0.jpg)

![557d0378ef7abff96d1edb891b6eacbcbaf7f02111ba01b99dabba114faaef28.jpg](../icml_results/4811_One%20Coin%20Has%20Two%20Sides_%20Single%20Positive%20Multi%20Label%20Learning%20from%20Salient%20Annotations/images/557d0378ef7abff96d1edb891b6eacbcbaf7f02111ba01b99dabba114faaef28.jpg)

![60b585cef9bd4f5bc7ff70caeca2ebcbec3d8482179f968ea64e091ab396fef8.jpg](../icml_results/4811_One%20Coin%20Has%20Two%20Sides_%20Single%20Positive%20Multi%20Label%20Learning%20from%20Salient%20Annotations/images/60b585cef9bd4f5bc7ff70caeca2ebcbec3d8482179f968ea64e091ab396fef8.jpg)

![7945a36288aa3f8c765030722efc481aa30e5e30ef1274125b6c953cce5278c0.jpg](../icml_results/4811_One%20Coin%20Has%20Two%20Sides_%20Single%20Positive%20Multi%20Label%20Learning%20from%20Salient%20Annotations/images/7945a36288aa3f8c765030722efc481aa30e5e30ef1274125b6c953cce5278c0.jpg)

![95ff05b4bfce37c7ed4acbdc78f0735f811e8afec43b6672468edcb5371712c8.jpg](../icml_results/4811_One%20Coin%20Has%20Two%20Sides_%20Single%20Positive%20Multi%20Label%20Learning%20from%20Salient%20Annotations/images/95ff05b4bfce37c7ed4acbdc78f0735f811e8afec43b6672468edcb5371712c8.jpg)

![b32586ba07475e773e3822ce2d084478abc6b53d536a9fd6fc936317174d76d0.jpg](../icml_results/4811_One%20Coin%20Has%20Two%20Sides_%20Single%20Positive%20Multi%20Label%20Learning%20from%20Salient%20Annotations/images/b32586ba07475e773e3822ce2d084478abc6b53d536a9fd6fc936317174d76d0.jpg)

![bd1ac669d756ae1cdea21044963c9d218d8364e1c2784a82435431887753fe3f.jpg](../icml_results/4811_One%20Coin%20Has%20Two%20Sides_%20Single%20Positive%20Multi%20Label%20Learning%20from%20Salient%20Annotations/images/bd1ac669d756ae1cdea21044963c9d218d8364e1c2784a82435431887753fe3f.jpg)

![fdff97d1f63f968bfa2af0e8eb49d6a2f79d8f29dfc70923a38ccd1aaf6911a6.jpg](../icml_results/4811_One%20Coin%20Has%20Two%20Sides_%20Single%20Positive%20Multi%20Label%20Learning%20from%20Salient%20Annotations/images/fdff97d1f63f968bfa2af0e8eb49d6a2f79d8f29dfc70923a38ccd1aaf6911a6.jpg)

### Tables

![1bae9e61548477eb50b1483ee933b19382eafb6ae0c30416157ada9034f825b5.jpg](../icml_results/4811_One%20Coin%20Has%20Two%20Sides_%20Single%20Positive%20Multi%20Label%20Learning%20from%20Salient%20Annotations/tables/1bae9e61548477eb50b1483ee933b19382eafb6ae0c30416157ada9034f825b5.jpg)

![7696df211adf0f7e0e452cec1ae3e8f3f04e08c7ff023a1dc32a8cb2948afea1.jpg](../icml_results/4811_One%20Coin%20Has%20Two%20Sides_%20Single%20Positive%20Multi%20Label%20Learning%20from%20Salient%20Annotations/tables/7696df211adf0f7e0e452cec1ae3e8f3f04e08c7ff023a1dc32a8cb2948afea1.jpg)

![791cc6941457f53315ca44a4eca305cc59ca05e362e44c94044163f5caf0ca11.jpg](../icml_results/4811_One%20Coin%20Has%20Two%20Sides_%20Single%20Positive%20Multi%20Label%20Learning%20from%20Salient%20Annotations/tables/791cc6941457f53315ca44a4eca305cc59ca05e362e44c94044163f5caf0ca11.jpg)

![8e1fc1bfcdc864828a3ea59b4bb01f26ea0ebf639cfa3dd7f9d3ab1d09127f87.jpg](../icml_results/4811_One%20Coin%20Has%20Two%20Sides_%20Single%20Positive%20Multi%20Label%20Learning%20from%20Salient%20Annotations/tables/8e1fc1bfcdc864828a3ea59b4bb01f26ea0ebf639cfa3dd7f9d3ab1d09127f87.jpg)

![92dc26d0711b22231ee067b0dd0430e961f341676454de623b081d10189e208a.jpg](../icml_results/4811_One%20Coin%20Has%20Two%20Sides_%20Single%20Positive%20Multi%20Label%20Learning%20from%20Salient%20Annotations/tables/92dc26d0711b22231ee067b0dd0430e961f341676454de623b081d10189e208a.jpg)

![a6b5a01bd3af92769dbb845d812f21de668b1341fcfbcd7ef7810ee0a4fe82e0.jpg](../icml_results/4811_One%20Coin%20Has%20Two%20Sides_%20Single%20Positive%20Multi%20Label%20Learning%20from%20Salient%20Annotations/tables/a6b5a01bd3af92769dbb845d812f21de668b1341fcfbcd7ef7810ee0a4fe82e0.jpg)

![abb920497e952f62711300fc1935cc536085182590e3ec098057025d77ab1a83.jpg](../icml_results/4811_One%20Coin%20Has%20Two%20Sides_%20Single%20Positive%20Multi%20Label%20Learning%20from%20Salient%20Annotations/tables/abb920497e952f62711300fc1935cc536085182590e3ec098057025d77ab1a83.jpg)

<a id="paper-4764"></a>

## From Coarse to Fine: Deep Prototype Refinement Network for Few-Shot Point Cloud Semantic Segmentation


### Images

![6f4f7080fe7dc45b9c41a9b9dbc589395bd0916302480eecca1c86312624652f.jpg](../icml_results/4812_From%20Coarse%20to%20Fine_%20Deep%20Prototype%20Refinement%20Network%20for%20Few-Shot%20Point%20Cloud%20Semantic%20Segmentatio/images/6f4f7080fe7dc45b9c41a9b9dbc589395bd0916302480eecca1c86312624652f.jpg)

![a48a7cdee8da56dbe28af546e1ec9c30438a28d3ac7aae407cc53cad41b57619.jpg](../icml_results/4812_From%20Coarse%20to%20Fine_%20Deep%20Prototype%20Refinement%20Network%20for%20Few-Shot%20Point%20Cloud%20Semantic%20Segmentatio/images/a48a7cdee8da56dbe28af546e1ec9c30438a28d3ac7aae407cc53cad41b57619.jpg)

![ebf0962679b8a4f36533b3a6edef20668dc4eb45590e8bb76a369d7960123910.jpg](../icml_results/4812_From%20Coarse%20to%20Fine_%20Deep%20Prototype%20Refinement%20Network%20for%20Few-Shot%20Point%20Cloud%20Semantic%20Segmentatio/images/ebf0962679b8a4f36533b3a6edef20668dc4eb45590e8bb76a369d7960123910.jpg)

![f05a73c10959fb57b57448787b3ad752974efcc7762ea98944ebe55877d95254.jpg](../icml_results/4812_From%20Coarse%20to%20Fine_%20Deep%20Prototype%20Refinement%20Network%20for%20Few-Shot%20Point%20Cloud%20Semantic%20Segmentatio/images/f05a73c10959fb57b57448787b3ad752974efcc7762ea98944ebe55877d95254.jpg)

![f8a563a26e6d95b29b679c42b199a208c6741dbf929cca03fc06357a7170c82a.jpg](../icml_results/4812_From%20Coarse%20to%20Fine_%20Deep%20Prototype%20Refinement%20Network%20for%20Few-Shot%20Point%20Cloud%20Semantic%20Segmentatio/images/f8a563a26e6d95b29b679c42b199a208c6741dbf929cca03fc06357a7170c82a.jpg)

### Tables

![27e77e2b81060d93ff54b1b04bda0b3fd60738025e639c68950d9e9c729040ff.jpg](../icml_results/4812_From%20Coarse%20to%20Fine_%20Deep%20Prototype%20Refinement%20Network%20for%20Few-Shot%20Point%20Cloud%20Semantic%20Segmentatio/tables/27e77e2b81060d93ff54b1b04bda0b3fd60738025e639c68950d9e9c729040ff.jpg)

![5ea852b8f251650ece71a664b75eec1edb8c21c1ad60e47858972a97cc7270f5.jpg](../icml_results/4812_From%20Coarse%20to%20Fine_%20Deep%20Prototype%20Refinement%20Network%20for%20Few-Shot%20Point%20Cloud%20Semantic%20Segmentatio/tables/5ea852b8f251650ece71a664b75eec1edb8c21c1ad60e47858972a97cc7270f5.jpg)

![92fb30ee051a2dda17f44e14a7c86b3caf3afb5336676ba1dfc696109fbe8696.jpg](../icml_results/4812_From%20Coarse%20to%20Fine_%20Deep%20Prototype%20Refinement%20Network%20for%20Few-Shot%20Point%20Cloud%20Semantic%20Segmentatio/tables/92fb30ee051a2dda17f44e14a7c86b3caf3afb5336676ba1dfc696109fbe8696.jpg)

![9552970453aa1c4ee5ad42485c0809d56b41044c8c06aefcf832d9ec3119fa36.jpg](../icml_results/4812_From%20Coarse%20to%20Fine_%20Deep%20Prototype%20Refinement%20Network%20for%20Few-Shot%20Point%20Cloud%20Semantic%20Segmentatio/tables/9552970453aa1c4ee5ad42485c0809d56b41044c8c06aefcf832d9ec3119fa36.jpg)

![ee90fe15337729133f0b99f4607aab6f9b2f3aa067efe31ee2f427789e2ae62e.jpg](../icml_results/4812_From%20Coarse%20to%20Fine_%20Deep%20Prototype%20Refinement%20Network%20for%20Few-Shot%20Point%20Cloud%20Semantic%20Segmentatio/tables/ee90fe15337729133f0b99f4607aab6f9b2f3aa067efe31ee2f427789e2ae62e.jpg)

![f09345bf3f8a265a5d5a1a7fed948f27e7230b4e9db83a9b0784bd6205846a99.jpg](../icml_results/4812_From%20Coarse%20to%20Fine_%20Deep%20Prototype%20Refinement%20Network%20for%20Few-Shot%20Point%20Cloud%20Semantic%20Segmentatio/tables/f09345bf3f8a265a5d5a1a7fed948f27e7230b4e9db83a9b0784bd6205846a99.jpg)

![fef1edc43cafc42ee5f8bb633d57c2d0af518f3598df89db9f6a6f2f25de8574.jpg](../icml_results/4812_From%20Coarse%20to%20Fine_%20Deep%20Prototype%20Refinement%20Network%20for%20Few-Shot%20Point%20Cloud%20Semantic%20Segmentatio/tables/fef1edc43cafc42ee5f8bb633d57c2d0af518f3598df89db9f6a6f2f25de8574.jpg)

<a id="paper-4765"></a>

## DeepHA: Scaling Action Chains Elicits Deep Hierarchical Agents


### Images

![24965796eeb8e3d233161b7fe6ba34bdc409f36908cad2a06fcc7cabe9c7f363.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/images/24965796eeb8e3d233161b7fe6ba34bdc409f36908cad2a06fcc7cabe9c7f363.jpg)

![3f1246daacdc88d5797d05a8bf681582475cb0a573f26ed7766db4b6305dff10.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/images/3f1246daacdc88d5797d05a8bf681582475cb0a573f26ed7766db4b6305dff10.jpg)

![4e80194a7d7e489b43fd56c5e71d1f1ca195fc2e1dde1fd3a56a49d7424e98b0.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/images/4e80194a7d7e489b43fd56c5e71d1f1ca195fc2e1dde1fd3a56a49d7424e98b0.jpg)

![687b1f524fefb1127b92d978c859c9528f4ab6eb3683b28bbe114cd79dd44662.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/images/687b1f524fefb1127b92d978c859c9528f4ab6eb3683b28bbe114cd79dd44662.jpg)

![6dbd81ff26d017d6a98dfddfc53a46bb873887e618695cdd801bd3bb3dc91e36.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/images/6dbd81ff26d017d6a98dfddfc53a46bb873887e618695cdd801bd3bb3dc91e36.jpg)

![7802beea65b3f8fe4dc6baa35a73f4e903c8baaadb29c952b3909dc8d8156d0c.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/images/7802beea65b3f8fe4dc6baa35a73f4e903c8baaadb29c952b3909dc8d8156d0c.jpg)

![87365df952051f2c501c0494d0fd973d3e3c489feb1731c2ef74774765e9cf5b.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/images/87365df952051f2c501c0494d0fd973d3e3c489feb1731c2ef74774765e9cf5b.jpg)

![a469f334355c5a2e74b6d70538df813468ab0c3c7113513a668a28f4a3329a4b.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/images/a469f334355c5a2e74b6d70538df813468ab0c3c7113513a668a28f4a3329a4b.jpg)

![adf9f9a46ed3bd85790d256d1701fe6339916c476a98c1aca5fe3aa1a2dfe26e.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/images/adf9f9a46ed3bd85790d256d1701fe6339916c476a98c1aca5fe3aa1a2dfe26e.jpg)

![af0819594177c94dde68dcd8ba3e79c46810cfb6d3857e3ec3c859c179b28690.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/images/af0819594177c94dde68dcd8ba3e79c46810cfb6d3857e3ec3c859c179b28690.jpg)

![bb8cde3a737b075dc41bb64de0a0008a97eab1eee5901aa3840b69edfede4c4a.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/images/bb8cde3a737b075dc41bb64de0a0008a97eab1eee5901aa3840b69edfede4c4a.jpg)

![c29587ca482c3371c4332403542fcdc4a48aac316555ed8e7c7e540c6584b19b.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/images/c29587ca482c3371c4332403542fcdc4a48aac316555ed8e7c7e540c6584b19b.jpg)

![c489df89c09ef954fe2358961bf1436cb87521117c27345223722578bc060d56.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/images/c489df89c09ef954fe2358961bf1436cb87521117c27345223722578bc060d56.jpg)

![cbfa86740d71f2a904e3c8d430cd054be5b225994f1fe3145904402f02ced641.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/images/cbfa86740d71f2a904e3c8d430cd054be5b225994f1fe3145904402f02ced641.jpg)

![d06ece78ff3399320dfd88b9b7df87e80fa7ba8d5f445bffd297838863af7368.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/images/d06ece78ff3399320dfd88b9b7df87e80fa7ba8d5f445bffd297838863af7368.jpg)

![deb3b9af31fc1184c18ab94accf4c61cc79de32aa3d274be91e15fd14ddc5bbc.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/images/deb3b9af31fc1184c18ab94accf4c61cc79de32aa3d274be91e15fd14ddc5bbc.jpg)

![e4613672712b6cbe2453aa1bd5f4c3f2a1b1f77db2e7b687f46bde3e391d7f19.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/images/e4613672712b6cbe2453aa1bd5f4c3f2a1b1f77db2e7b687f46bde3e391d7f19.jpg)

### Tables

![4723ecdccc23840f329b6875635c357c324ba9ef7e2c4d36f871844e7c42fde9.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/tables/4723ecdccc23840f329b6875635c357c324ba9ef7e2c4d36f871844e7c42fde9.jpg)

![53f6c085285b2859e220beeca79e9acf612ee84b87fe066ebc18c0d43ea282e8.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/tables/53f6c085285b2859e220beeca79e9acf612ee84b87fe066ebc18c0d43ea282e8.jpg)

![55c585a218419719841f1c9dd84de439db48089c6910a2d1680c6dd5ec56b11b.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/tables/55c585a218419719841f1c9dd84de439db48089c6910a2d1680c6dd5ec56b11b.jpg)

![599f0a46388f29cfaaf3fe0797a9834d6f1a872864b066be69e65b8f3e6a3c46.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/tables/599f0a46388f29cfaaf3fe0797a9834d6f1a872864b066be69e65b8f3e6a3c46.jpg)

![6a6b651170a863412cf9ee8bf75c81833438a1d9b1d3432a6c19e39fbd14bef3.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/tables/6a6b651170a863412cf9ee8bf75c81833438a1d9b1d3432a6c19e39fbd14bef3.jpg)

![e8ef512780e1418ed63f063b80eb470357e281cdba41cb70a3206591a4aa061e.jpg](../icml_results/4813_DeepHA_%20Scaling%20Action%20Chains%20Elicits%20Deep%20Hierarchical%20Agents/tables/e8ef512780e1418ed63f063b80eb470357e281cdba41cb70a3206591a4aa061e.jpg)

<a id="paper-4766"></a>

## OpenHA: A Series of Open-Source Hierarchical Agentic Models in Minecraft


### Images

![171cacb9a3f2f20648f8cf509fadedcabe997957f21d95beafd33e9bb21045d9.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/images/171cacb9a3f2f20648f8cf509fadedcabe997957f21d95beafd33e9bb21045d9.jpg)

![35d1d6c2bbfeaa2f7676911e1b7c6f42f76f0b95aa050b7f190b18e0d7d11387.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/images/35d1d6c2bbfeaa2f7676911e1b7c6f42f76f0b95aa050b7f190b18e0d7d11387.jpg)

![59f7d75a14bd96f129fc62317737dee4b51eb800ae80ea5a4d2649e76ac666c8.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/images/59f7d75a14bd96f129fc62317737dee4b51eb800ae80ea5a4d2649e76ac666c8.jpg)

![6c310b72f75678b6c1056bd99a7b9ee7dc1e0d1fd122891a1d5c3bbb0d295fa7.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/images/6c310b72f75678b6c1056bd99a7b9ee7dc1e0d1fd122891a1d5c3bbb0d295fa7.jpg)

![baade76095f2eef0ed152b1098d4f64873d839107bf3a9e4f0398bd5286f4a70.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/images/baade76095f2eef0ed152b1098d4f64873d839107bf3a9e4f0398bd5286f4a70.jpg)

![c0dbc862280b0cc7cf88b23f7b91729b8861f09d833d70e1eca2c6575594d58a.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/images/c0dbc862280b0cc7cf88b23f7b91729b8861f09d833d70e1eca2c6575594d58a.jpg)

![ec205f96f08ccee7dedfa6ba4ba5f703d5ea8af4bf8e5c4a45fc58d37c81514d.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/images/ec205f96f08ccee7dedfa6ba4ba5f703d5ea8af4bf8e5c4a45fc58d37c81514d.jpg)

![f928c655ce05de5e98ee69cb87ad8ecf91dab241d618edfb53952ada69846e85.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/images/f928c655ce05de5e98ee69cb87ad8ecf91dab241d618edfb53952ada69846e85.jpg)

### Tables

![0c5bd438f13ef80b5c1e02eb0c433348aa976a5d27d1d6339731496f798240d1.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/tables/0c5bd438f13ef80b5c1e02eb0c433348aa976a5d27d1d6339731496f798240d1.jpg)

![0d926b4137c1a592c608447cfc39c5260e5536c34eb7748709576b2839a007f1.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/tables/0d926b4137c1a592c608447cfc39c5260e5536c34eb7748709576b2839a007f1.jpg)

![0e8c7d8fedcce6a45ee1397178085cce09b94a054db95d7c51b6e1a794797914.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/tables/0e8c7d8fedcce6a45ee1397178085cce09b94a054db95d7c51b6e1a794797914.jpg)

![1756fdc2b97261ad6b811dd332ae7e672e73e3796c49a178bb7ce2ea12336bbe.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/tables/1756fdc2b97261ad6b811dd332ae7e672e73e3796c49a178bb7ce2ea12336bbe.jpg)

![343c5ffd642c5c5f13a9142a20218f7822882d684fa281b13131d086bd881825.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/tables/343c5ffd642c5c5f13a9142a20218f7822882d684fa281b13131d086bd881825.jpg)

![4739611416c6a33f56331c1996f19de061a259eb40f9d0830410a2b96ae9162b.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/tables/4739611416c6a33f56331c1996f19de061a259eb40f9d0830410a2b96ae9162b.jpg)

![59c33f76620ddc5790a8e42d4d02c8d6a71df0431a836e259cd23bc34d727e80.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/tables/59c33f76620ddc5790a8e42d4d02c8d6a71df0431a836e259cd23bc34d727e80.jpg)

![66aae75794393323d468bc0fc08f3ed1a26fa75694f72b98117405a5dd8d3b16.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/tables/66aae75794393323d468bc0fc08f3ed1a26fa75694f72b98117405a5dd8d3b16.jpg)

![7a3b0089fbe30099a3d0c5a82526b8a6d301c4edd11826937c74593151e3873f.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/tables/7a3b0089fbe30099a3d0c5a82526b8a6d301c4edd11826937c74593151e3873f.jpg)

![7e61778f09e350e023557d00f92b4bc218beda6abd1b80a56666c5208ac5e631.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/tables/7e61778f09e350e023557d00f92b4bc218beda6abd1b80a56666c5208ac5e631.jpg)

![9156f21f74266331372b87851053a3a95441749f1f27cf27fa11c8a93a34084a.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/tables/9156f21f74266331372b87851053a3a95441749f1f27cf27fa11c8a93a34084a.jpg)

![9bd0a7144c89835771b266fc5f8d4034486bc7f9c7ad3d3a318dc3dd24194cec.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/tables/9bd0a7144c89835771b266fc5f8d4034486bc7f9c7ad3d3a318dc3dd24194cec.jpg)

![a4d4dadf9fc94d0de5bd8189d548dea0cee775cab436b32f39705bae3fa1f4dd.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/tables/a4d4dadf9fc94d0de5bd8189d548dea0cee775cab436b32f39705bae3fa1f4dd.jpg)

![d0e16cc3ce7699823bb95177b98219b12d1d7892b5acc1722bc32c495b7a18f7.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/tables/d0e16cc3ce7699823bb95177b98219b12d1d7892b5acc1722bc32c495b7a18f7.jpg)

![d9b9491930356d6fbff787e70103249fec2690dea3216ce800415e420c6a0813.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/tables/d9b9491930356d6fbff787e70103249fec2690dea3216ce800415e420c6a0813.jpg)

![f2116f182a5d495ec0f6dcda03499f071d736bd33e9c58108297e55d85ceb2a9.jpg](../icml_results/4814_OpenHA_%20A%20Series%20of%20Open-Source%20Hierarchical%20Agentic%20Models%20in%20Minecraft/tables/f2116f182a5d495ec0f6dcda03499f071d736bd33e9c58108297e55d85ceb2a9.jpg)

<a id="paper-4767"></a>

## Reflect-then-Correct: Rebalancing Task Optimization for Generalizable Meta-Reinforcement Learning via Distributional Value Error Reduction


### Images

![1c28a72d664aa035ee889b9080604cbc9c95f26166f4dda9ec6f2055b1698815.jpg](../icml_results/4815_Reflect-then-Correct_%20Rebalancing%20Task%20Optimization%20for%20Generalizable%20Meta-Reinforcement%20Learning%20vi/images/1c28a72d664aa035ee889b9080604cbc9c95f26166f4dda9ec6f2055b1698815.jpg)

![7533dd514341326f8fcf58ba0fbd491384cefef10a1d21c13d56c6f56754a4e1.jpg](../icml_results/4815_Reflect-then-Correct_%20Rebalancing%20Task%20Optimization%20for%20Generalizable%20Meta-Reinforcement%20Learning%20vi/images/7533dd514341326f8fcf58ba0fbd491384cefef10a1d21c13d56c6f56754a4e1.jpg)

![c7223e91b4aa3ddac857d6c305349c5d787b717be0dfa68f3fde6d25cd84896b.jpg](../icml_results/4815_Reflect-then-Correct_%20Rebalancing%20Task%20Optimization%20for%20Generalizable%20Meta-Reinforcement%20Learning%20vi/images/c7223e91b4aa3ddac857d6c305349c5d787b717be0dfa68f3fde6d25cd84896b.jpg)

![cba9596056256d7ea0ba4b004fcb36cf20073afbb6dd76e40b8fd470ccaeccfb.jpg](../icml_results/4815_Reflect-then-Correct_%20Rebalancing%20Task%20Optimization%20for%20Generalizable%20Meta-Reinforcement%20Learning%20vi/images/cba9596056256d7ea0ba4b004fcb36cf20073afbb6dd76e40b8fd470ccaeccfb.jpg)

![e23f12789754568125a91b397363839c31ca457bf9c502f3ecf2e44c397b4258.jpg](../icml_results/4815_Reflect-then-Correct_%20Rebalancing%20Task%20Optimization%20for%20Generalizable%20Meta-Reinforcement%20Learning%20vi/images/e23f12789754568125a91b397363839c31ca457bf9c502f3ecf2e44c397b4258.jpg)

![f0b5b1d4bacc81d037a99b112d6facaaffac57eb18c71bd6dfbfcc26de64c81e.jpg](../icml_results/4815_Reflect-then-Correct_%20Rebalancing%20Task%20Optimization%20for%20Generalizable%20Meta-Reinforcement%20Learning%20vi/images/f0b5b1d4bacc81d037a99b112d6facaaffac57eb18c71bd6dfbfcc26de64c81e.jpg)

![f818abb537012cb990fa3c1efbcc76eef37b3cc6f6d8c970986f91a7e6c0d8bb.jpg](../icml_results/4815_Reflect-then-Correct_%20Rebalancing%20Task%20Optimization%20for%20Generalizable%20Meta-Reinforcement%20Learning%20vi/images/f818abb537012cb990fa3c1efbcc76eef37b3cc6f6d8c970986f91a7e6c0d8bb.jpg)

### Tables

![3e896d5345491f8c5490b0513c94c685afc41d4d0e242c13a1b1ec6359609c36.jpg](../icml_results/4815_Reflect-then-Correct_%20Rebalancing%20Task%20Optimization%20for%20Generalizable%20Meta-Reinforcement%20Learning%20vi/tables/3e896d5345491f8c5490b0513c94c685afc41d4d0e242c13a1b1ec6359609c36.jpg)

![6ebeebb05ffcd831f0c940c2bf60b5d7c25d307e6c954489b6d4b6d1cf8d0ff5.jpg](../icml_results/4815_Reflect-then-Correct_%20Rebalancing%20Task%20Optimization%20for%20Generalizable%20Meta-Reinforcement%20Learning%20vi/tables/6ebeebb05ffcd831f0c940c2bf60b5d7c25d307e6c954489b6d4b6d1cf8d0ff5.jpg)

![86a4de960f9a810c4f356ccab220d4b971af53aff118a321fdb9f8ead278583e.jpg](../icml_results/4815_Reflect-then-Correct_%20Rebalancing%20Task%20Optimization%20for%20Generalizable%20Meta-Reinforcement%20Learning%20vi/tables/86a4de960f9a810c4f356ccab220d4b971af53aff118a321fdb9f8ead278583e.jpg)

![aa14e7567df109c2a9a2c87ad294fbb523930e53f980923eb1ce5d2b2537d2e3.jpg](../icml_results/4815_Reflect-then-Correct_%20Rebalancing%20Task%20Optimization%20for%20Generalizable%20Meta-Reinforcement%20Learning%20vi/tables/aa14e7567df109c2a9a2c87ad294fbb523930e53f980923eb1ce5d2b2537d2e3.jpg)

![dc80968c9c4623e86e49f357fdd09c3e8af357a9afe835b5bafd1683d9e6add9.jpg](../icml_results/4815_Reflect-then-Correct_%20Rebalancing%20Task%20Optimization%20for%20Generalizable%20Meta-Reinforcement%20Learning%20vi/tables/dc80968c9c4623e86e49f357fdd09c3e8af357a9afe835b5bafd1683d9e6add9.jpg)

![e974b741cce64047a4eacf91ad69fbe4e95e4aa9f404642e5ee083bb92f6d0a7.jpg](../icml_results/4815_Reflect-then-Correct_%20Rebalancing%20Task%20Optimization%20for%20Generalizable%20Meta-Reinforcement%20Learning%20vi/tables/e974b741cce64047a4eacf91ad69fbe4e95e4aa9f404642e5ee083bb92f6d0a7.jpg)

<a id="paper-4768"></a>

## InfraRL: A Benchmark for Constrained Resource Allocation in Large-Scale Infrastructure Asset Management


### Images

![2eaa4165ae5ecc0c67f2963fe526c129b34415374f1a76df80065aaef5c97038.jpg](../icml_results/4816_InfraRL_%20A%20Benchmark%20for%20Constrained%20Resource%20Allocation%20in%20Large-Scale%20Infrastructure%20Asset%20Managem/images/2eaa4165ae5ecc0c67f2963fe526c129b34415374f1a76df80065aaef5c97038.jpg)

![67ee5fbbf1ba2244031fdb14e7bb659e27c217b6a808a3737ee6dd05fd2916e7.jpg](../icml_results/4816_InfraRL_%20A%20Benchmark%20for%20Constrained%20Resource%20Allocation%20in%20Large-Scale%20Infrastructure%20Asset%20Managem/images/67ee5fbbf1ba2244031fdb14e7bb659e27c217b6a808a3737ee6dd05fd2916e7.jpg)

![78c675d4e4811d4a93ac97cf5e0ff385de8f46015a3283099d70e72f92c3f2ac.jpg](../icml_results/4816_InfraRL_%20A%20Benchmark%20for%20Constrained%20Resource%20Allocation%20in%20Large-Scale%20Infrastructure%20Asset%20Managem/images/78c675d4e4811d4a93ac97cf5e0ff385de8f46015a3283099d70e72f92c3f2ac.jpg)

![817f9ee6506bc4408b5c39aeb07861a054ea707174347d6864380dc516111b90.jpg](../icml_results/4816_InfraRL_%20A%20Benchmark%20for%20Constrained%20Resource%20Allocation%20in%20Large-Scale%20Infrastructure%20Asset%20Managem/images/817f9ee6506bc4408b5c39aeb07861a054ea707174347d6864380dc516111b90.jpg)

![c3c0a211e2f8ed3f34858563d47d71aa6dbc2d1615fbdadd7fd913a16a29e50c.jpg](../icml_results/4816_InfraRL_%20A%20Benchmark%20for%20Constrained%20Resource%20Allocation%20in%20Large-Scale%20Infrastructure%20Asset%20Managem/images/c3c0a211e2f8ed3f34858563d47d71aa6dbc2d1615fbdadd7fd913a16a29e50c.jpg)

### Tables

![08ea93154ab0f2128db35b2651c4428a67ebca184776cbbeaf570ff91e98ec46.jpg](../icml_results/4816_InfraRL_%20A%20Benchmark%20for%20Constrained%20Resource%20Allocation%20in%20Large-Scale%20Infrastructure%20Asset%20Managem/tables/08ea93154ab0f2128db35b2651c4428a67ebca184776cbbeaf570ff91e98ec46.jpg)

![1b9dae29a8bfa3f49e77c841037e3554cb84d342f64a5a3f8edcb45e46dcb6c4.jpg](../icml_results/4816_InfraRL_%20A%20Benchmark%20for%20Constrained%20Resource%20Allocation%20in%20Large-Scale%20Infrastructure%20Asset%20Managem/tables/1b9dae29a8bfa3f49e77c841037e3554cb84d342f64a5a3f8edcb45e46dcb6c4.jpg)

![3112053ae94823a26d5af28e2795586b3b24a7cf3bf50d3550bba22f233dedc3.jpg](../icml_results/4816_InfraRL_%20A%20Benchmark%20for%20Constrained%20Resource%20Allocation%20in%20Large-Scale%20Infrastructure%20Asset%20Managem/tables/3112053ae94823a26d5af28e2795586b3b24a7cf3bf50d3550bba22f233dedc3.jpg)

![3bf624dd5a9c13a9d121df09114af34870a2b3c2f8642dd168629bccfb86dd1c.jpg](../icml_results/4816_InfraRL_%20A%20Benchmark%20for%20Constrained%20Resource%20Allocation%20in%20Large-Scale%20Infrastructure%20Asset%20Managem/tables/3bf624dd5a9c13a9d121df09114af34870a2b3c2f8642dd168629bccfb86dd1c.jpg)

![3e839467c75b14b57d94ab6bc8b22c8d9eb1e735c6852eac82abd872dcd55435.jpg](../icml_results/4816_InfraRL_%20A%20Benchmark%20for%20Constrained%20Resource%20Allocation%20in%20Large-Scale%20Infrastructure%20Asset%20Managem/tables/3e839467c75b14b57d94ab6bc8b22c8d9eb1e735c6852eac82abd872dcd55435.jpg)

![3fd5fd3787cada0f912bcef76da1135f1c304069cb356201639b73881d2940e7.jpg](../icml_results/4816_InfraRL_%20A%20Benchmark%20for%20Constrained%20Resource%20Allocation%20in%20Large-Scale%20Infrastructure%20Asset%20Managem/tables/3fd5fd3787cada0f912bcef76da1135f1c304069cb356201639b73881d2940e7.jpg)

![437ccbf8779afd1ecf001c24641e72e5ea5647273163ff2ef44ca10792c8bb60.jpg](../icml_results/4816_InfraRL_%20A%20Benchmark%20for%20Constrained%20Resource%20Allocation%20in%20Large-Scale%20Infrastructure%20Asset%20Managem/tables/437ccbf8779afd1ecf001c24641e72e5ea5647273163ff2ef44ca10792c8bb60.jpg)

![505b97742591dd4c77b73a5636c00b6b01ea2a2a37999ecbd9d0ab645faef778.jpg](../icml_results/4816_InfraRL_%20A%20Benchmark%20for%20Constrained%20Resource%20Allocation%20in%20Large-Scale%20Infrastructure%20Asset%20Managem/tables/505b97742591dd4c77b73a5636c00b6b01ea2a2a37999ecbd9d0ab645faef778.jpg)

![7a48b651c863a859655e3ad70a876ff2b93eb8139f62742dd606e07135407808.jpg](../icml_results/4816_InfraRL_%20A%20Benchmark%20for%20Constrained%20Resource%20Allocation%20in%20Large-Scale%20Infrastructure%20Asset%20Managem/tables/7a48b651c863a859655e3ad70a876ff2b93eb8139f62742dd606e07135407808.jpg)

![8541a0a53eb43d7d3cda306e51f13bc021366bc7677665017ee5304bf09582fe.jpg](../icml_results/4816_InfraRL_%20A%20Benchmark%20for%20Constrained%20Resource%20Allocation%20in%20Large-Scale%20Infrastructure%20Asset%20Managem/tables/8541a0a53eb43d7d3cda306e51f13bc021366bc7677665017ee5304bf09582fe.jpg)

![932f94bed0d594713c8c933aea9a38bc0656bbc0f4392e8c82d091889baf15e2.jpg](../icml_results/4816_InfraRL_%20A%20Benchmark%20for%20Constrained%20Resource%20Allocation%20in%20Large-Scale%20Infrastructure%20Asset%20Managem/tables/932f94bed0d594713c8c933aea9a38bc0656bbc0f4392e8c82d091889baf15e2.jpg)

![bdbada33bb358ed951951e11fe837d2e193af31aaf56d83e32c4c3bdf335f659.jpg](../icml_results/4816_InfraRL_%20A%20Benchmark%20for%20Constrained%20Resource%20Allocation%20in%20Large-Scale%20Infrastructure%20Asset%20Managem/tables/bdbada33bb358ed951951e11fe837d2e193af31aaf56d83e32c4c3bdf335f659.jpg)

<a id="paper-4769"></a>

## Decomposing the Basic Abilities of Large Language Models: Mitigating Cross-Task Interference in Multi-Task Instruct-Tuning


### Images

![01d2f15c445169dc3dd5264a567cc88803ac3dbab6d2cd975c3304efd89e45d0.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/images/01d2f15c445169dc3dd5264a567cc88803ac3dbab6d2cd975c3304efd89e45d0.jpg)

![54b1595ef1daa5dada1f053f1bc2f1c6142c601da8612840e34f3cd2b56c7768.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/images/54b1595ef1daa5dada1f053f1bc2f1c6142c601da8612840e34f3cd2b56c7768.jpg)

![5ba3ab8c225f717879afe38d4ed7f0f10e60e6469539584f3bc57418a5ecc201.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/images/5ba3ab8c225f717879afe38d4ed7f0f10e60e6469539584f3bc57418a5ecc201.jpg)

![5fd200fadd622e0d17a21fb445e337e11ad01daf0dd5892cad15d6ebacaa4435.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/images/5fd200fadd622e0d17a21fb445e337e11ad01daf0dd5892cad15d6ebacaa4435.jpg)

![63f76a39a5204f5807233887f21716b928eed3d6e2ab2567f39941a0c4af50f9.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/images/63f76a39a5204f5807233887f21716b928eed3d6e2ab2567f39941a0c4af50f9.jpg)

![7c5aae497ef438fca98947a3f804e69151424e81f31e36fff20266c9564e225c.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/images/7c5aae497ef438fca98947a3f804e69151424e81f31e36fff20266c9564e225c.jpg)

![94f2c694831e53b92b43524a94f73c03ab4b8d866a9b0029210dcaeaa42e7c05.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/images/94f2c694831e53b92b43524a94f73c03ab4b8d866a9b0029210dcaeaa42e7c05.jpg)

![c03886958c9f62fef2c55df1244f7668b96df37378f7c3c632772825b28cb269.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/images/c03886958c9f62fef2c55df1244f7668b96df37378f7c3c632772825b28cb269.jpg)

![c425ddfad0191666f24cd18b1c66421c5add4dbc5430bfb4f5e0a6421d11839a.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/images/c425ddfad0191666f24cd18b1c66421c5add4dbc5430bfb4f5e0a6421d11839a.jpg)

![cb093fe478a8008d74e2ca004b9df6d45ee53050668064545abb402c19df8242.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/images/cb093fe478a8008d74e2ca004b9df6d45ee53050668064545abb402c19df8242.jpg)

![de3ba4de8562860849412e55e11b6773c84c48ae2dc72d507e381f61096fa663.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/images/de3ba4de8562860849412e55e11b6773c84c48ae2dc72d507e381f61096fa663.jpg)

![ec4ee473a5b1841cca9a376918b626a15cdc607ddca8a2c2c3375d45af5b2a80.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/images/ec4ee473a5b1841cca9a376918b626a15cdc607ddca8a2c2c3375d45af5b2a80.jpg)

![f1b7eab03066ea5570f6f3b5fb7bce36094741a80836c88cf401829504a911e0.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/images/f1b7eab03066ea5570f6f3b5fb7bce36094741a80836c88cf401829504a911e0.jpg)

### Tables

![1e71d853cca0f5eab5361c3a45b134908e6b128e9e3f834ab3f1a171cd4df183.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/tables/1e71d853cca0f5eab5361c3a45b134908e6b128e9e3f834ab3f1a171cd4df183.jpg)

![2ae290007c459f0c402444f3488682288fef59b03c7bb57b789c834a378418eb.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/tables/2ae290007c459f0c402444f3488682288fef59b03c7bb57b789c834a378418eb.jpg)

![2c4e9adc4a36d6d1cbf3aede22b7bdda04027399b590d509a6649034502f424e.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/tables/2c4e9adc4a36d6d1cbf3aede22b7bdda04027399b590d509a6649034502f424e.jpg)

![57f0259a1f803a16733b6645680cbb0d91949518c9c25205157956e5cd2271ab.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/tables/57f0259a1f803a16733b6645680cbb0d91949518c9c25205157956e5cd2271ab.jpg)

![73d66bac51931ea4004474c220bc33fbb599057a78f836eb400b548bff4ad372.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/tables/73d66bac51931ea4004474c220bc33fbb599057a78f836eb400b548bff4ad372.jpg)

![d0d3b2f61eea6ba89a97c9ccc0f0facb3fe795d962840bdf11b8ee25de5ce693.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/tables/d0d3b2f61eea6ba89a97c9ccc0f0facb3fe795d962840bdf11b8ee25de5ce693.jpg)

![d8f17d2dae189a28afe846efbbb635a4dffd23306f254da68dbff4e5adb14fff.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/tables/d8f17d2dae189a28afe846efbbb635a4dffd23306f254da68dbff4e5adb14fff.jpg)

![e3c6c0dc5280460007c346534e967c7874a122c64eaf32995f4c29e6bc498761.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/tables/e3c6c0dc5280460007c346534e967c7874a122c64eaf32995f4c29e6bc498761.jpg)

![f54b3b034f0995c598b4bd3c783d090fadf3cca59cc2e284d11fdb1580390f5f.jpg](../icml_results/4817_Decomposing%20the%20Basic%20Abilities%20of%20Large%20Language%20Models_%20Mitigating%20Cross-Task%20Interference%20in%20Mult/tables/f54b3b034f0995c598b4bd3c783d090fadf3cca59cc2e284d11fdb1580390f5f.jpg)

<a id="paper-4770"></a>

## SLIP-RS: Structured-Attribute Language-Image Pre-Training for Remote Sensing Object Detection


### Images

![46285f9faa35e435c8c6efd81514bab15c6e76818ddc86ca75fdc81fae4f0a7b.jpg](../icml_results/4818_SLIP-RS_%20Structured-Attribute%20Language-Image%20Pre-Training%20for%20Remote%20Sensing%20Object%20Detection/images/46285f9faa35e435c8c6efd81514bab15c6e76818ddc86ca75fdc81fae4f0a7b.jpg)

![6b7127b7b4fa6bdc366b4db1d7e5663a588dc5b051b3789b49310b6d0db866d6.jpg](../icml_results/4818_SLIP-RS_%20Structured-Attribute%20Language-Image%20Pre-Training%20for%20Remote%20Sensing%20Object%20Detection/images/6b7127b7b4fa6bdc366b4db1d7e5663a588dc5b051b3789b49310b6d0db866d6.jpg)

![d1f4a0c61c32e5d0567355f99d2140fbef2e58b38dc0bb3f2de1f23f423a0024.jpg](../icml_results/4818_SLIP-RS_%20Structured-Attribute%20Language-Image%20Pre-Training%20for%20Remote%20Sensing%20Object%20Detection/images/d1f4a0c61c32e5d0567355f99d2140fbef2e58b38dc0bb3f2de1f23f423a0024.jpg)

![e0e703102454ffbf87ab49e558c457e21337fbc20e70b8a33ecead90d69d55a4.jpg](../icml_results/4818_SLIP-RS_%20Structured-Attribute%20Language-Image%20Pre-Training%20for%20Remote%20Sensing%20Object%20Detection/images/e0e703102454ffbf87ab49e558c457e21337fbc20e70b8a33ecead90d69d55a4.jpg)

![feb1d9332669eb8b3c7790af711eddb878cb5e39845becac687541e1e8f74623.jpg](../icml_results/4818_SLIP-RS_%20Structured-Attribute%20Language-Image%20Pre-Training%20for%20Remote%20Sensing%20Object%20Detection/images/feb1d9332669eb8b3c7790af711eddb878cb5e39845becac687541e1e8f74623.jpg)

### Tables

![0028905a6f54887e0bbb604b98be8701bad0c2ddfdb256c52485f64de5c39eec.jpg](../icml_results/4818_SLIP-RS_%20Structured-Attribute%20Language-Image%20Pre-Training%20for%20Remote%20Sensing%20Object%20Detection/tables/0028905a6f54887e0bbb604b98be8701bad0c2ddfdb256c52485f64de5c39eec.jpg)

![1562fe940dd953ce1b5a716265af087a9d5ce7c6f5183e6d4a17d4e3fba10856.jpg](../icml_results/4818_SLIP-RS_%20Structured-Attribute%20Language-Image%20Pre-Training%20for%20Remote%20Sensing%20Object%20Detection/tables/1562fe940dd953ce1b5a716265af087a9d5ce7c6f5183e6d4a17d4e3fba10856.jpg)

![303c5f46003ba3dcf58469ed32388d67d4406b85b84e3630b4b456274beaa59e.jpg](../icml_results/4818_SLIP-RS_%20Structured-Attribute%20Language-Image%20Pre-Training%20for%20Remote%20Sensing%20Object%20Detection/tables/303c5f46003ba3dcf58469ed32388d67d4406b85b84e3630b4b456274beaa59e.jpg)

![69a19b7e02ebfdedcc23dea9b7181a84005e9717175e92906fcd3c93f64f4e23.jpg](../icml_results/4818_SLIP-RS_%20Structured-Attribute%20Language-Image%20Pre-Training%20for%20Remote%20Sensing%20Object%20Detection/tables/69a19b7e02ebfdedcc23dea9b7181a84005e9717175e92906fcd3c93f64f4e23.jpg)

![9dc698f47bf96ed60c1a85932cb393e8248becfa3deff8b9bdaa8810b1b7d7f7.jpg](../icml_results/4818_SLIP-RS_%20Structured-Attribute%20Language-Image%20Pre-Training%20for%20Remote%20Sensing%20Object%20Detection/tables/9dc698f47bf96ed60c1a85932cb393e8248becfa3deff8b9bdaa8810b1b7d7f7.jpg)

![a4521dbcdf081eaf59c06b84e161f3a3bbdf6ed498b4ce1bde9b3163635e0604.jpg](../icml_results/4818_SLIP-RS_%20Structured-Attribute%20Language-Image%20Pre-Training%20for%20Remote%20Sensing%20Object%20Detection/tables/a4521dbcdf081eaf59c06b84e161f3a3bbdf6ed498b4ce1bde9b3163635e0604.jpg)

![b2f03dc2fb63f2931810def881f8df6770f9a0d9d19a15542f098141f385a04d.jpg](../icml_results/4818_SLIP-RS_%20Structured-Attribute%20Language-Image%20Pre-Training%20for%20Remote%20Sensing%20Object%20Detection/tables/b2f03dc2fb63f2931810def881f8df6770f9a0d9d19a15542f098141f385a04d.jpg)

![c4e3251c240b2faecd7f6a325b623423381dc2e3d7e10d0010380f96b8fb66fb.jpg](../icml_results/4818_SLIP-RS_%20Structured-Attribute%20Language-Image%20Pre-Training%20for%20Remote%20Sensing%20Object%20Detection/tables/c4e3251c240b2faecd7f6a325b623423381dc2e3d7e10d0010380f96b8fb66fb.jpg)

![ea7f6bbc75d99b9f93454bd163b80e50ad4fb55531af6a5c15f9ef6d531456b3.jpg](../icml_results/4818_SLIP-RS_%20Structured-Attribute%20Language-Image%20Pre-Training%20for%20Remote%20Sensing%20Object%20Detection/tables/ea7f6bbc75d99b9f93454bd163b80e50ad4fb55531af6a5c15f9ef6d531456b3.jpg)

<a id="paper-4771"></a>

## Lightweight Federated Incremental Learning via Decoupled Replay


### Images

![6412f74028e8acc6df8dfa1f62b952cad707c08dbb60980ca5adda1502db1d77.jpg](../icml_results/4819_Lightweight%20Federated%20Incremental%20Learning%20via%20Decoupled%20Replay/images/6412f74028e8acc6df8dfa1f62b952cad707c08dbb60980ca5adda1502db1d77.jpg)

![6b3ae19973f0d2d86418ac4c3a6d8e2c1a7a62c19d87bbf2c32986a4cd5311d9.jpg](../icml_results/4819_Lightweight%20Federated%20Incremental%20Learning%20via%20Decoupled%20Replay/images/6b3ae19973f0d2d86418ac4c3a6d8e2c1a7a62c19d87bbf2c32986a4cd5311d9.jpg)

![6cb3796d82191ac509095f7e22a7723c74d47b0860675fab1f8dbf7a0879ceb5.jpg](../icml_results/4819_Lightweight%20Federated%20Incremental%20Learning%20via%20Decoupled%20Replay/images/6cb3796d82191ac509095f7e22a7723c74d47b0860675fab1f8dbf7a0879ceb5.jpg)

![8832531e1facbbebe97dd372d7d16fbe7603a17f5f036dd530729ae77689b58a.jpg](../icml_results/4819_Lightweight%20Federated%20Incremental%20Learning%20via%20Decoupled%20Replay/images/8832531e1facbbebe97dd372d7d16fbe7603a17f5f036dd530729ae77689b58a.jpg)

### Tables

![38ac173128f618f68d0a1336e3ec8558c2981efddfa3b2bfe9d321574b535958.jpg](../icml_results/4819_Lightweight%20Federated%20Incremental%20Learning%20via%20Decoupled%20Replay/tables/38ac173128f618f68d0a1336e3ec8558c2981efddfa3b2bfe9d321574b535958.jpg)

![94712bc9357de9e62d329940f0139387234e2e97c4c49a2bdeb8675d600af90b.jpg](../icml_results/4819_Lightweight%20Federated%20Incremental%20Learning%20via%20Decoupled%20Replay/tables/94712bc9357de9e62d329940f0139387234e2e97c4c49a2bdeb8675d600af90b.jpg)

![a931f7cb985f5724749aa2d2721cb36437e9d0bfe4d776e20e8cb8317db21820.jpg](../icml_results/4819_Lightweight%20Federated%20Incremental%20Learning%20via%20Decoupled%20Replay/tables/a931f7cb985f5724749aa2d2721cb36437e9d0bfe4d776e20e8cb8317db21820.jpg)

![dae69adc3ff3b129e2f75dbb4a63664c1fe26fc0a953f82523984889224be2e1.jpg](../icml_results/4819_Lightweight%20Federated%20Incremental%20Learning%20via%20Decoupled%20Replay/tables/dae69adc3ff3b129e2f75dbb4a63664c1fe26fc0a953f82523984889224be2e1.jpg)

<a id="paper-4772"></a>

## Universal Skeleton Understanding via Differentiable Rendering and MLLMs


### Images

![04b0b7477d99228742f57a6b874e9e107a9732b08a28c50a7561aad124215ccb.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/images/04b0b7477d99228742f57a6b874e9e107a9732b08a28c50a7561aad124215ccb.jpg)

![17991c3ea04ae3c02f2bf3c6ebc840720a04aaf76bc84975990e2a876bb45371.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/images/17991c3ea04ae3c02f2bf3c6ebc840720a04aaf76bc84975990e2a876bb45371.jpg)

![3c86490c6d2f8894eab995299c8012b992dca68650d710472af6e70e26040169.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/images/3c86490c6d2f8894eab995299c8012b992dca68650d710472af6e70e26040169.jpg)

![461c085320df6ea74021491d0ce9c8637c5d08f382350daa4ba8f2cb0918a0a4.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/images/461c085320df6ea74021491d0ce9c8637c5d08f382350daa4ba8f2cb0918a0a4.jpg)

![63f57af673acea6da22f2de323cb496b49c1b13da71093f8911b28449f470e3a.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/images/63f57af673acea6da22f2de323cb496b49c1b13da71093f8911b28449f470e3a.jpg)

![652c7fab77a18562e99d627bc347ff3b79c1ff5bb575a2380f4f0929d32adacd.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/images/652c7fab77a18562e99d627bc347ff3b79c1ff5bb575a2380f4f0929d32adacd.jpg)

![96e3ae1ab9931d2348ac45d7f0a8ad48bf954f971ffb556171a111d4bc1e235e.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/images/96e3ae1ab9931d2348ac45d7f0a8ad48bf954f971ffb556171a111d4bc1e235e.jpg)

![afee920d267e5f37e64af8289a3cf2b257ed41d3ac6b2026cfdface13800f15f.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/images/afee920d267e5f37e64af8289a3cf2b257ed41d3ac6b2026cfdface13800f15f.jpg)

![b18c563b24dd779c7da94f0d43e431a4f0dae6254cca647d56fd796503438b24.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/images/b18c563b24dd779c7da94f0d43e431a4f0dae6254cca647d56fd796503438b24.jpg)

![bb103a6b09f08d5086953a5fe8bf22d55fe3abd7c132f730a2ffd51fe3c64967.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/images/bb103a6b09f08d5086953a5fe8bf22d55fe3abd7c132f730a2ffd51fe3c64967.jpg)

![d3456d416226a34da872c1933661af40285d725ce656e741e70608b5cb1e0cbe.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/images/d3456d416226a34da872c1933661af40285d725ce656e741e70608b5cb1e0cbe.jpg)

![f45b145b066f1defc01f730f27eaaa8ec86ed8fbd822abf8c0fc3ce499fc4ca7.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/images/f45b145b066f1defc01f730f27eaaa8ec86ed8fbd822abf8c0fc3ce499fc4ca7.jpg)

### Tables

![0026f44e526069838ca6b21f8c906d391ded82dbdd3ecb6e36fa499b18da2d63.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/0026f44e526069838ca6b21f8c906d391ded82dbdd3ecb6e36fa499b18da2d63.jpg)

![04a1411fdbc939dbdda510ebd054bef04f74de22554a1d774c5d36b05d4f20aa.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/04a1411fdbc939dbdda510ebd054bef04f74de22554a1d774c5d36b05d4f20aa.jpg)

![0b5fa7a0dfedafff94b4f24969be4b2fd1ac2fcf5aab7368a3eb0fda722cea23.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/0b5fa7a0dfedafff94b4f24969be4b2fd1ac2fcf5aab7368a3eb0fda722cea23.jpg)

![134e2bf4606a3740c84e246a40072d6b48a8a6521306388ec9a96ec7e1d14886.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/134e2bf4606a3740c84e246a40072d6b48a8a6521306388ec9a96ec7e1d14886.jpg)

![181a38b767769ca13b30c563cd058a752787083dc018f6c1273dbbe1bae59f1d.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/181a38b767769ca13b30c563cd058a752787083dc018f6c1273dbbe1bae59f1d.jpg)

![1c76c58c69b92df267e1c12cdd03dda45d96331926361c0b39515740269cfabc.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/1c76c58c69b92df267e1c12cdd03dda45d96331926361c0b39515740269cfabc.jpg)

![1ea1d055dadc553514fb30bd026b4975096f306b4b8ec3ea073e089199cee82b.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/1ea1d055dadc553514fb30bd026b4975096f306b4b8ec3ea073e089199cee82b.jpg)

![364b59524d7538d514ea5b979fa100385c433cae567419e347ba8c68007308c1.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/364b59524d7538d514ea5b979fa100385c433cae567419e347ba8c68007308c1.jpg)

![519ffc55999b401d42fe9bf0b1039926b88f3b803a593449ed2c2a1a77c98c14.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/519ffc55999b401d42fe9bf0b1039926b88f3b803a593449ed2c2a1a77c98c14.jpg)

![662b07710e4124b4f2895577de2bff1991702f541a77f0887460c5e1fb61f11c.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/662b07710e4124b4f2895577de2bff1991702f541a77f0887460c5e1fb61f11c.jpg)

![7074f55468e36767fb3146d82ec3a0348970b4f76b74dbbfcacd89ef907d466f.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/7074f55468e36767fb3146d82ec3a0348970b4f76b74dbbfcacd89ef907d466f.jpg)

![7e8437a97537649137efd7935be108f8511389bf5944e7372146b4a3d5fc6187.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/7e8437a97537649137efd7935be108f8511389bf5944e7372146b4a3d5fc6187.jpg)

![81b33dd5d48153e8cba18ed8cf3c9b8c5fe047dddddba8520ea13b3ee4f64de5.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/81b33dd5d48153e8cba18ed8cf3c9b8c5fe047dddddba8520ea13b3ee4f64de5.jpg)

![94b2005427749d4d33633221efd3853b3ebed2dce384b11694eb6089a5863d82.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/94b2005427749d4d33633221efd3853b3ebed2dce384b11694eb6089a5863d82.jpg)

![9a78ec3f8f471a67609640a7d211acf94e75a630fd9c0227a25e0ac24471f651.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/9a78ec3f8f471a67609640a7d211acf94e75a630fd9c0227a25e0ac24471f651.jpg)

![9f0ca004420d60897198ac0cfbc5b880d8f6d679a940e8af4b90e91a7d03d100.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/9f0ca004420d60897198ac0cfbc5b880d8f6d679a940e8af4b90e91a7d03d100.jpg)

![a445f162c279a7ffb088d681a15a670932fd1e6fa79ccd051d218ab4022521ad.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/a445f162c279a7ffb088d681a15a670932fd1e6fa79ccd051d218ab4022521ad.jpg)

![af7f78535835e6b576c032bd7662ae9fec18f4f81646fe937719ade207e8c1d3.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/af7f78535835e6b576c032bd7662ae9fec18f4f81646fe937719ade207e8c1d3.jpg)

![ba62a8aa114fb5d558800b8d12ab053d881a630e3e4c81f0312c1d3dde568811.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/ba62a8aa114fb5d558800b8d12ab053d881a630e3e4c81f0312c1d3dde568811.jpg)

![be78831cec692cd67da1ecacc8a7062cfc6ef08879229986ce1acfe07513fabf.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/be78831cec692cd67da1ecacc8a7062cfc6ef08879229986ce1acfe07513fabf.jpg)

![e95ebbd95a743706523e56ab17a5d437798e3e11dbfac768cc595c6e694aa7ba.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/e95ebbd95a743706523e56ab17a5d437798e3e11dbfac768cc595c6e694aa7ba.jpg)

![ffad8528488a221e304ed10b33562c0f5834f68e3ec60ac84565464f4962f987.jpg](../icml_results/4820_Universal%20Skeleton%20Understanding%20via%20Differentiable%20Rendering%20and%20MLLMs/tables/ffad8528488a221e304ed10b33562c0f5834f68e3ec60ac84565464f4962f987.jpg)

<a id="paper-4773"></a>

## CodeChemist: Test-Time Scaling for Low-Resource Code Generation via Functional Knowledge Transfer


### Images

![0a7b2446e6af707e772d63dd1bebd889abea73c6b7786291f79bcce928e8b823.jpg](../icml_results/4821_CodeChemist_%20Test-Time%20Scaling%20for%20Low-Resource%20Code%20Generation%20via%20Functional%20Knowledge%20Transfer/images/0a7b2446e6af707e772d63dd1bebd889abea73c6b7786291f79bcce928e8b823.jpg)

![0f499c97ba15ba4b2f4098fac80316a216867d083ab14c83bee6525d238294df.jpg](../icml_results/4821_CodeChemist_%20Test-Time%20Scaling%20for%20Low-Resource%20Code%20Generation%20via%20Functional%20Knowledge%20Transfer/images/0f499c97ba15ba4b2f4098fac80316a216867d083ab14c83bee6525d238294df.jpg)

![62bfec26de69fab863e285b913e1b4ab6b5568f0152659f462b479ae4318be0d.jpg](../icml_results/4821_CodeChemist_%20Test-Time%20Scaling%20for%20Low-Resource%20Code%20Generation%20via%20Functional%20Knowledge%20Transfer/images/62bfec26de69fab863e285b913e1b4ab6b5568f0152659f462b479ae4318be0d.jpg)

![a87d5d4caf7f039310e1810d6959499b15e76dcc1482b757036ec517f9a796ba.jpg](../icml_results/4821_CodeChemist_%20Test-Time%20Scaling%20for%20Low-Resource%20Code%20Generation%20via%20Functional%20Knowledge%20Transfer/images/a87d5d4caf7f039310e1810d6959499b15e76dcc1482b757036ec517f9a796ba.jpg)

### Tables

![0cccef1a4ec72ecbcafbb12c30f4c19b6b1233956ca59d427140ffcf0861fbd9.jpg](../icml_results/4821_CodeChemist_%20Test-Time%20Scaling%20for%20Low-Resource%20Code%20Generation%20via%20Functional%20Knowledge%20Transfer/tables/0cccef1a4ec72ecbcafbb12c30f4c19b6b1233956ca59d427140ffcf0861fbd9.jpg)

![24fb571ea58733c40a6d40811e19ffbb402a3560f2b48d090c77a1f53cc17a91.jpg](../icml_results/4821_CodeChemist_%20Test-Time%20Scaling%20for%20Low-Resource%20Code%20Generation%20via%20Functional%20Knowledge%20Transfer/tables/24fb571ea58733c40a6d40811e19ffbb402a3560f2b48d090c77a1f53cc17a91.jpg)

![3cae0d0287bb548b03bac3013c4625bf574ac863af229101347af4d234004376.jpg](../icml_results/4821_CodeChemist_%20Test-Time%20Scaling%20for%20Low-Resource%20Code%20Generation%20via%20Functional%20Knowledge%20Transfer/tables/3cae0d0287bb548b03bac3013c4625bf574ac863af229101347af4d234004376.jpg)

![59ed54c8414c21c57303c474329332dc4e81a64bc72f9da1c9a21223011f7774.jpg](../icml_results/4821_CodeChemist_%20Test-Time%20Scaling%20for%20Low-Resource%20Code%20Generation%20via%20Functional%20Knowledge%20Transfer/tables/59ed54c8414c21c57303c474329332dc4e81a64bc72f9da1c9a21223011f7774.jpg)

![7f6856b23085c159dbf6e815816a99da34a92024aea9b60a74c45c9fbf1576ae.jpg](../icml_results/4821_CodeChemist_%20Test-Time%20Scaling%20for%20Low-Resource%20Code%20Generation%20via%20Functional%20Knowledge%20Transfer/tables/7f6856b23085c159dbf6e815816a99da34a92024aea9b60a74c45c9fbf1576ae.jpg)

![86afaee10d8270de4acc423f0c632e74d74a2d454cbb49e1ed236c9460633b7d.jpg](../icml_results/4821_CodeChemist_%20Test-Time%20Scaling%20for%20Low-Resource%20Code%20Generation%20via%20Functional%20Knowledge%20Transfer/tables/86afaee10d8270de4acc423f0c632e74d74a2d454cbb49e1ed236c9460633b7d.jpg)

![8faa67894ab2db1673f8abaff99b6b119dfe4a6117625308e370e903438d025d.jpg](../icml_results/4821_CodeChemist_%20Test-Time%20Scaling%20for%20Low-Resource%20Code%20Generation%20via%20Functional%20Knowledge%20Transfer/tables/8faa67894ab2db1673f8abaff99b6b119dfe4a6117625308e370e903438d025d.jpg)

![af6e057652770e68f1cb4db55f366c9af1b2bb673a3dc1befcddc9d07fd9038e.jpg](../icml_results/4821_CodeChemist_%20Test-Time%20Scaling%20for%20Low-Resource%20Code%20Generation%20via%20Functional%20Knowledge%20Transfer/tables/af6e057652770e68f1cb4db55f366c9af1b2bb673a3dc1befcddc9d07fd9038e.jpg)

![b95166bec8818fcb058be2350de25f6b4fe25d203747b412965079527c0c4142.jpg](../icml_results/4821_CodeChemist_%20Test-Time%20Scaling%20for%20Low-Resource%20Code%20Generation%20via%20Functional%20Knowledge%20Transfer/tables/b95166bec8818fcb058be2350de25f6b4fe25d203747b412965079527c0c4142.jpg)

![bd526ab8e088f2062845b9a15e8de9f2ef7ecf832a5f9718f514748783dab7ac.jpg](../icml_results/4821_CodeChemist_%20Test-Time%20Scaling%20for%20Low-Resource%20Code%20Generation%20via%20Functional%20Knowledge%20Transfer/tables/bd526ab8e088f2062845b9a15e8de9f2ef7ecf832a5f9718f514748783dab7ac.jpg)

![c3ed7952b72163cda33d7be6d2200dadf2da3064331dc47a50d1f690e34c37f9.jpg](../icml_results/4821_CodeChemist_%20Test-Time%20Scaling%20for%20Low-Resource%20Code%20Generation%20via%20Functional%20Knowledge%20Transfer/tables/c3ed7952b72163cda33d7be6d2200dadf2da3064331dc47a50d1f690e34c37f9.jpg)

![c8151683593cc1499b0ee5d91b1cae16663e87fed8821cc97cdb835f63a45cdf.jpg](../icml_results/4821_CodeChemist_%20Test-Time%20Scaling%20for%20Low-Resource%20Code%20Generation%20via%20Functional%20Knowledge%20Transfer/tables/c8151683593cc1499b0ee5d91b1cae16663e87fed8821cc97cdb835f63a45cdf.jpg)

![cba69fc927e03938e70431ae28a235bf94a69de77403eaf375d79a3df029070a.jpg](../icml_results/4821_CodeChemist_%20Test-Time%20Scaling%20for%20Low-Resource%20Code%20Generation%20via%20Functional%20Knowledge%20Transfer/tables/cba69fc927e03938e70431ae28a235bf94a69de77403eaf375d79a3df029070a.jpg)

<a id="paper-4774"></a>

## Muon in Associative Memory Learning: Training Dynamics and Scaling Laws


### Images

![1a410d559377a17a49b2f53a20ba7c85c66782b912679c63a0b8ecb313e14b61.jpg](../icml_results/4822_Muon%20in%20Associative%20Memory%20Learning_%20Training%20Dynamics%20and%20Scaling%20Laws/images/1a410d559377a17a49b2f53a20ba7c85c66782b912679c63a0b8ecb313e14b61.jpg)

![5322e3e31d4863f6a29a948ff95d9e15eefe969f58c9d0b85ba38a206a405cb6.jpg](../icml_results/4822_Muon%20in%20Associative%20Memory%20Learning_%20Training%20Dynamics%20and%20Scaling%20Laws/images/5322e3e31d4863f6a29a948ff95d9e15eefe969f58c9d0b85ba38a206a405cb6.jpg)

![7264905f2da06f873b6a4a143ac49a41efb8ed96427c39044426bdc24c0659eb.jpg](../icml_results/4822_Muon%20in%20Associative%20Memory%20Learning_%20Training%20Dynamics%20and%20Scaling%20Laws/images/7264905f2da06f873b6a4a143ac49a41efb8ed96427c39044426bdc24c0659eb.jpg)

![a2c9adb023af60fd68c431fa41d5dde966edbc29938671a7fe73088c0289d0a5.jpg](../icml_results/4822_Muon%20in%20Associative%20Memory%20Learning_%20Training%20Dynamics%20and%20Scaling%20Laws/images/a2c9adb023af60fd68c431fa41d5dde966edbc29938671a7fe73088c0289d0a5.jpg)

![f641c21f9a3fee73e8da99fb33fbdf142f665698edadcb0b00ba60617751160d.jpg](../icml_results/4822_Muon%20in%20Associative%20Memory%20Learning_%20Training%20Dynamics%20and%20Scaling%20Laws/images/f641c21f9a3fee73e8da99fb33fbdf142f665698edadcb0b00ba60617751160d.jpg)

<a id="paper-4775"></a>

## Learning Attribute–Affordance Hierarchies in Hyperbolic Space for Open-Vocabulary 3D Object Affordance Grounding


### Images

![122ee4558b558a0a6752285ada989c5fbed9dad2f8b04348c76a686b569989a0.jpg](../icml_results/4823_Learning%20Attribute%E2%80%93Affordance%20Hierarchies%20in%20Hyperbolic%20Space%20for%20Open-Vocabulary%203D%20Object%20Affordan/images/122ee4558b558a0a6752285ada989c5fbed9dad2f8b04348c76a686b569989a0.jpg)

![28658596e762c7dc7922f5a88b482f8fefb79773e64309e1515c47c6c23d37d5.jpg](../icml_results/4823_Learning%20Attribute%E2%80%93Affordance%20Hierarchies%20in%20Hyperbolic%20Space%20for%20Open-Vocabulary%203D%20Object%20Affordan/images/28658596e762c7dc7922f5a88b482f8fefb79773e64309e1515c47c6c23d37d5.jpg)

![ae55f4148c2216faa658ca4d18d1d4e32b958e1bb02717c02b97d6cdb95fa06f.jpg](../icml_results/4823_Learning%20Attribute%E2%80%93Affordance%20Hierarchies%20in%20Hyperbolic%20Space%20for%20Open-Vocabulary%203D%20Object%20Affordan/images/ae55f4148c2216faa658ca4d18d1d4e32b958e1bb02717c02b97d6cdb95fa06f.jpg)

![c13aa1eb72d5c950998b189e44b034001a61db99fd9a336b4e99ad3134ffec11.jpg](../icml_results/4823_Learning%20Attribute%E2%80%93Affordance%20Hierarchies%20in%20Hyperbolic%20Space%20for%20Open-Vocabulary%203D%20Object%20Affordan/images/c13aa1eb72d5c950998b189e44b034001a61db99fd9a336b4e99ad3134ffec11.jpg)

![f37fb5a4f7656a0b29407e2354475deeb2aaf835965b2c432c61affc0514bb92.jpg](../icml_results/4823_Learning%20Attribute%E2%80%93Affordance%20Hierarchies%20in%20Hyperbolic%20Space%20for%20Open-Vocabulary%203D%20Object%20Affordan/images/f37fb5a4f7656a0b29407e2354475deeb2aaf835965b2c432c61affc0514bb92.jpg)

![f667b7f243ce97df42995d44190b96fa7ca04edea6052a1c2991f2a3e9959f2c.jpg](../icml_results/4823_Learning%20Attribute%E2%80%93Affordance%20Hierarchies%20in%20Hyperbolic%20Space%20for%20Open-Vocabulary%203D%20Object%20Affordan/images/f667b7f243ce97df42995d44190b96fa7ca04edea6052a1c2991f2a3e9959f2c.jpg)

### Tables

![294418cefd23f1e76feee78ec7807fabd7b4461fa980a47f1ffe151fb0953afe.jpg](../icml_results/4823_Learning%20Attribute%E2%80%93Affordance%20Hierarchies%20in%20Hyperbolic%20Space%20for%20Open-Vocabulary%203D%20Object%20Affordan/tables/294418cefd23f1e76feee78ec7807fabd7b4461fa980a47f1ffe151fb0953afe.jpg)

![38ff5f62d89b2b1132f7c2ed255949f0c0a2131dbe64df9d78e566939fde89da.jpg](../icml_results/4823_Learning%20Attribute%E2%80%93Affordance%20Hierarchies%20in%20Hyperbolic%20Space%20for%20Open-Vocabulary%203D%20Object%20Affordan/tables/38ff5f62d89b2b1132f7c2ed255949f0c0a2131dbe64df9d78e566939fde89da.jpg)

![6c01163bd98ca79373a0024aaaa5c6f4c0a60a90c726492b4bef336b8a3d6d90.jpg](../icml_results/4823_Learning%20Attribute%E2%80%93Affordance%20Hierarchies%20in%20Hyperbolic%20Space%20for%20Open-Vocabulary%203D%20Object%20Affordan/tables/6c01163bd98ca79373a0024aaaa5c6f4c0a60a90c726492b4bef336b8a3d6d90.jpg)

![9313e517322af26ca6222cf8545d722e17adb8821139ee41c854abf5d76c81bd.jpg](../icml_results/4823_Learning%20Attribute%E2%80%93Affordance%20Hierarchies%20in%20Hyperbolic%20Space%20for%20Open-Vocabulary%203D%20Object%20Affordan/tables/9313e517322af26ca6222cf8545d722e17adb8821139ee41c854abf5d76c81bd.jpg)

![9514ed6f372f785efa3370b54d5f628ad2a34a0cdaa2c60c24886cf95c5ef71e.jpg](../icml_results/4823_Learning%20Attribute%E2%80%93Affordance%20Hierarchies%20in%20Hyperbolic%20Space%20for%20Open-Vocabulary%203D%20Object%20Affordan/tables/9514ed6f372f785efa3370b54d5f628ad2a34a0cdaa2c60c24886cf95c5ef71e.jpg)

![a4a8f8b0948d7cfc291699c486ae9d9aa443c38619fff2ca4bf43c9bc27dbf65.jpg](../icml_results/4823_Learning%20Attribute%E2%80%93Affordance%20Hierarchies%20in%20Hyperbolic%20Space%20for%20Open-Vocabulary%203D%20Object%20Affordan/tables/a4a8f8b0948d7cfc291699c486ae9d9aa443c38619fff2ca4bf43c9bc27dbf65.jpg)

![dbef835ec4125159d5f677ab4ec6fbaa2f5b83871f741bac00f27ba00e045a1d.jpg](../icml_results/4823_Learning%20Attribute%E2%80%93Affordance%20Hierarchies%20in%20Hyperbolic%20Space%20for%20Open-Vocabulary%203D%20Object%20Affordan/tables/dbef835ec4125159d5f677ab4ec6fbaa2f5b83871f741bac00f27ba00e045a1d.jpg)

![dd7142055eac7bbe1024509862931c1968070702af27a19cd5572a80e00f5b2f.jpg](../icml_results/4823_Learning%20Attribute%E2%80%93Affordance%20Hierarchies%20in%20Hyperbolic%20Space%20for%20Open-Vocabulary%203D%20Object%20Affordan/tables/dd7142055eac7bbe1024509862931c1968070702af27a19cd5572a80e00f5b2f.jpg)

![f5e3221d193e6c7a8424c5f48433fdc0c1886adbb65008079f71e62ac453fa60.jpg](../icml_results/4823_Learning%20Attribute%E2%80%93Affordance%20Hierarchies%20in%20Hyperbolic%20Space%20for%20Open-Vocabulary%203D%20Object%20Affordan/tables/f5e3221d193e6c7a8424c5f48433fdc0c1886adbb65008079f71e62ac453fa60.jpg)

![fb28c3a2d5befa2181ba2d3a51ab0520e80421eda446e45bd19eeb168982a627.jpg](../icml_results/4823_Learning%20Attribute%E2%80%93Affordance%20Hierarchies%20in%20Hyperbolic%20Space%20for%20Open-Vocabulary%203D%20Object%20Affordan/tables/fb28c3a2d5befa2181ba2d3a51ab0520e80421eda446e45bd19eeb168982a627.jpg)

<a id="paper-4776"></a>

## SuCo: Sufficiency-guided Continuous Adaptive Reasoning


### Images

![4165dca7a4a481cde48fa5dfbce81510e5a97f848f9ac87155b680116b33a470.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/images/4165dca7a4a481cde48fa5dfbce81510e5a97f848f9ac87155b680116b33a470.jpg)

![69f170bc0fc148324940deb358202f782e58e2b2a27fe76ddfd692477be6f1c7.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/images/69f170bc0fc148324940deb358202f782e58e2b2a27fe76ddfd692477be6f1c7.jpg)

![6a47836c6d85cfd23fda6bdf128bccdbca8714acb08f075edaf63dac0f2f703c.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/images/6a47836c6d85cfd23fda6bdf128bccdbca8714acb08f075edaf63dac0f2f703c.jpg)

![87e0897f11b8fb502155e0d753ed0a7d55db92efde4457c756c073847ab94710.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/images/87e0897f11b8fb502155e0d753ed0a7d55db92efde4457c756c073847ab94710.jpg)

![a5beaca2cf7aae53260b3afd9dd506cb5a6cf68ef2c79dd13a273b2e32ffe36d.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/images/a5beaca2cf7aae53260b3afd9dd506cb5a6cf68ef2c79dd13a273b2e32ffe36d.jpg)

![b440efb6c01c1534703072880ed9f2f510ccf50b57c01bba8fa4d320ef0c4a0a.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/images/b440efb6c01c1534703072880ed9f2f510ccf50b57c01bba8fa4d320ef0c4a0a.jpg)

![b98449601b8bb8de4a8725948a3b96ac8aee1ec568ae9afe88560a1db1930a64.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/images/b98449601b8bb8de4a8725948a3b96ac8aee1ec568ae9afe88560a1db1930a64.jpg)

![c4be713b09b77b0053154bf5dbdb2a9074d2ad14821d5d3b856d4ad0be9d447d.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/images/c4be713b09b77b0053154bf5dbdb2a9074d2ad14821d5d3b856d4ad0be9d447d.jpg)

![d087423236269f1e30f8ae64457c2c3dcc8cf26eaed2eefec2d6f33c708db42e.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/images/d087423236269f1e30f8ae64457c2c3dcc8cf26eaed2eefec2d6f33c708db42e.jpg)

### Tables

![342be18c670bba447232771bd5ddbaec936a9b3dd69164c2de73dd6ab0e5721b.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/tables/342be18c670bba447232771bd5ddbaec936a9b3dd69164c2de73dd6ab0e5721b.jpg)

![5782fdebe9a0f27990632118bd479dc15314cb59509d62b229e688d9604aef92.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/tables/5782fdebe9a0f27990632118bd479dc15314cb59509d62b229e688d9604aef92.jpg)

![581c3601bb72d02e67aea7a39249a2652a5304ecac0ac687e30f32c5e41108f7.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/tables/581c3601bb72d02e67aea7a39249a2652a5304ecac0ac687e30f32c5e41108f7.jpg)

![b566fa50035801000d60915f475191711e4ee366b7a060f68314eb00989a0833.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/tables/b566fa50035801000d60915f475191711e4ee366b7a060f68314eb00989a0833.jpg)

![b9485630a3f122cfcf1030b01e2f8788bd02182045f8f07d5aa7ebbfc615d7c6.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/tables/b9485630a3f122cfcf1030b01e2f8788bd02182045f8f07d5aa7ebbfc615d7c6.jpg)

![c6937f0e48e2b73995353108fefca949ed2e5dbab244af4be94c5681acd8dc11.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/tables/c6937f0e48e2b73995353108fefca949ed2e5dbab244af4be94c5681acd8dc11.jpg)

![d167b987e11f5342d5070c3c18720fdef4aa874a110af87cb8a06d979856e2f2.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/tables/d167b987e11f5342d5070c3c18720fdef4aa874a110af87cb8a06d979856e2f2.jpg)

![d474c8e89e60adc7c9cdc18833fef406b84bcedef02401d76133171799baf398.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/tables/d474c8e89e60adc7c9cdc18833fef406b84bcedef02401d76133171799baf398.jpg)

![d6415dc42a74aef19d2886529c046665016e55f7f5b1b3b55a00e9151f84846c.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/tables/d6415dc42a74aef19d2886529c046665016e55f7f5b1b3b55a00e9151f84846c.jpg)

![e65f59ec3d3857cafdfb7967e915dbf6e9a0de8588e2929134ba0a7b90aec259.jpg](../icml_results/4824_SuCo_%20Sufficiency-guided%20Continuous%20Adaptive%20Reasoning/tables/e65f59ec3d3857cafdfb7967e915dbf6e9a0de8588e2929134ba0a7b90aec259.jpg)

<a id="paper-4777"></a>

## CausalRAG2: Hierarchical Causal Knowledge Graph Design for RAG


### Images

![0475dba9861ea8b0b45ab86f53e2050e0a83322502b70ffedb3a2b24aa376cc8.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/images/0475dba9861ea8b0b45ab86f53e2050e0a83322502b70ffedb3a2b24aa376cc8.jpg)

![11695e8768be8ade340cd7cc9f2caaf7a45cd4e2519e071034c60c8eb3799350.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/images/11695e8768be8ade340cd7cc9f2caaf7a45cd4e2519e071034c60c8eb3799350.jpg)

![3207e96f346ff4e60e360f68ab1e7b2447c553ce0eacc87fd439c88ad1cee250.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/images/3207e96f346ff4e60e360f68ab1e7b2447c553ce0eacc87fd439c88ad1cee250.jpg)

![4cfbc25cced343b9fccab245b8888fb6918e318a7e73d3c8b109c0c5af6f8620.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/images/4cfbc25cced343b9fccab245b8888fb6918e318a7e73d3c8b109c0c5af6f8620.jpg)

![57e9fc6adda74e32cc9e7660eb9e2564e558087e1de0247ab807091697fc76c2.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/images/57e9fc6adda74e32cc9e7660eb9e2564e558087e1de0247ab807091697fc76c2.jpg)

![8e6f06163513c07efdef610439e3256d22c44289ba18078dfa50006152234307.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/images/8e6f06163513c07efdef610439e3256d22c44289ba18078dfa50006152234307.jpg)

![ae8902a7dfd6c42ecebfe9e7a8e75a74744457dc8c72e2a09e4b10c7ef226524.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/images/ae8902a7dfd6c42ecebfe9e7a8e75a74744457dc8c72e2a09e4b10c7ef226524.jpg)

![b7ea9b1af87fbafd102b61a9de1c63599821209f048e1eddf4a7cd63f2c8096b.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/images/b7ea9b1af87fbafd102b61a9de1c63599821209f048e1eddf4a7cd63f2c8096b.jpg)

![cff76153a0a7be7560911c1019f80803e607c5c4c8304cff09132b154c88f190.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/images/cff76153a0a7be7560911c1019f80803e607c5c4c8304cff09132b154c88f190.jpg)

![d1416089b67dcb20fae9b12d70d8757bde8604d62f99ff89bdb3f1df8f903c94.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/images/d1416089b67dcb20fae9b12d70d8757bde8604d62f99ff89bdb3f1df8f903c94.jpg)

![eaf27ee5355f0294c4bb35c5d54d16a28d31b33b8ddaaec96ced45c60896644f.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/images/eaf27ee5355f0294c4bb35c5d54d16a28d31b33b8ddaaec96ced45c60896644f.jpg)

![f03a98c184e5e651581b100b817cc20d05685598e7863778929ef90c6ac4e38f.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/images/f03a98c184e5e651581b100b817cc20d05685598e7863778929ef90c6ac4e38f.jpg)

![f7c3f652e92f163cbcac6db51eeba2545865b43bef8cfaf670a4aab33e788612.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/images/f7c3f652e92f163cbcac6db51eeba2545865b43bef8cfaf670a4aab33e788612.jpg)

![f8a12bf29e0f1425183a1229b776749f19bc22f36696e1a76731808b1c09b990.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/images/f8a12bf29e0f1425183a1229b776749f19bc22f36696e1a76731808b1c09b990.jpg)

### Tables

![0656990fb82ab83cb3072ac8a75e52720880c3815c94352846e8ee4943ee9824.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/tables/0656990fb82ab83cb3072ac8a75e52720880c3815c94352846e8ee4943ee9824.jpg)

![4040da53d14e670072be189cc39d5816e26d0d429e626b4f0552a4d1b195e571.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/tables/4040da53d14e670072be189cc39d5816e26d0d429e626b4f0552a4d1b195e571.jpg)

![524ff2f977057344734d930eb6bea681616b8d06bd13171c378b6ad3c9036bf6.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/tables/524ff2f977057344734d930eb6bea681616b8d06bd13171c378b6ad3c9036bf6.jpg)

![53e471554ba17af2840df1b910bf81e8f1fa1c9906c5be23553e408494c26fbc.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/tables/53e471554ba17af2840df1b910bf81e8f1fa1c9906c5be23553e408494c26fbc.jpg)

![5ac348a1c5e598668126d6bb31e447785d7dcc87a8a156aab7d257bcd7d364a6.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/tables/5ac348a1c5e598668126d6bb31e447785d7dcc87a8a156aab7d257bcd7d364a6.jpg)

![63f141af1f2e9704d58f8ffe0ca8ef55d1b39a87e1348c1c02700bdeefc7be2e.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/tables/63f141af1f2e9704d58f8ffe0ca8ef55d1b39a87e1348c1c02700bdeefc7be2e.jpg)

![7b32b4154f2a54aba21ed75c529d37dbecfb5f7696f5831f4fbc891a70fd0ffc.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/tables/7b32b4154f2a54aba21ed75c529d37dbecfb5f7696f5831f4fbc891a70fd0ffc.jpg)

![a188523138c4db7dda7c53abd9a9a893a54b11689b9d2f3496f5a8f781cec607.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/tables/a188523138c4db7dda7c53abd9a9a893a54b11689b9d2f3496f5a8f781cec607.jpg)

![b690cb4851a66d11567ba386ea9d396cfb53d1e27994eb426e0eb8ac856d53d6.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/tables/b690cb4851a66d11567ba386ea9d396cfb53d1e27994eb426e0eb8ac856d53d6.jpg)

![d2080ae5f73c861f360b3f8b0397d5f8591160471bd132d08ad504acc199b5d4.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/tables/d2080ae5f73c861f360b3f8b0397d5f8591160471bd132d08ad504acc199b5d4.jpg)

![e90857928243f174a390918dac0bb815533d10be0ff2151d2160dc94491afc6d.jpg](../icml_results/4825_CausalRAG2_%20Hierarchical%20Causal%20Knowledge%20Graph%20Design%20for%20RAG/tables/e90857928243f174a390918dac0bb815533d10be0ff2151d2160dc94491afc6d.jpg)

<a id="paper-4778"></a>

## Reward Shaping for (Inference-Time) Alignment: A Stackelberg Game Perspective


### Images

![1ebb3c730b1b777db4f9846cc658444074c4c10753ceccf24a17af914737c8c7.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/images/1ebb3c730b1b777db4f9846cc658444074c4c10753ceccf24a17af914737c8c7.jpg)

![3df908f81c1965ae07dd353ed9babdde0feb4bf30d2339e8fa2269e6a4600f27.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/images/3df908f81c1965ae07dd353ed9babdde0feb4bf30d2339e8fa2269e6a4600f27.jpg)

![5253dde11a6e788407e940c5a1e24b95b65ed420e4b0f9ccece060a5de90c5de.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/images/5253dde11a6e788407e940c5a1e24b95b65ed420e4b0f9ccece060a5de90c5de.jpg)

![5c07c1d2229af202e76b4ee2b0a730a7b34b1df1f260e8d250d12374b3ad143e.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/images/5c07c1d2229af202e76b4ee2b0a730a7b34b1df1f260e8d250d12374b3ad143e.jpg)

![7a3a8a89971c77c2f4ad334065550bcad7e6fd7448ee78d5cdeb799e4c3da96f.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/images/7a3a8a89971c77c2f4ad334065550bcad7e6fd7448ee78d5cdeb799e4c3da96f.jpg)

![9599a2029e1223c79dafaa5d70593d47ed98c47272856727d3349b4e566940c1.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/images/9599a2029e1223c79dafaa5d70593d47ed98c47272856727d3349b4e566940c1.jpg)

![b2a0d1abfb2f394790d7e58edb130b6470a37cb6daa69feeb879e0700b05ef31.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/images/b2a0d1abfb2f394790d7e58edb130b6470a37cb6daa69feeb879e0700b05ef31.jpg)

![d9216bef3d253a87fc7ebfb0e4ce7c90985b35bc801e98aec8f0e427101db67b.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/images/d9216bef3d253a87fc7ebfb0e4ce7c90985b35bc801e98aec8f0e427101db67b.jpg)

### Tables

![06288313073b4ee0a56328358f9cc8e1c5d184cd25b5f60fa24bbb3246a74061.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/tables/06288313073b4ee0a56328358f9cc8e1c5d184cd25b5f60fa24bbb3246a74061.jpg)

![24deba967e1ea6f6c4aa7080305a569809f2dc5b2898b964b52ecf6fbe8c771e.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/tables/24deba967e1ea6f6c4aa7080305a569809f2dc5b2898b964b52ecf6fbe8c771e.jpg)

![2564f87b19932efd34e0a75b5f56bf7a39bd16783f4800bda82a5347d585d204.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/tables/2564f87b19932efd34e0a75b5f56bf7a39bd16783f4800bda82a5347d585d204.jpg)

![33dc799206741b2960effd5521e091c00341a9ebbd08351bddd1072e2841b687.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/tables/33dc799206741b2960effd5521e091c00341a9ebbd08351bddd1072e2841b687.jpg)

![40be556a4d8df96d3d53403cf078b48dd7b08fd0ab85c39df50a6069bf6272dd.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/tables/40be556a4d8df96d3d53403cf078b48dd7b08fd0ab85c39df50a6069bf6272dd.jpg)

![6453671a804eca0873dd6179543249e135cfd45edb8d8069fb79869f0808f998.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/tables/6453671a804eca0873dd6179543249e135cfd45edb8d8069fb79869f0808f998.jpg)

![64db2eb068c9ea3e6b1ed8a1eb10cb24b4759b3999ae48aae2c142557a83a9a4.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/tables/64db2eb068c9ea3e6b1ed8a1eb10cb24b4759b3999ae48aae2c142557a83a9a4.jpg)

![6effc6b776aa9e3dc0a56bc4949ed7d0080ec0d5984827137c28e69b4dbd301a.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/tables/6effc6b776aa9e3dc0a56bc4949ed7d0080ec0d5984827137c28e69b4dbd301a.jpg)

![7c862a9d38e0c100402c421c851db31fdd8374abdd3f603bd1815369fcc7052e.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/tables/7c862a9d38e0c100402c421c851db31fdd8374abdd3f603bd1815369fcc7052e.jpg)

![7fb44d7027c5954f91caece9c8a20cbb7dd7e9b40f7ba86cd1cd9b63932e936b.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/tables/7fb44d7027c5954f91caece9c8a20cbb7dd7e9b40f7ba86cd1cd9b63932e936b.jpg)

![8910cd45740b6a792f2aad42d0af4bd48d0a06469789a0a1b29ca4a0bf8feb12.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/tables/8910cd45740b6a792f2aad42d0af4bd48d0a06469789a0a1b29ca4a0bf8feb12.jpg)

![aaef4305f466f4a57943118c78bc08890d9304c09b1386b6cc5042b1dde1240e.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/tables/aaef4305f466f4a57943118c78bc08890d9304c09b1386b6cc5042b1dde1240e.jpg)

![d60ad0127e14e26af150534577154c3ab59d22ea5520d0948bde8e1ffc885e75.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/tables/d60ad0127e14e26af150534577154c3ab59d22ea5520d0948bde8e1ffc885e75.jpg)

![d7c09b838dc1aaca9b0498dc2046db5771498b8a7d741fc4c1eb066b4135196d.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/tables/d7c09b838dc1aaca9b0498dc2046db5771498b8a7d741fc4c1eb066b4135196d.jpg)

![eb59f544d5700b295a53763c1e1e2c04278c1f5d3d126a82563f472f7d724ab2.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/tables/eb59f544d5700b295a53763c1e1e2c04278c1f5d3d126a82563f472f7d724ab2.jpg)

![fe16e4c5619fd8f408f4ca239521e54f5158878ab35fdf507bb1c19864403894.jpg](../icml_results/4826_Reward%20Shaping%20for%20%28Inference-Time%29%20Alignment_%20A%20Stackelberg%20Game%20Perspective/tables/fe16e4c5619fd8f408f4ca239521e54f5158878ab35fdf507bb1c19864403894.jpg)

<a id="paper-4779"></a>

## Crisp: A Spectral-Based Interaction Strategy for Multivariate Time Series Forecasting


### Images

![2a39895c730f2a43bfab29d9bac99ea7e0ad83dbced16aa2e9e6f15ddc374471.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/images/2a39895c730f2a43bfab29d9bac99ea7e0ad83dbced16aa2e9e6f15ddc374471.jpg)

![49ad442170538e7e22959d0eff6575216377ca61708f36a3f9072249b1c0208b.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/images/49ad442170538e7e22959d0eff6575216377ca61708f36a3f9072249b1c0208b.jpg)

![532d31ead3b29e64b2b80bc22409ce526343495e763ef831f0da07621fab2753.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/images/532d31ead3b29e64b2b80bc22409ce526343495e763ef831f0da07621fab2753.jpg)

![54a5f8292d4bed902daa678fe198629d4fa096b5792829778dbc0533b8d07a48.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/images/54a5f8292d4bed902daa678fe198629d4fa096b5792829778dbc0533b8d07a48.jpg)

![6f1e0d6d88c663d5b619bcc5348ec51da3cfb0dd510690d8df9667a8f300c070.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/images/6f1e0d6d88c663d5b619bcc5348ec51da3cfb0dd510690d8df9667a8f300c070.jpg)

![79830437004c133b2bd8816ab9b27603cbbc78d2f9b3bd3c7e4dcb9701c01758.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/images/79830437004c133b2bd8816ab9b27603cbbc78d2f9b3bd3c7e4dcb9701c01758.jpg)

![842fa0d99384c86410f0cc8b17e42c88ac8584cedf577c81066a050d739f500b.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/images/842fa0d99384c86410f0cc8b17e42c88ac8584cedf577c81066a050d739f500b.jpg)

![a7bcdb47cd5cca2869edbc4fae9a440e78879acd5cfce3d9506f0353acf23424.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/images/a7bcdb47cd5cca2869edbc4fae9a440e78879acd5cfce3d9506f0353acf23424.jpg)

![b96bb485156f02df0fd8ae2d9569f46975a6a2d89432c4cf50ac6517cc883f95.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/images/b96bb485156f02df0fd8ae2d9569f46975a6a2d89432c4cf50ac6517cc883f95.jpg)

![d0852988ad86f31c9cb4e918b6027c70235dd900aa81729d85860aecf5da0f6f.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/images/d0852988ad86f31c9cb4e918b6027c70235dd900aa81729d85860aecf5da0f6f.jpg)

![d849acd073dda7b4200cc5f570e7030d3ad9429437b749095e0bccff1da03d3a.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/images/d849acd073dda7b4200cc5f570e7030d3ad9429437b749095e0bccff1da03d3a.jpg)

![fee25656727be57c13720da11e6ffb87811b7b3580bc9315b918e8c76976909b.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/images/fee25656727be57c13720da11e6ffb87811b7b3580bc9315b918e8c76976909b.jpg)

### Tables

![647573e5f5123667535f96d7472ec14fea0f305bae6a92e7a71a20e04c82a10c.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/tables/647573e5f5123667535f96d7472ec14fea0f305bae6a92e7a71a20e04c82a10c.jpg)

![8530d2ea2aec7440ca819bfa84b899b2f18f45766af43bfae8d73f447784b20c.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/tables/8530d2ea2aec7440ca819bfa84b899b2f18f45766af43bfae8d73f447784b20c.jpg)

![88f520a5498488169820134c45729704557143c9d2d84615818ef6bd739433d3.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/tables/88f520a5498488169820134c45729704557143c9d2d84615818ef6bd739433d3.jpg)

![d574f8d784529bcbbe6a9a5215c7f5429054a35d89efc56043e8febcc7a89e5c.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/tables/d574f8d784529bcbbe6a9a5215c7f5429054a35d89efc56043e8febcc7a89e5c.jpg)

![ecf25f6d64ef48f2231e6cd31b724662f5371202c4946edf513d0c92003baf18.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/tables/ecf25f6d64ef48f2231e6cd31b724662f5371202c4946edf513d0c92003baf18.jpg)

![f5423b72f5a483f713acc11086f364e2904d15be94a0384ccf5304cb8181973a.jpg](../icml_results/4827_Crisp_%20A%20Spectral-Based%20Interaction%20Strategy%20for%20Multivariate%20Time%20Series%20Forecasting/tables/f5423b72f5a483f713acc11086f364e2904d15be94a0384ccf5304cb8181973a.jpg)

<a id="paper-4780"></a>

## Towards Understanding Massive Activations in Attention Sink Mechanism


### Images

![03c7590763b468d528b1149c8f1d7de038353c09c8ddf71295b2ffbd572f43d8.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/images/03c7590763b468d528b1149c8f1d7de038353c09c8ddf71295b2ffbd572f43d8.jpg)

![18b8ce7c6b14a50d3171fdf86498af239d0e005d557548c7822739138e974d8c.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/images/18b8ce7c6b14a50d3171fdf86498af239d0e005d557548c7822739138e974d8c.jpg)

![1d276aaf39e6d0b3d7bbf736d6c4e6a2e4795b9307a824c49f6a479c08aa4492.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/images/1d276aaf39e6d0b3d7bbf736d6c4e6a2e4795b9307a824c49f6a479c08aa4492.jpg)

![20e7f2ae8f209550138e41a3b1c3e7bc0368ef9960cb614b67a363c5fe3fc71a.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/images/20e7f2ae8f209550138e41a3b1c3e7bc0368ef9960cb614b67a363c5fe3fc71a.jpg)

![3ff801936e001d4f84828b48bda47b7779c31feae6f29b7b5ee9237f9a539bd3.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/images/3ff801936e001d4f84828b48bda47b7779c31feae6f29b7b5ee9237f9a539bd3.jpg)

![5d71f37f49a05f809393fb60c64bfa466abdefed890af388987f7382a4047fc4.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/images/5d71f37f49a05f809393fb60c64bfa466abdefed890af388987f7382a4047fc4.jpg)

![5d8cf6322488d7a79c0214ef6821367101fb63db0dee8efe005884a2f9968b4b.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/images/5d8cf6322488d7a79c0214ef6821367101fb63db0dee8efe005884a2f9968b4b.jpg)

![6f639fd9b16fed6bab70851c0ceb8654945cad77a18cb90d65d2bfc4c7dcedd2.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/images/6f639fd9b16fed6bab70851c0ceb8654945cad77a18cb90d65d2bfc4c7dcedd2.jpg)

![74c78b2223a9eb9826273b751b8a867f62b3a27282ccc76b7d136b6d05b47fe8.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/images/74c78b2223a9eb9826273b751b8a867f62b3a27282ccc76b7d136b6d05b47fe8.jpg)

![922cf0e9b233c00e0d2fd0d160a120f2d615f0cec1d76885c7cbf754dbc2bb86.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/images/922cf0e9b233c00e0d2fd0d160a120f2d615f0cec1d76885c7cbf754dbc2bb86.jpg)

![ad101edb136e02d891c8ea4237cd573525a3a541d26a1043728b50a5dccd5deb.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/images/ad101edb136e02d891c8ea4237cd573525a3a541d26a1043728b50a5dccd5deb.jpg)

![b428d872cb1a80db01a650d1d4205a85b733ee95a58b475690508e32a28fee5c.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/images/b428d872cb1a80db01a650d1d4205a85b733ee95a58b475690508e32a28fee5c.jpg)

![c3fa7a1623095d4edd1eec1e1a0b019560934069697608538549d4265cb6508e.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/images/c3fa7a1623095d4edd1eec1e1a0b019560934069697608538549d4265cb6508e.jpg)

![fff1df708453122668f313f620df642f7eac7691a606e208a47b691e9151846a.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/images/fff1df708453122668f313f620df642f7eac7691a606e208a47b691e9151846a.jpg)

### Tables

![a77556867a1d4799dcd0823f18077339f7ae15c1dbd2cd7ffb31360c4342c4f7.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/tables/a77556867a1d4799dcd0823f18077339f7ae15c1dbd2cd7ffb31360c4342c4f7.jpg)

![b833bad58d7f9f6fbf63a8ebbe943ea0477d367dbde46ba24c153a2cdb705172.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/tables/b833bad58d7f9f6fbf63a8ebbe943ea0477d367dbde46ba24c153a2cdb705172.jpg)

![e8977744b67fd7e6f385381d8ee3b206d73b5a1106e932c9ed258231b0917bd4.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/tables/e8977744b67fd7e6f385381d8ee3b206d73b5a1106e932c9ed258231b0917bd4.jpg)

![fbef806733329faf68a2bde2e1e7a22a7b47b47531c02fdf6b7a949d630b7320.jpg](../icml_results/4828_Towards%20Understanding%20Massive%20Activations%20in%20Attention%20Sink%20Mechanism/tables/fbef806733329faf68a2bde2e1e7a22a7b47b47531c02fdf6b7a949d630b7320.jpg)

<a id="paper-4781"></a>

## E-mem: Multi-Agent Based Episodic Context Reconstruction for LLM Agent Memory


### Images

![5863537eb4081e38689bd6aa6b9be7b2cd553789072b1e6dafcf8558157f6db3.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/images/5863537eb4081e38689bd6aa6b9be7b2cd553789072b1e6dafcf8558157f6db3.jpg)

![67160bd1af76d5b063e703d22cdc95f45764b4ec7ed7c4d4b4177c2b718e8307.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/images/67160bd1af76d5b063e703d22cdc95f45764b4ec7ed7c4d4b4177c2b718e8307.jpg)

![ee5a00c8ba4d69059aa3c66624925310b85f3131df0eeef814c52faca6836de4.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/images/ee5a00c8ba4d69059aa3c66624925310b85f3131df0eeef814c52faca6836de4.jpg)

### Tables

![034b9b44949c5a4533dccc0ae39b93caa3525393c400848cfe2e1603afc8bcea.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/tables/034b9b44949c5a4533dccc0ae39b93caa3525393c400848cfe2e1603afc8bcea.jpg)

![1a4eb90280b3c53a1a4131571d6eb4f60ad3e3c4b9f297336630f6505cdb1019.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/tables/1a4eb90280b3c53a1a4131571d6eb4f60ad3e3c4b9f297336630f6505cdb1019.jpg)

![1bd3f8d926f22904dcf37ac7f91861dabdc9b478616f4b00ec76f80ed42f9f2b.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/tables/1bd3f8d926f22904dcf37ac7f91861dabdc9b478616f4b00ec76f80ed42f9f2b.jpg)

![48c1283442056f5128c8ae7b04b178c6fd498e188e55bec57bc755464218148c.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/tables/48c1283442056f5128c8ae7b04b178c6fd498e188e55bec57bc755464218148c.jpg)

![63e801416dd96d07973d0ff014d2aa5734faf823a9e9ca7192f671b7717a614b.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/tables/63e801416dd96d07973d0ff014d2aa5734faf823a9e9ca7192f671b7717a614b.jpg)

![8ee09d661a942f8a83413eaa7e69dc71e6c0e9a598a286eaa5aa3ab3d799de33.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/tables/8ee09d661a942f8a83413eaa7e69dc71e6c0e9a598a286eaa5aa3ab3d799de33.jpg)

![91218aa1b057246e54107b41a0af864f5290705341932e556b0412a55057fecd.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/tables/91218aa1b057246e54107b41a0af864f5290705341932e556b0412a55057fecd.jpg)

![a06276f659594b2169fe6976b01290b79e79ada5f5a04739d254ddd52527da3f.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/tables/a06276f659594b2169fe6976b01290b79e79ada5f5a04739d254ddd52527da3f.jpg)

![b2543edb7af650775e3415e53accfe585a3e0e8e7397e5b7bd628b4fe0af93e2.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/tables/b2543edb7af650775e3415e53accfe585a3e0e8e7397e5b7bd628b4fe0af93e2.jpg)

![c475b6b983a9fbae63bbad15563c2f0fc57d7c826aae483d7242a4d702a93950.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/tables/c475b6b983a9fbae63bbad15563c2f0fc57d7c826aae483d7242a4d702a93950.jpg)

![c9d3a23881328686172bce20cb4a80ce977c8fa62220ca11e418aa5c2533ee44.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/tables/c9d3a23881328686172bce20cb4a80ce977c8fa62220ca11e418aa5c2533ee44.jpg)

![d0cf3d75dccccd6994900f9f847ce6e31ae0d91e3e6cc1bfaba958ad3df83399.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/tables/d0cf3d75dccccd6994900f9f847ce6e31ae0d91e3e6cc1bfaba958ad3df83399.jpg)

![dd44d234fdfda5c090d148f575f6222bbcb3d829c3787d31e1c3211b6da2f9aa.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/tables/dd44d234fdfda5c090d148f575f6222bbcb3d829c3787d31e1c3211b6da2f9aa.jpg)

![e29d019cce1cde8e5630f11076b4dbfe304a908cba102e386fba46d8f0479eff.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/tables/e29d019cce1cde8e5630f11076b4dbfe304a908cba102e386fba46d8f0479eff.jpg)

![ec911604fcb9ef979c1ebe62a7a488cde181bb2ae35496e23cb14025595dd32e.jpg](../icml_results/4829_E-mem_%20Multi-Agent%20Based%20Episodic%20Context%20Reconstruction%20for%20LLM%20Agent%20Memory/tables/ec911604fcb9ef979c1ebe62a7a488cde181bb2ae35496e23cb14025595dd32e.jpg)

<a id="paper-4782"></a>

## Preference-Calibrated Optimization with Score-Level Distribution Alignment for Text-to-Image Diffusion Model Unlearning


### Images

![5b81bc4be40edb92eef8007aeaf3051323c4658ab8ba4b537629de0bc2aa4c69.jpg](../icml_results/4830_Preference-Calibrated%20Optimization%20with%20Score-Level%20Distribution%20Alignment%20for%20Text-to-Image%20Diffusi/images/5b81bc4be40edb92eef8007aeaf3051323c4658ab8ba4b537629de0bc2aa4c69.jpg)

![5faa49fe646e5268e7a2e9e4b1f2dc30128d0bdcba6bab2203156052932a4367.jpg](../icml_results/4830_Preference-Calibrated%20Optimization%20with%20Score-Level%20Distribution%20Alignment%20for%20Text-to-Image%20Diffusi/images/5faa49fe646e5268e7a2e9e4b1f2dc30128d0bdcba6bab2203156052932a4367.jpg)

![6dd72b2bc2e1bb64afb481fbfb0babecd9602bffba4d793316457b250e1aa810.jpg](../icml_results/4830_Preference-Calibrated%20Optimization%20with%20Score-Level%20Distribution%20Alignment%20for%20Text-to-Image%20Diffusi/images/6dd72b2bc2e1bb64afb481fbfb0babecd9602bffba4d793316457b250e1aa810.jpg)

![7b74c0abf2fa7cf8b62b6e15ec59f6601aca98e2e477b735a52658b02a702a5e.jpg](../icml_results/4830_Preference-Calibrated%20Optimization%20with%20Score-Level%20Distribution%20Alignment%20for%20Text-to-Image%20Diffusi/images/7b74c0abf2fa7cf8b62b6e15ec59f6601aca98e2e477b735a52658b02a702a5e.jpg)

![8b48af7171b0f0240f7613ce86078d75401802c45f287415dca3ac7cb088640a.jpg](../icml_results/4830_Preference-Calibrated%20Optimization%20with%20Score-Level%20Distribution%20Alignment%20for%20Text-to-Image%20Diffusi/images/8b48af7171b0f0240f7613ce86078d75401802c45f287415dca3ac7cb088640a.jpg)

![920a0a3d83fca17da21b761b6eeda5bbb897a2758436653e664069be2f69afb0.jpg](../icml_results/4830_Preference-Calibrated%20Optimization%20with%20Score-Level%20Distribution%20Alignment%20for%20Text-to-Image%20Diffusi/images/920a0a3d83fca17da21b761b6eeda5bbb897a2758436653e664069be2f69afb0.jpg)

![bd0003a1ce11a594f68b452b79aea1fbb6b192989a1bbe6da1093b70cf5f769a.jpg](../icml_results/4830_Preference-Calibrated%20Optimization%20with%20Score-Level%20Distribution%20Alignment%20for%20Text-to-Image%20Diffusi/images/bd0003a1ce11a594f68b452b79aea1fbb6b192989a1bbe6da1093b70cf5f769a.jpg)

![de50e9bf87d0562eb32eea3232553cb1b35f110c8b5f26de0ac0b2a0a0c1b18b.jpg](../icml_results/4830_Preference-Calibrated%20Optimization%20with%20Score-Level%20Distribution%20Alignment%20for%20Text-to-Image%20Diffusi/images/de50e9bf87d0562eb32eea3232553cb1b35f110c8b5f26de0ac0b2a0a0c1b18b.jpg)

### Tables

![7f4fb6f93ffc7fc566430c45abc83f76f73a2a16b799d542e0550436056851e6.jpg](../icml_results/4830_Preference-Calibrated%20Optimization%20with%20Score-Level%20Distribution%20Alignment%20for%20Text-to-Image%20Diffusi/tables/7f4fb6f93ffc7fc566430c45abc83f76f73a2a16b799d542e0550436056851e6.jpg)

![8cbffc6bfb3289f8a649707ed88694c44c71514ffff83e1b1d6fcf61d80c3832.jpg](../icml_results/4830_Preference-Calibrated%20Optimization%20with%20Score-Level%20Distribution%20Alignment%20for%20Text-to-Image%20Diffusi/tables/8cbffc6bfb3289f8a649707ed88694c44c71514ffff83e1b1d6fcf61d80c3832.jpg)

![daa7273232aa48804cc4d0555f2cea9d24ac5d53fc08b9cc68cfd19fd91be56a.jpg](../icml_results/4830_Preference-Calibrated%20Optimization%20with%20Score-Level%20Distribution%20Alignment%20for%20Text-to-Image%20Diffusi/tables/daa7273232aa48804cc4d0555f2cea9d24ac5d53fc08b9cc68cfd19fd91be56a.jpg)

<a id="paper-4783"></a>

## Beyond Static Pipelines: Learning Dynamic Workflows for Text-to-SQL


### Images

![385c1d0aece356ec78f70d43b4873602ecceed74436ddbdab1e66b466140cff1.jpg](../icml_results/4831_Beyond%20Static%20Pipelines_%20Learning%20Dynamic%20Workflows%20for%20Text-to-SQL/images/385c1d0aece356ec78f70d43b4873602ecceed74436ddbdab1e66b466140cff1.jpg)

![5257f9856a2bd1d6655e6db8ec942f6602daf934f82a10e2a6a7b8dffaa45cde.jpg](../icml_results/4831_Beyond%20Static%20Pipelines_%20Learning%20Dynamic%20Workflows%20for%20Text-to-SQL/images/5257f9856a2bd1d6655e6db8ec942f6602daf934f82a10e2a6a7b8dffaa45cde.jpg)

![6ad4522736fa559cc50e12ecfe5cc71dfa7b5fc9a584424dfe4a24a831a5e7db.jpg](../icml_results/4831_Beyond%20Static%20Pipelines_%20Learning%20Dynamic%20Workflows%20for%20Text-to-SQL/images/6ad4522736fa559cc50e12ecfe5cc71dfa7b5fc9a584424dfe4a24a831a5e7db.jpg)

![8d167108db5f4b86b24551af3767f1bc64a736f53a62b48ba85afd95b4fa1acb.jpg](../icml_results/4831_Beyond%20Static%20Pipelines_%20Learning%20Dynamic%20Workflows%20for%20Text-to-SQL/images/8d167108db5f4b86b24551af3767f1bc64a736f53a62b48ba85afd95b4fa1acb.jpg)

![a392c414f08832c19e419447ae95f05ca6f3f426674808d8c4344c6255bfe3bd.jpg](../icml_results/4831_Beyond%20Static%20Pipelines_%20Learning%20Dynamic%20Workflows%20for%20Text-to-SQL/images/a392c414f08832c19e419447ae95f05ca6f3f426674808d8c4344c6255bfe3bd.jpg)

### Tables

![0fea6ef98831cbd119a928d0a82416639a29885064e4dc2025f7f4e381c8ae5f.jpg](../icml_results/4831_Beyond%20Static%20Pipelines_%20Learning%20Dynamic%20Workflows%20for%20Text-to-SQL/tables/0fea6ef98831cbd119a928d0a82416639a29885064e4dc2025f7f4e381c8ae5f.jpg)

![26b580bfd40f9bfdcee8b7b9774a0f0332c3e6e9ea5ab59a7ad02c41183ba0e9.jpg](../icml_results/4831_Beyond%20Static%20Pipelines_%20Learning%20Dynamic%20Workflows%20for%20Text-to-SQL/tables/26b580bfd40f9bfdcee8b7b9774a0f0332c3e6e9ea5ab59a7ad02c41183ba0e9.jpg)

![42e9ac1e767e4cf2f83b060f8e07f3b60b7d39683d50b8fd0e521bddeb825b4d.jpg](../icml_results/4831_Beyond%20Static%20Pipelines_%20Learning%20Dynamic%20Workflows%20for%20Text-to-SQL/tables/42e9ac1e767e4cf2f83b060f8e07f3b60b7d39683d50b8fd0e521bddeb825b4d.jpg)

![4f4b5eaee5e2cfdc7592ada4f224a6373547c58d582b3a88ed81b16d7b3dde24.jpg](../icml_results/4831_Beyond%20Static%20Pipelines_%20Learning%20Dynamic%20Workflows%20for%20Text-to-SQL/tables/4f4b5eaee5e2cfdc7592ada4f224a6373547c58d582b3a88ed81b16d7b3dde24.jpg)

![6205c5d60fe38edaa5d9f3c6cef1324e40a684216cccf59592b9286be2782ac0.jpg](../icml_results/4831_Beyond%20Static%20Pipelines_%20Learning%20Dynamic%20Workflows%20for%20Text-to-SQL/tables/6205c5d60fe38edaa5d9f3c6cef1324e40a684216cccf59592b9286be2782ac0.jpg)

![9523534350e1d534c4d0000b868416aee5f7f3c6079b6e7ddd7147c0fec175f3.jpg](../icml_results/4831_Beyond%20Static%20Pipelines_%20Learning%20Dynamic%20Workflows%20for%20Text-to-SQL/tables/9523534350e1d534c4d0000b868416aee5f7f3c6079b6e7ddd7147c0fec175f3.jpg)

![990174dc236a4de84cd7f2a5c9ffcadba9fc8a6a87538e92f65ed1203471d99d.jpg](../icml_results/4831_Beyond%20Static%20Pipelines_%20Learning%20Dynamic%20Workflows%20for%20Text-to-SQL/tables/990174dc236a4de84cd7f2a5c9ffcadba9fc8a6a87538e92f65ed1203471d99d.jpg)

![d8a3879bb7367ad0d799c6e3f11fff65f845a66e60b7d8626cb52a91c7c057b5.jpg](../icml_results/4831_Beyond%20Static%20Pipelines_%20Learning%20Dynamic%20Workflows%20for%20Text-to-SQL/tables/d8a3879bb7367ad0d799c6e3f11fff65f845a66e60b7d8626cb52a91c7c057b5.jpg)

![f554d1dbf06b58e8c7bd73e73bb0a5c0cf3528b32b0b8a36f87ed33f0d560db0.jpg](../icml_results/4831_Beyond%20Static%20Pipelines_%20Learning%20Dynamic%20Workflows%20for%20Text-to-SQL/tables/f554d1dbf06b58e8c7bd73e73bb0a5c0cf3528b32b0b8a36f87ed33f0d560db0.jpg)

<a id="paper-4784"></a>

## Beyond Model Base Retrieval: Weaving Knowledge to Master Fine-grained Neural Network Design


### Images

![336d07b2048fd2fc66a9d60af73fab61bd87d80a299e03e82d09252563754379.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/images/336d07b2048fd2fc66a9d60af73fab61bd87d80a299e03e82d09252563754379.jpg)

![33d6c609fa47c1a3a8ca2b31613ce5b73b67f4de88853cddb02e09aba5666af1.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/images/33d6c609fa47c1a3a8ca2b31613ce5b73b67f4de88853cddb02e09aba5666af1.jpg)

![34e54f79f68b581cb572412fb94cf71d829deff21d5d8a5c900ad41480cc3087.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/images/34e54f79f68b581cb572412fb94cf71d829deff21d5d8a5c900ad41480cc3087.jpg)

![3d091b2fc23bd799031e56bcdf6574a545bad93d988368c364ba58f6cc8e44d1.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/images/3d091b2fc23bd799031e56bcdf6574a545bad93d988368c364ba58f6cc8e44d1.jpg)

![70c600f9b95ee35bdb91a6c141d4f5711bdb80f8eccdfc3e86dc4ebc19d06666.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/images/70c600f9b95ee35bdb91a6c141d4f5711bdb80f8eccdfc3e86dc4ebc19d06666.jpg)

![86371517502ae45854c2dd410e5df6abe5509aad3f6359cce86959698cafe67c.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/images/86371517502ae45854c2dd410e5df6abe5509aad3f6359cce86959698cafe67c.jpg)

![a87b5089e23d19d97272603e3ed73e84b56c0a641a818202344e04a08e9c3e25.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/images/a87b5089e23d19d97272603e3ed73e84b56c0a641a818202344e04a08e9c3e25.jpg)

![b2264bc67c3be928affb637829952f9f1651072591ad237bb288198593f49f60.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/images/b2264bc67c3be928affb637829952f9f1651072591ad237bb288198593f49f60.jpg)

![e8b717878d6cca09672e6299c3be7e3d571503fdaa9cb814c0c0bfd8f0eb4057.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/images/e8b717878d6cca09672e6299c3be7e3d571503fdaa9cb814c0c0bfd8f0eb4057.jpg)

### Tables

![0aa0410daa1c123ef38bae65c93adc0fc31bf5f114c42cc9cc36ab2491a82f72.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/tables/0aa0410daa1c123ef38bae65c93adc0fc31bf5f114c42cc9cc36ab2491a82f72.jpg)

![104cbbd936f8a59786ce9610422c931aa4c0d66fd7d2c6a5918b3b6ff05f6f74.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/tables/104cbbd936f8a59786ce9610422c931aa4c0d66fd7d2c6a5918b3b6ff05f6f74.jpg)

![1cc19877d8b83314f54c5cc0d70e0539c7a138237aefc2ae17c0008916503630.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/tables/1cc19877d8b83314f54c5cc0d70e0539c7a138237aefc2ae17c0008916503630.jpg)

![224653304032c9be7698a36b6d61e2c8a27b7275f5cc7e4ee08fc9503d0044e9.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/tables/224653304032c9be7698a36b6d61e2c8a27b7275f5cc7e4ee08fc9503d0044e9.jpg)

![48ead0ac99a761dda72bd75694f5c81110ac8befcd9b0dfd1fd19298406d7ac9.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/tables/48ead0ac99a761dda72bd75694f5c81110ac8befcd9b0dfd1fd19298406d7ac9.jpg)

![6bbc9d7f41190d1fa7ada6828422ec88c7dc52309682a9a31aeb9c1cf8ff71c0.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/tables/6bbc9d7f41190d1fa7ada6828422ec88c7dc52309682a9a31aeb9c1cf8ff71c0.jpg)

![6d71b1205c16a2483adc5796fadb076a3b0086e5dabb80c29b173296570b2519.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/tables/6d71b1205c16a2483adc5796fadb076a3b0086e5dabb80c29b173296570b2519.jpg)

![6e85166b0ee4273aea05efdb9fcb2c09c51f16ff175b53579865b220681d671c.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/tables/6e85166b0ee4273aea05efdb9fcb2c09c51f16ff175b53579865b220681d671c.jpg)

![852f3ac2ecf0b45fd23792ca434efc5f571f3d78b673ac45d1412bf6d2cbd769.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/tables/852f3ac2ecf0b45fd23792ca434efc5f571f3d78b673ac45d1412bf6d2cbd769.jpg)

![a8f36477666c66c37b8f41629256760a04521adc28e987043478aa945272c188.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/tables/a8f36477666c66c37b8f41629256760a04521adc28e987043478aa945272c188.jpg)

![ac41a06f2999607063cfbcf41fc46079b9ee84632558b5d0372ef1e28169d9c2.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/tables/ac41a06f2999607063cfbcf41fc46079b9ee84632558b5d0372ef1e28169d9c2.jpg)

![bd7cd5785b0c97974b7c6e4535ce8f0562bdecc167259cff34aa6223e29c37c8.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/tables/bd7cd5785b0c97974b7c6e4535ce8f0562bdecc167259cff34aa6223e29c37c8.jpg)

![df98d0abb3000e0b5c1b21e8a9e7866d4ac6ce7828d2d3024664792f35eb5dd1.jpg](../icml_results/4832_Beyond%20Model%20Base%20Retrieval_%20Weaving%20Knowledge%20to%20Master%20Fine-grained%20Neural%20Network%20Design/tables/df98d0abb3000e0b5c1b21e8a9e7866d4ac6ce7828d2d3024664792f35eb5dd1.jpg)

<a id="paper-4785"></a>

## DTKG: Dual-Track Knowledge Graph-Verified Reasoning Framework for Multi-Hop QA

### Images

![2a1bc91490023d2fe5a603d2fd3d71be1c65792ca68862a2cafe2e57a861bcbe.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/images/2a1bc91490023d2fe5a603d2fd3d71be1c65792ca68862a2cafe2e57a861bcbe.jpg)

![4c03813760d86aead3759eab57b721642ae7aa256c5be3befa59f785a5c7fe33.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/images/4c03813760d86aead3759eab57b721642ae7aa256c5be3befa59f785a5c7fe33.jpg)

![6e3df9511b29552099bc4efd22c9dfab766f5e0813a71e21fb8ccef77bab51b8.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/images/6e3df9511b29552099bc4efd22c9dfab766f5e0813a71e21fb8ccef77bab51b8.jpg)

![886b8718c5c5659c216b248d02177735b7479e6fedb6f7d71dbb059fee6573c2.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/images/886b8718c5c5659c216b248d02177735b7479e6fedb6f7d71dbb059fee6573c2.jpg)

![ddd25a86bacdf60879707d42366e331b32cc7c4d88fc0cebafd9921d62cd3b07.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/images/ddd25a86bacdf60879707d42366e331b32cc7c4d88fc0cebafd9921d62cd3b07.jpg)

![e050dd8d419130566297b9e0e4deeed1b5e4881b296564925d7bee8c14af28e9.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/images/e050dd8d419130566297b9e0e4deeed1b5e4881b296564925d7bee8c14af28e9.jpg)

![e5e2f37699a9757b45e866b50383f0bd7b6c43fcfe72004544c188ba8b418924.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/images/e5e2f37699a9757b45e866b50383f0bd7b6c43fcfe72004544c188ba8b418924.jpg)

### Tables

![1c6e586c47c70f6c834f595db8005359bdb21d213f763928aff39c0abf400a9f.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/tables/1c6e586c47c70f6c834f595db8005359bdb21d213f763928aff39c0abf400a9f.jpg)

![239e2f430210e14c0f48088097cc33ecb5d3e1521dfdac0fce2b7e3729376272.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/tables/239e2f430210e14c0f48088097cc33ecb5d3e1521dfdac0fce2b7e3729376272.jpg)

![3e8e845eb5a6979e04684db970d1a73787e840fe92adaf412235f495ac53bc81.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/tables/3e8e845eb5a6979e04684db970d1a73787e840fe92adaf412235f495ac53bc81.jpg)

![630f4d4eaf6457eae63fe000c9e84b135ca7f8c42ef800bcdcc33dad09504d4d.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/tables/630f4d4eaf6457eae63fe000c9e84b135ca7f8c42ef800bcdcc33dad09504d4d.jpg)

![85899db3f0735b85dccc2e5f8af4bf25682fcf9cba13e31e6443bade7f2898b6.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/tables/85899db3f0735b85dccc2e5f8af4bf25682fcf9cba13e31e6443bade7f2898b6.jpg)

![8629bbbb447a35f0aeb6dc497369fd748bfc03268dfa7dce583137e3198a1807.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/tables/8629bbbb447a35f0aeb6dc497369fd748bfc03268dfa7dce583137e3198a1807.jpg)

![8cadd19ec70c592f6c2086b067d28b9269dba5da6177228bd04ec4fa6f963d37.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/tables/8cadd19ec70c592f6c2086b067d28b9269dba5da6177228bd04ec4fa6f963d37.jpg)

![959f931b0c9287d556ee4324aaa11aa5d0e1474c08e3272fd3deb6e0de6e0478.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/tables/959f931b0c9287d556ee4324aaa11aa5d0e1474c08e3272fd3deb6e0de6e0478.jpg)

![97683213fc0e431f1064d68a4449e89ab991ec6b70acb22c9d01b50616c2cd1a.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/tables/97683213fc0e431f1064d68a4449e89ab991ec6b70acb22c9d01b50616c2cd1a.jpg)

![9c6ab4a9e6f2add8f6dbbab579181b2fb6ca93378eb7c349018aeaaed1f5451f.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/tables/9c6ab4a9e6f2add8f6dbbab579181b2fb6ca93378eb7c349018aeaaed1f5451f.jpg)

![a92d70d40a8948addf1defd1e7c0974905078b34b9cb91b5e1f9b3bf2b39d73e.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/tables/a92d70d40a8948addf1defd1e7c0974905078b34b9cb91b5e1f9b3bf2b39d73e.jpg)

![b35f274250834279f53349c5f0a585b843a142fc28c4df748011a05a3c210e50.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/tables/b35f274250834279f53349c5f0a585b843a142fc28c4df748011a05a3c210e50.jpg)

![d01a3c8b6bb08eb6ef89854fdfa64b77b6a9b2593bf1804b30cd6c7071e1f6bf.jpg](../icml_results/4833_DTKG_%20Dual-Track%20Knowledge%20Graph-Verified%20Reasoning%20Framework%20for%20Multi-Hop%20QA/tables/d01a3c8b6bb08eb6ef89854fdfa64b77b6a9b2593bf1804b30cd6c7071e1f6bf.jpg)
