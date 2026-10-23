# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 5 of 197

## 目录 (Table of Contents)

1. [On the Role of Computation in Reinforcement Learning](#paper-0133)
2. [Equivalence of Context and Parameter Updates in Modern Transformer Blocks](#paper-0134)
3. [VenusBench-Mobile: A Challenging and User-Centric Benchmark for Mobile GUI Agents with Capability Diagnostics](#paper-0135)
4. [FlashOptim: Optimizers for Memory-Efficient Training](#paper-0136)
5. [Decoupling The "What" and "Where" With Polar Coordinate Positional Embedding](#paper-0137)
6. [GEM: Geometric Erasure by Contrastive Velocity Matching in Rectified Flows](#paper-0138)
7. [What Do Agents Learn from Trajectory-SFT: Semantics or Interfaces?](#paper-0139)
8. [Detecting the Semantic Fixed Point: A Geometric Framework for Efficient Inference](#paper-0140)
9. [A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots](#paper-0141)
10. [Monitoring Monitorability](#paper-0142)
11. [Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing](#paper-0143)
12. [DRPBench: Evaluating LLMs in Concurrent Code Comprehension via Fine-Grained Data Race Prediction](#paper-0144)
13. [MEnvAgent: Scalable Polyglot Environment Construction for Verifiable Software Engineering](#paper-0145)
14. [Divide-and-Denoise: A Game-Theoretic Method for Fairly Composing Diffusion Models](#paper-0146)
15. [Why Deep Jacobian Spectra Separate: Depth-Induced Scaling and Singular-Vector Alignment](#paper-0147)
16. [Jailbreak to Protect: Buffering and Reinforcing via Temporary Jailbreaking for Safe Fine-Tuning in Large Language Models](#paper-0148)
17. [On the Optimization Trajectory of DeepWalk Embeddings](#paper-0149)
18. [From Feasible to Practical: Pareto-Optimal Synthesis Planning](#paper-0150)
19. [Flow Sampling : Learning to Sample from Unnormalized Densities via Denoising Conditional Processes](#paper-0151)
20. [Vision2Web: A Hierarchical Benchmark for Visual Website Development with Agent Verification](#paper-0152)
21. [FedPissa: Towards Federated Personalized Adaptation of Foundation Models via LoRA Subspace Mapping](#paper-0153)
22. [LiftQuant: Continuous Bit-Width LLM via Dimensional Lifting and Projection](#paper-0154)
23. [CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal](#paper-0155)
24. [ClinTutor-R1: Advancing Scalable and Robust One-to-Many Alignment in Clinical Socratic Education](#paper-0156)
25. [Joint-Space Empowerment for Dexterous Coordination in Tendon-Driven Hands](#paper-0157)
26. [Neural Feature Geometry Evolves as Discrete Ricci Flow](#paper-0158)
27. [Scalable Option Learning in High-Throughput Environments](#paper-0159)
28. [Security–Fidelity Tradeoffs: The Hidden Cost of Prompt Injection Defense](#paper-0160)
29. [Steer Like the LLM: Activation Steering that Mimics Prompting](#paper-0161)
30. [AdLift: Lifting Adversarial Perturbations to Safeguard 3D Gaussian Splatting Assets Against Instruction-Driven Editing](#paper-0162)
31. [Unifying Masked Diffusion Models with Various Generation Orders and Beyond](#paper-0163)
32. [MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks](#paper-0164)
33. [When the Prompt Becomes Visual: Vision-Centric Jailbreak Attacks for Large Image Editing Models](#paper-0165)

---


<a id="paper-0133"></a>

## On the Role of Computation in Reinforcement Learning

### Images

![069a88cc161fe64a52e594b4a54d48d3e0afd7dd772dc3a38e3162f50a7fdcfb.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/069a88cc161fe64a52e594b4a54d48d3e0afd7dd772dc3a38e3162f50a7fdcfb.jpg)

![08e014c57af086a84954167ffd952becf42854fee55613a7d0a79589e8600eaf.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/08e014c57af086a84954167ffd952becf42854fee55613a7d0a79589e8600eaf.jpg)

![1378044b0615e3f0687eecc7a61904f0cd6274f05739fc3fbf68c3ca2c3a13cf.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/1378044b0615e3f0687eecc7a61904f0cd6274f05739fc3fbf68c3ca2c3a13cf.jpg)

![197ac6c92ebf89f7eb423ee8ffe918ef261743df06d4fdf8639231cd1adeea89.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/197ac6c92ebf89f7eb423ee8ffe918ef261743df06d4fdf8639231cd1adeea89.jpg)

![2fdaaa91e5fd3a8fd809715b1128942b6bd23534673afa8db5e00286c947874e.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/2fdaaa91e5fd3a8fd809715b1128942b6bd23534673afa8db5e00286c947874e.jpg)

![459bb5f4818e056b6ef6efc35153966d69531f74c1336aa01ad1838f067e808c.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/459bb5f4818e056b6ef6efc35153966d69531f74c1336aa01ad1838f067e808c.jpg)

![4e63cf7b30a945f80159384aa78110ae48cbbc5fda7adc665d5aa9fb26838eae.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/4e63cf7b30a945f80159384aa78110ae48cbbc5fda7adc665d5aa9fb26838eae.jpg)

![51a52c85842163cf270d84cbf4d53bfa0f509b757c78a9cfd37bd4caf173a828.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/51a52c85842163cf270d84cbf4d53bfa0f509b757c78a9cfd37bd4caf173a828.jpg)

![54b744651f57cf2b05d4f0467ecd8a7911a45d4cfc5f349daa45b98ca7a0d7aa.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/54b744651f57cf2b05d4f0467ecd8a7911a45d4cfc5f349daa45b98ca7a0d7aa.jpg)

![88a0f5ee74983ebc2920891f1170b25edc98e9048ebe57b5642eecc4fe9cd430.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/88a0f5ee74983ebc2920891f1170b25edc98e9048ebe57b5642eecc4fe9cd430.jpg)

![bed1ffee8e6a171c8fa3d66f4b9fa8d4821eb98317b038e98453175f91ef7022.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/bed1ffee8e6a171c8fa3d66f4b9fa8d4821eb98317b038e98453175f91ef7022.jpg)

![d0a3db5612323b15dc9230f65ac36a5cfd0e1469894ff0d91d32e4d2d3fa8bce.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/d0a3db5612323b15dc9230f65ac36a5cfd0e1469894ff0d91d32e4d2d3fa8bce.jpg)

![ec61dd16eefc50477d4d46d1d69ba8e89a9f9e8e3aaa2584e7c31de8051c6619.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/images/ec61dd16eefc50477d4d46d1d69ba8e89a9f9e8e3aaa2584e7c31de8051c6619.jpg)

### Tables

![039e39797d5b9e893d9a011cc821b0c44406c1294d554a0dc26a06b26d52364e.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/tables/039e39797d5b9e893d9a011cc821b0c44406c1294d554a0dc26a06b26d52364e.jpg)

![2591e408d6d8b578d3617f60bce862dd952a707f5049b2c871e79dc68b845461.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/tables/2591e408d6d8b578d3617f60bce862dd952a707f5049b2c871e79dc68b845461.jpg)

![f775155e90bbc40781a1e449ee01f6c15105466424975e4c8d38964e288c2f86.jpg](../icml_results/134_Efficient%20Parallel%20Samplers%20for%20Recurrent-Depth%20Models%20and%20Their%20Connection%20to%20Diffusion%20Language%20Mo/tables/f775155e90bbc40781a1e449ee01f6c15105466424975e4c8d38964e288c2f86.jpg)

## On the Role of Computation in Reinforcement Learning


### Images

![359a0e7b71765c2b4f7a0d42014ff21a36826efeafbad0c1a73b17f42d65630f.jpg](../icml_results/135_On%20the%20Role%20of%20Computation%20in%20Reinforcement%20Learning/images/359a0e7b71765c2b4f7a0d42014ff21a36826efeafbad0c1a73b17f42d65630f.jpg)

![7ba7c894c0fbfd2ead377d4fcb85d99e78aba25d6fb4ce7aada4824b6c9a40c6.jpg](../icml_results/135_On%20the%20Role%20of%20Computation%20in%20Reinforcement%20Learning/images/7ba7c894c0fbfd2ead377d4fcb85d99e78aba25d6fb4ce7aada4824b6c9a40c6.jpg)

![981fcd87bfd83e36b402ea49d30e46797fcc18238583e2afa730d82f53bee770.jpg](../icml_results/135_On%20the%20Role%20of%20Computation%20in%20Reinforcement%20Learning/images/981fcd87bfd83e36b402ea49d30e46797fcc18238583e2afa730d82f53bee770.jpg)

![a3387d5e6ac82acba0213db81f0df643fc81e0547fc6c4102a4ee73efe66d791.jpg](../icml_results/135_On%20the%20Role%20of%20Computation%20in%20Reinforcement%20Learning/images/a3387d5e6ac82acba0213db81f0df643fc81e0547fc6c4102a4ee73efe66d791.jpg)

![b744e33428a6ea32acc56b9703fdc85f3bf80571cd64e401464fde588ec3795e.jpg](../icml_results/135_On%20the%20Role%20of%20Computation%20in%20Reinforcement%20Learning/images/b744e33428a6ea32acc56b9703fdc85f3bf80571cd64e401464fde588ec3795e.jpg)

![c4fb5dbd91275448d487144c6a0fa556cecf95c84b3dc99e08f18e6c6a2b2c15.jpg](../icml_results/135_On%20the%20Role%20of%20Computation%20in%20Reinforcement%20Learning/images/c4fb5dbd91275448d487144c6a0fa556cecf95c84b3dc99e08f18e6c6a2b2c15.jpg)

![ff4ce4de4b318f3279a0df4911cdfd6ae7dc710f47ee08caaf4e98a2502c3590.jpg](../icml_results/135_On%20the%20Role%20of%20Computation%20in%20Reinforcement%20Learning/images/ff4ce4de4b318f3279a0df4911cdfd6ae7dc710f47ee08caaf4e98a2502c3590.jpg)

### Tables

![1c1a224e23f737b3bf6fcde6c549f7108870e14cadaec740a8ac604eac018992.jpg](../icml_results/135_On%20the%20Role%20of%20Computation%20in%20Reinforcement%20Learning/tables/1c1a224e23f737b3bf6fcde6c549f7108870e14cadaec740a8ac604eac018992.jpg)

![3326a69b706b494cc98f9b979c31e5dc5eca76bce84497be734aa05408f16922.jpg](../icml_results/135_On%20the%20Role%20of%20Computation%20in%20Reinforcement%20Learning/tables/3326a69b706b494cc98f9b979c31e5dc5eca76bce84497be734aa05408f16922.jpg)

![4a70679c5873098d47f93c6d49e23ea0183090aa895dfa1b45ad39a7f86d8b26.jpg](../icml_results/135_On%20the%20Role%20of%20Computation%20in%20Reinforcement%20Learning/tables/4a70679c5873098d47f93c6d49e23ea0183090aa895dfa1b45ad39a7f86d8b26.jpg)

<a id="paper-0134"></a>

## Equivalence of Context and Parameter Updates in Modern Transformer Blocks


### Images

![0809fe18471c00ec0f62f0afe3028b3db297b00ab4781ea44764b873c5700b59.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/0809fe18471c00ec0f62f0afe3028b3db297b00ab4781ea44764b873c5700b59.jpg)

![135c29f6256be57d742e8a96ebccb529d6e41b7ceed5295dd5902f47c4e8467d.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/135c29f6256be57d742e8a96ebccb529d6e41b7ceed5295dd5902f47c4e8467d.jpg)

![1679c74825b9c17bdb0c9022b6784c147914726a972b793ef3adbc0f1b17fa8f.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/1679c74825b9c17bdb0c9022b6784c147914726a972b793ef3adbc0f1b17fa8f.jpg)

![29a22362438ce0e49ac19b9d45de5a20b33a1106ab57b0473dbc46939bb65e7d.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/29a22362438ce0e49ac19b9d45de5a20b33a1106ab57b0473dbc46939bb65e7d.jpg)

![441747db2b7c1ac67529968c0da133249e5a1afa0d992368241690c33768c8e8.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/441747db2b7c1ac67529968c0da133249e5a1afa0d992368241690c33768c8e8.jpg)

![4a94ae4c40ebbcd2cdea85ff047e9c8b9aa12f9993459929c16693499bfaa392.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/4a94ae4c40ebbcd2cdea85ff047e9c8b9aa12f9993459929c16693499bfaa392.jpg)

![623c40b2ecf422ce04e4be9d5fbe157f3235162e7a970b3895019def489372a4.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/623c40b2ecf422ce04e4be9d5fbe157f3235162e7a970b3895019def489372a4.jpg)

![690449601f0782f1ec4f8d79e10d5ca35a77f4401e2c810a0dc539f4e1b1d52e.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/690449601f0782f1ec4f8d79e10d5ca35a77f4401e2c810a0dc539f4e1b1d52e.jpg)

![6e6243eb04e6105e50b9579ea03e3a09043e8fe3be510806011c8c2046f1ee70.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/6e6243eb04e6105e50b9579ea03e3a09043e8fe3be510806011c8c2046f1ee70.jpg)

![752fb3e1b970b3e23192a3ab65b346869f789434c5abe16b73db81e4f8ab414f.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/752fb3e1b970b3e23192a3ab65b346869f789434c5abe16b73db81e4f8ab414f.jpg)

![77ba2aab124f58fdd9105dedbc08743b29fca11860a7c54ab835b085d7af8d1f.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/77ba2aab124f58fdd9105dedbc08743b29fca11860a7c54ab835b085d7af8d1f.jpg)

![7ec9a916b953742b15c1c65d5e2b044e262cf46dc62eeccb91554f408904aefc.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/7ec9a916b953742b15c1c65d5e2b044e262cf46dc62eeccb91554f408904aefc.jpg)

![998894c923296cd4b4c6c3e44bff12b7afadb69c1f945b624f57096de9ca2736.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/998894c923296cd4b4c6c3e44bff12b7afadb69c1f945b624f57096de9ca2736.jpg)

![b0a7785af46861fdf3c33d834a05d4eab2f974782aead897954987964fceff3b.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/b0a7785af46861fdf3c33d834a05d4eab2f974782aead897954987964fceff3b.jpg)

![b78261f40d68b7ee29b8558513d4d657a5e02d3f30a37f874581c9cc2cbe0d39.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/b78261f40d68b7ee29b8558513d4d657a5e02d3f30a37f874581c9cc2cbe0d39.jpg)

![c537e2a479cca1ec362c0add08f9aa35da93d0bcd91540aaac49129c6d5a21f0.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/c537e2a479cca1ec362c0add08f9aa35da93d0bcd91540aaac49129c6d5a21f0.jpg)

![d8fed9699d828998ff66cbe6f2646d7fc8bceaa1dbcd19f782d6668b32cb70bf.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/d8fed9699d828998ff66cbe6f2646d7fc8bceaa1dbcd19f782d6668b32cb70bf.jpg)

![e1be43622162c3f2015e1070413023978195b41b4525a7e6601f55f8d5e9bd63.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/e1be43622162c3f2015e1070413023978195b41b4525a7e6601f55f8d5e9bd63.jpg)

![eed0d18d8c5f422d01a39d100c4c68666b6e2ccd0814b0acd8ee856d6cc92246.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/eed0d18d8c5f422d01a39d100c4c68666b6e2ccd0814b0acd8ee856d6cc92246.jpg)

![f1bf09bf5d7f69961a17e0aca44ed400702618effb66b456d78517799a62968d.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/f1bf09bf5d7f69961a17e0aca44ed400702618effb66b456d78517799a62968d.jpg)

![fa790ab419371c34d1ebe8150f4f73f39924e16e98efccdaa4bf6316af0f336c.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/images/fa790ab419371c34d1ebe8150f4f73f39924e16e98efccdaa4bf6316af0f336c.jpg)

### Tables

![dcf7f63173a1455aecaef78efcd5b2a6407cce89dc2e3771a5e1492d0b685dd6.jpg](../icml_results/136_Equivalence%20of%20Context%20and%20Parameter%20Updates%20in%20Modern%20Transformer%20Blocks/tables/dcf7f63173a1455aecaef78efcd5b2a6407cce89dc2e3771a5e1492d0b685dd6.jpg)

<a id="paper-0135"></a>

## VenusBench-Mobile: A Challenging and User-Centric Benchmark for Mobile GUI Agents with Capability Diagnostics


### Images

![05ade1dda127e64db29d6e15a0f5ad19c4ad5cbd9ae242b1b187a0726fa591f7.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/images/05ade1dda127e64db29d6e15a0f5ad19c4ad5cbd9ae242b1b187a0726fa591f7.jpg)

![1167d7161124df62f7e3beb62bd2fe13a3bf163265fc5d5a19966e3e0698b555.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/images/1167d7161124df62f7e3beb62bd2fe13a3bf163265fc5d5a19966e3e0698b555.jpg)

![1ee1b5d0134f50df9eb26ad1d77511d5e48bb00361d87791b15228afcd67dcd9.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/images/1ee1b5d0134f50df9eb26ad1d77511d5e48bb00361d87791b15228afcd67dcd9.jpg)

![270cf34c41e4afbe85ee92b1f73494acd363321fb400a2ecdc214ecae043f435.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/images/270cf34c41e4afbe85ee92b1f73494acd363321fb400a2ecdc214ecae043f435.jpg)

![5225517a5177a30fa8d22aa27b0db7e985ee00e34d0c11770836e5caa51df84a.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/images/5225517a5177a30fa8d22aa27b0db7e985ee00e34d0c11770836e5caa51df84a.jpg)

![61bd8f6930d586737b4acf3cc239284e05a5eb7870b76fdfac9c6c9d0cf6296b.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/images/61bd8f6930d586737b4acf3cc239284e05a5eb7870b76fdfac9c6c9d0cf6296b.jpg)

![7b95257f2494aa9e4c9c779c9a3a5d72d3c470c60260256a1944649071eaf1d0.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/images/7b95257f2494aa9e4c9c779c9a3a5d72d3c470c60260256a1944649071eaf1d0.jpg)

![9769a7d736a032f306f994568443cec3c5b0921890bd1e233dbe033e22a4b243.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/images/9769a7d736a032f306f994568443cec3c5b0921890bd1e233dbe033e22a4b243.jpg)

![b98706a81361c88f6c966aa902593dc3e44f60d3ee7874b2010dd882d1cad66d.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/images/b98706a81361c88f6c966aa902593dc3e44f60d3ee7874b2010dd882d1cad66d.jpg)

![fbc9d6bf4c6fbbb04de13898ee7e6fc1cf38a182b4456f3c5427d67d41f2d771.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/images/fbc9d6bf4c6fbbb04de13898ee7e6fc1cf38a182b4456f3c5427d67d41f2d771.jpg)

### Tables

![1ed39f77c13eb766a0093b62f003b31be2b0ae4a8a18e5ee4e7991d798c7ac72.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/tables/1ed39f77c13eb766a0093b62f003b31be2b0ae4a8a18e5ee4e7991d798c7ac72.jpg)

![466dc733f40b1c84cf320745f4836843ba5c80ca38ecad19b804cf972b4dd468.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/tables/466dc733f40b1c84cf320745f4836843ba5c80ca38ecad19b804cf972b4dd468.jpg)

![64762de774a8bc6e0b64370632a2d887544ddedb54dace0010b1c67f82462812.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/tables/64762de774a8bc6e0b64370632a2d887544ddedb54dace0010b1c67f82462812.jpg)

![6f51350a7324cce4a7214f252dc8f1276b38940fddd51272b2f6ac3657bd8aee.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/tables/6f51350a7324cce4a7214f252dc8f1276b38940fddd51272b2f6ac3657bd8aee.jpg)

![77d79244a4d9aa4ebc72b5367171971599d8c424c3e76769685761638bc533af.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/tables/77d79244a4d9aa4ebc72b5367171971599d8c424c3e76769685761638bc533af.jpg)

![8514d9430494d7d5effd1758e7762a4e45d45d596d9699f4a3f7d802dd58c338.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/tables/8514d9430494d7d5effd1758e7762a4e45d45d596d9699f4a3f7d802dd58c338.jpg)

![9b8dc0d51502157c702c13e7f3fc2782c37d365ee98c5ee6c6a07d92fc0d14a4.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/tables/9b8dc0d51502157c702c13e7f3fc2782c37d365ee98c5ee6c6a07d92fc0d14a4.jpg)

![f88a1dd8a645ced7b00c56b2c162d0969d576eb0a8abdbba879c66522ac1e914.jpg](../icml_results/137_VenusBench-Mobile_%20A%20Challenging%20and%20User-Centric%20Benchmark%20for%20Mobile%20GUI%20Agents%20with%20Capability%20Di/tables/f88a1dd8a645ced7b00c56b2c162d0969d576eb0a8abdbba879c66522ac1e914.jpg)

<a id="paper-0136"></a>

## FlashOptim: Optimizers for Memory-Efficient Training


### Images

![2225b3028bad0a5daa3dc438aa00ffde945311064256b4275296ddad5a671bfe.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/images/2225b3028bad0a5daa3dc438aa00ffde945311064256b4275296ddad5a671bfe.jpg)

![6f355f0f839a4fa6c5ecb8eedce8e90fca421756737a80230ce4fa9f4572e0ee.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/images/6f355f0f839a4fa6c5ecb8eedce8e90fca421756737a80230ce4fa9f4572e0ee.jpg)

![804abe9a1162e4056d3019511751895cf9f19f5e4c9f4d78421c872fdc2d35dc.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/images/804abe9a1162e4056d3019511751895cf9f19f5e4c9f4d78421c872fdc2d35dc.jpg)

![862aae6b903c043b466c360fbcbb045c79cf9a8adbdb64ea55c4d61486fe092c.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/images/862aae6b903c043b466c360fbcbb045c79cf9a8adbdb64ea55c4d61486fe092c.jpg)

![90b28f84974ca2922c790db9ba5f1d8ab33150f045cbb4584ef6f67d66110f40.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/images/90b28f84974ca2922c790db9ba5f1d8ab33150f045cbb4584ef6f67d66110f40.jpg)

![9b7f7bbc4ebba7d3296c2ceb434461d02edb722e15cf8eb187913d4dd4891f63.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/images/9b7f7bbc4ebba7d3296c2ceb434461d02edb722e15cf8eb187913d4dd4891f63.jpg)

![ab0128b58d0992a8995a01e9e72c4cfa7a2733399c9a227c136dd14c8b228b69.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/images/ab0128b58d0992a8995a01e9e72c4cfa7a2733399c9a227c136dd14c8b228b69.jpg)

![b0d697c4bcae3c1c4a854cdf776855cb4b63fa35445c0f135635cc7735de9145.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/images/b0d697c4bcae3c1c4a854cdf776855cb4b63fa35445c0f135635cc7735de9145.jpg)

![e660e9944fc74d3f4b47d4747a99ec8cc3e446426ae84f8306ca8b70adc2e0f7.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/images/e660e9944fc74d3f4b47d4747a99ec8cc3e446426ae84f8306ca8b70adc2e0f7.jpg)

![e73638789ac364f71a79e861f647e4b9ccc86e9a355685ccf79a9f752b5b2569.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/images/e73638789ac364f71a79e861f647e4b9ccc86e9a355685ccf79a9f752b5b2569.jpg)

### Tables

![18d406271d74daebdcce3fef545e1df3da6a8e94d56217d036bafa4cdfa38a1c.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/tables/18d406271d74daebdcce3fef545e1df3da6a8e94d56217d036bafa4cdfa38a1c.jpg)

![20af7174703b09957bfc66e9acc7d78eb702e6698075b2b358074f6073cd44e7.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/tables/20af7174703b09957bfc66e9acc7d78eb702e6698075b2b358074f6073cd44e7.jpg)

![230b5de3e21fc2239fa8c042aa1627b942fae66f01138a36510a2d3d99d4e0fd.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/tables/230b5de3e21fc2239fa8c042aa1627b942fae66f01138a36510a2d3d99d4e0fd.jpg)

![4f6c6a549c629c68abab81880b3fb00900c5092a68ff4e69875913b1f656d267.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/tables/4f6c6a549c629c68abab81880b3fb00900c5092a68ff4e69875913b1f656d267.jpg)

![62a4ab3f3db59c80dc942e317ebfac941267ea02c79cbfe58f845d533b207585.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/tables/62a4ab3f3db59c80dc942e317ebfac941267ea02c79cbfe58f845d533b207585.jpg)

![7750e7e0f07c92bbae031b1e275514fabbfb750d6afef3dfbbef897edfd444c4.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/tables/7750e7e0f07c92bbae031b1e275514fabbfb750d6afef3dfbbef897edfd444c4.jpg)

![8ed8a8090f39eaf348169a0e6a2c8b66a626b8d96ceb8108acaf40672083c899.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/tables/8ed8a8090f39eaf348169a0e6a2c8b66a626b8d96ceb8108acaf40672083c899.jpg)

![a2792353878ec87f920c6eca45483c0a0e9ab8ece80a91afb06fa538ff331d25.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/tables/a2792353878ec87f920c6eca45483c0a0e9ab8ece80a91afb06fa538ff331d25.jpg)

![b330ef9452e62b65981822a41564da7900fcf9adb1aafd544a11067b29134e3b.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/tables/b330ef9452e62b65981822a41564da7900fcf9adb1aafd544a11067b29134e3b.jpg)

![b4c4d3a0b4e423b8c86c4e35f7ccb4df17ef4602976b713578eeddecb92fe00c.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/tables/b4c4d3a0b4e423b8c86c4e35f7ccb4df17ef4602976b713578eeddecb92fe00c.jpg)

![b7b46bda6b78ca1a66869c11b5df2a203a911d3343b876d49231eda6b3e78ac8.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/tables/b7b46bda6b78ca1a66869c11b5df2a203a911d3343b876d49231eda6b3e78ac8.jpg)

![c15ad9f4d096dbb36e2e7bacd545aba70b6b78320e0e7da940e2023d8f87c173.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/tables/c15ad9f4d096dbb36e2e7bacd545aba70b6b78320e0e7da940e2023d8f87c173.jpg)

![e47937256207d61d7c78ea09a35df302c6f1a5e14e8ce5137e7582442fdc7409.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/tables/e47937256207d61d7c78ea09a35df302c6f1a5e14e8ce5137e7582442fdc7409.jpg)

![f5d733066011a4f6c6310d57828a2948ae5f7396a9f11df7c1a8acaeff680b28.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/tables/f5d733066011a4f6c6310d57828a2948ae5f7396a9f11df7c1a8acaeff680b28.jpg)

![f8007be5c84425fa5ab813b2303b937eef49a0876ba8344aa3f76e8eff984bd8.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/tables/f8007be5c84425fa5ab813b2303b937eef49a0876ba8344aa3f76e8eff984bd8.jpg)

![f9d672666f4fa6a964f823e679a97fdb23b8df131ff0e36ebf5a276271cd1e53.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/tables/f9d672666f4fa6a964f823e679a97fdb23b8df131ff0e36ebf5a276271cd1e53.jpg)

![fed02ea0d121bd3bdac8805daa61665005c9e68ca899b11861f2f84ebbb20b24.jpg](../icml_results/138_FlashOptim_%20Optimizers%20for%20Memory-Efficient%20Training/tables/fed02ea0d121bd3bdac8805daa61665005c9e68ca899b11861f2f84ebbb20b24.jpg)

<a id="paper-0137"></a>

## Decoupling The "What" and "Where" With Polar Coordinate Positional Embedding


### Images

![03ef3a04917709d2519169a381886ea4cd790ce853b12fe8bf7315046ee86291.jpg](../icml_results/139_Decoupling%20The%20_What_%20and%20_Where_%20With%20Polar%20Coordinate%20Positional%20Embedding/images/03ef3a04917709d2519169a381886ea4cd790ce853b12fe8bf7315046ee86291.jpg)

![104544107bcb2d4e1a136633153bc27409a08d4f2b8e149815c3367a749dea38.jpg](../icml_results/139_Decoupling%20The%20_What_%20and%20_Where_%20With%20Polar%20Coordinate%20Positional%20Embedding/images/104544107bcb2d4e1a136633153bc27409a08d4f2b8e149815c3367a749dea38.jpg)

![12e6da17a85d49238348788765b9779dff2ca4b6fe2a2b28f5259eadbe3dcca5.jpg](../icml_results/139_Decoupling%20The%20_What_%20and%20_Where_%20With%20Polar%20Coordinate%20Positional%20Embedding/images/12e6da17a85d49238348788765b9779dff2ca4b6fe2a2b28f5259eadbe3dcca5.jpg)

![658e7e8c73c126c9f2ecb01f354ae3827ee005af6f627701d9112a3a5aedef31.jpg](../icml_results/139_Decoupling%20The%20_What_%20and%20_Where_%20With%20Polar%20Coordinate%20Positional%20Embedding/images/658e7e8c73c126c9f2ecb01f354ae3827ee005af6f627701d9112a3a5aedef31.jpg)

![9614935a71828ad876d16ed4e83a3432f8b6d28151fc3574d79402ee19874cb3.jpg](../icml_results/139_Decoupling%20The%20_What_%20and%20_Where_%20With%20Polar%20Coordinate%20Positional%20Embedding/images/9614935a71828ad876d16ed4e83a3432f8b6d28151fc3574d79402ee19874cb3.jpg)

![a48c37ee52dee62dba8d0b260634bd0a4d353edf8b601a1ff50c705f61d82f4f.jpg](../icml_results/139_Decoupling%20The%20_What_%20and%20_Where_%20With%20Polar%20Coordinate%20Positional%20Embedding/images/a48c37ee52dee62dba8d0b260634bd0a4d353edf8b601a1ff50c705f61d82f4f.jpg)

![e020a81f2966a99c3cd9667f9176d5e9b738f8139600601ed9874497d3d97baf.jpg](../icml_results/139_Decoupling%20The%20_What_%20and%20_Where_%20With%20Polar%20Coordinate%20Positional%20Embedding/images/e020a81f2966a99c3cd9667f9176d5e9b738f8139600601ed9874497d3d97baf.jpg)

![ec62305114b5c28e3872f38fbf98a8d42d2f762fe16896897e92f87c8ea75962.jpg](../icml_results/139_Decoupling%20The%20_What_%20and%20_Where_%20With%20Polar%20Coordinate%20Positional%20Embedding/images/ec62305114b5c28e3872f38fbf98a8d42d2f762fe16896897e92f87c8ea75962.jpg)

### Tables

![0cfb4af07d434080fafa1e0bcba430a6f946e884ec2ff8be090dd4c3097954a1.jpg](../icml_results/139_Decoupling%20The%20_What_%20and%20_Where_%20With%20Polar%20Coordinate%20Positional%20Embedding/tables/0cfb4af07d434080fafa1e0bcba430a6f946e884ec2ff8be090dd4c3097954a1.jpg)

![123e6bd657e9355ba9fb22d2d0f9dc7cd954d2cb8014c2608cbcccc9c18cb574.jpg](../icml_results/139_Decoupling%20The%20_What_%20and%20_Where_%20With%20Polar%20Coordinate%20Positional%20Embedding/tables/123e6bd657e9355ba9fb22d2d0f9dc7cd954d2cb8014c2608cbcccc9c18cb574.jpg)

![36d7fb371a43d9d917c3492a3854ded46bef53072ad207b0fbb579d2880d43f6.jpg](../icml_results/139_Decoupling%20The%20_What_%20and%20_Where_%20With%20Polar%20Coordinate%20Positional%20Embedding/tables/36d7fb371a43d9d917c3492a3854ded46bef53072ad207b0fbb579d2880d43f6.jpg)

![afca299b42f764271138f35bcb2d16fae39f4b0c582e3ac41165348aeaca9f01.jpg](../icml_results/139_Decoupling%20The%20_What_%20and%20_Where_%20With%20Polar%20Coordinate%20Positional%20Embedding/tables/afca299b42f764271138f35bcb2d16fae39f4b0c582e3ac41165348aeaca9f01.jpg)

![b8dfc3e5f76fd512f24426115c06ba8a3f29bb2e269edaeab92fd87e943346ac.jpg](../icml_results/139_Decoupling%20The%20_What_%20and%20_Where_%20With%20Polar%20Coordinate%20Positional%20Embedding/tables/b8dfc3e5f76fd512f24426115c06ba8a3f29bb2e269edaeab92fd87e943346ac.jpg)

![bea50e2e371d88bd69aeeeb91a5aab8930ac16eed59ad5ad310cebc18b47ac2a.jpg](../icml_results/139_Decoupling%20The%20_What_%20and%20_Where_%20With%20Polar%20Coordinate%20Positional%20Embedding/tables/bea50e2e371d88bd69aeeeb91a5aab8930ac16eed59ad5ad310cebc18b47ac2a.jpg)

![c6f140a2d34fbdd83d7b0c5d7fd17d13b8eb974a60436d244361128addbf7ece.jpg](../icml_results/139_Decoupling%20The%20_What_%20and%20_Where_%20With%20Polar%20Coordinate%20Positional%20Embedding/tables/c6f140a2d34fbdd83d7b0c5d7fd17d13b8eb974a60436d244361128addbf7ece.jpg)

![cfeb78dd8153fae19223cac21188487aac98b7b4c9c16d282526b81e0b128184.jpg](../icml_results/139_Decoupling%20The%20_What_%20and%20_Where_%20With%20Polar%20Coordinate%20Positional%20Embedding/tables/cfeb78dd8153fae19223cac21188487aac98b7b4c9c16d282526b81e0b128184.jpg)

![f6ddce7cf78850d1f860e0495cb50930c25b6d0aa537155be722fa48ab2030da.jpg](../icml_results/139_Decoupling%20The%20_What_%20and%20_Where_%20With%20Polar%20Coordinate%20Positional%20Embedding/tables/f6ddce7cf78850d1f860e0495cb50930c25b6d0aa537155be722fa48ab2030da.jpg)

<a id="paper-0138"></a>

## GEM: Geometric Erasure by Contrastive Velocity Matching in Rectified Flows


### Images

![19928ddee23ce3dd06e54ce0685191efbe83362f9c44ac976083dd44edfafc76.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/images/19928ddee23ce3dd06e54ce0685191efbe83362f9c44ac976083dd44edfafc76.jpg)

![349e9001f5af3f32235c97a30d7189c7ceb6541d206c86812b3afc24dc18ad08.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/images/349e9001f5af3f32235c97a30d7189c7ceb6541d206c86812b3afc24dc18ad08.jpg)

![381a848b11f8aa871bb5d2e3e559d1e3134b2a4dbeb7cd436b56c856c6330f8e.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/images/381a848b11f8aa871bb5d2e3e559d1e3134b2a4dbeb7cd436b56c856c6330f8e.jpg)

![404d97b8ab73719e8298fe7d37ae3d47fb4a7e4d0f460f44807ba5b2fa4097e0.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/images/404d97b8ab73719e8298fe7d37ae3d47fb4a7e4d0f460f44807ba5b2fa4097e0.jpg)

![7dd6580a372c0e2308fcf46257bf26f0ada8b48e9f78eabfaf53558ee840363f.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/images/7dd6580a372c0e2308fcf46257bf26f0ada8b48e9f78eabfaf53558ee840363f.jpg)

![835200e9aa19b391accedd208705e5ed76903c629dbc5e3f11c9fc015ed3e867.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/images/835200e9aa19b391accedd208705e5ed76903c629dbc5e3f11c9fc015ed3e867.jpg)

![a1960e3793f16568e3c6ba9d20a706c8118b4ae63d73ad79c128b6a55fa6bd87.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/images/a1960e3793f16568e3c6ba9d20a706c8118b4ae63d73ad79c128b6a55fa6bd87.jpg)

![b7df1ac370a8b349e64647b883c8a2e7e812bfa244ea63968661404628a25550.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/images/b7df1ac370a8b349e64647b883c8a2e7e812bfa244ea63968661404628a25550.jpg)

### Tables

![24ee544db9ec89075b92f99b57c4fb92bf3ffe40822a80adc3c52686d525656f.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/tables/24ee544db9ec89075b92f99b57c4fb92bf3ffe40822a80adc3c52686d525656f.jpg)

![381b7295b02090f4006a2edf7a50f9bf3a13df4aecd13165e1c2a213c1939d3b.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/tables/381b7295b02090f4006a2edf7a50f9bf3a13df4aecd13165e1c2a213c1939d3b.jpg)

![3d6c5bde4483cf6ba27dec5fc459dc0791931a503d7e48513d1c05843ef74c0c.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/tables/3d6c5bde4483cf6ba27dec5fc459dc0791931a503d7e48513d1c05843ef74c0c.jpg)

![5c63c0d460df48d8593ed76a758a179fee0797e46bd7b78bdd6e7332e9cf969d.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/tables/5c63c0d460df48d8593ed76a758a179fee0797e46bd7b78bdd6e7332e9cf969d.jpg)

![62a93845e2036955d9eb6df497547b915e1238b0467c0233ecbbe611399ea703.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/tables/62a93845e2036955d9eb6df497547b915e1238b0467c0233ecbbe611399ea703.jpg)

![73d3c9a7f4043c95a82c17ccb7291c8627d49ad6bfc35fbac4abfab505080747.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/tables/73d3c9a7f4043c95a82c17ccb7291c8627d49ad6bfc35fbac4abfab505080747.jpg)

![a154f784fefde60f738d75c558864ae6ff42ae9eb098d76243b13611ab51dd22.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/tables/a154f784fefde60f738d75c558864ae6ff42ae9eb098d76243b13611ab51dd22.jpg)

![d189937517d9914c39f431c3fede665e07398d29de95147ffb8d2a45423e139b.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/tables/d189937517d9914c39f431c3fede665e07398d29de95147ffb8d2a45423e139b.jpg)

![e16e5341a6b85dd6c7761c1e7e0748aaace18fb1618e44f63bd99bc13308d90d.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/tables/e16e5341a6b85dd6c7761c1e7e0748aaace18fb1618e44f63bd99bc13308d90d.jpg)

![ef42e768b659c6c16b2d4488f9e08a05706136780b0c98660639ef6dfc7bade0.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/tables/ef42e768b659c6c16b2d4488f9e08a05706136780b0c98660639ef6dfc7bade0.jpg)

![fea764968b05558b6f3bcd8be26ebf651ec7b40ee56646ddfceb483bba5ba390.jpg](../icml_results/140_GEM_%20Geometric%20Erasure%20by%20Contrastive%20Velocity%20Matching%20in%20Rectified%20Flows/tables/fea764968b05558b6f3bcd8be26ebf651ec7b40ee56646ddfceb483bba5ba390.jpg)

<a id="paper-0139"></a>

## What Do Agents Learn from Trajectory-SFT: Semantics or Interfaces?


### Images

![0eaaa145b197524bbf909fc6828b7964328bd26910798213e3c138513b0d9bbf.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/images/0eaaa145b197524bbf909fc6828b7964328bd26910798213e3c138513b0d9bbf.jpg)

![2799c2e9c26af01175656fcc2d58aba5a2086e74d0625b9d5c917818d9034639.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/images/2799c2e9c26af01175656fcc2d58aba5a2086e74d0625b9d5c917818d9034639.jpg)

![4d649fb453adc3ae8a4418ca14a59f40f3d2ccda0a588d75e26a019b5018fa77.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/images/4d649fb453adc3ae8a4418ca14a59f40f3d2ccda0a588d75e26a019b5018fa77.jpg)

![9e3bd5e909cfc83865135dfa54bab0bd934877b7619a49bfde2872805f2025d7.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/images/9e3bd5e909cfc83865135dfa54bab0bd934877b7619a49bfde2872805f2025d7.jpg)

![abc2704cef914ddc4e24bbc4c7f601edf246b08042c671be5702582bcbd6be75.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/images/abc2704cef914ddc4e24bbc4c7f601edf246b08042c671be5702582bcbd6be75.jpg)

### Tables

![05bb224955d5527324094ea676dfb1d5f33984c8554619bcbc37ab4d2bf19afe.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/05bb224955d5527324094ea676dfb1d5f33984c8554619bcbc37ab4d2bf19afe.jpg)

![0789f13e357b303b3313060e3347d5293f6a78d504de5912a48ad63c74c0933a.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/0789f13e357b303b3313060e3347d5293f6a78d504de5912a48ad63c74c0933a.jpg)

![0cc8ca5ff64d4fa9d86eeba3da2ea4ef7f588b28be6898474ce5220265d146ca.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/0cc8ca5ff64d4fa9d86eeba3da2ea4ef7f588b28be6898474ce5220265d146ca.jpg)

![10c4f9ba3d0054bc0280c9cd919e8475ea02986cc8abfdfaf4f548c10d908d4e.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/10c4f9ba3d0054bc0280c9cd919e8475ea02986cc8abfdfaf4f548c10d908d4e.jpg)

![113bbde940869b88d34c4998dcaaf034669e8790f437b797297ff3c7ffed5490.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/113bbde940869b88d34c4998dcaaf034669e8790f437b797297ff3c7ffed5490.jpg)

![117235a21f648a372334e65e55aa859d636bf1b49cd9151f9e4f348f9fb11700.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/117235a21f648a372334e65e55aa859d636bf1b49cd9151f9e4f348f9fb11700.jpg)

![19b81d42236e76f8a1024e76a00ebe14b5e902e7b769fe5faa0c1712af38263d.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/19b81d42236e76f8a1024e76a00ebe14b5e902e7b769fe5faa0c1712af38263d.jpg)

![2080096e1492c5961e0c87dfd5393a90ee480f0b7d0d27ab9884250ab4d11631.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/2080096e1492c5961e0c87dfd5393a90ee480f0b7d0d27ab9884250ab4d11631.jpg)

![2707535d6e7d09f858b189c752c2fcfe32ea5f46939d535217a65607aa302397.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/2707535d6e7d09f858b189c752c2fcfe32ea5f46939d535217a65607aa302397.jpg)

![271b8c76d046cea7bf16820d51695252b86a222652e3010b56b010e35b80dfc5.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/271b8c76d046cea7bf16820d51695252b86a222652e3010b56b010e35b80dfc5.jpg)

![32890a3b0c34a534cb7c2b960e6be064d6f31eb19f1905443dec0d2114fcd2c7.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/32890a3b0c34a534cb7c2b960e6be064d6f31eb19f1905443dec0d2114fcd2c7.jpg)

![36759b0a3cf776fbb24a92c52c2bbb6ad9815f3a937ea003bdddac198547b939.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/36759b0a3cf776fbb24a92c52c2bbb6ad9815f3a937ea003bdddac198547b939.jpg)

![3af3d9472c03b2f67078d8c80c521b66251ea85a068a626bbce2944b2e7eb0fc.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/3af3d9472c03b2f67078d8c80c521b66251ea85a068a626bbce2944b2e7eb0fc.jpg)

![3ca80bed795522b83a52dc394bf8ce22ab4086c2e6c011b65b332fde9abe24b7.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/3ca80bed795522b83a52dc394bf8ce22ab4086c2e6c011b65b332fde9abe24b7.jpg)

![4007b20607e0fe9fe770fd4f23f2cfa26e8bd64fc178104c1b534ba11c49a819.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/4007b20607e0fe9fe770fd4f23f2cfa26e8bd64fc178104c1b534ba11c49a819.jpg)

![41a0deb66b5987c4ab7ac12ce2fa369685312959bfc1f6acac7e8606096921a3.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/41a0deb66b5987c4ab7ac12ce2fa369685312959bfc1f6acac7e8606096921a3.jpg)

![4570a04b9cf9ea7b68410d013c28cc83e332a22e5ac91447e4c6c9c33c936809.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/4570a04b9cf9ea7b68410d013c28cc83e332a22e5ac91447e4c6c9c33c936809.jpg)

![55b1ffc8116922a7ccbaadc3760475d60fd08680504db9d0e0d933e32461ce8b.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/55b1ffc8116922a7ccbaadc3760475d60fd08680504db9d0e0d933e32461ce8b.jpg)

![57040261bb9ed97bb6a4ce64fd637b7c93a0ddc3d4549ca3791dc5edd931191f.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/57040261bb9ed97bb6a4ce64fd637b7c93a0ddc3d4549ca3791dc5edd931191f.jpg)

![5b88f2daa0d89edd8de1e386ea4cf13e5879b56f20c3b326ea1e93d7b46e843c.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/5b88f2daa0d89edd8de1e386ea4cf13e5879b56f20c3b326ea1e93d7b46e843c.jpg)

![5db21eb0d39baec14f417d724b8c4aa47ff45c7cc220fd484e6d03cd7bde1130.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/5db21eb0d39baec14f417d724b8c4aa47ff45c7cc220fd484e6d03cd7bde1130.jpg)

![5fb30ac42b09ac67bcc1a8ec07b14b1c39d7cd4e9127806f73da938f3bc2338f.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/5fb30ac42b09ac67bcc1a8ec07b14b1c39d7cd4e9127806f73da938f3bc2338f.jpg)

![675ce61f5a46ad64dbb37e50bf578fabfb0725be49da57318abefa20c3d28041.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/675ce61f5a46ad64dbb37e50bf578fabfb0725be49da57318abefa20c3d28041.jpg)

![75dac77047dc496f13384d40653c0ef697607c93959fb28ae266f1e8469f1506.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/75dac77047dc496f13384d40653c0ef697607c93959fb28ae266f1e8469f1506.jpg)

![8187ed780c7efe5bf5a948f076340265f16dcb00dbd5265dae768520a7d4fcb2.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/8187ed780c7efe5bf5a948f076340265f16dcb00dbd5265dae768520a7d4fcb2.jpg)

![81ce44f1ba19d6a0800495ba621fd1bba077ee39b1dc0e1387e6dc258a34bcb5.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/81ce44f1ba19d6a0800495ba621fd1bba077ee39b1dc0e1387e6dc258a34bcb5.jpg)

![8692fe23ab2dd6d12cc5d0826e1768d8508cd9ca9462b9cb01a11e1f15b1e0b7.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/8692fe23ab2dd6d12cc5d0826e1768d8508cd9ca9462b9cb01a11e1f15b1e0b7.jpg)

![8ee08a470b16127d2abd0e9766cc4ef9d9407bff230d2571ac64a1ff300461d2.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/8ee08a470b16127d2abd0e9766cc4ef9d9407bff230d2571ac64a1ff300461d2.jpg)

![8ef49dc446b825bb09e50e89e87527cb542b8657878c05406bb6f7415b438b7e.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/8ef49dc446b825bb09e50e89e87527cb542b8657878c05406bb6f7415b438b7e.jpg)

![923a0ca7fd35fddbfd0096182fc713757c22d53e299406018fa83acee9c97c9a.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/923a0ca7fd35fddbfd0096182fc713757c22d53e299406018fa83acee9c97c9a.jpg)

![9719cc331c1cb8b1b19e7cd3ceae52a059a2e3fee3bc24a378b6a77851543422.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/9719cc331c1cb8b1b19e7cd3ceae52a059a2e3fee3bc24a378b6a77851543422.jpg)

![98527b5e21c309f248ea6823aab1e7fc8e196f527c7b8f4649bd3ac29ca9d19f.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/98527b5e21c309f248ea6823aab1e7fc8e196f527c7b8f4649bd3ac29ca9d19f.jpg)

![aae959d79fc05796a7f36e8b0e4e1fd1d0229b7ff8855cdc93d00804e8eeaada.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/aae959d79fc05796a7f36e8b0e4e1fd1d0229b7ff8855cdc93d00804e8eeaada.jpg)

![ae8ce915ea8809f4e900eab3f1e61b810715a915350063bf543766df76fe33aa.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/ae8ce915ea8809f4e900eab3f1e61b810715a915350063bf543766df76fe33aa.jpg)

![b61df3217b9b3870cb480eae888cc9b880d0d4d89d728b3e15077863ceab0cdb.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/b61df3217b9b3870cb480eae888cc9b880d0d4d89d728b3e15077863ceab0cdb.jpg)

![bf89bf7da1d0bc3d7077076ecb3a851ca712c89d650add05df878f4d42da6606.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/bf89bf7da1d0bc3d7077076ecb3a851ca712c89d650add05df878f4d42da6606.jpg)

![cf1f2ead5ef8289ebdf3f70827254605d46fedaa6faa0234270e3106f9eedff5.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/cf1f2ead5ef8289ebdf3f70827254605d46fedaa6faa0234270e3106f9eedff5.jpg)

![cf70a2cdeda965ec5eda52a0fc08ef8ddaf16187751f8b60b1e6472994c4022d.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/cf70a2cdeda965ec5eda52a0fc08ef8ddaf16187751f8b60b1e6472994c4022d.jpg)

![dd295132907959f9fb35b4c1bbfa70d32d4b90689a6c6869c169de1dd12d216f.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/dd295132907959f9fb35b4c1bbfa70d32d4b90689a6c6869c169de1dd12d216f.jpg)

![df6681edf2746d1faa04d06a7ccfd149f6675ec72de3628284a5390b4f176d98.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/df6681edf2746d1faa04d06a7ccfd149f6675ec72de3628284a5390b4f176d98.jpg)

![e97f8081bf3667183a4a81170987a1bd499b1bff5f7009526f49c1caddd1e5de.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/e97f8081bf3667183a4a81170987a1bd499b1bff5f7009526f49c1caddd1e5de.jpg)

![eabdb8c94fbccbe498685aa9decba6871c73b23e1995c6c84e5db4bab4258ba3.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/eabdb8c94fbccbe498685aa9decba6871c73b23e1995c6c84e5db4bab4258ba3.jpg)

![ecb61bf86450dfb63158353afca0253ae801ac943b20b7c2a58671c841c3c715.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/ecb61bf86450dfb63158353afca0253ae801ac943b20b7c2a58671c841c3c715.jpg)

![f1d7e84a91567a01c8c642c6af02cb9a47a5b3ff1c84c633cffde0021709c5dd.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/f1d7e84a91567a01c8c642c6af02cb9a47a5b3ff1c84c633cffde0021709c5dd.jpg)

![f2ac1afbf50e2b1b69431c327628a933c045d80f5d278728862622febdcf5fde.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/f2ac1afbf50e2b1b69431c327628a933c045d80f5d278728862622febdcf5fde.jpg)

![f4bf7b6751be9672a67cee9bfd872cb717b8497a8e0706dcccc7f7b3e2cbf367.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/f4bf7b6751be9672a67cee9bfd872cb717b8497a8e0706dcccc7f7b3e2cbf367.jpg)

![faaf67d9671fb8f552d1c0c9702598dada106b6f8e29cdb431fb9899a5fc7840.jpg](../icml_results/141_What%20Do%20Agents%20Learn%20from%20Trajectory-SFT_%20Semantics%20or%20Interfaces_/tables/faaf67d9671fb8f552d1c0c9702598dada106b6f8e29cdb431fb9899a5fc7840.jpg)

<a id="paper-0140"></a>

## Detecting the Semantic Fixed Point: A Geometric Framework for Efficient Inference


### Images

![2a8bff3a9fb9e7688c19b3098b7b696ec8afd7f965ff2e94f1a789ad9f17d27a.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/images/2a8bff3a9fb9e7688c19b3098b7b696ec8afd7f965ff2e94f1a789ad9f17d27a.jpg)

![44b6f954a30295a3ad1d6daffd199a9f467a2870f42988ca0ff702ccb5870f84.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/images/44b6f954a30295a3ad1d6daffd199a9f467a2870f42988ca0ff702ccb5870f84.jpg)

![45de189333185c516a606dcf951b65452713d43347262a4c2efeb3e42799e8af.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/images/45de189333185c516a606dcf951b65452713d43347262a4c2efeb3e42799e8af.jpg)

![4a3551243242100f8ea704076d35ae258e91c4e531c82452491a3af7012e4216.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/images/4a3551243242100f8ea704076d35ae258e91c4e531c82452491a3af7012e4216.jpg)

![538c59a66fd7c3eaf8a6856462caeba088cb2d39869fa1e56cab505b10dbd799.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/images/538c59a66fd7c3eaf8a6856462caeba088cb2d39869fa1e56cab505b10dbd799.jpg)

![7156e4ba98ef081edaca8f6f5565d790af653db0d7fd0045d18434bc7e1cfce4.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/images/7156e4ba98ef081edaca8f6f5565d790af653db0d7fd0045d18434bc7e1cfce4.jpg)

![81538f3c73135e778d85421653aac36d3dd3c12e405766b411aba3303ce87f4b.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/images/81538f3c73135e778d85421653aac36d3dd3c12e405766b411aba3303ce87f4b.jpg)

![be5ce3fb433883baa02daafc54d8dcab461f063fe5ad4840850d4d85db066dd3.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/images/be5ce3fb433883baa02daafc54d8dcab461f063fe5ad4840850d4d85db066dd3.jpg)

![dc4d64809f571353bedeb48145ec79ca3ded18f63574e55785ef30a75d7770cf.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/images/dc4d64809f571353bedeb48145ec79ca3ded18f63574e55785ef30a75d7770cf.jpg)

![dcd999e00a301cb856b14200bae98c17c41b0e4b496a07f69a648c24b13c713c.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/images/dcd999e00a301cb856b14200bae98c17c41b0e4b496a07f69a648c24b13c713c.jpg)

![e1fb5f4fead630753e3f7380d39054ae809a04de6b7f34d011801f1d135f0db6.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/images/e1fb5f4fead630753e3f7380d39054ae809a04de6b7f34d011801f1d135f0db6.jpg)

### Tables

![05a71503cc1733bf147d3c1c8aefc6f897af3a57f7614240699b4edb36e597fe.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/05a71503cc1733bf147d3c1c8aefc6f897af3a57f7614240699b4edb36e597fe.jpg)

![08bcc107125d32d167cde18d3cc91b2b5f16b32fcdb8d199f4800905443ff736.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/08bcc107125d32d167cde18d3cc91b2b5f16b32fcdb8d199f4800905443ff736.jpg)

![1089f07d4298da945f5ea794c95e21ec4c1c90498117b027e778f369edd30727.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/1089f07d4298da945f5ea794c95e21ec4c1c90498117b027e778f369edd30727.jpg)

![17b107d2e4bbf8ff0aaef4621a2efe6d78197ef1a774a750a4a67656f8eb6197.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/17b107d2e4bbf8ff0aaef4621a2efe6d78197ef1a774a750a4a67656f8eb6197.jpg)

![20f0704f27822667b05caba789e70bdd824529ca25476c9d7322be12f0631baf.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/20f0704f27822667b05caba789e70bdd824529ca25476c9d7322be12f0631baf.jpg)

![2689b0bf2fba392c55e392e82a83899d7b528ca85dcf99b66dc8d379f8f1a37e.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/2689b0bf2fba392c55e392e82a83899d7b528ca85dcf99b66dc8d379f8f1a37e.jpg)

![31a444df98405a97da79fcef9a9825e340ac41aaa1ffcaddfa8357b288d39177.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/31a444df98405a97da79fcef9a9825e340ac41aaa1ffcaddfa8357b288d39177.jpg)

![32f713e049479134074ad80420a9ff6fc07b11221a8595a3b467eceee118a8af.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/32f713e049479134074ad80420a9ff6fc07b11221a8595a3b467eceee118a8af.jpg)

![3bc01e55f36df7344a673452838aac0e52b4805c1534ce012385423175fe10a7.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/3bc01e55f36df7344a673452838aac0e52b4805c1534ce012385423175fe10a7.jpg)

![3e92303839e4cd7f5ac2232a9f36708246f781a21c91a19ffbd4685eb674b8a8.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/3e92303839e4cd7f5ac2232a9f36708246f781a21c91a19ffbd4685eb674b8a8.jpg)

![5599307b5928c0d7413f04e76921b199d1778c988ee28bca3bcd39393cce2dad.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/5599307b5928c0d7413f04e76921b199d1778c988ee28bca3bcd39393cce2dad.jpg)

![7a82f282d953b99944353a5e38445539c738211bd21cf9760e035fc37e13bfd0.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/7a82f282d953b99944353a5e38445539c738211bd21cf9760e035fc37e13bfd0.jpg)

![95ccafbfd73d007a79a4e7a605fd62e3cf52e36a27f650a516a5d985dbd55f68.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/95ccafbfd73d007a79a4e7a605fd62e3cf52e36a27f650a516a5d985dbd55f68.jpg)

![b668feb2fc8b1e27f4d665842ae380529ca919a14f5e8dd425178ee126bc949d.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/b668feb2fc8b1e27f4d665842ae380529ca919a14f5e8dd425178ee126bc949d.jpg)

![c82340bd5f141893828eab70b652c37a7a69372c4d9a93dfb7936bf7b6016072.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/c82340bd5f141893828eab70b652c37a7a69372c4d9a93dfb7936bf7b6016072.jpg)

![cc26c78f364a5f3fe00b5c4dc3bbe0058c3e63f56ca47b4f9272a8d6aac06232.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/cc26c78f364a5f3fe00b5c4dc3bbe0058c3e63f56ca47b4f9272a8d6aac06232.jpg)

![e3a39738997a69e484995124bca890d2a69edae42d4090da4f33821ee073d758.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/e3a39738997a69e484995124bca890d2a69edae42d4090da4f33821ee073d758.jpg)

![ec983629095926567bee091342d2c8fa0e6fd81717ab8573986882f59f568c01.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/ec983629095926567bee091342d2c8fa0e6fd81717ab8573986882f59f568c01.jpg)

![f79d465dcf827a3ff7f6788d75e03a67dd348bf4c50bc0257eca2e3c0fedf858.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/f79d465dcf827a3ff7f6788d75e03a67dd348bf4c50bc0257eca2e3c0fedf858.jpg)

![f7f455ada35a00a28b8d3fc7796fc44a9df61ddb66f60665b9e6bbaef154624a.jpg](../icml_results/142_Detecting%20the%20Semantic%20Fixed%20Point_%20A%20Geometric%20Framework%20for%20Efficient%20Inference/tables/f7f455ada35a00a28b8d3fc7796fc44a9df61ddb66f60665b9e6bbaef154624a.jpg)

<a id="paper-0141"></a>

## A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots


### Images

![1c5145d76bfa58c3550c9796c7fea7a23726e7cc9f663e100f019ed875d819d8.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/images/1c5145d76bfa58c3550c9796c7fea7a23726e7cc9f663e100f019ed875d819d8.jpg)

![28af50af277448919c6e5b981d3c9c09e77ce0d595cedbe05d3b2a9beacc8d8a.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/images/28af50af277448919c6e5b981d3c9c09e77ce0d595cedbe05d3b2a9beacc8d8a.jpg)

![2cdd7015cb3e74fae8341e4f89ca645071698c9a3de7a6a8fc073cfcaef711b2.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/images/2cdd7015cb3e74fae8341e4f89ca645071698c9a3de7a6a8fc073cfcaef711b2.jpg)

![4b91e27ca4adf54c2bc37f892714cb0fbe9f4eabd14cb60cb6192028b2eb38c5.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/images/4b91e27ca4adf54c2bc37f892714cb0fbe9f4eabd14cb60cb6192028b2eb38c5.jpg)

![5473108fb13d9719e96d3bc4d04e66c37f6a3a2e795efaeec2ddf4728d3c12ff.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/images/5473108fb13d9719e96d3bc4d04e66c37f6a3a2e795efaeec2ddf4728d3c12ff.jpg)

![68794302543842e99334efaf8f11413251a5e5da2a614ef006e4b86163b0d2dc.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/images/68794302543842e99334efaf8f11413251a5e5da2a614ef006e4b86163b0d2dc.jpg)

![7079b1b6a5595c38e28c197411a9c347a947b6b5605fbe54305c740fde12d7aa.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/images/7079b1b6a5595c38e28c197411a9c347a947b6b5605fbe54305c740fde12d7aa.jpg)

![8e5149f01918ec53a35b04cb37fe98bfc16601da5d3f1b6b622e82980bc1fdee.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/images/8e5149f01918ec53a35b04cb37fe98bfc16601da5d3f1b6b622e82980bc1fdee.jpg)

![b6958f89d8e59c874cd792afcf0efa95481c19e177e0d2ba04cdb86d4b268ed4.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/images/b6958f89d8e59c874cd792afcf0efa95481c19e177e0d2ba04cdb86d4b268ed4.jpg)

![cc6c1662259486e838c1607bc02e506f9d058c51ff8251dcaa2f2a77a8b32964.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/images/cc6c1662259486e838c1607bc02e506f9d058c51ff8251dcaa2f2a77a8b32964.jpg)

![cd209fb80f7b6571c806b7aee359f5bfc64038deb9d9c6a912ec2e85479b73ab.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/images/cd209fb80f7b6571c806b7aee359f5bfc64038deb9d9c6a912ec2e85479b73ab.jpg)

![da614a84d9061b3ef7492434b73cb72e7b3839b9470ab2b00e978326ce5daf40.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/images/da614a84d9061b3ef7492434b73cb72e7b3839b9470ab2b00e978326ce5daf40.jpg)

### Tables

![4b19198d9c8ffaf91751f38dacd678c45152078bab11d58822dcbfe399ab8710.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/tables/4b19198d9c8ffaf91751f38dacd678c45152078bab11d58822dcbfe399ab8710.jpg)

![660c01538742f96c30049e7857bcfda4b814f9620ed3ad531312daacb8bf9c27.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/tables/660c01538742f96c30049e7857bcfda4b814f9620ed3ad531312daacb8bf9c27.jpg)

![8bf7dd561b46f2ed6e034e61a7559a5fc5e80815d3fcb2ac785b2f4ec0c5e19e.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/tables/8bf7dd561b46f2ed6e034e61a7559a5fc5e80815d3fcb2ac785b2f4ec0c5e19e.jpg)

![9b033b0e545825d0701c0431aa766ad423e98afa53fd95b0ffb14adca0a6439f.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/tables/9b033b0e545825d0701c0431aa766ad423e98afa53fd95b0ffb14adca0a6439f.jpg)

![b772e9fa50980bb530cfae672271cee6dbeba63c0decb986f9290fc52347840d.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/tables/b772e9fa50980bb530cfae672271cee6dbeba63c0decb986f9290fc52347840d.jpg)

![c65449e01d2dc3b261c5d99b60633d28ae26acfe232f9ae31f98f83d19f4b468.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/tables/c65449e01d2dc3b261c5d99b60633d28ae26acfe232f9ae31f98f83d19f4b468.jpg)

![efe42b97f18d78f1bfbce53c60bb6059a4d4b0a98dd8773b6882cfcbbc41e30d.jpg](../icml_results/143_A%20Call%20to%20Lagrangian%20Action_%20Learning%20Population%20Mechanics%20from%20Temporal%20Snapshots/tables/efe42b97f18d78f1bfbce53c60bb6059a4d4b0a98dd8773b6882cfcbbc41e30d.jpg)

<a id="paper-0142"></a>

## Monitoring Monitorability


### Images

![088b2a31d1015c235d6d84f0e64b2a3b318bafd956cfcd228183edc2dfc0147d.jpg](../icml_results/144_Monitoring%20Monitorability/images/088b2a31d1015c235d6d84f0e64b2a3b318bafd956cfcd228183edc2dfc0147d.jpg)

![0997d312e236d14e2a324a0bd521855dccf5315fc2519554dff5acb96db35b42.jpg](../icml_results/144_Monitoring%20Monitorability/images/0997d312e236d14e2a324a0bd521855dccf5315fc2519554dff5acb96db35b42.jpg)

![0a883c295ff3a3eb5c35f7a9a304bc059b8ea3542afd17c9b806377c916e4642.jpg](../icml_results/144_Monitoring%20Monitorability/images/0a883c295ff3a3eb5c35f7a9a304bc059b8ea3542afd17c9b806377c916e4642.jpg)

![0b5d26469b9a0594178d5890f972d1263d102cfc6664428ee3b36ab5eb6a61d6.jpg](../icml_results/144_Monitoring%20Monitorability/images/0b5d26469b9a0594178d5890f972d1263d102cfc6664428ee3b36ab5eb6a61d6.jpg)

![10dbbd5130168dc9896a4d30141db74f38d7f92641505b47198eb297eec6a23b.jpg](../icml_results/144_Monitoring%20Monitorability/images/10dbbd5130168dc9896a4d30141db74f38d7f92641505b47198eb297eec6a23b.jpg)

![216379533de3c6c56f76db580e56f9663643be6a6d7d3cb5e2599babada5d21e.jpg](../icml_results/144_Monitoring%20Monitorability/images/216379533de3c6c56f76db580e56f9663643be6a6d7d3cb5e2599babada5d21e.jpg)

![21ff5451f98bbbb508c3f16c842c251ed9ea78d9d9f94a1043b1625c10cf5196.jpg](../icml_results/144_Monitoring%20Monitorability/images/21ff5451f98bbbb508c3f16c842c251ed9ea78d9d9f94a1043b1625c10cf5196.jpg)

![2b31aa5c9eaac63bf056ed90522aba23203e2b9a92c804585cc27bcdcc2a9924.jpg](../icml_results/144_Monitoring%20Monitorability/images/2b31aa5c9eaac63bf056ed90522aba23203e2b9a92c804585cc27bcdcc2a9924.jpg)

![334ba91cdc76c076179c1d421e49d8606cad67f62c20de8da0c363b73a9e3923.jpg](../icml_results/144_Monitoring%20Monitorability/images/334ba91cdc76c076179c1d421e49d8606cad67f62c20de8da0c363b73a9e3923.jpg)

![3428a0da22aa45e9b1292de74149fb623ad506952745649b408c3338b571150f.jpg](../icml_results/144_Monitoring%20Monitorability/images/3428a0da22aa45e9b1292de74149fb623ad506952745649b408c3338b571150f.jpg)

![373e2565e0fb75ac9bea8d035ee815c1d13cca9b9d26db08d09a1b6ac85d7504.jpg](../icml_results/144_Monitoring%20Monitorability/images/373e2565e0fb75ac9bea8d035ee815c1d13cca9b9d26db08d09a1b6ac85d7504.jpg)

![3836187d234882def5c038a90c51e91804b7dde22293f0f2e37aa3c86b959d1f.jpg](../icml_results/144_Monitoring%20Monitorability/images/3836187d234882def5c038a90c51e91804b7dde22293f0f2e37aa3c86b959d1f.jpg)

![42426181a1462c1bdcc276998b84a2d9adfcf87f39f6ad886f48cca47f6456dd.jpg](../icml_results/144_Monitoring%20Monitorability/images/42426181a1462c1bdcc276998b84a2d9adfcf87f39f6ad886f48cca47f6456dd.jpg)

![4858bef7180586d8b410b88703de40999241dab5bbaa07e110abd8b97827b959.jpg](../icml_results/144_Monitoring%20Monitorability/images/4858bef7180586d8b410b88703de40999241dab5bbaa07e110abd8b97827b959.jpg)

![55cbe38db24d4cc352b117d537acda4a549a97c2ab2008b91e54be9a3cf8dfbe.jpg](../icml_results/144_Monitoring%20Monitorability/images/55cbe38db24d4cc352b117d537acda4a549a97c2ab2008b91e54be9a3cf8dfbe.jpg)

![68091bfafcc4419df1c58a04b8980fa37438c18fa2d06d4b80e2662960d42896.jpg](../icml_results/144_Monitoring%20Monitorability/images/68091bfafcc4419df1c58a04b8980fa37438c18fa2d06d4b80e2662960d42896.jpg)

![7012f7f0284f08549de110568811fbbbc87d192aca2c797ed65fdcac15aaa536.jpg](../icml_results/144_Monitoring%20Monitorability/images/7012f7f0284f08549de110568811fbbbc87d192aca2c797ed65fdcac15aaa536.jpg)

![7cf174462524f5c73d2af39babc92d34e5986cd523e75f590d91bec52ca4ac05.jpg](../icml_results/144_Monitoring%20Monitorability/images/7cf174462524f5c73d2af39babc92d34e5986cd523e75f590d91bec52ca4ac05.jpg)

![887c6ad6b2c017d1d6b33f5afff80d9e571a49fd452d58f29d4ac7319e89f508.jpg](../icml_results/144_Monitoring%20Monitorability/images/887c6ad6b2c017d1d6b33f5afff80d9e571a49fd452d58f29d4ac7319e89f508.jpg)

![8a64d4630d97aa7e1730ab8c19202d5fabc4435b0e74187c4b76accaed69242c.jpg](../icml_results/144_Monitoring%20Monitorability/images/8a64d4630d97aa7e1730ab8c19202d5fabc4435b0e74187c4b76accaed69242c.jpg)

![8c52be882071d9fc3c5ea974cc516aac6237a367f22b1cc993acaca98a7bc7da.jpg](../icml_results/144_Monitoring%20Monitorability/images/8c52be882071d9fc3c5ea974cc516aac6237a367f22b1cc993acaca98a7bc7da.jpg)

![8cf9373ac53e1ae1094828bc8321b68f82991774f3d0dd20e8b99b63e55bae22.jpg](../icml_results/144_Monitoring%20Monitorability/images/8cf9373ac53e1ae1094828bc8321b68f82991774f3d0dd20e8b99b63e55bae22.jpg)

![919b48b9c35efc36f266883e5879a21e6f319c7577aa9859a55e9c2b8c9d29e6.jpg](../icml_results/144_Monitoring%20Monitorability/images/919b48b9c35efc36f266883e5879a21e6f319c7577aa9859a55e9c2b8c9d29e6.jpg)

![94a00ce9e949a34aff88e4c888d65c5d1c72ecb5b0db2b56cca5cf6cadfa9e3f.jpg](../icml_results/144_Monitoring%20Monitorability/images/94a00ce9e949a34aff88e4c888d65c5d1c72ecb5b0db2b56cca5cf6cadfa9e3f.jpg)

![95c70800dde5d861668e616bfcff2437ee1d95a454e3b91e97cd49c4019dcae6.jpg](../icml_results/144_Monitoring%20Monitorability/images/95c70800dde5d861668e616bfcff2437ee1d95a454e3b91e97cd49c4019dcae6.jpg)

![a445e12aa42e7a8f097c3e9ff5c010627d491d749e55caf924a9f15b9e623122.jpg](../icml_results/144_Monitoring%20Monitorability/images/a445e12aa42e7a8f097c3e9ff5c010627d491d749e55caf924a9f15b9e623122.jpg)

![a9a17d15512610820fe8660eaee34602dd03deaf4183263326588decafcc4f2a.jpg](../icml_results/144_Monitoring%20Monitorability/images/a9a17d15512610820fe8660eaee34602dd03deaf4183263326588decafcc4f2a.jpg)

![ae78d920bc556cf3342a2bd49350ca279b6db8edc62d77e83b71401688565cfd.jpg](../icml_results/144_Monitoring%20Monitorability/images/ae78d920bc556cf3342a2bd49350ca279b6db8edc62d77e83b71401688565cfd.jpg)

![aee0e7704b39aab535d21455c9b968ad68472d6f2a32631296ddcefba9e44267.jpg](../icml_results/144_Monitoring%20Monitorability/images/aee0e7704b39aab535d21455c9b968ad68472d6f2a32631296ddcefba9e44267.jpg)

![ba4056004333e6fc1b5ae114193e3290f50f615e32d97c1c1564bcae860cdfc6.jpg](../icml_results/144_Monitoring%20Monitorability/images/ba4056004333e6fc1b5ae114193e3290f50f615e32d97c1c1564bcae860cdfc6.jpg)

![bcc99eb11ea1a3ac1bb7e2e01dfee74d74d7656294fd05016c8be8e02091363f.jpg](../icml_results/144_Monitoring%20Monitorability/images/bcc99eb11ea1a3ac1bb7e2e01dfee74d74d7656294fd05016c8be8e02091363f.jpg)

![cccff4f6c011aad907949d4100dcc3cfbef75cd5a7c5620c6822ceb44ac3f7db.jpg](../icml_results/144_Monitoring%20Monitorability/images/cccff4f6c011aad907949d4100dcc3cfbef75cd5a7c5620c6822ceb44ac3f7db.jpg)

![d6fbf550dd6be51d704b12b9ef1c7b6348f38a46d6f9876bd5cb9c196ba7ae57.jpg](../icml_results/144_Monitoring%20Monitorability/images/d6fbf550dd6be51d704b12b9ef1c7b6348f38a46d6f9876bd5cb9c196ba7ae57.jpg)

![e6689d5a74c8e5f3dd2b8b23487a85dc0bac38334a2868e5e5a60598f86ec5b3.jpg](../icml_results/144_Monitoring%20Monitorability/images/e6689d5a74c8e5f3dd2b8b23487a85dc0bac38334a2868e5e5a60598f86ec5b3.jpg)

![e9dc7fcae1ed3dcef160ef8e679a9491b5cdd923566bfcefa0aa7d9f3f0c4e2e.jpg](../icml_results/144_Monitoring%20Monitorability/images/e9dc7fcae1ed3dcef160ef8e679a9491b5cdd923566bfcefa0aa7d9f3f0c4e2e.jpg)

![eb95b79a1792d508b6843152b309d34313e6fd027bac3c2176c240f6c69570cf.jpg](../icml_results/144_Monitoring%20Monitorability/images/eb95b79a1792d508b6843152b309d34313e6fd027bac3c2176c240f6c69570cf.jpg)

### Tables

![0314ab540298e7dfa35646a62934ed28b67e95962b669d24e05e53c7bab610b4.jpg](../icml_results/144_Monitoring%20Monitorability/tables/0314ab540298e7dfa35646a62934ed28b67e95962b669d24e05e53c7bab610b4.jpg)

![2dc96c1ede9917a87de51c81781e8e6f5b91445c51034c4e38eabef9dcbbb6ef.jpg](../icml_results/144_Monitoring%20Monitorability/tables/2dc96c1ede9917a87de51c81781e8e6f5b91445c51034c4e38eabef9dcbbb6ef.jpg)

![a0242490b527262e906be2c51a470fd40a9e5fa5f39815ece7a5ff177706e3ac.jpg](../icml_results/144_Monitoring%20Monitorability/tables/a0242490b527262e906be2c51a470fd40a9e5fa5f39815ece7a5ff177706e3ac.jpg)

![bda32cea43c848f220cabf642a066fb11ac49c57f1a9c54364375282fdd0734a.jpg](../icml_results/144_Monitoring%20Monitorability/tables/bda32cea43c848f220cabf642a066fb11ac49c57f1a9c54364375282fdd0734a.jpg)

![df7fb8e1a9f492e171d4e6e98299866725ef8e688ae2b6789c8f712d6c355617.jpg](../icml_results/144_Monitoring%20Monitorability/tables/df7fb8e1a9f492e171d4e6e98299866725ef8e688ae2b6789c8f712d6c355617.jpg)

<a id="paper-0143"></a>

## Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing


### Images

![07b04faa0c6de786386caafba52e220faaac181aa227e5658aacec1807b1fb90.jpg](../icml_results/145_Hallucination%20is%20a%20Consequence%20of%20Space-Optimality_%20A%20Rate-Distortion%20Theorem%20for%20Membership%20Testing/images/07b04faa0c6de786386caafba52e220faaac181aa227e5658aacec1807b1fb90.jpg)

![124bdacf5dd7439088ca398b311940a9d92e692668d41655228502fc984382b5.jpg](../icml_results/145_Hallucination%20is%20a%20Consequence%20of%20Space-Optimality_%20A%20Rate-Distortion%20Theorem%20for%20Membership%20Testing/images/124bdacf5dd7439088ca398b311940a9d92e692668d41655228502fc984382b5.jpg)

![153b1c05dc4436d788d5eac5bcdb1545e32861170ae07ffbe1ae88dc1132dc80.jpg](../icml_results/145_Hallucination%20is%20a%20Consequence%20of%20Space-Optimality_%20A%20Rate-Distortion%20Theorem%20for%20Membership%20Testing/images/153b1c05dc4436d788d5eac5bcdb1545e32861170ae07ffbe1ae88dc1132dc80.jpg)

![5705ce9658bdc143cc331732b2df7e0f447f0611df93d7b788766ff970367600.jpg](../icml_results/145_Hallucination%20is%20a%20Consequence%20of%20Space-Optimality_%20A%20Rate-Distortion%20Theorem%20for%20Membership%20Testing/images/5705ce9658bdc143cc331732b2df7e0f447f0611df93d7b788766ff970367600.jpg)

![9d6f82572ac7c7e78f62f0fb37e49fea2c9da6b189d05db57cf9654bbc2234ea.jpg](../icml_results/145_Hallucination%20is%20a%20Consequence%20of%20Space-Optimality_%20A%20Rate-Distortion%20Theorem%20for%20Membership%20Testing/images/9d6f82572ac7c7e78f62f0fb37e49fea2c9da6b189d05db57cf9654bbc2234ea.jpg)

![c31df9896b04788c192f1b16a441fc1d3ede5675a676e2f4e84233b7b0e5b548.jpg](../icml_results/145_Hallucination%20is%20a%20Consequence%20of%20Space-Optimality_%20A%20Rate-Distortion%20Theorem%20for%20Membership%20Testing/images/c31df9896b04788c192f1b16a441fc1d3ede5675a676e2f4e84233b7b0e5b548.jpg)

![c6b3a1a0cba8bc0a2b8e6774045fc2a525c2cf94b1bfa31045d7f7ac109b8bfc.jpg](../icml_results/145_Hallucination%20is%20a%20Consequence%20of%20Space-Optimality_%20A%20Rate-Distortion%20Theorem%20for%20Membership%20Testing/images/c6b3a1a0cba8bc0a2b8e6774045fc2a525c2cf94b1bfa31045d7f7ac109b8bfc.jpg)

![dd00c918d210fede7af51ffc77703832b227e8adc60dd96f1fbf0110e35588c3.jpg](../icml_results/145_Hallucination%20is%20a%20Consequence%20of%20Space-Optimality_%20A%20Rate-Distortion%20Theorem%20for%20Membership%20Testing/images/dd00c918d210fede7af51ffc77703832b227e8adc60dd96f1fbf0110e35588c3.jpg)

### Tables

![05dbaa7722b55500316953c3a3d692f8e7115dba2c55796035593b91fb4bc929.jpg](../icml_results/145_Hallucination%20is%20a%20Consequence%20of%20Space-Optimality_%20A%20Rate-Distortion%20Theorem%20for%20Membership%20Testing/tables/05dbaa7722b55500316953c3a3d692f8e7115dba2c55796035593b91fb4bc929.jpg)

![0ba82fc347f561755eee2b339b0672b0fe3e761d18cdd9bafcb682514d431d3c.jpg](../icml_results/145_Hallucination%20is%20a%20Consequence%20of%20Space-Optimality_%20A%20Rate-Distortion%20Theorem%20for%20Membership%20Testing/tables/0ba82fc347f561755eee2b339b0672b0fe3e761d18cdd9bafcb682514d431d3c.jpg)

![16bb641ee494a181dbb852bd170daa03e9b3a52dfa91c53b8c43b3e57cd3f983.jpg](../icml_results/145_Hallucination%20is%20a%20Consequence%20of%20Space-Optimality_%20A%20Rate-Distortion%20Theorem%20for%20Membership%20Testing/tables/16bb641ee494a181dbb852bd170daa03e9b3a52dfa91c53b8c43b3e57cd3f983.jpg)

<a id="paper-0144"></a>

## DRPBench: Evaluating LLMs in Concurrent Code Comprehension via Fine-Grained Data Race Prediction


### Images

![155cbebf523a648922a9956d387387127975d625bc0fcee47a08fe24e1acb62d.jpg](../icml_results/146_DRPBench_%20Evaluating%20LLMs%20in%20Concurrent%20Code%20Comprehension%20via%20Fine-Grained%20Data%20Race%20Prediction/images/155cbebf523a648922a9956d387387127975d625bc0fcee47a08fe24e1acb62d.jpg)

![55c86b80bf8c0905691e20a573476ceabff65de1ff8dd97ae7da1638afe0c069.jpg](../icml_results/146_DRPBench_%20Evaluating%20LLMs%20in%20Concurrent%20Code%20Comprehension%20via%20Fine-Grained%20Data%20Race%20Prediction/images/55c86b80bf8c0905691e20a573476ceabff65de1ff8dd97ae7da1638afe0c069.jpg)

![673304ffb9c77462122c7217c2ecffa78ea96e8f5132287873be1bfcb9d06841.jpg](../icml_results/146_DRPBench_%20Evaluating%20LLMs%20in%20Concurrent%20Code%20Comprehension%20via%20Fine-Grained%20Data%20Race%20Prediction/images/673304ffb9c77462122c7217c2ecffa78ea96e8f5132287873be1bfcb9d06841.jpg)

![9df05887fdb9b66192b63843811273722964570ba90dd81152c5e2cc03e8b41e.jpg](../icml_results/146_DRPBench_%20Evaluating%20LLMs%20in%20Concurrent%20Code%20Comprehension%20via%20Fine-Grained%20Data%20Race%20Prediction/images/9df05887fdb9b66192b63843811273722964570ba90dd81152c5e2cc03e8b41e.jpg)

### Tables

![189aa084cafcf2a2a7c2ec8c50a561d28d5c16d572761278cd9d5f6c20e7a1bd.jpg](../icml_results/146_DRPBench_%20Evaluating%20LLMs%20in%20Concurrent%20Code%20Comprehension%20via%20Fine-Grained%20Data%20Race%20Prediction/tables/189aa084cafcf2a2a7c2ec8c50a561d28d5c16d572761278cd9d5f6c20e7a1bd.jpg)

![688c76cfaa21734420f595f881e83468272089b0ac63e69b9d80f8c8382ef1ba.jpg](../icml_results/146_DRPBench_%20Evaluating%20LLMs%20in%20Concurrent%20Code%20Comprehension%20via%20Fine-Grained%20Data%20Race%20Prediction/tables/688c76cfaa21734420f595f881e83468272089b0ac63e69b9d80f8c8382ef1ba.jpg)

![8fd55857fe2c224ea822de61fb598478099bbde6bb3757cfb6d8d3ebc4b46777.jpg](../icml_results/146_DRPBench_%20Evaluating%20LLMs%20in%20Concurrent%20Code%20Comprehension%20via%20Fine-Grained%20Data%20Race%20Prediction/tables/8fd55857fe2c224ea822de61fb598478099bbde6bb3757cfb6d8d3ebc4b46777.jpg)

![a7700377ea3374a1894a6c96307937b784e3c3f2526d2e6713845a769a95cc35.jpg](../icml_results/146_DRPBench_%20Evaluating%20LLMs%20in%20Concurrent%20Code%20Comprehension%20via%20Fine-Grained%20Data%20Race%20Prediction/tables/a7700377ea3374a1894a6c96307937b784e3c3f2526d2e6713845a769a95cc35.jpg)

![b99e34b1e0532d66724bd7ce9eb2b0dc1e684070e5bb70767abb118fc748ba79.jpg](../icml_results/146_DRPBench_%20Evaluating%20LLMs%20in%20Concurrent%20Code%20Comprehension%20via%20Fine-Grained%20Data%20Race%20Prediction/tables/b99e34b1e0532d66724bd7ce9eb2b0dc1e684070e5bb70767abb118fc748ba79.jpg)

![d659e4a4d566275cf67f78bb6dcddff0942adbcc8e915067f4c93fcd309a3a31.jpg](../icml_results/146_DRPBench_%20Evaluating%20LLMs%20in%20Concurrent%20Code%20Comprehension%20via%20Fine-Grained%20Data%20Race%20Prediction/tables/d659e4a4d566275cf67f78bb6dcddff0942adbcc8e915067f4c93fcd309a3a31.jpg)

![e585de45fdb01ec3615ad541cf142d5e3c9e9f74b698ac5626778d4874e043c2.jpg](../icml_results/146_DRPBench_%20Evaluating%20LLMs%20in%20Concurrent%20Code%20Comprehension%20via%20Fine-Grained%20Data%20Race%20Prediction/tables/e585de45fdb01ec3615ad541cf142d5e3c9e9f74b698ac5626778d4874e043c2.jpg)

![fd52b011b27809f9c5547c208b875d0621453e2df3ffe660814ecadb10c0ade2.jpg](../icml_results/146_DRPBench_%20Evaluating%20LLMs%20in%20Concurrent%20Code%20Comprehension%20via%20Fine-Grained%20Data%20Race%20Prediction/tables/fd52b011b27809f9c5547c208b875d0621453e2df3ffe660814ecadb10c0ade2.jpg)

<a id="paper-0145"></a>

## MEnvAgent: Scalable Polyglot Environment Construction for Verifiable Software Engineering


### Images

![24105ec664aeb06c855eb0d3efbe22dbc1ccba021ff4d2a6eaeaf137768d6a75.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/images/24105ec664aeb06c855eb0d3efbe22dbc1ccba021ff4d2a6eaeaf137768d6a75.jpg)

![33fa625e149a75d292f845693743d4df74a42d2a5800bfffc7a1078527ab3976.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/images/33fa625e149a75d292f845693743d4df74a42d2a5800bfffc7a1078527ab3976.jpg)

![3cdf17daa25acf5299ee69ea29bac5158795c40259311a81b47d99cd7567b7f3.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/images/3cdf17daa25acf5299ee69ea29bac5158795c40259311a81b47d99cd7567b7f3.jpg)

![50c0b7917c0478afc6e8a1921990b1e3b1659caeb51cb67faea3f81c96cc069b.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/images/50c0b7917c0478afc6e8a1921990b1e3b1659caeb51cb67faea3f81c96cc069b.jpg)

![8a865254117f7524d8c0c70586a7f531fc39e9168565d2fe1e2ec46eec7f7d77.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/images/8a865254117f7524d8c0c70586a7f531fc39e9168565d2fe1e2ec46eec7f7d77.jpg)

![939a9c322feeaaee52dc527b36164a6bcfe982eed427c70830ce5c49b5e993a6.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/images/939a9c322feeaaee52dc527b36164a6bcfe982eed427c70830ce5c49b5e993a6.jpg)

![9a52c72b9c0527e0f306c55fe3c49fd97525de1e5ad383a763e647ad86877eff.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/images/9a52c72b9c0527e0f306c55fe3c49fd97525de1e5ad383a763e647ad86877eff.jpg)

![ad1f97581bb2d39a60a3f2bfc72608bf1f7c0a72ee85883a8b5cf068a6ee5ccd.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/images/ad1f97581bb2d39a60a3f2bfc72608bf1f7c0a72ee85883a8b5cf068a6ee5ccd.jpg)

### Tables

![060bb6ec7217fde9b9fed3846ec4f1aa90dd6ecc73620949c9aac91a4e6e7905.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/tables/060bb6ec7217fde9b9fed3846ec4f1aa90dd6ecc73620949c9aac91a4e6e7905.jpg)

![1228fa1925aa8d1cc7506628d4ead05cb220780951f1adf87a31395a22562709.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/tables/1228fa1925aa8d1cc7506628d4ead05cb220780951f1adf87a31395a22562709.jpg)

![2ce77c990533737cf3cad4b9744fe9339d10b807a2fd0f345de40e0e73efa03a.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/tables/2ce77c990533737cf3cad4b9744fe9339d10b807a2fd0f345de40e0e73efa03a.jpg)

![4d88a4d3a5648acf9ff825a04f43c040b77d6fac7ef27fae562f89bcb975ed03.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/tables/4d88a4d3a5648acf9ff825a04f43c040b77d6fac7ef27fae562f89bcb975ed03.jpg)

![4d8cbca1200d3bc8e4a7bc57c04f5ad4c0f8947e87c6481e32722ab0e41ba1c4.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/tables/4d8cbca1200d3bc8e4a7bc57c04f5ad4c0f8947e87c6481e32722ab0e41ba1c4.jpg)

![5223fc45285d08c48ce9da61dc808af74a6b215dd762957f3be7f524dee62eeb.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/tables/5223fc45285d08c48ce9da61dc808af74a6b215dd762957f3be7f524dee62eeb.jpg)

![73fbbc1d3e00c5f0a3c7d39b2d511cf3e3ec697ba54761ad1c08bcbd51a9c9db.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/tables/73fbbc1d3e00c5f0a3c7d39b2d511cf3e3ec697ba54761ad1c08bcbd51a9c9db.jpg)

![9768bae393dd293b9a85bae4bcea00a043bf70f86004eac118b33056edab9619.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/tables/9768bae393dd293b9a85bae4bcea00a043bf70f86004eac118b33056edab9619.jpg)

![b0812a626e51a4e47ff5261b6ad7060547d2263a2b0fba72a2bb5c382d36e4e6.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/tables/b0812a626e51a4e47ff5261b6ad7060547d2263a2b0fba72a2bb5c382d36e4e6.jpg)

![bbd6c2fbb614364982ae16c988cd5ff4280585e88a0324b1f82ada9d66f73f7c.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/tables/bbd6c2fbb614364982ae16c988cd5ff4280585e88a0324b1f82ada9d66f73f7c.jpg)

![bbf2f6f470a1ba1cc65a3a453fc7f97ebdb4c9d5458d161aa9a5aeb9ee982704.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/tables/bbf2f6f470a1ba1cc65a3a453fc7f97ebdb4c9d5458d161aa9a5aeb9ee982704.jpg)

![c600df555dd136f09571ecfa989dbbee29951875d38e5ef28294375903e3395f.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/tables/c600df555dd136f09571ecfa989dbbee29951875d38e5ef28294375903e3395f.jpg)

![f5d8189dbae2a8bb6e4468bb571033e76571f8b47dd6a891d70517c1984f1931.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/tables/f5d8189dbae2a8bb6e4468bb571033e76571f8b47dd6a891d70517c1984f1931.jpg)

![fc91c6741bb1af8475e07d690c85a7c01e35052aa476250f157405f04ed972f9.jpg](../icml_results/147_MEnvAgent_%20Scalable%20Polyglot%20Environment%20Construction%20for%20Verifiable%20Software%20Engineering/tables/fc91c6741bb1af8475e07d690c85a7c01e35052aa476250f157405f04ed972f9.jpg)

<a id="paper-0146"></a>

## Divide-and-Denoise: A Game-Theoretic Method for Fairly Composing Diffusion Models


### Images

![1cca40104c3dd2daa5e4e91b9aff87381c929d79e2c6865d84e3e5415050a088.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/images/1cca40104c3dd2daa5e4e91b9aff87381c929d79e2c6865d84e3e5415050a088.jpg)

![324dc021d3a0cea03df7dfe27a397cf255a13e427d9d24a8564d3f139a1ce7d6.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/images/324dc021d3a0cea03df7dfe27a397cf255a13e427d9d24a8564d3f139a1ce7d6.jpg)

![55bf6394c7e14549998eca2bfc307ae7728c263310387f7c539a0d76aec53796.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/images/55bf6394c7e14549998eca2bfc307ae7728c263310387f7c539a0d76aec53796.jpg)

![678b4375f4b6ed7a623df13ad381e3dd96f4b3e3c7ade5973d1beb0ed3fbfdb9.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/images/678b4375f4b6ed7a623df13ad381e3dd96f4b3e3c7ade5973d1beb0ed3fbfdb9.jpg)

![6a7f56f93dd44c87b7949c66391a11ece172d34f952ff384a6cd07d832df7726.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/images/6a7f56f93dd44c87b7949c66391a11ece172d34f952ff384a6cd07d832df7726.jpg)

![c8731609d3eb3951a9743c1dd01e8010241f3399f52ee2476e1489a37a7c1f8b.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/images/c8731609d3eb3951a9743c1dd01e8010241f3399f52ee2476e1489a37a7c1f8b.jpg)

![d11fe82d4dc84f3449ba940b9d1980b9c7d50786f213cddf15faa391bd89b56e.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/images/d11fe82d4dc84f3449ba940b9d1980b9c7d50786f213cddf15faa391bd89b56e.jpg)

![dcd6efb8ad28460aef414b34c96a2369f6412885a318485fa58cc6bd18e0664d.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/images/dcd6efb8ad28460aef414b34c96a2369f6412885a318485fa58cc6bd18e0664d.jpg)

![e2ccb8a0137aee98db89d91e55639abfeff13ab43739e5556d16360c3da14284.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/images/e2ccb8a0137aee98db89d91e55639abfeff13ab43739e5556d16360c3da14284.jpg)

![e6b92f62970bbedf5278a74ee4c320727aca739272b233086be46e8c877daa24.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/images/e6b92f62970bbedf5278a74ee4c320727aca739272b233086be46e8c877daa24.jpg)

![eb7e7e3e9c9527829735a18807833ed288d2663e48222b37959fcd65885751f0.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/images/eb7e7e3e9c9527829735a18807833ed288d2663e48222b37959fcd65885751f0.jpg)

### Tables

![07dada6d0661b715033fdb817eac6f541f1e35fe2aaeae1846dd1fa0ec44c320.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/tables/07dada6d0661b715033fdb817eac6f541f1e35fe2aaeae1846dd1fa0ec44c320.jpg)

![17b634323a1d75a934ac51b368471fc1e91990a9e067daeab770ae236581e590.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/tables/17b634323a1d75a934ac51b368471fc1e91990a9e067daeab770ae236581e590.jpg)

![36aa922ddc705b5c789afa05cb7c0c98c7771e2faf8afaa51e48584465a2decb.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/tables/36aa922ddc705b5c789afa05cb7c0c98c7771e2faf8afaa51e48584465a2decb.jpg)

![4d5ee85bb63d8b9143f755e50e495a748c5afe145d66988b4f6e6b4b93aaa99b.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/tables/4d5ee85bb63d8b9143f755e50e495a748c5afe145d66988b4f6e6b4b93aaa99b.jpg)

![5f2ae54573c82a218303e4bbb1b0437820ab7fcd266007cac34fa97e4fd21b0c.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/tables/5f2ae54573c82a218303e4bbb1b0437820ab7fcd266007cac34fa97e4fd21b0c.jpg)

![77fb252021f0d70897ed1031158147991d868a192971dcd73613411db158cb2c.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/tables/77fb252021f0d70897ed1031158147991d868a192971dcd73613411db158cb2c.jpg)

![911d04230259d9f31495a653d141b155a8352d58c8036c795603a68cc200717c.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/tables/911d04230259d9f31495a653d141b155a8352d58c8036c795603a68cc200717c.jpg)

![a9d84fb8aae25e75cd09d710de3cda513a06365e4ebe1fb59fa7468b1abadca5.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/tables/a9d84fb8aae25e75cd09d710de3cda513a06365e4ebe1fb59fa7468b1abadca5.jpg)

![db84d9b4bd2b0bd8b26df7ba27c9a0e50711ddb5f10388f1b91dab981c7c6b22.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/tables/db84d9b4bd2b0bd8b26df7ba27c9a0e50711ddb5f10388f1b91dab981c7c6b22.jpg)

![ef5fbe4e56136b855089dce0f9684ccf21ff82577d230f5bb634c5f7da769ca3.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/tables/ef5fbe4e56136b855089dce0f9684ccf21ff82577d230f5bb634c5f7da769ca3.jpg)

![f4d195f54ff717939d9515ad2b5dbd4ae8434794a7d8c9964f79f4cffcfdd142.jpg](../icml_results/148_Divide-and-Denoise_%20A%20Game-Theoretic%20Method%20for%20Fairly%20Composing%20Diffusion%20Models/tables/f4d195f54ff717939d9515ad2b5dbd4ae8434794a7d8c9964f79f4cffcfdd142.jpg)

<a id="paper-0147"></a>

## Why Deep Jacobian Spectra Separate: Depth-Induced Scaling and Singular-Vector Alignment


### Images

![2487e9a59384e6060a2a38f5dda418e8538be6adf2be0d04035d223db9c772fb.jpg](../icml_results/149_Why%20Deep%20Jacobian%20Spectra%20Separate_%20Depth-Induced%20Scaling%20and%20Singular-Vector%20Alignment/images/2487e9a59384e6060a2a38f5dda418e8538be6adf2be0d04035d223db9c772fb.jpg)

![2547108905908f28a7dad7caa072b2b02b548044249dfbf3652e1b37c230c179.jpg](../icml_results/149_Why%20Deep%20Jacobian%20Spectra%20Separate_%20Depth-Induced%20Scaling%20and%20Singular-Vector%20Alignment/images/2547108905908f28a7dad7caa072b2b02b548044249dfbf3652e1b37c230c179.jpg)

![443990340cb9f76d3c38347f9d57fc9136a71e081af2859b3248b43ee961a71d.jpg](../icml_results/149_Why%20Deep%20Jacobian%20Spectra%20Separate_%20Depth-Induced%20Scaling%20and%20Singular-Vector%20Alignment/images/443990340cb9f76d3c38347f9d57fc9136a71e081af2859b3248b43ee961a71d.jpg)

![6f5511b43a5c84971d53b78c3741ed4bbd20ca524ac319b0a481a630349b95db.jpg](../icml_results/149_Why%20Deep%20Jacobian%20Spectra%20Separate_%20Depth-Induced%20Scaling%20and%20Singular-Vector%20Alignment/images/6f5511b43a5c84971d53b78c3741ed4bbd20ca524ac319b0a481a630349b95db.jpg)

![8a75970bf9680c11fa5e4cb2241fc1416ff383aa2c40be6a49815859d72b6d5c.jpg](../icml_results/149_Why%20Deep%20Jacobian%20Spectra%20Separate_%20Depth-Induced%20Scaling%20and%20Singular-Vector%20Alignment/images/8a75970bf9680c11fa5e4cb2241fc1416ff383aa2c40be6a49815859d72b6d5c.jpg)

![9bbb869eae0dd85f0633c78db0c439c9185f2d01d20f60de77357ce8f352528c.jpg](../icml_results/149_Why%20Deep%20Jacobian%20Spectra%20Separate_%20Depth-Induced%20Scaling%20and%20Singular-Vector%20Alignment/images/9bbb869eae0dd85f0633c78db0c439c9185f2d01d20f60de77357ce8f352528c.jpg)

![a1a7f73c495ab1a832185dd4362fad22981c051e91532602497eb74f276da566.jpg](../icml_results/149_Why%20Deep%20Jacobian%20Spectra%20Separate_%20Depth-Induced%20Scaling%20and%20Singular-Vector%20Alignment/images/a1a7f73c495ab1a832185dd4362fad22981c051e91532602497eb74f276da566.jpg)

![aec10f6dc2302505266be0ce4bd58ced72e052d69187624662e716ff85305ae0.jpg](../icml_results/149_Why%20Deep%20Jacobian%20Spectra%20Separate_%20Depth-Induced%20Scaling%20and%20Singular-Vector%20Alignment/images/aec10f6dc2302505266be0ce4bd58ced72e052d69187624662e716ff85305ae0.jpg)

![c1b4b8d4bf5c1eb2af1ffec07b3f8aca90748d9522a132899e391c886c2f5bee.jpg](../icml_results/149_Why%20Deep%20Jacobian%20Spectra%20Separate_%20Depth-Induced%20Scaling%20and%20Singular-Vector%20Alignment/images/c1b4b8d4bf5c1eb2af1ffec07b3f8aca90748d9522a132899e391c886c2f5bee.jpg)

![d3258cef9df89a43dfcb1507940881d89c50c50e845c0839dad6a94316a37aca.jpg](../icml_results/149_Why%20Deep%20Jacobian%20Spectra%20Separate_%20Depth-Induced%20Scaling%20and%20Singular-Vector%20Alignment/images/d3258cef9df89a43dfcb1507940881d89c50c50e845c0839dad6a94316a37aca.jpg)

![df246c12ef1369adcf6f760d82752a6ce95ede5cb029ba32848cb11899e8048b.jpg](../icml_results/149_Why%20Deep%20Jacobian%20Spectra%20Separate_%20Depth-Induced%20Scaling%20and%20Singular-Vector%20Alignment/images/df246c12ef1369adcf6f760d82752a6ce95ede5cb029ba32848cb11899e8048b.jpg)

![dff2592f2151205d61d4d6fb4c2d5db1df3e7df804f51d38b48b91c0d1006bb3.jpg](../icml_results/149_Why%20Deep%20Jacobian%20Spectra%20Separate_%20Depth-Induced%20Scaling%20and%20Singular-Vector%20Alignment/images/dff2592f2151205d61d4d6fb4c2d5db1df3e7df804f51d38b48b91c0d1006bb3.jpg)

![e38521c4cda13119f585f8838f8cd8b08bbf7146c87edeb95fe715a682ad19a3.jpg](../icml_results/149_Why%20Deep%20Jacobian%20Spectra%20Separate_%20Depth-Induced%20Scaling%20and%20Singular-Vector%20Alignment/images/e38521c4cda13119f585f8838f8cd8b08bbf7146c87edeb95fe715a682ad19a3.jpg)

![e45617ffdd7a819a651161cbab3c9ac785b3495eaa17282b67ec0e8ab6b97ab9.jpg](../icml_results/149_Why%20Deep%20Jacobian%20Spectra%20Separate_%20Depth-Induced%20Scaling%20and%20Singular-Vector%20Alignment/images/e45617ffdd7a819a651161cbab3c9ac785b3495eaa17282b67ec0e8ab6b97ab9.jpg)

![f28b45cb54c3909b3fa8b82782d9da9303fabd1064448477043fdde0d0484ed9.jpg](../icml_results/149_Why%20Deep%20Jacobian%20Spectra%20Separate_%20Depth-Induced%20Scaling%20and%20Singular-Vector%20Alignment/images/f28b45cb54c3909b3fa8b82782d9da9303fabd1064448477043fdde0d0484ed9.jpg)

![f6ac3c1c21ca524aaad217c0bb07fe2f160725a08006dc8d8e941312e2840f5d.jpg](../icml_results/149_Why%20Deep%20Jacobian%20Spectra%20Separate_%20Depth-Induced%20Scaling%20and%20Singular-Vector%20Alignment/images/f6ac3c1c21ca524aaad217c0bb07fe2f160725a08006dc8d8e941312e2840f5d.jpg)

<a id="paper-0148"></a>

## Jailbreak to Protect: Buffering and Reinforcing via Temporary Jailbreaking for Safe Fine-Tuning in Large Language Models


### Images

![08458391f9bc908d1534fd3d955c35d3a38404f2cc1fdcef87aeb123702ab552.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/images/08458391f9bc908d1534fd3d955c35d3a38404f2cc1fdcef87aeb123702ab552.jpg)

![5bbf74cf8dfe02a50441320473b167d59ea143a5ea7d64bc692f66a55ba7eba0.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/images/5bbf74cf8dfe02a50441320473b167d59ea143a5ea7d64bc692f66a55ba7eba0.jpg)

![82cc0881246e56493285f48d84af16694df2cd8e46a53267fbce56c63732fcec.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/images/82cc0881246e56493285f48d84af16694df2cd8e46a53267fbce56c63732fcec.jpg)

![89acab2302b277dd29d6ec9373acad11d8070cdcf9503240fbbcac5fa67bffd6.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/images/89acab2302b277dd29d6ec9373acad11d8070cdcf9503240fbbcac5fa67bffd6.jpg)

### Tables

![0cda76cb5297f063068556b68416b55dfdea65aacb848e3e4525212b3ff978aa.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/0cda76cb5297f063068556b68416b55dfdea65aacb848e3e4525212b3ff978aa.jpg)

![0fbe46560c35dc8c57927af9fff11fd637671760d91f5102a91614acfd8a1388.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/0fbe46560c35dc8c57927af9fff11fd637671760d91f5102a91614acfd8a1388.jpg)

![10424504c2659003600df14d25e8ca24b3ec44cbf47ccf81fd64150c22545dc8.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/10424504c2659003600df14d25e8ca24b3ec44cbf47ccf81fd64150c22545dc8.jpg)

![1e17824e559eb92000e108f8f1fbda42f1733d753d079a04bd03778be855e1c6.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/1e17824e559eb92000e108f8f1fbda42f1733d753d079a04bd03778be855e1c6.jpg)

![32c0ce65c3dc420f3d9211feb6d790bb71f6d389da8ccdbfabeebebede2aab3a.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/32c0ce65c3dc420f3d9211feb6d790bb71f6d389da8ccdbfabeebebede2aab3a.jpg)

![37bf5e1c1dd1372e9017c74d76a7dffd95f4e23c066ad1e6a7d85cc6e5ee007a.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/37bf5e1c1dd1372e9017c74d76a7dffd95f4e23c066ad1e6a7d85cc6e5ee007a.jpg)

![3ec42fb615f2efa4598be5367f13a0e335b67de93d6cfd9d3341ff54a974a1fd.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/3ec42fb615f2efa4598be5367f13a0e335b67de93d6cfd9d3341ff54a974a1fd.jpg)

![41e18171097087f0d9b8bd98a2497eca6dbe613029be74d9528047495daa17fd.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/41e18171097087f0d9b8bd98a2497eca6dbe613029be74d9528047495daa17fd.jpg)

![619b9ab8af74620079e5bdac44d9c840061369001e50ed8e3a79531058a59728.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/619b9ab8af74620079e5bdac44d9c840061369001e50ed8e3a79531058a59728.jpg)

![71ab22fb110eacb01bf472baf7c6ee8d64887d67fe3f3c7fccca6ae87981f1ac.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/71ab22fb110eacb01bf472baf7c6ee8d64887d67fe3f3c7fccca6ae87981f1ac.jpg)

![84f4f2a827bf6abc2d91e6d6e2abeb5202b1977f9d99c9e535dcfde589e12cfa.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/84f4f2a827bf6abc2d91e6d6e2abeb5202b1977f9d99c9e535dcfde589e12cfa.jpg)

![b3eb6dbac7ad98a39697ea00819d2744e7389f8f6860efee3f45908fe7a0a05e.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/b3eb6dbac7ad98a39697ea00819d2744e7389f8f6860efee3f45908fe7a0a05e.jpg)

![b3fb335850346a7a0f75c065a86f6b5d9d750753f35520b6ac048fbf262dcd0a.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/b3fb335850346a7a0f75c065a86f6b5d9d750753f35520b6ac048fbf262dcd0a.jpg)

![bd4ac127abae68c9da4fe546427e5fc18fa39b0d5b9b22b8a1fc64dfd64561e9.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/bd4ac127abae68c9da4fe546427e5fc18fa39b0d5b9b22b8a1fc64dfd64561e9.jpg)

![cfb36ffd3a92401a9b295b42b2691cc963bc19889977131d6604f6960ded7611.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/cfb36ffd3a92401a9b295b42b2691cc963bc19889977131d6604f6960ded7611.jpg)

![f60687fb539ef2a6456cd86d3a736f2c4a89a7d97c3979de771061e7d9bec260.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/f60687fb539ef2a6456cd86d3a736f2c4a89a7d97c3979de771061e7d9bec260.jpg)

![fb1afb1494ffcdcf2667a32044acf18c73c64dcf39af90b69a639429fdb7547c.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/fb1afb1494ffcdcf2667a32044acf18c73c64dcf39af90b69a639429fdb7547c.jpg)

![fff9b5b6ae1b5e9c3d7a5242f041b4e2b0028407182674c9a4de37a22a420e37.jpg](../icml_results/150_Jailbreak%20to%20Protect_%20Buffering%20and%20Reinforcing%20via%20Temporary%20Jailbreaking%20for%20Safe%20Fine-Tuning%20in%20L/tables/fff9b5b6ae1b5e9c3d7a5242f041b4e2b0028407182674c9a4de37a22a420e37.jpg)

<a id="paper-0149"></a>

## On the Optimization Trajectory of DeepWalk Embeddings


### Images

![08a71e1c9bb13211f224dd9dbadf08c333953c35a47cbe9efe13302897fcd772.jpg](../icml_results/151_On%20the%20Optimization%20Trajectory%20of%20DeepWalk%20Embeddings/images/08a71e1c9bb13211f224dd9dbadf08c333953c35a47cbe9efe13302897fcd772.jpg)

![3022ad73a4b80df16ee9823f4eb4a2027bc4981a195c96e0b9cdab789cdb3440.jpg](../icml_results/151_On%20the%20Optimization%20Trajectory%20of%20DeepWalk%20Embeddings/images/3022ad73a4b80df16ee9823f4eb4a2027bc4981a195c96e0b9cdab789cdb3440.jpg)

![4d2adaf379d5d2069b0770c45bd44ab6d503fad2425e181e2dcb3bd5183b4284.jpg](../icml_results/151_On%20the%20Optimization%20Trajectory%20of%20DeepWalk%20Embeddings/images/4d2adaf379d5d2069b0770c45bd44ab6d503fad2425e181e2dcb3bd5183b4284.jpg)

![6d958f96fa27c4cdf945b26163c8ddb2ccefc1cb90dee76b9382a1b604ff2ca7.jpg](../icml_results/151_On%20the%20Optimization%20Trajectory%20of%20DeepWalk%20Embeddings/images/6d958f96fa27c4cdf945b26163c8ddb2ccefc1cb90dee76b9382a1b604ff2ca7.jpg)

![991ff13b1e28361d2971a43954987200b4be8f284cb7bb430170242bbd4c76be.jpg](../icml_results/151_On%20the%20Optimization%20Trajectory%20of%20DeepWalk%20Embeddings/images/991ff13b1e28361d2971a43954987200b4be8f284cb7bb430170242bbd4c76be.jpg)

![e30c7d7a25acf84b23a8137b50f8d17aa2195070e23cef127c4fb3cf1e15399e.jpg](../icml_results/151_On%20the%20Optimization%20Trajectory%20of%20DeepWalk%20Embeddings/images/e30c7d7a25acf84b23a8137b50f8d17aa2195070e23cef127c4fb3cf1e15399e.jpg)

<a id="paper-0150"></a>

## From Feasible to Practical: Pareto-Optimal Synthesis Planning


### Images

![2f72d2b3462378be177fee3686dc88df96c77aa69e1444cb938faf49966d92b3.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/images/2f72d2b3462378be177fee3686dc88df96c77aa69e1444cb938faf49966d92b3.jpg)

![33e9c59519dd12d8d7697c13bda54ac8358e256711cbb326a92090d68f115271.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/images/33e9c59519dd12d8d7697c13bda54ac8358e256711cbb326a92090d68f115271.jpg)

![4c8d8f9645de2c4028ae2ffc834ba87a1dd9dd445220435300792af3891cd12d.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/images/4c8d8f9645de2c4028ae2ffc834ba87a1dd9dd445220435300792af3891cd12d.jpg)

![89bab69e0b0a15f6100e7acebe8147b68fdb267b045ff93936215e4941ae2240.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/images/89bab69e0b0a15f6100e7acebe8147b68fdb267b045ff93936215e4941ae2240.jpg)

![dfec9c9cf1ac5dbbab907ef95fc7aec802a348ffc18c67eecc2ef3ab7119420b.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/images/dfec9c9cf1ac5dbbab907ef95fc7aec802a348ffc18c67eecc2ef3ab7119420b.jpg)

![f2e1de4ec061ca7ae7c47c082c81c24f9181bb4df7084742bc54ca7aabb6066b.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/images/f2e1de4ec061ca7ae7c47c082c81c24f9181bb4df7084742bc54ca7aabb6066b.jpg)

### Tables

![07e83f6a4700540b5276288c7c2cd0039df5c7ffbcfe7a7a38766eba188c8277.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/tables/07e83f6a4700540b5276288c7c2cd0039df5c7ffbcfe7a7a38766eba188c8277.jpg)

![3650f25598264dfb8b5dc3a4f28308c6425a2241336262389a4cf8264584997e.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/tables/3650f25598264dfb8b5dc3a4f28308c6425a2241336262389a4cf8264584997e.jpg)

![37f321a9e7b9492540f6e641c261626526be5e5ca58ca2150c33368c751230e8.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/tables/37f321a9e7b9492540f6e641c261626526be5e5ca58ca2150c33368c751230e8.jpg)

![72690488e07bacdad906603ea7090123afbf1b9c61ff5e6e85ceef2a4bf0e910.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/tables/72690488e07bacdad906603ea7090123afbf1b9c61ff5e6e85ceef2a4bf0e910.jpg)

![8049a3300a5471e69f9c80f946d0073ff30411369c65dc66614256a65b712728.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/tables/8049a3300a5471e69f9c80f946d0073ff30411369c65dc66614256a65b712728.jpg)

![dac2c412d701f3a3bdb5185330b32a761d8f28283debdf017d37c42930cda1d9.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/tables/dac2c412d701f3a3bdb5185330b32a761d8f28283debdf017d37c42930cda1d9.jpg)

![dc126314dce69aeb4660184628370b4e2c0ab398d30545e9ad97cde4f1b0ab97.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/tables/dc126314dce69aeb4660184628370b4e2c0ab398d30545e9ad97cde4f1b0ab97.jpg)

![ed4f367711fd4478468f2487260a1040584bacf8fb041000373d03245628f3a6.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/tables/ed4f367711fd4478468f2487260a1040584bacf8fb041000373d03245628f3a6.jpg)

![f827a909cfdd900099205141e97d5700567e7bacd5e26c858000034766346478.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/tables/f827a909cfdd900099205141e97d5700567e7bacd5e26c858000034766346478.jpg)

![f9435e621228616878ea1f14c2d876975848baa49c9358c4b7a2dc51f1b26587.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/tables/f9435e621228616878ea1f14c2d876975848baa49c9358c4b7a2dc51f1b26587.jpg)

![fb59e19f85232f8beba8f2f632e956062ee8b0b11a17c230d5529f39fc27372e.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/tables/fb59e19f85232f8beba8f2f632e956062ee8b0b11a17c230d5529f39fc27372e.jpg)

![fed52baad808cc04041fa3bd77babb79b2ef6b9b9de545c13cc37cc95449dc57.jpg](../icml_results/152_From%20Feasible%20to%20Practical_%20Pareto-Optimal%20Synthesis%20Planning/tables/fed52baad808cc04041fa3bd77babb79b2ef6b9b9de545c13cc37cc95449dc57.jpg)

<a id="paper-0151"></a>

## Flow Sampling : Learning to Sample from Unnormalized Densities via Denoising Conditional Processes


### Images

![211c757e5a1b5ca57294f7b38ef49b71a0d25b418640094984b647d2e08e9469.jpg](../icml_results/153_Flow%20Sampling%20_%20Learning%20to%20Sample%20from%20Unnormalized%20Densities%20via%20Denoising%20Conditional%20Processes/images/211c757e5a1b5ca57294f7b38ef49b71a0d25b418640094984b647d2e08e9469.jpg)

![487fb8581fcb57f10f9f57ef01f4904b7e4c040834ca762bc9a9777a263bd743.jpg](../icml_results/153_Flow%20Sampling%20_%20Learning%20to%20Sample%20from%20Unnormalized%20Densities%20via%20Denoising%20Conditional%20Processes/images/487fb8581fcb57f10f9f57ef01f4904b7e4c040834ca762bc9a9777a263bd743.jpg)

![60c03b8a1b659ad8a5d29314c0df76bfcd4ef7aacbe32de19a2cbf14e627f86c.jpg](../icml_results/153_Flow%20Sampling%20_%20Learning%20to%20Sample%20from%20Unnormalized%20Densities%20via%20Denoising%20Conditional%20Processes/images/60c03b8a1b659ad8a5d29314c0df76bfcd4ef7aacbe32de19a2cbf14e627f86c.jpg)

![693da7177cf9aeed9ecc7e05eb55e423731edc1879d10520da8c6036c135da1e.jpg](../icml_results/153_Flow%20Sampling%20_%20Learning%20to%20Sample%20from%20Unnormalized%20Densities%20via%20Denoising%20Conditional%20Processes/images/693da7177cf9aeed9ecc7e05eb55e423731edc1879d10520da8c6036c135da1e.jpg)

![9ddfdb9710ff21dd7047ce9f4fa91829bd17fecb2e38e8ff4ebaf8a2711c31c8.jpg](../icml_results/153_Flow%20Sampling%20_%20Learning%20to%20Sample%20from%20Unnormalized%20Densities%20via%20Denoising%20Conditional%20Processes/images/9ddfdb9710ff21dd7047ce9f4fa91829bd17fecb2e38e8ff4ebaf8a2711c31c8.jpg)

### Tables

![0c7a550248b779f5b4840bf2ff32bbb805c7005a8477c8b600bff5af435ee930.jpg](../icml_results/153_Flow%20Sampling%20_%20Learning%20to%20Sample%20from%20Unnormalized%20Densities%20via%20Denoising%20Conditional%20Processes/tables/0c7a550248b779f5b4840bf2ff32bbb805c7005a8477c8b600bff5af435ee930.jpg)

![17f45135f538ee80d7c92844569c3be19072c100cefae0a0aa4a199692da18e5.jpg](../icml_results/153_Flow%20Sampling%20_%20Learning%20to%20Sample%20from%20Unnormalized%20Densities%20via%20Denoising%20Conditional%20Processes/tables/17f45135f538ee80d7c92844569c3be19072c100cefae0a0aa4a199692da18e5.jpg)

![18b110b928fe0a35c034293db3c94591d5472bf515bbbf277d0d4e02f12e61b1.jpg](../icml_results/153_Flow%20Sampling%20_%20Learning%20to%20Sample%20from%20Unnormalized%20Densities%20via%20Denoising%20Conditional%20Processes/tables/18b110b928fe0a35c034293db3c94591d5472bf515bbbf277d0d4e02f12e61b1.jpg)

![418bfebfb5a26968a95fadbcb6949e9b306210e91e39bf98fc5bb97e06e3e750.jpg](../icml_results/153_Flow%20Sampling%20_%20Learning%20to%20Sample%20from%20Unnormalized%20Densities%20via%20Denoising%20Conditional%20Processes/tables/418bfebfb5a26968a95fadbcb6949e9b306210e91e39bf98fc5bb97e06e3e750.jpg)

![7462444b8abefe20e64299ebcd987bf874e7af8cfbd704e6df82b2cc4c6a1d7d.jpg](../icml_results/153_Flow%20Sampling%20_%20Learning%20to%20Sample%20from%20Unnormalized%20Densities%20via%20Denoising%20Conditional%20Processes/tables/7462444b8abefe20e64299ebcd987bf874e7af8cfbd704e6df82b2cc4c6a1d7d.jpg)

<a id="paper-0152"></a>

## Vision2Web: A Hierarchical Benchmark for Visual Website Development with Agent Verification


### Images

![0473ff53fb2ccec6eb7345d69ad24796562bcf53dc84faf1a0f8ad132535b84c.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/images/0473ff53fb2ccec6eb7345d69ad24796562bcf53dc84faf1a0f8ad132535b84c.jpg)

![06a9ab630b7f313735f4d35e2c26128e33ad05df6fe9894a85170d43bc2f0a72.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/images/06a9ab630b7f313735f4d35e2c26128e33ad05df6fe9894a85170d43bc2f0a72.jpg)

![0a1706c79bbbe8a01de77c37521c499015b577a0e1b9e113b5a2660584baa431.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/images/0a1706c79bbbe8a01de77c37521c499015b577a0e1b9e113b5a2660584baa431.jpg)

![4e25605daec8fdc09ba33243075019df4ebe73e4db1bc88bb2380f026466e61d.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/images/4e25605daec8fdc09ba33243075019df4ebe73e4db1bc88bb2380f026466e61d.jpg)

![5cb4b27f03ec10f09f3d9d597df3e30599ca359c47a5836616fb8462306a746b.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/images/5cb4b27f03ec10f09f3d9d597df3e30599ca359c47a5836616fb8462306a746b.jpg)

![63b51f6aac2ab78f0eef308370e7fdf7266c6b608ab463d5e9b75171fa57bc7d.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/images/63b51f6aac2ab78f0eef308370e7fdf7266c6b608ab463d5e9b75171fa57bc7d.jpg)

![9174c7149c7516aff909f6a16731ab6bc2d0d052b48eb345fbd3c4e05c311277.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/images/9174c7149c7516aff909f6a16731ab6bc2d0d052b48eb345fbd3c4e05c311277.jpg)

![a86e5d76675cf5b1dd7f057ea67f3cb851907196b750b78e7454e6ea1b13f5a6.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/images/a86e5d76675cf5b1dd7f057ea67f3cb851907196b750b78e7454e6ea1b13f5a6.jpg)

![d1963a149802d3c456a334b2bf27caee52eacfc206b563eec4cd7ea98209140d.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/images/d1963a149802d3c456a334b2bf27caee52eacfc206b563eec4cd7ea98209140d.jpg)

![e2a6c8dcdf2321a00f8b71fd75d46af303ddb7efd33e4f5ba6412ee131363262.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/images/e2a6c8dcdf2321a00f8b71fd75d46af303ddb7efd33e4f5ba6412ee131363262.jpg)

![f7afaec82432efae70187ebbbe90e44cb4febf56b90d82537e8b3c776219b09b.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/images/f7afaec82432efae70187ebbbe90e44cb4febf56b90d82537e8b3c776219b09b.jpg)

![fde8993796a9982e01b42dece1d0225f904b0737dc6b7073b1a18195166969e2.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/images/fde8993796a9982e01b42dece1d0225f904b0737dc6b7073b1a18195166969e2.jpg)

### Tables

![20e7967bfe3323a9974598871e3fe2867008e4d7fce6b319914e5b469fa3d736.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/tables/20e7967bfe3323a9974598871e3fe2867008e4d7fce6b319914e5b469fa3d736.jpg)

![2ecfb771572af0c35ad02bd09432bed2cb3dd61609162f1fbe2480d4d0aa3169.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/tables/2ecfb771572af0c35ad02bd09432bed2cb3dd61609162f1fbe2480d4d0aa3169.jpg)

![3bd7a954621a04dbb209a10ecfaf6fa09e1b344dea713e7df8453fa707e6bb25.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/tables/3bd7a954621a04dbb209a10ecfaf6fa09e1b344dea713e7df8453fa707e6bb25.jpg)

![4dd4f8ffced22a21517b694f54127ee0e5642f06511677dbbbb001c14a1240d2.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/tables/4dd4f8ffced22a21517b694f54127ee0e5642f06511677dbbbb001c14a1240d2.jpg)

![5abf310f7ca444d1e0436dd30bc975d2b5086fdd4379ab3225b4dd33675cc3b8.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/tables/5abf310f7ca444d1e0436dd30bc975d2b5086fdd4379ab3225b4dd33675cc3b8.jpg)

![65670bf5996386279f8fd085bfb285605d59991e26c7cb398a98ca1800c1d36f.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/tables/65670bf5996386279f8fd085bfb285605d59991e26c7cb398a98ca1800c1d36f.jpg)

![684981e06af897c7bb4f7cedbaf66d1384fd645eb19184dd165d3cc98f62cbb7.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/tables/684981e06af897c7bb4f7cedbaf66d1384fd645eb19184dd165d3cc98f62cbb7.jpg)

![91ec5ca6126416ad4f02e933d867f5f95cdab912deb5c288f7d4f9185523970b.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/tables/91ec5ca6126416ad4f02e933d867f5f95cdab912deb5c288f7d4f9185523970b.jpg)

![924095f47580ef85162275cfbf8f879fb06f6c55b15d82e756070393a8cc9271.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/tables/924095f47580ef85162275cfbf8f879fb06f6c55b15d82e756070393a8cc9271.jpg)

![a7cfee2e7000fda746b7c4f818ecd68be097f3ca2db3f5fd87dd126d482123d7.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/tables/a7cfee2e7000fda746b7c4f818ecd68be097f3ca2db3f5fd87dd126d482123d7.jpg)

![c366af8ebcf3867c2f5feb1a7ad689bf2fd269574696d0c548bd66b83fd30471.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/tables/c366af8ebcf3867c2f5feb1a7ad689bf2fd269574696d0c548bd66b83fd30471.jpg)

![f760d9eb975828dc610fcfd1b0210a91483117035590f6bf9968977c3d1e357b.jpg](../icml_results/154_Vision2Web_%20A%20Hierarchical%20Benchmark%20for%20Visual%20Website%20Development%20with%20Agent%20Verification/tables/f760d9eb975828dc610fcfd1b0210a91483117035590f6bf9968977c3d1e357b.jpg)

<a id="paper-0153"></a>

## FedPissa: Towards Federated Personalized Adaptation of Foundation Models via LoRA Subspace Mapping


### Images

![0066c433f9713e0f6f712248c2a6aebf79d50aafbde5592edf5ab706f2495793.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/images/0066c433f9713e0f6f712248c2a6aebf79d50aafbde5592edf5ab706f2495793.jpg)

![29cb445ad5e523e2553fbc88b2e471db594dcd47ce5fcb714971c9b6b86c5708.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/images/29cb445ad5e523e2553fbc88b2e471db594dcd47ce5fcb714971c9b6b86c5708.jpg)

![91eb0119ca4845e039c3c404d49a50cb6ab4d519c9c97bd5e20059b941f5605b.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/images/91eb0119ca4845e039c3c404d49a50cb6ab4d519c9c97bd5e20059b941f5605b.jpg)

![a8052a99e64047956b24c5eea10dcda42e0ec65277ff8446f4a93c44da9db292.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/images/a8052a99e64047956b24c5eea10dcda42e0ec65277ff8446f4a93c44da9db292.jpg)

![bc165c1c818ec6915280865916e9bb3766c885a6177a40e324966c51ee9388c8.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/images/bc165c1c818ec6915280865916e9bb3766c885a6177a40e324966c51ee9388c8.jpg)

![cfb88bc2b7a36b250773306418031faa9ecf3d4e42e1bf1f5fbeeb54d20c1e08.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/images/cfb88bc2b7a36b250773306418031faa9ecf3d4e42e1bf1f5fbeeb54d20c1e08.jpg)

![d2e9fdc775427ec8661ff79e7317ab34f67c0085cb73640a9cbc6a9edcba040d.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/images/d2e9fdc775427ec8661ff79e7317ab34f67c0085cb73640a9cbc6a9edcba040d.jpg)

![d99a6330d6a49b0656362d251d527bdc9832b6656695074d97de025afdc3922b.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/images/d99a6330d6a49b0656362d251d527bdc9832b6656695074d97de025afdc3922b.jpg)

![f81f887681b2a02a00d964ed8e1cfda7eb5ee629bf78e11ff6c27212d311a7a6.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/images/f81f887681b2a02a00d964ed8e1cfda7eb5ee629bf78e11ff6c27212d311a7a6.jpg)

![fac3aff2d3ca39dead93281936f78689dbf545b3720e8846bbe0d8802235f564.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/images/fac3aff2d3ca39dead93281936f78689dbf545b3720e8846bbe0d8802235f564.jpg)

### Tables

![1105414084b6142d9f3db3d63f6c14c12f36bc0dda5dffa9b883e424da7da6f8.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/tables/1105414084b6142d9f3db3d63f6c14c12f36bc0dda5dffa9b883e424da7da6f8.jpg)

![1b38ff685a1d037961e76b4d816cad350d36892128d1b77faf790444eb1dcf51.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/tables/1b38ff685a1d037961e76b4d816cad350d36892128d1b77faf790444eb1dcf51.jpg)

![24934e8848f032cdce86bea3269d4530f5bf01cf2bc823a7442f55b94c589fcb.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/tables/24934e8848f032cdce86bea3269d4530f5bf01cf2bc823a7442f55b94c589fcb.jpg)

![36736e3a2eb2620ea27fe89e8c48ede330e35398d7fd8a485baa55ef5232ef6e.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/tables/36736e3a2eb2620ea27fe89e8c48ede330e35398d7fd8a485baa55ef5232ef6e.jpg)

![3855129a8dbbb2cd35b1adf4f021567306bafa77dc5ed0f48dfd8b124d605228.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/tables/3855129a8dbbb2cd35b1adf4f021567306bafa77dc5ed0f48dfd8b124d605228.jpg)

![3a2d301f7b1bebd19e8957c89cdad7691ae354ec23ba8e51889d107b730af83e.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/tables/3a2d301f7b1bebd19e8957c89cdad7691ae354ec23ba8e51889d107b730af83e.jpg)

![47ab12a88f353ff2ed7e31d000d7d6ec70f90a9d8e523130203cef8ea95be8a5.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/tables/47ab12a88f353ff2ed7e31d000d7d6ec70f90a9d8e523130203cef8ea95be8a5.jpg)

![6bcd190aa117324f4a13ddb1bb6113a373ec02bec1809c1d616311fcd1bcab65.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/tables/6bcd190aa117324f4a13ddb1bb6113a373ec02bec1809c1d616311fcd1bcab65.jpg)

![6ca89e0332783ac4623630199631026a0fcae0a28bcde57c5fe7ee5543708d99.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/tables/6ca89e0332783ac4623630199631026a0fcae0a28bcde57c5fe7ee5543708d99.jpg)

![ac4b243c4eb307c8d96db63405c2e44ffb3aac2ed6984455f9f81646c12e5890.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/tables/ac4b243c4eb307c8d96db63405c2e44ffb3aac2ed6984455f9f81646c12e5890.jpg)

![adecec7e073a8afe219a41ce2f7c14bdbec5a62a9af22057ce4c0411d55d14b5.jpg](../icml_results/155_FedPissa_%20Towards%20Federated%20Personalized%20Adaptation%20of%20Foundation%20Models%20via%20LoRA%20Subspace%20Mapping/tables/adecec7e073a8afe219a41ce2f7c14bdbec5a62a9af22057ce4c0411d55d14b5.jpg)

<a id="paper-0154"></a>

## LiftQuant: Continuous Bit-Width LLM via Dimensional Lifting and Projection


### Images

![4923379f8b8c89d7a81cff0fe54fd0bb833765ec8682f406af13fdd236d89e0e.jpg](../icml_results/156_LiftQuant_%20Continuous%20Bit-Width%20LLM%20via%20Dimensional%20Lifting%20and%20Projection/images/4923379f8b8c89d7a81cff0fe54fd0bb833765ec8682f406af13fdd236d89e0e.jpg)

![743ee122ff14775d269707165ac42a809fb9aaba7f26f903c36dca8f5e9f93cf.jpg](../icml_results/156_LiftQuant_%20Continuous%20Bit-Width%20LLM%20via%20Dimensional%20Lifting%20and%20Projection/images/743ee122ff14775d269707165ac42a809fb9aaba7f26f903c36dca8f5e9f93cf.jpg)

![87001f959b1110409f25e49bc8ad4d941385203847c069d88d857cfcdcde3877.jpg](../icml_results/156_LiftQuant_%20Continuous%20Bit-Width%20LLM%20via%20Dimensional%20Lifting%20and%20Projection/images/87001f959b1110409f25e49bc8ad4d941385203847c069d88d857cfcdcde3877.jpg)

![c6d9714e8df77ec2ec3edea9b0e778d996722302ada23cdd70b39872bf798655.jpg](../icml_results/156_LiftQuant_%20Continuous%20Bit-Width%20LLM%20via%20Dimensional%20Lifting%20and%20Projection/images/c6d9714e8df77ec2ec3edea9b0e778d996722302ada23cdd70b39872bf798655.jpg)

### Tables

![44b753c2b39e1fe43d65d53a9fd06606e869698910517a0e84e0e9385642eb9e.jpg](../icml_results/156_LiftQuant_%20Continuous%20Bit-Width%20LLM%20via%20Dimensional%20Lifting%20and%20Projection/tables/44b753c2b39e1fe43d65d53a9fd06606e869698910517a0e84e0e9385642eb9e.jpg)

![66be719875b19294d5c45d5afc356ca77df1cc1c66a577e934eb616142246035.jpg](../icml_results/156_LiftQuant_%20Continuous%20Bit-Width%20LLM%20via%20Dimensional%20Lifting%20and%20Projection/tables/66be719875b19294d5c45d5afc356ca77df1cc1c66a577e934eb616142246035.jpg)

![7cd1814aaeffb62e74272cb9e1f9b503d984f098fa09660d70db7e5cf799492f.jpg](../icml_results/156_LiftQuant_%20Continuous%20Bit-Width%20LLM%20via%20Dimensional%20Lifting%20and%20Projection/tables/7cd1814aaeffb62e74272cb9e1f9b503d984f098fa09660d70db7e5cf799492f.jpg)

![9b603072580ef1106af6dc83bfb55443398e6eb863e2a4ce3e4893125d528a5f.jpg](../icml_results/156_LiftQuant_%20Continuous%20Bit-Width%20LLM%20via%20Dimensional%20Lifting%20and%20Projection/tables/9b603072580ef1106af6dc83bfb55443398e6eb863e2a4ce3e4893125d528a5f.jpg)

![c44998e9735697515be92b2791217d5c01dc897481798ba888ca6395666c8c0c.jpg](../icml_results/156_LiftQuant_%20Continuous%20Bit-Width%20LLM%20via%20Dimensional%20Lifting%20and%20Projection/tables/c44998e9735697515be92b2791217d5c01dc897481798ba888ca6395666c8c0c.jpg)

![c4e88611f1beb13545dbe6b5a1ae4e9686c3c462556735a00831e7110242e7c0.jpg](../icml_results/156_LiftQuant_%20Continuous%20Bit-Width%20LLM%20via%20Dimensional%20Lifting%20and%20Projection/tables/c4e88611f1beb13545dbe6b5a1ae4e9686c3c462556735a00831e7110242e7c0.jpg)

![e91709ca19d8903cfa09a47741a6f244cb881a3e95b4fe3e84c3fd4ca5e3c212.jpg](../icml_results/156_LiftQuant_%20Continuous%20Bit-Width%20LLM%20via%20Dimensional%20Lifting%20and%20Projection/tables/e91709ca19d8903cfa09a47741a6f244cb881a3e95b4fe3e84c3fd4ca5e3c212.jpg)

![f378cad14f343582a62a005cb3135ea9919d34f84df7496b81b3199df8417da4.jpg](../icml_results/156_LiftQuant_%20Continuous%20Bit-Width%20LLM%20via%20Dimensional%20Lifting%20and%20Projection/tables/f378cad14f343582a62a005cb3135ea9919d34f84df7496b81b3199df8417da4.jpg)

![fe7e5bd8ac183fea97c5a24119cba52b03652e61deccbb91831a08efc90b4244.jpg](../icml_results/156_LiftQuant_%20Continuous%20Bit-Width%20LLM%20via%20Dimensional%20Lifting%20and%20Projection/tables/fe7e5bd8ac183fea97c5a24119cba52b03652e61deccbb91831a08efc90b4244.jpg)

<a id="paper-0155"></a>

## CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal


### Images

![11140e09a7ce901fbbd84637078defdfdbf6d0c4570da8d24601ca55952b8928.jpg](../icml_results/157_CLEAR_%20Context-Aware%20Learning%20with%20End-to-End%20Mask-Free%20Inference%20for%20Adaptive%20Video%20Subtitle%20Remova/images/11140e09a7ce901fbbd84637078defdfdbf6d0c4570da8d24601ca55952b8928.jpg)

![2ad13e1e4fbc2bc356a303466c9ca586e924f8ca49cc6406201f1c818e777aa2.jpg](../icml_results/157_CLEAR_%20Context-Aware%20Learning%20with%20End-to-End%20Mask-Free%20Inference%20for%20Adaptive%20Video%20Subtitle%20Remova/images/2ad13e1e4fbc2bc356a303466c9ca586e924f8ca49cc6406201f1c818e777aa2.jpg)

![2e2d780a425e65b836a0e3852638c3f4fe0f466f80242886dcc29e6b3c32e0b4.jpg](../icml_results/157_CLEAR_%20Context-Aware%20Learning%20with%20End-to-End%20Mask-Free%20Inference%20for%20Adaptive%20Video%20Subtitle%20Remova/images/2e2d780a425e65b836a0e3852638c3f4fe0f466f80242886dcc29e6b3c32e0b4.jpg)

![34e833eeddfe4a080c5970f3e325a36f1f43ea969fce8b1514856e643c6a9a56.jpg](../icml_results/157_CLEAR_%20Context-Aware%20Learning%20with%20End-to-End%20Mask-Free%20Inference%20for%20Adaptive%20Video%20Subtitle%20Remova/images/34e833eeddfe4a080c5970f3e325a36f1f43ea969fce8b1514856e643c6a9a56.jpg)

![3c9386c81af51a8d521a74721e307ce51fe583bb86c235925cd599c7a405470d.jpg](../icml_results/157_CLEAR_%20Context-Aware%20Learning%20with%20End-to-End%20Mask-Free%20Inference%20for%20Adaptive%20Video%20Subtitle%20Remova/images/3c9386c81af51a8d521a74721e307ce51fe583bb86c235925cd599c7a405470d.jpg)

![72fc242bdd47e4c8a84763df1332b5107c60b2d5c576c3105faeaec6f3877971.jpg](../icml_results/157_CLEAR_%20Context-Aware%20Learning%20with%20End-to-End%20Mask-Free%20Inference%20for%20Adaptive%20Video%20Subtitle%20Remova/images/72fc242bdd47e4c8a84763df1332b5107c60b2d5c576c3105faeaec6f3877971.jpg)

![88cef38b2dfad34ded6a0010de42ed62c7e7788c22bccd37790b1438331611b6.jpg](../icml_results/157_CLEAR_%20Context-Aware%20Learning%20with%20End-to-End%20Mask-Free%20Inference%20for%20Adaptive%20Video%20Subtitle%20Remova/images/88cef38b2dfad34ded6a0010de42ed62c7e7788c22bccd37790b1438331611b6.jpg)

![a231caf6af7329d12b65c44a6a58681af250d06f11be8e49100e4cfced9b2188.jpg](../icml_results/157_CLEAR_%20Context-Aware%20Learning%20with%20End-to-End%20Mask-Free%20Inference%20for%20Adaptive%20Video%20Subtitle%20Remova/images/a231caf6af7329d12b65c44a6a58681af250d06f11be8e49100e4cfced9b2188.jpg)

![b74095d5dfb07bde309f521275e0124f7a376142418978b1fd34c63dc83d15c3.jpg](../icml_results/157_CLEAR_%20Context-Aware%20Learning%20with%20End-to-End%20Mask-Free%20Inference%20for%20Adaptive%20Video%20Subtitle%20Remova/images/b74095d5dfb07bde309f521275e0124f7a376142418978b1fd34c63dc83d15c3.jpg)

![dadaa2bcb1501ca77f7b0b46b2779fbfc43b1033d7da05a511dd55ebe3ebf62a.jpg](../icml_results/157_CLEAR_%20Context-Aware%20Learning%20with%20End-to-End%20Mask-Free%20Inference%20for%20Adaptive%20Video%20Subtitle%20Remova/images/dadaa2bcb1501ca77f7b0b46b2779fbfc43b1033d7da05a511dd55ebe3ebf62a.jpg)

### Tables

![4d76d3ce803ab20b4c3972e4c259f26fa6f57208a79f0ce94a301d5edf7d497d.jpg](../icml_results/157_CLEAR_%20Context-Aware%20Learning%20with%20End-to-End%20Mask-Free%20Inference%20for%20Adaptive%20Video%20Subtitle%20Remova/tables/4d76d3ce803ab20b4c3972e4c259f26fa6f57208a79f0ce94a301d5edf7d497d.jpg)

![897596a86d1fb657b2452954a4316f7e742ea156c8ef63ac3ae6d44c53a1ade5.jpg](../icml_results/157_CLEAR_%20Context-Aware%20Learning%20with%20End-to-End%20Mask-Free%20Inference%20for%20Adaptive%20Video%20Subtitle%20Remova/tables/897596a86d1fb657b2452954a4316f7e742ea156c8ef63ac3ae6d44c53a1ade5.jpg)

![95ce6d15029aac3255280034c936060c78b25e635e1a0021e2e80e3b434f72ae.jpg](../icml_results/157_CLEAR_%20Context-Aware%20Learning%20with%20End-to-End%20Mask-Free%20Inference%20for%20Adaptive%20Video%20Subtitle%20Remova/tables/95ce6d15029aac3255280034c936060c78b25e635e1a0021e2e80e3b434f72ae.jpg)

<a id="paper-0156"></a>

## ClinTutor-R1: Advancing Scalable and Robust One-to-Many Alignment in Clinical Socratic Education


### Images

![0fc8ad82395dfea47591ce08fbc1f0f21f60e688d53bdb68aa7847582b9ac8d5.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/images/0fc8ad82395dfea47591ce08fbc1f0f21f60e688d53bdb68aa7847582b9ac8d5.jpg)

![15c20e046162b166e274196a3aceb467782932662234241f15833a329e136e42.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/images/15c20e046162b166e274196a3aceb467782932662234241f15833a329e136e42.jpg)

![2fb68bcc3aebc5d44477e2de395976e4b9a9b64342cb1febac52d8a333456bfc.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/images/2fb68bcc3aebc5d44477e2de395976e4b9a9b64342cb1febac52d8a333456bfc.jpg)

![3a631b4d280d877585f6d725c801916b22bc7106cb0d6898a979f26600d5429a.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/images/3a631b4d280d877585f6d725c801916b22bc7106cb0d6898a979f26600d5429a.jpg)

![4401dc1ec9e377a0ba86ecdb1cb4ef39117871b2c6adbd7bd849d77a422d5963.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/images/4401dc1ec9e377a0ba86ecdb1cb4ef39117871b2c6adbd7bd849d77a422d5963.jpg)

![674dde1ae8d3097473f3dbf5dffd928b2b7a761e0e14897bc4a8f40c1148e187.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/images/674dde1ae8d3097473f3dbf5dffd928b2b7a761e0e14897bc4a8f40c1148e187.jpg)

![6af45dbd882d35ab43dc3d3b1ffafaf46a0f8a2b5596019b2a02e1eafda81432.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/images/6af45dbd882d35ab43dc3d3b1ffafaf46a0f8a2b5596019b2a02e1eafda81432.jpg)

![73e75015b4122d72c84c95c93652714247e5dec34aa0a3d875ab86cd1f71700a.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/images/73e75015b4122d72c84c95c93652714247e5dec34aa0a3d875ab86cd1f71700a.jpg)

![7524fccfb258020f9f019b70d83f275edf41ad9b3ecaf518a8ada42edc8fc5eb.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/images/7524fccfb258020f9f019b70d83f275edf41ad9b3ecaf518a8ada42edc8fc5eb.jpg)

![878c2ebe76236b9c9e99abf9575ca61e27ca2ced80335364d633eaaa7df94cf7.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/images/878c2ebe76236b9c9e99abf9575ca61e27ca2ced80335364d633eaaa7df94cf7.jpg)

![9142e8384a95f3071324731a297805bd5cdbf81eab960a22bdeac85a0ff15566.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/images/9142e8384a95f3071324731a297805bd5cdbf81eab960a22bdeac85a0ff15566.jpg)

![c3cbf9ea3e28af81e37271d9514064dc5f711d0c25b1bf53635cf07809ac2643.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/images/c3cbf9ea3e28af81e37271d9514064dc5f711d0c25b1bf53635cf07809ac2643.jpg)

![dd2ee30786a24c36c7a987f4536567ac15e7781803d971f51ef3b67e771083a7.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/images/dd2ee30786a24c36c7a987f4536567ac15e7781803d971f51ef3b67e771083a7.jpg)

### Tables

![24882ab6bd5a9704afe4a5472f8009a00fece9fe5586a4cfe39d9a59723c55bc.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/tables/24882ab6bd5a9704afe4a5472f8009a00fece9fe5586a4cfe39d9a59723c55bc.jpg)

![2cb57ce326273af4258cdf04f95f97ae83eab04edddf8caa73b99619bd6ddf62.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/tables/2cb57ce326273af4258cdf04f95f97ae83eab04edddf8caa73b99619bd6ddf62.jpg)

![2dd9dc5e7070cd1bcbd34271f7e2b27a602cfa8f44c8bdd3f6e2c3f8bcdd91d5.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/tables/2dd9dc5e7070cd1bcbd34271f7e2b27a602cfa8f44c8bdd3f6e2c3f8bcdd91d5.jpg)

![3388a9eb02bcb10eb1c850485cc29b78c4a2988d7a182c49084652591f4bdb84.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/tables/3388a9eb02bcb10eb1c850485cc29b78c4a2988d7a182c49084652591f4bdb84.jpg)

![36672eb6f64e9b62fb21a72fe1ca5aa51a2a55ec63b54d5caaf2d67e3972b500.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/tables/36672eb6f64e9b62fb21a72fe1ca5aa51a2a55ec63b54d5caaf2d67e3972b500.jpg)

![4c220200b1a5356fc7b55544adba7a9b3ae3f5ba7fd8c2a8c51a518665ee0d79.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/tables/4c220200b1a5356fc7b55544adba7a9b3ae3f5ba7fd8c2a8c51a518665ee0d79.jpg)

![6497fc767398421056b3711bd0f35a33eaae18ff4654cf7e4db7d3fea0c791cb.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/tables/6497fc767398421056b3711bd0f35a33eaae18ff4654cf7e4db7d3fea0c791cb.jpg)

![9d56d3f1c3cf1b5a462f25ac726d2f4223c0543d4d7720402d1d483645ec6df3.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/tables/9d56d3f1c3cf1b5a462f25ac726d2f4223c0543d4d7720402d1d483645ec6df3.jpg)

![ae57a4d68b8c74cfb9fe39bd58a75631434a104de89179eb01a23478f6008fb4.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/tables/ae57a4d68b8c74cfb9fe39bd58a75631434a104de89179eb01a23478f6008fb4.jpg)

![be81912ac09ef05834b005a79b24f410f5f7fb3b9283f082c188d77a576915e2.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/tables/be81912ac09ef05834b005a79b24f410f5f7fb3b9283f082c188d77a576915e2.jpg)

![cee7d4ce0c53d927f0c059215e1219e9689f5421efd552cc3aeeb06749f838af.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/tables/cee7d4ce0c53d927f0c059215e1219e9689f5421efd552cc3aeeb06749f838af.jpg)

![f1d588489cd6907b6addde67c2aa70b8d230be9fc32652830029349d2fa506b3.jpg](../icml_results/158_ClinTutor-R1_%20Advancing%20Scalable%20and%20Robust%20One-to-Many%20Alignment%20in%20Clinical%20Socratic%20Education/tables/f1d588489cd6907b6addde67c2aa70b8d230be9fc32652830029349d2fa506b3.jpg)

<a id="paper-0157"></a>

## Joint-Space Empowerment for Dexterous Coordination in Tendon-Driven Hands


### Images

![21618ff81fd10fb24019f7ae7840305cdf1b22757a0d807291b5753098a918a8.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/images/21618ff81fd10fb24019f7ae7840305cdf1b22757a0d807291b5753098a918a8.jpg)

![3223ac5c573083b07575aec62a047af72935144ac4475a13e4d432cb508f9cc9.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/images/3223ac5c573083b07575aec62a047af72935144ac4475a13e4d432cb508f9cc9.jpg)

![479ce821ee6131c6cf0a0a9b5454cab0e9f337b718de8ff69c9457a6a60f574e.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/images/479ce821ee6131c6cf0a0a9b5454cab0e9f337b718de8ff69c9457a6a60f574e.jpg)

![4a082439bfe1363a40024b528ffb91d93538979634c2e32b6aa20630135f84f8.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/images/4a082439bfe1363a40024b528ffb91d93538979634c2e32b6aa20630135f84f8.jpg)

![53901ae07ed9d33288b5f7d6b4676ef7dad2d2f685af586c6f5b1af34b169d47.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/images/53901ae07ed9d33288b5f7d6b4676ef7dad2d2f685af586c6f5b1af34b169d47.jpg)

![643f341d140721d2059bf544231ee873a9e11a8fb4ab0e888ec403294d2b2a42.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/images/643f341d140721d2059bf544231ee873a9e11a8fb4ab0e888ec403294d2b2a42.jpg)

![7958b0c415e69a39cdcf9559593830d88ac1c55cf23ea89ebbcf830bc7fc8b8a.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/images/7958b0c415e69a39cdcf9559593830d88ac1c55cf23ea89ebbcf830bc7fc8b8a.jpg)

![99eb8e74f74e815439d1859f25ab7ceda815bdbf9da300d107bedec451f33bec.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/images/99eb8e74f74e815439d1859f25ab7ceda815bdbf9da300d107bedec451f33bec.jpg)

![9d5ff98d5eb36c2b31cde28de82f13f1e18a4919fa30377e937c1cd532b97e9d.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/images/9d5ff98d5eb36c2b31cde28de82f13f1e18a4919fa30377e937c1cd532b97e9d.jpg)

![c19e8dbbf990e837b16f900ff72cce25f5dda7a01c2a8f2d9586f1b05fa2e499.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/images/c19e8dbbf990e837b16f900ff72cce25f5dda7a01c2a8f2d9586f1b05fa2e499.jpg)

![c5a7ede4417fa046220c7943417b98833fa51ae436478676adcb1fd7eb816ca6.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/images/c5a7ede4417fa046220c7943417b98833fa51ae436478676adcb1fd7eb816ca6.jpg)

![daf26a21c9bfcd69cfa373cc3a1a57c6737337a99c0ca615c5825fa309896468.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/images/daf26a21c9bfcd69cfa373cc3a1a57c6737337a99c0ca615c5825fa309896468.jpg)

### Tables

![17467a8bacc06abbfba339818b5b32b985b726216420670fd34d10b0fd0d08fa.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/tables/17467a8bacc06abbfba339818b5b32b985b726216420670fd34d10b0fd0d08fa.jpg)

![3176a651630fa10db9408c74b3df0a909f33dbafde55dc3ec17023e44a9f2f8e.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/tables/3176a651630fa10db9408c74b3df0a909f33dbafde55dc3ec17023e44a9f2f8e.jpg)

![3e89db4a25847732f01f39b4322ea267f538b7de896d1c594cd1237ca298f37f.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/tables/3e89db4a25847732f01f39b4322ea267f538b7de896d1c594cd1237ca298f37f.jpg)

![5f903c065587417fe85a34acea2b43c981cf1cda1a2b585220678b494ff2f05c.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/tables/5f903c065587417fe85a34acea2b43c981cf1cda1a2b585220678b494ff2f05c.jpg)

![62a6550119bfc1cab7752744341daee45ef6c25efcfb7ed72e3eb8809757d6d8.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/tables/62a6550119bfc1cab7752744341daee45ef6c25efcfb7ed72e3eb8809757d6d8.jpg)

![755db000711e08ead05a7d47bb400a819dffe55459aab4a97027ed6a865f4e6f.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/tables/755db000711e08ead05a7d47bb400a819dffe55459aab4a97027ed6a865f4e6f.jpg)

![838ed8eca9e90bbe3454c7db3dfa826dfcd09104c6c6d5e1309017f452bd1d7e.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/tables/838ed8eca9e90bbe3454c7db3dfa826dfcd09104c6c6d5e1309017f452bd1d7e.jpg)

![898e2a65156014a78aa1b07c3a779ea71354338ff49aea47b81adb48472dfed6.jpg](../icml_results/159_Joint-Space%20Empowerment%20for%20Dexterous%20Coordination%20in%20Tendon-Driven%20Hands/tables/898e2a65156014a78aa1b07c3a779ea71354338ff49aea47b81adb48472dfed6.jpg)

<a id="paper-0158"></a>

## Neural Feature Geometry Evolves as Discrete Ricci Flow


### Images

![0c9de27ed44508195b6cc15c467dd16e94c20eac0d56d051d793dc2815d6e5d7.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/images/0c9de27ed44508195b6cc15c467dd16e94c20eac0d56d051d793dc2815d6e5d7.jpg)

![15fdbf09e3c41d2def7ae07768e365ef438a695924148ee5db5bed2b37b1a668.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/images/15fdbf09e3c41d2def7ae07768e365ef438a695924148ee5db5bed2b37b1a668.jpg)

![1f215aa5d7bb77b0f32b3cb61cf05b0ad45ba92481ba52ce3c6799b5c13761ef.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/images/1f215aa5d7bb77b0f32b3cb61cf05b0ad45ba92481ba52ce3c6799b5c13761ef.jpg)

![30ea2179f1e8f37c2ccff72e5b1c25dfc83c51b68c1cbec95f81f4f0c3fb319f.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/images/30ea2179f1e8f37c2ccff72e5b1c25dfc83c51b68c1cbec95f81f4f0c3fb319f.jpg)

![37916f9b975cc363711344b92f7449618f4105e5f976ee9ec6812dc523ce9d6f.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/images/37916f9b975cc363711344b92f7449618f4105e5f976ee9ec6812dc523ce9d6f.jpg)

![50b7e57f273be2b8914a4368fb84262525efd31305e10a5936f26345f373f913.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/images/50b7e57f273be2b8914a4368fb84262525efd31305e10a5936f26345f373f913.jpg)

![55c5dab1e14c6c6ca7788e8d754b22035bd4ace25df3b75b2084cacfe39d8307.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/images/55c5dab1e14c6c6ca7788e8d754b22035bd4ace25df3b75b2084cacfe39d8307.jpg)

![63df0cec55177332463fa5fe31c0fa15f1106c4cb71c9c4b3b92a4f09f7b61aa.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/images/63df0cec55177332463fa5fe31c0fa15f1106c4cb71c9c4b3b92a4f09f7b61aa.jpg)

![7262d36289f89ee1893fde63c5297065a58680f529dcf626545f31432ddd4aff.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/images/7262d36289f89ee1893fde63c5297065a58680f529dcf626545f31432ddd4aff.jpg)

![8e175355943b0300f0cc6409bfe71a7075205e90634fb5f1c8b1bac8f82d1c6d.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/images/8e175355943b0300f0cc6409bfe71a7075205e90634fb5f1c8b1bac8f82d1c6d.jpg)

![93bf744016a2b9d7d42fa50e48d179e498cc004dee39d27adcf5fda6dcfed2fb.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/images/93bf744016a2b9d7d42fa50e48d179e498cc004dee39d27adcf5fda6dcfed2fb.jpg)

![9a55a134082f99efc4e6072fea9b2fffb03b75dd2bcb08d7123759fb0dbbe58c.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/images/9a55a134082f99efc4e6072fea9b2fffb03b75dd2bcb08d7123759fb0dbbe58c.jpg)

![a7ed104d9025f2fb6cb2eb84aba9c40c11b9f207253d88ea35ae548beec2b521.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/images/a7ed104d9025f2fb6cb2eb84aba9c40c11b9f207253d88ea35ae548beec2b521.jpg)

![a99a74aebcc44373b4d74cd5d87c0501cd6a70e1c69c306808b1a31b947c2a90.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/images/a99a74aebcc44373b4d74cd5d87c0501cd6a70e1c69c306808b1a31b947c2a90.jpg)

![e236dfd50e480b1d2d469be3ccf620350ada5905dc90617aae652ada1a5ab11c.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/images/e236dfd50e480b1d2d469be3ccf620350ada5905dc90617aae652ada1a5ab11c.jpg)

### Tables

![022b248e56e105d09598e912e853603e39b92a2ed79407fa1b462538de38f797.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/tables/022b248e56e105d09598e912e853603e39b92a2ed79407fa1b462538de38f797.jpg)

![0e56447a68d38ff105d728e400c22437c52e73aba405609c36e801c53be907be.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/tables/0e56447a68d38ff105d728e400c22437c52e73aba405609c36e801c53be907be.jpg)

![0e7990e19fb965698667ab9149247146e498d97ff1bf81b93d025aa78201f112.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/tables/0e7990e19fb965698667ab9149247146e498d97ff1bf81b93d025aa78201f112.jpg)

![3156e6b02d6c73af88c1b21b4e5ec39e260cd89b332a86b4f06342debb29b413.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/tables/3156e6b02d6c73af88c1b21b4e5ec39e260cd89b332a86b4f06342debb29b413.jpg)

![3432224b62ab729fc1033b138c2729de5353562b3d7b15531325f1dacaeb4fd2.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/tables/3432224b62ab729fc1033b138c2729de5353562b3d7b15531325f1dacaeb4fd2.jpg)

![4c3614c9252d4cc8bf22b5eb5bc05bc5581412996e332f88af6a3cbc6fbe08b8.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/tables/4c3614c9252d4cc8bf22b5eb5bc05bc5581412996e332f88af6a3cbc6fbe08b8.jpg)

![512904fc170b189c6aacd8687045271210aaf011f3896ef275217fb94de051bb.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/tables/512904fc170b189c6aacd8687045271210aaf011f3896ef275217fb94de051bb.jpg)

![5191246e838ea7f4b21517fb86bf056a54f77064569269366993876a15b71f58.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/tables/5191246e838ea7f4b21517fb86bf056a54f77064569269366993876a15b71f58.jpg)

![5f96375461fb871214de5c8e3772091cf8acf491f5775b40591012caec941ff5.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/tables/5f96375461fb871214de5c8e3772091cf8acf491f5775b40591012caec941ff5.jpg)

![611cf3505a310a8d6abd0bff03f00ac1e19d3bdef3fb0719889e757dad9ad884.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/tables/611cf3505a310a8d6abd0bff03f00ac1e19d3bdef3fb0719889e757dad9ad884.jpg)

![a9b1726f504919f250e04798e6dee1fb135c871d9754ff34d07835bc6eb6a77f.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/tables/a9b1726f504919f250e04798e6dee1fb135c871d9754ff34d07835bc6eb6a77f.jpg)

![ae828672e096e6bb49437574b9c6294050d37dcafc36808619ccbbfbeea88e08.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/tables/ae828672e096e6bb49437574b9c6294050d37dcafc36808619ccbbfbeea88e08.jpg)

![b1f2701d1d3a4b300d16fcb8cceb9776960560ff5858d014f3c5d11312e49f63.jpg](../icml_results/160_Neural%20Feature%20Geometry%20Evolves%20as%20Discrete%20Ricci%20Flow/tables/b1f2701d1d3a4b300d16fcb8cceb9776960560ff5858d014f3c5d11312e49f63.jpg)

<a id="paper-0159"></a>

## Scalable Option Learning in High-Throughput Environments


### Images

![18922858388028b595b5ce6db866c7b81d023e22e83bc777658304f9570fcbe4.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/18922858388028b595b5ce6db866c7b81d023e22e83bc777658304f9570fcbe4.jpg)

![2f01acac81b75c2954b3ef0be57a00d9dc087db3d5e6481a1dbb32c9b2d43882.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/2f01acac81b75c2954b3ef0be57a00d9dc087db3d5e6481a1dbb32c9b2d43882.jpg)

![3b71f5c0bb8d115a2a487f658204844233fa6ac6065552fe68fa85bda56ab426.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/3b71f5c0bb8d115a2a487f658204844233fa6ac6065552fe68fa85bda56ab426.jpg)

![48c24af17da7185951ecfd4bca24ab8bf0e7191b4486b4820e289910222facf9.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/48c24af17da7185951ecfd4bca24ab8bf0e7191b4486b4820e289910222facf9.jpg)

![4cfd6fbf738b40c7c9b77791e7e8a0d11dff2ad32dba5d077d485282a5994468.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/4cfd6fbf738b40c7c9b77791e7e8a0d11dff2ad32dba5d077d485282a5994468.jpg)

![51fa5f47dc5ccd2c2d3c58c0e6b94db2cf090cb99bbbc714233e31950b55a954.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/51fa5f47dc5ccd2c2d3c58c0e6b94db2cf090cb99bbbc714233e31950b55a954.jpg)

![6ab0559ead4672d7fa23f30c5b6f86d2a8dc8918d6aec923330f44004e366277.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/6ab0559ead4672d7fa23f30c5b6f86d2a8dc8918d6aec923330f44004e366277.jpg)

![70cf5cec4732e48e25dc06419d8092c24897bfd7b0da19a6886a98e354ec2023.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/70cf5cec4732e48e25dc06419d8092c24897bfd7b0da19a6886a98e354ec2023.jpg)

![7415b197ec1acae7324d53aae345b291e7a195ef4b24141b264511d3cbcf34aa.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/7415b197ec1acae7324d53aae345b291e7a195ef4b24141b264511d3cbcf34aa.jpg)

![81d1b5b3f4dd94afed575c680d048dc12e72bb8098f1a275a928f1faa4c7e880.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/81d1b5b3f4dd94afed575c680d048dc12e72bb8098f1a275a928f1faa4c7e880.jpg)

![8204a5217f8dff4290d860a303249e55544a6baa27d06b5e8d91c53b58dcb9bf.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/8204a5217f8dff4290d860a303249e55544a6baa27d06b5e8d91c53b58dcb9bf.jpg)

![8af2d56f7654b9df5bf05db167e427b249646d2b37b66bc637c07028938b8836.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/8af2d56f7654b9df5bf05db167e427b249646d2b37b66bc637c07028938b8836.jpg)

![9c077f937fdee91028276b7f506e1a65953e6624dee633ac8290d3230327013e.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/9c077f937fdee91028276b7f506e1a65953e6624dee633ac8290d3230327013e.jpg)

![beeb1fa309917e7e8aaeae4851f5957a44ee8221198720424935775586f37d7c.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/beeb1fa309917e7e8aaeae4851f5957a44ee8221198720424935775586f37d7c.jpg)

![d02273d85417d3774eed2af1e945c60d6272264cc3c9c237d13502894b799a23.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/d02273d85417d3774eed2af1e945c60d6272264cc3c9c237d13502894b799a23.jpg)

![d73ed95d89fc355b84522aa027fe6ca6361f1c0620f105926843fcfc9809898a.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/d73ed95d89fc355b84522aa027fe6ca6361f1c0620f105926843fcfc9809898a.jpg)

![d7465ac7cac4207ee8f7e0fe967154e0629333c033150336dbc8945996015f43.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/d7465ac7cac4207ee8f7e0fe967154e0629333c033150336dbc8945996015f43.jpg)

![df904efa6140bd848d8cafb754d6b81f0cd63da7391c89eb0eb0bab5bf9c5ec2.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/df904efa6140bd848d8cafb754d6b81f0cd63da7391c89eb0eb0bab5bf9c5ec2.jpg)

![ec94e4e32034a4b6896f6847e39a06e95410271af46fc571e8da2481d5abfb5b.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/images/ec94e4e32034a4b6896f6847e39a06e95410271af46fc571e8da2481d5abfb5b.jpg)

### Tables

![3f14c1a53ed2475f6ed928288736c2d0a8b22339dbe3f4c395582c77fe8ecd2d.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/tables/3f14c1a53ed2475f6ed928288736c2d0a8b22339dbe3f4c395582c77fe8ecd2d.jpg)

![47c2b1e4c1d5501fa608be9dc12a4c191e62726d976dc007d449ea39405a1f6d.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/tables/47c2b1e4c1d5501fa608be9dc12a4c191e62726d976dc007d449ea39405a1f6d.jpg)

![490f904c9a92e267565f1be020a8aadef9be142aeff8c16ad4533be43bd81dd8.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/tables/490f904c9a92e267565f1be020a8aadef9be142aeff8c16ad4533be43bd81dd8.jpg)

![52f8b47e0e3b7236740f1a70bb775446b68ce8d890df7d82be3f7a451c74616d.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/tables/52f8b47e0e3b7236740f1a70bb775446b68ce8d890df7d82be3f7a451c74616d.jpg)

![75189ba440f5a23ee46438f8dde21e1ee51f30d5f12db77838e84731214e2f63.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/tables/75189ba440f5a23ee46438f8dde21e1ee51f30d5f12db77838e84731214e2f63.jpg)

![78e38a10d39cfb6130a9ee80db8493485e3b887fe14a6b70245340df18c80bb0.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/tables/78e38a10d39cfb6130a9ee80db8493485e3b887fe14a6b70245340df18c80bb0.jpg)

![9864569a4471387374413672079b4e7d2d87f72fc2b3403193d65c337eb9dbd3.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/tables/9864569a4471387374413672079b4e7d2d87f72fc2b3403193d65c337eb9dbd3.jpg)

![9f38f9ba2ec172dfc26bbc54931a0b6f058c6d91981be9139f91dad98ba3cdb3.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/tables/9f38f9ba2ec172dfc26bbc54931a0b6f058c6d91981be9139f91dad98ba3cdb3.jpg)

![cac9acf7ca075a44057a884e359c1bdcfe8ff3e1eaa6b7691d1a22e08265636a.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/tables/cac9acf7ca075a44057a884e359c1bdcfe8ff3e1eaa6b7691d1a22e08265636a.jpg)

![d2df67ca1be3c04ba888fc839e05d6275af66ab9f48d1e10a4c41fb05313e2f0.jpg](../icml_results/161_Scalable%20Option%20Learning%20in%20High-Throughput%20Environments/tables/d2df67ca1be3c04ba888fc839e05d6275af66ab9f48d1e10a4c41fb05313e2f0.jpg)

<a id="paper-0160"></a>

## Security–Fidelity Tradeoffs: The Hidden Cost of Prompt Injection Defense


### Images

![5224afe9f2a3b1f01d90a33c505770ed6b360ad76061e6a4e478ed76ee4d82ba.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/images/5224afe9f2a3b1f01d90a33c505770ed6b360ad76061e6a4e478ed76ee4d82ba.jpg)

![94a9f50b2004d0a137c6a546619bcc7d71e93c567c8dd0d121c59f630e0ad351.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/images/94a9f50b2004d0a137c6a546619bcc7d71e93c567c8dd0d121c59f630e0ad351.jpg)

![998074674b89cafd6ead4564277eb7ba3d7cc94c9fe2ce2e1433f648c6e3c4a0.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/images/998074674b89cafd6ead4564277eb7ba3d7cc94c9fe2ce2e1433f648c6e3c4a0.jpg)

![e55c0f0c0675b0e67a34649c0fd2678a21bb89893ec943756bd52bc056ddb079.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/images/e55c0f0c0675b0e67a34649c0fd2678a21bb89893ec943756bd52bc056ddb079.jpg)

### Tables

![0cf119f08f5870dd389f7aed85e81982f9df20f740d5d8b406577b4dac740b4f.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/0cf119f08f5870dd389f7aed85e81982f9df20f740d5d8b406577b4dac740b4f.jpg)

![10f1d06671294c112cad9196b00c08733031e6dc7f81d8e63fb41ebfe226b267.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/10f1d06671294c112cad9196b00c08733031e6dc7f81d8e63fb41ebfe226b267.jpg)

![197ed6d5db1801bca3a57f4ada1d9e9c77b400bf8326e48edeba6d094d350c0b.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/197ed6d5db1801bca3a57f4ada1d9e9c77b400bf8326e48edeba6d094d350c0b.jpg)

![219c9070e4cfbbb65c77a45e5ff9399753367191f6d44632aecd7afeb01130d2.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/219c9070e4cfbbb65c77a45e5ff9399753367191f6d44632aecd7afeb01130d2.jpg)

![2906faa4662b46c9e86a2eb57269204705b88a85bec0c3c57a38031572566a97.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/2906faa4662b46c9e86a2eb57269204705b88a85bec0c3c57a38031572566a97.jpg)

![2df007a318c0a2d1227da39dcea3fc3308bf2d6cd95cd749ad1f042e602b6f8e.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/2df007a318c0a2d1227da39dcea3fc3308bf2d6cd95cd749ad1f042e602b6f8e.jpg)

![2e975632800bfb3b523333cec447ad8e9cfd379d4c84a9f526e77bd6e097f3ec.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/2e975632800bfb3b523333cec447ad8e9cfd379d4c84a9f526e77bd6e097f3ec.jpg)

![3391003eda48090701cf4fc8b6b6691922068575879b2ca86c29cf514e21ed8b.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/3391003eda48090701cf4fc8b6b6691922068575879b2ca86c29cf514e21ed8b.jpg)

![3de376a69ff073c42a48023aa5fbe156aea1df32eb77c15191476aa16e35c8e1.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/3de376a69ff073c42a48023aa5fbe156aea1df32eb77c15191476aa16e35c8e1.jpg)

![587719593685e0d65eb4804011ae739f60304d1c1dc634aa9b809685f66cae95.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/587719593685e0d65eb4804011ae739f60304d1c1dc634aa9b809685f66cae95.jpg)

![5c56333edd717b71a01fe7b9ba1368accd7ad8f27c2a8db19f194bc31754e4c4.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/5c56333edd717b71a01fe7b9ba1368accd7ad8f27c2a8db19f194bc31754e4c4.jpg)

![5ee971021be2a73995e6dbccdce0455cf4ddd2086e0a4ed2a3be3222809fda6b.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/5ee971021be2a73995e6dbccdce0455cf4ddd2086e0a4ed2a3be3222809fda6b.jpg)

![61f8ebac1a6c2f159baeabc71ea7db36127249a24987382e34855e0954731aa0.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/61f8ebac1a6c2f159baeabc71ea7db36127249a24987382e34855e0954731aa0.jpg)

![678cc0ef9f3076eea5ba65c4a7d27838466a6ea0dbe09a2600b890a944123eaf.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/678cc0ef9f3076eea5ba65c4a7d27838466a6ea0dbe09a2600b890a944123eaf.jpg)

![6e4027179b27e6f43329f4d65ee1ea37c2164585a39050e69381ad840522dcb1.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/6e4027179b27e6f43329f4d65ee1ea37c2164585a39050e69381ad840522dcb1.jpg)

![85f0ae2bbb493722715c4bed4dd9849565dbb4363a6cb511cc0aa1d57f3281cd.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/85f0ae2bbb493722715c4bed4dd9849565dbb4363a6cb511cc0aa1d57f3281cd.jpg)

![9804a852a706151276e2766d6f469538b59f63b0fb8fe16e698fb6c2aa95772a.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/9804a852a706151276e2766d6f469538b59f63b0fb8fe16e698fb6c2aa95772a.jpg)

![a00a404eb73c3c3cafc3a16978133ddb527485df307afa0534b44d73194463c1.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/a00a404eb73c3c3cafc3a16978133ddb527485df307afa0534b44d73194463c1.jpg)

![a1d977b2b4e89d40df84a1508e917d027ceb03bd641a2f3c755641a3c0e69da4.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/a1d977b2b4e89d40df84a1508e917d027ceb03bd641a2f3c755641a3c0e69da4.jpg)

![a704c9e79536d42d70afff8e5cb4aa10b9abd935a6a2519fd56bf7d531df3a9e.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/a704c9e79536d42d70afff8e5cb4aa10b9abd935a6a2519fd56bf7d531df3a9e.jpg)

![a9c59a74fcf8108b17372093ec3c67297ec3d034c71da403313ea7c655a8fc0e.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/a9c59a74fcf8108b17372093ec3c67297ec3d034c71da403313ea7c655a8fc0e.jpg)

![befe9a841cc224580209af4713036877b84f2a6d9bb89951db4952493df9c739.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/befe9a841cc224580209af4713036877b84f2a6d9bb89951db4952493df9c739.jpg)

![c19a3fc78e19a647e2ea826f18e2d2aeb69a63711a21fe409762a5c071887f36.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/c19a3fc78e19a647e2ea826f18e2d2aeb69a63711a21fe409762a5c071887f36.jpg)

![c1cefcb834ce506ba21bfdfe5bba7ae07c66980a040d58393fdd9adff1d87895.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/c1cefcb834ce506ba21bfdfe5bba7ae07c66980a040d58393fdd9adff1d87895.jpg)

![c829a1921293af1837e9256cb788a0ac8d864b95f9852fb53349135cc875d535.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/c829a1921293af1837e9256cb788a0ac8d864b95f9852fb53349135cc875d535.jpg)

![c87b0643c1f1b455abf1e9127978b581e28b09a01b86ed548ec25b7de644e711.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/c87b0643c1f1b455abf1e9127978b581e28b09a01b86ed548ec25b7de644e711.jpg)

![cf904eed21f433bb9d19526a587437b6633d79f3444999bd9812ea66626044d4.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/cf904eed21f433bb9d19526a587437b6633d79f3444999bd9812ea66626044d4.jpg)

![d03f1be070a672c80006045f2bf307db303efe4e964fcc96e00b1bf705dabfb6.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/d03f1be070a672c80006045f2bf307db303efe4e964fcc96e00b1bf705dabfb6.jpg)

![d18e587de7b21beea1aaa9e66371147495837eeb0d5837f1b4e1bbf8e70cf332.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/d18e587de7b21beea1aaa9e66371147495837eeb0d5837f1b4e1bbf8e70cf332.jpg)

![d8080482eaf98eb8c78fab693c8edbdadf22d83f9886a07789bffc4aec065839.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/d8080482eaf98eb8c78fab693c8edbdadf22d83f9886a07789bffc4aec065839.jpg)

![e612aecc24d5221e1b17b6a6d0bc22d2e0d0cec1b33db0c2958b53c529aece20.jpg](../icml_results/162_Security%E2%80%93Fidelity%20Tradeoffs_%20The%20Hidden%20Cost%20of%20Prompt%20Injection%20Defense/tables/e612aecc24d5221e1b17b6a6d0bc22d2e0d0cec1b33db0c2958b53c529aece20.jpg)

<a id="paper-0161"></a>

## Steer Like the LLM: Activation Steering that Mimics Prompting


### Images

![019e4fb501e341d4887e578a0f38c35abea5320bc6de59ded9476a24848ea21b.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/019e4fb501e341d4887e578a0f38c35abea5320bc6de59ded9476a24848ea21b.jpg)

![09995f361f0b76f276c163d87395cabe6e457e77e96fd4a5cb6a046cfa9c93bc.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/09995f361f0b76f276c163d87395cabe6e457e77e96fd4a5cb6a046cfa9c93bc.jpg)

![1894544cea4e2f68a1bd200e110ca579514cd8c79ed29481962364acae44e4f6.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/1894544cea4e2f68a1bd200e110ca579514cd8c79ed29481962364acae44e4f6.jpg)

![32c5b79de5e8f4164c89480a0d17d3537a6f28d03761f44706fef6ad2dd27743.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/32c5b79de5e8f4164c89480a0d17d3537a6f28d03761f44706fef6ad2dd27743.jpg)

![48034642219fd504aab58ef092070f5ccd5ef83a1aa4be18d887b6389ecdc385.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/48034642219fd504aab58ef092070f5ccd5ef83a1aa4be18d887b6389ecdc385.jpg)

![5569f73f3c3057e1586181e1948fb95a5010d7c74fe102e07941f56d33211055.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/5569f73f3c3057e1586181e1948fb95a5010d7c74fe102e07941f56d33211055.jpg)

![59ea0f99a6de629df1dd214791835136dae498ed61887eb24855ddf0ccae12e9.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/59ea0f99a6de629df1dd214791835136dae498ed61887eb24855ddf0ccae12e9.jpg)

![652818f7b415c4c04e98b5552a3f104d2db3b51eb6079a6566d77347cabcdbde.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/652818f7b415c4c04e98b5552a3f104d2db3b51eb6079a6566d77347cabcdbde.jpg)

![72363b1fb65a1742d2812c0c7f877cc2e9cdb4d89a49c166a0432ca57167427c.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/72363b1fb65a1742d2812c0c7f877cc2e9cdb4d89a49c166a0432ca57167427c.jpg)

![7405539abc8826c803a8718298c681a5b87b31c393d8e7e153f7539b171d3989.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/7405539abc8826c803a8718298c681a5b87b31c393d8e7e153f7539b171d3989.jpg)

![79201a3ce043044c77851adc5e50f7926d3b75a56bb53ec27c74928ef9e24ef0.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/79201a3ce043044c77851adc5e50f7926d3b75a56bb53ec27c74928ef9e24ef0.jpg)

![a584c18656add3a1f6b19a10bbc09eb8a4aba27c6920c38b7c51ae289c039089.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/a584c18656add3a1f6b19a10bbc09eb8a4aba27c6920c38b7c51ae289c039089.jpg)

![c1f7a417a4364419e3d520a65844996051ffe1a69c3b9de4f96ef1ef73acc0a0.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/c1f7a417a4364419e3d520a65844996051ffe1a69c3b9de4f96ef1ef73acc0a0.jpg)

![c78b6535cb696bfdb6d14a76a155326238c0c2aeac29514a2e6387c9e01c505a.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/c78b6535cb696bfdb6d14a76a155326238c0c2aeac29514a2e6387c9e01c505a.jpg)

![d6c8d1f5d6ef8069cb0512bc5ae87627817cd1b2994e8b0555c0789a36916016.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/d6c8d1f5d6ef8069cb0512bc5ae87627817cd1b2994e8b0555c0789a36916016.jpg)

![dc7aa0b0a4741c17dacc4800f85245202e364f9a6a13ca06c87f3e2d947d71cb.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/dc7aa0b0a4741c17dacc4800f85245202e364f9a6a13ca06c87f3e2d947d71cb.jpg)

![dd7e2fa81414b90a9c27717dbe4d94c12357985ce242daec85881169b1f8e7aa.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/dd7e2fa81414b90a9c27717dbe4d94c12357985ce242daec85881169b1f8e7aa.jpg)

![e210f53c2e17a9fc24b941af9c4f28ed9021f2d7c5971906a897fa03d9ec3acd.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/e210f53c2e17a9fc24b941af9c4f28ed9021f2d7c5971906a897fa03d9ec3acd.jpg)

![e46bee6d1e0c4287cf02af4de8a2888df9dfefeaaa408c846b859eb689107921.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/e46bee6d1e0c4287cf02af4de8a2888df9dfefeaaa408c846b859eb689107921.jpg)

![e6b5e70a12619fe03940fa4a1644d7e677a96813abfef32453c37988dff7886b.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/e6b5e70a12619fe03940fa4a1644d7e677a96813abfef32453c37988dff7886b.jpg)

![f2665a2d9c8c6f5f71052297924fa0528a2a69f51f4a6f6eb54a27b9f1f0aa23.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/f2665a2d9c8c6f5f71052297924fa0528a2a69f51f4a6f6eb54a27b9f1f0aa23.jpg)

![f5487e32b159302510d8bef2f71b5ade9a5ea0b22ece4de72185866136fcc5f1.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/images/f5487e32b159302510d8bef2f71b5ade9a5ea0b22ece4de72185866136fcc5f1.jpg)

### Tables

![1ad8866adfd260ba2ddfa36a73fb00673c51b4ee11e81eace8da93781a21103f.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/tables/1ad8866adfd260ba2ddfa36a73fb00673c51b4ee11e81eace8da93781a21103f.jpg)

![3f882d84182ec7d62eb99acb9813ecc93a7a413319121f1611394f546d22c5a3.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/tables/3f882d84182ec7d62eb99acb9813ecc93a7a413319121f1611394f546d22c5a3.jpg)

![5ad9876799116ba95d0595e79d2f40c336cefa3d1c6256909d5a731bc0a4edae.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/tables/5ad9876799116ba95d0595e79d2f40c336cefa3d1c6256909d5a731bc0a4edae.jpg)

![6f5fa8b8d956ae445dcada3e5f2b8feb5e9b40915f85d9e3095d97efb569dfe6.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/tables/6f5fa8b8d956ae445dcada3e5f2b8feb5e9b40915f85d9e3095d97efb569dfe6.jpg)

![7507f655d4a3eef8233e5dce2b72e94534214bebd5d8955e1e932ec1f885aeb0.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/tables/7507f655d4a3eef8233e5dce2b72e94534214bebd5d8955e1e932ec1f885aeb0.jpg)

![7560e1509a0286d67f7e6e097f01cba1704c2acd95b7bce67c0ac5aee45a53d5.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/tables/7560e1509a0286d67f7e6e097f01cba1704c2acd95b7bce67c0ac5aee45a53d5.jpg)

![81844e9b8c0c018cedfe335e17e22ade2b134022bd32c7bc9d366c9f5acb5eb9.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/tables/81844e9b8c0c018cedfe335e17e22ade2b134022bd32c7bc9d366c9f5acb5eb9.jpg)

![a44e0173edf3136f193056afe4ff80e323484a6d408292d532074c1788bf0e22.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/tables/a44e0173edf3136f193056afe4ff80e323484a6d408292d532074c1788bf0e22.jpg)

![ab48cfcd205074c3982ce7f02377e67e671afeefe0f930837ece9a8f10813a30.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/tables/ab48cfcd205074c3982ce7f02377e67e671afeefe0f930837ece9a8f10813a30.jpg)

![ae34adee025809a60bd532b08c0f5edbdc88c09780d61547a78ccc54cf005b96.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/tables/ae34adee025809a60bd532b08c0f5edbdc88c09780d61547a78ccc54cf005b96.jpg)

![b2ee2f28e45918fb882f8695a90e4dd7f9fa7b358d39d48d881e2452bf62cf45.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/tables/b2ee2f28e45918fb882f8695a90e4dd7f9fa7b358d39d48d881e2452bf62cf45.jpg)

![d9c5daa0fbe88d981094e165506eadc97803c66b70cefaa659e4c1e12739e7cd.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/tables/d9c5daa0fbe88d981094e165506eadc97803c66b70cefaa659e4c1e12739e7cd.jpg)

![e04092137c122b484e794b3b553d9e0eacad303c25450b8e7320fc3b77965263.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/tables/e04092137c122b484e794b3b553d9e0eacad303c25450b8e7320fc3b77965263.jpg)

![e122e464c90770508bd9780abbaa5389d6770917af0b4387b61e0ccb9b496385.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/tables/e122e464c90770508bd9780abbaa5389d6770917af0b4387b61e0ccb9b496385.jpg)

![ee20014bb0bda8526e89f0fccb65f0a00384136cb8324a2523100e704be2a10b.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/tables/ee20014bb0bda8526e89f0fccb65f0a00384136cb8324a2523100e704be2a10b.jpg)

![ef7a134a749c6b56061b883d87df5c6c912d58c3ca1d4cbb9d0bc23badf02958.jpg](../icml_results/163_Steer%20Like%20the%20LLM_%20Activation%20Steering%20that%20Mimics%20Prompting/tables/ef7a134a749c6b56061b883d87df5c6c912d58c3ca1d4cbb9d0bc23badf02958.jpg)

<a id="paper-0162"></a>

## AdLift: Lifting Adversarial Perturbations to Safeguard 3D Gaussian Splatting Assets Against Instruction-Driven Editing


### Images

![0429f187d1c8f59a3881dc67b2b59998607524b7208aa2fb3f251f4d7d6e155a.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/0429f187d1c8f59a3881dc67b2b59998607524b7208aa2fb3f251f4d7d6e155a.jpg)

![05d4fb5f0b45c9c765bc0ad695a787c0795a6bf57c3ef8c4971b9ca59af5a2ba.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/05d4fb5f0b45c9c765bc0ad695a787c0795a6bf57c3ef8c4971b9ca59af5a2ba.jpg)

![083957e016cb6e5b15f6fc8616c4a82db0a1c87230e41309407e337e01b073d1.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/083957e016cb6e5b15f6fc8616c4a82db0a1c87230e41309407e337e01b073d1.jpg)

![0c712cb1142f0cfeb4e18f2fee5e69de41bd1a135b99b545b367be51f02536de.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/0c712cb1142f0cfeb4e18f2fee5e69de41bd1a135b99b545b367be51f02536de.jpg)

![11ebe199b00896735e6ac529ca508dbd750231ecfd6e77f57d0272438b11722f.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/11ebe199b00896735e6ac529ca508dbd750231ecfd6e77f57d0272438b11722f.jpg)

![18831c95d1758bbc607700c539d77be3b243e9255e258afd5015396f7ceea843.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/18831c95d1758bbc607700c539d77be3b243e9255e258afd5015396f7ceea843.jpg)

![18fc2bac12b18c7b14fbed3b065ba99fc79b72a4cef24a3da5fead6f8e7ed10a.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/18fc2bac12b18c7b14fbed3b065ba99fc79b72a4cef24a3da5fead6f8e7ed10a.jpg)

![33dcd35fd0fb547931324e7b3b47f6acaebafd3e6a3f009bd6135cd5ac451da6.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/33dcd35fd0fb547931324e7b3b47f6acaebafd3e6a3f009bd6135cd5ac451da6.jpg)

![42e76d8aac0ae11371b0477036c3dfa9e3571185834fa7581fcbb50967f851ac.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/42e76d8aac0ae11371b0477036c3dfa9e3571185834fa7581fcbb50967f851ac.jpg)

![44e1a7954d68ad605ed7cbeba1ecafc8b6da9b49e06e586d3bceb939b9ef1d75.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/44e1a7954d68ad605ed7cbeba1ecafc8b6da9b49e06e586d3bceb939b9ef1d75.jpg)

![460e2a722d2e42796446470631548edb392e7961d8019138cf9e633fcd42dc7c.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/460e2a722d2e42796446470631548edb392e7961d8019138cf9e633fcd42dc7c.jpg)

![4fa48592d8e2e7def50ab2d7b159877c68763342e0a950f80fd41d119e4595c7.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/4fa48592d8e2e7def50ab2d7b159877c68763342e0a950f80fd41d119e4595c7.jpg)

![50553d5d9a692a6b7213f29db1ba7cb409415a206aedc25fa47a8e74083fef7e.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/50553d5d9a692a6b7213f29db1ba7cb409415a206aedc25fa47a8e74083fef7e.jpg)

![55ce7698ec550e2392f99e8867b6113888be878a69ee15e8c1bdcaebf8f2a126.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/55ce7698ec550e2392f99e8867b6113888be878a69ee15e8c1bdcaebf8f2a126.jpg)

![5c29bc379d36190657d552f71ae4f329b5469167de2c2843a986f61b6a630195.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/5c29bc379d36190657d552f71ae4f329b5469167de2c2843a986f61b6a630195.jpg)

![5d0588d97b794b7454833084517005a13164dd4da7a822f62b16a03046a538bd.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/5d0588d97b794b7454833084517005a13164dd4da7a822f62b16a03046a538bd.jpg)

![6210cdf07d1bab928b8c3b0858e15111a8459982a53bce9d336492f421d9a0bf.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/6210cdf07d1bab928b8c3b0858e15111a8459982a53bce9d336492f421d9a0bf.jpg)

![6d2393051a90c734ebcc2746b611707b2320eb8bbe5dbd6dfa7eeeba033d5af7.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/6d2393051a90c734ebcc2746b611707b2320eb8bbe5dbd6dfa7eeeba033d5af7.jpg)

![734345d472ea80806840b1ace834ccc18ebcb07cbdc155b0c1f3465a98c71625.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/734345d472ea80806840b1ace834ccc18ebcb07cbdc155b0c1f3465a98c71625.jpg)

![7dcbe37212bf92b357d1941861aefcf1b056c4a0a7b26af9222d20e62bfda5a2.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/7dcbe37212bf92b357d1941861aefcf1b056c4a0a7b26af9222d20e62bfda5a2.jpg)

![816eee9f067c8a7454f0dccc691b02ba07303e7c09d70b2f5c48d53b882b1a80.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/816eee9f067c8a7454f0dccc691b02ba07303e7c09d70b2f5c48d53b882b1a80.jpg)

![8f59750267e0bd592d74bc413d2dae86ae19da748747e4577284384297cd4b68.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/8f59750267e0bd592d74bc413d2dae86ae19da748747e4577284384297cd4b68.jpg)

![966a80f1c4a68f51ee8bd519b65e39d3606831da16b778c220fc4a10dee1d483.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/966a80f1c4a68f51ee8bd519b65e39d3606831da16b778c220fc4a10dee1d483.jpg)

![971d396442bb764eaa3f39351b8758b643e3e562d809dc54de981c445248a1b4.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/971d396442bb764eaa3f39351b8758b643e3e562d809dc54de981c445248a1b4.jpg)

![99e431538fa463ca738c5fbbb684dba9a4464b493ffcce58c770cf6269204c05.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/99e431538fa463ca738c5fbbb684dba9a4464b493ffcce58c770cf6269204c05.jpg)

![9e78f48affd9110ec1163ac036fd247dc4f6576422d6e1efb192ce2c6c81d81d.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/9e78f48affd9110ec1163ac036fd247dc4f6576422d6e1efb192ce2c6c81d81d.jpg)

![a26b33506b1039d44e1f5ea7bf57ecd81e2785f270cd2c134418fc379772ee1c.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/a26b33506b1039d44e1f5ea7bf57ecd81e2785f270cd2c134418fc379772ee1c.jpg)

![a748cc3e7328fe17e6ea2777e222d7a5c4f9ef5bd6dcaf8f3dd4459c6de245e8.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/a748cc3e7328fe17e6ea2777e222d7a5c4f9ef5bd6dcaf8f3dd4459c6de245e8.jpg)

![b399e48324f94397bc723a59a997ff6b03ecdc6243c55eb8573b4ac85b9c6791.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/b399e48324f94397bc723a59a997ff6b03ecdc6243c55eb8573b4ac85b9c6791.jpg)

![b698c491cd0e33dcf4d7340150f8a4c6405c18f0843a0c53dd64cd88626023f6.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/b698c491cd0e33dcf4d7340150f8a4c6405c18f0843a0c53dd64cd88626023f6.jpg)

![b9e7f5550dcb5c21fd0702b17c1c9a5d4c2c4b015bb259d21069bab76d064c12.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/b9e7f5550dcb5c21fd0702b17c1c9a5d4c2c4b015bb259d21069bab76d064c12.jpg)

![bfe738e3793c0fb21f8437e9de84b4fee5646e39a2ff1edd0a3e777a0366dc63.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/bfe738e3793c0fb21f8437e9de84b4fee5646e39a2ff1edd0a3e777a0366dc63.jpg)

![cb9d74f2fd63d9aa56fe8bbb0551a017344f7b2f649e5beef804e2b6aa114127.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/cb9d74f2fd63d9aa56fe8bbb0551a017344f7b2f649e5beef804e2b6aa114127.jpg)

![ce253030af4acc167fe7f0e20685bfe16b690f269037d84a6cff978728869d43.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/ce253030af4acc167fe7f0e20685bfe16b690f269037d84a6cff978728869d43.jpg)

![d4ae4e921a6a69ddd04650ba45ddb44c40c72760f2bfa602e8a0773bd372ced8.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/d4ae4e921a6a69ddd04650ba45ddb44c40c72760f2bfa602e8a0773bd372ced8.jpg)

![de776cf819d3e112ab4c8c58d1a1e3dcf29556463e62a8f1ba7f9c05bef08683.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/de776cf819d3e112ab4c8c58d1a1e3dcf29556463e62a8f1ba7f9c05bef08683.jpg)

![e50d9f0b6f40b7a6f72cad087619fba8ce70433fe8c1d319f249a4e30c428a7b.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/e50d9f0b6f40b7a6f72cad087619fba8ce70433fe8c1d319f249a4e30c428a7b.jpg)

![ec7fda47419b0bf596af4b8a6a8041ef9055d90226acec243fb1055398154592.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/ec7fda47419b0bf596af4b8a6a8041ef9055d90226acec243fb1055398154592.jpg)

![ef5663aa71154b0e087405bb561feb5cbd19a98a21b4d31547796e6d26c93873.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/ef5663aa71154b0e087405bb561feb5cbd19a98a21b4d31547796e6d26c93873.jpg)

![f2d092496687654dc0fd72a1272e3541e5ab1949a93f71c453cf5043ad437d37.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/f2d092496687654dc0fd72a1272e3541e5ab1949a93f71c453cf5043ad437d37.jpg)

![f60e78816e91a9620484b3ee53a10335ac842977508de7e945cb3a69da465d9f.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/f60e78816e91a9620484b3ee53a10335ac842977508de7e945cb3a69da465d9f.jpg)

![fbd160775d42d14b92465008c6bd51b99221e71428280e882c648bfe0f485410.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/images/fbd160775d42d14b92465008c6bd51b99221e71428280e882c648bfe0f485410.jpg)

### Tables

![05e8674f7c81ac4a5af5f0d8c964bd448fba3d025ddf724c9ad0dac00fd64442.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/tables/05e8674f7c81ac4a5af5f0d8c964bd448fba3d025ddf724c9ad0dac00fd64442.jpg)

![06fe6099cc32315022d55c21e75079f3e7fd1485d8dc784a0b2c8fdbfd42e51f.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/tables/06fe6099cc32315022d55c21e75079f3e7fd1485d8dc784a0b2c8fdbfd42e51f.jpg)

![1c4983eef667ba2ec6639378560f541680a06f33b6296cb1e387b84da2427a55.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/tables/1c4983eef667ba2ec6639378560f541680a06f33b6296cb1e387b84da2427a55.jpg)

![31f944bb561d52e319dc3a6d11bd18216e357202a4de04357cfe7857a51c0d76.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/tables/31f944bb561d52e319dc3a6d11bd18216e357202a4de04357cfe7857a51c0d76.jpg)

![5c167fcf764830e860f66ae70df19f50fe64e653546200d8859dba69e1518c9b.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/tables/5c167fcf764830e860f66ae70df19f50fe64e653546200d8859dba69e1518c9b.jpg)

![670124926e1c024b36a5df1663ab35965bad53439574cfceac911c5ac7b85a02.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/tables/670124926e1c024b36a5df1663ab35965bad53439574cfceac911c5ac7b85a02.jpg)

![69df48f7061baa57f76c984356fd168ae9b0559428ff259b61f4041db1273bc4.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/tables/69df48f7061baa57f76c984356fd168ae9b0559428ff259b61f4041db1273bc4.jpg)

![6fb4fa08d16cea37890600ed665eae89947da561e0b0cb56a38a24bea8f8987d.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/tables/6fb4fa08d16cea37890600ed665eae89947da561e0b0cb56a38a24bea8f8987d.jpg)

![77fbdf9180776fce5d658b4b2951ac84328aa5d83aa9d8a6813413f72c532aaa.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/tables/77fbdf9180776fce5d658b4b2951ac84328aa5d83aa9d8a6813413f72c532aaa.jpg)

![7a21dac19470ddc4da935c9018e19e23fddfd8e3c32425e916ec7bd86aff697b.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/tables/7a21dac19470ddc4da935c9018e19e23fddfd8e3c32425e916ec7bd86aff697b.jpg)

![8465df59a38a6533fbcebfd0a756ef1f7d1484aab44a2045a1630b49648876cb.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/tables/8465df59a38a6533fbcebfd0a756ef1f7d1484aab44a2045a1630b49648876cb.jpg)

![9872362bd16e37667def7750a7f9b0809be5ee5d61f51b21ff39dcf9eeb55ab8.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/tables/9872362bd16e37667def7750a7f9b0809be5ee5d61f51b21ff39dcf9eeb55ab8.jpg)

![c7db088e4f0874312441e71ec6a031c1e82091fd80bc2f62080df089677f1c11.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/tables/c7db088e4f0874312441e71ec6a031c1e82091fd80bc2f62080df089677f1c11.jpg)

![cc46e478ccb34fb94024ca22bde104adc5d72d97d46b9233344c90df2f11d31d.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/tables/cc46e478ccb34fb94024ca22bde104adc5d72d97d46b9233344c90df2f11d31d.jpg)

![e18beb2c16753d5fe795de1108ea0537149bba3756fd03b2a64e72a2348093bd.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/tables/e18beb2c16753d5fe795de1108ea0537149bba3756fd03b2a64e72a2348093bd.jpg)

![eabb0ea36d7e82c8b46b3014e95eec7112174bee49664cfdf3920488e1f85c7f.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/tables/eabb0ea36d7e82c8b46b3014e95eec7112174bee49664cfdf3920488e1f85c7f.jpg)

![eec0b7d396a4eff9ba7c1730275ec66faa91d4459bee33a31cf3f3fbf0c437cd.jpg](../icml_results/164_AdLift_%20Lifting%20Adversarial%20Perturbations%20to%20Safeguard%203D%20Gaussian%20Splatting%20Assets%20Against%20Instruct/tables/eec0b7d396a4eff9ba7c1730275ec66faa91d4459bee33a31cf3f3fbf0c437cd.jpg)

<a id="paper-0163"></a>

## Unifying Masked Diffusion Models with Various Generation Orders and Beyond


### Images

![0a7bfcfd7bed9c72ea64b588426873576de0952e892f3307370989a3b66f3f21.jpg](../icml_results/165_Unifying%20Masked%20Diffusion%20Models%20with%20Various%20Generation%20Orders%20and%20Beyond/images/0a7bfcfd7bed9c72ea64b588426873576de0952e892f3307370989a3b66f3f21.jpg)

![1b96e84928bdcbdda640e00bc651902411a09b17b08660213c3ff3efe8bd91b3.jpg](../icml_results/165_Unifying%20Masked%20Diffusion%20Models%20with%20Various%20Generation%20Orders%20and%20Beyond/images/1b96e84928bdcbdda640e00bc651902411a09b17b08660213c3ff3efe8bd91b3.jpg)

![abcb22c7ae679a374597c0d8da6319a565e8455b6498227dd5442254ab10e57c.jpg](../icml_results/165_Unifying%20Masked%20Diffusion%20Models%20with%20Various%20Generation%20Orders%20and%20Beyond/images/abcb22c7ae679a374597c0d8da6319a565e8455b6498227dd5442254ab10e57c.jpg)

![ad95b546ada39f4618567db9f64483957a480eed0f312a4304f9ca9f520a5754.jpg](../icml_results/165_Unifying%20Masked%20Diffusion%20Models%20with%20Various%20Generation%20Orders%20and%20Beyond/images/ad95b546ada39f4618567db9f64483957a480eed0f312a4304f9ca9f520a5754.jpg)

![b5cbc33e44ff6953c2c81489987d53644fbb2dc8a8dd66fa9457121da5b99494.jpg](../icml_results/165_Unifying%20Masked%20Diffusion%20Models%20with%20Various%20Generation%20Orders%20and%20Beyond/images/b5cbc33e44ff6953c2c81489987d53644fbb2dc8a8dd66fa9457121da5b99494.jpg)

![b80e150d6e920f4eeadad4cc46e4748811b9be8c103e488d8949186210afe1fd.jpg](../icml_results/165_Unifying%20Masked%20Diffusion%20Models%20with%20Various%20Generation%20Orders%20and%20Beyond/images/b80e150d6e920f4eeadad4cc46e4748811b9be8c103e488d8949186210afe1fd.jpg)

![ddddf3cf55e4b9c1cb5aefec498223c4f7dbfd5483eebb068729b4181e6f02dd.jpg](../icml_results/165_Unifying%20Masked%20Diffusion%20Models%20with%20Various%20Generation%20Orders%20and%20Beyond/images/ddddf3cf55e4b9c1cb5aefec498223c4f7dbfd5483eebb068729b4181e6f02dd.jpg)

### Tables

![36a59c8e6db0890b4248caea0f68c69ff46ec6fa775ef10022570abb63d8d44f.jpg](../icml_results/165_Unifying%20Masked%20Diffusion%20Models%20with%20Various%20Generation%20Orders%20and%20Beyond/tables/36a59c8e6db0890b4248caea0f68c69ff46ec6fa775ef10022570abb63d8d44f.jpg)

![3cf15753a78df7e7fcab1abedd98b3144cdfee241dc2b4528a624eb77266f0fb.jpg](../icml_results/165_Unifying%20Masked%20Diffusion%20Models%20with%20Various%20Generation%20Orders%20and%20Beyond/tables/3cf15753a78df7e7fcab1abedd98b3144cdfee241dc2b4528a624eb77266f0fb.jpg)

![7ab7a731030ae68e6ce0879719a19869396e291792afe2efaea1a863f80c2ce5.jpg](../icml_results/165_Unifying%20Masked%20Diffusion%20Models%20with%20Various%20Generation%20Orders%20and%20Beyond/tables/7ab7a731030ae68e6ce0879719a19869396e291792afe2efaea1a863f80c2ce5.jpg)

![a082662007a436ae05c4f5302552ed0316ab32433a1a7cc55282235393dfd642.jpg](../icml_results/165_Unifying%20Masked%20Diffusion%20Models%20with%20Various%20Generation%20Orders%20and%20Beyond/tables/a082662007a436ae05c4f5302552ed0316ab32433a1a7cc55282235393dfd642.jpg)

![b3938c4de0bf3298745d47c89ae7cc01463f1a28c351c106f926dc2c9f3e8739.jpg](../icml_results/165_Unifying%20Masked%20Diffusion%20Models%20with%20Various%20Generation%20Orders%20and%20Beyond/tables/b3938c4de0bf3298745d47c89ae7cc01463f1a28c351c106f926dc2c9f3e8739.jpg)

![c8142337a23abedc2a89377938cab8fc41ae687e613a2ce99ca1188234565cf8.jpg](../icml_results/165_Unifying%20Masked%20Diffusion%20Models%20with%20Various%20Generation%20Orders%20and%20Beyond/tables/c8142337a23abedc2a89377938cab8fc41ae687e613a2ce99ca1188234565cf8.jpg)

![ca18d312c442b6611c8c2b369ed06ee63ca728c101e08aa8cdeb178e3bbb3694.jpg](../icml_results/165_Unifying%20Masked%20Diffusion%20Models%20with%20Various%20Generation%20Orders%20and%20Beyond/tables/ca18d312c442b6611c8c2b369ed06ee63ca728c101e08aa8cdeb178e3bbb3694.jpg)

![eda5e7a5f52123443850ad2905480e9774223e686dfda6ff941f7a1c258fee20.jpg](../icml_results/165_Unifying%20Masked%20Diffusion%20Models%20with%20Various%20Generation%20Orders%20and%20Beyond/tables/eda5e7a5f52123443850ad2905480e9774223e686dfda6ff941f7a1c258fee20.jpg)

<a id="paper-0164"></a>

## MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks


### Images

![0b7ae51ecbe79ce526cdabca8a0df522790d171392deb616bd225d7702a76871.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/images/0b7ae51ecbe79ce526cdabca8a0df522790d171392deb616bd225d7702a76871.jpg)

![0c41f92d5fc21a59e6b67755d26df2419aecfc087090fc14415effddb9266bd1.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/images/0c41f92d5fc21a59e6b67755d26df2419aecfc087090fc14415effddb9266bd1.jpg)

![28942d324c3f4c402a10ae499e6bee8c2938a3d9e8d218b9b247b3ed61f01d69.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/images/28942d324c3f4c402a10ae499e6bee8c2938a3d9e8d218b9b247b3ed61f01d69.jpg)

![5872e380f92c55f6924325751a20923c7ee624171349a2fb8bb93413d2216dbe.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/images/5872e380f92c55f6924325751a20923c7ee624171349a2fb8bb93413d2216dbe.jpg)

![660f46ebe55ad4ed39be7d8294af7be31e51b8170749db6bfb6ded3ecb6edf27.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/images/660f46ebe55ad4ed39be7d8294af7be31e51b8170749db6bfb6ded3ecb6edf27.jpg)

![8c54f0134fdc5812a04d2ca57d4d665d449d22be58032561ae33a35eb0a58a10.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/images/8c54f0134fdc5812a04d2ca57d4d665d449d22be58032561ae33a35eb0a58a10.jpg)

![bfc240dc4d8331c8707c80daaf15ab0f3f819e86f6d8b1e4e4648fb760647363.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/images/bfc240dc4d8331c8707c80daaf15ab0f3f819e86f6d8b1e4e4648fb760647363.jpg)

![fcf320af066ce01a6c8cef09fd60be98ff31e63480d74404f9a95bfd9fa68a97.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/images/fcf320af066ce01a6c8cef09fd60be98ff31e63480d74404f9a95bfd9fa68a97.jpg)

### Tables

![351cd423b3fb35fd42cd38fe790158c7592e9b8e114aa7dcbe14d6ebf1029fa3.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/tables/351cd423b3fb35fd42cd38fe790158c7592e9b8e114aa7dcbe14d6ebf1029fa3.jpg)

![3a36e4ba9ea592272db49cfe897dce5f860687698f2e142a2fa292ff5269625a.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/tables/3a36e4ba9ea592272db49cfe897dce5f860687698f2e142a2fa292ff5269625a.jpg)

![3f52fd1c691be6340b88ad4fca4af645b06e13fed628376891096241eaba054f.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/tables/3f52fd1c691be6340b88ad4fca4af645b06e13fed628376891096241eaba054f.jpg)

![417c96049e2c9cde64232f2fb9d96f941c8d7eb98b2240e95f28f1a275a3d85d.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/tables/417c96049e2c9cde64232f2fb9d96f941c8d7eb98b2240e95f28f1a275a3d85d.jpg)

![892c39b48ea749e5aa19def501432942a55d5f7fef7d87e340fc59a6569c4450.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/tables/892c39b48ea749e5aa19def501432942a55d5f7fef7d87e340fc59a6569c4450.jpg)

![90611f5a88d23a86b456e3434f64211c345704b25b609d4c8732176382d5bb05.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/tables/90611f5a88d23a86b456e3434f64211c345704b25b609d4c8732176382d5bb05.jpg)

![a5ffdb8786b64f381bec90d7b2f8e8929ed07eb87c95582fb0a1be006bb17130.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/tables/a5ffdb8786b64f381bec90d7b2f8e8929ed07eb87c95582fb0a1be006bb17130.jpg)

![af9d17659e1c9ce87ea07c7f1ba1e939e8093c9cc1cf487e04b650f467c56613.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/tables/af9d17659e1c9ce87ea07c7f1ba1e939e8093c9cc1cf487e04b650f467c56613.jpg)

![b9f75081de9647dd5765c7662225c7ece2d5c506cffeba1b7738c102e4e4e309.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/tables/b9f75081de9647dd5765c7662225c7ece2d5c506cffeba1b7738c102e4e4e309.jpg)

![dc122c4f92c88f3d5757d6bcf2c262435ee95026db4a221d59c18799f9ac5b55.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/tables/dc122c4f92c88f3d5757d6bcf2c262435ee95026db4a221d59c18799f9ac5b55.jpg)

![e05a5ad0d602103e7f075a13b8f97620d310c58bc91fecca6e439522ce8d53a3.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/tables/e05a5ad0d602103e7f075a13b8f97620d310c58bc91fecca6e439522ce8d53a3.jpg)

![e6f25ddca01a876b2ed24f08ae1e9065a22fc21cb620c26bafb7bf52375f2b36.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/tables/e6f25ddca01a876b2ed24f08ae1e9065a22fc21cb620c26bafb7bf52375f2b36.jpg)

![ea46f6d360d69b336efffdbe6ef3bccd4b8c37f1843874ba4b8fdb43c75ec959.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/tables/ea46f6d360d69b336efffdbe6ef3bccd4b8c37f1843874ba4b8fdb43c75ec959.jpg)

![fc350569b08da8259d84ba9a9777e9b1f0f139885fd41f41bd7155acccad5ae4.jpg](../icml_results/166_MASPOB_%20Bandit-Based%20Prompt%20Optimization%20for%20Multi-Agent%20Systems%20with%20Graph%20Neural%20Networks/tables/fc350569b08da8259d84ba9a9777e9b1f0f139885fd41f41bd7155acccad5ae4.jpg)

<a id="paper-0165"></a>

## When the Prompt Becomes Visual: Vision-Centric Jailbreak Attacks for Large Image Editing Models

### Images

![00a522731012cd772682f9a14a24e9fc74abc451f77853832ba4b8e94ad4f24d.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/00a522731012cd772682f9a14a24e9fc74abc451f77853832ba4b8e94ad4f24d.jpg)

![1e1676b6010e10ebe6f6ee31fd29c2886e0a9d46d34b93693025ed3961c948f0.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/1e1676b6010e10ebe6f6ee31fd29c2886e0a9d46d34b93693025ed3961c948f0.jpg)

![2025430ab85fcfa6a4efc6e2565ab250109cacb0b9cc00654e09339833cb34b2.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/2025430ab85fcfa6a4efc6e2565ab250109cacb0b9cc00654e09339833cb34b2.jpg)

![25de4396c31dc71f72a64b98032c9597707e2ae9988e48d1cb19ebb3eb7595ce.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/25de4396c31dc71f72a64b98032c9597707e2ae9988e48d1cb19ebb3eb7595ce.jpg)

![4b3b7ab5a07f0d476ffc6f0c0f52ad19177724f67590e44dda961de7a58e6e81.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/4b3b7ab5a07f0d476ffc6f0c0f52ad19177724f67590e44dda961de7a58e6e81.jpg)

![637aae2d8e7bf04d0a0356edd79efc17c0935a6b31180c92fb2f37e9c99d347e.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/637aae2d8e7bf04d0a0356edd79efc17c0935a6b31180c92fb2f37e9c99d347e.jpg)

![6d092a7c5e1252adb7f3594db948cf309084e64418bcbae3458537b4177c3d5e.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/6d092a7c5e1252adb7f3594db948cf309084e64418bcbae3458537b4177c3d5e.jpg)

![72725462d621ad031be50f40e3d07a58ba0692994bf143e3f12bf91fd5218864.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/72725462d621ad031be50f40e3d07a58ba0692994bf143e3f12bf91fd5218864.jpg)

![7cab15abedc3c3d9dfdcca6f38821997f4724466f531971b4b0059058c0413ee.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/7cab15abedc3c3d9dfdcca6f38821997f4724466f531971b4b0059058c0413ee.jpg)

![847d1964a8c14da84d1609f0ab7b02a6deaebfa188b982b4f4b94d7ef628b84f.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/847d1964a8c14da84d1609f0ab7b02a6deaebfa188b982b4f4b94d7ef628b84f.jpg)

![86e107732de741472421e56d290ecbb1fc68000151e04c7bdadb6cd68792a7b5.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/86e107732de741472421e56d290ecbb1fc68000151e04c7bdadb6cd68792a7b5.jpg)

![947856f54defa3f253d83692c3600aa85e3af96aae0db53c35d6d6b8849298fc.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/947856f54defa3f253d83692c3600aa85e3af96aae0db53c35d6d6b8849298fc.jpg)

![b4f6b1ceff97e0f2515dd2aac4e06d4c6e23be8760ce1d0b8fff5c2f6a74e241.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/b4f6b1ceff97e0f2515dd2aac4e06d4c6e23be8760ce1d0b8fff5c2f6a74e241.jpg)

![be5cf6dcc507de6a4dcc0049d8f58f4be52c856499e2a1369dc4186dfaf07594.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/be5cf6dcc507de6a4dcc0049d8f58f4be52c856499e2a1369dc4186dfaf07594.jpg)

![c6db180bf3e1310d8a84d926914f118bd83a7dc0349d3304414309d6dadb115d.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/c6db180bf3e1310d8a84d926914f118bd83a7dc0349d3304414309d6dadb115d.jpg)

![c919937b6f22c3134d77ea4a9294d02dd16362f4d5ebfed27e72287b3cccb6ba.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/c919937b6f22c3134d77ea4a9294d02dd16362f4d5ebfed27e72287b3cccb6ba.jpg)

![daa307217e365138687910fbad6eb5f615a2651da7e327cce0a3c49153aabb20.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/daa307217e365138687910fbad6eb5f615a2651da7e327cce0a3c49153aabb20.jpg)

![eb2eb3da0e78c3d19775bd154a39fc8d8895f2015b93674a92d94db5915d1df7.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/eb2eb3da0e78c3d19775bd154a39fc8d8895f2015b93674a92d94db5915d1df7.jpg)

![f59b25a1c2101bb18cf4f1207fb3a69f749cca37e3b06df2292c2ccb76f70c13.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/f59b25a1c2101bb18cf4f1207fb3a69f749cca37e3b06df2292c2ccb76f70c13.jpg)

![faa62ffcdb6cd0175cd190750206f1ad9efebf8b6392a99f1c12dbcb796f47ed.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/faa62ffcdb6cd0175cd190750206f1ad9efebf8b6392a99f1c12dbcb796f47ed.jpg)

![ff491ed9a1d0800ecf15457763017c1ac13fca4499a0b5ec5aab2892e8a6e846.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/images/ff491ed9a1d0800ecf15457763017c1ac13fca4499a0b5ec5aab2892e8a6e846.jpg)

### Tables

![3610ff7f0d2d8f22f15fdbfa42aced6fbd2fd47f274a2cdbdafa4d79f1be0056.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/3610ff7f0d2d8f22f15fdbfa42aced6fbd2fd47f274a2cdbdafa4d79f1be0056.jpg)

![3ecde2a8cf1bde448dac3aff6e05db2ade3ac8e8c31d404cf5e072dd967d9919.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/3ecde2a8cf1bde448dac3aff6e05db2ade3ac8e8c31d404cf5e072dd967d9919.jpg)

![43c8ed58f9da7a5651f9768c9b1238c2ec46fc1a8242ccecdc8a3717ed6b6185.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/43c8ed58f9da7a5651f9768c9b1238c2ec46fc1a8242ccecdc8a3717ed6b6185.jpg)

![44389b84efe1f2c5ec0ae843ba702fc8e0635b4216e8c5bba62e136eedc71619.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/44389b84efe1f2c5ec0ae843ba702fc8e0635b4216e8c5bba62e136eedc71619.jpg)

![a96249a7c7b78d6bb92e20c7c1b80ad8a54bae42b4da04e4cf999489bc03ace3.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/a96249a7c7b78d6bb92e20c7c1b80ad8a54bae42b4da04e4cf999489bc03ace3.jpg)

![c1a4b7e215fb3cb3c32c2ed5c9327a2bb4a5431d27e970cce7f477634c368e44.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/c1a4b7e215fb3cb3c32c2ed5c9327a2bb4a5431d27e970cce7f477634c368e44.jpg)

![c63fc48796585c5da743c0c4a8e40ed40f8f90c8f5b6cfd8273a72abdca23615.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/c63fc48796585c5da743c0c4a8e40ed40f8f90c8f5b6cfd8273a72abdca23615.jpg)

![e1a841fa39325acc983d73317ccf7cd9d43e0de570fffa2b412583d37413a940.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/e1a841fa39325acc983d73317ccf7cd9d43e0de570fffa2b412583d37413a940.jpg)

![f5e01baac9a07cda73989bb9a1a1e67cb93d63c4e28c978f7ea940e33acd5252.jpg](../icml_results/167_When%20the%20Prompt%20Becomes%20Visual_%20Vision-Centric%20Jailbreak%20Attacks%20for%20Large%20Image%20Editing%20Models/tables/f5e01baac9a07cda73989bb9a1a1e67cb93d63c4e28c978f7ea940e33acd5252.jpg)
