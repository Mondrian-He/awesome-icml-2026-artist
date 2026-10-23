# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 187 of 197

## 目录 (Table of Contents)

1. [Physiology-Aware Masked Cross-Modal Reconstruction for Biosignal Representation Learning](#paper-6139)
2. [Is Spurious Correlation Removal Always Learnable?](#paper-6140)
3. [EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts](#paper-6141)
4. [Time Series Reasoning via Process-Verifiable Thinking Data Synthesis and Scheduling for Tailored LLM Reasoning](#paper-6142)
5. [Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for Open-Ended LLM Reasoning](#paper-6143)
6. [DisPPO: Quantile-Based Distributional Reinforcement Learning for Large Language Models](#paper-6144)
7. [Revisiting ML Training under Fully Homomorphic Encryption: Convergence Guarantees, Differential Privacy, and Efficient Algorithms](#paper-6145)
8. [Positional Encoding for Spiking Transformers](#paper-6146)
9. [$\mathcalO(\log N)$ Latent Dimension Suffices for Universal Approximation of Permutation-invariant Function](#paper-6147)
10. [GeoAlign: Geometric Rollout Curation for Robust LLM Reinforcement Learning](#paper-6148)
11. [Meerkat-VL: Implicit Risk Safety Alignment in Multimodal LLMs via Perceptual Reasoning and Self-Verification](#paper-6149)
12. [InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](#paper-6150)
13. [OmniShow: Unifying Multimodal Conditions for Human-Object Interaction Video Generation](#paper-6151)
14. [The Easy, the Hard, and the Learnable: Confidence and Difficulty-Adaptive Policy Optimization for LLM Reasoning](#paper-6152)
15. [SMD: Multi-view Safety-Critical Driving Video Generation in the Real-world Domain](#paper-6153)
16. [DREAM: A Unified Framework for Drift-Corrected Federated Multi-Objective Learning](#paper-6154)
17. [WatchLog: Efficient and Interpretable Event Reasoning for Endpoint Detection and Response Logs with Multimodal LLMs](#paper-6155)
18. [Are Two Datasets Close Enough With Statistical Significance? A Kernel Distributional Closeness Testing Approach](#paper-6156)
19. [HTAC: Hierarchical Task-Aware Composition for Continual Offline Reinforcement Learning](#paper-6157)
20. [FormalJudge: A Neuro-Symbolic Paradigm for Agentic Oversight](#paper-6158)
21. [Trajectory Seriation via Spectral Tangent Alignment and Global Embedding](#paper-6159)
22. [Anti-Aliasing Matters: A Dynamic Network for Time Series Forecasting](#paper-6160)
23. [Segment Anything with Robust Uncertainty-Accuracy Correlation](#paper-6161)
24. [IMPACT: Influence Modeling for Open-Set Time Series Anomaly Detection](#paper-6162)
25. [How Transformers Represent Hierarchies: A Local-to-Global Mechanism](#paper-6163)
26. [Neural Vector Lyapunov–Razumikhin Certificates for Delayed Interconnected Systems](#paper-6164)
27. [Extra-Merge: Tracing the Rank-1 Subspace of Model Merging in Language Model Pre-Training](#paper-6165)
28. [Uni-DocRobust: Universal Plug-and-Play Robustness Enhancement for Multi-modal LLMs via Feature Restoration](#paper-6166)
29. [Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale Geometries](#paper-6167)
30. [Dismantling the Illusion of Vision-Language-Action Models Competence via Explicit Distributional Shifts](#paper-6168)
31. [SFedPO: Streaming Federated Learning with a Prediction Oracle under Temporal Shifts](#paper-6169)
32. [Winformer: Transcending Pairwise Similarity for Time-series Generation](#paper-6170)
33. [Ripple Perturbations Through Structure: Likelihood-Constrained Adversarial Attacks on Heterogeneous Tabular Data](#paper-6171)

---


<a id="paper-6139"></a>

## Physiology-Aware Masked Cross-Modal Reconstruction for Biosignal Representation Learning

### Images

![0062ee4be89fb6ea3a4bec0ddc9978223993d011fae632a1c52f1f33614064c0.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/0062ee4be89fb6ea3a4bec0ddc9978223993d011fae632a1c52f1f33614064c0.jpg)

![00ce244a2da87fea827c7378c41181f9b55f457e665d748b7e6147ca02753183.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/00ce244a2da87fea827c7378c41181f9b55f457e665d748b7e6147ca02753183.jpg)

![037987e0df44b2a6988c96824e3a81ddd3296f43c9ffd6b72ccdc77d48bd9d7c.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/037987e0df44b2a6988c96824e3a81ddd3296f43c9ffd6b72ccdc77d48bd9d7c.jpg)

![0e5f2a48d1a6cd27b191fa8b7d26acc77bd77bac1dd702d8cc944457154d59a7.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/0e5f2a48d1a6cd27b191fa8b7d26acc77bd77bac1dd702d8cc944457154d59a7.jpg)

![18e20ff101ae433f70979ac3e29ed043ec35a363faa6cd8b09d320956ebf1ab7.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/18e20ff101ae433f70979ac3e29ed043ec35a363faa6cd8b09d320956ebf1ab7.jpg)

![1ff89883f2504926fa9a9c343c2d75f70001870e961a61256914ffb25ea76ee0.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/1ff89883f2504926fa9a9c343c2d75f70001870e961a61256914ffb25ea76ee0.jpg)

![24a907356d4468a578fd5e9f7bbf84b378d051dbdb6c70730d61ea333b146ca9.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/24a907356d4468a578fd5e9f7bbf84b378d051dbdb6c70730d61ea333b146ca9.jpg)

![3213216cc310433c147eebf0bca3b3ac9533baed276baea980f68c3d66d6ccab.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/3213216cc310433c147eebf0bca3b3ac9533baed276baea980f68c3d66d6ccab.jpg)

![3665b8ab6da29fb6975e68c2b2eae91aba98b7f2a7bd51532be34ef6c8e1530b.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/3665b8ab6da29fb6975e68c2b2eae91aba98b7f2a7bd51532be34ef6c8e1530b.jpg)

![5619d74e5544db28f503377d5a83a161c0cf0fa226b92deda78e4eb6511ee0e1.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/5619d74e5544db28f503377d5a83a161c0cf0fa226b92deda78e4eb6511ee0e1.jpg)

![5bc6a9d600b321823b5dbf633bdf1ff08e95980514656dd7970108f428167435.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/5bc6a9d600b321823b5dbf633bdf1ff08e95980514656dd7970108f428167435.jpg)

![61dcf54e6ae0ec331114928e655c40389c426f3413258ae1445ffc48acf93a1f.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/61dcf54e6ae0ec331114928e655c40389c426f3413258ae1445ffc48acf93a1f.jpg)

![869e4abbbdbbb392926d8bfa3a182794f31d06153581ecee7ffe022f77eaee81.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/869e4abbbdbbb392926d8bfa3a182794f31d06153581ecee7ffe022f77eaee81.jpg)

![8a67ae4f79531235422ffaac3908be6d1fee447b45c4ca628a58e3d1acef454a.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/8a67ae4f79531235422ffaac3908be6d1fee447b45c4ca628a58e3d1acef454a.jpg)

![8b6fba1abedb6f32e85d968138fbc47df653de22a14107038702cf021cfe260c.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/8b6fba1abedb6f32e85d968138fbc47df653de22a14107038702cf021cfe260c.jpg)

![ad4ad4ef72e53b1ebe4e9151a6903aac6c207a37d61b90ea5a0092822a336afc.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/ad4ad4ef72e53b1ebe4e9151a6903aac6c207a37d61b90ea5a0092822a336afc.jpg)

![c0dc39f7019a72c0428707e75278dbe2768dcad1b4d6c70d06a40fd97df297f4.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/c0dc39f7019a72c0428707e75278dbe2768dcad1b4d6c70d06a40fd97df297f4.jpg)

![fecc726f107c6311c0b56d519a042bdadd503b7f7c27590629f27560bf202e05.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/images/fecc726f107c6311c0b56d519a042bdadd503b7f7c27590629f27560bf202e05.jpg)

### Tables

![062684559e1542dd87b0dc446aa372886cd23a525164f9f7ad037a92d3db9469.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/tables/062684559e1542dd87b0dc446aa372886cd23a525164f9f7ad037a92d3db9469.jpg)

![1c9e263bbd07e08f55488ffbf8846fd4e9a0c063da0137d4129545e5fcc2f45b.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/tables/1c9e263bbd07e08f55488ffbf8846fd4e9a0c063da0137d4129545e5fcc2f45b.jpg)

![47798d1cf82988afca6725ecd58f4f4f29ea53f74e9cd9245c4fddab7d0c31de.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/tables/47798d1cf82988afca6725ecd58f4f4f29ea53f74e9cd9245c4fddab7d0c31de.jpg)

![643a49d343536669a8aa7a73acf36f6b678b5990a57e90b00fbf83195e560976.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/tables/643a49d343536669a8aa7a73acf36f6b678b5990a57e90b00fbf83195e560976.jpg)

![ca5d1c6036b4b27ce2344d0f1fb43b3dcc5494f048464630fe7bfe00be499c95.jpg](../icml_results/6188_AVGen-Bench_%20A%20Task-Driven%20Benchmark%20for%20Multi-Granular%20Evaluation%20of%20Text-to-Audio-Video%20Generation/tables/ca5d1c6036b4b27ce2344d0f1fb43b3dcc5494f048464630fe7bfe00be499c95.jpg)

## Physiology-Aware Masked Cross-Modal Reconstruction for Biosignal Representation Learning


### Images

![070ede54b72066103efc8a33d2b2f601d2aa8d00591244c1b52494cb47187f08.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/070ede54b72066103efc8a33d2b2f601d2aa8d00591244c1b52494cb47187f08.jpg)

![0a7adbdc4cccfb70f0d93c305565170a137b219dfbce7cc74e5b4ce3d3c12699.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/0a7adbdc4cccfb70f0d93c305565170a137b219dfbce7cc74e5b4ce3d3c12699.jpg)

![11be07aef9010557ba0c15ed5e142a4c8da1b3f3631d655c4be52c510413bbb4.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/11be07aef9010557ba0c15ed5e142a4c8da1b3f3631d655c4be52c510413bbb4.jpg)

![15b7dd49c54735d49b2a741cc6892ff59e22ec6ebc4d9a44bfd4dfa9004759e5.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/15b7dd49c54735d49b2a741cc6892ff59e22ec6ebc4d9a44bfd4dfa9004759e5.jpg)

![18ebc6bb0f0ba7d386716e9191848c6dcca460bb720b26fa0470386d418a2e06.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/18ebc6bb0f0ba7d386716e9191848c6dcca460bb720b26fa0470386d418a2e06.jpg)

![1ba60834f69ebb4f7408a520c417dfa0c77d7b5e2a3408cd6d778235911d9bc6.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/1ba60834f69ebb4f7408a520c417dfa0c77d7b5e2a3408cd6d778235911d9bc6.jpg)

![3a15883a323c14bf291b9f8ef4eb8050442351ea94cdcb478b035ea817fa11aa.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/3a15883a323c14bf291b9f8ef4eb8050442351ea94cdcb478b035ea817fa11aa.jpg)

![40e6f82de244ed8f6432333503df3b3c1320e1866bc6661aca8f61d539aa5a63.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/40e6f82de244ed8f6432333503df3b3c1320e1866bc6661aca8f61d539aa5a63.jpg)

![42ccac3c3010fbf412b953bc4589d504f1de5731fdcf5b2055533aa50ac206a3.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/42ccac3c3010fbf412b953bc4589d504f1de5731fdcf5b2055533aa50ac206a3.jpg)

![48849758ce13c32fdd7a6875c79d9bafebc7b3208b0ca659fc253e2c2db62cc1.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/48849758ce13c32fdd7a6875c79d9bafebc7b3208b0ca659fc253e2c2db62cc1.jpg)

![619d08f9fab3fe1ff595f1a2f1acda187558a2b931d48494a505caac4b781956.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/619d08f9fab3fe1ff595f1a2f1acda187558a2b931d48494a505caac4b781956.jpg)

![6806cbbf90d000a36ac435685e11e19c2b3dceb82a71bbf1fb7564d72847b2a5.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/6806cbbf90d000a36ac435685e11e19c2b3dceb82a71bbf1fb7564d72847b2a5.jpg)

![800d9adb6ce865cb948b2c0f0b68d305db472156796c7522efce4b34b1ef2f8a.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/800d9adb6ce865cb948b2c0f0b68d305db472156796c7522efce4b34b1ef2f8a.jpg)

![8abfd2e49ddc306076c2fa83d8dba6dfe92cdc043f6609c9c754518fb716ded4.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/8abfd2e49ddc306076c2fa83d8dba6dfe92cdc043f6609c9c754518fb716ded4.jpg)

![95228526a343f9e13f1b1df70af6ab8907c14eaf4c7471b8ccb1e5c6cf546165.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/95228526a343f9e13f1b1df70af6ab8907c14eaf4c7471b8ccb1e5c6cf546165.jpg)

![9b4da938720814b8a00a525aaf187aac4acaf42954299b46de5918e126a123c1.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/9b4da938720814b8a00a525aaf187aac4acaf42954299b46de5918e126a123c1.jpg)

![b1e0d818e407319a255ecd2a800d2ffa5fbefe5f932b04e7f91a346a052919fa.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/b1e0d818e407319a255ecd2a800d2ffa5fbefe5f932b04e7f91a346a052919fa.jpg)

![b3df51fef44d0c12303bbcf0aaeee29c988649d1123ecaf9b7c94e847fedf38e.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/b3df51fef44d0c12303bbcf0aaeee29c988649d1123ecaf9b7c94e847fedf38e.jpg)

![b4e199b5750895e69fe4c7e3221c6a5a9e51eb43c71493151836ffd96213707e.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/b4e199b5750895e69fe4c7e3221c6a5a9e51eb43c71493151836ffd96213707e.jpg)

![c437f4edf208728d6ed1253e22ae25573a271fc79434ad98ffb3e3dab58b6f36.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/c437f4edf208728d6ed1253e22ae25573a271fc79434ad98ffb3e3dab58b6f36.jpg)

![c93f738fa21bdb18b9de03a114acccb24ef74ceff403a1f59089d478fbaf1dbf.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/c93f738fa21bdb18b9de03a114acccb24ef74ceff403a1f59089d478fbaf1dbf.jpg)

![d40f1a816065063f7e605162cf2642ac571821e5687dc707bbcf2ec64434cc96.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/d40f1a816065063f7e605162cf2642ac571821e5687dc707bbcf2ec64434cc96.jpg)

![e58494307838ad2b0c2bd31b0f2e03f0feee5700531084d5285209a69bcccaae.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/e58494307838ad2b0c2bd31b0f2e03f0feee5700531084d5285209a69bcccaae.jpg)

![e8f063b8b00010909de1e2cbd7c851156f75ec331c332cf9ff960e525c7dc0ca.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/images/e8f063b8b00010909de1e2cbd7c851156f75ec331c332cf9ff960e525c7dc0ca.jpg)

### Tables

![017e2b205cab41b21f8965bd780a2a3d0774cac9e8210c6fd341286ff18d7619.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/017e2b205cab41b21f8965bd780a2a3d0774cac9e8210c6fd341286ff18d7619.jpg)

![156c9f0a95410b07c717c45777884562def206e80a66b9a88abe1364b44f1f70.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/156c9f0a95410b07c717c45777884562def206e80a66b9a88abe1364b44f1f70.jpg)

![19e8b99d964cbc694544bb3148dfe6aff056a759a1509315dc677a464e1f7699.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/19e8b99d964cbc694544bb3148dfe6aff056a759a1509315dc677a464e1f7699.jpg)

![273ea5a7401b77184780ef8e2094536fe0b5d14e9fb4b8930c2e18d1581b8b61.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/273ea5a7401b77184780ef8e2094536fe0b5d14e9fb4b8930c2e18d1581b8b61.jpg)

![2a9e5a8c9ddf0c7c6c078a49d8841e195db56b33bc161e55122e8aea6d978ac2.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/2a9e5a8c9ddf0c7c6c078a49d8841e195db56b33bc161e55122e8aea6d978ac2.jpg)

![2ef2a40d0a3dc436b2c0184ba527661de6968d08edc04968cd456e69a418254c.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/2ef2a40d0a3dc436b2c0184ba527661de6968d08edc04968cd456e69a418254c.jpg)

![330d041cfc876a2b0e2715e3b80ff832c6881b9058c747c548c25cf42b297e55.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/330d041cfc876a2b0e2715e3b80ff832c6881b9058c747c548c25cf42b297e55.jpg)

![3f7c60b7c9d811df03b001a196ade45f23d5146f046df9fbd738c8bc328e13bf.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/3f7c60b7c9d811df03b001a196ade45f23d5146f046df9fbd738c8bc328e13bf.jpg)

![4ab06718f1f625389d0045acf0f0b2a89435c11373d2679cd7f5edacf9a94ffb.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/4ab06718f1f625389d0045acf0f0b2a89435c11373d2679cd7f5edacf9a94ffb.jpg)

![6457abe123d482d66bec268b97bb75398fd82b060334ed1c44082b404de7503c.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/6457abe123d482d66bec268b97bb75398fd82b060334ed1c44082b404de7503c.jpg)

![75f17ac3aa451e02726c33e528dafb63f13ee61e16d78cf5d572dfe7b8bf0c72.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/75f17ac3aa451e02726c33e528dafb63f13ee61e16d78cf5d572dfe7b8bf0c72.jpg)

![7bbd49b8feb9e7821a90490a8ae8a394f2e5d8bcdaee5bcf8a6f155d6568a5d7.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/7bbd49b8feb9e7821a90490a8ae8a394f2e5d8bcdaee5bcf8a6f155d6568a5d7.jpg)

![88ccab22560c140e10945844dffaff3c3d39cdd02f282efa751880809d4ec388.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/88ccab22560c140e10945844dffaff3c3d39cdd02f282efa751880809d4ec388.jpg)

![b0ff7684854a5f55ababc8f810b561e8a0eb4fcdf913395fc8b82f93aaaaacf5.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/b0ff7684854a5f55ababc8f810b561e8a0eb4fcdf913395fc8b82f93aaaaacf5.jpg)

![b79d2370d64c640d5008eb96c088aee73f29a2227e55017e941341cd7dc27916.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/b79d2370d64c640d5008eb96c088aee73f29a2227e55017e941341cd7dc27916.jpg)

![c3752125796bf333dbc64678399509e08d584e1946f184d2b56f417da1e35c9b.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/c3752125796bf333dbc64678399509e08d584e1946f184d2b56f417da1e35c9b.jpg)

![ccd2a864c413e0c5643d10e7cba155f6c337c2ca8606b17e6b55b21d70425a07.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/ccd2a864c413e0c5643d10e7cba155f6c337c2ca8606b17e6b55b21d70425a07.jpg)

![f49b9c9f25e28a61ce31c49e3692342c634721a6c4061043e28c593d980eed2c.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/f49b9c9f25e28a61ce31c49e3692342c634721a6c4061043e28c593d980eed2c.jpg)

![f84636564376fa623942e9ea3d85d1bb0a49000d2f1f1ca2e4d7d3ea8bec5dac.jpg](../icml_results/6189_Physiology-Aware%20Masked%20Cross-Modal%20Reconstruction%20for%20Biosignal%20Representation%20Learning/tables/f84636564376fa623942e9ea3d85d1bb0a49000d2f1f1ca2e4d7d3ea8bec5dac.jpg)

<a id="paper-6140"></a>

## Is Spurious Correlation Removal Always Learnable?


### Images

![52dc505cc6ca7f5e8c12a0f0f279d52833e02d34aa82536d9f39f0e8ea7b79be.jpg](../icml_results/6190_Is%20Spurious%20Correlation%20Removal%20Always%20Learnable_/images/52dc505cc6ca7f5e8c12a0f0f279d52833e02d34aa82536d9f39f0e8ea7b79be.jpg)

![653f5e9e783035100fb2019d92f9b7ad082d84ba245c19d0c7708578250b0585.jpg](../icml_results/6190_Is%20Spurious%20Correlation%20Removal%20Always%20Learnable_/images/653f5e9e783035100fb2019d92f9b7ad082d84ba245c19d0c7708578250b0585.jpg)

![747e099b137845b142b4d8e3389108ca8d9cb1fb082c3eefababb9224cd0c159.jpg](../icml_results/6190_Is%20Spurious%20Correlation%20Removal%20Always%20Learnable_/images/747e099b137845b142b4d8e3389108ca8d9cb1fb082c3eefababb9224cd0c159.jpg)

![ad7c3f00c332e337aee014bda7dcb3c133abbe916f07a8b7f1b495cde5b8d34a.jpg](../icml_results/6190_Is%20Spurious%20Correlation%20Removal%20Always%20Learnable_/images/ad7c3f00c332e337aee014bda7dcb3c133abbe916f07a8b7f1b495cde5b8d34a.jpg)

### Tables

![38bfe00ac5c4216e419c8630d0b7330d4fed6cb6028e6470416d01d9702d3b99.jpg](../icml_results/6190_Is%20Spurious%20Correlation%20Removal%20Always%20Learnable_/tables/38bfe00ac5c4216e419c8630d0b7330d4fed6cb6028e6470416d01d9702d3b99.jpg)

![3b16586e1ecc21e33a783fff17fa242861bd126e92a190cec878e93252c511f8.jpg](../icml_results/6190_Is%20Spurious%20Correlation%20Removal%20Always%20Learnable_/tables/3b16586e1ecc21e33a783fff17fa242861bd126e92a190cec878e93252c511f8.jpg)

![701b335889cd35596b314fb0276517d58ae561c5db9ca9fdc5acc144340b73cb.jpg](../icml_results/6190_Is%20Spurious%20Correlation%20Removal%20Always%20Learnable_/tables/701b335889cd35596b314fb0276517d58ae561c5db9ca9fdc5acc144340b73cb.jpg)

![70f3fa12c50ccc5161a247a27b22acf2d5537c26725d59b67f5351ea19524d20.jpg](../icml_results/6190_Is%20Spurious%20Correlation%20Removal%20Always%20Learnable_/tables/70f3fa12c50ccc5161a247a27b22acf2d5537c26725d59b67f5351ea19524d20.jpg)

![74452aed9529a7cc1b92bcf344593b9d7cfe975d8e814d9c0e3c27f3b3302d0e.jpg](../icml_results/6190_Is%20Spurious%20Correlation%20Removal%20Always%20Learnable_/tables/74452aed9529a7cc1b92bcf344593b9d7cfe975d8e814d9c0e3c27f3b3302d0e.jpg)

![9f24ac641326bed52db60861b8093c3d1770d292ca6178937013dab37058eb6c.jpg](../icml_results/6190_Is%20Spurious%20Correlation%20Removal%20Always%20Learnable_/tables/9f24ac641326bed52db60861b8093c3d1770d292ca6178937013dab37058eb6c.jpg)

![a20f88bb426f6648bf870bd13dac936e92dd2ec803ae7b2f496fd9ca40d153dc.jpg](../icml_results/6190_Is%20Spurious%20Correlation%20Removal%20Always%20Learnable_/tables/a20f88bb426f6648bf870bd13dac936e92dd2ec803ae7b2f496fd9ca40d153dc.jpg)

![bd336396685ad908f8a0460a2f2c344633c66c2eea2449b439a1eb178dc67ae0.jpg](../icml_results/6190_Is%20Spurious%20Correlation%20Removal%20Always%20Learnable_/tables/bd336396685ad908f8a0460a2f2c344633c66c2eea2449b439a1eb178dc67ae0.jpg)

<a id="paper-6141"></a>

## EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts


### Images

![2f87095874d2e994ceddc608b45fb209a0a0679a9763df2d7791c142f77b20c0.jpg](../icml_results/6191_EEG-Based%20Multimodal%20Learning%20via%20Hyperbolic%20Mixture-of-Curvature%20Experts/images/2f87095874d2e994ceddc608b45fb209a0a0679a9763df2d7791c142f77b20c0.jpg)

![3518291ffed1d04bf5bd644009df3517baa6af1e5b1931c3f8989f4ca6aae171.jpg](../icml_results/6191_EEG-Based%20Multimodal%20Learning%20via%20Hyperbolic%20Mixture-of-Curvature%20Experts/images/3518291ffed1d04bf5bd644009df3517baa6af1e5b1931c3f8989f4ca6aae171.jpg)

![b4e74e17f7dfbfe4b4642f283a224f45781f675f568b0956c39bd9917150070a.jpg](../icml_results/6191_EEG-Based%20Multimodal%20Learning%20via%20Hyperbolic%20Mixture-of-Curvature%20Experts/images/b4e74e17f7dfbfe4b4642f283a224f45781f675f568b0956c39bd9917150070a.jpg)

![c4c4f255802bd3c99cdbd107171f12452f11eb60ebf441d1597b9c57ecf49f9a.jpg](../icml_results/6191_EEG-Based%20Multimodal%20Learning%20via%20Hyperbolic%20Mixture-of-Curvature%20Experts/images/c4c4f255802bd3c99cdbd107171f12452f11eb60ebf441d1597b9c57ecf49f9a.jpg)

### Tables

![2fbea54756118205d810def957603770423b8a4fb95bb8e642ea71f09831af6a.jpg](../icml_results/6191_EEG-Based%20Multimodal%20Learning%20via%20Hyperbolic%20Mixture-of-Curvature%20Experts/tables/2fbea54756118205d810def957603770423b8a4fb95bb8e642ea71f09831af6a.jpg)

![3ae9113f8ba6a258257d7fd7d55a1474779eebbdc832ac3839f1410e35d54d90.jpg](../icml_results/6191_EEG-Based%20Multimodal%20Learning%20via%20Hyperbolic%20Mixture-of-Curvature%20Experts/tables/3ae9113f8ba6a258257d7fd7d55a1474779eebbdc832ac3839f1410e35d54d90.jpg)

![447e141a7b2da90faa2377d0b97a191b406859b63dfd83c019ff00f9b5c7c4d4.jpg](../icml_results/6191_EEG-Based%20Multimodal%20Learning%20via%20Hyperbolic%20Mixture-of-Curvature%20Experts/tables/447e141a7b2da90faa2377d0b97a191b406859b63dfd83c019ff00f9b5c7c4d4.jpg)

![599327a564ba991a85b99acfc83ba56a7a449de9b063ecf393b6785d348db278.jpg](../icml_results/6191_EEG-Based%20Multimodal%20Learning%20via%20Hyperbolic%20Mixture-of-Curvature%20Experts/tables/599327a564ba991a85b99acfc83ba56a7a449de9b063ecf393b6785d348db278.jpg)

![59ac8c00328e2d83b792e0956502015a7da6224f2300a56d7c16c309bef87a46.jpg](../icml_results/6191_EEG-Based%20Multimodal%20Learning%20via%20Hyperbolic%20Mixture-of-Curvature%20Experts/tables/59ac8c00328e2d83b792e0956502015a7da6224f2300a56d7c16c309bef87a46.jpg)

![813e73a874549bf334b922c1db23b43d6e48931c259c788aae9f3bb1a872ad38.jpg](../icml_results/6191_EEG-Based%20Multimodal%20Learning%20via%20Hyperbolic%20Mixture-of-Curvature%20Experts/tables/813e73a874549bf334b922c1db23b43d6e48931c259c788aae9f3bb1a872ad38.jpg)

![9d9ac679199d5d2f8c630de7a71cd6d8d0cec8c627382774bac8939b98089dff.jpg](../icml_results/6191_EEG-Based%20Multimodal%20Learning%20via%20Hyperbolic%20Mixture-of-Curvature%20Experts/tables/9d9ac679199d5d2f8c630de7a71cd6d8d0cec8c627382774bac8939b98089dff.jpg)

![a62e834322eb96e1461956fb76414f7879771d37697fb66c19d5b73f530bfcc7.jpg](../icml_results/6191_EEG-Based%20Multimodal%20Learning%20via%20Hyperbolic%20Mixture-of-Curvature%20Experts/tables/a62e834322eb96e1461956fb76414f7879771d37697fb66c19d5b73f530bfcc7.jpg)

![a699369b4c65af40d5d457e5c6af5c7c4833b200b8d506c2de75865faca8378f.jpg](../icml_results/6191_EEG-Based%20Multimodal%20Learning%20via%20Hyperbolic%20Mixture-of-Curvature%20Experts/tables/a699369b4c65af40d5d457e5c6af5c7c4833b200b8d506c2de75865faca8378f.jpg)

![a6fe7b522379a6815e1cae6979006f58ab7191fa8f818088bda8ab811a8f71cb.jpg](../icml_results/6191_EEG-Based%20Multimodal%20Learning%20via%20Hyperbolic%20Mixture-of-Curvature%20Experts/tables/a6fe7b522379a6815e1cae6979006f58ab7191fa8f818088bda8ab811a8f71cb.jpg)

![b92397d362c216f1465723923ce91b781b051f7182b80cd70218d0068303c066.jpg](../icml_results/6191_EEG-Based%20Multimodal%20Learning%20via%20Hyperbolic%20Mixture-of-Curvature%20Experts/tables/b92397d362c216f1465723923ce91b781b051f7182b80cd70218d0068303c066.jpg)

![e0f7e44bf0890fd7d4cdca3ab529688387cb9af397c64ef23d6bb129e7e29c33.jpg](../icml_results/6191_EEG-Based%20Multimodal%20Learning%20via%20Hyperbolic%20Mixture-of-Curvature%20Experts/tables/e0f7e44bf0890fd7d4cdca3ab529688387cb9af397c64ef23d6bb129e7e29c33.jpg)

![f0297044e09d8f5eec7e4e158856da00197ba1d2296c7637b93eed2c50adc877.jpg](../icml_results/6191_EEG-Based%20Multimodal%20Learning%20via%20Hyperbolic%20Mixture-of-Curvature%20Experts/tables/f0297044e09d8f5eec7e4e158856da00197ba1d2296c7637b93eed2c50adc877.jpg)

<a id="paper-6142"></a>

## Time Series Reasoning via Process-Verifiable Thinking Data Synthesis and Scheduling for Tailored LLM Reasoning


### Images

![2eeec536db56e81e4f281ac6bf0b955d6ecdcf3770b5f2244071f3fd7a833b57.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/images/2eeec536db56e81e4f281ac6bf0b955d6ecdcf3770b5f2244071f3fd7a833b57.jpg)

![44b649b62913a79de6482d0744a2164fca41cdccc8ef0d91f29f0c52fe148528.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/images/44b649b62913a79de6482d0744a2164fca41cdccc8ef0d91f29f0c52fe148528.jpg)

![4bb417d4998d7ab147f7e2d1facf9aecf54cbecb3b035bd5aa362aa4846cc49d.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/images/4bb417d4998d7ab147f7e2d1facf9aecf54cbecb3b035bd5aa362aa4846cc49d.jpg)

![4d2ea6f9499f31f257a43531d89a9238062858c1e5f6a3e20f207aea7f2d47f3.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/images/4d2ea6f9499f31f257a43531d89a9238062858c1e5f6a3e20f207aea7f2d47f3.jpg)

![6f245d0db86e20109907aa70be264876750f1b5882384d18bd0ef50b9b4493d3.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/images/6f245d0db86e20109907aa70be264876750f1b5882384d18bd0ef50b9b4493d3.jpg)

![be299c018845cff37bb255e68db2f50856d36476321376b0c6103d220d562483.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/images/be299c018845cff37bb255e68db2f50856d36476321376b0c6103d220d562483.jpg)

![bec71169aaf036a06a75a1808c46e913621050b485e177271c658f615227f27c.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/images/bec71169aaf036a06a75a1808c46e913621050b485e177271c658f615227f27c.jpg)

![cced6c5562c4ff206a56c9cd7498ce6582bacec98f6679543295c7c91f80755e.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/images/cced6c5562c4ff206a56c9cd7498ce6582bacec98f6679543295c7c91f80755e.jpg)

![cd19d846324abd6cc3f7fd397fde64066481fe15e39b4116f9f824787ef333f5.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/images/cd19d846324abd6cc3f7fd397fde64066481fe15e39b4116f9f824787ef333f5.jpg)

![e2a35f519b3855512d86723db508b5781f5ee6dbedf7a0e0641ee47566be6649.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/images/e2a35f519b3855512d86723db508b5781f5ee6dbedf7a0e0641ee47566be6649.jpg)

![f0db178d13f9281a11392096b84d5ddafb1e89531bee3337d4f5c776c27aae4b.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/images/f0db178d13f9281a11392096b84d5ddafb1e89531bee3337d4f5c776c27aae4b.jpg)

![f341f7001e61d3dd02d89a56ddf7e9993c99d256619302f97d59d550ca0d3f59.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/images/f341f7001e61d3dd02d89a56ddf7e9993c99d256619302f97d59d550ca0d3f59.jpg)

![f891c4fa4decaaad40fb96b30ad46098f45814c29e8bb1c000d0b877e5b6b544.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/images/f891c4fa4decaaad40fb96b30ad46098f45814c29e8bb1c000d0b877e5b6b544.jpg)

### Tables

![0a9b67a707e86800edada655e1bcc3158eaaf42cd4a0d3d5665794f7b93d66a8.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/tables/0a9b67a707e86800edada655e1bcc3158eaaf42cd4a0d3d5665794f7b93d66a8.jpg)

![3c899ffd7fcc6f7b10eca2e828e625393a8632b614b84901c3401294bf9ab898.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/tables/3c899ffd7fcc6f7b10eca2e828e625393a8632b614b84901c3401294bf9ab898.jpg)

![5ec95cf34feaac3d04b5739ea2f28b5c1aa45ce08b3fad275d089633193debb6.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/tables/5ec95cf34feaac3d04b5739ea2f28b5c1aa45ce08b3fad275d089633193debb6.jpg)

![63c53894fb33219b9ba5bd85b2cb072531ccc1468d877ae5e6d18a60184577e9.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/tables/63c53894fb33219b9ba5bd85b2cb072531ccc1468d877ae5e6d18a60184577e9.jpg)

![76ad805f0a94409d6bdf268139691b64f93db90786b3ad3e69de67074aaf3fca.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/tables/76ad805f0a94409d6bdf268139691b64f93db90786b3ad3e69de67074aaf3fca.jpg)

![7e4e9807077bddb467dd4b758f24902de2002d250b6ecfd86ea0ce975002e2a6.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/tables/7e4e9807077bddb467dd4b758f24902de2002d250b6ecfd86ea0ce975002e2a6.jpg)

![95f6ec8cce1f5c64338fe1a0fe83a28db9ffeec5e06f5ce44c90aa80c08d2b54.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/tables/95f6ec8cce1f5c64338fe1a0fe83a28db9ffeec5e06f5ce44c90aa80c08d2b54.jpg)

![990cdc9a639da94a02f2753c2998f4103360e98b39050295618f57fc884ac5f5.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/tables/990cdc9a639da94a02f2753c2998f4103360e98b39050295618f57fc884ac5f5.jpg)

![9da3d39fe22910278d289b9e19d13b1858873cef8f361b96508c278c6970b9eb.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/tables/9da3d39fe22910278d289b9e19d13b1858873cef8f361b96508c278c6970b9eb.jpg)

![a4b54e8c160c65f72db8f4f56f598e9158bd65c87fc31df82a5e0bfcbc0e864b.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/tables/a4b54e8c160c65f72db8f4f56f598e9158bd65c87fc31df82a5e0bfcbc0e864b.jpg)

![ac8fd00228e1f1e6c1f4b0d4608ce17e94f83d8fe3d73ec05ac8eeafe59db89b.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/tables/ac8fd00228e1f1e6c1f4b0d4608ce17e94f83d8fe3d73ec05ac8eeafe59db89b.jpg)

![addbdabacc74bd036d0d40dbc0f77dd45c398e49546894d44ff97f7c7150682a.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/tables/addbdabacc74bd036d0d40dbc0f77dd45c398e49546894d44ff97f7c7150682a.jpg)

![ae84fec4af45774bd18c1f65520fadab5ee134b3ac11b7ec1e3dd6df763e89f8.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/tables/ae84fec4af45774bd18c1f65520fadab5ee134b3ac11b7ec1e3dd6df763e89f8.jpg)

![b6511e16e8986c77d3bdf4c62992441aaeeaba1a54477a9e93a1ec6e5c05e457.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/tables/b6511e16e8986c77d3bdf4c62992441aaeeaba1a54477a9e93a1ec6e5c05e457.jpg)

![d34e0c4d41433245b0ba2bf2bd3c1405e7809f4af5a36a87d1e0cac0cfc18640.jpg](../icml_results/6192_Time%20Series%20Reasoning%20via%20Process-Verifiable%20Thinking%20Data%20Synthesis%20and%20Scheduling%20for%20Tailored%20LLM/tables/d34e0c4d41433245b0ba2bf2bd3c1405e7809f4af5a36a87d1e0cac0cfc18640.jpg)

<a id="paper-6143"></a>

## Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for Open-Ended LLM Reasoning


### Images

![7835da4744c40a46a4f694b1bf1ff8573846ae494a3218ca7297be064cc85bbf.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/images/7835da4744c40a46a4f694b1bf1ff8573846ae494a3218ca7297be064cc85bbf.jpg)

![9898ded70a5901fde62b9552eae26652645dbfec8e9fd28a092507f14047414e.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/images/9898ded70a5901fde62b9552eae26652645dbfec8e9fd28a092507f14047414e.jpg)

![9a4b9157ea678d787aa7cfef33159cbb472fa4c123e8332551383736d0638429.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/images/9a4b9157ea678d787aa7cfef33159cbb472fa4c123e8332551383736d0638429.jpg)

![ab3bfb3555fe08ddb14111d1d29a07cacd892cf45916353c169623b9c280de81.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/images/ab3bfb3555fe08ddb14111d1d29a07cacd892cf45916353c169623b9c280de81.jpg)

![ba5a026dd3392a63922c41535a4d42de06f37089c07de86fe78fda0241347c3f.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/images/ba5a026dd3392a63922c41535a4d42de06f37089c07de86fe78fda0241347c3f.jpg)

![dfc694526c6025680bbe91a4d115de079f069f7e94092e8f3c7feb25bf74cd78.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/images/dfc694526c6025680bbe91a4d115de079f069f7e94092e8f3c7feb25bf74cd78.jpg)

### Tables

![0750001d418548b3b7128289d54aad3bc56fa055d0ac0316993a5f533467af42.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/tables/0750001d418548b3b7128289d54aad3bc56fa055d0ac0316993a5f533467af42.jpg)

![1e6473576d845417ea3e54ca4fc7214f7938efcf67e087c93287d487778b4f89.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/tables/1e6473576d845417ea3e54ca4fc7214f7938efcf67e087c93287d487778b4f89.jpg)

![307831c5ec029198eecef638521c4fadc323b41441ae0e144fb4d6f25cf73e86.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/tables/307831c5ec029198eecef638521c4fadc323b41441ae0e144fb4d6f25cf73e86.jpg)

![3a3dbf71622391b407a69a3fa5bf8588b8b6220d1b4ce1007d94285da9828ade.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/tables/3a3dbf71622391b407a69a3fa5bf8588b8b6220d1b4ce1007d94285da9828ade.jpg)

![5102f1d9b6b7d67adfdb529abd150b6abe5cfb9ee0f0eae6bcb76f19eccba50f.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/tables/5102f1d9b6b7d67adfdb529abd150b6abe5cfb9ee0f0eae6bcb76f19eccba50f.jpg)

![6479c36693d752d821074f14c3e0476ac950e59b938ef254629317b7d1e02d74.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/tables/6479c36693d752d821074f14c3e0476ac950e59b938ef254629317b7d1e02d74.jpg)

![694dcf0352d49216c50a65253a90b4eb1a2a3c66d07ee941e86a32661c77c520.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/tables/694dcf0352d49216c50a65253a90b4eb1a2a3c66d07ee941e86a32661c77c520.jpg)

![78abf5df98fda94e9f34081b2ce1d7be7a3456a3039c3b5ce05d245943eac2e4.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/tables/78abf5df98fda94e9f34081b2ce1d7be7a3456a3039c3b5ce05d245943eac2e4.jpg)

![ab91fce7a3f8d30943dfd9fc90900e78c0a51017bf03b007de99cdd4b74089f9.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/tables/ab91fce7a3f8d30943dfd9fc90900e78c0a51017bf03b007de99cdd4b74089f9.jpg)

![adb078f38e2eaf0f499d1fb66d56c1ef255142c793e5c3794ef08bf17147afb4.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/tables/adb078f38e2eaf0f499d1fb66d56c1ef255142c793e5c3794ef08bf17147afb4.jpg)

![aed1d2cf83ef94e917624c9e2bad86087c4afbddc1080ce9227607a84d60e34b.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/tables/aed1d2cf83ef94e917624c9e2bad86087c4afbddc1080ce9227607a84d60e34b.jpg)

![b5a2ffc5de7208ba4728fdeb723e41a0383d8346cef86d971b01ed8f3fcef23a.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/tables/b5a2ffc5de7208ba4728fdeb723e41a0383d8346cef86d971b01ed8f3fcef23a.jpg)

![bae8e88891114bdfca0f2cbab253641095c37e3d87959f3e021d108b2d73d4e9.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/tables/bae8e88891114bdfca0f2cbab253641095c37e3d87959f3e021d108b2d73d4e9.jpg)

![e5a1085ebf7a960f718b076bfc5c774df99e4ae143c2b7d55bd3611c573d4f9d.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/tables/e5a1085ebf7a960f718b076bfc5c774df99e4ae143c2b7d55bd3611c573d4f9d.jpg)

![e851c35d7b3591e0416326337dd5c72df77e9433f22afa672922b81ca7ef1981.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/tables/e851c35d7b3591e0416326337dd5c72df77e9433f22afa672922b81ca7ef1981.jpg)

![fb0e5e63f3543ba27dba8107857339e99a6f8f061d4afcdf781d6ab0d02b1d0d.jpg](../icml_results/6193_Breaking%20the%20Exploration%20Bottleneck_%20Rubric-Scaffolded%20Reinforcement%20Learning%20for%20Open-Ended%20LLM%20Rea/tables/fb0e5e63f3543ba27dba8107857339e99a6f8f061d4afcdf781d6ab0d02b1d0d.jpg)

<a id="paper-6144"></a>

## DisPPO: Quantile-Based Distributional Reinforcement Learning for Large Language Models


### Images

![a15e73471afe97eb99921e852281cedc3aa32129779d30877b9fe61d525f2073.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/images/a15e73471afe97eb99921e852281cedc3aa32129779d30877b9fe61d525f2073.jpg)

![bc61624ca007ea40d794ca0dab631137928554096e30624048e7f3fb0833b268.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/images/bc61624ca007ea40d794ca0dab631137928554096e30624048e7f3fb0833b268.jpg)

### Tables

![036065317ddb21298029fad4c5e4bbc7758b49a621f544aaf5e4bc7730d61e33.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/036065317ddb21298029fad4c5e4bbc7758b49a621f544aaf5e4bc7730d61e33.jpg)

![210de9852a50a6955215b54ae569e4a88754a4521cc4e22c76317535870a6831.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/210de9852a50a6955215b54ae569e4a88754a4521cc4e22c76317535870a6831.jpg)

![2892c33f8f80339d615f1c9666c29967cd2aa4330d44784c6441af1f0e26af73.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/2892c33f8f80339d615f1c9666c29967cd2aa4330d44784c6441af1f0e26af73.jpg)

![4234277d60c73f827fc7ed2397c5e7b9a14f7a51faf62a7e1d8ffe1542d71c24.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/4234277d60c73f827fc7ed2397c5e7b9a14f7a51faf62a7e1d8ffe1542d71c24.jpg)

![47097d7c46b245f037b2a50ce6787bd518f752d35b9467e8382a0c6aad5b5d4f.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/47097d7c46b245f037b2a50ce6787bd518f752d35b9467e8382a0c6aad5b5d4f.jpg)

![484a1ef1ce4eb091526d83699b45fe057dc53c535eafe1cbef14b246ce023709.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/484a1ef1ce4eb091526d83699b45fe057dc53c535eafe1cbef14b246ce023709.jpg)

![4b27ba9125519157aadd0f2c28bf0f26f848a1b07dc0a790c09b1efd4fc911b8.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/4b27ba9125519157aadd0f2c28bf0f26f848a1b07dc0a790c09b1efd4fc911b8.jpg)

![50cf0a4f2afc7bb6df1e59d0cbf5a6f2aeca055e8d120cf15001e260c000e854.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/50cf0a4f2afc7bb6df1e59d0cbf5a6f2aeca055e8d120cf15001e260c000e854.jpg)

![540dbcdbbe1d935308d03cda10ab02b4bd5cb7c4def0a39021bdc0059724ecb8.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/540dbcdbbe1d935308d03cda10ab02b4bd5cb7c4def0a39021bdc0059724ecb8.jpg)

![56c1bb1efd6f7e88a7a6e874d36f965c78809e47acd0f6e57519e83b870e0a46.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/56c1bb1efd6f7e88a7a6e874d36f965c78809e47acd0f6e57519e83b870e0a46.jpg)

![6c5db2e8a9b89bc38bf5db2362bba828dc200bf53f2da7205018455c9902a9c0.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/6c5db2e8a9b89bc38bf5db2362bba828dc200bf53f2da7205018455c9902a9c0.jpg)

![7316a0b53137fcdc80e094b80cf97a107c8e5e84105640eba032c98f90a28b77.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/7316a0b53137fcdc80e094b80cf97a107c8e5e84105640eba032c98f90a28b77.jpg)

![7c77256265003c314feee1fb418f90eaab1cbd0cd2ad264b034682fe9dbb27d7.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/7c77256265003c314feee1fb418f90eaab1cbd0cd2ad264b034682fe9dbb27d7.jpg)

![961998e469756425a6e225e3d54cea59b03a3765a17423a8b39c017420d2a3cd.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/961998e469756425a6e225e3d54cea59b03a3765a17423a8b39c017420d2a3cd.jpg)

![9f7a025b42e0bd4838db86d3ee4edbbf1bd54c9dd49f5a23f32d881558ef485d.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/9f7a025b42e0bd4838db86d3ee4edbbf1bd54c9dd49f5a23f32d881558ef485d.jpg)

![a351fd0886287b3e8b310fd82613f82365d44edb63577308150e6cc61b2994b8.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/a351fd0886287b3e8b310fd82613f82365d44edb63577308150e6cc61b2994b8.jpg)

![af6110b6c1c899b89031fce9cf0a2738bd405b208d3a452e8bf815a34636c208.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/af6110b6c1c899b89031fce9cf0a2738bd405b208d3a452e8bf815a34636c208.jpg)

![b22a1d627b169361cfc657103e35936f16fb0e4c816ad36eca53f8357c52043a.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/b22a1d627b169361cfc657103e35936f16fb0e4c816ad36eca53f8357c52043a.jpg)

![b47ec96a4abdf650ce33e77308e6f33d7a16540c40c74db64689ac82aba776f4.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/b47ec96a4abdf650ce33e77308e6f33d7a16540c40c74db64689ac82aba776f4.jpg)

![d04b23f1f8d8cdbb9801305669378a1ebda53df93a8e9e2efd6f3e4811bfaead.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/d04b23f1f8d8cdbb9801305669378a1ebda53df93a8e9e2efd6f3e4811bfaead.jpg)

![d1cb4401515f3549633006b71685294ee27694bf25db03fc3e2256f0782dde5e.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/d1cb4401515f3549633006b71685294ee27694bf25db03fc3e2256f0782dde5e.jpg)

![d95132f57c77f72e37a575eaeb6d47d546038511a84c3e9a201c4fca101c9587.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/d95132f57c77f72e37a575eaeb6d47d546038511a84c3e9a201c4fca101c9587.jpg)

![f3411e3fad003ddf11dd449dc25f835012714f54298b68da0a72d64437a253c9.jpg](../icml_results/6194_DisPPO_%20Quantile-Based%20Distributional%20Reinforcement%20Learning%20for%20Large%20Language%20Models/tables/f3411e3fad003ddf11dd449dc25f835012714f54298b68da0a72d64437a253c9.jpg)

<a id="paper-6145"></a>

## Revisiting ML Training under Fully Homomorphic Encryption: Convergence Guarantees, Differential Privacy, and Efficient Algorithms


### Images

![7281511c67ba37c0c0de11a4c73cdd84810361cb329bcd0fcf22abbbbd3c8c48.jpg](../icml_results/6195_Revisiting%20ML%20Training%20under%20Fully%20Homomorphic%20Encryption_%20Convergence%20Guarantees%2C%20Differential%20Priv/images/7281511c67ba37c0c0de11a4c73cdd84810361cb329bcd0fcf22abbbbd3c8c48.jpg)

![80891d4c1711252ea9d4043761f1c06c4d1cdc8f565e89afac1e490204a5abd1.jpg](../icml_results/6195_Revisiting%20ML%20Training%20under%20Fully%20Homomorphic%20Encryption_%20Convergence%20Guarantees%2C%20Differential%20Priv/images/80891d4c1711252ea9d4043761f1c06c4d1cdc8f565e89afac1e490204a5abd1.jpg)

![8b3d9a4ca7e5a5a1fa6e5160e47c15fe8fc0fb4744470a247ca48199d35e37c5.jpg](../icml_results/6195_Revisiting%20ML%20Training%20under%20Fully%20Homomorphic%20Encryption_%20Convergence%20Guarantees%2C%20Differential%20Priv/images/8b3d9a4ca7e5a5a1fa6e5160e47c15fe8fc0fb4744470a247ca48199d35e37c5.jpg)

![a68f929c71424793513414dbc1646c4333bf62506c646de36dd7a1b494f493c0.jpg](../icml_results/6195_Revisiting%20ML%20Training%20under%20Fully%20Homomorphic%20Encryption_%20Convergence%20Guarantees%2C%20Differential%20Priv/images/a68f929c71424793513414dbc1646c4333bf62506c646de36dd7a1b494f493c0.jpg)

![e5de58d7ff08b396bcd405da94b514fa02838ca7be03b90ba8b4ebe1afebfe1a.jpg](../icml_results/6195_Revisiting%20ML%20Training%20under%20Fully%20Homomorphic%20Encryption_%20Convergence%20Guarantees%2C%20Differential%20Priv/images/e5de58d7ff08b396bcd405da94b514fa02838ca7be03b90ba8b4ebe1afebfe1a.jpg)

![fdb25f4fb8d3c8c0de2f7ea8d9c20d1e6be24b9fdb18c5131334f1beb6ad4d63.jpg](../icml_results/6195_Revisiting%20ML%20Training%20under%20Fully%20Homomorphic%20Encryption_%20Convergence%20Guarantees%2C%20Differential%20Priv/images/fdb25f4fb8d3c8c0de2f7ea8d9c20d1e6be24b9fdb18c5131334f1beb6ad4d63.jpg)

### Tables

![400ce1fb59f38234be582c16ac3c469fbbbe5f0ffca7e1822988297d1c6af3e0.jpg](../icml_results/6195_Revisiting%20ML%20Training%20under%20Fully%20Homomorphic%20Encryption_%20Convergence%20Guarantees%2C%20Differential%20Priv/tables/400ce1fb59f38234be582c16ac3c469fbbbe5f0ffca7e1822988297d1c6af3e0.jpg)

![58646da16398f17b2b27a9d63d36f24c52467a541e9d6708b43445c3a661dfd6.jpg](../icml_results/6195_Revisiting%20ML%20Training%20under%20Fully%20Homomorphic%20Encryption_%20Convergence%20Guarantees%2C%20Differential%20Priv/tables/58646da16398f17b2b27a9d63d36f24c52467a541e9d6708b43445c3a661dfd6.jpg)

![67a9349c4cec6eed2035997574cf194b6dd5b0c98096467b11905a835073c46a.jpg](../icml_results/6195_Revisiting%20ML%20Training%20under%20Fully%20Homomorphic%20Encryption_%20Convergence%20Guarantees%2C%20Differential%20Priv/tables/67a9349c4cec6eed2035997574cf194b6dd5b0c98096467b11905a835073c46a.jpg)

![8852777d6eb838f7817a11bb15ce6057c773fb09a36f5a40fa76b8bd839dbc1b.jpg](../icml_results/6195_Revisiting%20ML%20Training%20under%20Fully%20Homomorphic%20Encryption_%20Convergence%20Guarantees%2C%20Differential%20Priv/tables/8852777d6eb838f7817a11bb15ce6057c773fb09a36f5a40fa76b8bd839dbc1b.jpg)

![be99f1af4dc589e0c9d2dfece7be7605c23f5ea7042228caaeec28d13ca7e1fd.jpg](../icml_results/6195_Revisiting%20ML%20Training%20under%20Fully%20Homomorphic%20Encryption_%20Convergence%20Guarantees%2C%20Differential%20Priv/tables/be99f1af4dc589e0c9d2dfece7be7605c23f5ea7042228caaeec28d13ca7e1fd.jpg)

![ca80e7107f5e4dd9f6be0b94ca1876a83003a69eee0438b86c74a32c116b320a.jpg](../icml_results/6195_Revisiting%20ML%20Training%20under%20Fully%20Homomorphic%20Encryption_%20Convergence%20Guarantees%2C%20Differential%20Priv/tables/ca80e7107f5e4dd9f6be0b94ca1876a83003a69eee0438b86c74a32c116b320a.jpg)

![d8b2cb38fa0f717737e6efd916d98635225e0c7825c2236aa9115973a56fd071.jpg](../icml_results/6195_Revisiting%20ML%20Training%20under%20Fully%20Homomorphic%20Encryption_%20Convergence%20Guarantees%2C%20Differential%20Priv/tables/d8b2cb38fa0f717737e6efd916d98635225e0c7825c2236aa9115973a56fd071.jpg)

<a id="paper-6146"></a>

## Positional Encoding for Spiking Transformers


### Images

![02560b9ffe4c46d98ac0f33ba4e6ee173e388771d4041df8c3254ca284c0cb20.jpg](../icml_results/6196_Positional%20Encoding%20for%20Spiking%20Transformers/images/02560b9ffe4c46d98ac0f33ba4e6ee173e388771d4041df8c3254ca284c0cb20.jpg)

![02d41ce2eb7d5b8dee63ec75fe35ab6f7b3bb02767f891017729a7f7e08e2bed.jpg](../icml_results/6196_Positional%20Encoding%20for%20Spiking%20Transformers/images/02d41ce2eb7d5b8dee63ec75fe35ab6f7b3bb02767f891017729a7f7e08e2bed.jpg)

![0b858af8d6877dd7e6885bd974628a19dc8107150c001f80b6f1ad7d889da962.jpg](../icml_results/6196_Positional%20Encoding%20for%20Spiking%20Transformers/images/0b858af8d6877dd7e6885bd974628a19dc8107150c001f80b6f1ad7d889da962.jpg)

![18eb029e3a43540d2e11de4e2d9a14377ff2ac598140fb002ad6b43a2504619b.jpg](../icml_results/6196_Positional%20Encoding%20for%20Spiking%20Transformers/images/18eb029e3a43540d2e11de4e2d9a14377ff2ac598140fb002ad6b43a2504619b.jpg)

![54818f613d7365a4193619539a68d98a76148307ca4869a86ede8c76e5edec42.jpg](../icml_results/6196_Positional%20Encoding%20for%20Spiking%20Transformers/images/54818f613d7365a4193619539a68d98a76148307ca4869a86ede8c76e5edec42.jpg)

![582a04a3e388af1f3e12f51ef3628bf944f34daadd945600c09df980c664ae82.jpg](../icml_results/6196_Positional%20Encoding%20for%20Spiking%20Transformers/images/582a04a3e388af1f3e12f51ef3628bf944f34daadd945600c09df980c664ae82.jpg)

![5d9d82368a6bdfa409d75b0a46b9366b9415076ea152a76b540614d857842301.jpg](../icml_results/6196_Positional%20Encoding%20for%20Spiking%20Transformers/images/5d9d82368a6bdfa409d75b0a46b9366b9415076ea152a76b540614d857842301.jpg)

![649b8bd17e168c8f911fa3a482716e84c78cc3472e9681afd8d96b4182d3cd23.jpg](../icml_results/6196_Positional%20Encoding%20for%20Spiking%20Transformers/images/649b8bd17e168c8f911fa3a482716e84c78cc3472e9681afd8d96b4182d3cd23.jpg)

![c36b3cc8dca36239112b4cc2a0a43b7c3937126d15dd37466109f69ae609733e.jpg](../icml_results/6196_Positional%20Encoding%20for%20Spiking%20Transformers/images/c36b3cc8dca36239112b4cc2a0a43b7c3937126d15dd37466109f69ae609733e.jpg)

### Tables

![3092eb4e0a6405fc37d1d7dd16f56238244542e1a368667de83dc576573c4f33.jpg](../icml_results/6196_Positional%20Encoding%20for%20Spiking%20Transformers/tables/3092eb4e0a6405fc37d1d7dd16f56238244542e1a368667de83dc576573c4f33.jpg)

![512be4c37f3b3a562a16da78118f4029ddca1c917d2b211ec8b83f818cb6816f.jpg](../icml_results/6196_Positional%20Encoding%20for%20Spiking%20Transformers/tables/512be4c37f3b3a562a16da78118f4029ddca1c917d2b211ec8b83f818cb6816f.jpg)

![c1e689b59a176726fe1261dc6bc3e7bae322baa42cc9599eb8713ecad560dd2e.jpg](../icml_results/6196_Positional%20Encoding%20for%20Spiking%20Transformers/tables/c1e689b59a176726fe1261dc6bc3e7bae322baa42cc9599eb8713ecad560dd2e.jpg)

<a id="paper-6147"></a>

## $\mathcalO(\log N)$ Latent Dimension Suffices for Universal Approximation of Permutation-invariant Function


### Images

![83234b17a21452131f03e776a87035ed064034e3008a813f3b6b7e4b9f23312d.jpg](../icml_results/6197_%24_mathcalO%28_log%20N%29%24%20Latent%20Dimension%20Suffices%20for%20Universal%20Approximation%20of%20Permutation-invariant%20F/images/83234b17a21452131f03e776a87035ed064034e3008a813f3b6b7e4b9f23312d.jpg)

![ac93d79fcf11b3a5dbed1aee9891d90cc866f23cb2ee922a24d2cc8594e8267a.jpg](../icml_results/6197_%24_mathcalO%28_log%20N%29%24%20Latent%20Dimension%20Suffices%20for%20Universal%20Approximation%20of%20Permutation-invariant%20F/images/ac93d79fcf11b3a5dbed1aee9891d90cc866f23cb2ee922a24d2cc8594e8267a.jpg)

![babd870f0abd809b3db463a63d9e681ccc37f2c66af7043725a8814c9102ba87.jpg](../icml_results/6197_%24_mathcalO%28_log%20N%29%24%20Latent%20Dimension%20Suffices%20for%20Universal%20Approximation%20of%20Permutation-invariant%20F/images/babd870f0abd809b3db463a63d9e681ccc37f2c66af7043725a8814c9102ba87.jpg)

![ddeedc3e8bac833e05902ef912401c32cf23c95c953580961f90078000afde97.jpg](../icml_results/6197_%24_mathcalO%28_log%20N%29%24%20Latent%20Dimension%20Suffices%20for%20Universal%20Approximation%20of%20Permutation-invariant%20F/images/ddeedc3e8bac833e05902ef912401c32cf23c95c953580961f90078000afde97.jpg)

### Tables

![10032ae1f411e1ba4375360e55b1fdada7c982d5585c3c35177a5b9b8677e149.jpg](../icml_results/6197_%24_mathcalO%28_log%20N%29%24%20Latent%20Dimension%20Suffices%20for%20Universal%20Approximation%20of%20Permutation-invariant%20F/tables/10032ae1f411e1ba4375360e55b1fdada7c982d5585c3c35177a5b9b8677e149.jpg)

![10110e863eb436301efbf150d0e853f7d8f7c2321c8e50e49d47a3e139c2519a.jpg](../icml_results/6197_%24_mathcalO%28_log%20N%29%24%20Latent%20Dimension%20Suffices%20for%20Universal%20Approximation%20of%20Permutation-invariant%20F/tables/10110e863eb436301efbf150d0e853f7d8f7c2321c8e50e49d47a3e139c2519a.jpg)

<a id="paper-6148"></a>

## GeoAlign: Geometric Rollout Curation for Robust LLM Reinforcement Learning


### Images

![02485a4e9867818165bd0d8a3eae339298e3f0c94ce6daa1401420e90063d884.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/images/02485a4e9867818165bd0d8a3eae339298e3f0c94ce6daa1401420e90063d884.jpg)

![35abae1255becfcb18afe2f26b04c9180e6ccf605f6937fac4617472a2f5194a.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/images/35abae1255becfcb18afe2f26b04c9180e6ccf605f6937fac4617472a2f5194a.jpg)

![3b44768bcb3c96193aae6ed5c9d3f66e310ee777407c1a864f895cd3104ca801.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/images/3b44768bcb3c96193aae6ed5c9d3f66e310ee777407c1a864f895cd3104ca801.jpg)

![545980c3274a01e3fd631d2c5a567d150e39194aab3c9eb79ad499d8ebd5ee78.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/images/545980c3274a01e3fd631d2c5a567d150e39194aab3c9eb79ad499d8ebd5ee78.jpg)

![6c0155d4dd939231ef73c51f4a54000c2c4015d7f9a64afa25396cd0c0e305dc.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/images/6c0155d4dd939231ef73c51f4a54000c2c4015d7f9a64afa25396cd0c0e305dc.jpg)

![71b35f364389619e56ddda47d3c2a4cd6386feb57018b03be6df580f61a18159.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/images/71b35f364389619e56ddda47d3c2a4cd6386feb57018b03be6df580f61a18159.jpg)

![98bfbf02626f3c94943071390120d1a4a4eedc38f595fc8aa3fdee4a456ca172.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/images/98bfbf02626f3c94943071390120d1a4a4eedc38f595fc8aa3fdee4a456ca172.jpg)

![d6f7dcab3308bd36eefef7d727402071304bb51114d2c10bb8408c795d307d0f.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/images/d6f7dcab3308bd36eefef7d727402071304bb51114d2c10bb8408c795d307d0f.jpg)

![dd3bf05864534356dd57fe4e4e5cd7f1e6f8c8e57f68207cf7a11d6e3b0b704e.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/images/dd3bf05864534356dd57fe4e4e5cd7f1e6f8c8e57f68207cf7a11d6e3b0b704e.jpg)

![ea5d5dbab195adeaee58f3b7609fe5028bf27af1240fa62e10b8ab502ab406a8.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/images/ea5d5dbab195adeaee58f3b7609fe5028bf27af1240fa62e10b8ab502ab406a8.jpg)

![f8fcef35b250c6febeee5650c57fbc1dd283a5afdea3d2e360040a2dce25c71f.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/images/f8fcef35b250c6febeee5650c57fbc1dd283a5afdea3d2e360040a2dce25c71f.jpg)

![ff4041e1133a21d2e4cac71a756744db20a86e3d5eb4b8e4531c27216cf14caa.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/images/ff4041e1133a21d2e4cac71a756744db20a86e3d5eb4b8e4531c27216cf14caa.jpg)

### Tables

![0b683bd86a007b4c23ad32a8ce57e000246065b71f3b04765d623738eb35a6e3.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/tables/0b683bd86a007b4c23ad32a8ce57e000246065b71f3b04765d623738eb35a6e3.jpg)

![4d28c18cbb180c100b9b18b4d32d158d9887c4808ae54d8c4ee44d5acf742ee4.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/tables/4d28c18cbb180c100b9b18b4d32d158d9887c4808ae54d8c4ee44d5acf742ee4.jpg)

![4fbf5ab34dc1fffbb84ab4ed7bef9d17b5d8ffc0b840dc4f613620560e0c8915.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/tables/4fbf5ab34dc1fffbb84ab4ed7bef9d17b5d8ffc0b840dc4f613620560e0c8915.jpg)

![5bd422da3d31251b8ef4b930ab17d0758f3f78dbcb5c453d72e6910d0f1e9e9b.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/tables/5bd422da3d31251b8ef4b930ab17d0758f3f78dbcb5c453d72e6910d0f1e9e9b.jpg)

![5c73a498729b6d206b24190f5ca65c51b658afd6be7889b15d6965ab8a6b3123.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/tables/5c73a498729b6d206b24190f5ca65c51b658afd6be7889b15d6965ab8a6b3123.jpg)

![69731559fef96149568737b6e456b6a5ec86300702d83c10ad9bf802e526755e.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/tables/69731559fef96149568737b6e456b6a5ec86300702d83c10ad9bf802e526755e.jpg)

![734bd6c6436e83b12ae167ae2f58835417bff9950feb94697494c058ee8a1003.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/tables/734bd6c6436e83b12ae167ae2f58835417bff9950feb94697494c058ee8a1003.jpg)

![763bd7091de721f241cd35fe6e98b603d138d88394c17db9cf5af5879f3719ec.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/tables/763bd7091de721f241cd35fe6e98b603d138d88394c17db9cf5af5879f3719ec.jpg)

![765fb9474718e532988b85d7e123c2f2159507c99c819b91564a72605525593f.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/tables/765fb9474718e532988b85d7e123c2f2159507c99c819b91564a72605525593f.jpg)

![7bd42430a0c07ca5a1be22b12a18085ecf35b58c116fa57b04b0b6dac02648d9.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/tables/7bd42430a0c07ca5a1be22b12a18085ecf35b58c116fa57b04b0b6dac02648d9.jpg)

![8de09e75bb86fca09b52b71c0a8d170d41385a4cc9dcbdfa7b0d3ddffbfa9599.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/tables/8de09e75bb86fca09b52b71c0a8d170d41385a4cc9dcbdfa7b0d3ddffbfa9599.jpg)

![c1cfdb56eddbe2432eb266aedc1b99865c20156be7574f64f507ddd54ba90777.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/tables/c1cfdb56eddbe2432eb266aedc1b99865c20156be7574f64f507ddd54ba90777.jpg)

![d573116b83c1201bc3f55f09d80d5017ddd3564c6d4680f6ec1a861db2755a69.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/tables/d573116b83c1201bc3f55f09d80d5017ddd3564c6d4680f6ec1a861db2755a69.jpg)

![e8c12d8c5e3b2dd27c4addc243e829610e7fe895ea49003faac5bdafe5fd32bd.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/tables/e8c12d8c5e3b2dd27c4addc243e829610e7fe895ea49003faac5bdafe5fd32bd.jpg)

![edec0d995a4b0dd79c0b83b46044e8abe266c32dd74499260a62d9275c1d89a9.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/tables/edec0d995a4b0dd79c0b83b46044e8abe266c32dd74499260a62d9275c1d89a9.jpg)

![f2c263c6ff40b3ff4bd784e3468557b9357ed21a8f42534eab439f360982faab.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/tables/f2c263c6ff40b3ff4bd784e3468557b9357ed21a8f42534eab439f360982faab.jpg)

![fc3edf320e8ef39d3680a8479a7f5564ca8e7d47bb3d1f392e7fe33494e1e518.jpg](../icml_results/6198_GeoAlign_%20Geometric%20Rollout%20Curation%20for%20Robust%20LLM%20Reinforcement%20Learning/tables/fc3edf320e8ef39d3680a8479a7f5564ca8e7d47bb3d1f392e7fe33494e1e518.jpg)

<a id="paper-6149"></a>

## Meerkat-VL: Implicit Risk Safety Alignment in Multimodal LLMs via Perceptual Reasoning and Self-Verification


### Images

![03ac9f635ab622b089684515f44ded36925bdee157dc7443ecfa8c69e9d28aff.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/03ac9f635ab622b089684515f44ded36925bdee157dc7443ecfa8c69e9d28aff.jpg)

![03e5acab1d0bf06d2f26ff3e0ced0a8f82fb55db5749b22c48bf8174e6a99b64.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/03e5acab1d0bf06d2f26ff3e0ced0a8f82fb55db5749b22c48bf8174e6a99b64.jpg)

![1904197dce9af72b0c6e493af04c4776dcebb2691644afb11ca86e989259d3b4.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/1904197dce9af72b0c6e493af04c4776dcebb2691644afb11ca86e989259d3b4.jpg)

![22c8f73f63ee56caf0350de7ab9a32a00676bd91d435655d934049146c413d7c.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/22c8f73f63ee56caf0350de7ab9a32a00676bd91d435655d934049146c413d7c.jpg)

![2485e13602fe88342510d3a534dd029c860c073537f9464cb354eaed0a20090c.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/2485e13602fe88342510d3a534dd029c860c073537f9464cb354eaed0a20090c.jpg)

![3360be4b5d7c47285596adf604ed7352e58c7b8a55cb7f78c34937fc4d5bb20f.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/3360be4b5d7c47285596adf604ed7352e58c7b8a55cb7f78c34937fc4d5bb20f.jpg)

![3acc8fccd2479b3a196e00269496ceeb231faba8529b94684a6676ae06dcfed5.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/3acc8fccd2479b3a196e00269496ceeb231faba8529b94684a6676ae06dcfed5.jpg)

![418f9a19996ccdf5b8bd0dbcbb19f4e9a00b34c377b2182c82aaa8b76b88a79b.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/418f9a19996ccdf5b8bd0dbcbb19f4e9a00b34c377b2182c82aaa8b76b88a79b.jpg)

![4a081971f395815f9c6f0aeed6f45bd7952bcd23258823c6aba7cc244ee4b77c.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/4a081971f395815f9c6f0aeed6f45bd7952bcd23258823c6aba7cc244ee4b77c.jpg)

![4a0d2f1679efaf1b56575f7a40ba7a7d687dad38ee12e8f3a107bbde90b0e66d.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/4a0d2f1679efaf1b56575f7a40ba7a7d687dad38ee12e8f3a107bbde90b0e66d.jpg)

![4cee378552ea2feb0be713d234a88d853db51854ffb1540feb815aad92d76a49.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/4cee378552ea2feb0be713d234a88d853db51854ffb1540feb815aad92d76a49.jpg)

![76724d1629d58cb2e6eb833bbba67a9373c775d4958aa30b4d985133226fccda.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/76724d1629d58cb2e6eb833bbba67a9373c775d4958aa30b4d985133226fccda.jpg)

![84f26e6bc62bb0a2cd95ec1b5ddb692a02b53755bbb39487a15328ecc751730c.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/84f26e6bc62bb0a2cd95ec1b5ddb692a02b53755bbb39487a15328ecc751730c.jpg)

![84fcf4c6a3ba652bd2b2482c41f2f2d760e585b83aab8616d0f03be3cf724824.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/84fcf4c6a3ba652bd2b2482c41f2f2d760e585b83aab8616d0f03be3cf724824.jpg)

![89abf0d3dcbf9261470705ed0b823cb53d66f0ee17156f84d85a309e5bb05456.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/89abf0d3dcbf9261470705ed0b823cb53d66f0ee17156f84d85a309e5bb05456.jpg)

![925cf11ea8551b015806180f85a51413a3a1f60889a070ee966ce219a946459b.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/925cf11ea8551b015806180f85a51413a3a1f60889a070ee966ce219a946459b.jpg)

![992a5cbf4391331fb203b6e8cb9414cd9b5ecc33c40ecdcf622420c5a0020689.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/992a5cbf4391331fb203b6e8cb9414cd9b5ecc33c40ecdcf622420c5a0020689.jpg)

![b5c34cc959ad72d86b51cb58be3f6d106a43977f5ccb4abf4e096e863b44513d.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/b5c34cc959ad72d86b51cb58be3f6d106a43977f5ccb4abf4e096e863b44513d.jpg)

![babd40378bd24640c73f5d5b777aac6ab48841c79d004c073dc543b21c4a58ca.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/babd40378bd24640c73f5d5b777aac6ab48841c79d004c073dc543b21c4a58ca.jpg)

![c9c059942102976e3b28a99a91231c906b5189a7516e699c38eb27ef41a7e128.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/c9c059942102976e3b28a99a91231c906b5189a7516e699c38eb27ef41a7e128.jpg)

![d0479e14b3619d8d79948b2f64fb03781fb3ae123793ce2e2c12434ef8b1a181.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/d0479e14b3619d8d79948b2f64fb03781fb3ae123793ce2e2c12434ef8b1a181.jpg)

![dee28ba40ec9b4be11e3573d6ecb5efa5673f4d0c8e917c45e838c9f6714f7d4.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/dee28ba40ec9b4be11e3573d6ecb5efa5673f4d0c8e917c45e838c9f6714f7d4.jpg)

![df46920e15b836ad345ad6648484ed007bf50306cc373bb9dc5898dbbf7882ce.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/df46920e15b836ad345ad6648484ed007bf50306cc373bb9dc5898dbbf7882ce.jpg)

![e297531967e881f63646811eea7a5bc732efee809da9b90573b2251b9b27d2c7.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/e297531967e881f63646811eea7a5bc732efee809da9b90573b2251b9b27d2c7.jpg)

![eed5d4c03bca96212445ea0029017868bc852a79dad9509de329953d0788450c.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/eed5d4c03bca96212445ea0029017868bc852a79dad9509de329953d0788450c.jpg)

![f6dc78179cb8d9f046652ad187151c60c39d4d4715a3a29b4d6725552a8675a8.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/f6dc78179cb8d9f046652ad187151c60c39d4d4715a3a29b4d6725552a8675a8.jpg)

![f7a62d4bb48bb49b8ec6c0e3609993ba5cdcc98eefd1a32ee422d75a16426de8.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/f7a62d4bb48bb49b8ec6c0e3609993ba5cdcc98eefd1a32ee422d75a16426de8.jpg)

![f8d73c1d87c3ec7d8cea0ae140c2c0f0c6a9f5bcea6ca61a52b9dc5b3f798921.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/f8d73c1d87c3ec7d8cea0ae140c2c0f0c6a9f5bcea6ca61a52b9dc5b3f798921.jpg)

![f97bafa8391ac39fd2333e374ee7bc174503a700690b315a3de4e88dc4054eb6.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/f97bafa8391ac39fd2333e374ee7bc174503a700690b315a3de4e88dc4054eb6.jpg)

![ff597b5bd9e8c9bf8b28147701af8fae505933dac690c060407f14a67a1c5b2e.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/images/ff597b5bd9e8c9bf8b28147701af8fae505933dac690c060407f14a67a1c5b2e.jpg)

### Tables

![0d8e8d53a07d4a0de6d7130436e68fd6878c1343ce81b019b4aa646720c83300.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/tables/0d8e8d53a07d4a0de6d7130436e68fd6878c1343ce81b019b4aa646720c83300.jpg)

![0f7d774aac136516276bd9723f5a1d835a474238c1e66b068f45c249c0344467.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/tables/0f7d774aac136516276bd9723f5a1d835a474238c1e66b068f45c249c0344467.jpg)

![105c04f299e5a7731078720fe251195a215c0e93fb25ee479852105fc5d70038.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/tables/105c04f299e5a7731078720fe251195a215c0e93fb25ee479852105fc5d70038.jpg)

![29aed6b9cfcaa59afb0da66d3f29c1588bfeb0473fd8c373a2c401373eeae997.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/tables/29aed6b9cfcaa59afb0da66d3f29c1588bfeb0473fd8c373a2c401373eeae997.jpg)

![35e564511849d81081a1900ce84dfd44708060ee778afadd72eebd23b6510b9f.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/tables/35e564511849d81081a1900ce84dfd44708060ee778afadd72eebd23b6510b9f.jpg)

![736766c4df07b8d39a007abb9067cfffc5b4363190325c40a83f06c6987e7835.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/tables/736766c4df07b8d39a007abb9067cfffc5b4363190325c40a83f06c6987e7835.jpg)

![a495e75d17c485896d5f2a23e6bd9ba4135586b01b097b586cccb9a6541af607.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/tables/a495e75d17c485896d5f2a23e6bd9ba4135586b01b097b586cccb9a6541af607.jpg)

![debe3fbb072d4684bda050eccf1dc36ce937c4da9cf54643c971d6473db1bf66.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/tables/debe3fbb072d4684bda050eccf1dc36ce937c4da9cf54643c971d6473db1bf66.jpg)

![ec496cf69837214d7492d2230807574b3bc39209ed4ded1a0d524d5465fbc2eb.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/tables/ec496cf69837214d7492d2230807574b3bc39209ed4ded1a0d524d5465fbc2eb.jpg)

![ecdcdc867be5679717ab02c6e7ead5ae0d16a5ce00dfa32175758a29b351128e.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/tables/ecdcdc867be5679717ab02c6e7ead5ae0d16a5ce00dfa32175758a29b351128e.jpg)

![edc57f804741e9c73957b6a616b4ee7c343cb0cc9e55ba9f728ecf6a36fc0344.jpg](../icml_results/6199_Meerkat-VL_%20Implicit%20Risk%20Safety%20Alignment%20in%20Multimodal%20LLMs%20via%20Perceptual%20Reasoning%20and%20Self-Veri/tables/edc57f804741e9c73957b6a616b4ee7c343cb0cc9e55ba9f728ecf6a36fc0344.jpg)

<a id="paper-6150"></a>

## InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition


### Images

![0fe1fbee9217611275ecf635703a2b5d23e7f6e978d7814a4fee79bd4d123010.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/images/0fe1fbee9217611275ecf635703a2b5d23e7f6e978d7814a4fee79bd4d123010.jpg)

![2c7590de10bff85c700bd71ddb1ca6904dd150f58af0ccd6865fe1b72dded045.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/images/2c7590de10bff85c700bd71ddb1ca6904dd150f58af0ccd6865fe1b72dded045.jpg)

![5993fc071931ebfee6a91eae60a57e2c6d8e91e6465543d0620c1cf786f5eb4a.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/images/5993fc071931ebfee6a91eae60a57e2c6d8e91e6465543d0620c1cf786f5eb4a.jpg)

![5d0aecc30829689342660511f025cd7c1cbb6cfd192055c2c661fc29f5dd7f44.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/images/5d0aecc30829689342660511f025cd7c1cbb6cfd192055c2c661fc29f5dd7f44.jpg)

![62662d8f58a8f6a9c735a08bea3776efdf41577d73c28fb1ea069a57f2a046ef.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/images/62662d8f58a8f6a9c735a08bea3776efdf41577d73c28fb1ea069a57f2a046ef.jpg)

![6ef30ad19dfef9e6a208995726d05db932e2af4cb60d2a54a80160ba253b3af7.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/images/6ef30ad19dfef9e6a208995726d05db932e2af4cb60d2a54a80160ba253b3af7.jpg)

![8288a34c7f45c4f34821ebaddebecc6afd648256e23c393ad56c53add8507241.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/images/8288a34c7f45c4f34821ebaddebecc6afd648256e23c393ad56c53add8507241.jpg)

![832682edf6d6f8b0309a1fcc2b8b0783b8964bdd7110e8d044de4ff1aba45057.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/images/832682edf6d6f8b0309a1fcc2b8b0783b8964bdd7110e8d044de4ff1aba45057.jpg)

![abe546788b4c16c121662f40067191cb73ee2a6798c0d310eb96f5a743efe365.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/images/abe546788b4c16c121662f40067191cb73ee2a6798c0d310eb96f5a743efe365.jpg)

![b7064dc797f8247da77b1e6eaa7fa340978fd7d5046e983fc246a3c8c682a931.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/images/b7064dc797f8247da77b1e6eaa7fa340978fd7d5046e983fc246a3c8c682a931.jpg)

![cb7c87d9eea8faaba596ded8d4466cdb980cfe8404a3aa1a43002799e3f1d75d.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/images/cb7c87d9eea8faaba596ded8d4466cdb980cfe8404a3aa1a43002799e3f1d75d.jpg)

![f57641d17db7fba79eaa6e58c77a06c43801c3dc6f11e02a8949f7faf347f745.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/images/f57641d17db7fba79eaa6e58c77a06c43801c3dc6f11e02a8949f7faf347f745.jpg)

![f82493dbd599ab3bc16177c757e379d3f8d5052e737e154c17cb46a83aab4803.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/images/f82493dbd599ab3bc16177c757e379d3f8d5052e737e154c17cb46a83aab4803.jpg)

### Tables

![00b3a6203908dff573fc90ddc5198d91a1394ff6e43c9d5ef7bce9301b11bb9f.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/tables/00b3a6203908dff573fc90ddc5198d91a1394ff6e43c9d5ef7bce9301b11bb9f.jpg)

![00f42daf8dcae6f9459f8692bfa171c513c7fc935873d30177dc8f78ceca6df8.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/tables/00f42daf8dcae6f9459f8692bfa171c513c7fc935873d30177dc8f78ceca6df8.jpg)

![2ca49f1f7cf45f84ec76d0fda9f5773e3164e7cf20c21f28884718ed1489efac.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/tables/2ca49f1f7cf45f84ec76d0fda9f5773e3164e7cf20c21f28884718ed1489efac.jpg)

![447e46ca8b596652672c7f9b2df3edd9154a605fcedb873bbc27c8b4f6c51cd2.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/tables/447e46ca8b596652672c7f9b2df3edd9154a605fcedb873bbc27c8b4f6c51cd2.jpg)

![531288f7f0beb771376ab37e74b6539d674d5cf69dcfb25b1b2f385182ce1ab6.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/tables/531288f7f0beb771376ab37e74b6539d674d5cf69dcfb25b1b2f385182ce1ab6.jpg)

![5b08651581f18a5528fd7109dccbee14ebad6e3560aaf1b710f31e48d13d4fdb.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/tables/5b08651581f18a5528fd7109dccbee14ebad6e3560aaf1b710f31e48d13d4fdb.jpg)

![60e9bac966bdaaafd4ee2e68531400beb7594ddbb49990c3e5244177fabf9d76.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/tables/60e9bac966bdaaafd4ee2e68531400beb7594ddbb49990c3e5244177fabf9d76.jpg)

![7c8fce1b8f52d4b45851fd847addd3603b650b7b3b395433be7ae1412ca288f5.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/tables/7c8fce1b8f52d4b45851fd847addd3603b650b7b3b395433be7ae1412ca288f5.jpg)

![a13c333cccff8bc0843fce3cc375c3cffd1e64568d6cf7ab62b3d86358895e67.jpg](../icml_results/6200_InfoLaw_%20Information%20Scaling%20Laws%20for%20Large%20Language%20Models%20with%20Quality-Weighted%20Mixture%20Data%20and%20R/tables/a13c333cccff8bc0843fce3cc375c3cffd1e64568d6cf7ab62b3d86358895e67.jpg)

<a id="paper-6151"></a>

## OmniShow: Unifying Multimodal Conditions for Human-Object Interaction Video Generation


### Images

![2c387bfac81e1d447c41ee303a7e7dadb9e427883943ee690fadb10452fe1e1a.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/images/2c387bfac81e1d447c41ee303a7e7dadb9e427883943ee690fadb10452fe1e1a.jpg)

![3259754de6d32daa46cce5533f6a926be58b95bd22a5e6d9dddc5279b26b2907.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/images/3259754de6d32daa46cce5533f6a926be58b95bd22a5e6d9dddc5279b26b2907.jpg)

![39275672f30bee0adab36ecb340fc42e62a7d74aa9f49258c94cb5edffb40c9b.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/images/39275672f30bee0adab36ecb340fc42e62a7d74aa9f49258c94cb5edffb40c9b.jpg)

![621f42733c818c96c8e5ac86e896cfa2244ba526cd4838ca64840a1a7e6f0c85.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/images/621f42733c818c96c8e5ac86e896cfa2244ba526cd4838ca64840a1a7e6f0c85.jpg)

![644e46bcf24002da3b1dea2520ca239d6d022ae25709e205e96db2a515b8afa7.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/images/644e46bcf24002da3b1dea2520ca239d6d022ae25709e205e96db2a515b8afa7.jpg)

![6dcf92bc024702dee9092d66d9f2fd4c0b3bc72d589277fed6ade7d12a41ccf3.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/images/6dcf92bc024702dee9092d66d9f2fd4c0b3bc72d589277fed6ade7d12a41ccf3.jpg)

![8f81fc48eb069f407c1c7659b0c660fa58ddbdbbd79b73d9c7abdf400fc1df02.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/images/8f81fc48eb069f407c1c7659b0c660fa58ddbdbbd79b73d9c7abdf400fc1df02.jpg)

![a4399c4208ea3438287c0b071ba2ee748044050f68e18993a68d25394512beac.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/images/a4399c4208ea3438287c0b071ba2ee748044050f68e18993a68d25394512beac.jpg)

![b2e86b10a18c392c66527c48d6fb1f94823ede48a9d3f4ea51c8796480b6e483.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/images/b2e86b10a18c392c66527c48d6fb1f94823ede48a9d3f4ea51c8796480b6e483.jpg)

![b4d291b5d1c4b47637c1e57a774ddb62d8983dff403539fa6dd6732fe67b50c7.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/images/b4d291b5d1c4b47637c1e57a774ddb62d8983dff403539fa6dd6732fe67b50c7.jpg)

![b97e6bf13eb2f16e5ad3f115234a8e3cd41c9e04ffde17ff1681039629a36ee1.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/images/b97e6bf13eb2f16e5ad3f115234a8e3cd41c9e04ffde17ff1681039629a36ee1.jpg)

![bd2a6d22568707a70fed9720b50c359b48e75ae17a56474a42f14bd03bdcda61.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/images/bd2a6d22568707a70fed9720b50c359b48e75ae17a56474a42f14bd03bdcda61.jpg)

![f8f0c401e015429468af11dc3121393c61e112b2b554539d2c6b5f676879cfe5.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/images/f8f0c401e015429468af11dc3121393c61e112b2b554539d2c6b5f676879cfe5.jpg)

![ff7671b55c979f0bc855eaf686f3d0992a88436e4035d2117d232209ae3eba61.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/images/ff7671b55c979f0bc855eaf686f3d0992a88436e4035d2117d232209ae3eba61.jpg)

### Tables

![111652953665184d65911995e859f782bde0031320a2b89bfad328bd34e9868a.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/tables/111652953665184d65911995e859f782bde0031320a2b89bfad328bd34e9868a.jpg)

![4711b9c33a68513f4bbc231fd2d996e5cd4f3bcec73ce3fd87bc9db4ffedb6d4.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/tables/4711b9c33a68513f4bbc231fd2d996e5cd4f3bcec73ce3fd87bc9db4ffedb6d4.jpg)

![5bce3d4e9d737bebdaedfe5786cbaa67ac8aee7c4c306da1508083fae4d0891f.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/tables/5bce3d4e9d737bebdaedfe5786cbaa67ac8aee7c4c306da1508083fae4d0891f.jpg)

![7f80c6fcb907abea7dc0afd53048b3828b22fc6787004ec2cc156325cec955d0.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/tables/7f80c6fcb907abea7dc0afd53048b3828b22fc6787004ec2cc156325cec955d0.jpg)

![8f26313dbed2874b3a6efcd71b9c89417dd30a431420164e32ca279d1762cdff.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/tables/8f26313dbed2874b3a6efcd71b9c89417dd30a431420164e32ca279d1762cdff.jpg)

![a605d46fd670805a913aef99e5cb3d3d232dc0747057571a9ea31d541052dddf.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/tables/a605d46fd670805a913aef99e5cb3d3d232dc0747057571a9ea31d541052dddf.jpg)

![b7f80b63f7f3180088f5fb4def1bb27786a2bd715e44f67896b9a975bad0db66.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/tables/b7f80b63f7f3180088f5fb4def1bb27786a2bd715e44f67896b9a975bad0db66.jpg)

![dde2cdee5e403999ebd333dabc3b6ac7a122dc4815bed1424bfac7dff0dbf3a9.jpg](../icml_results/6201_OmniShow_%20Unifying%20Multimodal%20Conditions%20for%20Human-Object%20Interaction%20Video%20Generation/tables/dde2cdee5e403999ebd333dabc3b6ac7a122dc4815bed1424bfac7dff0dbf3a9.jpg)

<a id="paper-6152"></a>

## The Easy, the Hard, and the Learnable: Confidence and Difficulty-Adaptive Policy Optimization for LLM Reasoning


### Images

![0407c29705ada34da6fefbd7dfeac84f1d51e80bda3cc7d70d54ddd837364183.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/images/0407c29705ada34da6fefbd7dfeac84f1d51e80bda3cc7d70d54ddd837364183.jpg)

![42ffff7e8c07e1dc7def87f18626d6b115740f0a51e2694b8a8840fb37609bb4.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/images/42ffff7e8c07e1dc7def87f18626d6b115740f0a51e2694b8a8840fb37609bb4.jpg)

![4679460b37d26f4479ef3844d442334a9d900cdef8eceb2966bcf52a6a2991e7.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/images/4679460b37d26f4479ef3844d442334a9d900cdef8eceb2966bcf52a6a2991e7.jpg)

![5b4deeb0122e1b068ddd37b5cdec8505e482035a6eb76c4f42a996f8cd96812c.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/images/5b4deeb0122e1b068ddd37b5cdec8505e482035a6eb76c4f42a996f8cd96812c.jpg)

![70c4f14acbabe24ef1be3d487905312397dde2838e9c998c053e7433b56331e6.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/images/70c4f14acbabe24ef1be3d487905312397dde2838e9c998c053e7433b56331e6.jpg)

![7634de1fdc9942c33b73d2e87e782df320540f9a0d5e3bed5892faf46a382bd8.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/images/7634de1fdc9942c33b73d2e87e782df320540f9a0d5e3bed5892faf46a382bd8.jpg)

![9053cb970bdeabd1049a4733bf49c4fba9d40711f0cee2422c2e0061623d17b4.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/images/9053cb970bdeabd1049a4733bf49c4fba9d40711f0cee2422c2e0061623d17b4.jpg)

![9624099431f19127d504f8ee185e38c72ae61f47d40045711a67e56893109d61.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/images/9624099431f19127d504f8ee185e38c72ae61f47d40045711a67e56893109d61.jpg)

![aa70d45e459e28b189e5e1d7cf8ebc88a5ab521740a746b4f70d8e0dcabcfabc.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/images/aa70d45e459e28b189e5e1d7cf8ebc88a5ab521740a746b4f70d8e0dcabcfabc.jpg)

![f69cb0064789c0cb83dd5c82213b21c483a66dc312491182c69b75cd95c2c8fa.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/images/f69cb0064789c0cb83dd5c82213b21c483a66dc312491182c69b75cd95c2c8fa.jpg)

### Tables

![0ee3c17a0184df599b880e0966ef9d67381c150d11266513ba16a5dd8f131a6f.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/tables/0ee3c17a0184df599b880e0966ef9d67381c150d11266513ba16a5dd8f131a6f.jpg)

![1bfb92e0b7c87ce7e3c11973315ebdaa4c27754d751e193f38225f2d196016c3.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/tables/1bfb92e0b7c87ce7e3c11973315ebdaa4c27754d751e193f38225f2d196016c3.jpg)

![1d981f8445febad6baf1ecb50645abe805f81d8b2b70e9b4709f6819df3b5a51.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/tables/1d981f8445febad6baf1ecb50645abe805f81d8b2b70e9b4709f6819df3b5a51.jpg)

![1f83dce12ca3a1c767fa67f613ae9375041d4a63c854b000e0704ef247a1630f.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/tables/1f83dce12ca3a1c767fa67f613ae9375041d4a63c854b000e0704ef247a1630f.jpg)

![32ed270e0254bcc130f5401c1712ba3612818a16bd72d43a24ac2f0d8b19281e.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/tables/32ed270e0254bcc130f5401c1712ba3612818a16bd72d43a24ac2f0d8b19281e.jpg)

![3bcb9d7e87e4921fc908cf0567b0e7972ac7518820f65a3ef7ce0c15b73a122b.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/tables/3bcb9d7e87e4921fc908cf0567b0e7972ac7518820f65a3ef7ce0c15b73a122b.jpg)

![56cc4eaddb0f22da15d73b2cec7b182518ef7b438bd06db42d20b75745826522.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/tables/56cc4eaddb0f22da15d73b2cec7b182518ef7b438bd06db42d20b75745826522.jpg)

![640da5c25f8e2cc74831ad62561e0a1a142290148038d9529088f6e1d33d2dae.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/tables/640da5c25f8e2cc74831ad62561e0a1a142290148038d9529088f6e1d33d2dae.jpg)

![6ad6b284783c437e955bcccad97e2ac6edaeb2fc1413e433e6d70076b05a3760.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/tables/6ad6b284783c437e955bcccad97e2ac6edaeb2fc1413e433e6d70076b05a3760.jpg)

![7238949d4cebec979deba6dd4dcc61a06f6b475cf574b1fdfbc07e6d5c76273b.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/tables/7238949d4cebec979deba6dd4dcc61a06f6b475cf574b1fdfbc07e6d5c76273b.jpg)

![7f489eabe3b2bc899d6109b189e08554ecca0b517dbf6d2413ecbd1bb68b1d73.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/tables/7f489eabe3b2bc899d6109b189e08554ecca0b517dbf6d2413ecbd1bb68b1d73.jpg)

![b6f9550e89377b9e845a67e6b9e2d781fa194ff2e24d94d9ecc50e15f38c3475.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/tables/b6f9550e89377b9e845a67e6b9e2d781fa194ff2e24d94d9ecc50e15f38c3475.jpg)

![cb0876971e09285469d0088f06b793fba7a9ce071415baa3d995d3e08c875125.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/tables/cb0876971e09285469d0088f06b793fba7a9ce071415baa3d995d3e08c875125.jpg)

![df3556e87f8a719c4702d7a480e0631cc5b837dd305ca2a5bda16bebe718f0b7.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/tables/df3556e87f8a719c4702d7a480e0631cc5b837dd305ca2a5bda16bebe718f0b7.jpg)

![f4e61b535da3dc09fce5652789f34095a00c602db39ca925b66cbbc24cdb34ce.jpg](../icml_results/6202_The%20Easy%2C%20the%20Hard%2C%20and%20the%20Learnable_%20Confidence%20and%20Difficulty-Adaptive%20Policy%20Optimization%20for%20LL/tables/f4e61b535da3dc09fce5652789f34095a00c602db39ca925b66cbbc24cdb34ce.jpg)

<a id="paper-6153"></a>

## SMD: Multi-view Safety-Critical Driving Video Generation in the Real-world Domain


### Images

![01fda24e26d16eef9b394eff70fae2b4dedff4e172678ca568449c4b9f715b64.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/images/01fda24e26d16eef9b394eff70fae2b4dedff4e172678ca568449c4b9f715b64.jpg)

![2b8688f7277188ab408661a5c891b7bc475bce288073c9557b05965016ab06a9.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/images/2b8688f7277188ab408661a5c891b7bc475bce288073c9557b05965016ab06a9.jpg)

![4a527ffd7c592cae94e87ca6240df980df3a8f1d55d6af91e12328cc0ba6cf80.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/images/4a527ffd7c592cae94e87ca6240df980df3a8f1d55d6af91e12328cc0ba6cf80.jpg)

![54cfe8f8dcf5f02ffe97a3af7343cea9a9bd4ade73ba64d234f4a42a4a091766.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/images/54cfe8f8dcf5f02ffe97a3af7343cea9a9bd4ade73ba64d234f4a42a4a091766.jpg)

![55ad1ae9d461dfcfb66e8ffaeb5d5a2bafb49afb0ade598f3b2d81b399f73987.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/images/55ad1ae9d461dfcfb66e8ffaeb5d5a2bafb49afb0ade598f3b2d81b399f73987.jpg)

![5ce7fb1216fb1090b5363bb8ca663bd4c57d9d245581f2d6822e355a7431be4d.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/images/5ce7fb1216fb1090b5363bb8ca663bd4c57d9d245581f2d6822e355a7431be4d.jpg)

![923716d1616ce9ffb8016693a88ac5ecfa974a81f0942aca02b764d1d1ec5388.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/images/923716d1616ce9ffb8016693a88ac5ecfa974a81f0942aca02b764d1d1ec5388.jpg)

![adff42ced72ad5ebfa7946ba8cc6167d40fb2343cee4401785a988dc068efda7.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/images/adff42ced72ad5ebfa7946ba8cc6167d40fb2343cee4401785a988dc068efda7.jpg)

![bf92ac330eda1e3038d73d6f07ec8021fca1ce70cae3f1fedac25dbd3996ff5b.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/images/bf92ac330eda1e3038d73d6f07ec8021fca1ce70cae3f1fedac25dbd3996ff5b.jpg)

### Tables

![1d8e9e676a9450cb00a500010036192e02e5f97c2320219e4a43f3f15d6657c1.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/1d8e9e676a9450cb00a500010036192e02e5f97c2320219e4a43f3f15d6657c1.jpg)

![21b6b9cdc5db443a0c8c62089054cbb19c1cee4b896654fab7fbb0a1bc644c8e.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/21b6b9cdc5db443a0c8c62089054cbb19c1cee4b896654fab7fbb0a1bc644c8e.jpg)

![26eb7da382324dd9a56047c2cf62d09dfe4c585c4b87312812c26df9b112b40b.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/26eb7da382324dd9a56047c2cf62d09dfe4c585c4b87312812c26df9b112b40b.jpg)

![2e7026f4e6537157dcf52804cb830db5ffb4bf5a92a10fb198e9b78764924961.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/2e7026f4e6537157dcf52804cb830db5ffb4bf5a92a10fb198e9b78764924961.jpg)

![2e7a06d410d4c7edb7a4303f1e8e3b7ac7191026442efdc72bca23a4bc12121f.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/2e7a06d410d4c7edb7a4303f1e8e3b7ac7191026442efdc72bca23a4bc12121f.jpg)

![3ca22e23297fbb69d48c6b125701f8c7f94361674455bdbc2cfee83c4a677a4d.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/3ca22e23297fbb69d48c6b125701f8c7f94361674455bdbc2cfee83c4a677a4d.jpg)

![42a421fce2967f0724f070ee2647bf06f313b7875c99f856ce85a12e8aa59cb5.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/42a421fce2967f0724f070ee2647bf06f313b7875c99f856ce85a12e8aa59cb5.jpg)

![4c9c665c8a5a939beb9f956caf5dcca0a4e59f276560e56cf4a017983ddc864e.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/4c9c665c8a5a939beb9f956caf5dcca0a4e59f276560e56cf4a017983ddc864e.jpg)

![595b4f9349029d73fdc8524057d45588f33d02bc8ad96332bd2a2b37ce3043ff.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/595b4f9349029d73fdc8524057d45588f33d02bc8ad96332bd2a2b37ce3043ff.jpg)

![9615a44a6a233e168ca28bfa49bfdf4095a4c639d6ad465df6523d9c0ce46842.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/9615a44a6a233e168ca28bfa49bfdf4095a4c639d6ad465df6523d9c0ce46842.jpg)

![b271ef330d9bc4e68946ad1953f899d7d874164794e28198caeb13a6a5e05c5d.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/b271ef330d9bc4e68946ad1953f899d7d874164794e28198caeb13a6a5e05c5d.jpg)

![c1bc32e50365707584229ec0f7c86f8ca062b90701ea5bc009f71aea72f2cb28.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/c1bc32e50365707584229ec0f7c86f8ca062b90701ea5bc009f71aea72f2cb28.jpg)

![c1de99c0fc19c705a3d1d0c4b6620b7f33997891a147ce3aff3ee7cf9dff7de5.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/c1de99c0fc19c705a3d1d0c4b6620b7f33997891a147ce3aff3ee7cf9dff7de5.jpg)

![cd3e16a283b53b809d603815db134e8cdd8293f21d9fd7346a7e6b7090f1deeb.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/cd3e16a283b53b809d603815db134e8cdd8293f21d9fd7346a7e6b7090f1deeb.jpg)

![cf979fb2f19dedfb9caae8f39d82d09f57accda314fdcba9517e5cf057f9687c.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/cf979fb2f19dedfb9caae8f39d82d09f57accda314fdcba9517e5cf057f9687c.jpg)

![d1d0803c49b154643bc0b9ddd42eed3d21a82cc234776f8be8571a64e281ea5a.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/d1d0803c49b154643bc0b9ddd42eed3d21a82cc234776f8be8571a64e281ea5a.jpg)

![e228999f0dc37e18903af91f5e934c75fad432df9f4f7299c657b2db25e2d32e.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/e228999f0dc37e18903af91f5e934c75fad432df9f4f7299c657b2db25e2d32e.jpg)

![e4c427b5d272d619bc1fd09fa6bab396f81aa5f643ef35365f3e7282db66c568.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/e4c427b5d272d619bc1fd09fa6bab396f81aa5f643ef35365f3e7282db66c568.jpg)

![e8ba0950faa79976b2933ebe2b2e7c5c8b9778a1aaf45f55c4e4531f8e5aa8d0.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/e8ba0950faa79976b2933ebe2b2e7c5c8b9778a1aaf45f55c4e4531f8e5aa8d0.jpg)

![ecc7f7af24876f17534f338c25235d00791ee950329ecde84392ec91d7a0d6a2.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/ecc7f7af24876f17534f338c25235d00791ee950329ecde84392ec91d7a0d6a2.jpg)

![fbfc5e04ee49d93ac5657d12137caa50a2a66cc572b3d77727330d2f1205a8a4.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/fbfc5e04ee49d93ac5657d12137caa50a2a66cc572b3d77727330d2f1205a8a4.jpg)

![fc17b9a19aeaf8cb1a465ede0bd7bcfa1bd99fd8082ddf7ee4463d2ef085a1b4.jpg](../icml_results/6203_SMD_%20Multi-view%20Safety-Critical%20Driving%20Video%20Generation%20in%20the%20Real-world%20Domain/tables/fc17b9a19aeaf8cb1a465ede0bd7bcfa1bd99fd8082ddf7ee4463d2ef085a1b4.jpg)

<a id="paper-6154"></a>

## DREAM: A Unified Framework for Drift-Corrected Federated Multi-Objective Learning


### Images

![14407cbd724694f183283fe472c89c960a7e181b5176daf9e825bbedcda550e8.jpg](../icml_results/6204_DREAM_%20A%20Unified%20Framework%20for%20Drift-Corrected%20Federated%20Multi-Objective%20Learning/images/14407cbd724694f183283fe472c89c960a7e181b5176daf9e825bbedcda550e8.jpg)

![15740ea81c8c5b6515517c472036c1c7b050d7020ff05054350cc98f6cfac873.jpg](../icml_results/6204_DREAM_%20A%20Unified%20Framework%20for%20Drift-Corrected%20Federated%20Multi-Objective%20Learning/images/15740ea81c8c5b6515517c472036c1c7b050d7020ff05054350cc98f6cfac873.jpg)

![20ac27f0492b5ba0494f469d8ce95cc61abf297c81c7f9ecfde6d43f646720dd.jpg](../icml_results/6204_DREAM_%20A%20Unified%20Framework%20for%20Drift-Corrected%20Federated%20Multi-Objective%20Learning/images/20ac27f0492b5ba0494f469d8ce95cc61abf297c81c7f9ecfde6d43f646720dd.jpg)

![4bce35a0e3fd7ddaca47ff868e8c28c342f9861b34f184c54e8afa4d19d93645.jpg](../icml_results/6204_DREAM_%20A%20Unified%20Framework%20for%20Drift-Corrected%20Federated%20Multi-Objective%20Learning/images/4bce35a0e3fd7ddaca47ff868e8c28c342f9861b34f184c54e8afa4d19d93645.jpg)

![56716d260b1d9f97465bfe1a729d097c92a49963e2c19115ffcdd6fbe3484b58.jpg](../icml_results/6204_DREAM_%20A%20Unified%20Framework%20for%20Drift-Corrected%20Federated%20Multi-Objective%20Learning/images/56716d260b1d9f97465bfe1a729d097c92a49963e2c19115ffcdd6fbe3484b58.jpg)

![7167fdab559200cee8c1a5e7e513f4840c6c3f8b50dfa66b6949fabdc90513d6.jpg](../icml_results/6204_DREAM_%20A%20Unified%20Framework%20for%20Drift-Corrected%20Federated%20Multi-Objective%20Learning/images/7167fdab559200cee8c1a5e7e513f4840c6c3f8b50dfa66b6949fabdc90513d6.jpg)

![aa9596340050cfc824f281af30535552c92c5094c83701c1db3f37c133792c3e.jpg](../icml_results/6204_DREAM_%20A%20Unified%20Framework%20for%20Drift-Corrected%20Federated%20Multi-Objective%20Learning/images/aa9596340050cfc824f281af30535552c92c5094c83701c1db3f37c133792c3e.jpg)

![d33c8163304cbecd576a2b97e52e21e50e40e3fef11ae003fd7481c6d73cc8b9.jpg](../icml_results/6204_DREAM_%20A%20Unified%20Framework%20for%20Drift-Corrected%20Federated%20Multi-Objective%20Learning/images/d33c8163304cbecd576a2b97e52e21e50e40e3fef11ae003fd7481c6d73cc8b9.jpg)

![d577a8b7a75fdd515708017aa19d15b8e945fed4983dbde6f54392180ecbcaf4.jpg](../icml_results/6204_DREAM_%20A%20Unified%20Framework%20for%20Drift-Corrected%20Federated%20Multi-Objective%20Learning/images/d577a8b7a75fdd515708017aa19d15b8e945fed4983dbde6f54392180ecbcaf4.jpg)

![d6994f004bd573b4109a5911b9acc725aa99f8292cbcdbc1534139607ffd5017.jpg](../icml_results/6204_DREAM_%20A%20Unified%20Framework%20for%20Drift-Corrected%20Federated%20Multi-Objective%20Learning/images/d6994f004bd573b4109a5911b9acc725aa99f8292cbcdbc1534139607ffd5017.jpg)

![e2d22c985b920811d03d757ec7e4e6fc7d9becafd9b0ddea45e5376034469673.jpg](../icml_results/6204_DREAM_%20A%20Unified%20Framework%20for%20Drift-Corrected%20Federated%20Multi-Objective%20Learning/images/e2d22c985b920811d03d757ec7e4e6fc7d9becafd9b0ddea45e5376034469673.jpg)

![e3b6c00db9312ae06014aaf093e85bfca7fd7f4371615d2e2fa4ecbe5db68f7f.jpg](../icml_results/6204_DREAM_%20A%20Unified%20Framework%20for%20Drift-Corrected%20Federated%20Multi-Objective%20Learning/images/e3b6c00db9312ae06014aaf093e85bfca7fd7f4371615d2e2fa4ecbe5db68f7f.jpg)

### Tables

![7809a914443975fdb6047d43ffda6922e5d7a3ee8c9f6422e57ea0ff4395283b.jpg](../icml_results/6204_DREAM_%20A%20Unified%20Framework%20for%20Drift-Corrected%20Federated%20Multi-Objective%20Learning/tables/7809a914443975fdb6047d43ffda6922e5d7a3ee8c9f6422e57ea0ff4395283b.jpg)

<a id="paper-6155"></a>

## WatchLog: Efficient and Interpretable Event Reasoning for Endpoint Detection and Response Logs with Multimodal LLMs


### Images

![1f8defab25137692a515b8a26ec0208ebc751d95efaa1f5f4a2072368061ba2b.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/images/1f8defab25137692a515b8a26ec0208ebc751d95efaa1f5f4a2072368061ba2b.jpg)

![af016540a4f0937122a0b76e1da9419335971015be5c62bc46a2a61208f6da36.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/images/af016540a4f0937122a0b76e1da9419335971015be5c62bc46a2a61208f6da36.jpg)

![b3190f1a280d7e5bcaa4eb5b905699409a0ad83101afcad7f07cffe7cd487802.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/images/b3190f1a280d7e5bcaa4eb5b905699409a0ad83101afcad7f07cffe7cd487802.jpg)

### Tables

![306eb1ac491ef698d583faee0944d6e2b0c0656a61b07cacc7eae1410867a8ac.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/tables/306eb1ac491ef698d583faee0944d6e2b0c0656a61b07cacc7eae1410867a8ac.jpg)

![5263f664858464724ba4076e49010750e42e906c01416b78cd0370d44cc9a63d.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/tables/5263f664858464724ba4076e49010750e42e906c01416b78cd0370d44cc9a63d.jpg)

![627bf6913a4f25468b1c7f606e9c77d4b14c511500a42896c33bbc0d61df88c4.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/tables/627bf6913a4f25468b1c7f606e9c77d4b14c511500a42896c33bbc0d61df88c4.jpg)

![67206be9dd24903f051d837b8c075bcc2c4f2c5ddd249900238288009d8c67a9.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/tables/67206be9dd24903f051d837b8c075bcc2c4f2c5ddd249900238288009d8c67a9.jpg)

![68bb644b611cd9cb3251b246c94897c14d288c3423a9be8002427673d223435c.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/tables/68bb644b611cd9cb3251b246c94897c14d288c3423a9be8002427673d223435c.jpg)

![7d1526769b8a1426d2f97b9ba48b1eb1901b6c06041e559e75b9e0f40ce54bed.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/tables/7d1526769b8a1426d2f97b9ba48b1eb1901b6c06041e559e75b9e0f40ce54bed.jpg)

![80a4ad4e7cceab8e5552af792d30f3a8f180f90be3cbe7deb0424fcff52c693d.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/tables/80a4ad4e7cceab8e5552af792d30f3a8f180f90be3cbe7deb0424fcff52c693d.jpg)

![811d634f9f22673f80ed077c21b528a8366ac71ed8432b37f8aa21f9506b4a23.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/tables/811d634f9f22673f80ed077c21b528a8366ac71ed8432b37f8aa21f9506b4a23.jpg)

![86bc1102c800e6966dc9883f77864153cb54d6855e897999c4702179d0b13f7b.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/tables/86bc1102c800e6966dc9883f77864153cb54d6855e897999c4702179d0b13f7b.jpg)

![b4e8fb50c69de113ff089d9221d1b2176eb060ab3629efbcb79a19dd95708db7.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/tables/b4e8fb50c69de113ff089d9221d1b2176eb060ab3629efbcb79a19dd95708db7.jpg)

![d382aca59283dd575c772c90e83050daa36b34b91540301f814bc19eb754635f.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/tables/d382aca59283dd575c772c90e83050daa36b34b91540301f814bc19eb754635f.jpg)

![d6147ed07bf7408d2418e5c2f50e35d526bdfface96dbdc136a60def74eb13fb.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/tables/d6147ed07bf7408d2418e5c2f50e35d526bdfface96dbdc136a60def74eb13fb.jpg)

![ecaa1e4e3c63482c760c9ddcc14e612a9096896e4b4fb89664dd790ac9975e5b.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/tables/ecaa1e4e3c63482c760c9ddcc14e612a9096896e4b4fb89664dd790ac9975e5b.jpg)

![f7a6eb8531c097e26e541cfd4456bae05108511428306ee7924f4397b20afdab.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/tables/f7a6eb8531c097e26e541cfd4456bae05108511428306ee7924f4397b20afdab.jpg)

![f850d770855b4bc409140c602fba37cd305294b28178393cd9a362fe0df8b086.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/tables/f850d770855b4bc409140c602fba37cd305294b28178393cd9a362fe0df8b086.jpg)

![f86e26749fc6b715024fd3ed827d718871f2f4d590f5ad2a742754de804cb6f8.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/tables/f86e26749fc6b715024fd3ed827d718871f2f4d590f5ad2a742754de804cb6f8.jpg)

![fafb516fd1bd770092f47e7989ee47d2d8b48aebd482705339e4768b90013b1f.jpg](../icml_results/6205_WatchLog_%20Efficient%20and%20Interpretable%20Event%20Reasoning%20for%20Endpoint%20Detection%20and%20Response%20Logs%20with%20/tables/fafb516fd1bd770092f47e7989ee47d2d8b48aebd482705339e4768b90013b1f.jpg)

<a id="paper-6156"></a>

## Are Two Datasets Close Enough With Statistical Significance? A Kernel Distributional Closeness Testing Approach


### Images

![0c9ec8212fec4f553f7f6d2eaf74514e66d0e6e0047de66875caef13c5e5f8b2.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/images/0c9ec8212fec4f553f7f6d2eaf74514e66d0e6e0047de66875caef13c5e5f8b2.jpg)

![2b012ae4b46fd6b42dd5295b2ed8f81b72e057d3be93c63e278e0f8a9de78db0.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/images/2b012ae4b46fd6b42dd5295b2ed8f81b72e057d3be93c63e278e0f8a9de78db0.jpg)

![8a56b75b5bd52f7c22597fb929a30eafab89300dbf433b6c36f629ffe2f34bb6.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/images/8a56b75b5bd52f7c22597fb929a30eafab89300dbf433b6c36f629ffe2f34bb6.jpg)

![9ee333f8eb4201c461d08a23859eae7f6a09d87c314d785f038633efe134e406.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/images/9ee333f8eb4201c461d08a23859eae7f6a09d87c314d785f038633efe134e406.jpg)

![ca7f7a7bb5977a99c531977d8b1b89a21a13958de5d7952c5354ba82b381ea4b.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/images/ca7f7a7bb5977a99c531977d8b1b89a21a13958de5d7952c5354ba82b381ea4b.jpg)

![cbd600e42161eb07384a9b1153821e248a7d3d9dbd03778dc87f7000ad26ea35.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/images/cbd600e42161eb07384a9b1153821e248a7d3d9dbd03778dc87f7000ad26ea35.jpg)

![e8af52fd3db81c0c768fb170717704e29ae119d5bc7961895a59a0765d1467dd.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/images/e8af52fd3db81c0c768fb170717704e29ae119d5bc7961895a59a0765d1467dd.jpg)

### Tables

![05ec079a594c60b67337de4d9a647139a92fdf138974d5872da24b51a30f31d8.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/tables/05ec079a594c60b67337de4d9a647139a92fdf138974d5872da24b51a30f31d8.jpg)

![2088225845ae1c79a241bb99a8b2a06270125f58938b6c1f51724e14b7c34e36.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/tables/2088225845ae1c79a241bb99a8b2a06270125f58938b6c1f51724e14b7c34e36.jpg)

![25d07c702e71ed636b83cc72e25f113d004cff3c74bcc011122eb404df916f54.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/tables/25d07c702e71ed636b83cc72e25f113d004cff3c74bcc011122eb404df916f54.jpg)

![27ebbabac5facd685d2695f5904cf304d48d1a4521b81a64365faaff26790fd6.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/tables/27ebbabac5facd685d2695f5904cf304d48d1a4521b81a64365faaff26790fd6.jpg)

![2f3cb47095ce596bbd82eb9fd5e93431cba61965431aca0bb181949e04463fff.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/tables/2f3cb47095ce596bbd82eb9fd5e93431cba61965431aca0bb181949e04463fff.jpg)

![3180e7c1154e43ccbcbff2c5005079b28ee5a57353468479de4b3bbaeb6f36cb.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/tables/3180e7c1154e43ccbcbff2c5005079b28ee5a57353468479de4b3bbaeb6f36cb.jpg)

![4643b1a671ce141ce03ae60c6fb9fc0e934bbc82972d8cd93ef236275466ea3b.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/tables/4643b1a671ce141ce03ae60c6fb9fc0e934bbc82972d8cd93ef236275466ea3b.jpg)

![54af5c11640d4054bdf2789f8d2b57baad144db72612cab1a8c104e733f07e3b.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/tables/54af5c11640d4054bdf2789f8d2b57baad144db72612cab1a8c104e733f07e3b.jpg)

![62b8ead65564d31951db0a65d712f3175f12ffea2b9d1abdc2108a80d1077b64.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/tables/62b8ead65564d31951db0a65d712f3175f12ffea2b9d1abdc2108a80d1077b64.jpg)

![7506193147e74ef49f9de34bd594c074719cd61fa89ef1fd9961a318c84a015b.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/tables/7506193147e74ef49f9de34bd594c074719cd61fa89ef1fd9961a318c84a015b.jpg)

![775db7f992d053bdd27046b415c4932aa6d49489e5c8693b3206a0e3cf01c663.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/tables/775db7f992d053bdd27046b415c4932aa6d49489e5c8693b3206a0e3cf01c663.jpg)

![7baaef8471134a7d8639af2ddce039c058f874ddc3da8680de5747be1195c01e.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/tables/7baaef8471134a7d8639af2ddce039c058f874ddc3da8680de5747be1195c01e.jpg)

![a7c9c0eb400a360690c0a8f05c573940f75b7216df8795287f1d1a3f627dea8c.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/tables/a7c9c0eb400a360690c0a8f05c573940f75b7216df8795287f1d1a3f627dea8c.jpg)

![c9ed13fc1868adbc2a21d20f4135ff48fa9a61c871a28322415681ae792905ba.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/tables/c9ed13fc1868adbc2a21d20f4135ff48fa9a61c871a28322415681ae792905ba.jpg)

![dd3559a62e4347b93f7f9a38158e20254e7045e36128e5ea5ea60585c080dc05.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/tables/dd3559a62e4347b93f7f9a38158e20254e7045e36128e5ea5ea60585c080dc05.jpg)

![e10f43f799438a200cfe6af0ae4393c09b534b1ae69f93369a672a5ffadcf4a4.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/tables/e10f43f799438a200cfe6af0ae4393c09b534b1ae69f93369a672a5ffadcf4a4.jpg)

![ea37be9106c73cc8ffd5d1efe6746981aef727a70d369bee8e6a2e1adb72e5e9.jpg](../icml_results/6206_Are%20Two%20Datasets%20Close%20Enough%20With%20Statistical%20Significance_%20A%20Kernel%20Distributional%20Closeness%20Testi/tables/ea37be9106c73cc8ffd5d1efe6746981aef727a70d369bee8e6a2e1adb72e5e9.jpg)

<a id="paper-6157"></a>

## HTAC: Hierarchical Task-Aware Composition for Continual Offline Reinforcement Learning


### Images

![43d8e1213f987d5816a162f8e35e56e656ea8380fb7537c5ab75e1c5b51d0ab1.jpg](../icml_results/6207_HTAC_%20Hierarchical%20Task-Aware%20Composition%20for%20Continual%20Offline%20Reinforcement%20Learning/images/43d8e1213f987d5816a162f8e35e56e656ea8380fb7537c5ab75e1c5b51d0ab1.jpg)

![6fe5f64120fe7baeeef1cfaba5ea7824f467532f5da02bef441ed623e0a64a98.jpg](../icml_results/6207_HTAC_%20Hierarchical%20Task-Aware%20Composition%20for%20Continual%20Offline%20Reinforcement%20Learning/images/6fe5f64120fe7baeeef1cfaba5ea7824f467532f5da02bef441ed623e0a64a98.jpg)

![95233f0b2404fae94a63dfb0632156926c0710594b43c080bfb66b0b0d7beb77.jpg](../icml_results/6207_HTAC_%20Hierarchical%20Task-Aware%20Composition%20for%20Continual%20Offline%20Reinforcement%20Learning/images/95233f0b2404fae94a63dfb0632156926c0710594b43c080bfb66b0b0d7beb77.jpg)

![9c60e41e0990be7954ccc91e3683df8280ca9f1e87b8fac83e635d149809bead.jpg](../icml_results/6207_HTAC_%20Hierarchical%20Task-Aware%20Composition%20for%20Continual%20Offline%20Reinforcement%20Learning/images/9c60e41e0990be7954ccc91e3683df8280ca9f1e87b8fac83e635d149809bead.jpg)

![9e01c61eff412621179b6ca7da07bb6ad912f4b0e6ed72a4cd3d5803ddfff8c6.jpg](../icml_results/6207_HTAC_%20Hierarchical%20Task-Aware%20Composition%20for%20Continual%20Offline%20Reinforcement%20Learning/images/9e01c61eff412621179b6ca7da07bb6ad912f4b0e6ed72a4cd3d5803ddfff8c6.jpg)

![e1051997774ad444218d80f2a1003098ad85547512e3d7b2dd57ef1b48b4a5ae.jpg](../icml_results/6207_HTAC_%20Hierarchical%20Task-Aware%20Composition%20for%20Continual%20Offline%20Reinforcement%20Learning/images/e1051997774ad444218d80f2a1003098ad85547512e3d7b2dd57ef1b48b4a5ae.jpg)

### Tables

![41d2c621dfa9b40f739d937c9aa42c9cd5de9adef603f8f0511beb50e586be08.jpg](../icml_results/6207_HTAC_%20Hierarchical%20Task-Aware%20Composition%20for%20Continual%20Offline%20Reinforcement%20Learning/tables/41d2c621dfa9b40f739d937c9aa42c9cd5de9adef603f8f0511beb50e586be08.jpg)

![4ee69f51c29ddeb769045a8ce4a974d19d4d6b303e112a30ec40e66f78898856.jpg](../icml_results/6207_HTAC_%20Hierarchical%20Task-Aware%20Composition%20for%20Continual%20Offline%20Reinforcement%20Learning/tables/4ee69f51c29ddeb769045a8ce4a974d19d4d6b303e112a30ec40e66f78898856.jpg)

![5bdcf150d410e909c66ceabdbc475cb96604f7884153e624a784028bd6846cdb.jpg](../icml_results/6207_HTAC_%20Hierarchical%20Task-Aware%20Composition%20for%20Continual%20Offline%20Reinforcement%20Learning/tables/5bdcf150d410e909c66ceabdbc475cb96604f7884153e624a784028bd6846cdb.jpg)

![5ecc357805353fd9af36fa81efc3f43eed85c529f4fb8e87de47ccec25dcad1c.jpg](../icml_results/6207_HTAC_%20Hierarchical%20Task-Aware%20Composition%20for%20Continual%20Offline%20Reinforcement%20Learning/tables/5ecc357805353fd9af36fa81efc3f43eed85c529f4fb8e87de47ccec25dcad1c.jpg)

![8c2e801602d94b3e909bd3d4ea3f81bfc1a90acbd954f605930a2d88f34f307e.jpg](../icml_results/6207_HTAC_%20Hierarchical%20Task-Aware%20Composition%20for%20Continual%20Offline%20Reinforcement%20Learning/tables/8c2e801602d94b3e909bd3d4ea3f81bfc1a90acbd954f605930a2d88f34f307e.jpg)

![9e1116ea4cc12dd0888d77f2e0e23d7017030931167bae61617fc5192cb3da9b.jpg](../icml_results/6207_HTAC_%20Hierarchical%20Task-Aware%20Composition%20for%20Continual%20Offline%20Reinforcement%20Learning/tables/9e1116ea4cc12dd0888d77f2e0e23d7017030931167bae61617fc5192cb3da9b.jpg)

![be0a68f8dc9ee025403255f0879acbeb2570bb71dc03642899cd1a7832c66eee.jpg](../icml_results/6207_HTAC_%20Hierarchical%20Task-Aware%20Composition%20for%20Continual%20Offline%20Reinforcement%20Learning/tables/be0a68f8dc9ee025403255f0879acbeb2570bb71dc03642899cd1a7832c66eee.jpg)

![df5f58cd446ffdaf57fbf94f7abfa2c21fd8ddb2d9eaa8885f07f9c042e7d338.jpg](../icml_results/6207_HTAC_%20Hierarchical%20Task-Aware%20Composition%20for%20Continual%20Offline%20Reinforcement%20Learning/tables/df5f58cd446ffdaf57fbf94f7abfa2c21fd8ddb2d9eaa8885f07f9c042e7d338.jpg)

![f331c4d1b36b0feb39d8d727b2885c4bcfa7d86ece6aeddb75cad54b7f5ff216.jpg](../icml_results/6207_HTAC_%20Hierarchical%20Task-Aware%20Composition%20for%20Continual%20Offline%20Reinforcement%20Learning/tables/f331c4d1b36b0feb39d8d727b2885c4bcfa7d86ece6aeddb75cad54b7f5ff216.jpg)

![f65b58c23ade7eaba780437b163801f5e5087469370c973ab97a05f0ed0d266e.jpg](../icml_results/6207_HTAC_%20Hierarchical%20Task-Aware%20Composition%20for%20Continual%20Offline%20Reinforcement%20Learning/tables/f65b58c23ade7eaba780437b163801f5e5087469370c973ab97a05f0ed0d266e.jpg)

<a id="paper-6158"></a>

## FormalJudge: A Neuro-Symbolic Paradigm for Agentic Oversight


### Images

![3f0249f808f4da62a56c6bd7417a9a186a151a6b3e51d35483e4e17b3f424433.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/images/3f0249f808f4da62a56c6bd7417a9a186a151a6b3e51d35483e4e17b3f424433.jpg)

![5d432c4db1b29a29b7c9d5c44fb78da826a3fe6741c437181577b822a12e9e62.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/images/5d432c4db1b29a29b7c9d5c44fb78da826a3fe6741c437181577b822a12e9e62.jpg)

![6e83a15bb853c403f22e9f1f8bd99ad215663dca4dc633dda97291e9498f6fbb.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/images/6e83a15bb853c403f22e9f1f8bd99ad215663dca4dc633dda97291e9498f6fbb.jpg)

![7330d09b1b020791b03a663ae7444f4dd886bf7a823530d59cf7668bcf6b2968.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/images/7330d09b1b020791b03a663ae7444f4dd886bf7a823530d59cf7668bcf6b2968.jpg)

![83d81881806a8e39da56638644424cbe97654f3be9619c87e0444406588b92b0.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/images/83d81881806a8e39da56638644424cbe97654f3be9619c87e0444406588b92b0.jpg)

![8526f9f816bf215dca59925fd87496fd7438d22fbd201ee670920f89b7982c9d.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/images/8526f9f816bf215dca59925fd87496fd7438d22fbd201ee670920f89b7982c9d.jpg)

![b3bd6bf944d81ead5eac38c8f12510afe4d8cba2843d62763f42d6c59d4dfea3.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/images/b3bd6bf944d81ead5eac38c8f12510afe4d8cba2843d62763f42d6c59d4dfea3.jpg)

![d222d07a74a85f2a0442634e3c435046f151de4f25c1388f79f3abffe63b1af9.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/images/d222d07a74a85f2a0442634e3c435046f151de4f25c1388f79f3abffe63b1af9.jpg)

![e121c1c86b652df2a2d89394fa262c87984c3515c2885fb05c6672924d1358ca.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/images/e121c1c86b652df2a2d89394fa262c87984c3515c2885fb05c6672924d1358ca.jpg)

### Tables

![0528a227046bc0f7ccafeebd6ada3643e2bbe3cbbb71ff8022f609d119307a60.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/0528a227046bc0f7ccafeebd6ada3643e2bbe3cbbb71ff8022f609d119307a60.jpg)

![0e25e674a8c8926dcb04d8c5b671e4477cffd70da488e52e515f6124b204bae1.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/0e25e674a8c8926dcb04d8c5b671e4477cffd70da488e52e515f6124b204bae1.jpg)

![10c0e56c70b64b10551b4c4704d1576825052615f386ca873dd397c186ba19cb.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/10c0e56c70b64b10551b4c4704d1576825052615f386ca873dd397c186ba19cb.jpg)

![1ad8b411d93dc6b58cf037389b8ec8132e077c7864a923c4db9538b5123400da.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/1ad8b411d93dc6b58cf037389b8ec8132e077c7864a923c4db9538b5123400da.jpg)

![2bb1a5191a803cda6097115fbdb36d51b83542fc49e84e99b916458e79c46981.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/2bb1a5191a803cda6097115fbdb36d51b83542fc49e84e99b916458e79c46981.jpg)

![354cdb047944cefabe50bd768b34d5f7bf4e9d78dbc557af02619ebf2a79509a.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/354cdb047944cefabe50bd768b34d5f7bf4e9d78dbc557af02619ebf2a79509a.jpg)

![36636945cb5bbedcd0b7ffea1409830074eaaf0b73e3ac24261f2a4f566d5494.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/36636945cb5bbedcd0b7ffea1409830074eaaf0b73e3ac24261f2a4f566d5494.jpg)

![497823ca634939d06760f25e91f2d5c9919080c42cfed8cce99a940648abcebb.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/497823ca634939d06760f25e91f2d5c9919080c42cfed8cce99a940648abcebb.jpg)

![4c9deb8b8cf2815de3015d0dd9fc4749754aa2b301871fac2aba5bd5f9113653.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/4c9deb8b8cf2815de3015d0dd9fc4749754aa2b301871fac2aba5bd5f9113653.jpg)

![50f7237b7e39d2d14e7b23325443ea7516627306db3a1ad75bb12b35a12471c7.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/50f7237b7e39d2d14e7b23325443ea7516627306db3a1ad75bb12b35a12471c7.jpg)

![5c1a5a4a13894104266ced6fb585ca3ad380eeb430871a874b937d74749bf963.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/5c1a5a4a13894104266ced6fb585ca3ad380eeb430871a874b937d74749bf963.jpg)

![7e3f726d4e03b9b3109ea1b948350231e0ff509125545e523c2c6e39e5c56686.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/7e3f726d4e03b9b3109ea1b948350231e0ff509125545e523c2c6e39e5c56686.jpg)

![8b30caf967cb93fbf852eae8355d21c5d45b4cd4f80f5f366b8f182ad3dd8524.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/8b30caf967cb93fbf852eae8355d21c5d45b4cd4f80f5f366b8f182ad3dd8524.jpg)

![8b4704a7cad1e69a09803c9ecc14e88c621554d8e732e5f4908f36392b94952a.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/8b4704a7cad1e69a09803c9ecc14e88c621554d8e732e5f4908f36392b94952a.jpg)

![9089c111bdee77e768c92b2e8b3a8cfa4a0926dd342a2b663f0a37163cb092b2.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/9089c111bdee77e768c92b2e8b3a8cfa4a0926dd342a2b663f0a37163cb092b2.jpg)

![912e257ff6c4a78e61fc756e7d181fd484dfea8f65856332327938fac40ad8e0.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/912e257ff6c4a78e61fc756e7d181fd484dfea8f65856332327938fac40ad8e0.jpg)

![9b8840636e4a07b9961155fb5c418f772adbd759617f8c7762ffde347464333a.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/9b8840636e4a07b9961155fb5c418f772adbd759617f8c7762ffde347464333a.jpg)

![9ba903a85a3640338dc7bf015d68c97a43a20ea6b69312609ff0dd94811da2cc.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/9ba903a85a3640338dc7bf015d68c97a43a20ea6b69312609ff0dd94811da2cc.jpg)

![9bafb1b346c4cd16fe9cc5dbf168f15390dbafa5750344d58868062a8803ce08.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/9bafb1b346c4cd16fe9cc5dbf168f15390dbafa5750344d58868062a8803ce08.jpg)

![a144c5f21398ba6cf280681450ec7cfb5cfd3e31243fccbff08be771ebcf13f9.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/a144c5f21398ba6cf280681450ec7cfb5cfd3e31243fccbff08be771ebcf13f9.jpg)

![b1627a6fce89455e9d25c28f1ccef516b297c5ecf42ecc4f40e7507fe74a5093.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/b1627a6fce89455e9d25c28f1ccef516b297c5ecf42ecc4f40e7507fe74a5093.jpg)

![b5f1939eb180f5d6ac60b514c9404dccdee6378329ced9d012b35d09473284df.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/b5f1939eb180f5d6ac60b514c9404dccdee6378329ced9d012b35d09473284df.jpg)

![bd58b8c8ff656bbefb8583c497fdfa11a30b47be531f70bbf82bb78e5785747f.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/bd58b8c8ff656bbefb8583c497fdfa11a30b47be531f70bbf82bb78e5785747f.jpg)

![c3a04cc02cce93acb18f169a32c10e27478d30dae811a091f6028d7d44c8f717.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/c3a04cc02cce93acb18f169a32c10e27478d30dae811a091f6028d7d44c8f717.jpg)

![cabf73792c5afcdc7322bdbb5c173a5015a06ed393d88a8cadb63aaec4eadaff.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/cabf73792c5afcdc7322bdbb5c173a5015a06ed393d88a8cadb63aaec4eadaff.jpg)

![d9969c5bf25a127b65077ecbb22c5af673668f53df883b48156e8a20572ff8b8.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/d9969c5bf25a127b65077ecbb22c5af673668f53df883b48156e8a20572ff8b8.jpg)

![db7ba5da0d88c104eebe99f99dec380d0317ea82b79c5f5ed14e7b8e77fc1822.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/db7ba5da0d88c104eebe99f99dec380d0317ea82b79c5f5ed14e7b8e77fc1822.jpg)

![f25e174acde95fd20d08935a7a460d70b8a9c648c4beef973c58d51906e371a9.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/f25e174acde95fd20d08935a7a460d70b8a9c648c4beef973c58d51906e371a9.jpg)

![f27c23cdcd93c0f4863e6ff742d9ca4efbc459f0f58fb1a8356fec97bed79812.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/f27c23cdcd93c0f4863e6ff742d9ca4efbc459f0f58fb1a8356fec97bed79812.jpg)

![f305cf0efdfd839b272a224cb4a3fbc0b0cd61972222a2dcdc60d9c5753eb574.jpg](../icml_results/6208_FormalJudge_%20A%20Neuro-Symbolic%20Paradigm%20for%20Agentic%20Oversight/tables/f305cf0efdfd839b272a224cb4a3fbc0b0cd61972222a2dcdc60d9c5753eb574.jpg)

<a id="paper-6159"></a>

## Trajectory Seriation via Spectral Tangent Alignment and Global Embedding


### Images

![00744bcf5bc7be97d9090c5fd0a2d6fe0a563b3d1ddb8815210d28ad36a98da8.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/00744bcf5bc7be97d9090c5fd0a2d6fe0a563b3d1ddb8815210d28ad36a98da8.jpg)

![0ccbc45bfacf9bfccbc7f09c6dd17f6e502d8012cca14326bf3833e7e478e7c8.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/0ccbc45bfacf9bfccbc7f09c6dd17f6e502d8012cca14326bf3833e7e478e7c8.jpg)

![2a9eebe1c2bfc545884d3d9697a6d3de7bc0722402b6f657fa1d578bf2babf0c.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/2a9eebe1c2bfc545884d3d9697a6d3de7bc0722402b6f657fa1d578bf2babf0c.jpg)

![342bed813515b61c9820e7bccd1a8a4bf802db147e2b5c0222e404ea03be0a00.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/342bed813515b61c9820e7bccd1a8a4bf802db147e2b5c0222e404ea03be0a00.jpg)

![505794c287a26a3b4e4215f6b72275e6941fecc30c686e4d9d3dc286703d6c24.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/505794c287a26a3b4e4215f6b72275e6941fecc30c686e4d9d3dc286703d6c24.jpg)

![5d80042ad2fd46da170ffbb5eea6353c3c7f9454e2d013bc14f0c1bf7155750f.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/5d80042ad2fd46da170ffbb5eea6353c3c7f9454e2d013bc14f0c1bf7155750f.jpg)

![5e2bfb19673bf750efc8fb20b3fcbd9bd6c6ae5bafc89aba4dbc335e67acbd9d.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/5e2bfb19673bf750efc8fb20b3fcbd9bd6c6ae5bafc89aba4dbc335e67acbd9d.jpg)

![68a306cf448bacadc26df13d5f973ea8214dce78752918653513e4b79cc4de0b.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/68a306cf448bacadc26df13d5f973ea8214dce78752918653513e4b79cc4de0b.jpg)

![891973a038e9f5be5eb60381a5c9761ecfa6f35083a25844c75fdf6624d93867.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/891973a038e9f5be5eb60381a5c9761ecfa6f35083a25844c75fdf6624d93867.jpg)

![95201c9e9c3292cf7c310a9f547f9dac73cbe35c27f998f41751a1ba469374d6.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/95201c9e9c3292cf7c310a9f547f9dac73cbe35c27f998f41751a1ba469374d6.jpg)

![b534a7ed599176452176da3c4623b972c0c69583f977fee7a7333999503506f6.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/b534a7ed599176452176da3c4623b972c0c69583f977fee7a7333999503506f6.jpg)

![c1d75701c4c334d38092dc099f56b750fae512b08d387c2404b9713310c9160d.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/c1d75701c4c334d38092dc099f56b750fae512b08d387c2404b9713310c9160d.jpg)

![da94f9199802828c49e822ef6c252f636e44946a3aef48b028beab7d7b686340.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/da94f9199802828c49e822ef6c252f636e44946a3aef48b028beab7d7b686340.jpg)

![df59e8d0795fdbd473dd0be680f01d4a52984e6ddb309706eeaf87fc75235cdc.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/df59e8d0795fdbd473dd0be680f01d4a52984e6ddb309706eeaf87fc75235cdc.jpg)

![e0c37ecb0fd085b1ebb8b8e0d997ff21222bea23130eb0470bc0e5bbbffacbef.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/e0c37ecb0fd085b1ebb8b8e0d997ff21222bea23130eb0470bc0e5bbbffacbef.jpg)

![e152d31f10ac6b89df13be27c5a3fd90d5b8f8314f3d334335c5b60420e2dfe1.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/e152d31f10ac6b89df13be27c5a3fd90d5b8f8314f3d334335c5b60420e2dfe1.jpg)

![e45224ea853a3cb9fca4db03b64a5d533fad2ef83257db412955e29112f20e98.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/e45224ea853a3cb9fca4db03b64a5d533fad2ef83257db412955e29112f20e98.jpg)

![e54fbb105e2c511fee11ba7b48173f1501e979aa6b42c359bf0f5def14674a53.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/images/e54fbb105e2c511fee11ba7b48173f1501e979aa6b42c359bf0f5def14674a53.jpg)

### Tables

![09460c73c82ec5770fb75b491141fe48e48f4c11d0ee569203364e33068780c2.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/tables/09460c73c82ec5770fb75b491141fe48e48f4c11d0ee569203364e33068780c2.jpg)

![137fec66feea0c680d7ae1f208c480da8fbae7e3b1221c7c3b42d145bd76ea9c.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/tables/137fec66feea0c680d7ae1f208c480da8fbae7e3b1221c7c3b42d145bd76ea9c.jpg)

![4838ccb0aea66ca912e4487a8802ce47980be4dcd34f4e19f67ca2d083da5caf.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/tables/4838ccb0aea66ca912e4487a8802ce47980be4dcd34f4e19f67ca2d083da5caf.jpg)

![589bf46dbca00b4e79862ac93da9b58bad2b143ea2ea497423d8f318048bf97a.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/tables/589bf46dbca00b4e79862ac93da9b58bad2b143ea2ea497423d8f318048bf97a.jpg)

![688104da02a095172304693ceae00245392598b129ab166eede133772c78e3e4.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/tables/688104da02a095172304693ceae00245392598b129ab166eede133772c78e3e4.jpg)

![699472371c2954e29137fe22fda24d8fe1ab507e7201c2dea14ecba4e18506e5.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/tables/699472371c2954e29137fe22fda24d8fe1ab507e7201c2dea14ecba4e18506e5.jpg)

![7dedb49fe74eadfd02e9b908b26b9929ea3ef631a17d0f6a454d340866e4dcd2.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/tables/7dedb49fe74eadfd02e9b908b26b9929ea3ef631a17d0f6a454d340866e4dcd2.jpg)

![b450ba6255935aa75d78c900ad2d3a244dbef29dab29f89e17e61ad8eea4b609.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/tables/b450ba6255935aa75d78c900ad2d3a244dbef29dab29f89e17e61ad8eea4b609.jpg)

![b91811f08b9f25ba8f4e045c4cf41190d4b53c4f0e1bc21d346a0944b4ae2ee3.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/tables/b91811f08b9f25ba8f4e045c4cf41190d4b53c4f0e1bc21d346a0944b4ae2ee3.jpg)

![fcf916d7da95b3c79e6011b9c090f7fa35c4f456f0e1052ebc8012f177944972.jpg](../icml_results/6209_Trajectory%20Seriation%20via%20Spectral%20Tangent%20Alignment%20and%20Global%20Embedding/tables/fcf916d7da95b3c79e6011b9c090f7fa35c4f456f0e1052ebc8012f177944972.jpg)

<a id="paper-6160"></a>

## Anti-Aliasing Matters: A Dynamic Network for Time Series Forecasting


### Images

![0016a6162ac1325bf64da484312e4600a1b67af846f436e256157ce0714ea9c4.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/images/0016a6162ac1325bf64da484312e4600a1b67af846f436e256157ce0714ea9c4.jpg)

![0603ba9da5f285ae8611b1f2236caa78047bdce1c52d13a2a711b24cbb8a1885.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/images/0603ba9da5f285ae8611b1f2236caa78047bdce1c52d13a2a711b24cbb8a1885.jpg)

![28dff12ecba74c07ee2771b3f491a511b65a2033119a93f9979a6a7d97095a78.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/images/28dff12ecba74c07ee2771b3f491a511b65a2033119a93f9979a6a7d97095a78.jpg)

![2ae1212d8273d679a63f343f7922b98d929e2e5a24c7505e562c6ebcf75ef51e.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/images/2ae1212d8273d679a63f343f7922b98d929e2e5a24c7505e562c6ebcf75ef51e.jpg)

![2f38f8b5cfe9f9b3904f1d25b0b5971eaca1685a2ca8518b51717df0c0baef2e.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/images/2f38f8b5cfe9f9b3904f1d25b0b5971eaca1685a2ca8518b51717df0c0baef2e.jpg)

![32bc6b67279bd9630554e607b7989fce84217568d20543ae57da32098dbb2e2a.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/images/32bc6b67279bd9630554e607b7989fce84217568d20543ae57da32098dbb2e2a.jpg)

![437274c6b7fe2587e0483e864b8224519b470db0b41663280582f8399370071b.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/images/437274c6b7fe2587e0483e864b8224519b470db0b41663280582f8399370071b.jpg)

![44fdcbab337206c82d1c5a6cec7f1c115aa744448eb5d83fbc51304b15aed2f1.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/images/44fdcbab337206c82d1c5a6cec7f1c115aa744448eb5d83fbc51304b15aed2f1.jpg)

![52d33273cef055f9dbf7e2c76a7526d4b0a056ee9eab39d9fdf115285c691dbb.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/images/52d33273cef055f9dbf7e2c76a7526d4b0a056ee9eab39d9fdf115285c691dbb.jpg)

![55647c1258303bf4b826bfc95906b47fcb057c40a291f76e3bfb709885c98526.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/images/55647c1258303bf4b826bfc95906b47fcb057c40a291f76e3bfb709885c98526.jpg)

![55985d622de172c9debc62c4826806437c9ae0168faf07aac98730df9a16a7fd.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/images/55985d622de172c9debc62c4826806437c9ae0168faf07aac98730df9a16a7fd.jpg)

![8047f11c2f8f8a20fb81f18f1dbc9a85e2fefeabfdebf3c34793f182355e031c.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/images/8047f11c2f8f8a20fb81f18f1dbc9a85e2fefeabfdebf3c34793f182355e031c.jpg)

![979defaabd2726004ada181a72c594b1dfcabdf52d642bfb209ec65aeed40208.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/images/979defaabd2726004ada181a72c594b1dfcabdf52d642bfb209ec65aeed40208.jpg)

![9af6ec2dac6fcbb87a1410ed05d00892205c14c42a672f17a7aea2ff05094b97.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/images/9af6ec2dac6fcbb87a1410ed05d00892205c14c42a672f17a7aea2ff05094b97.jpg)

![abe116c8bc885461301b0387bc0be3f00db3774bbd7a023f56f23e860ea4720c.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/images/abe116c8bc885461301b0387bc0be3f00db3774bbd7a023f56f23e860ea4720c.jpg)

![cefd579a10c04380b7d0e36662887f15318443e9df4b46ab191cb60df4eed502.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/images/cefd579a10c04380b7d0e36662887f15318443e9df4b46ab191cb60df4eed502.jpg)

![e7b3cdfb19048fff170f3ddb107fbf128bda0f1a704b22de45433d7c8dd5475e.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/images/e7b3cdfb19048fff170f3ddb107fbf128bda0f1a704b22de45433d7c8dd5475e.jpg)

### Tables

![0a411059113cb1c38107264db66b4eae0debb3fc03769c40242f1f5a55c13b82.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/0a411059113cb1c38107264db66b4eae0debb3fc03769c40242f1f5a55c13b82.jpg)

![19a352b3bb6454265b1b01c27792d6aeee5472f7058bd28ead0177c16c4f8697.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/19a352b3bb6454265b1b01c27792d6aeee5472f7058bd28ead0177c16c4f8697.jpg)

![1fb99d5b44ac25416f4453057a17623166c59f60fc49ca9698b21c27a007eae0.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/1fb99d5b44ac25416f4453057a17623166c59f60fc49ca9698b21c27a007eae0.jpg)

![2260aa3741ff0a338075d7493d7c3a7f5c2fb002ab3cc1c6ef3765fc65c96e1a.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/2260aa3741ff0a338075d7493d7c3a7f5c2fb002ab3cc1c6ef3765fc65c96e1a.jpg)

![34629f5799bf18ccce15f7b8fd06266826876f33c17e58252a97040f5354f9f5.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/34629f5799bf18ccce15f7b8fd06266826876f33c17e58252a97040f5354f9f5.jpg)

![3f5a2f1d78bab9002f7032fe60540812f8bfe08437b15a532d96243da9a2300b.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/3f5a2f1d78bab9002f7032fe60540812f8bfe08437b15a532d96243da9a2300b.jpg)

![5589dbc9be2105c3f8ba5edd5303e381213012f60a0b938e6c1e1260e9c4c1bb.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/5589dbc9be2105c3f8ba5edd5303e381213012f60a0b938e6c1e1260e9c4c1bb.jpg)

![57de98339ca3fa5c2d1088cd4e06f2c9ef90cd0c97f8378d79a189450fe5800d.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/57de98339ca3fa5c2d1088cd4e06f2c9ef90cd0c97f8378d79a189450fe5800d.jpg)

![5ee80ecd1526910b292175ed2daa365c42d636493cb2379ea98f9e72c6f95faf.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/5ee80ecd1526910b292175ed2daa365c42d636493cb2379ea98f9e72c6f95faf.jpg)

![6219d9079c8c8226b44fd20bb23120fac0e97ad9c295815778b0569737685089.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/6219d9079c8c8226b44fd20bb23120fac0e97ad9c295815778b0569737685089.jpg)

![82172408964741b622b726d6b41268660d31e2990e5c1b85c02a8731a30f2fdf.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/82172408964741b622b726d6b41268660d31e2990e5c1b85c02a8731a30f2fdf.jpg)

![84643a8c8e1332182e2c5155264b6eec1e498db16de1cb02a062bbab329967e9.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/84643a8c8e1332182e2c5155264b6eec1e498db16de1cb02a062bbab329967e9.jpg)

![8762e5fb16e4cd5014e282a4df81dae7337801ff49620d2771ea005ba2d42322.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/8762e5fb16e4cd5014e282a4df81dae7337801ff49620d2771ea005ba2d42322.jpg)

![968955ee3deef11e4ef4f9759eb8587bed5662451a65d36845ca565512c0c92d.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/968955ee3deef11e4ef4f9759eb8587bed5662451a65d36845ca565512c0c92d.jpg)

![997a5028a226e6b6395233fe48961ae22590734dc6787a83ca9c596d3c5124c2.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/997a5028a226e6b6395233fe48961ae22590734dc6787a83ca9c596d3c5124c2.jpg)

![a8ab4b7927e02607779e542c7b57e7679ec2428af6285b91bfffff2168835314.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/a8ab4b7927e02607779e542c7b57e7679ec2428af6285b91bfffff2168835314.jpg)

![b7198700d9470c5552c57d00debfd8c9f3f50d029bdcdaf057da9bd01337bac1.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/b7198700d9470c5552c57d00debfd8c9f3f50d029bdcdaf057da9bd01337bac1.jpg)

![c8af978f9f352cd195780e75d738802e9860ae471472b7c6bef1b74f46c793bb.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/c8af978f9f352cd195780e75d738802e9860ae471472b7c6bef1b74f46c793bb.jpg)

![ce0163caf6aad802ddd178d063b85cb8a2cd837b5e75d02ee4397379c3857fbb.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/ce0163caf6aad802ddd178d063b85cb8a2cd837b5e75d02ee4397379c3857fbb.jpg)

![d50875c755115bafa6cda558416eb1cc9ff4d4213be602cc3c671dcb0c244150.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/d50875c755115bafa6cda558416eb1cc9ff4d4213be602cc3c671dcb0c244150.jpg)

![e71824d301e85ea166fac62191f1d35560b1ad315d4fc36591c00bbe862b007c.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/e71824d301e85ea166fac62191f1d35560b1ad315d4fc36591c00bbe862b007c.jpg)

![eacbbb3aec4f40a42411b928f3c28907b0b20b9fb4c88df5b7f5a8fd5519cc6c.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/eacbbb3aec4f40a42411b928f3c28907b0b20b9fb4c88df5b7f5a8fd5519cc6c.jpg)

![ef2eb14ba4e646e0fc20333478078eff94519420f6e5066b76109b9585742645.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/ef2eb14ba4e646e0fc20333478078eff94519420f6e5066b76109b9585742645.jpg)

![f095655479fb130203b4e30cc7acafcb5e19a79660be4ab582f37e09e1b8a834.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/f095655479fb130203b4e30cc7acafcb5e19a79660be4ab582f37e09e1b8a834.jpg)

![f188d0cc6b3e77ef8bbd672ce97775dd705f39871bd4e518c0d9645b8f71d5f7.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/f188d0cc6b3e77ef8bbd672ce97775dd705f39871bd4e518c0d9645b8f71d5f7.jpg)

![ffe5368e97e2f8c5da123ee703013730962545b7e7443c4a8d535a11bfbe320b.jpg](../icml_results/6210_Anti-Aliasing%20Matters_%20A%20Dynamic%20Network%20for%20Time%20Series%20Forecasting/tables/ffe5368e97e2f8c5da123ee703013730962545b7e7443c4a8d535a11bfbe320b.jpg)

<a id="paper-6161"></a>

## Segment Anything with Robust Uncertainty-Accuracy Correlation


### Images

![0187602696d3572eafa9a7f9c6b971db9338f80f4a8cd0981d64f0a745bcac16.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/images/0187602696d3572eafa9a7f9c6b971db9338f80f4a8cd0981d64f0a745bcac16.jpg)

![0c3bbdbe8cc6a4987e18e4c8eb07555eb2d018794274832dbd4b6e4e70a33e08.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/images/0c3bbdbe8cc6a4987e18e4c8eb07555eb2d018794274832dbd4b6e4e70a33e08.jpg)

![1422ee90d231fe9e5ea3211daa3ac60f87fe9071a6cfb9b810a8434d13175e8f.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/images/1422ee90d231fe9e5ea3211daa3ac60f87fe9071a6cfb9b810a8434d13175e8f.jpg)

![872004d84d72dc8ea03b56f1941033ebbcf707961295db0247b61a54efdb2937.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/images/872004d84d72dc8ea03b56f1941033ebbcf707961295db0247b61a54efdb2937.jpg)

![88a58c3039e70d9c5044b7a54a4b688380a709e3872d04ce90b95246d016cf29.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/images/88a58c3039e70d9c5044b7a54a4b688380a709e3872d04ce90b95246d016cf29.jpg)

![ac98260736a1036ad539ddcf9fb37df0e26ab4378b4b4fdf25b920c00d84b786.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/images/ac98260736a1036ad539ddcf9fb37df0e26ab4378b4b4fdf25b920c00d84b786.jpg)

![add778665284145e6eb75fae3fd8cdc3fb80b3f8f4c9ffe2f15f21507b8eed76.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/images/add778665284145e6eb75fae3fd8cdc3fb80b3f8f4c9ffe2f15f21507b8eed76.jpg)

![c80a6c39a8aab9ef550452a1ba6188b30c38470c5ce92f4169e262706264357e.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/images/c80a6c39a8aab9ef550452a1ba6188b30c38470c5ce92f4169e262706264357e.jpg)

![e2eaa4833ec23edcc18bb440ff268d2a7e340cf0dbe10ff8ee6ffff0d1e14f7b.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/images/e2eaa4833ec23edcc18bb440ff268d2a7e340cf0dbe10ff8ee6ffff0d1e14f7b.jpg)

![ebbc463179f5683de90ae99c6a0938e7b21006e36553822f51a1f640092815e9.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/images/ebbc463179f5683de90ae99c6a0938e7b21006e36553822f51a1f640092815e9.jpg)

### Tables

![0ccb9139359341980e7285a02a2d333cb3d432f18760f613e9ac6452bb09fbbd.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/tables/0ccb9139359341980e7285a02a2d333cb3d432f18760f613e9ac6452bb09fbbd.jpg)

![0e7ee5f69628ca39c6eae0a23cdaf2ee25fc9539a7454b062cb95ba69991fd56.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/tables/0e7ee5f69628ca39c6eae0a23cdaf2ee25fc9539a7454b062cb95ba69991fd56.jpg)

![1c45b014bf03fffcbcbd4d162be3e12b0c2c2beb31d4ad086703930e39874ced.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/tables/1c45b014bf03fffcbcbd4d162be3e12b0c2c2beb31d4ad086703930e39874ced.jpg)

![1d0107e971f5f58d923c80d81fd4a40957e66c1207de4491c613421a80780a1d.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/tables/1d0107e971f5f58d923c80d81fd4a40957e66c1207de4491c613421a80780a1d.jpg)

![3a70a2d4fac76f8cd89d35c218cd13963fc2076fd22eb7df739a6d6ffc3cce71.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/tables/3a70a2d4fac76f8cd89d35c218cd13963fc2076fd22eb7df739a6d6ffc3cce71.jpg)

![3ce0d436468dcc66c139033381cd8aae96630d993ac298a7aed5d71674ce223f.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/tables/3ce0d436468dcc66c139033381cd8aae96630d993ac298a7aed5d71674ce223f.jpg)

![4d29f413c3e175bda527d093daa45b95a0871a4faad837d4d1fa2f522eb5f425.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/tables/4d29f413c3e175bda527d093daa45b95a0871a4faad837d4d1fa2f522eb5f425.jpg)

![67b9a6805f38be65c5f696f58d10eb85a77301a06e34d1acbd081443af38c253.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/tables/67b9a6805f38be65c5f696f58d10eb85a77301a06e34d1acbd081443af38c253.jpg)

![6f8026d0e99ce7fe5321caec34f906e641d4abb92aa56f35247852d32281245d.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/tables/6f8026d0e99ce7fe5321caec34f906e641d4abb92aa56f35247852d32281245d.jpg)

![8d1592f16cd18e55f3fd284646310aebdfa098212b6feb4f76dbfcc713e5a72f.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/tables/8d1592f16cd18e55f3fd284646310aebdfa098212b6feb4f76dbfcc713e5a72f.jpg)

![9fb2c53591b816d6877a23c4d48bc3f481deb6adddb35a5407aa7ae92beadec0.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/tables/9fb2c53591b816d6877a23c4d48bc3f481deb6adddb35a5407aa7ae92beadec0.jpg)

![ab3e084f3050a3635fb9e0fc2d51e145c439a5733767813f42042653f27c9f7e.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/tables/ab3e084f3050a3635fb9e0fc2d51e145c439a5733767813f42042653f27c9f7e.jpg)

![d144015822d0898b712e8e306cd955e24c69131d8f91edd7935265f17bd612a1.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/tables/d144015822d0898b712e8e306cd955e24c69131d8f91edd7935265f17bd612a1.jpg)

![dfaa3c6369cf8fc5468bab1de34105d7ab141c8ca0a0fff64794ffa0ec66024c.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/tables/dfaa3c6369cf8fc5468bab1de34105d7ab141c8ca0a0fff64794ffa0ec66024c.jpg)

![e41839052e0faa4e9bfa92ca4d16ee884c3f3e5303e538d272d8bf33dba30b46.jpg](../icml_results/6211_Segment%20Anything%20with%20Robust%20Uncertainty-Accuracy%20Correlation/tables/e41839052e0faa4e9bfa92ca4d16ee884c3f3e5303e538d272d8bf33dba30b46.jpg)

<a id="paper-6162"></a>

## IMPACT: Influence Modeling for Open-Set Time Series Anomaly Detection


### Images

![0e8c45b90f9fabcca978c563cfb70e5085b364851a4772f58b96a07375ac9719.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/images/0e8c45b90f9fabcca978c563cfb70e5085b364851a4772f58b96a07375ac9719.jpg)

![1bf224514a00a590602fcf6115c768db8fd16eb096fe5aebba506a3708708482.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/images/1bf224514a00a590602fcf6115c768db8fd16eb096fe5aebba506a3708708482.jpg)

![269e39518998b7a1b067442e3e41aed9c7b965f4f1fd910bd445618016f76b59.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/images/269e39518998b7a1b067442e3e41aed9c7b965f4f1fd910bd445618016f76b59.jpg)

![5e58b681a544560dce31eafaa8fbd1ee4b5dc32164d7776be00997985645e106.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/images/5e58b681a544560dce31eafaa8fbd1ee4b5dc32164d7776be00997985645e106.jpg)

![63f2abb08e1dee22ada24b81e36324880e4ee8b082bfa985455232c59ed0de04.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/images/63f2abb08e1dee22ada24b81e36324880e4ee8b082bfa985455232c59ed0de04.jpg)

![667e0bd3a6a9100b707d7813084b2137623d80a4b29f6078114250f813c3272c.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/images/667e0bd3a6a9100b707d7813084b2137623d80a4b29f6078114250f813c3272c.jpg)

![767bf04b8b6dd6e3e9515ef595a4518797969b76f52ace251d791904135c4b5b.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/images/767bf04b8b6dd6e3e9515ef595a4518797969b76f52ace251d791904135c4b5b.jpg)

![9c14298af7882267e6c2defaa84232c46548c16963badf7a63a7d1ffa5d3e9c8.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/images/9c14298af7882267e6c2defaa84232c46548c16963badf7a63a7d1ffa5d3e9c8.jpg)

![a69c38af0d4cf8f6b1625187c8160cc9cea279dec819e60fc86c21bf8d496c17.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/images/a69c38af0d4cf8f6b1625187c8160cc9cea279dec819e60fc86c21bf8d496c17.jpg)

![b9a4f72eb7c3da41db8e31394186a85dbb6c17b101f052ad8e561d5b6765806d.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/images/b9a4f72eb7c3da41db8e31394186a85dbb6c17b101f052ad8e561d5b6765806d.jpg)

![cd8c433aa6cabd8e6503e832c2ecd70c4b7f326d6a841adf6d47b2db062a0df8.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/images/cd8c433aa6cabd8e6503e832c2ecd70c4b7f326d6a841adf6d47b2db062a0df8.jpg)

![cff83c369bf8f86ca47ddf383c827cf52c4ad8457a0eb6a396e0cc208526e004.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/images/cff83c369bf8f86ca47ddf383c827cf52c4ad8457a0eb6a396e0cc208526e004.jpg)

![d71fa821525e93c2ce18bf0de2884d2f6cba8572d639417c9255b66fa3a5ab91.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/images/d71fa821525e93c2ce18bf0de2884d2f6cba8572d639417c9255b66fa3a5ab91.jpg)

![df89b10c7169ff2ec7544db0f093da5b1cc489ce0551e74779be6e83a4d21b35.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/images/df89b10c7169ff2ec7544db0f093da5b1cc489ce0551e74779be6e83a4d21b35.jpg)

![e2b42631b33ce115f9777608346e1aedecfdb66e641d19e6d2a932243fcccd86.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/images/e2b42631b33ce115f9777608346e1aedecfdb66e641d19e6d2a932243fcccd86.jpg)

### Tables

![4aaa6e1581d5f6157604c382bb160cffcd6721f6a352807b4d79cc637e91bcf7.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/tables/4aaa6e1581d5f6157604c382bb160cffcd6721f6a352807b4d79cc637e91bcf7.jpg)

![5d8d7780b4616e94fce17849c1815e7ff6eb1677912e89d9fb732781a96aa8c0.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/tables/5d8d7780b4616e94fce17849c1815e7ff6eb1677912e89d9fb732781a96aa8c0.jpg)

![8a7edc0aa52e6f14430d5b83974c939ac3692ab813772ec4a1f2d06eaa4fb6fa.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/tables/8a7edc0aa52e6f14430d5b83974c939ac3692ab813772ec4a1f2d06eaa4fb6fa.jpg)

![8d35287a34802b73dfcbce2f44fa27dbc755f25b5013670b9a349208d11b32dd.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/tables/8d35287a34802b73dfcbce2f44fa27dbc755f25b5013670b9a349208d11b32dd.jpg)

![c5495ec995b276b9fea5cd5f5536385d84f31fedb80482a668c33cc9fe1c4513.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/tables/c5495ec995b276b9fea5cd5f5536385d84f31fedb80482a668c33cc9fe1c4513.jpg)

![df6391c9e7d85dde1ded0915f18d19c7281e12f359869d9fc5026f4cbed25cf6.jpg](../icml_results/6212_IMPACT_%20Influence%20Modeling%20for%20Open-Set%20Time%20Series%20Anomaly%20Detection/tables/df6391c9e7d85dde1ded0915f18d19c7281e12f359869d9fc5026f4cbed25cf6.jpg)

<a id="paper-6163"></a>

## How Transformers Represent Hierarchies: A Local-to-Global Mechanism


### Images

![02466f580037d49e044d3ad2001f223c8ef6bd9507069c66d748c00d1d2108c6.jpg](../icml_results/6213_How%20Transformers%20Represent%20Hierarchies_%20A%20Local-to-Global%20Mechanism/images/02466f580037d49e044d3ad2001f223c8ef6bd9507069c66d748c00d1d2108c6.jpg)

![0db61fe8d2e4c9577624f41ea2e301ba0891d9146144ca2b81e6603dbe0f8c0e.jpg](../icml_results/6213_How%20Transformers%20Represent%20Hierarchies_%20A%20Local-to-Global%20Mechanism/images/0db61fe8d2e4c9577624f41ea2e301ba0891d9146144ca2b81e6603dbe0f8c0e.jpg)

![188f615eadea148cccdb5465498d05fab78bbe6db5def3d34bc1e7333ed8634a.jpg](../icml_results/6213_How%20Transformers%20Represent%20Hierarchies_%20A%20Local-to-Global%20Mechanism/images/188f615eadea148cccdb5465498d05fab78bbe6db5def3d34bc1e7333ed8634a.jpg)

![19844786fcb5dd630dab9e47a2e1808624c0269d6484907a5e06ce108eb43e52.jpg](../icml_results/6213_How%20Transformers%20Represent%20Hierarchies_%20A%20Local-to-Global%20Mechanism/images/19844786fcb5dd630dab9e47a2e1808624c0269d6484907a5e06ce108eb43e52.jpg)

![1d090b8b88da4656b94988280b1f60afaa3788824dd6bf523fefea9b983acd82.jpg](../icml_results/6213_How%20Transformers%20Represent%20Hierarchies_%20A%20Local-to-Global%20Mechanism/images/1d090b8b88da4656b94988280b1f60afaa3788824dd6bf523fefea9b983acd82.jpg)

![4edd723f1f5f073a6a71179fc62640a5660f4cbcefec123cb6a7f63b02456562.jpg](../icml_results/6213_How%20Transformers%20Represent%20Hierarchies_%20A%20Local-to-Global%20Mechanism/images/4edd723f1f5f073a6a71179fc62640a5660f4cbcefec123cb6a7f63b02456562.jpg)

![5dfd7c36bead22e1b0e3efa2bb973423bf5874c19fab51ed8324bb99fed87e5b.jpg](../icml_results/6213_How%20Transformers%20Represent%20Hierarchies_%20A%20Local-to-Global%20Mechanism/images/5dfd7c36bead22e1b0e3efa2bb973423bf5874c19fab51ed8324bb99fed87e5b.jpg)

![948d26d6eb9035fa1e36565b0ac54df17b5b788b285d015745aa01a1faad262d.jpg](../icml_results/6213_How%20Transformers%20Represent%20Hierarchies_%20A%20Local-to-Global%20Mechanism/images/948d26d6eb9035fa1e36565b0ac54df17b5b788b285d015745aa01a1faad262d.jpg)

![9c9ae80d43a8159c05a112cd13a8369fbbd9f5c230bf6af20f13a80ab1ef3f1b.jpg](../icml_results/6213_How%20Transformers%20Represent%20Hierarchies_%20A%20Local-to-Global%20Mechanism/images/9c9ae80d43a8159c05a112cd13a8369fbbd9f5c230bf6af20f13a80ab1ef3f1b.jpg)

![a155562a84b69a00c2f4cfe09303014cbf4e1b4ba094dc7929176e5c92ee215c.jpg](../icml_results/6213_How%20Transformers%20Represent%20Hierarchies_%20A%20Local-to-Global%20Mechanism/images/a155562a84b69a00c2f4cfe09303014cbf4e1b4ba094dc7929176e5c92ee215c.jpg)

![a9b7d8d1a4eb8224d45f7b19d572fba9598c7330290f05280954df2b48bda02b.jpg](../icml_results/6213_How%20Transformers%20Represent%20Hierarchies_%20A%20Local-to-Global%20Mechanism/images/a9b7d8d1a4eb8224d45f7b19d572fba9598c7330290f05280954df2b48bda02b.jpg)

![efd196a1255f47bf39e70db6ca868f86f2f32a85f0a4abc91759c1f95376f6c2.jpg](../icml_results/6213_How%20Transformers%20Represent%20Hierarchies_%20A%20Local-to-Global%20Mechanism/images/efd196a1255f47bf39e70db6ca868f86f2f32a85f0a4abc91759c1f95376f6c2.jpg)

![f0fed26b3d24e71ced4b6c067e03e7141d07d6e188924e6153ccfb0243f17339.jpg](../icml_results/6213_How%20Transformers%20Represent%20Hierarchies_%20A%20Local-to-Global%20Mechanism/images/f0fed26b3d24e71ced4b6c067e03e7141d07d6e188924e6153ccfb0243f17339.jpg)

### Tables

![1bf6ab5f495e48aa48a212a825183d0c6ee03198dadb19cb44969414699f2d3e.jpg](../icml_results/6213_How%20Transformers%20Represent%20Hierarchies_%20A%20Local-to-Global%20Mechanism/tables/1bf6ab5f495e48aa48a212a825183d0c6ee03198dadb19cb44969414699f2d3e.jpg)

![79ac306dd603c787678af343b80d542bd50d416be5607fa94d191f497a4e3854.jpg](../icml_results/6213_How%20Transformers%20Represent%20Hierarchies_%20A%20Local-to-Global%20Mechanism/tables/79ac306dd603c787678af343b80d542bd50d416be5607fa94d191f497a4e3854.jpg)

![f5fef08872a97abc073c4e9b29a76c102cd2c29af98713e7f220936b22f16068.jpg](../icml_results/6213_How%20Transformers%20Represent%20Hierarchies_%20A%20Local-to-Global%20Mechanism/tables/f5fef08872a97abc073c4e9b29a76c102cd2c29af98713e7f220936b22f16068.jpg)

<a id="paper-6164"></a>

## Neural Vector Lyapunov–Razumikhin Certificates for Delayed Interconnected Systems


### Images

![7f9bd9082d2e781e803492d73193164edc72f1d34ebcc0dc8eaa06a6eafea5fe.jpg](../icml_results/6214_Neural%20Vector%20Lyapunov%E2%80%93Razumikhin%20Certificates%20for%20Delayed%20Interconnected%20Systems/images/7f9bd9082d2e781e803492d73193164edc72f1d34ebcc0dc8eaa06a6eafea5fe.jpg)

![993c69b6e5208c83f0bdda6838ff07b46128a9319b63ba23a855585d20c80a3b.jpg](../icml_results/6214_Neural%20Vector%20Lyapunov%E2%80%93Razumikhin%20Certificates%20for%20Delayed%20Interconnected%20Systems/images/993c69b6e5208c83f0bdda6838ff07b46128a9319b63ba23a855585d20c80a3b.jpg)

![cf942b1ecdf339afd23f4ee83b07fb53faa4ec28957b5f7e8575151432e7a929.jpg](../icml_results/6214_Neural%20Vector%20Lyapunov%E2%80%93Razumikhin%20Certificates%20for%20Delayed%20Interconnected%20Systems/images/cf942b1ecdf339afd23f4ee83b07fb53faa4ec28957b5f7e8575151432e7a929.jpg)

![e8a7f1c6bed4c7a87358fe113a3a2b38f217a12b91e0b0513c827926850b7add.jpg](../icml_results/6214_Neural%20Vector%20Lyapunov%E2%80%93Razumikhin%20Certificates%20for%20Delayed%20Interconnected%20Systems/images/e8a7f1c6bed4c7a87358fe113a3a2b38f217a12b91e0b0513c827926850b7add.jpg)

### Tables

![3c8f299959d9198ba910a75f2261a479a69806f64f6a39d14d0927ca64de58d9.jpg](../icml_results/6214_Neural%20Vector%20Lyapunov%E2%80%93Razumikhin%20Certificates%20for%20Delayed%20Interconnected%20Systems/tables/3c8f299959d9198ba910a75f2261a479a69806f64f6a39d14d0927ca64de58d9.jpg)

![b5e3666b20be67eefa5b59c413986c0017f2325236b8fdd64c89a8f6e4b2b417.jpg](../icml_results/6214_Neural%20Vector%20Lyapunov%E2%80%93Razumikhin%20Certificates%20for%20Delayed%20Interconnected%20Systems/tables/b5e3666b20be67eefa5b59c413986c0017f2325236b8fdd64c89a8f6e4b2b417.jpg)

![c8cd944f329423db2cc66a1d32dd32c9fff9da2fc5cac80ede3bb35b1411fc19.jpg](../icml_results/6214_Neural%20Vector%20Lyapunov%E2%80%93Razumikhin%20Certificates%20for%20Delayed%20Interconnected%20Systems/tables/c8cd944f329423db2cc66a1d32dd32c9fff9da2fc5cac80ede3bb35b1411fc19.jpg)

![f16b1906e52957c49777507e87e47fcb8c00f797b36c714e70125546ba3d58e5.jpg](../icml_results/6214_Neural%20Vector%20Lyapunov%E2%80%93Razumikhin%20Certificates%20for%20Delayed%20Interconnected%20Systems/tables/f16b1906e52957c49777507e87e47fcb8c00f797b36c714e70125546ba3d58e5.jpg)

<a id="paper-6165"></a>

## Extra-Merge: Tracing the Rank-1 Subspace of Model Merging in Language Model Pre-Training


### Images

![31d9769491f838452b57aac6f08d0388ddb778d8f6b6d0e15f12d041593226c9.jpg](../icml_results/6215_Extra-Merge_%20Tracing%20the%20Rank-1%20Subspace%20of%20Model%20Merging%20in%20Language%20Model%20Pre-Training/images/31d9769491f838452b57aac6f08d0388ddb778d8f6b6d0e15f12d041593226c9.jpg)

![45f1076b8419f6b474eecefeb403e6b27595ea279e53bb8acfff5e2c9336a2e6.jpg](../icml_results/6215_Extra-Merge_%20Tracing%20the%20Rank-1%20Subspace%20of%20Model%20Merging%20in%20Language%20Model%20Pre-Training/images/45f1076b8419f6b474eecefeb403e6b27595ea279e53bb8acfff5e2c9336a2e6.jpg)

![48553e0d64cea07a477b13e17d9ba729500ff4eac594e76464c666b353648000.jpg](../icml_results/6215_Extra-Merge_%20Tracing%20the%20Rank-1%20Subspace%20of%20Model%20Merging%20in%20Language%20Model%20Pre-Training/images/48553e0d64cea07a477b13e17d9ba729500ff4eac594e76464c666b353648000.jpg)

![8be3a334e01c757a8eb96f23d7b1ae4d680b3e6a78dbdadd5bbe188379dc043c.jpg](../icml_results/6215_Extra-Merge_%20Tracing%20the%20Rank-1%20Subspace%20of%20Model%20Merging%20in%20Language%20Model%20Pre-Training/images/8be3a334e01c757a8eb96f23d7b1ae4d680b3e6a78dbdadd5bbe188379dc043c.jpg)

![a9eaf1108bc859f35aef0208e81b53ed3c59d3f7a2d84fd5abcdc94af59e957f.jpg](../icml_results/6215_Extra-Merge_%20Tracing%20the%20Rank-1%20Subspace%20of%20Model%20Merging%20in%20Language%20Model%20Pre-Training/images/a9eaf1108bc859f35aef0208e81b53ed3c59d3f7a2d84fd5abcdc94af59e957f.jpg)

![db20e34aed3e7e976d080784eec78e1cc29c281ba245af063cf9030f87746bee.jpg](../icml_results/6215_Extra-Merge_%20Tracing%20the%20Rank-1%20Subspace%20of%20Model%20Merging%20in%20Language%20Model%20Pre-Training/images/db20e34aed3e7e976d080784eec78e1cc29c281ba245af063cf9030f87746bee.jpg)

![f5d6cc567b05209bbbe6b94668a1e2c030fe5fe0aaa38b102065333cc1d24a4d.jpg](../icml_results/6215_Extra-Merge_%20Tracing%20the%20Rank-1%20Subspace%20of%20Model%20Merging%20in%20Language%20Model%20Pre-Training/images/f5d6cc567b05209bbbe6b94668a1e2c030fe5fe0aaa38b102065333cc1d24a4d.jpg)

### Tables

![0d920726e8f6a81f972f5ee3bcd15eca579ae75d2ea01f94d06c50f1ba0d3620.jpg](../icml_results/6215_Extra-Merge_%20Tracing%20the%20Rank-1%20Subspace%20of%20Model%20Merging%20in%20Language%20Model%20Pre-Training/tables/0d920726e8f6a81f972f5ee3bcd15eca579ae75d2ea01f94d06c50f1ba0d3620.jpg)

![63e8eabce70aaa6927793d5f4cbf51f1b3dfa3a6b67ecf5eaba750a9c4fb2d72.jpg](../icml_results/6215_Extra-Merge_%20Tracing%20the%20Rank-1%20Subspace%20of%20Model%20Merging%20in%20Language%20Model%20Pre-Training/tables/63e8eabce70aaa6927793d5f4cbf51f1b3dfa3a6b67ecf5eaba750a9c4fb2d72.jpg)

![f48e6dfc3a18a9c03b46bff6c66c8bdeced55405c040f9c8ecdf699bc8778ce9.jpg](../icml_results/6215_Extra-Merge_%20Tracing%20the%20Rank-1%20Subspace%20of%20Model%20Merging%20in%20Language%20Model%20Pre-Training/tables/f48e6dfc3a18a9c03b46bff6c66c8bdeced55405c040f9c8ecdf699bc8778ce9.jpg)

<a id="paper-6166"></a>

## Uni-DocRobust: Universal Plug-and-Play Robustness Enhancement for Multi-modal LLMs via Feature Restoration


### Images

![46931206001efcd9109f9010022a0cb868c5e678d5f0e0a9bca1590bf2c92a51.jpg](../icml_results/6216_Uni-DocRobust_%20Universal%20Plug-and-Play%20Robustness%20Enhancement%20for%20Multi-modal%20LLMs%20via%20Feature%20Resto/images/46931206001efcd9109f9010022a0cb868c5e678d5f0e0a9bca1590bf2c92a51.jpg)

![62084b3881a5886a1553f92bb7bd581948260e69dbad3eadb59e3c2c156dd52f.jpg](../icml_results/6216_Uni-DocRobust_%20Universal%20Plug-and-Play%20Robustness%20Enhancement%20for%20Multi-modal%20LLMs%20via%20Feature%20Resto/images/62084b3881a5886a1553f92bb7bd581948260e69dbad3eadb59e3c2c156dd52f.jpg)

![719af3afd56c3c370d1d5c39cce3135f2fec8095402b170dc16a65f822ff193f.jpg](../icml_results/6216_Uni-DocRobust_%20Universal%20Plug-and-Play%20Robustness%20Enhancement%20for%20Multi-modal%20LLMs%20via%20Feature%20Resto/images/719af3afd56c3c370d1d5c39cce3135f2fec8095402b170dc16a65f822ff193f.jpg)

![928e8820f7aeb3a4d310b5540e51401bb89d390c3591a3b6d16de2dd81000bdf.jpg](../icml_results/6216_Uni-DocRobust_%20Universal%20Plug-and-Play%20Robustness%20Enhancement%20for%20Multi-modal%20LLMs%20via%20Feature%20Resto/images/928e8820f7aeb3a4d310b5540e51401bb89d390c3591a3b6d16de2dd81000bdf.jpg)

![e6c874dfc7ed32629bb281d9fa61a6bc793f8ebcfd14be50bc5aaa8676a0967e.jpg](../icml_results/6216_Uni-DocRobust_%20Universal%20Plug-and-Play%20Robustness%20Enhancement%20for%20Multi-modal%20LLMs%20via%20Feature%20Resto/images/e6c874dfc7ed32629bb281d9fa61a6bc793f8ebcfd14be50bc5aaa8676a0967e.jpg)

### Tables

![2fc92750f8f81bc4e24cc4f38269ced1fefe1c28221395a1ef89852348ad46d4.jpg](../icml_results/6216_Uni-DocRobust_%20Universal%20Plug-and-Play%20Robustness%20Enhancement%20for%20Multi-modal%20LLMs%20via%20Feature%20Resto/tables/2fc92750f8f81bc4e24cc4f38269ced1fefe1c28221395a1ef89852348ad46d4.jpg)

![54fcf62dc9057c88e95382c14eeb73297e1a5be803481de3a57883191d923298.jpg](../icml_results/6216_Uni-DocRobust_%20Universal%20Plug-and-Play%20Robustness%20Enhancement%20for%20Multi-modal%20LLMs%20via%20Feature%20Resto/tables/54fcf62dc9057c88e95382c14eeb73297e1a5be803481de3a57883191d923298.jpg)

![7b6c40ab4094c56b6e67d0e5479d68cb84bbcc012e147e4e8e5b1095464251cd.jpg](../icml_results/6216_Uni-DocRobust_%20Universal%20Plug-and-Play%20Robustness%20Enhancement%20for%20Multi-modal%20LLMs%20via%20Feature%20Resto/tables/7b6c40ab4094c56b6e67d0e5479d68cb84bbcc012e147e4e8e5b1095464251cd.jpg)

![bcb04289f2bdfab9d129fc8374fb6ac61e0bf67035006c4fff93d752c6a120b6.jpg](../icml_results/6216_Uni-DocRobust_%20Universal%20Plug-and-Play%20Robustness%20Enhancement%20for%20Multi-modal%20LLMs%20via%20Feature%20Resto/tables/bcb04289f2bdfab9d129fc8374fb6ac61e0bf67035006c4fff93d752c6a120b6.jpg)

![c953c2dd6b0c07e42d1d6723006c2ae6571bc156ef7ec411c28c092f15ee2df7.jpg](../icml_results/6216_Uni-DocRobust_%20Universal%20Plug-and-Play%20Robustness%20Enhancement%20for%20Multi-modal%20LLMs%20via%20Feature%20Resto/tables/c953c2dd6b0c07e42d1d6723006c2ae6571bc156ef7ec411c28c092f15ee2df7.jpg)

![eacf8b4f06f51fbccad87903c73d2a437c980e01eca6b236834ec7e22f0ba629.jpg](../icml_results/6216_Uni-DocRobust_%20Universal%20Plug-and-Play%20Robustness%20Enhancement%20for%20Multi-modal%20LLMs%20via%20Feature%20Resto/tables/eacf8b4f06f51fbccad87903c73d2a437c980e01eca6b236834ec7e22f0ba629.jpg)

![ffaa170e15559a072b46f87a99333187b964d656d05c4ea4d85d057cbb852f64.jpg](../icml_results/6216_Uni-DocRobust_%20Universal%20Plug-and-Play%20Robustness%20Enhancement%20for%20Multi-modal%20LLMs%20via%20Feature%20Resto/tables/ffaa170e15559a072b46f87a99333187b964d656d05c4ea4d85d057cbb852f64.jpg)

<a id="paper-6167"></a>

## Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale Geometries


### Images

![0225763e01bf45f105ce2e6e89abce15299088bb9e3f2bb7b4d5a9136da4e486.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/images/0225763e01bf45f105ce2e6e89abce15299088bb9e3f2bb7b4d5a9136da4e486.jpg)

![07d6a86bba55912eb8760054799e015fe10273d738c510d2f9358761da0606d6.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/images/07d6a86bba55912eb8760054799e015fe10273d738c510d2f9358761da0606d6.jpg)

![10c165f001dbce82d481ae044226d550ae7030a9bd0656e6336f09d1efdff28d.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/images/10c165f001dbce82d481ae044226d550ae7030a9bd0656e6336f09d1efdff28d.jpg)

![1ac223bb5f1f942eae7370329d2a6b4534d982635713b4017e62c9ee2667dc9f.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/images/1ac223bb5f1f942eae7370329d2a6b4534d982635713b4017e62c9ee2667dc9f.jpg)

![4bf36980a6daaae50e4cc9fbec639680afc0861de6641efc1cee746a1c505125.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/images/4bf36980a6daaae50e4cc9fbec639680afc0861de6641efc1cee746a1c505125.jpg)

![5073be339fdb824a4a853ac69319005c66c5c8e2ce467f7897ef1fa32a3de859.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/images/5073be339fdb824a4a853ac69319005c66c5c8e2ce467f7897ef1fa32a3de859.jpg)

![5579f64d92c42366a46fa11441ed7e311807ccd49c8d399a6f5ea9dce766eb3d.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/images/5579f64d92c42366a46fa11441ed7e311807ccd49c8d399a6f5ea9dce766eb3d.jpg)

![596ad7215b7dae5615a34825bc804aa2776317be727513affaa288fdcd6115eb.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/images/596ad7215b7dae5615a34825bc804aa2776317be727513affaa288fdcd6115eb.jpg)

![658a54fe4c3a9b3b4fbd57f5819f6ac6cd6f376f61937be65af9be346b6464d1.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/images/658a54fe4c3a9b3b4fbd57f5819f6ac6cd6f376f61937be65af9be346b6464d1.jpg)

![841e028a94e65a109cfb6d84f166592a1485b846cfde4439303fb7de567bac16.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/images/841e028a94e65a109cfb6d84f166592a1485b846cfde4439303fb7de567bac16.jpg)

![8a93fdaebfde9a3441df256535eebb6ed17514f99e18834ab4db698cc497acde.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/images/8a93fdaebfde9a3441df256535eebb6ed17514f99e18834ab4db698cc497acde.jpg)

![981a251838eb4d825b19e7676a246b198bf91665bfa926fe836e80506b57db80.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/images/981a251838eb4d825b19e7676a246b198bf91665bfa926fe836e80506b57db80.jpg)

![a28e510736a5dabaf2661042990877f2a4476ea8db401b08083c7e96d59cebb8.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/images/a28e510736a5dabaf2661042990877f2a4476ea8db401b08083c7e96d59cebb8.jpg)

![a3a06e0e81f7a0ff2038a7ce0e1659ebda6685b7ac9940ef7e8af1b74f8b4caa.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/images/a3a06e0e81f7a0ff2038a7ce0e1659ebda6685b7ac9940ef7e8af1b74f8b4caa.jpg)

![b2625a25574086bd0aaaa27279dc166d0ff3f0faaebc47532289461f4d33e09b.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/images/b2625a25574086bd0aaaa27279dc166d0ff3f0faaebc47532289461f4d33e09b.jpg)

### Tables

![080de1b530eb7a6a21615b3919096876794a49bf8741621952c8566d742ad723.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/tables/080de1b530eb7a6a21615b3919096876794a49bf8741621952c8566d742ad723.jpg)

![3317e5adeb7d541fa118e495da206df654ba751ecc89758fca8811f8d64c4f88.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/tables/3317e5adeb7d541fa118e495da206df654ba751ecc89758fca8811f8d64c4f88.jpg)

![51cf5cf1c016f37942a939eec3e26c45f64c026a86ef22a348f1e25d276af34d.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/tables/51cf5cf1c016f37942a939eec3e26c45f64c026a86ef22a348f1e25d276af34d.jpg)

![6113737fb30b09f1ff82a17a850d716253232bb5ac6fc57460a02253cbd563da.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/tables/6113737fb30b09f1ff82a17a850d716253232bb5ac6fc57460a02253cbd563da.jpg)

![61dd17b7d2b3491003b53bb7ebd111b26879b32855d855f33453fdfebc84412c.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/tables/61dd17b7d2b3491003b53bb7ebd111b26879b32855d855f33453fdfebc84412c.jpg)

![627a2bf96a80904248635d517fc4b86c341abea44a35bdc262c410f27ea370df.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/tables/627a2bf96a80904248635d517fc4b86c341abea44a35bdc262c410f27ea370df.jpg)

![9b1e29730be9ef1743ce5d888107544157cea3676870e23e97464f54f9d43474.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/tables/9b1e29730be9ef1743ce5d888107544157cea3676870e23e97464f54f9d43474.jpg)

![b5239cd568f842d08e26e114852cb18946c9a78bce8f056f1c445dfee4fe9262.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/tables/b5239cd568f842d08e26e114852cb18946c9a78bce8f056f1c445dfee4fe9262.jpg)

![cb327c67ee7a7a392d307c83ba603d45e9cd76236d930c3c2bf4f3e0fd1482c5.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/tables/cb327c67ee7a7a392d307c83ba603d45e9cd76236d930c3c2bf4f3e0fd1482c5.jpg)

![f1c84469b74d95b2d7f544a94b6dd5fb552b74cf95af4ea0f7066e9a1155d366.jpg](../icml_results/6217_Transolver-3_%20Scaling%20Up%20Transformer%20Solvers%20to%20Industrial-Scale%20Geometries/tables/f1c84469b74d95b2d7f544a94b6dd5fb552b74cf95af4ea0f7066e9a1155d366.jpg)

<a id="paper-6168"></a>

## Dismantling the Illusion of Vision-Language-Action Models Competence via Explicit Distributional Shifts


### Images

![196da98cf03aae7fb41c281be4173372e106b77dec6a0c0f1f943052b5f47e36.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/images/196da98cf03aae7fb41c281be4173372e106b77dec6a0c0f1f943052b5f47e36.jpg)

![19835bc468f5890acca55922dd66d6773a29e4d4466eedfad88798703f0332d3.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/images/19835bc468f5890acca55922dd66d6773a29e4d4466eedfad88798703f0332d3.jpg)

![19b47f5611dc489f4abd31ba4e14bce60f42b02737b076bdcd34ceae9d8021e9.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/images/19b47f5611dc489f4abd31ba4e14bce60f42b02737b076bdcd34ceae9d8021e9.jpg)

![3c478cb771731b64f49c0311da9c4b4a63618828a95e2e066117ac1ecc640f5b.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/images/3c478cb771731b64f49c0311da9c4b4a63618828a95e2e066117ac1ecc640f5b.jpg)

![42ca6c87b0b68ee3b848993ca222b95c26dff3bbc1e0ea5e6cf20c5dd3326b53.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/images/42ca6c87b0b68ee3b848993ca222b95c26dff3bbc1e0ea5e6cf20c5dd3326b53.jpg)

![633e189a0fa5ac61f99e65ccbbd6bf6bfef2b6526986ce00b0a1aff19baee186.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/images/633e189a0fa5ac61f99e65ccbbd6bf6bfef2b6526986ce00b0a1aff19baee186.jpg)

![6c508c9123b3f98b464720fcaf46dbba85def28d326527cec87556ff80ac5be9.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/images/6c508c9123b3f98b464720fcaf46dbba85def28d326527cec87556ff80ac5be9.jpg)

![742f036a53a11d7457c2fee081f117a1792a6d9963a152e726c14e755b45e92a.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/images/742f036a53a11d7457c2fee081f117a1792a6d9963a152e726c14e755b45e92a.jpg)

![804319242d320f42733cd2991b4586c2f9cc40a51cc0558b2bdf62c52fd8a497.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/images/804319242d320f42733cd2991b4586c2f9cc40a51cc0558b2bdf62c52fd8a497.jpg)

![b010621c82752eca70351f3242acba387f2936ad2a9a25dbc4b5794460f86ae1.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/images/b010621c82752eca70351f3242acba387f2936ad2a9a25dbc4b5794460f86ae1.jpg)

![b04586aaa743f8f3cdba40fa958c6499e4770cf7b59d0470b9e62ebe78963009.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/images/b04586aaa743f8f3cdba40fa958c6499e4770cf7b59d0470b9e62ebe78963009.jpg)

![b3e50ca3a219076f1fe294779baff43de2810804b30a05deb42591888f986e5d.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/images/b3e50ca3a219076f1fe294779baff43de2810804b30a05deb42591888f986e5d.jpg)

![d2e3d19ab0fc2c84179ca53f1455db3fd50c1101a91becbb571e7f2606e560fb.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/images/d2e3d19ab0fc2c84179ca53f1455db3fd50c1101a91becbb571e7f2606e560fb.jpg)

![e2811dbdc1fe520b7dedfe7f5cacd11495ce0d9448004801f53c91c3461427ef.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/images/e2811dbdc1fe520b7dedfe7f5cacd11495ce0d9448004801f53c91c3461427ef.jpg)

![fc42df107cdb4f6f34fcc40b816ccc2596a9c2fdca223ca518ff777023349d32.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/images/fc42df107cdb4f6f34fcc40b816ccc2596a9c2fdca223ca518ff777023349d32.jpg)

### Tables

![090946775019fbfe1f24ecc8cda563202d8feed850bcc4824ff4f4f349686962.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/tables/090946775019fbfe1f24ecc8cda563202d8feed850bcc4824ff4f4f349686962.jpg)

![1d646f342b4fb4cb6b7d8fb10f6fc23d026db1592b5d1000e6ea74857155a251.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/tables/1d646f342b4fb4cb6b7d8fb10f6fc23d026db1592b5d1000e6ea74857155a251.jpg)

![340a7b68f6fdb5d3ee0aa70994713eb8e8a667c0309fa744cadf6353cf853da3.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/tables/340a7b68f6fdb5d3ee0aa70994713eb8e8a667c0309fa744cadf6353cf853da3.jpg)

![3bfc226b6b92052c889cf0212e0e40a3a0937c91dc522d2e9db077191d2e4105.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/tables/3bfc226b6b92052c889cf0212e0e40a3a0937c91dc522d2e9db077191d2e4105.jpg)

![51de147b9d1a959c0e3ae9ce58e3a18d50689d5f7449f8a2292337016b3b48bb.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/tables/51de147b9d1a959c0e3ae9ce58e3a18d50689d5f7449f8a2292337016b3b48bb.jpg)

![885cabe094815ec0f94694ff2f73bcf41e0d3e43174f9ab4d89967295f396ba4.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/tables/885cabe094815ec0f94694ff2f73bcf41e0d3e43174f9ab4d89967295f396ba4.jpg)

![958340b25588404a17c031b7cae3248ef5b7fbd0146aa11331968a3ba4c3e184.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/tables/958340b25588404a17c031b7cae3248ef5b7fbd0146aa11331968a3ba4c3e184.jpg)

![a753474dd583cb505aee5c984748aa0208ea142e9c88b0861288dcd4c0f606fc.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/tables/a753474dd583cb505aee5c984748aa0208ea142e9c88b0861288dcd4c0f606fc.jpg)

![dc5f678e3d683f5d583b36878f9db1ee58a7ca6a1affebc1408e329b69b575ab.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/tables/dc5f678e3d683f5d583b36878f9db1ee58a7ca6a1affebc1408e329b69b575ab.jpg)

![f2418ca6bdaaa999d8629edef1eee98d41a3d6abd2294274197a9cd0e21bb883.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/tables/f2418ca6bdaaa999d8629edef1eee98d41a3d6abd2294274197a9cd0e21bb883.jpg)

![f2525311f9618eb4e3c8e436ed49cbbedd2b4572517d691bb7efb6c546d0c6fc.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/tables/f2525311f9618eb4e3c8e436ed49cbbedd2b4572517d691bb7efb6c546d0c6fc.jpg)

![f47dfb104d27f93fd3ee1a87814b8198bed94e015305deb4242556f8edb5222a.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/tables/f47dfb104d27f93fd3ee1a87814b8198bed94e015305deb4242556f8edb5222a.jpg)

![f9e52769e015912b804334d3576f6d434532d147312490f7d823002b6f60deb4.jpg](../icml_results/6218_Dismantling%20the%20Illusion%20of%20Vision-Language-Action%20Models%20Competence%20via%20Explicit%20Distributional%20Shi/tables/f9e52769e015912b804334d3576f6d434532d147312490f7d823002b6f60deb4.jpg)

<a id="paper-6169"></a>

## SFedPO: Streaming Federated Learning with a Prediction Oracle under Temporal Shifts


### Images

![003a3a008904cc09491a51e3cde0691f606e70bd03759892442567ed0b36bd2b.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/images/003a3a008904cc09491a51e3cde0691f606e70bd03759892442567ed0b36bd2b.jpg)

![24bc309b63d5946d6b9e492aae0f6d247e4cf44e5fd43696bdafc7c60838c3e5.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/images/24bc309b63d5946d6b9e492aae0f6d247e4cf44e5fd43696bdafc7c60838c3e5.jpg)

![3e0a15e01817b4582b10695bf86d15c353b748648cbcf7c92ce3bb7d2253ddce.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/images/3e0a15e01817b4582b10695bf86d15c353b748648cbcf7c92ce3bb7d2253ddce.jpg)

![70ce1c40b60d5a46ffa09f4d729f0a1791a2fc43681abac8f51c5c14cb73830d.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/images/70ce1c40b60d5a46ffa09f4d729f0a1791a2fc43681abac8f51c5c14cb73830d.jpg)

![752d82379e0d774b3393603971e984ff00025ebee99c983c6ebedc6badb52c02.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/images/752d82379e0d774b3393603971e984ff00025ebee99c983c6ebedc6badb52c02.jpg)

![85829ea51160efebcde8efeaf3f1ad4587bdd48fe662ebf6fe17e7cb9a6b3392.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/images/85829ea51160efebcde8efeaf3f1ad4587bdd48fe662ebf6fe17e7cb9a6b3392.jpg)

![aa14a0c45f5b9a2cd716c72963c813a21327c498494a1168fa2874418e2c64d0.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/images/aa14a0c45f5b9a2cd716c72963c813a21327c498494a1168fa2874418e2c64d0.jpg)

![c99c12cc19c33326f1f767707feb2e209236df0da783d57dfa0a4825960518cc.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/images/c99c12cc19c33326f1f767707feb2e209236df0da783d57dfa0a4825960518cc.jpg)

![e37cdaacc27b16b7a6a176d0bd8aae0c462986fe70e9bab9edf8d99fb4e248d1.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/images/e37cdaacc27b16b7a6a176d0bd8aae0c462986fe70e9bab9edf8d99fb4e248d1.jpg)

![e8a0b9cd78a2f43f3b55ceff9e61f1db3a388b6ed72c050aceef009e6ca19dca.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/images/e8a0b9cd78a2f43f3b55ceff9e61f1db3a388b6ed72c050aceef009e6ca19dca.jpg)

### Tables

![23102301cbf07a4e60e2411fe85a848134dfee9ec9a2da83e624cd2b0dfc5f87.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/tables/23102301cbf07a4e60e2411fe85a848134dfee9ec9a2da83e624cd2b0dfc5f87.jpg)

![2401f1dd353ddead51a5e13eb5329141cb390bfe6ded929b9b5e05c809df9584.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/tables/2401f1dd353ddead51a5e13eb5329141cb390bfe6ded929b9b5e05c809df9584.jpg)

![651b29ec9b40b164988207a5d80aabc2e3f1cfbf0aff8cb2da751da3b542547e.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/tables/651b29ec9b40b164988207a5d80aabc2e3f1cfbf0aff8cb2da751da3b542547e.jpg)

![99f4717cb6a9b91c39b01cb529715ff8c91e69a9f6dcdec96b86f1bdea33865e.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/tables/99f4717cb6a9b91c39b01cb529715ff8c91e69a9f6dcdec96b86f1bdea33865e.jpg)

![a38ed0deb3ba5e24f1c87d530e4af590757ab616cbec272fbc200dcbb65b8cb1.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/tables/a38ed0deb3ba5e24f1c87d530e4af590757ab616cbec272fbc200dcbb65b8cb1.jpg)

![b225bd017b10a2d734b6a55dd120ba37e7fb67b6b0d3a9ce36a43ea9a7f3a7d9.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/tables/b225bd017b10a2d734b6a55dd120ba37e7fb67b6b0d3a9ce36a43ea9a7f3a7d9.jpg)

![f3dc0d58f778a85789638d365c7a18046fc6f911f27c4eef7cbd7393586c9309.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/tables/f3dc0d58f778a85789638d365c7a18046fc6f911f27c4eef7cbd7393586c9309.jpg)

![f9e5b85030f09c132935e75fe03c47d79a56281739d259c127576e27ffb1f9fd.jpg](../icml_results/6219_SFedPO_%20Streaming%20Federated%20Learning%20with%20a%20Prediction%20Oracle%20under%20Temporal%20Shifts/tables/f9e5b85030f09c132935e75fe03c47d79a56281739d259c127576e27ffb1f9fd.jpg)

<a id="paper-6170"></a>

## Winformer: Transcending Pairwise Similarity for Time-series Generation


### Images

![0acc18b4cb9229e3119f2576f59db93d63bfb60e11bb5dc9554147848b4f1d6f.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/images/0acc18b4cb9229e3119f2576f59db93d63bfb60e11bb5dc9554147848b4f1d6f.jpg)

![3359034549439d2e9737921f066d9b6891f21c970be9efb6e7ba9fd8d074dfc3.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/images/3359034549439d2e9737921f066d9b6891f21c970be9efb6e7ba9fd8d074dfc3.jpg)

![3a5701d4741e02925a781ea944d47538cbf05c4292e38c560efe78797e004ee9.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/images/3a5701d4741e02925a781ea944d47538cbf05c4292e38c560efe78797e004ee9.jpg)

![3c1183bce30a1273f7985ba7a8df62f4fce26dbc586fbb3f52d61759ab5b5236.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/images/3c1183bce30a1273f7985ba7a8df62f4fce26dbc586fbb3f52d61759ab5b5236.jpg)

![4178b64b3906b585dd529ec61730c2e2650950d3ab8d6267f9410202863f37a2.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/images/4178b64b3906b585dd529ec61730c2e2650950d3ab8d6267f9410202863f37a2.jpg)

![4e16ac84d3223002849b549ab9770cb3b2b828303cab5f434557a8827218b900.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/images/4e16ac84d3223002849b549ab9770cb3b2b828303cab5f434557a8827218b900.jpg)

![635d9a791bd5f47074f70970132839e7bc80d3ec7513695d7beed7daa0a1baa7.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/images/635d9a791bd5f47074f70970132839e7bc80d3ec7513695d7beed7daa0a1baa7.jpg)

![67ad9d20b909aac778f4ab2266971954af63718b10acfca0d728a9ce0ab21500.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/images/67ad9d20b909aac778f4ab2266971954af63718b10acfca0d728a9ce0ab21500.jpg)

![6e70723f3b8e7590436752649f5720312f0cab8dc9f2c1f0fa539428611a4ff7.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/images/6e70723f3b8e7590436752649f5720312f0cab8dc9f2c1f0fa539428611a4ff7.jpg)

![899b41fbf024955cff5718de35666fe128584ff95b64bfc3f0aae6ae3cfdacaa.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/images/899b41fbf024955cff5718de35666fe128584ff95b64bfc3f0aae6ae3cfdacaa.jpg)

![8a2d16911233f6d7d06455a35ef645bf86ebcf765fc21eb4ce768536f34bc670.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/images/8a2d16911233f6d7d06455a35ef645bf86ebcf765fc21eb4ce768536f34bc670.jpg)

![9ddfbd88837ee57b37c3d59799b29d64a30a6253f1443bbcbaa40310096bbca5.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/images/9ddfbd88837ee57b37c3d59799b29d64a30a6253f1443bbcbaa40310096bbca5.jpg)

![9f8b493823e08ce71307db6d08d88dce01ecc75924b451dcd2158446b4b113c2.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/images/9f8b493823e08ce71307db6d08d88dce01ecc75924b451dcd2158446b4b113c2.jpg)

![c5a00c121090506a10f340298a72337ceb386f765b88693c4dc6709039a70766.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/images/c5a00c121090506a10f340298a72337ceb386f765b88693c4dc6709039a70766.jpg)

![ff25a6f4a68a1353037affa9b13bfb0ad5c9bae852c829d4027662bffb7c85e6.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/images/ff25a6f4a68a1353037affa9b13bfb0ad5c9bae852c829d4027662bffb7c85e6.jpg)

### Tables

![010c5fc3095e138b89f5884542a3d02e34bee488ab2d704b08d6c72c71a2d5b2.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/tables/010c5fc3095e138b89f5884542a3d02e34bee488ab2d704b08d6c72c71a2d5b2.jpg)

![01d2770e44ca7afa3b801710af80004f0b4f1b3933af765ee047fca2b783563f.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/tables/01d2770e44ca7afa3b801710af80004f0b4f1b3933af765ee047fca2b783563f.jpg)

![109f996c5ddfe0f4250e64d0e1f8bf1550f6d9193af67ae9f6033a3a1d5e205d.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/tables/109f996c5ddfe0f4250e64d0e1f8bf1550f6d9193af67ae9f6033a3a1d5e205d.jpg)

![12e763154ad63dc169b05b0610b5010dd858945e167ea9c3ece7954da4807c29.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/tables/12e763154ad63dc169b05b0610b5010dd858945e167ea9c3ece7954da4807c29.jpg)

![33e900bbd1a9d2c35f0eade6a7b9ea8f2e3b32606293fddae76eb48f2db067e2.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/tables/33e900bbd1a9d2c35f0eade6a7b9ea8f2e3b32606293fddae76eb48f2db067e2.jpg)

![49ab642654cfc2ce9ebd0e1f1741fc3d954ee23ece9b411d4fa0251d5c69c36b.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/tables/49ab642654cfc2ce9ebd0e1f1741fc3d954ee23ece9b411d4fa0251d5c69c36b.jpg)

![9632f82d09a54362066ac752d24adb08fa687d8b7fd8f5700c32b354210600ac.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/tables/9632f82d09a54362066ac752d24adb08fa687d8b7fd8f5700c32b354210600ac.jpg)

![967527fc555c55be09b3e2923f98dc17aed737b334847576c462bdb310f0d209.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/tables/967527fc555c55be09b3e2923f98dc17aed737b334847576c462bdb310f0d209.jpg)

![a1c8b448a64885386f44becccb36e014ec61255cf323d0659d7934e0093a81f6.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/tables/a1c8b448a64885386f44becccb36e014ec61255cf323d0659d7934e0093a81f6.jpg)

![bd9ebcb0891b468a0d2afa839328dc6d9f12e6014a05936c439f54b341ceaaa8.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/tables/bd9ebcb0891b468a0d2afa839328dc6d9f12e6014a05936c439f54b341ceaaa8.jpg)

![df2710983c62ab7684029e41fe934f516498a5c23aefd5750a0c548cf0106de0.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/tables/df2710983c62ab7684029e41fe934f516498a5c23aefd5750a0c548cf0106de0.jpg)

![e123e701cebf6e15dd96f2ecf26c08cde2862b9e11cfb7be2390a92afe0619de.jpg](../icml_results/6220_Winformer_%20Transcending%20Pairwise%20Similarity%20for%20Time-series%20Generation/tables/e123e701cebf6e15dd96f2ecf26c08cde2862b9e11cfb7be2390a92afe0619de.jpg)

<a id="paper-6171"></a>

## Ripple Perturbations Through Structure: Likelihood-Constrained Adversarial Attacks on Heterogeneous Tabular Data

### Images

![27d7ae20e18a549d38aa475678a731dce7ed10d8da71cb40c83c4a10684bde11.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/images/27d7ae20e18a549d38aa475678a731dce7ed10d8da71cb40c83c4a10684bde11.jpg)

![5bf89a3a3b9f724c8cd1d5404421452f125a622ed8dd696e473f36ba92b573c5.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/images/5bf89a3a3b9f724c8cd1d5404421452f125a622ed8dd696e473f36ba92b573c5.jpg)

![83aad9cd536f26c32e2fefb61b52565d45466315dae9fa954c2b2a26e8a98d7d.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/images/83aad9cd536f26c32e2fefb61b52565d45466315dae9fa954c2b2a26e8a98d7d.jpg)

![8a7ac348b7e529ae662500285b297668a00e645e11b080009ecf5ef049b0577f.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/images/8a7ac348b7e529ae662500285b297668a00e645e11b080009ecf5ef049b0577f.jpg)

![b96e27815d54b54f71149a81ee667761f5edbeab60464da1eb40ba30b16e48f9.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/images/b96e27815d54b54f71149a81ee667761f5edbeab60464da1eb40ba30b16e48f9.jpg)

![f5fe0862e94a3fdf31f3dcdae72ef0514bfdd497bcfa8b847b12313327dc0eef.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/images/f5fe0862e94a3fdf31f3dcdae72ef0514bfdd497bcfa8b847b12313327dc0eef.jpg)

### Tables

![2bb39541f8bc00a61b9b7b99cb4bd4d2981be3b0e47944b348666d105d7c398a.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/2bb39541f8bc00a61b9b7b99cb4bd4d2981be3b0e47944b348666d105d7c398a.jpg)

![68b00f06189d0fc201df98b356b11f37208ffb5fefabfa9b93a7880ce26510b3.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/68b00f06189d0fc201df98b356b11f37208ffb5fefabfa9b93a7880ce26510b3.jpg)

![75e8564389d75aac5198fdea4eb17ab7d460f33a2abaea047daa07fd20e99b86.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/75e8564389d75aac5198fdea4eb17ab7d460f33a2abaea047daa07fd20e99b86.jpg)

![80e1e62022e1dc5a3c88568d3542913afd6a02244fcc11745e179cdc06748f38.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/80e1e62022e1dc5a3c88568d3542913afd6a02244fcc11745e179cdc06748f38.jpg)

![9cf1ff36a523149b26a41a633c76d296d6a30f4cf86c15fc813866de7205051a.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/9cf1ff36a523149b26a41a633c76d296d6a30f4cf86c15fc813866de7205051a.jpg)

![9d03468c515e65bb2d10d7f90219f775766e3c7a597a11fb08a62d24a4257688.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/9d03468c515e65bb2d10d7f90219f775766e3c7a597a11fb08a62d24a4257688.jpg)

![9d527c78dcfa0e7d2ed5dc56cad0d5f06030c0dc128ec0ff1aaa50f2ec4f71f2.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/9d527c78dcfa0e7d2ed5dc56cad0d5f06030c0dc128ec0ff1aaa50f2ec4f71f2.jpg)

![b0f1ff95c5eea0b207685f3be306fd48d20f8c450c55eaed39d3e9f8b2010f58.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/b0f1ff95c5eea0b207685f3be306fd48d20f8c450c55eaed39d3e9f8b2010f58.jpg)

![bb8db74b32be8ec3e2f1005fd4f2d06406ab078af6a7a01f64d08d80bbe626ce.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/bb8db74b32be8ec3e2f1005fd4f2d06406ab078af6a7a01f64d08d80bbe626ce.jpg)

![ca085a5a7a2b75310504696eee88c092229a9830b0fe022ce808eeb0cdc71161.jpg](../icml_results/6221_Ripple%20Perturbations%20Through%20Structure_%20Likelihood-Constrained%20Adversarial%20Attacks%20on%20Heterogeneous%20/tables/ca085a5a7a2b75310504696eee88c092229a9830b0fe022ce808eeb0cdc71161.jpg)
