# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 161 of 197

## 目录 (Table of Contents)

1. [Once-for-All: Scalable Simultaneous Forecasting via Equilibrium State Estimation](#paper-5281)
2. [Semantic-Enriched Latent Visual Reasoning](#paper-5282)
3. [When Agents Go Rogue: Activation-Based Detection of Malicious Behaviors in Multi-Agent Systems](#paper-5283)
4. [A Judge-Aware Ranking Framework for Evaluating Large Language Models without Ground Truth](#paper-5284)
5. [Beyond Independence: Learning Correlated Views for Variational Incomplete Multi-View Clustering](#paper-5285)
6. [Finding Differentially Private Second Order Stationary Points in Stochastic Minimax Optimization](#paper-5286)
7. [A Two-Tier Perspective on Inference-Time Parallelism in Multi-Agent LLM Systems](#paper-5287)
8. [BubbleSpec: Turning Long-Tail Bubbles into Speculative Rollout Drafts for Synchronous Reinforcement Learning](#paper-5288)
9. [EigenCache: Rethinking Diffusion Acceleration as Covariance-Optimal Forecasting and Submodular Information Allocation](#paper-5289)
10. [Fingerprinting Pre-trained Encoders under Arbitrary Downstream Fine-Tuning via Adversarial Shifting](#paper-5290)
11. [CoGeoAD: Hierarchical Color-Geometric Fusion with Multi-View Attention for Zero-Shot 3D Anomaly Detection](#paper-5291)
12. [Visual Para-Thinker: Divide-and-Conquer Reasoning for Visual Comprehension](#paper-5292)
13. [TIME: Tensor-Factorized Mixture-of-Experts with Intrinsic Routing for Lifelong Multimodal Knowledge Editing](#paper-5293)
14. [Unlocking Cross-Modal Biosignal Synthesis: A Temporally-Aware VAE-Diffusion Model](#paper-5294)
15. [Non-Adversarial Imitation Learning Provably Free of Compounding Errors: The Value Flow Mechanism](#paper-5295)
16. [Hearing Without Noticing? Attention-Aware Stealthy Black-Box Adversarial Audio Attacks](#paper-5296)
17. [PointDiT: Pixel-Space Diffusion for Monocular Geometry Estimation](#paper-5297)
18. [D-CORE: Incentivizing Task Decomposition in Large Reasoning Models for Complex Tool Use](#paper-5298)
19. [LMCleaner: Efficient and Certified Online Unlearning via Influence Propagation Truncation](#paper-5299)
20. [Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models](#paper-5300)
21. [A Cartesian-3j Framework for Machine Learning Interatomic Potentials](#paper-5301)
22. [MORE: A Multilingual Document Parsing Benchmark and Evaluation](#paper-5302)
23. [Neural-Inspired Modeling of Auditory Selection and Compensation for Audio-Visual Speech Separation](#paper-5303)
24. [Feature Resemblance: Towards a Theoretical Understanding of Analogical Reasoning in Transformers](#paper-5304)
25. [From Internal Diagnosis to External Auditing: A VLM-Driven Paradigm for Data-Free Online Backdoor Defense](#paper-5305)
26. [Resting Neurons, Active Insights: Robustifying Activation Sparsity in LLMs via Spontaneity](#paper-5306)
27. [Theoretical Analysis of Sparse Optimization with Reparameterization, Weight Decay, and Adaptive Learning Rate](#paper-5307)
28. [Weak-to-Strong Generalization via Bregman Bias–Variance Decomposition](#paper-5308)
29. [Doubly Robust Distributionally Robust Offline Contextual Pricing](#paper-5309)
30. [Unlocking Zero-Shot Geospatial Reasoning via Indirect Rewards](#paper-5310)
31. [How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning](#paper-5311)
32. [Towards One-to-Many Temporal Grounding](#paper-5312)
33. [Unlearning Isn’t Deletion: Investigating Reversibility of Machine Unlearning in LLMs](#paper-5313)

---


<a id="paper-5281"></a>

## Once-for-All: Scalable Simultaneous Forecasting via Equilibrium State Estimation

### Images

![02906772cfd59703cb05e7d3b9f0d43499c4f5794c991265bfe220bd082daf47.jpg](../icml_results/5328_A%20Bi-metric%20Framework%20for%20Efficient%20Nearest%20Neighbor%20Search/images/02906772cfd59703cb05e7d3b9f0d43499c4f5794c991265bfe220bd082daf47.jpg)

![0eb3cc054ee7b1ed7e3573a43b2d4c7d19981b83c061922ebd0df501c1c4b7af.jpg](../icml_results/5328_A%20Bi-metric%20Framework%20for%20Efficient%20Nearest%20Neighbor%20Search/images/0eb3cc054ee7b1ed7e3573a43b2d4c7d19981b83c061922ebd0df501c1c4b7af.jpg)

![2280bd3c30c276547603def0ce07923dea74267c213956d11628701e686e5e40.jpg](../icml_results/5328_A%20Bi-metric%20Framework%20for%20Efficient%20Nearest%20Neighbor%20Search/images/2280bd3c30c276547603def0ce07923dea74267c213956d11628701e686e5e40.jpg)

![3b05ac6856d26c499ec9998a4b9fbbf7974acb322aab1ea1b09661871497c9b0.jpg](../icml_results/5328_A%20Bi-metric%20Framework%20for%20Efficient%20Nearest%20Neighbor%20Search/images/3b05ac6856d26c499ec9998a4b9fbbf7974acb322aab1ea1b09661871497c9b0.jpg)

![3dd6b3018b15cb612c85c5c52e49f612d8c3ee231b68e436488df62c37da96d2.jpg](../icml_results/5328_A%20Bi-metric%20Framework%20for%20Efficient%20Nearest%20Neighbor%20Search/images/3dd6b3018b15cb612c85c5c52e49f612d8c3ee231b68e436488df62c37da96d2.jpg)

![4a9adbb00b253d22f67a9875b363e102f2006ff00087b0584f804ef576d6e137.jpg](../icml_results/5328_A%20Bi-metric%20Framework%20for%20Efficient%20Nearest%20Neighbor%20Search/images/4a9adbb00b253d22f67a9875b363e102f2006ff00087b0584f804ef576d6e137.jpg)

![553fe0de0d6b4696f0eb63a6a793e58fbfc58cc350a6caaede9c9e19921c31dc.jpg](../icml_results/5328_A%20Bi-metric%20Framework%20for%20Efficient%20Nearest%20Neighbor%20Search/images/553fe0de0d6b4696f0eb63a6a793e58fbfc58cc350a6caaede9c9e19921c31dc.jpg)

![624d72679c85fcfd47b84f39132da033ac18cb3d9b5861b9436d2ba5e006f9f0.jpg](../icml_results/5328_A%20Bi-metric%20Framework%20for%20Efficient%20Nearest%20Neighbor%20Search/images/624d72679c85fcfd47b84f39132da033ac18cb3d9b5861b9436d2ba5e006f9f0.jpg)

![6496dcb2f07f78f2bb1d25f3f24cd2c89553856b65fbd21b2c230bc80c608029.jpg](../icml_results/5328_A%20Bi-metric%20Framework%20for%20Efficient%20Nearest%20Neighbor%20Search/images/6496dcb2f07f78f2bb1d25f3f24cd2c89553856b65fbd21b2c230bc80c608029.jpg)

![7dd7f908982ebe71bc42c191fda82d4a9bfd4ffcb8d69c68e0ac623b8f6fa00f.jpg](../icml_results/5328_A%20Bi-metric%20Framework%20for%20Efficient%20Nearest%20Neighbor%20Search/images/7dd7f908982ebe71bc42c191fda82d4a9bfd4ffcb8d69c68e0ac623b8f6fa00f.jpg)

![940f0fcbe690c4e5b129cfde4bbb3c4ff3e074889565cd09c06066b45326aa30.jpg](../icml_results/5328_A%20Bi-metric%20Framework%20for%20Efficient%20Nearest%20Neighbor%20Search/images/940f0fcbe690c4e5b129cfde4bbb3c4ff3e074889565cd09c06066b45326aa30.jpg)

![a8f893e0e72968225541d2adbffb44b3525be10c9da12e172a2206c8f3da12ec.jpg](../icml_results/5328_A%20Bi-metric%20Framework%20for%20Efficient%20Nearest%20Neighbor%20Search/images/a8f893e0e72968225541d2adbffb44b3525be10c9da12e172a2206c8f3da12ec.jpg)

![c14a7313eb73415d6f328cf63ea437bf33186721de8a10f46f7f1498b8ab95dc.jpg](../icml_results/5328_A%20Bi-metric%20Framework%20for%20Efficient%20Nearest%20Neighbor%20Search/images/c14a7313eb73415d6f328cf63ea437bf33186721de8a10f46f7f1498b8ab95dc.jpg)

![cb47c87c02ec3de5111bf4d42cd94597894b3aed8f8d14d0a5b305de449a5cea.jpg](../icml_results/5328_A%20Bi-metric%20Framework%20for%20Efficient%20Nearest%20Neighbor%20Search/images/cb47c87c02ec3de5111bf4d42cd94597894b3aed8f8d14d0a5b305de449a5cea.jpg)

![e86384cb43b89be8e624334ff2ef4950ccfbeb977192cb3b2d6e0d1d92d94ba7.jpg](../icml_results/5328_A%20Bi-metric%20Framework%20for%20Efficient%20Nearest%20Neighbor%20Search/images/e86384cb43b89be8e624334ff2ef4950ccfbeb977192cb3b2d6e0d1d92d94ba7.jpg)

### Tables

![2c72b41801fe9de78714a4cd7bf93aa49fcf76cff301acfe7230809c3fec8f31.jpg](../icml_results/5328_A%20Bi-metric%20Framework%20for%20Efficient%20Nearest%20Neighbor%20Search/tables/2c72b41801fe9de78714a4cd7bf93aa49fcf76cff301acfe7230809c3fec8f31.jpg)

![f9f147d653877eb3f37aa48874a790551ebfbdbb981de369c1a85459fd1a0bc8.jpg](../icml_results/5328_A%20Bi-metric%20Framework%20for%20Efficient%20Nearest%20Neighbor%20Search/tables/f9f147d653877eb3f37aa48874a790551ebfbdbb981de369c1a85459fd1a0bc8.jpg)

## Once-for-All: Scalable Simultaneous Forecasting via Equilibrium State Estimation


### Images

![2d8a02d9636dfae4ce5f447a68c54c1ecd5c9a5af42445d98f8e3868437abfd1.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/images/2d8a02d9636dfae4ce5f447a68c54c1ecd5c9a5af42445d98f8e3868437abfd1.jpg)

![62b841239a844d3008a889690e613abcef90a2011e946f7f6373f23b9d3d75a0.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/images/62b841239a844d3008a889690e613abcef90a2011e946f7f6373f23b9d3d75a0.jpg)

![a401448da7bd92fe97c36637e8b59f21c29af43defdb70a86a9191a90a3ed91f.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/images/a401448da7bd92fe97c36637e8b59f21c29af43defdb70a86a9191a90a3ed91f.jpg)

![a4596dfc0bc27a981119331f0be93914bba35093e174945be5cdaedc30c29b21.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/images/a4596dfc0bc27a981119331f0be93914bba35093e174945be5cdaedc30c29b21.jpg)

![dca1f26eb6e0c4b48dfbc01f04070ff1431e8043bd1617fccd118719e69250c9.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/images/dca1f26eb6e0c4b48dfbc01f04070ff1431e8043bd1617fccd118719e69250c9.jpg)

![dff70c4071ae733f48cd601f17fdb6821fb8908fae27ed788149c86d4de6e10c.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/images/dff70c4071ae733f48cd601f17fdb6821fb8908fae27ed788149c86d4de6e10c.jpg)

![f6a10f9c46c9241260e13ef195a370f399876a17eb255c092fd8e23a7c301297.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/images/f6a10f9c46c9241260e13ef195a370f399876a17eb255c092fd8e23a7c301297.jpg)

### Tables

![06f61adeeb93cd86b8d5123656998c8a2e34f9ff786821f16df25ab51322b7a7.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/06f61adeeb93cd86b8d5123656998c8a2e34f9ff786821f16df25ab51322b7a7.jpg)

![0a5685e290a9a0532c7a39a8f23e48cf0b697bb7008d30696e77855f5bd787e6.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/0a5685e290a9a0532c7a39a8f23e48cf0b697bb7008d30696e77855f5bd787e6.jpg)

![0eaad9fe2ff6ef9fe30abe905c6efcb3c766646b249f16bab13b41a56eb69025.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/0eaad9fe2ff6ef9fe30abe905c6efcb3c766646b249f16bab13b41a56eb69025.jpg)

![15956891195cee52988896c47428fe7beccc011c303c68622b1110267c3b4e65.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/15956891195cee52988896c47428fe7beccc011c303c68622b1110267c3b4e65.jpg)

![197d002e430f3d44f6088cfe1d6d647804ae4621f8c00ed1161f91b88a7ec324.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/197d002e430f3d44f6088cfe1d6d647804ae4621f8c00ed1161f91b88a7ec324.jpg)

![1dae5d04634f5728320933f485e2f14b99f32c47d00cb9887a550a5dfaae2f58.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/1dae5d04634f5728320933f485e2f14b99f32c47d00cb9887a550a5dfaae2f58.jpg)

![21854c229d43e1e4ad3449761ae097c60e921dcec772b09dbe553fbf76f6dd4f.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/21854c229d43e1e4ad3449761ae097c60e921dcec772b09dbe553fbf76f6dd4f.jpg)

![2864fb65930ac6eb715f1bb248452dbaba679fbeb7ab831a3e36ee82ed606817.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/2864fb65930ac6eb715f1bb248452dbaba679fbeb7ab831a3e36ee82ed606817.jpg)

![498fa5c2d1e89799b5006cd9e5f6112fb36735f97bd394a4da4bee63cecd453c.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/498fa5c2d1e89799b5006cd9e5f6112fb36735f97bd394a4da4bee63cecd453c.jpg)

![4aa9fca2e22fd8360923afbbd407b41aca299e6e3d85b7c25d6d44b8cab12b17.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/4aa9fca2e22fd8360923afbbd407b41aca299e6e3d85b7c25d6d44b8cab12b17.jpg)

![4b8276c607b2593a0edebba2b3b930be60281bae020119e2b21e9bb6ddf3a26d.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/4b8276c607b2593a0edebba2b3b930be60281bae020119e2b21e9bb6ddf3a26d.jpg)

![4e3f7810bc1ba5587a6339b54f4e46d74e1585706f9cd3551e9279cb1c5a9783.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/4e3f7810bc1ba5587a6339b54f4e46d74e1585706f9cd3551e9279cb1c5a9783.jpg)

![4eb0ce8d1c5c11efb008318b82cdfe7333865d6b7779f2dec0d33debd8f2ef05.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/4eb0ce8d1c5c11efb008318b82cdfe7333865d6b7779f2dec0d33debd8f2ef05.jpg)

![548668a5e8aa0025e09791fbe4035f378ea62f0df4ca73f1afbc672d71da6edf.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/548668a5e8aa0025e09791fbe4035f378ea62f0df4ca73f1afbc672d71da6edf.jpg)

![5f80f63886373a292c7f19fcd8e05911b3f974d3a914e2e330abb213f29f42f0.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/5f80f63886373a292c7f19fcd8e05911b3f974d3a914e2e330abb213f29f42f0.jpg)

![777c80dc179d5ad27c0e5abad6996c48e97660ec6f145e4410e0f784e91154cd.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/777c80dc179d5ad27c0e5abad6996c48e97660ec6f145e4410e0f784e91154cd.jpg)

![7fea603d2c2b0b4e9d23add9bc23c4c8912250ea4cb9c33dbfcbeb47a089f8a3.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/7fea603d2c2b0b4e9d23add9bc23c4c8912250ea4cb9c33dbfcbeb47a089f8a3.jpg)

![802d30443ab0e5b7f00e2bf2b3c9789bef9d295bafb338179c716e040f92a48a.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/802d30443ab0e5b7f00e2bf2b3c9789bef9d295bafb338179c716e040f92a48a.jpg)

![83fe8e7a265687d607694dba257af91561ef4615bee8d0cf81d318b6879747f2.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/83fe8e7a265687d607694dba257af91561ef4615bee8d0cf81d318b6879747f2.jpg)

![8916e66c06e61f63873c3ae654686ec3dd8fac59eb2ed8af4778e202c083909d.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/8916e66c06e61f63873c3ae654686ec3dd8fac59eb2ed8af4778e202c083909d.jpg)

![95dd1cbe5db20c43e2c3a522965885457ed61b995692be15818a66899db22d5e.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/95dd1cbe5db20c43e2c3a522965885457ed61b995692be15818a66899db22d5e.jpg)

![97e2d93594b7ea9751629bef0fb0ba66e8e968878ec61143625b1d9ddabdd8ee.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/97e2d93594b7ea9751629bef0fb0ba66e8e968878ec61143625b1d9ddabdd8ee.jpg)

![97fea55c00c91c7f32a6b60170e74cb2ea1d173117995cdde9dba51ebcec9db5.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/97fea55c00c91c7f32a6b60170e74cb2ea1d173117995cdde9dba51ebcec9db5.jpg)

![9f257ad5368d6570a2e34e037b1dfb42fe0fecdd6e6b8ec56dc501178b93d425.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/9f257ad5368d6570a2e34e037b1dfb42fe0fecdd6e6b8ec56dc501178b93d425.jpg)

![9fabb8982c8fce4c20ea435be00fcf91d15d9f205721b13a84e7c72a674dab38.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/9fabb8982c8fce4c20ea435be00fcf91d15d9f205721b13a84e7c72a674dab38.jpg)

![b64540a4bf63630230727ce1b9e05c515b910c33f8dc2d2a5aaed06d5d068c82.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/b64540a4bf63630230727ce1b9e05c515b910c33f8dc2d2a5aaed06d5d068c82.jpg)

![c506fe69949a16b57909cfa92caf88a5d998218cd1b33fab3899b5d9b41d975d.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/c506fe69949a16b57909cfa92caf88a5d998218cd1b33fab3899b5d9b41d975d.jpg)

![c62666ce85d341fdd318800003d55893490e77b0b0c3d4f4ac33e0f403e7455d.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/c62666ce85d341fdd318800003d55893490e77b0b0c3d4f4ac33e0f403e7455d.jpg)

![ca544462d56f315b1483551ff750045f0e9f716809795b7fad1259852af2452d.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/ca544462d56f315b1483551ff750045f0e9f716809795b7fad1259852af2452d.jpg)

![dc3bef0f1e0cc359a2406b50ed061872e5b54f31404ea25f6145057adb310314.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/dc3bef0f1e0cc359a2406b50ed061872e5b54f31404ea25f6145057adb310314.jpg)

![dd99ec131044cbf45994a1cdffba50552a19cd799f2ef02f9b54fcedf9c4ed22.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/dd99ec131044cbf45994a1cdffba50552a19cd799f2ef02f9b54fcedf9c4ed22.jpg)

![de87bf60eb5082c5b053af43bde1d70ebfbb3b76213301e3b931dd0999053046.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/de87bf60eb5082c5b053af43bde1d70ebfbb3b76213301e3b931dd0999053046.jpg)

![e740e697da1a1d77025f93ef82b87b37cf0049d982f5990ad3e3e4583960d8d6.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/e740e697da1a1d77025f93ef82b87b37cf0049d982f5990ad3e3e4583960d8d6.jpg)

![e7ace55942368acafa57509322e672bcb254c31477934c70a13df5f5eba73291.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/e7ace55942368acafa57509322e672bcb254c31477934c70a13df5f5eba73291.jpg)

![e9b77e290e063fd3ee33bcad0773261044537833a7a5e68bc86c6c4c7f3a5e4f.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/e9b77e290e063fd3ee33bcad0773261044537833a7a5e68bc86c6c4c7f3a5e4f.jpg)

![ecbf8a826ac4ed3f473478ccb6033c02404f88ed4fdfd37d9394151b4935628f.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/ecbf8a826ac4ed3f473478ccb6033c02404f88ed4fdfd37d9394151b4935628f.jpg)

![ecfbb2a01c67ad7ddc4d51b0df746fec4a73d47a3d664b8633d81768ffb48f11.jpg](../icml_results/5329_Once-for-All_%20Scalable%20Simultaneous%20Forecasting%20via%20Equilibrium%20State%20Estimation/tables/ecfbb2a01c67ad7ddc4d51b0df746fec4a73d47a3d664b8633d81768ffb48f11.jpg)

<a id="paper-5282"></a>

## Semantic-Enriched Latent Visual Reasoning


### Images

![0df8f5535b201b50a23bca3324813dedc09ffe10194dc0254477f14397b26ca8.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/0df8f5535b201b50a23bca3324813dedc09ffe10194dc0254477f14397b26ca8.jpg)

![1b083e75334d078f33ffcaa0da65200042406d02e82ab19ee26595fc4a528bfb.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/1b083e75334d078f33ffcaa0da65200042406d02e82ab19ee26595fc4a528bfb.jpg)

![267f522e3e96766d18f66b3b3ffdd07e973df86753e727b3adf56991ecca00fd.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/267f522e3e96766d18f66b3b3ffdd07e973df86753e727b3adf56991ecca00fd.jpg)

![32e515cc3d50360b1c945ea7e3a5e5f30c9b924c82c9d212b3dc2500eccab70d.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/32e515cc3d50360b1c945ea7e3a5e5f30c9b924c82c9d212b3dc2500eccab70d.jpg)

![33ee8509e56e6a0de2bc7fc71c2ff00fdcf9a16783d9ad500dc788eb92ad0b19.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/33ee8509e56e6a0de2bc7fc71c2ff00fdcf9a16783d9ad500dc788eb92ad0b19.jpg)

![38f2e8b56c47e5ccb68e137536148ac2968b816dd4e8b6f67e80ac1ae3d8423b.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/38f2e8b56c47e5ccb68e137536148ac2968b816dd4e8b6f67e80ac1ae3d8423b.jpg)

![504cfbcdd604963a4fbb34620c26679c15ba0307508973d5240e6685b5850077.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/504cfbcdd604963a4fbb34620c26679c15ba0307508973d5240e6685b5850077.jpg)

![50d44757cb4af0e33e548031fc538cd902ecf492689aa917d621c825116202aa.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/50d44757cb4af0e33e548031fc538cd902ecf492689aa917d621c825116202aa.jpg)

![5563cfd0a912fef0a48d2c86ecc64ac9f66e219ae9e35f5c7f5fd9957b2e3029.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/5563cfd0a912fef0a48d2c86ecc64ac9f66e219ae9e35f5c7f5fd9957b2e3029.jpg)

![59a969d0d9195c0820ab74dcd1ee429d93cddd61f96ea4dec3a26d1cedc438dd.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/59a969d0d9195c0820ab74dcd1ee429d93cddd61f96ea4dec3a26d1cedc438dd.jpg)

![60c1b2b0480198233fe1621de453d0f7be854bb5ba01613d47a0681a5aee8143.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/60c1b2b0480198233fe1621de453d0f7be854bb5ba01613d47a0681a5aee8143.jpg)

![7578cda866c6c88c6ae6d905e8920777a7f9a34897d7b81d982fb561494cf9de.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/7578cda866c6c88c6ae6d905e8920777a7f9a34897d7b81d982fb561494cf9de.jpg)

![80803f00e32fe03a2c3b18963ed7a21db4be78d519acf98c9e97cd43b2a69d26.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/80803f00e32fe03a2c3b18963ed7a21db4be78d519acf98c9e97cd43b2a69d26.jpg)

![83a64a1eb3fad4d09b334a80b5eb7b208330142aac6ae5feda226361152756d3.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/83a64a1eb3fad4d09b334a80b5eb7b208330142aac6ae5feda226361152756d3.jpg)

![9b46e24e8b2f2b4f01ce3e272ea2e0b0f1997c8ea946ead656443c687653e76f.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/9b46e24e8b2f2b4f01ce3e272ea2e0b0f1997c8ea946ead656443c687653e76f.jpg)

![b9cd15a063ae68fb56b08321821b8b3929b7956e11c2ac665c87b50fcf5d52eb.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/b9cd15a063ae68fb56b08321821b8b3929b7956e11c2ac665c87b50fcf5d52eb.jpg)

![c80e5c77bd325f19c3bbfd4ce480a33ed737e75b96ed60b03dd99e85f25a7999.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/c80e5c77bd325f19c3bbfd4ce480a33ed737e75b96ed60b03dd99e85f25a7999.jpg)

![d279f43344db7e599c5402b8beb18fde9e28171c94ee4869a00f057d5b9b8fbb.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/d279f43344db7e599c5402b8beb18fde9e28171c94ee4869a00f057d5b9b8fbb.jpg)

![e81c601db548efa9e13f7b2d46f1c2c4fe47b0f6b7efdada85b414ca79fac3e6.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/e81c601db548efa9e13f7b2d46f1c2c4fe47b0f6b7efdada85b414ca79fac3e6.jpg)

![f3297af700ee05e59ab51389170cdfdb842abb3e292d76f813b1b6eb11807e06.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/images/f3297af700ee05e59ab51389170cdfdb842abb3e292d76f813b1b6eb11807e06.jpg)

### Tables

![0133871dad37651985f5c2e84dc75676fdb1d61d185f02ce769a9f8265c02079.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/tables/0133871dad37651985f5c2e84dc75676fdb1d61d185f02ce769a9f8265c02079.jpg)

![372df40471ea04bf5fc1afdb935962f7a4cb6e5cd8b9d14c993f88aaff4f111f.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/tables/372df40471ea04bf5fc1afdb935962f7a4cb6e5cd8b9d14c993f88aaff4f111f.jpg)

![4690e02d409344bbf8ce69c4bbec883635bf9568d6051713c32a203764221771.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/tables/4690e02d409344bbf8ce69c4bbec883635bf9568d6051713c32a203764221771.jpg)

![661e6be4396b28d78bd7428e95630b1583a1d8c51608b0c460cff6723eb41018.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/tables/661e6be4396b28d78bd7428e95630b1583a1d8c51608b0c460cff6723eb41018.jpg)

![8543eda7a215e2441cf319096ce46d6af8ddb00ef2b27fb42431e4428e8f5a78.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/tables/8543eda7a215e2441cf319096ce46d6af8ddb00ef2b27fb42431e4428e8f5a78.jpg)

![99afbd1377ada2ac8e2760736486c9b4ab267515d5345f79d70a38d53d4460e1.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/tables/99afbd1377ada2ac8e2760736486c9b4ab267515d5345f79d70a38d53d4460e1.jpg)

![a60870104ac831df5e30fffff6bfd78a6e63449943c1a6efe99214c4f9cd1db6.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/tables/a60870104ac831df5e30fffff6bfd78a6e63449943c1a6efe99214c4f9cd1db6.jpg)

![b926995c81eef5e6d29e173e0eaddbde94265925c6bc35d73b82068ced121697.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/tables/b926995c81eef5e6d29e173e0eaddbde94265925c6bc35d73b82068ced121697.jpg)

![e97a921afba775c5837df2e5cac55a651c7f59258a56141fbfeb7bcda4756ace.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/tables/e97a921afba775c5837df2e5cac55a651c7f59258a56141fbfeb7bcda4756ace.jpg)

![ff546dbe6a896725a6f75e3fa8515fad381404a03dc35afd3c30b1be3a893efd.jpg](../icml_results/5330_Semantic-Enriched%20Latent%20Visual%20Reasoning/tables/ff546dbe6a896725a6f75e3fa8515fad381404a03dc35afd3c30b1be3a893efd.jpg)

<a id="paper-5283"></a>

## When Agents Go Rogue: Activation-Based Detection of Malicious Behaviors in Multi-Agent Systems


### Images

![22dc402132d23b28616e1abdb2bd5625dff219e8f85457743cc13f8358ee4d88.jpg](../icml_results/5331_When%20Agents%20Go%20Rogue_%20Activation-Based%20Detection%20of%20Malicious%20Behaviors%20in%20Multi-Agent%20Systems/images/22dc402132d23b28616e1abdb2bd5625dff219e8f85457743cc13f8358ee4d88.jpg)

![32b1a8b90418407cc75f9373dbb9a0f10fc264473786029707d0aa1d1fce099b.jpg](../icml_results/5331_When%20Agents%20Go%20Rogue_%20Activation-Based%20Detection%20of%20Malicious%20Behaviors%20in%20Multi-Agent%20Systems/images/32b1a8b90418407cc75f9373dbb9a0f10fc264473786029707d0aa1d1fce099b.jpg)

![5c27a8f74145be731fc4b57c5ac8b79add20a852441f67d791d9208c956fe9ad.jpg](../icml_results/5331_When%20Agents%20Go%20Rogue_%20Activation-Based%20Detection%20of%20Malicious%20Behaviors%20in%20Multi-Agent%20Systems/images/5c27a8f74145be731fc4b57c5ac8b79add20a852441f67d791d9208c956fe9ad.jpg)

![a85827eb7d7360829958f6e9c0eabed5c56c53e349024f3a89670cf723bbe047.jpg](../icml_results/5331_When%20Agents%20Go%20Rogue_%20Activation-Based%20Detection%20of%20Malicious%20Behaviors%20in%20Multi-Agent%20Systems/images/a85827eb7d7360829958f6e9c0eabed5c56c53e349024f3a89670cf723bbe047.jpg)

![a9598d16c3f28078975c62f91c800a5a892f691981a764f46b2290ecfdad2ae5.jpg](../icml_results/5331_When%20Agents%20Go%20Rogue_%20Activation-Based%20Detection%20of%20Malicious%20Behaviors%20in%20Multi-Agent%20Systems/images/a9598d16c3f28078975c62f91c800a5a892f691981a764f46b2290ecfdad2ae5.jpg)

![ab7ffe448f885564a81b7dccc7e0dcd7b01d264df8a5cd50d73e554ed43ca0e6.jpg](../icml_results/5331_When%20Agents%20Go%20Rogue_%20Activation-Based%20Detection%20of%20Malicious%20Behaviors%20in%20Multi-Agent%20Systems/images/ab7ffe448f885564a81b7dccc7e0dcd7b01d264df8a5cd50d73e554ed43ca0e6.jpg)

### Tables

![2fdfd003289024bd5a28b4ffeab99df3c1e719a92b0315c9906f1f9feffe7233.jpg](../icml_results/5331_When%20Agents%20Go%20Rogue_%20Activation-Based%20Detection%20of%20Malicious%20Behaviors%20in%20Multi-Agent%20Systems/tables/2fdfd003289024bd5a28b4ffeab99df3c1e719a92b0315c9906f1f9feffe7233.jpg)

![611d60145bd4c40c5ed59c8fcb650d0b47d64b8218650ec7cef432dfdc7b893d.jpg](../icml_results/5331_When%20Agents%20Go%20Rogue_%20Activation-Based%20Detection%20of%20Malicious%20Behaviors%20in%20Multi-Agent%20Systems/tables/611d60145bd4c40c5ed59c8fcb650d0b47d64b8218650ec7cef432dfdc7b893d.jpg)

![8414e8bbecde6c4ae2a8c3005d5092341ed7edb8686dbe2badc5daf4ebbf62b3.jpg](../icml_results/5331_When%20Agents%20Go%20Rogue_%20Activation-Based%20Detection%20of%20Malicious%20Behaviors%20in%20Multi-Agent%20Systems/tables/8414e8bbecde6c4ae2a8c3005d5092341ed7edb8686dbe2badc5daf4ebbf62b3.jpg)

![8c71501365f3700644e9acd25a9a24392cba305a2ca1653ad9b46fafde383643.jpg](../icml_results/5331_When%20Agents%20Go%20Rogue_%20Activation-Based%20Detection%20of%20Malicious%20Behaviors%20in%20Multi-Agent%20Systems/tables/8c71501365f3700644e9acd25a9a24392cba305a2ca1653ad9b46fafde383643.jpg)

![941e8e28d88f2dba78b78c35cf521bb9f8ec6e57fbfa5a0d54312346dbeec9f8.jpg](../icml_results/5331_When%20Agents%20Go%20Rogue_%20Activation-Based%20Detection%20of%20Malicious%20Behaviors%20in%20Multi-Agent%20Systems/tables/941e8e28d88f2dba78b78c35cf521bb9f8ec6e57fbfa5a0d54312346dbeec9f8.jpg)

![9a7b52748e98b791dfd8a58ff932a1a2eee96c88ad87b80df49703ac21f29ced.jpg](../icml_results/5331_When%20Agents%20Go%20Rogue_%20Activation-Based%20Detection%20of%20Malicious%20Behaviors%20in%20Multi-Agent%20Systems/tables/9a7b52748e98b791dfd8a58ff932a1a2eee96c88ad87b80df49703ac21f29ced.jpg)

![b91b84134c3c4734dd06126367069c5f43907a89ebf0ae682981ca5e3d5f4c5a.jpg](../icml_results/5331_When%20Agents%20Go%20Rogue_%20Activation-Based%20Detection%20of%20Malicious%20Behaviors%20in%20Multi-Agent%20Systems/tables/b91b84134c3c4734dd06126367069c5f43907a89ebf0ae682981ca5e3d5f4c5a.jpg)

![c72165bf12e9024a2e31e6df92384685fc77eda84df68144523df9a2df83448f.jpg](../icml_results/5331_When%20Agents%20Go%20Rogue_%20Activation-Based%20Detection%20of%20Malicious%20Behaviors%20in%20Multi-Agent%20Systems/tables/c72165bf12e9024a2e31e6df92384685fc77eda84df68144523df9a2df83448f.jpg)

<a id="paper-5284"></a>

## A Judge-Aware Ranking Framework for Evaluating Large Language Models without Ground Truth


### Images

![06c5021a87db77286d8abd063979e5b51b810946bf4c3aecf83b7e8e6adf8ad4.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/images/06c5021a87db77286d8abd063979e5b51b810946bf4c3aecf83b7e8e6adf8ad4.jpg)

![1708e3b40f9678e02c64369cdae8944b713b32f7afc21f8771958e5b8e16b149.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/images/1708e3b40f9678e02c64369cdae8944b713b32f7afc21f8771958e5b8e16b149.jpg)

![2dcc723e1a18ecfc306cbca55204510593374f215c611c68affd3650d9cadfd6.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/images/2dcc723e1a18ecfc306cbca55204510593374f215c611c68affd3650d9cadfd6.jpg)

![5a94d657ada0776571f65bfc1e33914fa38ff025efbe0c40a6abfb992c2b5fe7.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/images/5a94d657ada0776571f65bfc1e33914fa38ff025efbe0c40a6abfb992c2b5fe7.jpg)

![5e30ade4ec1e61698b6eb956a00a3bcaf62058cdf3d124954a66158b761b46f4.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/images/5e30ade4ec1e61698b6eb956a00a3bcaf62058cdf3d124954a66158b761b46f4.jpg)

![8104d30edc9081e6e5bd916ffa5001e50dfad377505fb8854f760e8cfc591c2c.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/images/8104d30edc9081e6e5bd916ffa5001e50dfad377505fb8854f760e8cfc591c2c.jpg)

![854ac40a9212d253045eb5cb3c3a6eb14d1bd7ebd73c1eb35c8115609b13611d.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/images/854ac40a9212d253045eb5cb3c3a6eb14d1bd7ebd73c1eb35c8115609b13611d.jpg)

![8751a4b3d7db374d57c7d05cb990863db256d67a50d79c5ccca6bc2b6ccf87bc.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/images/8751a4b3d7db374d57c7d05cb990863db256d67a50d79c5ccca6bc2b6ccf87bc.jpg)

![90793da80404b3e0de348f4d9f05e651390ed52b4448402da1c2459a1c652c85.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/images/90793da80404b3e0de348f4d9f05e651390ed52b4448402da1c2459a1c652c85.jpg)

![e84a50a35e265afdddaaec953a10a7bc8efaadce04ce0ebe9373c516fe723114.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/images/e84a50a35e265afdddaaec953a10a7bc8efaadce04ce0ebe9373c516fe723114.jpg)

![e9e7f2c408d13d97557cb2380ad7c53b65ce4223face2cf8ecb2fcd3f7ff7759.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/images/e9e7f2c408d13d97557cb2380ad7c53b65ce4223face2cf8ecb2fcd3f7ff7759.jpg)

![ec4eaba71740ca6d27a075061332554716f6e7d71ec855a1563e5689471e4059.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/images/ec4eaba71740ca6d27a075061332554716f6e7d71ec855a1563e5689471e4059.jpg)

![f721b6bd7d9b9601a0c05da846c31e9fb64d96ff1d2b445383c0cae6f90d81d2.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/images/f721b6bd7d9b9601a0c05da846c31e9fb64d96ff1d2b445383c0cae6f90d81d2.jpg)

### Tables

![1f96dce9949dd42766c20eeb3c32e5277d8a4a88f987c00abb499517c5df54c2.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/tables/1f96dce9949dd42766c20eeb3c32e5277d8a4a88f987c00abb499517c5df54c2.jpg)

![36cadd72b8fef2a2c73d4e04d9cda9d2b2b66252382eca19711a6297ec840987.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/tables/36cadd72b8fef2a2c73d4e04d9cda9d2b2b66252382eca19711a6297ec840987.jpg)

![37b80d9e423aaab5f946d6dbfc030eda34474c087c21ac5611ab9bd2fbea3d03.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/tables/37b80d9e423aaab5f946d6dbfc030eda34474c087c21ac5611ab9bd2fbea3d03.jpg)

![54a97ce2a2185f4d385a8e60256bf722c842ad5bb3965c556d07b7880fd2d594.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/tables/54a97ce2a2185f4d385a8e60256bf722c842ad5bb3965c556d07b7880fd2d594.jpg)

![630fa2554fcdd25ecf29bb22b8861e1f09d3860f869016cb7674ba136abce95a.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/tables/630fa2554fcdd25ecf29bb22b8861e1f09d3860f869016cb7674ba136abce95a.jpg)

![8f33822f3810d55f6d942ee7d7f0024a2070c4f9a741066235e731dac41d63ea.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/tables/8f33822f3810d55f6d942ee7d7f0024a2070c4f9a741066235e731dac41d63ea.jpg)

![aa5c60755462fb40454d1e02d59078fb4dbce1025a04a962e3436ec40b5e51cf.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/tables/aa5c60755462fb40454d1e02d59078fb4dbce1025a04a962e3436ec40b5e51cf.jpg)

![ba076b9876bc655441ba55f0c321a0475c59e97911845b3859407f297e56f663.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/tables/ba076b9876bc655441ba55f0c321a0475c59e97911845b3859407f297e56f663.jpg)

![bccdbdb93664b53d9cd021a9d37cd756392150b2005241e0693cbe4e35ac4b75.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/tables/bccdbdb93664b53d9cd021a9d37cd756392150b2005241e0693cbe4e35ac4b75.jpg)

![d0871ad54f34fee7a671982c234fa85a47c7f8e6a68437848cba37e951d8b876.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/tables/d0871ad54f34fee7a671982c234fa85a47c7f8e6a68437848cba37e951d8b876.jpg)

![deb0d7e19ab0bb2643f7e86037b46c683830ac8ad6e8d9e359001d155d1a0d6d.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/tables/deb0d7e19ab0bb2643f7e86037b46c683830ac8ad6e8d9e359001d155d1a0d6d.jpg)

![e82a02409c7f331c3bb463aed012bc45f17c238f765072bd075cf077c564a6d4.jpg](../icml_results/5332_A%20Judge-Aware%20Ranking%20Framework%20for%20Evaluating%20Large%20Language%20Models%20without%20Ground%20Truth/tables/e82a02409c7f331c3bb463aed012bc45f17c238f765072bd075cf077c564a6d4.jpg)

<a id="paper-5285"></a>

## Beyond Independence: Learning Correlated Views for Variational Incomplete Multi-View Clustering


### Images

![15f30a9806b33ff1fc2065cd59b7d3de3a951c6c804b159cce0f6b5dafedee9e.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/images/15f30a9806b33ff1fc2065cd59b7d3de3a951c6c804b159cce0f6b5dafedee9e.jpg)

![1c403464f780eacc4bcdc566027621fda6a10320a2fae177254ee11d1ebcb2b6.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/images/1c403464f780eacc4bcdc566027621fda6a10320a2fae177254ee11d1ebcb2b6.jpg)

![2969448f920185ea805a249cb366e065791914a402699ddcd6c61492116931d5.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/images/2969448f920185ea805a249cb366e065791914a402699ddcd6c61492116931d5.jpg)

![56137804dc5a95a727e5d59a98ee9ab858ace4566d183f4de123217f7395478e.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/images/56137804dc5a95a727e5d59a98ee9ab858ace4566d183f4de123217f7395478e.jpg)

![65a19f3a422a7f3bb7c765bdc9610ba9d5b00920037b85e6a452fec4c2c14d84.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/images/65a19f3a422a7f3bb7c765bdc9610ba9d5b00920037b85e6a452fec4c2c14d84.jpg)

![9ae60386352dc40641c6984f4b0ef6eb3f386829c4a778db6c305b57b39f2e0e.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/images/9ae60386352dc40641c6984f4b0ef6eb3f386829c4a778db6c305b57b39f2e0e.jpg)

![9d6fd7cd83648bce21426b853c77fea6e4e7c523f46b59895d9de41aa3c0e6de.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/images/9d6fd7cd83648bce21426b853c77fea6e4e7c523f46b59895d9de41aa3c0e6de.jpg)

![b1fe38486c7f5c620f6a59b899991ee07e93b5185bb8545e6b55549f6031c54e.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/images/b1fe38486c7f5c620f6a59b899991ee07e93b5185bb8545e6b55549f6031c54e.jpg)

![be8441098377050b39ee441b9e61c69a89d55955bacf0cf26cbded3eb17246f8.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/images/be8441098377050b39ee441b9e61c69a89d55955bacf0cf26cbded3eb17246f8.jpg)

![cda8506067a417a7c340d50221d19074264f9cefd8ee0e46f7a271c032a021e1.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/images/cda8506067a417a7c340d50221d19074264f9cefd8ee0e46f7a271c032a021e1.jpg)

![d0575c8c2eda7987afd1366375f910f79e8c0f9237f733b3f6056d56767f7e18.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/images/d0575c8c2eda7987afd1366375f910f79e8c0f9237f733b3f6056d56767f7e18.jpg)

![d4cf39a3b6638cd942050400a29bfcbfd016f775b2fd20d9da19b9cf7d789b56.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/images/d4cf39a3b6638cd942050400a29bfcbfd016f775b2fd20d9da19b9cf7d789b56.jpg)

![e4aa0ae766a83466646192ba8cc34dee67316c02c05112cc5dd81370652e93c2.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/images/e4aa0ae766a83466646192ba8cc34dee67316c02c05112cc5dd81370652e93c2.jpg)

### Tables

![2048d08d9f1104523b1e61547551e00630840baf3ae2329891c4d89bce02e63e.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/tables/2048d08d9f1104523b1e61547551e00630840baf3ae2329891c4d89bce02e63e.jpg)

![2922d8841660177adbaeb979ac2166514442df15e25fae6aae0a503104839df3.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/tables/2922d8841660177adbaeb979ac2166514442df15e25fae6aae0a503104839df3.jpg)

![6c4b603f6a2eb95f57e21f6fd12fcee8d38daa47c6fe03ac0ba8403a8936b2a1.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/tables/6c4b603f6a2eb95f57e21f6fd12fcee8d38daa47c6fe03ac0ba8403a8936b2a1.jpg)

![7841ce9540218543d7df84ea7ea436b8cf961b84a8e433b848bd72f8cc6beb65.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/tables/7841ce9540218543d7df84ea7ea436b8cf961b84a8e433b848bd72f8cc6beb65.jpg)

![7b9293d134c14ff01f3800a008f5ada54c8fcf3ccb697f41fe315a6536091d90.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/tables/7b9293d134c14ff01f3800a008f5ada54c8fcf3ccb697f41fe315a6536091d90.jpg)

![94dfbc17036ae19e0558b5e62c249f34ff18adc17cef93e28fb13c2820138975.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/tables/94dfbc17036ae19e0558b5e62c249f34ff18adc17cef93e28fb13c2820138975.jpg)

![b2eb71d061df6bcf159bfa7988b65cbc071dcb19a6863a39005a804613a2448f.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/tables/b2eb71d061df6bcf159bfa7988b65cbc071dcb19a6863a39005a804613a2448f.jpg)

![c510a0a79cde529646f8273536bdd874e5715234038b99e8d366a4b1966cf2e0.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/tables/c510a0a79cde529646f8273536bdd874e5715234038b99e8d366a4b1966cf2e0.jpg)

![daed82038777eba2d0eb47d85e3e2747de8415b92b89a6ae091f108744376baf.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/tables/daed82038777eba2d0eb47d85e3e2747de8415b92b89a6ae091f108744376baf.jpg)

![dfe44b71d2f4493671f773bb9d77e53a1999338cf50c6dbbdeec64421874afb1.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/tables/dfe44b71d2f4493671f773bb9d77e53a1999338cf50c6dbbdeec64421874afb1.jpg)

![edcd4179cc21fb455d0f73bfc187578877981b0c48a621055619523c2a199edf.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/tables/edcd4179cc21fb455d0f73bfc187578877981b0c48a621055619523c2a199edf.jpg)

![f5f46d4a3bf42d6acd0f093e87af090edfe2307bf86d004a57c9e458d98175db.jpg](../icml_results/5333_Beyond%20Independence_%20Learning%20Correlated%20Views%20for%20Variational%20Incomplete%20Multi-View%20Clustering/tables/f5f46d4a3bf42d6acd0f093e87af090edfe2307bf86d004a57c9e458d98175db.jpg)

<a id="paper-5286"></a>

## Finding Differentially Private Second Order Stationary Points in Stochastic Minimax Optimization


### Images

![5988ea35eda7a01d47a79cb880481497ad737c07da6a9ea196d6ec3b0fc9c49b.jpg](../icml_results/5334_Finding%20Differentially%20Private%20Second%20Order%20Stationary%20Points%20in%20Stochastic%20Minimax%20Optimization/images/5988ea35eda7a01d47a79cb880481497ad737c07da6a9ea196d6ec3b0fc9c49b.jpg)

### Tables

![08c407b1bdf3dd8fe3c45143d847454a33375bc5fc98550c71dda19b7875b552.jpg](../icml_results/5334_Finding%20Differentially%20Private%20Second%20Order%20Stationary%20Points%20in%20Stochastic%20Minimax%20Optimization/tables/08c407b1bdf3dd8fe3c45143d847454a33375bc5fc98550c71dda19b7875b552.jpg)

![3310af81234c80f77af8cd56f423264528ba245badbf1069a21a9e61b2aa4556.jpg](../icml_results/5334_Finding%20Differentially%20Private%20Second%20Order%20Stationary%20Points%20in%20Stochastic%20Minimax%20Optimization/tables/3310af81234c80f77af8cd56f423264528ba245badbf1069a21a9e61b2aa4556.jpg)

<a id="paper-5287"></a>

## A Two-Tier Perspective on Inference-Time Parallelism in Multi-Agent LLM Systems


### Images

![8a7345c3f682d37df76510e539fa35956fad92e533b6fc7a1d958deee7a6f5e7.jpg](../icml_results/5335_A%20Two-Tier%20Perspective%20on%20Inference-Time%20Parallelism%20in%20Multi-Agent%20LLM%20Systems/images/8a7345c3f682d37df76510e539fa35956fad92e533b6fc7a1d958deee7a6f5e7.jpg)

![a32b3bc0ec5e8a4a2d480d34a41fc3f442540afd2341d7c1bf66e5e5e2a34a89.jpg](../icml_results/5335_A%20Two-Tier%20Perspective%20on%20Inference-Time%20Parallelism%20in%20Multi-Agent%20LLM%20Systems/images/a32b3bc0ec5e8a4a2d480d34a41fc3f442540afd2341d7c1bf66e5e5e2a34a89.jpg)

![a9abbcb267c1c5065a92358e82c98fed34430eb8bf24b56b314a580177e5a2f4.jpg](../icml_results/5335_A%20Two-Tier%20Perspective%20on%20Inference-Time%20Parallelism%20in%20Multi-Agent%20LLM%20Systems/images/a9abbcb267c1c5065a92358e82c98fed34430eb8bf24b56b314a580177e5a2f4.jpg)

### Tables

![01a114dc98b77b4dd07874904ba1fcdc21ef3dc2fab08d3851287678ed81a180.jpg](../icml_results/5335_A%20Two-Tier%20Perspective%20on%20Inference-Time%20Parallelism%20in%20Multi-Agent%20LLM%20Systems/tables/01a114dc98b77b4dd07874904ba1fcdc21ef3dc2fab08d3851287678ed81a180.jpg)

![0bd1503c2a937f0f54e0fc391d996f700237aa1c1700453641b388f4e39c3b3e.jpg](../icml_results/5335_A%20Two-Tier%20Perspective%20on%20Inference-Time%20Parallelism%20in%20Multi-Agent%20LLM%20Systems/tables/0bd1503c2a937f0f54e0fc391d996f700237aa1c1700453641b388f4e39c3b3e.jpg)

![2eab9bc7fb00af26f95f93cca7f1cd971f63d54de5cfa73f4fce7be9eb68ca23.jpg](../icml_results/5335_A%20Two-Tier%20Perspective%20on%20Inference-Time%20Parallelism%20in%20Multi-Agent%20LLM%20Systems/tables/2eab9bc7fb00af26f95f93cca7f1cd971f63d54de5cfa73f4fce7be9eb68ca23.jpg)

![572645d267487e96fe6272f1aad604539157a39573abe3ec4ca3f3318f22e386.jpg](../icml_results/5335_A%20Two-Tier%20Perspective%20on%20Inference-Time%20Parallelism%20in%20Multi-Agent%20LLM%20Systems/tables/572645d267487e96fe6272f1aad604539157a39573abe3ec4ca3f3318f22e386.jpg)

![67211dd0ce4630e0b3e83d8ba990192a637bc6302824b2db41304c04a12b52b3.jpg](../icml_results/5335_A%20Two-Tier%20Perspective%20on%20Inference-Time%20Parallelism%20in%20Multi-Agent%20LLM%20Systems/tables/67211dd0ce4630e0b3e83d8ba990192a637bc6302824b2db41304c04a12b52b3.jpg)

![674ddfa34c7a84e7620bf4f489dcf3577c834677273953ff2a7ee261ee41b5f9.jpg](../icml_results/5335_A%20Two-Tier%20Perspective%20on%20Inference-Time%20Parallelism%20in%20Multi-Agent%20LLM%20Systems/tables/674ddfa34c7a84e7620bf4f489dcf3577c834677273953ff2a7ee261ee41b5f9.jpg)

![8d00c3b4bb61a3a2b44a4d70398aab80cb2bca442844c43541384c587afe3771.jpg](../icml_results/5335_A%20Two-Tier%20Perspective%20on%20Inference-Time%20Parallelism%20in%20Multi-Agent%20LLM%20Systems/tables/8d00c3b4bb61a3a2b44a4d70398aab80cb2bca442844c43541384c587afe3771.jpg)

![bc9dfeff2ae495173651fcdb65310b0fec5616a119e66896b9a9b5ebda6d11ef.jpg](../icml_results/5335_A%20Two-Tier%20Perspective%20on%20Inference-Time%20Parallelism%20in%20Multi-Agent%20LLM%20Systems/tables/bc9dfeff2ae495173651fcdb65310b0fec5616a119e66896b9a9b5ebda6d11ef.jpg)

![d47f714d1703da13047433a25b689d8c02dba3ad053225e4d70ca336c1d10849.jpg](../icml_results/5335_A%20Two-Tier%20Perspective%20on%20Inference-Time%20Parallelism%20in%20Multi-Agent%20LLM%20Systems/tables/d47f714d1703da13047433a25b689d8c02dba3ad053225e4d70ca336c1d10849.jpg)

![e487f0016843ebea105eff3c436db183b21d570fdefaddd89bf491a26cbbb219.jpg](../icml_results/5335_A%20Two-Tier%20Perspective%20on%20Inference-Time%20Parallelism%20in%20Multi-Agent%20LLM%20Systems/tables/e487f0016843ebea105eff3c436db183b21d570fdefaddd89bf491a26cbbb219.jpg)

![e6700767117cfbcc0cbb7b9162b35f6fecf8cda381bce7f8032f4cee56e46c67.jpg](../icml_results/5335_A%20Two-Tier%20Perspective%20on%20Inference-Time%20Parallelism%20in%20Multi-Agent%20LLM%20Systems/tables/e6700767117cfbcc0cbb7b9162b35f6fecf8cda381bce7f8032f4cee56e46c67.jpg)

![e75bbc973363d0f82f87a6eb42f99e68d0390cf2e9f2d2e6bcc1e28fc53a9902.jpg](../icml_results/5335_A%20Two-Tier%20Perspective%20on%20Inference-Time%20Parallelism%20in%20Multi-Agent%20LLM%20Systems/tables/e75bbc973363d0f82f87a6eb42f99e68d0390cf2e9f2d2e6bcc1e28fc53a9902.jpg)

![e879697389e4a04a746cc6056bd916930f8669bc6c9901678c12285aa57a109e.jpg](../icml_results/5335_A%20Two-Tier%20Perspective%20on%20Inference-Time%20Parallelism%20in%20Multi-Agent%20LLM%20Systems/tables/e879697389e4a04a746cc6056bd916930f8669bc6c9901678c12285aa57a109e.jpg)

![f42bf164a7c5c3288654ee9901257db52e1579cf22adca541cc2583cb248d0dc.jpg](../icml_results/5335_A%20Two-Tier%20Perspective%20on%20Inference-Time%20Parallelism%20in%20Multi-Agent%20LLM%20Systems/tables/f42bf164a7c5c3288654ee9901257db52e1579cf22adca541cc2583cb248d0dc.jpg)

<a id="paper-5288"></a>

## BubbleSpec: Turning Long-Tail Bubbles into Speculative Rollout Drafts for Synchronous Reinforcement Learning


### Images

![0132dbfa5a4884a456da3b0cfbdcd2f2fb1d2ee946986017ecf28e5bdc897d02.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/images/0132dbfa5a4884a456da3b0cfbdcd2f2fb1d2ee946986017ecf28e5bdc897d02.jpg)

![1f539d21fdbaaed965dc17d6fdbd5d1d197945a25ada5e0681e29de8f17da64e.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/images/1f539d21fdbaaed965dc17d6fdbd5d1d197945a25ada5e0681e29de8f17da64e.jpg)

![3d1247385b3035028099f47a5d4e17e3dcfaf88984dcdfc182ce0406080c284c.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/images/3d1247385b3035028099f47a5d4e17e3dcfaf88984dcdfc182ce0406080c284c.jpg)

![7603a3fb2eb04937f1a8b37cf3d71b9f9f61c8008da44c6087f20fe295e4fc67.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/images/7603a3fb2eb04937f1a8b37cf3d71b9f9f61c8008da44c6087f20fe295e4fc67.jpg)

![89e135dc373d98116160750dd3affcc97d0be30354d35ed3fe57d35b94a36322.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/images/89e135dc373d98116160750dd3affcc97d0be30354d35ed3fe57d35b94a36322.jpg)

![8e3e4a1f5d8df277adb6d98da7d60995e3fdb38c6908ff3ae2d17cabb17d6f50.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/images/8e3e4a1f5d8df277adb6d98da7d60995e3fdb38c6908ff3ae2d17cabb17d6f50.jpg)

![986a3cbdbb2c5105f34d5b1ec97d90820cc4d587255bec4f833e27c27db78c0a.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/images/986a3cbdbb2c5105f34d5b1ec97d90820cc4d587255bec4f833e27c27db78c0a.jpg)

![e1349175d8c9ada6d280c8ca62df97e145f5f5373aab5318766b1ba25261e58d.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/images/e1349175d8c9ada6d280c8ca62df97e145f5f5373aab5318766b1ba25261e58d.jpg)

### Tables

![00ed1e77b21e80d853b74c3d5010c688dd366bed0da7f8a47d573811c0626148.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/tables/00ed1e77b21e80d853b74c3d5010c688dd366bed0da7f8a47d573811c0626148.jpg)

![0e23e3d99f563888dbfa17bdf4ae4871dbc49ad2b9c1516f3004c95508a48992.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/tables/0e23e3d99f563888dbfa17bdf4ae4871dbc49ad2b9c1516f3004c95508a48992.jpg)

![1d4db702637766a990fb2315e9f79d83514fb72a2c93856d506c11ab8156fc9e.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/tables/1d4db702637766a990fb2315e9f79d83514fb72a2c93856d506c11ab8156fc9e.jpg)

![2bba9a4fc4a9cff49a179f163969b23e407e2347e5c2c63371e233a52e311205.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/tables/2bba9a4fc4a9cff49a179f163969b23e407e2347e5c2c63371e233a52e311205.jpg)

![3e6b3672262bfdeb0a1ae7d0542a04a7e76476326e5b4be024c53d424618125f.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/tables/3e6b3672262bfdeb0a1ae7d0542a04a7e76476326e5b4be024c53d424618125f.jpg)

![4f98497165781f952b924639c9f8ba9255702a19578e53ffb245f5907c10e105.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/tables/4f98497165781f952b924639c9f8ba9255702a19578e53ffb245f5907c10e105.jpg)

![5a9cf90b247006af490b02b03172eeeae41feb1aa1fa98bb4e7db84c2ee472cd.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/tables/5a9cf90b247006af490b02b03172eeeae41feb1aa1fa98bb4e7db84c2ee472cd.jpg)

![5aec3b0740813e79a61d5920a600fa3948dd703f7835bc97c5d5b3c5eb5bc0ff.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/tables/5aec3b0740813e79a61d5920a600fa3948dd703f7835bc97c5d5b3c5eb5bc0ff.jpg)

![b4daa6637d3a5840551ea63184685070eadfe35bc0cfd7846c5f0b002e4e1560.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/tables/b4daa6637d3a5840551ea63184685070eadfe35bc0cfd7846c5f0b002e4e1560.jpg)

![d8e327d72252138caa686d5c37b9fecec5f2d8b4cef6cd1e7d938d6a2608f931.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/tables/d8e327d72252138caa686d5c37b9fecec5f2d8b4cef6cd1e7d938d6a2608f931.jpg)

![eb22b58481910fdd35466b0e7accbbf01bea125e06a0f8cee9db1aed2cd4683b.jpg](../icml_results/5336_BubbleSpec_%20Turning%20Long-Tail%20Bubbles%20into%20Speculative%20Rollout%20Drafts%20for%20Synchronous%20Reinforcement%20/tables/eb22b58481910fdd35466b0e7accbbf01bea125e06a0f8cee9db1aed2cd4683b.jpg)

<a id="paper-5289"></a>

## EigenCache: Rethinking Diffusion Acceleration as Covariance-Optimal Forecasting and Submodular Information Allocation


### Images

![2d5d3f6b4c84f47130026a56e412d4874e2f01aec0d3dff4d2f1b41f5155e82c.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/images/2d5d3f6b4c84f47130026a56e412d4874e2f01aec0d3dff4d2f1b41f5155e82c.jpg)

![4e06f4e5913a9744157a40c169aca87837beed41e825e8c593dd379b44a21a72.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/images/4e06f4e5913a9744157a40c169aca87837beed41e825e8c593dd379b44a21a72.jpg)

![6ee4bf47e75062547a166f39511b94fc19387811fbd3bcffca3ce915094a7db5.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/images/6ee4bf47e75062547a166f39511b94fc19387811fbd3bcffca3ce915094a7db5.jpg)

![79283f2c5efadf51ff3f76bf8f361c13685ecc3076ee2ab6f2501f872fe984ea.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/images/79283f2c5efadf51ff3f76bf8f361c13685ecc3076ee2ab6f2501f872fe984ea.jpg)

![cbb3740244b9feec9637245fdfc231df4a6ce931cb663c825aacfcfa58318177.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/images/cbb3740244b9feec9637245fdfc231df4a6ce931cb663c825aacfcfa58318177.jpg)

![dd2f0e08c3474ce839cca8cc77de94bca71c68439fdadbc0072d3aa72a4907e3.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/images/dd2f0e08c3474ce839cca8cc77de94bca71c68439fdadbc0072d3aa72a4907e3.jpg)

![f23a7952e7caa790312c4c6d4f9afa7a222efce2ec6f2fe3a6353211a842cdc6.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/images/f23a7952e7caa790312c4c6d4f9afa7a222efce2ec6f2fe3a6353211a842cdc6.jpg)

### Tables

![07f2ab99044515ed888b0d277d96e3e573cefbbea041f6435223fbbaad6085d3.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/tables/07f2ab99044515ed888b0d277d96e3e573cefbbea041f6435223fbbaad6085d3.jpg)

![10dc5fb5c436fa5b5253f83310e58dbb549d7784198bb2e60273547f0cb800e3.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/tables/10dc5fb5c436fa5b5253f83310e58dbb549d7784198bb2e60273547f0cb800e3.jpg)

![156f07d9ada54324e837b24fee4db3157e1abef5d50c637450d7336ccdf8380c.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/tables/156f07d9ada54324e837b24fee4db3157e1abef5d50c637450d7336ccdf8380c.jpg)

![2302fc20ae3c966ae0e19dba8439bd0c898fde1e02d9426dad9d6414766d0954.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/tables/2302fc20ae3c966ae0e19dba8439bd0c898fde1e02d9426dad9d6414766d0954.jpg)

![2901089e06648469704b7ecf051c0294a2501061d26f561f571d72ee35397d6a.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/tables/2901089e06648469704b7ecf051c0294a2501061d26f561f571d72ee35397d6a.jpg)

![3abe26ffaf020f8d29af6676d541714b606c224fa77e1690551e70ea891c864f.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/tables/3abe26ffaf020f8d29af6676d541714b606c224fa77e1690551e70ea891c864f.jpg)

![4129ba8c17d3fa2fdefd94c987214b2e4bf3d6d0c98ae10ef0edff3ed2bad13a.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/tables/4129ba8c17d3fa2fdefd94c987214b2e4bf3d6d0c98ae10ef0edff3ed2bad13a.jpg)

![625bcfc844080f2a7e90ba8e0823200bcdd86f97a36ca523d38832e863b1ab27.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/tables/625bcfc844080f2a7e90ba8e0823200bcdd86f97a36ca523d38832e863b1ab27.jpg)

![7789577e2a8088af44a15a59d1c0cae55f2add174f0c0f55b13aeb4dad8f75b3.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/tables/7789577e2a8088af44a15a59d1c0cae55f2add174f0c0f55b13aeb4dad8f75b3.jpg)

![789d7e77721bcf06ddbe149895a541642aeeed48cdce649f95adacf85c1d8035.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/tables/789d7e77721bcf06ddbe149895a541642aeeed48cdce649f95adacf85c1d8035.jpg)

![82821fc65a1dd4495becd2d87df3bb565f65ce0606ea506562a8ec07ed8561a6.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/tables/82821fc65a1dd4495becd2d87df3bb565f65ce0606ea506562a8ec07ed8561a6.jpg)

![b4e41f7440af64d4f1bee538a4c15a80ea62ec919ef2545b8e0e8eda3b1f4ba5.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/tables/b4e41f7440af64d4f1bee538a4c15a80ea62ec919ef2545b8e0e8eda3b1f4ba5.jpg)

![bd2276c69afa04700a28e793f7bed2435e6802020ccf3e310adbc111f97222c9.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/tables/bd2276c69afa04700a28e793f7bed2435e6802020ccf3e310adbc111f97222c9.jpg)

![d1d4f5bc27b80b6a1805044a5f34b40ec7642b3abd93bd4962ecf3b446c189d8.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/tables/d1d4f5bc27b80b6a1805044a5f34b40ec7642b3abd93bd4962ecf3b446c189d8.jpg)

![fc472b4811c0eddd6c5cfb54b54c38b9b0b9dc65f1e08704d25309c3f20e0b16.jpg](../icml_results/5337_EigenCache_%20Rethinking%20Diffusion%20Acceleration%20as%20Covariance-Optimal%20Forecasting%20and%20Submodular%20Infor/tables/fc472b4811c0eddd6c5cfb54b54c38b9b0b9dc65f1e08704d25309c3f20e0b16.jpg)

<a id="paper-5290"></a>

## Fingerprinting Pre-trained Encoders under Arbitrary Downstream Fine-Tuning via Adversarial Shifting


### Images

![03d16c07741f4b91fa18a33e92a70ff1559bdb529b959ef6ac71f54408d92bb2.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/images/03d16c07741f4b91fa18a33e92a70ff1559bdb529b959ef6ac71f54408d92bb2.jpg)

![2824d53984e1e7e44193d0dfc32ab4b775cd58c888c26f837e8ea9f619fdedf8.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/images/2824d53984e1e7e44193d0dfc32ab4b775cd58c888c26f837e8ea9f619fdedf8.jpg)

![2c0b1d754b3d8045ef67481b8058a185d9ba136a1f28578ef9b52017ba328618.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/images/2c0b1d754b3d8045ef67481b8058a185d9ba136a1f28578ef9b52017ba328618.jpg)

![425d4c5a22a0c520f02dc1333f9ce0303aac9259a428971e68134a8983522f76.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/images/425d4c5a22a0c520f02dc1333f9ce0303aac9259a428971e68134a8983522f76.jpg)

![43cbb19377ea9e8f90654170ecafa033fd3e5cc978d0b6d6737d72612aae1cf1.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/images/43cbb19377ea9e8f90654170ecafa033fd3e5cc978d0b6d6737d72612aae1cf1.jpg)

![4d060efc9e5a6c0f8185546c03c4932622fa78ce082202a28a37c705373c5c23.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/images/4d060efc9e5a6c0f8185546c03c4932622fa78ce082202a28a37c705373c5c23.jpg)

![8bef3a1661395f466731c2707ac6eb96ff8ebf339b9ba0c47027582986f83f6e.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/images/8bef3a1661395f466731c2707ac6eb96ff8ebf339b9ba0c47027582986f83f6e.jpg)

![90403c44c1702e531d5f3ea24835684b2208279f6c30f5e67b412c00eb869397.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/images/90403c44c1702e531d5f3ea24835684b2208279f6c30f5e67b412c00eb869397.jpg)

![e883dd44e6c951b30d1e66730bcaadfd9d43191a7996268a755fcde82a2f74fa.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/images/e883dd44e6c951b30d1e66730bcaadfd9d43191a7996268a755fcde82a2f74fa.jpg)

![ef2aa9d84f26661cbb8d49916df8bd3ca7603983520cff4ed18ddc85333ebb82.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/images/ef2aa9d84f26661cbb8d49916df8bd3ca7603983520cff4ed18ddc85333ebb82.jpg)

![fe99701003a345e51b0d0fd5096dceffa6110390f37f3660e25c0382bd8d3802.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/images/fe99701003a345e51b0d0fd5096dceffa6110390f37f3660e25c0382bd8d3802.jpg)

### Tables

![02558758e90063b68004fc97679f077cb35f451fc548c8f0683a80c9e6b27532.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/tables/02558758e90063b68004fc97679f077cb35f451fc548c8f0683a80c9e6b27532.jpg)

![47de8076c156bafaa4bde87a47bb5747e3c0913f1c086b5a67224af82c7650fa.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/tables/47de8076c156bafaa4bde87a47bb5747e3c0913f1c086b5a67224af82c7650fa.jpg)

![636e9680a31a8eac065ebac54dae6f4eca0686e1b061918e434a22a68e4850b1.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/tables/636e9680a31a8eac065ebac54dae6f4eca0686e1b061918e434a22a68e4850b1.jpg)

![663de8127515c4f0d4afb213ef5e6a3f4aa6e3afe21a6adffd6423ce0ba29e29.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/tables/663de8127515c4f0d4afb213ef5e6a3f4aa6e3afe21a6adffd6423ce0ba29e29.jpg)

![98eb1ec92ba03a210f1be3e5763bd76c6732c579bb6bd3bb2115441ac33aaba1.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/tables/98eb1ec92ba03a210f1be3e5763bd76c6732c579bb6bd3bb2115441ac33aaba1.jpg)

![cc3ae54727be37688d93879426e026781bd48fbe1923c0dea2ccc046a0711e56.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/tables/cc3ae54727be37688d93879426e026781bd48fbe1923c0dea2ccc046a0711e56.jpg)

![d0b40831b970bdacc6bef7195e0d6357e9ed2af450c6485e31ee4a070940047e.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/tables/d0b40831b970bdacc6bef7195e0d6357e9ed2af450c6485e31ee4a070940047e.jpg)

![f0384d7a582f68686cab3ecac1e36498af557207c31eaae306afb28cea78c4aa.jpg](../icml_results/5338_Fingerprinting%20Pre-trained%20Encoders%20under%20Arbitrary%20Downstream%20Fine-Tuning%20via%20Adversarial%20Shifting/tables/f0384d7a582f68686cab3ecac1e36498af557207c31eaae306afb28cea78c4aa.jpg)

<a id="paper-5291"></a>

## CoGeoAD: Hierarchical Color-Geometric Fusion with Multi-View Attention for Zero-Shot 3D Anomaly Detection


### Images

![24ac135fea3a099831d75e52fca755bbdf5379ac23acd7af985e43667ffce550.jpg](../icml_results/5339_CoGeoAD_%20Hierarchical%20Color-Geometric%20Fusion%20with%20Multi-View%20Attention%20for%20Zero-Shot%203D%20Anomaly%20Dete/images/24ac135fea3a099831d75e52fca755bbdf5379ac23acd7af985e43667ffce550.jpg)

![87304b8b7998db814b6e1d1d5be3118d32e7948ea238fe08290b201431e44fd1.jpg](../icml_results/5339_CoGeoAD_%20Hierarchical%20Color-Geometric%20Fusion%20with%20Multi-View%20Attention%20for%20Zero-Shot%203D%20Anomaly%20Dete/images/87304b8b7998db814b6e1d1d5be3118d32e7948ea238fe08290b201431e44fd1.jpg)

![a574d67593c8b4fbe2f27884ee7c2c7f11ee3c82b9fbf3b270d5eb1afb2cef4d.jpg](../icml_results/5339_CoGeoAD_%20Hierarchical%20Color-Geometric%20Fusion%20with%20Multi-View%20Attention%20for%20Zero-Shot%203D%20Anomaly%20Dete/images/a574d67593c8b4fbe2f27884ee7c2c7f11ee3c82b9fbf3b270d5eb1afb2cef4d.jpg)

![b83c5b3ffe66d0b5fd2251c44ff821c6d3fde2abe9c7fc6449f576f983e98c81.jpg](../icml_results/5339_CoGeoAD_%20Hierarchical%20Color-Geometric%20Fusion%20with%20Multi-View%20Attention%20for%20Zero-Shot%203D%20Anomaly%20Dete/images/b83c5b3ffe66d0b5fd2251c44ff821c6d3fde2abe9c7fc6449f576f983e98c81.jpg)

![ce79a0d12b5c1baa82fe7f9fca6303427e07c86939cbb4d0d25902345275c9e0.jpg](../icml_results/5339_CoGeoAD_%20Hierarchical%20Color-Geometric%20Fusion%20with%20Multi-View%20Attention%20for%20Zero-Shot%203D%20Anomaly%20Dete/images/ce79a0d12b5c1baa82fe7f9fca6303427e07c86939cbb4d0d25902345275c9e0.jpg)

![cf594a11872b28ed107502f6744466d87ab41527a39d2fff9b838b3e19c91d24.jpg](../icml_results/5339_CoGeoAD_%20Hierarchical%20Color-Geometric%20Fusion%20with%20Multi-View%20Attention%20for%20Zero-Shot%203D%20Anomaly%20Dete/images/cf594a11872b28ed107502f6744466d87ab41527a39d2fff9b838b3e19c91d24.jpg)

### Tables

![2fafa3c351b7684457ab3b3fd80b590e2acdbeb581dc138c7b9732cd58d4880d.jpg](../icml_results/5339_CoGeoAD_%20Hierarchical%20Color-Geometric%20Fusion%20with%20Multi-View%20Attention%20for%20Zero-Shot%203D%20Anomaly%20Dete/tables/2fafa3c351b7684457ab3b3fd80b590e2acdbeb581dc138c7b9732cd58d4880d.jpg)

![40810447a6cf75ffca5c969aee7c601b0634f39c1296e1110d9b97bb4229d25a.jpg](../icml_results/5339_CoGeoAD_%20Hierarchical%20Color-Geometric%20Fusion%20with%20Multi-View%20Attention%20for%20Zero-Shot%203D%20Anomaly%20Dete/tables/40810447a6cf75ffca5c969aee7c601b0634f39c1296e1110d9b97bb4229d25a.jpg)

![40f12a8c0ee170264e2ff02f09abc5810bfc7016edc2ffdd6ab88c64987ff0a9.jpg](../icml_results/5339_CoGeoAD_%20Hierarchical%20Color-Geometric%20Fusion%20with%20Multi-View%20Attention%20for%20Zero-Shot%203D%20Anomaly%20Dete/tables/40f12a8c0ee170264e2ff02f09abc5810bfc7016edc2ffdd6ab88c64987ff0a9.jpg)

![4f68b6d00ac0b3b44ea0acf78194b5a11650404054ad00027645e286b29f8851.jpg](../icml_results/5339_CoGeoAD_%20Hierarchical%20Color-Geometric%20Fusion%20with%20Multi-View%20Attention%20for%20Zero-Shot%203D%20Anomaly%20Dete/tables/4f68b6d00ac0b3b44ea0acf78194b5a11650404054ad00027645e286b29f8851.jpg)

![52914aae1a70d5625f550f763442915e1c0621d09da75b4026b34f6b9fcd2751.jpg](../icml_results/5339_CoGeoAD_%20Hierarchical%20Color-Geometric%20Fusion%20with%20Multi-View%20Attention%20for%20Zero-Shot%203D%20Anomaly%20Dete/tables/52914aae1a70d5625f550f763442915e1c0621d09da75b4026b34f6b9fcd2751.jpg)

![6290464727fd7361c1532bdaa16eef8882adacb07e8021188ed392d148e2d6ea.jpg](../icml_results/5339_CoGeoAD_%20Hierarchical%20Color-Geometric%20Fusion%20with%20Multi-View%20Attention%20for%20Zero-Shot%203D%20Anomaly%20Dete/tables/6290464727fd7361c1532bdaa16eef8882adacb07e8021188ed392d148e2d6ea.jpg)

![b372b49f6969ef1ddc35f5259e7c0ce7f7b7deba1cdc15da4a9402409e786013.jpg](../icml_results/5339_CoGeoAD_%20Hierarchical%20Color-Geometric%20Fusion%20with%20Multi-View%20Attention%20for%20Zero-Shot%203D%20Anomaly%20Dete/tables/b372b49f6969ef1ddc35f5259e7c0ce7f7b7deba1cdc15da4a9402409e786013.jpg)

![c4071fe86b376137453ffadcb6a0d73a6403b749396546c1a99fde62b8d68e21.jpg](../icml_results/5339_CoGeoAD_%20Hierarchical%20Color-Geometric%20Fusion%20with%20Multi-View%20Attention%20for%20Zero-Shot%203D%20Anomaly%20Dete/tables/c4071fe86b376137453ffadcb6a0d73a6403b749396546c1a99fde62b8d68e21.jpg)

![d65a6917f9dcc33b08fec87f409b451ecf35cec391582e34355c7b61b1cf1e82.jpg](../icml_results/5339_CoGeoAD_%20Hierarchical%20Color-Geometric%20Fusion%20with%20Multi-View%20Attention%20for%20Zero-Shot%203D%20Anomaly%20Dete/tables/d65a6917f9dcc33b08fec87f409b451ecf35cec391582e34355c7b61b1cf1e82.jpg)

![ef0bf3822247307c1d9265c09acdfe4a5dd09e273aebcd3de2eae657e763566b.jpg](../icml_results/5339_CoGeoAD_%20Hierarchical%20Color-Geometric%20Fusion%20with%20Multi-View%20Attention%20for%20Zero-Shot%203D%20Anomaly%20Dete/tables/ef0bf3822247307c1d9265c09acdfe4a5dd09e273aebcd3de2eae657e763566b.jpg)

<a id="paper-5292"></a>

## Visual Para-Thinker: Divide-and-Conquer Reasoning for Visual Comprehension


### Images

![0552171039a934b88e8b2728b42808f5e69764d36c91e8d42e9665167d52db0e.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/images/0552171039a934b88e8b2728b42808f5e69764d36c91e8d42e9665167d52db0e.jpg)

![2f67709999d363cfc02023607995759c9bccacee251cc5459b8378f24c0937b1.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/images/2f67709999d363cfc02023607995759c9bccacee251cc5459b8378f24c0937b1.jpg)

![32e275e5d62562e54938ea9b7e1536b2ab77af6435c12484f889e6c74ae547f6.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/images/32e275e5d62562e54938ea9b7e1536b2ab77af6435c12484f889e6c74ae547f6.jpg)

![6b937fd4ad170c9ab6f36a1ff6a0121effa443b209dca8595a5504608223007a.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/images/6b937fd4ad170c9ab6f36a1ff6a0121effa443b209dca8595a5504608223007a.jpg)

![7cae07b528b44db7ac39e846881cf708ccb85483b684d993e9809fe57e60fa9f.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/images/7cae07b528b44db7ac39e846881cf708ccb85483b684d993e9809fe57e60fa9f.jpg)

![8ab1ecce68c33343793dadf8ac63250d5fc11b788db6ec201826d2ea8d570c89.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/images/8ab1ecce68c33343793dadf8ac63250d5fc11b788db6ec201826d2ea8d570c89.jpg)

![8b2bc621ee66647b9937804d2af5e5f657f5c05b7c131500fe702f089dae111e.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/images/8b2bc621ee66647b9937804d2af5e5f657f5c05b7c131500fe702f089dae111e.jpg)

![8e60704bb8532872d6e103ec6232f7f63c7a5c98eaddeeb89960c577b5b85a37.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/images/8e60704bb8532872d6e103ec6232f7f63c7a5c98eaddeeb89960c577b5b85a37.jpg)

![9c7fcb490679b4e300e7c12f1de105ad1d3f42ea7f0300f73d892b99a0b78b4d.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/images/9c7fcb490679b4e300e7c12f1de105ad1d3f42ea7f0300f73d892b99a0b78b4d.jpg)

![a1635f9d739f372d9421c2bb51464a32ce014a4ac2723f019b6de7f6923be15c.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/images/a1635f9d739f372d9421c2bb51464a32ce014a4ac2723f019b6de7f6923be15c.jpg)

![fbad243001241ef12c2b716311a3a1b1586d751115840c46294d45da649bfe5f.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/images/fbad243001241ef12c2b716311a3a1b1586d751115840c46294d45da649bfe5f.jpg)

### Tables

![09083b85857a9b37721db43c8bdfc5ef3deeefd751ad8a58b28aeb047f22bf52.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/tables/09083b85857a9b37721db43c8bdfc5ef3deeefd751ad8a58b28aeb047f22bf52.jpg)

![1a8fd35ba43a90f80b01bf81d95ab7eb8973f310e6ae3172929977a35e3e61d1.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/tables/1a8fd35ba43a90f80b01bf81d95ab7eb8973f310e6ae3172929977a35e3e61d1.jpg)

![1c4b4a779fd2ee60743acae8bff96e1e032c1b45c708844747d67d2a973a55c6.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/tables/1c4b4a779fd2ee60743acae8bff96e1e032c1b45c708844747d67d2a973a55c6.jpg)

![4ed796a3be7b9c66310e4ba5c26bc405a20984aa4efd7db73795b0e1087b7203.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/tables/4ed796a3be7b9c66310e4ba5c26bc405a20984aa4efd7db73795b0e1087b7203.jpg)

![5fbb52eee8a7decfc59bf34bf779092dbcb204c64ed0f36d2d1b4f348437e9f0.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/tables/5fbb52eee8a7decfc59bf34bf779092dbcb204c64ed0f36d2d1b4f348437e9f0.jpg)

![a9c390e6a73e63f067b49b8a4f3137b7c9c7558d128bde0da10b03729f23e208.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/tables/a9c390e6a73e63f067b49b8a4f3137b7c9c7558d128bde0da10b03729f23e208.jpg)

![c6d9c176a1b4c018860bfedc612844d024eea3fd160ab04d24f0cc29fbc986df.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/tables/c6d9c176a1b4c018860bfedc612844d024eea3fd160ab04d24f0cc29fbc986df.jpg)

![ff372a884cbab5f4f4a5589dc464a9f802def326f442468f883456a079cd6fd8.jpg](../icml_results/5340_Visual%20Para-Thinker_%20Divide-and-Conquer%20Reasoning%20for%20Visual%20Comprehension/tables/ff372a884cbab5f4f4a5589dc464a9f802def326f442468f883456a079cd6fd8.jpg)

<a id="paper-5293"></a>

## TIME: Tensor-Factorized Mixture-of-Experts with Intrinsic Routing for Lifelong Multimodal Knowledge Editing


### Images

![0378d5197f4fce4b7338bb696c8a1e1c87a7d069a5285f1e8a0e523aa4c647ef.jpg](../icml_results/5341_TIME_%20Tensor-Factorized%20Mixture-of-Experts%20with%20Intrinsic%20Routing%20for%20Lifelong%20Multimodal%20Knowledge%20/images/0378d5197f4fce4b7338bb696c8a1e1c87a7d069a5285f1e8a0e523aa4c647ef.jpg)

![06a113f2540bf54961c4adff1ccc1313780c82053eab64f29d55096c0a92010e.jpg](../icml_results/5341_TIME_%20Tensor-Factorized%20Mixture-of-Experts%20with%20Intrinsic%20Routing%20for%20Lifelong%20Multimodal%20Knowledge%20/images/06a113f2540bf54961c4adff1ccc1313780c82053eab64f29d55096c0a92010e.jpg)

![261f656f70764c8a4bc81af714ceb19baef04814b7128502bbac8ecbde4e835c.jpg](../icml_results/5341_TIME_%20Tensor-Factorized%20Mixture-of-Experts%20with%20Intrinsic%20Routing%20for%20Lifelong%20Multimodal%20Knowledge%20/images/261f656f70764c8a4bc81af714ceb19baef04814b7128502bbac8ecbde4e835c.jpg)

![4801d2604c8327fe7a2a90a89442b16946509c5e90068a613ad9fe815e1718be.jpg](../icml_results/5341_TIME_%20Tensor-Factorized%20Mixture-of-Experts%20with%20Intrinsic%20Routing%20for%20Lifelong%20Multimodal%20Knowledge%20/images/4801d2604c8327fe7a2a90a89442b16946509c5e90068a613ad9fe815e1718be.jpg)

![9097a0513ce4601204e72ee1d041e016c9061913b2973be20b4fb11a035431f8.jpg](../icml_results/5341_TIME_%20Tensor-Factorized%20Mixture-of-Experts%20with%20Intrinsic%20Routing%20for%20Lifelong%20Multimodal%20Knowledge%20/images/9097a0513ce4601204e72ee1d041e016c9061913b2973be20b4fb11a035431f8.jpg)

![956d5e5de4add47504a1aa88a54d50812181651e88832354150692dc5dd061fe.jpg](../icml_results/5341_TIME_%20Tensor-Factorized%20Mixture-of-Experts%20with%20Intrinsic%20Routing%20for%20Lifelong%20Multimodal%20Knowledge%20/images/956d5e5de4add47504a1aa88a54d50812181651e88832354150692dc5dd061fe.jpg)

![98cd609601f3fec0a3e9bf47307f7cdcb90dbaa07c2d28f8b128b315032b8a59.jpg](../icml_results/5341_TIME_%20Tensor-Factorized%20Mixture-of-Experts%20with%20Intrinsic%20Routing%20for%20Lifelong%20Multimodal%20Knowledge%20/images/98cd609601f3fec0a3e9bf47307f7cdcb90dbaa07c2d28f8b128b315032b8a59.jpg)

![c4e040436d0121d676a889d99ea61d623b26509a6ae7051ff385e907098224bb.jpg](../icml_results/5341_TIME_%20Tensor-Factorized%20Mixture-of-Experts%20with%20Intrinsic%20Routing%20for%20Lifelong%20Multimodal%20Knowledge%20/images/c4e040436d0121d676a889d99ea61d623b26509a6ae7051ff385e907098224bb.jpg)

![e2c13c933525489ed0bc7272c90806d85e5147f9d965332b568b1a278b16c865.jpg](../icml_results/5341_TIME_%20Tensor-Factorized%20Mixture-of-Experts%20with%20Intrinsic%20Routing%20for%20Lifelong%20Multimodal%20Knowledge%20/images/e2c13c933525489ed0bc7272c90806d85e5147f9d965332b568b1a278b16c865.jpg)

### Tables

![057579a400baeb89aa2c57bb1bb098c8d1bd3d57ceea630ec21076637b826565.jpg](../icml_results/5341_TIME_%20Tensor-Factorized%20Mixture-of-Experts%20with%20Intrinsic%20Routing%20for%20Lifelong%20Multimodal%20Knowledge%20/tables/057579a400baeb89aa2c57bb1bb098c8d1bd3d57ceea630ec21076637b826565.jpg)

![2433916bc7de694c2db95c56f544dca96aa5d205907958db6e14ef98d26eb817.jpg](../icml_results/5341_TIME_%20Tensor-Factorized%20Mixture-of-Experts%20with%20Intrinsic%20Routing%20for%20Lifelong%20Multimodal%20Knowledge%20/tables/2433916bc7de694c2db95c56f544dca96aa5d205907958db6e14ef98d26eb817.jpg)

![472093a4637e131907246bc298404de4e53025412506d39e96d0db3ccbd8f3de.jpg](../icml_results/5341_TIME_%20Tensor-Factorized%20Mixture-of-Experts%20with%20Intrinsic%20Routing%20for%20Lifelong%20Multimodal%20Knowledge%20/tables/472093a4637e131907246bc298404de4e53025412506d39e96d0db3ccbd8f3de.jpg)

![49a50f6a0b4a965fa4de2b55ebfd3db87e5b09ca2e2fea86aec1e6d22152bb10.jpg](../icml_results/5341_TIME_%20Tensor-Factorized%20Mixture-of-Experts%20with%20Intrinsic%20Routing%20for%20Lifelong%20Multimodal%20Knowledge%20/tables/49a50f6a0b4a965fa4de2b55ebfd3db87e5b09ca2e2fea86aec1e6d22152bb10.jpg)

![533b1c91be55d4cd16c2cd26d5fbdecb5cddd1b13be55e894114ccfe94c4f54b.jpg](../icml_results/5341_TIME_%20Tensor-Factorized%20Mixture-of-Experts%20with%20Intrinsic%20Routing%20for%20Lifelong%20Multimodal%20Knowledge%20/tables/533b1c91be55d4cd16c2cd26d5fbdecb5cddd1b13be55e894114ccfe94c4f54b.jpg)

![5a9fc6b73c7b581c23a541a78e37fb4e824c4988dcac573c8fbeaa07f7e7f5e7.jpg](../icml_results/5341_TIME_%20Tensor-Factorized%20Mixture-of-Experts%20with%20Intrinsic%20Routing%20for%20Lifelong%20Multimodal%20Knowledge%20/tables/5a9fc6b73c7b581c23a541a78e37fb4e824c4988dcac573c8fbeaa07f7e7f5e7.jpg)

![5caf581e799233e9f28aed4a4f0e5c6b63c6d447079b0f79d6e78d852ae1d3f6.jpg](../icml_results/5341_TIME_%20Tensor-Factorized%20Mixture-of-Experts%20with%20Intrinsic%20Routing%20for%20Lifelong%20Multimodal%20Knowledge%20/tables/5caf581e799233e9f28aed4a4f0e5c6b63c6d447079b0f79d6e78d852ae1d3f6.jpg)

![61c3eab69d74e46e8abe93d602c12704939f83ea101e1c6eee431c1ba302e35c.jpg](../icml_results/5341_TIME_%20Tensor-Factorized%20Mixture-of-Experts%20with%20Intrinsic%20Routing%20for%20Lifelong%20Multimodal%20Knowledge%20/tables/61c3eab69d74e46e8abe93d602c12704939f83ea101e1c6eee431c1ba302e35c.jpg)

<a id="paper-5294"></a>

## Unlocking Cross-Modal Biosignal Synthesis: A Temporally-Aware VAE-Diffusion Model


### Images

![b1bf7d7981e19a44124631fdd1cffc50a98f6145b6dc759b70c56ebfbb473c25.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/images/b1bf7d7981e19a44124631fdd1cffc50a98f6145b6dc759b70c56ebfbb473c25.jpg)

![e473878fc64744e4259066229a6439adaac9c29249fbfa429783e017809547b5.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/images/e473878fc64744e4259066229a6439adaac9c29249fbfa429783e017809547b5.jpg)

![fb1ddacd4d404bfd580bb87af625b200fbd86ccfb3d29ccaa1b63a8c7f6f86e7.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/images/fb1ddacd4d404bfd580bb87af625b200fbd86ccfb3d29ccaa1b63a8c7f6f86e7.jpg)

![fb9943e80f479b1e5a2b1d24c0601ff0bd1f90c9f329af6185d5d18ed4e72eb1.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/images/fb9943e80f479b1e5a2b1d24c0601ff0bd1f90c9f329af6185d5d18ed4e72eb1.jpg)

### Tables

![011cc3b3ba647e2c7482218526458e37b2d797aff61c215500dcd3758293931f.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/011cc3b3ba647e2c7482218526458e37b2d797aff61c215500dcd3758293931f.jpg)

![0c0c8ceb2158247604ee7e5693e503634146e2e5381e14bd54aa05d754fffe04.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/0c0c8ceb2158247604ee7e5693e503634146e2e5381e14bd54aa05d754fffe04.jpg)

![0d0c6cf0b79fba9a9fad43b82d11b92ecb450cdd7d086c41cd6c9815f30cdb43.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/0d0c6cf0b79fba9a9fad43b82d11b92ecb450cdd7d086c41cd6c9815f30cdb43.jpg)

![2dadd5beabc1133993ad75523d3a33108fb96978bcda998b1d9de5d6fe958847.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/2dadd5beabc1133993ad75523d3a33108fb96978bcda998b1d9de5d6fe958847.jpg)

![3554bfb3f5ff1162f046917c2dae79512ef68902b12d4352265481f9a049269c.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/3554bfb3f5ff1162f046917c2dae79512ef68902b12d4352265481f9a049269c.jpg)

![3b530b29d58182be3c6e03ea79595773e97c514dff1a0ee663a09693589ff414.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/3b530b29d58182be3c6e03ea79595773e97c514dff1a0ee663a09693589ff414.jpg)

![3fc4963dba02cf8ddc55b927d352434ba3967127a052efe1485d720d506ed689.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/3fc4963dba02cf8ddc55b927d352434ba3967127a052efe1485d720d506ed689.jpg)

![4a4a323095dadf020081d847638e8147e67491f81d8f63e6e82db7d5483c762c.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/4a4a323095dadf020081d847638e8147e67491f81d8f63e6e82db7d5483c762c.jpg)

![4be6e3b8ef74d6792da92a835dc78256d8369d4a25c94e65ee91106ac151ec76.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/4be6e3b8ef74d6792da92a835dc78256d8369d4a25c94e65ee91106ac151ec76.jpg)

![5bc87de155542efec14ed34e65e94abbbced076f05b02970b74274fccd96f3e5.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/5bc87de155542efec14ed34e65e94abbbced076f05b02970b74274fccd96f3e5.jpg)

![750d6ea1452bd4439d9318c9e8232bdc5032d0f4173642400eb01cb0500a60eb.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/750d6ea1452bd4439d9318c9e8232bdc5032d0f4173642400eb01cb0500a60eb.jpg)

![7ace0a233c988457085b8b754ae6ce138f19b9b62f3ea829545f33739e83a319.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/7ace0a233c988457085b8b754ae6ce138f19b9b62f3ea829545f33739e83a319.jpg)

![7f63f6a12145fb07571cc3687d1d362ceb455154443f871af0a7df764302d776.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/7f63f6a12145fb07571cc3687d1d362ceb455154443f871af0a7df764302d776.jpg)

![847d03a5cb65753c4504cad523fda244f5cc71b1bd4a2796daaef6fd122df3e1.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/847d03a5cb65753c4504cad523fda244f5cc71b1bd4a2796daaef6fd122df3e1.jpg)

![8ec03b962158c2a08f28ad84e0f81f67590cea09973d5f2c2b40b76c5030fd07.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/8ec03b962158c2a08f28ad84e0f81f67590cea09973d5f2c2b40b76c5030fd07.jpg)

![a8e22cec2949eff148ede9b7bfb39fa0eda67e2d5a5a41a9dcb7e8b80910359e.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/a8e22cec2949eff148ede9b7bfb39fa0eda67e2d5a5a41a9dcb7e8b80910359e.jpg)

![ce3a4d5fbdf7835d76d5c52edfc45969fc3ff6a4be0421c21ee89b8bef17903f.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/ce3a4d5fbdf7835d76d5c52edfc45969fc3ff6a4be0421c21ee89b8bef17903f.jpg)

![eaa8e3118c3686ea10cac9b10710912e0c46d2df4d786011879ed47a205b784f.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/eaa8e3118c3686ea10cac9b10710912e0c46d2df4d786011879ed47a205b784f.jpg)

![f183b741860cc3922a2d920c47c33d8acd276f3f03bbd87cefbc12997529dfc2.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/f183b741860cc3922a2d920c47c33d8acd276f3f03bbd87cefbc12997529dfc2.jpg)

![fcb6f724e5b49ddca38826f5648eca7ed6b71d66a63ce5657e80cd9869fbafd4.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/fcb6f724e5b49ddca38826f5648eca7ed6b71d66a63ce5657e80cd9869fbafd4.jpg)

![ff0e6f62b260570b99f541900591ca4d752cd83658b40500d7a0eafeab37b14a.jpg](../icml_results/5342_Unlocking%20Cross-Modal%20Biosignal%20Synthesis_%20A%20Temporally-Aware%20VAE-Diffusion%20Model/tables/ff0e6f62b260570b99f541900591ca4d752cd83658b40500d7a0eafeab37b14a.jpg)

<a id="paper-5295"></a>

## Non-Adversarial Imitation Learning Provably Free of Compounding Errors: The Value Flow Mechanism


### Images

![0e54d49a913e71659329e194db3325f352aae486f11da088292b523b1d48fce9.jpg](../icml_results/5343_Non-Adversarial%20Imitation%20Learning%20Provably%20Free%20of%20Compounding%20Errors_%20The%20Value%20Flow%20Mechanism/images/0e54d49a913e71659329e194db3325f352aae486f11da088292b523b1d48fce9.jpg)

![3425f759650a1bdc2540119df840a672dc66e50cd1cbbf75a99c1aa02c4ae2af.jpg](../icml_results/5343_Non-Adversarial%20Imitation%20Learning%20Provably%20Free%20of%20Compounding%20Errors_%20The%20Value%20Flow%20Mechanism/images/3425f759650a1bdc2540119df840a672dc66e50cd1cbbf75a99c1aa02c4ae2af.jpg)

![5ac301c8076fedfa5b30884aacf9431deb796f54441a269ceb1ada0af5b9d387.jpg](../icml_results/5343_Non-Adversarial%20Imitation%20Learning%20Provably%20Free%20of%20Compounding%20Errors_%20The%20Value%20Flow%20Mechanism/images/5ac301c8076fedfa5b30884aacf9431deb796f54441a269ceb1ada0af5b9d387.jpg)

![b85d484f0b17f1bef6039b3de389169a6797e609f2be8669380b782598fc94d1.jpg](../icml_results/5343_Non-Adversarial%20Imitation%20Learning%20Provably%20Free%20of%20Compounding%20Errors_%20The%20Value%20Flow%20Mechanism/images/b85d484f0b17f1bef6039b3de389169a6797e609f2be8669380b782598fc94d1.jpg)

![c6f74ddb003b325846bd1013af8d148276608a8755c36450275e880e5e3507a3.jpg](../icml_results/5343_Non-Adversarial%20Imitation%20Learning%20Provably%20Free%20of%20Compounding%20Errors_%20The%20Value%20Flow%20Mechanism/images/c6f74ddb003b325846bd1013af8d148276608a8755c36450275e880e5e3507a3.jpg)

![d78cfe5ee35caeba45805cc98ab687daa672c47946104bfa0d00035a6933dfcd.jpg](../icml_results/5343_Non-Adversarial%20Imitation%20Learning%20Provably%20Free%20of%20Compounding%20Errors_%20The%20Value%20Flow%20Mechanism/images/d78cfe5ee35caeba45805cc98ab687daa672c47946104bfa0d00035a6933dfcd.jpg)

![e46f99bf08ceeec1df9a0a1f887bf25124cdded0a8cb08b5917f0ca526eaaf69.jpg](../icml_results/5343_Non-Adversarial%20Imitation%20Learning%20Provably%20Free%20of%20Compounding%20Errors_%20The%20Value%20Flow%20Mechanism/images/e46f99bf08ceeec1df9a0a1f887bf25124cdded0a8cb08b5917f0ca526eaaf69.jpg)

### Tables

![10f50082b4ed9ff39e223795f5849fff11c242ad70def068588808b3f26b530f.jpg](../icml_results/5343_Non-Adversarial%20Imitation%20Learning%20Provably%20Free%20of%20Compounding%20Errors_%20The%20Value%20Flow%20Mechanism/tables/10f50082b4ed9ff39e223795f5849fff11c242ad70def068588808b3f26b530f.jpg)

![16428b8b207a0610ddf6bd59d2d34ec8cba8493aa66ee0301d83a06fc8f43838.jpg](../icml_results/5343_Non-Adversarial%20Imitation%20Learning%20Provably%20Free%20of%20Compounding%20Errors_%20The%20Value%20Flow%20Mechanism/tables/16428b8b207a0610ddf6bd59d2d34ec8cba8493aa66ee0301d83a06fc8f43838.jpg)

![32a000050ff23f1126200073611008cd4307f1a59228dea3a313f5b7cbdb2a95.jpg](../icml_results/5343_Non-Adversarial%20Imitation%20Learning%20Provably%20Free%20of%20Compounding%20Errors_%20The%20Value%20Flow%20Mechanism/tables/32a000050ff23f1126200073611008cd4307f1a59228dea3a313f5b7cbdb2a95.jpg)

![42c2c528c8b5f5e3b549ab335c7157003142a05604a5bb156a13438c33e7a444.jpg](../icml_results/5343_Non-Adversarial%20Imitation%20Learning%20Provably%20Free%20of%20Compounding%20Errors_%20The%20Value%20Flow%20Mechanism/tables/42c2c528c8b5f5e3b549ab335c7157003142a05604a5bb156a13438c33e7a444.jpg)

![45fa5e87149baccc70230498bfeb33b5b2053e639b3cec9ff0e65d7c8f74e696.jpg](../icml_results/5343_Non-Adversarial%20Imitation%20Learning%20Provably%20Free%20of%20Compounding%20Errors_%20The%20Value%20Flow%20Mechanism/tables/45fa5e87149baccc70230498bfeb33b5b2053e639b3cec9ff0e65d7c8f74e696.jpg)

![4b120063d766a3ac77749defdc462df2b0d4cc44a7e58c208094bc9e164dc2be.jpg](../icml_results/5343_Non-Adversarial%20Imitation%20Learning%20Provably%20Free%20of%20Compounding%20Errors_%20The%20Value%20Flow%20Mechanism/tables/4b120063d766a3ac77749defdc462df2b0d4cc44a7e58c208094bc9e164dc2be.jpg)

![52b2fb161c60c41674af39b4209c57fc367a4275f16ad742d6e50e2c2a309b9a.jpg](../icml_results/5343_Non-Adversarial%20Imitation%20Learning%20Provably%20Free%20of%20Compounding%20Errors_%20The%20Value%20Flow%20Mechanism/tables/52b2fb161c60c41674af39b4209c57fc367a4275f16ad742d6e50e2c2a309b9a.jpg)

![7439d087074c64413811e3f66e5527e9d523fdf7b32e09da0073fdd04c7a70cf.jpg](../icml_results/5343_Non-Adversarial%20Imitation%20Learning%20Provably%20Free%20of%20Compounding%20Errors_%20The%20Value%20Flow%20Mechanism/tables/7439d087074c64413811e3f66e5527e9d523fdf7b32e09da0073fdd04c7a70cf.jpg)

![e930bc6449ed1c03af06ef989e39d97d018ef45543b0d4cc70bf47ea4175a2b7.jpg](../icml_results/5343_Non-Adversarial%20Imitation%20Learning%20Provably%20Free%20of%20Compounding%20Errors_%20The%20Value%20Flow%20Mechanism/tables/e930bc6449ed1c03af06ef989e39d97d018ef45543b0d4cc70bf47ea4175a2b7.jpg)

<a id="paper-5296"></a>

## Hearing Without Noticing? Attention-Aware Stealthy Black-Box Adversarial Audio Attacks


### Images

![22e3403836723401b9e74cab59d4e0d2828881d2e62fc5b17280b9dc8225f97b.jpg](../icml_results/5344_Hearing%20Without%20Noticing_%20Attention-Aware%20Stealthy%20Black-Box%20Adversarial%20Audio%20Attacks/images/22e3403836723401b9e74cab59d4e0d2828881d2e62fc5b17280b9dc8225f97b.jpg)

![3591462f3c70ad4347b1833c2898e0f73297fee3458baab9219a2f64a245d685.jpg](../icml_results/5344_Hearing%20Without%20Noticing_%20Attention-Aware%20Stealthy%20Black-Box%20Adversarial%20Audio%20Attacks/images/3591462f3c70ad4347b1833c2898e0f73297fee3458baab9219a2f64a245d685.jpg)

![5825398e9354caa80c71bdb488c3edfe441acda6bd218ea05d2c25638ed075c0.jpg](../icml_results/5344_Hearing%20Without%20Noticing_%20Attention-Aware%20Stealthy%20Black-Box%20Adversarial%20Audio%20Attacks/images/5825398e9354caa80c71bdb488c3edfe441acda6bd218ea05d2c25638ed075c0.jpg)

### Tables

![0965e166606f2dabc0005f85ccbdacd977107fa60a8ec7c73f1e1ff183eca5ec.jpg](../icml_results/5344_Hearing%20Without%20Noticing_%20Attention-Aware%20Stealthy%20Black-Box%20Adversarial%20Audio%20Attacks/tables/0965e166606f2dabc0005f85ccbdacd977107fa60a8ec7c73f1e1ff183eca5ec.jpg)

![13bb9638c96bfbfaa11b37586e1c1fa667723f699ff16e59672351b96b0d1228.jpg](../icml_results/5344_Hearing%20Without%20Noticing_%20Attention-Aware%20Stealthy%20Black-Box%20Adversarial%20Audio%20Attacks/tables/13bb9638c96bfbfaa11b37586e1c1fa667723f699ff16e59672351b96b0d1228.jpg)

![2563984f233ec9532b60f0fd00ea85d997b79fa1da0b47a2e7279470908590ad.jpg](../icml_results/5344_Hearing%20Without%20Noticing_%20Attention-Aware%20Stealthy%20Black-Box%20Adversarial%20Audio%20Attacks/tables/2563984f233ec9532b60f0fd00ea85d997b79fa1da0b47a2e7279470908590ad.jpg)

![29b38ab2c78d71de759d5a0c802d4c803e4b34e5d4109ad007ae416915ed9938.jpg](../icml_results/5344_Hearing%20Without%20Noticing_%20Attention-Aware%20Stealthy%20Black-Box%20Adversarial%20Audio%20Attacks/tables/29b38ab2c78d71de759d5a0c802d4c803e4b34e5d4109ad007ae416915ed9938.jpg)

![6027b47e8e312d3758106105fce0258154a45ef9b6ef751a160bce5231861af1.jpg](../icml_results/5344_Hearing%20Without%20Noticing_%20Attention-Aware%20Stealthy%20Black-Box%20Adversarial%20Audio%20Attacks/tables/6027b47e8e312d3758106105fce0258154a45ef9b6ef751a160bce5231861af1.jpg)

![76d810e18a45848d3d7a4e4bb3e4947c24596a46ebab123f024f8516c80a68d6.jpg](../icml_results/5344_Hearing%20Without%20Noticing_%20Attention-Aware%20Stealthy%20Black-Box%20Adversarial%20Audio%20Attacks/tables/76d810e18a45848d3d7a4e4bb3e4947c24596a46ebab123f024f8516c80a68d6.jpg)

![81777f9ca2ad2aeffea1fa1648858f568bdb6649de15d957c4a6bcf3693dc975.jpg](../icml_results/5344_Hearing%20Without%20Noticing_%20Attention-Aware%20Stealthy%20Black-Box%20Adversarial%20Audio%20Attacks/tables/81777f9ca2ad2aeffea1fa1648858f568bdb6649de15d957c4a6bcf3693dc975.jpg)

![9ceb49ded0361225bad9380bfe66ac6f1dc67f9369b116cbd33243d9e3c7f863.jpg](../icml_results/5344_Hearing%20Without%20Noticing_%20Attention-Aware%20Stealthy%20Black-Box%20Adversarial%20Audio%20Attacks/tables/9ceb49ded0361225bad9380bfe66ac6f1dc67f9369b116cbd33243d9e3c7f863.jpg)

![a56633d5a87a551e0310a8e557b063a4867cea96e5109ca9992bf24e705f84dc.jpg](../icml_results/5344_Hearing%20Without%20Noticing_%20Attention-Aware%20Stealthy%20Black-Box%20Adversarial%20Audio%20Attacks/tables/a56633d5a87a551e0310a8e557b063a4867cea96e5109ca9992bf24e705f84dc.jpg)

![bbeb7ed23ca0c1ad0059ae3c0805ad38f9438274fdf49eeb4d71bffc48a3c489.jpg](../icml_results/5344_Hearing%20Without%20Noticing_%20Attention-Aware%20Stealthy%20Black-Box%20Adversarial%20Audio%20Attacks/tables/bbeb7ed23ca0c1ad0059ae3c0805ad38f9438274fdf49eeb4d71bffc48a3c489.jpg)

![bcb9053d0d14b58c6bb5502fb4dc725ebf7349fd4273b9b6ca91a2a24e8c7640.jpg](../icml_results/5344_Hearing%20Without%20Noticing_%20Attention-Aware%20Stealthy%20Black-Box%20Adversarial%20Audio%20Attacks/tables/bcb9053d0d14b58c6bb5502fb4dc725ebf7349fd4273b9b6ca91a2a24e8c7640.jpg)

![c8429d26601046e970d3f7805524acbc1aa146e8c2d57633da4c36964a94fd7f.jpg](../icml_results/5344_Hearing%20Without%20Noticing_%20Attention-Aware%20Stealthy%20Black-Box%20Adversarial%20Audio%20Attacks/tables/c8429d26601046e970d3f7805524acbc1aa146e8c2d57633da4c36964a94fd7f.jpg)

<a id="paper-5297"></a>

## PointDiT: Pixel-Space Diffusion for Monocular Geometry Estimation


### Images

![0aae6fd8474d903dc13affa98ac7028bdb9015a144dcaf21abef249a9b2cf407.jpg](../icml_results/5345_PointDiT_%20Pixel-Space%20Diffusion%20for%20Monocular%20Geometry%20Estimation/images/0aae6fd8474d903dc13affa98ac7028bdb9015a144dcaf21abef249a9b2cf407.jpg)

![18ffdc93033f0abffb50c299771d0b6101a31af2ea3e6dd82403ef307d98e096.jpg](../icml_results/5345_PointDiT_%20Pixel-Space%20Diffusion%20for%20Monocular%20Geometry%20Estimation/images/18ffdc93033f0abffb50c299771d0b6101a31af2ea3e6dd82403ef307d98e096.jpg)

![31a7210696f7d62a8bf12c05b13194cee66bb3c8725ecbf0db5dbe05a1edcbe7.jpg](../icml_results/5345_PointDiT_%20Pixel-Space%20Diffusion%20for%20Monocular%20Geometry%20Estimation/images/31a7210696f7d62a8bf12c05b13194cee66bb3c8725ecbf0db5dbe05a1edcbe7.jpg)

![827d1b71a11b004e19cb1d8c11d4844f6e6d1f390e884858dfa655f14b6d2f1a.jpg](../icml_results/5345_PointDiT_%20Pixel-Space%20Diffusion%20for%20Monocular%20Geometry%20Estimation/images/827d1b71a11b004e19cb1d8c11d4844f6e6d1f390e884858dfa655f14b6d2f1a.jpg)

![a8b7efb54b98a344af87bd761dafb24b5e7ecfc89d8313f7cfb96bcd3d5eee5d.jpg](../icml_results/5345_PointDiT_%20Pixel-Space%20Diffusion%20for%20Monocular%20Geometry%20Estimation/images/a8b7efb54b98a344af87bd761dafb24b5e7ecfc89d8313f7cfb96bcd3d5eee5d.jpg)

![ac968c708339e6bc94863350c0bba8c3eafd0fd2cd521fc6a3397bdfa4a69576.jpg](../icml_results/5345_PointDiT_%20Pixel-Space%20Diffusion%20for%20Monocular%20Geometry%20Estimation/images/ac968c708339e6bc94863350c0bba8c3eafd0fd2cd521fc6a3397bdfa4a69576.jpg)

![edf125080778a2af0a91a7dc8162473a51ff13a584fec400af0f0d13f8e12bf5.jpg](../icml_results/5345_PointDiT_%20Pixel-Space%20Diffusion%20for%20Monocular%20Geometry%20Estimation/images/edf125080778a2af0a91a7dc8162473a51ff13a584fec400af0f0d13f8e12bf5.jpg)

![fae28bf0685d5bd651fcaf140b0d95f88ede38fa2dfc488f158454c6e111feb5.jpg](../icml_results/5345_PointDiT_%20Pixel-Space%20Diffusion%20for%20Monocular%20Geometry%20Estimation/images/fae28bf0685d5bd651fcaf140b0d95f88ede38fa2dfc488f158454c6e111feb5.jpg)

### Tables

![043ea4032a9e21b4d106a67c75e422f0a382fbd817bacecc9cb5151af576aeec.jpg](../icml_results/5345_PointDiT_%20Pixel-Space%20Diffusion%20for%20Monocular%20Geometry%20Estimation/tables/043ea4032a9e21b4d106a67c75e422f0a382fbd817bacecc9cb5151af576aeec.jpg)

![095404a0f2998ffa5fc8a77c92a46aa3ab9ecfff7d2a0fa0b2e21f344cef83f0.jpg](../icml_results/5345_PointDiT_%20Pixel-Space%20Diffusion%20for%20Monocular%20Geometry%20Estimation/tables/095404a0f2998ffa5fc8a77c92a46aa3ab9ecfff7d2a0fa0b2e21f344cef83f0.jpg)

![3ddc43a328d31d846a9759c1b2bf9c2ddc8b9694230c033941c10cfbbedc8469.jpg](../icml_results/5345_PointDiT_%20Pixel-Space%20Diffusion%20for%20Monocular%20Geometry%20Estimation/tables/3ddc43a328d31d846a9759c1b2bf9c2ddc8b9694230c033941c10cfbbedc8469.jpg)

![58c8e35ade5b8a16d8a70b9447157c3e357bf836fcdf82e311e3a8313786a941.jpg](../icml_results/5345_PointDiT_%20Pixel-Space%20Diffusion%20for%20Monocular%20Geometry%20Estimation/tables/58c8e35ade5b8a16d8a70b9447157c3e357bf836fcdf82e311e3a8313786a941.jpg)

![855323a74fb345708337fd49824631691a1931eeb73ae5fbfe0c9d80cd30cdf7.jpg](../icml_results/5345_PointDiT_%20Pixel-Space%20Diffusion%20for%20Monocular%20Geometry%20Estimation/tables/855323a74fb345708337fd49824631691a1931eeb73ae5fbfe0c9d80cd30cdf7.jpg)

![a2029e11acf13854ef399247f46914915ddb6844f3f17f77ceecf04000354936.jpg](../icml_results/5345_PointDiT_%20Pixel-Space%20Diffusion%20for%20Monocular%20Geometry%20Estimation/tables/a2029e11acf13854ef399247f46914915ddb6844f3f17f77ceecf04000354936.jpg)

![f7518a5a69690009cf9a3236c2f71e612e884d327a9d75f366c8fa11058afb37.jpg](../icml_results/5345_PointDiT_%20Pixel-Space%20Diffusion%20for%20Monocular%20Geometry%20Estimation/tables/f7518a5a69690009cf9a3236c2f71e612e884d327a9d75f366c8fa11058afb37.jpg)

![f905ede97ce3bb839396a0cbe7e6b36047f962d955c7478c787bc8f108ac2ed6.jpg](../icml_results/5345_PointDiT_%20Pixel-Space%20Diffusion%20for%20Monocular%20Geometry%20Estimation/tables/f905ede97ce3bb839396a0cbe7e6b36047f962d955c7478c787bc8f108ac2ed6.jpg)

![ffca6036918e72b4188938b3feb93b542d0b2321ccc1b9073828734b8b13a9ff.jpg](../icml_results/5345_PointDiT_%20Pixel-Space%20Diffusion%20for%20Monocular%20Geometry%20Estimation/tables/ffca6036918e72b4188938b3feb93b542d0b2321ccc1b9073828734b8b13a9ff.jpg)

<a id="paper-5298"></a>

## D-CORE: Incentivizing Task Decomposition in Large Reasoning Models for Complex Tool Use


### Images

![0cc995d9d21c0331885b19c7f68198e4918b85e98267543d488cb1a852f2e310.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/images/0cc995d9d21c0331885b19c7f68198e4918b85e98267543d488cb1a852f2e310.jpg)

![1e470c2ea40f901b87d8151b1c6521b9be95f64c09bba33f5cf46c1dec5c2d39.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/images/1e470c2ea40f901b87d8151b1c6521b9be95f64c09bba33f5cf46c1dec5c2d39.jpg)

![2f93ddff42177178f89fe6664f9c0de6083b0a10c035b807615da26c5b376525.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/images/2f93ddff42177178f89fe6664f9c0de6083b0a10c035b807615da26c5b376525.jpg)

![330a70ba0cb5b172657550bb564d274dbc0879381b842db8f48a16b35d4b014b.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/images/330a70ba0cb5b172657550bb564d274dbc0879381b842db8f48a16b35d4b014b.jpg)

![3433c51a2ba30c29a6efb0845c1ba11115480776117368a559bc4f1ae7e8bff0.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/images/3433c51a2ba30c29a6efb0845c1ba11115480776117368a559bc4f1ae7e8bff0.jpg)

![3501a4391efb13534ebb2ff40533f940ed787497f0db332532c6b5d4d522c13b.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/images/3501a4391efb13534ebb2ff40533f940ed787497f0db332532c6b5d4d522c13b.jpg)

![3ef3574cbde9ff8f87d5b14cd05e8b33cb591433dfef4a12e87da8f9ce2b6d4e.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/images/3ef3574cbde9ff8f87d5b14cd05e8b33cb591433dfef4a12e87da8f9ce2b6d4e.jpg)

![4585d130fc3d5f78b280b0a4b79c592692b16f591265d103b09df266766e3d47.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/images/4585d130fc3d5f78b280b0a4b79c592692b16f591265d103b09df266766e3d47.jpg)

![582a32bc995467bf42ae9efc139ff3efc9fddcc1fc6693070c8a2aa357448634.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/images/582a32bc995467bf42ae9efc139ff3efc9fddcc1fc6693070c8a2aa357448634.jpg)

![65e4ad8bf38b6d176160d4b06a213686c50f27cf93547ac766e60afcead9f0e1.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/images/65e4ad8bf38b6d176160d4b06a213686c50f27cf93547ac766e60afcead9f0e1.jpg)

![7ccde8025e2b7ed34927493bcdf9b8989315e0b5ad871d6c85e46befdced303a.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/images/7ccde8025e2b7ed34927493bcdf9b8989315e0b5ad871d6c85e46befdced303a.jpg)

![7f5678f1462288ac83f0c5d3d70aea96df271dc0106f9437538dab076974fbf8.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/images/7f5678f1462288ac83f0c5d3d70aea96df271dc0106f9437538dab076974fbf8.jpg)

![95d5cf6d55b66d9db8e8edb696a38de6e65ba9572b494b38d1de1f30cfe572d4.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/images/95d5cf6d55b66d9db8e8edb696a38de6e65ba9572b494b38d1de1f30cfe572d4.jpg)

![d0a2cd14f256f167383f060598bef3a7b116d0185fb9e6ecec20ba0e0f1747c4.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/images/d0a2cd14f256f167383f060598bef3a7b116d0185fb9e6ecec20ba0e0f1747c4.jpg)

### Tables

![04a485aca6985706ecefd12ed570694e415bb4f59942888422ecc8ab95557c24.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/tables/04a485aca6985706ecefd12ed570694e415bb4f59942888422ecc8ab95557c24.jpg)

![09ca0f289d9729761de9cc298619aa092f49cfa7dbb41566cf672e396092e757.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/tables/09ca0f289d9729761de9cc298619aa092f49cfa7dbb41566cf672e396092e757.jpg)

![1dd9e8a7d63344017d8dda02bba77c63ada37eec7aa8d11416c8feb3c33a4486.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/tables/1dd9e8a7d63344017d8dda02bba77c63ada37eec7aa8d11416c8feb3c33a4486.jpg)

![1f154857976b7e537b3652b4656d02fa79a1f71d11d141166af897377db6fcd8.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/tables/1f154857976b7e537b3652b4656d02fa79a1f71d11d141166af897377db6fcd8.jpg)

![257850a30cbc3e7bf152bbdbd8c716ad67b625c5da7794d1a189d17ac038dcb0.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/tables/257850a30cbc3e7bf152bbdbd8c716ad67b625c5da7794d1a189d17ac038dcb0.jpg)

![4738852bf4d675b7c259a613e33b964f616adeed78fb46be1c1d412e1829f46d.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/tables/4738852bf4d675b7c259a613e33b964f616adeed78fb46be1c1d412e1829f46d.jpg)

![8730a7981844e52a504cd6bab9b1e4901090367113655d37ecc272d83c566498.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/tables/8730a7981844e52a504cd6bab9b1e4901090367113655d37ecc272d83c566498.jpg)

![8ed0b754e050abd55c6f28737b3b0d93c6e01c06e6c60e3d53d025a442fba3fe.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/tables/8ed0b754e050abd55c6f28737b3b0d93c6e01c06e6c60e3d53d025a442fba3fe.jpg)

![98e3dd44a9d9c2ade076b425554b47d9c34e47a067f3f65e0e1df3d23ced9f40.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/tables/98e3dd44a9d9c2ade076b425554b47d9c34e47a067f3f65e0e1df3d23ced9f40.jpg)

![b51ee3a9dabfa46471a83a5ee32e0a9350e88f9c783d46b0edfb4322a5cbc20d.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/tables/b51ee3a9dabfa46471a83a5ee32e0a9350e88f9c783d46b0edfb4322a5cbc20d.jpg)

![f4992d032893e14df76c871759db3dbb5750cf60025592f33db19107388364c0.jpg](../icml_results/5346_D-CORE_%20Incentivizing%20Task%20Decomposition%20in%20Large%20Reasoning%20Models%20for%20Complex%20Tool%20Use/tables/f4992d032893e14df76c871759db3dbb5750cf60025592f33db19107388364c0.jpg)

<a id="paper-5299"></a>

## LMCleaner: Efficient and Certified Online Unlearning via Influence Propagation Truncation


### Images

![16d03f685bc0a47bdf38cb04551fd03c75261d870cc1a14cd616a720dab1fa9f.jpg](../icml_results/5347_LMCleaner_%20Efficient%20and%20Certified%20Online%20Unlearning%20via%20Influence%20Propagation%20Truncation/images/16d03f685bc0a47bdf38cb04551fd03c75261d870cc1a14cd616a720dab1fa9f.jpg)

![340ada3f2db2b5e5848913fd3e54f12ef786044332dd07eef67b532aea77cf23.jpg](../icml_results/5347_LMCleaner_%20Efficient%20and%20Certified%20Online%20Unlearning%20via%20Influence%20Propagation%20Truncation/images/340ada3f2db2b5e5848913fd3e54f12ef786044332dd07eef67b532aea77cf23.jpg)

![66508af566581a69cb4be685b6d229d601f7eeed054fe1608d412995769d8451.jpg](../icml_results/5347_LMCleaner_%20Efficient%20and%20Certified%20Online%20Unlearning%20via%20Influence%20Propagation%20Truncation/images/66508af566581a69cb4be685b6d229d601f7eeed054fe1608d412995769d8451.jpg)

![6f5479c96d526f5ebc01a56624c7e0542a7697d94c562d79eebcf2ca652be62e.jpg](../icml_results/5347_LMCleaner_%20Efficient%20and%20Certified%20Online%20Unlearning%20via%20Influence%20Propagation%20Truncation/images/6f5479c96d526f5ebc01a56624c7e0542a7697d94c562d79eebcf2ca652be62e.jpg)

![70de80af9fc0833bb05f1064c2e31ca9716f2a364cee11dab0f316a60fa4bd19.jpg](../icml_results/5347_LMCleaner_%20Efficient%20and%20Certified%20Online%20Unlearning%20via%20Influence%20Propagation%20Truncation/images/70de80af9fc0833bb05f1064c2e31ca9716f2a364cee11dab0f316a60fa4bd19.jpg)

![9102b1684daecc1a45f96db286e9ed9b2da7eb95e79a9ee8590c542dd32652cf.jpg](../icml_results/5347_LMCleaner_%20Efficient%20and%20Certified%20Online%20Unlearning%20via%20Influence%20Propagation%20Truncation/images/9102b1684daecc1a45f96db286e9ed9b2da7eb95e79a9ee8590c542dd32652cf.jpg)

![ddb2fd92e84165a2fed047fde64cfafea2c964c1c16d47bc70afbb6e50a378f2.jpg](../icml_results/5347_LMCleaner_%20Efficient%20and%20Certified%20Online%20Unlearning%20via%20Influence%20Propagation%20Truncation/images/ddb2fd92e84165a2fed047fde64cfafea2c964c1c16d47bc70afbb6e50a378f2.jpg)

### Tables

![4e1a4dd9589c93c81fe072ae76fe9083f5a90bd2a7094369800412a72b823c07.jpg](../icml_results/5347_LMCleaner_%20Efficient%20and%20Certified%20Online%20Unlearning%20via%20Influence%20Propagation%20Truncation/tables/4e1a4dd9589c93c81fe072ae76fe9083f5a90bd2a7094369800412a72b823c07.jpg)

![5612ccbe311a14f8c59f452ce60673048247ed41022f330ceb92ecfd80ff9590.jpg](../icml_results/5347_LMCleaner_%20Efficient%20and%20Certified%20Online%20Unlearning%20via%20Influence%20Propagation%20Truncation/tables/5612ccbe311a14f8c59f452ce60673048247ed41022f330ceb92ecfd80ff9590.jpg)

![5fbe9e6bfbec98006decaaf12d78417f05b3af5b31f3fb246f05cf129e859226.jpg](../icml_results/5347_LMCleaner_%20Efficient%20and%20Certified%20Online%20Unlearning%20via%20Influence%20Propagation%20Truncation/tables/5fbe9e6bfbec98006decaaf12d78417f05b3af5b31f3fb246f05cf129e859226.jpg)

![7b85104ece15afd93568788fb797bc41137c21631567a420e41ff06e7fda3e7f.jpg](../icml_results/5347_LMCleaner_%20Efficient%20and%20Certified%20Online%20Unlearning%20via%20Influence%20Propagation%20Truncation/tables/7b85104ece15afd93568788fb797bc41137c21631567a420e41ff06e7fda3e7f.jpg)

![8fc37a5062e8968a1dbcb70548ae30a85fec763775a3e892f16a56f8b75d73fc.jpg](../icml_results/5347_LMCleaner_%20Efficient%20and%20Certified%20Online%20Unlearning%20via%20Influence%20Propagation%20Truncation/tables/8fc37a5062e8968a1dbcb70548ae30a85fec763775a3e892f16a56f8b75d73fc.jpg)

![ff5f8a91b1524bcaa86681bf25e61f0b6b87dfc3dcb115dd085888586dfe0082.jpg](../icml_results/5347_LMCleaner_%20Efficient%20and%20Certified%20Online%20Unlearning%20via%20Influence%20Propagation%20Truncation/tables/ff5f8a91b1524bcaa86681bf25e61f0b6b87dfc3dcb115dd085888586dfe0082.jpg)

<a id="paper-5300"></a>

## Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models


### Images

![1fdb8b3a79b969d0a8966716b58e7dba7c44b54dd3f0db2e824f904444b2a429.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/1fdb8b3a79b969d0a8966716b58e7dba7c44b54dd3f0db2e824f904444b2a429.jpg)

![21c6739c54f959acc2ac45f234fd906398749b180b4a81ee4d43e5f5132b5f92.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/21c6739c54f959acc2ac45f234fd906398749b180b4a81ee4d43e5f5132b5f92.jpg)

![2899f083d9b25b08f87b27460b79a7a482ee9e869491ba229eca9832470ce06e.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/2899f083d9b25b08f87b27460b79a7a482ee9e869491ba229eca9832470ce06e.jpg)

![3777c40480f195e8f6f21cb3b375024f53a5c223f5f80d94b55e1c6d1a07410d.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/3777c40480f195e8f6f21cb3b375024f53a5c223f5f80d94b55e1c6d1a07410d.jpg)

![3f1e34d4207f27b877b41be59894828265b3f7a03df06e0907a9e835b159d0c7.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/3f1e34d4207f27b877b41be59894828265b3f7a03df06e0907a9e835b159d0c7.jpg)

![43188d29a26318d4cc773b45bd559cbc2b479d07221a372eafb58f722701c01f.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/43188d29a26318d4cc773b45bd559cbc2b479d07221a372eafb58f722701c01f.jpg)

![52f11cdc53dff83d9bb99be0aa89efd25bcf519acf3bc17246879430555ac817.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/52f11cdc53dff83d9bb99be0aa89efd25bcf519acf3bc17246879430555ac817.jpg)

![591acd5c2210ed459c92ed05c040e5268ce8a7b9f9f83d6d63504a2666767873.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/591acd5c2210ed459c92ed05c040e5268ce8a7b9f9f83d6d63504a2666767873.jpg)

![5e26e5332923c49096603526172745f782f8634a9328978aec401cbaacf33c30.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/5e26e5332923c49096603526172745f782f8634a9328978aec401cbaacf33c30.jpg)

![6667c8216c4d8a3b5a69048054dce1c7f13fe0ba424576f6ced2284a5b75e142.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/6667c8216c4d8a3b5a69048054dce1c7f13fe0ba424576f6ced2284a5b75e142.jpg)

![87f6d577c7e5d087540821446c801df07acc34c02ffe005f53fa9d59ac1bca5f.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/87f6d577c7e5d087540821446c801df07acc34c02ffe005f53fa9d59ac1bca5f.jpg)

![a53b3bc8545d2960d8244d64935b688c81dadd0f5ba27e26d9f469026eac2a3b.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/a53b3bc8545d2960d8244d64935b688c81dadd0f5ba27e26d9f469026eac2a3b.jpg)

![b581348cd71fe822ed206221395ddcebe322a287b9fe657e17a995958922c4fb.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/b581348cd71fe822ed206221395ddcebe322a287b9fe657e17a995958922c4fb.jpg)

![bc7bc91084aba765534f5734882970295d3f5fba005bf89af8e4f6aafe1dd94f.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/bc7bc91084aba765534f5734882970295d3f5fba005bf89af8e4f6aafe1dd94f.jpg)

![c1bc173105cd8adbe22164176f65fc58f4a1a3bf5d74811cc280697819352c5a.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/c1bc173105cd8adbe22164176f65fc58f4a1a3bf5d74811cc280697819352c5a.jpg)

![c367a56bb2c75408cc1e3e672f07f4b54e53e46436db5c9a75933e3c9255799e.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/c367a56bb2c75408cc1e3e672f07f4b54e53e46436db5c9a75933e3c9255799e.jpg)

![da165aab930742e0284e028d942700bf57a045cba9f0c382667b949891aeece9.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/da165aab930742e0284e028d942700bf57a045cba9f0c382667b949891aeece9.jpg)

![e2064fb9dee4cf7cf8ebe484f04a62215cf980354b2b7a890dde020a5a7cf815.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/e2064fb9dee4cf7cf8ebe484f04a62215cf980354b2b7a890dde020a5a7cf815.jpg)

![f26f6f17ce5f800cdfdbab443281adbc1fd0fd68717ebc0bc638b3839c076e7f.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/images/f26f6f17ce5f800cdfdbab443281adbc1fd0fd68717ebc0bc638b3839c076e7f.jpg)

### Tables

![0b82c69d49d5e229d049e85f73a761273ff9e109dc4cdc74081a983ead41df90.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/tables/0b82c69d49d5e229d049e85f73a761273ff9e109dc4cdc74081a983ead41df90.jpg)

![5c1617aa208a406b157b17dddbe7eb9a46372f90e98c432672a81d1243f6b616.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/tables/5c1617aa208a406b157b17dddbe7eb9a46372f90e98c432672a81d1243f6b616.jpg)

![74d57924ced96bf4b1b90e6e92c71578396baee4087023c33fd6eb2eb6376cac.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/tables/74d57924ced96bf4b1b90e6e92c71578396baee4087023c33fd6eb2eb6376cac.jpg)

![93600454642294d8b34a99a411bc1e66388ce4b1cd484476acbb98a5c9714ad6.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/tables/93600454642294d8b34a99a411bc1e66388ce4b1cd484476acbb98a5c9714ad6.jpg)

![eae61309beec855de43d7c4f9a584711e61892017065bf6c84ff92ae1bd4001e.jpg](../icml_results/5348_Temporal%20Score%20Rescaling%20for%20Temperature%20Sampling%20in%20Diffusion%20and%20Flow%20Models/tables/eae61309beec855de43d7c4f9a584711e61892017065bf6c84ff92ae1bd4001e.jpg)

<a id="paper-5301"></a>

## A Cartesian-3j Framework for Machine Learning Interatomic Potentials


### Tables

![2d67174064d831f92c7f2eeba5a2e84cd388f5f2b9c4c681187ce4e38d6a2540.jpg](../icml_results/5349_A%20Cartesian-3j%20Framework%20for%20Machine%20Learning%20Interatomic%20Potentials/tables/2d67174064d831f92c7f2eeba5a2e84cd388f5f2b9c4c681187ce4e38d6a2540.jpg)

![2e536a67dc135b83f08efa535788c1889f952c1846e219c34187e3720aaca410.jpg](../icml_results/5349_A%20Cartesian-3j%20Framework%20for%20Machine%20Learning%20Interatomic%20Potentials/tables/2e536a67dc135b83f08efa535788c1889f952c1846e219c34187e3720aaca410.jpg)

![5c1ce4faa4d0e0ecea1692946047810a4c42de91c90a71e8931ec0c72bcbf441.jpg](../icml_results/5349_A%20Cartesian-3j%20Framework%20for%20Machine%20Learning%20Interatomic%20Potentials/tables/5c1ce4faa4d0e0ecea1692946047810a4c42de91c90a71e8931ec0c72bcbf441.jpg)

![5c7dbf5f1282bd790db89894a1a10b917c0cc63a07091650c4f87bc99795a4a9.jpg](../icml_results/5349_A%20Cartesian-3j%20Framework%20for%20Machine%20Learning%20Interatomic%20Potentials/tables/5c7dbf5f1282bd790db89894a1a10b917c0cc63a07091650c4f87bc99795a4a9.jpg)

![5e56af3cfb01c4cf9e65573b79ad27675b683e0ef64bcbca6068f043ff8c93d2.jpg](../icml_results/5349_A%20Cartesian-3j%20Framework%20for%20Machine%20Learning%20Interatomic%20Potentials/tables/5e56af3cfb01c4cf9e65573b79ad27675b683e0ef64bcbca6068f043ff8c93d2.jpg)

![bc4f8e823ed8f92443cf708e7d8100ff3fca22e5f49e7ee69cad005979e6f673.jpg](../icml_results/5349_A%20Cartesian-3j%20Framework%20for%20Machine%20Learning%20Interatomic%20Potentials/tables/bc4f8e823ed8f92443cf708e7d8100ff3fca22e5f49e7ee69cad005979e6f673.jpg)

![c74cab8ba8d30f4f3eacd4b1b3fcb3708f9d01641cb9118b62a2637f3c963378.jpg](../icml_results/5349_A%20Cartesian-3j%20Framework%20for%20Machine%20Learning%20Interatomic%20Potentials/tables/c74cab8ba8d30f4f3eacd4b1b3fcb3708f9d01641cb9118b62a2637f3c963378.jpg)

![d988e280f7bf8514c2421d573140cfce9b77976879b3ae725ff507c5d65d4f95.jpg](../icml_results/5349_A%20Cartesian-3j%20Framework%20for%20Machine%20Learning%20Interatomic%20Potentials/tables/d988e280f7bf8514c2421d573140cfce9b77976879b3ae725ff507c5d65d4f95.jpg)

![eb212a69cb80afe9df865e4e71e1ca0aa1f6f686d39f78b9c8f04f1949d91865.jpg](../icml_results/5349_A%20Cartesian-3j%20Framework%20for%20Machine%20Learning%20Interatomic%20Potentials/tables/eb212a69cb80afe9df865e4e71e1ca0aa1f6f686d39f78b9c8f04f1949d91865.jpg)

<a id="paper-5302"></a>

## MORE: A Multilingual Document Parsing Benchmark and Evaluation


### Images

![20bd3ffa7e0e8f9df857508c55f065dbc972a9594163d7aeeb55cafe2ff33263.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/images/20bd3ffa7e0e8f9df857508c55f065dbc972a9594163d7aeeb55cafe2ff33263.jpg)

![2639dc85217f95ee4e7f678a1cbf88c6f0702ed3e3b28e5a747cab9be1de49ab.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/images/2639dc85217f95ee4e7f678a1cbf88c6f0702ed3e3b28e5a747cab9be1de49ab.jpg)

![28919a0e7fb04898ac73780bef7b0c21ffb53852a950919165f4eeef2141f4e4.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/images/28919a0e7fb04898ac73780bef7b0c21ffb53852a950919165f4eeef2141f4e4.jpg)

![33c9a80f471d4d982ef75849cb1a2d883e9899f1732be7805794774bef72ff8f.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/images/33c9a80f471d4d982ef75849cb1a2d883e9899f1732be7805794774bef72ff8f.jpg)

![624443d95aa42a509cb48ddbe4022327cca2617563e88eace6c70f537fa32eba.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/images/624443d95aa42a509cb48ddbe4022327cca2617563e88eace6c70f537fa32eba.jpg)

![8d02a5450ca3bb08cdc296ef91edac0cdc246ce349d5889ecd7f4a2e43158061.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/images/8d02a5450ca3bb08cdc296ef91edac0cdc246ce349d5889ecd7f4a2e43158061.jpg)

![b51d02d94a65d1427d3ae8d4acaa38581879f16a9701ec23359a9316a455c69f.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/images/b51d02d94a65d1427d3ae8d4acaa38581879f16a9701ec23359a9316a455c69f.jpg)

![f4b61835e7687ab01c0bd7fbcdd805a55d475e425d1d840db779f5a2f6fba261.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/images/f4b61835e7687ab01c0bd7fbcdd805a55d475e425d1d840db779f5a2f6fba261.jpg)

### Tables

![0215d49fb0bcac7cbc4cc766e2d1f6346c026cb2dd895cab3ef3f32b99b346fb.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/0215d49fb0bcac7cbc4cc766e2d1f6346c026cb2dd895cab3ef3f32b99b346fb.jpg)

![0d19a7e5f851b7b890e9381f702e826fd588f2b7ac10a0f3abf040e5b18bd409.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/0d19a7e5f851b7b890e9381f702e826fd588f2b7ac10a0f3abf040e5b18bd409.jpg)

![1156e72ca03d1083c597e708241a0118fa0652e609c08334ec82064299ed163f.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/1156e72ca03d1083c597e708241a0118fa0652e609c08334ec82064299ed163f.jpg)

![1e955c6464d7b5d56eeaac1bd0bdcf4b2cecb91a365ca7b34996791f48e56d6c.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/1e955c6464d7b5d56eeaac1bd0bdcf4b2cecb91a365ca7b34996791f48e56d6c.jpg)

![247f6abc252f5cf3bd5afbd520f2fccfac5190052dcb1c0f4a653397767c2383.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/247f6abc252f5cf3bd5afbd520f2fccfac5190052dcb1c0f4a653397767c2383.jpg)

![2775e5b5a9e6345cc5822b161df5a1d95ad93ce8d662ee666a16382121b52ec9.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/2775e5b5a9e6345cc5822b161df5a1d95ad93ce8d662ee666a16382121b52ec9.jpg)

![423bd2e266373cabab8c99f86fb3323ef2b6db6d9e5af307cb55a5a9f7b5c3ad.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/423bd2e266373cabab8c99f86fb3323ef2b6db6d9e5af307cb55a5a9f7b5c3ad.jpg)

![5948eefcabf1b1e05f00270b48c9a737e39e9b5eebcb389d8dc423b2d12be15f.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/5948eefcabf1b1e05f00270b48c9a737e39e9b5eebcb389d8dc423b2d12be15f.jpg)

![5977188ef62279bdd7f53633960e35c3913419d543fd4f3e1e032dd6cff71d42.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/5977188ef62279bdd7f53633960e35c3913419d543fd4f3e1e032dd6cff71d42.jpg)

![6e748af85543993fc969348b391a97b63134ae1499024db200787b17c484febf.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/6e748af85543993fc969348b391a97b63134ae1499024db200787b17c484febf.jpg)

![82c24b1c774173382d0b31ee51d9cb2c45b436f7b0eda734cac349f4d625a858.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/82c24b1c774173382d0b31ee51d9cb2c45b436f7b0eda734cac349f4d625a858.jpg)

![87bd04d60272e629f1df82acf6f3752d8fbf18d40bfdeedc472073a62f8b8a07.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/87bd04d60272e629f1df82acf6f3752d8fbf18d40bfdeedc472073a62f8b8a07.jpg)

![8b68a271201a3eec0c07f9e0fa85ec26fdb33ecc7c76f7142a54642587cfdf64.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/8b68a271201a3eec0c07f9e0fa85ec26fdb33ecc7c76f7142a54642587cfdf64.jpg)

![9204a44cedc93e5c058280fea27579868a60fe8eed072f62aead16a8569d227f.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/9204a44cedc93e5c058280fea27579868a60fe8eed072f62aead16a8569d227f.jpg)

![a0cd6429ea87814ef11f36d10d13cebcff6709392c4055ebb71d822ef0fef45e.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/a0cd6429ea87814ef11f36d10d13cebcff6709392c4055ebb71d822ef0fef45e.jpg)

![af94ef3ef2c7192dc69f606a85201bf787812eb8ce0f23f6c5def99f074cd331.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/af94ef3ef2c7192dc69f606a85201bf787812eb8ce0f23f6c5def99f074cd331.jpg)

![bd1e70b3e1ce48cca98e3aa6b1a52d5ce4744446230092add396352e49c86141.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/bd1e70b3e1ce48cca98e3aa6b1a52d5ce4744446230092add396352e49c86141.jpg)

![ccb4e4bb75d5394207177397d3540234a77d4633b13e0638e095d368a092a3cf.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/ccb4e4bb75d5394207177397d3540234a77d4633b13e0638e095d368a092a3cf.jpg)

![ccba2e968d0505cabf6bea2f829a6b7a1952b2228b72033135f215b576bb298d.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/ccba2e968d0505cabf6bea2f829a6b7a1952b2228b72033135f215b576bb298d.jpg)

![d9aeef0bdccc8646154e173528f8b9776539aae6795751496d65045b56c2b1ff.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/d9aeef0bdccc8646154e173528f8b9776539aae6795751496d65045b56c2b1ff.jpg)

![d9f965505d1912c8c3a43a1440e74ccb2491a76b1d4f42bb4404249e4a2cf798.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/d9f965505d1912c8c3a43a1440e74ccb2491a76b1d4f42bb4404249e4a2cf798.jpg)

![e03403d1abe28f375b28dbf6370c0b1adbedb31fc2698e25a615762f65a53ce7.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/e03403d1abe28f375b28dbf6370c0b1adbedb31fc2698e25a615762f65a53ce7.jpg)

![f8caa1a89ebcf879ff778d4dccf6d0e07623e654d7ab56a66ab9b742cac87930.jpg](../icml_results/5350_MORE_%20A%20Multilingual%20Document%20Parsing%20Benchmark%20and%20Evaluation/tables/f8caa1a89ebcf879ff778d4dccf6d0e07623e654d7ab56a66ab9b742cac87930.jpg)

<a id="paper-5303"></a>

## Neural-Inspired Modeling of Auditory Selection and Compensation for Audio-Visual Speech Separation


### Images

![1cf1acf54552651a523233f22112c97f82665369167ae96510785ab81feaced3.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/images/1cf1acf54552651a523233f22112c97f82665369167ae96510785ab81feaced3.jpg)

![394747291e99e55577199cba1643ed80e5950487c154666544a877270aac6633.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/images/394747291e99e55577199cba1643ed80e5950487c154666544a877270aac6633.jpg)

![5aa73fbc753d36fb7c9aaebb309286d682476279b9a5ca329523d13f0d3d473e.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/images/5aa73fbc753d36fb7c9aaebb309286d682476279b9a5ca329523d13f0d3d473e.jpg)

![7a2b93d95ecee2e57d0e7cdaf420d351c186657ac2c187034b79f8c91fbcde1b.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/images/7a2b93d95ecee2e57d0e7cdaf420d351c186657ac2c187034b79f8c91fbcde1b.jpg)

![8b0f581c94bbcf99ee0d8a9a34998c8afeb44ecc73637fde785197eee2990fd9.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/images/8b0f581c94bbcf99ee0d8a9a34998c8afeb44ecc73637fde785197eee2990fd9.jpg)

![def9dcaa82f1625fc463022ab711df564f8d449b0a18c46ae1119bdb13c39076.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/images/def9dcaa82f1625fc463022ab711df564f8d449b0a18c46ae1119bdb13c39076.jpg)

![e30c78309788cae0ce985b909bfb20c30e94ebe70459e790f58853b4c7d9e21a.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/images/e30c78309788cae0ce985b909bfb20c30e94ebe70459e790f58853b4c7d9e21a.jpg)

### Tables

![0702e40b31117c42565dc96d1b29f767583184baff497ea08537b1da8bca1b16.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/0702e40b31117c42565dc96d1b29f767583184baff497ea08537b1da8bca1b16.jpg)

![0f9edbef2c4a00d3088c17599d38102467812b0b2c8d201a22badf8ea7bcac30.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/0f9edbef2c4a00d3088c17599d38102467812b0b2c8d201a22badf8ea7bcac30.jpg)

![1423d3db8b015a0d270ee22b2352b08ddff364d2158dd8c430e1e05862993885.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/1423d3db8b015a0d270ee22b2352b08ddff364d2158dd8c430e1e05862993885.jpg)

![149f9c356aba18fcb70278d7ceb93824e6ac6ea731c7b46038be8a52503876ab.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/149f9c356aba18fcb70278d7ceb93824e6ac6ea731c7b46038be8a52503876ab.jpg)

![58b1d025c94dcad1b1a170c87abd539c73a6b631ff2bcfccafe8d034d745d08a.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/58b1d025c94dcad1b1a170c87abd539c73a6b631ff2bcfccafe8d034d745d08a.jpg)

![64521cb4da982017f40051281290d3e261f6c8e8aef21b5a9e91b96181465171.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/64521cb4da982017f40051281290d3e261f6c8e8aef21b5a9e91b96181465171.jpg)

![8839ef82b25c153b22a78b9f53689f70f81c91b054c619d3ef1e243374aaa114.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/8839ef82b25c153b22a78b9f53689f70f81c91b054c619d3ef1e243374aaa114.jpg)

![8bde87c6e08c82338d85946a9534e83f41c1682ffcb6b22b194e3dc3d17e8f15.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/8bde87c6e08c82338d85946a9534e83f41c1682ffcb6b22b194e3dc3d17e8f15.jpg)

![8fcaf6100631dacc401426fd699b1b6c30c2cafb5bdcd814dea4f2e7bced7692.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/8fcaf6100631dacc401426fd699b1b6c30c2cafb5bdcd814dea4f2e7bced7692.jpg)

![99ada54a039fe37b51febf8b228ed328392ab8ff0fa8cda1d2461460650498f4.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/99ada54a039fe37b51febf8b228ed328392ab8ff0fa8cda1d2461460650498f4.jpg)

![9bcac58987974bd9bf129ee732a59717df17d325c6fba5e0032c245e05b3c790.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/9bcac58987974bd9bf129ee732a59717df17d325c6fba5e0032c245e05b3c790.jpg)

![9bdafe2806e7ebca825cf6f3a68dbe0b56da316d83e61e9e9b47ed51b50ffe5d.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/9bdafe2806e7ebca825cf6f3a68dbe0b56da316d83e61e9e9b47ed51b50ffe5d.jpg)

![a43d6cb33f014366d323c394c2460f35a453aa73a4abd050dd305ab4e1eee333.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/a43d6cb33f014366d323c394c2460f35a453aa73a4abd050dd305ab4e1eee333.jpg)

![a97e63991f76b8f0b56fc8d03df8f450d57b33ad4288f3cb7ce7573a4f8b0d9c.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/a97e63991f76b8f0b56fc8d03df8f450d57b33ad4288f3cb7ce7573a4f8b0d9c.jpg)

![b421495425b0f4fa0f4ae7b474a4b7959620fc4f94c84f09841d27c91f310678.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/b421495425b0f4fa0f4ae7b474a4b7959620fc4f94c84f09841d27c91f310678.jpg)

![b7a9f78d87622f4a86436b5026bd8303438ed7497b89822c15529648beadf5b6.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/b7a9f78d87622f4a86436b5026bd8303438ed7497b89822c15529648beadf5b6.jpg)

![bcdc5617e344d0381b702e600e0a7d3cd71ba6fec079080b3200922f8d64c315.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/bcdc5617e344d0381b702e600e0a7d3cd71ba6fec079080b3200922f8d64c315.jpg)

![c0e8959cde316c2930445478be165820b0a7e168f7ebddfa1d9081bca8978d96.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/c0e8959cde316c2930445478be165820b0a7e168f7ebddfa1d9081bca8978d96.jpg)

![f6de11628f96dcf4c41e0c79c77cbfe507699671094368862fb114053015b204.jpg](../icml_results/5351_Neural-Inspired%20Modeling%20of%20Auditory%20Selection%20and%20Compensation%20for%20Audio-Visual%20Speech%20Separation/tables/f6de11628f96dcf4c41e0c79c77cbfe507699671094368862fb114053015b204.jpg)

<a id="paper-5304"></a>

## Feature Resemblance: Towards a Theoretical Understanding of Analogical Reasoning in Transformers


### Images

![4c1ae612386cff6484837f4b822d60c5f1f578e3fd4d9512982fd73c737bb1b5.jpg](../icml_results/5352_Feature%20Resemblance_%20Towards%20a%20Theoretical%20Understanding%20of%20Analogical%20Reasoning%20in%20Transformers/images/4c1ae612386cff6484837f4b822d60c5f1f578e3fd4d9512982fd73c737bb1b5.jpg)

### Tables

![0d53b2e604dad8d821014a5f3d9238bcba9fbd2306ea69c7233570c4fd781cc0.jpg](../icml_results/5352_Feature%20Resemblance_%20Towards%20a%20Theoretical%20Understanding%20of%20Analogical%20Reasoning%20in%20Transformers/tables/0d53b2e604dad8d821014a5f3d9238bcba9fbd2306ea69c7233570c4fd781cc0.jpg)

![41eca93128eb9bc8e371bea0966b3bbe75b6d75da1e7c484f8d7454e2b74685e.jpg](../icml_results/5352_Feature%20Resemblance_%20Towards%20a%20Theoretical%20Understanding%20of%20Analogical%20Reasoning%20in%20Transformers/tables/41eca93128eb9bc8e371bea0966b3bbe75b6d75da1e7c484f8d7454e2b74685e.jpg)

![42a75c9bdc1f5db9d43a7b90561a99386f11d46e2bcef4c60bf599d092587cb6.jpg](../icml_results/5352_Feature%20Resemblance_%20Towards%20a%20Theoretical%20Understanding%20of%20Analogical%20Reasoning%20in%20Transformers/tables/42a75c9bdc1f5db9d43a7b90561a99386f11d46e2bcef4c60bf599d092587cb6.jpg)

![4b2adda8133e2456b7f2044322c8c3e81b3a4ef5b734989da239dd55efcca735.jpg](../icml_results/5352_Feature%20Resemblance_%20Towards%20a%20Theoretical%20Understanding%20of%20Analogical%20Reasoning%20in%20Transformers/tables/4b2adda8133e2456b7f2044322c8c3e81b3a4ef5b734989da239dd55efcca735.jpg)

![636ebe2f707dfe605d306137b9664eac668f77ca177aeb46746a69acb36edd5d.jpg](../icml_results/5352_Feature%20Resemblance_%20Towards%20a%20Theoretical%20Understanding%20of%20Analogical%20Reasoning%20in%20Transformers/tables/636ebe2f707dfe605d306137b9664eac668f77ca177aeb46746a69acb36edd5d.jpg)

![6e1ee2b6031e2bd417f62ea44909f637a565c2148e6003c012f0bbb177a8630c.jpg](../icml_results/5352_Feature%20Resemblance_%20Towards%20a%20Theoretical%20Understanding%20of%20Analogical%20Reasoning%20in%20Transformers/tables/6e1ee2b6031e2bd417f62ea44909f637a565c2148e6003c012f0bbb177a8630c.jpg)

![9251d41f593e77e8aea78c5e98b48400c5ebb1f307e0cd76aec1ff2e34f452f7.jpg](../icml_results/5352_Feature%20Resemblance_%20Towards%20a%20Theoretical%20Understanding%20of%20Analogical%20Reasoning%20in%20Transformers/tables/9251d41f593e77e8aea78c5e98b48400c5ebb1f307e0cd76aec1ff2e34f452f7.jpg)

<a id="paper-5305"></a>

## From Internal Diagnosis to External Auditing: A VLM-Driven Paradigm for Data-Free Online Backdoor Defense


### Images

![0a26a7d9eed74206ce1c2793edf1f8f5f6c62ba7e5aa317a4dfad9733bdc4386.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/images/0a26a7d9eed74206ce1c2793edf1f8f5f6c62ba7e5aa317a4dfad9733bdc4386.jpg)

![1aee2a5a37a26a212e2dce32dedd977681efabe2015e08791fdf61f13a1ed9da.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/images/1aee2a5a37a26a212e2dce32dedd977681efabe2015e08791fdf61f13a1ed9da.jpg)

![1afa052eeb16c68435c1846b97189dc15d4aaf0e290f399cc96b0eb3b9f5cc73.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/images/1afa052eeb16c68435c1846b97189dc15d4aaf0e290f399cc96b0eb3b9f5cc73.jpg)

![31caef18d7b50138f3591fcdc8204820abf1b5e29b399f70ed7b523ea0a6a4fe.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/images/31caef18d7b50138f3591fcdc8204820abf1b5e29b399f70ed7b523ea0a6a4fe.jpg)

![3a99932324d19755b4d64d1815e9a79edba19ae416f8eea7feba40ce337d6064.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/images/3a99932324d19755b4d64d1815e9a79edba19ae416f8eea7feba40ce337d6064.jpg)

![44855987e8cfb1359047f5d6ebca14ae5205495eff73d682a204595810a4b701.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/images/44855987e8cfb1359047f5d6ebca14ae5205495eff73d682a204595810a4b701.jpg)

![552610cf8ad97e92e3cc277c746a3e02f71364ca843d807f3791a7f7647b4826.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/images/552610cf8ad97e92e3cc277c746a3e02f71364ca843d807f3791a7f7647b4826.jpg)

![5c2d9b97e2155c148571da6ddab067ff715721812309a9d27fd36a295b7259fc.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/images/5c2d9b97e2155c148571da6ddab067ff715721812309a9d27fd36a295b7259fc.jpg)

![b05b974fe6f078c2c4799e94bc90997d4c7f41109aa80340ea0f2995f5c6aaea.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/images/b05b974fe6f078c2c4799e94bc90997d4c7f41109aa80340ea0f2995f5c6aaea.jpg)

![d8f169ab34d3ec56af03d6ddee73823714cd5fb899140f72b882bdb32064bc51.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/images/d8f169ab34d3ec56af03d6ddee73823714cd5fb899140f72b882bdb32064bc51.jpg)

### Tables

![0743b9149912bd750a425495c31347e0435b4c9c40a939e1ad32e406bc1068d3.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/0743b9149912bd750a425495c31347e0435b4c9c40a939e1ad32e406bc1068d3.jpg)

![1fd10cd8491b5f093ea3d3931ad72934248b2fd8372ff5564e2fb2e94c2e2ce8.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/1fd10cd8491b5f093ea3d3931ad72934248b2fd8372ff5564e2fb2e94c2e2ce8.jpg)

![2173e013e27383d319ac69d4514f25f85b7b5b9cc2a264516734bdf408fb209c.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/2173e013e27383d319ac69d4514f25f85b7b5b9cc2a264516734bdf408fb209c.jpg)

![21b3e9ad525e95d65432c900c546584f76b64b98859ba39bf4fa0b79fd2134da.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/21b3e9ad525e95d65432c900c546584f76b64b98859ba39bf4fa0b79fd2134da.jpg)

![418356e219955e3913f6e4c38c9670f96c6cca3e44006e5f18dcc8b55966c96f.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/418356e219955e3913f6e4c38c9670f96c6cca3e44006e5f18dcc8b55966c96f.jpg)

![4189ecd752be4e06b55ac5834f6c24eff3074be54164e30a2d650aba8cf4d253.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/4189ecd752be4e06b55ac5834f6c24eff3074be54164e30a2d650aba8cf4d253.jpg)

![53654656ec96c0d1dca4c340604718949cd9ced27fb26581e4230a182a81164d.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/53654656ec96c0d1dca4c340604718949cd9ced27fb26581e4230a182a81164d.jpg)

![633fb6733bb8f8ad74fc19bcf449d4bbf5698094ae3e44c91f733a452b1a6fb0.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/633fb6733bb8f8ad74fc19bcf449d4bbf5698094ae3e44c91f733a452b1a6fb0.jpg)

![6c47a07b230db0509e9c4f74f4565389ba627c48087943d70f35bc5de4d07be0.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/6c47a07b230db0509e9c4f74f4565389ba627c48087943d70f35bc5de4d07be0.jpg)

![7d7b128155a51eaf6a8550ee7c54e307a2a5f8b6d6a091d6ba81cb41c2b42833.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/7d7b128155a51eaf6a8550ee7c54e307a2a5f8b6d6a091d6ba81cb41c2b42833.jpg)

![8f185bfd6261014c94af1cbc8ca856311dc428721de614f6fdf3df1953ab2554.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/8f185bfd6261014c94af1cbc8ca856311dc428721de614f6fdf3df1953ab2554.jpg)

![a29c2a5cfb61e949bbb1c4603c0f97ab6df45f576e08f90c7632ac9cd7786cce.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/a29c2a5cfb61e949bbb1c4603c0f97ab6df45f576e08f90c7632ac9cd7786cce.jpg)

![aa1ed07cff8c5a72f2ef7bedd96a5f439746ceb6f12b099b99ad92915206dd3d.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/aa1ed07cff8c5a72f2ef7bedd96a5f439746ceb6f12b099b99ad92915206dd3d.jpg)

![c50d930c5e5508fc02ab5a02ee1f031d10b9d89ff91f5b708f67fb9e37297981.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/c50d930c5e5508fc02ab5a02ee1f031d10b9d89ff91f5b708f67fb9e37297981.jpg)

![cd58f2dd8c86bf004294d63c1ab1d2bb7970be065130386fb8cfd4e1bd68b4c5.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/cd58f2dd8c86bf004294d63c1ab1d2bb7970be065130386fb8cfd4e1bd68b4c5.jpg)

![cf60c48b082734093dd7efe7e8c89298646c92ea9ad2a2e5fb6b8433f4a346c1.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/cf60c48b082734093dd7efe7e8c89298646c92ea9ad2a2e5fb6b8433f4a346c1.jpg)

![d1b630af5609ec9c40e4e22a20217da29509f0f4424caf0dce8a35fe9c95c2cb.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/d1b630af5609ec9c40e4e22a20217da29509f0f4424caf0dce8a35fe9c95c2cb.jpg)

![da432596082eecf9fe0f128765f1ac2089b4daf7149de4d3c4794f24c0c59a1b.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/da432596082eecf9fe0f128765f1ac2089b4daf7149de4d3c4794f24c0c59a1b.jpg)

![ff7486a3eb695c57bf7a1424254cce1f8b71a10002cfb19c613151c3c6d47bb2.jpg](../icml_results/5353_From%20Internal%20Diagnosis%20to%20External%20Auditing_%20A%20VLM-Driven%20Paradigm%20for%20Data-Free%20Online%20Backdoor%20De/tables/ff7486a3eb695c57bf7a1424254cce1f8b71a10002cfb19c613151c3c6d47bb2.jpg)

<a id="paper-5306"></a>

## Resting Neurons, Active Insights: Robustifying Activation Sparsity in LLMs via Spontaneity


### Images

![1430006840112056ac2ffb30f075cf736358385e786aa72dff23f10cef3247da.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/images/1430006840112056ac2ffb30f075cf736358385e786aa72dff23f10cef3247da.jpg)

![1a6a8ba55f0942488f58762fbb63e67ab9961a804de03fc947ba5e9c3673bf38.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/images/1a6a8ba55f0942488f58762fbb63e67ab9961a804de03fc947ba5e9c3673bf38.jpg)

![283657842cc94db4b641635258a3f96ea40b80b4e370b72c800b1f4d4047c44b.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/images/283657842cc94db4b641635258a3f96ea40b80b4e370b72c800b1f4d4047c44b.jpg)

![384d53631a6e07ad5aef00729983220bc0b43ef0fb62733d7f5262c0e5c3bd8f.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/images/384d53631a6e07ad5aef00729983220bc0b43ef0fb62733d7f5262c0e5c3bd8f.jpg)

![3cfa22d3c7d22c05b008adffbdfbb0ee5d491f3f266c1aba6ec5aa2c73abb7b3.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/images/3cfa22d3c7d22c05b008adffbdfbb0ee5d491f3f266c1aba6ec5aa2c73abb7b3.jpg)

![3d7684658c01f7a476214866cf27c1e568092731160f9dcae5d4ae50aa2ecc09.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/images/3d7684658c01f7a476214866cf27c1e568092731160f9dcae5d4ae50aa2ecc09.jpg)

![45736c7eaef05eb5eca2038279acc2965bcdf1b47d21e6112a6d745c889f57e2.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/images/45736c7eaef05eb5eca2038279acc2965bcdf1b47d21e6112a6d745c889f57e2.jpg)

![5c44204c1c282a1209c5258a6da00b1efec3b6f1c4ef404a26115f938a1e1380.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/images/5c44204c1c282a1209c5258a6da00b1efec3b6f1c4ef404a26115f938a1e1380.jpg)

![63a30afc766d8e9d40b9dec195f230d842b15c3111c249aebd346e054b0f2a26.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/images/63a30afc766d8e9d40b9dec195f230d842b15c3111c249aebd346e054b0f2a26.jpg)

![6b90dc7c2625d51f039da28a6a675aa65ba65735a18bbd54a17efb8409fd0d83.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/images/6b90dc7c2625d51f039da28a6a675aa65ba65735a18bbd54a17efb8409fd0d83.jpg)

![6d0d143cfc1232cd3d322ffe45fa543dcb8c973743e09e394cda2598dd1ae812.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/images/6d0d143cfc1232cd3d322ffe45fa543dcb8c973743e09e394cda2598dd1ae812.jpg)

![9d0b8b00402b14101830ae2a8b5694497edf97aa1c10fdb7e711223d119116ba.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/images/9d0b8b00402b14101830ae2a8b5694497edf97aa1c10fdb7e711223d119116ba.jpg)

![a185b2cb5ebe702a1fd81368b318e81b5a3518866d4403dbc10f4208707dc1f7.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/images/a185b2cb5ebe702a1fd81368b318e81b5a3518866d4403dbc10f4208707dc1f7.jpg)

### Tables

![174666467fe19fcf2f1efa06ed973e30a6d5edb3b42757543e293084d7feaa74.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/tables/174666467fe19fcf2f1efa06ed973e30a6d5edb3b42757543e293084d7feaa74.jpg)

![25bbd96bea185ef335f784265f5d45ef39825f05ed2b9238eff0604d9629d878.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/tables/25bbd96bea185ef335f784265f5d45ef39825f05ed2b9238eff0604d9629d878.jpg)

![26356f7e155ec07061391ed4c431cc19e283261ac612b13608eb330df3b9a73a.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/tables/26356f7e155ec07061391ed4c431cc19e283261ac612b13608eb330df3b9a73a.jpg)

![3e841866295e0830c23cc641c50baef54d79207f24c168bab08065dca0fa98ef.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/tables/3e841866295e0830c23cc641c50baef54d79207f24c168bab08065dca0fa98ef.jpg)

![52062247c232c4d3ca4654d50af782ed926f54cf47236b10c1040d374a0cf02e.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/tables/52062247c232c4d3ca4654d50af782ed926f54cf47236b10c1040d374a0cf02e.jpg)

![599df8c9438e677f159c03e19ff46c39cb1d69c00edd834de3f4f818c190947e.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/tables/599df8c9438e677f159c03e19ff46c39cb1d69c00edd834de3f4f818c190947e.jpg)

![5d9d7f8d2882ecb8b76fe089139e6951c9100884a3a916cc376b576857ea5444.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/tables/5d9d7f8d2882ecb8b76fe089139e6951c9100884a3a916cc376b576857ea5444.jpg)

![73571b2635dc05a0c9d1bd10c22ea1c3b074f9c7de4014ca8988f31fb37c5c27.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/tables/73571b2635dc05a0c9d1bd10c22ea1c3b074f9c7de4014ca8988f31fb37c5c27.jpg)

![82ed654d2a6d50cab13b1dd830480da08121ec2d0878620a2fa40d127d14c879.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/tables/82ed654d2a6d50cab13b1dd830480da08121ec2d0878620a2fa40d127d14c879.jpg)

![dcdae25d1ea14e2c652c83b1f230e3606aa3952a118da8a1a53a88f1a90ec49a.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/tables/dcdae25d1ea14e2c652c83b1f230e3606aa3952a118da8a1a53a88f1a90ec49a.jpg)

![e5982aa06b1904bacadc9dc59bfbacdfd89b000d2d53894c78bbbafc9ee7926d.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/tables/e5982aa06b1904bacadc9dc59bfbacdfd89b000d2d53894c78bbbafc9ee7926d.jpg)

![fe496d000ec2c2ce7b4f00b33001d6f03b64e4609e099dddad8437f87ce9f7ba.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/tables/fe496d000ec2c2ce7b4f00b33001d6f03b64e4609e099dddad8437f87ce9f7ba.jpg)

![ff82ad9ec4687beab078ab58b39b4c28550e6f3fdba3912e12ab8d5d5d991c2d.jpg](../icml_results/5354_Resting%20Neurons%2C%20Active%20Insights_%20Robustifying%20Activation%20Sparsity%20in%20LLMs%20via%20Spontaneity/tables/ff82ad9ec4687beab078ab58b39b4c28550e6f3fdba3912e12ab8d5d5d991c2d.jpg)

<a id="paper-5307"></a>

## Theoretical Analysis of Sparse Optimization with Reparameterization, Weight Decay, and Adaptive Learning Rate


### Images

![3467df6c89f1b76aae0157af0d1281b5d847cf881154a4c2bbcda5595046e58c.jpg](../icml_results/5355_Theoretical%20Analysis%20of%20Sparse%20Optimization%20with%20Reparameterization%2C%20Weight%20Decay%2C%20and%20Adaptive%20Lear/images/3467df6c89f1b76aae0157af0d1281b5d847cf881154a4c2bbcda5595046e58c.jpg)

![5d2aec69764091bc5e5bd35a5bfcf945e3432ca827a5cead419db35b7c3b109d.jpg](../icml_results/5355_Theoretical%20Analysis%20of%20Sparse%20Optimization%20with%20Reparameterization%2C%20Weight%20Decay%2C%20and%20Adaptive%20Lear/images/5d2aec69764091bc5e5bd35a5bfcf945e3432ca827a5cead419db35b7c3b109d.jpg)

![84d1e4efb6e3d9a6190b1a8fb819a96e6a74d66ac15bad8f736b9295aabd8650.jpg](../icml_results/5355_Theoretical%20Analysis%20of%20Sparse%20Optimization%20with%20Reparameterization%2C%20Weight%20Decay%2C%20and%20Adaptive%20Lear/images/84d1e4efb6e3d9a6190b1a8fb819a96e6a74d66ac15bad8f736b9295aabd8650.jpg)

![9ff51be74bf07ef599ac9c940961cfe0a2b7db779bc674da70d21df858d3289f.jpg](../icml_results/5355_Theoretical%20Analysis%20of%20Sparse%20Optimization%20with%20Reparameterization%2C%20Weight%20Decay%2C%20and%20Adaptive%20Lear/images/9ff51be74bf07ef599ac9c940961cfe0a2b7db779bc674da70d21df858d3289f.jpg)

![d8d845da1176275df8b55b5abfb1a309a9ee2c4118c1351a4505fdc57bb65f04.jpg](../icml_results/5355_Theoretical%20Analysis%20of%20Sparse%20Optimization%20with%20Reparameterization%2C%20Weight%20Decay%2C%20and%20Adaptive%20Lear/images/d8d845da1176275df8b55b5abfb1a309a9ee2c4118c1351a4505fdc57bb65f04.jpg)

### Tables

![658a12716dbf8f40fa7880618a6b80a2d8a40345281310ae246b6f0be3416a3b.jpg](../icml_results/5355_Theoretical%20Analysis%20of%20Sparse%20Optimization%20with%20Reparameterization%2C%20Weight%20Decay%2C%20and%20Adaptive%20Lear/tables/658a12716dbf8f40fa7880618a6b80a2d8a40345281310ae246b6f0be3416a3b.jpg)

![66d63c980bca3321f7f9a67ce10027afb448854ae0921fef9ec67e324cf24fa3.jpg](../icml_results/5355_Theoretical%20Analysis%20of%20Sparse%20Optimization%20with%20Reparameterization%2C%20Weight%20Decay%2C%20and%20Adaptive%20Lear/tables/66d63c980bca3321f7f9a67ce10027afb448854ae0921fef9ec67e324cf24fa3.jpg)

![d025c197f1c9b459ceca9725a5bba625bc988d30f2ac97ef08b97229e8dc2678.jpg](../icml_results/5355_Theoretical%20Analysis%20of%20Sparse%20Optimization%20with%20Reparameterization%2C%20Weight%20Decay%2C%20and%20Adaptive%20Lear/tables/d025c197f1c9b459ceca9725a5bba625bc988d30f2ac97ef08b97229e8dc2678.jpg)

<a id="paper-5308"></a>

## Weak-to-Strong Generalization via Bregman Bias–Variance Decomposition


### Images

![207dfffd67f0e21e3733daf63efd0e75fb84e4c40d1093b8606645eb179cebd0.jpg](../icml_results/5356_Weak-to-Strong%20Generalization%20via%20Bregman%20Bias%E2%80%93Variance%20Decomposition/images/207dfffd67f0e21e3733daf63efd0e75fb84e4c40d1093b8606645eb179cebd0.jpg)

![619e2f029b655c9e6bf1699ff17d5be79d7a34076da7e3374b77da6b700a3325.jpg](../icml_results/5356_Weak-to-Strong%20Generalization%20via%20Bregman%20Bias%E2%80%93Variance%20Decomposition/images/619e2f029b655c9e6bf1699ff17d5be79d7a34076da7e3374b77da6b700a3325.jpg)

![65f5b34e416901a68de69594381685db81e54d6d2726d7971d8bca4ae40b8e64.jpg](../icml_results/5356_Weak-to-Strong%20Generalization%20via%20Bregman%20Bias%E2%80%93Variance%20Decomposition/images/65f5b34e416901a68de69594381685db81e54d6d2726d7971d8bca4ae40b8e64.jpg)

![72987b3f3af3d382ad67b9fb769f6c798b039e6778ded56b252453845f9961d1.jpg](../icml_results/5356_Weak-to-Strong%20Generalization%20via%20Bregman%20Bias%E2%80%93Variance%20Decomposition/images/72987b3f3af3d382ad67b9fb769f6c798b039e6778ded56b252453845f9961d1.jpg)

![73473a0cef1b6a50be8679bb2c43e2ccf25e52772ab62836d2ea1305015f604d.jpg](../icml_results/5356_Weak-to-Strong%20Generalization%20via%20Bregman%20Bias%E2%80%93Variance%20Decomposition/images/73473a0cef1b6a50be8679bb2c43e2ccf25e52772ab62836d2ea1305015f604d.jpg)

![b1fafa230e43737c67ed53eeffa4f327cd53227d5b914bcf921d0497f9676c29.jpg](../icml_results/5356_Weak-to-Strong%20Generalization%20via%20Bregman%20Bias%E2%80%93Variance%20Decomposition/images/b1fafa230e43737c67ed53eeffa4f327cd53227d5b914bcf921d0497f9676c29.jpg)

![b70f82624ae33fd2158ca3393aff85f89448c3ef095cdcd2742f6875dfa4f63e.jpg](../icml_results/5356_Weak-to-Strong%20Generalization%20via%20Bregman%20Bias%E2%80%93Variance%20Decomposition/images/b70f82624ae33fd2158ca3393aff85f89448c3ef095cdcd2742f6875dfa4f63e.jpg)

![b996949cbcf752606ee9481a2c2674c4de17f7fea1b3b289ca7e7344d923d09e.jpg](../icml_results/5356_Weak-to-Strong%20Generalization%20via%20Bregman%20Bias%E2%80%93Variance%20Decomposition/images/b996949cbcf752606ee9481a2c2674c4de17f7fea1b3b289ca7e7344d923d09e.jpg)

### Tables

![04f64b88b97f67407e7ad23525c62b39636390ed3e26a3334f53286ad3f7411d.jpg](../icml_results/5356_Weak-to-Strong%20Generalization%20via%20Bregman%20Bias%E2%80%93Variance%20Decomposition/tables/04f64b88b97f67407e7ad23525c62b39636390ed3e26a3334f53286ad3f7411d.jpg)

![8bf792c1ca8b1ba220aa4d6b795a814f67cffcd172d313bc22e755e6ea1e8ffb.jpg](../icml_results/5356_Weak-to-Strong%20Generalization%20via%20Bregman%20Bias%E2%80%93Variance%20Decomposition/tables/8bf792c1ca8b1ba220aa4d6b795a814f67cffcd172d313bc22e755e6ea1e8ffb.jpg)

![be52a57d992e1b4e32374321e27d4c37575e80ba0bb4df6b1dab3394f9ef690b.jpg](../icml_results/5356_Weak-to-Strong%20Generalization%20via%20Bregman%20Bias%E2%80%93Variance%20Decomposition/tables/be52a57d992e1b4e32374321e27d4c37575e80ba0bb4df6b1dab3394f9ef690b.jpg)

<a id="paper-5309"></a>

## Doubly Robust Distributionally Robust Offline Contextual Pricing


### Images

![8a9106dcd5fb4bd713608d087477343fa91a3b47906944843abde95c5c8ca1b5.jpg](../icml_results/5357_Doubly%20Robust%20Distributionally%20Robust%20Offline%20Contextual%20Pricing/images/8a9106dcd5fb4bd713608d087477343fa91a3b47906944843abde95c5c8ca1b5.jpg)

![c955f017b633627180f6ad5736e2e6ec9c8988c54e4527b5c92152673cbaee30.jpg](../icml_results/5357_Doubly%20Robust%20Distributionally%20Robust%20Offline%20Contextual%20Pricing/images/c955f017b633627180f6ad5736e2e6ec9c8988c54e4527b5c92152673cbaee30.jpg)

### Tables

![5e585bcffe2db18919fe080538fe48b3463d0ec1c9e53ad228b23abf707b8f0b.jpg](../icml_results/5357_Doubly%20Robust%20Distributionally%20Robust%20Offline%20Contextual%20Pricing/tables/5e585bcffe2db18919fe080538fe48b3463d0ec1c9e53ad228b23abf707b8f0b.jpg)

![61841bbe03ae0f834b453a2bcd0ee3c94455377b972fa839ae4dccd32a6de0fd.jpg](../icml_results/5357_Doubly%20Robust%20Distributionally%20Robust%20Offline%20Contextual%20Pricing/tables/61841bbe03ae0f834b453a2bcd0ee3c94455377b972fa839ae4dccd32a6de0fd.jpg)

<a id="paper-5310"></a>

## Unlocking Zero-Shot Geospatial Reasoning via Indirect Rewards


### Images

![00f9edbe59c708795604d9080abffbde3d73ed482602e90e9d468d7f661c0302.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/00f9edbe59c708795604d9080abffbde3d73ed482602e90e9d468d7f661c0302.jpg)

![02f85d691d244ece0ebb25ff51dd5fa45a55bbf9cac09584c5cf744a27bdf35e.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/02f85d691d244ece0ebb25ff51dd5fa45a55bbf9cac09584c5cf744a27bdf35e.jpg)

![13e7e6da7e353884e29e0379aec97fff33e4aaad469ad39fc11c4ba0d153e216.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/13e7e6da7e353884e29e0379aec97fff33e4aaad469ad39fc11c4ba0d153e216.jpg)

![142cb0eadeff7544d830a822fb759ae26b6e962a4697eb9473181a111e9b328d.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/142cb0eadeff7544d830a822fb759ae26b6e962a4697eb9473181a111e9b328d.jpg)

![1c1ed19a17c61fedd196b4225ba370cb72567c3031a50ccac5b660b74338e083.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/1c1ed19a17c61fedd196b4225ba370cb72567c3031a50ccac5b660b74338e083.jpg)

![3bd75537d17391087c4cb6021e47265b887b8f10022d96ecb1e8845c013c1bae.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/3bd75537d17391087c4cb6021e47265b887b8f10022d96ecb1e8845c013c1bae.jpg)

![3f2f927a2c5e4ac9449474c6248bec983a4b9f8985d396de6ca3afc11db7e660.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/3f2f927a2c5e4ac9449474c6248bec983a4b9f8985d396de6ca3afc11db7e660.jpg)

![51eba986df73c1a5bc8a902d0652a5dafa8fd56b46d1d08ea780aeabdefc133d.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/51eba986df73c1a5bc8a902d0652a5dafa8fd56b46d1d08ea780aeabdefc133d.jpg)

![5223a1c9e6c0ac95c69272543add1cc04ad95a5e127cd72f1e2c0e629d9010a2.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/5223a1c9e6c0ac95c69272543add1cc04ad95a5e127cd72f1e2c0e629d9010a2.jpg)

![6fa3c4554fc506fdd11173a58147fcdd14a5e8e952f846773bda09bc44f50e4b.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/6fa3c4554fc506fdd11173a58147fcdd14a5e8e952f846773bda09bc44f50e4b.jpg)

![75ac5dde328d98d9251bb0e5d1c299f69efba0527a47c7e2a70db155250f7168.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/75ac5dde328d98d9251bb0e5d1c299f69efba0527a47c7e2a70db155250f7168.jpg)

![76f491bceb425f330dc42ce4435c67ecdb604476c663c8f459a5fffc1d2a9617.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/76f491bceb425f330dc42ce4435c67ecdb604476c663c8f459a5fffc1d2a9617.jpg)

![80c5bf621d3b0b3e155afc6c7e1c609a5ba63d10a6b1aa83e336fba4d0160e5f.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/80c5bf621d3b0b3e155afc6c7e1c609a5ba63d10a6b1aa83e336fba4d0160e5f.jpg)

![80cb649c8f183596e5410c4f64981b02c4a54c56f6e91a2337e348ab616b7c78.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/80cb649c8f183596e5410c4f64981b02c4a54c56f6e91a2337e348ab616b7c78.jpg)

![81eacec9fb795749e115c8b0bd07cd8bbebe7a5e53da77bf8574029e7461c8ab.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/81eacec9fb795749e115c8b0bd07cd8bbebe7a5e53da77bf8574029e7461c8ab.jpg)

![8b1be26555c3c6bc6ec07684e78ecd3aed9a54fb2ea1f15fa361629854ab9d76.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/8b1be26555c3c6bc6ec07684e78ecd3aed9a54fb2ea1f15fa361629854ab9d76.jpg)

![8f05dd689ce71223052fae0ee7272934069a5330a38e5f0ad38578ad698c753d.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/8f05dd689ce71223052fae0ee7272934069a5330a38e5f0ad38578ad698c753d.jpg)

![ba88cf775559c7af676d3a6306ce9d9bbe5943d28ca949f8a002f7fc057794e4.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/ba88cf775559c7af676d3a6306ce9d9bbe5943d28ca949f8a002f7fc057794e4.jpg)

![bbb4c7a5d2f0f7ebcf02ce1f0c7118be66e04d91a193c69652bcbb117702b926.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/bbb4c7a5d2f0f7ebcf02ce1f0c7118be66e04d91a193c69652bcbb117702b926.jpg)

![bea7c1da2a722b944a4d19c76576b54e5ccb6e54f6d9119aeff8f2293b46ecbf.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/bea7c1da2a722b944a4d19c76576b54e5ccb6e54f6d9119aeff8f2293b46ecbf.jpg)

![bfbf35fc9fb620e978c05a7e3485f22f3a02145f352f641619988df521cd7226.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/bfbf35fc9fb620e978c05a7e3485f22f3a02145f352f641619988df521cd7226.jpg)

![cb9ea3f4ac0bded38fec73d519271867d568792571cec511d67c3d5e6dd87275.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/cb9ea3f4ac0bded38fec73d519271867d568792571cec511d67c3d5e6dd87275.jpg)

![cc52a74e9fe070e0fdf71b7cb358864cdcbd76d608365bfedbaadf681c4fd238.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/cc52a74e9fe070e0fdf71b7cb358864cdcbd76d608365bfedbaadf681c4fd238.jpg)

![d0e835e655688aab61d91ed3ba33926cbff68fbe48a0e69a77b350f3db1456d7.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/d0e835e655688aab61d91ed3ba33926cbff68fbe48a0e69a77b350f3db1456d7.jpg)

![d28612a6d2a1f3046ee42695583784e229f048faf94af98063e32409b9729d7e.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/d28612a6d2a1f3046ee42695583784e229f048faf94af98063e32409b9729d7e.jpg)

![df51ed72702261ce626e2ebc74f654b85ffdab26850f9af33fa866de66b966b8.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/df51ed72702261ce626e2ebc74f654b85ffdab26850f9af33fa866de66b966b8.jpg)

![ec6a540ed3ad68278d5bc9e3f3fbb11d7f4ddc4586b1614f0e5896dab722573f.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/ec6a540ed3ad68278d5bc9e3f3fbb11d7f4ddc4586b1614f0e5896dab722573f.jpg)

![f147f1e8919ab7d2831941d396673d84522d58d1de36a7030ca3a077ea64b9a1.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/f147f1e8919ab7d2831941d396673d84522d58d1de36a7030ca3a077ea64b9a1.jpg)

![fe770e49b1bfa3eb0d69a6e477bb886e4f14a034c1627998e8b54431064a427a.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/images/fe770e49b1bfa3eb0d69a6e477bb886e4f14a034c1627998e8b54431064a427a.jpg)

### Tables

![09ca9bc5908706c17280dba56585cff64d1311935a009d902ffbf10790c7cef8.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/tables/09ca9bc5908706c17280dba56585cff64d1311935a009d902ffbf10790c7cef8.jpg)

![0b39cda3067f0905c8eff05790b8e3be5e072763d3f35016f93b2b42753a9751.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/tables/0b39cda3067f0905c8eff05790b8e3be5e072763d3f35016f93b2b42753a9751.jpg)

![24f810e446ddd3bf813f2376230abb67bef5d4d1df4f35b99229a81b69339b33.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/tables/24f810e446ddd3bf813f2376230abb67bef5d4d1df4f35b99229a81b69339b33.jpg)

![475326768fcc98bb7236fd2b0164478d26202b21e9cbfee4633345d646f7ced3.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/tables/475326768fcc98bb7236fd2b0164478d26202b21e9cbfee4633345d646f7ced3.jpg)

![5d50e31b20657409e916668c41cabadd1b103cd59d4c57e81657fb457acf47c9.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/tables/5d50e31b20657409e916668c41cabadd1b103cd59d4c57e81657fb457acf47c9.jpg)

![663c63436650c178ca22d682fb1775c1fa3137f93117bb9e57487cfe96ece040.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/tables/663c63436650c178ca22d682fb1775c1fa3137f93117bb9e57487cfe96ece040.jpg)

![6e5107da04d326ed65f282929aa07ba0a1a47a152783b2008f6cd7274ff3ec67.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/tables/6e5107da04d326ed65f282929aa07ba0a1a47a152783b2008f6cd7274ff3ec67.jpg)

![7a6ac2c151a6d146e05741ad9aac0ccbad187847622662cf489cf8cbcb497864.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/tables/7a6ac2c151a6d146e05741ad9aac0ccbad187847622662cf489cf8cbcb497864.jpg)

![7ea8a163f8e5285525fa4edfb686c4d3c263b8424f69ccb2303fda124fbc49b0.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/tables/7ea8a163f8e5285525fa4edfb686c4d3c263b8424f69ccb2303fda124fbc49b0.jpg)

![7f4a249f62582cbef3dc3563855c3a92dc6aaa4305652ebd6c2844df0a0540e5.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/tables/7f4a249f62582cbef3dc3563855c3a92dc6aaa4305652ebd6c2844df0a0540e5.jpg)

![827dffff102680f3f5bd286bb7583dc53a5f4df059fc7fcecf2fd85d06912a2d.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/tables/827dffff102680f3f5bd286bb7583dc53a5f4df059fc7fcecf2fd85d06912a2d.jpg)

![8b23a39342bcc22584d62fd1eecc71178932a02e831cfe0e9a56116f004d0b69.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/tables/8b23a39342bcc22584d62fd1eecc71178932a02e831cfe0e9a56116f004d0b69.jpg)

![a607fd2c152f63778dedb43a6e9ecb9b4b8c3e81faae7aa898e0fcc802ab2fa0.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/tables/a607fd2c152f63778dedb43a6e9ecb9b4b8c3e81faae7aa898e0fcc802ab2fa0.jpg)

![bb655ca54bbd1dfa460f5cdb88eb5bc9c6119f9c59e55cdff56fe6dfc73a40ea.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/tables/bb655ca54bbd1dfa460f5cdb88eb5bc9c6119f9c59e55cdff56fe6dfc73a40ea.jpg)

![bb701faffdb29390a900f07da18fe3a82109dadccc8c5a1413d6b41f18837137.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/tables/bb701faffdb29390a900f07da18fe3a82109dadccc8c5a1413d6b41f18837137.jpg)

![cb4ce2d6434f35f7604e5e9368a98c5dafbe3f80e610dbb42a9ad2ebc4c82d87.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/tables/cb4ce2d6434f35f7604e5e9368a98c5dafbe3f80e610dbb42a9ad2ebc4c82d87.jpg)

![ea749ec44e5f4109203141ce3275eb334a488dc4a4871399dc0e7d9f90bc80d5.jpg](../icml_results/5358_Unlocking%20Zero-Shot%20Geospatial%20Reasoning%20via%20Indirect%20Rewards/tables/ea749ec44e5f4109203141ce3275eb334a488dc4a4871399dc0e7d9f90bc80d5.jpg)

<a id="paper-5311"></a>

## How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning


### Images

![00d85bb1dc345b0d323d4bc6e26c71cc54b7115fb956b5f2d2781a7ab6dd8f09.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/00d85bb1dc345b0d323d4bc6e26c71cc54b7115fb956b5f2d2781a7ab6dd8f09.jpg)

![0136cf0b23e884af9273dd89a1e935141620136816e454d2059a5919d39bc71c.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/0136cf0b23e884af9273dd89a1e935141620136816e454d2059a5919d39bc71c.jpg)

![1a9ff96f27e977e85486a152ca402da987c1c8b37518f5675f14d3a78f882d9a.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/1a9ff96f27e977e85486a152ca402da987c1c8b37518f5675f14d3a78f882d9a.jpg)

![1ebd75251f3ae62ed18e519e534e2e6b603492871757c339223d646d385a82f9.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/1ebd75251f3ae62ed18e519e534e2e6b603492871757c339223d646d385a82f9.jpg)

![38bf74bff005b0249612315692b1b980693f242c1bd4a0c4fdd6870e9fee0dbd.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/38bf74bff005b0249612315692b1b980693f242c1bd4a0c4fdd6870e9fee0dbd.jpg)

![3a6ef7348a0384131bf6b59ff029f2a982d1c168161d736604697f651acde1b2.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/3a6ef7348a0384131bf6b59ff029f2a982d1c168161d736604697f651acde1b2.jpg)

![4bfb8689cbac5d2e1a0b9dc80bd418eb2b4906425f5bc9304e28d01e295c8ced.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/4bfb8689cbac5d2e1a0b9dc80bd418eb2b4906425f5bc9304e28d01e295c8ced.jpg)

![561e687a6100410a1a64c3954b375d745c2e12a9829d76ded0db7970f5a17675.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/561e687a6100410a1a64c3954b375d745c2e12a9829d76ded0db7970f5a17675.jpg)

![62b55bc0ab4a4887c4f38c830a65177fabb5d9b4d55f8338cd6cabe816c4081d.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/62b55bc0ab4a4887c4f38c830a65177fabb5d9b4d55f8338cd6cabe816c4081d.jpg)

![63b0c8524ac72b3114cb0b40d50bf972442111cac23080b57c28349f817855e3.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/63b0c8524ac72b3114cb0b40d50bf972442111cac23080b57c28349f817855e3.jpg)

![67998592a668b471eea09b593e61810bdb3e2fd38c543cb0226cb1b5861a32a6.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/67998592a668b471eea09b593e61810bdb3e2fd38c543cb0226cb1b5861a32a6.jpg)

![6d91802c57d3ea30a4939e45b02713dcb78e7c8bbdffac074f53295c8e206be0.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/6d91802c57d3ea30a4939e45b02713dcb78e7c8bbdffac074f53295c8e206be0.jpg)

![73d19b00fc413baeceb0709d1f86359383c1421e3b761bf3880441de5b55ff28.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/73d19b00fc413baeceb0709d1f86359383c1421e3b761bf3880441de5b55ff28.jpg)

![76bd04dd726ef7522e6d2bdfd38c286d428333f0b07242c5fd8aa3cb6f59c8ff.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/76bd04dd726ef7522e6d2bdfd38c286d428333f0b07242c5fd8aa3cb6f59c8ff.jpg)

![7fcdc11f972b198bccd61848b870779a212ed6d5b086b0c9a55e467ca6ba56c9.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/7fcdc11f972b198bccd61848b870779a212ed6d5b086b0c9a55e467ca6ba56c9.jpg)

![81724457e524bd5b37660d1c0e6c38b725f390d0ef92d4de5685b621ffb1487d.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/81724457e524bd5b37660d1c0e6c38b725f390d0ef92d4de5685b621ffb1487d.jpg)

![826d60c688eaae1f72b67c5aceb597d953b0b7d1c0e53c4b081d871b034009b5.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/826d60c688eaae1f72b67c5aceb597d953b0b7d1c0e53c4b081d871b034009b5.jpg)

![8dfde381c351998a0b5ecae02eecc353aeb2e93cb244b993e92f4ab38ca2d94a.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/8dfde381c351998a0b5ecae02eecc353aeb2e93cb244b993e92f4ab38ca2d94a.jpg)

![9109e84935a2064ce832985d5f72b3acb3608fb6fb6e496a1e63cea47c5b8db2.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/9109e84935a2064ce832985d5f72b3acb3608fb6fb6e496a1e63cea47c5b8db2.jpg)

![ab75d65ccc321d8a1387bb14114c11ffc39438ed95f7e7460959b99733c1bfd1.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/ab75d65ccc321d8a1387bb14114c11ffc39438ed95f7e7460959b99733c1bfd1.jpg)

![b64f3cdb2ee058147a5cd923eb5eeb2514d0c3cb0d7fb4ea7827196e96296e56.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/b64f3cdb2ee058147a5cd923eb5eeb2514d0c3cb0d7fb4ea7827196e96296e56.jpg)

![c779df052d7c7bf2a6b8ab832b753dddedd39e628793bc70ebdc43eee8f5bab7.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/c779df052d7c7bf2a6b8ab832b753dddedd39e628793bc70ebdc43eee8f5bab7.jpg)

![d2194650cb7850454ab5ada6699d2e19cdfdea523cfcc835b6b2a0dd2971f10f.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/images/d2194650cb7850454ab5ada6699d2e19cdfdea523cfcc835b6b2a0dd2971f10f.jpg)

### Tables

![0b081d87027fc4d8b7afdc9236197e26147bbcaf5106765f37d214ab17abe7de.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/tables/0b081d87027fc4d8b7afdc9236197e26147bbcaf5106765f37d214ab17abe7de.jpg)

![1faf216c81613e07ade14cc49f0da112f7b7f1d7ae6f793d1c05be7467c5d461.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/tables/1faf216c81613e07ade14cc49f0da112f7b7f1d7ae6f793d1c05be7467c5d461.jpg)

![2ea4ebd74c075d5a7cb2e3771b5efb359f0b26304f462f75b28afe34dfeda543.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/tables/2ea4ebd74c075d5a7cb2e3771b5efb359f0b26304f462f75b28afe34dfeda543.jpg)

![43bf8630cb9129a6b7d75bf64ebb0d407869fa5cb8c110ea579bd5aa9a5979b2.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/tables/43bf8630cb9129a6b7d75bf64ebb0d407869fa5cb8c110ea579bd5aa9a5979b2.jpg)

![48c3a4a3250ca663a8d4818f640cea606fd6c7806c80707d3db43a1c29ddb10b.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/tables/48c3a4a3250ca663a8d4818f640cea606fd6c7806c80707d3db43a1c29ddb10b.jpg)

![9846afbad9dffe02e354b32ad3aedc30e2676721c068a1625537a5ae538bdb84.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/tables/9846afbad9dffe02e354b32ad3aedc30e2676721c068a1625537a5ae538bdb84.jpg)

![dd727d6489ecc4717e58e99556638a0befe548def4dc7620de64712fba696611.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/tables/dd727d6489ecc4717e58e99556638a0befe548def4dc7620de64712fba696611.jpg)

![e2f311bb222b7c331c92cb861531d944b1f1467469161cbf1e4b0e6aec8ddb06.jpg](../icml_results/5359_How%20Far%20Ahead%20Do%20LLMs%20Plan_%20Uncovering%20the%20Latent%20Horizon%20in%20Chain-of-Thought%20Reasoning/tables/e2f311bb222b7c331c92cb861531d944b1f1467469161cbf1e4b0e6aec8ddb06.jpg)

<a id="paper-5312"></a>

## Towards One-to-Many Temporal Grounding


### Images

![409be73fa90f7eeddf83fca5390d0f611a181ed46aa07edc882f49798339d39f.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/images/409be73fa90f7eeddf83fca5390d0f611a181ed46aa07edc882f49798339d39f.jpg)

![64bc2e6a3a1a721aed408616d563a08ac832b860c47826a1eb6dfe2e9e1163cc.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/images/64bc2e6a3a1a721aed408616d563a08ac832b860c47826a1eb6dfe2e9e1163cc.jpg)

![65f34052ad40432242faa77ca511ae3ecee7b46655f794eeabccd92041699e7c.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/images/65f34052ad40432242faa77ca511ae3ecee7b46655f794eeabccd92041699e7c.jpg)

![75406af4e9ee29a574cec8cb7f79f251290a75b79a8b82bde9982565d080877e.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/images/75406af4e9ee29a574cec8cb7f79f251290a75b79a8b82bde9982565d080877e.jpg)

![773326227bf1a013b169836c05e746b22f9105eff66ba2980a2193ccbc314b32.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/images/773326227bf1a013b169836c05e746b22f9105eff66ba2980a2193ccbc314b32.jpg)

![891aa75c51d35e3b31f5b47f6ddac3d02735f2df35a79497f48b06636947483c.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/images/891aa75c51d35e3b31f5b47f6ddac3d02735f2df35a79497f48b06636947483c.jpg)

![ab8ae81f29a9dcbe0fb48d7d039a345ec6cee1779a648e1d8a07a1d61465a555.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/images/ab8ae81f29a9dcbe0fb48d7d039a345ec6cee1779a648e1d8a07a1d61465a555.jpg)

![c0377c5c878cdafb662a02ceec5599b781969d9466361305b879be5e71888b1d.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/images/c0377c5c878cdafb662a02ceec5599b781969d9466361305b879be5e71888b1d.jpg)

![cc2fe8b3d244aeb43748927f73716d112e3aa5c51c7c823843e8cee8f3c6036c.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/images/cc2fe8b3d244aeb43748927f73716d112e3aa5c51c7c823843e8cee8f3c6036c.jpg)

![fdf2de7416975cd4b75de122c0c949c262673fa332e04a5bc9ca91be52705733.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/images/fdf2de7416975cd4b75de122c0c949c262673fa332e04a5bc9ca91be52705733.jpg)

### Tables

![3ad9be427cb605a3332b0bc726e43c2e17e7d10b2be9ed8a5c5f7c36aa851ea4.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/tables/3ad9be427cb605a3332b0bc726e43c2e17e7d10b2be9ed8a5c5f7c36aa851ea4.jpg)

![434338d92ba92345687839476da302b4d32c3e07b344f318bdfa82e77005d628.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/tables/434338d92ba92345687839476da302b4d32c3e07b344f318bdfa82e77005d628.jpg)

![6b0e811e15ebf5315055c582d01566c4968fe571e5a7b5fbca6f594068868b48.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/tables/6b0e811e15ebf5315055c582d01566c4968fe571e5a7b5fbca6f594068868b48.jpg)

![810b3375e0747bcf1f04f0c5ffebb23fb2089559a291973b03ee8730c1f60251.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/tables/810b3375e0747bcf1f04f0c5ffebb23fb2089559a291973b03ee8730c1f60251.jpg)

![85651aec1507aec8e9af4de848b44eec668151ab5eb24bde37a766ac98cc19da.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/tables/85651aec1507aec8e9af4de848b44eec668151ab5eb24bde37a766ac98cc19da.jpg)

![869d7174e03cfdc288de961899b9e06df0ab619debaadc2c55544ce7ae19532d.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/tables/869d7174e03cfdc288de961899b9e06df0ab619debaadc2c55544ce7ae19532d.jpg)

![987d9a4da9511495a8b8486357c752ccb7854a26059710d7ecdd7d5a67b31ae7.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/tables/987d9a4da9511495a8b8486357c752ccb7854a26059710d7ecdd7d5a67b31ae7.jpg)

![b571302f38d70106415066a786dab2b924ab3939301a8e536e0cd0ade91f6870.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/tables/b571302f38d70106415066a786dab2b924ab3939301a8e536e0cd0ade91f6870.jpg)

![b60180b3dd5f9e5c274ef6f0918b75e515ee6459ac2505c625510a9ac12cbd3a.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/tables/b60180b3dd5f9e5c274ef6f0918b75e515ee6459ac2505c625510a9ac12cbd3a.jpg)

![ba46e8aecac372c0c15184c27d70267436089c7b8fe327233483cb3969ee6f63.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/tables/ba46e8aecac372c0c15184c27d70267436089c7b8fe327233483cb3969ee6f63.jpg)

![d05298dae9551d6ef4475e06ef146322b7a9017d2f2e12e08db283633a2bf566.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/tables/d05298dae9551d6ef4475e06ef146322b7a9017d2f2e12e08db283633a2bf566.jpg)

![deb5fed162795918fc1d0b7880d0e21072ce9de12878693b31f8821e3e760b47.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/tables/deb5fed162795918fc1d0b7880d0e21072ce9de12878693b31f8821e3e760b47.jpg)

![e11cd7aa14aeed321bd368ce07eb1613ac064d62b0c8fc7b633d68d994857701.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/tables/e11cd7aa14aeed321bd368ce07eb1613ac064d62b0c8fc7b633d68d994857701.jpg)

![e260d8950f982234f7170b2ae451a18b476476821c56ed1980dd6b1dfeb7e1b0.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/tables/e260d8950f982234f7170b2ae451a18b476476821c56ed1980dd6b1dfeb7e1b0.jpg)

![e3db1743ae2f5b889ec50aa42e7547a35e5be88a60242536e1a09838c68e9508.jpg](../icml_results/5360_Towards%20One-to-Many%20Temporal%20Grounding/tables/e3db1743ae2f5b889ec50aa42e7547a35e5be88a60242536e1a09838c68e9508.jpg)

<a id="paper-5313"></a>

## Unlearning Isn’t Deletion: Investigating Reversibility of Machine Unlearning in LLMs

### Images

![03d7bc98e16387b8cb5871cae64c17ead5378ee01dd043d8efd06bb6e8be4491.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/03d7bc98e16387b8cb5871cae64c17ead5378ee01dd043d8efd06bb6e8be4491.jpg)

![054bd2e3498bb86139cc9cef316667d3463d1353bf9574074db204a6d1ccd0dd.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/054bd2e3498bb86139cc9cef316667d3463d1353bf9574074db204a6d1ccd0dd.jpg)

![130be4ac2c25f1c3c60cb9f9b15cf1e79e85f9b939a1181ceb45b490be654854.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/130be4ac2c25f1c3c60cb9f9b15cf1e79e85f9b939a1181ceb45b490be654854.jpg)

![256049b895f53ae39f816655a44025095b04e7492f744901d1b8df97945d9859.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/256049b895f53ae39f816655a44025095b04e7492f744901d1b8df97945d9859.jpg)

![2d1adacd064600efb0d3f1d0a55c06b72e476f0602c2ba666452917aa2a50a88.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/2d1adacd064600efb0d3f1d0a55c06b72e476f0602c2ba666452917aa2a50a88.jpg)

![31850d5b3361fb232cd6478ee05cde6c104ac2c9384168fcd293d80863a5dec8.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/31850d5b3361fb232cd6478ee05cde6c104ac2c9384168fcd293d80863a5dec8.jpg)

![371c7ededc930eed639da73ff1be951548fc4215c167f3e395b7b5b7c43cfd58.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/371c7ededc930eed639da73ff1be951548fc4215c167f3e395b7b5b7c43cfd58.jpg)

![4d4257a454d370278dd096e2048ae734c875c2c415979bc84aaeb4ea61d7b850.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/4d4257a454d370278dd096e2048ae734c875c2c415979bc84aaeb4ea61d7b850.jpg)

![5aec3e89ccec4adb9c691d63665cc798aaa5ebaf1dcf9fdedbf0ed49f4dbaca7.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/5aec3e89ccec4adb9c691d63665cc798aaa5ebaf1dcf9fdedbf0ed49f4dbaca7.jpg)

![720db3f9225da48b9d6ffdee60a8a0a0120bbb4c948c1c84f80b7a446e21074e.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/720db3f9225da48b9d6ffdee60a8a0a0120bbb4c948c1c84f80b7a446e21074e.jpg)

![74bc02116051a4d7a39a063526b283180e6aa318603d54adf5bae2b7e4ff7342.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/74bc02116051a4d7a39a063526b283180e6aa318603d54adf5bae2b7e4ff7342.jpg)

![8623e754890eea82584b0d4211bb1302602394cdab9597d885effc23cd10599b.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/8623e754890eea82584b0d4211bb1302602394cdab9597d885effc23cd10599b.jpg)

![8c9a81756ced9677b647f9b61537506245e2090429a32e1fe73c37d947a759d8.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/8c9a81756ced9677b647f9b61537506245e2090429a32e1fe73c37d947a759d8.jpg)

![8d6a7788608df8e01b6e13eeaecb92b5ebf138d100aef258d1cc3e16f068d3f4.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/8d6a7788608df8e01b6e13eeaecb92b5ebf138d100aef258d1cc3e16f068d3f4.jpg)

![8d7bd936deec202b5a3caed92611a0378e3a5dd6eda7a06e6203abd51b1d7a70.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/8d7bd936deec202b5a3caed92611a0378e3a5dd6eda7a06e6203abd51b1d7a70.jpg)

![9828060fda044f0a9373e86451a83825c79cd22611ef5d42416feccfa2a393e7.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/9828060fda044f0a9373e86451a83825c79cd22611ef5d42416feccfa2a393e7.jpg)

![99c55cd15013b450fa1358c7338078fef783ed2ef4112fb19d3db29acd966771.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/99c55cd15013b450fa1358c7338078fef783ed2ef4112fb19d3db29acd966771.jpg)

![9ae92efc8f3b67ee53ff5e4570be3266cce060a7ee21a73b0a83176719d378a3.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/9ae92efc8f3b67ee53ff5e4570be3266cce060a7ee21a73b0a83176719d378a3.jpg)

![a049d13ef695cef3e9215bf2329b82f2e06ba08e0984ce53858e42870d484bc2.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/a049d13ef695cef3e9215bf2329b82f2e06ba08e0984ce53858e42870d484bc2.jpg)

![a171050885fd52546e4061996369f1909b14b5fd8888ff2d109eb31d39cafcb5.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/a171050885fd52546e4061996369f1909b14b5fd8888ff2d109eb31d39cafcb5.jpg)

![a41b4ae7273daa3b4a90211cfe1f900280d517b6873df810511820cfdac47431.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/a41b4ae7273daa3b4a90211cfe1f900280d517b6873df810511820cfdac47431.jpg)

![a9b6e53251026ab35a01bfd165e50522a3ea260045bef402c85a05f3281015ee.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/a9b6e53251026ab35a01bfd165e50522a3ea260045bef402c85a05f3281015ee.jpg)

![b25eb2227c7c7cf68083685b3efde35591636b8ffe7920baaf6ce88868c56b92.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/b25eb2227c7c7cf68083685b3efde35591636b8ffe7920baaf6ce88868c56b92.jpg)

![b9882e67eb87fc20faa5e7d4b73081afc142eb2f87408c95cc113eb3b005c539.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/b9882e67eb87fc20faa5e7d4b73081afc142eb2f87408c95cc113eb3b005c539.jpg)

![ba8fdcd00915a3b3d53a90647b68535691bc0430cc4a155498b9b4f2f709f41e.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/ba8fdcd00915a3b3d53a90647b68535691bc0430cc4a155498b9b4f2f709f41e.jpg)

![bbc76c8f5ab94964614500b0b4e5b8f714f801e0cf7b0c4d05afc47ced24526e.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/bbc76c8f5ab94964614500b0b4e5b8f714f801e0cf7b0c4d05afc47ced24526e.jpg)

![bce740b39687915d6fc54411d2752014baf1b3d4b6fbf01deecaeb2d0704255b.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/bce740b39687915d6fc54411d2752014baf1b3d4b6fbf01deecaeb2d0704255b.jpg)

![c380134164b48ca4301cf648afbd75b9cb8cb1bc7bd84856f8562782824e240e.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/c380134164b48ca4301cf648afbd75b9cb8cb1bc7bd84856f8562782824e240e.jpg)

![c64e6abf6248ad3f5e7bf4c3012a95d9531b9132c41c0a5d6fb7f3bd42260b3f.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/c64e6abf6248ad3f5e7bf4c3012a95d9531b9132c41c0a5d6fb7f3bd42260b3f.jpg)

![c877e66e6aaa31c9f02a3b80ddf30b5a506418cf2c9890352103d3df203c5247.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/c877e66e6aaa31c9f02a3b80ddf30b5a506418cf2c9890352103d3df203c5247.jpg)

![d0b356089652c56e55c3072a31aff0c26da006f13504166c0a94fc06ce499945.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/d0b356089652c56e55c3072a31aff0c26da006f13504166c0a94fc06ce499945.jpg)

![d6241ad7c1fad8d38484b295c4622734f574818713efd0d1c7cf7ea4c8d08ee2.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/d6241ad7c1fad8d38484b295c4622734f574818713efd0d1c7cf7ea4c8d08ee2.jpg)

![d89df9e982348eec9494f84649c0b7399c5bb5d225f15f04a7d65c36ab241978.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/d89df9e982348eec9494f84649c0b7399c5bb5d225f15f04a7d65c36ab241978.jpg)

![e2ef2dddc075985806cb2ac8ff7c2667cfe3f1be70504589567f9362af1b89dc.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/e2ef2dddc075985806cb2ac8ff7c2667cfe3f1be70504589567f9362af1b89dc.jpg)

![e8637dbe56a0d25115d80a70b7a04382087de4439de7505a94d123fa0bd532ba.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/e8637dbe56a0d25115d80a70b7a04382087de4439de7505a94d123fa0bd532ba.jpg)

![f630c1df8ecf8ed3d1fe5c6ab29f15d80115df8282864d3f4756395902826b04.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/f630c1df8ecf8ed3d1fe5c6ab29f15d80115df8282864d3f4756395902826b04.jpg)

![fdbece28324308350d21b22effb9bb0da19b0ea07b136d2032c32f8180f33d25.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/fdbece28324308350d21b22effb9bb0da19b0ea07b136d2032c32f8180f33d25.jpg)

![fffe9bd1c9abc97f53666012156ea70e1bb636398ece7de07e0b8c5787a98f1b.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/images/fffe9bd1c9abc97f53666012156ea70e1bb636398ece7de07e0b8c5787a98f1b.jpg)

### Tables

![116cf33c81bef1f85549a8c3c29315b35be4bfab598b6018bfc02e0dc55cfd09.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/tables/116cf33c81bef1f85549a8c3c29315b35be4bfab598b6018bfc02e0dc55cfd09.jpg)

![1806e8886fd753971a659be097e3c8579e15c1a79950b6e5f985a5fcc9ccdc27.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/tables/1806e8886fd753971a659be097e3c8579e15c1a79950b6e5f985a5fcc9ccdc27.jpg)

![222fdca32a241466e32d8c0a270451c2d21a8dc1a6a2cf2095dc9bf69248ed09.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/tables/222fdca32a241466e32d8c0a270451c2d21a8dc1a6a2cf2095dc9bf69248ed09.jpg)

![28f42d872b93903ba9a823c0caf6b308059ffcbd78387dc18d0588760cfe0101.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/tables/28f42d872b93903ba9a823c0caf6b308059ffcbd78387dc18d0588760cfe0101.jpg)

![2fbea2d72ca686a94813f256660fdd5c348e1f69bf75cdf90c3049326761976e.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/tables/2fbea2d72ca686a94813f256660fdd5c348e1f69bf75cdf90c3049326761976e.jpg)

![432fa917106bd36b69645debbfce610cfb3d94e954066121d1e4a4ef5ed5230a.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/tables/432fa917106bd36b69645debbfce610cfb3d94e954066121d1e4a4ef5ed5230a.jpg)

![5fbd196706601e424d5e21401bf2568d4c327d0b25e62aec0dc3bfea8d1905eb.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/tables/5fbd196706601e424d5e21401bf2568d4c327d0b25e62aec0dc3bfea8d1905eb.jpg)

![7adb65ad4b276de3cebcd954bf845474ebd716a4b949e29bbdc6e1b90f24588e.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/tables/7adb65ad4b276de3cebcd954bf845474ebd716a4b949e29bbdc6e1b90f24588e.jpg)

![83049888001ef4069a4e19350746e75b1fa7626ebfce5c126b7dff6674ff79e8.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/tables/83049888001ef4069a4e19350746e75b1fa7626ebfce5c126b7dff6674ff79e8.jpg)

![bc2efc8c61a5deb452b4e0c18cf0ad430f68def9fff126fd62707fa61ee4c016.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/tables/bc2efc8c61a5deb452b4e0c18cf0ad430f68def9fff126fd62707fa61ee4c016.jpg)

![c00a3a9940af2266f7b176118d8298dd17d9a3d811577fb55d708929ccb531bb.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/tables/c00a3a9940af2266f7b176118d8298dd17d9a3d811577fb55d708929ccb531bb.jpg)

![e3cf1b076b1fa084c8f58ebf5f43fbf956e4118b60b5303a2cd129bd97115199.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/tables/e3cf1b076b1fa084c8f58ebf5f43fbf956e4118b60b5303a2cd129bd97115199.jpg)

![ee337bca38930e99e00dadc2a67ce98091ca8188152a2174f2c9ccf34ac1d6f0.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/tables/ee337bca38930e99e00dadc2a67ce98091ca8188152a2174f2c9ccf34ac1d6f0.jpg)

![f6ebf606566d004a260114ae79d4b28ca67a118385741f696738dbadf1569ea0.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/tables/f6ebf606566d004a260114ae79d4b28ca67a118385741f696738dbadf1569ea0.jpg)

![fd6a103557e318f5a71dbea2dc90c442e3fbbb368ab6b2c2eae181feaee2fdc6.jpg](../icml_results/5361_Unlearning%20Isn%E2%80%99t%20Deletion_%20Investigating%20Reversibility%20of%20Machine%20Unlearning%20in%20LLMs/tables/fd6a103557e318f5a71dbea2dc90c442e3fbbb368ab6b2c2eae181feaee2fdc6.jpg)
