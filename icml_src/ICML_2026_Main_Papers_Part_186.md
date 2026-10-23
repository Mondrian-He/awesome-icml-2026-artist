# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 186 of 197

## 目录 (Table of Contents)

1. [RulePlanner: All-in-One Reinforcement Learner for Unifying Design Rules in 3D Floorplanning](#paper-6106)
2. [From Noise to Intent: Anchoring Generative VLA Policies with Residual Bridges](#paper-6107)
3. [RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression](#paper-6108)
4. [From Shortcuts to Reasoning: Robust Post-Training of Theory of Mind with Reinforcement Learning](#paper-6109)
5. [DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models](#paper-6110)
6. [Reparameterization Flow Policy Optimization](#paper-6111)
7. [DropoutTS: Sample-Adaptive Dropout for Robust Time Series Forecasting](#paper-6112)
8. [FlowNar: Scalable Streaming Narration for Long-Form Videos](#paper-6113)
9. [AgentWebBench: Benchmarking Multi-Agent Coordination in Agentic Web](#paper-6114)
10. [Riemannian MeanFlow for One-Step Generation on Manifolds](#paper-6115)
11. [Stabilizing Reinforcement Learning for Diffusion Language Models](#paper-6116)
12. [Reparameterization Proximal Policy Optimization](#paper-6117)
13. [Graph-Link: Bridging the Semantic-Structural Gap in Text-to-SQL via Constrained Subgraph Induction](#paper-6118)
14. [SWE-IF: Aligning Code Evaluation with Human Preference](#paper-6119)
15. [Black-Box Assisted Regression: Phase Transitions and Minimax Optimality](#paper-6120)
16. [ImpQuant: Fine-Grained Importance-Aware Quantization for Large Vision-Language Models](#paper-6121)
17. [AdaS: Adaptive Gradient Descent for Spiking Transformers](#paper-6122)
18. [Characterizing Vision-Language-Action Models across XPUs: Constraints and Acceleration for On-Robot Deployment](#paper-6123)
19. [TOM-SWE: User Mental Modeling For Software Engineering Agents](#paper-6124)
20. [SpikeCLR: Self-Supervised Contrastive Learning for Visual Representations with Spiking Neural Networks](#paper-6125)
21. [PGC: Peak-Guided Calibration for Generalizable AI-Generated Image Detection](#paper-6126)
22. [LLawCo: Learning Laws of Cooperation for Modeling Embodied Multi-Agent Behavior](#paper-6127)
23. [How Out-of-Distribution Detection Learning Theory Enhances Transformer: Learnability and Reliability](#paper-6128)
24. [KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices](#paper-6129)
25. [OPTION: Optimal Transport–Guided Flow Matching for Incomplete and Unaligned Multi-View Clustering](#paper-6130)
26. [QPoint: End-to-End Lightweight Point Cloud Processing via Robust Quaternion Feature Learning](#paper-6131)
27. [SpikeNet: Sparse Spike-Driven Mask Vector Transformer for Energy-Efficient and Stable Spiking Point Cloud Processing](#paper-6132)
28. [Deep Neural Network Regression with Functional Covariates](#paper-6133)
29. [Alignment Risks from Capability-Seeking RL Training](#paper-6134)
30. [Preference-Enhanced Reinforcement Learning for Pluralistic Image Inpainting](#paper-6135)
31. [SpecPL: Disentangling Spectral Granularity for Prompt Learning](#paper-6136)
32. [Towards Completeness in Causal Discovery from Soft Interventions with Known Targets](#paper-6137)
33. [AVGen-Bench: A Task-Driven Benchmark for Multi-Granular Evaluation of Text-to-Audio-Video Generation](#paper-6138)

---


<a id="paper-6106"></a>

## RulePlanner: All-in-One Reinforcement Learner for Unifying Design Rules in 3D Floorplanning

### Images

![079ce7f54c29e440433ed3e41b06ac070417df446859ddfee5fe7830263f546a.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/079ce7f54c29e440433ed3e41b06ac070417df446859ddfee5fe7830263f546a.jpg)

![095cd12af125cd436cba68ec25b5abbf7d28ff8c4384f75525ed7210f5273c03.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/095cd12af125cd436cba68ec25b5abbf7d28ff8c4384f75525ed7210f5273c03.jpg)

![237a05e454b4b52e8b98c78a292effbd70eb777cba5f59c6c4ce1371e575f44c.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/237a05e454b4b52e8b98c78a292effbd70eb777cba5f59c6c4ce1371e575f44c.jpg)

![2621baaca78e41ea5bee1ba5899b6af8112db7f331b8466490a7abb14abce836.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/2621baaca78e41ea5bee1ba5899b6af8112db7f331b8466490a7abb14abce836.jpg)

![344c4c158ff1fe1377a1cdb1c07ef8fa06d980ea5560aec0c4a200f89bc0948c.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/344c4c158ff1fe1377a1cdb1c07ef8fa06d980ea5560aec0c4a200f89bc0948c.jpg)

![34cfe3fc9f265a619c5848acb30fc5f878be1836bfb8685f66d56694049ca461.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/34cfe3fc9f265a619c5848acb30fc5f878be1836bfb8685f66d56694049ca461.jpg)

![3818a7ff6f9a44226f28cc62edf346a37aac84b570f4fe5d866c3fc3daa1b02d.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/3818a7ff6f9a44226f28cc62edf346a37aac84b570f4fe5d866c3fc3daa1b02d.jpg)

![3abbd7ea7f1e32ca62d4550da59eb9447fa80eb5465b4f43f363374844177808.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/3abbd7ea7f1e32ca62d4550da59eb9447fa80eb5465b4f43f363374844177808.jpg)

![47318fdee13924079033b9387e800df3307cc6020e8c67df8e3c32a48ee8552b.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/47318fdee13924079033b9387e800df3307cc6020e8c67df8e3c32a48ee8552b.jpg)

![523ab46bb20bee039762942d69a8cb8ca91e7dd86e7568484470617139c62987.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/523ab46bb20bee039762942d69a8cb8ca91e7dd86e7568484470617139c62987.jpg)

![54685fa476ea9ab11bcb41a5eccf8f5401633d1bbf779dbe164c03dc8c59ba58.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/54685fa476ea9ab11bcb41a5eccf8f5401633d1bbf779dbe164c03dc8c59ba58.jpg)

![55e8fb82e648d1b13e65f5c2af568a0acc9d60232c85141cbbfaa399e4a9fe79.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/55e8fb82e648d1b13e65f5c2af568a0acc9d60232c85141cbbfaa399e4a9fe79.jpg)

![58ff2a8ee4fba57a478cd5b89a3e1f2c3eedc1adfdc6449c558089cf6ef3e0d0.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/58ff2a8ee4fba57a478cd5b89a3e1f2c3eedc1adfdc6449c558089cf6ef3e0d0.jpg)

![5f8b84f128dec83dadc871c0287c2fdaa8ff65e96793f24705d5edf81eacf6dd.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/5f8b84f128dec83dadc871c0287c2fdaa8ff65e96793f24705d5edf81eacf6dd.jpg)

![684b9306c3f570bb01228a1a1db017af352fb8dee97f9011ac16d529973c602b.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/684b9306c3f570bb01228a1a1db017af352fb8dee97f9011ac16d529973c602b.jpg)

![75c983c0a67891f94760e9b7eb05921eea078e5a9b0b2ac98cdffd9461fc23ff.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/75c983c0a67891f94760e9b7eb05921eea078e5a9b0b2ac98cdffd9461fc23ff.jpg)

![78a0cbcf0d80f8df3499e7272846ff6cf768539f9589ef3c5691c4da51f424d5.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/78a0cbcf0d80f8df3499e7272846ff6cf768539f9589ef3c5691c4da51f424d5.jpg)

![7caa81cdf2ff10abccd996904969ccc04b2cf2b01bb1467f23f3f2b515163190.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/7caa81cdf2ff10abccd996904969ccc04b2cf2b01bb1467f23f3f2b515163190.jpg)

![80bb31e876114c109c591ac245ae81893d974595396f858c01624e6987dd49be.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/80bb31e876114c109c591ac245ae81893d974595396f858c01624e6987dd49be.jpg)

![83b89ed4cded861b27562cc8595a1d6bf3ddc52e74f1f82c8a5451d3cdc8da37.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/83b89ed4cded861b27562cc8595a1d6bf3ddc52e74f1f82c8a5451d3cdc8da37.jpg)

![8e92e7da2fd7334710263802bfb3b9f8e1cda1c475d460c8d6667ed020a044c8.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/8e92e7da2fd7334710263802bfb3b9f8e1cda1c475d460c8d6667ed020a044c8.jpg)

![a263fd56cccc389486926eefbcb7a68901488cc23f564caeda75cd43ddb7c808.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/a263fd56cccc389486926eefbcb7a68901488cc23f564caeda75cd43ddb7c808.jpg)

![a7b5cbc1b8ad7b65e945bced8ea5d0093da43c4eb6276019b3995fd2aaf35b93.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/a7b5cbc1b8ad7b65e945bced8ea5d0093da43c4eb6276019b3995fd2aaf35b93.jpg)

![aa09821fe0cdddfed2e25a36ec82facd6b9a949d53816fa10647a2077a62751d.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/aa09821fe0cdddfed2e25a36ec82facd6b9a949d53816fa10647a2077a62751d.jpg)

![af47f103b3f1c37124d675c5cb68aa83ef5556a016aa9bafab0290e7de4c0972.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/af47f103b3f1c37124d675c5cb68aa83ef5556a016aa9bafab0290e7de4c0972.jpg)

![b2a0c380b0e7190393ccc41f001b4556c0d34c98d3cd2caef4f0f3238f9af5d6.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/b2a0c380b0e7190393ccc41f001b4556c0d34c98d3cd2caef4f0f3238f9af5d6.jpg)

![bcdcc8f7dd1b4f180c2a1de4a975b3522629c98e8884cecb8d158367c6c4429c.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/bcdcc8f7dd1b4f180c2a1de4a975b3522629c98e8884cecb8d158367c6c4429c.jpg)

![d7895f71b310b01934b319dc3852c2d0bd1853eb6bb58563ef435f0771fd04a7.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/d7895f71b310b01934b319dc3852c2d0bd1853eb6bb58563ef435f0771fd04a7.jpg)

![d8fe96acfc886c2b82fff19cbc9255e5a021d841b8379bfac5f78ae19204f6cf.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/d8fe96acfc886c2b82fff19cbc9255e5a021d841b8379bfac5f78ae19204f6cf.jpg)

![dc7556a41b33e0f9bb86295c6b9ff3d25e0755257a5b4fec2f9f58ce432aaeae.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/dc7556a41b33e0f9bb86295c6b9ff3d25e0755257a5b4fec2f9f58ce432aaeae.jpg)

![e53622ecd0dcd032b23d431235df2fc415af754e451cd6658ce0a29d534a35bc.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/e53622ecd0dcd032b23d431235df2fc415af754e451cd6658ce0a29d534a35bc.jpg)

![e700aceaec773830fc50ee870d36784223b62ca9c0861ef9bb0ec164ea45678c.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/e700aceaec773830fc50ee870d36784223b62ca9c0861ef9bb0ec164ea45678c.jpg)

![ee225cbe65dbadfaa8038ea26ca0b5a49d5a505902580456c0b6a6399e08394b.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/ee225cbe65dbadfaa8038ea26ca0b5a49d5a505902580456c0b6a6399e08394b.jpg)

![f43c8251db15abb7cd6614d086f9ce48f15cbd55712d7dfcd636e5d836ac392d.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/f43c8251db15abb7cd6614d086f9ce48f15cbd55712d7dfcd636e5d836ac392d.jpg)

### Tables

![012e087a0fd48c893926a3af9c1bccd341d1639ed9e903aa224e812c12332b01.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/tables/012e087a0fd48c893926a3af9c1bccd341d1639ed9e903aa224e812c12332b01.jpg)

![1c8bba6d1a1f402204d1dd83788c0199bcfa7e15b4d4d5a82ec6835be02fd4c4.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/tables/1c8bba6d1a1f402204d1dd83788c0199bcfa7e15b4d4d5a82ec6835be02fd4c4.jpg)

![28bd524492879fe767534f93024a9669b670972c135828c3ecb326aa01b78888.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/tables/28bd524492879fe767534f93024a9669b670972c135828c3ecb326aa01b78888.jpg)

![3288a204fad0c9167a84ca766f3d531d92940254e7d6e45340272983cabac6b6.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/tables/3288a204fad0c9167a84ca766f3d531d92940254e7d6e45340272983cabac6b6.jpg)

![8ace9761dcb23662098f544b51ece8578fbc932933a10dc98402bf45a41f8a48.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/tables/8ace9761dcb23662098f544b51ece8578fbc932933a10dc98402bf45a41f8a48.jpg)

![94e20fc72ca2b0f12e684ef632c18f406a8bfc3fd9e987bebaa72e96e35a4971.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/tables/94e20fc72ca2b0f12e684ef632c18f406a8bfc3fd9e987bebaa72e96e35a4971.jpg)

![db02b7f79cd2b63c82e9a50ba15ed294f92314e0f5b179261c4965a59dcb90b9.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/tables/db02b7f79cd2b63c82e9a50ba15ed294f92314e0f5b179261c4965a59dcb90b9.jpg)

![f1fe4962914bae0389bdce335d1ba3aa02fa473b84cdb89467a1094654fe4bb0.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/tables/f1fe4962914bae0389bdce335d1ba3aa02fa473b84cdb89467a1094654fe4bb0.jpg)

## RulePlanner: All-in-One Reinforcement Learner for Unifying Design Rules in 3D Floorplanning


### Images

![2d44ea26b501c7c15d13011e31b64839bd4915a38d0280cf24e89707b4ab14af.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/images/2d44ea26b501c7c15d13011e31b64839bd4915a38d0280cf24e89707b4ab14af.jpg)

![43c90781e3befb5892dd983800ee980853bd72d4c5f15a8c03575108263f6941.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/images/43c90781e3befb5892dd983800ee980853bd72d4c5f15a8c03575108263f6941.jpg)

![662f7fd62bd4ea6247d4baef7203c6c6e61645b00e222959c367a54ed0a4c228.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/images/662f7fd62bd4ea6247d4baef7203c6c6e61645b00e222959c367a54ed0a4c228.jpg)

![7e4493643b20f3869430b57b843b709f31e7e0e5fc7509c9235f9737fe1f23e1.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/images/7e4493643b20f3869430b57b843b709f31e7e0e5fc7509c9235f9737fe1f23e1.jpg)

![96aa26e174c34af236bd6ef4b1666f8f63a2c9d2a0d206cf64c9a0abe424dde2.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/images/96aa26e174c34af236bd6ef4b1666f8f63a2c9d2a0d206cf64c9a0abe424dde2.jpg)

![9bb837de3b509b447dd0b64829aa5a2d1866d0bca4f210554f75dfdaa900244e.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/images/9bb837de3b509b447dd0b64829aa5a2d1866d0bca4f210554f75dfdaa900244e.jpg)

![a7849988c77ac2d54e0fd9e12ffd791ee09ddac5ffc1a750c45fc3cdc24012aa.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/images/a7849988c77ac2d54e0fd9e12ffd791ee09ddac5ffc1a750c45fc3cdc24012aa.jpg)

![a954c0a78857eec08bede5a98f761639d1962145a705c763eafa19a11fc7b8a4.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/images/a954c0a78857eec08bede5a98f761639d1962145a705c763eafa19a11fc7b8a4.jpg)

![aa52baac0709cb6488f83897b3b5cdae2647f57e436bde61e220016331aa4923.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/images/aa52baac0709cb6488f83897b3b5cdae2647f57e436bde61e220016331aa4923.jpg)

![bd93cd8b9ac49e232715b1bc0da2a12ca45f310d4b69d43f12bd9adc69375f7f.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/images/bd93cd8b9ac49e232715b1bc0da2a12ca45f310d4b69d43f12bd9adc69375f7f.jpg)

![c0aba2cd3e88b29427bf085c15f798d046ea669cc4180b636b712dbfcd9b4599.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/images/c0aba2cd3e88b29427bf085c15f798d046ea669cc4180b636b712dbfcd9b4599.jpg)

![ced831e82756a42ba7cc315ec668539a8b1f66ec419c98c8a635493a1a09909e.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/images/ced831e82756a42ba7cc315ec668539a8b1f66ec419c98c8a635493a1a09909e.jpg)

### Tables

![05a06dfc17c06e0683a7191ff029e9963b6ba0866cf3025e6255f505620ff5fd.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/05a06dfc17c06e0683a7191ff029e9963b6ba0866cf3025e6255f505620ff5fd.jpg)

![0c085c80800875a60e32acc1dadc3855019232a1ec6c05b639b400a4a1a02326.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/0c085c80800875a60e32acc1dadc3855019232a1ec6c05b639b400a4a1a02326.jpg)

![1d844ec66d60496ccbe0a4d794965bc67fcd2885e3f928489b98446515fc079c.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/1d844ec66d60496ccbe0a4d794965bc67fcd2885e3f928489b98446515fc079c.jpg)

![5afc6a13c75c855f0fea533b52ac7a9492cd1e0789c3f9f0a088dfc8c4b08555.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/5afc6a13c75c855f0fea533b52ac7a9492cd1e0789c3f9f0a088dfc8c4b08555.jpg)

![5b5e59fa0c473d4b59a2f8df2291141e929ce2214b32daa27e22c4b5a7c5c7d0.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/5b5e59fa0c473d4b59a2f8df2291141e929ce2214b32daa27e22c4b5a7c5c7d0.jpg)

![716e98349034806f3b9857f7adbec201f571e7ddede7898446c2babb8f3ceaa0.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/716e98349034806f3b9857f7adbec201f571e7ddede7898446c2babb8f3ceaa0.jpg)

![73d1df67c29cac46faf412da478db519a827af35bf097f01d871186fe78f8570.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/73d1df67c29cac46faf412da478db519a827af35bf097f01d871186fe78f8570.jpg)

![78e6d423853c8f32bb7f6b8bf844e6bc89341d30fd74a4246f54a04033db6f19.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/78e6d423853c8f32bb7f6b8bf844e6bc89341d30fd74a4246f54a04033db6f19.jpg)

![821cd4343627e3a7fe93ff51b53676138d3d98bf46eea843e74b300d9e3d4a43.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/821cd4343627e3a7fe93ff51b53676138d3d98bf46eea843e74b300d9e3d4a43.jpg)

![9788b2aeb129395e26e1118b2c349d4b27691d595c72fcae4ea9de345c5dfa7d.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/9788b2aeb129395e26e1118b2c349d4b27691d595c72fcae4ea9de345c5dfa7d.jpg)

![a7b14aa374c494371cb07c183f823e19da3eb979289ca2c3beb6bcc5b850bf4b.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/a7b14aa374c494371cb07c183f823e19da3eb979289ca2c3beb6bcc5b850bf4b.jpg)

![a83a64357af4a7454917a3d8e3be4aa00dc01342145dbe44486c6322788fcba8.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/a83a64357af4a7454917a3d8e3be4aa00dc01342145dbe44486c6322788fcba8.jpg)

![ad6759316c17e1382010a337bead3677091c7b8d3c50a6003499fe54b927ca2d.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/ad6759316c17e1382010a337bead3677091c7b8d3c50a6003499fe54b927ca2d.jpg)

![b4139e5c727b6a2b3bb84767f74148498ec9cd598dc26be86c112999c914ed09.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/b4139e5c727b6a2b3bb84767f74148498ec9cd598dc26be86c112999c914ed09.jpg)

![b4c337ad424eb3db6e73530e11cfcad237378b39564aafcec62c9132a0944bf2.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/b4c337ad424eb3db6e73530e11cfcad237378b39564aafcec62c9132a0944bf2.jpg)

![b5d80dc71c6930be912591f33d36eecfd80a5acc67738dbe498a3130af72070f.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/b5d80dc71c6930be912591f33d36eecfd80a5acc67738dbe498a3130af72070f.jpg)

![bb85b330b48c64688c6956ed32f382fdfc7c4f4ece815c9a9719307a47249098.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/bb85b330b48c64688c6956ed32f382fdfc7c4f4ece815c9a9719307a47249098.jpg)

![d17e83c70614572b2fa0800e7dedc7615990d390c27b5fcb06feaec826837017.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/d17e83c70614572b2fa0800e7dedc7615990d390c27b5fcb06feaec826837017.jpg)

![de15e646aab5314915d44bd0deac2993f0d8724baa6c34ff34a33d8e95e89c2c.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/de15e646aab5314915d44bd0deac2993f0d8724baa6c34ff34a33d8e95e89c2c.jpg)

![e07f4e16a62e895781ad7e9ada87077af421ed538f7babb1f44a63ed1672daad.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/e07f4e16a62e895781ad7e9ada87077af421ed538f7babb1f44a63ed1672daad.jpg)

![eb9cb0ed5d4a6cc366c99db62c9c93eef4f33e1b677acdefe4db06f66ae9cd7a.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/eb9cb0ed5d4a6cc366c99db62c9c93eef4f33e1b677acdefe4db06f66ae9cd7a.jpg)

![f47d979c3615ecc443387aaf786f721574c53cca92197f960db2894b3d8d54ca.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/f47d979c3615ecc443387aaf786f721574c53cca92197f960db2894b3d8d54ca.jpg)

![f76a9652f62ebdbaf7ab62208af173fc0274a0dd1884a36db4f00f218643c465.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/f76a9652f62ebdbaf7ab62208af173fc0274a0dd1884a36db4f00f218643c465.jpg)

![f8e0c81db81dcb71a799e3b173c1c64fe00c4b09a98eaa318227b48590d803c9.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/f8e0c81db81dcb71a799e3b173c1c64fe00c4b09a98eaa318227b48590d803c9.jpg)

![fd55c03d49d5310bf151b3f7d2ab9a4ab43ca51a6eb36eee2d23b1d87bae4253.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/fd55c03d49d5310bf151b3f7d2ab9a4ab43ca51a6eb36eee2d23b1d87bae4253.jpg)

![ffc02422cdc019bad138bc37a2631019385e531f3b6f37434bcf60d2dada2f2b.jpg](../icml_results/6156_RulePlanner_%20All-in-One%20Reinforcement%20Learner%20for%20Unifying%20Design%20Rules%20in%203D%20Floorplanning/tables/ffc02422cdc019bad138bc37a2631019385e531f3b6f37434bcf60d2dada2f2b.jpg)

<a id="paper-6107"></a>

## From Noise to Intent: Anchoring Generative VLA Policies with Residual Bridges


### Images

![1c0c7afe328e1a6a88fb56a6dcd8c2469661bd487d4751049805c4dd5626897b.jpg](../icml_results/6157_From%20Noise%20to%20Intent_%20Anchoring%20Generative%20VLA%20Policies%20with%20Residual%20Bridges/images/1c0c7afe328e1a6a88fb56a6dcd8c2469661bd487d4751049805c4dd5626897b.jpg)

![28fdfad4fbe1e5f33548345ca78528c054b7fb41af137da2115319e1f60cb8f3.jpg](../icml_results/6157_From%20Noise%20to%20Intent_%20Anchoring%20Generative%20VLA%20Policies%20with%20Residual%20Bridges/images/28fdfad4fbe1e5f33548345ca78528c054b7fb41af137da2115319e1f60cb8f3.jpg)

![40cd613a16c46d0b8b6b5ef584e4f106e8434ab8e38db39124d2f652d76ee61c.jpg](../icml_results/6157_From%20Noise%20to%20Intent_%20Anchoring%20Generative%20VLA%20Policies%20with%20Residual%20Bridges/images/40cd613a16c46d0b8b6b5ef584e4f106e8434ab8e38db39124d2f652d76ee61c.jpg)

![4a9d6b534690460da481cd7240ea23ce0bdfd93dae9643271e19ba764c37c1f4.jpg](../icml_results/6157_From%20Noise%20to%20Intent_%20Anchoring%20Generative%20VLA%20Policies%20with%20Residual%20Bridges/images/4a9d6b534690460da481cd7240ea23ce0bdfd93dae9643271e19ba764c37c1f4.jpg)

![94e0405a47e2d143338ee042f0e067f3a105c8db399ea1b478d344202ce874c1.jpg](../icml_results/6157_From%20Noise%20to%20Intent_%20Anchoring%20Generative%20VLA%20Policies%20with%20Residual%20Bridges/images/94e0405a47e2d143338ee042f0e067f3a105c8db399ea1b478d344202ce874c1.jpg)

![af3ad7f11f99d5df4fada3d5d53dd9e5617572fe6e4caa9f9d353da9f309d915.jpg](../icml_results/6157_From%20Noise%20to%20Intent_%20Anchoring%20Generative%20VLA%20Policies%20with%20Residual%20Bridges/images/af3ad7f11f99d5df4fada3d5d53dd9e5617572fe6e4caa9f9d353da9f309d915.jpg)

![dacda97ce7232f77b2fcf6560223d86075e97d73cf83d68b2627c3dd241352c0.jpg](../icml_results/6157_From%20Noise%20to%20Intent_%20Anchoring%20Generative%20VLA%20Policies%20with%20Residual%20Bridges/images/dacda97ce7232f77b2fcf6560223d86075e97d73cf83d68b2627c3dd241352c0.jpg)

![e602d018a558b7135391c9ad4292f460859f9c3e3f1d6f6bb87b2a59f868dfb9.jpg](../icml_results/6157_From%20Noise%20to%20Intent_%20Anchoring%20Generative%20VLA%20Policies%20with%20Residual%20Bridges/images/e602d018a558b7135391c9ad4292f460859f9c3e3f1d6f6bb87b2a59f868dfb9.jpg)

![ee50cddccbb1005913c18216ed0fc67131b6a86becb90e9fa15d2e0a57f15893.jpg](../icml_results/6157_From%20Noise%20to%20Intent_%20Anchoring%20Generative%20VLA%20Policies%20with%20Residual%20Bridges/images/ee50cddccbb1005913c18216ed0fc67131b6a86becb90e9fa15d2e0a57f15893.jpg)

### Tables

![005bded9bbb431b25261358040b4cd5e890cce8b1ab93fa48dd61c28d230211b.jpg](../icml_results/6157_From%20Noise%20to%20Intent_%20Anchoring%20Generative%20VLA%20Policies%20with%20Residual%20Bridges/tables/005bded9bbb431b25261358040b4cd5e890cce8b1ab93fa48dd61c28d230211b.jpg)

![1119ff3621d5fd678ff07b568cc78a9edc8bafcff432fdf13778cc19aad81d75.jpg](../icml_results/6157_From%20Noise%20to%20Intent_%20Anchoring%20Generative%20VLA%20Policies%20with%20Residual%20Bridges/tables/1119ff3621d5fd678ff07b568cc78a9edc8bafcff432fdf13778cc19aad81d75.jpg)

![54d1f1d6bc5c60447afb8466edec23dbbda8492b82ae5265ed62be7696d29174.jpg](../icml_results/6157_From%20Noise%20to%20Intent_%20Anchoring%20Generative%20VLA%20Policies%20with%20Residual%20Bridges/tables/54d1f1d6bc5c60447afb8466edec23dbbda8492b82ae5265ed62be7696d29174.jpg)

![e80035a32a44a7aaf711ae11116ae8b5298f15a90407b658eb585d34812a9e0f.jpg](../icml_results/6157_From%20Noise%20to%20Intent_%20Anchoring%20Generative%20VLA%20Policies%20with%20Residual%20Bridges/tables/e80035a32a44a7aaf711ae11116ae8b5298f15a90407b658eb585d34812a9e0f.jpg)

![eb54cb213ca988108020869976ce9549f7007020622ab503a97a4873ce972a2a.jpg](../icml_results/6157_From%20Noise%20to%20Intent_%20Anchoring%20Generative%20VLA%20Policies%20with%20Residual%20Bridges/tables/eb54cb213ca988108020869976ce9549f7007020622ab503a97a4873ce972a2a.jpg)

![fad0bb0cc41e1afd885eace9ee58ab4a3e5bef6518d7de8c1c861e97854a674c.jpg](../icml_results/6157_From%20Noise%20to%20Intent_%20Anchoring%20Generative%20VLA%20Policies%20with%20Residual%20Bridges/tables/fad0bb0cc41e1afd885eace9ee58ab4a3e5bef6518d7de8c1c861e97854a674c.jpg)

<a id="paper-6108"></a>

## RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression


### Images

![0121303af5ce023d1b5a74eb3fc892427a20323bbc09d9a8fd34fc3d2a2ba73f.jpg](../icml_results/6158_RQ-MoE_%20Residual%20Quantization%20via%20Mixture%20of%20Experts%20for%20Efficient%20Input-Dependent%20Vector%20Compressio/images/0121303af5ce023d1b5a74eb3fc892427a20323bbc09d9a8fd34fc3d2a2ba73f.jpg)

![62879d313012d6d3c5a38747953d3603f04079a7ae8c4fe300ba0bc2cd2a74fb.jpg](../icml_results/6158_RQ-MoE_%20Residual%20Quantization%20via%20Mixture%20of%20Experts%20for%20Efficient%20Input-Dependent%20Vector%20Compressio/images/62879d313012d6d3c5a38747953d3603f04079a7ae8c4fe300ba0bc2cd2a74fb.jpg)

![73f2ad5c175e1701c1cbe1f8c45604418846b78c094fffd08d88ae76f496fb6c.jpg](../icml_results/6158_RQ-MoE_%20Residual%20Quantization%20via%20Mixture%20of%20Experts%20for%20Efficient%20Input-Dependent%20Vector%20Compressio/images/73f2ad5c175e1701c1cbe1f8c45604418846b78c094fffd08d88ae76f496fb6c.jpg)

![80fce11c0e08bcc08c6ea533ef55f8e7e68899ff5b0f0f3f8d472c7ffc33bb55.jpg](../icml_results/6158_RQ-MoE_%20Residual%20Quantization%20via%20Mixture%20of%20Experts%20for%20Efficient%20Input-Dependent%20Vector%20Compressio/images/80fce11c0e08bcc08c6ea533ef55f8e7e68899ff5b0f0f3f8d472c7ffc33bb55.jpg)

![8a6a584f46b32d12f6752f91a00d2bbeeab0cfc95654fae9927ac3573f1fb158.jpg](../icml_results/6158_RQ-MoE_%20Residual%20Quantization%20via%20Mixture%20of%20Experts%20for%20Efficient%20Input-Dependent%20Vector%20Compressio/images/8a6a584f46b32d12f6752f91a00d2bbeeab0cfc95654fae9927ac3573f1fb158.jpg)

![94890e5cc764ae0465e89f914d2ed39e4eed1c8955ea297464870c8f53886ce4.jpg](../icml_results/6158_RQ-MoE_%20Residual%20Quantization%20via%20Mixture%20of%20Experts%20for%20Efficient%20Input-Dependent%20Vector%20Compressio/images/94890e5cc764ae0465e89f914d2ed39e4eed1c8955ea297464870c8f53886ce4.jpg)

![9d41b2ba3c587f8da537bfc9b5b9baea812e219239b5a6d6060b62c0fc90c670.jpg](../icml_results/6158_RQ-MoE_%20Residual%20Quantization%20via%20Mixture%20of%20Experts%20for%20Efficient%20Input-Dependent%20Vector%20Compressio/images/9d41b2ba3c587f8da537bfc9b5b9baea812e219239b5a6d6060b62c0fc90c670.jpg)

![d831063f837e5a83e7f1d56087aef82035d00d6e772a0a0c6f84da3cea4d866b.jpg](../icml_results/6158_RQ-MoE_%20Residual%20Quantization%20via%20Mixture%20of%20Experts%20for%20Efficient%20Input-Dependent%20Vector%20Compressio/images/d831063f837e5a83e7f1d56087aef82035d00d6e772a0a0c6f84da3cea4d866b.jpg)

### Tables

![00d4cc14da0a0c25db5f5198662e231357f41c9b0f364ab7aa2e0d3eac0054b5.jpg](../icml_results/6158_RQ-MoE_%20Residual%20Quantization%20via%20Mixture%20of%20Experts%20for%20Efficient%20Input-Dependent%20Vector%20Compressio/tables/00d4cc14da0a0c25db5f5198662e231357f41c9b0f364ab7aa2e0d3eac0054b5.jpg)

![1867848f77b449c38aa8af42711805693cdb0985a01df609c50db5d0a177a1e0.jpg](../icml_results/6158_RQ-MoE_%20Residual%20Quantization%20via%20Mixture%20of%20Experts%20for%20Efficient%20Input-Dependent%20Vector%20Compressio/tables/1867848f77b449c38aa8af42711805693cdb0985a01df609c50db5d0a177a1e0.jpg)

![3c042415905e2485f39456a29fbacd12bded582344a2dd5c32a7924a874427cc.jpg](../icml_results/6158_RQ-MoE_%20Residual%20Quantization%20via%20Mixture%20of%20Experts%20for%20Efficient%20Input-Dependent%20Vector%20Compressio/tables/3c042415905e2485f39456a29fbacd12bded582344a2dd5c32a7924a874427cc.jpg)

![61bc84bd356b3480aede046afce4fdb48478ec8f61bdbbd4a79a294b77dbc9ba.jpg](../icml_results/6158_RQ-MoE_%20Residual%20Quantization%20via%20Mixture%20of%20Experts%20for%20Efficient%20Input-Dependent%20Vector%20Compressio/tables/61bc84bd356b3480aede046afce4fdb48478ec8f61bdbbd4a79a294b77dbc9ba.jpg)

![66b8be2ba771bedc091bad1b7f7605631cae580ac2351caa62229b83183e5629.jpg](../icml_results/6158_RQ-MoE_%20Residual%20Quantization%20via%20Mixture%20of%20Experts%20for%20Efficient%20Input-Dependent%20Vector%20Compressio/tables/66b8be2ba771bedc091bad1b7f7605631cae580ac2351caa62229b83183e5629.jpg)

![6e57f9522c1533b1e2ad10c068e7e2ebfdc666869e695f4610599e1f652a2577.jpg](../icml_results/6158_RQ-MoE_%20Residual%20Quantization%20via%20Mixture%20of%20Experts%20for%20Efficient%20Input-Dependent%20Vector%20Compressio/tables/6e57f9522c1533b1e2ad10c068e7e2ebfdc666869e695f4610599e1f652a2577.jpg)

![c43b07f30fe7d78d283ee8605908f7709a508de4be25a7a1e8ea1399392e9ea1.jpg](../icml_results/6158_RQ-MoE_%20Residual%20Quantization%20via%20Mixture%20of%20Experts%20for%20Efficient%20Input-Dependent%20Vector%20Compressio/tables/c43b07f30fe7d78d283ee8605908f7709a508de4be25a7a1e8ea1399392e9ea1.jpg)

![e11249cd1333f02963e37d46caf509301e15efe5c62d1ba3e083b60127856559.jpg](../icml_results/6158_RQ-MoE_%20Residual%20Quantization%20via%20Mixture%20of%20Experts%20for%20Efficient%20Input-Dependent%20Vector%20Compressio/tables/e11249cd1333f02963e37d46caf509301e15efe5c62d1ba3e083b60127856559.jpg)

<a id="paper-6109"></a>

## From Shortcuts to Reasoning: Robust Post-Training of Theory of Mind with Reinforcement Learning


### Images

![0128b7eff36dcbde209f1cf3b776c40591f667b719d4de523fdd78cb0d3d9616.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/images/0128b7eff36dcbde209f1cf3b776c40591f667b719d4de523fdd78cb0d3d9616.jpg)

![2028fbc142bbca59a487de1a6e1404fe2a9b6815180eb70dceaa6ddc74ef3bb5.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/images/2028fbc142bbca59a487de1a6e1404fe2a9b6815180eb70dceaa6ddc74ef3bb5.jpg)

![238d2f50320f454edfbe4607542a86046fc364a5b7cd7385c761f2d6263abeba.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/images/238d2f50320f454edfbe4607542a86046fc364a5b7cd7385c761f2d6263abeba.jpg)

![3e792ae9824066900793e7db4ca2f1b5e6c56cb6dec772ef70f6c87a22ca9062.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/images/3e792ae9824066900793e7db4ca2f1b5e6c56cb6dec772ef70f6c87a22ca9062.jpg)

![62e27971158355ef48af27c37c96c2b73cf60a6059bb425a1bc1d6dca7ba0ca7.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/images/62e27971158355ef48af27c37c96c2b73cf60a6059bb425a1bc1d6dca7ba0ca7.jpg)

![8cd669b64a71b2b7ee364c029b0bfc3ac94e44b6c48e59b07984383ba348a044.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/images/8cd669b64a71b2b7ee364c029b0bfc3ac94e44b6c48e59b07984383ba348a044.jpg)

![9e2c9adf8037dc85e0b2b7f881509052d2914d1d7059eb048c1ebcc1c167a5c7.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/images/9e2c9adf8037dc85e0b2b7f881509052d2914d1d7059eb048c1ebcc1c167a5c7.jpg)

![badc2a4447c2899c748351fbbd40a1fdd19b6750da03509d849d34322eeb39a4.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/images/badc2a4447c2899c748351fbbd40a1fdd19b6750da03509d849d34322eeb39a4.jpg)

![bc97aba57f42f7de8f9cf6f1048215e843e2f2b399a7ed6f462d78f85d55ff7f.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/images/bc97aba57f42f7de8f9cf6f1048215e843e2f2b399a7ed6f462d78f85d55ff7f.jpg)

![e68efe8aab138f68e711da587858a353cc8af68e34c2d54fcfa741345afef489.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/images/e68efe8aab138f68e711da587858a353cc8af68e34c2d54fcfa741345afef489.jpg)

### Tables

![1abf19b3dbed957fc8b927e2617f32e23e31c7ea0ae59df0d380b56547f52225.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/tables/1abf19b3dbed957fc8b927e2617f32e23e31c7ea0ae59df0d380b56547f52225.jpg)

![1d845c40e153692da630e54d9e47a9881b672d2e47e47b95fde6f87505712c65.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/tables/1d845c40e153692da630e54d9e47a9881b672d2e47e47b95fde6f87505712c65.jpg)

![2073254a7bcf2323fc8c2199a44549519bd4ae85f42381a16f34e01d25d80944.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/tables/2073254a7bcf2323fc8c2199a44549519bd4ae85f42381a16f34e01d25d80944.jpg)

![3fa848d462a98f4bf117d408b95731fc6ee4f5e271b57560789cbdb0c9cb6d1d.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/tables/3fa848d462a98f4bf117d408b95731fc6ee4f5e271b57560789cbdb0c9cb6d1d.jpg)

![4e3f7b3306f383ccbf9bb492ca21968824ba54db74f7dc9f156e7272d8cc932b.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/tables/4e3f7b3306f383ccbf9bb492ca21968824ba54db74f7dc9f156e7272d8cc932b.jpg)

![50dafc82e3388b2bce0c4ab71cf0f80909ee14070de4990668180e91b888bca3.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/tables/50dafc82e3388b2bce0c4ab71cf0f80909ee14070de4990668180e91b888bca3.jpg)

![569d195db0890b85c3abcd166e0102772525d0b593824b949e37daa058f6c846.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/tables/569d195db0890b85c3abcd166e0102772525d0b593824b949e37daa058f6c846.jpg)

![6962704f50a8eca2a6ad28ad3fd22726a9dc51e206c80c23b07c386d05f33653.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/tables/6962704f50a8eca2a6ad28ad3fd22726a9dc51e206c80c23b07c386d05f33653.jpg)

![84566ac30ed88b8b52ffd935b62205e91737cfcbc666cb4a670396c604f9534a.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/tables/84566ac30ed88b8b52ffd935b62205e91737cfcbc666cb4a670396c604f9534a.jpg)

![84d8bc105e9fb2fab315700ddd8e63eb46621289d60a7151bc1b97f962e67650.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/tables/84d8bc105e9fb2fab315700ddd8e63eb46621289d60a7151bc1b97f962e67650.jpg)

![980850602efa8dd6d18c51723c9a503c82f5b13b75e25d902eb76093e4421183.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/tables/980850602efa8dd6d18c51723c9a503c82f5b13b75e25d902eb76093e4421183.jpg)

![a13c5a68353bb19274a4c4c8e6c389901364944581f7534d732a2aa93c098dfa.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/tables/a13c5a68353bb19274a4c4c8e6c389901364944581f7534d732a2aa93c098dfa.jpg)

![ac271c63edb938a87be21352506781b1d5c5bc9d7821e8f869855f31fd0335ac.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/tables/ac271c63edb938a87be21352506781b1d5c5bc9d7821e8f869855f31fd0335ac.jpg)

![b8acde3259909605a1739670fa7911eb6934abcf92df285087b90bacd43d39b7.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/tables/b8acde3259909605a1739670fa7911eb6934abcf92df285087b90bacd43d39b7.jpg)

![d1a7e489e47ba7101ffcb1bac5d24a5831b394c50f60a616820278b6c3ddbd77.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/tables/d1a7e489e47ba7101ffcb1bac5d24a5831b394c50f60a616820278b6c3ddbd77.jpg)

![d6f637894c7eb6bf418ec363dcd84ec19beb5f64095c1f81dec89990757caebb.jpg](../icml_results/6159_From%20Shortcuts%20to%20Reasoning_%20Robust%20Post-Training%20of%20Theory%20of%20Mind%20with%20Reinforcement%20Learning/tables/d6f637894c7eb6bf418ec363dcd84ec19beb5f64095c1f81dec89990757caebb.jpg)

<a id="paper-6110"></a>

## DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models


### Images

![0f790c24078b41a1e6cd715311f84acaabd7383c72f21eb5df150c6fb8cc9fb1.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/images/0f790c24078b41a1e6cd715311f84acaabd7383c72f21eb5df150c6fb8cc9fb1.jpg)

![f92ccb5789b03c4ba3239091b18ca170c577d2fd1e6eecadbf54b1ff12eb4d0b.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/images/f92ccb5789b03c4ba3239091b18ca170c577d2fd1e6eecadbf54b1ff12eb4d0b.jpg)

![fa941229b00556aa799dc0d8e0b6a1dbaab366321c5e1fef158e82d795d89b2d.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/images/fa941229b00556aa799dc0d8e0b6a1dbaab366321c5e1fef158e82d795d89b2d.jpg)

### Tables

![04360d070d40f7f3c498db614b0e818932c804e67346873560ef4436f30c20c5.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/tables/04360d070d40f7f3c498db614b0e818932c804e67346873560ef4436f30c20c5.jpg)

![103f19f2b214570b4bc0e4133251efa270e18b0412d7165f5231db6746cfa8fc.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/tables/103f19f2b214570b4bc0e4133251efa270e18b0412d7165f5231db6746cfa8fc.jpg)

![31c956d14d7e5b2c512c30598fa3125ef85a03d603c2103ca2b02304e47f4837.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/tables/31c956d14d7e5b2c512c30598fa3125ef85a03d603c2103ca2b02304e47f4837.jpg)

![40022dd1e638a7b576be48ef95e74e5e74c5b610a08fcc66c7324d25604a430c.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/tables/40022dd1e638a7b576be48ef95e74e5e74c5b610a08fcc66c7324d25604a430c.jpg)

![433f7e514c000179cf140b49a6e104a785ec74e63fa289699ddb837f99ad1ad7.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/tables/433f7e514c000179cf140b49a6e104a785ec74e63fa289699ddb837f99ad1ad7.jpg)

![4e125289856f0aaf61abe045735c7c0eb69fdec3283cffb73b3cd543b5e581d5.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/tables/4e125289856f0aaf61abe045735c7c0eb69fdec3283cffb73b3cd543b5e581d5.jpg)

![5da52dfd074036999199474d8432882c9f401859aff01e5c952eff18192b383a.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/tables/5da52dfd074036999199474d8432882c9f401859aff01e5c952eff18192b383a.jpg)

![6f092121853f9a098642e81be258b520075085783fcc730f4de6069f60f0e894.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/tables/6f092121853f9a098642e81be258b520075085783fcc730f4de6069f60f0e894.jpg)

![7d2f6ed33c571ff3b7b5afe02866e2820f999593f4632c4aef4c6a2ce7ab4e9d.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/tables/7d2f6ed33c571ff3b7b5afe02866e2820f999593f4632c4aef4c6a2ce7ab4e9d.jpg)

![867e1d1507535791de014cf099522a6cc986d4d34ba0724e46b0f8bfdc7c5ade.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/tables/867e1d1507535791de014cf099522a6cc986d4d34ba0724e46b0f8bfdc7c5ade.jpg)

![999420fe29716fcb04b88c06febda58b00d6afee349aa1e472d70a6f219d6f83.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/tables/999420fe29716fcb04b88c06febda58b00d6afee349aa1e472d70a6f219d6f83.jpg)

![aa12966ef4979a8d78d8ee66c0e6e5bce1e868578a383959c8fc9e44deec3b6e.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/tables/aa12966ef4979a8d78d8ee66c0e6e5bce1e868578a383959c8fc9e44deec3b6e.jpg)

![b9c3d319d697e3ef46e3eafc24a6f8f9dd4d261f1344b882911b00130683795a.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/tables/b9c3d319d697e3ef46e3eafc24a6f8f9dd4d261f1344b882911b00130683795a.jpg)

![ba7a66b836ad2a827a62fffbee3519fcc23ffde3f49853863e3930abf175aaef.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/tables/ba7a66b836ad2a827a62fffbee3519fcc23ffde3f49853863e3930abf175aaef.jpg)

![c77d72f6da54859b57b89bdad25dbe3afce8bfb0c5ac39d75fbb40d617123d74.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/tables/c77d72f6da54859b57b89bdad25dbe3afce8bfb0c5ac39d75fbb40d617123d74.jpg)

![db6a7c45786adcadff032da2842ae0a3243a92ae1a2e970ff3d0dabe41e2d6f4.jpg](../icml_results/6160_DualOptim%2B_%20Bridging%20Shared%20and%20Decoupled%20Optimizer%20States%20for%20Better%20Machine%20Unlearning%20in%20Large%20La/tables/db6a7c45786adcadff032da2842ae0a3243a92ae1a2e970ff3d0dabe41e2d6f4.jpg)

<a id="paper-6111"></a>

## Reparameterization Flow Policy Optimization


### Images

![0abb91153b7a833ae22290a1a6b4264cabe3b95f61581f80f2337eec29c68d1d.jpg](../icml_results/6161_Reparameterization%20Flow%20Policy%20Optimization/images/0abb91153b7a833ae22290a1a6b4264cabe3b95f61581f80f2337eec29c68d1d.jpg)

![0efc33ecc6d3dfd01082b585cba20b03d510da87fc393b4717e06978b4f3d8a2.jpg](../icml_results/6161_Reparameterization%20Flow%20Policy%20Optimization/images/0efc33ecc6d3dfd01082b585cba20b03d510da87fc393b4717e06978b4f3d8a2.jpg)

![4515af9f5ad74cd3d6e79ca03bbaac167c50ebd45a7ad1426901276d1377dfe0.jpg](../icml_results/6161_Reparameterization%20Flow%20Policy%20Optimization/images/4515af9f5ad74cd3d6e79ca03bbaac167c50ebd45a7ad1426901276d1377dfe0.jpg)

![51251682e2258e271a0eb603b7f587111236a24c067f22ae70712eabb6de59b7.jpg](../icml_results/6161_Reparameterization%20Flow%20Policy%20Optimization/images/51251682e2258e271a0eb603b7f587111236a24c067f22ae70712eabb6de59b7.jpg)

![a64702ba903971eb3ba443609974022bc6d352e3a198c20fc7c4b3d13c4c60a4.jpg](../icml_results/6161_Reparameterization%20Flow%20Policy%20Optimization/images/a64702ba903971eb3ba443609974022bc6d352e3a198c20fc7c4b3d13c4c60a4.jpg)

![cb7a37d46fef597e4039fc3557cf09a2b8e6f8fb1242b77648ebb1babca1f40a.jpg](../icml_results/6161_Reparameterization%20Flow%20Policy%20Optimization/images/cb7a37d46fef597e4039fc3557cf09a2b8e6f8fb1242b77648ebb1babca1f40a.jpg)

![d5bdb16435bd541682f14e786b071c0c2941e8cb56e41fe847a2302124ac8db9.jpg](../icml_results/6161_Reparameterization%20Flow%20Policy%20Optimization/images/d5bdb16435bd541682f14e786b071c0c2941e8cb56e41fe847a2302124ac8db9.jpg)

### Tables

![0f2864447343b0f54dce1409e8f0a94c9720992b9d314704e6b31b611de7157a.jpg](../icml_results/6161_Reparameterization%20Flow%20Policy%20Optimization/tables/0f2864447343b0f54dce1409e8f0a94c9720992b9d314704e6b31b611de7157a.jpg)

![179cafc82755bf304e8f7be32dcc36ae1d7749eb644ab3dc3e5bf3abcf9b4339.jpg](../icml_results/6161_Reparameterization%20Flow%20Policy%20Optimization/tables/179cafc82755bf304e8f7be32dcc36ae1d7749eb644ab3dc3e5bf3abcf9b4339.jpg)

![43818c91237be1a3fdc6d150923ccfa8b94fbac1cd799abbdac079d187a2b665.jpg](../icml_results/6161_Reparameterization%20Flow%20Policy%20Optimization/tables/43818c91237be1a3fdc6d150923ccfa8b94fbac1cd799abbdac079d187a2b665.jpg)

![706f56c8e1663ed7c342fe194c9886521798de76800d421ff1569549c5a0aeba.jpg](../icml_results/6161_Reparameterization%20Flow%20Policy%20Optimization/tables/706f56c8e1663ed7c342fe194c9886521798de76800d421ff1569549c5a0aeba.jpg)

![92486cbafbbaa36f3fe5c28b23b84a55df05cee6ab17e723819818319c29659a.jpg](../icml_results/6161_Reparameterization%20Flow%20Policy%20Optimization/tables/92486cbafbbaa36f3fe5c28b23b84a55df05cee6ab17e723819818319c29659a.jpg)

![aa69f307cd41fc5e504b32aba9751c75e1e35ad23e8a9f95a82e45c76dd39a73.jpg](../icml_results/6161_Reparameterization%20Flow%20Policy%20Optimization/tables/aa69f307cd41fc5e504b32aba9751c75e1e35ad23e8a9f95a82e45c76dd39a73.jpg)

![be5cf4b1515b8f37cf9e1bd4476136c3ae6f332ef3a0d493926796d4553632f4.jpg](../icml_results/6161_Reparameterization%20Flow%20Policy%20Optimization/tables/be5cf4b1515b8f37cf9e1bd4476136c3ae6f332ef3a0d493926796d4553632f4.jpg)

<a id="paper-6112"></a>

## DropoutTS: Sample-Adaptive Dropout for Robust Time Series Forecasting


### Images

![064d5b77d485b20eed7a457a74da846366221fd682fa68fbbfe81ac31d42fb5d.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/images/064d5b77d485b20eed7a457a74da846366221fd682fa68fbbfe81ac31d42fb5d.jpg)

![078f53a7a8b8ba7b6b716d995a3934aeaf5f46af7da53fd5ec32d95c4fa36adf.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/images/078f53a7a8b8ba7b6b716d995a3934aeaf5f46af7da53fd5ec32d95c4fa36adf.jpg)

![231c6f0996ede8cf723cb4b9520fb53053d23eb6944fe7b9df31ba9ccd56ddd5.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/images/231c6f0996ede8cf723cb4b9520fb53053d23eb6944fe7b9df31ba9ccd56ddd5.jpg)

![24053cc0dd99c40fd53620d1d5c3f8177a2ba0ad4a7bedf11cbfc1b9c168922f.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/images/24053cc0dd99c40fd53620d1d5c3f8177a2ba0ad4a7bedf11cbfc1b9c168922f.jpg)

![4f7a084f5937f8effadfddafdd463dbbc2629bb401871c12ac082164bf9491fd.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/images/4f7a084f5937f8effadfddafdd463dbbc2629bb401871c12ac082164bf9491fd.jpg)

![5a0fd95bf5d7da64baf7904143c6f0da07e96d50212696ab31debee15966ec92.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/images/5a0fd95bf5d7da64baf7904143c6f0da07e96d50212696ab31debee15966ec92.jpg)

![77cfe670d34f99ce31ca59d525b4f8d69e965e13d77633a6375ac5e507bc69f0.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/images/77cfe670d34f99ce31ca59d525b4f8d69e965e13d77633a6375ac5e507bc69f0.jpg)

![81730a443625a8cdd88b2f2038e5a2e9895173ba57859cc462f4be22461b7e22.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/images/81730a443625a8cdd88b2f2038e5a2e9895173ba57859cc462f4be22461b7e22.jpg)

![88ede5811dd8511d10779f0b6ab8102bb3198ae8c882ae9f10a312677a528350.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/images/88ede5811dd8511d10779f0b6ab8102bb3198ae8c882ae9f10a312677a528350.jpg)

![8a220ef7b1f06531b0f769a88e86bcc34e4c289f487e720a7386f792737e7a9d.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/images/8a220ef7b1f06531b0f769a88e86bcc34e4c289f487e720a7386f792737e7a9d.jpg)

![a15d3f46cd71ed81580c8f0181e11a51eb695848c7ca9ff674be6a9704bbda65.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/images/a15d3f46cd71ed81580c8f0181e11a51eb695848c7ca9ff674be6a9704bbda65.jpg)

![a25fca288fa689093bd1b2e772e6a4066ef99c753644ecf3517beea4077f8126.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/images/a25fca288fa689093bd1b2e772e6a4066ef99c753644ecf3517beea4077f8126.jpg)

![c21c024e04210309406f7b1f8fa8840a551976c2c10ee75f30486cd2ebbad411.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/images/c21c024e04210309406f7b1f8fa8840a551976c2c10ee75f30486cd2ebbad411.jpg)

![cddcc52af055197ea5802f6d98073782598bae7e8ed390bf270aefddf52399b8.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/images/cddcc52af055197ea5802f6d98073782598bae7e8ed390bf270aefddf52399b8.jpg)

### Tables

![0b6b8080777b4945920cd35a6248787640d6da85f6b2bd395bbea4c82fa583f6.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/tables/0b6b8080777b4945920cd35a6248787640d6da85f6b2bd395bbea4c82fa583f6.jpg)

![115be7dec80650f7efffc888cb6c02dc4ab0b1d26f424ba700fe1c615a3b0a39.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/tables/115be7dec80650f7efffc888cb6c02dc4ab0b1d26f424ba700fe1c615a3b0a39.jpg)

![13989199f367455630003cc4fe01dc5a409a79aafd57ae40a6da4c11a9743328.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/tables/13989199f367455630003cc4fe01dc5a409a79aafd57ae40a6da4c11a9743328.jpg)

![148b6e9cd806a92fdc70622b66debc57918207ef5c282f0f2eee57495bee131d.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/tables/148b6e9cd806a92fdc70622b66debc57918207ef5c282f0f2eee57495bee131d.jpg)

![17568df78cd9da7ede3dbd3d096f73af87cc88c90afb5ab1c038c3c99accf2c3.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/tables/17568df78cd9da7ede3dbd3d096f73af87cc88c90afb5ab1c038c3c99accf2c3.jpg)

![295940a881e95653ff2ccefefa79cd1df748a8d16a61e703b215475dd7dcbc4d.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/tables/295940a881e95653ff2ccefefa79cd1df748a8d16a61e703b215475dd7dcbc4d.jpg)

![3a434cdb348e2dd315afd82448eb03f7425af926497ef0b533332e103bc1b5f7.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/tables/3a434cdb348e2dd315afd82448eb03f7425af926497ef0b533332e103bc1b5f7.jpg)

![6323b5432504ca55a0219062f2a353b19feae878c7b1d2ff19cfa7456e498230.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/tables/6323b5432504ca55a0219062f2a353b19feae878c7b1d2ff19cfa7456e498230.jpg)

![801d6fe5f2394e3411952eaed04fcc4f508aeb091d2e894961469c26444e359a.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/tables/801d6fe5f2394e3411952eaed04fcc4f508aeb091d2e894961469c26444e359a.jpg)

![89e6d06f83c1ac8426a76c05dc77183558909ce4d697f9cb56b4fe5c2a8302ab.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/tables/89e6d06f83c1ac8426a76c05dc77183558909ce4d697f9cb56b4fe5c2a8302ab.jpg)

![9e68f018d17682273b07a424aced9e8829d8a1e8179c4cd09cd69f9a342bdb3a.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/tables/9e68f018d17682273b07a424aced9e8829d8a1e8179c4cd09cd69f9a342bdb3a.jpg)

![a4cfdcfe4cef156651b1945c94c841518b339a47da82da90971367b6d79f70d1.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/tables/a4cfdcfe4cef156651b1945c94c841518b339a47da82da90971367b6d79f70d1.jpg)

![bb7c4bd40bd81db667a3cae9e77e0eec9157614644c7fdbccd8c8d5e05cfda7c.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/tables/bb7c4bd40bd81db667a3cae9e77e0eec9157614644c7fdbccd8c8d5e05cfda7c.jpg)

![c898441bd62e6f8b61f1ed31a8edf1c418bc1c21969aca2868f5846ff6f3c815.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/tables/c898441bd62e6f8b61f1ed31a8edf1c418bc1c21969aca2868f5846ff6f3c815.jpg)

![daea823618dd881f7f5168171c91dc7594c244b4b88376cd7acf6ff272914cd4.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/tables/daea823618dd881f7f5168171c91dc7594c244b4b88376cd7acf6ff272914cd4.jpg)

![f484402341bc9fbef484ebdbbf602bdab00bbacafc3e551f7398773a8038a416.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/tables/f484402341bc9fbef484ebdbbf602bdab00bbacafc3e551f7398773a8038a416.jpg)

![fc42660f98f87180821cfb810a85e53263edd4987d953403c6e9ad0ac808ed4f.jpg](../icml_results/6162_DropoutTS_%20Sample-Adaptive%20Dropout%20for%20Robust%20Time%20Series%20Forecasting/tables/fc42660f98f87180821cfb810a85e53263edd4987d953403c6e9ad0ac808ed4f.jpg)

<a id="paper-6113"></a>

## FlowNar: Scalable Streaming Narration for Long-Form Videos


### Images

![0272b4405d1f4b3e156c8c8f4adb251449e5391790aef20ffcc8329bb5baf046.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/images/0272b4405d1f4b3e156c8c8f4adb251449e5391790aef20ffcc8329bb5baf046.jpg)

![07956a0801500176edc47e7d6564a54a3a06850972c0d2eb3407148935f99d79.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/images/07956a0801500176edc47e7d6564a54a3a06850972c0d2eb3407148935f99d79.jpg)

![0c2ef5756260dee662d1bc4e40ad92f816699d32f08d0fcf0f483f8d0f12873f.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/images/0c2ef5756260dee662d1bc4e40ad92f816699d32f08d0fcf0f483f8d0f12873f.jpg)

![12ea1c421fd2cb06ed409c1401b280ca12ba94231b289fadbffab00e9b4a2ec7.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/images/12ea1c421fd2cb06ed409c1401b280ca12ba94231b289fadbffab00e9b4a2ec7.jpg)

![1f6c4e9670abb831b93d9ff1f00c0821fc2aac41bb215f9e215c1db3397bb8f0.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/images/1f6c4e9670abb831b93d9ff1f00c0821fc2aac41bb215f9e215c1db3397bb8f0.jpg)

![264857eacd39cd99f8581626ed89188e80d9b187fde81cca349eaff3db45dd30.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/images/264857eacd39cd99f8581626ed89188e80d9b187fde81cca349eaff3db45dd30.jpg)

![541e35ba6e295c10a19aa0f75042205cbb424e1ef4474603792c5d72b02788de.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/images/541e35ba6e295c10a19aa0f75042205cbb424e1ef4474603792c5d72b02788de.jpg)

![5b97124c5aaabdefdd3aa4cedbd8cbdf695ca06279c793b07d0c39d203c3ad25.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/images/5b97124c5aaabdefdd3aa4cedbd8cbdf695ca06279c793b07d0c39d203c3ad25.jpg)

![5badb45541df4bed670d3494cbe6b65f3cf2f98fd64a0f6f26d7b35526ec625f.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/images/5badb45541df4bed670d3494cbe6b65f3cf2f98fd64a0f6f26d7b35526ec625f.jpg)

![61fb34bbfc96cc420919d8ae4096b470e54514edadb186ee8b3cf66c7f6b82b2.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/images/61fb34bbfc96cc420919d8ae4096b470e54514edadb186ee8b3cf66c7f6b82b2.jpg)

![8e0c5457027ad58abd134dec4dbd5807f6c77740413812e06b162b2bf636b0f0.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/images/8e0c5457027ad58abd134dec4dbd5807f6c77740413812e06b162b2bf636b0f0.jpg)

![cef3ea22c5833b5ee0ff08720878ecf675a7c19e1e8f5591e655e8481a2e67e8.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/images/cef3ea22c5833b5ee0ff08720878ecf675a7c19e1e8f5591e655e8481a2e67e8.jpg)

![e7de1dd9ea7141cf8f801f4e6835c4ffc9fb1063287bbb2af5d9ff297c01755c.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/images/e7de1dd9ea7141cf8f801f4e6835c4ffc9fb1063287bbb2af5d9ff297c01755c.jpg)

### Tables

![1b14e0be8b404326d6791018be70fd7d5bb9bc40dc60eda941ef0f0cca79973e.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/tables/1b14e0be8b404326d6791018be70fd7d5bb9bc40dc60eda941ef0f0cca79973e.jpg)

![2175d9a543fb3447c514d54d711eab07e1744e8fa09997fb30ffd82ef43b7fe0.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/tables/2175d9a543fb3447c514d54d711eab07e1744e8fa09997fb30ffd82ef43b7fe0.jpg)

![229a391e97187315498e43e5ef8bc80924056f13eedb4437630e3fa97c5e8ba5.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/tables/229a391e97187315498e43e5ef8bc80924056f13eedb4437630e3fa97c5e8ba5.jpg)

![2717c9ea96c2f7294f725f74bdaf3d97da0f1931f752a1a858a96fc29e8eda65.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/tables/2717c9ea96c2f7294f725f74bdaf3d97da0f1931f752a1a858a96fc29e8eda65.jpg)

![300acdd3fc55a463c85732a6d50ca0094486a4e28cbba4c8faff7619019bc9e3.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/tables/300acdd3fc55a463c85732a6d50ca0094486a4e28cbba4c8faff7619019bc9e3.jpg)

![340fe0c7309d42fade9603bc8c54c3cdbaf4e9a456649455d61b058869e7c04f.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/tables/340fe0c7309d42fade9603bc8c54c3cdbaf4e9a456649455d61b058869e7c04f.jpg)

![498cdf01a4c6511a4ecc9c59830e885c44ff17c2bd249e0a93232354139a3118.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/tables/498cdf01a4c6511a4ecc9c59830e885c44ff17c2bd249e0a93232354139a3118.jpg)

![5f75c780aab9c3cb61f0fc3f652d259f5220519c732c5899e6e3dc283ad3a6ef.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/tables/5f75c780aab9c3cb61f0fc3f652d259f5220519c732c5899e6e3dc283ad3a6ef.jpg)

![781bb01af1c88a8a0503d4a64282df4081df68186c8d6eba33fca5a38a5cde73.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/tables/781bb01af1c88a8a0503d4a64282df4081df68186c8d6eba33fca5a38a5cde73.jpg)

![7b25e3410200c82193d147bdef7132517b5334b50c5e5dea78c36168ad4358bb.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/tables/7b25e3410200c82193d147bdef7132517b5334b50c5e5dea78c36168ad4358bb.jpg)

![8319d31569346e61f7a708906f13b4af3a29f08e9e8c973627d300992ac4c706.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/tables/8319d31569346e61f7a708906f13b4af3a29f08e9e8c973627d300992ac4c706.jpg)

![ae657030fa0c05d6aeec11539897405c0641eb6c957600ed8bf7bff89a418602.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/tables/ae657030fa0c05d6aeec11539897405c0641eb6c957600ed8bf7bff89a418602.jpg)

![b8257ec6fb2bc196e1f8e2944c9d221c98947374a189ed2b13c01e021dbca67d.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/tables/b8257ec6fb2bc196e1f8e2944c9d221c98947374a189ed2b13c01e021dbca67d.jpg)

![cae240c16aeaf7f61db22b290add5b5b98bb89beb60af506f9a7686cc07928e5.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/tables/cae240c16aeaf7f61db22b290add5b5b98bb89beb60af506f9a7686cc07928e5.jpg)

![d07bf4515d9c554a233b9488578fe1254c30124146a72d3aea53ff6df22ac2d8.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/tables/d07bf4515d9c554a233b9488578fe1254c30124146a72d3aea53ff6df22ac2d8.jpg)

![efc7ed89ad0b75358b0124aa68c3540cfb37069a2d061ae43ba5320c99f36582.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/tables/efc7ed89ad0b75358b0124aa68c3540cfb37069a2d061ae43ba5320c99f36582.jpg)

![fca49d54fb9c25cd5c8b591f73121a6d5d433e88d4175eafff4af2a41c5ece28.jpg](../icml_results/6163_FlowNar_%20Scalable%20Streaming%20Narration%20for%20Long-Form%20Videos/tables/fca49d54fb9c25cd5c8b591f73121a6d5d433e88d4175eafff4af2a41c5ece28.jpg)

<a id="paper-6114"></a>

## AgentWebBench: Benchmarking Multi-Agent Coordination in Agentic Web


### Images

![281cdfc9e0190af49b7fe3558f5ccc6cd204faa50637510023e77a9eb0416faf.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/images/281cdfc9e0190af49b7fe3558f5ccc6cd204faa50637510023e77a9eb0416faf.jpg)

![39b4d4e22702b90ebc8fbf4db24009749db2e74733e295e2cbd00b0624675ecc.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/images/39b4d4e22702b90ebc8fbf4db24009749db2e74733e295e2cbd00b0624675ecc.jpg)

![4c7051695e4a87cdeba7f20685041870d15402a26180ece07be1b7a4f751876a.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/images/4c7051695e4a87cdeba7f20685041870d15402a26180ece07be1b7a4f751876a.jpg)

![5ec29c62742fe24411d02c6d38bcf1a11e872cd450aca6716d29212875e4e15c.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/images/5ec29c62742fe24411d02c6d38bcf1a11e872cd450aca6716d29212875e4e15c.jpg)

![7cb623a7ce214fd72ff3467fc0923bd01ffbcc08ba5e99edfb598257e28f3d69.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/images/7cb623a7ce214fd72ff3467fc0923bd01ffbcc08ba5e99edfb598257e28f3d69.jpg)

### Tables

![093141e1f01fed7d38e37ee8ad863c95faa5f60d4cafd18aca45b04b978906de.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/093141e1f01fed7d38e37ee8ad863c95faa5f60d4cafd18aca45b04b978906de.jpg)

![14431aa3a996f06b1e4c0dad968ad8bac655fba5e982a754812dc1d33094d47f.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/14431aa3a996f06b1e4c0dad968ad8bac655fba5e982a754812dc1d33094d47f.jpg)

![161f73347c0a67252c6b21295c15fdd81fd476c1ae022aa0621dcdd63a373c16.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/161f73347c0a67252c6b21295c15fdd81fd476c1ae022aa0621dcdd63a373c16.jpg)

![21da3b4741b4567e6e37544d16e0f717864d9e3b7309398f81e7ebbf73efa29d.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/21da3b4741b4567e6e37544d16e0f717864d9e3b7309398f81e7ebbf73efa29d.jpg)

![2ac04652ef26232ef191cf405373da90541bb8108cffaec4d13e8ed45fe8b377.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/2ac04652ef26232ef191cf405373da90541bb8108cffaec4d13e8ed45fe8b377.jpg)

![3be9a44a0ad80bf5209b66b795eea02a729c166d102702ce45d2a6186b7b4cd5.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/3be9a44a0ad80bf5209b66b795eea02a729c166d102702ce45d2a6186b7b4cd5.jpg)

![593a758a551be9188b9917ea2f979f3956acf2360d83959503dccb2f49273a9d.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/593a758a551be9188b9917ea2f979f3956acf2360d83959503dccb2f49273a9d.jpg)

![5ebb75920c3367bc99adf38c2363083921bd38a304bbed2c3eac2cc27ba18060.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/5ebb75920c3367bc99adf38c2363083921bd38a304bbed2c3eac2cc27ba18060.jpg)

![62173600cfc983d8fb185897d1c46dcad8f19982bd15120710d2f356ebf978da.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/62173600cfc983d8fb185897d1c46dcad8f19982bd15120710d2f356ebf978da.jpg)

![73428778ddf8c7888a0ec64bcafc23b233fcedf5b9e1989d0c7cb41da22343cf.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/73428778ddf8c7888a0ec64bcafc23b233fcedf5b9e1989d0c7cb41da22343cf.jpg)

![8c58dcf2502a4368bc6d87ad7c56d71c7b9bb8bef1be5c64e1a6dd3f8cd5c7bb.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/8c58dcf2502a4368bc6d87ad7c56d71c7b9bb8bef1be5c64e1a6dd3f8cd5c7bb.jpg)

![92a4eb20140885a623baae7702e169cd4252315e7a2fd4ed35dcb1c70995e9d9.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/92a4eb20140885a623baae7702e169cd4252315e7a2fd4ed35dcb1c70995e9d9.jpg)

![a3b6bf85b0594b8dba8a080d3032213a22f75d1ab4c6c675b82d0ebacb020ac6.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/a3b6bf85b0594b8dba8a080d3032213a22f75d1ab4c6c675b82d0ebacb020ac6.jpg)

![adee75577ae1f6cf1a027616defaddc00edbe70ee847a6b286b0abee59002bb8.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/adee75577ae1f6cf1a027616defaddc00edbe70ee847a6b286b0abee59002bb8.jpg)

![bf32c9d678e15819fecd9710294a973f44e4dabe81908a0924fdc842028ff635.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/bf32c9d678e15819fecd9710294a973f44e4dabe81908a0924fdc842028ff635.jpg)

![cc69538da62ab297bc6b266a53916a7cb952f65add278b182521fa3cd0c4c764.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/cc69538da62ab297bc6b266a53916a7cb952f65add278b182521fa3cd0c4c764.jpg)

![eb8ebad7371ef65f9b921bd803c8136c6b43443aef01b089a64f63b7029da54e.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/eb8ebad7371ef65f9b921bd803c8136c6b43443aef01b089a64f63b7029da54e.jpg)

![f4de7bc654e14b96ed63e9ee5e7da54ae50f8c056184fc4ca00a21ee56f3ee94.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/f4de7bc654e14b96ed63e9ee5e7da54ae50f8c056184fc4ca00a21ee56f3ee94.jpg)

![f99f50ed617d66f1f7129aff8cb36f44eb88ead9c6016dc889a69a87aefd605d.jpg](../icml_results/6164_AgentWebBench_%20Benchmarking%20Multi-Agent%20Coordination%20in%20Agentic%20Web/tables/f99f50ed617d66f1f7129aff8cb36f44eb88ead9c6016dc889a69a87aefd605d.jpg)

<a id="paper-6115"></a>

## Riemannian MeanFlow for One-Step Generation on Manifolds


### Images

![0e6d6a8bb79a4501d2f59a5233306e546e8787832615709ed88b79f9e29ab1cb.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/images/0e6d6a8bb79a4501d2f59a5233306e546e8787832615709ed88b79f9e29ab1cb.jpg)

![172b4564e887b83403711025ef477380b89382884624e83672d486209e0fb5ce.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/images/172b4564e887b83403711025ef477380b89382884624e83672d486209e0fb5ce.jpg)

![2c733526a27ccad47aba1b98dcfd284842b6ca853759a51a82c8ace30bb9463e.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/images/2c733526a27ccad47aba1b98dcfd284842b6ca853759a51a82c8ace30bb9463e.jpg)

![3194a8a5f9b4521f365f79772c0045d05cbc3bd61d55fbaae9dcb62d4f7e9aaf.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/images/3194a8a5f9b4521f365f79772c0045d05cbc3bd61d55fbaae9dcb62d4f7e9aaf.jpg)

![40597d2cdab4bdccce733f87d8b0f40b94ab127e8c40891d7500f1ce0512a304.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/images/40597d2cdab4bdccce733f87d8b0f40b94ab127e8c40891d7500f1ce0512a304.jpg)

![42a5b7fb3f735da64b5f1c5bdc975122cc679038a90a00236c82c3a84be05427.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/images/42a5b7fb3f735da64b5f1c5bdc975122cc679038a90a00236c82c3a84be05427.jpg)

![553d7f0eaf332f285c9dceaf5c92b2df8e988992855b3790af11b49280ac0902.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/images/553d7f0eaf332f285c9dceaf5c92b2df8e988992855b3790af11b49280ac0902.jpg)

![ed718c4cf00cd10583b15b090a8f3a95995552824df233844e70cfcedc1b245f.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/images/ed718c4cf00cd10583b15b090a8f3a95995552824df233844e70cfcedc1b245f.jpg)

![fd23125bd6008f29d42cd74ad083833a159ac2325a721ee5a44bf15e633606bf.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/images/fd23125bd6008f29d42cd74ad083833a159ac2325a721ee5a44bf15e633606bf.jpg)

### Tables

![312c26e0a3f0493d5ab9d39c9a9fafc0c68810c3f6738cbd9b294c345ae43658.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/312c26e0a3f0493d5ab9d39c9a9fafc0c68810c3f6738cbd9b294c345ae43658.jpg)

![40996cf3c524abecbe6c77b1e52a52339caf34d7a009d495fa4776a736d54842.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/40996cf3c524abecbe6c77b1e52a52339caf34d7a009d495fa4776a736d54842.jpg)

![40fb49951c57d7ef5604fca366b3c78e793d9551f2f29e82c7a68241f4b83e24.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/40fb49951c57d7ef5604fca366b3c78e793d9551f2f29e82c7a68241f4b83e24.jpg)

![438a13fcd5f791574b4c5e91af7a3d36dc91dd4ad4a9a1faa64adb1e23189180.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/438a13fcd5f791574b4c5e91af7a3d36dc91dd4ad4a9a1faa64adb1e23189180.jpg)

![4ea5c9e86c3b8f0d0375ba236efe150963e1ef7eef10dc666ec3eeaf48fe59eb.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/4ea5c9e86c3b8f0d0375ba236efe150963e1ef7eef10dc666ec3eeaf48fe59eb.jpg)

![707040b4c8a091cbaa0eb034c81d6e02234838fc3b3b1155693be6ab5cba86bb.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/707040b4c8a091cbaa0eb034c81d6e02234838fc3b3b1155693be6ab5cba86bb.jpg)

![7d9673fd3cd6989430bacd26f4c71acad24a5c46256522e50dc978ae74c0e98f.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/7d9673fd3cd6989430bacd26f4c71acad24a5c46256522e50dc978ae74c0e98f.jpg)

![888121cac4853f5be40e85d38399600dd290c137b423b8be860a901009833b2e.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/888121cac4853f5be40e85d38399600dd290c137b423b8be860a901009833b2e.jpg)

![8a2b331b020f90f1cb1a562489b60954090ae583d432073d5fc7abf97a2faa5f.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/8a2b331b020f90f1cb1a562489b60954090ae583d432073d5fc7abf97a2faa5f.jpg)

![8e6ec8c43e8a74ae4438f7e8130e30ca84f25b61ee5c8ff811e1e3574e7e8926.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/8e6ec8c43e8a74ae4438f7e8130e30ca84f25b61ee5c8ff811e1e3574e7e8926.jpg)

![958e27de9d30ee505b5f4fe76999b7a31ff91d13aa0bf4850dedd0b692606c95.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/958e27de9d30ee505b5f4fe76999b7a31ff91d13aa0bf4850dedd0b692606c95.jpg)

![a5ff655399cb50d540b85599b22e6709adae33d359353af71c1f047f83fd72d3.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/a5ff655399cb50d540b85599b22e6709adae33d359353af71c1f047f83fd72d3.jpg)

![af51a9c2c40585021ec1cb23b908c0430676d33ebf3cefcb0cf07e4efd63752a.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/af51a9c2c40585021ec1cb23b908c0430676d33ebf3cefcb0cf07e4efd63752a.jpg)

![b07c19072eca0e85d43bbf9cb660d4102bae2237e245b005a865561d26c630ac.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/b07c19072eca0e85d43bbf9cb660d4102bae2237e245b005a865561d26c630ac.jpg)

![b41056ba91d34f852b5f050df82b1798b5f99c92fc0c4db312659084ad1924b2.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/b41056ba91d34f852b5f050df82b1798b5f99c92fc0c4db312659084ad1924b2.jpg)

![b7f2a22791e56998d8be84e34126caa719cb06765324b9e4b99b935188e7bd5b.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/b7f2a22791e56998d8be84e34126caa719cb06765324b9e4b99b935188e7bd5b.jpg)

![d71deee8bd4125bfdd502284e915a9b62bbbe1c1ae1c3d84ef7090d6f77802bf.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/d71deee8bd4125bfdd502284e915a9b62bbbe1c1ae1c3d84ef7090d6f77802bf.jpg)

![d7819176ce762a29bb80dbf6e76f11148d0c098f2e4877379617ab52050b3d7f.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/d7819176ce762a29bb80dbf6e76f11148d0c098f2e4877379617ab52050b3d7f.jpg)

![f36d19fd5125818b5aefac2f32e40ab7578811b0a157a948dcc56d2c10b0029c.jpg](../icml_results/6165_Riemannian%20MeanFlow%20for%20One-Step%20Generation%20on%20Manifolds/tables/f36d19fd5125818b5aefac2f32e40ab7578811b0a157a948dcc56d2c10b0029c.jpg)

<a id="paper-6116"></a>

## Stabilizing Reinforcement Learning for Diffusion Language Models


### Images

![024227b2339155382462fb7db15703b15814ff32a8256892401bce54b0518667.jpg](../icml_results/6166_Stabilizing%20Reinforcement%20Learning%20for%20Diffusion%20Language%20Models/images/024227b2339155382462fb7db15703b15814ff32a8256892401bce54b0518667.jpg)

![3beb64d3dc21fd175a782c57597d9abc73b29af7d2fd96fd5c378a246ca04ddc.jpg](../icml_results/6166_Stabilizing%20Reinforcement%20Learning%20for%20Diffusion%20Language%20Models/images/3beb64d3dc21fd175a782c57597d9abc73b29af7d2fd96fd5c378a246ca04ddc.jpg)

![58dd0b8bbac4271fed0837289d486342bb40f83a3e68afad627dfe7a2e7ee87a.jpg](../icml_results/6166_Stabilizing%20Reinforcement%20Learning%20for%20Diffusion%20Language%20Models/images/58dd0b8bbac4271fed0837289d486342bb40f83a3e68afad627dfe7a2e7ee87a.jpg)

![77b80b48be8c08a3d6b48d92baaa4fe088ebf908434006d54175e4e6b8005e54.jpg](../icml_results/6166_Stabilizing%20Reinforcement%20Learning%20for%20Diffusion%20Language%20Models/images/77b80b48be8c08a3d6b48d92baaa4fe088ebf908434006d54175e4e6b8005e54.jpg)

![8fdff83dfe46eda64b148e7b381737256ac13aba9b35a60842fa386d59b85b99.jpg](../icml_results/6166_Stabilizing%20Reinforcement%20Learning%20for%20Diffusion%20Language%20Models/images/8fdff83dfe46eda64b148e7b381737256ac13aba9b35a60842fa386d59b85b99.jpg)

![9db17eb82bd77499fb48f7f18933af13048a091391b6e0ca31fd5e74dd95d807.jpg](../icml_results/6166_Stabilizing%20Reinforcement%20Learning%20for%20Diffusion%20Language%20Models/images/9db17eb82bd77499fb48f7f18933af13048a091391b6e0ca31fd5e74dd95d807.jpg)

![df298aaecd1da98184b193fa0cd7a86fb42b38cefd98400f7c1b83fa2ed41a87.jpg](../icml_results/6166_Stabilizing%20Reinforcement%20Learning%20for%20Diffusion%20Language%20Models/images/df298aaecd1da98184b193fa0cd7a86fb42b38cefd98400f7c1b83fa2ed41a87.jpg)

![f21b70fe6f72017b01732ca45c3d2692385e99316dd3b2ab5afef6adf8bd4270.jpg](../icml_results/6166_Stabilizing%20Reinforcement%20Learning%20for%20Diffusion%20Language%20Models/images/f21b70fe6f72017b01732ca45c3d2692385e99316dd3b2ab5afef6adf8bd4270.jpg)

### Tables

![3ff6b8a3e7feea47257c2806f99d79768ecad4c3cbf0317a95cf741cbd02ac41.jpg](../icml_results/6166_Stabilizing%20Reinforcement%20Learning%20for%20Diffusion%20Language%20Models/tables/3ff6b8a3e7feea47257c2806f99d79768ecad4c3cbf0317a95cf741cbd02ac41.jpg)

![472f7fd949b4974b21be74a46826eb3f63635ee760819361fbe37ad1178dddea.jpg](../icml_results/6166_Stabilizing%20Reinforcement%20Learning%20for%20Diffusion%20Language%20Models/tables/472f7fd949b4974b21be74a46826eb3f63635ee760819361fbe37ad1178dddea.jpg)

![572f55ae7531822b8a795e856dbf7cf0fdb427b600318737ad75a288a9e122de.jpg](../icml_results/6166_Stabilizing%20Reinforcement%20Learning%20for%20Diffusion%20Language%20Models/tables/572f55ae7531822b8a795e856dbf7cf0fdb427b600318737ad75a288a9e122de.jpg)

![5b61af0f480ccf1c56f18d45224d1af24152ed149229ee4aab118e936beef58f.jpg](../icml_results/6166_Stabilizing%20Reinforcement%20Learning%20for%20Diffusion%20Language%20Models/tables/5b61af0f480ccf1c56f18d45224d1af24152ed149229ee4aab118e936beef58f.jpg)

![6abc3306b47125c625ffaab7368de633371ab5ca20a1b25bb36cba685026f0b6.jpg](../icml_results/6166_Stabilizing%20Reinforcement%20Learning%20for%20Diffusion%20Language%20Models/tables/6abc3306b47125c625ffaab7368de633371ab5ca20a1b25bb36cba685026f0b6.jpg)

![8ee2eb98fd845d69adda1c23a972b5223286beeea3d98380e5df1e1b87be3f27.jpg](../icml_results/6166_Stabilizing%20Reinforcement%20Learning%20for%20Diffusion%20Language%20Models/tables/8ee2eb98fd845d69adda1c23a972b5223286beeea3d98380e5df1e1b87be3f27.jpg)

![b761652b444bf6782a8086922b68f4e686d877d3f303ed3c8c5e9be53c04bd71.jpg](../icml_results/6166_Stabilizing%20Reinforcement%20Learning%20for%20Diffusion%20Language%20Models/tables/b761652b444bf6782a8086922b68f4e686d877d3f303ed3c8c5e9be53c04bd71.jpg)

![cd8a6a7ce3e68d861cae6cc854c1d0b103b8d864119b82790c54ad8be28775d6.jpg](../icml_results/6166_Stabilizing%20Reinforcement%20Learning%20for%20Diffusion%20Language%20Models/tables/cd8a6a7ce3e68d861cae6cc854c1d0b103b8d864119b82790c54ad8be28775d6.jpg)

<a id="paper-6117"></a>

## Reparameterization Proximal Policy Optimization


### Images

![160882b8d36273c835ee4936fda32d9499de1a475c4d6e0a74e30577fa9568c3.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/images/160882b8d36273c835ee4936fda32d9499de1a475c4d6e0a74e30577fa9568c3.jpg)

![1619181ce384bcd2ff180d36115d84322aeb7c04219529fd82738740628e08b8.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/images/1619181ce384bcd2ff180d36115d84322aeb7c04219529fd82738740628e08b8.jpg)

![19d9bcf372cf34adecd4d634211c64bfe52660a6dc8b227e99c2e0db98b718e2.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/images/19d9bcf372cf34adecd4d634211c64bfe52660a6dc8b227e99c2e0db98b718e2.jpg)

![3005c2dfba768c07cb92d452c316fa0e7db86d59c349a8403e0d7fb233894809.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/images/3005c2dfba768c07cb92d452c316fa0e7db86d59c349a8403e0d7fb233894809.jpg)

![396d19c2bccd3caec56c23972603c8db845f50e383fa1eb591db783b43b83a0f.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/images/396d19c2bccd3caec56c23972603c8db845f50e383fa1eb591db783b43b83a0f.jpg)

![3efac1704fbf1e53cb6598aa86112eede8d05d6c0936d5cb0f43a5bb5f9e46cf.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/images/3efac1704fbf1e53cb6598aa86112eede8d05d6c0936d5cb0f43a5bb5f9e46cf.jpg)

![57df1cb1710f5dd67c0ba51e0911f820343f0a42dc6b3083a2f4cd7727764278.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/images/57df1cb1710f5dd67c0ba51e0911f820343f0a42dc6b3083a2f4cd7727764278.jpg)

![5deba41af718a5a8a6c37b818e8089bb28b0e98425620551366fef4084c36081.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/images/5deba41af718a5a8a6c37b818e8089bb28b0e98425620551366fef4084c36081.jpg)

![6fcac35aae08ff7fb751d516a1eb8015cb52b394643adf62923a369d19fe6c3a.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/images/6fcac35aae08ff7fb751d516a1eb8015cb52b394643adf62923a369d19fe6c3a.jpg)

![a5cb2f8db7f2791996b33022be333d8c34bba913aeb94d01afff929a64f9eff6.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/images/a5cb2f8db7f2791996b33022be333d8c34bba913aeb94d01afff929a64f9eff6.jpg)

![cfe982f9e16d673e5dbe862596110a34ace6a4e038476c8678907ba4f7741718.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/images/cfe982f9e16d673e5dbe862596110a34ace6a4e038476c8678907ba4f7741718.jpg)

![e0fa7938384107b073ce55e8fcbe1ec835e514a95346fef0a83365679e88493d.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/images/e0fa7938384107b073ce55e8fcbe1ec835e514a95346fef0a83365679e88493d.jpg)

![ec18790e4200e98c0005d66dacc81fd30fa78b6ddc1cb084588f840870c59f71.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/images/ec18790e4200e98c0005d66dacc81fd30fa78b6ddc1cb084588f840870c59f71.jpg)

### Tables

![17491c52d0596f035b4ca72fc1168ee833d8995de410ea24138a96390f398bdb.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/tables/17491c52d0596f035b4ca72fc1168ee833d8995de410ea24138a96390f398bdb.jpg)

![30c3ba999b6ed13be8727bb7abd71c7fdccd595d9d153238428308b067cdecf3.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/tables/30c3ba999b6ed13be8727bb7abd71c7fdccd595d9d153238428308b067cdecf3.jpg)

![4b785b2e97d42cbe8ee69f2f75dc07e8ecb748f7adf5d68d5f34eb3d9aa657c9.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/tables/4b785b2e97d42cbe8ee69f2f75dc07e8ecb748f7adf5d68d5f34eb3d9aa657c9.jpg)

![751396b3c9a4b819b5a7dd3ad51485a6c87854ab59142a0b6fb0950e70a9e16a.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/tables/751396b3c9a4b819b5a7dd3ad51485a6c87854ab59142a0b6fb0950e70a9e16a.jpg)

![7f9084be551f38e4bb113c15a323b186f871e94961c3ab6766e467c63b8a41ce.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/tables/7f9084be551f38e4bb113c15a323b186f871e94961c3ab6766e467c63b8a41ce.jpg)

![a2c534e55ae7f5f7bdc632242c617baeeb25cbabdb009fb36c412405e6b68cd9.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/tables/a2c534e55ae7f5f7bdc632242c617baeeb25cbabdb009fb36c412405e6b68cd9.jpg)

![bddf95bbdb989cd81d52af2e79b10bbb73a4bd70121141b357cfbc4a561233e5.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/tables/bddf95bbdb989cd81d52af2e79b10bbb73a4bd70121141b357cfbc4a561233e5.jpg)

![f2199bbeaa03d299adfeb9a19b46e63f54347799563f07d93b58f35e13668f24.jpg](../icml_results/6167_Reparameterization%20Proximal%20Policy%20Optimization/tables/f2199bbeaa03d299adfeb9a19b46e63f54347799563f07d93b58f35e13668f24.jpg)

<a id="paper-6118"></a>

## Graph-Link: Bridging the Semantic-Structural Gap in Text-to-SQL via Constrained Subgraph Induction


### Images

![0183a95c697deabaad0d142064c95e15dcc4f7a4d4276eccc41645be18a51f56.jpg](../icml_results/6168_Graph-Link_%20Bridging%20the%20Semantic-Structural%20Gap%20in%20Text-to-SQL%20via%20Constrained%20Subgraph%20Induction/images/0183a95c697deabaad0d142064c95e15dcc4f7a4d4276eccc41645be18a51f56.jpg)

![6e930271576439b1ed7fc6aff6b270a4097b9f5bed65f80f1f8d2eb0c0a776a3.jpg](../icml_results/6168_Graph-Link_%20Bridging%20the%20Semantic-Structural%20Gap%20in%20Text-to-SQL%20via%20Constrained%20Subgraph%20Induction/images/6e930271576439b1ed7fc6aff6b270a4097b9f5bed65f80f1f8d2eb0c0a776a3.jpg)

![824be8fc87da22c1a938f06d2a6ce0d3cdd7339b8254b0f5347a33d0d1b84d88.jpg](../icml_results/6168_Graph-Link_%20Bridging%20the%20Semantic-Structural%20Gap%20in%20Text-to-SQL%20via%20Constrained%20Subgraph%20Induction/images/824be8fc87da22c1a938f06d2a6ce0d3cdd7339b8254b0f5347a33d0d1b84d88.jpg)

![b243a843acabfb507a5746ede99ac7c5bb9ad924781dea224ed6cdef21c028db.jpg](../icml_results/6168_Graph-Link_%20Bridging%20the%20Semantic-Structural%20Gap%20in%20Text-to-SQL%20via%20Constrained%20Subgraph%20Induction/images/b243a843acabfb507a5746ede99ac7c5bb9ad924781dea224ed6cdef21c028db.jpg)

![cd19c685118ac821b107d2ef8e8ae8c861a6d8895414e51104db952b26f76fab.jpg](../icml_results/6168_Graph-Link_%20Bridging%20the%20Semantic-Structural%20Gap%20in%20Text-to-SQL%20via%20Constrained%20Subgraph%20Induction/images/cd19c685118ac821b107d2ef8e8ae8c861a6d8895414e51104db952b26f76fab.jpg)

![cf3cf31381b8d1ff5ecad3d3287f4a3d54bd30e8952957aed1c494f040318b1d.jpg](../icml_results/6168_Graph-Link_%20Bridging%20the%20Semantic-Structural%20Gap%20in%20Text-to-SQL%20via%20Constrained%20Subgraph%20Induction/images/cf3cf31381b8d1ff5ecad3d3287f4a3d54bd30e8952957aed1c494f040318b1d.jpg)

### Tables

![049413b34f1279e33dd8b7c4a405bf57d09278e0aecda621e7c831afe35903c0.jpg](../icml_results/6168_Graph-Link_%20Bridging%20the%20Semantic-Structural%20Gap%20in%20Text-to-SQL%20via%20Constrained%20Subgraph%20Induction/tables/049413b34f1279e33dd8b7c4a405bf57d09278e0aecda621e7c831afe35903c0.jpg)

![23594f8fb41e90d95afffea47215dc0734b728a609ae6ff6022d5e749cb794cc.jpg](../icml_results/6168_Graph-Link_%20Bridging%20the%20Semantic-Structural%20Gap%20in%20Text-to-SQL%20via%20Constrained%20Subgraph%20Induction/tables/23594f8fb41e90d95afffea47215dc0734b728a609ae6ff6022d5e749cb794cc.jpg)

![2752063c85cc1fdee8c92cea72798fac703531a76008623833cbe8bccc01fd25.jpg](../icml_results/6168_Graph-Link_%20Bridging%20the%20Semantic-Structural%20Gap%20in%20Text-to-SQL%20via%20Constrained%20Subgraph%20Induction/tables/2752063c85cc1fdee8c92cea72798fac703531a76008623833cbe8bccc01fd25.jpg)

![49255dd5e8fc6097bebecabc0b6def5ba88fc1686abaf11dbcd10b1e0639b1f3.jpg](../icml_results/6168_Graph-Link_%20Bridging%20the%20Semantic-Structural%20Gap%20in%20Text-to-SQL%20via%20Constrained%20Subgraph%20Induction/tables/49255dd5e8fc6097bebecabc0b6def5ba88fc1686abaf11dbcd10b1e0639b1f3.jpg)

![4face77d2536f4fbf934acc594899bfe5bd9dc87fe69705a25af1b9c97ea1b75.jpg](../icml_results/6168_Graph-Link_%20Bridging%20the%20Semantic-Structural%20Gap%20in%20Text-to-SQL%20via%20Constrained%20Subgraph%20Induction/tables/4face77d2536f4fbf934acc594899bfe5bd9dc87fe69705a25af1b9c97ea1b75.jpg)

![60dcd9703f8da74bd7e77b44e7728b9624842e1413377830a3ebaa621e98caa7.jpg](../icml_results/6168_Graph-Link_%20Bridging%20the%20Semantic-Structural%20Gap%20in%20Text-to-SQL%20via%20Constrained%20Subgraph%20Induction/tables/60dcd9703f8da74bd7e77b44e7728b9624842e1413377830a3ebaa621e98caa7.jpg)

![84e55771a3ac587ad302b7fb1371fb81b5a7ca96e8ef35cec9b96d9bf3eeed39.jpg](../icml_results/6168_Graph-Link_%20Bridging%20the%20Semantic-Structural%20Gap%20in%20Text-to-SQL%20via%20Constrained%20Subgraph%20Induction/tables/84e55771a3ac587ad302b7fb1371fb81b5a7ca96e8ef35cec9b96d9bf3eeed39.jpg)

![ba24f33beedb7fc5a3bd64385e7d3cc4909d45666fe06f2bcd8c60cd22681c4a.jpg](../icml_results/6168_Graph-Link_%20Bridging%20the%20Semantic-Structural%20Gap%20in%20Text-to-SQL%20via%20Constrained%20Subgraph%20Induction/tables/ba24f33beedb7fc5a3bd64385e7d3cc4909d45666fe06f2bcd8c60cd22681c4a.jpg)

![ca03769466fd05924b5ca244ec2ffcf66cb8f66344442162bd13a2dec89a1371.jpg](../icml_results/6168_Graph-Link_%20Bridging%20the%20Semantic-Structural%20Gap%20in%20Text-to-SQL%20via%20Constrained%20Subgraph%20Induction/tables/ca03769466fd05924b5ca244ec2ffcf66cb8f66344442162bd13a2dec89a1371.jpg)

![d4c164121b17949a104b85f239ef059fa47494b515be7c8e3aaa3a29ae2cff8c.jpg](../icml_results/6168_Graph-Link_%20Bridging%20the%20Semantic-Structural%20Gap%20in%20Text-to-SQL%20via%20Constrained%20Subgraph%20Induction/tables/d4c164121b17949a104b85f239ef059fa47494b515be7c8e3aaa3a29ae2cff8c.jpg)

![fc0a6aba9f9804543c1b5bcd7b0d6c2aa28d45faf442a008b668059bad3a9486.jpg](../icml_results/6168_Graph-Link_%20Bridging%20the%20Semantic-Structural%20Gap%20in%20Text-to-SQL%20via%20Constrained%20Subgraph%20Induction/tables/fc0a6aba9f9804543c1b5bcd7b0d6c2aa28d45faf442a008b668059bad3a9486.jpg)

<a id="paper-6119"></a>

## SWE-IF: Aligning Code Evaluation with Human Preference


### Images

![049519d2f60e64a3b8258b8b295bbdf750ff742492637daab0eaa7b6b1bd44ba.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/images/049519d2f60e64a3b8258b8b295bbdf750ff742492637daab0eaa7b6b1bd44ba.jpg)

![0cce90dce558a39975b0af6ebc81773a6d6cb8b384d13dd6a27c08d05c10031c.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/images/0cce90dce558a39975b0af6ebc81773a6d6cb8b384d13dd6a27c08d05c10031c.jpg)

![277f957e9f4d34759deefedd8f75957c06f2e526bd19635bab959f0a91615b72.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/images/277f957e9f4d34759deefedd8f75957c06f2e526bd19635bab959f0a91615b72.jpg)

![3b9eb350cd359fa38f3df2ad5a91527db524d1ab79699e0bfa4537c288253c9e.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/images/3b9eb350cd359fa38f3df2ad5a91527db524d1ab79699e0bfa4537c288253c9e.jpg)

![4af696808b283e457438e68f7a6aad97f64dddf4b0feb934bc54fd2a9ea9a164.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/images/4af696808b283e457438e68f7a6aad97f64dddf4b0feb934bc54fd2a9ea9a164.jpg)

![4ef37bf293e9ffe5662ce1664bd1a54aa81f9523ab8e9cfc7a0387465caaf056.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/images/4ef37bf293e9ffe5662ce1664bd1a54aa81f9523ab8e9cfc7a0387465caaf056.jpg)

![8673133bad770b1615beee4b31ae4b04bf59cab9559a38df868d6ef0d3b29ccc.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/images/8673133bad770b1615beee4b31ae4b04bf59cab9559a38df868d6ef0d3b29ccc.jpg)

![b228c3115e9fa25cc87f1853c9e63b3dddf423f68ef1f8f04a53f52ca90b1e2d.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/images/b228c3115e9fa25cc87f1853c9e63b3dddf423f68ef1f8f04a53f52ca90b1e2d.jpg)

![db3785c6bbbaf48b8d44c2367acf94e7df2c7aff82ad3df3f3c80fb79e026e87.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/images/db3785c6bbbaf48b8d44c2367acf94e7df2c7aff82ad3df3f3c80fb79e026e87.jpg)

### Tables

![164eb9562ebdebe6e1354aa940103f872d0a612ba80b0fceb82c13ae2e37d681.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/tables/164eb9562ebdebe6e1354aa940103f872d0a612ba80b0fceb82c13ae2e37d681.jpg)

![22aaba7e73f321cb1be709e32772f4a1dda541ed40f71d3a5521617e45d29a42.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/tables/22aaba7e73f321cb1be709e32772f4a1dda541ed40f71d3a5521617e45d29a42.jpg)

![3e3f53b5056d0bda70065130b8b586e5166392c55293219e59ba75daffd10e32.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/tables/3e3f53b5056d0bda70065130b8b586e5166392c55293219e59ba75daffd10e32.jpg)

![4cba71c645192827d5890983bee21601857c87653aa2c0acc5cd65e35fd4175e.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/tables/4cba71c645192827d5890983bee21601857c87653aa2c0acc5cd65e35fd4175e.jpg)

![6502e33d7ebb2667146ca3b33d9688bb14eafc5298cf57788f591c0c3cf15eee.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/tables/6502e33d7ebb2667146ca3b33d9688bb14eafc5298cf57788f591c0c3cf15eee.jpg)

![6e66df98bda6c631d851fda433d3703505067b786ab42285ef0efb8718fed445.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/tables/6e66df98bda6c631d851fda433d3703505067b786ab42285ef0efb8718fed445.jpg)

![9083829d92e94e1a35103fb275cb8aca4ac02a7abb2a0a1fba0bcdc3b34b2638.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/tables/9083829d92e94e1a35103fb275cb8aca4ac02a7abb2a0a1fba0bcdc3b34b2638.jpg)

![99283be1fe056633211820dfba5fa5fd52faac09948bfb776101a20eaeb95eb4.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/tables/99283be1fe056633211820dfba5fa5fd52faac09948bfb776101a20eaeb95eb4.jpg)

![c2e24ff8a049172562ca04e6017d8e465e0538cb3818f3be7eac1642c734479a.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/tables/c2e24ff8a049172562ca04e6017d8e465e0538cb3818f3be7eac1642c734479a.jpg)

![c5965ba83a62607aa916ac2d60e5b30e00d891d82997a887fca407252d127f30.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/tables/c5965ba83a62607aa916ac2d60e5b30e00d891d82997a887fca407252d127f30.jpg)

![df60cf0f722ef9fe320f8a06d4d4daf26287a9f88008a60d91fc646a32682839.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/tables/df60cf0f722ef9fe320f8a06d4d4daf26287a9f88008a60d91fc646a32682839.jpg)

![ed97bb178895779ca7b4b8835b1073e7b3c05d5a6d4ae3825511a5d2cfe76758.jpg](../icml_results/6169_SWE-IF_%20Aligning%20Code%20Evaluation%20with%20Human%20Preference/tables/ed97bb178895779ca7b4b8835b1073e7b3c05d5a6d4ae3825511a5d2cfe76758.jpg)

<a id="paper-6120"></a>

## Black-Box Assisted Regression: Phase Transitions and Minimax Optimality


### Images

![42e72049d93e7ea26736de59844b0f7faaffabf00b34d821ae094b8969f4e848.jpg](../icml_results/6170_Black-Box%20Assisted%20Regression_%20Phase%20Transitions%20and%20Minimax%20Optimality/images/42e72049d93e7ea26736de59844b0f7faaffabf00b34d821ae094b8969f4e848.jpg)

![b984991a83a3362eb01530a08ea7cf20c22c1647db66f247418effee29ba5cfc.jpg](../icml_results/6170_Black-Box%20Assisted%20Regression_%20Phase%20Transitions%20and%20Minimax%20Optimality/images/b984991a83a3362eb01530a08ea7cf20c22c1647db66f247418effee29ba5cfc.jpg)

![cccaedf4681399f87bf2872120b69ac18207e4a93378a36d43c34f3c7f791033.jpg](../icml_results/6170_Black-Box%20Assisted%20Regression_%20Phase%20Transitions%20and%20Minimax%20Optimality/images/cccaedf4681399f87bf2872120b69ac18207e4a93378a36d43c34f3c7f791033.jpg)

![f95418c37a1cee4f85d0edb5362ada775d7e7034b4ee2d72e97960583a1e4888.jpg](../icml_results/6170_Black-Box%20Assisted%20Regression_%20Phase%20Transitions%20and%20Minimax%20Optimality/images/f95418c37a1cee4f85d0edb5362ada775d7e7034b4ee2d72e97960583a1e4888.jpg)

### Tables

![3836376c7645035e14a263e3e7f1fa01bd8615825b6a52f9628e0f479e0a7123.jpg](../icml_results/6170_Black-Box%20Assisted%20Regression_%20Phase%20Transitions%20and%20Minimax%20Optimality/tables/3836376c7645035e14a263e3e7f1fa01bd8615825b6a52f9628e0f479e0a7123.jpg)

![3df42ee6669ae23e7a1c39203d18c5c14a4a07303394905f5e82fe617441723e.jpg](../icml_results/6170_Black-Box%20Assisted%20Regression_%20Phase%20Transitions%20and%20Minimax%20Optimality/tables/3df42ee6669ae23e7a1c39203d18c5c14a4a07303394905f5e82fe617441723e.jpg)

![58c49f33d513e7d2b4be9ea04a124f7cac66889c71f47c034840d80fae3c786a.jpg](../icml_results/6170_Black-Box%20Assisted%20Regression_%20Phase%20Transitions%20and%20Minimax%20Optimality/tables/58c49f33d513e7d2b4be9ea04a124f7cac66889c71f47c034840d80fae3c786a.jpg)

![999a2f906d1add7eef652c6aa88fdd87c371e15e052695b0f373ac93b55e9fd1.jpg](../icml_results/6170_Black-Box%20Assisted%20Regression_%20Phase%20Transitions%20and%20Minimax%20Optimality/tables/999a2f906d1add7eef652c6aa88fdd87c371e15e052695b0f373ac93b55e9fd1.jpg)

![9c11aa913a8a3e96da6feb582de80e0cca26d5c584c2d24ed0d0c06767b9d55b.jpg](../icml_results/6170_Black-Box%20Assisted%20Regression_%20Phase%20Transitions%20and%20Minimax%20Optimality/tables/9c11aa913a8a3e96da6feb582de80e0cca26d5c584c2d24ed0d0c06767b9d55b.jpg)

![e0b2d1df587c424d0e2bcc63dbcb9a4928f10e406483dd707f4cf578720d070d.jpg](../icml_results/6170_Black-Box%20Assisted%20Regression_%20Phase%20Transitions%20and%20Minimax%20Optimality/tables/e0b2d1df587c424d0e2bcc63dbcb9a4928f10e406483dd707f4cf578720d070d.jpg)

<a id="paper-6121"></a>

## ImpQuant: Fine-Grained Importance-Aware Quantization for Large Vision-Language Models


### Images

![037dcf6ed27ee206fb9538d57c104d2c55902766f32eafbf8ef62ddfc8dae59d.jpg](../icml_results/6171_ImpQuant_%20Fine-Grained%20Importance-Aware%20Quantization%20for%20Large%20Vision-Language%20Models/images/037dcf6ed27ee206fb9538d57c104d2c55902766f32eafbf8ef62ddfc8dae59d.jpg)

![933343c481951469f4d2e99596a771327c3b034a3f7e47c9909c9889f394605a.jpg](../icml_results/6171_ImpQuant_%20Fine-Grained%20Importance-Aware%20Quantization%20for%20Large%20Vision-Language%20Models/images/933343c481951469f4d2e99596a771327c3b034a3f7e47c9909c9889f394605a.jpg)

![d0d0bb96f0c9b867e7ee9a71a2bc853322c26afd611bbb557bc2fe11decd4c55.jpg](../icml_results/6171_ImpQuant_%20Fine-Grained%20Importance-Aware%20Quantization%20for%20Large%20Vision-Language%20Models/images/d0d0bb96f0c9b867e7ee9a71a2bc853322c26afd611bbb557bc2fe11decd4c55.jpg)

![ddcb0333ac62b2bb528393960c4c93ccfe8650d58913ea078eef573cf46c01de.jpg](../icml_results/6171_ImpQuant_%20Fine-Grained%20Importance-Aware%20Quantization%20for%20Large%20Vision-Language%20Models/images/ddcb0333ac62b2bb528393960c4c93ccfe8650d58913ea078eef573cf46c01de.jpg)

### Tables

![137a91cef274a317fe1ad9b7f9f0889c547bbc54f2ea167dc9243baa3546a502.jpg](../icml_results/6171_ImpQuant_%20Fine-Grained%20Importance-Aware%20Quantization%20for%20Large%20Vision-Language%20Models/tables/137a91cef274a317fe1ad9b7f9f0889c547bbc54f2ea167dc9243baa3546a502.jpg)

![1622963ff5e1785e80a9bb7d4a23eac9eda65b086b12352a63b3068c3577fe45.jpg](../icml_results/6171_ImpQuant_%20Fine-Grained%20Importance-Aware%20Quantization%20for%20Large%20Vision-Language%20Models/tables/1622963ff5e1785e80a9bb7d4a23eac9eda65b086b12352a63b3068c3577fe45.jpg)

![3c0d37c800f0c534585616678b7f9a06c56ad273e3f2bb3e05ec906772453c01.jpg](../icml_results/6171_ImpQuant_%20Fine-Grained%20Importance-Aware%20Quantization%20for%20Large%20Vision-Language%20Models/tables/3c0d37c800f0c534585616678b7f9a06c56ad273e3f2bb3e05ec906772453c01.jpg)

![5c12200076838ce1eda936b417bcedbc5df136a94c058cf981c9754ea5fe62a5.jpg](../icml_results/6171_ImpQuant_%20Fine-Grained%20Importance-Aware%20Quantization%20for%20Large%20Vision-Language%20Models/tables/5c12200076838ce1eda936b417bcedbc5df136a94c058cf981c9754ea5fe62a5.jpg)

![61ba320d57d97b272a37e65f97947c93b76795dc0b6492cbbb9460d5bd75f74d.jpg](../icml_results/6171_ImpQuant_%20Fine-Grained%20Importance-Aware%20Quantization%20for%20Large%20Vision-Language%20Models/tables/61ba320d57d97b272a37e65f97947c93b76795dc0b6492cbbb9460d5bd75f74d.jpg)

![a9d163665132bcf3e1d26705ef58c2aba4cf837713a999eebea676cf06825068.jpg](../icml_results/6171_ImpQuant_%20Fine-Grained%20Importance-Aware%20Quantization%20for%20Large%20Vision-Language%20Models/tables/a9d163665132bcf3e1d26705ef58c2aba4cf837713a999eebea676cf06825068.jpg)

![b5395d3ea2cd3b83bfb9fd8321a7fb6aced2cd8f973d655f28c0c20c03f17039.jpg](../icml_results/6171_ImpQuant_%20Fine-Grained%20Importance-Aware%20Quantization%20for%20Large%20Vision-Language%20Models/tables/b5395d3ea2cd3b83bfb9fd8321a7fb6aced2cd8f973d655f28c0c20c03f17039.jpg)

![dc255cfbd2a038371cf5f43b20640267f8c30cdf5f8a6f9800fb92b85d1df173.jpg](../icml_results/6171_ImpQuant_%20Fine-Grained%20Importance-Aware%20Quantization%20for%20Large%20Vision-Language%20Models/tables/dc255cfbd2a038371cf5f43b20640267f8c30cdf5f8a6f9800fb92b85d1df173.jpg)

![ddf18dd116a92718a49f372d77e52c4d206c13f77eaf7d760090bdab14d0c2ca.jpg](../icml_results/6171_ImpQuant_%20Fine-Grained%20Importance-Aware%20Quantization%20for%20Large%20Vision-Language%20Models/tables/ddf18dd116a92718a49f372d77e52c4d206c13f77eaf7d760090bdab14d0c2ca.jpg)

![e0dd900aee4572a6dc245f575454e763d247339f1a8a26e0a92afe4954cce035.jpg](../icml_results/6171_ImpQuant_%20Fine-Grained%20Importance-Aware%20Quantization%20for%20Large%20Vision-Language%20Models/tables/e0dd900aee4572a6dc245f575454e763d247339f1a8a26e0a92afe4954cce035.jpg)

<a id="paper-6122"></a>

## AdaS: Adaptive Gradient Descent for Spiking Transformers


### Images

![6819d306b052765b7c685e8f31d4244fea33322a99a18c4978bf49a9ad7bf276.jpg](../icml_results/6172_AdaS_%20Adaptive%20Gradient%20Descent%20for%20Spiking%20Transformers/images/6819d306b052765b7c685e8f31d4244fea33322a99a18c4978bf49a9ad7bf276.jpg)

![881045a8abab84429b1a0eab97374e523dc79db6f86084ff31b08d39daba6eef.jpg](../icml_results/6172_AdaS_%20Adaptive%20Gradient%20Descent%20for%20Spiking%20Transformers/images/881045a8abab84429b1a0eab97374e523dc79db6f86084ff31b08d39daba6eef.jpg)

![8a2b0034bd1b4c3d325f0429ddcf6d44a964a3b74ed34356b4348be285553409.jpg](../icml_results/6172_AdaS_%20Adaptive%20Gradient%20Descent%20for%20Spiking%20Transformers/images/8a2b0034bd1b4c3d325f0429ddcf6d44a964a3b74ed34356b4348be285553409.jpg)

![e6f8f9f135f4742c621ad779d1a6aa3be469195b96e6f26c9920ee74cb070974.jpg](../icml_results/6172_AdaS_%20Adaptive%20Gradient%20Descent%20for%20Spiking%20Transformers/images/e6f8f9f135f4742c621ad779d1a6aa3be469195b96e6f26c9920ee74cb070974.jpg)

![e747e06a9421903ca85e7c7fd6440c05081696e6da0fa94134e1e480ac5b69bb.jpg](../icml_results/6172_AdaS_%20Adaptive%20Gradient%20Descent%20for%20Spiking%20Transformers/images/e747e06a9421903ca85e7c7fd6440c05081696e6da0fa94134e1e480ac5b69bb.jpg)

### Tables

![64269288fbb4109c7c7e203fb67b9a2a24871fb24264bdf621ff86152ab4fe14.jpg](../icml_results/6172_AdaS_%20Adaptive%20Gradient%20Descent%20for%20Spiking%20Transformers/tables/64269288fbb4109c7c7e203fb67b9a2a24871fb24264bdf621ff86152ab4fe14.jpg)

![99e9f281ad4f1de861d583f624c07576ab58c1d4d8872792016b8ea9e2b3f104.jpg](../icml_results/6172_AdaS_%20Adaptive%20Gradient%20Descent%20for%20Spiking%20Transformers/tables/99e9f281ad4f1de861d583f624c07576ab58c1d4d8872792016b8ea9e2b3f104.jpg)

![c1fcdebb2bab7db8eaf976f87f7c1490a931ec291c8c79ae4ae2f98d7021eef4.jpg](../icml_results/6172_AdaS_%20Adaptive%20Gradient%20Descent%20for%20Spiking%20Transformers/tables/c1fcdebb2bab7db8eaf976f87f7c1490a931ec291c8c79ae4ae2f98d7021eef4.jpg)

![d8cd6d285bdd2773d0a7ae00c99954167c59ff776db24ec48662a48cc7e732d4.jpg](../icml_results/6172_AdaS_%20Adaptive%20Gradient%20Descent%20for%20Spiking%20Transformers/tables/d8cd6d285bdd2773d0a7ae00c99954167c59ff776db24ec48662a48cc7e732d4.jpg)

![db78a0fe3f7e8789fe4b9accae02ca3f0398dce322eb1f42972a1b2e71af2dd9.jpg](../icml_results/6172_AdaS_%20Adaptive%20Gradient%20Descent%20for%20Spiking%20Transformers/tables/db78a0fe3f7e8789fe4b9accae02ca3f0398dce322eb1f42972a1b2e71af2dd9.jpg)

![f179e59c9619cb14f526a1ac21d63af323978c44086332c1e7521f06b7dea0ad.jpg](../icml_results/6172_AdaS_%20Adaptive%20Gradient%20Descent%20for%20Spiking%20Transformers/tables/f179e59c9619cb14f526a1ac21d63af323978c44086332c1e7521f06b7dea0ad.jpg)

![f71bf69f8be00ce8fa52f455dd8e348fd644eb6ed6150da548955d8ebe203bf1.jpg](../icml_results/6172_AdaS_%20Adaptive%20Gradient%20Descent%20for%20Spiking%20Transformers/tables/f71bf69f8be00ce8fa52f455dd8e348fd644eb6ed6150da548955d8ebe203bf1.jpg)

<a id="paper-6123"></a>

## Characterizing Vision-Language-Action Models across XPUs: Constraints and Acceleration for On-Robot Deployment


### Images

![02eefa9b4d999929c638fa68138b1a35c9997201e77064d938b90b10b719f337.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/images/02eefa9b4d999929c638fa68138b1a35c9997201e77064d938b90b10b719f337.jpg)

![175cb44d9ac21f608a6c09edeec89b400b32f7aa5d536776246fcbea4f2dcc37.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/images/175cb44d9ac21f608a6c09edeec89b400b32f7aa5d536776246fcbea4f2dcc37.jpg)

![1fa9745629307a1d8d27b1d0cb76c55c2b826403e797d1aab5fd4e15875c3261.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/images/1fa9745629307a1d8d27b1d0cb76c55c2b826403e797d1aab5fd4e15875c3261.jpg)

![4f533a25d8b8df26380161ad641e9e8f7144b26c66dfbaa6bedefe74317541fa.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/images/4f533a25d8b8df26380161ad641e9e8f7144b26c66dfbaa6bedefe74317541fa.jpg)

![705b056f0a68e0c624a57c721f669bac0ed3de4fbe11284b502a1da47c7de0ce.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/images/705b056f0a68e0c624a57c721f669bac0ed3de4fbe11284b502a1da47c7de0ce.jpg)

![7a7cb1af976639d812cc8f092063e5a3cbd73e519e384e2d4cc24079efb8ceea.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/images/7a7cb1af976639d812cc8f092063e5a3cbd73e519e384e2d4cc24079efb8ceea.jpg)

![92f72fe81725ea9292f4c86e0b81570cac2530cfc4541e456e2ba53ef240203c.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/images/92f72fe81725ea9292f4c86e0b81570cac2530cfc4541e456e2ba53ef240203c.jpg)

![932f4e71f407af2440f40aedace41372a864acc95c2b3bca4e0aa5a866c693b9.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/images/932f4e71f407af2440f40aedace41372a864acc95c2b3bca4e0aa5a866c693b9.jpg)

![ce532c992bf55b7e8efcf7767110fff9f53afe16ca8ac315a9d307d8ba39d60d.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/images/ce532c992bf55b7e8efcf7767110fff9f53afe16ca8ac315a9d307d8ba39d60d.jpg)

![e248c3a27391a3b656e4e1a2ca8432504e0aff1e766e145ffe9743b252a7d888.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/images/e248c3a27391a3b656e4e1a2ca8432504e0aff1e766e145ffe9743b252a7d888.jpg)

![ed8b17ea0e53ed380382d63598451f90fc938d1476c8d9fcf0c662d513065ce1.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/images/ed8b17ea0e53ed380382d63598451f90fc938d1476c8d9fcf0c662d513065ce1.jpg)

![f3967e692f73f6209d8dcff1ce97c8fc9315a6ce01489f7bc852e33932071181.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/images/f3967e692f73f6209d8dcff1ce97c8fc9315a6ce01489f7bc852e33932071181.jpg)

### Tables

![08f716b049cea702c79b28f9f00514dbff28234d2d00974c29cfe8736215534b.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/tables/08f716b049cea702c79b28f9f00514dbff28234d2d00974c29cfe8736215534b.jpg)

![20f55a51d42f6b2c623b5483d0391f162b10933c22e30d17ded6fd86a15f5e11.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/tables/20f55a51d42f6b2c623b5483d0391f162b10933c22e30d17ded6fd86a15f5e11.jpg)

![28c08f37e8bcde75fe20ed7268ed1adc13846e23388d33d01dd8f4fd4ce64545.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/tables/28c08f37e8bcde75fe20ed7268ed1adc13846e23388d33d01dd8f4fd4ce64545.jpg)

![298027777f2280ff442b1f62807b9553cc46d9dc895cb74dc10c85647f8bd0db.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/tables/298027777f2280ff442b1f62807b9553cc46d9dc895cb74dc10c85647f8bd0db.jpg)

![450c42511e341afd72a6f6168ba3f96e268dc0723d85a94ff5a3ce57d3ce5b85.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/tables/450c42511e341afd72a6f6168ba3f96e268dc0723d85a94ff5a3ce57d3ce5b85.jpg)

![51954081868b060e4779438dfb7f607a5dcd7aa2f7ef318253176546609449a8.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/tables/51954081868b060e4779438dfb7f607a5dcd7aa2f7ef318253176546609449a8.jpg)

![681bcd61bdcd0718696361c2da091679c2f51bd17f9df2d5ff766fbcff776493.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/tables/681bcd61bdcd0718696361c2da091679c2f51bd17f9df2d5ff766fbcff776493.jpg)

![6c20aeb5f709201577e7727cdfc95b8bd8798271ef47ef84eba0c23b2dfe1e6c.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/tables/6c20aeb5f709201577e7727cdfc95b8bd8798271ef47ef84eba0c23b2dfe1e6c.jpg)

![6fc043b567c8f779fe08364a48c0674dab0325e3604db1bf904525b3aac4712e.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/tables/6fc043b567c8f779fe08364a48c0674dab0325e3604db1bf904525b3aac4712e.jpg)

![c19183ae9302dd8f77dd68cfeb3aa2eb92c5f7f67e8186d9bc54c337687cefbc.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/tables/c19183ae9302dd8f77dd68cfeb3aa2eb92c5f7f67e8186d9bc54c337687cefbc.jpg)

![e886b1ef91660eeebd554df5fc240edfa16c69e97a467aacab50eda13cbbf4e5.jpg](../icml_results/6173_Characterizing%20Vision-Language-Action%20Models%20across%20XPUs_%20Constraints%20and%20Acceleration%20for%20On-Robot%20/tables/e886b1ef91660eeebd554df5fc240edfa16c69e97a467aacab50eda13cbbf4e5.jpg)

<a id="paper-6124"></a>

## TOM-SWE: User Mental Modeling For Software Engineering Agents


### Images

![2cf5ef37d74a3a6c6bcab39ffac9a3d0a7657d331a65060e8392c337d6fe70ae.jpg](../icml_results/6174_TOM-SWE_%20User%20Mental%20Modeling%20For%20Software%20Engineering%20Agents/images/2cf5ef37d74a3a6c6bcab39ffac9a3d0a7657d331a65060e8392c337d6fe70ae.jpg)

![31ee55a851703ae18c22e6faef49c6aa4511153e3973b431c1fda5c3f8aec1bc.jpg](../icml_results/6174_TOM-SWE_%20User%20Mental%20Modeling%20For%20Software%20Engineering%20Agents/images/31ee55a851703ae18c22e6faef49c6aa4511153e3973b431c1fda5c3f8aec1bc.jpg)

![42aa69b80435edd675d8fa263b0c2b2df7d102663dea389fed9b15d69e3cc762.jpg](../icml_results/6174_TOM-SWE_%20User%20Mental%20Modeling%20For%20Software%20Engineering%20Agents/images/42aa69b80435edd675d8fa263b0c2b2df7d102663dea389fed9b15d69e3cc762.jpg)

![4ca8ff758a7c1d1d0062e1d7cdbfe3e614a1fb00c3b82b3b1bfaa57d1d33c8da.jpg](../icml_results/6174_TOM-SWE_%20User%20Mental%20Modeling%20For%20Software%20Engineering%20Agents/images/4ca8ff758a7c1d1d0062e1d7cdbfe3e614a1fb00c3b82b3b1bfaa57d1d33c8da.jpg)

![69005bded8f306a6e2a4327105bbd49defac0bd094db8b5ea5866badf8d9de70.jpg](../icml_results/6174_TOM-SWE_%20User%20Mental%20Modeling%20For%20Software%20Engineering%20Agents/images/69005bded8f306a6e2a4327105bbd49defac0bd094db8b5ea5866badf8d9de70.jpg)

![8f5db63a7e55a0f7626b1d62545a6b2c621f03f983571875eb948bd632923b5d.jpg](../icml_results/6174_TOM-SWE_%20User%20Mental%20Modeling%20For%20Software%20Engineering%20Agents/images/8f5db63a7e55a0f7626b1d62545a6b2c621f03f983571875eb948bd632923b5d.jpg)

![8f607329fea5115e9fe1f36488879c53e1c674d91e2bab872855a647f20f1333.jpg](../icml_results/6174_TOM-SWE_%20User%20Mental%20Modeling%20For%20Software%20Engineering%20Agents/images/8f607329fea5115e9fe1f36488879c53e1c674d91e2bab872855a647f20f1333.jpg)

### Tables

![06481fc5b0b0ec319b9916dbb6399abf6fa1c7d59e40505b5adeef13dcd82e35.jpg](../icml_results/6174_TOM-SWE_%20User%20Mental%20Modeling%20For%20Software%20Engineering%20Agents/tables/06481fc5b0b0ec319b9916dbb6399abf6fa1c7d59e40505b5adeef13dcd82e35.jpg)

![a062bd844041c306fd5daadb73cc4601ad6e57de96bf98512bd02d43decf964e.jpg](../icml_results/6174_TOM-SWE_%20User%20Mental%20Modeling%20For%20Software%20Engineering%20Agents/tables/a062bd844041c306fd5daadb73cc4601ad6e57de96bf98512bd02d43decf964e.jpg)

![d0a5f0a14ac59e3b584718dbab6abd84d9709b4b252d7345c4e4bcf11f8a55aa.jpg](../icml_results/6174_TOM-SWE_%20User%20Mental%20Modeling%20For%20Software%20Engineering%20Agents/tables/d0a5f0a14ac59e3b584718dbab6abd84d9709b4b252d7345c4e4bcf11f8a55aa.jpg)

<a id="paper-6125"></a>

## SpikeCLR: Self-Supervised Contrastive Learning for Visual Representations with Spiking Neural Networks


### Images

![441bcc7394a2427967ca780e0308cb4e1662f620c7ac8e5ef50f356bd62f5a76.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/images/441bcc7394a2427967ca780e0308cb4e1662f620c7ac8e5ef50f356bd62f5a76.jpg)

![6af3722fc7d848d376d6126b3d2d8e640a60d48ef71a39c97c894c11e22af6f2.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/images/6af3722fc7d848d376d6126b3d2d8e640a60d48ef71a39c97c894c11e22af6f2.jpg)

![79a3a2680e327d8dd317df02a7fd8c7b140cb5b4c8bc044270bc16a13b0e3a6b.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/images/79a3a2680e327d8dd317df02a7fd8c7b140cb5b4c8bc044270bc16a13b0e3a6b.jpg)

![914222634795727857adc5157e1ab138f064c0d6020e948f584eb51dcc9c797e.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/images/914222634795727857adc5157e1ab138f064c0d6020e948f584eb51dcc9c797e.jpg)

![c5aecf67d413b78a140f7329a1ca224e6974979e102ccfd81b2df84830c3a395.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/images/c5aecf67d413b78a140f7329a1ca224e6974979e102ccfd81b2df84830c3a395.jpg)

![ca669c9de9487501b7ef32ea2122f260b250be742f4a722917ccaf26a875d449.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/images/ca669c9de9487501b7ef32ea2122f260b250be742f4a722917ccaf26a875d449.jpg)

![d6e2bd2d9ac4eba503cac74ec39494d7dabba63302dbc4a2874a0b8324125dde.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/images/d6e2bd2d9ac4eba503cac74ec39494d7dabba63302dbc4a2874a0b8324125dde.jpg)

![dbf43724a48e462e10bc96e6576833a5e7134ecc97ee354cf3cb72fa3b123497.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/images/dbf43724a48e462e10bc96e6576833a5e7134ecc97ee354cf3cb72fa3b123497.jpg)

### Tables

![12aec0fc200486d1fa2bf656bd8be509cefc3902d847910be5b395b79c1d0bf6.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/tables/12aec0fc200486d1fa2bf656bd8be509cefc3902d847910be5b395b79c1d0bf6.jpg)

![1832238fc4d61373200eb3aa7a255e43e817bd9fccf76b28579d534e17b012c2.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/tables/1832238fc4d61373200eb3aa7a255e43e817bd9fccf76b28579d534e17b012c2.jpg)

![22d85e4462ef58ca18385fec2d9a0f6cbd43f84bdcc9fba5ff3f54701abf14ce.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/tables/22d85e4462ef58ca18385fec2d9a0f6cbd43f84bdcc9fba5ff3f54701abf14ce.jpg)

![233086600456155b3f20bef1719fd51723f1652a389cd0b11e111df37a76114d.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/tables/233086600456155b3f20bef1719fd51723f1652a389cd0b11e111df37a76114d.jpg)

![38ea82c5c113dbda0a9c8013209e425b0b7256e8fd41752fdb2ae48b93d618d0.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/tables/38ea82c5c113dbda0a9c8013209e425b0b7256e8fd41752fdb2ae48b93d618d0.jpg)

![46cfa93ad53cb3a35b881353e721e95e3c0e83a551ced96de8503ead7d6bc3aa.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/tables/46cfa93ad53cb3a35b881353e721e95e3c0e83a551ced96de8503ead7d6bc3aa.jpg)

![7a9d26f419df549ecc06e07f147ead38758fb6d0b0d5f9f2b6c302d31279b2e8.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/tables/7a9d26f419df549ecc06e07f147ead38758fb6d0b0d5f9f2b6c302d31279b2e8.jpg)

![89ddc9bee5df4abe784870db81987e0bea34b4873d04d5633ba78d6cc41d35fb.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/tables/89ddc9bee5df4abe784870db81987e0bea34b4873d04d5633ba78d6cc41d35fb.jpg)

![af3ecd5d756d918de0206538f04ac32b2dadb70c11b1502f44bf3900c20da07a.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/tables/af3ecd5d756d918de0206538f04ac32b2dadb70c11b1502f44bf3900c20da07a.jpg)

![c748a62ed30c70843f3b128bd56cc540a7dae53dacdb7374e6cdd35f39c5a751.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/tables/c748a62ed30c70843f3b128bd56cc540a7dae53dacdb7374e6cdd35f39c5a751.jpg)

![d154ec5a5f3b597ccc736942fb093f36288c076e42a0d4dc4c124a87d3a7ad84.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/tables/d154ec5a5f3b597ccc736942fb093f36288c076e42a0d4dc4c124a87d3a7ad84.jpg)

![d661d85a280274f4b5156740aaf0fbc6d5e749ef09cb002cbcd37bfc05c23e35.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/tables/d661d85a280274f4b5156740aaf0fbc6d5e749ef09cb002cbcd37bfc05c23e35.jpg)

![d996a7530641d8e9730d175c0ac4e2e3b5c092b4133e8d970d1c0b84eea2c74d.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/tables/d996a7530641d8e9730d175c0ac4e2e3b5c092b4133e8d970d1c0b84eea2c74d.jpg)

![e45d558720dd50ef1fffb199107bb5bee07b315f93aeedd5bb9c628f8743e412.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/tables/e45d558720dd50ef1fffb199107bb5bee07b315f93aeedd5bb9c628f8743e412.jpg)

![e76709ab333db5fd0891f67e6dd12d61d8c2deb909378966d0b470002f1a18e2.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/tables/e76709ab333db5fd0891f67e6dd12d61d8c2deb909378966d0b470002f1a18e2.jpg)

![ea1e191b3b8a366e5353d53c9dc7cf9544e94e24304be3bf2f864365c3b9b295.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/tables/ea1e191b3b8a366e5353d53c9dc7cf9544e94e24304be3bf2f864365c3b9b295.jpg)

![fcebf5442ea74a3a51d539dc76feaea93dc99565a8661c6a3e2c6634c552fd8d.jpg](../icml_results/6175_SpikeCLR_%20Self-Supervised%20Contrastive%20Learning%20for%20Visual%20Representations%20with%20Spiking%20Neural%20Networ/tables/fcebf5442ea74a3a51d539dc76feaea93dc99565a8661c6a3e2c6634c552fd8d.jpg)

<a id="paper-6126"></a>

## PGC: Peak-Guided Calibration for Generalizable AI-Generated Image Detection


### Images

![096457e30ac5a6e45de527e44331bd5d8e86715c9545cc2d257e6a4b18d63841.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/images/096457e30ac5a6e45de527e44331bd5d8e86715c9545cc2d257e6a4b18d63841.jpg)

![0a04df1733d696f8e81fa99ef34f0a44b658b1a63df5ef898d23d1d4ccbf896a.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/images/0a04df1733d696f8e81fa99ef34f0a44b658b1a63df5ef898d23d1d4ccbf896a.jpg)

![289860233ca98496df460f7177dc73bcb0ea974c255a1130ab193b00c037714a.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/images/289860233ca98496df460f7177dc73bcb0ea974c255a1130ab193b00c037714a.jpg)

![3762aa6f45526c56b7c9d5242980e6995c2c4ae1d319e5ee35b492c48208713b.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/images/3762aa6f45526c56b7c9d5242980e6995c2c4ae1d319e5ee35b492c48208713b.jpg)

![6248efef53d530005271ea7e08ea636704d64f6ad299fa7aba24b61349ff8eed.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/images/6248efef53d530005271ea7e08ea636704d64f6ad299fa7aba24b61349ff8eed.jpg)

![6dfd1b897b233277fae0005b739d9b1eaf12db60fb53735eee199ee46c9181b7.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/images/6dfd1b897b233277fae0005b739d9b1eaf12db60fb53735eee199ee46c9181b7.jpg)

![811eacd4ed55f536ad2f87def597dbe6aea5c03021da25d66ecd6ed46956ec01.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/images/811eacd4ed55f536ad2f87def597dbe6aea5c03021da25d66ecd6ed46956ec01.jpg)

![a6bbfe385997d5cd571c4631194c21f00bba16f6c036d7e8264475cbce81fd61.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/images/a6bbfe385997d5cd571c4631194c21f00bba16f6c036d7e8264475cbce81fd61.jpg)

![b51854b20ba7dc38c72398537c32a303cd9b27e6e202d3a6d0044e3d137c6e3e.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/images/b51854b20ba7dc38c72398537c32a303cd9b27e6e202d3a6d0044e3d137c6e3e.jpg)

![d8931599c1f4e759d83c2724a513b07a07e081c280fefa0493e1dbe50aafecd0.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/images/d8931599c1f4e759d83c2724a513b07a07e081c280fefa0493e1dbe50aafecd0.jpg)

![efb9d95d264d34ad0cc42643800421084b81e57da9d5e32452e05dfb7f1df640.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/images/efb9d95d264d34ad0cc42643800421084b81e57da9d5e32452e05dfb7f1df640.jpg)

### Tables

![06b5d85dd9d38e570833e41e1217402409378b1958a782c1ceeb3f2c979fac3c.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/tables/06b5d85dd9d38e570833e41e1217402409378b1958a782c1ceeb3f2c979fac3c.jpg)

![0ff778d6d0d253deb88af3073a62829ff1cbf2f20f81590a8d8860ddc7e8f901.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/tables/0ff778d6d0d253deb88af3073a62829ff1cbf2f20f81590a8d8860ddc7e8f901.jpg)

![22c7c267aa05463dcf58b029202214d171d356fe31d0e2a7ac064667b1ca95f4.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/tables/22c7c267aa05463dcf58b029202214d171d356fe31d0e2a7ac064667b1ca95f4.jpg)

![26e86e155b7c26d4015c98a3000cf9e178f481f50783432246ec9d3b305f42ab.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/tables/26e86e155b7c26d4015c98a3000cf9e178f481f50783432246ec9d3b305f42ab.jpg)

![4339610d4a7180704d1c7af859a3cc5d370212ddb29c9072d4fc17c720e66a22.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/tables/4339610d4a7180704d1c7af859a3cc5d370212ddb29c9072d4fc17c720e66a22.jpg)

![749ebd5b88bce9a228c8b845d6f30695aa758b3caf5e1ccdd3f82106dc781554.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/tables/749ebd5b88bce9a228c8b845d6f30695aa758b3caf5e1ccdd3f82106dc781554.jpg)

![77042383913a4284da9613e4a7b3d0dab53e33d5977f2404b4c67c8758b17fa1.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/tables/77042383913a4284da9613e4a7b3d0dab53e33d5977f2404b4c67c8758b17fa1.jpg)

![9c5699970c5227b96cd43d92e9fc7fe31ad6904566264b3e74fce3b43f073808.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/tables/9c5699970c5227b96cd43d92e9fc7fe31ad6904566264b3e74fce3b43f073808.jpg)

![a271eee5d8f2f622f19d10794fa7a74104ae0d1b87e9c40931bfbbbdbbf5dcf7.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/tables/a271eee5d8f2f622f19d10794fa7a74104ae0d1b87e9c40931bfbbbdbbf5dcf7.jpg)

![b2fe9f5b93faa5f6bc2a25c6499d59ff8b7505967d39f02b88e88ea95e487e75.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/tables/b2fe9f5b93faa5f6bc2a25c6499d59ff8b7505967d39f02b88e88ea95e487e75.jpg)

![b3a3054d5b8cda23253af565f63a00e34dadd1b1ff8771a8d0c33941c81e4da2.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/tables/b3a3054d5b8cda23253af565f63a00e34dadd1b1ff8771a8d0c33941c81e4da2.jpg)

![c00008f5ab549027d5deac39a60cf6b03c25f4e2f430871e4efe639e591477f4.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/tables/c00008f5ab549027d5deac39a60cf6b03c25f4e2f430871e4efe639e591477f4.jpg)

![c50f59994d88d4afc3030e795187658697033dbe136276dedbf065cc9ee9a04a.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/tables/c50f59994d88d4afc3030e795187658697033dbe136276dedbf065cc9ee9a04a.jpg)

![ed9bbb19040fde5c062407002920ab778f1ae85166705c937e8eb8bdc8883fb9.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/tables/ed9bbb19040fde5c062407002920ab778f1ae85166705c937e8eb8bdc8883fb9.jpg)

![f57caac4ca8561e708f7cf97ef044b9ae81144a478d754d757c2c9accef1ccb4.jpg](../icml_results/6176_PGC_%20Peak-Guided%20Calibration%20for%20Generalizable%20AI-Generated%20Image%20Detection/tables/f57caac4ca8561e708f7cf97ef044b9ae81144a478d754d757c2c9accef1ccb4.jpg)

<a id="paper-6127"></a>

## LLawCo: Learning Laws of Cooperation for Modeling Embodied Multi-Agent Behavior


### Images

![05868c26a7ee6d751ddb9a4b2c7bb5d705c1fc556566fde3fa74ec7f310bf516.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/images/05868c26a7ee6d751ddb9a4b2c7bb5d705c1fc556566fde3fa74ec7f310bf516.jpg)

![09a9130d89fedecb3b0c10297c3ef37e46ce6fa88d01cdff4f603324ddc22017.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/images/09a9130d89fedecb3b0c10297c3ef37e46ce6fa88d01cdff4f603324ddc22017.jpg)

![22b500b8f875e4c6501048e43b3768ff785d794c99af5b23fb130ddae2cbba9e.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/images/22b500b8f875e4c6501048e43b3768ff785d794c99af5b23fb130ddae2cbba9e.jpg)

![67f6175e689280fa257595ef74ba496d87df6093cd191476b63764519a4444eb.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/images/67f6175e689280fa257595ef74ba496d87df6093cd191476b63764519a4444eb.jpg)

![9ba40a61e9d8680196e4fe590a69d3f53b5e6f9fbe931378e3dc1e0aa81f17fa.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/images/9ba40a61e9d8680196e4fe590a69d3f53b5e6f9fbe931378e3dc1e0aa81f17fa.jpg)

![e2f5a6629be743776f2dd470bd0e58f7ff69635d5adc6fdbbed04096848a94ed.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/images/e2f5a6629be743776f2dd470bd0e58f7ff69635d5adc6fdbbed04096848a94ed.jpg)

![e33a14c154921658075e3e422f03be9e1edc01d8bfe9a0fa29025d973c74e6ff.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/images/e33a14c154921658075e3e422f03be9e1edc01d8bfe9a0fa29025d973c74e6ff.jpg)

![e977a1470a542e9a9ee021f1df6fe842780362467c6907e38f6c38c8ea125eef.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/images/e977a1470a542e9a9ee021f1df6fe842780362467c6907e38f6c38c8ea125eef.jpg)

### Tables

![0d32070e2d013de462b100fbcb06c4817eebae94ca0608c124b45e09064bf596.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/tables/0d32070e2d013de462b100fbcb06c4817eebae94ca0608c124b45e09064bf596.jpg)

![19b972cd6a5e806811445cba9c843c8e5b306f1ae28e101bc105b9bfe2ad10ff.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/tables/19b972cd6a5e806811445cba9c843c8e5b306f1ae28e101bc105b9bfe2ad10ff.jpg)

![2374f96a5c44e57f88c44adb9152244e9acf8d4f9d07f7875db88c6f4e0668bd.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/tables/2374f96a5c44e57f88c44adb9152244e9acf8d4f9d07f7875db88c6f4e0668bd.jpg)

![29fcdd1d860f575ac3e7e9392f51a3eed20cb6ae5e24d869378ca038fb9b762f.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/tables/29fcdd1d860f575ac3e7e9392f51a3eed20cb6ae5e24d869378ca038fb9b762f.jpg)

![4928b9c406c0f3160efa86f78358461d49c6d381f0b69767db7cee8fdb03c394.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/tables/4928b9c406c0f3160efa86f78358461d49c6d381f0b69767db7cee8fdb03c394.jpg)

![4f2d8e3c4a98f4719fc38cd077e9236534bcfa5bafd0cb8513e6d7ef81f3eead.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/tables/4f2d8e3c4a98f4719fc38cd077e9236534bcfa5bafd0cb8513e6d7ef81f3eead.jpg)

![5a812c17b5683f7cb7b666ebb9185513d67002e8a0d1f4558cf70538e3c1c855.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/tables/5a812c17b5683f7cb7b666ebb9185513d67002e8a0d1f4558cf70538e3c1c855.jpg)

![6464845e190a524a60b062508ea2b9704b8044dbbe5dad1c5216616ae2cee6ca.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/tables/6464845e190a524a60b062508ea2b9704b8044dbbe5dad1c5216616ae2cee6ca.jpg)

![6d90f582329eb24155610dcf81223a6b848bba15cbcee5a55a803bc01e635a46.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/tables/6d90f582329eb24155610dcf81223a6b848bba15cbcee5a55a803bc01e635a46.jpg)

![9953ec6dcac3f4091698097f8c2c5d84f631a418bf4dea0192b6756c81fffca2.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/tables/9953ec6dcac3f4091698097f8c2c5d84f631a418bf4dea0192b6756c81fffca2.jpg)

![9d9ad75f7cc6962f6d11b63fe13f31b759764cc4e6c6605797156bb24037bf52.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/tables/9d9ad75f7cc6962f6d11b63fe13f31b759764cc4e6c6605797156bb24037bf52.jpg)

![b7b08d233c5dbafc733efc98835e00224f398f70989319daf44daf4b9484a1a9.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/tables/b7b08d233c5dbafc733efc98835e00224f398f70989319daf44daf4b9484a1a9.jpg)

![b908d02ab1ee35df0648f938bb46a58406b55f491466085f6386ad8c61859537.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/tables/b908d02ab1ee35df0648f938bb46a58406b55f491466085f6386ad8c61859537.jpg)

![e229eb69a99019c1a38819c6ff842ebfea1c2e6778e07cc9705d0b75bfaea337.jpg](../icml_results/6177_LLawCo_%20Learning%20Laws%20of%20Cooperation%20for%20Modeling%20Embodied%20Multi-Agent%20Behavior/tables/e229eb69a99019c1a38819c6ff842ebfea1c2e6778e07cc9705d0b75bfaea337.jpg)

<a id="paper-6128"></a>

## How Out-of-Distribution Detection Learning Theory Enhances Transformer: Learnability and Reliability


### Images

![23bcfecb2d70e19e48b375ca682341336dc2d5bfab3155faacc2fca9c31ea353.jpg](../icml_results/6178_How%20Out-of-Distribution%20Detection%20Learning%20Theory%20Enhances%20Transformer_%20Learnability%20and%20Reliability/images/23bcfecb2d70e19e48b375ca682341336dc2d5bfab3155faacc2fca9c31ea353.jpg)

![5e75d389c996eec210cff037891ddba2bbba0cfad6b746cf3c72574401f55f37.jpg](../icml_results/6178_How%20Out-of-Distribution%20Detection%20Learning%20Theory%20Enhances%20Transformer_%20Learnability%20and%20Reliability/images/5e75d389c996eec210cff037891ddba2bbba0cfad6b746cf3c72574401f55f37.jpg)

![6f3cd01ebf2f1b2666e0cff2863f8574e55bb5e74c42062a3beacf675d4a6618.jpg](../icml_results/6178_How%20Out-of-Distribution%20Detection%20Learning%20Theory%20Enhances%20Transformer_%20Learnability%20and%20Reliability/images/6f3cd01ebf2f1b2666e0cff2863f8574e55bb5e74c42062a3beacf675d4a6618.jpg)

![782d88e7219d4d73023221abf32257b8ed1d3d30074e7bb3e8cefa062bd05337.jpg](../icml_results/6178_How%20Out-of-Distribution%20Detection%20Learning%20Theory%20Enhances%20Transformer_%20Learnability%20and%20Reliability/images/782d88e7219d4d73023221abf32257b8ed1d3d30074e7bb3e8cefa062bd05337.jpg)

![86bdb24d4839b1b3a32fb5145f9f1586d49f02632d20354276c985cac7d8727b.jpg](../icml_results/6178_How%20Out-of-Distribution%20Detection%20Learning%20Theory%20Enhances%20Transformer_%20Learnability%20and%20Reliability/images/86bdb24d4839b1b3a32fb5145f9f1586d49f02632d20354276c985cac7d8727b.jpg)

![88f4725519ea71859d5266bec9b5d97ae93bb8cbbcd729f2c721f20766983db4.jpg](../icml_results/6178_How%20Out-of-Distribution%20Detection%20Learning%20Theory%20Enhances%20Transformer_%20Learnability%20and%20Reliability/images/88f4725519ea71859d5266bec9b5d97ae93bb8cbbcd729f2c721f20766983db4.jpg)

![f5b55e10485d70359eafd6637a21f2ae44fe63a59b6a8488a74122427a6b8187.jpg](../icml_results/6178_How%20Out-of-Distribution%20Detection%20Learning%20Theory%20Enhances%20Transformer_%20Learnability%20and%20Reliability/images/f5b55e10485d70359eafd6637a21f2ae44fe63a59b6a8488a74122427a6b8187.jpg)

### Tables

![33e7862b0238fc9b6a8ad10ee155616efce47db939630580937b8d3642c0ffb9.jpg](../icml_results/6178_How%20Out-of-Distribution%20Detection%20Learning%20Theory%20Enhances%20Transformer_%20Learnability%20and%20Reliability/tables/33e7862b0238fc9b6a8ad10ee155616efce47db939630580937b8d3642c0ffb9.jpg)

![3524b25283007e83b5377f66854c0989ef7bb8ae48fc76d57b0d7795a0d45900.jpg](../icml_results/6178_How%20Out-of-Distribution%20Detection%20Learning%20Theory%20Enhances%20Transformer_%20Learnability%20and%20Reliability/tables/3524b25283007e83b5377f66854c0989ef7bb8ae48fc76d57b0d7795a0d45900.jpg)

![3eff033c588b6934848a030aa6fe56355b3e09830bf3426e5dc20f9b602dad4c.jpg](../icml_results/6178_How%20Out-of-Distribution%20Detection%20Learning%20Theory%20Enhances%20Transformer_%20Learnability%20and%20Reliability/tables/3eff033c588b6934848a030aa6fe56355b3e09830bf3426e5dc20f9b602dad4c.jpg)

![61cc71a1d98bf1dc3c681e4b0a5b3565e067f63d8ec2760c12b13c872279d181.jpg](../icml_results/6178_How%20Out-of-Distribution%20Detection%20Learning%20Theory%20Enhances%20Transformer_%20Learnability%20and%20Reliability/tables/61cc71a1d98bf1dc3c681e4b0a5b3565e067f63d8ec2760c12b13c872279d181.jpg)

![752a82527ffebefcc649f14f0dc5a07bedd0d337886c4e595432da92b49f1f23.jpg](../icml_results/6178_How%20Out-of-Distribution%20Detection%20Learning%20Theory%20Enhances%20Transformer_%20Learnability%20and%20Reliability/tables/752a82527ffebefcc649f14f0dc5a07bedd0d337886c4e595432da92b49f1f23.jpg)

![8d23356b1320e6612e50804b6a8a9e186b71bfe6b1e1d9a9528eefafc464d1ec.jpg](../icml_results/6178_How%20Out-of-Distribution%20Detection%20Learning%20Theory%20Enhances%20Transformer_%20Learnability%20and%20Reliability/tables/8d23356b1320e6612e50804b6a8a9e186b71bfe6b1e1d9a9528eefafc464d1ec.jpg)

![9b554b463ac051b843456f9c788ec6b8527578b36139d3f76fa7766c3fdd4171.jpg](../icml_results/6178_How%20Out-of-Distribution%20Detection%20Learning%20Theory%20Enhances%20Transformer_%20Learnability%20and%20Reliability/tables/9b554b463ac051b843456f9c788ec6b8527578b36139d3f76fa7766c3fdd4171.jpg)

![a8abf038f8ecffe04829c17abd7458c8f4c49e10e023a3e68dae31f87eeca697.jpg](../icml_results/6178_How%20Out-of-Distribution%20Detection%20Learning%20Theory%20Enhances%20Transformer_%20Learnability%20and%20Reliability/tables/a8abf038f8ecffe04829c17abd7458c8f4c49e10e023a3e68dae31f87eeca697.jpg)

<a id="paper-6129"></a>

## KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices


### Images

![0d4dfd26e252a0ed293bca93a97a68a195fea86a75579d2b827a03ec39603367.jpg](../icml_results/6179_KromHC_%20Manifold-Constrained%20Hyper-Connections%20with%20Kronecker-Product%20Residual%20Matrices/images/0d4dfd26e252a0ed293bca93a97a68a195fea86a75579d2b827a03ec39603367.jpg)

![139590b2fa6e988edf59498bbd3e0f70635e839ef08666342c44ec03f8fddcb2.jpg](../icml_results/6179_KromHC_%20Manifold-Constrained%20Hyper-Connections%20with%20Kronecker-Product%20Residual%20Matrices/images/139590b2fa6e988edf59498bbd3e0f70635e839ef08666342c44ec03f8fddcb2.jpg)

![35bb0917e7363b80c8a7384cfbfc83f9645f6e9a61b6384fcfa5e5ac34ae812f.jpg](../icml_results/6179_KromHC_%20Manifold-Constrained%20Hyper-Connections%20with%20Kronecker-Product%20Residual%20Matrices/images/35bb0917e7363b80c8a7384cfbfc83f9645f6e9a61b6384fcfa5e5ac34ae812f.jpg)

![7181feae935dacd79ecd0d3af6a8a1ae7eb88060f9776d7e8c1e843ee70e3ea3.jpg](../icml_results/6179_KromHC_%20Manifold-Constrained%20Hyper-Connections%20with%20Kronecker-Product%20Residual%20Matrices/images/7181feae935dacd79ecd0d3af6a8a1ae7eb88060f9776d7e8c1e843ee70e3ea3.jpg)

![823e98765ae9606f4155def5cc38d0fce635d5b18683d32cd3031b25aa7160cb.jpg](../icml_results/6179_KromHC_%20Manifold-Constrained%20Hyper-Connections%20with%20Kronecker-Product%20Residual%20Matrices/images/823e98765ae9606f4155def5cc38d0fce635d5b18683d32cd3031b25aa7160cb.jpg)

![c2ad00fa6e3ffec00038a106d236519baedf1f9a9041cb2b411d94c8420107f3.jpg](../icml_results/6179_KromHC_%20Manifold-Constrained%20Hyper-Connections%20with%20Kronecker-Product%20Residual%20Matrices/images/c2ad00fa6e3ffec00038a106d236519baedf1f9a9041cb2b411d94c8420107f3.jpg)

![d29db126a8d2795edcd8c1bd51b0c0730d71ae46c8dc6d7ca3a62854d9d3795a.jpg](../icml_results/6179_KromHC_%20Manifold-Constrained%20Hyper-Connections%20with%20Kronecker-Product%20Residual%20Matrices/images/d29db126a8d2795edcd8c1bd51b0c0730d71ae46c8dc6d7ca3a62854d9d3795a.jpg)

![d6ef2ae125b2a8deb11c99cf5abdb3eb7dd810f4ad43008ab3c007d7a07ccada.jpg](../icml_results/6179_KromHC_%20Manifold-Constrained%20Hyper-Connections%20with%20Kronecker-Product%20Residual%20Matrices/images/d6ef2ae125b2a8deb11c99cf5abdb3eb7dd810f4ad43008ab3c007d7a07ccada.jpg)

### Tables

![0482980850fa06a2f8a29e5a2391942a54c0ebcb7bca8f30349232b5556b3d8b.jpg](../icml_results/6179_KromHC_%20Manifold-Constrained%20Hyper-Connections%20with%20Kronecker-Product%20Residual%20Matrices/tables/0482980850fa06a2f8a29e5a2391942a54c0ebcb7bca8f30349232b5556b3d8b.jpg)

![1449ab8c28b781d18cdead02a6fa51496034e502a4ea651c94ac7ad7b2f866dc.jpg](../icml_results/6179_KromHC_%20Manifold-Constrained%20Hyper-Connections%20with%20Kronecker-Product%20Residual%20Matrices/tables/1449ab8c28b781d18cdead02a6fa51496034e502a4ea651c94ac7ad7b2f866dc.jpg)

![2ffd399be5f7996a6d79c4057484cfaae8681def42862d41f2cc719d13634bc2.jpg](../icml_results/6179_KromHC_%20Manifold-Constrained%20Hyper-Connections%20with%20Kronecker-Product%20Residual%20Matrices/tables/2ffd399be5f7996a6d79c4057484cfaae8681def42862d41f2cc719d13634bc2.jpg)

![63a0b10f18781a563eac5a0eac65d92fc0308398b628664d03ad7c1b5e415abe.jpg](../icml_results/6179_KromHC_%20Manifold-Constrained%20Hyper-Connections%20with%20Kronecker-Product%20Residual%20Matrices/tables/63a0b10f18781a563eac5a0eac65d92fc0308398b628664d03ad7c1b5e415abe.jpg)

![8a83588ff1c81a02f3d6136d38449a35f1c0f186b0514356d9ee6b668f8f5716.jpg](../icml_results/6179_KromHC_%20Manifold-Constrained%20Hyper-Connections%20with%20Kronecker-Product%20Residual%20Matrices/tables/8a83588ff1c81a02f3d6136d38449a35f1c0f186b0514356d9ee6b668f8f5716.jpg)

![8aa08519849392cae6f507234dddbfecfcda4a3ef9193651f1827944460de307.jpg](../icml_results/6179_KromHC_%20Manifold-Constrained%20Hyper-Connections%20with%20Kronecker-Product%20Residual%20Matrices/tables/8aa08519849392cae6f507234dddbfecfcda4a3ef9193651f1827944460de307.jpg)

![c512dfcb762bbd7e6fc4bf6b19799c1c6de74de2c82db63189d9cc68667601e6.jpg](../icml_results/6179_KromHC_%20Manifold-Constrained%20Hyper-Connections%20with%20Kronecker-Product%20Residual%20Matrices/tables/c512dfcb762bbd7e6fc4bf6b19799c1c6de74de2c82db63189d9cc68667601e6.jpg)

![d13e269035de8940652872a59442114611c4ef6242b5334e30e6dae0ceaacfea.jpg](../icml_results/6179_KromHC_%20Manifold-Constrained%20Hyper-Connections%20with%20Kronecker-Product%20Residual%20Matrices/tables/d13e269035de8940652872a59442114611c4ef6242b5334e30e6dae0ceaacfea.jpg)

![f5f9cdab2ad4ccf7a2a3bf2e56dcea14a04b2a6d4e6ea59aa385aea29780cf72.jpg](../icml_results/6179_KromHC_%20Manifold-Constrained%20Hyper-Connections%20with%20Kronecker-Product%20Residual%20Matrices/tables/f5f9cdab2ad4ccf7a2a3bf2e56dcea14a04b2a6d4e6ea59aa385aea29780cf72.jpg)

<a id="paper-6130"></a>

## OPTION: Optimal Transport–Guided Flow Matching for Incomplete and Unaligned Multi-View Clustering


### Images

![1db344fa924db1f8352dd07ac1e352d605021e45cf6b57835f657d5a38c543e2.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/images/1db344fa924db1f8352dd07ac1e352d605021e45cf6b57835f657d5a38c543e2.jpg)

![223bd7182bc5079bcb54ad5fb0752d78a942762ede528e916fa4a1a6bc467175.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/images/223bd7182bc5079bcb54ad5fb0752d78a942762ede528e916fa4a1a6bc467175.jpg)

![273109a5162b6964d425f08616d3eb27c34a8fbe15b6f3927cc9d964bb1d68fc.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/images/273109a5162b6964d425f08616d3eb27c34a8fbe15b6f3927cc9d964bb1d68fc.jpg)

![4000d04bf558e2cb52053c4e2d10983276d3fa58de10e023b938440229cbac98.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/images/4000d04bf558e2cb52053c4e2d10983276d3fa58de10e023b938440229cbac98.jpg)

![40065e310ba3a2d44d31425fa072e5a96694e730b11f1a4c74da4d1dfa5cc300.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/images/40065e310ba3a2d44d31425fa072e5a96694e730b11f1a4c74da4d1dfa5cc300.jpg)

![6e8e6414725baec5cd331875499f5a0a1868a51b42b2bf2abd0f78d5e7ecfca0.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/images/6e8e6414725baec5cd331875499f5a0a1868a51b42b2bf2abd0f78d5e7ecfca0.jpg)

![7e8ee0624dd11031b2edbc07f53aa3e49026b83cec416d78ae49dc8895f82ebc.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/images/7e8ee0624dd11031b2edbc07f53aa3e49026b83cec416d78ae49dc8895f82ebc.jpg)

![9624fb84347de2ce0b7cdf86ab4f4bd52c91fac431b74aec048616854fe3fec2.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/images/9624fb84347de2ce0b7cdf86ab4f4bd52c91fac431b74aec048616854fe3fec2.jpg)

![9e872900c855c8f9ebba1154a5ede88b4cdb06df9787b1bdb12c4dedb3752a19.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/images/9e872900c855c8f9ebba1154a5ede88b4cdb06df9787b1bdb12c4dedb3752a19.jpg)

![9fe1f422c80694b5c1e8b8ab1f0dd5565e48409486d69bf5214dc3a5d5c73693.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/images/9fe1f422c80694b5c1e8b8ab1f0dd5565e48409486d69bf5214dc3a5d5c73693.jpg)

![dd90c061f76c7ac46290e4eedf8eb92bab7c2678ee79d58dd0b3ba9ab807bcf7.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/images/dd90c061f76c7ac46290e4eedf8eb92bab7c2678ee79d58dd0b3ba9ab807bcf7.jpg)

![e9dad2bfd045182abe2c70569ed3ea590c8d66efa96e6df4ebd19ce506388282.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/images/e9dad2bfd045182abe2c70569ed3ea590c8d66efa96e6df4ebd19ce506388282.jpg)

### Tables

![056d07fd62149de2d930f2472ff8df2e15f25114aa7b3df3c0f97ca0ae774649.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/tables/056d07fd62149de2d930f2472ff8df2e15f25114aa7b3df3c0f97ca0ae774649.jpg)

![1163ca5e35237fb3f02e2d69710e6cc6c6ba49ffcd4fb44bd6a78bf3c501eb1b.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/tables/1163ca5e35237fb3f02e2d69710e6cc6c6ba49ffcd4fb44bd6a78bf3c501eb1b.jpg)

![18933906718423719fbd9eb69ba933e61a706a85a1a8e49dc2d31f7d154f7fc5.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/tables/18933906718423719fbd9eb69ba933e61a706a85a1a8e49dc2d31f7d154f7fc5.jpg)

![4b3d79ef7626654515a359ccf3e793a9d5b9453f86912075ce8d05190471b725.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/tables/4b3d79ef7626654515a359ccf3e793a9d5b9453f86912075ce8d05190471b725.jpg)

![4e4b44fb752a65ef1684071674e476b588795a6a0433347c970119cbb495b777.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/tables/4e4b44fb752a65ef1684071674e476b588795a6a0433347c970119cbb495b777.jpg)

![5b4c7e1bb7b7ec343c02a049ff1324441b3d1ca27896a187cc576aa84e4e9ae3.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/tables/5b4c7e1bb7b7ec343c02a049ff1324441b3d1ca27896a187cc576aa84e4e9ae3.jpg)

![6282827eb5171b97e2b70c8faf08116b917f70ff471cb4ead2767cc23c9fd49e.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/tables/6282827eb5171b97e2b70c8faf08116b917f70ff471cb4ead2767cc23c9fd49e.jpg)

![6ead24cb5c19f55791bb2b34ef00a988ea50ae3d3ce12a7dbe4fbf1b4af5a85a.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/tables/6ead24cb5c19f55791bb2b34ef00a988ea50ae3d3ce12a7dbe4fbf1b4af5a85a.jpg)

![89170819daf08d8fd2b700c300302f8537c15fd4f6f33e30f0a51ef5e9531042.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/tables/89170819daf08d8fd2b700c300302f8537c15fd4f6f33e30f0a51ef5e9531042.jpg)

![8aa7a00eb6f1fbb8464ca8beed96b8309b2595aefeef528b37652ea559c7a4a6.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/tables/8aa7a00eb6f1fbb8464ca8beed96b8309b2595aefeef528b37652ea559c7a4a6.jpg)

![969baf38d0105e170ea5c8e192e641167bf3fbdc2e9821c9cf2375d19f69a20b.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/tables/969baf38d0105e170ea5c8e192e641167bf3fbdc2e9821c9cf2375d19f69a20b.jpg)

![affbf26ca32ced535b11dfe7350c26febba280d6efc093d7a700682a78a58ee7.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/tables/affbf26ca32ced535b11dfe7350c26febba280d6efc093d7a700682a78a58ee7.jpg)

![d53de3d88367ab14208d13d955ad3df7bb4638f98ab11dc191131085d09bcba8.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/tables/d53de3d88367ab14208d13d955ad3df7bb4638f98ab11dc191131085d09bcba8.jpg)

![dd756167c698a68fa98f9163970f479229db2742072676d9100a3e698ebc6718.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/tables/dd756167c698a68fa98f9163970f479229db2742072676d9100a3e698ebc6718.jpg)

![ea45b457272221f2ecc97457f704b80ca8c5aaf47d0e46f0e814e50497c5cb25.jpg](../icml_results/6180_OPTION_%20Optimal%20Transport%E2%80%93Guided%20Flow%20Matching%20for%20Incomplete%20and%20Unaligned%20Multi-View%20Clustering/tables/ea45b457272221f2ecc97457f704b80ca8c5aaf47d0e46f0e814e50497c5cb25.jpg)

<a id="paper-6131"></a>

## QPoint: End-to-End Lightweight Point Cloud Processing via Robust Quaternion Feature Learning


### Images

![0c45f2d685ad8d9b2a4ef766eebbbf360e7fc9eeb353d0da0814cc65d4af51f7.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/images/0c45f2d685ad8d9b2a4ef766eebbbf360e7fc9eeb353d0da0814cc65d4af51f7.jpg)

![12be30995cf7992b8d4f525724e36786ca68d21d68365caaad096877d3b4cb17.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/images/12be30995cf7992b8d4f525724e36786ca68d21d68365caaad096877d3b4cb17.jpg)

![5a80b9b5673f4639025941f12f7df3b69f2cff80f255a9d5be998e4a3ce0fb0b.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/images/5a80b9b5673f4639025941f12f7df3b69f2cff80f255a9d5be998e4a3ce0fb0b.jpg)

![ab96c10b359923e1a881a75eca21e4d2fa178db1a6e524a05351a13f5ef9f322.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/images/ab96c10b359923e1a881a75eca21e4d2fa178db1a6e524a05351a13f5ef9f322.jpg)

![cd49e52c85f73fae363bbae2971a606d320f7608536601aeb3f9e0d7374ec5a8.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/images/cd49e52c85f73fae363bbae2971a606d320f7608536601aeb3f9e0d7374ec5a8.jpg)

![d183c8597292dd22256f3419588d762009393ccb1fd909c5a6edb8d1e266371e.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/images/d183c8597292dd22256f3419588d762009393ccb1fd909c5a6edb8d1e266371e.jpg)

![ffa780d763fa1146bc672849d67bdc934923c836cfb4742d6e49a998a5f682bf.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/images/ffa780d763fa1146bc672849d67bdc934923c836cfb4742d6e49a998a5f682bf.jpg)

### Tables

![0411903f128887c451fc1420a0bb8945fa7de4cbd0bb4c1b5e1e5928403065d7.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/tables/0411903f128887c451fc1420a0bb8945fa7de4cbd0bb4c1b5e1e5928403065d7.jpg)

![238106858fb88fb9de2c44459a4ed4efda98af532357096eb408b8e14c7548ff.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/tables/238106858fb88fb9de2c44459a4ed4efda98af532357096eb408b8e14c7548ff.jpg)

![30e6804e846bbc0fc2bbfae8b35c481e8899b445a37c8bf54fcbf8a69b35e385.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/tables/30e6804e846bbc0fc2bbfae8b35c481e8899b445a37c8bf54fcbf8a69b35e385.jpg)

![44b3aaf1381cee78d6ef2a358be9d2b07e0f6738073e8a9ae52a3a3cbbd3f243.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/tables/44b3aaf1381cee78d6ef2a358be9d2b07e0f6738073e8a9ae52a3a3cbbd3f243.jpg)

![5dfaf478b02f16f4bc6e14f35851237232cb2708338d246b6900cae043a7f21c.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/tables/5dfaf478b02f16f4bc6e14f35851237232cb2708338d246b6900cae043a7f21c.jpg)

![94c7b292ccb89a35b834ae55c2c957d96f8942ffdc718263c4a6f164e2f649b7.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/tables/94c7b292ccb89a35b834ae55c2c957d96f8942ffdc718263c4a6f164e2f649b7.jpg)

![acec3e132d5cf6e82d3ed1a6387f8d7a70bd8566b8161e9bc370352545c88d92.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/tables/acec3e132d5cf6e82d3ed1a6387f8d7a70bd8566b8161e9bc370352545c88d92.jpg)

![d1a1a8e51ddd20e66f9498ecdcefeb4724f4f934d0cf1458e3b3c9769815690d.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/tables/d1a1a8e51ddd20e66f9498ecdcefeb4724f4f934d0cf1458e3b3c9769815690d.jpg)

![f17eb187d837d57ccd1415ff41fa3bd4744b3779c2e6c9b8bd6aab2948218fc6.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/tables/f17eb187d837d57ccd1415ff41fa3bd4744b3779c2e6c9b8bd6aab2948218fc6.jpg)

![f7e3bdfbd40244be444c0692180cbd63b3f89d90a9eb4fe1acc9f6c0cc035fac.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/tables/f7e3bdfbd40244be444c0692180cbd63b3f89d90a9eb4fe1acc9f6c0cc035fac.jpg)

![feed2d29f2cfdaf66f78dcb729592c78e9280ffb26ee61e4e5ce6fd91d39334e.jpg](../icml_results/6181_QPoint_%20End-to-End%20Lightweight%20Point%20Cloud%20Processing%20via%20Robust%20Quaternion%20Feature%20Learning/tables/feed2d29f2cfdaf66f78dcb729592c78e9280ffb26ee61e4e5ce6fd91d39334e.jpg)

<a id="paper-6132"></a>

## SpikeNet: Sparse Spike-Driven Mask Vector Transformer for Energy-Efficient and Stable Spiking Point Cloud Processing


### Images

![3079f1ad9fe5ffce0afd54605cd5d6e2840380e0f2eea75b03e47bc40505f860.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/images/3079f1ad9fe5ffce0afd54605cd5d6e2840380e0f2eea75b03e47bc40505f860.jpg)

![3e4980086cded846b76eeac3fb2d43220b66b9d33dd83b18393d00035f9cac1c.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/images/3e4980086cded846b76eeac3fb2d43220b66b9d33dd83b18393d00035f9cac1c.jpg)

![532f64c374987da7ee0c5b465ef32f1e5e2cb06798befa107ba5ee4555c17967.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/images/532f64c374987da7ee0c5b465ef32f1e5e2cb06798befa107ba5ee4555c17967.jpg)

![8d17efc81cc9cfc85d830718105e2c629c2d5d39fb88f2d25a1b6b904185be04.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/images/8d17efc81cc9cfc85d830718105e2c629c2d5d39fb88f2d25a1b6b904185be04.jpg)

![90df6b2575445d0b59fd40b5abc5172d85e2b9b18904aae91617ebe48fb78906.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/images/90df6b2575445d0b59fd40b5abc5172d85e2b9b18904aae91617ebe48fb78906.jpg)

![a825595706417054897894074f0fc5ea64ddde33f772773eb90fe09a097b2ab3.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/images/a825595706417054897894074f0fc5ea64ddde33f772773eb90fe09a097b2ab3.jpg)

![c955e66ca4d150eec698271e21fc05a8f50059edabf6550256eed8b0b9027e88.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/images/c955e66ca4d150eec698271e21fc05a8f50059edabf6550256eed8b0b9027e88.jpg)

![e46f6d837b3a7c3dbd5c3cb3307c45d63abdfaacb07dc7c10da37e16525338e4.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/images/e46f6d837b3a7c3dbd5c3cb3307c45d63abdfaacb07dc7c10da37e16525338e4.jpg)

### Tables

![0b9c07e0105e17cb946fe0dab5925b1f4d2a37890658da266beab5346002e3a3.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/0b9c07e0105e17cb946fe0dab5925b1f4d2a37890658da266beab5346002e3a3.jpg)

![1b280645eacdc94be7ef221c2890143de76b41092ffcf00d3b836f2bc9ae1ade.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/1b280645eacdc94be7ef221c2890143de76b41092ffcf00d3b836f2bc9ae1ade.jpg)

![22afa90edfa055637f1c2e2df8fb4c2608ace40f0747d727691470e50cd1b4ba.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/22afa90edfa055637f1c2e2df8fb4c2608ace40f0747d727691470e50cd1b4ba.jpg)

![280b938ae718be88916b727452ebc5851dddff0d9e4940fba9318c73ba7c282b.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/280b938ae718be88916b727452ebc5851dddff0d9e4940fba9318c73ba7c282b.jpg)

![29918212f69e848bf84c65c2981e91add54afa4795f4d78e78c7a1c3b14db4be.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/29918212f69e848bf84c65c2981e91add54afa4795f4d78e78c7a1c3b14db4be.jpg)

![2e2732b707d4170f4ce45adf36d09e1fa73c8e2018c73d620fb03467c969dd51.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/2e2732b707d4170f4ce45adf36d09e1fa73c8e2018c73d620fb03467c969dd51.jpg)

![3c2b17af3723c39844977c7dad22b756254105b2ebe31206a17ede6ff38b49bd.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/3c2b17af3723c39844977c7dad22b756254105b2ebe31206a17ede6ff38b49bd.jpg)

![40e2b7df1874e17592f3746fc93a0bb3cab6e724cf572877a3fc8e9e52430737.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/40e2b7df1874e17592f3746fc93a0bb3cab6e724cf572877a3fc8e9e52430737.jpg)

![4fed1408b04bd240df936a86bcd83c34fb14026c138562aafbdf0486f678cb02.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/4fed1408b04bd240df936a86bcd83c34fb14026c138562aafbdf0486f678cb02.jpg)

![7027a791c4ef24b2de903e3803c1f5e83b3b534fb3845e035923d6b656e8d486.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/7027a791c4ef24b2de903e3803c1f5e83b3b534fb3845e035923d6b656e8d486.jpg)

![7461c9d815b7d13710678ec6ce58db168aa7a066b891f3a4e097270ccbc5ba6a.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/7461c9d815b7d13710678ec6ce58db168aa7a066b891f3a4e097270ccbc5ba6a.jpg)

![81a04748be327c991853558edc1e2c46e970a09007da599af8742c17c22a0d0d.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/81a04748be327c991853558edc1e2c46e970a09007da599af8742c17c22a0d0d.jpg)

![8becbbfe48ae482fef30d0aedd0375afeaa8ac508005ea230d5e26e3f6bccd94.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/8becbbfe48ae482fef30d0aedd0375afeaa8ac508005ea230d5e26e3f6bccd94.jpg)

![9bbd41240097ada3ea99a0f3e2c87794822ccd15b68a04c4705146afc5e1659c.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/9bbd41240097ada3ea99a0f3e2c87794822ccd15b68a04c4705146afc5e1659c.jpg)

![9e9bab8a24d00c9172ca25cb6d8c5701c67ae3e2374109b8d9d1302a74df1c58.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/9e9bab8a24d00c9172ca25cb6d8c5701c67ae3e2374109b8d9d1302a74df1c58.jpg)

![a10a53dc42c35f868aeeefe32857fbb23745ad74a77208fc7b9343d7f002c381.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/a10a53dc42c35f868aeeefe32857fbb23745ad74a77208fc7b9343d7f002c381.jpg)

![b16c83d33bdb4a7d745ce07543fedfc0d723b65145e3d7fa3ec00df8e66be482.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/b16c83d33bdb4a7d745ce07543fedfc0d723b65145e3d7fa3ec00df8e66be482.jpg)

![d4a4adcec2057b4e3b38ccff0f5f98cc0e31ab08e93a29896537c81e3e93650d.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/d4a4adcec2057b4e3b38ccff0f5f98cc0e31ab08e93a29896537c81e3e93650d.jpg)

![e2ac209bbbd8aab9b19bfddacef72067a15f87a28f04f052872d7136f19b68ef.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/e2ac209bbbd8aab9b19bfddacef72067a15f87a28f04f052872d7136f19b68ef.jpg)

![f5e4b5941a7a492f3eca8ebf637296d413cb778194d8b87866d85925474f58b7.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/f5e4b5941a7a492f3eca8ebf637296d413cb778194d8b87866d85925474f58b7.jpg)

![f7d594e7709a7e91e23b5c94e27ed9ee41ff67c7988284bb7fcca28626bf9ef0.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/f7d594e7709a7e91e23b5c94e27ed9ee41ff67c7988284bb7fcca28626bf9ef0.jpg)

![f82dab1c08ba11a61a617d8d2e9f5b95acf0601704f2218dbfbafe879eb2f6f1.jpg](../icml_results/6182_SpikeNet_%20Sparse%20Spike-Driven%20Mask%20Vector%20Transformer%20for%20Energy-Efficient%20and%20Stable%20Spiking%20Point%20/tables/f82dab1c08ba11a61a617d8d2e9f5b95acf0601704f2218dbfbafe879eb2f6f1.jpg)

<a id="paper-6133"></a>

## Deep Neural Network Regression with Functional Covariates


### Images

![045ca761556f3f8140a739c2687d1aef2f621fa4ef4edbeee98ddcd54a4789bf.jpg](../icml_results/6183_Deep%20Neural%20Network%20Regression%20with%20Functional%20Covariates/images/045ca761556f3f8140a739c2687d1aef2f621fa4ef4edbeee98ddcd54a4789bf.jpg)

![0c20472d0bc2570cf03bc6ba8ffee3fda8b74087539d44ee5ab3dd0cbc271a18.jpg](../icml_results/6183_Deep%20Neural%20Network%20Regression%20with%20Functional%20Covariates/images/0c20472d0bc2570cf03bc6ba8ffee3fda8b74087539d44ee5ab3dd0cbc271a18.jpg)

![2887ab40802fb8809f0f06a9ca00f3b355f724ab838c8003d737020486d6e876.jpg](../icml_results/6183_Deep%20Neural%20Network%20Regression%20with%20Functional%20Covariates/images/2887ab40802fb8809f0f06a9ca00f3b355f724ab838c8003d737020486d6e876.jpg)

![350a4c24d5e9fabd4c7b74d7135c0eb1056a292a08135da22c28056481685efd.jpg](../icml_results/6183_Deep%20Neural%20Network%20Regression%20with%20Functional%20Covariates/images/350a4c24d5e9fabd4c7b74d7135c0eb1056a292a08135da22c28056481685efd.jpg)

![7c92d60dd9761969679be6ec230692a2a872cd5536d4775bdbcfdfe8995df285.jpg](../icml_results/6183_Deep%20Neural%20Network%20Regression%20with%20Functional%20Covariates/images/7c92d60dd9761969679be6ec230692a2a872cd5536d4775bdbcfdfe8995df285.jpg)

![be83e2a1c4c28ff8acb0275dd512ac31c7a6895797cf23a4c646ada776f6d135.jpg](../icml_results/6183_Deep%20Neural%20Network%20Regression%20with%20Functional%20Covariates/images/be83e2a1c4c28ff8acb0275dd512ac31c7a6895797cf23a4c646ada776f6d135.jpg)

![cc9965a03c26f8822beb7a7f2315fedb635ede9496df0da254606c4b65662bbe.jpg](../icml_results/6183_Deep%20Neural%20Network%20Regression%20with%20Functional%20Covariates/images/cc9965a03c26f8822beb7a7f2315fedb635ede9496df0da254606c4b65662bbe.jpg)

![dcfdc9487d71f3dbc968738e649e85e520cfc49be4100dd9ba3b569cc096b9db.jpg](../icml_results/6183_Deep%20Neural%20Network%20Regression%20with%20Functional%20Covariates/images/dcfdc9487d71f3dbc968738e649e85e520cfc49be4100dd9ba3b569cc096b9db.jpg)

![e0f033e86a9c47c14772e61f1069b5420f82f827537cb88f7646214fd984567e.jpg](../icml_results/6183_Deep%20Neural%20Network%20Regression%20with%20Functional%20Covariates/images/e0f033e86a9c47c14772e61f1069b5420f82f827537cb88f7646214fd984567e.jpg)

![ec318020db5006e834dd10da14cc7ce5cdda08a17aba1646972cb77116b81998.jpg](../icml_results/6183_Deep%20Neural%20Network%20Regression%20with%20Functional%20Covariates/images/ec318020db5006e834dd10da14cc7ce5cdda08a17aba1646972cb77116b81998.jpg)

<a id="paper-6134"></a>

## Alignment Risks from Capability-Seeking RL Training


### Images

![3eb846402cbed5290e602e4b69a78e8423219288c2fbe0aaec50954456bb0805.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/images/3eb846402cbed5290e602e4b69a78e8423219288c2fbe0aaec50954456bb0805.jpg)

![40f2bd586083ded63dca04ae6cffa6210e11f72e1055617db30afbbfd3676dd1.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/images/40f2bd586083ded63dca04ae6cffa6210e11f72e1055617db30afbbfd3676dd1.jpg)

![74191c62e86dfa3467d9dfb68397c08461542098524642d345e835953b2cabd8.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/images/74191c62e86dfa3467d9dfb68397c08461542098524642d345e835953b2cabd8.jpg)

![9255a78da9e7c5bc08f3e1184cd3f87579540fd78ba12dfa2799cc3078b4d42f.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/images/9255a78da9e7c5bc08f3e1184cd3f87579540fd78ba12dfa2799cc3078b4d42f.jpg)

![93df398628a117156559eb3918844f3cc0582a4059120f315516fc19c376271c.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/images/93df398628a117156559eb3918844f3cc0582a4059120f315516fc19c376271c.jpg)

![955f28e0b4bd4f5076a34c3d453d1a5d1d8f50c9ac71662e04b3efe963137c56.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/images/955f28e0b4bd4f5076a34c3d453d1a5d1d8f50c9ac71662e04b3efe963137c56.jpg)

### Tables

![01799143799607a5e6dc7bc8cdcb8b3c25b2aa8d20e00134d4004c1a2d0ec926.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/tables/01799143799607a5e6dc7bc8cdcb8b3c25b2aa8d20e00134d4004c1a2d0ec926.jpg)

![0714aa86f4ef1955463f354219f0ecd55c7d0c76b99c4cf22f4a14c8cb270a3f.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/tables/0714aa86f4ef1955463f354219f0ecd55c7d0c76b99c4cf22f4a14c8cb270a3f.jpg)

![1134f3d806d4f8e1c34cc7f5091b4dc78783b3630047c5a8a03d73d9fb18a346.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/tables/1134f3d806d4f8e1c34cc7f5091b4dc78783b3630047c5a8a03d73d9fb18a346.jpg)

![37ad4edd84fe18dbf0d374896b9507b86d787a50b9a59fc64620127cf6536147.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/tables/37ad4edd84fe18dbf0d374896b9507b86d787a50b9a59fc64620127cf6536147.jpg)

![4a94bbb87ec5a7bcfbbda36df3e8491dbbc200394defb28425510f4dc9d030c2.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/tables/4a94bbb87ec5a7bcfbbda36df3e8491dbbc200394defb28425510f4dc9d030c2.jpg)

![53c6035d00a101468d218ddb13beec91119af14ea9c2787de4c435f853db7c03.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/tables/53c6035d00a101468d218ddb13beec91119af14ea9c2787de4c435f853db7c03.jpg)

![5c9e58f3bcb2e19ef0ce3247a735b51f23bdb31ccde1bfeaa9162abc1000be44.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/tables/5c9e58f3bcb2e19ef0ce3247a735b51f23bdb31ccde1bfeaa9162abc1000be44.jpg)

![6154dc53e7bc5b4c64cc86e3330dbef2bfe34fb50a1dd5e74592c21df91f5971.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/tables/6154dc53e7bc5b4c64cc86e3330dbef2bfe34fb50a1dd5e74592c21df91f5971.jpg)

![68e895686311f02368833d3e276bd3548b0d8680f380b19a27e9136861f58207.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/tables/68e895686311f02368833d3e276bd3548b0d8680f380b19a27e9136861f58207.jpg)

![6fbd5146df304c6a881dbb3a7957ea9384d73e34f8656a8e228768c6171195c0.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/tables/6fbd5146df304c6a881dbb3a7957ea9384d73e34f8656a8e228768c6171195c0.jpg)

![8dd71f00a8be61b9b71d1b32a9caed38eb831faee79bfe96a144fbe6e17e7c2a.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/tables/8dd71f00a8be61b9b71d1b32a9caed38eb831faee79bfe96a144fbe6e17e7c2a.jpg)

![9bb98d60a22dd49e0978d8ac0612be64e38391171edb8a8e14a9dcd40ec43e7d.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/tables/9bb98d60a22dd49e0978d8ac0612be64e38391171edb8a8e14a9dcd40ec43e7d.jpg)

![e4d9b7c7ad43a2cf317cf5412449f2fe712555a4b1020c57001a7af7d2ff24b0.jpg](../icml_results/6184_Alignment%20Risks%20from%20Capability-Seeking%20RL%20Training/tables/e4d9b7c7ad43a2cf317cf5412449f2fe712555a4b1020c57001a7af7d2ff24b0.jpg)

<a id="paper-6135"></a>

## Preference-Enhanced Reinforcement Learning for Pluralistic Image Inpainting


### Images

![051f0c99e6bb451def4b5bf5659cd7dc7cc879712e6a6c339fe9319be7688771.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/051f0c99e6bb451def4b5bf5659cd7dc7cc879712e6a6c339fe9319be7688771.jpg)

![105a28dcab044c53c4ae839b002028e700cc43d7914cf825531a1f100ac67a10.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/105a28dcab044c53c4ae839b002028e700cc43d7914cf825531a1f100ac67a10.jpg)

![31f53bdb70d73356f698b199e76a4d0b73f508c5016d991c032f5359e526973c.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/31f53bdb70d73356f698b199e76a4d0b73f508c5016d991c032f5359e526973c.jpg)

![47c52784c5f4f2cce7499c9714d265e158f408708efb7861731e933c22be1eeb.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/47c52784c5f4f2cce7499c9714d265e158f408708efb7861731e933c22be1eeb.jpg)

![59c7c819066c0678bf0690f5f24cabb4d5b808298d0ce16b02582c2f680c9ee4.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/59c7c819066c0678bf0690f5f24cabb4d5b808298d0ce16b02582c2f680c9ee4.jpg)

![5aab561736a3c83e793e98d4d30f53e98d4fcb33d102e39a265e9ee7839dfaa8.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/5aab561736a3c83e793e98d4d30f53e98d4fcb33d102e39a265e9ee7839dfaa8.jpg)

![5ae0524ba1bc8770178f264729ee8baaba3486a5e6e1cd5bc6b5f61d39c1e5d7.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/5ae0524ba1bc8770178f264729ee8baaba3486a5e6e1cd5bc6b5f61d39c1e5d7.jpg)

![689b658811ce1ca2b93dfe5a4f178851948a16ed24984a12848fec035c046c65.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/689b658811ce1ca2b93dfe5a4f178851948a16ed24984a12848fec035c046c65.jpg)

![6cd2a29179ddf4cbeee8af9f2b98ba7ce95f1f3a850f100c4201e3ab266efe59.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/6cd2a29179ddf4cbeee8af9f2b98ba7ce95f1f3a850f100c4201e3ab266efe59.jpg)

![6db0f9b2cb8158128bc3690e33aab7e2aa47a1488f631d44e64948f71422624e.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/6db0f9b2cb8158128bc3690e33aab7e2aa47a1488f631d44e64948f71422624e.jpg)

![702776547f5c91138f1133cabd0c7f1005c2feb05058a969fa10e0b9d5bc32d4.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/702776547f5c91138f1133cabd0c7f1005c2feb05058a969fa10e0b9d5bc32d4.jpg)

![742f8192586600ff3a7d579a32cd1c5df77092d16ea91e51ca247b05d7b5f852.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/742f8192586600ff3a7d579a32cd1c5df77092d16ea91e51ca247b05d7b5f852.jpg)

![7aa96b30c44e2c1eb260c4a9a56d3dbe7d754f46fb9efae0e4644cb407fdb2d1.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/7aa96b30c44e2c1eb260c4a9a56d3dbe7d754f46fb9efae0e4644cb407fdb2d1.jpg)

![8c3b7b2d8a805eb4e750038f3e7eba36673cfb69e3ee95f6000d16b48d1d7b62.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/8c3b7b2d8a805eb4e750038f3e7eba36673cfb69e3ee95f6000d16b48d1d7b62.jpg)

![952a24b0e1c2f4524cc30a1220effafffbc57e8513816e7109ae4899f0c3e519.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/952a24b0e1c2f4524cc30a1220effafffbc57e8513816e7109ae4899f0c3e519.jpg)

![b08d677393d5efc33cddb1ff0cb6918223f86603b75893ffb83f85720c638320.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/b08d677393d5efc33cddb1ff0cb6918223f86603b75893ffb83f85720c638320.jpg)

![b0b2ad480b3ec6f9f0f2ad477c0c89d131c1c83999d59236b345c424ab33f180.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/b0b2ad480b3ec6f9f0f2ad477c0c89d131c1c83999d59236b345c424ab33f180.jpg)

![c3055b11438355f36485fedb6b2774defb577754dbcd4c2956e6aa23e0997d2a.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/c3055b11438355f36485fedb6b2774defb577754dbcd4c2956e6aa23e0997d2a.jpg)

![cdaf43b45118e11eb57bcc2c6e7796c6d0ecfb5c758cbe801766029dfc583924.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/cdaf43b45118e11eb57bcc2c6e7796c6d0ecfb5c758cbe801766029dfc583924.jpg)

![cdc99ec5d1c52d4bef71588811dadbe3b3563a562bea1541ffdaf6cbf6b2dabf.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/cdc99ec5d1c52d4bef71588811dadbe3b3563a562bea1541ffdaf6cbf6b2dabf.jpg)

![e68dbef5ee033696c459becbfefff2a5fd21283d5be88689bb17cec0ae778d0a.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/e68dbef5ee033696c459becbfefff2a5fd21283d5be88689bb17cec0ae778d0a.jpg)

![fb9d1d11780b5295a2f76104307a3bab867e5efb374d38636174d1a3266165bb.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/fb9d1d11780b5295a2f76104307a3bab867e5efb374d38636174d1a3266165bb.jpg)

![fc460a8c9d72f332b0e29413e7f660debf0ae2b4d48d1dff091c3bf468934872.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/fc460a8c9d72f332b0e29413e7f660debf0ae2b4d48d1dff091c3bf468934872.jpg)

![fe32056ff01cf534f4829aa73193c33976d02d818c1ae4a9bdc0713a2ece5e65.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/images/fe32056ff01cf534f4829aa73193c33976d02d818c1ae4a9bdc0713a2ece5e65.jpg)

### Tables

![09c77b815b60f163f99b23463177b5b5caa4b703633f2c35592e7c52f62bea39.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/tables/09c77b815b60f163f99b23463177b5b5caa4b703633f2c35592e7c52f62bea39.jpg)

![3647dd79329e3090784705942246f0f18593abd50acaa1498e6143216bd42bad.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/tables/3647dd79329e3090784705942246f0f18593abd50acaa1498e6143216bd42bad.jpg)

![77ee2cb176755e444637d13928e6bd93e61bc83909435b828f398654041bbfab.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/tables/77ee2cb176755e444637d13928e6bd93e61bc83909435b828f398654041bbfab.jpg)

![c909a029c12317004bb1dd9ac819cd608bdf798260409faebc5c1626db4a9c72.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/tables/c909a029c12317004bb1dd9ac819cd608bdf798260409faebc5c1626db4a9c72.jpg)

![f913c86d7416f3acddb03e2e43aa5af7c6b85904302412bde4fe38214068d6d1.jpg](../icml_results/6185_Preference-Enhanced%20Reinforcement%20Learning%20for%20Pluralistic%20Image%20Inpainting/tables/f913c86d7416f3acddb03e2e43aa5af7c6b85904302412bde4fe38214068d6d1.jpg)

<a id="paper-6136"></a>

## SpecPL: Disentangling Spectral Granularity for Prompt Learning


### Images

![016a600027a30c3d18b7c380b5593aeb067307dfd47f63e72385557a7c262086.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/images/016a600027a30c3d18b7c380b5593aeb067307dfd47f63e72385557a7c262086.jpg)

![193bfe4d29c94d85d528af1a1e0884aedb9c2e073ba896d9aa6d6e7f81e20e28.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/images/193bfe4d29c94d85d528af1a1e0884aedb9c2e073ba896d9aa6d6e7f81e20e28.jpg)

![321b22f1f92eaffc5d4bb58b090e1150781965ae669d7e7bc98c14de3f625464.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/images/321b22f1f92eaffc5d4bb58b090e1150781965ae669d7e7bc98c14de3f625464.jpg)

![3dc9b9acb0e683df5bdab1b160430dd3864155797f0e43197dbb353bff5cd15f.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/images/3dc9b9acb0e683df5bdab1b160430dd3864155797f0e43197dbb353bff5cd15f.jpg)

![4eb9389b7f991763847fa176c6711be368dcb71231cf17e1fab7026accf98822.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/images/4eb9389b7f991763847fa176c6711be368dcb71231cf17e1fab7026accf98822.jpg)

![842932504a5b9f8f82ebdc6ab9d8b8ef272297ee5a7c2b95a0b0a7df67ede899.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/images/842932504a5b9f8f82ebdc6ab9d8b8ef272297ee5a7c2b95a0b0a7df67ede899.jpg)

![8f6b26f85642012246dec7dcb0553c509b3b6b9a95f9219491629a4c1f24c65c.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/images/8f6b26f85642012246dec7dcb0553c509b3b6b9a95f9219491629a4c1f24c65c.jpg)

![a68d15a89be3bcb7a47bcf7d8a7f0906704cc83d2bf99f50d4863d9cdd76de36.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/images/a68d15a89be3bcb7a47bcf7d8a7f0906704cc83d2bf99f50d4863d9cdd76de36.jpg)

![a97236565ca282362a6879ac0062f89c07affe66a4815941c398043ab7fbe148.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/images/a97236565ca282362a6879ac0062f89c07affe66a4815941c398043ab7fbe148.jpg)

![b02d029ff62dbfd401ab87b183ef0f1a7cc09535ca0910729dee44f91777ee02.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/images/b02d029ff62dbfd401ab87b183ef0f1a7cc09535ca0910729dee44f91777ee02.jpg)

![e65894cc08dde939d4e711cde5b5b281329d8c9cb7c78329b5050e59db542935.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/images/e65894cc08dde939d4e711cde5b5b281329d8c9cb7c78329b5050e59db542935.jpg)

### Tables

![097a44c6f748670d9a188d02cd6c98d878cd138d68219da86e9c45f28ab5c17b.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/tables/097a44c6f748670d9a188d02cd6c98d878cd138d68219da86e9c45f28ab5c17b.jpg)

![1801d537a39d835b423eb9fb959ab4f8ca08dd280b5978dd395174c979686b5f.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/tables/1801d537a39d835b423eb9fb959ab4f8ca08dd280b5978dd395174c979686b5f.jpg)

![1d4306421a2d0f30a05184bc0a5d3dc95aa62c8c829e14118240dc6d2459a503.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/tables/1d4306421a2d0f30a05184bc0a5d3dc95aa62c8c829e14118240dc6d2459a503.jpg)

![21f64f2620fee49a50244bca9876d4010e9d26a1d1bff9e04176ab4b754c1fe1.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/tables/21f64f2620fee49a50244bca9876d4010e9d26a1d1bff9e04176ab4b754c1fe1.jpg)

![397990393ef20dd3fdd5de23ef1bc32791313abe262d11d35344289ca699e6ea.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/tables/397990393ef20dd3fdd5de23ef1bc32791313abe262d11d35344289ca699e6ea.jpg)

![694ff43f51c7bc792b8e5381a5b331fa18510a703ecfbde181dd95744e4d8e28.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/tables/694ff43f51c7bc792b8e5381a5b331fa18510a703ecfbde181dd95744e4d8e28.jpg)

![77f2e7b52d704b4dc3b26073aac0663807d161e0fd33e73756d261687e52ac13.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/tables/77f2e7b52d704b4dc3b26073aac0663807d161e0fd33e73756d261687e52ac13.jpg)

![85f59409cb6f4e4de02125fd480f75829d7850d32f2110fb0c0f31226eee8168.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/tables/85f59409cb6f4e4de02125fd480f75829d7850d32f2110fb0c0f31226eee8168.jpg)

![998e29d5f04e81cfaf618c2a00deb36cf92f847bf106ba6452164edeb214f0fd.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/tables/998e29d5f04e81cfaf618c2a00deb36cf92f847bf106ba6452164edeb214f0fd.jpg)

![a954aea26629806028d3a53dc613facaa6875b4ddbf90df395f3d865f5329f37.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/tables/a954aea26629806028d3a53dc613facaa6875b4ddbf90df395f3d865f5329f37.jpg)

![daa64cb789b9be8190ebea55dc5d25fd421c5bae61f87410b9d8fc32aa04549f.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/tables/daa64cb789b9be8190ebea55dc5d25fd421c5bae61f87410b9d8fc32aa04549f.jpg)

![db6037450db0670f304bf49243215856a513bf050e5ad0d5a99705d40d514ee3.jpg](../icml_results/6186_SpecPL_%20Disentangling%20Spectral%20Granularity%20for%20Prompt%20Learning/tables/db6037450db0670f304bf49243215856a513bf050e5ad0d5a99705d40d514ee3.jpg)

<a id="paper-6137"></a>

## Towards Completeness in Causal Discovery from Soft Interventions with Known Targets


### Images

![3deb9a9db42e1bac751600c8120a70e08bd64dcb970b48b07072b57d0ef7ef92.jpg](../icml_results/6187_Towards%20Completeness%20in%20Causal%20Discovery%20from%20Soft%20Interventions%20with%20Known%20Targets/images/3deb9a9db42e1bac751600c8120a70e08bd64dcb970b48b07072b57d0ef7ef92.jpg)

![b4070b51cf0ce6f5b541cbd74a2b0fbe460d6b527c03b891e92e78cb962d0903.jpg](../icml_results/6187_Towards%20Completeness%20in%20Causal%20Discovery%20from%20Soft%20Interventions%20with%20Known%20Targets/images/b4070b51cf0ce6f5b541cbd74a2b0fbe460d6b527c03b891e92e78cb962d0903.jpg)

![c2d482c157350a308c074560a1a0cdd8d1a203922e7ffca587e7fd83f1a95749.jpg](../icml_results/6187_Towards%20Completeness%20in%20Causal%20Discovery%20from%20Soft%20Interventions%20with%20Known%20Targets/images/c2d482c157350a308c074560a1a0cdd8d1a203922e7ffca587e7fd83f1a95749.jpg)

### Tables

![163af3517812600b88d6fd9d472cda1f106f8a84d5398b8912fce9ccfccd2bec.jpg](../icml_results/6187_Towards%20Completeness%20in%20Causal%20Discovery%20from%20Soft%20Interventions%20with%20Known%20Targets/tables/163af3517812600b88d6fd9d472cda1f106f8a84d5398b8912fce9ccfccd2bec.jpg)

![31f3176219237856d2f8e6b804cd3570e5f89737fef688e8e26f1487aa5f9ea7.jpg](../icml_results/6187_Towards%20Completeness%20in%20Causal%20Discovery%20from%20Soft%20Interventions%20with%20Known%20Targets/tables/31f3176219237856d2f8e6b804cd3570e5f89737fef688e8e26f1487aa5f9ea7.jpg)

![40a1f17865a2f5aa24b8927ab4901f144f16f7c2e7215d54007540bd6255a9c7.jpg](../icml_results/6187_Towards%20Completeness%20in%20Causal%20Discovery%20from%20Soft%20Interventions%20with%20Known%20Targets/tables/40a1f17865a2f5aa24b8927ab4901f144f16f7c2e7215d54007540bd6255a9c7.jpg)

![56000d299e452047e30c0430830d4057893279ca311399733d68f8ef792a1bb7.jpg](../icml_results/6187_Towards%20Completeness%20in%20Causal%20Discovery%20from%20Soft%20Interventions%20with%20Known%20Targets/tables/56000d299e452047e30c0430830d4057893279ca311399733d68f8ef792a1bb7.jpg)

![72fa17e751855a5aa0736a504388d8a7fa888d028d1495cec8264cd3fad014b9.jpg](../icml_results/6187_Towards%20Completeness%20in%20Causal%20Discovery%20from%20Soft%20Interventions%20with%20Known%20Targets/tables/72fa17e751855a5aa0736a504388d8a7fa888d028d1495cec8264cd3fad014b9.jpg)

![98894a0030e3307e848ca38a5186bade0573eb7536253fc5c43ca5333733b2da.jpg](../icml_results/6187_Towards%20Completeness%20in%20Causal%20Discovery%20from%20Soft%20Interventions%20with%20Known%20Targets/tables/98894a0030e3307e848ca38a5186bade0573eb7536253fc5c43ca5333733b2da.jpg)

![b66ddfaad6ac09ca1eb1344e687bddcd019cab276b270bbde6d315e5734a876d.jpg](../icml_results/6187_Towards%20Completeness%20in%20Causal%20Discovery%20from%20Soft%20Interventions%20with%20Known%20Targets/tables/b66ddfaad6ac09ca1eb1344e687bddcd019cab276b270bbde6d315e5734a876d.jpg)

![e51def67689e38b086dc4cd8d03c69f0218075d301a50b1b140440ba5c7b3621.jpg](../icml_results/6187_Towards%20Completeness%20in%20Causal%20Discovery%20from%20Soft%20Interventions%20with%20Known%20Targets/tables/e51def67689e38b086dc4cd8d03c69f0218075d301a50b1b140440ba5c7b3621.jpg)

![f212ce050bee1a758756352ee47438f2d754b3750897cef699433f3d7c7e052d.jpg](../icml_results/6187_Towards%20Completeness%20in%20Causal%20Discovery%20from%20Soft%20Interventions%20with%20Known%20Targets/tables/f212ce050bee1a758756352ee47438f2d754b3750897cef699433f3d7c7e052d.jpg)

<a id="paper-6138"></a>

## AVGen-Bench: A Task-Driven Benchmark for Multi-Granular Evaluation of Text-to-Audio-Video Generation

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
