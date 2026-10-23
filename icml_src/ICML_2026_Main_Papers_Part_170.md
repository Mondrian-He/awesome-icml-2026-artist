# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 170 of 197

## 目录 (Table of Contents)

1. [RaBiT: Residual Aware Binarization Training for Accurate and Efficient LLMs](#paper-5578)
2. [LLMInertia: Adaptive Counter-Inertial Reasoning to Improve Evidence Faithfulness in Large Language Models](#paper-5579)
3. [Pianist Transformer: Towards Expressive Piano Performance Rendering via Scalable Self-Supervised Pre-Training](#paper-5580)
4. [DDGA: Dirichlet Distributional Gradient Aggregation for Transferable Vision-Language Adversarial Attacks](#paper-5581)
5. [Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards](#paper-5582)
6. [SWE-ABS: Adversarial Benchmark Strengthening Exposes Inflated Success Rates on Test-based Benchmark](#paper-5583)
7. [Dismantling Pathological Shortcuts: A Causal Framework for Faithful LVLM Decoding](#paper-5584)
8. [Branch Scaling Manifests as Implicit Architectural Regularization for Improving Generalization in Overparameterized ResNets](#paper-5585)
9. [CacheEdit: Efficient Multi-round Image Editing via Adaptive Token-wise Reuse.](#paper-5586)
10. [ClimateAR: Multi-Scale Autoregressive Generative Modeling for Climate Forecasting](#paper-5587)
11. [NeurIPS: Neuro-anatomical Inductive Priors for Sphere-based Brain Decoding](#paper-5588)
12. [On the Limits of Test-Time Compute: Sequential Reward Filtering for Better Inference](#paper-5589)
13. [PathwayLLM: Explainable Clinical Trajectory Modeling with Structured Pathways for Sepsis Prediction](#paper-5590)
14. [Deep Pre-Alignment for VLMs](#paper-5591)
15. [SafeSeek: Universal Attribution of Safety Circuits in Language Models](#paper-5592)
16. [Reliable Confidence Alignment for Generalized Category Discovery](#paper-5593)
17. [TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models](#paper-5594)
18. [Robust Inter-Series Dependency Modeling for Time Series Forecasting via Information-Theoretic Alignment](#paper-5595)
19. [CARD: Coarse-to-fine Autoregressive Modeling with Radix-based Decomposition for Transferable Free Energy Estimation](#paper-5596)
20. [Geometry-Preserving Unsupervised Alignment for Heterogeneous Foundation Models](#paper-5597)
21. [GFedCL: Graph-Based Federated Continual Learning with Spatial and Temporal Awareness](#paper-5598)
22. [Scaling Laws of Global Weather Models](#paper-5599)
23. [Mirror Descent Under Generalized Smoothness](#paper-5600)
24. [FlowBot: Inducing LLM Workflows with Bilevel Optimization and Textual Gradients](#paper-5601)
25. [Multi-Adapter Representation Interventions via Energy Calibration](#paper-5602)
26. [From Prompts to Tokens: Internalizing Causal Supervision in Vision-Language Model for Multi-Image Causal Reasoning](#paper-5603)
27. [CORRECT: COndensed eRror RECognition via knowledge Transfer in multi-agent systems](#paper-5604)
28. [Efficient Prediction of SO(3)-Equivariant Hamiltonian Matrices via SO(2) Local Frames](#paper-5605)
29. [Autoregressive Image Generation with Masked Bit Modeling](#paper-5606)
30. [PATCHCODE: Discrete Latent Predictive Learning for EEG Foundation Model](#paper-5607)
31. [Bridging Structure and Semantics: Uncertainty-Modulated Dual-Path Diffusion for Robust Text-Attributed Graph Learning](#paper-5608)
32. [Evaluating AI Grading on Real-World Handwritten College Mathematics: A Large-Scale Study Toward a Benchmark](#paper-5609)
33. [Fine-to-Coarse Fairness-Informed Multi-View Clustering](#paper-5610)

---


<a id="paper-5578"></a>

## RaBiT: Residual Aware Binarization Training for Accurate and Efficient LLMs

### Images

![1124ad98018c8c892e1de9f103d9459954620e304c67cbc66f952c71b4acf95c.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/images/1124ad98018c8c892e1de9f103d9459954620e304c67cbc66f952c71b4acf95c.jpg)

![3ece8d338c9264c33b28e1bf0b532f8fcbd6f270eb82e2ba0ecf12847032e03c.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/images/3ece8d338c9264c33b28e1bf0b532f8fcbd6f270eb82e2ba0ecf12847032e03c.jpg)

![4175a8f611244f57b5d9aee284ffa7d5aac43a9b95c0529ba7aa593ea2e530d5.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/images/4175a8f611244f57b5d9aee284ffa7d5aac43a9b95c0529ba7aa593ea2e530d5.jpg)

![441142f69e4a372ea130870bd13277fdd6a8512a1bed2f618048b6a14c5734cb.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/images/441142f69e4a372ea130870bd13277fdd6a8512a1bed2f618048b6a14c5734cb.jpg)

![72d0884f3dbbf3f42f36fb1e84a295a40586dfd187e5e3e571fdb158c8a4b99f.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/images/72d0884f3dbbf3f42f36fb1e84a295a40586dfd187e5e3e571fdb158c8a4b99f.jpg)

![b0a6f530a9fb804d50a0484cfe9ffead4c384a17689e829a7ae3cea20788fae4.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/images/b0a6f530a9fb804d50a0484cfe9ffead4c384a17689e829a7ae3cea20788fae4.jpg)

![c5f955df9c6ea6b5dcadbba4f0304c28387216d1c78b7bed4a817bdefae74358.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/images/c5f955df9c6ea6b5dcadbba4f0304c28387216d1c78b7bed4a817bdefae74358.jpg)

![d382f6eae90b57cb91abd3b1f85630c9c1487068dd093f433d6610fbfd20289c.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/images/d382f6eae90b57cb91abd3b1f85630c9c1487068dd093f433d6610fbfd20289c.jpg)

### Tables

![2ae2a734acb546c4e6d01e460790aaef8888789ecb08c2e42d20c5a624843907.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/tables/2ae2a734acb546c4e6d01e460790aaef8888789ecb08c2e42d20c5a624843907.jpg)

![426e52ab7785ffedc0f68e49da7898aa4ac2b5fdc9886523fad003c1857c3e87.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/tables/426e52ab7785ffedc0f68e49da7898aa4ac2b5fdc9886523fad003c1857c3e87.jpg)

![58cd17da61fc5688ee324327dd040b1fe98d10b709ebd98a8147aee9aaebb45b.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/tables/58cd17da61fc5688ee324327dd040b1fe98d10b709ebd98a8147aee9aaebb45b.jpg)

![9453e02995be43ae2b5f2e685dc11ab2e8dae101209b17fb1429294868f4ef73.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/tables/9453e02995be43ae2b5f2e685dc11ab2e8dae101209b17fb1429294868f4ef73.jpg)

![a384704843ebe74f98ce074596e823563b4251d1d9d574104ba8560b9fb13c8b.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/tables/a384704843ebe74f98ce074596e823563b4251d1d9d574104ba8560b9fb13c8b.jpg)

![a882c2eabde6210355e263e6c973862c90057e169d951ef3dc5c6fa30ae5f07f.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/tables/a882c2eabde6210355e263e6c973862c90057e169d951ef3dc5c6fa30ae5f07f.jpg)

![c4728d4af3a9bf7a634039ce957869d7596141d89bdab9126c9ded6fa851ce1b.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/tables/c4728d4af3a9bf7a634039ce957869d7596141d89bdab9126c9ded6fa851ce1b.jpg)

![d6ddb70ddab4417c951e634933f6074408a62035b9e5a92ec84c2d5fcd677fcc.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/tables/d6ddb70ddab4417c951e634933f6074408a62035b9e5a92ec84c2d5fcd677fcc.jpg)

![e1437e1d63a7808335951548af1cd809548b6e37c9252c72eaaeb221d90e1cb4.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/tables/e1437e1d63a7808335951548af1cd809548b6e37c9252c72eaaeb221d90e1cb4.jpg)

![e1bffd401f956c0c53fe12b84b7792362325a7884fc867a10c899490573928dc.jpg](../icml_results/5626_Spherical%20Steering_%20Geometry-Aware%20Activation%20Rotation%20for%20Language%20Models/tables/e1bffd401f956c0c53fe12b84b7792362325a7884fc867a10c899490573928dc.jpg)

## RaBiT: Residual Aware Binarization Training for Accurate and Efficient LLMs


### Images

![0c158058b1479b0ac940587f536ffa0256afa0569f6d3696a84c4460fbe28141.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/images/0c158058b1479b0ac940587f536ffa0256afa0569f6d3696a84c4460fbe28141.jpg)

![0c933bcce5d15f7b38057819973ce70e7537c04ad6e6afda73835342788a2e7f.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/images/0c933bcce5d15f7b38057819973ce70e7537c04ad6e6afda73835342788a2e7f.jpg)

![4f959cf230d14c1531b9321c882898f53e14b52b94792c72cecf0ff1eefd0af8.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/images/4f959cf230d14c1531b9321c882898f53e14b52b94792c72cecf0ff1eefd0af8.jpg)

![7fa3e425c827eeffd1272ec1721e3c6e189dfb0b6c8925c3892b940897d3ac04.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/images/7fa3e425c827eeffd1272ec1721e3c6e189dfb0b6c8925c3892b940897d3ac04.jpg)

![8012f96b6263982a971b9703a65039360e00eef5b021c330339e7201f31f50b3.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/images/8012f96b6263982a971b9703a65039360e00eef5b021c330339e7201f31f50b3.jpg)

![bcd23def293eba846ccc5cc480580e8ef0f7962b388cf955c89988fcfb3f51f7.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/images/bcd23def293eba846ccc5cc480580e8ef0f7962b388cf955c89988fcfb3f51f7.jpg)

### Tables

![259f625be5f86498657434a941d33cd58d89809cbce605c7b15ddaa6485b61a1.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/259f625be5f86498657434a941d33cd58d89809cbce605c7b15ddaa6485b61a1.jpg)

![318616c1b0f32b64e8bf6df6385a16fcdbba89cc1d5efe4fa938cf88a2a1c13b.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/318616c1b0f32b64e8bf6df6385a16fcdbba89cc1d5efe4fa938cf88a2a1c13b.jpg)

![335ce220174be5686c984eac64a2f2ec604e80ff27d2286e7a7fdfa56b89cbd3.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/335ce220174be5686c984eac64a2f2ec604e80ff27d2286e7a7fdfa56b89cbd3.jpg)

![3ed50fab24ad9c53b837217f07a313f68bcc12647153e5a389b0061b07322788.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/3ed50fab24ad9c53b837217f07a313f68bcc12647153e5a389b0061b07322788.jpg)

![4952070d94b47278415dd092c6ed54e1f78f5633c9b5c2c61cede7088836f326.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/4952070d94b47278415dd092c6ed54e1f78f5633c9b5c2c61cede7088836f326.jpg)

![56a43c211bfae248ca55eba4a6097f649119f4e1d287fd3d2f27e1fd82d795e4.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/56a43c211bfae248ca55eba4a6097f649119f4e1d287fd3d2f27e1fd82d795e4.jpg)

![6c8a9008b33c6d6bbe500a0e4ecbd6a38551ec3e82c279f0cea76804c0f10573.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/6c8a9008b33c6d6bbe500a0e4ecbd6a38551ec3e82c279f0cea76804c0f10573.jpg)

![6e1f36cf94e0150a28052dd18be508519d5b2de808c792c284097ec5863d1f0c.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/6e1f36cf94e0150a28052dd18be508519d5b2de808c792c284097ec5863d1f0c.jpg)

![82919325127c807358bf363f70d5ff68325b74616c9dd0551c333d0ae6a49aca.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/82919325127c807358bf363f70d5ff68325b74616c9dd0551c333d0ae6a49aca.jpg)

![93bf1c98f2f492ac90a1e56a1610f7d1d10d1a53b0f6e2c11bb8adab860eae75.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/93bf1c98f2f492ac90a1e56a1610f7d1d10d1a53b0f6e2c11bb8adab860eae75.jpg)

![a4a3f494e722329c221bfc7684f11dfda65142a2a17ab56715e99cffbf4903ad.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/a4a3f494e722329c221bfc7684f11dfda65142a2a17ab56715e99cffbf4903ad.jpg)

![c7714939f013ffc3e897d2dfdf3d288a92e4facd2ac5cb0c69047b730e878e93.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/c7714939f013ffc3e897d2dfdf3d288a92e4facd2ac5cb0c69047b730e878e93.jpg)

![c9a2d26bec3ddc954a243c93b266b623da90b027aa7d8b113b399d2dc3c3ae2c.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/c9a2d26bec3ddc954a243c93b266b623da90b027aa7d8b113b399d2dc3c3ae2c.jpg)

![dcf3f05f0190f4f6d406d19dabf29139c5eb0d8aafd4df1b6e9f075d26558ac0.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/dcf3f05f0190f4f6d406d19dabf29139c5eb0d8aafd4df1b6e9f075d26558ac0.jpg)

![de11bce9f1917c69aec1afa976cc55d85ac4ae3b65438c8ff3fda40151672959.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/de11bce9f1917c69aec1afa976cc55d85ac4ae3b65438c8ff3fda40151672959.jpg)

![e0e1108cfffe9d6ae7730121c670090c1ec2d28492e13a4ac82a7aa58baca4c7.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/e0e1108cfffe9d6ae7730121c670090c1ec2d28492e13a4ac82a7aa58baca4c7.jpg)

![e13c2a551edac9afc188d6b038b559b2e727e93595942617dfb74a1675e7f858.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/e13c2a551edac9afc188d6b038b559b2e727e93595942617dfb74a1675e7f858.jpg)

![ee457b189885a12459cf0818830315afc3c60dc5e5bb2972c2b8efa9d4dcdfa8.jpg](../icml_results/5627_RaBiT_%20Residual%20Aware%20Binarization%20Training%20for%20Accurate%20and%20Efficient%20LLMs/tables/ee457b189885a12459cf0818830315afc3c60dc5e5bb2972c2b8efa9d4dcdfa8.jpg)

<a id="paper-5579"></a>

## LLMInertia: Adaptive Counter-Inertial Reasoning to Improve Evidence Faithfulness in Large Language Models


### Images

![03acf0f36bdf9ebc82e4b76bb2d8786782333d684d80c8d2828fecb005d6b60e.jpg](../icml_results/5628_LLMInertia_%20Adaptive%20Counter-Inertial%20Reasoning%20to%20Improve%20Evidence%20Faithfulness%20in%20Large%20Language%20M/images/03acf0f36bdf9ebc82e4b76bb2d8786782333d684d80c8d2828fecb005d6b60e.jpg)

![1018e43412e9d70adc7998398bf69148ff8f1aa7715b60c8305f4723e3cc0579.jpg](../icml_results/5628_LLMInertia_%20Adaptive%20Counter-Inertial%20Reasoning%20to%20Improve%20Evidence%20Faithfulness%20in%20Large%20Language%20M/images/1018e43412e9d70adc7998398bf69148ff8f1aa7715b60c8305f4723e3cc0579.jpg)

![9affc2c9330c12a38dc942a7166d17a17733e666d1ce204a9baab2557838f971.jpg](../icml_results/5628_LLMInertia_%20Adaptive%20Counter-Inertial%20Reasoning%20to%20Improve%20Evidence%20Faithfulness%20in%20Large%20Language%20M/images/9affc2c9330c12a38dc942a7166d17a17733e666d1ce204a9baab2557838f971.jpg)

![ad128b74538a19a38c736447a975fc8d58032afd0481c3134e6416ade755c2e2.jpg](../icml_results/5628_LLMInertia_%20Adaptive%20Counter-Inertial%20Reasoning%20to%20Improve%20Evidence%20Faithfulness%20in%20Large%20Language%20M/images/ad128b74538a19a38c736447a975fc8d58032afd0481c3134e6416ade755c2e2.jpg)

![d352556437c3ec31fe1940527b7881475d2edbed6b913c2595aa4c6f1a36956c.jpg](../icml_results/5628_LLMInertia_%20Adaptive%20Counter-Inertial%20Reasoning%20to%20Improve%20Evidence%20Faithfulness%20in%20Large%20Language%20M/images/d352556437c3ec31fe1940527b7881475d2edbed6b913c2595aa4c6f1a36956c.jpg)

![e0e5a9198902d9223cd3eaa433c02c01537415f978ca346e4f780df5313af8e2.jpg](../icml_results/5628_LLMInertia_%20Adaptive%20Counter-Inertial%20Reasoning%20to%20Improve%20Evidence%20Faithfulness%20in%20Large%20Language%20M/images/e0e5a9198902d9223cd3eaa433c02c01537415f978ca346e4f780df5313af8e2.jpg)

![fce62d3d5f5f63cd6af5bbf8b09d27eabb2ce7d3811d286f19ead5e9950d099a.jpg](../icml_results/5628_LLMInertia_%20Adaptive%20Counter-Inertial%20Reasoning%20to%20Improve%20Evidence%20Faithfulness%20in%20Large%20Language%20M/images/fce62d3d5f5f63cd6af5bbf8b09d27eabb2ce7d3811d286f19ead5e9950d099a.jpg)

### Tables

![06f7d204f04196cb3c0bc029f5a688f542532236b44aa4a5b02ae4769f5c04cc.jpg](../icml_results/5628_LLMInertia_%20Adaptive%20Counter-Inertial%20Reasoning%20to%20Improve%20Evidence%20Faithfulness%20in%20Large%20Language%20M/tables/06f7d204f04196cb3c0bc029f5a688f542532236b44aa4a5b02ae4769f5c04cc.jpg)

![3ca487c5dcf27b838bde9095d1de07ce8910945c0d222bda3c244d196e03e5cf.jpg](../icml_results/5628_LLMInertia_%20Adaptive%20Counter-Inertial%20Reasoning%20to%20Improve%20Evidence%20Faithfulness%20in%20Large%20Language%20M/tables/3ca487c5dcf27b838bde9095d1de07ce8910945c0d222bda3c244d196e03e5cf.jpg)

![530fae3593f6a73c68a9c291bb0fba799a3e2a6ddec13fbffa450c7b33e31b6c.jpg](../icml_results/5628_LLMInertia_%20Adaptive%20Counter-Inertial%20Reasoning%20to%20Improve%20Evidence%20Faithfulness%20in%20Large%20Language%20M/tables/530fae3593f6a73c68a9c291bb0fba799a3e2a6ddec13fbffa450c7b33e31b6c.jpg)

![6969f5dac1134bdea61c391ab726488eca3cd33698b0fac89553db471424bc48.jpg](../icml_results/5628_LLMInertia_%20Adaptive%20Counter-Inertial%20Reasoning%20to%20Improve%20Evidence%20Faithfulness%20in%20Large%20Language%20M/tables/6969f5dac1134bdea61c391ab726488eca3cd33698b0fac89553db471424bc48.jpg)

![7e76ccec521fb7872e488dc74068527e62c4f3f6a66b70a8d4fa28f1cc2e731b.jpg](../icml_results/5628_LLMInertia_%20Adaptive%20Counter-Inertial%20Reasoning%20to%20Improve%20Evidence%20Faithfulness%20in%20Large%20Language%20M/tables/7e76ccec521fb7872e488dc74068527e62c4f3f6a66b70a8d4fa28f1cc2e731b.jpg)

![81de766dcbdbe334e2da2d1d0407e7232b6b86bacc1f9383c58bd1b6fc87daed.jpg](../icml_results/5628_LLMInertia_%20Adaptive%20Counter-Inertial%20Reasoning%20to%20Improve%20Evidence%20Faithfulness%20in%20Large%20Language%20M/tables/81de766dcbdbe334e2da2d1d0407e7232b6b86bacc1f9383c58bd1b6fc87daed.jpg)

![a922fbf7f00bef5247e6740c341c7c2aac1933e03e8f671e31f1613e1dbddcf7.jpg](../icml_results/5628_LLMInertia_%20Adaptive%20Counter-Inertial%20Reasoning%20to%20Improve%20Evidence%20Faithfulness%20in%20Large%20Language%20M/tables/a922fbf7f00bef5247e6740c341c7c2aac1933e03e8f671e31f1613e1dbddcf7.jpg)

![ae5e816547df3985d654173614e285eab19b2b5e7b811c24fdca32109c889509.jpg](../icml_results/5628_LLMInertia_%20Adaptive%20Counter-Inertial%20Reasoning%20to%20Improve%20Evidence%20Faithfulness%20in%20Large%20Language%20M/tables/ae5e816547df3985d654173614e285eab19b2b5e7b811c24fdca32109c889509.jpg)

![c4f860356bb3bcd228c3bf689642b9f2a8d02f1e817591588c6385ad77ebe60d.jpg](../icml_results/5628_LLMInertia_%20Adaptive%20Counter-Inertial%20Reasoning%20to%20Improve%20Evidence%20Faithfulness%20in%20Large%20Language%20M/tables/c4f860356bb3bcd228c3bf689642b9f2a8d02f1e817591588c6385ad77ebe60d.jpg)

![ce38de10df05bcbf6ebae08cb96f47f06f181f3a4a747f3575df5606aaca9986.jpg](../icml_results/5628_LLMInertia_%20Adaptive%20Counter-Inertial%20Reasoning%20to%20Improve%20Evidence%20Faithfulness%20in%20Large%20Language%20M/tables/ce38de10df05bcbf6ebae08cb96f47f06f181f3a4a747f3575df5606aaca9986.jpg)

<a id="paper-5580"></a>

## Pianist Transformer: Towards Expressive Piano Performance Rendering via Scalable Self-Supervised Pre-Training


### Images

![36e53ac1538f5fa8fa50882cf475c3800607c06d827922836095a9a5cf7ed083.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/images/36e53ac1538f5fa8fa50882cf475c3800607c06d827922836095a9a5cf7ed083.jpg)

![467c4d38db3a796fc097618c32a72ba8cd616e3c9cbb7d7bcee46fc63bb58ba9.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/images/467c4d38db3a796fc097618c32a72ba8cd616e3c9cbb7d7bcee46fc63bb58ba9.jpg)

![58dba9c93f8a879cd9953531332ae5caf32b8e8c6e8f658e884f9d024a490a4c.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/images/58dba9c93f8a879cd9953531332ae5caf32b8e8c6e8f658e884f9d024a490a4c.jpg)

![6cffa9e1e169ecab3737065398810623fe6b94a45fcc8613093c23e3cd8c1018.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/images/6cffa9e1e169ecab3737065398810623fe6b94a45fcc8613093c23e3cd8c1018.jpg)

![79a0bc6ccd6ca620c8967cc71811ed7547d2514050ef5d741279c54f6f31f09f.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/images/79a0bc6ccd6ca620c8967cc71811ed7547d2514050ef5d741279c54f6f31f09f.jpg)

![81b1298675fe153201c2726a141bc66acb990973324b2420aa6932b18548b6a1.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/images/81b1298675fe153201c2726a141bc66acb990973324b2420aa6932b18548b6a1.jpg)

![8e846a6da6327d1ce0f80fb4387622a7da543aa4f156d263782d3ef90c90a47f.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/images/8e846a6da6327d1ce0f80fb4387622a7da543aa4f156d263782d3ef90c90a47f.jpg)

![9aec638cc6034296507ac19febc8a4e373e74fa148ef81a7258a64c1a31cd284.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/images/9aec638cc6034296507ac19febc8a4e373e74fa148ef81a7258a64c1a31cd284.jpg)

![c33975812b87b029e4736091c1d3d4b5b8344e971c68e1775cfdbef671638cb0.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/images/c33975812b87b029e4736091c1d3d4b5b8344e971c68e1775cfdbef671638cb0.jpg)

![fbcc979160ef1de87af9fd09cebfe9e32a686dcbdcca343eb4a38a7f1859b41c.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/images/fbcc979160ef1de87af9fd09cebfe9e32a686dcbdcca343eb4a38a7f1859b41c.jpg)

![ff5f79e705eecde29f0a623bddb49b93375abb09ea8bb8f3866d0ca7440f8313.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/images/ff5f79e705eecde29f0a623bddb49b93375abb09ea8bb8f3866d0ca7440f8313.jpg)

### Tables

![01da04d9e45b745b5be8e70130ac6ff86fb560ba1aa9a8670e04f2181b83709a.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/tables/01da04d9e45b745b5be8e70130ac6ff86fb560ba1aa9a8670e04f2181b83709a.jpg)

![2da144298b3e7fd07639f540f1b2900f44db9396f0c931c8f6a4f8fc25b3a6ce.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/tables/2da144298b3e7fd07639f540f1b2900f44db9396f0c931c8f6a4f8fc25b3a6ce.jpg)

![3b01718e137baebbfef7062df62635cf4c58d027cb4b66cf83efcb18fe7df2b2.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/tables/3b01718e137baebbfef7062df62635cf4c58d027cb4b66cf83efcb18fe7df2b2.jpg)

![69a26f9a96b17f9f593a0898fcc5bfbd2562a12cde405b7f0db77b1ac267420a.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/tables/69a26f9a96b17f9f593a0898fcc5bfbd2562a12cde405b7f0db77b1ac267420a.jpg)

![798cb2a0f16517b702f460278f24e24682c219bf67f817ada70ddc2253f0e65b.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/tables/798cb2a0f16517b702f460278f24e24682c219bf67f817ada70ddc2253f0e65b.jpg)

![86f6acef7626859ac52bdbf415eef4fb71df760df38d9b9243f69ffc374aa334.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/tables/86f6acef7626859ac52bdbf415eef4fb71df760df38d9b9243f69ffc374aa334.jpg)

![b8924bce5d3f0ff0749b63cf9f940ef07a6a5b478bd089cd17cfe458657fa83d.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/tables/b8924bce5d3f0ff0749b63cf9f940ef07a6a5b478bd089cd17cfe458657fa83d.jpg)

![d51dd0430c41df1ba8a5066debf77c6298483ec82b58234647321a486b3a4300.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/tables/d51dd0430c41df1ba8a5066debf77c6298483ec82b58234647321a486b3a4300.jpg)

![dd20f3ddcd323f5e107cdecd299402c01c0fde67b3121fa052c79bdae277d9b8.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/tables/dd20f3ddcd323f5e107cdecd299402c01c0fde67b3121fa052c79bdae277d9b8.jpg)

![eb9619c6a1540a17f39d69500c274a3e93b00ce58d609a7d79d97b046308d25f.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/tables/eb9619c6a1540a17f39d69500c274a3e93b00ce58d609a7d79d97b046308d25f.jpg)

![edcc8ce7a650990468e37dcdae6ecf0b9abd9b4bbc4e20ffb4a7dc9ee6238760.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/tables/edcc8ce7a650990468e37dcdae6ecf0b9abd9b4bbc4e20ffb4a7dc9ee6238760.jpg)

![f377bade24978cca4790c757166f377905992e3bd7e561e823b6fd93490fdf15.jpg](../icml_results/5629_Pianist%20Transformer_%20Towards%20Expressive%20Piano%20Performance%20Rendering%20via%20Scalable%20Self-Supervised%20Pre/tables/f377bade24978cca4790c757166f377905992e3bd7e561e823b6fd93490fdf15.jpg)

<a id="paper-5581"></a>

## DDGA: Dirichlet Distributional Gradient Aggregation for Transferable Vision-Language Adversarial Attacks


### Images

![1eb947952bf9f7abc4fdf844a2dd05b72cb36da12f80ca14b5c4ca569dbc6327.jpg](../icml_results/5630_DDGA_%20Dirichlet%20Distributional%20Gradient%20Aggregation%20for%20Transferable%20Vision-Language%20Adversarial%20Att/images/1eb947952bf9f7abc4fdf844a2dd05b72cb36da12f80ca14b5c4ca569dbc6327.jpg)

![314473a24b1499104b8310c4d8e9e720f0a092f631b3a019cd6704c4a28ac3b0.jpg](../icml_results/5630_DDGA_%20Dirichlet%20Distributional%20Gradient%20Aggregation%20for%20Transferable%20Vision-Language%20Adversarial%20Att/images/314473a24b1499104b8310c4d8e9e720f0a092f631b3a019cd6704c4a28ac3b0.jpg)

![3d19c9b9135803c8de5082573694968faae671faf3eda20707fc525e6c4fc890.jpg](../icml_results/5630_DDGA_%20Dirichlet%20Distributional%20Gradient%20Aggregation%20for%20Transferable%20Vision-Language%20Adversarial%20Att/images/3d19c9b9135803c8de5082573694968faae671faf3eda20707fc525e6c4fc890.jpg)

![430d72ce3201ba8cea3e1a43329f64c60328db148ef073c76d41b5cbc7183dec.jpg](../icml_results/5630_DDGA_%20Dirichlet%20Distributional%20Gradient%20Aggregation%20for%20Transferable%20Vision-Language%20Adversarial%20Att/images/430d72ce3201ba8cea3e1a43329f64c60328db148ef073c76d41b5cbc7183dec.jpg)

![4a141421a536c2e534879937792ba8c456ec7e5feb313b46ddbc3990fae184a0.jpg](../icml_results/5630_DDGA_%20Dirichlet%20Distributional%20Gradient%20Aggregation%20for%20Transferable%20Vision-Language%20Adversarial%20Att/images/4a141421a536c2e534879937792ba8c456ec7e5feb313b46ddbc3990fae184a0.jpg)

![a38ea0877a66295f9433f146ce19cce93a8f736dfbebe9b3c34d37e3446aad1f.jpg](../icml_results/5630_DDGA_%20Dirichlet%20Distributional%20Gradient%20Aggregation%20for%20Transferable%20Vision-Language%20Adversarial%20Att/images/a38ea0877a66295f9433f146ce19cce93a8f736dfbebe9b3c34d37e3446aad1f.jpg)

![de611c8e3eb4248bd70e65da2087e392373b60b717fb73a7bbc765e1a98b574f.jpg](../icml_results/5630_DDGA_%20Dirichlet%20Distributional%20Gradient%20Aggregation%20for%20Transferable%20Vision-Language%20Adversarial%20Att/images/de611c8e3eb4248bd70e65da2087e392373b60b717fb73a7bbc765e1a98b574f.jpg)

![e0ea8395220c417fa031cf147e673d27901aca48da16416cd10db25c15034f0e.jpg](../icml_results/5630_DDGA_%20Dirichlet%20Distributional%20Gradient%20Aggregation%20for%20Transferable%20Vision-Language%20Adversarial%20Att/images/e0ea8395220c417fa031cf147e673d27901aca48da16416cd10db25c15034f0e.jpg)

### Tables

![0eec8016045afb3f0808c1f179ccac486d4cda8e68e2782db2bb42c48c211960.jpg](../icml_results/5630_DDGA_%20Dirichlet%20Distributional%20Gradient%20Aggregation%20for%20Transferable%20Vision-Language%20Adversarial%20Att/tables/0eec8016045afb3f0808c1f179ccac486d4cda8e68e2782db2bb42c48c211960.jpg)

![1684d24fe82bca2b7399d71dcbb82c8d71a80ee1370612c5ce4348fd132e90b9.jpg](../icml_results/5630_DDGA_%20Dirichlet%20Distributional%20Gradient%20Aggregation%20for%20Transferable%20Vision-Language%20Adversarial%20Att/tables/1684d24fe82bca2b7399d71dcbb82c8d71a80ee1370612c5ce4348fd132e90b9.jpg)

![3b49317fc6329693a82ccdc9d5f52420384acc5d28c6317148419e1e47a8e53a.jpg](../icml_results/5630_DDGA_%20Dirichlet%20Distributional%20Gradient%20Aggregation%20for%20Transferable%20Vision-Language%20Adversarial%20Att/tables/3b49317fc6329693a82ccdc9d5f52420384acc5d28c6317148419e1e47a8e53a.jpg)

![8074adde08e8a4df907a79d37a3c289f4475ccbe5d9c61b68337a9ac83623ad7.jpg](../icml_results/5630_DDGA_%20Dirichlet%20Distributional%20Gradient%20Aggregation%20for%20Transferable%20Vision-Language%20Adversarial%20Att/tables/8074adde08e8a4df907a79d37a3c289f4475ccbe5d9c61b68337a9ac83623ad7.jpg)

![980ccca1e22205d6d11b4d07c4c35b45a1539655c9ab6c4d3f6ea862840b5db3.jpg](../icml_results/5630_DDGA_%20Dirichlet%20Distributional%20Gradient%20Aggregation%20for%20Transferable%20Vision-Language%20Adversarial%20Att/tables/980ccca1e22205d6d11b4d07c4c35b45a1539655c9ab6c4d3f6ea862840b5db3.jpg)

![b55a1133d3c3d0757395d9812d402b4feb2224b648404035af6079b478adb532.jpg](../icml_results/5630_DDGA_%20Dirichlet%20Distributional%20Gradient%20Aggregation%20for%20Transferable%20Vision-Language%20Adversarial%20Att/tables/b55a1133d3c3d0757395d9812d402b4feb2224b648404035af6079b478adb532.jpg)

![d66473a33467c80674111e56863702e56a1cc7396b12d9d425b7fe574cd627c3.jpg](../icml_results/5630_DDGA_%20Dirichlet%20Distributional%20Gradient%20Aggregation%20for%20Transferable%20Vision-Language%20Adversarial%20Att/tables/d66473a33467c80674111e56863702e56a1cc7396b12d9d425b7fe574cd627c3.jpg)

![eccadea1fe82dd0688f2f6585d10ed3b778347fddd5c2b44106fc3c53a0b8d38.jpg](../icml_results/5630_DDGA_%20Dirichlet%20Distributional%20Gradient%20Aggregation%20for%20Transferable%20Vision-Language%20Adversarial%20Att/tables/eccadea1fe82dd0688f2f6585d10ed3b778347fddd5c2b44106fc3c53a0b8d38.jpg)

<a id="paper-5582"></a>

## Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards


### Images

![1f23f727adb6420f9618b073b9cc15103d92daee31d648751fa48de53fca3247.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/1f23f727adb6420f9618b073b9cc15103d92daee31d648751fa48de53fca3247.jpg)

![202d3ddabd6a031d7b117d83038003b81addf263cbcbd35fbc1626b1bd8e9575.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/202d3ddabd6a031d7b117d83038003b81addf263cbcbd35fbc1626b1bd8e9575.jpg)

![2f38e6e4341d6e4311285fb86cbc0dcc5e7682b32672433e11d239cc3d4c88b2.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/2f38e6e4341d6e4311285fb86cbc0dcc5e7682b32672433e11d239cc3d4c88b2.jpg)

![378904e8b269d65fa5c53d400ad6e650244d8301d402470eb14b96600eb09be3.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/378904e8b269d65fa5c53d400ad6e650244d8301d402470eb14b96600eb09be3.jpg)

![3a1d11f10bfac5f74963acd3835e5d3be85d9a91f0246041608ba13fa56ab408.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/3a1d11f10bfac5f74963acd3835e5d3be85d9a91f0246041608ba13fa56ab408.jpg)

![540d654572a331e4a7fd1b9ae363a6f8043ca80f9c239fa2c32f67e036b5a5c9.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/540d654572a331e4a7fd1b9ae363a6f8043ca80f9c239fa2c32f67e036b5a5c9.jpg)

![5b8f09d960a0fe46d3d94f9fae28d008ef3aeac0681547235da2e428c0ca512c.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/5b8f09d960a0fe46d3d94f9fae28d008ef3aeac0681547235da2e428c0ca512c.jpg)

![60c4b56f97ed3141d978ac7a3261c874b57f4a44fcc45515969ba9f2f0a64312.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/60c4b56f97ed3141d978ac7a3261c874b57f4a44fcc45515969ba9f2f0a64312.jpg)

![63dc4e32938a529147114599a00c5115330f55efd3973b4c3e17e25e5af7fb2f.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/63dc4e32938a529147114599a00c5115330f55efd3973b4c3e17e25e5af7fb2f.jpg)

![7b88e14dfe9c74af2791029095150df9b8daadb478174b4ccae901df094b064e.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/7b88e14dfe9c74af2791029095150df9b8daadb478174b4ccae901df094b064e.jpg)

![922ec09a9df5b55d8ee9f2dedd040928dae16b16f48929051ea8eec0e7f08ba5.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/922ec09a9df5b55d8ee9f2dedd040928dae16b16f48929051ea8eec0e7f08ba5.jpg)

![9eabf21e990a91242f233b99fcb0e0fa49892575a8a349c2eee5fec012943755.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/9eabf21e990a91242f233b99fcb0e0fa49892575a8a349c2eee5fec012943755.jpg)

![a8d1cb0591a0e02b323b0123be7d0aefa3db3636844abb5835c9c7f6c0d46bd0.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/a8d1cb0591a0e02b323b0123be7d0aefa3db3636844abb5835c9c7f6c0d46bd0.jpg)

![be1d581ae1f936e2416065c4d699708c307891166947a79b275dcb55c295f4f2.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/be1d581ae1f936e2416065c4d699708c307891166947a79b275dcb55c295f4f2.jpg)

![cc5d8ec058d6cc3aabeea58be4de28ca6b5429085e840ca35f5ea22d6823a0d8.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/cc5d8ec058d6cc3aabeea58be4de28ca6b5429085e840ca35f5ea22d6823a0d8.jpg)

![eeed7281b25c8abce1c62b16e76e99b3b21cda3b2c4d9e8658b269463286986b.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/eeed7281b25c8abce1c62b16e76e99b3b21cda3b2c4d9e8658b269463286986b.jpg)

![f21bf62faa604abe599fae3dc1676fe11b3296bb6f11f59d67e62502e91e9582.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/f21bf62faa604abe599fae3dc1676fe11b3296bb6f11f59d67e62502e91e9582.jpg)

![f56bc9fc0f964dd0c481c05c6ccf8920524eb2d295678c2c84d80092940d01ca.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/f56bc9fc0f964dd0c481c05c6ccf8920524eb2d295678c2c84d80092940d01ca.jpg)

![fa01862941807f4b5690e486f639fcfd624974d835c7394b5c4f9ad275d19b6a.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/fa01862941807f4b5690e486f639fcfd624974d835c7394b5c4f9ad275d19b6a.jpg)

![fb529e2c708c43a417da6e417be3279a223b55d7bdb75a9eab30e0b6e8e1d125.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/fb529e2c708c43a417da6e417be3279a223b55d7bdb75a9eab30e0b6e8e1d125.jpg)

![fc82bf9e70f659b003112774814cfd57e4d6ee9365cec001ee29a11a9539d4ed.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/images/fc82bf9e70f659b003112774814cfd57e4d6ee9365cec001ee29a11a9539d4ed.jpg)

### Tables

![0f65ebfc4cefc4fe0751642ffc33b562492425bd1ef568bf547ba28e669aefed.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/tables/0f65ebfc4cefc4fe0751642ffc33b562492425bd1ef568bf547ba28e669aefed.jpg)

![27b96caf14baeb04f33138cc7fddd8f5d1ff0395c83f08953d660ffef6bbd7ed.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/tables/27b96caf14baeb04f33138cc7fddd8f5d1ff0395c83f08953d660ffef6bbd7ed.jpg)

![514f826fed2d86ba92e7e93365fdfd4dc4421674437fa02daa59946bd58fd8aa.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/tables/514f826fed2d86ba92e7e93365fdfd4dc4421674437fa02daa59946bd58fd8aa.jpg)

![564989854ff73dc97f495c09db3b9feb634a5d0189e0d9fd2d72a584144c810a.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/tables/564989854ff73dc97f495c09db3b9feb634a5d0189e0d9fd2d72a584144c810a.jpg)

![8186dabb5d5bc4de5daca1e577ea194d8d6d399bc1d6524c820b53c0ed73c30e.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/tables/8186dabb5d5bc4de5daca1e577ea194d8d6d399bc1d6524c820b53c0ed73c30e.jpg)

![cd670bd8d4a1447b2ad669ee4d4720b80393a1aad377fbf629c33505292f37f9.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/tables/cd670bd8d4a1447b2ad669ee4d4720b80393a1aad377fbf629c33505292f37f9.jpg)

![d222208d4867b113556c9e3a6d0baf090e7d81942bef2d4bb279a40430866e16.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/tables/d222208d4867b113556c9e3a6d0baf090e7d81942bef2d4bb279a40430866e16.jpg)

![d9485c6b6e12d3cb9f2550db9458a5c154dd7df747802b0107fb58a6aa10f01c.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/tables/d9485c6b6e12d3cb9f2550db9458a5c154dd7df747802b0107fb58a6aa10f01c.jpg)

![d967c4f104b221080f7705faea8808e063e65ae2214e781801106c27f88c098d.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/tables/d967c4f104b221080f7705faea8808e063e65ae2214e781801106c27f88c098d.jpg)

![f98e54fdec98bc9cf31e4a7f9da85ca143f55a57f4d21a0c28244de08a628524.jpg](../icml_results/5631_Conflict-Aware%20Additive%20Guidance%20for%20Flow%20Models%20under%20Compositional%20Rewards/tables/f98e54fdec98bc9cf31e4a7f9da85ca143f55a57f4d21a0c28244de08a628524.jpg)

<a id="paper-5583"></a>

## SWE-ABS: Adversarial Benchmark Strengthening Exposes Inflated Success Rates on Test-based Benchmark


### Images

![078587c5e515983388036af5787cccabc76e4b0b480053a0c3cc2317c5691cf5.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/images/078587c5e515983388036af5787cccabc76e4b0b480053a0c3cc2317c5691cf5.jpg)

![1f92df7f77b8843de5d8642513bf31136f3dec395afc402c800c296dbb6fc193.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/images/1f92df7f77b8843de5d8642513bf31136f3dec395afc402c800c296dbb6fc193.jpg)

![6f2ff9cb8f9e59322a0e40e10871036a0bcf896e914d544520c91efe9166429d.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/images/6f2ff9cb8f9e59322a0e40e10871036a0bcf896e914d544520c91efe9166429d.jpg)

![7a6e95b81133f40c593da1fe4f6e72582d7aef57038745f29d493c182c180dda.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/images/7a6e95b81133f40c593da1fe4f6e72582d7aef57038745f29d493c182c180dda.jpg)

![fd441c70c420396b4d68ae651601d1ac0bf84267e6a5f11b0dac7ac70af6a738.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/images/fd441c70c420396b4d68ae651601d1ac0bf84267e6a5f11b0dac7ac70af6a738.jpg)

![fe336e48818c440043222e54701414b1ca250a730332cd525fc686c235d7d145.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/images/fe336e48818c440043222e54701414b1ca250a730332cd525fc686c235d7d145.jpg)

### Tables

![10c53729f2bd6ee96c7f25bcbc652a2eb014decb57fd32e7a69b32cad4a501f9.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/10c53729f2bd6ee96c7f25bcbc652a2eb014decb57fd32e7a69b32cad4a501f9.jpg)

![19203a52b331926577b631c24a5bd9aca4debc0e4111f8dc0578bedb949e18f6.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/19203a52b331926577b631c24a5bd9aca4debc0e4111f8dc0578bedb949e18f6.jpg)

![23803bf6f2e2372c6280640d3052f66bb6456c330217b44f92b6ed9967562d3c.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/23803bf6f2e2372c6280640d3052f66bb6456c330217b44f92b6ed9967562d3c.jpg)

![27c7d00544affd52bd46d5a0325bcd8297467d31c73124fad73ea5556481adc5.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/27c7d00544affd52bd46d5a0325bcd8297467d31c73124fad73ea5556481adc5.jpg)

![2cdc6ea2ae21f863e0aee202bda4945afb1d5ca4ec3885af4bdd0c45ec336fea.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/2cdc6ea2ae21f863e0aee202bda4945afb1d5ca4ec3885af4bdd0c45ec336fea.jpg)

![341e2dcaf14fa41e25cce7de7b9abc81d0c3da56802031b7233d6027e0f4a3c6.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/341e2dcaf14fa41e25cce7de7b9abc81d0c3da56802031b7233d6027e0f4a3c6.jpg)

![355190cabfb014b433854601f04d7fe3adc7bd2e7313d377b8d9e38c91775062.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/355190cabfb014b433854601f04d7fe3adc7bd2e7313d377b8d9e38c91775062.jpg)

![3dd499287fc766a556e02e5b50d39c1ca4d074831dbb73ef13a7846b5ff11b09.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/3dd499287fc766a556e02e5b50d39c1ca4d074831dbb73ef13a7846b5ff11b09.jpg)

![3e0bcebf2ff7849c53a3a3e4a6afa3b7f63f0f4f0e63d9015531891b98be6170.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/3e0bcebf2ff7849c53a3a3e4a6afa3b7f63f0f4f0e63d9015531891b98be6170.jpg)

![4653712a3cba407dfd53dc96d3ebe5321dbe530a60e1d4cfd7a0cf94a1547d7b.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/4653712a3cba407dfd53dc96d3ebe5321dbe530a60e1d4cfd7a0cf94a1547d7b.jpg)

![4a8757b9c0b4315b3ba2a7ad54e5f2c3d59565459886acb88e0e419738fcd405.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/4a8757b9c0b4315b3ba2a7ad54e5f2c3d59565459886acb88e0e419738fcd405.jpg)

![518e7f5f96ba4b27be893ded114b73278476fd0568393b8de6277315bbb872fc.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/518e7f5f96ba4b27be893ded114b73278476fd0568393b8de6277315bbb872fc.jpg)

![52dfd9f097bc598cf7307d3772b8b5ca3f2919bde5a8300db5a276f1b41aab3b.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/52dfd9f097bc598cf7307d3772b8b5ca3f2919bde5a8300db5a276f1b41aab3b.jpg)

![6e4715e9d8f90fded909dad957d5d619fd80b3864d89960e14a2e1bae9280392.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/6e4715e9d8f90fded909dad957d5d619fd80b3864d89960e14a2e1bae9280392.jpg)

![7399dc2aca73f4826753867a5b072580d3ade340c986e43a2877318c59528d54.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/7399dc2aca73f4826753867a5b072580d3ade340c986e43a2877318c59528d54.jpg)

![859780b87164007efa1e15468d680faf094a16f516746f5267dc82d8e19ae0df.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/859780b87164007efa1e15468d680faf094a16f516746f5267dc82d8e19ae0df.jpg)

![8ab4445607e69f77ba57dca898a8ba30ce09ca6996bcd3f0a65a047aebef172a.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/8ab4445607e69f77ba57dca898a8ba30ce09ca6996bcd3f0a65a047aebef172a.jpg)

![8b67ffb1f8931d426d81fd73ab48d0ba2dbe7531f5ae209d164fbfcf5001d2da.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/8b67ffb1f8931d426d81fd73ab48d0ba2dbe7531f5ae209d164fbfcf5001d2da.jpg)

![a2af60fa0558243110f30c4f314d990da6ce01c0a3ac457701016daa1455557b.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/a2af60fa0558243110f30c4f314d990da6ce01c0a3ac457701016daa1455557b.jpg)

![ae2fbaa8be4f677b94e61f0c791764e098f10027cf1e90acae5129588ba18498.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/ae2fbaa8be4f677b94e61f0c791764e098f10027cf1e90acae5129588ba18498.jpg)

![b090fdb5e4006abb5184c051ec08d9c071c3c1f47e6b51090004da3178797c11.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/b090fdb5e4006abb5184c051ec08d9c071c3c1f47e6b51090004da3178797c11.jpg)

![bd77faf6d4b0cab5d0b8cdd427bead1a473137fbb21a8d4408ffb48c866c69c2.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/bd77faf6d4b0cab5d0b8cdd427bead1a473137fbb21a8d4408ffb48c866c69c2.jpg)

![bde12e9f03e4ac3aa99933c3833d6a6a0d8f1d2558d37ca24b349f214f93e072.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/bde12e9f03e4ac3aa99933c3833d6a6a0d8f1d2558d37ca24b349f214f93e072.jpg)

![d8a008768ab6d6be043d96ed9b3adf74898f7f57f7977357d1b4d673dd37b0c8.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/d8a008768ab6d6be043d96ed9b3adf74898f7f57f7977357d1b4d673dd37b0c8.jpg)

![e7d7c09ee7f2b0e15badf30d4ef6d05bc392ba024ded58df14b297003317e96a.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/e7d7c09ee7f2b0e15badf30d4ef6d05bc392ba024ded58df14b297003317e96a.jpg)

![ee74d2c80e3d4ebc2e042e2b6f8df5bf70c79751bbbed35ff10f78b1cc2f8d4d.jpg](../icml_results/5632_SWE-ABS_%20Adversarial%20Benchmark%20Strengthening%20Exposes%20Inflated%20Success%20Rates%20on%20Test-based%20Benchmark/tables/ee74d2c80e3d4ebc2e042e2b6f8df5bf70c79751bbbed35ff10f78b1cc2f8d4d.jpg)

<a id="paper-5584"></a>

## Dismantling Pathological Shortcuts: A Causal Framework for Faithful LVLM Decoding


### Images

![0a0ec8671ae3f2f8c48a6ee2a931c8253e17f69b39f4d1606f1cc0e64672232b.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/0a0ec8671ae3f2f8c48a6ee2a931c8253e17f69b39f4d1606f1cc0e64672232b.jpg)

![0a12c5cff90ade37794d4bc1f50a27dae895d488773e446762acd0bc229ffee0.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/0a12c5cff90ade37794d4bc1f50a27dae895d488773e446762acd0bc229ffee0.jpg)

![0e490c9348029feb5a7757428450fec5e9f64bc004490527a083256ebffffb06.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/0e490c9348029feb5a7757428450fec5e9f64bc004490527a083256ebffffb06.jpg)

![119eb52404295253739ff738236fd6b03ab727d9eddeeca5b90958558dd6a7c8.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/119eb52404295253739ff738236fd6b03ab727d9eddeeca5b90958558dd6a7c8.jpg)

![14411105dbddaebbd2a54fa581a4d6c4cc36da2ea175747b78510751352ce195.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/14411105dbddaebbd2a54fa581a4d6c4cc36da2ea175747b78510751352ce195.jpg)

![190d4f846e3701c314897e0fed11c84ab555850c7fd27bc6806b3ea20d902283.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/190d4f846e3701c314897e0fed11c84ab555850c7fd27bc6806b3ea20d902283.jpg)

![22e85caf9738b462cf64703e0d7635d379456929bdc7239a30fd7bdb0a331b33.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/22e85caf9738b462cf64703e0d7635d379456929bdc7239a30fd7bdb0a331b33.jpg)

![4104f921d72c3a141c7fdbdb4b85722e5be988fc8809d9dae174bec73c52140e.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/4104f921d72c3a141c7fdbdb4b85722e5be988fc8809d9dae174bec73c52140e.jpg)

![41b34aa8e4c34d7e700a45bfe793d3177e283d85387cd55c5017f12497df6c47.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/41b34aa8e4c34d7e700a45bfe793d3177e283d85387cd55c5017f12497df6c47.jpg)

![440bb157291ad996fbdfc95a602b474507f32b73f73a4bc2b7955b618ba112e1.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/440bb157291ad996fbdfc95a602b474507f32b73f73a4bc2b7955b618ba112e1.jpg)

![475b924514bec4d0d40b1ff4f614f6aaa04e0b87214c90f397f0ec88f6c14fa0.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/475b924514bec4d0d40b1ff4f614f6aaa04e0b87214c90f397f0ec88f6c14fa0.jpg)

![4e92570651c70122776ab27256848373363a33d0c1b96d43602f346a2fedddfe.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/4e92570651c70122776ab27256848373363a33d0c1b96d43602f346a2fedddfe.jpg)

![5422ee8057829d8226057d62c2301c5bcf9f08e21eb5e517eb497ba515a05093.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/5422ee8057829d8226057d62c2301c5bcf9f08e21eb5e517eb497ba515a05093.jpg)

![58ca5310a6ed0c80647bbc2810841224dea17779195941b7d3036f32d920e38e.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/58ca5310a6ed0c80647bbc2810841224dea17779195941b7d3036f32d920e38e.jpg)

![5ba65fdf54bcec5d322107bc52e5dcf0dded26a8e5b7c68f0a772cfd18fa36c1.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/5ba65fdf54bcec5d322107bc52e5dcf0dded26a8e5b7c68f0a772cfd18fa36c1.jpg)

![622ea5ec60956de0dcd3ceef3548cb29a9dacdc270bacbb65a375180589ce567.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/622ea5ec60956de0dcd3ceef3548cb29a9dacdc270bacbb65a375180589ce567.jpg)

![6aa567d7d6f0f1559c57f7ecbb4e7fefa6971b0e5e1eab9327e31b2877d78ba3.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/6aa567d7d6f0f1559c57f7ecbb4e7fefa6971b0e5e1eab9327e31b2877d78ba3.jpg)

![781fb15c0a02723d497f22827c37dec4fec6172de0363e90d0ae034bab66eab7.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/781fb15c0a02723d497f22827c37dec4fec6172de0363e90d0ae034bab66eab7.jpg)

![8f0a5eab203fdba8ab44a2db7146be14c576560b3c0063c366acdf64800a74c5.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/8f0a5eab203fdba8ab44a2db7146be14c576560b3c0063c366acdf64800a74c5.jpg)

![a5b2696fbe26a106b5b26f3572ca8b94b0d46aa64489fd0fae7ef54266ac27fb.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/a5b2696fbe26a106b5b26f3572ca8b94b0d46aa64489fd0fae7ef54266ac27fb.jpg)

![a731b13266a88dbd1c49c29e6153e8d95672b0b1f6252089d9b493c4765250ac.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/a731b13266a88dbd1c49c29e6153e8d95672b0b1f6252089d9b493c4765250ac.jpg)

![ad9e21716f79f3e29dca29958e53dc9ecee5c23019f6f5226f890c8721b5e0cd.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/ad9e21716f79f3e29dca29958e53dc9ecee5c23019f6f5226f890c8721b5e0cd.jpg)

![b4026987b0ee8726535bd94c514037e5e8d9589820066f1e0a559338614d5e52.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/b4026987b0ee8726535bd94c514037e5e8d9589820066f1e0a559338614d5e52.jpg)

![c66d145fd3249bf5e88c71f7cfb0a7f974850861eb0e6745a908068fe304624f.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/c66d145fd3249bf5e88c71f7cfb0a7f974850861eb0e6745a908068fe304624f.jpg)

![caecd05256dc31978e08ddb21d685c3df8fda8924014c6edd14d299c093a6a0a.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/caecd05256dc31978e08ddb21d685c3df8fda8924014c6edd14d299c093a6a0a.jpg)

![d844aaa9c6ba49cab62ceb86302ebc04b1fa824ef4a398591047a92d55ceab33.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/d844aaa9c6ba49cab62ceb86302ebc04b1fa824ef4a398591047a92d55ceab33.jpg)

![d9aaeed1f3ce88d5cdf49b81a6a40ee548580d3ca2bdaf62050b25725045b90c.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/d9aaeed1f3ce88d5cdf49b81a6a40ee548580d3ca2bdaf62050b25725045b90c.jpg)

![da10e122dd9067700b6f78bd7786dee4e1ac4c4766c08fbee87c93ce8853c276.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/da10e122dd9067700b6f78bd7786dee4e1ac4c4766c08fbee87c93ce8853c276.jpg)

![eb3727e18e62113fdf1c87fe822518d4ca55d443acba61c32e1eff10f7473a4f.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/eb3727e18e62113fdf1c87fe822518d4ca55d443acba61c32e1eff10f7473a4f.jpg)

![f14eeb2fc1d81df5ea9174d56c54af9e7c6d87db0a08fb7304221479a6030640.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/f14eeb2fc1d81df5ea9174d56c54af9e7c6d87db0a08fb7304221479a6030640.jpg)

![f34497fedfe69f0b508fb83bb651e20e126db4333c3a61f92cfae8b415856ac3.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/f34497fedfe69f0b508fb83bb651e20e126db4333c3a61f92cfae8b415856ac3.jpg)

![f82c26500ce0e1888f593bc7ae1f44cfaa72a89720874d75103e7864b1dfa172.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/f82c26500ce0e1888f593bc7ae1f44cfaa72a89720874d75103e7864b1dfa172.jpg)

![fab2c4d872b7596a6d95167caa49506de53234c9f3d3d71c89ea16148ca5529f.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/images/fab2c4d872b7596a6d95167caa49506de53234c9f3d3d71c89ea16148ca5529f.jpg)

### Tables

![0bd253a241485e50e4084cec676534b86aafee4ca1514a15536e143b038c81b2.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/tables/0bd253a241485e50e4084cec676534b86aafee4ca1514a15536e143b038c81b2.jpg)

![0ea70b23ac8570ada5f069a6e46879ddbe203602fd1f97ba414b9defd8326d79.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/tables/0ea70b23ac8570ada5f069a6e46879ddbe203602fd1f97ba414b9defd8326d79.jpg)

![2c985aeb2e2aadd266bf84c8ae633f53f7f316b9633b94610d8fbbe1fc446722.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/tables/2c985aeb2e2aadd266bf84c8ae633f53f7f316b9633b94610d8fbbe1fc446722.jpg)

![309f531b82066e602ae9157f91ec4d209f6cd9654d7741e0fd315bc7822ab319.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/tables/309f531b82066e602ae9157f91ec4d209f6cd9654d7741e0fd315bc7822ab319.jpg)

![4e039bec122505e37e84872d266e669e44736fcefbc059d5488a82d1abd9a525.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/tables/4e039bec122505e37e84872d266e669e44736fcefbc059d5488a82d1abd9a525.jpg)

![783ac9de5eb0972b4f4161d1a90fbd7f0d018372bedd51e1343a0330bfba4b77.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/tables/783ac9de5eb0972b4f4161d1a90fbd7f0d018372bedd51e1343a0330bfba4b77.jpg)

![79d1d76a788bd081c67d26760f39fd13dc0884e3145687277325d8442c6a7518.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/tables/79d1d76a788bd081c67d26760f39fd13dc0884e3145687277325d8442c6a7518.jpg)

![888c0f329ae8c25d41a533122b32c89f874483a9f670ece4494d33977e9c8535.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/tables/888c0f329ae8c25d41a533122b32c89f874483a9f670ece4494d33977e9c8535.jpg)

![996e3cf1d9e43389c19382fa3f0a1c7c294a8f4085482f19661550773fea60fc.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/tables/996e3cf1d9e43389c19382fa3f0a1c7c294a8f4085482f19661550773fea60fc.jpg)

![9bfd36a15da55d5f00caffc1c73b9df1dd535c913633e47caa14d475f5e04ea7.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/tables/9bfd36a15da55d5f00caffc1c73b9df1dd535c913633e47caa14d475f5e04ea7.jpg)

![ab74ab4e0a4f4f1eec5c76fb7559b2475d7fb631288401be30d243991e88d994.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/tables/ab74ab4e0a4f4f1eec5c76fb7559b2475d7fb631288401be30d243991e88d994.jpg)

![bbc4cfec1a8c5cb8c766289206794ac1437064b046c33946b647cc2fde3ec895.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/tables/bbc4cfec1a8c5cb8c766289206794ac1437064b046c33946b647cc2fde3ec895.jpg)

![d7d2841044708910aaafce1c4b56be1c89b7876b056451dba039be8ca9d4eda4.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/tables/d7d2841044708910aaafce1c4b56be1c89b7876b056451dba039be8ca9d4eda4.jpg)

![db59cfcbdbb6347c3b8a695e28a45f8740e813d1961174d56a893bd0f0bc087f.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/tables/db59cfcbdbb6347c3b8a695e28a45f8740e813d1961174d56a893bd0f0bc087f.jpg)

![fff47996adc03ca99283c08e865a4f1127f0795c02d5726aea52bfc9cf65cca1.jpg](../icml_results/5633_Dismantling%20Pathological%20Shortcuts_%20A%20Causal%20Framework%20for%20Faithful%20LVLM%20Decoding/tables/fff47996adc03ca99283c08e865a4f1127f0795c02d5726aea52bfc9cf65cca1.jpg)

<a id="paper-5585"></a>

## Branch Scaling Manifests as Implicit Architectural Regularization for Improving Generalization in Overparameterized ResNets


### Images

![0f3e05d0e1d1cb3fe54c6b1395bbc0ecfd46a95307a164845375a3359f78f958.jpg](../icml_results/5634_Branch%20Scaling%20Manifests%20as%20Implicit%20Architectural%20Regularization%20for%20Improving%20Generalization%20in%20Ov/images/0f3e05d0e1d1cb3fe54c6b1395bbc0ecfd46a95307a164845375a3359f78f958.jpg)

![10523364911a7c08f0b87f80673ebb9a00b399d9f683c438e4a696f47f2ce5ba.jpg](../icml_results/5634_Branch%20Scaling%20Manifests%20as%20Implicit%20Architectural%20Regularization%20for%20Improving%20Generalization%20in%20Ov/images/10523364911a7c08f0b87f80673ebb9a00b399d9f683c438e4a696f47f2ce5ba.jpg)

![333ee0b5c5163badce1b4fa38a16838e9a8c8f4d8cc0dcdd8c4a1debea1fa6cc.jpg](../icml_results/5634_Branch%20Scaling%20Manifests%20as%20Implicit%20Architectural%20Regularization%20for%20Improving%20Generalization%20in%20Ov/images/333ee0b5c5163badce1b4fa38a16838e9a8c8f4d8cc0dcdd8c4a1debea1fa6cc.jpg)

![46029563159715d1bfc0979937ec397aa83f1fba0238fc429b6b18057c7496db.jpg](../icml_results/5634_Branch%20Scaling%20Manifests%20as%20Implicit%20Architectural%20Regularization%20for%20Improving%20Generalization%20in%20Ov/images/46029563159715d1bfc0979937ec397aa83f1fba0238fc429b6b18057c7496db.jpg)

![46461711c14bfc723dbe3612f7e697c1181642afde7288084375e696342a2f3f.jpg](../icml_results/5634_Branch%20Scaling%20Manifests%20as%20Implicit%20Architectural%20Regularization%20for%20Improving%20Generalization%20in%20Ov/images/46461711c14bfc723dbe3612f7e697c1181642afde7288084375e696342a2f3f.jpg)

![7a87daefd01e21e9de6295296665266a7c36424694bcb8b7d8f2a6e0ecbf93c8.jpg](../icml_results/5634_Branch%20Scaling%20Manifests%20as%20Implicit%20Architectural%20Regularization%20for%20Improving%20Generalization%20in%20Ov/images/7a87daefd01e21e9de6295296665266a7c36424694bcb8b7d8f2a6e0ecbf93c8.jpg)

![9e802080e9cf84e9011319a86c82445f4ee3fd6c168b1d37f07bb846f9e48841.jpg](../icml_results/5634_Branch%20Scaling%20Manifests%20as%20Implicit%20Architectural%20Regularization%20for%20Improving%20Generalization%20in%20Ov/images/9e802080e9cf84e9011319a86c82445f4ee3fd6c168b1d37f07bb846f9e48841.jpg)

### Tables

![4b0210d0fccfe7c88575256e7c2f46ca562c3885650f50ab07ceed1552ace5d1.jpg](../icml_results/5634_Branch%20Scaling%20Manifests%20as%20Implicit%20Architectural%20Regularization%20for%20Improving%20Generalization%20in%20Ov/tables/4b0210d0fccfe7c88575256e7c2f46ca562c3885650f50ab07ceed1552ace5d1.jpg)

<a id="paper-5586"></a>

## CacheEdit: Efficient Multi-round Image Editing via Adaptive Token-wise Reuse.


### Images

![143a614c2395dd62d146c2debcade30644e5e73f7572ad00f5f42b8cd0779489.jpg](../icml_results/5635_CacheEdit_%20Efficient%20Multi-round%20Image%20Editing%20via%20Adaptive%20Token-wise%20Reuse./images/143a614c2395dd62d146c2debcade30644e5e73f7572ad00f5f42b8cd0779489.jpg)

![1862f6fc536235dabc5034c7de01e02848d0e09e842fd8175d1d76c962468236.jpg](../icml_results/5635_CacheEdit_%20Efficient%20Multi-round%20Image%20Editing%20via%20Adaptive%20Token-wise%20Reuse./images/1862f6fc536235dabc5034c7de01e02848d0e09e842fd8175d1d76c962468236.jpg)

![1e0140c8172d0286da1d1e6d5cf7a2f15be19b9267a85f6ca35445e273c53132.jpg](../icml_results/5635_CacheEdit_%20Efficient%20Multi-round%20Image%20Editing%20via%20Adaptive%20Token-wise%20Reuse./images/1e0140c8172d0286da1d1e6d5cf7a2f15be19b9267a85f6ca35445e273c53132.jpg)

![3675c039e1cf2413939dd251f79937ce40bb19b9ceff42e21148bd48a1567731.jpg](../icml_results/5635_CacheEdit_%20Efficient%20Multi-round%20Image%20Editing%20via%20Adaptive%20Token-wise%20Reuse./images/3675c039e1cf2413939dd251f79937ce40bb19b9ceff42e21148bd48a1567731.jpg)

![39c836faa47ac5d329518f9cc2082fa65f7ca8ec97ae928499462102688cc4c2.jpg](../icml_results/5635_CacheEdit_%20Efficient%20Multi-round%20Image%20Editing%20via%20Adaptive%20Token-wise%20Reuse./images/39c836faa47ac5d329518f9cc2082fa65f7ca8ec97ae928499462102688cc4c2.jpg)

![3e6c7d097d4dfb1e4521abca4800df8ef97cb16a17f15ecee92735ff75a0b873.jpg](../icml_results/5635_CacheEdit_%20Efficient%20Multi-round%20Image%20Editing%20via%20Adaptive%20Token-wise%20Reuse./images/3e6c7d097d4dfb1e4521abca4800df8ef97cb16a17f15ecee92735ff75a0b873.jpg)

![834701b83651239960c2379864386798198f6b06e69db8c71384dfcf27241f6e.jpg](../icml_results/5635_CacheEdit_%20Efficient%20Multi-round%20Image%20Editing%20via%20Adaptive%20Token-wise%20Reuse./images/834701b83651239960c2379864386798198f6b06e69db8c71384dfcf27241f6e.jpg)

![84fbcc123137e961942f1a5caaf65f16431cd3fd36feac4cf9225690d8a62154.jpg](../icml_results/5635_CacheEdit_%20Efficient%20Multi-round%20Image%20Editing%20via%20Adaptive%20Token-wise%20Reuse./images/84fbcc123137e961942f1a5caaf65f16431cd3fd36feac4cf9225690d8a62154.jpg)

![ca99ef89c78a8a712ebe0b5abc43321bd9a4c1d88d3f54483bdb66753da05ab0.jpg](../icml_results/5635_CacheEdit_%20Efficient%20Multi-round%20Image%20Editing%20via%20Adaptive%20Token-wise%20Reuse./images/ca99ef89c78a8a712ebe0b5abc43321bd9a4c1d88d3f54483bdb66753da05ab0.jpg)

![d9dcb62a33f54c869628b42dad56de698f396f9510b45fdc726549743148445f.jpg](../icml_results/5635_CacheEdit_%20Efficient%20Multi-round%20Image%20Editing%20via%20Adaptive%20Token-wise%20Reuse./images/d9dcb62a33f54c869628b42dad56de698f396f9510b45fdc726549743148445f.jpg)

![f0395f182a75b2cef232ef5f5b17ec7ddb52c77e35e87fe8b744a78392cbd17b.jpg](../icml_results/5635_CacheEdit_%20Efficient%20Multi-round%20Image%20Editing%20via%20Adaptive%20Token-wise%20Reuse./images/f0395f182a75b2cef232ef5f5b17ec7ddb52c77e35e87fe8b744a78392cbd17b.jpg)

### Tables

![d64de265f7103299a17e617138c59c72444fbe4c90a7b3818e1ac0de87b6ede8.jpg](../icml_results/5635_CacheEdit_%20Efficient%20Multi-round%20Image%20Editing%20via%20Adaptive%20Token-wise%20Reuse./tables/d64de265f7103299a17e617138c59c72444fbe4c90a7b3818e1ac0de87b6ede8.jpg)

![d74cb2f76961cd09f556ab796069114a567a7716f6c6c15c66332428f58de821.jpg](../icml_results/5635_CacheEdit_%20Efficient%20Multi-round%20Image%20Editing%20via%20Adaptive%20Token-wise%20Reuse./tables/d74cb2f76961cd09f556ab796069114a567a7716f6c6c15c66332428f58de821.jpg)

<a id="paper-5587"></a>

## ClimateAR: Multi-Scale Autoregressive Generative Modeling for Climate Forecasting


### Images

![01fb3d0a851a59b38e943ee68a199a58fd02ace0b5588fb0e029311bc5c17ab0.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/01fb3d0a851a59b38e943ee68a199a58fd02ace0b5588fb0e029311bc5c17ab0.jpg)

![098587bfaa988669ab8b45e1262324b5c12a81ef4c327bbde77268b62f12ec33.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/098587bfaa988669ab8b45e1262324b5c12a81ef4c327bbde77268b62f12ec33.jpg)

![23749cb5117624dd58cf422d036a73c998d4e4aa1e0c0d603726d237335a6bb1.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/23749cb5117624dd58cf422d036a73c998d4e4aa1e0c0d603726d237335a6bb1.jpg)

![2ace23adb488bf46beda6ab51d05c5f789046dd228dedab450d6f2d1e49460cd.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/2ace23adb488bf46beda6ab51d05c5f789046dd228dedab450d6f2d1e49460cd.jpg)

![2d86cb0b8b15460089f609971c54d216b54f1b02fad651e9a3c09b32f5cbbbd2.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/2d86cb0b8b15460089f609971c54d216b54f1b02fad651e9a3c09b32f5cbbbd2.jpg)

![427d719d717ab56ecea8f744dbfa985d833b1a9402a4a4037aabf2b5d2e90b7f.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/427d719d717ab56ecea8f744dbfa985d833b1a9402a4a4037aabf2b5d2e90b7f.jpg)

![57878f3f2f49c51f2e6223b985a6e4277c622fe784ec9e9da5475d367d2da792.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/57878f3f2f49c51f2e6223b985a6e4277c622fe784ec9e9da5475d367d2da792.jpg)

![58639bc46d69f7260a6ce3aa98b316a444b092b530cfadce701a803a098a931d.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/58639bc46d69f7260a6ce3aa98b316a444b092b530cfadce701a803a098a931d.jpg)

![5b0e4d3a3e26fc02fe243a0b01db3905f2dbd9b683b1762ee4dd4173b275bee0.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/5b0e4d3a3e26fc02fe243a0b01db3905f2dbd9b683b1762ee4dd4173b275bee0.jpg)

![7fc26f1a7eef3fbcd4d9eed5d1adf2dcdf16d6578b575b3dbd8727f15588e898.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/7fc26f1a7eef3fbcd4d9eed5d1adf2dcdf16d6578b575b3dbd8727f15588e898.jpg)

![91ae5fa77aae3990cef8a331e64c22ded7fc8022fd949d0234588be01afdf753.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/91ae5fa77aae3990cef8a331e64c22ded7fc8022fd949d0234588be01afdf753.jpg)

![91cc716bcd418de3324b3a600dfb63509e716c2db2938dac3621e0571504814c.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/91cc716bcd418de3324b3a600dfb63509e716c2db2938dac3621e0571504814c.jpg)

![971e75fd97eb601647819466086d1a3eaf6aee574a419ddc72327da675257d45.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/971e75fd97eb601647819466086d1a3eaf6aee574a419ddc72327da675257d45.jpg)

![a82abfd21a72171ca96e9a1a6c71905bb95f0e93887e98ea2c8068dde0fc77d8.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/a82abfd21a72171ca96e9a1a6c71905bb95f0e93887e98ea2c8068dde0fc77d8.jpg)

![aebfe7dd66f771533ea352258a19b10585f0ef839361b5c90a7efd75bc0851e9.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/aebfe7dd66f771533ea352258a19b10585f0ef839361b5c90a7efd75bc0851e9.jpg)

![c4f6d2a43ebd9e301543d0082e6d0cb986435564ef30b4cab63b6a29e8c13eeb.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/c4f6d2a43ebd9e301543d0082e6d0cb986435564ef30b4cab63b6a29e8c13eeb.jpg)

![c9b0e8c8a5674e1d767fdfe5a3ac138752469ecc54688bc1f4d3d59e7a543c1c.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/c9b0e8c8a5674e1d767fdfe5a3ac138752469ecc54688bc1f4d3d59e7a543c1c.jpg)

![d9f2c17d58560a58c4cdf383f8cb528fb582775bbb7264a8c1268e9b6e8894a7.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/d9f2c17d58560a58c4cdf383f8cb528fb582775bbb7264a8c1268e9b6e8894a7.jpg)

![e0ccbedb3bb0259b0529c5efc61fc5520d0dc54a878dfc15a51a2b813ea5af3d.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/e0ccbedb3bb0259b0529c5efc61fc5520d0dc54a878dfc15a51a2b813ea5af3d.jpg)

![e76680ce9f37c0068de04049965ad51c623a180d53c75d183e8f05a2b31ce055.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/e76680ce9f37c0068de04049965ad51c623a180d53c75d183e8f05a2b31ce055.jpg)

![ebf96317bac85aff4797477b81c9234953d35e37b2178cf9fc533fdf800853ed.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/ebf96317bac85aff4797477b81c9234953d35e37b2178cf9fc533fdf800853ed.jpg)

![ed43f1b228a3c7670c2d38d01423bdc6a5c1e76c104d0df847df93fc5be18064.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/images/ed43f1b228a3c7670c2d38d01423bdc6a5c1e76c104d0df847df93fc5be18064.jpg)

### Tables

![2bba75b906c98bf102b201df15b78f58f3a754d17ed852e7f0908a442a4c3057.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/tables/2bba75b906c98bf102b201df15b78f58f3a754d17ed852e7f0908a442a4c3057.jpg)

![3a9d81e095d2cd82764c3274312286d2c88553d20e69463b1858d5d41f9f92f8.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/tables/3a9d81e095d2cd82764c3274312286d2c88553d20e69463b1858d5d41f9f92f8.jpg)

![7c87cd20466f83421b4e99cd4d9788eb94a72d520fca11ba7c1e26986a58561c.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/tables/7c87cd20466f83421b4e99cd4d9788eb94a72d520fca11ba7c1e26986a58561c.jpg)

![81a7f42d2d1936be9a6444eb03fdd18e083222a4bc04db837ef9e3c0784b7ee2.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/tables/81a7f42d2d1936be9a6444eb03fdd18e083222a4bc04db837ef9e3c0784b7ee2.jpg)

![923edf55cacafefee0ada3daa8f4740aa4b9da6ab0433cd4113f413bc7c5b8dd.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/tables/923edf55cacafefee0ada3daa8f4740aa4b9da6ab0433cd4113f413bc7c5b8dd.jpg)

![9cd029b3a5596c4af529e9785c0ce5f5f5099039ee95484f5145631fad2ba686.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/tables/9cd029b3a5596c4af529e9785c0ce5f5f5099039ee95484f5145631fad2ba686.jpg)

![a9fa78595058bb6a6bf744ca0e1b418b0e6bf03093be0b996239217c6ba1eda7.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/tables/a9fa78595058bb6a6bf744ca0e1b418b0e6bf03093be0b996239217c6ba1eda7.jpg)

![e7a6931897383c64facf09940e4d3f019edab4cf81b306d896086de080534308.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/tables/e7a6931897383c64facf09940e4d3f019edab4cf81b306d896086de080534308.jpg)

![f0b62bcb50db6a0af14e05be295040bbe8ab7100c7873cf9250a86e8d9bb51d5.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/tables/f0b62bcb50db6a0af14e05be295040bbe8ab7100c7873cf9250a86e8d9bb51d5.jpg)

![fe0a0b1b8da2957ac4acbbc0e72a55138f5aabda061b0aea4be232a964823d50.jpg](../icml_results/5636_ClimateAR_%20Multi-Scale%20Autoregressive%20Generative%20Modeling%20for%20Climate%20Forecasting/tables/fe0a0b1b8da2957ac4acbbc0e72a55138f5aabda061b0aea4be232a964823d50.jpg)

<a id="paper-5588"></a>

## NeurIPS: Neuro-anatomical Inductive Priors for Sphere-based Brain Decoding


### Images

![048d6422a0dc0e4462d69da32125e42ab05cfa6451cd960a16110f43806dda1e.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/images/048d6422a0dc0e4462d69da32125e42ab05cfa6451cd960a16110f43806dda1e.jpg)

![074a65359455fa616041b73a6f492f522bdb997e74005b0bd6c27ce2f1de6dfe.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/images/074a65359455fa616041b73a6f492f522bdb997e74005b0bd6c27ce2f1de6dfe.jpg)

![27344b0b94aa274271e7cee15f5d9eb1c5217860c7d3d8658086aa5136b119e5.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/images/27344b0b94aa274271e7cee15f5d9eb1c5217860c7d3d8658086aa5136b119e5.jpg)

![2b4fb58a6bdf3239f33c93142faeee80e100922fe149f0035b6f0dd97bcc05c3.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/images/2b4fb58a6bdf3239f33c93142faeee80e100922fe149f0035b6f0dd97bcc05c3.jpg)

![3b2be8559d0c953e2c6666e47eb01ef5f01fc86c7eb56a1084188e4ef3fd2f24.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/images/3b2be8559d0c953e2c6666e47eb01ef5f01fc86c7eb56a1084188e4ef3fd2f24.jpg)

![41920a37a96ab8830d7315dd45afb16aaf32e71e4f9099af7fb6dc6e68067295.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/images/41920a37a96ab8830d7315dd45afb16aaf32e71e4f9099af7fb6dc6e68067295.jpg)

![4beddd51d86a552c9b777cb12b26dceb5ff82a6ead61de1137f66fb1c75b5bba.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/images/4beddd51d86a552c9b777cb12b26dceb5ff82a6ead61de1137f66fb1c75b5bba.jpg)

![581b6decf50555e218473f43eea7486df53deca673d24a9512e9bd4829267c17.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/images/581b6decf50555e218473f43eea7486df53deca673d24a9512e9bd4829267c17.jpg)

![82835f7ec4ef41ce357f85b8f1d982a1c99734a1bda47dc864181eff851406a8.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/images/82835f7ec4ef41ce357f85b8f1d982a1c99734a1bda47dc864181eff851406a8.jpg)

![bdb33b2cec8fdf7000fd51394f29a8de18d4a8cefd4647c147a482c18aa75566.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/images/bdb33b2cec8fdf7000fd51394f29a8de18d4a8cefd4647c147a482c18aa75566.jpg)

![bf28a3845ece81d9b41a29b6e7e13a997a76534b9d3c563f41277073588dec91.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/images/bf28a3845ece81d9b41a29b6e7e13a997a76534b9d3c563f41277073588dec91.jpg)

![d801aa1eed1c112e3832993907d77d05d1b35c3433e25d3700476c45b4a2d1d3.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/images/d801aa1eed1c112e3832993907d77d05d1b35c3433e25d3700476c45b4a2d1d3.jpg)

![df0b1a7021df045df7b98d44486c311ac7dd3bb173840b19e7047dbb05361f4d.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/images/df0b1a7021df045df7b98d44486c311ac7dd3bb173840b19e7047dbb05361f4d.jpg)

![f9f98e509c558a8d012f7a2d9820d13f04d40da677369d759bce9c522c37c6ec.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/images/f9f98e509c558a8d012f7a2d9820d13f04d40da677369d759bce9c522c37c6ec.jpg)

### Tables

![12ce14f05b63e9186f7e6a57209cb4c11258960e3ee6b0a74e772db7e0b2e98a.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/tables/12ce14f05b63e9186f7e6a57209cb4c11258960e3ee6b0a74e772db7e0b2e98a.jpg)

![1aa98b042d05453f2b0f059462312176c44b6ee9c1f8e96362cc66bf1501a126.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/tables/1aa98b042d05453f2b0f059462312176c44b6ee9c1f8e96362cc66bf1501a126.jpg)

![516354c88584318b59bc07230235ec12d4cc8accdea6c4ab72f70c87ef7f20be.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/tables/516354c88584318b59bc07230235ec12d4cc8accdea6c4ab72f70c87ef7f20be.jpg)

![5311ed4f91fbe661c03e1b43264a4583d4bd68d83a0b3402a49c3d3770e4bafe.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/tables/5311ed4f91fbe661c03e1b43264a4583d4bd68d83a0b3402a49c3d3770e4bafe.jpg)

![56de2b3d7dc932d8bc53e61be241a07ca085664349cde117bece2bfe12a33382.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/tables/56de2b3d7dc932d8bc53e61be241a07ca085664349cde117bece2bfe12a33382.jpg)

![5d2c4d1e969af57a32252b1115db75a941ec396aa5f0f560a3e35e7309b72f99.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/tables/5d2c4d1e969af57a32252b1115db75a941ec396aa5f0f560a3e35e7309b72f99.jpg)

![88d6ade2000cf8b2059a819559ff177b586d9979a5bfd2b0c6008c237fa636e4.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/tables/88d6ade2000cf8b2059a819559ff177b586d9979a5bfd2b0c6008c237fa636e4.jpg)

![9de6a8aeceddeb48e12957a647fcdcfd804f77d8bd5a9d2678dfd966aa932347.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/tables/9de6a8aeceddeb48e12957a647fcdcfd804f77d8bd5a9d2678dfd966aa932347.jpg)

![ab711b5b624aeb2fae5d86aaf02b1438cff550847f017da0122f31d2eb2e1c90.jpg](../icml_results/5637_NeurIPS_%20Neuro-anatomical%20Inductive%20Priors%20for%20Sphere-based%20Brain%20Decoding/tables/ab711b5b624aeb2fae5d86aaf02b1438cff550847f017da0122f31d2eb2e1c90.jpg)

<a id="paper-5589"></a>

## On the Limits of Test-Time Compute: Sequential Reward Filtering for Better Inference


### Images

![01048e9a99b7718841b4d82f221c1c2ba199be4dd71e9e2f728f8118b2348681.jpg](../icml_results/5638_On%20the%20Limits%20of%20Test-Time%20Compute_%20Sequential%20Reward%20Filtering%20for%20Better%20Inference/images/01048e9a99b7718841b4d82f221c1c2ba199be4dd71e9e2f728f8118b2348681.jpg)

![2e0ff39a71d7adaa5b29f08a289bcf6d96b381c7a4fe0ba1c5bcf7abffbdc9fa.jpg](../icml_results/5638_On%20the%20Limits%20of%20Test-Time%20Compute_%20Sequential%20Reward%20Filtering%20for%20Better%20Inference/images/2e0ff39a71d7adaa5b29f08a289bcf6d96b381c7a4fe0ba1c5bcf7abffbdc9fa.jpg)

![44b94d0fcc26e0ec73d7b2d1ce67f00a2e0bfe47816c78a4e4364724f7670d3f.jpg](../icml_results/5638_On%20the%20Limits%20of%20Test-Time%20Compute_%20Sequential%20Reward%20Filtering%20for%20Better%20Inference/images/44b94d0fcc26e0ec73d7b2d1ce67f00a2e0bfe47816c78a4e4364724f7670d3f.jpg)

![62abb73555f12020cb01a8434a90c879b2bfa6c391c20e77fb7e8d1c56df4cd8.jpg](../icml_results/5638_On%20the%20Limits%20of%20Test-Time%20Compute_%20Sequential%20Reward%20Filtering%20for%20Better%20Inference/images/62abb73555f12020cb01a8434a90c879b2bfa6c391c20e77fb7e8d1c56df4cd8.jpg)

![75447a91cfce1bca91dc9b83da35438658ad87c33568b2a614421cc6cc0d62f7.jpg](../icml_results/5638_On%20the%20Limits%20of%20Test-Time%20Compute_%20Sequential%20Reward%20Filtering%20for%20Better%20Inference/images/75447a91cfce1bca91dc9b83da35438658ad87c33568b2a614421cc6cc0d62f7.jpg)

![a7ce8dab0849c906e2d496fe6363b2b6612f75618422ceda1bbe3d8ee93e80d8.jpg](../icml_results/5638_On%20the%20Limits%20of%20Test-Time%20Compute_%20Sequential%20Reward%20Filtering%20for%20Better%20Inference/images/a7ce8dab0849c906e2d496fe6363b2b6612f75618422ceda1bbe3d8ee93e80d8.jpg)

![be622c656ccf7847eea51f977c14b8b2983558fcc7f70bba2438e2969b968d18.jpg](../icml_results/5638_On%20the%20Limits%20of%20Test-Time%20Compute_%20Sequential%20Reward%20Filtering%20for%20Better%20Inference/images/be622c656ccf7847eea51f977c14b8b2983558fcc7f70bba2438e2969b968d18.jpg)

![d926cfd60942f16d7f6291d84f5e9bdfdb98fa0da3ab906e4d0c29d76b64e500.jpg](../icml_results/5638_On%20the%20Limits%20of%20Test-Time%20Compute_%20Sequential%20Reward%20Filtering%20for%20Better%20Inference/images/d926cfd60942f16d7f6291d84f5e9bdfdb98fa0da3ab906e4d0c29d76b64e500.jpg)

### Tables

![10ef22f450a5f6b68b07925a7f5c3dbf2b4dfb0915c69da30a7df40c4a29cdde.jpg](../icml_results/5638_On%20the%20Limits%20of%20Test-Time%20Compute_%20Sequential%20Reward%20Filtering%20for%20Better%20Inference/tables/10ef22f450a5f6b68b07925a7f5c3dbf2b4dfb0915c69da30a7df40c4a29cdde.jpg)

![6f4fcd1b27812ee76d9a30e8edd110a95ef961e3d72e82e29be1f0de4bcd6b8d.jpg](../icml_results/5638_On%20the%20Limits%20of%20Test-Time%20Compute_%20Sequential%20Reward%20Filtering%20for%20Better%20Inference/tables/6f4fcd1b27812ee76d9a30e8edd110a95ef961e3d72e82e29be1f0de4bcd6b8d.jpg)

![928baf60f00c246b2fed4a31ac3bf04004725c33d18a934b3d6c86673d9e31ad.jpg](../icml_results/5638_On%20the%20Limits%20of%20Test-Time%20Compute_%20Sequential%20Reward%20Filtering%20for%20Better%20Inference/tables/928baf60f00c246b2fed4a31ac3bf04004725c33d18a934b3d6c86673d9e31ad.jpg)

<a id="paper-5590"></a>

## PathwayLLM: Explainable Clinical Trajectory Modeling with Structured Pathways for Sepsis Prediction


### Images

![1c178034614ac68e24dc9c9bf220a9d15c77b3d144e2e306900858596d070a38.jpg](../icml_results/5639_PathwayLLM_%20Explainable%20Clinical%20Trajectory%20Modeling%20with%20Structured%20Pathways%20for%20Sepsis%20Prediction/images/1c178034614ac68e24dc9c9bf220a9d15c77b3d144e2e306900858596d070a38.jpg)

![3e6e4501c0f18d0d020246c98581d6f4e97cc208a50a65b4b4e5803ea5147838.jpg](../icml_results/5639_PathwayLLM_%20Explainable%20Clinical%20Trajectory%20Modeling%20with%20Structured%20Pathways%20for%20Sepsis%20Prediction/images/3e6e4501c0f18d0d020246c98581d6f4e97cc208a50a65b4b4e5803ea5147838.jpg)

![6b7f1970f066503ba03c6ff31530dcdd92b8e9406243aabb0aa24e4ff881b849.jpg](../icml_results/5639_PathwayLLM_%20Explainable%20Clinical%20Trajectory%20Modeling%20with%20Structured%20Pathways%20for%20Sepsis%20Prediction/images/6b7f1970f066503ba03c6ff31530dcdd92b8e9406243aabb0aa24e4ff881b849.jpg)

![7fe3a09613b200d403d6df5d1ba314c5af872c7b5fad252e277778d6b56557db.jpg](../icml_results/5639_PathwayLLM_%20Explainable%20Clinical%20Trajectory%20Modeling%20with%20Structured%20Pathways%20for%20Sepsis%20Prediction/images/7fe3a09613b200d403d6df5d1ba314c5af872c7b5fad252e277778d6b56557db.jpg)

### Tables

![14d4d3faceca873adfc878ab1757618d76a48e859b014dce05c5eb57800cdef3.jpg](../icml_results/5639_PathwayLLM_%20Explainable%20Clinical%20Trajectory%20Modeling%20with%20Structured%20Pathways%20for%20Sepsis%20Prediction/tables/14d4d3faceca873adfc878ab1757618d76a48e859b014dce05c5eb57800cdef3.jpg)

![19510d9c0e8381cfaa66f5665fea56514f2884bc458b82db674e75eec1e95545.jpg](../icml_results/5639_PathwayLLM_%20Explainable%20Clinical%20Trajectory%20Modeling%20with%20Structured%20Pathways%20for%20Sepsis%20Prediction/tables/19510d9c0e8381cfaa66f5665fea56514f2884bc458b82db674e75eec1e95545.jpg)

![39ef8aed374bafe88ec1d6ac4c1dd7da971b09c86a8b4550f151beb7d0b00aa0.jpg](../icml_results/5639_PathwayLLM_%20Explainable%20Clinical%20Trajectory%20Modeling%20with%20Structured%20Pathways%20for%20Sepsis%20Prediction/tables/39ef8aed374bafe88ec1d6ac4c1dd7da971b09c86a8b4550f151beb7d0b00aa0.jpg)

![3e12cb50621a4294e1e3bfab85ae071789b63bfe2e1fc713e38f622e49e76e68.jpg](../icml_results/5639_PathwayLLM_%20Explainable%20Clinical%20Trajectory%20Modeling%20with%20Structured%20Pathways%20for%20Sepsis%20Prediction/tables/3e12cb50621a4294e1e3bfab85ae071789b63bfe2e1fc713e38f622e49e76e68.jpg)

![4170005b694cfface61c764a3e4d5fdeae1e49180f5cd5efcb7401aa68dc371d.jpg](../icml_results/5639_PathwayLLM_%20Explainable%20Clinical%20Trajectory%20Modeling%20with%20Structured%20Pathways%20for%20Sepsis%20Prediction/tables/4170005b694cfface61c764a3e4d5fdeae1e49180f5cd5efcb7401aa68dc371d.jpg)

![4789901053868910f60e4cfbbade0026bbfc6638b5137318984058fa0c781445.jpg](../icml_results/5639_PathwayLLM_%20Explainable%20Clinical%20Trajectory%20Modeling%20with%20Structured%20Pathways%20for%20Sepsis%20Prediction/tables/4789901053868910f60e4cfbbade0026bbfc6638b5137318984058fa0c781445.jpg)

![5287fe3ef66990620b0755aac0f813371c5884e24b2787e917ae0102b0fed8e0.jpg](../icml_results/5639_PathwayLLM_%20Explainable%20Clinical%20Trajectory%20Modeling%20with%20Structured%20Pathways%20for%20Sepsis%20Prediction/tables/5287fe3ef66990620b0755aac0f813371c5884e24b2787e917ae0102b0fed8e0.jpg)

![602cc66ad20f6f2e4d5d9b01365a524663092a4c5483f3a5df3f402db18aeaf7.jpg](../icml_results/5639_PathwayLLM_%20Explainable%20Clinical%20Trajectory%20Modeling%20with%20Structured%20Pathways%20for%20Sepsis%20Prediction/tables/602cc66ad20f6f2e4d5d9b01365a524663092a4c5483f3a5df3f402db18aeaf7.jpg)

![8cc6007d225c25c3ac0726121d967e1ba93d6e8c4318d646425b351382854a49.jpg](../icml_results/5639_PathwayLLM_%20Explainable%20Clinical%20Trajectory%20Modeling%20with%20Structured%20Pathways%20for%20Sepsis%20Prediction/tables/8cc6007d225c25c3ac0726121d967e1ba93d6e8c4318d646425b351382854a49.jpg)

![9a9b92349eeb0fd2abee60d47cca94c258b9304c6d131108c7345cc3d881fcd6.jpg](../icml_results/5639_PathwayLLM_%20Explainable%20Clinical%20Trajectory%20Modeling%20with%20Structured%20Pathways%20for%20Sepsis%20Prediction/tables/9a9b92349eeb0fd2abee60d47cca94c258b9304c6d131108c7345cc3d881fcd6.jpg)

![a4e55b9f69307a73cd8bb83b5acfa731f9fc48bb60b62ea29aaa96f2e39ccc03.jpg](../icml_results/5639_PathwayLLM_%20Explainable%20Clinical%20Trajectory%20Modeling%20with%20Structured%20Pathways%20for%20Sepsis%20Prediction/tables/a4e55b9f69307a73cd8bb83b5acfa731f9fc48bb60b62ea29aaa96f2e39ccc03.jpg)

![af50828be06987841ad2d284d434efaecb77a5ac88ea4223df71dc2c0087d0ae.jpg](../icml_results/5639_PathwayLLM_%20Explainable%20Clinical%20Trajectory%20Modeling%20with%20Structured%20Pathways%20for%20Sepsis%20Prediction/tables/af50828be06987841ad2d284d434efaecb77a5ac88ea4223df71dc2c0087d0ae.jpg)

![e869d9c32bd9ed26891d6aae089c7c76157c59e6956465710d1dc7ec6cbf7cb8.jpg](../icml_results/5639_PathwayLLM_%20Explainable%20Clinical%20Trajectory%20Modeling%20with%20Structured%20Pathways%20for%20Sepsis%20Prediction/tables/e869d9c32bd9ed26891d6aae089c7c76157c59e6956465710d1dc7ec6cbf7cb8.jpg)

<a id="paper-5591"></a>

## Deep Pre-Alignment for VLMs


### Images

![07ff881942d54d0ec69bb3df9a86373ca4da5326d8f1944f5c552702ac79a0ab.jpg](../icml_results/5640_Deep%20Pre-Alignment%20for%20VLMs/images/07ff881942d54d0ec69bb3df9a86373ca4da5326d8f1944f5c552702ac79a0ab.jpg)

![20057da5d278abc7f5052e0201e7c3336863ec21bc2f0d6164148f52e66fb5e2.jpg](../icml_results/5640_Deep%20Pre-Alignment%20for%20VLMs/images/20057da5d278abc7f5052e0201e7c3336863ec21bc2f0d6164148f52e66fb5e2.jpg)

![7673b5a60d6fee2bb90f5790fde8f11bdd8ce31c1f2f1899fa5ffc69b80a2510.jpg](../icml_results/5640_Deep%20Pre-Alignment%20for%20VLMs/images/7673b5a60d6fee2bb90f5790fde8f11bdd8ce31c1f2f1899fa5ffc69b80a2510.jpg)

![b4f9bc3480a524347c9262dc78836901725e3ea67a900b697bfde773f6009155.jpg](../icml_results/5640_Deep%20Pre-Alignment%20for%20VLMs/images/b4f9bc3480a524347c9262dc78836901725e3ea67a900b697bfde773f6009155.jpg)

![c19ec448a57175fd617b6127027793c469b8fb40c988baaa1ef8248c1cf51d1b.jpg](../icml_results/5640_Deep%20Pre-Alignment%20for%20VLMs/images/c19ec448a57175fd617b6127027793c469b8fb40c988baaa1ef8248c1cf51d1b.jpg)

![e78c2121f0ca1832cce75d908b1de604e731ab5f7bec10c56d400151ecf1231e.jpg](../icml_results/5640_Deep%20Pre-Alignment%20for%20VLMs/images/e78c2121f0ca1832cce75d908b1de604e731ab5f7bec10c56d400151ecf1231e.jpg)

### Tables

![3167307ad6595a50f4d31ac3e04f7506a68580384cf19384c99f8d0269330c3c.jpg](../icml_results/5640_Deep%20Pre-Alignment%20for%20VLMs/tables/3167307ad6595a50f4d31ac3e04f7506a68580384cf19384c99f8d0269330c3c.jpg)

![56a5f36eb4b03087ef128232888afe65665c9232f3d4e47e08abf50a4d4ac572.jpg](../icml_results/5640_Deep%20Pre-Alignment%20for%20VLMs/tables/56a5f36eb4b03087ef128232888afe65665c9232f3d4e47e08abf50a4d4ac572.jpg)

![6619da5aa369c2c781c17ee41bdb57635ed6115fd835a380b572e49924cadd88.jpg](../icml_results/5640_Deep%20Pre-Alignment%20for%20VLMs/tables/6619da5aa369c2c781c17ee41bdb57635ed6115fd835a380b572e49924cadd88.jpg)

![76f0dba4a4613eab650133ee6cf5ea4d4e9796d257a99f088c373c905967739c.jpg](../icml_results/5640_Deep%20Pre-Alignment%20for%20VLMs/tables/76f0dba4a4613eab650133ee6cf5ea4d4e9796d257a99f088c373c905967739c.jpg)

![851a5b9b0b9481e624a3462e909d3462a5248dd9def51025e7522b21b0f3627a.jpg](../icml_results/5640_Deep%20Pre-Alignment%20for%20VLMs/tables/851a5b9b0b9481e624a3462e909d3462a5248dd9def51025e7522b21b0f3627a.jpg)

![86a64eed5485ad9f753300a37d5565404ed04e6bd4d7fcd04a933df10a14a857.jpg](../icml_results/5640_Deep%20Pre-Alignment%20for%20VLMs/tables/86a64eed5485ad9f753300a37d5565404ed04e6bd4d7fcd04a933df10a14a857.jpg)

![943d26103b76c1938873a4038549e475152117551fa23c15c2792324ccb2f59d.jpg](../icml_results/5640_Deep%20Pre-Alignment%20for%20VLMs/tables/943d26103b76c1938873a4038549e475152117551fa23c15c2792324ccb2f59d.jpg)

![b15b7fa26e0ef2287a85911e0c33aa97b561b6357546075446fd7b78f25e34bb.jpg](../icml_results/5640_Deep%20Pre-Alignment%20for%20VLMs/tables/b15b7fa26e0ef2287a85911e0c33aa97b561b6357546075446fd7b78f25e34bb.jpg)

![c209ecc4f354ab6606c00759cbca76828959a9364008de118b3773d655505714.jpg](../icml_results/5640_Deep%20Pre-Alignment%20for%20VLMs/tables/c209ecc4f354ab6606c00759cbca76828959a9364008de118b3773d655505714.jpg)

![d58a4037da1d54f77f39f5323286809ec2a4e23450e4cc2db6990cb8e847b11f.jpg](../icml_results/5640_Deep%20Pre-Alignment%20for%20VLMs/tables/d58a4037da1d54f77f39f5323286809ec2a4e23450e4cc2db6990cb8e847b11f.jpg)

![e62cca99bc28393c99c48450f26220ba575b95e8c64fbac86167486ff10a7c1f.jpg](../icml_results/5640_Deep%20Pre-Alignment%20for%20VLMs/tables/e62cca99bc28393c99c48450f26220ba575b95e8c64fbac86167486ff10a7c1f.jpg)

<a id="paper-5592"></a>

## SafeSeek: Universal Attribution of Safety Circuits in Language Models


### Images

![1467502590f40e904b76d275a4a8658d4c443c3c4f173b758dd34a1d50c213d7.jpg](../icml_results/5641_SafeSeek_%20Universal%20Attribution%20of%20Safety%20Circuits%20in%20Language%20Models/images/1467502590f40e904b76d275a4a8658d4c443c3c4f173b758dd34a1d50c213d7.jpg)

![3266f892aa2bbca3330ecab5060a068960ff0c7cfa5e63b83e5fd4593d8570a4.jpg](../icml_results/5641_SafeSeek_%20Universal%20Attribution%20of%20Safety%20Circuits%20in%20Language%20Models/images/3266f892aa2bbca3330ecab5060a068960ff0c7cfa5e63b83e5fd4593d8570a4.jpg)

![75c6d772734a5523ad23fcd03381c38abef206ecbafe92307e7b6c36afc19734.jpg](../icml_results/5641_SafeSeek_%20Universal%20Attribution%20of%20Safety%20Circuits%20in%20Language%20Models/images/75c6d772734a5523ad23fcd03381c38abef206ecbafe92307e7b6c36afc19734.jpg)

![7c099318a7517f0698a8a9e3a649fd642d04c3e4daf7dd597bf676c3c6c73fd8.jpg](../icml_results/5641_SafeSeek_%20Universal%20Attribution%20of%20Safety%20Circuits%20in%20Language%20Models/images/7c099318a7517f0698a8a9e3a649fd642d04c3e4daf7dd597bf676c3c6c73fd8.jpg)

![ca5a6ba454874c846ebf9842da4ae09ac5aeae73afba835c4d207011844ef879.jpg](../icml_results/5641_SafeSeek_%20Universal%20Attribution%20of%20Safety%20Circuits%20in%20Language%20Models/images/ca5a6ba454874c846ebf9842da4ae09ac5aeae73afba835c4d207011844ef879.jpg)

![e2dc4bd238a83d1cf788c5efa2086d44c943d6f19af2c470785b56beef6f816f.jpg](../icml_results/5641_SafeSeek_%20Universal%20Attribution%20of%20Safety%20Circuits%20in%20Language%20Models/images/e2dc4bd238a83d1cf788c5efa2086d44c943d6f19af2c470785b56beef6f816f.jpg)

![ef440ce9216f2050774b80c1dd3938a5a666643cae2059c9583ff38e35cc0f71.jpg](../icml_results/5641_SafeSeek_%20Universal%20Attribution%20of%20Safety%20Circuits%20in%20Language%20Models/images/ef440ce9216f2050774b80c1dd3938a5a666643cae2059c9583ff38e35cc0f71.jpg)

### Tables

![01d429ddd3ba4a69dfda4964f8b15d50f7ce4a2b8f8944c142fc5a2359d6b2b1.jpg](../icml_results/5641_SafeSeek_%20Universal%20Attribution%20of%20Safety%20Circuits%20in%20Language%20Models/tables/01d429ddd3ba4a69dfda4964f8b15d50f7ce4a2b8f8944c142fc5a2359d6b2b1.jpg)

![11d638952f1efe5c165e595535481e44576b6bf46bee13d4073851e7e4401963.jpg](../icml_results/5641_SafeSeek_%20Universal%20Attribution%20of%20Safety%20Circuits%20in%20Language%20Models/tables/11d638952f1efe5c165e595535481e44576b6bf46bee13d4073851e7e4401963.jpg)

![192fa16caed40f723220a71ac1ac3c81fd12ea98681cfda1feb034e0e52b910f.jpg](../icml_results/5641_SafeSeek_%20Universal%20Attribution%20of%20Safety%20Circuits%20in%20Language%20Models/tables/192fa16caed40f723220a71ac1ac3c81fd12ea98681cfda1feb034e0e52b910f.jpg)

![1cd7862229465e44c18601c48600efff106e56a74ab0c7bd6146bdc7db3a1fb8.jpg](../icml_results/5641_SafeSeek_%20Universal%20Attribution%20of%20Safety%20Circuits%20in%20Language%20Models/tables/1cd7862229465e44c18601c48600efff106e56a74ab0c7bd6146bdc7db3a1fb8.jpg)

![1f433f94f2c02a26cdb6ffd355a4717ac722e1da3663c8291792ca53c15f3f69.jpg](../icml_results/5641_SafeSeek_%20Universal%20Attribution%20of%20Safety%20Circuits%20in%20Language%20Models/tables/1f433f94f2c02a26cdb6ffd355a4717ac722e1da3663c8291792ca53c15f3f69.jpg)

![3a280bbd2373b7201aafaab5e68d9870df9385309ff7495f6d836b8c4ec42022.jpg](../icml_results/5641_SafeSeek_%20Universal%20Attribution%20of%20Safety%20Circuits%20in%20Language%20Models/tables/3a280bbd2373b7201aafaab5e68d9870df9385309ff7495f6d836b8c4ec42022.jpg)

![903fa6f2e91496751ec9af7d94739276afa50a56a6c06145846fe7e1dc9ac8ef.jpg](../icml_results/5641_SafeSeek_%20Universal%20Attribution%20of%20Safety%20Circuits%20in%20Language%20Models/tables/903fa6f2e91496751ec9af7d94739276afa50a56a6c06145846fe7e1dc9ac8ef.jpg)

![973362ee93b14957fec704c11e4c5635d99bc345f2b46643005dc2f6903938f2.jpg](../icml_results/5641_SafeSeek_%20Universal%20Attribution%20of%20Safety%20Circuits%20in%20Language%20Models/tables/973362ee93b14957fec704c11e4c5635d99bc345f2b46643005dc2f6903938f2.jpg)

![db6d5aee7d6ff36601b9cf01d57ac7423f1fabe06f8267cb959220b5fdb5f9b2.jpg](../icml_results/5641_SafeSeek_%20Universal%20Attribution%20of%20Safety%20Circuits%20in%20Language%20Models/tables/db6d5aee7d6ff36601b9cf01d57ac7423f1fabe06f8267cb959220b5fdb5f9b2.jpg)

<a id="paper-5593"></a>

## Reliable Confidence Alignment for Generalized Category Discovery


### Images

![1b4ca1f7a89cba7ec22c2ba4a6d9c03a0d8a051a21ca97c5815ff5c95a8c1cc2.jpg](../icml_results/5642_Reliable%20Confidence%20Alignment%20for%20Generalized%20Category%20Discovery/images/1b4ca1f7a89cba7ec22c2ba4a6d9c03a0d8a051a21ca97c5815ff5c95a8c1cc2.jpg)

![27c09f2f990a96e635fecd1b5f5b1ca244d285638f0e45ea6fa52bea0a9db042.jpg](../icml_results/5642_Reliable%20Confidence%20Alignment%20for%20Generalized%20Category%20Discovery/images/27c09f2f990a96e635fecd1b5f5b1ca244d285638f0e45ea6fa52bea0a9db042.jpg)

![2ca929c9efbe707bfd4fa356dd76db1089551d63af245330761dc5a0bf65a09a.jpg](../icml_results/5642_Reliable%20Confidence%20Alignment%20for%20Generalized%20Category%20Discovery/images/2ca929c9efbe707bfd4fa356dd76db1089551d63af245330761dc5a0bf65a09a.jpg)

![335a0fae0c05a29949e3afeae9c87c05735c46900e23dad1cc22005c058931e4.jpg](../icml_results/5642_Reliable%20Confidence%20Alignment%20for%20Generalized%20Category%20Discovery/images/335a0fae0c05a29949e3afeae9c87c05735c46900e23dad1cc22005c058931e4.jpg)

![6303fd0b8f95385e7efed89e12ec454210ec0e78583c737d19e3e22bd88ea6c9.jpg](../icml_results/5642_Reliable%20Confidence%20Alignment%20for%20Generalized%20Category%20Discovery/images/6303fd0b8f95385e7efed89e12ec454210ec0e78583c737d19e3e22bd88ea6c9.jpg)

![81096861ee0bfb135fd6872756396f89316d473d884463bbbd540740ff5bdd3a.jpg](../icml_results/5642_Reliable%20Confidence%20Alignment%20for%20Generalized%20Category%20Discovery/images/81096861ee0bfb135fd6872756396f89316d473d884463bbbd540740ff5bdd3a.jpg)

### Tables

![107a98ab116a16476c63c0bd943f4a20bbc32de48a36cceffe7d56619bda04f9.jpg](../icml_results/5642_Reliable%20Confidence%20Alignment%20for%20Generalized%20Category%20Discovery/tables/107a98ab116a16476c63c0bd943f4a20bbc32de48a36cceffe7d56619bda04f9.jpg)

![15642fba77dacee40d56f7687f7397951306d38f1e8db3b1efd1748937694ab1.jpg](../icml_results/5642_Reliable%20Confidence%20Alignment%20for%20Generalized%20Category%20Discovery/tables/15642fba77dacee40d56f7687f7397951306d38f1e8db3b1efd1748937694ab1.jpg)

![a6f469baa05072ab7a3d90ff6b9e6310495eb9b3c7cf243cdc43ec9729e616ca.jpg](../icml_results/5642_Reliable%20Confidence%20Alignment%20for%20Generalized%20Category%20Discovery/tables/a6f469baa05072ab7a3d90ff6b9e6310495eb9b3c7cf243cdc43ec9729e616ca.jpg)

![cb8df21898e614cee00da0ec0cb9087872fdaafd3647b217bd1274a1942bb177.jpg](../icml_results/5642_Reliable%20Confidence%20Alignment%20for%20Generalized%20Category%20Discovery/tables/cb8df21898e614cee00da0ec0cb9087872fdaafd3647b217bd1274a1942bb177.jpg)

![e165945b45978767363ab3d1c2fd35d4d9b03a1eca9b0a8e9a035c215c8c1878.jpg](../icml_results/5642_Reliable%20Confidence%20Alignment%20for%20Generalized%20Category%20Discovery/tables/e165945b45978767363ab3d1c2fd35d4d9b03a1eca9b0a8e9a035c215c8c1878.jpg)

![e39ea81162be1b8f7a702e0d61ae8fa7f1fc94ea9933b897bf794f1f35f3dde8.jpg](../icml_results/5642_Reliable%20Confidence%20Alignment%20for%20Generalized%20Category%20Discovery/tables/e39ea81162be1b8f7a702e0d61ae8fa7f1fc94ea9933b897bf794f1f35f3dde8.jpg)

<a id="paper-5594"></a>

## TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models


### Images

![0053273d5d2ac47ba2aa4922e9d964f14df89af1fba5612a7d9ff3e106d26a8c.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/0053273d5d2ac47ba2aa4922e9d964f14df89af1fba5612a7d9ff3e106d26a8c.jpg)

![185fc03d3168bd67e2eb2d4e3f99625e1e5e3474f66fdb71bf8fef87a8b61c83.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/185fc03d3168bd67e2eb2d4e3f99625e1e5e3474f66fdb71bf8fef87a8b61c83.jpg)

![1e41a6e01f0ef306e7806bee5e54a481af848075af26e7b039f38d8c47c6887f.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/1e41a6e01f0ef306e7806bee5e54a481af848075af26e7b039f38d8c47c6887f.jpg)

![26b7159460682aea523bc1558e12840c76317b3de3232c66fd55abeaf553f0f8.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/26b7159460682aea523bc1558e12840c76317b3de3232c66fd55abeaf553f0f8.jpg)

![27208316b0fd141f731e55461a8a67d6198e2661b5a1378468dafeb823a4a45e.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/27208316b0fd141f731e55461a8a67d6198e2661b5a1378468dafeb823a4a45e.jpg)

![29207413e85aa985902b4fddd126d51ead9dc6984afa02f4c25b82e374665d35.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/29207413e85aa985902b4fddd126d51ead9dc6984afa02f4c25b82e374665d35.jpg)

![29f5e4073ba65e0d4b857278af0165273537b627e97998d9d2584545dbf1c530.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/29f5e4073ba65e0d4b857278af0165273537b627e97998d9d2584545dbf1c530.jpg)

![2bd0641fd19491d2f43ef92b0de27339ca3f9e9d7bcca0e695cf3adc9f77ca17.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/2bd0641fd19491d2f43ef92b0de27339ca3f9e9d7bcca0e695cf3adc9f77ca17.jpg)

![373a6901599eba1e8cf3031e93963f8d608aff08cccfefa58eea25ce6e72b36f.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/373a6901599eba1e8cf3031e93963f8d608aff08cccfefa58eea25ce6e72b36f.jpg)

![4c8f13067a61bcd8c4776308ac63eddb2c872fb1f9a24e040d65200b38933f62.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/4c8f13067a61bcd8c4776308ac63eddb2c872fb1f9a24e040d65200b38933f62.jpg)

![5a33ee919697d92bdd4d0c491fcb1e6442851876035372557734e61baf096d07.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/5a33ee919697d92bdd4d0c491fcb1e6442851876035372557734e61baf096d07.jpg)

![74cae90082e643a9822cb00d4c2fd3a3792305fcf47f25d7188101f290027bb5.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/74cae90082e643a9822cb00d4c2fd3a3792305fcf47f25d7188101f290027bb5.jpg)

![7d5d1b14cf1720179364e4e6431792c6257e112f3528e7898b9b3f05318b85c9.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/7d5d1b14cf1720179364e4e6431792c6257e112f3528e7898b9b3f05318b85c9.jpg)

![84edeb5c303cb6fa1564817a6364973840c07d7b0776efdc3be12e9d10b15f42.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/84edeb5c303cb6fa1564817a6364973840c07d7b0776efdc3be12e9d10b15f42.jpg)

![9694e4001931a1b0543aca39c4acc86e002b30104f4978be99a2546853fe4210.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/9694e4001931a1b0543aca39c4acc86e002b30104f4978be99a2546853fe4210.jpg)

![a1644e70b5002c68ad060165ba1cb5d34133dc45a5472423cd4a97648236fae5.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/a1644e70b5002c68ad060165ba1cb5d34133dc45a5472423cd4a97648236fae5.jpg)

![a167f3f2d9e2e822594910ac656011eb6efefb13d171743a4edc27d06f376da6.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/a167f3f2d9e2e822594910ac656011eb6efefb13d171743a4edc27d06f376da6.jpg)

![a430cad76fe49c4c52f122e6c52f8b04fb8d9c15fb496f5154b74350b78ef631.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/a430cad76fe49c4c52f122e6c52f8b04fb8d9c15fb496f5154b74350b78ef631.jpg)

![a462e213b4654874e40f2fec8d449c983dabfafcbd38456d81be292f2567f733.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/a462e213b4654874e40f2fec8d449c983dabfafcbd38456d81be292f2567f733.jpg)

![b5cc35c082059c69cbc420b0ef2f16fe4f79dd3754603d94188bb4c6f243a61a.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/b5cc35c082059c69cbc420b0ef2f16fe4f79dd3754603d94188bb4c6f243a61a.jpg)

![b869910c9efbb17250b36c2ad86e7884ca681a754600321c7fadd2d5eca0b1cb.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/b869910c9efbb17250b36c2ad86e7884ca681a754600321c7fadd2d5eca0b1cb.jpg)

![bbfce269eb12ac33f2ce4b09285f7de5149fdd29bf4706d3e93930ecda66518c.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/bbfce269eb12ac33f2ce4b09285f7de5149fdd29bf4706d3e93930ecda66518c.jpg)

![c43bec894b9d24a29a31479d115b8ca3b2161b6466a4eb03116fd34cb4ab408d.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/c43bec894b9d24a29a31479d115b8ca3b2161b6466a4eb03116fd34cb4ab408d.jpg)

![c614af5b22b2a8f1fa83c846e914eeb46f21144fabeaff8b7b489ad88153d88b.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/c614af5b22b2a8f1fa83c846e914eeb46f21144fabeaff8b7b489ad88153d88b.jpg)

![ccd218e1f338fa9ac32e407666af6404ae63acd0def7227667254e5fc48f1f30.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/ccd218e1f338fa9ac32e407666af6404ae63acd0def7227667254e5fc48f1f30.jpg)

![cdfb1717b0b217feec9461ddd118a21c2038ab3f6ace1acf45912bb45e7fa004.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/cdfb1717b0b217feec9461ddd118a21c2038ab3f6ace1acf45912bb45e7fa004.jpg)

![d454ac15443067851b3c44d714e11ca85865b86e8279bddc54d2627b5deeea92.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/d454ac15443067851b3c44d714e11ca85865b86e8279bddc54d2627b5deeea92.jpg)

![d462f87fd83474b8abba902aa9ff742a92cdefdb1981e3e76d31f46c8c1202b8.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/d462f87fd83474b8abba902aa9ff742a92cdefdb1981e3e76d31f46c8c1202b8.jpg)

![f7c5042091824123eb281df81719e05f6b3728f4be5d6faeea98eeab3d027c8a.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/f7c5042091824123eb281df81719e05f6b3728f4be5d6faeea98eeab3d027c8a.jpg)

![fd42f50f21d7121fbf6c82c4965e9977acb47f09cfe1c683659070442dd6a22c.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/fd42f50f21d7121fbf6c82c4965e9977acb47f09cfe1c683659070442dd6a22c.jpg)

![fe3af6fadbe688991311bda72808c654e47f53b2192077165d837dc88dc3ec61.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/images/fe3af6fadbe688991311bda72808c654e47f53b2192077165d837dc88dc3ec61.jpg)

### Tables

![20312f8166f59f50960adba851f94d90e3dce3c484e27db8f2b632ca96c87b0a.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/tables/20312f8166f59f50960adba851f94d90e3dce3c484e27db8f2b632ca96c87b0a.jpg)

![25028c9768a01857918214eee10dd6a5580f495f65596cfb4d5eee5b21241f32.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/tables/25028c9768a01857918214eee10dd6a5580f495f65596cfb4d5eee5b21241f32.jpg)

![267ff344066d99d6acd54cd146d476850da0e9f4d69d638b1d217aef4dc5e3d0.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/tables/267ff344066d99d6acd54cd146d476850da0e9f4d69d638b1d217aef4dc5e3d0.jpg)

![2db230a82494b041f4074cfc0046390797861a0eb0b87c547afadd8a42d0bbdf.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/tables/2db230a82494b041f4074cfc0046390797861a0eb0b87c547afadd8a42d0bbdf.jpg)

![5f46a7800a9343bce1745a93b0a43543b994680c16502225a7dcd014d1a66b9b.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/tables/5f46a7800a9343bce1745a93b0a43543b994680c16502225a7dcd014d1a66b9b.jpg)

![793e14accecd5cd9934b7e7ee76b556314feee87dd57a9f844c162a0eabc82af.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/tables/793e14accecd5cd9934b7e7ee76b556314feee87dd57a9f844c162a0eabc82af.jpg)

![95965050d0a4f1a1b1678926a24090200c7dc0d4bacdb0ac87ee2527bf840942.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/tables/95965050d0a4f1a1b1678926a24090200c7dc0d4bacdb0ac87ee2527bf840942.jpg)

![9e625daa64ed219dc7ea1ac9447c693ffe14eafeb8eac90bc4c846c3d4c07f69.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/tables/9e625daa64ed219dc7ea1ac9447c693ffe14eafeb8eac90bc4c846c3d4c07f69.jpg)

![9f086e70a883af5a54638c110dbc50a9ded83eb71a2d89888d033e459388ef53.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/tables/9f086e70a883af5a54638c110dbc50a9ded83eb71a2d89888d033e459388ef53.jpg)

![d58de8866088a3b6a705b049e3899bcc899503748a179f9385da706d0e249b58.jpg](../icml_results/5643_TSRBench_%20A%20Comprehensive%20Multi-task%20Multi-modal%20Time%20Series%20Reasoning%20Benchmark%20for%20Generalist%20Mode/tables/d58de8866088a3b6a705b049e3899bcc899503748a179f9385da706d0e249b58.jpg)

<a id="paper-5595"></a>

## Robust Inter-Series Dependency Modeling for Time Series Forecasting via Information-Theoretic Alignment


### Images

![1017038595c5994251edfa96a9cbfeca9a5e116560ec5f8c8c072346c5f6f01b.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/1017038595c5994251edfa96a9cbfeca9a5e116560ec5f8c8c072346c5f6f01b.jpg)

![11ca7702a09e4612416e02ef2f92274cdd065f8c57ac7c28e42cdd07459b4e5f.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/11ca7702a09e4612416e02ef2f92274cdd065f8c57ac7c28e42cdd07459b4e5f.jpg)

![1de5b576064d90b0a7f2e47c21756963232e93f73c8cf988ff0d9cf2f17495c3.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/1de5b576064d90b0a7f2e47c21756963232e93f73c8cf988ff0d9cf2f17495c3.jpg)

![27ad858445a68e733d87339b9053dda9a9a1744daf4229e92eb4783211c952bd.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/27ad858445a68e733d87339b9053dda9a9a1744daf4229e92eb4783211c952bd.jpg)

![2f69a8aaf8dfc2d5920eee398aaf84d5991c5f7bde0db8e8885451c490795be5.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/2f69a8aaf8dfc2d5920eee398aaf84d5991c5f7bde0db8e8885451c490795be5.jpg)

![384589c33f25d7c2d76ed5ee4be5917f205189e19ea1fb965f6cac68ebf13edf.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/384589c33f25d7c2d76ed5ee4be5917f205189e19ea1fb965f6cac68ebf13edf.jpg)

![4268763e94e01cc98e1ee9bd6ebd7004b78414a60cef374621e579b280cce038.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/4268763e94e01cc98e1ee9bd6ebd7004b78414a60cef374621e579b280cce038.jpg)

![4b33f179d206076e7a1b4fa749505b387f435208584f68d96f3e88792f30f080.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/4b33f179d206076e7a1b4fa749505b387f435208584f68d96f3e88792f30f080.jpg)

![4ff2dc1d136403560a4f2b104f51251099abdbc6831a734e833e6e650031d498.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/4ff2dc1d136403560a4f2b104f51251099abdbc6831a734e833e6e650031d498.jpg)

![654dd8050ce6635995c233f8c314f3dc15199364d568b8735330ed5dfb8dd2e9.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/654dd8050ce6635995c233f8c314f3dc15199364d568b8735330ed5dfb8dd2e9.jpg)

![6a4fc8021b3261b2beeb49026ee621d60e33fb93975fe00e18a4ca4d8a0d8ecb.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/6a4fc8021b3261b2beeb49026ee621d60e33fb93975fe00e18a4ca4d8a0d8ecb.jpg)

![6e95f3e287414b9fcab197f249060943598471c128ed21a477679a1df7fa7287.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/6e95f3e287414b9fcab197f249060943598471c128ed21a477679a1df7fa7287.jpg)

![7215123f533ed6fb27fff0c92e12818b79155d00bda582f93ca60354829c0812.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/7215123f533ed6fb27fff0c92e12818b79155d00bda582f93ca60354829c0812.jpg)

![855a0150f855ddfe195a377b95f31877d83af1d631fa628827947abb87aad944.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/855a0150f855ddfe195a377b95f31877d83af1d631fa628827947abb87aad944.jpg)

![8c6d079870f687a478918752e6b303e969bb7de680dde7488b773da4f2edd742.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/8c6d079870f687a478918752e6b303e969bb7de680dde7488b773da4f2edd742.jpg)

![8d7d8289ee23a3dc781ef4548e12537b1717ec6f23ddd7486abc9ad196c3a1e0.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/8d7d8289ee23a3dc781ef4548e12537b1717ec6f23ddd7486abc9ad196c3a1e0.jpg)

![97403d9b4a166fb2d34de1facba09df1c2b850c24bac64a92d29bdd0959fc605.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/97403d9b4a166fb2d34de1facba09df1c2b850c24bac64a92d29bdd0959fc605.jpg)

![a0623b2b896d7e2caf50b02c60e4e4ee06dd60d7d48458763810e23e8370c3fa.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/a0623b2b896d7e2caf50b02c60e4e4ee06dd60d7d48458763810e23e8370c3fa.jpg)

![aa1f536a1f2c82187396b2dc889920b4ead3a3db65699ac742d31d4114807b07.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/aa1f536a1f2c82187396b2dc889920b4ead3a3db65699ac742d31d4114807b07.jpg)

![bf835a8107ea8eaaf1c9bd03d4482494b4e6438b0b52a099dd528f2d1651a55a.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/bf835a8107ea8eaaf1c9bd03d4482494b4e6438b0b52a099dd528f2d1651a55a.jpg)

![d0e837964af5be7aa56c4f4e0e8a456dcbf5d8ec3150fb8016c25aeeff306e45.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/d0e837964af5be7aa56c4f4e0e8a456dcbf5d8ec3150fb8016c25aeeff306e45.jpg)

![e05bda6dcce900d45fe96907faca1a47e2a90fd208665e49fc8f4ce28b158654.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/e05bda6dcce900d45fe96907faca1a47e2a90fd208665e49fc8f4ce28b158654.jpg)

![e347f62c888f1e8df5afa95695b0ba71e9ca7fb3fa2b47e6830f169466f88c0f.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/e347f62c888f1e8df5afa95695b0ba71e9ca7fb3fa2b47e6830f169466f88c0f.jpg)

![e94dfefee105bdc743eb4d9befe5aad3e51c94b4b36a67c139be081ea74c68d1.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/e94dfefee105bdc743eb4d9befe5aad3e51c94b4b36a67c139be081ea74c68d1.jpg)

![ed8e0b0bfe6d6cacf2e5cd2feae7352456975d13eda816a910c5c85e53fe4153.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/ed8e0b0bfe6d6cacf2e5cd2feae7352456975d13eda816a910c5c85e53fe4153.jpg)

![efb97af63f0aac08510aa3de0d57047bad60e52f346c46ed2e1a88f0e56a3374.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/efb97af63f0aac08510aa3de0d57047bad60e52f346c46ed2e1a88f0e56a3374.jpg)

![f35a300b06cfee5329f3fb765d86c5ed510c9175b7b0be455658de33d44c813e.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/f35a300b06cfee5329f3fb765d86c5ed510c9175b7b0be455658de33d44c813e.jpg)

![fd5abb0742bbfcf2f600df9879814b113df202985a8ec9f5066d907ed3210983.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/images/fd5abb0742bbfcf2f600df9879814b113df202985a8ec9f5066d907ed3210983.jpg)

### Tables

![0079c22dad773c1b62f645e8e617469eddb3ddcd13d7a7b78edd985a97ddc8ac.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/tables/0079c22dad773c1b62f645e8e617469eddb3ddcd13d7a7b78edd985a97ddc8ac.jpg)

![03058a4708a4271651a6055e34f8e9cace62f61b37812eae7e90b84a6090e6d4.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/tables/03058a4708a4271651a6055e34f8e9cace62f61b37812eae7e90b84a6090e6d4.jpg)

![0b1efe82dc89b17a3a64c45900cd20027bd3066d1745e1e2fb955a9dbd621ec7.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/tables/0b1efe82dc89b17a3a64c45900cd20027bd3066d1745e1e2fb955a9dbd621ec7.jpg)

![192e8ca32841c96c8fe37dd1888691cba778d06ac35bd9a5f34f17fad651f933.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/tables/192e8ca32841c96c8fe37dd1888691cba778d06ac35bd9a5f34f17fad651f933.jpg)

![2266d0af346425707fdcbd32000e2ad9a59e49d62c5c42e84c7c6c5c8b6786f5.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/tables/2266d0af346425707fdcbd32000e2ad9a59e49d62c5c42e84c7c6c5c8b6786f5.jpg)

![469f639a9be92d5d21cfd659b5690af25aa25122c2f37ab0758a2172feb0deb8.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/tables/469f639a9be92d5d21cfd659b5690af25aa25122c2f37ab0758a2172feb0deb8.jpg)

![4ecce3b06981650763cc92bcbe36b9f807d630eca6f460696d9e887548aaf170.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/tables/4ecce3b06981650763cc92bcbe36b9f807d630eca6f460696d9e887548aaf170.jpg)

![51a0d73303ad16f1a6b53bc7f4e8c77a73a6ca5a6bb4ae4f159242de73b5d480.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/tables/51a0d73303ad16f1a6b53bc7f4e8c77a73a6ca5a6bb4ae4f159242de73b5d480.jpg)

![68be7e2fc8f5df221242b9d96354fd16343e28b737d9240b2d0b89b733165652.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/tables/68be7e2fc8f5df221242b9d96354fd16343e28b737d9240b2d0b89b733165652.jpg)

![80155b62cf89a1ed19c66de4aa73da08515278b6a3b911ba6cdc6c1e0e2a4071.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/tables/80155b62cf89a1ed19c66de4aa73da08515278b6a3b911ba6cdc6c1e0e2a4071.jpg)

![8654a6c1d996d5832e4567455412ed720813fd0387f355ef176b0b701ec99dbd.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/tables/8654a6c1d996d5832e4567455412ed720813fd0387f355ef176b0b701ec99dbd.jpg)

![94b05c8eabb64f301495942d5075bcc195ae4be0c307c7687435f827ac47d69d.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/tables/94b05c8eabb64f301495942d5075bcc195ae4be0c307c7687435f827ac47d69d.jpg)

![96f57c9e30dcc014cbb3f05e158fe6f0d4094a6e1f13bfad0edd4d1c3a41e851.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/tables/96f57c9e30dcc014cbb3f05e158fe6f0d4094a6e1f13bfad0edd4d1c3a41e851.jpg)

![e507ec4090fcaeb9e4af6adbbe818efebb41fa6b42152ba3ba5115bce0b96248.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/tables/e507ec4090fcaeb9e4af6adbbe818efebb41fa6b42152ba3ba5115bce0b96248.jpg)

![f7c5487cf83f1b9de9a65765187cda87a797375a14ac237ab9ab4008873b6888.jpg](../icml_results/5644_Robust%20Inter-Series%20Dependency%20Modeling%20for%20Time%20Series%20Forecasting%20via%20Information-Theoretic%20Alignm/tables/f7c5487cf83f1b9de9a65765187cda87a797375a14ac237ab9ab4008873b6888.jpg)

<a id="paper-5596"></a>

## CARD: Coarse-to-fine Autoregressive Modeling with Radix-based Decomposition for Transferable Free Energy Estimation


### Images

![13738224986f6adc82ae9474d9084e6b8138fb4ea3c21f2bcc3d4cdc82793409.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/images/13738224986f6adc82ae9474d9084e6b8138fb4ea3c21f2bcc3d4cdc82793409.jpg)

![39ae57e420fd5e1e3cd4519b5ddf9c2ae7271cf604275146b1fe27811dc50b7a.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/images/39ae57e420fd5e1e3cd4519b5ddf9c2ae7271cf604275146b1fe27811dc50b7a.jpg)

![6c89d1f2fb548114d5b24e3741c7cb54dfc5d3f7f48ccc7d2f4760d2888b3308.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/images/6c89d1f2fb548114d5b24e3741c7cb54dfc5d3f7f48ccc7d2f4760d2888b3308.jpg)

![903a09b51074bd2e96adc5a59c18d4d88bb7cb54aca602ac208a0e1e2ba587c3.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/images/903a09b51074bd2e96adc5a59c18d4d88bb7cb54aca602ac208a0e1e2ba587c3.jpg)

![9e307fc763985c8e5fa3619d1517c9053916751ca616ae5c914b1420a8df1e58.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/images/9e307fc763985c8e5fa3619d1517c9053916751ca616ae5c914b1420a8df1e58.jpg)

![ab399dbf9934c205d3779b516a1ab0aa30546be437c23642f0aefac8fee414cd.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/images/ab399dbf9934c205d3779b516a1ab0aa30546be437c23642f0aefac8fee414cd.jpg)

![c7321c182e502182b9d32fae9f5a937495c12d7073b817603f66f3e82398b84c.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/images/c7321c182e502182b9d32fae9f5a937495c12d7073b817603f66f3e82398b84c.jpg)

![fa09c4d5f9f36d4059f7f30085248646543245aa40cf659501ff3c65f8f9490e.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/images/fa09c4d5f9f36d4059f7f30085248646543245aa40cf659501ff3c65f8f9490e.jpg)

### Tables

![58e7a8e9dd3ff4565227185a210394fb8be235e782d38855b0cf14bf57050d78.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/tables/58e7a8e9dd3ff4565227185a210394fb8be235e782d38855b0cf14bf57050d78.jpg)

![5ddcc1210166ecc1e53060b66d100ade22f5e1dde546c75fd3cae1337744f5c7.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/tables/5ddcc1210166ecc1e53060b66d100ade22f5e1dde546c75fd3cae1337744f5c7.jpg)

![75f2a49c469c970f8131bf8f7b40e5271deacf5cd07059acdfcff610a09c820b.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/tables/75f2a49c469c970f8131bf8f7b40e5271deacf5cd07059acdfcff610a09c820b.jpg)

![916b6e95851c0e60d004ece039a119e8096c9315de067f1b43952acfced4e13e.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/tables/916b6e95851c0e60d004ece039a119e8096c9315de067f1b43952acfced4e13e.jpg)

![9eb41ea2b153bb4ed1e2f718f0bffbe1443d958a18e0068d8c55cf96c693c837.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/tables/9eb41ea2b153bb4ed1e2f718f0bffbe1443d958a18e0068d8c55cf96c693c837.jpg)

![b12221d692d4bc45338c185ffdb024ac2ac67c8a19b22b2debc88c760eccdc61.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/tables/b12221d692d4bc45338c185ffdb024ac2ac67c8a19b22b2debc88c760eccdc61.jpg)

![b1890b59345c886c520ac2701d79a703ad42ccbb95b4c02246d7d7091f048ec1.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/tables/b1890b59345c886c520ac2701d79a703ad42ccbb95b4c02246d7d7091f048ec1.jpg)

![c34c09200a027c48d6da329e4b094ec22dc6600d0e4f2e3564c009493eb3ce23.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/tables/c34c09200a027c48d6da329e4b094ec22dc6600d0e4f2e3564c009493eb3ce23.jpg)

![c43def368f7f42cb16b2330d1a53404e258b8b35b13253ce4b4fca9b2e8a5037.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/tables/c43def368f7f42cb16b2330d1a53404e258b8b35b13253ce4b4fca9b2e8a5037.jpg)

![cd13e1af0664430a5fc2ce5a97a97f63116f3c70234cb901d1ff8887893653e5.jpg](../icml_results/5645_CARD_%20Coarse-to-fine%20Autoregressive%20Modeling%20with%20Radix-based%20Decomposition%20for%20Transferable%20Free%20En/tables/cd13e1af0664430a5fc2ce5a97a97f63116f3c70234cb901d1ff8887893653e5.jpg)

<a id="paper-5597"></a>

## Geometry-Preserving Unsupervised Alignment for Heterogeneous Foundation Models


### Images

![111c50bb0991bd27281bd4c6f175fda18eb086bf03988559f5ed70e6cc5f3e9a.jpg](../icml_results/5646_Geometry-Preserving%20Unsupervised%20Alignment%20for%20Heterogeneous%20Foundation%20Models/images/111c50bb0991bd27281bd4c6f175fda18eb086bf03988559f5ed70e6cc5f3e9a.jpg)

![291cd9da62df7e7be36b9ce1e14b63c62d5e78519efce7d8593b742bc6ed5859.jpg](../icml_results/5646_Geometry-Preserving%20Unsupervised%20Alignment%20for%20Heterogeneous%20Foundation%20Models/images/291cd9da62df7e7be36b9ce1e14b63c62d5e78519efce7d8593b742bc6ed5859.jpg)

![2f919960fba90a6b40bcbe6668dbb9c499b5bdfe154a4a9ccd2b66d1832b3965.jpg](../icml_results/5646_Geometry-Preserving%20Unsupervised%20Alignment%20for%20Heterogeneous%20Foundation%20Models/images/2f919960fba90a6b40bcbe6668dbb9c499b5bdfe154a4a9ccd2b66d1832b3965.jpg)

![4bb85b87f0201eace3921d5e2fe89d9d5aa6d4f79219cac709c67a88025fc6a8.jpg](../icml_results/5646_Geometry-Preserving%20Unsupervised%20Alignment%20for%20Heterogeneous%20Foundation%20Models/images/4bb85b87f0201eace3921d5e2fe89d9d5aa6d4f79219cac709c67a88025fc6a8.jpg)

![f7b285010e6da1808f43504ddb42d267098f78925b234891729b2adae491b462.jpg](../icml_results/5646_Geometry-Preserving%20Unsupervised%20Alignment%20for%20Heterogeneous%20Foundation%20Models/images/f7b285010e6da1808f43504ddb42d267098f78925b234891729b2adae491b462.jpg)

### Tables

![1c175df480f843515d36a0428b9332d41b343f8a5883c7073d00622535b5ec39.jpg](../icml_results/5646_Geometry-Preserving%20Unsupervised%20Alignment%20for%20Heterogeneous%20Foundation%20Models/tables/1c175df480f843515d36a0428b9332d41b343f8a5883c7073d00622535b5ec39.jpg)

![1ee8574f892c4e5f53c58005ed225d682bcdb9d3a87ec1e365f577a35b0d4685.jpg](../icml_results/5646_Geometry-Preserving%20Unsupervised%20Alignment%20for%20Heterogeneous%20Foundation%20Models/tables/1ee8574f892c4e5f53c58005ed225d682bcdb9d3a87ec1e365f577a35b0d4685.jpg)

![217b81ec888a9a90f5ac71702ba060c3538028b1e37dd326ed8b5d4173a95dfa.jpg](../icml_results/5646_Geometry-Preserving%20Unsupervised%20Alignment%20for%20Heterogeneous%20Foundation%20Models/tables/217b81ec888a9a90f5ac71702ba060c3538028b1e37dd326ed8b5d4173a95dfa.jpg)

![2f06e2785ae6b2a100d89211e06c9badd935814cf7df7408f8cc698e86ce5ba9.jpg](../icml_results/5646_Geometry-Preserving%20Unsupervised%20Alignment%20for%20Heterogeneous%20Foundation%20Models/tables/2f06e2785ae6b2a100d89211e06c9badd935814cf7df7408f8cc698e86ce5ba9.jpg)

![73119482104ecedcb10aa11720ae8b92960117dd40acb451c7ef657877c0e6bb.jpg](../icml_results/5646_Geometry-Preserving%20Unsupervised%20Alignment%20for%20Heterogeneous%20Foundation%20Models/tables/73119482104ecedcb10aa11720ae8b92960117dd40acb451c7ef657877c0e6bb.jpg)

![77d0d1c2254eeac8a12742e5f545ffee3e9b9cd686ee3524a71b7d39516dcdad.jpg](../icml_results/5646_Geometry-Preserving%20Unsupervised%20Alignment%20for%20Heterogeneous%20Foundation%20Models/tables/77d0d1c2254eeac8a12742e5f545ffee3e9b9cd686ee3524a71b7d39516dcdad.jpg)

![9ed85bd4ffb440cebf9191a8fc4999fb363281654f3f4149dac728744befaeb0.jpg](../icml_results/5646_Geometry-Preserving%20Unsupervised%20Alignment%20for%20Heterogeneous%20Foundation%20Models/tables/9ed85bd4ffb440cebf9191a8fc4999fb363281654f3f4149dac728744befaeb0.jpg)

![ab24cb573b84cb57cdf8a871cb401ead3c02a1ef57e6e8d070aea9afe167dd02.jpg](../icml_results/5646_Geometry-Preserving%20Unsupervised%20Alignment%20for%20Heterogeneous%20Foundation%20Models/tables/ab24cb573b84cb57cdf8a871cb401ead3c02a1ef57e6e8d070aea9afe167dd02.jpg)

<a id="paper-5598"></a>

## GFedCL: Graph-Based Federated Continual Learning with Spatial and Temporal Awareness


### Images

![1c5d6212db2b02df68f313b0b2f00b1e579aae49eb6c03cf7f8ee74cb63fbea5.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/images/1c5d6212db2b02df68f313b0b2f00b1e579aae49eb6c03cf7f8ee74cb63fbea5.jpg)

![242e68052024e67cbf2e842eac9e8e0ec2de8853b751989b611d9b9d9abc83bd.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/images/242e68052024e67cbf2e842eac9e8e0ec2de8853b751989b611d9b9d9abc83bd.jpg)

![54c99ab1c935517a98c604423a0837196326d082c50a7efcdc822eac3822f654.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/images/54c99ab1c935517a98c604423a0837196326d082c50a7efcdc822eac3822f654.jpg)

![69d906c102ea3ddc3c99817e3274c67fb5fb790a6bcd1929454de86237cb82e9.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/images/69d906c102ea3ddc3c99817e3274c67fb5fb790a6bcd1929454de86237cb82e9.jpg)

![7460d10670458b0170bd6f1ffa398ebfba9a5d1a4b6048d075c7e38fca982a71.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/images/7460d10670458b0170bd6f1ffa398ebfba9a5d1a4b6048d075c7e38fca982a71.jpg)

![753a81760b2a4d1897db29db3bff4925c9e91b7bbfa2c3d13f68dab3de3e203f.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/images/753a81760b2a4d1897db29db3bff4925c9e91b7bbfa2c3d13f68dab3de3e203f.jpg)

![a33b153a92b288a90660a079f7fac8c89466dc510aa9585f242c35f1c7d60db5.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/images/a33b153a92b288a90660a079f7fac8c89466dc510aa9585f242c35f1c7d60db5.jpg)

![a6ba411e050c791d4025f7560a44abee50c979bf648692cf46ff2e030731f750.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/images/a6ba411e050c791d4025f7560a44abee50c979bf648692cf46ff2e030731f750.jpg)

![bec71cd1b33a11842079971a3cc7c0c2e8792ce4863dbd48334fb70e182a0f6b.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/images/bec71cd1b33a11842079971a3cc7c0c2e8792ce4863dbd48334fb70e182a0f6b.jpg)

![cebc667f15c66fbd1563494b2ab2f045a3e78bceba305c906b2f60e250237dac.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/images/cebc667f15c66fbd1563494b2ab2f045a3e78bceba305c906b2f60e250237dac.jpg)

### Tables

![1742e09e1182f9c57963c9a1297d517e100243819b75358cdb20688f62706ca1.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/tables/1742e09e1182f9c57963c9a1297d517e100243819b75358cdb20688f62706ca1.jpg)

![1b99033ac1e5db59c3ec6245433944015cae7aa2c5a473ada23de1190734aae5.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/tables/1b99033ac1e5db59c3ec6245433944015cae7aa2c5a473ada23de1190734aae5.jpg)

![1f8c9652d982fc95b60047f2018ef8fec8b5b380e1327009ec556847641acce6.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/tables/1f8c9652d982fc95b60047f2018ef8fec8b5b380e1327009ec556847641acce6.jpg)

![2717a8f782625735ac1cb6287bc8726cb234e4f8f791da7d9723082697e9aecd.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/tables/2717a8f782625735ac1cb6287bc8726cb234e4f8f791da7d9723082697e9aecd.jpg)

![51138d776061a41a5efc0c4d9030656fbf0d7a5e4858db4929c5c070dd6f106e.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/tables/51138d776061a41a5efc0c4d9030656fbf0d7a5e4858db4929c5c070dd6f106e.jpg)

![5f806e2f1aeebd3a01891939770628d577141f1777b8c3a0facb5ab96097f2b4.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/tables/5f806e2f1aeebd3a01891939770628d577141f1777b8c3a0facb5ab96097f2b4.jpg)

![858029b4899282405b9d2724cf151788436cf06860e84ef6daadfc1c90e543c7.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/tables/858029b4899282405b9d2724cf151788436cf06860e84ef6daadfc1c90e543c7.jpg)

![862dc42bf95d3a78aae7a77ed25971a9da7ec213f149f1557d9813c6ec0a777d.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/tables/862dc42bf95d3a78aae7a77ed25971a9da7ec213f149f1557d9813c6ec0a777d.jpg)

![bf3ad30e61afa1ba51a0d539cc1279085e81e3b71555c41b983ff08b4b93ffd8.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/tables/bf3ad30e61afa1ba51a0d539cc1279085e81e3b71555c41b983ff08b4b93ffd8.jpg)

![da3de4635b17666e28ba6d026a8ddbb7e3dd35cbde49e88eba869f73ab091459.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/tables/da3de4635b17666e28ba6d026a8ddbb7e3dd35cbde49e88eba869f73ab091459.jpg)

![e6f73425d8b9e7892fd6fd3c27e8a0f4595536ba865c281e6b0bafb604c89c96.jpg](../icml_results/5647_GFedCL_%20Graph-Based%20Federated%20Continual%20Learning%20with%20Spatial%20and%20Temporal%20Awareness/tables/e6f73425d8b9e7892fd6fd3c27e8a0f4595536ba865c281e6b0bafb604c89c96.jpg)

<a id="paper-5599"></a>

## Scaling Laws of Global Weather Models


### Images

![0c150a809314f8dea90dfea50254e45f8c77d85379f6a3b058f2538c27cb21d0.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/images/0c150a809314f8dea90dfea50254e45f8c77d85379f6a3b058f2538c27cb21d0.jpg)

![0c89df76ea459e448850dd421c19131824d01dbdd18dc80263824421130dc559.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/images/0c89df76ea459e448850dd421c19131824d01dbdd18dc80263824421130dc559.jpg)

![3cdc74b83b2bec7be52b58c41a42f2b68df8c01ebd29454000b66b1cd5df38bf.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/images/3cdc74b83b2bec7be52b58c41a42f2b68df8c01ebd29454000b66b1cd5df38bf.jpg)

![883012ab1bfb2b12fb8687e37c0119e0dc4a49905c93f9f72958afce316d4c93.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/images/883012ab1bfb2b12fb8687e37c0119e0dc4a49905c93f9f72958afce316d4c93.jpg)

![970c68db0643f1cb1dc35fec0ca7e70fd637e2686c62d75d5224cf36157655d9.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/images/970c68db0643f1cb1dc35fec0ca7e70fd637e2686c62d75d5224cf36157655d9.jpg)

![a182b289436aed4e8557500010d54d54c5738403d72685e1054224b4b5c02dfc.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/images/a182b289436aed4e8557500010d54d54c5738403d72685e1054224b4b5c02dfc.jpg)

![d3d6d96569822317d678c3be6e9ccd7d14069c9dd2eedcdf2736fa6c39014904.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/images/d3d6d96569822317d678c3be6e9ccd7d14069c9dd2eedcdf2736fa6c39014904.jpg)

### Tables

![29278ada5a6430b469028a23f7fefcac009d70ee303b5732140e1a26fddc3863.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/tables/29278ada5a6430b469028a23f7fefcac009d70ee303b5732140e1a26fddc3863.jpg)

![5f2fb717f5aabdbf5f6297d8bd69e354d3142a88f02599adad026c3177bcb7d1.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/tables/5f2fb717f5aabdbf5f6297d8bd69e354d3142a88f02599adad026c3177bcb7d1.jpg)

![6c7a41aa8fb90ef214b4dd9a106b9b6062a8efbcdab801c66d96962d9fa95bf5.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/tables/6c7a41aa8fb90ef214b4dd9a106b9b6062a8efbcdab801c66d96962d9fa95bf5.jpg)

![75aa803dc70a89859c92235718588e389ea147c9c59de36301da17a6a247d778.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/tables/75aa803dc70a89859c92235718588e389ea147c9c59de36301da17a6a247d778.jpg)

![8f1f76c4b8cc791a7966334787042ea115c3f8ec49b7d81640a4aa9817adac83.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/tables/8f1f76c4b8cc791a7966334787042ea115c3f8ec49b7d81640a4aa9817adac83.jpg)

![a37d1d462bc4a30d925e246cc056f0d79fa10a94b91fc1b046c41dcf592fe848.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/tables/a37d1d462bc4a30d925e246cc056f0d79fa10a94b91fc1b046c41dcf592fe848.jpg)

![b49ab35da2cbd4eaacb11c44abe63a26ee7556fcad9c058aee5fae7d5d803a59.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/tables/b49ab35da2cbd4eaacb11c44abe63a26ee7556fcad9c058aee5fae7d5d803a59.jpg)

![c834fc0e0254308dc2f8e8315fbdd076f3e227736d249638f9e19c1dc766c224.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/tables/c834fc0e0254308dc2f8e8315fbdd076f3e227736d249638f9e19c1dc766c224.jpg)

![ca99ede7bcd0d059b579ad803b9e682b232624ed6ca8a55339df7fa7679812d1.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/tables/ca99ede7bcd0d059b579ad803b9e682b232624ed6ca8a55339df7fa7679812d1.jpg)

![cfc61421fc5be6eb6bd46967ddbaf9eaa7d121cf914a61f11b363fa197fbde2d.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/tables/cfc61421fc5be6eb6bd46967ddbaf9eaa7d121cf914a61f11b363fa197fbde2d.jpg)

![defa4673c23ab515134a0dc077c97dea5588d320394ea8cbfe4afd8419e07bef.jpg](../icml_results/5648_Scaling%20Laws%20of%20Global%20Weather%20Models/tables/defa4673c23ab515134a0dc077c97dea5588d320394ea8cbfe4afd8419e07bef.jpg)

<a id="paper-5600"></a>

## Mirror Descent Under Generalized Smoothness


### Images

![0191597427341bf8acbc09c699e6eacf2292b2e9c50f8aa48316e85e4b276337.jpg](../icml_results/5649_Mirror%20Descent%20Under%20Generalized%20Smoothness/images/0191597427341bf8acbc09c699e6eacf2292b2e9c50f8aa48316e85e4b276337.jpg)

![0f21f1bcd6a2a11cc96d7884daab2792ab61522e1d09b5dc708bc3abd1abfe1c.jpg](../icml_results/5649_Mirror%20Descent%20Under%20Generalized%20Smoothness/images/0f21f1bcd6a2a11cc96d7884daab2792ab61522e1d09b5dc708bc3abd1abfe1c.jpg)

![109adf12d6f7a6db004b4a0146778e23bb68f9aacaab0d156af90bef41644a9e.jpg](../icml_results/5649_Mirror%20Descent%20Under%20Generalized%20Smoothness/images/109adf12d6f7a6db004b4a0146778e23bb68f9aacaab0d156af90bef41644a9e.jpg)

![45c8ae24ab09a8e6a85079dfa578a5f22231537a8a5493bf49a81d53e87906fa.jpg](../icml_results/5649_Mirror%20Descent%20Under%20Generalized%20Smoothness/images/45c8ae24ab09a8e6a85079dfa578a5f22231537a8a5493bf49a81d53e87906fa.jpg)

![ab5bdd971b8a3fc763637c31ef959bfc5ad73318acd99a4238089ca82ca5922b.jpg](../icml_results/5649_Mirror%20Descent%20Under%20Generalized%20Smoothness/images/ab5bdd971b8a3fc763637c31ef959bfc5ad73318acd99a4238089ca82ca5922b.jpg)

![fc7ac342bba4811dcc5db8370568b468b24071cbf6a691ec11758920c9640ca9.jpg](../icml_results/5649_Mirror%20Descent%20Under%20Generalized%20Smoothness/images/fc7ac342bba4811dcc5db8370568b468b24071cbf6a691ec11758920c9640ca9.jpg)

### Tables

![73e6034048334c16200b2759dbd9f061c2e262111f6578e8e1e42ce88c358e74.jpg](../icml_results/5649_Mirror%20Descent%20Under%20Generalized%20Smoothness/tables/73e6034048334c16200b2759dbd9f061c2e262111f6578e8e1e42ce88c358e74.jpg)

<a id="paper-5601"></a>

## FlowBot: Inducing LLM Workflows with Bilevel Optimization and Textual Gradients


### Images

![1f1a1b8ce6cd96f6bed6ca95e5d036b66bc86fd0d70abfbe5ed0b1ffe06176c2.jpg](../icml_results/5650_FlowBot_%20Inducing%20LLM%20Workflows%20with%20Bilevel%20Optimization%20and%20Textual%20Gradients/images/1f1a1b8ce6cd96f6bed6ca95e5d036b66bc86fd0d70abfbe5ed0b1ffe06176c2.jpg)

![44d40526cfc811b6b1b1f38f5ac3abba478817fc2607e10c3e35fcd41c7fda97.jpg](../icml_results/5650_FlowBot_%20Inducing%20LLM%20Workflows%20with%20Bilevel%20Optimization%20and%20Textual%20Gradients/images/44d40526cfc811b6b1b1f38f5ac3abba478817fc2607e10c3e35fcd41c7fda97.jpg)

![721813837a66581708a38d6a27073c3a438d0b6989de8e4c7cd4a62b1a84debb.jpg](../icml_results/5650_FlowBot_%20Inducing%20LLM%20Workflows%20with%20Bilevel%20Optimization%20and%20Textual%20Gradients/images/721813837a66581708a38d6a27073c3a438d0b6989de8e4c7cd4a62b1a84debb.jpg)

![f7c4af443627a877b4db4548d94fe5b6a9dd2de23208771f3499979a52f4e7fd.jpg](../icml_results/5650_FlowBot_%20Inducing%20LLM%20Workflows%20with%20Bilevel%20Optimization%20and%20Textual%20Gradients/images/f7c4af443627a877b4db4548d94fe5b6a9dd2de23208771f3499979a52f4e7fd.jpg)

### Tables

![0ef8814397a82eb6370e406d40b5b3956b02106f1d5057ddb9e9bd2e349a8095.jpg](../icml_results/5650_FlowBot_%20Inducing%20LLM%20Workflows%20with%20Bilevel%20Optimization%20and%20Textual%20Gradients/tables/0ef8814397a82eb6370e406d40b5b3956b02106f1d5057ddb9e9bd2e349a8095.jpg)

![19bce41ae9ef73f59ffdd731d7f9d80b12f0c567fd0d7cfc52214cbd141556f7.jpg](../icml_results/5650_FlowBot_%20Inducing%20LLM%20Workflows%20with%20Bilevel%20Optimization%20and%20Textual%20Gradients/tables/19bce41ae9ef73f59ffdd731d7f9d80b12f0c567fd0d7cfc52214cbd141556f7.jpg)

![19caf339520a9d6850590280fe08c968fd564babdc488e2062da126a901d0d75.jpg](../icml_results/5650_FlowBot_%20Inducing%20LLM%20Workflows%20with%20Bilevel%20Optimization%20and%20Textual%20Gradients/tables/19caf339520a9d6850590280fe08c968fd564babdc488e2062da126a901d0d75.jpg)

![32dce8d13315dd9476689bd8f62d470e0ab2b8f8d23205f1e9c768749a30f54e.jpg](../icml_results/5650_FlowBot_%20Inducing%20LLM%20Workflows%20with%20Bilevel%20Optimization%20and%20Textual%20Gradients/tables/32dce8d13315dd9476689bd8f62d470e0ab2b8f8d23205f1e9c768749a30f54e.jpg)

![4181dbccfdef1bf1e372ce8683fc865b414056606d1d4fee331be8aa6d0aacca.jpg](../icml_results/5650_FlowBot_%20Inducing%20LLM%20Workflows%20with%20Bilevel%20Optimization%20and%20Textual%20Gradients/tables/4181dbccfdef1bf1e372ce8683fc865b414056606d1d4fee331be8aa6d0aacca.jpg)

![448b287ddd21881975ad39e8fef8b35860d36e722034b8829ef2ce22e3c6a2ab.jpg](../icml_results/5650_FlowBot_%20Inducing%20LLM%20Workflows%20with%20Bilevel%20Optimization%20and%20Textual%20Gradients/tables/448b287ddd21881975ad39e8fef8b35860d36e722034b8829ef2ce22e3c6a2ab.jpg)

![83ee07218e2cc981206d451d89c596b9528f87a5a8a9bf395bf6ea0d41a4b238.jpg](../icml_results/5650_FlowBot_%20Inducing%20LLM%20Workflows%20with%20Bilevel%20Optimization%20and%20Textual%20Gradients/tables/83ee07218e2cc981206d451d89c596b9528f87a5a8a9bf395bf6ea0d41a4b238.jpg)

![b76d9fe0144d0aed1c39f7aa56aab20ccb0e8abc11cda97f663403e013813e7b.jpg](../icml_results/5650_FlowBot_%20Inducing%20LLM%20Workflows%20with%20Bilevel%20Optimization%20and%20Textual%20Gradients/tables/b76d9fe0144d0aed1c39f7aa56aab20ccb0e8abc11cda97f663403e013813e7b.jpg)

![c0d3c6a025b010cdb71b39ab6a7c52fb4137eba7732777662df4b592470e5179.jpg](../icml_results/5650_FlowBot_%20Inducing%20LLM%20Workflows%20with%20Bilevel%20Optimization%20and%20Textual%20Gradients/tables/c0d3c6a025b010cdb71b39ab6a7c52fb4137eba7732777662df4b592470e5179.jpg)

![ccb4cac4f416479ce8718d6dda4552a62ba3ed40da92d6cc6e762b0cb81c007c.jpg](../icml_results/5650_FlowBot_%20Inducing%20LLM%20Workflows%20with%20Bilevel%20Optimization%20and%20Textual%20Gradients/tables/ccb4cac4f416479ce8718d6dda4552a62ba3ed40da92d6cc6e762b0cb81c007c.jpg)

![d0639b042128e66af645721aacffb2faeed1f2270b03d004dd7cbe904b0bc364.jpg](../icml_results/5650_FlowBot_%20Inducing%20LLM%20Workflows%20with%20Bilevel%20Optimization%20and%20Textual%20Gradients/tables/d0639b042128e66af645721aacffb2faeed1f2270b03d004dd7cbe904b0bc364.jpg)

![f41a564ee837271ae8c4f6aabb322809fcd1300ae1228c38d4452b3fc91f6265.jpg](../icml_results/5650_FlowBot_%20Inducing%20LLM%20Workflows%20with%20Bilevel%20Optimization%20and%20Textual%20Gradients/tables/f41a564ee837271ae8c4f6aabb322809fcd1300ae1228c38d4452b3fc91f6265.jpg)

<a id="paper-5602"></a>

## Multi-Adapter Representation Interventions via Energy Calibration


### Images

![37fa3cbe8e083a55dfaad37eac84577a0c4c88c7bb43b6d2cc9f5332997ac44d.jpg](../icml_results/5651_Multi-Adapter%20Representation%20Interventions%20via%20Energy%20Calibration/images/37fa3cbe8e083a55dfaad37eac84577a0c4c88c7bb43b6d2cc9f5332997ac44d.jpg)

![3c639dac5dd4892d7a40c001a0719e1ca7a4966d582601862a38463e45b64f03.jpg](../icml_results/5651_Multi-Adapter%20Representation%20Interventions%20via%20Energy%20Calibration/images/3c639dac5dd4892d7a40c001a0719e1ca7a4966d582601862a38463e45b64f03.jpg)

![75ae87aaaeacb8bc5eda18f9e855c30772b84683444dfef591d819f2b2f9073f.jpg](../icml_results/5651_Multi-Adapter%20Representation%20Interventions%20via%20Energy%20Calibration/images/75ae87aaaeacb8bc5eda18f9e855c30772b84683444dfef591d819f2b2f9073f.jpg)

![ab6a7f8f0dd8d2190909dfef118f97cced4097ffc2d5565004334c70458724ec.jpg](../icml_results/5651_Multi-Adapter%20Representation%20Interventions%20via%20Energy%20Calibration/images/ab6a7f8f0dd8d2190909dfef118f97cced4097ffc2d5565004334c70458724ec.jpg)

![b4e111e1bd30993d679c5f2ba2c1540620283edbd92b030fe97e8a7c7b830be8.jpg](../icml_results/5651_Multi-Adapter%20Representation%20Interventions%20via%20Energy%20Calibration/images/b4e111e1bd30993d679c5f2ba2c1540620283edbd92b030fe97e8a7c7b830be8.jpg)

![b758429508370659a8d17807c244028474be670aa9132e869baeaf69b42a4323.jpg](../icml_results/5651_Multi-Adapter%20Representation%20Interventions%20via%20Energy%20Calibration/images/b758429508370659a8d17807c244028474be670aa9132e869baeaf69b42a4323.jpg)

![e7eaf389714a3167d384d5d50f16de3edd0c7ee48ffda3a6551f800c5a4844b4.jpg](../icml_results/5651_Multi-Adapter%20Representation%20Interventions%20via%20Energy%20Calibration/images/e7eaf389714a3167d384d5d50f16de3edd0c7ee48ffda3a6551f800c5a4844b4.jpg)

### Tables

![05bb0d6431b51061b73c3f1613ce3e2b98b4c6952f896ad477dd21d7b2dcfbaf.jpg](../icml_results/5651_Multi-Adapter%20Representation%20Interventions%20via%20Energy%20Calibration/tables/05bb0d6431b51061b73c3f1613ce3e2b98b4c6952f896ad477dd21d7b2dcfbaf.jpg)

![152acee957d6afdd6ff2a52a22ed8817316a109c4df8d0d8e6c0e32d0733691b.jpg](../icml_results/5651_Multi-Adapter%20Representation%20Interventions%20via%20Energy%20Calibration/tables/152acee957d6afdd6ff2a52a22ed8817316a109c4df8d0d8e6c0e32d0733691b.jpg)

![7ed58d00885c12bacffcc307944b8f836adf75b641ffff5125ea000f9483a962.jpg](../icml_results/5651_Multi-Adapter%20Representation%20Interventions%20via%20Energy%20Calibration/tables/7ed58d00885c12bacffcc307944b8f836adf75b641ffff5125ea000f9483a962.jpg)

![900c40c3883e30d2f935aaa477126d68eda535ef94f830659d8e409e26786d69.jpg](../icml_results/5651_Multi-Adapter%20Representation%20Interventions%20via%20Energy%20Calibration/tables/900c40c3883e30d2f935aaa477126d68eda535ef94f830659d8e409e26786d69.jpg)

<a id="paper-5603"></a>

## From Prompts to Tokens: Internalizing Causal Supervision in Vision-Language Model for Multi-Image Causal Reasoning


### Images

![3050c34ac15d859f362527835053f496a67d9b5d7610323463df90d03272858b.jpg](../icml_results/5652_From%20Prompts%20to%20Tokens_%20Internalizing%20Causal%20Supervision%20in%20Vision-Language%20Model%20for%20Multi-Image%20Ca/images/3050c34ac15d859f362527835053f496a67d9b5d7610323463df90d03272858b.jpg)

![53dc3e3ca510a0b312f995bbe55addf6c551ae07d8fd1aab0cee44b81ec75449.jpg](../icml_results/5652_From%20Prompts%20to%20Tokens_%20Internalizing%20Causal%20Supervision%20in%20Vision-Language%20Model%20for%20Multi-Image%20Ca/images/53dc3e3ca510a0b312f995bbe55addf6c551ae07d8fd1aab0cee44b81ec75449.jpg)

![64688e06aedf315f08c4ee2a5e5648e436d4ef04f23e54c3c43065d0583ef491.jpg](../icml_results/5652_From%20Prompts%20to%20Tokens_%20Internalizing%20Causal%20Supervision%20in%20Vision-Language%20Model%20for%20Multi-Image%20Ca/images/64688e06aedf315f08c4ee2a5e5648e436d4ef04f23e54c3c43065d0583ef491.jpg)

![d5272431fb87ef2e545cd22848b65ec035ade6ded8398962acefb06b0ede26a0.jpg](../icml_results/5652_From%20Prompts%20to%20Tokens_%20Internalizing%20Causal%20Supervision%20in%20Vision-Language%20Model%20for%20Multi-Image%20Ca/images/d5272431fb87ef2e545cd22848b65ec035ade6ded8398962acefb06b0ede26a0.jpg)

![ec450f8c5d601bc029422d424f63d2c62d5a132c37df248b07c03200f66f8393.jpg](../icml_results/5652_From%20Prompts%20to%20Tokens_%20Internalizing%20Causal%20Supervision%20in%20Vision-Language%20Model%20for%20Multi-Image%20Ca/images/ec450f8c5d601bc029422d424f63d2c62d5a132c37df248b07c03200f66f8393.jpg)

### Tables

![1371d3aff02f173c9964bfca166bcfa6031437162125a085124b2580d63ed44e.jpg](../icml_results/5652_From%20Prompts%20to%20Tokens_%20Internalizing%20Causal%20Supervision%20in%20Vision-Language%20Model%20for%20Multi-Image%20Ca/tables/1371d3aff02f173c9964bfca166bcfa6031437162125a085124b2580d63ed44e.jpg)

![1ccab7a1cb79c047825e0b6346d6416e42a904dd31329c418aacce1ef5ad7d47.jpg](../icml_results/5652_From%20Prompts%20to%20Tokens_%20Internalizing%20Causal%20Supervision%20in%20Vision-Language%20Model%20for%20Multi-Image%20Ca/tables/1ccab7a1cb79c047825e0b6346d6416e42a904dd31329c418aacce1ef5ad7d47.jpg)

![3ad0ae4273c35dde96f2c744da13229d59fe416863ea15583d239bb0efc5f800.jpg](../icml_results/5652_From%20Prompts%20to%20Tokens_%20Internalizing%20Causal%20Supervision%20in%20Vision-Language%20Model%20for%20Multi-Image%20Ca/tables/3ad0ae4273c35dde96f2c744da13229d59fe416863ea15583d239bb0efc5f800.jpg)

![5874ab634036e947914261a78eed26a850581314bffeb90415e0984edce666e9.jpg](../icml_results/5652_From%20Prompts%20to%20Tokens_%20Internalizing%20Causal%20Supervision%20in%20Vision-Language%20Model%20for%20Multi-Image%20Ca/tables/5874ab634036e947914261a78eed26a850581314bffeb90415e0984edce666e9.jpg)

![75d5270f7a9fefca2545a558289a07c98e9ad4a4993f8c44e06569f6fcce6345.jpg](../icml_results/5652_From%20Prompts%20to%20Tokens_%20Internalizing%20Causal%20Supervision%20in%20Vision-Language%20Model%20for%20Multi-Image%20Ca/tables/75d5270f7a9fefca2545a558289a07c98e9ad4a4993f8c44e06569f6fcce6345.jpg)

![a346e23069010d366f1458e36871104e90d5a485f25885e067f9210960f2b4f6.jpg](../icml_results/5652_From%20Prompts%20to%20Tokens_%20Internalizing%20Causal%20Supervision%20in%20Vision-Language%20Model%20for%20Multi-Image%20Ca/tables/a346e23069010d366f1458e36871104e90d5a485f25885e067f9210960f2b4f6.jpg)

![ba4dec1d88c4ee72e564371cdec0978ac41a03cb9316c2fd2354ff15ca2b269a.jpg](../icml_results/5652_From%20Prompts%20to%20Tokens_%20Internalizing%20Causal%20Supervision%20in%20Vision-Language%20Model%20for%20Multi-Image%20Ca/tables/ba4dec1d88c4ee72e564371cdec0978ac41a03cb9316c2fd2354ff15ca2b269a.jpg)

![c1e9a35ff3fbbea9b5048075cb4cca1cc6c83ba95ee16edff280957af59d4715.jpg](../icml_results/5652_From%20Prompts%20to%20Tokens_%20Internalizing%20Causal%20Supervision%20in%20Vision-Language%20Model%20for%20Multi-Image%20Ca/tables/c1e9a35ff3fbbea9b5048075cb4cca1cc6c83ba95ee16edff280957af59d4715.jpg)

![da72459ba8e065fe3df9bb19d44e5b7fd82c477fa08fedbe8d51753447c8244a.jpg](../icml_results/5652_From%20Prompts%20to%20Tokens_%20Internalizing%20Causal%20Supervision%20in%20Vision-Language%20Model%20for%20Multi-Image%20Ca/tables/da72459ba8e065fe3df9bb19d44e5b7fd82c477fa08fedbe8d51753447c8244a.jpg)

![e4437aa7d7571bae81e354b7fdd7b35ae5304f459c141b5348ba8a8484dacce3.jpg](../icml_results/5652_From%20Prompts%20to%20Tokens_%20Internalizing%20Causal%20Supervision%20in%20Vision-Language%20Model%20for%20Multi-Image%20Ca/tables/e4437aa7d7571bae81e354b7fdd7b35ae5304f459c141b5348ba8a8484dacce3.jpg)

![ec55c81ba0cd785dca6eb621ab62fa4f4937204adfe88f8bb58dc6d19253d2ef.jpg](../icml_results/5652_From%20Prompts%20to%20Tokens_%20Internalizing%20Causal%20Supervision%20in%20Vision-Language%20Model%20for%20Multi-Image%20Ca/tables/ec55c81ba0cd785dca6eb621ab62fa4f4937204adfe88f8bb58dc6d19253d2ef.jpg)

<a id="paper-5604"></a>

## CORRECT: COndensed eRror RECognition via knowledge Transfer in multi-agent systems


### Images

![0ee760d9923990826359b2f3f6cac4a7b47eb261a008db2dfcdeb8885439e2dc.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/images/0ee760d9923990826359b2f3f6cac4a7b47eb261a008db2dfcdeb8885439e2dc.jpg)

![5037363b45d1b29e50341698bb652cbf8236e9f1e8ffef3222daff0c34bf7cc2.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/images/5037363b45d1b29e50341698bb652cbf8236e9f1e8ffef3222daff0c34bf7cc2.jpg)

![5b858b2a97732529ac0c7ead43c64748c2ec974406292baf7d02e3b24a1cb237.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/images/5b858b2a97732529ac0c7ead43c64748c2ec974406292baf7d02e3b24a1cb237.jpg)

![60f90abac46bc7161576f9b964b887f2aef39fe3ac42f562104ee1bf59e2e8cb.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/images/60f90abac46bc7161576f9b964b887f2aef39fe3ac42f562104ee1bf59e2e8cb.jpg)

![708fb5dd1bc7b55cc0895fbd0287c7cfe1f54ffc9ffc073b083ee193d293e996.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/images/708fb5dd1bc7b55cc0895fbd0287c7cfe1f54ffc9ffc073b083ee193d293e996.jpg)

![7db6534de3329fec7a3286da3b14c5b2b7971425b7e72c6a9161858c22781cf0.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/images/7db6534de3329fec7a3286da3b14c5b2b7971425b7e72c6a9161858c22781cf0.jpg)

![8b6512dc031748b5eb7b45ddf1f46b40871a1b250aa250ea25b8898df77fd639.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/images/8b6512dc031748b5eb7b45ddf1f46b40871a1b250aa250ea25b8898df77fd639.jpg)

![fbb780d77581edaf86502f0a9575ef2a4ad064be66e3ba062f46bc6c5d436a09.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/images/fbb780d77581edaf86502f0a9575ef2a4ad064be66e3ba062f46bc6c5d436a09.jpg)

### Tables

![07f2f416e43e6dff326d827b56a4d9abb6a64de8136d7400dcd65e9393579243.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/tables/07f2f416e43e6dff326d827b56a4d9abb6a64de8136d7400dcd65e9393579243.jpg)

![3299df4078525bc0ffc1cbcc244a54d7b16953c58f91eb654a998e4e8e05f17b.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/tables/3299df4078525bc0ffc1cbcc244a54d7b16953c58f91eb654a998e4e8e05f17b.jpg)

![3663f6b1d722f0168465bb11bcb10eee3293846b6cb83d72e23acfd633034a7f.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/tables/3663f6b1d722f0168465bb11bcb10eee3293846b6cb83d72e23acfd633034a7f.jpg)

![3808e61647f833e8ad7cfdc0c80e8d86545e425a635ba19ea6fd1c172f0a1e37.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/tables/3808e61647f833e8ad7cfdc0c80e8d86545e425a635ba19ea6fd1c172f0a1e37.jpg)

![3fff86dc1dcabb1e7cd3e7fc3b2edbbbb48265a4c0807967357a44d77771be1d.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/tables/3fff86dc1dcabb1e7cd3e7fc3b2edbbbb48265a4c0807967357a44d77771be1d.jpg)

![56412b5397c300cd9c28e141688364ee4258937606869859c220eb9e8f596134.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/tables/56412b5397c300cd9c28e141688364ee4258937606869859c220eb9e8f596134.jpg)

![69ee15bfab031f1b72eab5ab4cab1dcc078b1a3630540110851b432ffa333fd3.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/tables/69ee15bfab031f1b72eab5ab4cab1dcc078b1a3630540110851b432ffa333fd3.jpg)

![9116c9754378f6afb7c3b76b5454c5515e32703fb92b49cd28b8269aef61c9ee.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/tables/9116c9754378f6afb7c3b76b5454c5515e32703fb92b49cd28b8269aef61c9ee.jpg)

![92c0f0f6930645207ed1692503296499e00f392ef2d3b4cd61c48d237c27c4cb.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/tables/92c0f0f6930645207ed1692503296499e00f392ef2d3b4cd61c48d237c27c4cb.jpg)

![b016abde863a3433a4bf6d08b0cbf0d994f50e34fa0bcbca258e5b8f21c31451.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/tables/b016abde863a3433a4bf6d08b0cbf0d994f50e34fa0bcbca258e5b8f21c31451.jpg)

![bb653cd4f12b637b6695df2b7f759e44bc045e910031cf838e42b242747d571e.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/tables/bb653cd4f12b637b6695df2b7f759e44bc045e910031cf838e42b242747d571e.jpg)

![d0b55c8428640ded05a27b2ef704e922bc2f6ce526288dc7e503039fdc9161d2.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/tables/d0b55c8428640ded05a27b2ef704e922bc2f6ce526288dc7e503039fdc9161d2.jpg)

![dd1db814fb082e7b6746e677d2821b0f29d7ab9872a5033d82dc4d15295b814f.jpg](../icml_results/5653_CORRECT_%20COndensed%20eRror%20RECognition%20via%20knowledge%20Transfer%20in%20multi-agent%20systems/tables/dd1db814fb082e7b6746e677d2821b0f29d7ab9872a5033d82dc4d15295b814f.jpg)

<a id="paper-5605"></a>

## Efficient Prediction of SO(3)-Equivariant Hamiltonian Matrices via SO(2) Local Frames


### Images

![5ccb763fd8b5b251a2ea2f5591392315aec0947aa263bc39d53f5697b8f44193.jpg](../icml_results/5654_Efficient%20Prediction%20of%20SO%283%29-Equivariant%20Hamiltonian%20Matrices%20via%20SO%282%29%20Local%20Frames/images/5ccb763fd8b5b251a2ea2f5591392315aec0947aa263bc39d53f5697b8f44193.jpg)

![71e2c2a200174ae90b1ce0d78e23a9f1ef482dcfb72ceaebf50ded34f435f90d.jpg](../icml_results/5654_Efficient%20Prediction%20of%20SO%283%29-Equivariant%20Hamiltonian%20Matrices%20via%20SO%282%29%20Local%20Frames/images/71e2c2a200174ae90b1ce0d78e23a9f1ef482dcfb72ceaebf50ded34f435f90d.jpg)

### Tables

![08e9ddffebd8642c647c06e1e49fbb57a6a98b6eb7d96d18e11b40298f44ed8d.jpg](../icml_results/5654_Efficient%20Prediction%20of%20SO%283%29-Equivariant%20Hamiltonian%20Matrices%20via%20SO%282%29%20Local%20Frames/tables/08e9ddffebd8642c647c06e1e49fbb57a6a98b6eb7d96d18e11b40298f44ed8d.jpg)

![0d0d7b4171f2049a9d54b9d165ad903e8c2f9be06b2640a6b6d6e2c4710fc85c.jpg](../icml_results/5654_Efficient%20Prediction%20of%20SO%283%29-Equivariant%20Hamiltonian%20Matrices%20via%20SO%282%29%20Local%20Frames/tables/0d0d7b4171f2049a9d54b9d165ad903e8c2f9be06b2640a6b6d6e2c4710fc85c.jpg)

![1faf0449ca1d256e76ae60dfd2544a4d95a0c0a51d9b68f22c880ee3f3b37c49.jpg](../icml_results/5654_Efficient%20Prediction%20of%20SO%283%29-Equivariant%20Hamiltonian%20Matrices%20via%20SO%282%29%20Local%20Frames/tables/1faf0449ca1d256e76ae60dfd2544a4d95a0c0a51d9b68f22c880ee3f3b37c49.jpg)

![6eceab0f7b2524f7d069efa97d2fbc14731d373b5398dd5ac2f04141e449261f.jpg](../icml_results/5654_Efficient%20Prediction%20of%20SO%283%29-Equivariant%20Hamiltonian%20Matrices%20via%20SO%282%29%20Local%20Frames/tables/6eceab0f7b2524f7d069efa97d2fbc14731d373b5398dd5ac2f04141e449261f.jpg)

![793ea2c34fb1b50caa9052130184433b9a2893e23029d7e635c86834ed935d2e.jpg](../icml_results/5654_Efficient%20Prediction%20of%20SO%283%29-Equivariant%20Hamiltonian%20Matrices%20via%20SO%282%29%20Local%20Frames/tables/793ea2c34fb1b50caa9052130184433b9a2893e23029d7e635c86834ed935d2e.jpg)

![8d32e94e398c437fecfe63ab4410f49f71b66afc139b29a6cfb3e86cd09e4ff5.jpg](../icml_results/5654_Efficient%20Prediction%20of%20SO%283%29-Equivariant%20Hamiltonian%20Matrices%20via%20SO%282%29%20Local%20Frames/tables/8d32e94e398c437fecfe63ab4410f49f71b66afc139b29a6cfb3e86cd09e4ff5.jpg)

![b794ea675a18988d76e0934a8955605322d90c24adeeb7beb76594d22d3aa787.jpg](../icml_results/5654_Efficient%20Prediction%20of%20SO%283%29-Equivariant%20Hamiltonian%20Matrices%20via%20SO%282%29%20Local%20Frames/tables/b794ea675a18988d76e0934a8955605322d90c24adeeb7beb76594d22d3aa787.jpg)

![d0fa144ea7eb0963dac26c75d232baf8c38913cc89bdadefaad755491cf170bd.jpg](../icml_results/5654_Efficient%20Prediction%20of%20SO%283%29-Equivariant%20Hamiltonian%20Matrices%20via%20SO%282%29%20Local%20Frames/tables/d0fa144ea7eb0963dac26c75d232baf8c38913cc89bdadefaad755491cf170bd.jpg)

<a id="paper-5606"></a>

## Autoregressive Image Generation with Masked Bit Modeling


### Images

![096cc8c2a80ee43f0d4c334546da6e7e113f6f1006ca7384710edc1b8a483155.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/096cc8c2a80ee43f0d4c334546da6e7e113f6f1006ca7384710edc1b8a483155.jpg)

![2518be1f07c91dd9cd94b6ab02eef3ded9f3510a4e03e93c4f5b7c469a6593df.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/2518be1f07c91dd9cd94b6ab02eef3ded9f3510a4e03e93c4f5b7c469a6593df.jpg)

![2661389a9aee36695841accaa9168359a38928043a281cbbd9b654476e593bbd.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/2661389a9aee36695841accaa9168359a38928043a281cbbd9b654476e593bbd.jpg)

![3cdb09541f3898da704e714ad5d92808fbc2e72d3be18302bc8eea062474e476.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/3cdb09541f3898da704e714ad5d92808fbc2e72d3be18302bc8eea062474e476.jpg)

![461f9696bc94455e6ed74e2e56f7aed2500d1b181311728d6721799630c83266.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/461f9696bc94455e6ed74e2e56f7aed2500d1b181311728d6721799630c83266.jpg)

![502bc2274a012275dcebea3cf39c67970fbe91e8afbc65c3b733663394a43a8c.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/502bc2274a012275dcebea3cf39c67970fbe91e8afbc65c3b733663394a43a8c.jpg)

![5bb56189c4ba33981a110c0c1e71d044d7e339a3b71b0b03627e15cf3f7f0cc3.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/5bb56189c4ba33981a110c0c1e71d044d7e339a3b71b0b03627e15cf3f7f0cc3.jpg)

![62063cbc974e4e4f56b07f69cf1010a2cd6ebc4ab409b3e88911bad47b5f54a2.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/62063cbc974e4e4f56b07f69cf1010a2cd6ebc4ab409b3e88911bad47b5f54a2.jpg)

![65b76770684a1ec9a87d35ba5f3e24191836e7142798de48c18786b162019e7b.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/65b76770684a1ec9a87d35ba5f3e24191836e7142798de48c18786b162019e7b.jpg)

![6c53ebe17d1db09f8c6646a36f6af92e2a813ab444b3e44410f584cdf8a06541.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/6c53ebe17d1db09f8c6646a36f6af92e2a813ab444b3e44410f584cdf8a06541.jpg)

![a383c766d0bef13f6b786b5342844152847106fddefc0f8c89a44d58f447b29e.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/a383c766d0bef13f6b786b5342844152847106fddefc0f8c89a44d58f447b29e.jpg)

![ad22ab45de94716b73530a8abbfba518f1b36dd17d18e26807876a314e808ae7.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/ad22ab45de94716b73530a8abbfba518f1b36dd17d18e26807876a314e808ae7.jpg)

![ae7bcb408d14ae9d4761ade1908e045a7140dd50ce2399329c48569210d372d7.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/ae7bcb408d14ae9d4761ade1908e045a7140dd50ce2399329c48569210d372d7.jpg)

![c94abaf946f3c1ea8f6657a43516c28034932e04231fba777e2d8e788e859e46.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/c94abaf946f3c1ea8f6657a43516c28034932e04231fba777e2d8e788e859e46.jpg)

![d348411a6334202706207eafee81fc0e3a8324f5d11ea514fddf569e1864fe94.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/d348411a6334202706207eafee81fc0e3a8324f5d11ea514fddf569e1864fe94.jpg)

![e6d9c488c6871154fb65f1b48e12e8beceea82676e2ac12d73e95b5a2dffb8e9.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/e6d9c488c6871154fb65f1b48e12e8beceea82676e2ac12d73e95b5a2dffb8e9.jpg)

![ed1da7717ca1a57447e187015b99940b7294e47728ca816f541106d8bdd88a00.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/ed1da7717ca1a57447e187015b99940b7294e47728ca816f541106d8bdd88a00.jpg)

![ed5e2a30dd4eecd2914e1cd597f23176a1ab6595698418a889f84a28428d5212.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/images/ed5e2a30dd4eecd2914e1cd597f23176a1ab6595698418a889f84a28428d5212.jpg)

### Tables

![01769c430cb87663c7a61c6db59969ebf74dfd1e1a0b057cd012c9f2ccb24eee.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/tables/01769c430cb87663c7a61c6db59969ebf74dfd1e1a0b057cd012c9f2ccb24eee.jpg)

![29af651ec58ef2e26a95c29c03599e110aa087452afacfeeef9bb3d144f6ebeb.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/tables/29af651ec58ef2e26a95c29c03599e110aa087452afacfeeef9bb3d144f6ebeb.jpg)

![341caa04275bc7361cd8a09caa4ab942373520d1a04db0e4efe60b01b2453f80.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/tables/341caa04275bc7361cd8a09caa4ab942373520d1a04db0e4efe60b01b2453f80.jpg)

![43ee923d6d87ce5fb7f5644dbac04fd6f6a8667e8a1893adde81e58dcdd2d756.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/tables/43ee923d6d87ce5fb7f5644dbac04fd6f6a8667e8a1893adde81e58dcdd2d756.jpg)

![5291a7a3ecf39816547e1e3511637be34c0359c3f4fbaedb1b76091dd604994a.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/tables/5291a7a3ecf39816547e1e3511637be34c0359c3f4fbaedb1b76091dd604994a.jpg)

![753a32e31339ccb2dd94d4a7e197827be99d525e83b893bfe1d7ac8e5a25b397.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/tables/753a32e31339ccb2dd94d4a7e197827be99d525e83b893bfe1d7ac8e5a25b397.jpg)

![84a711bb9c1deafea748cb334006c18daeb2f5728b83b4a15af841feff90f896.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/tables/84a711bb9c1deafea748cb334006c18daeb2f5728b83b4a15af841feff90f896.jpg)

![aa263db5cdb8d68d3e93b6f0abaecfee84dabc0f22492a9b92c42e7870814b25.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/tables/aa263db5cdb8d68d3e93b6f0abaecfee84dabc0f22492a9b92c42e7870814b25.jpg)

![b12654bdb3a889ed9452201e8f4411891d6709624056760bbce60fe984ba7a0b.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/tables/b12654bdb3a889ed9452201e8f4411891d6709624056760bbce60fe984ba7a0b.jpg)

![faf961953001ae864b3382e95d1fa0c6152bb92d37ddab70f8171ef16714f252.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/tables/faf961953001ae864b3382e95d1fa0c6152bb92d37ddab70f8171ef16714f252.jpg)

![fe63e9caa4b6a71f3c37455d99127fcd483c89bbfa7d0afa42e88e3c2bf30b83.jpg](../icml_results/5655_Autoregressive%20Image%20Generation%20with%20Masked%20Bit%20Modeling/tables/fe63e9caa4b6a71f3c37455d99127fcd483c89bbfa7d0afa42e88e3c2bf30b83.jpg)

<a id="paper-5607"></a>

## PATCHCODE: Discrete Latent Predictive Learning for EEG Foundation Model


### Images

![13ce96e64a17008388688dde812e67b509f8e37569bb65c3d7e79453024cccaa.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/images/13ce96e64a17008388688dde812e67b509f8e37569bb65c3d7e79453024cccaa.jpg)

![1674ffd70f7146da4a6b8195ea0ccbe475b8c2300fbec4500d03cb65efafa378.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/images/1674ffd70f7146da4a6b8195ea0ccbe475b8c2300fbec4500d03cb65efafa378.jpg)

![182c769eb26931e863575f8decb58b124ee5d287bd3b5b6f609184a59919add6.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/images/182c769eb26931e863575f8decb58b124ee5d287bd3b5b6f609184a59919add6.jpg)

![75976f774a1f6adc319cb5b9666d84d1ba862212cf3c9dcd18764b5e80890740.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/images/75976f774a1f6adc319cb5b9666d84d1ba862212cf3c9dcd18764b5e80890740.jpg)

![7e919238c6e45fd2f4748b4c3ec426163680a20052c7c9973a134f2aeb8253a9.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/images/7e919238c6e45fd2f4748b4c3ec426163680a20052c7c9973a134f2aeb8253a9.jpg)

![d150e26cab452c09620ec997664490c21ad903e554e6a073d93b2a45b59903e2.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/images/d150e26cab452c09620ec997664490c21ad903e554e6a073d93b2a45b59903e2.jpg)

### Tables

![1113ea36cea851685115aee5d942dc6467133f0cedb134d4ed94a78391810cf2.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/1113ea36cea851685115aee5d942dc6467133f0cedb134d4ed94a78391810cf2.jpg)

![1b0d6549ea1b205b6380584679dd51dd544a4ed574fc60616edea737cc930962.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/1b0d6549ea1b205b6380584679dd51dd544a4ed574fc60616edea737cc930962.jpg)

![2e285fa962be5e3e97eb2f8d58423088b7b6745220cd4f66401dea9bf31cd507.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/2e285fa962be5e3e97eb2f8d58423088b7b6745220cd4f66401dea9bf31cd507.jpg)

![34822b48bf76f29ae91b9f0d8d6965f444bf707d2d516f7b4d940d07cbe03b08.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/34822b48bf76f29ae91b9f0d8d6965f444bf707d2d516f7b4d940d07cbe03b08.jpg)

![3812020c950d18a3c1a05e17ba959acac0c332b9dc88986c05541f85b89528b7.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/3812020c950d18a3c1a05e17ba959acac0c332b9dc88986c05541f85b89528b7.jpg)

![4d001d24a9d6320f8456b19ea8843404b0e12277bfc066a82be403966d5d5333.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/4d001d24a9d6320f8456b19ea8843404b0e12277bfc066a82be403966d5d5333.jpg)

![4edb878beacd10b143bd7773c0ef95d9236fb32ccb7cb7810690ea2117694ec0.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/4edb878beacd10b143bd7773c0ef95d9236fb32ccb7cb7810690ea2117694ec0.jpg)

![5b11b6d2c54e84bbf28b921bf89a81397a52b0821c3c2e44c3eef2ad54cc1239.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/5b11b6d2c54e84bbf28b921bf89a81397a52b0821c3c2e44c3eef2ad54cc1239.jpg)

![5d5cedd1036daf48935a1bf8666a54d73270a5b30311ec17a0d096392de025d6.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/5d5cedd1036daf48935a1bf8666a54d73270a5b30311ec17a0d096392de025d6.jpg)

![695eb38b0d3fcd8aafde0bb7d01b9cd64a3414038ba914005d34b43b7a0dc15b.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/695eb38b0d3fcd8aafde0bb7d01b9cd64a3414038ba914005d34b43b7a0dc15b.jpg)

![6ea7353597725b229e613aa0ac16651bae16f5f4cc2e7fd3b18c45031dc3182b.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/6ea7353597725b229e613aa0ac16651bae16f5f4cc2e7fd3b18c45031dc3182b.jpg)

![73345e891e8920f35bfebab8da8399e75174f78e4d029d08aa5fcbca0c78b621.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/73345e891e8920f35bfebab8da8399e75174f78e4d029d08aa5fcbca0c78b621.jpg)

![75b50e818d869be1ea64b01a0c8bee9cdca7e56338e245eec1540b873cc7e3f0.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/75b50e818d869be1ea64b01a0c8bee9cdca7e56338e245eec1540b873cc7e3f0.jpg)

![7f0c3df9ea94e9d8b2c02506c0b2f8b101f113000ab19a4bb8e46d9335538553.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/7f0c3df9ea94e9d8b2c02506c0b2f8b101f113000ab19a4bb8e46d9335538553.jpg)

![825452c8f69fe7036fcc42f9520ea5056b4e075261cc8e524790ef3d52cd531b.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/825452c8f69fe7036fcc42f9520ea5056b4e075261cc8e524790ef3d52cd531b.jpg)

![9141046123a71620d74949baedf47a77a31fde74d3d4d8a4871609620e0a3c72.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/9141046123a71620d74949baedf47a77a31fde74d3d4d8a4871609620e0a3c72.jpg)

![af8cf0254855d15c73ddeea1ad6a037335eb852baad9fb080aafd45ba3844818.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/af8cf0254855d15c73ddeea1ad6a037335eb852baad9fb080aafd45ba3844818.jpg)

![b0b6c72e915ad90cac94eaab377ae49a2d0fa87e1eac57ad8459e48bb2e33f3a.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/b0b6c72e915ad90cac94eaab377ae49a2d0fa87e1eac57ad8459e48bb2e33f3a.jpg)

![b17e23826a895a3e026fcb97c1875977a8bba67458cd3e58f049db3522be8254.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/b17e23826a895a3e026fcb97c1875977a8bba67458cd3e58f049db3522be8254.jpg)

![b46cafcf1405e64cb9043b0e02c5fbeb9b0b631826b0ae56bf7030b5ad062320.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/b46cafcf1405e64cb9043b0e02c5fbeb9b0b631826b0ae56bf7030b5ad062320.jpg)

![baf4c5bca9724b23f7c7000988652dd42b63ec934663502bf97ee1b60fb3923b.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/baf4c5bca9724b23f7c7000988652dd42b63ec934663502bf97ee1b60fb3923b.jpg)

![c0029038cd8a222851f40b032ecac2253042a72ffc1c4ade2a2166416a37bf5e.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/c0029038cd8a222851f40b032ecac2253042a72ffc1c4ade2a2166416a37bf5e.jpg)

![c17d8290d1a0df02530e9c9d4149117545e3f3664bf624fb8b52bda93cf97a94.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/c17d8290d1a0df02530e9c9d4149117545e3f3664bf624fb8b52bda93cf97a94.jpg)

![c5ebb0533a38227252745742e8f4830ce3010cbe0d7fe06d632e7286df91f90e.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/c5ebb0533a38227252745742e8f4830ce3010cbe0d7fe06d632e7286df91f90e.jpg)

![cc9481baef5a2962fe3c41f3d89ec9bae1fa8316480bc8dba303b28cf46134e8.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/cc9481baef5a2962fe3c41f3d89ec9bae1fa8316480bc8dba303b28cf46134e8.jpg)

![d52422c30a7a817b05fe98a4b73f01b359912949bcb6a4670323529287116968.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/d52422c30a7a817b05fe98a4b73f01b359912949bcb6a4670323529287116968.jpg)

![e3570015f783fd2934244040c33836af4c888ac2eb9ad0cb86459599c248c6b4.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/e3570015f783fd2934244040c33836af4c888ac2eb9ad0cb86459599c248c6b4.jpg)

![e43b8be130aa43a473e0a436f9f815db4fce25896f6f2f488a3db55499c3a7aa.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/e43b8be130aa43a473e0a436f9f815db4fce25896f6f2f488a3db55499c3a7aa.jpg)

![e5074c9e5fb30914521061515f97bb0b52bce17baba90c8ef88af1184d688808.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/e5074c9e5fb30914521061515f97bb0b52bce17baba90c8ef88af1184d688808.jpg)

![fa014a881feb2ad01a4ce634cdbd98d3500976baff13f6b9ae89556f163f7347.jpg](../icml_results/5656_PATCHCODE_%20Discrete%20Latent%20Predictive%20Learning%20for%20EEG%20Foundation%20Model/tables/fa014a881feb2ad01a4ce634cdbd98d3500976baff13f6b9ae89556f163f7347.jpg)

<a id="paper-5608"></a>

## Bridging Structure and Semantics: Uncertainty-Modulated Dual-Path Diffusion for Robust Text-Attributed Graph Learning


### Images

![0eb234bff25de7cff8b96405de9c97b52fa15d4eafad449941f299629c4ec402.jpg](../icml_results/5657_Bridging%20Structure%20and%20Semantics_%20Uncertainty-Modulated%20Dual-Path%20Diffusion%20for%20Robust%20Text-Attribut/images/0eb234bff25de7cff8b96405de9c97b52fa15d4eafad449941f299629c4ec402.jpg)

![874d2db9f8ce35f8e46f6021a4a69c01d405e69b921f7623ac93310079523539.jpg](../icml_results/5657_Bridging%20Structure%20and%20Semantics_%20Uncertainty-Modulated%20Dual-Path%20Diffusion%20for%20Robust%20Text-Attribut/images/874d2db9f8ce35f8e46f6021a4a69c01d405e69b921f7623ac93310079523539.jpg)

![a45a50d34c096032ae75d428aef02a433ba0193351a0d38ea4777ae113f1f889.jpg](../icml_results/5657_Bridging%20Structure%20and%20Semantics_%20Uncertainty-Modulated%20Dual-Path%20Diffusion%20for%20Robust%20Text-Attribut/images/a45a50d34c096032ae75d428aef02a433ba0193351a0d38ea4777ae113f1f889.jpg)

![dfe88cd982a2f0128893f4153e237f5370a4d21db1767f68b6af826781f172dd.jpg](../icml_results/5657_Bridging%20Structure%20and%20Semantics_%20Uncertainty-Modulated%20Dual-Path%20Diffusion%20for%20Robust%20Text-Attribut/images/dfe88cd982a2f0128893f4153e237f5370a4d21db1767f68b6af826781f172dd.jpg)

### Tables

![093d107164f06b9ccdcd9a5d1a02cfb5ddd767df67b0dbd2ef7aece3c6803871.jpg](../icml_results/5657_Bridging%20Structure%20and%20Semantics_%20Uncertainty-Modulated%20Dual-Path%20Diffusion%20for%20Robust%20Text-Attribut/tables/093d107164f06b9ccdcd9a5d1a02cfb5ddd767df67b0dbd2ef7aece3c6803871.jpg)

![3ed0633249055e238b3d49cf2dc23bd0a1fe4a5210d7e495c6667231c3721c34.jpg](../icml_results/5657_Bridging%20Structure%20and%20Semantics_%20Uncertainty-Modulated%20Dual-Path%20Diffusion%20for%20Robust%20Text-Attribut/tables/3ed0633249055e238b3d49cf2dc23bd0a1fe4a5210d7e495c6667231c3721c34.jpg)

![47e0faf892e558e5f29343b03da66e9b84e6d0367a91112a68fe233a9d7eb8df.jpg](../icml_results/5657_Bridging%20Structure%20and%20Semantics_%20Uncertainty-Modulated%20Dual-Path%20Diffusion%20for%20Robust%20Text-Attribut/tables/47e0faf892e558e5f29343b03da66e9b84e6d0367a91112a68fe233a9d7eb8df.jpg)

![98ee612978b8d15ef00591d01139912eb9b51e5e1ce4dfc5f9615a647e4aacfe.jpg](../icml_results/5657_Bridging%20Structure%20and%20Semantics_%20Uncertainty-Modulated%20Dual-Path%20Diffusion%20for%20Robust%20Text-Attribut/tables/98ee612978b8d15ef00591d01139912eb9b51e5e1ce4dfc5f9615a647e4aacfe.jpg)

<a id="paper-5609"></a>

## Evaluating AI Grading on Real-World Handwritten College Mathematics: A Large-Scale Study Toward a Benchmark


### Images

![05a18ba9ebebbac1bb7a9d068bd52979ac049c1b5614f6a5861a58c2f7ab96c0.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/images/05a18ba9ebebbac1bb7a9d068bd52979ac049c1b5614f6a5861a58c2f7ab96c0.jpg)

![12163ae2ca9ab32033355b143d1b2ac2997ab6395535c37ffe3f6d798c5233f8.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/images/12163ae2ca9ab32033355b143d1b2ac2997ab6395535c37ffe3f6d798c5233f8.jpg)

![14b0a2e013d720c773e4700caf0bc96a10c9b1d4a397a719ece9a9dd7257eb0e.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/images/14b0a2e013d720c773e4700caf0bc96a10c9b1d4a397a719ece9a9dd7257eb0e.jpg)

![365c3ae60189ea32f4ba90b11dd5ef229b23cd1c257df6a11386a0543c10d773.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/images/365c3ae60189ea32f4ba90b11dd5ef229b23cd1c257df6a11386a0543c10d773.jpg)

![3d4f828af2786b19b0202ac4c43ad24983314a40dd18f2688bb560353bcc80c3.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/images/3d4f828af2786b19b0202ac4c43ad24983314a40dd18f2688bb560353bcc80c3.jpg)

![402764dda7abe975753b017d58930e344463b026b586d54ba411f6d3d3a243ce.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/images/402764dda7abe975753b017d58930e344463b026b586d54ba411f6d3d3a243ce.jpg)

![46e342a178373edaf136c6a27d185692aa703aa619daecf88541d5fceed914e2.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/images/46e342a178373edaf136c6a27d185692aa703aa619daecf88541d5fceed914e2.jpg)

![4a041e3aad3e762517438196bdbfea1070d25c42a698dd372f96a0e9aa7096f3.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/images/4a041e3aad3e762517438196bdbfea1070d25c42a698dd372f96a0e9aa7096f3.jpg)

![55033272f7d773de09a8a111f4702e24a25a6e6207c735a8c48a62fb05b4f693.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/images/55033272f7d773de09a8a111f4702e24a25a6e6207c735a8c48a62fb05b4f693.jpg)

![5ae92d40e7c8977714f99778b201aea323714b5140e24e7768324b805c674b91.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/images/5ae92d40e7c8977714f99778b201aea323714b5140e24e7768324b805c674b91.jpg)

![5d50cd2e1229a42d51dcb5ffd620fa4d87a25af30637842b0867f6cd7e047765.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/images/5d50cd2e1229a42d51dcb5ffd620fa4d87a25af30637842b0867f6cd7e047765.jpg)

![7726340b8a4241ea03aa74dab0a2e3af6a05b05f7a1db8d79c7b0f8f67fc38cf.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/images/7726340b8a4241ea03aa74dab0a2e3af6a05b05f7a1db8d79c7b0f8f67fc38cf.jpg)

![7a64e18507de718a192e149de271bd153253438a3e5f6da30894a544a4c76afa.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/images/7a64e18507de718a192e149de271bd153253438a3e5f6da30894a544a4c76afa.jpg)

![85dd95d7796a7f0dd78dce89d32dd1bc4ab8d72e3856ba3b661bc193f71771ff.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/images/85dd95d7796a7f0dd78dce89d32dd1bc4ab8d72e3856ba3b661bc193f71771ff.jpg)

![ce101ac1d870aee611a3a769fd6a36adb10bab997df2b1bd1a6496c5eaf5f847.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/images/ce101ac1d870aee611a3a769fd6a36adb10bab997df2b1bd1a6496c5eaf5f847.jpg)

![e40d38f3283e3638771cca968f3767ac0bef26baeba7aef6439228e578566d50.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/images/e40d38f3283e3638771cca968f3767ac0bef26baeba7aef6439228e578566d50.jpg)

![f168e5bb1dc9ef1bb20c8fda94b82daea301ae901e5df4b0e7daf044391a055f.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/images/f168e5bb1dc9ef1bb20c8fda94b82daea301ae901e5df4b0e7daf044391a055f.jpg)

### Tables

![179e9fbe8a9a6e95c79f266883aaf0d007363e1c77409d11803efe0f986fce85.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/tables/179e9fbe8a9a6e95c79f266883aaf0d007363e1c77409d11803efe0f986fce85.jpg)

![19f17d6e55b573affd059c24a9ecaddbcb14ee62cb0e1c3b30819267465635bc.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/tables/19f17d6e55b573affd059c24a9ecaddbcb14ee62cb0e1c3b30819267465635bc.jpg)

![23392de15edab04fa79f8b72eef38e180809bc1d391cea7ef35221ed8b250623.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/tables/23392de15edab04fa79f8b72eef38e180809bc1d391cea7ef35221ed8b250623.jpg)

![30a741d63a3457f3e1237dc99bc228936e51f9b25c7343a3da4299e3636d49fd.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/tables/30a741d63a3457f3e1237dc99bc228936e51f9b25c7343a3da4299e3636d49fd.jpg)

![459606908ba6bd18e248ca8a61826356e043bccc4c683eb4f8a39ea4553cc6d6.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/tables/459606908ba6bd18e248ca8a61826356e043bccc4c683eb4f8a39ea4553cc6d6.jpg)

![6914c65a5ccef29c2c5b031c84dbb80db04b6aa9953318126646ab8836ced656.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/tables/6914c65a5ccef29c2c5b031c84dbb80db04b6aa9953318126646ab8836ced656.jpg)

![7d4edefdff9b5887ec32c53b020513ca1d594e4fcf9817d10c424360382317fa.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/tables/7d4edefdff9b5887ec32c53b020513ca1d594e4fcf9817d10c424360382317fa.jpg)

![82be61989867f0d0ea009f61ce3147e0ac501a8c9152449a0fb66ec0f1a9dbef.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/tables/82be61989867f0d0ea009f61ce3147e0ac501a8c9152449a0fb66ec0f1a9dbef.jpg)

![c6c03e6046fc707127455a811b9f6a43b8bd6b947f9a1fb7a38eae37dece2260.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/tables/c6c03e6046fc707127455a811b9f6a43b8bd6b947f9a1fb7a38eae37dece2260.jpg)

![d9de5cc90815f86fd85e2f8d6e4f772daaa42c780cc6609bb39e5835df0b3412.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/tables/d9de5cc90815f86fd85e2f8d6e4f772daaa42c780cc6609bb39e5835df0b3412.jpg)

![eb2fadc72669f0f1027ad3f0ed470ff506bd5f9242c0146b6985599dfc05e262.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/tables/eb2fadc72669f0f1027ad3f0ed470ff506bd5f9242c0146b6985599dfc05e262.jpg)

![ebf28b5203ea1bf4dcc0db8184bafdc709f12814a5b9aebf8ef9884bdb6389ab.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/tables/ebf28b5203ea1bf4dcc0db8184bafdc709f12814a5b9aebf8ef9884bdb6389ab.jpg)

![f556bd5035a4b913204de3ff6ddc52ce0d072aadd1b20a33fa189755e8611779.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/tables/f556bd5035a4b913204de3ff6ddc52ce0d072aadd1b20a33fa189755e8611779.jpg)

![fd05024780f7b49e2a351e671b6b093a8da75c11822428b622724055230b0aa1.jpg](../icml_results/5658_Evaluating%20AI%20Grading%20on%20Real-World%20Handwritten%20College%20Mathematics_%20A%20Large-Scale%20Study%20Toward%20a%20Be/tables/fd05024780f7b49e2a351e671b6b093a8da75c11822428b622724055230b0aa1.jpg)

<a id="paper-5610"></a>

## Fine-to-Coarse Fairness-Informed Multi-View Clustering

### Images

![1382823f7e45187812da3920b576d61a952f6c6127b0aed213123b28b2e5a2c3.jpg](../icml_results/5659_Fine-to-Coarse%20Fairness-Informed%20Multi-View%20Clustering/images/1382823f7e45187812da3920b576d61a952f6c6127b0aed213123b28b2e5a2c3.jpg)

![3824b36950eeca85030807691d194d1b93909195fefbd3a5e091a6bb2829d221.jpg](../icml_results/5659_Fine-to-Coarse%20Fairness-Informed%20Multi-View%20Clustering/images/3824b36950eeca85030807691d194d1b93909195fefbd3a5e091a6bb2829d221.jpg)

![4b38fc635868c542587c432410f128809c52d9c759b8fb64c6dd7ba3be1dc67f.jpg](../icml_results/5659_Fine-to-Coarse%20Fairness-Informed%20Multi-View%20Clustering/images/4b38fc635868c542587c432410f128809c52d9c759b8fb64c6dd7ba3be1dc67f.jpg)

![4e732ea300d0315a30331750960b52b63d769d3f452ade34f0ec2407a223a1bf.jpg](../icml_results/5659_Fine-to-Coarse%20Fairness-Informed%20Multi-View%20Clustering/images/4e732ea300d0315a30331750960b52b63d769d3f452ade34f0ec2407a223a1bf.jpg)

![526e843ffa815eb5fa8f132bf307b5dad25bc625518ee5f2a5abaf150b6e550f.jpg](../icml_results/5659_Fine-to-Coarse%20Fairness-Informed%20Multi-View%20Clustering/images/526e843ffa815eb5fa8f132bf307b5dad25bc625518ee5f2a5abaf150b6e550f.jpg)

![55897878d52071924f723f4ada6ba64ba0a8769649844383a478f61aa5166208.jpg](../icml_results/5659_Fine-to-Coarse%20Fairness-Informed%20Multi-View%20Clustering/images/55897878d52071924f723f4ada6ba64ba0a8769649844383a478f61aa5166208.jpg)

![5f1b9b25b539cf6776aa57f0fe7b72b7db1ec554b8206ddfa188b74f5a475955.jpg](../icml_results/5659_Fine-to-Coarse%20Fairness-Informed%20Multi-View%20Clustering/images/5f1b9b25b539cf6776aa57f0fe7b72b7db1ec554b8206ddfa188b74f5a475955.jpg)

![deb80d68e618d7d1017b297d2019ce3a18f949f2b98fc06d20cda0839e620429.jpg](../icml_results/5659_Fine-to-Coarse%20Fairness-Informed%20Multi-View%20Clustering/images/deb80d68e618d7d1017b297d2019ce3a18f949f2b98fc06d20cda0839e620429.jpg)

![e0526d4dd77aaca3dc0cad2dfbbf827a37d8422db4cd36ad8232b74f6d6b1a79.jpg](../icml_results/5659_Fine-to-Coarse%20Fairness-Informed%20Multi-View%20Clustering/images/e0526d4dd77aaca3dc0cad2dfbbf827a37d8422db4cd36ad8232b74f6d6b1a79.jpg)

![e8571c0af5fe0565d7065f8e7a5d1f4010526b0ae47c9e83dff28cfc90162391.jpg](../icml_results/5659_Fine-to-Coarse%20Fairness-Informed%20Multi-View%20Clustering/images/e8571c0af5fe0565d7065f8e7a5d1f4010526b0ae47c9e83dff28cfc90162391.jpg)

### Tables

![64e342035bcf179a9484af424f230371c68799814ecd7b344a0b12e42bb1022b.jpg](../icml_results/5659_Fine-to-Coarse%20Fairness-Informed%20Multi-View%20Clustering/tables/64e342035bcf179a9484af424f230371c68799814ecd7b344a0b12e42bb1022b.jpg)

![837f295f56648be663255b63717ebb484891f77f8989f56f8b94dc750b448309.jpg](../icml_results/5659_Fine-to-Coarse%20Fairness-Informed%20Multi-View%20Clustering/tables/837f295f56648be663255b63717ebb484891f77f8989f56f8b94dc750b448309.jpg)

![90d40b4c9f9860cfcddb4feee95a76dd035ddb92a5666ef46ff64af50fef1c35.jpg](../icml_results/5659_Fine-to-Coarse%20Fairness-Informed%20Multi-View%20Clustering/tables/90d40b4c9f9860cfcddb4feee95a76dd035ddb92a5666ef46ff64af50fef1c35.jpg)

![c7d76b6940110ed255b0088c424cc1085d52357f10ee435253514ae0d40bcd63.jpg](../icml_results/5659_Fine-to-Coarse%20Fairness-Informed%20Multi-View%20Clustering/tables/c7d76b6940110ed255b0088c424cc1085d52357f10ee435253514ae0d40bcd63.jpg)
