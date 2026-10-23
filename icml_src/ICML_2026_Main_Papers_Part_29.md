# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 29 of 197

## 目录 (Table of Contents)

1. [The Geometry of Projection Heads: Conditioning, Invariance, and Collapse](#paper-0925)
2. [Independent Component Discovery in Temporal Count Data](#paper-0926)
3. [Token-Efficient Change Detection in LLM APIs](#paper-0927)
4. [NanoFLUX: Distillation-Driven Compression of Large Text-to-Image Generation Models for Mobile Devices](#paper-0928)
5. [MetaOthello: A Controlled Study of Multiple World Models in Transformers](#paper-0929)
6. [Iterated Population Based Training with Task-Agnostic Restarts](#paper-0930)
7. [Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis](#paper-0931)
8. [Scalable Kronecker-Factored Fisher Approximation for Neural Network Parameter Sensitivity](#paper-0932)
9. [Adapting Noise to Data: Generative Flows from Learned 1D Processes](#paper-0933)
10. [Differentially Private Range Subgraph Counting](#paper-0934)
11. [OpenGPT-4o-Image: A Comprehensive Dataset for Advanced Image Generation and Editing](#paper-0935)
12. [NOMAD: Lifelong Trajectory Planning via Non-Parametric Bayesian Memory-Adaptive Diffusion Experts](#paper-0936)
13. [Proximal-IMH: Proximal Posterior Proposals for Independent Metropolis–Hastings with Approximate Operators](#paper-0937)
14. [FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion](#paper-0938)
15. [Vector Linking via Cross-Model Local Isometric Consistency](#paper-0939)
16. [PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making](#paper-0940)
17. [RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation](#paper-0941)
18. [PowerFlow: Unlocking the Dual Nature of LLMs via Principled Distribution Matching](#paper-0942)
19. [Task-Aware Structured Memory for Dynamic Multi-modal In-Context Learning](#paper-0943)
20. [Mind the Gap: Catching Hallucinations via Evidence Drop on the Reasoning Manifold](#paper-0944)
21. [OPT-Engine: Benchmarking the Limits of LLMs in Optimization Modeling via Complexity Scaling](#paper-0945)
22. [L-CUBE: Isolating Long-Context Capacity from Knowledge with Controllable Mutual Information Scaling](#paper-0946)
23. [MARS: Modular Agent with Reflective Search for Automated AI Research](#paper-0947)
24. [When Preference Labels Fall Short: Aligning Diffusion Models from Real Data](#paper-0948)
25. [Test-Time Learning of Causal Structure from Interventional Data](#paper-0949)
26. [Foundation VAEs for CT Reconstruction, Augmentation, and Generation](#paper-0950)
27. [PRISM: Training-Free Video Anomaly Detection via Intrinsic Statistical Modeling](#paper-0951)
28. [Adaptive Probe-based Steering for Robust LLM Jailbreaking](#paper-0952)
29. [SGERA: Stein-Guided ECG-Report Alignment for ECG Representation Learning](#paper-0953)
30. [WUSH: Near-Optimal Adaptive Transforms for LLM Quantization](#paper-0954)
31. [StormInsight: Hierarchical Environmental Forcing and Vertical Coupling for Convective Systems Evolution](#paper-0955)
32. [PhyScene3D: Physically Consistent Interactive 3D Tabletop Scene Generation](#paper-0956)
33. [Let the Prototype Guide You: Robust Aggregation of Sparse Multi-Class Annotations via Annotator Prototype Learning](#paper-0957)

---


<a id="paper-0925"></a>

## The Geometry of Projection Heads: Conditioning, Invariance, and Collapse

### Images

![0180ce0b4c0c8b0906983a8d640491d3ae6d343eb68b8a03cb809766257b03d0.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/0180ce0b4c0c8b0906983a8d640491d3ae6d343eb68b8a03cb809766257b03d0.jpg)

![199c9a84d3f854f0875af3ff81bb8e23eb77496a12f6f83f7bdc084d8c0b5485.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/199c9a84d3f854f0875af3ff81bb8e23eb77496a12f6f83f7bdc084d8c0b5485.jpg)

![2aed4260ebdb0690e9985e42b2d0392c4e8c962303c844fdd052405429ec65da.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/2aed4260ebdb0690e9985e42b2d0392c4e8c962303c844fdd052405429ec65da.jpg)

![3a8f1968639772e3162c260b7bb514f9007be0a3d0a652e079548478a0e5c433.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/3a8f1968639772e3162c260b7bb514f9007be0a3d0a652e079548478a0e5c433.jpg)

![4fdf527141ba7dd1057844e5541e958fbb8ac9575662098bbc9463e0b8f3cac3.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/4fdf527141ba7dd1057844e5541e958fbb8ac9575662098bbc9463e0b8f3cac3.jpg)

![50a81b8a6c4b71fa7c3feb40a4946af994015afb8a71717ce01e92caf493eaa2.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/50a81b8a6c4b71fa7c3feb40a4946af994015afb8a71717ce01e92caf493eaa2.jpg)

![518c03f13a89432fe4df624e13b12ebf2eacc225b43f36f0f28ec0d43636cb09.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/518c03f13a89432fe4df624e13b12ebf2eacc225b43f36f0f28ec0d43636cb09.jpg)

![65edef592cc14e73af398fecc22b5645018100dd3aac9c3b599acb24378316dc.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/65edef592cc14e73af398fecc22b5645018100dd3aac9c3b599acb24378316dc.jpg)

![66237e86c329da5ac1f536bc76008c6412b914cf6c932c9917d53ae97812a591.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/66237e86c329da5ac1f536bc76008c6412b914cf6c932c9917d53ae97812a591.jpg)

![a87341bc0c5be727faec43ccf74b43b1c03aa8794093f1280f28f4ef3928b1a1.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/a87341bc0c5be727faec43ccf74b43b1c03aa8794093f1280f28f4ef3928b1a1.jpg)

![bfa3c6f43bc343eece93aca937c55302ede101004e095381d4edd75722267ef6.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/bfa3c6f43bc343eece93aca937c55302ede101004e095381d4edd75722267ef6.jpg)

![d409aa96327b35012a8860d8e3d3610ad27b84bad8e36e14117ea0bf8c45fd4c.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/d409aa96327b35012a8860d8e3d3610ad27b84bad8e36e14117ea0bf8c45fd4c.jpg)

![ef56a1c3c17b0b3aae2c9b388a209be5feffdff9575f04ab19be2e07c5e3d6f3.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/images/ef56a1c3c17b0b3aae2c9b388a209be5feffdff9575f04ab19be2e07c5e3d6f3.jpg)

### Tables

![1df1ecc8f36ec6ac2f6636aa941b804b8cfa7b217a982b3e460a1eb0d4c77933.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/tables/1df1ecc8f36ec6ac2f6636aa941b804b8cfa7b217a982b3e460a1eb0d4c77933.jpg)

![ef82468835d5d72009fa07c79cc342bca2257e4f586e1c61d024d02993602680.jpg](../icml_results/941_Blending%20Neural%20Control%20Density%20Functions%20for%20Stabilization%20and%20Safety/tables/ef82468835d5d72009fa07c79cc342bca2257e4f586e1c61d024d02993602680.jpg)

## The Geometry of Projection Heads: Conditioning, Invariance, and Collapse


### Images

![037a3d73b288f803b45c6aa7f841a587a2135b0ca4aa123e3bda48001d8dc223.jpg](../icml_results/942_The%20Geometry%20of%20Projection%20Heads_%20Conditioning%2C%20Invariance%2C%20and%20Collapse/images/037a3d73b288f803b45c6aa7f841a587a2135b0ca4aa123e3bda48001d8dc223.jpg)

![0c47ed026e79896f72c5fe77484c6ae4a08a1edb9e17ae0db329930e320582f7.jpg](../icml_results/942_The%20Geometry%20of%20Projection%20Heads_%20Conditioning%2C%20Invariance%2C%20and%20Collapse/images/0c47ed026e79896f72c5fe77484c6ae4a08a1edb9e17ae0db329930e320582f7.jpg)

![0dfd9eb56947a43f09094329a10d10471b3d3618d137f196230eb27a91b3342b.jpg](../icml_results/942_The%20Geometry%20of%20Projection%20Heads_%20Conditioning%2C%20Invariance%2C%20and%20Collapse/images/0dfd9eb56947a43f09094329a10d10471b3d3618d137f196230eb27a91b3342b.jpg)

![1189fe6022ed8015b3b49df6d8056f40904d29cfb215f8b1e1e03eae7075e49c.jpg](../icml_results/942_The%20Geometry%20of%20Projection%20Heads_%20Conditioning%2C%20Invariance%2C%20and%20Collapse/images/1189fe6022ed8015b3b49df6d8056f40904d29cfb215f8b1e1e03eae7075e49c.jpg)

![60f6e6736f7b062aa8165b78a465d854b5829cfd81ad063b17ae75cb133ecf84.jpg](../icml_results/942_The%20Geometry%20of%20Projection%20Heads_%20Conditioning%2C%20Invariance%2C%20and%20Collapse/images/60f6e6736f7b062aa8165b78a465d854b5829cfd81ad063b17ae75cb133ecf84.jpg)

![bb779fe166f4a9269a22ecec632c1085a314419e91d00737f328e0e8f71f6b69.jpg](../icml_results/942_The%20Geometry%20of%20Projection%20Heads_%20Conditioning%2C%20Invariance%2C%20and%20Collapse/images/bb779fe166f4a9269a22ecec632c1085a314419e91d00737f328e0e8f71f6b69.jpg)

![c90087295368066e1f3cae197c011c98e3d1310704a6b174e8f5d85cfadf9bf2.jpg](../icml_results/942_The%20Geometry%20of%20Projection%20Heads_%20Conditioning%2C%20Invariance%2C%20and%20Collapse/images/c90087295368066e1f3cae197c011c98e3d1310704a6b174e8f5d85cfadf9bf2.jpg)

![d033aacb68104b0e233280d76b55dfbef06aa4f35ee25d6f750ee19135bee954.jpg](../icml_results/942_The%20Geometry%20of%20Projection%20Heads_%20Conditioning%2C%20Invariance%2C%20and%20Collapse/images/d033aacb68104b0e233280d76b55dfbef06aa4f35ee25d6f750ee19135bee954.jpg)

### Tables

![41fba25bcb8ba77a41ee563e45e80e98148d50d91d2baa32bbd04554ed026639.jpg](../icml_results/942_The%20Geometry%20of%20Projection%20Heads_%20Conditioning%2C%20Invariance%2C%20and%20Collapse/tables/41fba25bcb8ba77a41ee563e45e80e98148d50d91d2baa32bbd04554ed026639.jpg)

![48910dbdaac3598231ca19275c204369994af8252cd6cc6038393a656bb879d8.jpg](../icml_results/942_The%20Geometry%20of%20Projection%20Heads_%20Conditioning%2C%20Invariance%2C%20and%20Collapse/tables/48910dbdaac3598231ca19275c204369994af8252cd6cc6038393a656bb879d8.jpg)

![99a15400298f7a5919fa5ae9d8c0e601a232600fb77ed6c21a2b74bc8d9a3cd6.jpg](../icml_results/942_The%20Geometry%20of%20Projection%20Heads_%20Conditioning%2C%20Invariance%2C%20and%20Collapse/tables/99a15400298f7a5919fa5ae9d8c0e601a232600fb77ed6c21a2b74bc8d9a3cd6.jpg)

![a6a6eeb9d12c5e5f7e33787ee40c74dedb82a57cb461ddf8c3b6b0ea8c354ac4.jpg](../icml_results/942_The%20Geometry%20of%20Projection%20Heads_%20Conditioning%2C%20Invariance%2C%20and%20Collapse/tables/a6a6eeb9d12c5e5f7e33787ee40c74dedb82a57cb461ddf8c3b6b0ea8c354ac4.jpg)

![e6b8b7f65f51292aafcc144d898190d1086c48d9686d8b71818ce21957eb1215.jpg](../icml_results/942_The%20Geometry%20of%20Projection%20Heads_%20Conditioning%2C%20Invariance%2C%20and%20Collapse/tables/e6b8b7f65f51292aafcc144d898190d1086c48d9686d8b71818ce21957eb1215.jpg)

![eda98c507eb79b2b461d4a20c6606e726e4f356c39d1d9c88f3e369c7434d7e6.jpg](../icml_results/942_The%20Geometry%20of%20Projection%20Heads_%20Conditioning%2C%20Invariance%2C%20and%20Collapse/tables/eda98c507eb79b2b461d4a20c6606e726e4f356c39d1d9c88f3e369c7434d7e6.jpg)

![f654885037ce4cf3334f80d7bb26def8e5ffa6cf9f191be274ff933085bc2de7.jpg](../icml_results/942_The%20Geometry%20of%20Projection%20Heads_%20Conditioning%2C%20Invariance%2C%20and%20Collapse/tables/f654885037ce4cf3334f80d7bb26def8e5ffa6cf9f191be274ff933085bc2de7.jpg)

<a id="paper-0926"></a>

## Independent Component Discovery in Temporal Count Data


### Images

![31c773e7dcd745e5bbfc534ac5872f1cfa8a97abe2e0db84fa001e50905da908.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/31c773e7dcd745e5bbfc534ac5872f1cfa8a97abe2e0db84fa001e50905da908.jpg)

![326daff41dd3e99073a99f9718098d9d06471adc62cb9b630edba9c646035bf0.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/326daff41dd3e99073a99f9718098d9d06471adc62cb9b630edba9c646035bf0.jpg)

![3cc416d438e44f8b3a858b6db2cdf3e90eb1468843f116f5da830f070e545d6f.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/3cc416d438e44f8b3a858b6db2cdf3e90eb1468843f116f5da830f070e545d6f.jpg)

![495968ae1603a8bbd8b41f12fa5706b9007e818f28e13bd114cb2aedbb234b42.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/495968ae1603a8bbd8b41f12fa5706b9007e818f28e13bd114cb2aedbb234b42.jpg)

![51dd188d815c953a8c8199bc91d34a4cbbe91676080030ce26793a446a160883.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/51dd188d815c953a8c8199bc91d34a4cbbe91676080030ce26793a446a160883.jpg)

![6cbc190729d585c150dc7cfbb6118c5c3cacf67c992ae06959a022609714c1bb.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/6cbc190729d585c150dc7cfbb6118c5c3cacf67c992ae06959a022609714c1bb.jpg)

![6f04ca521576d66874e449aab5698d7754cee3c05e2e089620cacef4d889e932.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/6f04ca521576d66874e449aab5698d7754cee3c05e2e089620cacef4d889e932.jpg)

![709b60d018d5c72c53425a8c84d924d8b71577b6a2e7191b5e95f6beaf427185.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/709b60d018d5c72c53425a8c84d924d8b71577b6a2e7191b5e95f6beaf427185.jpg)

![77fc3e3ba86a2f5c94c63282722456f4fc0900f96c63f52f5b4d2b12d7c0b8f9.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/77fc3e3ba86a2f5c94c63282722456f4fc0900f96c63f52f5b4d2b12d7c0b8f9.jpg)

![82e1eda0554b05c828b5c924c993df2a03b944720e2d9576580fd07e2ace1b60.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/82e1eda0554b05c828b5c924c993df2a03b944720e2d9576580fd07e2ace1b60.jpg)

![89c4d83bddc6ce8c929c88c8729dabc4b6951a7a8c83c63cfa030a9140b6b699.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/89c4d83bddc6ce8c929c88c8729dabc4b6951a7a8c83c63cfa030a9140b6b699.jpg)

![8aae1a0b4a0531cfce09cc909ceca7a48c47a6640b09c527af916478188b9c4e.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/8aae1a0b4a0531cfce09cc909ceca7a48c47a6640b09c527af916478188b9c4e.jpg)

![9220095d19974f1a2272568e850082497b63a803d7d1d1bfd263a880628080f9.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/9220095d19974f1a2272568e850082497b63a803d7d1d1bfd263a880628080f9.jpg)

![9be0d8bd6bb602b323ac0342b5af680b48e48fdb5c96b181b3172ab5f9cad3dc.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/9be0d8bd6bb602b323ac0342b5af680b48e48fdb5c96b181b3172ab5f9cad3dc.jpg)

![aa22fcf4b3aea6e986d91374b1cf39a1ff3cc4149f6ac1f6c1e2cdb9dc0e1dfe.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/aa22fcf4b3aea6e986d91374b1cf39a1ff3cc4149f6ac1f6c1e2cdb9dc0e1dfe.jpg)

![aefe7c9daed9de044f1a8b60fe5a83444bcaf58dd75956cf5791a756eea36f65.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/aefe7c9daed9de044f1a8b60fe5a83444bcaf58dd75956cf5791a756eea36f65.jpg)

![b13b918adcbc70bd4298ca9eb61dcb72ca2adef46b504a3c3af919bd721af192.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/b13b918adcbc70bd4298ca9eb61dcb72ca2adef46b504a3c3af919bd721af192.jpg)

![b4f4311dd61d3f23543883af48ba081dd1f39ebb9083be366de03f86d6c034bd.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/b4f4311dd61d3f23543883af48ba081dd1f39ebb9083be366de03f86d6c034bd.jpg)

![bbe28015a0834ccc3f86bf4f2269cf21b2272a71e4bac32a42ba5debe933f2c9.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/bbe28015a0834ccc3f86bf4f2269cf21b2272a71e4bac32a42ba5debe933f2c9.jpg)

![c0947b5a72faab5c360d0418add69a7e0cc9ed316caa061a73faf0e8ceaf571e.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/c0947b5a72faab5c360d0418add69a7e0cc9ed316caa061a73faf0e8ceaf571e.jpg)

![c5bc2d073ef25a643c33fdc312cfe73ab79700c22ec0e506d45afb5471e36662.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/c5bc2d073ef25a643c33fdc312cfe73ab79700c22ec0e506d45afb5471e36662.jpg)

![e8f687090c7a8d21c98e5c3159f719082f0de0d5b61e6fd8f312c1c2ccf42874.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/images/e8f687090c7a8d21c98e5c3159f719082f0de0d5b61e6fd8f312c1c2ccf42874.jpg)

### Tables

![a28cd5a9fa9506a3b72a9f9e15ad7eff4cec32bc77ed3b2ea8abfe8bf9a76e66.jpg](../icml_results/943_Independent%20Component%20Discovery%20in%20Temporal%20Count%20Data/tables/a28cd5a9fa9506a3b72a9f9e15ad7eff4cec32bc77ed3b2ea8abfe8bf9a76e66.jpg)

<a id="paper-0927"></a>

## Token-Efficient Change Detection in LLM APIs


### Images

![0206d96b502f51e9c36c33c058769d53a3e33b07420b514a9c100b24810c1ad7.jpg](../icml_results/944_Token-Efficient%20Change%20Detection%20in%20LLM%20APIs/images/0206d96b502f51e9c36c33c058769d53a3e33b07420b514a9c100b24810c1ad7.jpg)

![2e5eff8b245e532601c35ae5e2df599c645b5b3a248521ff9993111965a419e7.jpg](../icml_results/944_Token-Efficient%20Change%20Detection%20in%20LLM%20APIs/images/2e5eff8b245e532601c35ae5e2df599c645b5b3a248521ff9993111965a419e7.jpg)

![3b4bb7d53d0176c8c37acd08ffaf3fc447c6e8b15ddc1adfd6b6c8e46e466728.jpg](../icml_results/944_Token-Efficient%20Change%20Detection%20in%20LLM%20APIs/images/3b4bb7d53d0176c8c37acd08ffaf3fc447c6e8b15ddc1adfd6b6c8e46e466728.jpg)

![7a1f3a1afe7f519dc7c302bc3c4443ad7fdc8298656baf6e5fa9cb6b5329bfa1.jpg](../icml_results/944_Token-Efficient%20Change%20Detection%20in%20LLM%20APIs/images/7a1f3a1afe7f519dc7c302bc3c4443ad7fdc8298656baf6e5fa9cb6b5329bfa1.jpg)

![952bb4604383abfff8fed91f9d00fc920ae7dc2760c1a30c677f91772fc15ccb.jpg](../icml_results/944_Token-Efficient%20Change%20Detection%20in%20LLM%20APIs/images/952bb4604383abfff8fed91f9d00fc920ae7dc2760c1a30c677f91772fc15ccb.jpg)

![af89fc1d371f5943454c8aa76124cb13a52451c308f0361e1e86ee430fc903d6.jpg](../icml_results/944_Token-Efficient%20Change%20Detection%20in%20LLM%20APIs/images/af89fc1d371f5943454c8aa76124cb13a52451c308f0361e1e86ee430fc903d6.jpg)

![cbeec8e0bb273f34ef53dab097b96c7c935b86e77e6697c2d0a4182d14731b57.jpg](../icml_results/944_Token-Efficient%20Change%20Detection%20in%20LLM%20APIs/images/cbeec8e0bb273f34ef53dab097b96c7c935b86e77e6697c2d0a4182d14731b57.jpg)

![d416a81b1e1f0fb3b7e26ebc4c8a95ff643c7354d1371e52ac14835949e960b4.jpg](../icml_results/944_Token-Efficient%20Change%20Detection%20in%20LLM%20APIs/images/d416a81b1e1f0fb3b7e26ebc4c8a95ff643c7354d1371e52ac14835949e960b4.jpg)

![e558e8f13b2406e52f75264405b12e6a4ff9bf26b7f160ef221dfec1e4e5b03b.jpg](../icml_results/944_Token-Efficient%20Change%20Detection%20in%20LLM%20APIs/images/e558e8f13b2406e52f75264405b12e6a4ff9bf26b7f160ef221dfec1e4e5b03b.jpg)

![e7714772362a6fbfacbd53b9ca791ceec578b0458be1f4dcd6d8f812d0e8c39e.jpg](../icml_results/944_Token-Efficient%20Change%20Detection%20in%20LLM%20APIs/images/e7714772362a6fbfacbd53b9ca791ceec578b0458be1f4dcd6d8f812d0e8c39e.jpg)

### Tables

![0d719665cb0262f59b7d6552d1683b799a247605a6b3a98a4128f97286cd38e4.jpg](../icml_results/944_Token-Efficient%20Change%20Detection%20in%20LLM%20APIs/tables/0d719665cb0262f59b7d6552d1683b799a247605a6b3a98a4128f97286cd38e4.jpg)

![726231da017b3c9fcfb11e6e78ade790018ec01795af5979aad18868f4fabbdd.jpg](../icml_results/944_Token-Efficient%20Change%20Detection%20in%20LLM%20APIs/tables/726231da017b3c9fcfb11e6e78ade790018ec01795af5979aad18868f4fabbdd.jpg)

![92261c8aa7e54ef1c4df0c1432632af483038d524cac4f69fd97614c06174469.jpg](../icml_results/944_Token-Efficient%20Change%20Detection%20in%20LLM%20APIs/tables/92261c8aa7e54ef1c4df0c1432632af483038d524cac4f69fd97614c06174469.jpg)

<a id="paper-0928"></a>

## NanoFLUX: Distillation-Driven Compression of Large Text-to-Image Generation Models for Mobile Devices


### Images

![026965e2d66b798e4944b3f02b229d81e9b39053db69fa6c95fc6b9d0fa84539.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/images/026965e2d66b798e4944b3f02b229d81e9b39053db69fa6c95fc6b9d0fa84539.jpg)

![06e865beeea116641290d01bbc39bc2973e0d875f45664840913dfa3da4ace2b.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/images/06e865beeea116641290d01bbc39bc2973e0d875f45664840913dfa3da4ace2b.jpg)

![21d510a3a10f39e03a9544eae4578ec4d0115f7d0dac7da00466d476c17291fc.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/images/21d510a3a10f39e03a9544eae4578ec4d0115f7d0dac7da00466d476c17291fc.jpg)

![2f86a2464a125b1bd15f814dad1eefbb0799878227b086ea13afc23878672223.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/images/2f86a2464a125b1bd15f814dad1eefbb0799878227b086ea13afc23878672223.jpg)

![4d698475964295ac26c4dec4e57e484357216f43a1a0e0f8280c8f0fa16844bb.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/images/4d698475964295ac26c4dec4e57e484357216f43a1a0e0f8280c8f0fa16844bb.jpg)

![58703e1670d14cf3711c0e4e683fdd4566c6c7b2b6c55684fbf05a691bf541a7.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/images/58703e1670d14cf3711c0e4e683fdd4566c6c7b2b6c55684fbf05a691bf541a7.jpg)

![72ed8560597ff0d950da25ca99be164076395849ca81bee727e3b122cfefbab0.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/images/72ed8560597ff0d950da25ca99be164076395849ca81bee727e3b122cfefbab0.jpg)

![8cd246ba83777d270566d38b7f64b2ec71d4caf82c4d74b93ba5e09306daa28a.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/images/8cd246ba83777d270566d38b7f64b2ec71d4caf82c4d74b93ba5e09306daa28a.jpg)

![f6d286ec2abfd81b40023dedfa6c2975879cca9b6e47c5255bdbc5d18253457d.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/images/f6d286ec2abfd81b40023dedfa6c2975879cca9b6e47c5255bdbc5d18253457d.jpg)

![fd2dd594e1617de3891d92a1744e589dadb534e71b23dd1b00fb2076ae4b270b.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/images/fd2dd594e1617de3891d92a1744e589dadb534e71b23dd1b00fb2076ae4b270b.jpg)

### Tables

![3651f551b326e71399be1fc40bf46a9e2e5072e286fb88a498522db83c8aeda7.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/tables/3651f551b326e71399be1fc40bf46a9e2e5072e286fb88a498522db83c8aeda7.jpg)

![381f2167a2d1979020b2762a69fa0ff4532ceebe04d085fe4451a4e4fc9a2ffd.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/tables/381f2167a2d1979020b2762a69fa0ff4532ceebe04d085fe4451a4e4fc9a2ffd.jpg)

![52e2aeeb9ff981049c3268ddc1320fc10269c453a7889f4fe5444f976a2a21cb.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/tables/52e2aeeb9ff981049c3268ddc1320fc10269c453a7889f4fe5444f976a2a21cb.jpg)

![67180976ae74bcd75fd41147eef9fd46fab85b64e41a3132a9a3d4dc00073013.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/tables/67180976ae74bcd75fd41147eef9fd46fab85b64e41a3132a9a3d4dc00073013.jpg)

![6c9ccb7cd22c4059bc836e4a5f36e95d41aad7e8556df86ff3d502be0290f542.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/tables/6c9ccb7cd22c4059bc836e4a5f36e95d41aad7e8556df86ff3d502be0290f542.jpg)

![742f37d9d4df42794199c85139079e32d7fd0b539ce7be5e3d0c561b63e98819.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/tables/742f37d9d4df42794199c85139079e32d7fd0b539ce7be5e3d0c561b63e98819.jpg)

![9fdb488c10d3a58e11b42773862045b1ac7256486d6cd2a48ecc47ad1af84210.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/tables/9fdb488c10d3a58e11b42773862045b1ac7256486d6cd2a48ecc47ad1af84210.jpg)

![a4fc0c7cada0ce95315c24609d2b0365ca2558ed1ea0ba7bbc29ae8d4b4fa486.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/tables/a4fc0c7cada0ce95315c24609d2b0365ca2558ed1ea0ba7bbc29ae8d4b4fa486.jpg)

![b0ad9940334dac6f62992135b77da7ad77cdcff5fb5fa3f74d014f85b675246a.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/tables/b0ad9940334dac6f62992135b77da7ad77cdcff5fb5fa3f74d014f85b675246a.jpg)

![c97bac6a9fc8cdf8aa3a86f84df14bfbf43ba8a9cb6b485347cf5e5dd4313713.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/tables/c97bac6a9fc8cdf8aa3a86f84df14bfbf43ba8a9cb6b485347cf5e5dd4313713.jpg)

![cc7f2af43c2e2d9d91e733995acb66e6b587be9195fd5897bf000633de74585e.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/tables/cc7f2af43c2e2d9d91e733995acb66e6b587be9195fd5897bf000633de74585e.jpg)

![ce336106209a49d4599c12282d9497e5427ffc269fc9faf36c8ab8e3c3deb142.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/tables/ce336106209a49d4599c12282d9497e5427ffc269fc9faf36c8ab8e3c3deb142.jpg)

![de422ea9c20c6c5b989ec16d35436a591bfd95a935cb6c83d8003bb4104b5991.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/tables/de422ea9c20c6c5b989ec16d35436a591bfd95a935cb6c83d8003bb4104b5991.jpg)

![fcbc949fd80ba2f01fc80c8f7f89652117f4e3adecc89d62e2044ee18590d268.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/tables/fcbc949fd80ba2f01fc80c8f7f89652117f4e3adecc89d62e2044ee18590d268.jpg)

![ff0b62c805111b78642f0f588f67031f3d9f9594d34f05e042569cec62f5737e.jpg](../icml_results/945_NanoFLUX_%20Distillation-Driven%20Compression%20of%20Large%20Text-to-Image%20Generation%20Models%20for%20Mobile%20Device/tables/ff0b62c805111b78642f0f588f67031f3d9f9594d34f05e042569cec62f5737e.jpg)

<a id="paper-0929"></a>

## MetaOthello: A Controlled Study of Multiple World Models in Transformers


### Images

![2f96866e79df49eb160593a44f32eb7f2bdb9d884a30e05bdd51eb8ae8c96352.jpg](../icml_results/946_MetaOthello_%20A%20Controlled%20Study%20of%20Multiple%20World%20Models%20in%20Transformers/images/2f96866e79df49eb160593a44f32eb7f2bdb9d884a30e05bdd51eb8ae8c96352.jpg)

![34a66bbaced377c17a2255c68b5cf3d7dfc7f3cdaa53457adb3c4b805edcf0f5.jpg](../icml_results/946_MetaOthello_%20A%20Controlled%20Study%20of%20Multiple%20World%20Models%20in%20Transformers/images/34a66bbaced377c17a2255c68b5cf3d7dfc7f3cdaa53457adb3c4b805edcf0f5.jpg)

![35430ffd9b7e77d406745d86fee07758398448a73cafb7cf1cf69a95a1467f70.jpg](../icml_results/946_MetaOthello_%20A%20Controlled%20Study%20of%20Multiple%20World%20Models%20in%20Transformers/images/35430ffd9b7e77d406745d86fee07758398448a73cafb7cf1cf69a95a1467f70.jpg)

![59892f21dd9290c6fe3775584339660ef7b7df0f54672d6342464683f5df557e.jpg](../icml_results/946_MetaOthello_%20A%20Controlled%20Study%20of%20Multiple%20World%20Models%20in%20Transformers/images/59892f21dd9290c6fe3775584339660ef7b7df0f54672d6342464683f5df557e.jpg)

![7a1d079e8340b4e439e91e41895a89c7d3aec9643ff0f916820b432ec17bf96e.jpg](../icml_results/946_MetaOthello_%20A%20Controlled%20Study%20of%20Multiple%20World%20Models%20in%20Transformers/images/7a1d079e8340b4e439e91e41895a89c7d3aec9643ff0f916820b432ec17bf96e.jpg)

![7dc3e31570eec6ee92af1a19518cca919e86b505d0003b0e02d4ea8996a4b271.jpg](../icml_results/946_MetaOthello_%20A%20Controlled%20Study%20of%20Multiple%20World%20Models%20in%20Transformers/images/7dc3e31570eec6ee92af1a19518cca919e86b505d0003b0e02d4ea8996a4b271.jpg)

![882a39f3d38c5af49ebb3c431eab938d04da5ea36c96de28399268f6d3d552df.jpg](../icml_results/946_MetaOthello_%20A%20Controlled%20Study%20of%20Multiple%20World%20Models%20in%20Transformers/images/882a39f3d38c5af49ebb3c431eab938d04da5ea36c96de28399268f6d3d552df.jpg)

![a4c5e4e031ad2a4b5637c493305779d4fdc7c7b912b47522f6b83f98fb45e630.jpg](../icml_results/946_MetaOthello_%20A%20Controlled%20Study%20of%20Multiple%20World%20Models%20in%20Transformers/images/a4c5e4e031ad2a4b5637c493305779d4fdc7c7b912b47522f6b83f98fb45e630.jpg)

![afe7aee32e728a846939bcc0f7da73ed1db01dd0d4e047bd881b2d4e768fc1bf.jpg](../icml_results/946_MetaOthello_%20A%20Controlled%20Study%20of%20Multiple%20World%20Models%20in%20Transformers/images/afe7aee32e728a846939bcc0f7da73ed1db01dd0d4e047bd881b2d4e768fc1bf.jpg)

![d60a831def4d7271ecedf5d814de6e889b77df35575f903fbb15146c292ab21e.jpg](../icml_results/946_MetaOthello_%20A%20Controlled%20Study%20of%20Multiple%20World%20Models%20in%20Transformers/images/d60a831def4d7271ecedf5d814de6e889b77df35575f903fbb15146c292ab21e.jpg)

![f65d2c52a7f93ed26da01b75e94a569e78bd96136aac34e5e65f23b721275e2f.jpg](../icml_results/946_MetaOthello_%20A%20Controlled%20Study%20of%20Multiple%20World%20Models%20in%20Transformers/images/f65d2c52a7f93ed26da01b75e94a569e78bd96136aac34e5e65f23b721275e2f.jpg)

![fe9004a2b7971b3089a936054cc445ff5383d56c0c6e4323ad653d4c80a1aedf.jpg](../icml_results/946_MetaOthello_%20A%20Controlled%20Study%20of%20Multiple%20World%20Models%20in%20Transformers/images/fe9004a2b7971b3089a936054cc445ff5383d56c0c6e4323ad653d4c80a1aedf.jpg)

### Tables

![8bf21eb591bbce51b58ee1c2dab7b615acc6e21cc2393963842345de9cc96cf0.jpg](../icml_results/946_MetaOthello_%20A%20Controlled%20Study%20of%20Multiple%20World%20Models%20in%20Transformers/tables/8bf21eb591bbce51b58ee1c2dab7b615acc6e21cc2393963842345de9cc96cf0.jpg)

<a id="paper-0930"></a>

## Iterated Population Based Training with Task-Agnostic Restarts


### Images

![159e70a8fa0062a5662b01e621cc1a0adbeeac968aafc583494ced677c65e0e6.jpg](../icml_results/947_Iterated%20Population%20Based%20Training%20with%20Task-Agnostic%20Restarts/images/159e70a8fa0062a5662b01e621cc1a0adbeeac968aafc583494ced677c65e0e6.jpg)

![45ede079216fa14dbcb54d549509af60db2c1dc812554f9dff85cf85ebf9936c.jpg](../icml_results/947_Iterated%20Population%20Based%20Training%20with%20Task-Agnostic%20Restarts/images/45ede079216fa14dbcb54d549509af60db2c1dc812554f9dff85cf85ebf9936c.jpg)

![5668c6b7bab16b36baa203289908fc1bb24b0b07aeeb7e5db06cb6b4c6b9b904.jpg](../icml_results/947_Iterated%20Population%20Based%20Training%20with%20Task-Agnostic%20Restarts/images/5668c6b7bab16b36baa203289908fc1bb24b0b07aeeb7e5db06cb6b4c6b9b904.jpg)

![5780ee1988f1d09397d25e36f6be24634149bdb9ecec1996b3a9c4d9dae31b40.jpg](../icml_results/947_Iterated%20Population%20Based%20Training%20with%20Task-Agnostic%20Restarts/images/5780ee1988f1d09397d25e36f6be24634149bdb9ecec1996b3a9c4d9dae31b40.jpg)

![5df9cd915da00f026e1d1e6cc0e218da1b721c91ff9d33a4a789ce347f31f2fd.jpg](../icml_results/947_Iterated%20Population%20Based%20Training%20with%20Task-Agnostic%20Restarts/images/5df9cd915da00f026e1d1e6cc0e218da1b721c91ff9d33a4a789ce347f31f2fd.jpg)

![88d77678652997530accb97599853eec39eb40587bbd97934a93a4cdc9ea0c32.jpg](../icml_results/947_Iterated%20Population%20Based%20Training%20with%20Task-Agnostic%20Restarts/images/88d77678652997530accb97599853eec39eb40587bbd97934a93a4cdc9ea0c32.jpg)

![8908a723616057ae9bdb0d08bdcb23c9cfc7b145f04d09f2876f07a41ae7973a.jpg](../icml_results/947_Iterated%20Population%20Based%20Training%20with%20Task-Agnostic%20Restarts/images/8908a723616057ae9bdb0d08bdcb23c9cfc7b145f04d09f2876f07a41ae7973a.jpg)

![a56996c4edcbde33d32fe5121fa1e9e52a358d9d7e4ff54c538560a36cb359ff.jpg](../icml_results/947_Iterated%20Population%20Based%20Training%20with%20Task-Agnostic%20Restarts/images/a56996c4edcbde33d32fe5121fa1e9e52a358d9d7e4ff54c538560a36cb359ff.jpg)

![e26e62b67d327dbb9d6adee56a75df686f7eb227fc2eec8e4f6ba0ddb94cd2ae.jpg](../icml_results/947_Iterated%20Population%20Based%20Training%20with%20Task-Agnostic%20Restarts/images/e26e62b67d327dbb9d6adee56a75df686f7eb227fc2eec8e4f6ba0ddb94cd2ae.jpg)

![e3e67ad24d3d389eaa4f254096c54a7280d3f456d0c12fc11842433bcd0d3d2a.jpg](../icml_results/947_Iterated%20Population%20Based%20Training%20with%20Task-Agnostic%20Restarts/images/e3e67ad24d3d389eaa4f254096c54a7280d3f456d0c12fc11842433bcd0d3d2a.jpg)

![ed7fff63fd79549096fb486dd6f478e5b3727e96cf203297fc09f1af1e1dd27c.jpg](../icml_results/947_Iterated%20Population%20Based%20Training%20with%20Task-Agnostic%20Restarts/images/ed7fff63fd79549096fb486dd6f478e5b3727e96cf203297fc09f1af1e1dd27c.jpg)

### Tables

![0c30194f8b3654a583f97dc665ebca4717f2262a4469697f007f938edf7fddea.jpg](../icml_results/947_Iterated%20Population%20Based%20Training%20with%20Task-Agnostic%20Restarts/tables/0c30194f8b3654a583f97dc665ebca4717f2262a4469697f007f938edf7fddea.jpg)

![148b63478db96b2ec23bb1bcace049684cae5f1ba105474edfc45ce093bf396b.jpg](../icml_results/947_Iterated%20Population%20Based%20Training%20with%20Task-Agnostic%20Restarts/tables/148b63478db96b2ec23bb1bcace049684cae5f1ba105474edfc45ce093bf396b.jpg)

![5e13cafbda913c15cb2cdbb6753db665e08dc7ee3c2d3553baed9a4de989b939.jpg](../icml_results/947_Iterated%20Population%20Based%20Training%20with%20Task-Agnostic%20Restarts/tables/5e13cafbda913c15cb2cdbb6753db665e08dc7ee3c2d3553baed9a4de989b939.jpg)

![bedccf334099c8ed525b0450f198ef449abeee8932274c5531e3938a420e99fe.jpg](../icml_results/947_Iterated%20Population%20Based%20Training%20with%20Task-Agnostic%20Restarts/tables/bedccf334099c8ed525b0450f198ef449abeee8932274c5531e3938a420e99fe.jpg)

![cfad4088d805bff4651ba8c7c6b6e5e5d8287ab8e323da242f75236303da338d.jpg](../icml_results/947_Iterated%20Population%20Based%20Training%20with%20Task-Agnostic%20Restarts/tables/cfad4088d805bff4651ba8c7c6b6e5e5d8287ab8e323da242f75236303da338d.jpg)

![d1246adfa5ee3f574e5e74b078c8643f01951f1b7a02b904bf4f663f39682718.jpg](../icml_results/947_Iterated%20Population%20Based%20Training%20with%20Task-Agnostic%20Restarts/tables/d1246adfa5ee3f574e5e74b078c8643f01951f1b7a02b904bf4f663f39682718.jpg)

<a id="paper-0931"></a>

## Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis


### Images

![049f439cefbc57c777ef4e68319ae49d89776fc3d3b22d3364092ab60f888892.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/049f439cefbc57c777ef4e68319ae49d89776fc3d3b22d3364092ab60f888892.jpg)

![060e8a77a0ea5114ad0f7ac85832b2278b94ba9158d99a3af8a2f62091799e38.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/060e8a77a0ea5114ad0f7ac85832b2278b94ba9158d99a3af8a2f62091799e38.jpg)

![06be0d6281c8a39bc7f23761b6763764a56da8de3d202b60544503d925cc2d16.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/06be0d6281c8a39bc7f23761b6763764a56da8de3d202b60544503d925cc2d16.jpg)

![0703313c7886311b684a2eef861b1ce8a5330f25c7ed39bb0053a5abc35e7b8c.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/0703313c7886311b684a2eef861b1ce8a5330f25c7ed39bb0053a5abc35e7b8c.jpg)

![078a99fb109d515228ae2f1ebe235f0dbddfcd38fc79bbbd33a99653afc855f2.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/078a99fb109d515228ae2f1ebe235f0dbddfcd38fc79bbbd33a99653afc855f2.jpg)

![09413131f6b5933b557d34bafb1158da8db74d37d4273fe36455e37b0c2990c4.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/09413131f6b5933b557d34bafb1158da8db74d37d4273fe36455e37b0c2990c4.jpg)

![205259b04c1b86e1d6109079f08e8b591c76ca7cd3974b9023f44d0a9233712b.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/205259b04c1b86e1d6109079f08e8b591c76ca7cd3974b9023f44d0a9233712b.jpg)

![27d96282d6a617ebf0c02692695a19ae216e4897f9fa562fc3765240005febc5.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/27d96282d6a617ebf0c02692695a19ae216e4897f9fa562fc3765240005febc5.jpg)

![2e62d2995fbf407ef7b4f428c019209d437cb10e3706e71e4802b2bd288f629b.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/2e62d2995fbf407ef7b4f428c019209d437cb10e3706e71e4802b2bd288f629b.jpg)

![32b8084070fa0e2ee75e12abfc4ade600c4c27fe0dacdf524c4b3556769aa8a5.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/32b8084070fa0e2ee75e12abfc4ade600c4c27fe0dacdf524c4b3556769aa8a5.jpg)

![34f75257fec0d3e2bd2e067b1a48d60d4ed5a2c8c909ae45309a126cf090b861.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/34f75257fec0d3e2bd2e067b1a48d60d4ed5a2c8c909ae45309a126cf090b861.jpg)

![36a4e3545c77a94db8be568f0ac7483d5a7b94c462ef0384e8b7820fc978347e.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/36a4e3545c77a94db8be568f0ac7483d5a7b94c462ef0384e8b7820fc978347e.jpg)

![378beea430fe72a6ef26cc1aa536dea2254599d198579d71c6dba8bca164e4e6.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/378beea430fe72a6ef26cc1aa536dea2254599d198579d71c6dba8bca164e4e6.jpg)

![39ba13f1b3639ff1092bc80188e4ac7d87ebdbab2c29a0c01b526ed82387a9e2.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/39ba13f1b3639ff1092bc80188e4ac7d87ebdbab2c29a0c01b526ed82387a9e2.jpg)

![44b1a67eef45e27b8de184b899186f2d1e78f03c6b4a64822965bf5417cceb7d.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/44b1a67eef45e27b8de184b899186f2d1e78f03c6b4a64822965bf5417cceb7d.jpg)

![47d247d32f74089f3683d9e3e3f02e6d1f3d3af9e3b9e669cfe51c1238062ed1.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/47d247d32f74089f3683d9e3e3f02e6d1f3d3af9e3b9e669cfe51c1238062ed1.jpg)

![4b8089bcaa079e1d3b01287406713f163096b547657431c0dc47d16b19450e7e.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/4b8089bcaa079e1d3b01287406713f163096b547657431c0dc47d16b19450e7e.jpg)

![4c5a463755ea66fc30491522e90bedcfad2731824516faa7759c69bad799c24f.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/4c5a463755ea66fc30491522e90bedcfad2731824516faa7759c69bad799c24f.jpg)

![4d7ace94511183e82d50cc592d2281d0670ad4350ee4509975b9a454384fd1df.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/4d7ace94511183e82d50cc592d2281d0670ad4350ee4509975b9a454384fd1df.jpg)

![4f9757aa289691e1e88920ecebb4ca3220273ec77bb5998c478253629f46206d.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/4f9757aa289691e1e88920ecebb4ca3220273ec77bb5998c478253629f46206d.jpg)

![4f9e586511351e4447a30df74604c117c8433de448fd2f99342254fe2ad54131.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/4f9e586511351e4447a30df74604c117c8433de448fd2f99342254fe2ad54131.jpg)

![518800b742d70ce1eed91dbb1532fee350aea9244b977b058171c2df8416efd1.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/518800b742d70ce1eed91dbb1532fee350aea9244b977b058171c2df8416efd1.jpg)

![567afe1e8cb1076f285792803528f27eb0e35ed94d67d9ef312b0da0cdac1973.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/567afe1e8cb1076f285792803528f27eb0e35ed94d67d9ef312b0da0cdac1973.jpg)

![57a28cf43eec253fb5005bfc5ff1514ca82a9702151f35887c8a4df2aa47cb3c.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/57a28cf43eec253fb5005bfc5ff1514ca82a9702151f35887c8a4df2aa47cb3c.jpg)

![5e53159772c423d8103b258eabedead21a571439bdd01780e25a08c53f81ae24.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/5e53159772c423d8103b258eabedead21a571439bdd01780e25a08c53f81ae24.jpg)

![64bae2f63c307ccb46d1d95bdc83c205e5ace118367215a420cd0ae4162751d6.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/64bae2f63c307ccb46d1d95bdc83c205e5ace118367215a420cd0ae4162751d6.jpg)

![64fdf704e8e3e5e1fa0afce76df0f5a181af9029a5b5546814b430b1305a8719.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/64fdf704e8e3e5e1fa0afce76df0f5a181af9029a5b5546814b430b1305a8719.jpg)

![65c877ac961b75ec30a6ef59f05400295ce1bf82ca1298bdbbb170bdfb9eff0c.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/65c877ac961b75ec30a6ef59f05400295ce1bf82ca1298bdbbb170bdfb9eff0c.jpg)

![68fa7fd57efe27dbfd5622965cff3125d43fe0ee10d30e17a713ed4df5b8d62a.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/68fa7fd57efe27dbfd5622965cff3125d43fe0ee10d30e17a713ed4df5b8d62a.jpg)

![69bf2e0cd4e61d3abf7df24addf37ae16cd9ee1af463773b9aac4eccad106268.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/69bf2e0cd4e61d3abf7df24addf37ae16cd9ee1af463773b9aac4eccad106268.jpg)

![6ca160b1de797028ba9503c7cdaa46f7cb793ac01bb09ab8e001c8719b4bdee2.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/6ca160b1de797028ba9503c7cdaa46f7cb793ac01bb09ab8e001c8719b4bdee2.jpg)

![6f8ea2cc1b66fdfad3c4c4fb1e320cfef845e00dfdfa2727ff9e088b5fd4bf08.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/6f8ea2cc1b66fdfad3c4c4fb1e320cfef845e00dfdfa2727ff9e088b5fd4bf08.jpg)

![70d99e1cf1e25cb2770f0a2778764fbfde9ee32b32bd80b1b870484b2120551a.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/70d99e1cf1e25cb2770f0a2778764fbfde9ee32b32bd80b1b870484b2120551a.jpg)

![70f1b22bec64f6ca372427297a2114a1b489d5203afd1c4c2d390f97fa67491c.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/70f1b22bec64f6ca372427297a2114a1b489d5203afd1c4c2d390f97fa67491c.jpg)

![7b60e19d42a3ab971c3a4b2574d43539f6e23a6190d81a8cc5adcb4ae28d04d6.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/7b60e19d42a3ab971c3a4b2574d43539f6e23a6190d81a8cc5adcb4ae28d04d6.jpg)

![7e2ab0a496459fe3dd7071b801798a554457ac3ec57d0851982eaec76dfcf245.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/7e2ab0a496459fe3dd7071b801798a554457ac3ec57d0851982eaec76dfcf245.jpg)

![7e500269fed383074171b0d311ae90480cebe77a1f3b51e47ac2787140edc10f.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/7e500269fed383074171b0d311ae90480cebe77a1f3b51e47ac2787140edc10f.jpg)

![80796bad6fd20d2727025004d5d1f5b3501c1c78285af9a97fe06d6629584baa.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/80796bad6fd20d2727025004d5d1f5b3501c1c78285af9a97fe06d6629584baa.jpg)

![81232334a6f4f46f3ce59faea1c8ce28fc6ba5bfda69a5eb4a0a3a78a4096caf.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/81232334a6f4f46f3ce59faea1c8ce28fc6ba5bfda69a5eb4a0a3a78a4096caf.jpg)

![8cc244f06fad8ec7e7f3374ad522d218feda5f6bbaf4a78f5f20ef5a1cf7cbe3.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/8cc244f06fad8ec7e7f3374ad522d218feda5f6bbaf4a78f5f20ef5a1cf7cbe3.jpg)

![8da0966e430eba2c7055f5670c8235db6e3c81c9fd55a55b49a8c85d2b6bd72b.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/8da0966e430eba2c7055f5670c8235db6e3c81c9fd55a55b49a8c85d2b6bd72b.jpg)

![90b39245cd2c85d96a65057b2fe97d878dd7bf795e25420756da287bdbd30a75.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/90b39245cd2c85d96a65057b2fe97d878dd7bf795e25420756da287bdbd30a75.jpg)

![a358bb4ba488b3332b563b573ff29b0197eaec0f710a9d13324b4a7b931a2631.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/a358bb4ba488b3332b563b573ff29b0197eaec0f710a9d13324b4a7b931a2631.jpg)

![b5219d12f0cac675ac2724da9687672d1b4feeafa28e2117874de003a32a380b.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/b5219d12f0cac675ac2724da9687672d1b4feeafa28e2117874de003a32a380b.jpg)

![b79155f53c289a8d41cfbf5c22be36b1bd62b003326db06de33ca88809b837f5.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/b79155f53c289a8d41cfbf5c22be36b1bd62b003326db06de33ca88809b837f5.jpg)

![bc0c02b06c5d0eeb5eeff362fbb192050ae533f23ee78f9b634af0856a3d7da0.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/bc0c02b06c5d0eeb5eeff362fbb192050ae533f23ee78f9b634af0856a3d7da0.jpg)

![bf9b1e353a171d1cc37f32b9b4bfea6bc5b40301a508bfd604d59f40328e7fe0.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/bf9b1e353a171d1cc37f32b9b4bfea6bc5b40301a508bfd604d59f40328e7fe0.jpg)

![c101b566b2b083ebc0e692250c7a7b9eb49a2cd030f63e9491f64b2464bb204c.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/c101b566b2b083ebc0e692250c7a7b9eb49a2cd030f63e9491f64b2464bb204c.jpg)

![c2b3f9153952874977be5a45df89bc78fe4083af34119e9f84da879e85c5e080.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/c2b3f9153952874977be5a45df89bc78fe4083af34119e9f84da879e85c5e080.jpg)

![c47130a5b683d2853791ed17a6ba8c0e602c7a342285ed18d70f985d1170ae8b.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/c47130a5b683d2853791ed17a6ba8c0e602c7a342285ed18d70f985d1170ae8b.jpg)

![ca8e303d67ba023faa12497f4ba1d7675b35cb346de1b608b2b43900f8fe95d2.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/ca8e303d67ba023faa12497f4ba1d7675b35cb346de1b608b2b43900f8fe95d2.jpg)

![cdf54f2f22a49047ea476ec19406227881ed3e5f59ebc5d613efb0ae8a4a02c9.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/cdf54f2f22a49047ea476ec19406227881ed3e5f59ebc5d613efb0ae8a4a02c9.jpg)

![ced8f5ebf23d6ebf0f0cdde9eabc804c8594275d0c6d33c5950f0efc9c634d2d.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/ced8f5ebf23d6ebf0f0cdde9eabc804c8594275d0c6d33c5950f0efc9c634d2d.jpg)

![cf780807a608852a14db7b99432fd74b95881084e789a46ba71f1a8f4b9e0ce7.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/cf780807a608852a14db7b99432fd74b95881084e789a46ba71f1a8f4b9e0ce7.jpg)

![d05198ad60e7ca1e665487043983c75c8472ee827d96221e98e56faed9c3b9f2.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/d05198ad60e7ca1e665487043983c75c8472ee827d96221e98e56faed9c3b9f2.jpg)

![d2104465e461bf098eb214e49209e2366b077d8d63a740504fe9b639e5abcb06.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/d2104465e461bf098eb214e49209e2366b077d8d63a740504fe9b639e5abcb06.jpg)

![d93b7f0865663ef846df0432a8490fcc04c46c4138cf734805c679699feac4b3.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/d93b7f0865663ef846df0432a8490fcc04c46c4138cf734805c679699feac4b3.jpg)

![d9a86456b22656deb6355def749f48a94d176620b840c1faf5f6fb6be8268f7e.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/d9a86456b22656deb6355def749f48a94d176620b840c1faf5f6fb6be8268f7e.jpg)

![da9878d4108128cea4985356683d7e327ed29867790c58e8e744db4d8c1cc619.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/da9878d4108128cea4985356683d7e327ed29867790c58e8e744db4d8c1cc619.jpg)

![e581e5407ebc9c55b9ab4fbf8a6bff27ace4824d9bb740b84de2445c95be3a8a.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/e581e5407ebc9c55b9ab4fbf8a6bff27ace4824d9bb740b84de2445c95be3a8a.jpg)

![f3521511ed5541ee98d8a45d34030a4510b268f20d2fb0615fcc05d490dc6188.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/f3521511ed5541ee98d8a45d34030a4510b268f20d2fb0615fcc05d490dc6188.jpg)

![f7fd70e4366ac5968ec8a9b4963a7a1b20538f4785baddfc3809486e32a88431.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/f7fd70e4366ac5968ec8a9b4963a7a1b20538f4785baddfc3809486e32a88431.jpg)

![fa0db91cff0c1f3785e05a052395b271e619b7584124ffd15ee7d080fb9985cd.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/fa0db91cff0c1f3785e05a052395b271e619b7584124ffd15ee7d080fb9985cd.jpg)

![fde10d12f553347c6975522554918993bd2ad19270176db93780d07e598356f7.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/images/fde10d12f553347c6975522554918993bd2ad19270176db93780d07e598356f7.jpg)

### Tables

![1e9134852ee5d01917c9b2cf603b94e76b5533569947eb96f19c75b3632486ea.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/tables/1e9134852ee5d01917c9b2cf603b94e76b5533569947eb96f19c75b3632486ea.jpg)

![28b8ca93aeff7e4b81dcc62af186a3aa0acf3b7316ddb7101266025d90d0014e.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/tables/28b8ca93aeff7e4b81dcc62af186a3aa0acf3b7316ddb7101266025d90d0014e.jpg)

![34c8b5dc8d5a81ba7e2ab25101a1275fc94dabc0f7aeb395a897324be7be04c3.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/tables/34c8b5dc8d5a81ba7e2ab25101a1275fc94dabc0f7aeb395a897324be7be04c3.jpg)

![852d7052bfcd8175a92bcb8065936216a7b23af860919f5b44202a330b978cff.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/tables/852d7052bfcd8175a92bcb8065936216a7b23af860919f5b44202a330b978cff.jpg)

![9daf9de38889c315953a5eee9cfee8d2f483901b29d3d20bd18a701640e3c2e5.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/tables/9daf9de38889c315953a5eee9cfee8d2f483901b29d3d20bd18a701640e3c2e5.jpg)

![b9de2ffe77f4abda531bbb3bfa1b677aeffe55a6cb2b96f38b70599a71a361b2.jpg](../icml_results/948_Self-Supervised%20Flow%20Matching%20for%20Scalable%20Multi-Modal%20Synthesis/tables/b9de2ffe77f4abda531bbb3bfa1b677aeffe55a6cb2b96f38b70599a71a361b2.jpg)

<a id="paper-0932"></a>

## Scalable Kronecker-Factored Fisher Approximation for Neural Network Parameter Sensitivity


### Images

![1e38c31b99fef31772711f83b8f1ad59d19950c9a8bd65f7303b06396f8eaef0.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/images/1e38c31b99fef31772711f83b8f1ad59d19950c9a8bd65f7303b06396f8eaef0.jpg)

![36f230092c693176f37a77c38fa04d02268212edb98aa43abf6454862a02f265.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/images/36f230092c693176f37a77c38fa04d02268212edb98aa43abf6454862a02f265.jpg)

![697f665c9221465eb7a25c8220f15e05dc8a93662d598da613617702c09c1fdf.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/images/697f665c9221465eb7a25c8220f15e05dc8a93662d598da613617702c09c1fdf.jpg)

![9207b37b6804cd7f4a4f7e779ebaeb7097ac247a194c21ba7d9be1176c429f22.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/images/9207b37b6804cd7f4a4f7e779ebaeb7097ac247a194c21ba7d9be1176c429f22.jpg)

![98f090b4809e3f0abb4b3780b8b7b8fbff015be498c8b119c7311696b89974dc.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/images/98f090b4809e3f0abb4b3780b8b7b8fbff015be498c8b119c7311696b89974dc.jpg)

![a0099e726a0c88326ffc0c9411338a3aaccab259da66fbd925f808d67accd4c4.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/images/a0099e726a0c88326ffc0c9411338a3aaccab259da66fbd925f808d67accd4c4.jpg)

![be54b6405b940aea57a12becfc417f9c738ddeaa4f18e8eab4ac713801cdfd30.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/images/be54b6405b940aea57a12becfc417f9c738ddeaa4f18e8eab4ac713801cdfd30.jpg)

### Tables

![1646b2859e265e7e0c913920309798a684b6a3ebb06b22d6cc1b9bbdc44f0040.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/tables/1646b2859e265e7e0c913920309798a684b6a3ebb06b22d6cc1b9bbdc44f0040.jpg)

![373f7ea0ac7d38774726b1c93e273b277ccd71da7e5434143324ac1696abcaa0.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/tables/373f7ea0ac7d38774726b1c93e273b277ccd71da7e5434143324ac1696abcaa0.jpg)

![49ac69f61d487382a7c6007071a916cff5bf75fb7cfde6d863321d7338f2a81f.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/tables/49ac69f61d487382a7c6007071a916cff5bf75fb7cfde6d863321d7338f2a81f.jpg)

![5e4097f6ae344d8e0f71f36696ab58550e3dbaa96664230444e51e8c2a0c8bef.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/tables/5e4097f6ae344d8e0f71f36696ab58550e3dbaa96664230444e51e8c2a0c8bef.jpg)

![6220ae6c5c1cef879a4e3c147c78340983e390a93e668b6aaf655417aa3f47a9.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/tables/6220ae6c5c1cef879a4e3c147c78340983e390a93e668b6aaf655417aa3f47a9.jpg)

![78253a1700e9948a2dd5d336acb7ebea61b80aa9e9f724a99963acb4994146fb.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/tables/78253a1700e9948a2dd5d336acb7ebea61b80aa9e9f724a99963acb4994146fb.jpg)

![a2eb1e9f8529537d3f354890f6ba6156ddda352c8bfbda5336bd414e38720c35.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/tables/a2eb1e9f8529537d3f354890f6ba6156ddda352c8bfbda5336bd414e38720c35.jpg)

![c08e7c4f140117289cf9e42b40bafb141e7e6a633c41394fa57b7a8487ba8370.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/tables/c08e7c4f140117289cf9e42b40bafb141e7e6a633c41394fa57b7a8487ba8370.jpg)

![c19642c408ee4b78d220554ee92929eba1571b3dd95642fc78040744ecef52b4.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/tables/c19642c408ee4b78d220554ee92929eba1571b3dd95642fc78040744ecef52b4.jpg)

![cc287d11d4ad289c20cdb99486c2a9ee13b4e8ed628c1c4dc6e0d962e20a3a3a.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/tables/cc287d11d4ad289c20cdb99486c2a9ee13b4e8ed628c1c4dc6e0d962e20a3a3a.jpg)

![ce435feaec721c61f1ee46836e23281252304367ba8f6b734f7e533c5c8bb32a.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/tables/ce435feaec721c61f1ee46836e23281252304367ba8f6b734f7e533c5c8bb32a.jpg)

![e6cf06a9c97ae2cb5e4ed47a04c17e10d5f0833fb1073872d84bc40b15366bea.jpg](../icml_results/949_Scalable%20Kronecker-Factored%20Fisher%20Approximation%20for%20Neural%20Network%20Parameter%20Sensitivity/tables/e6cf06a9c97ae2cb5e4ed47a04c17e10d5f0833fb1073872d84bc40b15366bea.jpg)

<a id="paper-0933"></a>

## Adapting Noise to Data: Generative Flows from Learned 1D Processes


### Images

![03edd17f11d9b8d1ded76b7b9c85dbe751457e697d1b004aa512164cd7f4163c.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/03edd17f11d9b8d1ded76b7b9c85dbe751457e697d1b004aa512164cd7f4163c.jpg)

![1e42a89fe4f92437a5770699baf7f3e98288ea7cac4449f6cf54f22772895eb7.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/1e42a89fe4f92437a5770699baf7f3e98288ea7cac4449f6cf54f22772895eb7.jpg)

![290d46ac51db8fb4e917956f15fe51f2d56adc1c5fab7a229c98748e0629a4c3.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/290d46ac51db8fb4e917956f15fe51f2d56adc1c5fab7a229c98748e0629a4c3.jpg)

![2d3877b723a336a6df61b23b1ff5b65aa19a97d8653c04e915f31b6a23cd021a.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/2d3877b723a336a6df61b23b1ff5b65aa19a97d8653c04e915f31b6a23cd021a.jpg)

![3dc53ed72b12c705b944c02fe15afb70577647f67f88a30a35695ddedad0059f.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/3dc53ed72b12c705b944c02fe15afb70577647f67f88a30a35695ddedad0059f.jpg)

![439bb0b0f93ea2ca66fe34205bda79404a52601680141f24bc81273bc4829048.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/439bb0b0f93ea2ca66fe34205bda79404a52601680141f24bc81273bc4829048.jpg)

![526730a5b1e60b60932e8d566136a53d3a5b7ddb15c2cf029467ac6a14b60539.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/526730a5b1e60b60932e8d566136a53d3a5b7ddb15c2cf029467ac6a14b60539.jpg)

![5baf0417727da391df25fffc831a9f5a77f876649b7df8a2763be0ce4e96efcb.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/5baf0417727da391df25fffc831a9f5a77f876649b7df8a2763be0ce4e96efcb.jpg)

![5d4f3a7e48734f4a27eb05ef8c19c2762bc8f30d07d0a8838ff007cd7067a058.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/5d4f3a7e48734f4a27eb05ef8c19c2762bc8f30d07d0a8838ff007cd7067a058.jpg)

![65cce2737471ebb2944dcb0571efb9a419f19107e7f0fc41f3ffbc9208365d93.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/65cce2737471ebb2944dcb0571efb9a419f19107e7f0fc41f3ffbc9208365d93.jpg)

![77b2b8b4403fe626c3a497a19ccde14a8b66975a33118fa4300a96d80bb72c5c.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/77b2b8b4403fe626c3a497a19ccde14a8b66975a33118fa4300a96d80bb72c5c.jpg)

![7a958fb51453766344a17fd37b4232a9a236e14849fddd9c10bc5555769217b5.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/7a958fb51453766344a17fd37b4232a9a236e14849fddd9c10bc5555769217b5.jpg)

![7f41e540a6db5595e17ca550fd1afdc638e333102de31603a4cab01cff43c60f.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/7f41e540a6db5595e17ca550fd1afdc638e333102de31603a4cab01cff43c60f.jpg)

![900393d34e63feff4d436e18fa8133d4b08ec78ff3b7f00f3fd9f124051c0104.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/900393d34e63feff4d436e18fa8133d4b08ec78ff3b7f00f3fd9f124051c0104.jpg)

![92cf4ab41696a18687b5aa8edc563c9bf12726612340e6156f5980276dd8837c.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/92cf4ab41696a18687b5aa8edc563c9bf12726612340e6156f5980276dd8837c.jpg)

![92d6d9a6e91a0aa24b3f7a7e4baeedcae12e078318287cd08a0c70910514a7ca.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/92d6d9a6e91a0aa24b3f7a7e4baeedcae12e078318287cd08a0c70910514a7ca.jpg)

![961c4e76e822cb17672a6bd2446bd38cee24e0b9203b998c251e7f7470e745f6.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/961c4e76e822cb17672a6bd2446bd38cee24e0b9203b998c251e7f7470e745f6.jpg)

![b473a46249f180f8b503cc5b80c489f1b15d4f8f1dda0250ada8b9e03bd813d6.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/b473a46249f180f8b503cc5b80c489f1b15d4f8f1dda0250ada8b9e03bd813d6.jpg)

![b6b466a7669dc608200a772458ae88392dc0095483eab95a0a7272091444fc4b.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/b6b466a7669dc608200a772458ae88392dc0095483eab95a0a7272091444fc4b.jpg)

![bd4173d0a778ab6bb48c36d19c28bf4757e40c974cd4587b7177a143cff9a43c.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/bd4173d0a778ab6bb48c36d19c28bf4757e40c974cd4587b7177a143cff9a43c.jpg)

![d2e5a06b34eeb507d64c4c75e45f57e5eadebcaf13c31e96f68f3e0b3a9cb248.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/d2e5a06b34eeb507d64c4c75e45f57e5eadebcaf13c31e96f68f3e0b3a9cb248.jpg)

![d39d72006942bd23777931d4eda49531c4a6503926cfd3e4629ab005a24f5978.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/d39d72006942bd23777931d4eda49531c4a6503926cfd3e4629ab005a24f5978.jpg)

![d56cc28947c2ee230c15cec55c2ca205ac66f4c2f37f8264ae59f2cff022e486.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/d56cc28947c2ee230c15cec55c2ca205ac66f4c2f37f8264ae59f2cff022e486.jpg)

![e41e70d146244bde60fcbcd08a70186cac3506a462feeceac53c77647417df63.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/images/e41e70d146244bde60fcbcd08a70186cac3506a462feeceac53c77647417df63.jpg)

### Tables

![03b321ddc3e95c3d252daff574364f5ba3debeb1500da2633127223292b7e637.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/tables/03b321ddc3e95c3d252daff574364f5ba3debeb1500da2633127223292b7e637.jpg)

![84ec4fb95716e61b9125b8bd5ef3e66c7b1c722b7a82bbb5cf9bff6a95b2e188.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/tables/84ec4fb95716e61b9125b8bd5ef3e66c7b1c722b7a82bbb5cf9bff6a95b2e188.jpg)

![a53a0ffbf7dfdaa7a2dd4464e6f8a02cd05f44cfeeaec12fc23fdc4eabcdda3f.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/tables/a53a0ffbf7dfdaa7a2dd4464e6f8a02cd05f44cfeeaec12fc23fdc4eabcdda3f.jpg)

![b409ad2b1d8308ea1d9c0eaafd724ceb7e90bf86788efc213e22f789be015024.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/tables/b409ad2b1d8308ea1d9c0eaafd724ceb7e90bf86788efc213e22f789be015024.jpg)

![cfe68b2094c03ffee024a0972b97eaa79841ad96c2d044f47bacda72a80f2075.jpg](../icml_results/950_Adapting%20Noise%20to%20Data_%20Generative%20Flows%20from%20Learned%201D%20Processes/tables/cfe68b2094c03ffee024a0972b97eaa79841ad96c2d044f47bacda72a80f2075.jpg)

<a id="paper-0934"></a>

## Differentially Private Range Subgraph Counting


### Images

![01656b886a077fcde3594481f11f27d372d1ed96cee053f7fb1004697a11ce46.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/images/01656b886a077fcde3594481f11f27d372d1ed96cee053f7fb1004697a11ce46.jpg)

![2dc4a7f0a2f921c9ee6deb3c0d32f91d75b4d5873e44f1e05176213f32f69ee6.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/images/2dc4a7f0a2f921c9ee6deb3c0d32f91d75b4d5873e44f1e05176213f32f69ee6.jpg)

![3378c13458ce8d7162167d230719e04b446c6f0dd30f92f85b4d83af68bbe814.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/images/3378c13458ce8d7162167d230719e04b446c6f0dd30f92f85b4d83af68bbe814.jpg)

![42da067be17f4976031a7fef599b9bcb33402579bf4e74353d89137c77424c2f.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/images/42da067be17f4976031a7fef599b9bcb33402579bf4e74353d89137c77424c2f.jpg)

![5aaeb64512b870145f81042d5851006b077e10f967f5384bcf28e9bdab9ec468.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/images/5aaeb64512b870145f81042d5851006b077e10f967f5384bcf28e9bdab9ec468.jpg)

![74fedcf6de3e801d0156beb37f2040a55d0c80c84ea2f906eafdedfb020c8e54.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/images/74fedcf6de3e801d0156beb37f2040a55d0c80c84ea2f906eafdedfb020c8e54.jpg)

![8775279ae6861affa5020d79e39db2b825993572f03ecf0dad19c3192c4129e9.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/images/8775279ae6861affa5020d79e39db2b825993572f03ecf0dad19c3192c4129e9.jpg)

![89007f6350accec41d9c0cc982a67611ba890c6469c92867ce1a0504b000b7a0.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/images/89007f6350accec41d9c0cc982a67611ba890c6469c92867ce1a0504b000b7a0.jpg)

![8bfa6b8ccb42bf7fb0864c3f185ce5a8c68a977d2b965046e310749927373f71.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/images/8bfa6b8ccb42bf7fb0864c3f185ce5a8c68a977d2b965046e310749927373f71.jpg)

![9e677f33036c7fb2cf41dbb433935fc4fcc8c91620d6b32df5d37b8261ca6fef.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/images/9e677f33036c7fb2cf41dbb433935fc4fcc8c91620d6b32df5d37b8261ca6fef.jpg)

![b17f44731cc8fe8e2cc7b8bb50bc4faf45ae855523ac5241b68398e470edb809.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/images/b17f44731cc8fe8e2cc7b8bb50bc4faf45ae855523ac5241b68398e470edb809.jpg)

![c2b35a914bf290e0e874a1d40d70ebc4a17032e3104498eafaaa787df26d8bfc.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/images/c2b35a914bf290e0e874a1d40d70ebc4a17032e3104498eafaaa787df26d8bfc.jpg)

![c865734d28725e81b23327680ff0f76761b5116dac815045493097926ecf9c3e.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/images/c865734d28725e81b23327680ff0f76761b5116dac815045493097926ecf9c3e.jpg)

![e0b612a8d92db0303f6534aee3857e4a0a169c88236de912a1f2afdee8682f04.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/images/e0b612a8d92db0303f6534aee3857e4a0a169c88236de912a1f2afdee8682f04.jpg)

![e99367a6892118096e548b70ddf5d5f2faaa18dad286bb2939ec96c87bb0c2f1.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/images/e99367a6892118096e548b70ddf5d5f2faaa18dad286bb2939ec96c87bb0c2f1.jpg)

![ebc7396543249061603001884b84274a5f56d36693dcab879dc9b919bf1c0441.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/images/ebc7396543249061603001884b84274a5f56d36693dcab879dc9b919bf1c0441.jpg)

![f94d65eb4ca60fdf2981dcabbc51acf4c33c632bac0eb97b37b82c88efe73ca3.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/images/f94d65eb4ca60fdf2981dcabbc51acf4c33c632bac0eb97b37b82c88efe73ca3.jpg)

### Tables

![4d29a13c874993c729c36e67f57e25eed6bc74cd18c9788e43a49021e1dcaba9.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/tables/4d29a13c874993c729c36e67f57e25eed6bc74cd18c9788e43a49021e1dcaba9.jpg)

![4f519777b95c9f364c1223ca0bd6204d22e1a8d7776157da862029d1a5d38871.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/tables/4f519777b95c9f364c1223ca0bd6204d22e1a8d7776157da862029d1a5d38871.jpg)

![58f442e9dfbc2437172562a28ce0c937540ad3bfbefb7eb06bd599f41f1ee698.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/tables/58f442e9dfbc2437172562a28ce0c937540ad3bfbefb7eb06bd599f41f1ee698.jpg)

![774f549bd38593a8a88b53e6d09ba6a7654d09ce8f5d223a1210c17e8cc720b9.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/tables/774f549bd38593a8a88b53e6d09ba6a7654d09ce8f5d223a1210c17e8cc720b9.jpg)

![a989f4a919958d8c6047ee3b824004e510e4a39b7f677c64435a4fba47232316.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/tables/a989f4a919958d8c6047ee3b824004e510e4a39b7f677c64435a4fba47232316.jpg)

![dedc1788813df98ba464cd1116b4b9965abcf23108f8243657c11ba190e72045.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/tables/dedc1788813df98ba464cd1116b4b9965abcf23108f8243657c11ba190e72045.jpg)

![dfb0708e64720a2b45819477899ed193e1fd09af058897f6692c0b524e418632.jpg](../icml_results/951_Differentially%20Private%20Range%20Subgraph%20Counting/tables/dfb0708e64720a2b45819477899ed193e1fd09af058897f6692c0b524e418632.jpg)

<a id="paper-0935"></a>

## OpenGPT-4o-Image: A Comprehensive Dataset for Advanced Image Generation and Editing


### Images

![1a7866f053ab291a216e85b33242a34e135f130f759066bbe7b66d48c3702c9f.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/images/1a7866f053ab291a216e85b33242a34e135f130f759066bbe7b66d48c3702c9f.jpg)

![26a69dbd202980ad390ce16264dae3bc5846235a2628b2876677c5bda8b664a8.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/images/26a69dbd202980ad390ce16264dae3bc5846235a2628b2876677c5bda8b664a8.jpg)

![31d3870404702bd06abf6b9b33856da0bc80909a225679c00ffeeb92966a56ae.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/images/31d3870404702bd06abf6b9b33856da0bc80909a225679c00ffeeb92966a56ae.jpg)

![582c3239695e3f710d69a9a79038aade3860299d5cc60a54d60bb52226344be8.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/images/582c3239695e3f710d69a9a79038aade3860299d5cc60a54d60bb52226344be8.jpg)

![9286ff2565a6266f457470271ccb84dd708fd0dbb03247c101053a990712a07b.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/images/9286ff2565a6266f457470271ccb84dd708fd0dbb03247c101053a990712a07b.jpg)

![e824e7ff9637b1249897ab1ea5beb0c3d76076eabb9529356a7f7aa875dfb53e.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/images/e824e7ff9637b1249897ab1ea5beb0c3d76076eabb9529356a7f7aa875dfb53e.jpg)

![fbf51d0da908e845fad0dbec647213d4ec0b7fb3bff577f3f31b396169137edb.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/images/fbf51d0da908e845fad0dbec647213d4ec0b7fb3bff577f3f31b396169137edb.jpg)

### Tables

![0d1c0e5de1391be7af7683a025ded3ee4bd389ae695ae8660c62e0898cc863f4.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/tables/0d1c0e5de1391be7af7683a025ded3ee4bd389ae695ae8660c62e0898cc863f4.jpg)

![146058fff0bab8377d1741f278409ba46e328acfdb2ffd55a56085864e5921f4.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/tables/146058fff0bab8377d1741f278409ba46e328acfdb2ffd55a56085864e5921f4.jpg)

![21f78806bf9e31d278fe44ba3fd97720d5f18ab735878b994e4ab9506dfac589.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/tables/21f78806bf9e31d278fe44ba3fd97720d5f18ab735878b994e4ab9506dfac589.jpg)

![25792d136588b3feea0fa1cc1a3d915aa403273535ccbd843e6661d2003eb961.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/tables/25792d136588b3feea0fa1cc1a3d915aa403273535ccbd843e6661d2003eb961.jpg)

![3f56732750996ca45be0908ee6e6b114d57aa2bdada701a37b676b0655cf8550.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/tables/3f56732750996ca45be0908ee6e6b114d57aa2bdada701a37b676b0655cf8550.jpg)

![5efb740a2e0027c569bfbb12ce281c740aa5cbe328ed7d2cea7b2ec8db095fb1.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/tables/5efb740a2e0027c569bfbb12ce281c740aa5cbe328ed7d2cea7b2ec8db095fb1.jpg)

![60d65f8fb18d7d26aa62c9919ab79b44d910364c99aa6510527f24268f2781cc.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/tables/60d65f8fb18d7d26aa62c9919ab79b44d910364c99aa6510527f24268f2781cc.jpg)

![7cc0cdd7ffb33ab857c161c408f528cca0dd2976eaf4bfa3aba9472d45bdefa1.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/tables/7cc0cdd7ffb33ab857c161c408f528cca0dd2976eaf4bfa3aba9472d45bdefa1.jpg)

![987e3865e366a950f88bc29881add5a9615a9d096473d72688ebf1283df593b3.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/tables/987e3865e366a950f88bc29881add5a9615a9d096473d72688ebf1283df593b3.jpg)

![9f63d315483b7d5429ba0ae47222e1a3d2e593d3d2d672d6f79bb54aebc42f06.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/tables/9f63d315483b7d5429ba0ae47222e1a3d2e593d3d2d672d6f79bb54aebc42f06.jpg)

![a268184603785c430873ddd3d5f95f79bbfacdae331174486bf6c8a1d8867930.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/tables/a268184603785c430873ddd3d5f95f79bbfacdae331174486bf6c8a1d8867930.jpg)

![bea370ec570388187aa245f6c02b4da7dac7424c463f55fe8591a2063a368d4a.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/tables/bea370ec570388187aa245f6c02b4da7dac7424c463f55fe8591a2063a368d4a.jpg)

![bf8fc582b410a459aa119063ab3703353a835fb5f190eec0158024b81482149f.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/tables/bf8fc582b410a459aa119063ab3703353a835fb5f190eec0158024b81482149f.jpg)

![c5f0c91c7dfdde6e6846cc333e4c715bb44ac69120ec337b65a844d3c18f9ba3.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/tables/c5f0c91c7dfdde6e6846cc333e4c715bb44ac69120ec337b65a844d3c18f9ba3.jpg)

![fce8154929e50d5914c4bdab6947714fab707d191615fd0cc236ee9725deeb4e.jpg](../icml_results/952_OpenGPT-4o-Image_%20A%20Comprehensive%20Dataset%20for%20Advanced%20Image%20Generation%20and%20Editing/tables/fce8154929e50d5914c4bdab6947714fab707d191615fd0cc236ee9725deeb4e.jpg)

<a id="paper-0936"></a>

## NOMAD: Lifelong Trajectory Planning via Non-Parametric Bayesian Memory-Adaptive Diffusion Experts


### Images

![2bfff9d1e4e006db09b0f83a8b3e0f7ecb7104e7b9febd41bc27fb31d047061b.jpg](../icml_results/953_NOMAD_%20Lifelong%20Trajectory%20Planning%20via%20Non-Parametric%20Bayesian%20Memory-Adaptive%20Diffusion%20Experts/images/2bfff9d1e4e006db09b0f83a8b3e0f7ecb7104e7b9febd41bc27fb31d047061b.jpg)

![43d3a0ccbc250b5470018d47397dc4c7f57ae9cbe4a9c2bfe1ef41eabf30d19d.jpg](../icml_results/953_NOMAD_%20Lifelong%20Trajectory%20Planning%20via%20Non-Parametric%20Bayesian%20Memory-Adaptive%20Diffusion%20Experts/images/43d3a0ccbc250b5470018d47397dc4c7f57ae9cbe4a9c2bfe1ef41eabf30d19d.jpg)

![5abc3938a5e684f7786ddb1fa0dccf734391fecbc59ebf66355fde064617c969.jpg](../icml_results/953_NOMAD_%20Lifelong%20Trajectory%20Planning%20via%20Non-Parametric%20Bayesian%20Memory-Adaptive%20Diffusion%20Experts/images/5abc3938a5e684f7786ddb1fa0dccf734391fecbc59ebf66355fde064617c969.jpg)

![5c98a1284b39b889b992b97aa7932c732a98350e413cc33075bbf6d35c6faf46.jpg](../icml_results/953_NOMAD_%20Lifelong%20Trajectory%20Planning%20via%20Non-Parametric%20Bayesian%20Memory-Adaptive%20Diffusion%20Experts/images/5c98a1284b39b889b992b97aa7932c732a98350e413cc33075bbf6d35c6faf46.jpg)

![620dc9e910269d3e12f99055793dd839ca51981043bedb8bf2ac7c2a269fb007.jpg](../icml_results/953_NOMAD_%20Lifelong%20Trajectory%20Planning%20via%20Non-Parametric%20Bayesian%20Memory-Adaptive%20Diffusion%20Experts/images/620dc9e910269d3e12f99055793dd839ca51981043bedb8bf2ac7c2a269fb007.jpg)

![a8b11ed3724dc3a800a153904d81c23d0dc538c855943419a04a3db2d5b86a79.jpg](../icml_results/953_NOMAD_%20Lifelong%20Trajectory%20Planning%20via%20Non-Parametric%20Bayesian%20Memory-Adaptive%20Diffusion%20Experts/images/a8b11ed3724dc3a800a153904d81c23d0dc538c855943419a04a3db2d5b86a79.jpg)

![dca35ab8fff401a596019d9909ecb247a1579c1b40d5505373a7942fdd05977d.jpg](../icml_results/953_NOMAD_%20Lifelong%20Trajectory%20Planning%20via%20Non-Parametric%20Bayesian%20Memory-Adaptive%20Diffusion%20Experts/images/dca35ab8fff401a596019d9909ecb247a1579c1b40d5505373a7942fdd05977d.jpg)

### Tables

![01d880dafbd55bae25501e45890d311c6ce542098c54db92f917bf3259279814.jpg](../icml_results/953_NOMAD_%20Lifelong%20Trajectory%20Planning%20via%20Non-Parametric%20Bayesian%20Memory-Adaptive%20Diffusion%20Experts/tables/01d880dafbd55bae25501e45890d311c6ce542098c54db92f917bf3259279814.jpg)

![049c54e955e60e88004c722ce44c3a6236326cb552f8f68396ed3db78cacfc39.jpg](../icml_results/953_NOMAD_%20Lifelong%20Trajectory%20Planning%20via%20Non-Parametric%20Bayesian%20Memory-Adaptive%20Diffusion%20Experts/tables/049c54e955e60e88004c722ce44c3a6236326cb552f8f68396ed3db78cacfc39.jpg)

![289f6ffd4222c557c9d1212b2ba9e68076a1b06cd33100dd9c54f0ed7bedd518.jpg](../icml_results/953_NOMAD_%20Lifelong%20Trajectory%20Planning%20via%20Non-Parametric%20Bayesian%20Memory-Adaptive%20Diffusion%20Experts/tables/289f6ffd4222c557c9d1212b2ba9e68076a1b06cd33100dd9c54f0ed7bedd518.jpg)

![2b422f5ef431a065af410b6ca3170f7a8d3a784515c9873e7089a2e51ff9521a.jpg](../icml_results/953_NOMAD_%20Lifelong%20Trajectory%20Planning%20via%20Non-Parametric%20Bayesian%20Memory-Adaptive%20Diffusion%20Experts/tables/2b422f5ef431a065af410b6ca3170f7a8d3a784515c9873e7089a2e51ff9521a.jpg)

![3783ca6f41d92b47f71c3b6810866fccd54408f05d351f1d05f3389fa0801046.jpg](../icml_results/953_NOMAD_%20Lifelong%20Trajectory%20Planning%20via%20Non-Parametric%20Bayesian%20Memory-Adaptive%20Diffusion%20Experts/tables/3783ca6f41d92b47f71c3b6810866fccd54408f05d351f1d05f3389fa0801046.jpg)

![84d5696c341ce6862a86506a41f48e050ed3071a772b0b089199e4e9169f06df.jpg](../icml_results/953_NOMAD_%20Lifelong%20Trajectory%20Planning%20via%20Non-Parametric%20Bayesian%20Memory-Adaptive%20Diffusion%20Experts/tables/84d5696c341ce6862a86506a41f48e050ed3071a772b0b089199e4e9169f06df.jpg)

![98c9a7bf26034156164a6dfa6f49e91553a5d321b74010c45f955d2aa962a800.jpg](../icml_results/953_NOMAD_%20Lifelong%20Trajectory%20Planning%20via%20Non-Parametric%20Bayesian%20Memory-Adaptive%20Diffusion%20Experts/tables/98c9a7bf26034156164a6dfa6f49e91553a5d321b74010c45f955d2aa962a800.jpg)

![b88ed16a77e26d539b3f3d87da43e065524664c1e18205febfc2980774220e85.jpg](../icml_results/953_NOMAD_%20Lifelong%20Trajectory%20Planning%20via%20Non-Parametric%20Bayesian%20Memory-Adaptive%20Diffusion%20Experts/tables/b88ed16a77e26d539b3f3d87da43e065524664c1e18205febfc2980774220e85.jpg)

![d50aa695e0ffc1f21432d53d1bc4ac644ab6016d572b38081701d1e7abfa888f.jpg](../icml_results/953_NOMAD_%20Lifelong%20Trajectory%20Planning%20via%20Non-Parametric%20Bayesian%20Memory-Adaptive%20Diffusion%20Experts/tables/d50aa695e0ffc1f21432d53d1bc4ac644ab6016d572b38081701d1e7abfa888f.jpg)

![e494c58c935482cc533042d2ecbe6268207ac9fd59c257b0dcffcf47d5ced5d2.jpg](../icml_results/953_NOMAD_%20Lifelong%20Trajectory%20Planning%20via%20Non-Parametric%20Bayesian%20Memory-Adaptive%20Diffusion%20Experts/tables/e494c58c935482cc533042d2ecbe6268207ac9fd59c257b0dcffcf47d5ced5d2.jpg)

<a id="paper-0937"></a>

## Proximal-IMH: Proximal Posterior Proposals for Independent Metropolis–Hastings with Approximate Operators


### Images

![07b3e17b35a92ef9f2ca6190ec57636de4f361e6c9f0b13a45e77a8815aa0f6f.jpg](../icml_results/954_Proximal-IMH_%20Proximal%20Posterior%20Proposals%20for%20Independent%20Metropolis%E2%80%93Hastings%20with%20Approximate%20Oper/images/07b3e17b35a92ef9f2ca6190ec57636de4f361e6c9f0b13a45e77a8815aa0f6f.jpg)

![1c1b9dd73562e32de2da1496a732cad5837bebd503bbe513cf6b62d1168b90eb.jpg](../icml_results/954_Proximal-IMH_%20Proximal%20Posterior%20Proposals%20for%20Independent%20Metropolis%E2%80%93Hastings%20with%20Approximate%20Oper/images/1c1b9dd73562e32de2da1496a732cad5837bebd503bbe513cf6b62d1168b90eb.jpg)

![23c513e143abb340a8f55c6c10137e790fa7b2bc87cabd52cb58bfe8ce03e84e.jpg](../icml_results/954_Proximal-IMH_%20Proximal%20Posterior%20Proposals%20for%20Independent%20Metropolis%E2%80%93Hastings%20with%20Approximate%20Oper/images/23c513e143abb340a8f55c6c10137e790fa7b2bc87cabd52cb58bfe8ce03e84e.jpg)

![39dad0e7264b4a558c2c36ae9bd9a4f6365a4138ad0c74a4828d95a0620a62e3.jpg](../icml_results/954_Proximal-IMH_%20Proximal%20Posterior%20Proposals%20for%20Independent%20Metropolis%E2%80%93Hastings%20with%20Approximate%20Oper/images/39dad0e7264b4a558c2c36ae9bd9a4f6365a4138ad0c74a4828d95a0620a62e3.jpg)

![5b367a0bac7071ba87b4473d4e650c489d3c6a98302a7df324babef001e0ad70.jpg](../icml_results/954_Proximal-IMH_%20Proximal%20Posterior%20Proposals%20for%20Independent%20Metropolis%E2%80%93Hastings%20with%20Approximate%20Oper/images/5b367a0bac7071ba87b4473d4e650c489d3c6a98302a7df324babef001e0ad70.jpg)

![75fa88f642d008633e3207554fe9286f75a7e633d4b12736862c8513c66d80a9.jpg](../icml_results/954_Proximal-IMH_%20Proximal%20Posterior%20Proposals%20for%20Independent%20Metropolis%E2%80%93Hastings%20with%20Approximate%20Oper/images/75fa88f642d008633e3207554fe9286f75a7e633d4b12736862c8513c66d80a9.jpg)

![7ac9049826a78c24a21585b6e82a1e6004a9f04b374ca5dc41e4382ccf461e40.jpg](../icml_results/954_Proximal-IMH_%20Proximal%20Posterior%20Proposals%20for%20Independent%20Metropolis%E2%80%93Hastings%20with%20Approximate%20Oper/images/7ac9049826a78c24a21585b6e82a1e6004a9f04b374ca5dc41e4382ccf461e40.jpg)

![86a8abe85d0ab292fd6ee3fe24d2e2880600f9181a7aa1021ddf3fe2a8c6b409.jpg](../icml_results/954_Proximal-IMH_%20Proximal%20Posterior%20Proposals%20for%20Independent%20Metropolis%E2%80%93Hastings%20with%20Approximate%20Oper/images/86a8abe85d0ab292fd6ee3fe24d2e2880600f9181a7aa1021ddf3fe2a8c6b409.jpg)

![93a11bf2b491fbdbde9424f55fb4d5c792fc99097eb891889ac5235f726934e0.jpg](../icml_results/954_Proximal-IMH_%20Proximal%20Posterior%20Proposals%20for%20Independent%20Metropolis%E2%80%93Hastings%20with%20Approximate%20Oper/images/93a11bf2b491fbdbde9424f55fb4d5c792fc99097eb891889ac5235f726934e0.jpg)

![9dfe5d9e6710b93c93ef58091171d8753a5d4c35e40b6fde7ea3bd9726c03692.jpg](../icml_results/954_Proximal-IMH_%20Proximal%20Posterior%20Proposals%20for%20Independent%20Metropolis%E2%80%93Hastings%20with%20Approximate%20Oper/images/9dfe5d9e6710b93c93ef58091171d8753a5d4c35e40b6fde7ea3bd9726c03692.jpg)

![a042b182ca351fe0eac8802cca36dfd321312db283783040436df6aa32bac913.jpg](../icml_results/954_Proximal-IMH_%20Proximal%20Posterior%20Proposals%20for%20Independent%20Metropolis%E2%80%93Hastings%20with%20Approximate%20Oper/images/a042b182ca351fe0eac8802cca36dfd321312db283783040436df6aa32bac913.jpg)

![a2e50cb73fa37abb2bc2fefdc0aa768790c98aadfb1c14741c4fe13ba2acd3d1.jpg](../icml_results/954_Proximal-IMH_%20Proximal%20Posterior%20Proposals%20for%20Independent%20Metropolis%E2%80%93Hastings%20with%20Approximate%20Oper/images/a2e50cb73fa37abb2bc2fefdc0aa768790c98aadfb1c14741c4fe13ba2acd3d1.jpg)

![d59d9369b97839b7c44e101180717a0ee30cee4716b88a8bf8406999f728d345.jpg](../icml_results/954_Proximal-IMH_%20Proximal%20Posterior%20Proposals%20for%20Independent%20Metropolis%E2%80%93Hastings%20with%20Approximate%20Oper/images/d59d9369b97839b7c44e101180717a0ee30cee4716b88a8bf8406999f728d345.jpg)

![e597e1b8aab1deeff527d4aa17ff753ceb6cd7a8eb12b4e73649f40340c7d6aa.jpg](../icml_results/954_Proximal-IMH_%20Proximal%20Posterior%20Proposals%20for%20Independent%20Metropolis%E2%80%93Hastings%20with%20Approximate%20Oper/images/e597e1b8aab1deeff527d4aa17ff753ceb6cd7a8eb12b4e73649f40340c7d6aa.jpg)

![f1a1323dc13d423e4fdd19bcf05157fa585d3c0c7349eb9528c84a6b1e82f41a.jpg](../icml_results/954_Proximal-IMH_%20Proximal%20Posterior%20Proposals%20for%20Independent%20Metropolis%E2%80%93Hastings%20with%20Approximate%20Oper/images/f1a1323dc13d423e4fdd19bcf05157fa585d3c0c7349eb9528c84a6b1e82f41a.jpg)

### Tables

![af37d3fac52fc87584d8324e5efbc992038a967d0d554daf484dacfbf279f76d.jpg](../icml_results/954_Proximal-IMH_%20Proximal%20Posterior%20Proposals%20for%20Independent%20Metropolis%E2%80%93Hastings%20with%20Approximate%20Oper/tables/af37d3fac52fc87584d8324e5efbc992038a967d0d554daf484dacfbf279f76d.jpg)

![ceacf8d41f782a91e125373d7d5a0b6bc866d443baf88f6e1f90b070fe3f3db1.jpg](../icml_results/954_Proximal-IMH_%20Proximal%20Posterior%20Proposals%20for%20Independent%20Metropolis%E2%80%93Hastings%20with%20Approximate%20Oper/tables/ceacf8d41f782a91e125373d7d5a0b6bc866d443baf88f6e1f90b070fe3f3db1.jpg)

<a id="paper-0938"></a>

## FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion


### Images

![00b734d9ec524fb2269de72dc94fa20f88d2fac9c866ceaca8be54d2249ae3f7.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/images/00b734d9ec524fb2269de72dc94fa20f88d2fac9c866ceaca8be54d2249ae3f7.jpg)

![1981a4ed17ac7b9dc9eb32ddfb03e647668bf7bdd12a35c0cc67bf16b46435c1.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/images/1981a4ed17ac7b9dc9eb32ddfb03e647668bf7bdd12a35c0cc67bf16b46435c1.jpg)

![216ef24f3e428807dee206d872813bd1a252735665a06cbcf96db7ba24ba28e1.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/images/216ef24f3e428807dee206d872813bd1a252735665a06cbcf96db7ba24ba28e1.jpg)

![7393061753af284746bbb28d273acc716c412331a20088f330a2816c526e9b00.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/images/7393061753af284746bbb28d273acc716c412331a20088f330a2816c526e9b00.jpg)

![83c2592ab935e62f167cda8f7146f4c85c98a24b1228ca9d376c4d319ea08a37.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/images/83c2592ab935e62f167cda8f7146f4c85c98a24b1228ca9d376c4d319ea08a37.jpg)

![89aca2aa8c9c8afca3929b7d9bf543f52e9212623379b0ece38fee0f7623009e.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/images/89aca2aa8c9c8afca3929b7d9bf543f52e9212623379b0ece38fee0f7623009e.jpg)

![97ef05bcff41630550c8d94e68c57419189d0ac0f918cc45f719aa6db2674f19.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/images/97ef05bcff41630550c8d94e68c57419189d0ac0f918cc45f719aa6db2674f19.jpg)

![b2c2a84015dd0913f9a4d2bad240fcd54788a8df70ed19fd8cd4d8c2057c8705.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/images/b2c2a84015dd0913f9a4d2bad240fcd54788a8df70ed19fd8cd4d8c2057c8705.jpg)

![b787ca38a3ba385ee196c02020f0b05bd232699d2fd21dcb4a42f20b38e121cf.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/images/b787ca38a3ba385ee196c02020f0b05bd232699d2fd21dcb4a42f20b38e121cf.jpg)

![be32e4efff7fc59c7aebe47d4867682365b8dd87746aa435cbc4eda6287d5e07.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/images/be32e4efff7fc59c7aebe47d4867682365b8dd87746aa435cbc4eda6287d5e07.jpg)

### Tables

![1c99217fe904cb85aec48b55521754ead3560d10a066cfa627d21b9f2c79ee4e.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/tables/1c99217fe904cb85aec48b55521754ead3560d10a066cfa627d21b9f2c79ee4e.jpg)

![21a8c392857739f627b5e7f37b10279485daa7ac38fd5039cfc046c5195e722e.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/tables/21a8c392857739f627b5e7f37b10279485daa7ac38fd5039cfc046c5195e722e.jpg)

![22457e9eab4e6addab53d08a10be67335fea04852241b16c30860ff384db6526.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/tables/22457e9eab4e6addab53d08a10be67335fea04852241b16c30860ff384db6526.jpg)

![43874b9242db1ca2fcaa58dc8257fe0bdbbca1f583ae2aa807eefab0de520150.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/tables/43874b9242db1ca2fcaa58dc8257fe0bdbbca1f583ae2aa807eefab0de520150.jpg)

![4c0f0c0043442c9a83b4b6041c28b451c8a4cfb335f0be14f4ea3f57dff2116a.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/tables/4c0f0c0043442c9a83b4b6041c28b451c8a4cfb335f0be14f4ea3f57dff2116a.jpg)

![67065c8938a7fc7df1461a81e9e45aff5866815be36d4451049b78a7c06710a6.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/tables/67065c8938a7fc7df1461a81e9e45aff5866815be36d4451049b78a7c06710a6.jpg)

![9d75fb8a7356d884c0e75f33621d8185880a7212bbf12d3eaa0dc52627659b46.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/tables/9d75fb8a7356d884c0e75f33621d8185880a7212bbf12d3eaa0dc52627659b46.jpg)

![b4238c3f41928d4645d865a529ee89b29ebb3facc081be418775b9d3936164f6.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/tables/b4238c3f41928d4645d865a529ee89b29ebb3facc081be418775b9d3936164f6.jpg)

![b75c3247d09bf0ea8c8ddd5831b0ebf933c5031ab8a4d5eca6fe2632bb0fa5c4.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/tables/b75c3247d09bf0ea8c8ddd5831b0ebf933c5031ab8a4d5eca6fe2632bb0fa5c4.jpg)

![be639490aa4c19c6d511f31e709d705ae4334d47dfa23526c9a5d0b3846f396d.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/tables/be639490aa4c19c6d511f31e709d705ae4334d47dfa23526c9a5d0b3846f396d.jpg)

![d84abb2992145af99d4ca6ae67d0889dabcf195a2ae53bae2ac7fd86f2f970dc.jpg](../icml_results/955_FlashBlock_%20Attention%20Caching%20for%20Efficient%20Long-Context%20Block%20Diffusion/tables/d84abb2992145af99d4ca6ae67d0889dabcf195a2ae53bae2ac7fd86f2f970dc.jpg)

<a id="paper-0939"></a>

## Vector Linking via Cross-Model Local Isometric Consistency


### Images

![05fa5a27b1ebab37aaf0f76038cbf2ea510611ef2220322ecb4c8ed46e32b5ce.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/images/05fa5a27b1ebab37aaf0f76038cbf2ea510611ef2220322ecb4c8ed46e32b5ce.jpg)

![1b483d8a32196653e5438e2bd5647a67791ace48299f689bfda1eced8f7268da.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/images/1b483d8a32196653e5438e2bd5647a67791ace48299f689bfda1eced8f7268da.jpg)

![392ac5742f969e301377ae5feb5cf0f61c4c92f7b01948415dafb8fb7f93a8e3.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/images/392ac5742f969e301377ae5feb5cf0f61c4c92f7b01948415dafb8fb7f93a8e3.jpg)

![3e34183e2257c3cc97c9d79654beb0de9caee370b57e7e213af93d605eb14874.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/images/3e34183e2257c3cc97c9d79654beb0de9caee370b57e7e213af93d605eb14874.jpg)

![42c3d3104164bfa3e0fbcd873e54199ac652fa9b80ebcadd526e23390387065f.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/images/42c3d3104164bfa3e0fbcd873e54199ac652fa9b80ebcadd526e23390387065f.jpg)

![4f98db3c9f3584e9dd23fd32223c989dd2c438d3cacfc364e74de284b0f474d7.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/images/4f98db3c9f3584e9dd23fd32223c989dd2c438d3cacfc364e74de284b0f474d7.jpg)

![a00807fb1f0e5e683050eb7998b6070fb8b79ba018bf82df8b701a5a0895db79.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/images/a00807fb1f0e5e683050eb7998b6070fb8b79ba018bf82df8b701a5a0895db79.jpg)

![a39b097450ff0c1f804c5da6107196ef4b1c7f46cb0c9407bbde1bf6345c26c4.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/images/a39b097450ff0c1f804c5da6107196ef4b1c7f46cb0c9407bbde1bf6345c26c4.jpg)

![b6eca3723d5d469c7a754fd7e587aad153e2d67bb7046f3755722435e49083f6.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/images/b6eca3723d5d469c7a754fd7e587aad153e2d67bb7046f3755722435e49083f6.jpg)

![bc7a9ea8d418ad4d313bd3a7f4af6964ed4155712233b1050a0e7162293d88bd.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/images/bc7a9ea8d418ad4d313bd3a7f4af6964ed4155712233b1050a0e7162293d88bd.jpg)

![e57a9603ceb3d8c1012973fb222f618d0f4256f7d52dfe6115e47ffe9d9b39bf.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/images/e57a9603ceb3d8c1012973fb222f618d0f4256f7d52dfe6115e47ffe9d9b39bf.jpg)

### Tables

![0c188381de2c4010f0a090141d98f8d51f290ce61ac489fdc12efe99fddb8a37.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/0c188381de2c4010f0a090141d98f8d51f290ce61ac489fdc12efe99fddb8a37.jpg)

![0d9f39068ff7885f0b2d87442bc3e93b887d4c445ee60a87415b97c2325b318b.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/0d9f39068ff7885f0b2d87442bc3e93b887d4c445ee60a87415b97c2325b318b.jpg)

![1ef2b9341cf26a6a89b3efdde5386410910f44d90384991ac64d1ae1e883a6ed.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/1ef2b9341cf26a6a89b3efdde5386410910f44d90384991ac64d1ae1e883a6ed.jpg)

![365e4c3022b0c051b58e0ec435e80a57137d45874de05ce40a729b4365219ba2.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/365e4c3022b0c051b58e0ec435e80a57137d45874de05ce40a729b4365219ba2.jpg)

![3e729514fca8efc5cc7d112d99e2173f34a7124e4b03cbb575002c7b8a46a96a.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/3e729514fca8efc5cc7d112d99e2173f34a7124e4b03cbb575002c7b8a46a96a.jpg)

![61e0c1ba6693a4f76c0fc7671f1c41cf5d26191b3651a82dc2e1625ac1b33372.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/61e0c1ba6693a4f76c0fc7671f1c41cf5d26191b3651a82dc2e1625ac1b33372.jpg)

![6b8acfeb495e1f5e5bc7097189f285133dbcce32fd26d11b3c431ca9b7dcb1e7.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/6b8acfeb495e1f5e5bc7097189f285133dbcce32fd26d11b3c431ca9b7dcb1e7.jpg)

![6d3ea861776510354fb47b7147c372e4a6d99a1dd055627dc3a656e449dca8ad.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/6d3ea861776510354fb47b7147c372e4a6d99a1dd055627dc3a656e449dca8ad.jpg)

![708cfc86e1abe42a36329b0e66ab0e2eb9ad68aae4fd178199fadeda4c76ccb6.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/708cfc86e1abe42a36329b0e66ab0e2eb9ad68aae4fd178199fadeda4c76ccb6.jpg)

![7107836840a05150c9ef058b8ce920d1a484edf32fdc6fe1e58b056b4358ad3f.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/7107836840a05150c9ef058b8ce920d1a484edf32fdc6fe1e58b056b4358ad3f.jpg)

![71e20def520841f993216c48f566bfc39da91a494a7f6a336b875aee252e2844.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/71e20def520841f993216c48f566bfc39da91a494a7f6a336b875aee252e2844.jpg)

![772891eafaa657e04a5e75a85889c13f6b195fcaa6e7faada5a9e563743a1557.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/772891eafaa657e04a5e75a85889c13f6b195fcaa6e7faada5a9e563743a1557.jpg)

![805b7501b6d43e2f1b26bd462f8d2844ca3db465d5e7da62fb2c062cc9218f44.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/805b7501b6d43e2f1b26bd462f8d2844ca3db465d5e7da62fb2c062cc9218f44.jpg)

![827ce45284a85226e440f8f1851dd5365529f04a6e317ad305f3a05e21d71b78.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/827ce45284a85226e440f8f1851dd5365529f04a6e317ad305f3a05e21d71b78.jpg)

![889c467e10d7c9b34ab3f9f3e87ac6ea0d2ff1fe61621947184a3705a6288ed3.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/889c467e10d7c9b34ab3f9f3e87ac6ea0d2ff1fe61621947184a3705a6288ed3.jpg)

![8ded5c30eaefeeb89d86b35412ae320ebb9361191e165710eb0289a7989c0736.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/8ded5c30eaefeeb89d86b35412ae320ebb9361191e165710eb0289a7989c0736.jpg)

![9c2102dce14679f2d9201b5ff8524d996ea6a9be5ed7dd5d6f2df91d0efc8061.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/9c2102dce14679f2d9201b5ff8524d996ea6a9be5ed7dd5d6f2df91d0efc8061.jpg)

![aac3d1df18eed83fc17f9680913fff7d7ce7c4f775b77b34c6067095074091b3.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/aac3d1df18eed83fc17f9680913fff7d7ce7c4f775b77b34c6067095074091b3.jpg)

![ad950af3a8515a97be78874acb5e38a1c682c8cd58065a7c829135b2309c5e95.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/ad950af3a8515a97be78874acb5e38a1c682c8cd58065a7c829135b2309c5e95.jpg)

![b731b85eeec6772b23616f07e3bcf0f288f9cf3347025d0c7186ec7a40dc389d.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/b731b85eeec6772b23616f07e3bcf0f288f9cf3347025d0c7186ec7a40dc389d.jpg)

![b832fe9853720ddaf8dd7a876a6efe9b7f23512c10a9ddaa760d35b82c2281c5.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/b832fe9853720ddaf8dd7a876a6efe9b7f23512c10a9ddaa760d35b82c2281c5.jpg)

![c354d6743e1a24d55e460b1f6443f795dda4c1b855c4d1652c140595b4ffe29b.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/c354d6743e1a24d55e460b1f6443f795dda4c1b855c4d1652c140595b4ffe29b.jpg)

![c7b9e39606606ebdfd478611c7d1d77d93e4d9222f66d2e5d4d08917b7d2790d.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/c7b9e39606606ebdfd478611c7d1d77d93e4d9222f66d2e5d4d08917b7d2790d.jpg)

![dae601720cd85eadfe3d3e09882156cd0a378840896ba9c901609b555ffb0831.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/dae601720cd85eadfe3d3e09882156cd0a378840896ba9c901609b555ffb0831.jpg)

![e1b0e2b5dda765e6876fcbdf8b89bc5b86446eca686ea376ab328d531d53242f.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/e1b0e2b5dda765e6876fcbdf8b89bc5b86446eca686ea376ab328d531d53242f.jpg)

![e27555699b99378d4db3826ede70bffb1aa37b3ec30d96a3da66aabdd5db87b4.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/e27555699b99378d4db3826ede70bffb1aa37b3ec30d96a3da66aabdd5db87b4.jpg)

![e4be8d672bb96156eef9b109c5df32fbf5a1d228e8b9e3da7ff933959ed73487.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/e4be8d672bb96156eef9b109c5df32fbf5a1d228e8b9e3da7ff933959ed73487.jpg)

![ee3b7b2f0056c343c2a011c32b2ef0e5c795d69c6a9f115d7fdfa0a110977f57.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/ee3b7b2f0056c343c2a011c32b2ef0e5c795d69c6a9f115d7fdfa0a110977f57.jpg)

![f2889916af06107548f94003f7205b0ffb08d696d9a92a95baf1e9ef964e0d45.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/f2889916af06107548f94003f7205b0ffb08d696d9a92a95baf1e9ef964e0d45.jpg)

![f33505af9eafd76c99855f76748a67e61d3878fa71008fce82b94ad8304d5b4b.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/f33505af9eafd76c99855f76748a67e61d3878fa71008fce82b94ad8304d5b4b.jpg)

![f3bd0be96b1a956dad332f6f8e92599adff394a999e18c6917da69bdf92bf1cb.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/f3bd0be96b1a956dad332f6f8e92599adff394a999e18c6917da69bdf92bf1cb.jpg)

![f5f8561635a18872bb28f240d2e6244fadb00af31fc5e05a3b92ae84af3c8ce8.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/f5f8561635a18872bb28f240d2e6244fadb00af31fc5e05a3b92ae84af3c8ce8.jpg)

![fc0e904bbe17a141f24383ea8beb5d547df8435283490f4b8c9894ee09193641.jpg](../icml_results/956_Vector%20Linking%20via%20Cross-Model%20Local%20Isometric%20Consistency/tables/fc0e904bbe17a141f24383ea8beb5d547df8435283490f4b8c9894ee09193641.jpg)

<a id="paper-0940"></a>

## PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making


### Images

![3ccfe2d4eee9483365f7cef1afd9f78e5e6d7bc951221ee8c3e8ec15f188ff4d.jpg](../icml_results/957_PSG-Nav_%20Probabilistic%20Scene%20Graph%20Navigation%20via%20Multiverse%20Decision%20Making/images/3ccfe2d4eee9483365f7cef1afd9f78e5e6d7bc951221ee8c3e8ec15f188ff4d.jpg)

![443da9b1ae28216a49b39adbd43a7a54dab8ace93c13f0ef233d621b029bf1ff.jpg](../icml_results/957_PSG-Nav_%20Probabilistic%20Scene%20Graph%20Navigation%20via%20Multiverse%20Decision%20Making/images/443da9b1ae28216a49b39adbd43a7a54dab8ace93c13f0ef233d621b029bf1ff.jpg)

![4796042c24cd82eb630ebd80b34da6228b02fd3f1b45de4289fad5e59a3e0fb3.jpg](../icml_results/957_PSG-Nav_%20Probabilistic%20Scene%20Graph%20Navigation%20via%20Multiverse%20Decision%20Making/images/4796042c24cd82eb630ebd80b34da6228b02fd3f1b45de4289fad5e59a3e0fb3.jpg)

![485666908863244cf7e9215f64c886e7e1234800c95b888e213cd472fdc0cfb8.jpg](../icml_results/957_PSG-Nav_%20Probabilistic%20Scene%20Graph%20Navigation%20via%20Multiverse%20Decision%20Making/images/485666908863244cf7e9215f64c886e7e1234800c95b888e213cd472fdc0cfb8.jpg)

![53cd41dbe036463682fe2235b6886e0636e01e607db5098a1ee9fa5534254de3.jpg](../icml_results/957_PSG-Nav_%20Probabilistic%20Scene%20Graph%20Navigation%20via%20Multiverse%20Decision%20Making/images/53cd41dbe036463682fe2235b6886e0636e01e607db5098a1ee9fa5534254de3.jpg)

![bc0d07e4d1b28d8c32d49b139bc3e1beeea8a23b7c6319aadb335c9e666dad1d.jpg](../icml_results/957_PSG-Nav_%20Probabilistic%20Scene%20Graph%20Navigation%20via%20Multiverse%20Decision%20Making/images/bc0d07e4d1b28d8c32d49b139bc3e1beeea8a23b7c6319aadb335c9e666dad1d.jpg)

![f58dd9c0119fe0077a97b57c1ae8a8a3ede2da8dbd3383397dae430db8d4792e.jpg](../icml_results/957_PSG-Nav_%20Probabilistic%20Scene%20Graph%20Navigation%20via%20Multiverse%20Decision%20Making/images/f58dd9c0119fe0077a97b57c1ae8a8a3ede2da8dbd3383397dae430db8d4792e.jpg)

### Tables

![164dd2e65850f4bd93e2b66f9ca54ea3f3cbb22bb719f3264f5a13cb62eeca1d.jpg](../icml_results/957_PSG-Nav_%20Probabilistic%20Scene%20Graph%20Navigation%20via%20Multiverse%20Decision%20Making/tables/164dd2e65850f4bd93e2b66f9ca54ea3f3cbb22bb719f3264f5a13cb62eeca1d.jpg)

![16a88e7f6b17e7c54605b1d41a6ded93017515c877704af38cec1c80cf0fc81b.jpg](../icml_results/957_PSG-Nav_%20Probabilistic%20Scene%20Graph%20Navigation%20via%20Multiverse%20Decision%20Making/tables/16a88e7f6b17e7c54605b1d41a6ded93017515c877704af38cec1c80cf0fc81b.jpg)

![387de560b3d0f9fe06bb9dc0fb342b35c1de748ff0ffed52c2781c47684e9dca.jpg](../icml_results/957_PSG-Nav_%20Probabilistic%20Scene%20Graph%20Navigation%20via%20Multiverse%20Decision%20Making/tables/387de560b3d0f9fe06bb9dc0fb342b35c1de748ff0ffed52c2781c47684e9dca.jpg)

![4f5d050106c92d4a75c6f2beee901f46541d91abc67199a436239f6c609e1710.jpg](../icml_results/957_PSG-Nav_%20Probabilistic%20Scene%20Graph%20Navigation%20via%20Multiverse%20Decision%20Making/tables/4f5d050106c92d4a75c6f2beee901f46541d91abc67199a436239f6c609e1710.jpg)

![7a2865b518c4f8413058cf0eb8af751fd171668285d93ff53a403c70f24b71c8.jpg](../icml_results/957_PSG-Nav_%20Probabilistic%20Scene%20Graph%20Navigation%20via%20Multiverse%20Decision%20Making/tables/7a2865b518c4f8413058cf0eb8af751fd171668285d93ff53a403c70f24b71c8.jpg)

![8262382ebe0d70d1b8bbae26d3798ce498f580e34e74958b143ab55205cd51eb.jpg](../icml_results/957_PSG-Nav_%20Probabilistic%20Scene%20Graph%20Navigation%20via%20Multiverse%20Decision%20Making/tables/8262382ebe0d70d1b8bbae26d3798ce498f580e34e74958b143ab55205cd51eb.jpg)

![fb6e808221359b622ad4760a5c5bf9ff1c40170440d66f3f3e1b2bba0f6e04ec.jpg](../icml_results/957_PSG-Nav_%20Probabilistic%20Scene%20Graph%20Navigation%20via%20Multiverse%20Decision%20Making/tables/fb6e808221359b622ad4760a5c5bf9ff1c40170440d66f3f3e1b2bba0f6e04ec.jpg)

![fea17821b1de09c5a409b5b8ec11f7b94b8730fb238975d4d8672121e7f4b44f.jpg](../icml_results/957_PSG-Nav_%20Probabilistic%20Scene%20Graph%20Navigation%20via%20Multiverse%20Decision%20Making/tables/fea17821b1de09c5a409b5b8ec11f7b94b8730fb238975d4d8672121e7f4b44f.jpg)

<a id="paper-0941"></a>

## RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation


### Images

![111bd425b1deb4191c6e3e6d7373c53ec4a965f8bc88e4c23f62d903e3cc34c4.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/images/111bd425b1deb4191c6e3e6d7373c53ec4a965f8bc88e4c23f62d903e3cc34c4.jpg)

![5ec0b6a934caf377346fdcb5b9ef4abd117a3025fd2154c8c48ae49acc052d1a.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/images/5ec0b6a934caf377346fdcb5b9ef4abd117a3025fd2154c8c48ae49acc052d1a.jpg)

![78531b537b391315d7099ecf56a2562436e5359a6f6d6da5dbee11009acdb87a.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/images/78531b537b391315d7099ecf56a2562436e5359a6f6d6da5dbee11009acdb87a.jpg)

![7b9b7eb6c3aecf2cc3da9740c64164ad2c11a82fb6d51bc3c82f9ec71a459477.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/images/7b9b7eb6c3aecf2cc3da9740c64164ad2c11a82fb6d51bc3c82f9ec71a459477.jpg)

![82c7707c7e91614422b536d469433763f6402be7a079decbdfdcc61ebafa50b5.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/images/82c7707c7e91614422b536d469433763f6402be7a079decbdfdcc61ebafa50b5.jpg)

![a7cff315343dde1c745ef6670b1b00ab72079b7af5ad5e71b3049eec4a4a3997.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/images/a7cff315343dde1c745ef6670b1b00ab72079b7af5ad5e71b3049eec4a4a3997.jpg)

![b3eec2c2be579d59d893cd50954396de618854597460e9d3074973f0f179bbcd.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/images/b3eec2c2be579d59d893cd50954396de618854597460e9d3074973f0f179bbcd.jpg)

![b3f512fd352046b2eea0cec2f36ef29592144d16e446fc5cb9d321a03bc54bf5.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/images/b3f512fd352046b2eea0cec2f36ef29592144d16e446fc5cb9d321a03bc54bf5.jpg)

![c01d332fa0d64a128a550a6be0a825bc9ee6c9f1892792291a17930176c934e2.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/images/c01d332fa0d64a128a550a6be0a825bc9ee6c9f1892792291a17930176c934e2.jpg)

![e5bc9149ba907b12f8726f0871ca5f4e70ac55c79edba8221e852113173d017f.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/images/e5bc9149ba907b12f8726f0871ca5f4e70ac55c79edba8221e852113173d017f.jpg)

![fb8157a97b198378c5c467fc98a8a3ef49637843df0b56b23726fdf410187ef7.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/images/fb8157a97b198378c5c467fc98a8a3ef49637843df0b56b23726fdf410187ef7.jpg)

### Tables

![15da00e01681c5f8ed8b514a85fe13f6814b92341d2d6cb9e262826237c73d4c.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/15da00e01681c5f8ed8b514a85fe13f6814b92341d2d6cb9e262826237c73d4c.jpg)

![162c55bcfa0063b18dc6d7cf246b0e34a0a8caa8d8db9c60449d89a044675585.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/162c55bcfa0063b18dc6d7cf246b0e34a0a8caa8d8db9c60449d89a044675585.jpg)

![24d6efbf13f8c7d25535eba81074577ba7c8d92c398f744e38c6405d71f7de4f.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/24d6efbf13f8c7d25535eba81074577ba7c8d92c398f744e38c6405d71f7de4f.jpg)

![2cb162ea7cb470492bc524b51c7268841ed1efa961e8953a07d2f0fc75054ca4.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/2cb162ea7cb470492bc524b51c7268841ed1efa961e8953a07d2f0fc75054ca4.jpg)

![4120e9c50cbef258332360b066b63ace6849a13ed588037ae24844cddda2e1c9.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/4120e9c50cbef258332360b066b63ace6849a13ed588037ae24844cddda2e1c9.jpg)

![78c1fb35f16d1d7b85afeb5a92c162d32239ff2df87de0faf1d16b037c219fc5.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/78c1fb35f16d1d7b85afeb5a92c162d32239ff2df87de0faf1d16b037c219fc5.jpg)

![7cc6deeb42d1ebfd994c15e4d3cc1a5763ce08f3c3789a03b4b567e679ac9895.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/7cc6deeb42d1ebfd994c15e4d3cc1a5763ce08f3c3789a03b4b567e679ac9895.jpg)

![7d70f0a4e3eab9437c7f439831bbe577bdc34335cdcf83452b9789b7471b596b.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/7d70f0a4e3eab9437c7f439831bbe577bdc34335cdcf83452b9789b7471b596b.jpg)

![8b96d7fb5fdad659be81f942a8e8cd60adffbdbb38b24aca970f8dd97d355587.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/8b96d7fb5fdad659be81f942a8e8cd60adffbdbb38b24aca970f8dd97d355587.jpg)

![93244e6829b7b5b66cd600a7c2c7d09d11504d18f6f59b6f690063e077a6badb.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/93244e6829b7b5b66cd600a7c2c7d09d11504d18f6f59b6f690063e077a6badb.jpg)

![ad674a279ec66353007747cd1760a34ad6251dd3491c6f263875263662aa9168.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/ad674a279ec66353007747cd1760a34ad6251dd3491c6f263875263662aa9168.jpg)

![b8837d12bb95c3e40aabaa077bd482a2e7a225cee75c051cfa89c2e4d4d74fb0.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/b8837d12bb95c3e40aabaa077bd482a2e7a225cee75c051cfa89c2e4d4d74fb0.jpg)

![b959e2cf5adc947dbcfaec115f12b2c7d86162787fb5ecb2678645c23fc16850.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/b959e2cf5adc947dbcfaec115f12b2c7d86162787fb5ecb2678645c23fc16850.jpg)

![bdf476469ee31fecacb6904b42e39f20f8847f96f61a7a273f4722421bf06a83.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/bdf476469ee31fecacb6904b42e39f20f8847f96f61a7a273f4722421bf06a83.jpg)

![c00f46b146177fc9e218f5130cdd0857bd8afe03188d30fadb528516ecbdbeb1.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/c00f46b146177fc9e218f5130cdd0857bd8afe03188d30fadb528516ecbdbeb1.jpg)

![ebb7cef9d13cb56dedab36f0b59d6644df4d6c51af857a3a070f61e06d9de47e.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/ebb7cef9d13cb56dedab36f0b59d6644df4d6c51af857a3a070f61e06d9de47e.jpg)

![f8d5395d87bfaa79d81de8e734ee857e99cfa1da0f625500947ae5ae095ebaae.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/f8d5395d87bfaa79d81de8e734ee857e99cfa1da0f625500947ae5ae095ebaae.jpg)

![ffe8911cf0550e8802154fa98d989b6d07e15b185a00445232a4e585ae7cf33d.jpg](../icml_results/958_RoboTwin%202.0_%20A%20Scalable%20Data%20Generator%20and%20Benchmark%20with%20Strong%20Domain%20Randomization%20for%20Robust%20Bi/tables/ffe8911cf0550e8802154fa98d989b6d07e15b185a00445232a4e585ae7cf33d.jpg)

<a id="paper-0942"></a>

## PowerFlow: Unlocking the Dual Nature of LLMs via Principled Distribution Matching


### Images

![122c6ebd7819f3e1e84c8c658165b6887f8dae7259335812d98efd6daa053e7e.jpg](../icml_results/959_PowerFlow_%20Unlocking%20the%20Dual%20Nature%20of%20LLMs%20via%20Principled%20Distribution%20Matching/images/122c6ebd7819f3e1e84c8c658165b6887f8dae7259335812d98efd6daa053e7e.jpg)

![5b2d20e845ff116cf1664138fa8fa13e98daa8ea83fabb262b3532e128134ca0.jpg](../icml_results/959_PowerFlow_%20Unlocking%20the%20Dual%20Nature%20of%20LLMs%20via%20Principled%20Distribution%20Matching/images/5b2d20e845ff116cf1664138fa8fa13e98daa8ea83fabb262b3532e128134ca0.jpg)

![6660a8a593e59f74e01e4246826de518e0931d105818fac070c5f0d563843dc3.jpg](../icml_results/959_PowerFlow_%20Unlocking%20the%20Dual%20Nature%20of%20LLMs%20via%20Principled%20Distribution%20Matching/images/6660a8a593e59f74e01e4246826de518e0931d105818fac070c5f0d563843dc3.jpg)

![7d5251298232efc3a6151c27f8886143fc7ab0a14b24e20c63c004e8e62d8a92.jpg](../icml_results/959_PowerFlow_%20Unlocking%20the%20Dual%20Nature%20of%20LLMs%20via%20Principled%20Distribution%20Matching/images/7d5251298232efc3a6151c27f8886143fc7ab0a14b24e20c63c004e8e62d8a92.jpg)

![90394043665fbf4daa639e013316537b76254d73f8a5c76a8da7a26a00bd099a.jpg](../icml_results/959_PowerFlow_%20Unlocking%20the%20Dual%20Nature%20of%20LLMs%20via%20Principled%20Distribution%20Matching/images/90394043665fbf4daa639e013316537b76254d73f8a5c76a8da7a26a00bd099a.jpg)

![cada92e3141950fe8d0d9fdb04f12a6392f17245dce6faa4684bd5eb664a5c4e.jpg](../icml_results/959_PowerFlow_%20Unlocking%20the%20Dual%20Nature%20of%20LLMs%20via%20Principled%20Distribution%20Matching/images/cada92e3141950fe8d0d9fdb04f12a6392f17245dce6faa4684bd5eb664a5c4e.jpg)

![da9acf61d7e789fc1616d25a6ca8661665694b2f1acaf6080afbb42a158450c9.jpg](../icml_results/959_PowerFlow_%20Unlocking%20the%20Dual%20Nature%20of%20LLMs%20via%20Principled%20Distribution%20Matching/images/da9acf61d7e789fc1616d25a6ca8661665694b2f1acaf6080afbb42a158450c9.jpg)

![f38b832a1d5b28b4ec7e24decb56a8c49001c7bd7b3d361450b78e7af3f48cc0.jpg](../icml_results/959_PowerFlow_%20Unlocking%20the%20Dual%20Nature%20of%20LLMs%20via%20Principled%20Distribution%20Matching/images/f38b832a1d5b28b4ec7e24decb56a8c49001c7bd7b3d361450b78e7af3f48cc0.jpg)

### Tables

![6940876601da0b6c928cec4d34b4ab2bee1eb319ee4077a81d26bd547b35daa2.jpg](../icml_results/959_PowerFlow_%20Unlocking%20the%20Dual%20Nature%20of%20LLMs%20via%20Principled%20Distribution%20Matching/tables/6940876601da0b6c928cec4d34b4ab2bee1eb319ee4077a81d26bd547b35daa2.jpg)

![6b610b07169533629c8c02dc1a7e8ee74b71f7dc4f77d7c6ac91c94f2b9b65d3.jpg](../icml_results/959_PowerFlow_%20Unlocking%20the%20Dual%20Nature%20of%20LLMs%20via%20Principled%20Distribution%20Matching/tables/6b610b07169533629c8c02dc1a7e8ee74b71f7dc4f77d7c6ac91c94f2b9b65d3.jpg)

![d3fb8f162a58625b158d86227eb2d7556ce9a43e691c1eed1af6b4786d14232c.jpg](../icml_results/959_PowerFlow_%20Unlocking%20the%20Dual%20Nature%20of%20LLMs%20via%20Principled%20Distribution%20Matching/tables/d3fb8f162a58625b158d86227eb2d7556ce9a43e691c1eed1af6b4786d14232c.jpg)

<a id="paper-0943"></a>

## Task-Aware Structured Memory for Dynamic Multi-modal In-Context Learning


### Images

![1a84c591655ce82c8fd05f9550ea033f5a9b606a327efd29a16c834a1900dce7.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/images/1a84c591655ce82c8fd05f9550ea033f5a9b606a327efd29a16c834a1900dce7.jpg)

![1f1563bd19113c5bc75cf5966e652601eecbaf63ff64bd30d76d2ebc21fa6f86.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/images/1f1563bd19113c5bc75cf5966e652601eecbaf63ff64bd30d76d2ebc21fa6f86.jpg)

![638007f714c395082de9f5c62b54043c5cafd43a4b972e89b2ca2a8eb76ddc48.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/images/638007f714c395082de9f5c62b54043c5cafd43a4b972e89b2ca2a8eb76ddc48.jpg)

![6b95831e0adfabd23b7306d260931733105193936fe5f9146430dde10cea69ad.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/images/6b95831e0adfabd23b7306d260931733105193936fe5f9146430dde10cea69ad.jpg)

![b2db2c88116fc0f1d49c022694b7f0f317462ec883bf02085a1c839d02068b2c.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/images/b2db2c88116fc0f1d49c022694b7f0f317462ec883bf02085a1c839d02068b2c.jpg)

![b7e68ea712a2434f7567a2aeabf2f0c694648e06909ea77151f0474a6511b718.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/images/b7e68ea712a2434f7567a2aeabf2f0c694648e06909ea77151f0474a6511b718.jpg)

### Tables

![2c0807e8dd9e075e663ddefc2fc3d3b7e27a44a00e1d3b14c8e41b75a241f7cf.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/tables/2c0807e8dd9e075e663ddefc2fc3d3b7e27a44a00e1d3b14c8e41b75a241f7cf.jpg)

![2f087dc01c9e8e0e72f941d9a8bcfe463906b3807c84353c8b8b68c473778793.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/tables/2f087dc01c9e8e0e72f941d9a8bcfe463906b3807c84353c8b8b68c473778793.jpg)

![32ff8a0c6a6e2044e0b8d1015f169f1c85f345800992dba0d6d3ceb3ef8dc6ac.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/tables/32ff8a0c6a6e2044e0b8d1015f169f1c85f345800992dba0d6d3ceb3ef8dc6ac.jpg)

![447840c63ded6bc0bfadbb157cfa4d64e5b6fbc9150c2bae443a7730d07426e6.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/tables/447840c63ded6bc0bfadbb157cfa4d64e5b6fbc9150c2bae443a7730d07426e6.jpg)

![5c9e94e9543361047502b2661fac67bfe9659215e57b4dc6a3da4dfff65e8eee.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/tables/5c9e94e9543361047502b2661fac67bfe9659215e57b4dc6a3da4dfff65e8eee.jpg)

![7033ef7dca8a4cc0e383d800a937bcd330f49e131ad61294d8b320eeed847f88.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/tables/7033ef7dca8a4cc0e383d800a937bcd330f49e131ad61294d8b320eeed847f88.jpg)

![7617edfacca461c2b8e256e0257485ba2a1665cf6fb6967af065e2ff88400c8d.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/tables/7617edfacca461c2b8e256e0257485ba2a1665cf6fb6967af065e2ff88400c8d.jpg)

![8e006315eecdca23ce2e414fb96515f6ba2dd0050b0b1987d63b2409c53a3b2c.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/tables/8e006315eecdca23ce2e414fb96515f6ba2dd0050b0b1987d63b2409c53a3b2c.jpg)

![aec456b8fbab2f79f18aadf5616a4d86438719e67ad2ae6ae2890e2a9a83888f.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/tables/aec456b8fbab2f79f18aadf5616a4d86438719e67ad2ae6ae2890e2a9a83888f.jpg)

![c914d48c21b13d2b41c79c94c442715710dd8df5d1c04255d57995f908b5179e.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/tables/c914d48c21b13d2b41c79c94c442715710dd8df5d1c04255d57995f908b5179e.jpg)

![e0c838bca5489ca9b80283feecaca48e77ec36a4d239a3ce5820929ed562af77.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/tables/e0c838bca5489ca9b80283feecaca48e77ec36a4d239a3ce5820929ed562af77.jpg)

![f1fe36e90b6a19772f931bc61b1d9d155f4644fd82e57490548b44ca0f01f25f.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/tables/f1fe36e90b6a19772f931bc61b1d9d155f4644fd82e57490548b44ca0f01f25f.jpg)

![fd9b137c77bbd498bf46ff98d8275b0b19636aba6367477335b1fda10bfee2c9.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/tables/fd9b137c77bbd498bf46ff98d8275b0b19636aba6367477335b1fda10bfee2c9.jpg)

![ff3dae64ef6bf882428aae608bf0e1520151e2f3c764cd55de3c7d04b1e73767.jpg](../icml_results/960_Task-Aware%20Structured%20Memory%20for%20Dynamic%20Multi-modal%20In-Context%20Learning/tables/ff3dae64ef6bf882428aae608bf0e1520151e2f3c764cd55de3c7d04b1e73767.jpg)

<a id="paper-0944"></a>

## Mind the Gap: Catching Hallucinations via Evidence Drop on the Reasoning Manifold


### Images

![0071e71a2476516115d368807618648d6b53e278e61244a04bd2e43111c8a269.jpg](../icml_results/961_Mind%20the%20Gap_%20Catching%20Hallucinations%20via%20Evidence%20Drop%20on%20the%20Reasoning%20Manifold/images/0071e71a2476516115d368807618648d6b53e278e61244a04bd2e43111c8a269.jpg)

![81f9dd1800794dfe3a94bdf5f7c631d02349853b2a84967c7ba7e54b879a988a.jpg](../icml_results/961_Mind%20the%20Gap_%20Catching%20Hallucinations%20via%20Evidence%20Drop%20on%20the%20Reasoning%20Manifold/images/81f9dd1800794dfe3a94bdf5f7c631d02349853b2a84967c7ba7e54b879a988a.jpg)

![9f88bea0b2769ca8c36a9187c0d8717b7ab64e00a5cd8012621344484ed75f7e.jpg](../icml_results/961_Mind%20the%20Gap_%20Catching%20Hallucinations%20via%20Evidence%20Drop%20on%20the%20Reasoning%20Manifold/images/9f88bea0b2769ca8c36a9187c0d8717b7ab64e00a5cd8012621344484ed75f7e.jpg)

![d23070c1ad6114fafd6cc0fd77407cd30d3b37fa5b0d15491a13c2dfad9a5be8.jpg](../icml_results/961_Mind%20the%20Gap_%20Catching%20Hallucinations%20via%20Evidence%20Drop%20on%20the%20Reasoning%20Manifold/images/d23070c1ad6114fafd6cc0fd77407cd30d3b37fa5b0d15491a13c2dfad9a5be8.jpg)

### Tables

![1571a8a0c5456bba64acd2cd22b96a25e3c338aa84d083a0e83de03c2a20c6c4.jpg](../icml_results/961_Mind%20the%20Gap_%20Catching%20Hallucinations%20via%20Evidence%20Drop%20on%20the%20Reasoning%20Manifold/tables/1571a8a0c5456bba64acd2cd22b96a25e3c338aa84d083a0e83de03c2a20c6c4.jpg)

![4e2a6d6288e819cf37c85f1449e18c26dd55b9fe500184a8e88131a97e84a3c2.jpg](../icml_results/961_Mind%20the%20Gap_%20Catching%20Hallucinations%20via%20Evidence%20Drop%20on%20the%20Reasoning%20Manifold/tables/4e2a6d6288e819cf37c85f1449e18c26dd55b9fe500184a8e88131a97e84a3c2.jpg)

![514f81dffe2933a000a68618ced511ec0eebffbaf7086761558ca6fb3611c06d.jpg](../icml_results/961_Mind%20the%20Gap_%20Catching%20Hallucinations%20via%20Evidence%20Drop%20on%20the%20Reasoning%20Manifold/tables/514f81dffe2933a000a68618ced511ec0eebffbaf7086761558ca6fb3611c06d.jpg)

![740f46feaaaf3fe9911a954a35b47f1b054406af672d96c20eb7baacb1989c1f.jpg](../icml_results/961_Mind%20the%20Gap_%20Catching%20Hallucinations%20via%20Evidence%20Drop%20on%20the%20Reasoning%20Manifold/tables/740f46feaaaf3fe9911a954a35b47f1b054406af672d96c20eb7baacb1989c1f.jpg)

![7786fd1c9920b7a3a1768af72e914a977b343ac3b2ffb73bdecea6413c4959f7.jpg](../icml_results/961_Mind%20the%20Gap_%20Catching%20Hallucinations%20via%20Evidence%20Drop%20on%20the%20Reasoning%20Manifold/tables/7786fd1c9920b7a3a1768af72e914a977b343ac3b2ffb73bdecea6413c4959f7.jpg)

![850e586641912e840227e9b2e47398f5af3586ba3361fb1a858d1589b6d9f711.jpg](../icml_results/961_Mind%20the%20Gap_%20Catching%20Hallucinations%20via%20Evidence%20Drop%20on%20the%20Reasoning%20Manifold/tables/850e586641912e840227e9b2e47398f5af3586ba3361fb1a858d1589b6d9f711.jpg)

![a6331edf58e611497aa865eeab526c5c849aeddae970f454e99ccea9f22448e3.jpg](../icml_results/961_Mind%20the%20Gap_%20Catching%20Hallucinations%20via%20Evidence%20Drop%20on%20the%20Reasoning%20Manifold/tables/a6331edf58e611497aa865eeab526c5c849aeddae970f454e99ccea9f22448e3.jpg)

![afc16dc2223894d8c0f1fd9bc53e85d1768b3ff07dadedae6271dfe7f5408dcd.jpg](../icml_results/961_Mind%20the%20Gap_%20Catching%20Hallucinations%20via%20Evidence%20Drop%20on%20the%20Reasoning%20Manifold/tables/afc16dc2223894d8c0f1fd9bc53e85d1768b3ff07dadedae6271dfe7f5408dcd.jpg)

![b63dff6f5ca845ba19f89d46a325ef83fecdb866cedbf1882b2ab364864cc26e.jpg](../icml_results/961_Mind%20the%20Gap_%20Catching%20Hallucinations%20via%20Evidence%20Drop%20on%20the%20Reasoning%20Manifold/tables/b63dff6f5ca845ba19f89d46a325ef83fecdb866cedbf1882b2ab364864cc26e.jpg)

![bbbc6ae8c86f1bc4874285e05f5cff3fa22a4466d2bafcc01cd979e64224decc.jpg](../icml_results/961_Mind%20the%20Gap_%20Catching%20Hallucinations%20via%20Evidence%20Drop%20on%20the%20Reasoning%20Manifold/tables/bbbc6ae8c86f1bc4874285e05f5cff3fa22a4466d2bafcc01cd979e64224decc.jpg)

<a id="paper-0945"></a>

## OPT-Engine: Benchmarking the Limits of LLMs in Optimization Modeling via Complexity Scaling


### Images

![0e02465078a1faef9c1308d81e4b4f38590a458955f93b84d6d16e82bda4f2d7.jpg](../icml_results/962_OPT-Engine_%20Benchmarking%20the%20Limits%20of%20LLMs%20in%20Optimization%20Modeling%20via%20Complexity%20Scaling/images/0e02465078a1faef9c1308d81e4b4f38590a458955f93b84d6d16e82bda4f2d7.jpg)

![333d409228fe5451732719847c0f9be2548eb0293fbe62d898c104e4de7e60d7.jpg](../icml_results/962_OPT-Engine_%20Benchmarking%20the%20Limits%20of%20LLMs%20in%20Optimization%20Modeling%20via%20Complexity%20Scaling/images/333d409228fe5451732719847c0f9be2548eb0293fbe62d898c104e4de7e60d7.jpg)

![3817b89f3c9134b3fa86b70009fa6c6bc894ad5fa40a60f7dba6fabe884b5209.jpg](../icml_results/962_OPT-Engine_%20Benchmarking%20the%20Limits%20of%20LLMs%20in%20Optimization%20Modeling%20via%20Complexity%20Scaling/images/3817b89f3c9134b3fa86b70009fa6c6bc894ad5fa40a60f7dba6fabe884b5209.jpg)

![436015c6ccd2c9b43005a656f3d0568f67d710ee3a07ac6140551ed6d832d361.jpg](../icml_results/962_OPT-Engine_%20Benchmarking%20the%20Limits%20of%20LLMs%20in%20Optimization%20Modeling%20via%20Complexity%20Scaling/images/436015c6ccd2c9b43005a656f3d0568f67d710ee3a07ac6140551ed6d832d361.jpg)

![8f5dc214cf49997721fb3d64ed5129db64aa41ec46de90f0276eec696b6397cc.jpg](../icml_results/962_OPT-Engine_%20Benchmarking%20the%20Limits%20of%20LLMs%20in%20Optimization%20Modeling%20via%20Complexity%20Scaling/images/8f5dc214cf49997721fb3d64ed5129db64aa41ec46de90f0276eec696b6397cc.jpg)

![95895faa3a492ed7f0f8d2a582b18142e7805c0515f1adb3436bf2512ce1f141.jpg](../icml_results/962_OPT-Engine_%20Benchmarking%20the%20Limits%20of%20LLMs%20in%20Optimization%20Modeling%20via%20Complexity%20Scaling/images/95895faa3a492ed7f0f8d2a582b18142e7805c0515f1adb3436bf2512ce1f141.jpg)

![abdff5000b28906b66d1cb286aa616ba5e947f596bb8abf7a7f75819d6343a8e.jpg](../icml_results/962_OPT-Engine_%20Benchmarking%20the%20Limits%20of%20LLMs%20in%20Optimization%20Modeling%20via%20Complexity%20Scaling/images/abdff5000b28906b66d1cb286aa616ba5e947f596bb8abf7a7f75819d6343a8e.jpg)

![b90117134ef0e2fa19e90d44a9a36865ebef6307e14472c94fa78a04996149ef.jpg](../icml_results/962_OPT-Engine_%20Benchmarking%20the%20Limits%20of%20LLMs%20in%20Optimization%20Modeling%20via%20Complexity%20Scaling/images/b90117134ef0e2fa19e90d44a9a36865ebef6307e14472c94fa78a04996149ef.jpg)

![e4d2029902964671857fed2dee66d5372e7eb3d6665a06cd4f11b9a9ef1fa2e3.jpg](../icml_results/962_OPT-Engine_%20Benchmarking%20the%20Limits%20of%20LLMs%20in%20Optimization%20Modeling%20via%20Complexity%20Scaling/images/e4d2029902964671857fed2dee66d5372e7eb3d6665a06cd4f11b9a9ef1fa2e3.jpg)

![f9b548c755c654721de174a9979e8fde90991d959c85b4c52ccd24011103da1c.jpg](../icml_results/962_OPT-Engine_%20Benchmarking%20the%20Limits%20of%20LLMs%20in%20Optimization%20Modeling%20via%20Complexity%20Scaling/images/f9b548c755c654721de174a9979e8fde90991d959c85b4c52ccd24011103da1c.jpg)

![fa5437ffb333fd64a781a8daf85a1480a507b9d3c013b8d1da4f26d863a4ee99.jpg](../icml_results/962_OPT-Engine_%20Benchmarking%20the%20Limits%20of%20LLMs%20in%20Optimization%20Modeling%20via%20Complexity%20Scaling/images/fa5437ffb333fd64a781a8daf85a1480a507b9d3c013b8d1da4f26d863a4ee99.jpg)

### Tables

![26252471d29501e1e703babc1bec2c9978f184820e93f9faa89e50b6471e97b1.jpg](../icml_results/962_OPT-Engine_%20Benchmarking%20the%20Limits%20of%20LLMs%20in%20Optimization%20Modeling%20via%20Complexity%20Scaling/tables/26252471d29501e1e703babc1bec2c9978f184820e93f9faa89e50b6471e97b1.jpg)

![4b69deba5a0026ad409d9b83053f78481329733e153e7b4c465da043a2cc3bfb.jpg](../icml_results/962_OPT-Engine_%20Benchmarking%20the%20Limits%20of%20LLMs%20in%20Optimization%20Modeling%20via%20Complexity%20Scaling/tables/4b69deba5a0026ad409d9b83053f78481329733e153e7b4c465da043a2cc3bfb.jpg)

![9119d2b179f99ac858f47aa707dff84ba8d7ba2a080378c8988262a2795d2b4a.jpg](../icml_results/962_OPT-Engine_%20Benchmarking%20the%20Limits%20of%20LLMs%20in%20Optimization%20Modeling%20via%20Complexity%20Scaling/tables/9119d2b179f99ac858f47aa707dff84ba8d7ba2a080378c8988262a2795d2b4a.jpg)

![ddb1fc84c61501ae22df25970cb2a3b46bd15ac1bb9551b706e82cc2097bc070.jpg](../icml_results/962_OPT-Engine_%20Benchmarking%20the%20Limits%20of%20LLMs%20in%20Optimization%20Modeling%20via%20Complexity%20Scaling/tables/ddb1fc84c61501ae22df25970cb2a3b46bd15ac1bb9551b706e82cc2097bc070.jpg)

![e9c3875a334576996857788ed37e8f5bc2a3671fe6058b889c7ca3891fa93f78.jpg](../icml_results/962_OPT-Engine_%20Benchmarking%20the%20Limits%20of%20LLMs%20in%20Optimization%20Modeling%20via%20Complexity%20Scaling/tables/e9c3875a334576996857788ed37e8f5bc2a3671fe6058b889c7ca3891fa93f78.jpg)

<a id="paper-0946"></a>

## L-CUBE: Isolating Long-Context Capacity from Knowledge with Controllable Mutual Information Scaling


### Images

![02d1b7a85886b0217f612cafa0c1abc3ee303f89c625e6d6780fe1b4cce7cb1f.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/02d1b7a85886b0217f612cafa0c1abc3ee303f89c625e6d6780fe1b4cce7cb1f.jpg)

![1660af0dc68c82c0325c1a01653190f641da4b5754bdab1e3b81386948fef480.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/1660af0dc68c82c0325c1a01653190f641da4b5754bdab1e3b81386948fef480.jpg)

![1c575a93f77dec5c628da48d6cc30ee71a935a6f734741187ec56c6300f323fb.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/1c575a93f77dec5c628da48d6cc30ee71a935a6f734741187ec56c6300f323fb.jpg)

![1e48276c5b80649e698c9d19a5e7c1ad2e67f618dfee08fe3449c98f1c8f31ec.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/1e48276c5b80649e698c9d19a5e7c1ad2e67f618dfee08fe3449c98f1c8f31ec.jpg)

![1f7ef3b688b24b6b5adb5b8d19c2e0d2b96def20a9322c219385470b7d4e70dc.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/1f7ef3b688b24b6b5adb5b8d19c2e0d2b96def20a9322c219385470b7d4e70dc.jpg)

![2503826f50d98bba7ab33d8a5cf26b8cf48cddb03b51187e0d05b63fbb569932.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/2503826f50d98bba7ab33d8a5cf26b8cf48cddb03b51187e0d05b63fbb569932.jpg)

![2976c5fc52e518c13aef83fb7e0d60c830f2cb8cf9d12eb684ce484d0f390ec7.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/2976c5fc52e518c13aef83fb7e0d60c830f2cb8cf9d12eb684ce484d0f390ec7.jpg)

![2ab7549328ce9cd5bf5b603ce10af6a7e536bf3cea98702aa6bd3d4dabd2aa59.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/2ab7549328ce9cd5bf5b603ce10af6a7e536bf3cea98702aa6bd3d4dabd2aa59.jpg)

![2bf882cf7b9bb7f117b40ba3e61b71525f9e5f0bad3ca8d6b47d102920fedb50.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/2bf882cf7b9bb7f117b40ba3e61b71525f9e5f0bad3ca8d6b47d102920fedb50.jpg)

![344cb6b97c3004b53725c38966ba01875ca0227f81689ab2f15c71c28321d759.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/344cb6b97c3004b53725c38966ba01875ca0227f81689ab2f15c71c28321d759.jpg)

![34ef4bc4523f68933525ecf256e0ca85fc8cf4da868641c623daeb5ad6077b54.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/34ef4bc4523f68933525ecf256e0ca85fc8cf4da868641c623daeb5ad6077b54.jpg)

![3bf28b2fdc24acf6d1ecdf8b24186cc6eb9d3e00fac1328d4c199480b6e3a7bd.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/3bf28b2fdc24acf6d1ecdf8b24186cc6eb9d3e00fac1328d4c199480b6e3a7bd.jpg)

![3d4b5d088768729b49599740aa8b2b26e694f305a940c093917121335956494c.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/3d4b5d088768729b49599740aa8b2b26e694f305a940c093917121335956494c.jpg)

![41d0acd35e2d6846955ac3e6fd3ace07117d88da6526fb5289259dc91d05d301.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/41d0acd35e2d6846955ac3e6fd3ace07117d88da6526fb5289259dc91d05d301.jpg)

![4a3d0204fe7f33c6d54cab865d41c8c1ddc9241b273a5ad81e2a529fb13fff46.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/4a3d0204fe7f33c6d54cab865d41c8c1ddc9241b273a5ad81e2a529fb13fff46.jpg)

![4bb95d1e41d771e4f2cd6b15c36cc0b712971d87cc06b656a34795015a67fa1b.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/4bb95d1e41d771e4f2cd6b15c36cc0b712971d87cc06b656a34795015a67fa1b.jpg)

![52ed8e8a3ef2aa5b82d8c9394b575db6816a754da829eaac84d14f922135fe03.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/52ed8e8a3ef2aa5b82d8c9394b575db6816a754da829eaac84d14f922135fe03.jpg)

![64c83fdfa698fbb0de4db363824f818649bb0a8b2c575178d21debed3ea8c4cd.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/64c83fdfa698fbb0de4db363824f818649bb0a8b2c575178d21debed3ea8c4cd.jpg)

![6b12b18370716dea79e28c380a13b82922059e8deb03785c2b9197032bf9732b.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/6b12b18370716dea79e28c380a13b82922059e8deb03785c2b9197032bf9732b.jpg)

![6c897865c9a53dc4ed735d12392c5c7cf84fda12d919b99a999cfcb9a8d13619.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/6c897865c9a53dc4ed735d12392c5c7cf84fda12d919b99a999cfcb9a8d13619.jpg)

![753a790726d11faced579528d75a797f05073d1eda7f32b51ed41ad58b57ba98.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/753a790726d11faced579528d75a797f05073d1eda7f32b51ed41ad58b57ba98.jpg)

![7926bf7628c83eeeaea2858788cb4f9908325958312bf48e10be7720437abd69.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/7926bf7628c83eeeaea2858788cb4f9908325958312bf48e10be7720437abd69.jpg)

![7ab2219d45190278a8bf84d5e8a615b771f752cfa360163c9736740a6a278d64.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/7ab2219d45190278a8bf84d5e8a615b771f752cfa360163c9736740a6a278d64.jpg)

![7e007a04fd14407fb2de2eb8e98eac87e4d2e827e4470f019e5f0d9368d26b3a.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/7e007a04fd14407fb2de2eb8e98eac87e4d2e827e4470f019e5f0d9368d26b3a.jpg)

![9100f33cb4e79216382cf21bf4d31c8966d06f690c03da2f2002264fa0295dd4.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/9100f33cb4e79216382cf21bf4d31c8966d06f690c03da2f2002264fa0295dd4.jpg)

![9352a3aae59c34a6868c28916a7ca9293f069f8f7f661dba511e14e24da6a769.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/9352a3aae59c34a6868c28916a7ca9293f069f8f7f661dba511e14e24da6a769.jpg)

![94d4ec71e38e6bc5f4c65602cdf457db5ee53e2f1c628ef9637830c8ceb4589b.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/94d4ec71e38e6bc5f4c65602cdf457db5ee53e2f1c628ef9637830c8ceb4589b.jpg)

![96b2cecac666385ee340d327fe6ce208cb241bac84420847eb4b2591f5b55e13.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/96b2cecac666385ee340d327fe6ce208cb241bac84420847eb4b2591f5b55e13.jpg)

![a26169c62d0e81863da4ee127b4e9c14cac118cbf7fe14dd7ceecffe72b681b5.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/a26169c62d0e81863da4ee127b4e9c14cac118cbf7fe14dd7ceecffe72b681b5.jpg)

![a610e846f98a5c0333dda3e79c5147ac9894eb4888cc16df4fa69f70e998a675.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/a610e846f98a5c0333dda3e79c5147ac9894eb4888cc16df4fa69f70e998a675.jpg)

![ac98eaf74c4c0c24b8a336c49609ebbb2a18a9dacf0207bd30a182e619fe5775.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/ac98eaf74c4c0c24b8a336c49609ebbb2a18a9dacf0207bd30a182e619fe5775.jpg)

![aeda64424dcf6fb6bcaa560939c325452eec449cee940f10bf40738100da293f.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/aeda64424dcf6fb6bcaa560939c325452eec449cee940f10bf40738100da293f.jpg)

![b6b8c949e8b7a92f2986c6002771a9057eaa63218e1bed4466c25e1498050d81.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/b6b8c949e8b7a92f2986c6002771a9057eaa63218e1bed4466c25e1498050d81.jpg)

![b93adb3f45b484bb617805dd73088effed3ecd777a17b2ef4a24720eb4251dfe.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/b93adb3f45b484bb617805dd73088effed3ecd777a17b2ef4a24720eb4251dfe.jpg)

![bc1ca028ad62295409759e4e706a98a1ba7c0763ab8cdb91d154d8a0bf90ac66.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/bc1ca028ad62295409759e4e706a98a1ba7c0763ab8cdb91d154d8a0bf90ac66.jpg)

![ced162df5d4a17a5a2ccabedff1640c2bb178044280e219e34dada15e22b3292.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/ced162df5d4a17a5a2ccabedff1640c2bb178044280e219e34dada15e22b3292.jpg)

![dd04b8c75c743e4948fa6c7643c07919774db91d55750a9a27dae2de56e459de.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/dd04b8c75c743e4948fa6c7643c07919774db91d55750a9a27dae2de56e459de.jpg)

![e4b454e58bbeb9b196137ee8bf5216426d84b5a6bb1534a62f004592b4b0f88b.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/e4b454e58bbeb9b196137ee8bf5216426d84b5a6bb1534a62f004592b4b0f88b.jpg)

![ed121d382b577eb51c761657a9e9a0e70b85de8f6d6b053e3be2c7069d67d34c.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/ed121d382b577eb51c761657a9e9a0e70b85de8f6d6b053e3be2c7069d67d34c.jpg)

![ee3623e76d20f0ee2221022974ed7b905e263d6ced7f7a9f2e057930330b019a.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/ee3623e76d20f0ee2221022974ed7b905e263d6ced7f7a9f2e057930330b019a.jpg)

![f5b003fdb36d8de54ae95b163f8145a7d7b65d111ca7b55a76164faa39d19970.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/f5b003fdb36d8de54ae95b163f8145a7d7b65d111ca7b55a76164faa39d19970.jpg)

![f823c4dbbd73fc232d4df95e4a2695a84f7450a0f3c145eed241fd8e0fae7596.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/images/f823c4dbbd73fc232d4df95e4a2695a84f7450a0f3c145eed241fd8e0fae7596.jpg)

### Tables

![3a27673d8039f463b21d864612635ba403d844a139e9834fcf3e0f564cb19147.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/tables/3a27673d8039f463b21d864612635ba403d844a139e9834fcf3e0f564cb19147.jpg)

![61bee4e8d016d64927570f90bd1d705534896ebb741789284355d17ef2322d3d.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/tables/61bee4e8d016d64927570f90bd1d705534896ebb741789284355d17ef2322d3d.jpg)

![c99d258fee349269abd8ff4f96cf271566faf70e2f229141bf3665d35c3e1497.jpg](../icml_results/963_L-CUBE_%20Isolating%20Long-Context%20Capacity%20from%20Knowledge%20with%20Controllable%20Mutual%20Information%20Scaling/tables/c99d258fee349269abd8ff4f96cf271566faf70e2f229141bf3665d35c3e1497.jpg)

<a id="paper-0947"></a>

## MARS: Modular Agent with Reflective Search for Automated AI Research


### Images

![0161390afa7995d06e7024cf253e17efe6473d9db773bce5618883a08d358176.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/images/0161390afa7995d06e7024cf253e17efe6473d9db773bce5618883a08d358176.jpg)

![388744d03d67325f801502a1099af4415b11c596384031f71e810cf4e70f5341.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/images/388744d03d67325f801502a1099af4415b11c596384031f71e810cf4e70f5341.jpg)

![734ea7098990e429945656bde1ab0e45e2701c4832e4ee0ec70fefccc49e353a.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/images/734ea7098990e429945656bde1ab0e45e2701c4832e4ee0ec70fefccc49e353a.jpg)

![920ba3b100d7af8c2dfeceee528ba43b894022f28406d20960649d56a452a446.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/images/920ba3b100d7af8c2dfeceee528ba43b894022f28406d20960649d56a452a446.jpg)

![9e2f3ea44a25d4a15bddcb4ab0dcc840a4d352e3dcf266a3200c7abfde6523f6.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/images/9e2f3ea44a25d4a15bddcb4ab0dcc840a4d352e3dcf266a3200c7abfde6523f6.jpg)

![afac1956a6888ad300478e761dba232a5494a0e4899a8dc85f7ccbe76af70c1e.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/images/afac1956a6888ad300478e761dba232a5494a0e4899a8dc85f7ccbe76af70c1e.jpg)

![b176b7aad4d28329214f330c4f5453ea89ce6b5dda9f7dfc31079d076bd9c4b7.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/images/b176b7aad4d28329214f330c4f5453ea89ce6b5dda9f7dfc31079d076bd9c4b7.jpg)

![f1ac85ff46fb9be9b756de27dad254eaba2e3e12d90fa0543cf3ab90003ba838.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/images/f1ac85ff46fb9be9b756de27dad254eaba2e3e12d90fa0543cf3ab90003ba838.jpg)

### Tables

![236d1f2dedbfa0b4bc9512c04bad7cb7df754461205d41a2446b2cf13d0f14ec.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/tables/236d1f2dedbfa0b4bc9512c04bad7cb7df754461205d41a2446b2cf13d0f14ec.jpg)

![4d74abe2cccd24d621b23dec95a340fc01c6452f9d43bed4165a97c8b599849f.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/tables/4d74abe2cccd24d621b23dec95a340fc01c6452f9d43bed4165a97c8b599849f.jpg)

![59286702878b4213792c326dd83b9c2fd28e406f40d4d7f19bce4bca962fcfe6.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/tables/59286702878b4213792c326dd83b9c2fd28e406f40d4d7f19bce4bca962fcfe6.jpg)

![72652de496647b1404ff6f3e2024835debbcd8f112515785deb8fe92bf63cc81.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/tables/72652de496647b1404ff6f3e2024835debbcd8f112515785deb8fe92bf63cc81.jpg)

![816e49c0d17955b9dee2a433b5de5adaf0937fcb43937b74c5e905c9f012b580.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/tables/816e49c0d17955b9dee2a433b5de5adaf0937fcb43937b74c5e905c9f012b580.jpg)

![8577059b681582b81ce59643b1a6963b36cae4cf3521f5d0f8ead11a3e73bcd8.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/tables/8577059b681582b81ce59643b1a6963b36cae4cf3521f5d0f8ead11a3e73bcd8.jpg)

![89d250e5ec7410408f494a761edbbea327fea3b8e216edb579e18045a9d77cf2.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/tables/89d250e5ec7410408f494a761edbbea327fea3b8e216edb579e18045a9d77cf2.jpg)

![9777e5265e99870f7a3cea60a4c41e71a1cea360ce9017f323ad80ecfc4d60c8.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/tables/9777e5265e99870f7a3cea60a4c41e71a1cea360ce9017f323ad80ecfc4d60c8.jpg)

![b0501213ca453d867b9e0d5c3d6e1280352307de6314490b64ec0e1ddd379485.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/tables/b0501213ca453d867b9e0d5c3d6e1280352307de6314490b64ec0e1ddd379485.jpg)

![beb93213ad5f91c0ac49806e82f6c55519c197bb2d19369d4b13b749c2db4e45.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/tables/beb93213ad5f91c0ac49806e82f6c55519c197bb2d19369d4b13b749c2db4e45.jpg)

![f465e9ad8c01b693a3868a7b43fb5ecade395e4d4b427670eeeb4993d472ff16.jpg](../icml_results/964_MARS_%20Modular%20Agent%20with%20Reflective%20Search%20for%20Automated%20AI%20Research/tables/f465e9ad8c01b693a3868a7b43fb5ecade395e4d4b427670eeeb4993d472ff16.jpg)

<a id="paper-0948"></a>

## When Preference Labels Fall Short: Aligning Diffusion Models from Real Data


### Images

![01c69415b63214857e179dd82fd5275892e6f72e3afd742360f1ad5df7baca54.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/images/01c69415b63214857e179dd82fd5275892e6f72e3afd742360f1ad5df7baca54.jpg)

![0802aa7525be849d91db494e126eb5815873cc6e3b49021ee03617cad0555575.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/images/0802aa7525be849d91db494e126eb5815873cc6e3b49021ee03617cad0555575.jpg)

![168bcca02f746286a6c70cec36ec293fa00b49add2eec7182b76c6f7090a56f0.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/images/168bcca02f746286a6c70cec36ec293fa00b49add2eec7182b76c6f7090a56f0.jpg)

![49ca001ded3453729dbb4e4d3f4a6aa328574de4ec41202aa0bd356447ea5d22.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/images/49ca001ded3453729dbb4e4d3f4a6aa328574de4ec41202aa0bd356447ea5d22.jpg)

![53a1efaa6c3480db691edd5771e45d3ef3dcfb3a1e532517e16a737618b00fd8.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/images/53a1efaa6c3480db691edd5771e45d3ef3dcfb3a1e532517e16a737618b00fd8.jpg)

![8e56478c64a3c2c16192ec0c58c37d83ee5a77ebb3977a3cc03c6130efc41be0.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/images/8e56478c64a3c2c16192ec0c58c37d83ee5a77ebb3977a3cc03c6130efc41be0.jpg)

![983da7e23718a0df7bcdb6998ecc493d0ffe6bbcd592916660f33c32e5329733.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/images/983da7e23718a0df7bcdb6998ecc493d0ffe6bbcd592916660f33c32e5329733.jpg)

![a38447e6b1ef89e13e42e078fae5b0a7ed8ac191421a6604637f870a569c42d0.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/images/a38447e6b1ef89e13e42e078fae5b0a7ed8ac191421a6604637f870a569c42d0.jpg)

![bcbd054088740e717bc2f9645b327c60cf9187c5ceb153e88c8572987485dfe3.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/images/bcbd054088740e717bc2f9645b327c60cf9187c5ceb153e88c8572987485dfe3.jpg)

![d4c2b0cc910a6168ffb941e97505e04c3896370f2bdd7a578a908674ede8f025.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/images/d4c2b0cc910a6168ffb941e97505e04c3896370f2bdd7a578a908674ede8f025.jpg)

![dcb8ed44ea3b40a90acf55aaa84cb20df53bdbf78856d457dc6769017e2fbd45.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/images/dcb8ed44ea3b40a90acf55aaa84cb20df53bdbf78856d457dc6769017e2fbd45.jpg)

![ed65f423410661594bca8fce1ed42f90fed787327d7729f7b887f4792f009daf.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/images/ed65f423410661594bca8fce1ed42f90fed787327d7729f7b887f4792f009daf.jpg)

### Tables

![1b2a57f61699598cdd3ecd03b98784e008c65eee4ad02a6b45dbfdc2f842cafd.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/tables/1b2a57f61699598cdd3ecd03b98784e008c65eee4ad02a6b45dbfdc2f842cafd.jpg)

![331d7ba6d5140f89d9bd04b43d318d696c3244963c7b9d7d0d54a3b360abf55b.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/tables/331d7ba6d5140f89d9bd04b43d318d696c3244963c7b9d7d0d54a3b360abf55b.jpg)

![411c75d329ec447cec9081a2ed07ea6d7e8eecf62dc99af9a8e0125786c92c03.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/tables/411c75d329ec447cec9081a2ed07ea6d7e8eecf62dc99af9a8e0125786c92c03.jpg)

![4210fec8d211e3de60276c0227f53d712b279173c4eccb7559f15656d845ea69.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/tables/4210fec8d211e3de60276c0227f53d712b279173c4eccb7559f15656d845ea69.jpg)

![44950c37721c7a115083e9615cabc852c34fa6fed14876a7ae2a9c5a7a167abc.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/tables/44950c37721c7a115083e9615cabc852c34fa6fed14876a7ae2a9c5a7a167abc.jpg)

![4fec965658168b260c176305059a08e8f78cf4611732c91dc7eae50d3757ab55.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/tables/4fec965658168b260c176305059a08e8f78cf4611732c91dc7eae50d3757ab55.jpg)

![600125382cc4a8338df1ec56e60c0f00379ba333c05cbe11aeae5b1dcb24e9cf.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/tables/600125382cc4a8338df1ec56e60c0f00379ba333c05cbe11aeae5b1dcb24e9cf.jpg)

![82af749981db17a98bf1eb2903623d4caa076cc2e5684bed2be89ad677db8617.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/tables/82af749981db17a98bf1eb2903623d4caa076cc2e5684bed2be89ad677db8617.jpg)

![9985cd354b2365ac1eb687b6f0dee94d00cdcabac466a8cc14c9407caa384668.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/tables/9985cd354b2365ac1eb687b6f0dee94d00cdcabac466a8cc14c9407caa384668.jpg)

![e7b3211c47d4f14415cb2a90b7f212b3fc57747a7b0c4655868c56a2fe71c136.jpg](../icml_results/965_When%20Preference%20Labels%20Fall%20Short_%20Aligning%20Diffusion%20Models%20from%20Real%20Data/tables/e7b3211c47d4f14415cb2a90b7f212b3fc57747a7b0c4655868c56a2fe71c136.jpg)

<a id="paper-0949"></a>

## Test-Time Learning of Causal Structure from Interventional Data


### Images

![2ca3ebb818b904dae1b4a7e4ab4930b5f1efdbb28c51b6ec005e33a7707f7245.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/images/2ca3ebb818b904dae1b4a7e4ab4930b5f1efdbb28c51b6ec005e33a7707f7245.jpg)

![410a0862a975f12a291a7fbe845c6dbec7ba2ce1f89fc132750425d80ec2dee3.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/images/410a0862a975f12a291a7fbe845c6dbec7ba2ce1f89fc132750425d80ec2dee3.jpg)

![4901c9901a51b44f9c776a21d62711c63585d024c8e1d3560aec0fdce4b7e98a.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/images/4901c9901a51b44f9c776a21d62711c63585d024c8e1d3560aec0fdce4b7e98a.jpg)

![5760930d005610cf4f8872a2a3832da36e9b74fe4ae8c06322fa55fb9aae6d7b.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/images/5760930d005610cf4f8872a2a3832da36e9b74fe4ae8c06322fa55fb9aae6d7b.jpg)

![5d4b52de59f17a46fd0d10bea2f4b5df3d4912de19a3e858b5f9667735ec124e.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/images/5d4b52de59f17a46fd0d10bea2f4b5df3d4912de19a3e858b5f9667735ec124e.jpg)

![647086554fe26c80be3717adf34be4fe25e7a35b4a60724a147bdd68adbcc85a.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/images/647086554fe26c80be3717adf34be4fe25e7a35b4a60724a147bdd68adbcc85a.jpg)

![9a6c3d7c8ae9d2753a9117d6f2b3efcf052243187f781fb75d4e64ada193afbe.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/images/9a6c3d7c8ae9d2753a9117d6f2b3efcf052243187f781fb75d4e64ada193afbe.jpg)

![b4550d2f986ae3b1bebe5e671a63f15381d5d5ffb0c92c720a06bccbebf94513.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/images/b4550d2f986ae3b1bebe5e671a63f15381d5d5ffb0c92c720a06bccbebf94513.jpg)

![cfd571abdf8f9c49ddcf7ddbbc73409d916e49103a4f59d115f7dd0080cc8574.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/images/cfd571abdf8f9c49ddcf7ddbbc73409d916e49103a4f59d115f7dd0080cc8574.jpg)

![d93c25aae7f65830ad90d9b8151ee0a033cf3f58c18672d7c768d9f798ce8986.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/images/d93c25aae7f65830ad90d9b8151ee0a033cf3f58c18672d7c768d9f798ce8986.jpg)

![ef1a0f2c86d269f3bc03bb2c04c6b676033e8638b7e260f7dc6ff4d76719c696.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/images/ef1a0f2c86d269f3bc03bb2c04c6b676033e8638b7e260f7dc6ff4d76719c696.jpg)

![f65c3c42f295e74e7b3328386a0121f346b7b5230dfd78cd3c07aedecb686de3.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/images/f65c3c42f295e74e7b3328386a0121f346b7b5230dfd78cd3c07aedecb686de3.jpg)

![fb2c6e2aad8dbc768b5be8f98e24974827ca1ad8002817046674b9802eec9960.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/images/fb2c6e2aad8dbc768b5be8f98e24974827ca1ad8002817046674b9802eec9960.jpg)

### Tables

![2c6bdef3157097c824eea1d99868e5a535395611b3f149752d14f1d922e237fd.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/tables/2c6bdef3157097c824eea1d99868e5a535395611b3f149752d14f1d922e237fd.jpg)

![2f772b076b1e610f9714be06174aef8a31cf552036a12eb55354da4edc8110aa.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/tables/2f772b076b1e610f9714be06174aef8a31cf552036a12eb55354da4edc8110aa.jpg)

![322d9bfadcecb3ab9d2dcb36d3a4b312213ec0c1090cb418343d7437ea3286a9.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/tables/322d9bfadcecb3ab9d2dcb36d3a4b312213ec0c1090cb418343d7437ea3286a9.jpg)

![54ee0604ef9fffff492cb0f8a53c1cf9f231eb4749cfe227b200a41eab71506a.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/tables/54ee0604ef9fffff492cb0f8a53c1cf9f231eb4749cfe227b200a41eab71506a.jpg)

![617f838585636813a3f78ab8fc94128515afd2697ba4a537d69a992a071e5112.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/tables/617f838585636813a3f78ab8fc94128515afd2697ba4a537d69a992a071e5112.jpg)

![80dc2a9c482e6300126617576d3d078bb50d9df7d77d72b0aa648d7c40bc2a71.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/tables/80dc2a9c482e6300126617576d3d078bb50d9df7d77d72b0aa648d7c40bc2a71.jpg)

![88e2b4b2cb101d9da4e7d8b93cdc9feec841cb53385b1a3ffc5d293f4ee024ab.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/tables/88e2b4b2cb101d9da4e7d8b93cdc9feec841cb53385b1a3ffc5d293f4ee024ab.jpg)

![8f155a56eb04709047375374168f909e87fa7a0c3c7b49486e6b6284f02b75b3.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/tables/8f155a56eb04709047375374168f909e87fa7a0c3c7b49486e6b6284f02b75b3.jpg)

![b77ad2e99eb1e554a59c8887c062d45636d592d43ad31b8185dd77a721b6e455.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/tables/b77ad2e99eb1e554a59c8887c062d45636d592d43ad31b8185dd77a721b6e455.jpg)

![b840b9334ddae5f5dbb2da9893f072df8c3f021269aad9f491aa1e81ae9d7230.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/tables/b840b9334ddae5f5dbb2da9893f072df8c3f021269aad9f491aa1e81ae9d7230.jpg)

![cac490850a2ed34b11aa32f658568713733f022ca3701ef6958e4e0aed9b75c4.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/tables/cac490850a2ed34b11aa32f658568713733f022ca3701ef6958e4e0aed9b75c4.jpg)

![df5600cc1cd3b7a940d5d0a72a1be9e54d3e5ebca57bac0b2c811eef27d701b7.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/tables/df5600cc1cd3b7a940d5d0a72a1be9e54d3e5ebca57bac0b2c811eef27d701b7.jpg)

![e0883d2e54b2ce88c8d5410de0d3299e266d354578d68f008c898ad52b7d8ed4.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/tables/e0883d2e54b2ce88c8d5410de0d3299e266d354578d68f008c898ad52b7d8ed4.jpg)

![e2f9c2c77df3c70a25d02848f87ba6b73852e8a6f97527cd7686fd44c5660792.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/tables/e2f9c2c77df3c70a25d02848f87ba6b73852e8a6f97527cd7686fd44c5660792.jpg)

![ee51ff56231336ddbed56d41144ab2be65e5c2aeb0da2c37e56ff33cad1effd7.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/tables/ee51ff56231336ddbed56d41144ab2be65e5c2aeb0da2c37e56ff33cad1effd7.jpg)

![f0c52e06cc9cfeaabb21d3cb5c8051faf7f437c35b0c5cfbcbed34aa80f93f67.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/tables/f0c52e06cc9cfeaabb21d3cb5c8051faf7f437c35b0c5cfbcbed34aa80f93f67.jpg)

![f152bc4574fed0215f062dc6f56160f5969aaf77fc19a7a63c0c34b383a1e296.jpg](../icml_results/966_Test-Time%20Learning%20of%20Causal%20Structure%20from%20Interventional%20Data/tables/f152bc4574fed0215f062dc6f56160f5969aaf77fc19a7a63c0c34b383a1e296.jpg)

<a id="paper-0950"></a>

## Foundation VAEs for CT Reconstruction, Augmentation, and Generation


### Images

![085ef60553f775e24ba74531be1622d3b29c50f9daaee89ae4e660e94d380038.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/images/085ef60553f775e24ba74531be1622d3b29c50f9daaee89ae4e660e94d380038.jpg)

![0a7abba0d05ce2f458b53f4406820a8b01acabf9a360228b80a6a265116a7d42.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/images/0a7abba0d05ce2f458b53f4406820a8b01acabf9a360228b80a6a265116a7d42.jpg)

![0fd6be61a4228bbe13add9d43884a5da6323421e4337bcba67aea2b365de212f.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/images/0fd6be61a4228bbe13add9d43884a5da6323421e4337bcba67aea2b365de212f.jpg)

![3765229ea12b3693c3ac4f545a8f90f0f3be2d429dbff1003aefd1b2a0735173.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/images/3765229ea12b3693c3ac4f545a8f90f0f3be2d429dbff1003aefd1b2a0735173.jpg)

![5acfb0a77004e5aaf31ff5b4422e729e66a2acb5c1425ecfd969c9f3146a273c.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/images/5acfb0a77004e5aaf31ff5b4422e729e66a2acb5c1425ecfd969c9f3146a273c.jpg)

![89f64018fac1fa24b139d92a6f186fa79be0af053413bfd38e3c5c57084ad96b.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/images/89f64018fac1fa24b139d92a6f186fa79be0af053413bfd38e3c5c57084ad96b.jpg)

![90931ef6abe05afe6bcfec4d6767440a9bf91a0c1b9083234da9a69b327269f1.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/images/90931ef6abe05afe6bcfec4d6767440a9bf91a0c1b9083234da9a69b327269f1.jpg)

![a5ad7b784afb1cf31416c91089e312a72e228f8d7110b38cdf531e932f37dfe6.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/images/a5ad7b784afb1cf31416c91089e312a72e228f8d7110b38cdf531e932f37dfe6.jpg)

![a7559163661966ba5b6e5a3b9a096758c27ff908fc9b8c17988e706b7cb14a58.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/images/a7559163661966ba5b6e5a3b9a096758c27ff908fc9b8c17988e706b7cb14a58.jpg)

![edcb6ebc96d501369c23b3d5b188d32d1f6c8149c6efa8e4e769d4735af3c147.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/images/edcb6ebc96d501369c23b3d5b188d32d1f6c8149c6efa8e4e769d4735af3c147.jpg)

### Tables

![14be795e34fc055227ba2cd37bf64f74a63abdb0e6f132087df10f4e47bec274.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/tables/14be795e34fc055227ba2cd37bf64f74a63abdb0e6f132087df10f4e47bec274.jpg)

![249e98e392e34bce7312533e1732014832045a0cb4921ca14571580ab2f84286.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/tables/249e98e392e34bce7312533e1732014832045a0cb4921ca14571580ab2f84286.jpg)

![49209a38057e78df1488a0ccfee5500d9f35c85c2fd396c5dfdb05e3eed9208f.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/tables/49209a38057e78df1488a0ccfee5500d9f35c85c2fd396c5dfdb05e3eed9208f.jpg)

![542c3721ba9b9360338f24d149f69f3643e82df84ead9e0216742ed7823ed5b2.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/tables/542c3721ba9b9360338f24d149f69f3643e82df84ead9e0216742ed7823ed5b2.jpg)

![5aac1820c159a480d3afde460465843f52756f7ba368d71128dcc4163804f1b3.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/tables/5aac1820c159a480d3afde460465843f52756f7ba368d71128dcc4163804f1b3.jpg)

![69b9b1f3af689ec364b99b9a31d96b55fed1aa0afb1f6d7ffdad63e26347ce43.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/tables/69b9b1f3af689ec364b99b9a31d96b55fed1aa0afb1f6d7ffdad63e26347ce43.jpg)

![82a01a20f8f918d29b0f08238f2641a5e26ed7f9017c7ea6df1731c7fdf27c37.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/tables/82a01a20f8f918d29b0f08238f2641a5e26ed7f9017c7ea6df1731c7fdf27c37.jpg)

![b5f0372409c805d98e4c824f868beae27b5b817664a7c7d444a7bb6bd0eae1bf.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/tables/b5f0372409c805d98e4c824f868beae27b5b817664a7c7d444a7bb6bd0eae1bf.jpg)

![cbe25611874cf7e0c8ef42b0aa619d326900638c73395cd03cb7cf2e350c1650.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/tables/cbe25611874cf7e0c8ef42b0aa619d326900638c73395cd03cb7cf2e350c1650.jpg)

![edd1aae004a962a0f6d64a7f6e28beb030495bca9ba5c22b1f0d20a10c699a7c.jpg](../icml_results/967_Foundation%20VAEs%20for%20CT%20Reconstruction%2C%20Augmentation%2C%20and%20Generation/tables/edd1aae004a962a0f6d64a7f6e28beb030495bca9ba5c22b1f0d20a10c699a7c.jpg)

<a id="paper-0951"></a>

## PRISM: Training-Free Video Anomaly Detection via Intrinsic Statistical Modeling


### Images

![1c923d4ff742f27d6ea8cf1d6c544474745baa81cf266fa01e6e1b15b133f2ae.jpg](../icml_results/968_PRISM_%20Training-Free%20Video%20Anomaly%20Detection%20via%20Intrinsic%20Statistical%20Modeling/images/1c923d4ff742f27d6ea8cf1d6c544474745baa81cf266fa01e6e1b15b133f2ae.jpg)

![31913112ddcca8b061f41d9b55dcb734d8557d2fd394c88aab7124508175a22f.jpg](../icml_results/968_PRISM_%20Training-Free%20Video%20Anomaly%20Detection%20via%20Intrinsic%20Statistical%20Modeling/images/31913112ddcca8b061f41d9b55dcb734d8557d2fd394c88aab7124508175a22f.jpg)

![35ca4834ef7e999d37ec57956df113e0159664315baa9a3adfbbe25417bf6990.jpg](../icml_results/968_PRISM_%20Training-Free%20Video%20Anomaly%20Detection%20via%20Intrinsic%20Statistical%20Modeling/images/35ca4834ef7e999d37ec57956df113e0159664315baa9a3adfbbe25417bf6990.jpg)

![71ebc474dd4e37aabfb5c1f2fb0dcba4d7eb1f46093f52e35a7f4bb1ef2b00df.jpg](../icml_results/968_PRISM_%20Training-Free%20Video%20Anomaly%20Detection%20via%20Intrinsic%20Statistical%20Modeling/images/71ebc474dd4e37aabfb5c1f2fb0dcba4d7eb1f46093f52e35a7f4bb1ef2b00df.jpg)

![806091fd0dc6c88850519cbcc7aaff680b4e4026b95b6af387a3b10e696da376.jpg](../icml_results/968_PRISM_%20Training-Free%20Video%20Anomaly%20Detection%20via%20Intrinsic%20Statistical%20Modeling/images/806091fd0dc6c88850519cbcc7aaff680b4e4026b95b6af387a3b10e696da376.jpg)

![8dd733c09fc7757e9a669ae75157caff1ef6adc6cfae994c7423cb2f172ebccb.jpg](../icml_results/968_PRISM_%20Training-Free%20Video%20Anomaly%20Detection%20via%20Intrinsic%20Statistical%20Modeling/images/8dd733c09fc7757e9a669ae75157caff1ef6adc6cfae994c7423cb2f172ebccb.jpg)

![d062a19d0ffbebf031c6878e0535e290e833dd21056e26b950870dfb9b6f1843.jpg](../icml_results/968_PRISM_%20Training-Free%20Video%20Anomaly%20Detection%20via%20Intrinsic%20Statistical%20Modeling/images/d062a19d0ffbebf031c6878e0535e290e833dd21056e26b950870dfb9b6f1843.jpg)

![dbabf0607b7fd2de96436bf3711e97aad70f3f289273b1a897aef49c9f2c1c6e.jpg](../icml_results/968_PRISM_%20Training-Free%20Video%20Anomaly%20Detection%20via%20Intrinsic%20Statistical%20Modeling/images/dbabf0607b7fd2de96436bf3711e97aad70f3f289273b1a897aef49c9f2c1c6e.jpg)

![e20d31ddf85c47e3a3c873446d3e7029d8cdac58589e40f8bf8f1682282a72c5.jpg](../icml_results/968_PRISM_%20Training-Free%20Video%20Anomaly%20Detection%20via%20Intrinsic%20Statistical%20Modeling/images/e20d31ddf85c47e3a3c873446d3e7029d8cdac58589e40f8bf8f1682282a72c5.jpg)

![f787a48a4732632e74d72abcf415fcec656f21e8dc65dafecc66b16f39b73bac.jpg](../icml_results/968_PRISM_%20Training-Free%20Video%20Anomaly%20Detection%20via%20Intrinsic%20Statistical%20Modeling/images/f787a48a4732632e74d72abcf415fcec656f21e8dc65dafecc66b16f39b73bac.jpg)

### Tables

![119d785339941efc3139acf412245c8c9c56ff39b371c737510d2a844364aa58.jpg](../icml_results/968_PRISM_%20Training-Free%20Video%20Anomaly%20Detection%20via%20Intrinsic%20Statistical%20Modeling/tables/119d785339941efc3139acf412245c8c9c56ff39b371c737510d2a844364aa58.jpg)

![276f247ebb4a7cad1ad23d986520b1f0c698ad1c672a197b646de2e7f08a1581.jpg](../icml_results/968_PRISM_%20Training-Free%20Video%20Anomaly%20Detection%20via%20Intrinsic%20Statistical%20Modeling/tables/276f247ebb4a7cad1ad23d986520b1f0c698ad1c672a197b646de2e7f08a1581.jpg)

![380664228083cf4f09a6fb8878fd2656d352c6e2cb0589a4d1ad95f28ae40dae.jpg](../icml_results/968_PRISM_%20Training-Free%20Video%20Anomaly%20Detection%20via%20Intrinsic%20Statistical%20Modeling/tables/380664228083cf4f09a6fb8878fd2656d352c6e2cb0589a4d1ad95f28ae40dae.jpg)

![5f0263827320e26371d639d8a2241b65ad05c4ebc530c69a1571830c9b51752f.jpg](../icml_results/968_PRISM_%20Training-Free%20Video%20Anomaly%20Detection%20via%20Intrinsic%20Statistical%20Modeling/tables/5f0263827320e26371d639d8a2241b65ad05c4ebc530c69a1571830c9b51752f.jpg)

![9ca27de761eef5d9c145c8d5c8bceb1af8ac42522c6d6050be53ef2610701b6a.jpg](../icml_results/968_PRISM_%20Training-Free%20Video%20Anomaly%20Detection%20via%20Intrinsic%20Statistical%20Modeling/tables/9ca27de761eef5d9c145c8d5c8bceb1af8ac42522c6d6050be53ef2610701b6a.jpg)

![9fb1640157c8ade83cdcc80fc82e757a8a2cd2e21b769bb91a9d9f5ff1765447.jpg](../icml_results/968_PRISM_%20Training-Free%20Video%20Anomaly%20Detection%20via%20Intrinsic%20Statistical%20Modeling/tables/9fb1640157c8ade83cdcc80fc82e757a8a2cd2e21b769bb91a9d9f5ff1765447.jpg)

![9feb7af558bdc64675af204581b4245f4481a3cfbdb4b530601e801378585f65.jpg](../icml_results/968_PRISM_%20Training-Free%20Video%20Anomaly%20Detection%20via%20Intrinsic%20Statistical%20Modeling/tables/9feb7af558bdc64675af204581b4245f4481a3cfbdb4b530601e801378585f65.jpg)

<a id="paper-0952"></a>

## Adaptive Probe-based Steering for Robust LLM Jailbreaking


### Images

![1163869001f98d98d1f63794e58d3f221d60ba8c376ece1193f5a8164c582d00.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/images/1163869001f98d98d1f63794e58d3f221d60ba8c376ece1193f5a8164c582d00.jpg)

![2548c8e0f482149bae4e3f210fc09e77134da1ef0723333fb3956e7c94d96853.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/images/2548c8e0f482149bae4e3f210fc09e77134da1ef0723333fb3956e7c94d96853.jpg)

![5455e9498e7771d27541141636b6baa11d51f3857fb32079d9545c7850bea964.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/images/5455e9498e7771d27541141636b6baa11d51f3857fb32079d9545c7850bea964.jpg)

![54a15e53a393f3fa0a9798e152d5dd9bef24d435e708b25b865c0fe387ebf096.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/images/54a15e53a393f3fa0a9798e152d5dd9bef24d435e708b25b865c0fe387ebf096.jpg)

![577888324690cd421283f50d8915d9697e3dd828a2b26c82a4497ada8ded166a.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/images/577888324690cd421283f50d8915d9697e3dd828a2b26c82a4497ada8ded166a.jpg)

![6d41323fc5b96942d01c6e7b23edf2999ab8541d57962479a2107afc29b5bad3.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/images/6d41323fc5b96942d01c6e7b23edf2999ab8541d57962479a2107afc29b5bad3.jpg)

![7864b524b67e6e96aa40d9c7a8fbcf465ae2caaa79757b0767ef38c393ac8717.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/images/7864b524b67e6e96aa40d9c7a8fbcf465ae2caaa79757b0767ef38c393ac8717.jpg)

![9c7d02de31b41c993c4832d71be8096a5d9d385bafab6e0c369db6f811fddc44.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/images/9c7d02de31b41c993c4832d71be8096a5d9d385bafab6e0c369db6f811fddc44.jpg)

![ca6a80d58c42b8cc6e6f2ba132105446206510cbcbdb27357c749beb5da80bd5.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/images/ca6a80d58c42b8cc6e6f2ba132105446206510cbcbdb27357c749beb5da80bd5.jpg)

![d8905fdcd4deb79dc88cb51c588be082aad305d8c985d9ed112f3b8427c371d9.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/images/d8905fdcd4deb79dc88cb51c588be082aad305d8c985d9ed112f3b8427c371d9.jpg)

![e13a3464fa122612c4e82a4c1d1a60c967b17307f19b9c6306cfb7fb1c19eff6.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/images/e13a3464fa122612c4e82a4c1d1a60c967b17307f19b9c6306cfb7fb1c19eff6.jpg)

![f791b2d6ca20bb15f9973dcd8fc3ce727befc9d3fd9d714383b647d47e8c257f.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/images/f791b2d6ca20bb15f9973dcd8fc3ce727befc9d3fd9d714383b647d47e8c257f.jpg)

### Tables

![05cc1b31a65205821aa7bb6fa41c16fa4b927c3dee176b0f14742189bb3995c9.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/tables/05cc1b31a65205821aa7bb6fa41c16fa4b927c3dee176b0f14742189bb3995c9.jpg)

![0be5327d8a6292aa5df238c3e7c15d07579be1a440c6ce0f26991df453fea283.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/tables/0be5327d8a6292aa5df238c3e7c15d07579be1a440c6ce0f26991df453fea283.jpg)

![44b2800e1a9c9289113cb084d53877c06ee1202a53d004578e0f464bfd110352.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/tables/44b2800e1a9c9289113cb084d53877c06ee1202a53d004578e0f464bfd110352.jpg)

![4d3c161ac933621ed6b5d0660a11ddef40b57856c4df6ea2d5f2719838cd09ed.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/tables/4d3c161ac933621ed6b5d0660a11ddef40b57856c4df6ea2d5f2719838cd09ed.jpg)

![61bee68c7aabd9ca3c67f8f28090dd99820b91ddbb04c2a304b4e94e7546b684.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/tables/61bee68c7aabd9ca3c67f8f28090dd99820b91ddbb04c2a304b4e94e7546b684.jpg)

![a622cb6f738b68fa4c4fa5d072513985d0d447d3c6f2a9e85951f6ba98d941bd.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/tables/a622cb6f738b68fa4c4fa5d072513985d0d447d3c6f2a9e85951f6ba98d941bd.jpg)

![a94c151c9fa8a9a4b7598dca73788d851af7c9539dcee956df0b43204a95899c.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/tables/a94c151c9fa8a9a4b7598dca73788d851af7c9539dcee956df0b43204a95899c.jpg)

![c75ba02423e5e8bf5a65947b47f721c883fd6f89d012e39747ce193ff1b6617f.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/tables/c75ba02423e5e8bf5a65947b47f721c883fd6f89d012e39747ce193ff1b6617f.jpg)

![ed611645a922669809cd9b7da0725b181ac94c935b44e927b8db9c4f8966f447.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/tables/ed611645a922669809cd9b7da0725b181ac94c935b44e927b8db9c4f8966f447.jpg)

![f13383d656207a3adf771b7e710857bdbffeac3ebf1efede09af64ba53920c2b.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/tables/f13383d656207a3adf771b7e710857bdbffeac3ebf1efede09af64ba53920c2b.jpg)

![f906d43b0d4b492b293028f937db616ae635b3e9a6a2dae22e2c20d9fba52402.jpg](../icml_results/969_Adaptive%20Probe-based%20Steering%20for%20Robust%20LLM%20Jailbreaking/tables/f906d43b0d4b492b293028f937db616ae635b3e9a6a2dae22e2c20d9fba52402.jpg)

<a id="paper-0953"></a>

## SGERA: Stein-Guided ECG-Report Alignment for ECG Representation Learning


### Images

![0e8e25b9d7c23d820b9b414ef7888684ea9c30d4f80eebf3db53cb473df0eaf7.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/images/0e8e25b9d7c23d820b9b414ef7888684ea9c30d4f80eebf3db53cb473df0eaf7.jpg)

![2326a9d0e08eb1a567a6c10275c3d150075ac412dc520dd08760c0af8fd32e0d.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/images/2326a9d0e08eb1a567a6c10275c3d150075ac412dc520dd08760c0af8fd32e0d.jpg)

![2f99d3e91f2864466d1a087c35d245129219dba8ef913d6657e957a8ae4ab645.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/images/2f99d3e91f2864466d1a087c35d245129219dba8ef913d6657e957a8ae4ab645.jpg)

![5136dad7ec4f8c8c94f9ec67479acd13149dd47c3f550e00d9b407877fd1305e.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/images/5136dad7ec4f8c8c94f9ec67479acd13149dd47c3f550e00d9b407877fd1305e.jpg)

![83fb2995c2acdc718abb15cada5a8128d413d6246182267c8f172b122f44e740.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/images/83fb2995c2acdc718abb15cada5a8128d413d6246182267c8f172b122f44e740.jpg)

![8d672bafb76f6ee73f8d1286273e01070c6b666bf7cdc55577af8883ba8feea2.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/images/8d672bafb76f6ee73f8d1286273e01070c6b666bf7cdc55577af8883ba8feea2.jpg)

![c8de0fd8cda507f2749f19fda4039dff5e2f665830f1313b75a4ba88be45a0ec.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/images/c8de0fd8cda507f2749f19fda4039dff5e2f665830f1313b75a4ba88be45a0ec.jpg)

![ef714c64e23da2c54c388f524f41d5cd4087788f3f46c38ad875e9a61092b0df.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/images/ef714c64e23da2c54c388f524f41d5cd4087788f3f46c38ad875e9a61092b0df.jpg)

### Tables

![03160a992bbc3cbf610a88b07a53db9d5a079b3154367edd832c32aaebde913e.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/tables/03160a992bbc3cbf610a88b07a53db9d5a079b3154367edd832c32aaebde913e.jpg)

![534c1764d3f647216979e90f3be8a608e5ea0e3c675e073a5f01886ff8f541db.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/tables/534c1764d3f647216979e90f3be8a608e5ea0e3c675e073a5f01886ff8f541db.jpg)

![6ce5ba4bcaee3ae23c751bd94bd2ce64d9b90e4a8cf3da78ccaf7cc568663df9.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/tables/6ce5ba4bcaee3ae23c751bd94bd2ce64d9b90e4a8cf3da78ccaf7cc568663df9.jpg)

![7461bbacabd4d11810a38778005a2962d52eedf136510e3bacf8f20790785113.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/tables/7461bbacabd4d11810a38778005a2962d52eedf136510e3bacf8f20790785113.jpg)

![857e893d0facdc16e19fb346f4ebc657495a0c4af3254afabde141be6738f64c.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/tables/857e893d0facdc16e19fb346f4ebc657495a0c4af3254afabde141be6738f64c.jpg)

![8efb919392727b6df9bc171c02d13ea6d909a6bf7424620917ef943634025f5e.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/tables/8efb919392727b6df9bc171c02d13ea6d909a6bf7424620917ef943634025f5e.jpg)

![a35351b64c3b4e89a180e50dc7d425898a6f232eeb827d30e0eb6232e9fc2476.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/tables/a35351b64c3b4e89a180e50dc7d425898a6f232eeb827d30e0eb6232e9fc2476.jpg)

![c7463b83aaaa67cdf0fe7bb022e8a35f25a6aa2933813340446d3fdb16dd8878.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/tables/c7463b83aaaa67cdf0fe7bb022e8a35f25a6aa2933813340446d3fdb16dd8878.jpg)

![e388d443fdc76b712efde5bba424238b464a7f02d1f36b121ad2c08e27fafc28.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/tables/e388d443fdc76b712efde5bba424238b464a7f02d1f36b121ad2c08e27fafc28.jpg)

![f95c4bfeb4642114dd6525d103052a626e257d83eccf64ef74abb0b1a31836be.jpg](../icml_results/970_SGERA_%20Stein-Guided%20ECG-Report%20Alignment%20for%20ECG%20Representation%20Learning/tables/f95c4bfeb4642114dd6525d103052a626e257d83eccf64ef74abb0b1a31836be.jpg)

<a id="paper-0954"></a>

## WUSH: Near-Optimal Adaptive Transforms for LLM Quantization


### Images

![0af7609c9753372c4381bbab2fc9d6e8666fba624ec4ed2faff1de38586fa186.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/images/0af7609c9753372c4381bbab2fc9d6e8666fba624ec4ed2faff1de38586fa186.jpg)

![100a2f1a5e515d989332cb4751b38f6b1ba9a8b31b8c10c03028499e15e396cb.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/images/100a2f1a5e515d989332cb4751b38f6b1ba9a8b31b8c10c03028499e15e396cb.jpg)

![13864cc2a6787b61874544fe873781b3ff5e5babbe548c7918be25ba60c68651.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/images/13864cc2a6787b61874544fe873781b3ff5e5babbe548c7918be25ba60c68651.jpg)

![33eb599508fabfcb4ba56a134ee07427f8b332a8f96c43ae0f2630802b35da3c.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/images/33eb599508fabfcb4ba56a134ee07427f8b332a8f96c43ae0f2630802b35da3c.jpg)

![3554c0a945a981411b0b2236419f692b5747aae2e3f105d690d4c2cd1a71fb16.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/images/3554c0a945a981411b0b2236419f692b5747aae2e3f105d690d4c2cd1a71fb16.jpg)

![361d8abe94a3078bc393ffd61ab26ae5eb5359c00799432650c0f4c8c1473a0d.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/images/361d8abe94a3078bc393ffd61ab26ae5eb5359c00799432650c0f4c8c1473a0d.jpg)

![6d0bd587d582b66dee3a464c6e3664a40e40a57d01ad0eabce6541cfebee46ce.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/images/6d0bd587d582b66dee3a464c6e3664a40e40a57d01ad0eabce6541cfebee46ce.jpg)

![745694425a4c57b332d847901e2cf2171f6fdc6ceac5c58e263bd3164b7032fd.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/images/745694425a4c57b332d847901e2cf2171f6fdc6ceac5c58e263bd3164b7032fd.jpg)

![9a4ce3d8e4141312bc67ade8b7ae74c8a976f4154089ea59bcdc3221e8d6ce13.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/images/9a4ce3d8e4141312bc67ade8b7ae74c8a976f4154089ea59bcdc3221e8d6ce13.jpg)

![c77691200ccea8e96ee4f85dd0e6e435c499487ae081c434dabb68acca346598.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/images/c77691200ccea8e96ee4f85dd0e6e435c499487ae081c434dabb68acca346598.jpg)

![db6d40f393c2faf0af839d51e8c13008c31de9e5f4138ded8876dd059a42aac1.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/images/db6d40f393c2faf0af839d51e8c13008c31de9e5f4138ded8876dd059a42aac1.jpg)

### Tables

![130270cb7496e79c4b508fe2395bb512501320a27d5e70646af87fd0486ed131.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/tables/130270cb7496e79c4b508fe2395bb512501320a27d5e70646af87fd0486ed131.jpg)

![1c608517b6c85364d6f038f067fe60985c554b103e47d3ef341f70f9536fc25b.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/tables/1c608517b6c85364d6f038f067fe60985c554b103e47d3ef341f70f9536fc25b.jpg)

![2cccaab775a520a03839f280afeb2699a1a60db91ec6a244860e08867fc27553.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/tables/2cccaab775a520a03839f280afeb2699a1a60db91ec6a244860e08867fc27553.jpg)

![6c1adfba83edcc4e7afad5ffb4028dd6a1c488ebe8508195814623ec773a3ed7.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/tables/6c1adfba83edcc4e7afad5ffb4028dd6a1c488ebe8508195814623ec773a3ed7.jpg)

![6d94d56fea8daf63c480ef47b77ae8f40828997b6389026a3e07bebb7177e9a3.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/tables/6d94d56fea8daf63c480ef47b77ae8f40828997b6389026a3e07bebb7177e9a3.jpg)

![948a55b2b7b28fd694ef5f2318eb816586ee92ec733d5ddfe4fde50939ecb966.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/tables/948a55b2b7b28fd694ef5f2318eb816586ee92ec733d5ddfe4fde50939ecb966.jpg)

![de4cd90f1c6677ed29f694117ce86988f8b816775a35bf808ae8ba70b4cf282b.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/tables/de4cd90f1c6677ed29f694117ce86988f8b816775a35bf808ae8ba70b4cf282b.jpg)

![e894ccca46407709a58fb8826df8c64d33897b3a1a734d4388abc3f10524a76c.jpg](../icml_results/971_WUSH_%20Near-Optimal%20Adaptive%20Transforms%20for%20LLM%20Quantization/tables/e894ccca46407709a58fb8826df8c64d33897b3a1a734d4388abc3f10524a76c.jpg)

<a id="paper-0955"></a>

## StormInsight: Hierarchical Environmental Forcing and Vertical Coupling for Convective Systems Evolution


### Images

![07e00fb15aaacd67c079bf4370dbcadf92c82c5503a292f6bc9b5ec05e0ca9a5.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/images/07e00fb15aaacd67c079bf4370dbcadf92c82c5503a292f6bc9b5ec05e0ca9a5.jpg)

![0a4565dded9c0bfa4be874ce34c4886ee0688691bad24814b3f6736ed87fc258.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/images/0a4565dded9c0bfa4be874ce34c4886ee0688691bad24814b3f6736ed87fc258.jpg)

![285ff61021b2cc1e63793ba6023f4267c84aa0316161930f99aff09b213c8102.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/images/285ff61021b2cc1e63793ba6023f4267c84aa0316161930f99aff09b213c8102.jpg)

![28aa0b03809e8164237415cbf8a890cd61875383578f344ae9cbb83cb089690a.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/images/28aa0b03809e8164237415cbf8a890cd61875383578f344ae9cbb83cb089690a.jpg)

![3700e5d926e6744efbf89644ecc4ac25b9b0590fdd016974423132891fd20de7.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/images/3700e5d926e6744efbf89644ecc4ac25b9b0590fdd016974423132891fd20de7.jpg)

![67289a7d9fafd3c2eacc5088937bd5a92f1df7d063ea7ad808ec43ff88b54712.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/images/67289a7d9fafd3c2eacc5088937bd5a92f1df7d063ea7ad808ec43ff88b54712.jpg)

![7089f7058b9f2c89f1c18b8d6250484025d0db17c105377366cc2bd5856754bc.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/images/7089f7058b9f2c89f1c18b8d6250484025d0db17c105377366cc2bd5856754bc.jpg)

![84d4aacf57b11c2fef28e45c51ac2409e9c13e76367e7fd75bcca640a1f82901.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/images/84d4aacf57b11c2fef28e45c51ac2409e9c13e76367e7fd75bcca640a1f82901.jpg)

![9f3a5058695890fcc59eee4761e045b1af3476ee7180d765b5595993a3866210.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/images/9f3a5058695890fcc59eee4761e045b1af3476ee7180d765b5595993a3866210.jpg)

![a48c1c8a1b57840a042108d384b171da0cc55176c006636a18c61a90a920bad3.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/images/a48c1c8a1b57840a042108d384b171da0cc55176c006636a18c61a90a920bad3.jpg)

![b277fb80fc2e0946369d587c7f20149fdfa8705b60eb0e336986e419bcc5dd3b.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/images/b277fb80fc2e0946369d587c7f20149fdfa8705b60eb0e336986e419bcc5dd3b.jpg)

![d6b6d7ca2715463caf8f128370bf84f65ebef59c354e39314e5184e6ddb7ef2a.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/images/d6b6d7ca2715463caf8f128370bf84f65ebef59c354e39314e5184e6ddb7ef2a.jpg)

![f3365bf4a841a071c836336821485d4f0efbe8dc1d56cdd79feb19f4c31c7b5a.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/images/f3365bf4a841a071c836336821485d4f0efbe8dc1d56cdd79feb19f4c31c7b5a.jpg)

### Tables

![06733cf599cb588c96b8c6726f299288f6aecca239dc08ecfba15312a8cdd8fe.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/tables/06733cf599cb588c96b8c6726f299288f6aecca239dc08ecfba15312a8cdd8fe.jpg)

![09ce97e9fd71217d33e56ae0dc1cc3f1d54d78a223077618243a12e3c13415b5.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/tables/09ce97e9fd71217d33e56ae0dc1cc3f1d54d78a223077618243a12e3c13415b5.jpg)

![149c75e7e7b2a3139962b5476ecf1b722db8333854e3f3b3b378211a9d7e3f2d.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/tables/149c75e7e7b2a3139962b5476ecf1b722db8333854e3f3b3b378211a9d7e3f2d.jpg)

![1a48d6ae904bc7ea369f6dac7bd5a5aef7bfce96fba4fcf1f162583db1c2cfb4.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/tables/1a48d6ae904bc7ea369f6dac7bd5a5aef7bfce96fba4fcf1f162583db1c2cfb4.jpg)

![226f13056f69027a2729976385919554abfe00e691846ed66b4ff1fc819861ff.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/tables/226f13056f69027a2729976385919554abfe00e691846ed66b4ff1fc819861ff.jpg)

![3a4bdf8c280ce59a552d0b53338e6b148e99c41bad21932d32a024323202b062.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/tables/3a4bdf8c280ce59a552d0b53338e6b148e99c41bad21932d32a024323202b062.jpg)

![3db9231a1366f58ca8804db84033b6e7744a3f830597b9be9ca08837522828aa.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/tables/3db9231a1366f58ca8804db84033b6e7744a3f830597b9be9ca08837522828aa.jpg)

![67126af87cd2ad8e91eaa1d5de1676e5809928fb0dfc733fad4e1f2760244cb4.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/tables/67126af87cd2ad8e91eaa1d5de1676e5809928fb0dfc733fad4e1f2760244cb4.jpg)

![6dbef26d0827ef366bbe3f9b1ae0396ac1745b0845da9f5560316a1aa7cb08c8.jpg](../icml_results/972_StormInsight_%20Hierarchical%20Environmental%20Forcing%20and%20Vertical%20Coupling%20for%20Convective%20Systems%20Evolut/tables/6dbef26d0827ef366bbe3f9b1ae0396ac1745b0845da9f5560316a1aa7cb08c8.jpg)

<a id="paper-0956"></a>

## PhyScene3D: Physically Consistent Interactive 3D Tabletop Scene Generation


### Images

![1c4cfd6413ee100326fe76127dde2f645d020d69542fa76046f25d4bfcf4f3d1.jpg](../icml_results/973_PhyScene3D_%20Physically%20Consistent%20Interactive%203D%20Tabletop%20Scene%20Generation/images/1c4cfd6413ee100326fe76127dde2f645d020d69542fa76046f25d4bfcf4f3d1.jpg)

![2d77e7ae2526d57bfced44eb2f770ced59274ffc5a09826cb3b87d9e46c8f31f.jpg](../icml_results/973_PhyScene3D_%20Physically%20Consistent%20Interactive%203D%20Tabletop%20Scene%20Generation/images/2d77e7ae2526d57bfced44eb2f770ced59274ffc5a09826cb3b87d9e46c8f31f.jpg)

![2dd3cdba653cdf5786ac42ccb0f4240e8b05b34a045e5ea8bd8b98ca16f88798.jpg](../icml_results/973_PhyScene3D_%20Physically%20Consistent%20Interactive%203D%20Tabletop%20Scene%20Generation/images/2dd3cdba653cdf5786ac42ccb0f4240e8b05b34a045e5ea8bd8b98ca16f88798.jpg)

![73f20ff6a352126c798280ad216ef8c2094a50215bedc91dbcf26b07bdb3d5a8.jpg](../icml_results/973_PhyScene3D_%20Physically%20Consistent%20Interactive%203D%20Tabletop%20Scene%20Generation/images/73f20ff6a352126c798280ad216ef8c2094a50215bedc91dbcf26b07bdb3d5a8.jpg)

![b414e650a2a7b94ddc17ea39d12b3998ce46a4b79ece316a8aab72f2bd1c7024.jpg](../icml_results/973_PhyScene3D_%20Physically%20Consistent%20Interactive%203D%20Tabletop%20Scene%20Generation/images/b414e650a2a7b94ddc17ea39d12b3998ce46a4b79ece316a8aab72f2bd1c7024.jpg)

### Tables

![1c9166652601cfe550226749996571ea5610d493fbf42a146b4c56aca71b564c.jpg](../icml_results/973_PhyScene3D_%20Physically%20Consistent%20Interactive%203D%20Tabletop%20Scene%20Generation/tables/1c9166652601cfe550226749996571ea5610d493fbf42a146b4c56aca71b564c.jpg)

![3c711a6817dddd1367b65a8773b392fb7f15d256de42b27f66a28274eb4a1ecd.jpg](../icml_results/973_PhyScene3D_%20Physically%20Consistent%20Interactive%203D%20Tabletop%20Scene%20Generation/tables/3c711a6817dddd1367b65a8773b392fb7f15d256de42b27f66a28274eb4a1ecd.jpg)

![455ffb5558098d35f20dad29c7e876c177033f062105b5dbfa85117b6d9fba17.jpg](../icml_results/973_PhyScene3D_%20Physically%20Consistent%20Interactive%203D%20Tabletop%20Scene%20Generation/tables/455ffb5558098d35f20dad29c7e876c177033f062105b5dbfa85117b6d9fba17.jpg)

![457676a59584b3a008f9fcd01b67d4ac699af9deb467ecb24bffaceb4bb76ede.jpg](../icml_results/973_PhyScene3D_%20Physically%20Consistent%20Interactive%203D%20Tabletop%20Scene%20Generation/tables/457676a59584b3a008f9fcd01b67d4ac699af9deb467ecb24bffaceb4bb76ede.jpg)

![4a5b19f11790208dcb7e241ad2eb6b3ac5f0afe3d2eb0f29208332e2b1573ef6.jpg](../icml_results/973_PhyScene3D_%20Physically%20Consistent%20Interactive%203D%20Tabletop%20Scene%20Generation/tables/4a5b19f11790208dcb7e241ad2eb6b3ac5f0afe3d2eb0f29208332e2b1573ef6.jpg)

![738f9c0e0d16429bf6614a5ef9c68887ca9d62c6eb7cf1ffe7cfb7e596d132ce.jpg](../icml_results/973_PhyScene3D_%20Physically%20Consistent%20Interactive%203D%20Tabletop%20Scene%20Generation/tables/738f9c0e0d16429bf6614a5ef9c68887ca9d62c6eb7cf1ffe7cfb7e596d132ce.jpg)

![bb8a82a9f2e3ff8b18082f5e0f5e44dd39438532f74c4393711f99b6efac2d2b.jpg](../icml_results/973_PhyScene3D_%20Physically%20Consistent%20Interactive%203D%20Tabletop%20Scene%20Generation/tables/bb8a82a9f2e3ff8b18082f5e0f5e44dd39438532f74c4393711f99b6efac2d2b.jpg)

![dcde6f0b426ed6ac246222fbd7f3d5985066de713a5db9c85de563588e84cb13.jpg](../icml_results/973_PhyScene3D_%20Physically%20Consistent%20Interactive%203D%20Tabletop%20Scene%20Generation/tables/dcde6f0b426ed6ac246222fbd7f3d5985066de713a5db9c85de563588e84cb13.jpg)

![e892cae071a031bde06d39710133d54dbfa001dd0d9183a7ab32ceacaeb0f3fa.jpg](../icml_results/973_PhyScene3D_%20Physically%20Consistent%20Interactive%203D%20Tabletop%20Scene%20Generation/tables/e892cae071a031bde06d39710133d54dbfa001dd0d9183a7ab32ceacaeb0f3fa.jpg)

![f2329225d9e18bca7f27d82dc0f164bbb9f26ac5ca022936c9534e875c379068.jpg](../icml_results/973_PhyScene3D_%20Physically%20Consistent%20Interactive%203D%20Tabletop%20Scene%20Generation/tables/f2329225d9e18bca7f27d82dc0f164bbb9f26ac5ca022936c9534e875c379068.jpg)

<a id="paper-0957"></a>

## Let the Prototype Guide You: Robust Aggregation of Sparse Multi-Class Annotations via Annotator Prototype Learning

### Images

![4fefd37a033075a905ec1e02303a20c59fd2db02d591395166dce06a5b33b064.jpg](../icml_results/974_Let%20the%20Prototype%20Guide%20You_%20Robust%20Aggregation%20of%20Sparse%20Multi-Class%20Annotations%20via%20Annotator%20Prot/images/4fefd37a033075a905ec1e02303a20c59fd2db02d591395166dce06a5b33b064.jpg)

![700c59555ad01aecc2644fa7d2bf4b2f1f7207e54cb321dc81b45f3481e1ec26.jpg](../icml_results/974_Let%20the%20Prototype%20Guide%20You_%20Robust%20Aggregation%20of%20Sparse%20Multi-Class%20Annotations%20via%20Annotator%20Prot/images/700c59555ad01aecc2644fa7d2bf4b2f1f7207e54cb321dc81b45f3481e1ec26.jpg)

![7979dbec08c35e930f31a278268f2b5946e74b31298c26e477e6caafff76ba31.jpg](../icml_results/974_Let%20the%20Prototype%20Guide%20You_%20Robust%20Aggregation%20of%20Sparse%20Multi-Class%20Annotations%20via%20Annotator%20Prot/images/7979dbec08c35e930f31a278268f2b5946e74b31298c26e477e6caafff76ba31.jpg)

![b4ec2875380d5ece06d274613fb9ee0369f84c4404ece692365587f4af83511c.jpg](../icml_results/974_Let%20the%20Prototype%20Guide%20You_%20Robust%20Aggregation%20of%20Sparse%20Multi-Class%20Annotations%20via%20Annotator%20Prot/images/b4ec2875380d5ece06d274613fb9ee0369f84c4404ece692365587f4af83511c.jpg)

![edd46f4c4c674e93ddda8ea21b3441153a996f5ad47036e97463eb9eb97edfb9.jpg](../icml_results/974_Let%20the%20Prototype%20Guide%20You_%20Robust%20Aggregation%20of%20Sparse%20Multi-Class%20Annotations%20via%20Annotator%20Prot/images/edd46f4c4c674e93ddda8ea21b3441153a996f5ad47036e97463eb9eb97edfb9.jpg)

![ff9151097eba93778a32379a1b99a9ab300f29981e7dfd547f4153df64b2ab3c.jpg](../icml_results/974_Let%20the%20Prototype%20Guide%20You_%20Robust%20Aggregation%20of%20Sparse%20Multi-Class%20Annotations%20via%20Annotator%20Prot/images/ff9151097eba93778a32379a1b99a9ab300f29981e7dfd547f4153df64b2ab3c.jpg)

### Tables

![38f3b578c6cebd1b03363a8eb17cb62822ef1b1ffc52fdb29b23e6a6657fd981.jpg](../icml_results/974_Let%20the%20Prototype%20Guide%20You_%20Robust%20Aggregation%20of%20Sparse%20Multi-Class%20Annotations%20via%20Annotator%20Prot/tables/38f3b578c6cebd1b03363a8eb17cb62822ef1b1ffc52fdb29b23e6a6657fd981.jpg)

![3b3c13a7908074c6e0ed151844556b9c13d01a94d45cde19ad5c89558fa7a823.jpg](../icml_results/974_Let%20the%20Prototype%20Guide%20You_%20Robust%20Aggregation%20of%20Sparse%20Multi-Class%20Annotations%20via%20Annotator%20Prot/tables/3b3c13a7908074c6e0ed151844556b9c13d01a94d45cde19ad5c89558fa7a823.jpg)

![6113336925ed39360f311811aa492873dd43de61c2aaa41fd6ac8e614ed96868.jpg](../icml_results/974_Let%20the%20Prototype%20Guide%20You_%20Robust%20Aggregation%20of%20Sparse%20Multi-Class%20Annotations%20via%20Annotator%20Prot/tables/6113336925ed39360f311811aa492873dd43de61c2aaa41fd6ac8e614ed96868.jpg)

![7159bbb307a56031d3bc064422f12aff333593eec14ead081581018ec22b2e74.jpg](../icml_results/974_Let%20the%20Prototype%20Guide%20You_%20Robust%20Aggregation%20of%20Sparse%20Multi-Class%20Annotations%20via%20Annotator%20Prot/tables/7159bbb307a56031d3bc064422f12aff333593eec14ead081581018ec22b2e74.jpg)

![8900c328e552550ae51c325fba545104b7628e72021597f238cc39b9c06f317e.jpg](../icml_results/974_Let%20the%20Prototype%20Guide%20You_%20Robust%20Aggregation%20of%20Sparse%20Multi-Class%20Annotations%20via%20Annotator%20Prot/tables/8900c328e552550ae51c325fba545104b7628e72021597f238cc39b9c06f317e.jpg)

![9c3b03d40923e62e34d69b5e2c96d6b21c9cbbfa44ee90f031c85808e69b269c.jpg](../icml_results/974_Let%20the%20Prototype%20Guide%20You_%20Robust%20Aggregation%20of%20Sparse%20Multi-Class%20Annotations%20via%20Annotator%20Prot/tables/9c3b03d40923e62e34d69b5e2c96d6b21c9cbbfa44ee90f031c85808e69b269c.jpg)

![a5ee4de9e8bb97444b0bc826a96834dee6b974e227e9c2d0936fbce3c5de31b7.jpg](../icml_results/974_Let%20the%20Prototype%20Guide%20You_%20Robust%20Aggregation%20of%20Sparse%20Multi-Class%20Annotations%20via%20Annotator%20Prot/tables/a5ee4de9e8bb97444b0bc826a96834dee6b974e227e9c2d0936fbce3c5de31b7.jpg)

![d01c60b9898d45507be8e5833157143556322d5d150bf147e3928b0d20fd1751.jpg](../icml_results/974_Let%20the%20Prototype%20Guide%20You_%20Robust%20Aggregation%20of%20Sparse%20Multi-Class%20Annotations%20via%20Annotator%20Prot/tables/d01c60b9898d45507be8e5833157143556322d5d150bf147e3928b0d20fd1751.jpg)

![f44bb47e89d89304aacb37fedc453d645264ee8424d1faad96f4967d090c096d.jpg](../icml_results/974_Let%20the%20Prototype%20Guide%20You_%20Robust%20Aggregation%20of%20Sparse%20Multi-Class%20Annotations%20via%20Annotator%20Prot/tables/f44bb47e89d89304aacb37fedc453d645264ee8424d1faad96f4967d090c096d.jpg)
