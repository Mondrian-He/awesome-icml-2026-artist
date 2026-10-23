# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 73 of 197

## 目录 (Table of Contents)

1. [Expanding the AI Evaluation Toolbox with Statistical Models](#paper-2377)
2. [Backward Oversmoothing: why is it hard to train deep Graph Neural Networks?](#paper-2378)
3. [Sinkhorn Normalization of Diffusion Kernels](#paper-2379)
4. [Robust AI Evaluation through Maximal Lotteries](#paper-2380)
5. [Unlearning in Diffusion Models: A Unified Framework with KL Divergence and Likelihood Constraints](#paper-2381)
6. [Reward Learning through Ranking Mean Squared Error](#paper-2382)
7. [A Deep Learning Model of Mental Rotation Informed by Interactive VR Experiments](#paper-2383)
8. [Quantifying Error Propagation and Model Collapse in Diffusion Models](#paper-2384)
9. [Federated Causal Inference on Multi-Site Observational Data via Propensity Score Aggregation](#paper-2385)
10. [Minimum Distance Summaries for Robust Neural Posterior Estimation](#paper-2386)
11. [Spatial Deconfounder: Interference-Aware Deconfounding for Spatial Causal Inference](#paper-2387)
12. [Align Forward, Adapt Backward: Closing the Discretization Gap in Logic Gate Networks](#paper-2388)
13. [Continuous Diffusion Models Can Obey Formal Syntax](#paper-2389)
14. [Explicit representation of germline and non-germline residues improves antibody language modeling](#paper-2390)
15. [Rule2DRC: Benchmarking LLM Agents for DRC Script Synthesis with Execution-Guided Test Generation](#paper-2391)
16. [Align Your Trajectory Tangent: Training Better Consistency Models via Manifold-Aligned Tangents](#paper-2392)
17. [Universal Reasoner: A Single, Composable Plug-and-Play Reasoner for Frozen LLMs](#paper-2393)
18. [On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](#paper-2394)
19. [Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models](#paper-2395)
20. [LeakGFN: Robust Molecular Generation in Generative Flow Networks via Flow Decomposition](#paper-2396)
21. [Dissect and Prune: Enhancing Robustness in AI-Generated Image Detection](#paper-2397)
22. [Robust and Consistent Ski Rental with Distributional Advice](#paper-2398)
23. [Stochastic Gradient Variational Inference with Price’s Gradient Estimator from Bures-Wasserstein to Parameter Space](#paper-2399)
24. [Stop Training for the Worst: Progressive Unmasking Accelerates Masked Diffusion Training](#paper-2400)
25. [Equivariant Latent Alignment via Flow Matching under Group Symmetries](#paper-2401)
26. [Semi-Supervised Neural Super-Resolution for Mesh-Based Simulations](#paper-2402)
27. [Verifying Meta-Awareness via Predictive Rewards in Reasoning Models](#paper-2403)
28. [DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs](#paper-2404)
29. [Inverting Data Transformations via Diffusion Sampling](#paper-2405)
30. [AlienLM: Alienization of Language for API-Boundary Privacy in Black-Box LLMs](#paper-2406)
31. [Unlocking Noise-Resistant Vision: Key Architectural Secrets for Robust Models Against Gaussian Noise](#paper-2407)
32. [Compositional Transduction with Latent Analogies for Offline Goal-Conditioned Reinforcement Learning](#paper-2408)
33. [Block-wise Codeword Embedding for Reliable Multi-bit Text Watermarking](#paper-2409)

---


<a id="paper-2377"></a>

## Expanding the AI Evaluation Toolbox with Statistical Models

### Images

![09d83aa1795c596c0c7e48aa9514497f2922d2040dd6c33af30311f99ebf9790.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/images/09d83aa1795c596c0c7e48aa9514497f2922d2040dd6c33af30311f99ebf9790.jpg)

![0fffb14dab88ee9a296c0c52c442f8770d7559877304fdcdb605662463164b92.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/images/0fffb14dab88ee9a296c0c52c442f8770d7559877304fdcdb605662463164b92.jpg)

![6576c250f89531d3284fca44de83ce515d286b25b494ad9bd074de38e6881191.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/images/6576c250f89531d3284fca44de83ce515d286b25b494ad9bd074de38e6881191.jpg)

![686a55776b19a74afefb45f571e91ac4e5ae1f8d8347ad9aec86a9b71303d420.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/images/686a55776b19a74afefb45f571e91ac4e5ae1f8d8347ad9aec86a9b71303d420.jpg)

![78b6ffcd42f383a36c19aea0fbd8a6c94d2791cc4fa094084be7f4080c137218.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/images/78b6ffcd42f383a36c19aea0fbd8a6c94d2791cc4fa094084be7f4080c137218.jpg)

![85419660cdf70ef6ca8e8f280f656f90f7dece693414c4678c9e80befa5fb5d0.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/images/85419660cdf70ef6ca8e8f280f656f90f7dece693414c4678c9e80befa5fb5d0.jpg)

![a41b125965e979cb60a56b046a430afd57b9e52944b0400d5a45654c20961426.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/images/a41b125965e979cb60a56b046a430afd57b9e52944b0400d5a45654c20961426.jpg)

![d831b020532cfba2ecf3c38a11af214eb81b014a4412cc5c0dc83d2ca3c8b2b0.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/images/d831b020532cfba2ecf3c38a11af214eb81b014a4412cc5c0dc83d2ca3c8b2b0.jpg)

### Tables

![26125d82b7e7e4c74b01dae1754da4ba16993211898ffbe67025b8c5ccd8eaa5.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/tables/26125d82b7e7e4c74b01dae1754da4ba16993211898ffbe67025b8c5ccd8eaa5.jpg)

![36872e30459b5cc47f5eeb2c9b74eabb9d13c5939ae2339dfacc0bf0f4d10e41.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/tables/36872e30459b5cc47f5eeb2c9b74eabb9d13c5939ae2339dfacc0bf0f4d10e41.jpg)

![4bb0f6e9b13c1bd88bb4f37e362ed6d0e3e888e0ab0a58022bdee8b2511844fb.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/tables/4bb0f6e9b13c1bd88bb4f37e362ed6d0e3e888e0ab0a58022bdee8b2511844fb.jpg)

![5a49b74ae547bcc3ab4b08afc65f184fa509a07ffb431131cbd24376d59112a6.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/tables/5a49b74ae547bcc3ab4b08afc65f184fa509a07ffb431131cbd24376d59112a6.jpg)

![5c24541169f8dd4d5a320c31328817eaece7e9007614fddca2875fb585417b5d.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/tables/5c24541169f8dd4d5a320c31328817eaece7e9007614fddca2875fb585417b5d.jpg)

![98b0d232d4a790b4c0561b1020cf09caf3decaa01bcb1cec099578ce9b4d1fd7.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/tables/98b0d232d4a790b4c0561b1020cf09caf3decaa01bcb1cec099578ce9b4d1fd7.jpg)

![9f6bcdfdd5d513ebcd3090d04fdff93ac2b54a388a206b2438fe80dab4c77fbe.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/tables/9f6bcdfdd5d513ebcd3090d04fdff93ac2b54a388a206b2438fe80dab4c77fbe.jpg)

![f51fdfa50e0863a1b5525447912cd6e6bebda715b1d9fcafe525ef8e2ad81e8f.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/tables/f51fdfa50e0863a1b5525447912cd6e6bebda715b1d9fcafe525ef8e2ad81e8f.jpg)

![f7f80b9fbcdc6d15e0dce6923eb5cec6963fae512c456f3fc0b2e63a7e160f04.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/tables/f7f80b9fbcdc6d15e0dce6923eb5cec6963fae512c456f3fc0b2e63a7e160f04.jpg)

![f8d52181b1f6932256844a78bfb1581884e8e54445a534c4e0b357df1a6b0dc2.jpg](../icml_results/2406_Mechanistic%20Anomaly%20Detection%20via%20Functional%20Attribution/tables/f8d52181b1f6932256844a78bfb1581884e8e54445a534c4e0b357df1a6b0dc2.jpg)

## Expanding the AI Evaluation Toolbox with Statistical Models


### Images

![0e6b000fa65a2983769c2fb1c37fdbbe72129e0a5bbf2cc95037160cfa9d6003.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/images/0e6b000fa65a2983769c2fb1c37fdbbe72129e0a5bbf2cc95037160cfa9d6003.jpg)

![1556899356c70bb97e378bcb761d8d555ccf0779a569638868175c2bc61fc32b.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/images/1556899356c70bb97e378bcb761d8d555ccf0779a569638868175c2bc61fc32b.jpg)

![2594f04838b23de1fd301e87660e7a045d3171406ab2266e7b3face1277e0b4a.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/images/2594f04838b23de1fd301e87660e7a045d3171406ab2266e7b3face1277e0b4a.jpg)

![25f2fd849787baf2efed5ec9c0337279ef0d0ee52d4264e613a0ebbf8a99ff15.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/images/25f2fd849787baf2efed5ec9c0337279ef0d0ee52d4264e613a0ebbf8a99ff15.jpg)

![4b3c62058d36643e8591eeb8c9f3683c8409c33aa70166890be96812d3d74275.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/images/4b3c62058d36643e8591eeb8c9f3683c8409c33aa70166890be96812d3d74275.jpg)

![5d4ca1c14e37d924e8a256678175eca0e54de56559fbafdfb14116ff41767d46.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/images/5d4ca1c14e37d924e8a256678175eca0e54de56559fbafdfb14116ff41767d46.jpg)

![800211134210eb2313fce679e94c968a822b8d98587d00d0e10781b9e47b5909.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/images/800211134210eb2313fce679e94c968a822b8d98587d00d0e10781b9e47b5909.jpg)

![94b0c961df94d9062ad3de781d25f55959e1e8cd3b1f1a97fb34a9ee79e21691.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/images/94b0c961df94d9062ad3de781d25f55959e1e8cd3b1f1a97fb34a9ee79e21691.jpg)

![b7445367b2e3587756871a2a2f0c94588b8ca5a70023df113726edb02bea0d34.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/images/b7445367b2e3587756871a2a2f0c94588b8ca5a70023df113726edb02bea0d34.jpg)

![e91afa6f82f45a3a74ba9ec27f10d81ba0dfb47438250d98c03e5e1dc98f1ad1.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/images/e91afa6f82f45a3a74ba9ec27f10d81ba0dfb47438250d98c03e5e1dc98f1ad1.jpg)

### Tables

![3201b349fcf2728ada0379fee8c279ca2c84e6b97a37d807e4a99c431d2d3347.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/tables/3201b349fcf2728ada0379fee8c279ca2c84e6b97a37d807e4a99c431d2d3347.jpg)

![6c78b4b8db04d71c21dd6fb6c985c01aed06dca8fd2b7c40235c83457cf4c9ac.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/tables/6c78b4b8db04d71c21dd6fb6c985c01aed06dca8fd2b7c40235c83457cf4c9ac.jpg)

![7094372ba1523a2b13812dfe19a40cc724df17c2b543f1a0d47ff47a6fa97331.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/tables/7094372ba1523a2b13812dfe19a40cc724df17c2b543f1a0d47ff47a6fa97331.jpg)

![7dc466bb521030b77776dd63bd598b2950c7c6a8cef1303a34806d348101ff55.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/tables/7dc466bb521030b77776dd63bd598b2950c7c6a8cef1303a34806d348101ff55.jpg)

![81932d31cf13201e517fd022e3645a2c3485a79188ad6396d98841d9a74cd9f8.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/tables/81932d31cf13201e517fd022e3645a2c3485a79188ad6396d98841d9a74cd9f8.jpg)

![82abb32a92bc418b3228fa9b454a67960494b2f0652774fd8acf683b26378995.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/tables/82abb32a92bc418b3228fa9b454a67960494b2f0652774fd8acf683b26378995.jpg)

![9b48500859386c374b4447968412640f7c8cd6d2825c55750b7e09b606533d3b.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/tables/9b48500859386c374b4447968412640f7c8cd6d2825c55750b7e09b606533d3b.jpg)

![9f2d9e3c93d59d6e518826b4986f6557477d7cfca3dcd3246b93f2537bfe37c8.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/tables/9f2d9e3c93d59d6e518826b4986f6557477d7cfca3dcd3246b93f2537bfe37c8.jpg)

![b1650c328b6e4798ce51d0c67f51d7291e70885036e98915e08de97bc9fe3e3a.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/tables/b1650c328b6e4798ce51d0c67f51d7291e70885036e98915e08de97bc9fe3e3a.jpg)

![b86d27cfd2c9aef203ef631d3711d52a943159b2dfc84cc04eb945eabcc5975d.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/tables/b86d27cfd2c9aef203ef631d3711d52a943159b2dfc84cc04eb945eabcc5975d.jpg)

![d6835d385002e829d14064f3720c11eb2a475a7f10eb9308c7aaf513b63f5d6d.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/tables/d6835d385002e829d14064f3720c11eb2a475a7f10eb9308c7aaf513b63f5d6d.jpg)

![fae7f74fa7b50498485847a8f939e16efd156a5a94937fca1d39e92c581299c5.jpg](../icml_results/2407_Expanding%20the%20AI%20Evaluation%20Toolbox%20with%20Statistical%20Models/tables/fae7f74fa7b50498485847a8f939e16efd156a5a94937fca1d39e92c581299c5.jpg)

<a id="paper-2378"></a>

## Backward Oversmoothing: why is it hard to train deep Graph Neural Networks?


### Images

![016bd5ccf63944fa1d3f27632d7cd36ea378006f4ecf27ca05978719ab8173d9.jpg](../icml_results/2408_Backward%20Oversmoothing_%20why%20is%20it%20hard%20to%20train%20deep%20Graph%20Neural%20Networks_/images/016bd5ccf63944fa1d3f27632d7cd36ea378006f4ecf27ca05978719ab8173d9.jpg)

![0e90e72e392026e120b8b600d041cf2fef1a1174813b621789429ee4412c2fbe.jpg](../icml_results/2408_Backward%20Oversmoothing_%20why%20is%20it%20hard%20to%20train%20deep%20Graph%20Neural%20Networks_/images/0e90e72e392026e120b8b600d041cf2fef1a1174813b621789429ee4412c2fbe.jpg)

![1a09459639fb08fb66f7942e6566d44616b51c32c52a71f206bfe1650a7d55c2.jpg](../icml_results/2408_Backward%20Oversmoothing_%20why%20is%20it%20hard%20to%20train%20deep%20Graph%20Neural%20Networks_/images/1a09459639fb08fb66f7942e6566d44616b51c32c52a71f206bfe1650a7d55c2.jpg)

![286a7f3f68a1c5fcbec7898fecf07e88e3cd5582d9dcbc4beb8c3c4186ab6009.jpg](../icml_results/2408_Backward%20Oversmoothing_%20why%20is%20it%20hard%20to%20train%20deep%20Graph%20Neural%20Networks_/images/286a7f3f68a1c5fcbec7898fecf07e88e3cd5582d9dcbc4beb8c3c4186ab6009.jpg)

![e4ee2db4fb638df0d0a76448964ff01685ff79768adcf99128db9d943701c4e7.jpg](../icml_results/2408_Backward%20Oversmoothing_%20why%20is%20it%20hard%20to%20train%20deep%20Graph%20Neural%20Networks_/images/e4ee2db4fb638df0d0a76448964ff01685ff79768adcf99128db9d943701c4e7.jpg)

### Tables

![15eb7f2719fa23fd09c692d5ba8926bf42e411e714c8723081c7434f5c9f1389.jpg](../icml_results/2408_Backward%20Oversmoothing_%20why%20is%20it%20hard%20to%20train%20deep%20Graph%20Neural%20Networks_/tables/15eb7f2719fa23fd09c692d5ba8926bf42e411e714c8723081c7434f5c9f1389.jpg)

<a id="paper-2379"></a>

## Sinkhorn Normalization of Diffusion Kernels


### Images

![00dee96108efea341cbab74209342dc33969fd5880dcc41b593c7a4f64ba6365.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/00dee96108efea341cbab74209342dc33969fd5880dcc41b593c7a4f64ba6365.jpg)

![19409d194368fafb635d28d4063ad420e1d6d651e71ba2f0d6a287e719b3f4a3.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/19409d194368fafb635d28d4063ad420e1d6d651e71ba2f0d6a287e719b3f4a3.jpg)

![20f37cd4d262561360ee1674b840495d25b23099e459023aa45bcc7b41c4eaa8.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/20f37cd4d262561360ee1674b840495d25b23099e459023aa45bcc7b41c4eaa8.jpg)

![3053105724508c9d3128e32f7fda441d0ca045cf2d53644081ac3ae06a5a5c59.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/3053105724508c9d3128e32f7fda441d0ca045cf2d53644081ac3ae06a5a5c59.jpg)

![39650b6c139cfb556fd565e25b4667fe64c3cff3bdf52cd69fcf5d6847dd806f.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/39650b6c139cfb556fd565e25b4667fe64c3cff3bdf52cd69fcf5d6847dd806f.jpg)

![3a3d0f7b7212b9f1f05515c5087462a9637dfc70de1ac6cb47c4a325e12e8a70.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/3a3d0f7b7212b9f1f05515c5087462a9637dfc70de1ac6cb47c4a325e12e8a70.jpg)

![3af043dcd5f3443c7caaf072c4c69870abc605fec4e12c1983235e4aec76b15f.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/3af043dcd5f3443c7caaf072c4c69870abc605fec4e12c1983235e4aec76b15f.jpg)

![3bb85be185ff3a44e7fd8a1e8dc9f857abecf0ef9c5acae432e1bfb74e03ea5a.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/3bb85be185ff3a44e7fd8a1e8dc9f857abecf0ef9c5acae432e1bfb74e03ea5a.jpg)

![42019a464a4efde7e9891667163fd75748b7380e10e0ec2a8179eb075c416740.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/42019a464a4efde7e9891667163fd75748b7380e10e0ec2a8179eb075c416740.jpg)

![745f824d6559d74af12eb3c4275bff2d6536fe01caaccc85c37e269bfa2940ad.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/745f824d6559d74af12eb3c4275bff2d6536fe01caaccc85c37e269bfa2940ad.jpg)

![7feb97b7b5d3d304a5f1fbdc27fee666382e2e89477255d1b35bdf77dc51cddf.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/7feb97b7b5d3d304a5f1fbdc27fee666382e2e89477255d1b35bdf77dc51cddf.jpg)

![9e5889f1a2549cefa2e3544f4e13bfba807717018854846b1715f40674855000.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/9e5889f1a2549cefa2e3544f4e13bfba807717018854846b1715f40674855000.jpg)

![a3d1097b8810f4e91ba6e962cbb8fe2d40ebabb7ab4091cf8e1702540e137b09.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/a3d1097b8810f4e91ba6e962cbb8fe2d40ebabb7ab4091cf8e1702540e137b09.jpg)

![a909eb2a164f285018d51cd5d7a60acc6f5a7045e22edca38b2fa3deed8ae043.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/a909eb2a164f285018d51cd5d7a60acc6f5a7045e22edca38b2fa3deed8ae043.jpg)

![b84c47ef5d5e373ff97e4d2ab1d499a4d18dbc47066fcf0c252d3fa70e26e288.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/b84c47ef5d5e373ff97e4d2ab1d499a4d18dbc47066fcf0c252d3fa70e26e288.jpg)

![b8b32dc4357e332d04c9db3e5b041c1e6861dd60490f08ecd737daea30df1b94.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/b8b32dc4357e332d04c9db3e5b041c1e6861dd60490f08ecd737daea30df1b94.jpg)

![cbc8b4b165b0a0e0a4bbee471d4af3fdf87653ee3fa59bc86d57d8e1d2351749.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/cbc8b4b165b0a0e0a4bbee471d4af3fdf87653ee3fa59bc86d57d8e1d2351749.jpg)

![cdfe926e26e4696cc029efbbdcd30c3dd5cd9fa19a49b982f23fd46ec8cfae0b.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/cdfe926e26e4696cc029efbbdcd30c3dd5cd9fa19a49b982f23fd46ec8cfae0b.jpg)

![e58235ab0d545904833073dfa99d93028a9683525cee55fda6cc1dc9f318bc7c.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/e58235ab0d545904833073dfa99d93028a9683525cee55fda6cc1dc9f318bc7c.jpg)

![ec7c34161d15adc3c1decc862f7cb0734f5c524802ce3978b2ade132a536745f.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/images/ec7c34161d15adc3c1decc862f7cb0734f5c524802ce3978b2ade132a536745f.jpg)

### Tables

![1875cd284301371a8554fc81d8259400124b2d74b64e63dd0dd011825ea3cf1c.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/tables/1875cd284301371a8554fc81d8259400124b2d74b64e63dd0dd011825ea3cf1c.jpg)

![1f91c915c511fe0425224df70eb7b8aef763a88b74c6c18a58f6a922b305ecc6.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/tables/1f91c915c511fe0425224df70eb7b8aef763a88b74c6c18a58f6a922b305ecc6.jpg)

![693884ac8951bff9837b56035fd3d6add273f5e5fdeb5f4b3efc248137ff117a.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/tables/693884ac8951bff9837b56035fd3d6add273f5e5fdeb5f4b3efc248137ff117a.jpg)

![cf00d9e8939f9fe9240f1a7585fe516b0d9efb67171831830ec6bd21dfca276f.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/tables/cf00d9e8939f9fe9240f1a7585fe516b0d9efb67171831830ec6bd21dfca276f.jpg)

![f7a88a20db34787b3b83160fea62b7b22750111478aa4c1abbabc4f7c8e2fc0b.jpg](../icml_results/2409_Sinkhorn%20Normalization%20of%20Diffusion%20Kernels/tables/f7a88a20db34787b3b83160fea62b7b22750111478aa4c1abbabc4f7c8e2fc0b.jpg)

<a id="paper-2380"></a>

## Robust AI Evaluation through Maximal Lotteries


### Images

![2c70452b0caf17a5b1051b21d58cb0e50d7cadfb0b1d3c08ccc72241c7b73587.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/2c70452b0caf17a5b1051b21d58cb0e50d7cadfb0b1d3c08ccc72241c7b73587.jpg)

![2c8d04778a55618a7e2c011496c261c595aeb887500804ead8dcb111143482dc.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/2c8d04778a55618a7e2c011496c261c595aeb887500804ead8dcb111143482dc.jpg)

![3b0cf71724cd7c6cd9db4db756cd2042dba042071e73aad1229fbc5d33ea6e02.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/3b0cf71724cd7c6cd9db4db756cd2042dba042071e73aad1229fbc5d33ea6e02.jpg)

![40c1de59086e3fee496f6e51c46d7d526bf6f6812525a0495859ae2c24cf9920.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/40c1de59086e3fee496f6e51c46d7d526bf6f6812525a0495859ae2c24cf9920.jpg)

![44b07761b68d93ba737da3b0760badd0d50d8d02131eefcb80b412d9c7d3f7ea.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/44b07761b68d93ba737da3b0760badd0d50d8d02131eefcb80b412d9c7d3f7ea.jpg)

![519cd4b9c9813feabc52e176daeab151065cbd4dd0138014224ca2418a35cc71.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/519cd4b9c9813feabc52e176daeab151065cbd4dd0138014224ca2418a35cc71.jpg)

![523d26b27e04e72fe8389c14de6a44c8984811b2bb620126df87112ad9877dbe.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/523d26b27e04e72fe8389c14de6a44c8984811b2bb620126df87112ad9877dbe.jpg)

![7dc4826db2976fc6f7bae85a67c5f95551bcc5d3ea3da32ccda3864287502fe8.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/7dc4826db2976fc6f7bae85a67c5f95551bcc5d3ea3da32ccda3864287502fe8.jpg)

![7f20d4af0330bfb620b70ff5e95fd291625dc88b680a1cbc519b0bab0a943046.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/7f20d4af0330bfb620b70ff5e95fd291625dc88b680a1cbc519b0bab0a943046.jpg)

![81ced3064094b634711ff503930c633e478f646a0ab3a8ff09fbc4c716763c24.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/81ced3064094b634711ff503930c633e478f646a0ab3a8ff09fbc4c716763c24.jpg)

![876e5fe404961f8399ae1337d8a38e72e05fc07e7456d491ddbed1709ea9d0e4.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/876e5fe404961f8399ae1337d8a38e72e05fc07e7456d491ddbed1709ea9d0e4.jpg)

![8c95aff193079b6c0ce8777f68b95e0bd19f53ef5f8a0faa1015c7264c2e8a81.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/8c95aff193079b6c0ce8777f68b95e0bd19f53ef5f8a0faa1015c7264c2e8a81.jpg)

![8fc1002530419b97dae6987a7689a66d2cc5162e0cc7dc17d61afd5a5988e638.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/8fc1002530419b97dae6987a7689a66d2cc5162e0cc7dc17d61afd5a5988e638.jpg)

![99ef8fbc35afe731f9bd43dfb9038acac20053e10c43364c382d4210c46336ff.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/99ef8fbc35afe731f9bd43dfb9038acac20053e10c43364c382d4210c46336ff.jpg)

![9ecc914722e0cd48adb102201fa4f554eea430f17b557885d7cfefee49323ac8.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/9ecc914722e0cd48adb102201fa4f554eea430f17b557885d7cfefee49323ac8.jpg)

![b17e63dedd93baa9e8c511b53b6e507ef13b35bc7846c30cfcf1bfc611c48a49.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/b17e63dedd93baa9e8c511b53b6e507ef13b35bc7846c30cfcf1bfc611c48a49.jpg)

![c1642aeebb5aa968416858f0319d2c46a037a46ece3435132e4adf556b960b95.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/c1642aeebb5aa968416858f0319d2c46a037a46ece3435132e4adf556b960b95.jpg)

![c3805c16676284ce0aec32c6cb73afad45355c4ce3521b7f7180b1b05dcbd1ff.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/c3805c16676284ce0aec32c6cb73afad45355c4ce3521b7f7180b1b05dcbd1ff.jpg)

![d0cc58910621358ad2920d975fa4dea1a46722692aca04a75ea02bf709fb63fb.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/d0cc58910621358ad2920d975fa4dea1a46722692aca04a75ea02bf709fb63fb.jpg)

![eaa83030d117fb6390dcacc1b5d162dbf045337d91d3d6f3f5a408de73d9337e.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/eaa83030d117fb6390dcacc1b5d162dbf045337d91d3d6f3f5a408de73d9337e.jpg)

![edb32a5f7c372fd38830704caae307271c2a26dbe71611973d960cf37b68ab01.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/edb32a5f7c372fd38830704caae307271c2a26dbe71611973d960cf37b68ab01.jpg)

![fcda3001eb009e28d1faa5ed81bea463e40a013e7829f7c54eee26a2531d4bed.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/images/fcda3001eb009e28d1faa5ed81bea463e40a013e7829f7c54eee26a2531d4bed.jpg)

### Tables

![140d30e047d32412b4857e5a25c15281dc975f4a5827c39358d3a71d3e4d2ac6.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/tables/140d30e047d32412b4857e5a25c15281dc975f4a5827c39358d3a71d3e4d2ac6.jpg)

![3b17ce15951febac9485db3def1ee67e4b3a6b1b742ffa61f5e95348786d2a71.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/tables/3b17ce15951febac9485db3def1ee67e4b3a6b1b742ffa61f5e95348786d2a71.jpg)

![43ac3a61f33dde00f9e069fe8a9069c4813e894038deff62b490c37386c14e29.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/tables/43ac3a61f33dde00f9e069fe8a9069c4813e894038deff62b490c37386c14e29.jpg)

![8ca585225213dbd1f890d033bfd5d8c7056b5354cc0aaa894b75a1a45e79c72c.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/tables/8ca585225213dbd1f890d033bfd5d8c7056b5354cc0aaa894b75a1a45e79c72c.jpg)

![91bb744b5c8dab1b5b7ca6814f39c4f010ecb1be2a510b463faa3edf9ccadf84.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/tables/91bb744b5c8dab1b5b7ca6814f39c4f010ecb1be2a510b463faa3edf9ccadf84.jpg)

![969a69d3d7e81803e0dc217c25e498be103254a5134f611385428cf44cde940a.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/tables/969a69d3d7e81803e0dc217c25e498be103254a5134f611385428cf44cde940a.jpg)

![a57b5a929b9f78d177f412fae590b3f04f139ab335e178b410fe579bdd187e7d.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/tables/a57b5a929b9f78d177f412fae590b3f04f139ab335e178b410fe579bdd187e7d.jpg)

![c63cb521475ef4dfa9137a6f2764527f60c9f9f2edcea194877a8416db1dc876.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/tables/c63cb521475ef4dfa9137a6f2764527f60c9f9f2edcea194877a8416db1dc876.jpg)

![f2e168d6ab8b731d52ea69802adb7714842ae4282028ff61ca556a5eab2562c1.jpg](../icml_results/2410_Robust%20AI%20Evaluation%20through%20Maximal%20Lotteries/tables/f2e168d6ab8b731d52ea69802adb7714842ae4282028ff61ca556a5eab2562c1.jpg)

<a id="paper-2381"></a>

## Unlearning in Diffusion Models: A Unified Framework with KL Divergence and Likelihood Constraints


### Images

![00d36b5efe7a7def85cd70e25f3bfaeb2f1a33a31aafaeed4e805a222d7da3d2.jpg](../icml_results/2411_Unlearning%20in%20Diffusion%20Models_%20A%20Unified%20Framework%20with%20KL%20Divergence%20and%20Likelihood%20Constraints/images/00d36b5efe7a7def85cd70e25f3bfaeb2f1a33a31aafaeed4e805a222d7da3d2.jpg)

![1eb7745c9013b6b9417cff63738fbcbdbd82d998b524cd6e190d536f23485d38.jpg](../icml_results/2411_Unlearning%20in%20Diffusion%20Models_%20A%20Unified%20Framework%20with%20KL%20Divergence%20and%20Likelihood%20Constraints/images/1eb7745c9013b6b9417cff63738fbcbdbd82d998b524cd6e190d536f23485d38.jpg)

![4d48b6bdbd1e4388ddb1bc76b60002089a8b167a4ccc89d587b3f8e7384f60ca.jpg](../icml_results/2411_Unlearning%20in%20Diffusion%20Models_%20A%20Unified%20Framework%20with%20KL%20Divergence%20and%20Likelihood%20Constraints/images/4d48b6bdbd1e4388ddb1bc76b60002089a8b167a4ccc89d587b3f8e7384f60ca.jpg)

![a0e020a451b7410eb03f64c6dd7cd775d60a70db2ab3540b23cfbe4e42f2513d.jpg](../icml_results/2411_Unlearning%20in%20Diffusion%20Models_%20A%20Unified%20Framework%20with%20KL%20Divergence%20and%20Likelihood%20Constraints/images/a0e020a451b7410eb03f64c6dd7cd775d60a70db2ab3540b23cfbe4e42f2513d.jpg)

![aa3fd95588f6c5d7fdaa53b75b0415d3287aebf050b682e96e99a2dd35c824e2.jpg](../icml_results/2411_Unlearning%20in%20Diffusion%20Models_%20A%20Unified%20Framework%20with%20KL%20Divergence%20and%20Likelihood%20Constraints/images/aa3fd95588f6c5d7fdaa53b75b0415d3287aebf050b682e96e99a2dd35c824e2.jpg)

![b30b97ce2b359843620d8aafaaef3e3943c37acc41f1344263316766441c18c8.jpg](../icml_results/2411_Unlearning%20in%20Diffusion%20Models_%20A%20Unified%20Framework%20with%20KL%20Divergence%20and%20Likelihood%20Constraints/images/b30b97ce2b359843620d8aafaaef3e3943c37acc41f1344263316766441c18c8.jpg)

### Tables

![06c2c86492223115cc6e07de58a63d1731d91da2fa77591e647d1219adcaa630.jpg](../icml_results/2411_Unlearning%20in%20Diffusion%20Models_%20A%20Unified%20Framework%20with%20KL%20Divergence%20and%20Likelihood%20Constraints/tables/06c2c86492223115cc6e07de58a63d1731d91da2fa77591e647d1219adcaa630.jpg)

![09886de74120fcd10a4df3a73aa7bd97908bcd12a8c413ad816014a7c3b67591.jpg](../icml_results/2411_Unlearning%20in%20Diffusion%20Models_%20A%20Unified%20Framework%20with%20KL%20Divergence%20and%20Likelihood%20Constraints/tables/09886de74120fcd10a4df3a73aa7bd97908bcd12a8c413ad816014a7c3b67591.jpg)

![774d80a841d34be482f073452208df0d82c94f00115035abe437b08848ee61fa.jpg](../icml_results/2411_Unlearning%20in%20Diffusion%20Models_%20A%20Unified%20Framework%20with%20KL%20Divergence%20and%20Likelihood%20Constraints/tables/774d80a841d34be482f073452208df0d82c94f00115035abe437b08848ee61fa.jpg)

![af8b4acd42edcce44ea81c63834b58c636be352cc4b4729b5f4ebee384fa5d7d.jpg](../icml_results/2411_Unlearning%20in%20Diffusion%20Models_%20A%20Unified%20Framework%20with%20KL%20Divergence%20and%20Likelihood%20Constraints/tables/af8b4acd42edcce44ea81c63834b58c636be352cc4b4729b5f4ebee384fa5d7d.jpg)

<a id="paper-2382"></a>

## Reward Learning through Ranking Mean Squared Error


### Images

![0238d8ce4883d27a43cbbbeedef7cdd01c0bdda9d23ed9fd559213302f40b6d8.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/0238d8ce4883d27a43cbbbeedef7cdd01c0bdda9d23ed9fd559213302f40b6d8.jpg)

![066110d3308596df70bf80b6e8402ef63770a0e859865b98627cf780d7819561.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/066110d3308596df70bf80b6e8402ef63770a0e859865b98627cf780d7819561.jpg)

![0ee0bbe326da77ec4fd775f6ac590082a292de9d8796b09e8d20a847867267d6.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/0ee0bbe326da77ec4fd775f6ac590082a292de9d8796b09e8d20a847867267d6.jpg)

![165732c9dcf8c31e6f44497f71c97f2e8c76c28e72e90e4dcfd834632f1bfb5f.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/165732c9dcf8c31e6f44497f71c97f2e8c76c28e72e90e4dcfd834632f1bfb5f.jpg)

![2ee37f08ffbc702ae91b15035384a08470d9a65f5534094b1d1fc09f95e8af88.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/2ee37f08ffbc702ae91b15035384a08470d9a65f5534094b1d1fc09f95e8af88.jpg)

![37ddc79a0cf5810bf796e70cc23455c3a736abf22034f0f297b0332f423e0ed0.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/37ddc79a0cf5810bf796e70cc23455c3a736abf22034f0f297b0332f423e0ed0.jpg)

![5127f21bac88a0542bce5719737c6c6252ac56833d3b684cf5d9a2c1b2d58845.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/5127f21bac88a0542bce5719737c6c6252ac56833d3b684cf5d9a2c1b2d58845.jpg)

![5fb8f2d77174d459c36b5fc38785b354353e185475dca5c2bf580fc23e90474f.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/5fb8f2d77174d459c36b5fc38785b354353e185475dca5c2bf580fc23e90474f.jpg)

![80673d8eea017c49263adf2130737439331e45a3938191ffdc9ba9bbfc46a2c0.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/80673d8eea017c49263adf2130737439331e45a3938191ffdc9ba9bbfc46a2c0.jpg)

![813d92d8dc0273d0f97ad8fa5ac8575267f07d91d3f4b019def22d177a770a90.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/813d92d8dc0273d0f97ad8fa5ac8575267f07d91d3f4b019def22d177a770a90.jpg)

![81f41b3bbd63ee21c9f536e8f186ef3a02caf9c99da9793e95c4793ae336d755.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/81f41b3bbd63ee21c9f536e8f186ef3a02caf9c99da9793e95c4793ae336d755.jpg)

![82851998824a3d03f655a4bdb132d19bd167806564e71df0c790ba88a843b349.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/82851998824a3d03f655a4bdb132d19bd167806564e71df0c790ba88a843b349.jpg)

![83c789efb67f0ca2af7977a98281151717acdb65c09d791aae2ae327eed06771.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/83c789efb67f0ca2af7977a98281151717acdb65c09d791aae2ae327eed06771.jpg)

![8cce29e30be8a1f31f75a6c08801f9deb225e02e2005053dce1db1f58fee4821.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/8cce29e30be8a1f31f75a6c08801f9deb225e02e2005053dce1db1f58fee4821.jpg)

![9120810aa3b2b8a37e29d3732c3d53292bbefb7f46cb2dec4441f43b6bc5a1a4.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/9120810aa3b2b8a37e29d3732c3d53292bbefb7f46cb2dec4441f43b6bc5a1a4.jpg)

![9773f495bb77ea7dfd814e1cf26ffaf291e89e7797758f8edeb1df65e6d95e59.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/9773f495bb77ea7dfd814e1cf26ffaf291e89e7797758f8edeb1df65e6d95e59.jpg)

![9bbf27644bb93436fba56811d0c28816b3d8f6eca0fb0fc7c223e77aca9838bc.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/9bbf27644bb93436fba56811d0c28816b3d8f6eca0fb0fc7c223e77aca9838bc.jpg)

![ac0a6339c2c8d3d992ddfb74d535bba091211d2c56c8e40f5bf4a017b9206ce1.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/ac0a6339c2c8d3d992ddfb74d535bba091211d2c56c8e40f5bf4a017b9206ce1.jpg)

![aed9feaa47ac6773f39fc2649dec8056da7ee01fadcff6aef169b5ed9e69fa7c.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/aed9feaa47ac6773f39fc2649dec8056da7ee01fadcff6aef169b5ed9e69fa7c.jpg)

![b3846ee8b114b22451d877331d365d2183890aeb117b8f1f99b08b111ec6b514.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/b3846ee8b114b22451d877331d365d2183890aeb117b8f1f99b08b111ec6b514.jpg)

![d1706de2752784c3a8e00317177d9a21163868f3e4925d93e782f5c73dea1ad9.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/d1706de2752784c3a8e00317177d9a21163868f3e4925d93e782f5c73dea1ad9.jpg)

![deed72cd1b36fead471fbf3aa059776721e2961f5c4aa8c08a03677c6dfb6f74.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/deed72cd1b36fead471fbf3aa059776721e2961f5c4aa8c08a03677c6dfb6f74.jpg)

![df4b16c0a9ef61e5fbe1afe86c215e439b5b6f60082a03ef724c7d9e0f31b381.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/df4b16c0a9ef61e5fbe1afe86c215e439b5b6f60082a03ef724c7d9e0f31b381.jpg)

![dffd21e4047b23ca1cf7a71e1671361d7c1809a2b2930a63feae0ee12496043c.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/dffd21e4047b23ca1cf7a71e1671361d7c1809a2b2930a63feae0ee12496043c.jpg)

![eb3e40b74dc31b2ff9dfc607f2fcb1c472b329e8b3d9a3cf6018be794e4218de.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/eb3e40b74dc31b2ff9dfc607f2fcb1c472b329e8b3d9a3cf6018be794e4218de.jpg)

![fe10ec670c72642d751c2dd0ee2f4ca9f8da023d5c5c1e423e2d96787881d940.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/images/fe10ec670c72642d751c2dd0ee2f4ca9f8da023d5c5c1e423e2d96787881d940.jpg)

### Tables

![083534f9040dd6829f30eb33e8e794d35a79b8bd681ff6bc55dcc3a86279deee.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/tables/083534f9040dd6829f30eb33e8e794d35a79b8bd681ff6bc55dcc3a86279deee.jpg)

![377940afc42aa5569391947588f795bff84b51f75a81d65fa0b8de177337beb8.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/tables/377940afc42aa5569391947588f795bff84b51f75a81d65fa0b8de177337beb8.jpg)

![65cbdc9704fbab8631d4ae8d5239fb4eaa6828da5bb4854e4d00cdd4f6331adb.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/tables/65cbdc9704fbab8631d4ae8d5239fb4eaa6828da5bb4854e4d00cdd4f6331adb.jpg)

![6c777e1de264af8fc9e3af8ce58cacd210dd6e0af8aefc72c87b75bb1ce0450c.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/tables/6c777e1de264af8fc9e3af8ce58cacd210dd6e0af8aefc72c87b75bb1ce0450c.jpg)

![6fd8fa36403964bac41efe3b2ef840d13d80b0a26e4143d1c1320b7d9c537b19.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/tables/6fd8fa36403964bac41efe3b2ef840d13d80b0a26e4143d1c1320b7d9c537b19.jpg)

![7077d6de0e9e3679f992a8bd728143e01ee79dac3175b820dde5896916f5e837.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/tables/7077d6de0e9e3679f992a8bd728143e01ee79dac3175b820dde5896916f5e837.jpg)

![778897e7d2a4b0e493e9a25544fa6b21812d221eaed5d98b79eb77eb50635f38.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/tables/778897e7d2a4b0e493e9a25544fa6b21812d221eaed5d98b79eb77eb50635f38.jpg)

![791ce902c15daedf70d37e7be208c035361d5c65a5c202049451972ed8f4e0f1.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/tables/791ce902c15daedf70d37e7be208c035361d5c65a5c202049451972ed8f4e0f1.jpg)

![9ec54c3bf8a00573fbaba9c61be32493f593263e3c7905640cd51d37fe382a5f.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/tables/9ec54c3bf8a00573fbaba9c61be32493f593263e3c7905640cd51d37fe382a5f.jpg)

![b369c17926df0d6f60ffe9a3f278cbc22070cd0f132b0ea970d746795697866f.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/tables/b369c17926df0d6f60ffe9a3f278cbc22070cd0f132b0ea970d746795697866f.jpg)

![bdc7e778a0ad6b2032077bded244944ba7bc9ca086380dc3dd0f3f61e95a39a0.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/tables/bdc7e778a0ad6b2032077bded244944ba7bc9ca086380dc3dd0f3f61e95a39a0.jpg)

![cff7b8a6f8aeb507e72f58d2e0f01becd5efb3c0101f4e5a9125e71fc9aebf27.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/tables/cff7b8a6f8aeb507e72f58d2e0f01becd5efb3c0101f4e5a9125e71fc9aebf27.jpg)

![da9d36845ca88c63f777e8ead21c8eb32870eab41641c71fcb68ea5ab86f3ba7.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/tables/da9d36845ca88c63f777e8ead21c8eb32870eab41641c71fcb68ea5ab86f3ba7.jpg)

![ddc787898dd2cb0aa86f4d427e3776b290259beb6fdeb86245b884a64a4e5d6c.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/tables/ddc787898dd2cb0aa86f4d427e3776b290259beb6fdeb86245b884a64a4e5d6c.jpg)

![f81d4d8e239780d52c4caa05aef5ed41afa5a5d7cdcad0b0570f212b013f4981.jpg](../icml_results/2412_Reward%20Learning%20through%20Ranking%20Mean%20Squared%20Error/tables/f81d4d8e239780d52c4caa05aef5ed41afa5a5d7cdcad0b0570f212b013f4981.jpg)

<a id="paper-2383"></a>

## A Deep Learning Model of Mental Rotation Informed by Interactive VR Experiments


### Images

![1c207e20c0da73c1b9ed0de86a3fb9604594712a26f44540d151eef4600bce69.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/images/1c207e20c0da73c1b9ed0de86a3fb9604594712a26f44540d151eef4600bce69.jpg)

![207085e08d95231e1ed3c66fb0b5c01692eaf48dc303de59cdaf47fbd11d9e70.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/images/207085e08d95231e1ed3c66fb0b5c01692eaf48dc303de59cdaf47fbd11d9e70.jpg)

![3372e57aeaa07b1cd4ec44cd5072cfedf991667ecd7ca16ba5e8c8933c9dedf5.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/images/3372e57aeaa07b1cd4ec44cd5072cfedf991667ecd7ca16ba5e8c8933c9dedf5.jpg)

![363a25be601990c7b42e10cbbdb219ff218121bd263db2a7c6b1cffa52f4da75.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/images/363a25be601990c7b42e10cbbdb219ff218121bd263db2a7c6b1cffa52f4da75.jpg)

![4a54442c11009c646fd30c2144b8f448d81022a322a06a00086f842bacdb5d90.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/images/4a54442c11009c646fd30c2144b8f448d81022a322a06a00086f842bacdb5d90.jpg)

![5c6e021053934b2b2ce6f0ac60ac6a019583fb85c7d010346f67ea44b21f7335.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/images/5c6e021053934b2b2ce6f0ac60ac6a019583fb85c7d010346f67ea44b21f7335.jpg)

![60401d3c4e78105892e1938cd2d816a76421587df6455e26a86156d9b7f26fa6.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/images/60401d3c4e78105892e1938cd2d816a76421587df6455e26a86156d9b7f26fa6.jpg)

![6ea1c9bfa9702d1ba2ac1c5ec8a7845c4661791c27e0ad936d82420920b22968.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/images/6ea1c9bfa9702d1ba2ac1c5ec8a7845c4661791c27e0ad936d82420920b22968.jpg)

![9690d6371ebde22eda021ca31c3f6be2cf76827ad6659d659a2a5e1982506f3b.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/images/9690d6371ebde22eda021ca31c3f6be2cf76827ad6659d659a2a5e1982506f3b.jpg)

![99696c2354723eaeca75b91ebf66e0da1cd8d8772530412426eeb1fe00c7ec37.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/images/99696c2354723eaeca75b91ebf66e0da1cd8d8772530412426eeb1fe00c7ec37.jpg)

![a2bf9837ca4cbd4bda840dd1727af7d0fea0b2ceca2e159c4cd17e157d244c3a.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/images/a2bf9837ca4cbd4bda840dd1727af7d0fea0b2ceca2e159c4cd17e157d244c3a.jpg)

![ce987e7aa87cdb6f0229e843638cb35471387d57b0232bf34ce75e358d6c25d0.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/images/ce987e7aa87cdb6f0229e843638cb35471387d57b0232bf34ce75e358d6c25d0.jpg)

![d8f5c27f20f26618e7c21fe4eb50337f618342d3c80b55f8d569b2cf67a78bd8.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/images/d8f5c27f20f26618e7c21fe4eb50337f618342d3c80b55f8d569b2cf67a78bd8.jpg)

![ec1e131e65bdc472367d0c29615fdfdde4e133532ac67e81ca7dd6e6a866b67a.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/images/ec1e131e65bdc472367d0c29615fdfdde4e133532ac67e81ca7dd6e6a866b67a.jpg)

![f3a85b9946d6b1e4b1c2b8ef65a4d9756ef8702f8f30e33e362ab4981899682c.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/images/f3a85b9946d6b1e4b1c2b8ef65a4d9756ef8702f8f30e33e362ab4981899682c.jpg)

![f4f61ec58a7efa20a18f315010862bf258572d44d4bbeb2faaae1d2636f032bf.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/images/f4f61ec58a7efa20a18f315010862bf258572d44d4bbeb2faaae1d2636f032bf.jpg)

![fa0da0b311fffea5a26bc230f401216cc38201212d3fbe9e4c562ed13e619266.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/images/fa0da0b311fffea5a26bc230f401216cc38201212d3fbe9e4c562ed13e619266.jpg)

### Tables

![285897217e40d84c4e53825c905d12d74bdf620e5846f86185638f913daee908.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/tables/285897217e40d84c4e53825c905d12d74bdf620e5846f86185638f913daee908.jpg)

![c0e5eadd5690ae47f1e0a52fc1aa9b48543d2a548b7e02fcd9967360d38e7d2b.jpg](../icml_results/2413_A%20Deep%20Learning%20Model%20of%20Mental%20Rotation%20Informed%20by%20Interactive%20VR%20Experiments/tables/c0e5eadd5690ae47f1e0a52fc1aa9b48543d2a548b7e02fcd9967360d38e7d2b.jpg)

<a id="paper-2384"></a>

## Quantifying Error Propagation and Model Collapse in Diffusion Models


### Images

![17a04f61f9f35b706e9926762b8a568dac2faf0ac9a041c67424722fca5c8b18.jpg](../icml_results/2414_Quantifying%20Error%20Propagation%20and%20Model%20Collapse%20in%20Diffusion%20Models/images/17a04f61f9f35b706e9926762b8a568dac2faf0ac9a041c67424722fca5c8b18.jpg)

![1b75247b0e723c1c942b1f15f7d5503d5cf42d954bb73d3ffb63c1dd54c8b4ea.jpg](../icml_results/2414_Quantifying%20Error%20Propagation%20and%20Model%20Collapse%20in%20Diffusion%20Models/images/1b75247b0e723c1c942b1f15f7d5503d5cf42d954bb73d3ffb63c1dd54c8b4ea.jpg)

![38e42b10aa4e6a078b99586a077274cbe4e39735a28a5f49378b517eb63c3813.jpg](../icml_results/2414_Quantifying%20Error%20Propagation%20and%20Model%20Collapse%20in%20Diffusion%20Models/images/38e42b10aa4e6a078b99586a077274cbe4e39735a28a5f49378b517eb63c3813.jpg)

![3c2566789958ede6cf01575c29935990347e9684d795d2ee192f7e84ed234ee2.jpg](../icml_results/2414_Quantifying%20Error%20Propagation%20and%20Model%20Collapse%20in%20Diffusion%20Models/images/3c2566789958ede6cf01575c29935990347e9684d795d2ee192f7e84ed234ee2.jpg)

![3fb40e12a01f96b2e14e64376336ffa3cb38065277244a774b0f655b99f90a1e.jpg](../icml_results/2414_Quantifying%20Error%20Propagation%20and%20Model%20Collapse%20in%20Diffusion%20Models/images/3fb40e12a01f96b2e14e64376336ffa3cb38065277244a774b0f655b99f90a1e.jpg)

![5ac6e5498ef8d2fee43942237bd7f21b11695d60ddf41e98655bf185768ad3d9.jpg](../icml_results/2414_Quantifying%20Error%20Propagation%20and%20Model%20Collapse%20in%20Diffusion%20Models/images/5ac6e5498ef8d2fee43942237bd7f21b11695d60ddf41e98655bf185768ad3d9.jpg)

![6e3286805d3245f908747892a64dab3fcbce465ba9d53007d89dde7be28bb78d.jpg](../icml_results/2414_Quantifying%20Error%20Propagation%20and%20Model%20Collapse%20in%20Diffusion%20Models/images/6e3286805d3245f908747892a64dab3fcbce465ba9d53007d89dde7be28bb78d.jpg)

![b0821408de342a79b880f4b94370a36c5a8bb64c251bd65120f536ac877b31f0.jpg](../icml_results/2414_Quantifying%20Error%20Propagation%20and%20Model%20Collapse%20in%20Diffusion%20Models/images/b0821408de342a79b880f4b94370a36c5a8bb64c251bd65120f536ac877b31f0.jpg)

![d29b73cda514a6d5b8986d82aebead08da0a7740fff134238bb01b1dca8463f5.jpg](../icml_results/2414_Quantifying%20Error%20Propagation%20and%20Model%20Collapse%20in%20Diffusion%20Models/images/d29b73cda514a6d5b8986d82aebead08da0a7740fff134238bb01b1dca8463f5.jpg)

![ee942f5930297a47c6a6f0cb16362da6d5684f1b545b32fbfa469c925e546c3c.jpg](../icml_results/2414_Quantifying%20Error%20Propagation%20and%20Model%20Collapse%20in%20Diffusion%20Models/images/ee942f5930297a47c6a6f0cb16362da6d5684f1b545b32fbfa469c925e546c3c.jpg)

![f9989d05ffad64467f2130f7a40dfade8fd742ffda05506b5072ca7475ddb63f.jpg](../icml_results/2414_Quantifying%20Error%20Propagation%20and%20Model%20Collapse%20in%20Diffusion%20Models/images/f9989d05ffad64467f2130f7a40dfade8fd742ffda05506b5072ca7475ddb63f.jpg)

<a id="paper-2385"></a>

## Federated Causal Inference on Multi-Site Observational Data via Propensity Score Aggregation


### Images

![1ddcde75664806e88ee2acc1cc638f61be8a41c166e6fe52b3b4f3e01d2381fa.jpg](../icml_results/2415_Federated%20Causal%20Inference%20on%20Multi-Site%20Observational%20Data%20via%20Propensity%20Score%20Aggregation/images/1ddcde75664806e88ee2acc1cc638f61be8a41c166e6fe52b3b4f3e01d2381fa.jpg)

![2944c02ad26fd1cd57e77618b1f19ff698d9dc44498a6264d081123435530918.jpg](../icml_results/2415_Federated%20Causal%20Inference%20on%20Multi-Site%20Observational%20Data%20via%20Propensity%20Score%20Aggregation/images/2944c02ad26fd1cd57e77618b1f19ff698d9dc44498a6264d081123435530918.jpg)

![2fb00e648b816fa6c13ab5f8a822a6203bb40f652bee0d0ca63a0dcf2e4c830f.jpg](../icml_results/2415_Federated%20Causal%20Inference%20on%20Multi-Site%20Observational%20Data%20via%20Propensity%20Score%20Aggregation/images/2fb00e648b816fa6c13ab5f8a822a6203bb40f652bee0d0ca63a0dcf2e4c830f.jpg)

![391c9adf22cd133f348e200b58786fe3d9b8682b037fce30857edc06220722f9.jpg](../icml_results/2415_Federated%20Causal%20Inference%20on%20Multi-Site%20Observational%20Data%20via%20Propensity%20Score%20Aggregation/images/391c9adf22cd133f348e200b58786fe3d9b8682b037fce30857edc06220722f9.jpg)

![4d4446534ad076b86f62ee10c7a7743e625cbb5c0656e1b89747c1d7d1f9be0a.jpg](../icml_results/2415_Federated%20Causal%20Inference%20on%20Multi-Site%20Observational%20Data%20via%20Propensity%20Score%20Aggregation/images/4d4446534ad076b86f62ee10c7a7743e625cbb5c0656e1b89747c1d7d1f9be0a.jpg)

![665f2851e811e80a234a63468ddcda8c838f87ba0d7b04e6a690e0f355a5de7f.jpg](../icml_results/2415_Federated%20Causal%20Inference%20on%20Multi-Site%20Observational%20Data%20via%20Propensity%20Score%20Aggregation/images/665f2851e811e80a234a63468ddcda8c838f87ba0d7b04e6a690e0f355a5de7f.jpg)

![6f1098dac3c3fbd3fb2a8d7fcbaf9b23682f378d6ba90743a0549066134f0a6f.jpg](../icml_results/2415_Federated%20Causal%20Inference%20on%20Multi-Site%20Observational%20Data%20via%20Propensity%20Score%20Aggregation/images/6f1098dac3c3fbd3fb2a8d7fcbaf9b23682f378d6ba90743a0549066134f0a6f.jpg)

![d6880e1578b6acc9b6c01e031a14aabf9a94afb129b2354c494e6627cc323724.jpg](../icml_results/2415_Federated%20Causal%20Inference%20on%20Multi-Site%20Observational%20Data%20via%20Propensity%20Score%20Aggregation/images/d6880e1578b6acc9b6c01e031a14aabf9a94afb129b2354c494e6627cc323724.jpg)

### Tables

![40c3f4b4cb62519d0bff79175cf703fcd16a38c04e0bed2907f0e67ffde80c07.jpg](../icml_results/2415_Federated%20Causal%20Inference%20on%20Multi-Site%20Observational%20Data%20via%20Propensity%20Score%20Aggregation/tables/40c3f4b4cb62519d0bff79175cf703fcd16a38c04e0bed2907f0e67ffde80c07.jpg)

![8968a44a7b1d0753049d9d0278c1fc51a45413c27d605af8a544994934f020ac.jpg](../icml_results/2415_Federated%20Causal%20Inference%20on%20Multi-Site%20Observational%20Data%20via%20Propensity%20Score%20Aggregation/tables/8968a44a7b1d0753049d9d0278c1fc51a45413c27d605af8a544994934f020ac.jpg)

![9802a7770ccc5219e881c828e27124f5d78df50c4497983ec217e9f0caa2fb56.jpg](../icml_results/2415_Federated%20Causal%20Inference%20on%20Multi-Site%20Observational%20Data%20via%20Propensity%20Score%20Aggregation/tables/9802a7770ccc5219e881c828e27124f5d78df50c4497983ec217e9f0caa2fb56.jpg)

![c5a418a14043f28de90af2ab42b3f386d3302705cf890d2823988e187a2d996f.jpg](../icml_results/2415_Federated%20Causal%20Inference%20on%20Multi-Site%20Observational%20Data%20via%20Propensity%20Score%20Aggregation/tables/c5a418a14043f28de90af2ab42b3f386d3302705cf890d2823988e187a2d996f.jpg)

![ce009c7afcb8d57cf192275cea0c07aa3e73353b34112ec25c23204a9512de6e.jpg](../icml_results/2415_Federated%20Causal%20Inference%20on%20Multi-Site%20Observational%20Data%20via%20Propensity%20Score%20Aggregation/tables/ce009c7afcb8d57cf192275cea0c07aa3e73353b34112ec25c23204a9512de6e.jpg)

![e8cef6abb681695d8c869a539595d0a2a09056e2b417a873016da2df8e3749ea.jpg](../icml_results/2415_Federated%20Causal%20Inference%20on%20Multi-Site%20Observational%20Data%20via%20Propensity%20Score%20Aggregation/tables/e8cef6abb681695d8c869a539595d0a2a09056e2b417a873016da2df8e3749ea.jpg)

![eb8bcf67e97e61804f1b0f7c2ea0eb69a691f1fe6c4ac85aa1e0c20b646b688a.jpg](../icml_results/2415_Federated%20Causal%20Inference%20on%20Multi-Site%20Observational%20Data%20via%20Propensity%20Score%20Aggregation/tables/eb8bcf67e97e61804f1b0f7c2ea0eb69a691f1fe6c4ac85aa1e0c20b646b688a.jpg)

<a id="paper-2386"></a>

## Minimum Distance Summaries for Robust Neural Posterior Estimation


### Images

![0b7ac6dc06f699eb02906e57b5f13a373d28fbf1fc02d86d21f8050d32311143.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/0b7ac6dc06f699eb02906e57b5f13a373d28fbf1fc02d86d21f8050d32311143.jpg)

![0eb3cbb6bf3ed56d67727cc867331c0c64c2fcd8425589fdd352d117e17e41fd.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/0eb3cbb6bf3ed56d67727cc867331c0c64c2fcd8425589fdd352d117e17e41fd.jpg)

![1238ceccc7756db97e3d5ff4a93124a1fef4f2492e321d3ce236e2f5f9faefb6.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/1238ceccc7756db97e3d5ff4a93124a1fef4f2492e321d3ce236e2f5f9faefb6.jpg)

![29d181d1538b7f56763074aacedf90b56628974531b91145c74908bd8edfca29.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/29d181d1538b7f56763074aacedf90b56628974531b91145c74908bd8edfca29.jpg)

![2be70db84dcace3695ad48ec5f575e91a6f714578cc916b1292832c1727205d5.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/2be70db84dcace3695ad48ec5f575e91a6f714578cc916b1292832c1727205d5.jpg)

![407b0bfabe412ce40028dd892b44debfc66761bb8384876d84b8dacfb0343063.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/407b0bfabe412ce40028dd892b44debfc66761bb8384876d84b8dacfb0343063.jpg)

![4985d8b5517660ac85f5fb565e0b73bd2cad45da4b0c55456f6515e6e36d8333.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/4985d8b5517660ac85f5fb565e0b73bd2cad45da4b0c55456f6515e6e36d8333.jpg)

![534f7c25df2c8ccafdd01a9b5892ab61d1b13a1bd4609ae80fa707ec9c6a2a81.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/534f7c25df2c8ccafdd01a9b5892ab61d1b13a1bd4609ae80fa707ec9c6a2a81.jpg)

![59234f62d403170b13f9df3d3de0be0487cf648e8e58bd52c4cbf8f4cc64779e.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/59234f62d403170b13f9df3d3de0be0487cf648e8e58bd52c4cbf8f4cc64779e.jpg)

![5fdae59d9091ac01a7a2ff3470f806f0d2ffce3fa18961bad03029b6a1541279.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/5fdae59d9091ac01a7a2ff3470f806f0d2ffce3fa18961bad03029b6a1541279.jpg)

![6fe195f33ae9b9b87f7807811a79415837776b4d97eb9871cbf05fb3749951a9.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/6fe195f33ae9b9b87f7807811a79415837776b4d97eb9871cbf05fb3749951a9.jpg)

![73fc318607f19484c8b1e47a9a16557c463e8d73894e9273428a9ae0a1935999.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/73fc318607f19484c8b1e47a9a16557c463e8d73894e9273428a9ae0a1935999.jpg)

![7f3051b3583267b2b96c417a922c066c098016a1dca8132edfd2272c9bc02498.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/7f3051b3583267b2b96c417a922c066c098016a1dca8132edfd2272c9bc02498.jpg)

![7f5681b3b730f25a5c465e807da794f43dc38825029da3fb5fec64cb8b4b40fc.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/7f5681b3b730f25a5c465e807da794f43dc38825029da3fb5fec64cb8b4b40fc.jpg)

![89283a09b06731a33081b8b589904effab4f38ba9496b28e013d64f38f3e9f85.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/89283a09b06731a33081b8b589904effab4f38ba9496b28e013d64f38f3e9f85.jpg)

![9006fdb0ac64b613edab67caeedc0baa121732e0478ee8ace32487fb28baaf6c.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/9006fdb0ac64b613edab67caeedc0baa121732e0478ee8ace32487fb28baaf6c.jpg)

![90cd20b54ad8d3e211633f5aba38ad7260fe03401ab5d0f6b31785c852ef4760.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/90cd20b54ad8d3e211633f5aba38ad7260fe03401ab5d0f6b31785c852ef4760.jpg)

![9ec909c796801592778d20ea4c3d689d59ade7a9ab15699118ea010c878d4972.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/9ec909c796801592778d20ea4c3d689d59ade7a9ab15699118ea010c878d4972.jpg)

![a44b2c51c0be7a07059c9d9c4bb41f10809f0e83d3fff30b30f578971825a9fe.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/a44b2c51c0be7a07059c9d9c4bb41f10809f0e83d3fff30b30f578971825a9fe.jpg)

![ae22e236644dd54f269acfe8c9ebd5afd5599115d18952ebabcaf2a54d753a34.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/ae22e236644dd54f269acfe8c9ebd5afd5599115d18952ebabcaf2a54d753a34.jpg)

![b16984c8022347c724381a51c1339e5e7b95ed03b9cd78a8e2193804ed82ea22.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/b16984c8022347c724381a51c1339e5e7b95ed03b9cd78a8e2193804ed82ea22.jpg)

![b17d1662d95d973abba6461cf3ca1e4c4a9884ab2bda485f183a2e0fddfbae19.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/b17d1662d95d973abba6461cf3ca1e4c4a9884ab2bda485f183a2e0fddfbae19.jpg)

![b6f1c6e2ee0df67d0d345dacffd289569b358eae697ccf20614574041d4c3200.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/b6f1c6e2ee0df67d0d345dacffd289569b358eae697ccf20614574041d4c3200.jpg)

![b9e47cd633d27a5a9e8d53d1dc94adb3dd67461f31c313ba952769993913c343.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/b9e47cd633d27a5a9e8d53d1dc94adb3dd67461f31c313ba952769993913c343.jpg)

![cb86ac23cf174c72d53cc1c17d6ead8d548a9d48eabaa7cd2542b3b706219de7.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/cb86ac23cf174c72d53cc1c17d6ead8d548a9d48eabaa7cd2542b3b706219de7.jpg)

![d54bca6860738cbd0f07590abc6c8b319ea2ca3c03195bcb8d454118960fab59.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/d54bca6860738cbd0f07590abc6c8b319ea2ca3c03195bcb8d454118960fab59.jpg)

![d635fd03a8628cebf61215c35eb70e6ce5fa3dba874e08de29c34a5f144a822f.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/d635fd03a8628cebf61215c35eb70e6ce5fa3dba874e08de29c34a5f144a822f.jpg)

![dc0c29cd14aaaca94d9b9173a0c269ced9c78be8ec52058d09e027078f127d4e.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/dc0c29cd14aaaca94d9b9173a0c269ced9c78be8ec52058d09e027078f127d4e.jpg)

![ea91dc08ba663f6b566cc1f3aac72d37d0d217c238e07d4c5ba4c27f9ce37d9d.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/ea91dc08ba663f6b566cc1f3aac72d37d0d217c238e07d4c5ba4c27f9ce37d9d.jpg)

![ee85e4a5e17c28955b7a905c00d71ba3f9ee0d50841bc8758506a9dbd53aaddb.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/images/ee85e4a5e17c28955b7a905c00d71ba3f9ee0d50841bc8758506a9dbd53aaddb.jpg)

### Tables

![3f9910e78e2bcb566058ad6185b127b52aa6de9ccd37dbca5d4898353bafe39a.jpg](../icml_results/2416_Minimum%20Distance%20Summaries%20for%20Robust%20Neural%20Posterior%20Estimation/tables/3f9910e78e2bcb566058ad6185b127b52aa6de9ccd37dbca5d4898353bafe39a.jpg)

<a id="paper-2387"></a>

## Spatial Deconfounder: Interference-Aware Deconfounding for Spatial Causal Inference


### Images

![0722d249ca292c97dc9af397db3ebe39c81218e139322ae67e42ba79716c6b84.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/images/0722d249ca292c97dc9af397db3ebe39c81218e139322ae67e42ba79716c6b84.jpg)

![0cca51416c78e0a2316788db1ec9103a8698fd552f09661a9f18717443f7d463.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/images/0cca51416c78e0a2316788db1ec9103a8698fd552f09661a9f18717443f7d463.jpg)

![44c5f292f7b796d8a60c2c0a88c841b2066d36bbf753b1e93b8f428ae141fb1e.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/images/44c5f292f7b796d8a60c2c0a88c841b2066d36bbf753b1e93b8f428ae141fb1e.jpg)

![4ba0341fe7bda0e24f9f67a8fa920cd2d79aeae7ddff364e9def7c207ca4f025.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/images/4ba0341fe7bda0e24f9f67a8fa920cd2d79aeae7ddff364e9def7c207ca4f025.jpg)

![801e30dcaf84f920157ed7263e1174ba63639c80c6a9965adf8720e2d4f1a7ba.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/images/801e30dcaf84f920157ed7263e1174ba63639c80c6a9965adf8720e2d4f1a7ba.jpg)

![81beb767912b36f13ad4053d68e71db6ed7b51484cabbc371de2b21521804c06.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/images/81beb767912b36f13ad4053d68e71db6ed7b51484cabbc371de2b21521804c06.jpg)

![bcf63127e079cc222a2810a069bb45114f40e1382a8ae45e18d0feb41d0ba9fd.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/images/bcf63127e079cc222a2810a069bb45114f40e1382a8ae45e18d0feb41d0ba9fd.jpg)

![cfd6a8aa7fdc452c6f4346e9f0eebb40fdf70c0c11d8fb25606593c4991fbb37.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/images/cfd6a8aa7fdc452c6f4346e9f0eebb40fdf70c0c11d8fb25606593c4991fbb37.jpg)

### Tables

![0efa38a3f2a0201f39e4e45c6867d1f73f3c841b743452e866f606086eb30457.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/tables/0efa38a3f2a0201f39e4e45c6867d1f73f3c841b743452e866f606086eb30457.jpg)

![2a70c98c4b53e84f4e3d58119bea6a72cdbdca807779222c37bbc639e6bc22b0.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/tables/2a70c98c4b53e84f4e3d58119bea6a72cdbdca807779222c37bbc639e6bc22b0.jpg)

![2e709e4f7926267de4bd0d7776db845dd48fd3e5a48b48a5042adc88b3c8efab.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/tables/2e709e4f7926267de4bd0d7776db845dd48fd3e5a48b48a5042adc88b3c8efab.jpg)

![2e90361093a4c019b5b9b7d43f4607e19abde576b42c11a46aaef1d5bedd9502.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/tables/2e90361093a4c019b5b9b7d43f4607e19abde576b42c11a46aaef1d5bedd9502.jpg)

![558a1ed4221f6674c8341cb1f54f2f32a250230683c843be3020bc00171a9bb9.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/tables/558a1ed4221f6674c8341cb1f54f2f32a250230683c843be3020bc00171a9bb9.jpg)

![55cd4b724575a1be16a317e507cf6c6d6e3a2001eed67e32d2a7f424d653b5e2.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/tables/55cd4b724575a1be16a317e507cf6c6d6e3a2001eed67e32d2a7f424d653b5e2.jpg)

![67a6fd7fd620278a0eb7974496c938947e84684e238a400213cd8cb97006d6a4.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/tables/67a6fd7fd620278a0eb7974496c938947e84684e238a400213cd8cb97006d6a4.jpg)

![b2e22198a789b6c268278a0e58340f8443c9ea531599c4488b0b13e6fae4c573.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/tables/b2e22198a789b6c268278a0e58340f8443c9ea531599c4488b0b13e6fae4c573.jpg)

![d23485bfc8eebeb16d07a9e4f01b478ad073270898c4eb5fd20c5febb9884e38.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/tables/d23485bfc8eebeb16d07a9e4f01b478ad073270898c4eb5fd20c5febb9884e38.jpg)

![d693e8173e2e7afaf649eb82f4bda6c8c10b9d4b73f55ca8730c01343e143cb9.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/tables/d693e8173e2e7afaf649eb82f4bda6c8c10b9d4b73f55ca8730c01343e143cb9.jpg)

![d7c3e8b926e2be9183a7d733403e298149c95771ff4e9868864a84adcbd3dd01.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/tables/d7c3e8b926e2be9183a7d733403e298149c95771ff4e9868864a84adcbd3dd01.jpg)

![f61bf6d5e4d49e1d29115f4779ed5f37d8903915e927f5d27067d07b423e76ec.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/tables/f61bf6d5e4d49e1d29115f4779ed5f37d8903915e927f5d27067d07b423e76ec.jpg)

![f9d3ed98979217448dceab5aad97397b19e2d6c14ca53423f25787239b7d6ac2.jpg](../icml_results/2417_Spatial%20Deconfounder_%20Interference-Aware%20Deconfounding%20for%20Spatial%20Causal%20Inference/tables/f9d3ed98979217448dceab5aad97397b19e2d6c14ca53423f25787239b7d6ac2.jpg)

<a id="paper-2388"></a>

## Align Forward, Adapt Backward: Closing the Discretization Gap in Logic Gate Networks


### Images

![1e301e29450f7465736a96bfec2a03dd3a89fa3e7706736adf65256d5e819a10.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/images/1e301e29450f7465736a96bfec2a03dd3a89fa3e7706736adf65256d5e819a10.jpg)

![6a20828efb4cdddb601e85654fc3e4dda366eba96e022be71632f8fc21c96b55.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/images/6a20828efb4cdddb601e85654fc3e4dda366eba96e022be71632f8fc21c96b55.jpg)

![8916068fe7fc95d7780b1b31c057b18fe467928ef28c944a732086e0646960aa.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/images/8916068fe7fc95d7780b1b31c057b18fe467928ef28c944a732086e0646960aa.jpg)

![942dc3ae9c91129b2960edab667df6f94f2447a348f88ae2a569d6715a947474.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/images/942dc3ae9c91129b2960edab667df6f94f2447a348f88ae2a569d6715a947474.jpg)

![c479f8ff9fffd8abc313503099e0fe74555ecbe41c56f87806282518b433aaff.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/images/c479f8ff9fffd8abc313503099e0fe74555ecbe41c56f87806282518b433aaff.jpg)

### Tables

![0d9316b25e1989796f3255da5c83f5874c67ea6e648f6955ccce8ad223120e97.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/0d9316b25e1989796f3255da5c83f5874c67ea6e648f6955ccce8ad223120e97.jpg)

![0e480d3d1af9e1fac0f279602947a2b3ef462085ae47be6c69ceb6da4310acb0.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/0e480d3d1af9e1fac0f279602947a2b3ef462085ae47be6c69ceb6da4310acb0.jpg)

![1350838dfe2471a48bcce0ec98530eca93cb042b28bca9bcbba8a4faf4d3a147.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/1350838dfe2471a48bcce0ec98530eca93cb042b28bca9bcbba8a4faf4d3a147.jpg)

![1788a92b57d31b710d7b3961af4bfb872047589d1bdeefa2cc79331db698c133.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/1788a92b57d31b710d7b3961af4bfb872047589d1bdeefa2cc79331db698c133.jpg)

![1c2b8511abfac8dc9c3b9286b51d30bc5074a8f910c9830f2f3ed2058b367e5d.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/1c2b8511abfac8dc9c3b9286b51d30bc5074a8f910c9830f2f3ed2058b367e5d.jpg)

![288d99b242bf8ad03e658f299c257617dc6b7bd38a8db966b41c9754074bb6af.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/288d99b242bf8ad03e658f299c257617dc6b7bd38a8db966b41c9754074bb6af.jpg)

![590a694aea3978796199df05f7cf05223880f481aec54fca51fcc47b41c5a4e9.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/590a694aea3978796199df05f7cf05223880f481aec54fca51fcc47b41c5a4e9.jpg)

![5bbf868bca0faf4a150582c7a31121645db8a81d50d6179ff0f79ecd4caa557f.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/5bbf868bca0faf4a150582c7a31121645db8a81d50d6179ff0f79ecd4caa557f.jpg)

![6208c59d997eabbb53e46d72b00921d04e694a54d0bc69b8648d874606444aa1.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/6208c59d997eabbb53e46d72b00921d04e694a54d0bc69b8648d874606444aa1.jpg)

![69923d2cc48ad779a0b39b00482ecde61fde5d8d6b9741e4f29670e5c422cf51.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/69923d2cc48ad779a0b39b00482ecde61fde5d8d6b9741e4f29670e5c422cf51.jpg)

![7ea64a69f9dcaa302b6971f85ddaad6909eefd1066bcdc20bb9ad27fa1b691c9.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/7ea64a69f9dcaa302b6971f85ddaad6909eefd1066bcdc20bb9ad27fa1b691c9.jpg)

![8142bda66f0c6a917adc203eacbd12a686583986b52ebae635d4d708ec72f778.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/8142bda66f0c6a917adc203eacbd12a686583986b52ebae635d4d708ec72f778.jpg)

![92c535e4aa2de29395b68de4aa4eaa3789375dc24b538ec87f2106979699b3d6.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/92c535e4aa2de29395b68de4aa4eaa3789375dc24b538ec87f2106979699b3d6.jpg)

![9e4578c7db504d69f02a6ccee18286f9b061909bd08af3b8dc76c191c120d5a4.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/9e4578c7db504d69f02a6ccee18286f9b061909bd08af3b8dc76c191c120d5a4.jpg)

![be99012b2e7a98e3a3f6937da4f86c4f37161dad980b2689623c633c5da41e45.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/be99012b2e7a98e3a3f6937da4f86c4f37161dad980b2689623c633c5da41e45.jpg)

![c9acf5980b504e4bc842388f3e8edc3e48aa141e4b7302ac25028d2cf5fac548.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/c9acf5980b504e4bc842388f3e8edc3e48aa141e4b7302ac25028d2cf5fac548.jpg)

![d8041f1b087ac081512d35db60bd5f2224c3f5bc682650e838acee9769948896.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/d8041f1b087ac081512d35db60bd5f2224c3f5bc682650e838acee9769948896.jpg)

![f09c05a33a2f0c508ba01d4cd3a6bcf0857b8f29f18a8f9614c85ef6b2e258c9.jpg](../icml_results/2418_Align%20Forward%2C%20Adapt%20Backward_%20Closing%20the%20Discretization%20Gap%20in%20Logic%20Gate%20Networks/tables/f09c05a33a2f0c508ba01d4cd3a6bcf0857b8f29f18a8f9614c85ef6b2e258c9.jpg)

<a id="paper-2389"></a>

## Continuous Diffusion Models Can Obey Formal Syntax


### Images

![44717716db3451bcafe0878691086a72628f4074e1626514be8b628cb0461082.jpg](../icml_results/2419_Continuous%20Diffusion%20Models%20Can%20Obey%20Formal%20Syntax/images/44717716db3451bcafe0878691086a72628f4074e1626514be8b628cb0461082.jpg)

![81cf6805bd2515863dcc81e60093d02dc419e11f2b1bdc85c67e7484931873bc.jpg](../icml_results/2419_Continuous%20Diffusion%20Models%20Can%20Obey%20Formal%20Syntax/images/81cf6805bd2515863dcc81e60093d02dc419e11f2b1bdc85c67e7484931873bc.jpg)

### Tables

![41f8264ea320ea30d5d19b37d69e974a4c4556fe36b8427a58c8532011a09148.jpg](../icml_results/2419_Continuous%20Diffusion%20Models%20Can%20Obey%20Formal%20Syntax/tables/41f8264ea320ea30d5d19b37d69e974a4c4556fe36b8427a58c8532011a09148.jpg)

![74c54d9efeb696f8170784ac6cc97ae5ac0416bab051972bffa84426d54cd109.jpg](../icml_results/2419_Continuous%20Diffusion%20Models%20Can%20Obey%20Formal%20Syntax/tables/74c54d9efeb696f8170784ac6cc97ae5ac0416bab051972bffa84426d54cd109.jpg)

![91f39bbfef5d5cfa7864f1fc1ea8e4744525c4317a9402c07046e38dc298a049.jpg](../icml_results/2419_Continuous%20Diffusion%20Models%20Can%20Obey%20Formal%20Syntax/tables/91f39bbfef5d5cfa7864f1fc1ea8e4744525c4317a9402c07046e38dc298a049.jpg)

![94eb92bbc0ee6275542b810dcc1d936161b5d5cd8a0b9adc51ec50027387d647.jpg](../icml_results/2419_Continuous%20Diffusion%20Models%20Can%20Obey%20Formal%20Syntax/tables/94eb92bbc0ee6275542b810dcc1d936161b5d5cd8a0b9adc51ec50027387d647.jpg)

![9936187e4d5bc67193a807f404ba8f98acdf9e7ebd66615873feb472ad85b9b0.jpg](../icml_results/2419_Continuous%20Diffusion%20Models%20Can%20Obey%20Formal%20Syntax/tables/9936187e4d5bc67193a807f404ba8f98acdf9e7ebd66615873feb472ad85b9b0.jpg)

![d702aa3b1b0df8c3651bfe96270fa45718d767304843cbd4c4c007c3c59e7639.jpg](../icml_results/2419_Continuous%20Diffusion%20Models%20Can%20Obey%20Formal%20Syntax/tables/d702aa3b1b0df8c3651bfe96270fa45718d767304843cbd4c4c007c3c59e7639.jpg)

<a id="paper-2390"></a>

## Explicit representation of germline and non-germline residues improves antibody language modeling


### Images

![0570303dfabbde063b2f4d12ae38e32a00613b11cd70a1aaa3f6046535bc3f6e.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/0570303dfabbde063b2f4d12ae38e32a00613b11cd70a1aaa3f6046535bc3f6e.jpg)

![0d2bfdaa94edfa962da0ac2d18f1d2010b3773d8966e70f339081059dae48504.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/0d2bfdaa94edfa962da0ac2d18f1d2010b3773d8966e70f339081059dae48504.jpg)

![345055b5fba06940f9890b3f0e023b3fa527be54ad9dd113310158d9898bbc06.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/345055b5fba06940f9890b3f0e023b3fa527be54ad9dd113310158d9898bbc06.jpg)

![3c916d98631bb1c7c72a9b9770af7844379eeda555a2b0aa926cc0f48e07505c.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/3c916d98631bb1c7c72a9b9770af7844379eeda555a2b0aa926cc0f48e07505c.jpg)

![4037c9b2ef0b72915e3c435e8f89a7f8680a11552c4f77803fb7b20d0214f00e.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/4037c9b2ef0b72915e3c435e8f89a7f8680a11552c4f77803fb7b20d0214f00e.jpg)

![56142ddc1545daf8087bcfe3e678940aa45045f83fe444ec8d69b56532e8f876.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/56142ddc1545daf8087bcfe3e678940aa45045f83fe444ec8d69b56532e8f876.jpg)

![58d898afaf849a31b12f9b3083182a03883f466a2e6debf74d54c5703bddd552.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/58d898afaf849a31b12f9b3083182a03883f466a2e6debf74d54c5703bddd552.jpg)

![60d54d269ce6835d0e31a2b64e561c4cf8315ae51b7214a6d04cea8459d89649.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/60d54d269ce6835d0e31a2b64e561c4cf8315ae51b7214a6d04cea8459d89649.jpg)

![6365b0cc7888401e7332797e65e04c791d46b0253b1bad176c2360d630a0a923.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/6365b0cc7888401e7332797e65e04c791d46b0253b1bad176c2360d630a0a923.jpg)

![6461a227b53a6b2678d9fb12e946d7dfb5f8305496945f2751eade91acdc21f5.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/6461a227b53a6b2678d9fb12e946d7dfb5f8305496945f2751eade91acdc21f5.jpg)

![7213f34fc5040f8c78c46b2cba606e9e726b36211a6e7b070145d1de5793c6df.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/7213f34fc5040f8c78c46b2cba606e9e726b36211a6e7b070145d1de5793c6df.jpg)

![74095f6b3478634e378f361e4444fd3626dd82edc8a689345cdf7e42d951a116.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/74095f6b3478634e378f361e4444fd3626dd82edc8a689345cdf7e42d951a116.jpg)

![777317925fffe697ba21737e4b25dce7a961b13b6caadbbbe58e8f36f2d886cc.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/777317925fffe697ba21737e4b25dce7a961b13b6caadbbbe58e8f36f2d886cc.jpg)

![9354a76f4f624dc746e156020d66df619b3fba7c5b3d8ab67ffec4dabd8138a1.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/9354a76f4f624dc746e156020d66df619b3fba7c5b3d8ab67ffec4dabd8138a1.jpg)

![9749c94bfc8bbfe1d555558f518d0cf6fe419919227af529af480e5bfe9ce3f6.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/9749c94bfc8bbfe1d555558f518d0cf6fe419919227af529af480e5bfe9ce3f6.jpg)

![a440c875eecac143021c7175b4ecd2a9823826fc3e59530655f5500f9974e31b.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/a440c875eecac143021c7175b4ecd2a9823826fc3e59530655f5500f9974e31b.jpg)

![a6dd40ffceab7de1535ed9f4142d8c0bd600f000aa565fa17fb8fee5f30c86ea.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/a6dd40ffceab7de1535ed9f4142d8c0bd600f000aa565fa17fb8fee5f30c86ea.jpg)

![acd8248f7e4303bb6d986518ca2817237113ea590c628f5d96ae54256d6fbd4e.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/acd8248f7e4303bb6d986518ca2817237113ea590c628f5d96ae54256d6fbd4e.jpg)

![b9fcea0f935171938fa03a2fa44c13f2391016266e037602efe173a502525a36.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/b9fcea0f935171938fa03a2fa44c13f2391016266e037602efe173a502525a36.jpg)

![cfa76451a4d77e32a5838dfec71bbbd89ee1f126cc62161da408d809f7cc550e.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/cfa76451a4d77e32a5838dfec71bbbd89ee1f126cc62161da408d809f7cc550e.jpg)

![d5a321a6f09adef823e28a099c90c2d7caa26948478238aa5866d8fed1e779fb.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/d5a321a6f09adef823e28a099c90c2d7caa26948478238aa5866d8fed1e779fb.jpg)

![e08df6e972ef2aa40ecca46f420d139a2260b08e1b4f93267db20a31c2fe4277.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/e08df6e972ef2aa40ecca46f420d139a2260b08e1b4f93267db20a31c2fe4277.jpg)

![e8dc6c1c9c7f1332c701926192f90b1d8834291efef00e638a692736af0bab22.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/images/e8dc6c1c9c7f1332c701926192f90b1d8834291efef00e638a692736af0bab22.jpg)

### Tables

![0a5d011a077b7b94baeb60b72eeaa6ad96a08cee75bdcb3fd0a51bd0431a8004.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/tables/0a5d011a077b7b94baeb60b72eeaa6ad96a08cee75bdcb3fd0a51bd0431a8004.jpg)

![125fc71287f20a6e37617eaa5a214aa6ed6b51a95c04871c16439db582f85753.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/tables/125fc71287f20a6e37617eaa5a214aa6ed6b51a95c04871c16439db582f85753.jpg)

![1a12f2f1e8f7e5c1073a35ae63d16cb3c8b7fe09bd9dd896f4d36a8061ac5635.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/tables/1a12f2f1e8f7e5c1073a35ae63d16cb3c8b7fe09bd9dd896f4d36a8061ac5635.jpg)

![27f919547c7a238902b15dbc04b3042e82ff7cd8651bd756208eec9869a99d4b.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/tables/27f919547c7a238902b15dbc04b3042e82ff7cd8651bd756208eec9869a99d4b.jpg)

![2fef0cbc8f41a71cb979d435d559a0ae20920ae8e60058a093680524b5e6e61a.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/tables/2fef0cbc8f41a71cb979d435d559a0ae20920ae8e60058a093680524b5e6e61a.jpg)

![4cd9a08862e6d5d25a6cad194f4f04912c189b936cb3863d42b645462f67e029.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/tables/4cd9a08862e6d5d25a6cad194f4f04912c189b936cb3863d42b645462f67e029.jpg)

![595e15e7c68715b8fa2d0ca893d5d5f635db74e544ac3a40294927247c7a3d92.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/tables/595e15e7c68715b8fa2d0ca893d5d5f635db74e544ac3a40294927247c7a3d92.jpg)

![66e4b6448fb9dff54d4d610df8fdba27a6af0b0526f23e5b250b70decf13175b.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/tables/66e4b6448fb9dff54d4d610df8fdba27a6af0b0526f23e5b250b70decf13175b.jpg)

![a09cc8345b41c5ea4114395abdd065ffca5396c1f4e5c5cdf4c7e98e769db4f4.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/tables/a09cc8345b41c5ea4114395abdd065ffca5396c1f4e5c5cdf4c7e98e769db4f4.jpg)

![bb3918466e1e64cf4a8f85976edef5ec95957d2caa577b44f4abeb216d20c60d.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/tables/bb3918466e1e64cf4a8f85976edef5ec95957d2caa577b44f4abeb216d20c60d.jpg)

![bcf060b84b6ee7cd90b1df5a19df74a6997b78f954a5de5208a388d299080d2d.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/tables/bcf060b84b6ee7cd90b1df5a19df74a6997b78f954a5de5208a388d299080d2d.jpg)

![c1d79e730b6c63d004adbf0e0c7cc6a5ffe91aa3a30059f5f7c129a319fc3514.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/tables/c1d79e730b6c63d004adbf0e0c7cc6a5ffe91aa3a30059f5f7c129a319fc3514.jpg)

![ce0ad5f442dc30c86c863e7ab7f0e85517ad02176d3f85c6acc21704acbec3a6.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/tables/ce0ad5f442dc30c86c863e7ab7f0e85517ad02176d3f85c6acc21704acbec3a6.jpg)

![d5d2ee3c563021fdfcaf7aa0b74957260629fa1dd86ec1c0e57a54778dd95b52.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/tables/d5d2ee3c563021fdfcaf7aa0b74957260629fa1dd86ec1c0e57a54778dd95b52.jpg)

![d828219c98e71780791b0c5be85a0ab57c55c9b25dcdbcec7d132d3ebba22ff2.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/tables/d828219c98e71780791b0c5be85a0ab57c55c9b25dcdbcec7d132d3ebba22ff2.jpg)

![dbdc2638af9046349d9bc9ba0769c5a13f5409f34a4fa31269fa1f4f1811cfe8.jpg](../icml_results/2420_Explicit%20representation%20of%20germline%20and%20non-germline%20residues%20improves%20antibody%20language%20modeling/tables/dbdc2638af9046349d9bc9ba0769c5a13f5409f34a4fa31269fa1f4f1811cfe8.jpg)

<a id="paper-2391"></a>

## Rule2DRC: Benchmarking LLM Agents for DRC Script Synthesis with Execution-Guided Test Generation


### Images

![0971de2271608d927a4e1d32d8dde28a3e4ea25d96563d8fe879e880fbb01a66.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/images/0971de2271608d927a4e1d32d8dde28a3e4ea25d96563d8fe879e880fbb01a66.jpg)

![4f58e92e0379600719fc396ba0a3d9957f5df8cbe596c2fc477554d0c3ef8062.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/images/4f58e92e0379600719fc396ba0a3d9957f5df8cbe596c2fc477554d0c3ef8062.jpg)

![b07c33b768079993e5f7d3feb09c73e5379fe0a7c1126ba5d737faddb738e225.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/images/b07c33b768079993e5f7d3feb09c73e5379fe0a7c1126ba5d737faddb738e225.jpg)

![b764d82c29c17d99150877cb0f6720fe5c476691a400aa10f2cb38993e516a09.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/images/b764d82c29c17d99150877cb0f6720fe5c476691a400aa10f2cb38993e516a09.jpg)

![d19cd87ba0c8c51d6a340b0f14f1605bfd312484567315c31cb9ac14db3125a1.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/images/d19cd87ba0c8c51d6a340b0f14f1605bfd312484567315c31cb9ac14db3125a1.jpg)

![f0a41d8c8f193a7b87be65b73e422c0854e3ed54233bcfb4c69f3adbabd208a7.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/images/f0a41d8c8f193a7b87be65b73e422c0854e3ed54233bcfb4c69f3adbabd208a7.jpg)

![f0b54a46f84561efed7c04735591cf58d9bda04ab1f793b2875cbdb97bcfd330.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/images/f0b54a46f84561efed7c04735591cf58d9bda04ab1f793b2875cbdb97bcfd330.jpg)

![f10a26a6b040e3bb5e578df58cfb4371f50d4828cfdbc85986683e0d3b52fde5.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/images/f10a26a6b040e3bb5e578df58cfb4371f50d4828cfdbc85986683e0d3b52fde5.jpg)

### Tables

![3a34531cbfad6281327ca287a40b5f147599c1b83a331d13016f0954f9a14df1.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/tables/3a34531cbfad6281327ca287a40b5f147599c1b83a331d13016f0954f9a14df1.jpg)

![48f508b260fdde2432360295d906bb98a160f25a77cf097b0ec7bb6cce834234.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/tables/48f508b260fdde2432360295d906bb98a160f25a77cf097b0ec7bb6cce834234.jpg)

![52e926cacbadc52edba79cad0f7594cec7fb3b594281f4d9c0480065ec96f510.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/tables/52e926cacbadc52edba79cad0f7594cec7fb3b594281f4d9c0480065ec96f510.jpg)

![533dc07088c7cc879b0d349a1f821dacfacb3d57acb52b5bdbe54e240373e2ec.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/tables/533dc07088c7cc879b0d349a1f821dacfacb3d57acb52b5bdbe54e240373e2ec.jpg)

![60e26525426882fe241b2d568639b74da1156ba1db89eec22eca25707a65300b.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/tables/60e26525426882fe241b2d568639b74da1156ba1db89eec22eca25707a65300b.jpg)

![6224a07a64a2272d38ed25ee499d8712c0647edbc6a36a2d6a111be530a07fb6.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/tables/6224a07a64a2272d38ed25ee499d8712c0647edbc6a36a2d6a111be530a07fb6.jpg)

![a10f2c2b709e38b14905fb03faac3b2d002cf2cfa5fd2ad45f3119c356eac35c.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/tables/a10f2c2b709e38b14905fb03faac3b2d002cf2cfa5fd2ad45f3119c356eac35c.jpg)

![a6405188eeff20930b04ab7dd7e1c0879bd40b312a65a49a1c6cd1fd1454ec31.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/tables/a6405188eeff20930b04ab7dd7e1c0879bd40b312a65a49a1c6cd1fd1454ec31.jpg)

![d93e8dcef0e5d376dce83319760a16af61fc78a2c7af62be689d75a2c3d7f066.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/tables/d93e8dcef0e5d376dce83319760a16af61fc78a2c7af62be689d75a2c3d7f066.jpg)

![ede8e2dec25db201287327da27f725b564d3c12a88aace5759f58428c2a51e30.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/tables/ede8e2dec25db201287327da27f725b564d3c12a88aace5759f58428c2a51e30.jpg)

![f070e5bc8602fb0e74d83397d9e82fe61a9929ec10b837ce863bc9e33fd45b0e.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/tables/f070e5bc8602fb0e74d83397d9e82fe61a9929ec10b837ce863bc9e33fd45b0e.jpg)

![fe71e97d8cff95cae4978384e976d8645895ecb6b07d194d9263428263cf686c.jpg](../icml_results/2421_Rule2DRC_%20Benchmarking%20LLM%20Agents%20for%20DRC%20Script%20Synthesis%20with%20Execution-Guided%20Test%20Generation/tables/fe71e97d8cff95cae4978384e976d8645895ecb6b07d194d9263428263cf686c.jpg)

<a id="paper-2392"></a>

## Align Your Trajectory Tangent: Training Better Consistency Models via Manifold-Aligned Tangents


### Images

![013fe6c9d654ca1f9dfb7a41e673d0357812680bf6a4990a805ddadd037a7b7f.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/images/013fe6c9d654ca1f9dfb7a41e673d0357812680bf6a4990a805ddadd037a7b7f.jpg)

![0dd08d94db854776557ceccefd8b7ba29c70b0411ab82008d75d339a8f69ad59.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/images/0dd08d94db854776557ceccefd8b7ba29c70b0411ab82008d75d339a8f69ad59.jpg)

![1eb3a34fb8c7f9d172f388abb9ca131ade63f91c872f136175d3ecf407e8d1d4.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/images/1eb3a34fb8c7f9d172f388abb9ca131ade63f91c872f136175d3ecf407e8d1d4.jpg)

![286ad12083234d37c59fadd63edf53fa2637b357a4565b55583b32425f8cae83.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/images/286ad12083234d37c59fadd63edf53fa2637b357a4565b55583b32425f8cae83.jpg)

![3d5e19ba963b762a2d0a6b9a675ac5d4464c518ca7105acc6dfd9c0dea081ebc.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/images/3d5e19ba963b762a2d0a6b9a675ac5d4464c518ca7105acc6dfd9c0dea081ebc.jpg)

![462fc8ea75f6694a446601403b6eaf7014298f0378d75fb9a2a6d0c391d7d200.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/images/462fc8ea75f6694a446601403b6eaf7014298f0378d75fb9a2a6d0c391d7d200.jpg)

![47a3bbf6fa6ee23e418756f06c877528d7a5e669b1bc4ac17eb590b00feebe07.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/images/47a3bbf6fa6ee23e418756f06c877528d7a5e669b1bc4ac17eb590b00feebe07.jpg)

![63a8ac3aeee28ee4555d3fd2f8263cb360e7e6fda25d252b381aecdcc2e7feec.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/images/63a8ac3aeee28ee4555d3fd2f8263cb360e7e6fda25d252b381aecdcc2e7feec.jpg)

![65b27bc164ebfdcbd8839d926e4e5f8f434d9a7157862c483306444cc38f9ce5.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/images/65b27bc164ebfdcbd8839d926e4e5f8f434d9a7157862c483306444cc38f9ce5.jpg)

![6b5ba06300e818c8127ed0e75483575e62052d152bd16d04e8a1976d7fa6a187.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/images/6b5ba06300e818c8127ed0e75483575e62052d152bd16d04e8a1976d7fa6a187.jpg)

![818736a169eea62075d38cc49c3adda42635ff9759c072282e9df0525ed8c5dc.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/images/818736a169eea62075d38cc49c3adda42635ff9759c072282e9df0525ed8c5dc.jpg)

![a0c0c5a792ba44997fad7f51023d2c4bbaf26ad023007734ecf1e89fccb17391.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/images/a0c0c5a792ba44997fad7f51023d2c4bbaf26ad023007734ecf1e89fccb17391.jpg)

![afca9d25646f8ffc2ff44734d0bf7f0d9c927a0b5bad3102e2068e4b864e72ea.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/images/afca9d25646f8ffc2ff44734d0bf7f0d9c927a0b5bad3102e2068e4b864e72ea.jpg)

![bc848661b596ae27dd818cadff6646cce5d1127ca9b062c713d144a6edbe755e.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/images/bc848661b596ae27dd818cadff6646cce5d1127ca9b062c713d144a6edbe755e.jpg)

![cf9d2df127d0dbbc4fa15acfc9bdb38a299b8aea5f6fc5479e6a0e593d9b7cd6.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/images/cf9d2df127d0dbbc4fa15acfc9bdb38a299b8aea5f6fc5479e6a0e593d9b7cd6.jpg)

![d45f518cfd67057d5a15f8eee2fd2fc1191312cb8838f1abe7e34bdef441ba43.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/images/d45f518cfd67057d5a15f8eee2fd2fc1191312cb8838f1abe7e34bdef441ba43.jpg)

### Tables

![1789c8ea1f0d78c1077dc4fd096ae6abd1f4f5c11f323a88453eb8f52c5f7f26.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/tables/1789c8ea1f0d78c1077dc4fd096ae6abd1f4f5c11f323a88453eb8f52c5f7f26.jpg)

![1b7d4c518e219a68e177a6af6f41eb39407d1d0c92eb31613dfff71bfd713a34.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/tables/1b7d4c518e219a68e177a6af6f41eb39407d1d0c92eb31613dfff71bfd713a34.jpg)

![2a54332c586830eccdbfd8fd2fe39184d707421410c886d8e0d8698e52369fe0.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/tables/2a54332c586830eccdbfd8fd2fe39184d707421410c886d8e0d8698e52369fe0.jpg)

![5caaa023255e12981e127a3faee99d73e9a70ebe22c4ff80a8d407e2f964507d.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/tables/5caaa023255e12981e127a3faee99d73e9a70ebe22c4ff80a8d407e2f964507d.jpg)

![65b8a91776b9fac138ae89d3e4dd61339f0241685ece36458c937e38094ef534.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/tables/65b8a91776b9fac138ae89d3e4dd61339f0241685ece36458c937e38094ef534.jpg)

![90a4dade43c0714c6471caf8843fc025f0395a59b42591f16fde6f636a1f469b.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/tables/90a4dade43c0714c6471caf8843fc025f0395a59b42591f16fde6f636a1f469b.jpg)

![90abd0bb6dbf234de07d060b70b66fbe57e1362aaa0e0d33b6ff966206f8ea1d.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/tables/90abd0bb6dbf234de07d060b70b66fbe57e1362aaa0e0d33b6ff966206f8ea1d.jpg)

![e0f7e60d30011905b4830be38e9c5584f00391fed3d35f3acf43c3667ec72e43.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/tables/e0f7e60d30011905b4830be38e9c5584f00391fed3d35f3acf43c3667ec72e43.jpg)

![fb692f72e68345b7e93df342c6a6ad4a67e0cdc4357d00d1963ca01f6c00d731.jpg](../icml_results/2422_Align%20Your%20Trajectory%20Tangent_%20Training%20Better%20Consistency%20Models%20via%20Manifold-Aligned%20Tangents/tables/fb692f72e68345b7e93df342c6a6ad4a67e0cdc4357d00d1963ca01f6c00d731.jpg)

<a id="paper-2393"></a>

## Universal Reasoner: A Single, Composable Plug-and-Play Reasoner for Frozen LLMs


### Images

![05259f97304f259265b94e29b9215ee1ebc2c0478e26ee3d8b9d62c5abccc5a1.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/images/05259f97304f259265b94e29b9215ee1ebc2c0478e26ee3d8b9d62c5abccc5a1.jpg)

![0908307f8846cbd7b2a7328e78656b71095f8a297ebb7dcaa4fee4efa69a42e1.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/images/0908307f8846cbd7b2a7328e78656b71095f8a297ebb7dcaa4fee4efa69a42e1.jpg)

![210514a2b7b13f107fb0d32d41f49948991715eddefc030dea59616109528718.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/images/210514a2b7b13f107fb0d32d41f49948991715eddefc030dea59616109528718.jpg)

![373430ac5358a14702c28b80289aabb357f66a60b5a76e26eed304baa80611f0.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/images/373430ac5358a14702c28b80289aabb357f66a60b5a76e26eed304baa80611f0.jpg)

![53d2ab65a8243f1786818c8f1e0382d5556f930fd0af77169211ee9d0e31f4b2.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/images/53d2ab65a8243f1786818c8f1e0382d5556f930fd0af77169211ee9d0e31f4b2.jpg)

![62f00ac1d5a9b6d0d6ea39b2f5e2d106006e2a25b0834622dd3d1757805823a4.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/images/62f00ac1d5a9b6d0d6ea39b2f5e2d106006e2a25b0834622dd3d1757805823a4.jpg)

![77ec11939abc5ea575b969fb5771bdf87103c1bf84fc28b9ba55afc4d633f150.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/images/77ec11939abc5ea575b969fb5771bdf87103c1bf84fc28b9ba55afc4d633f150.jpg)

![837953826037cbdcf6dbf50112b3018fe6ddff4406a58e64b9ab2abd7c76dcfb.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/images/837953826037cbdcf6dbf50112b3018fe6ddff4406a58e64b9ab2abd7c76dcfb.jpg)

![9f5193a44d67d057babce4f836bf71ada2e0a5beb032c5e9beb621ba8218a161.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/images/9f5193a44d67d057babce4f836bf71ada2e0a5beb032c5e9beb621ba8218a161.jpg)

![ca416d5ca337ec39389b1a9b0272c6bdf18a82fd435ec2552298d2a63357b1a8.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/images/ca416d5ca337ec39389b1a9b0272c6bdf18a82fd435ec2552298d2a63357b1a8.jpg)

![d8aaea61ae0ec4f8fb83d45a4fed7e78b494c26b7934b2283589847f34cc4030.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/images/d8aaea61ae0ec4f8fb83d45a4fed7e78b494c26b7934b2283589847f34cc4030.jpg)

![dbbac6ec0ff03ebc982b1a6d8069eab2d68a8bb0706a72d2089dce48bfdb6b26.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/images/dbbac6ec0ff03ebc982b1a6d8069eab2d68a8bb0706a72d2089dce48bfdb6b26.jpg)

![f59cc07c5e845f4c35cb3992b8f7157ea34b58d949a5b385594bd3cf715c46ad.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/images/f59cc07c5e845f4c35cb3992b8f7157ea34b58d949a5b385594bd3cf715c46ad.jpg)

### Tables

![162bd3b94c1736c34f7cc862eac5d3a981039249213c1adc78802944952e77cf.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/162bd3b94c1736c34f7cc862eac5d3a981039249213c1adc78802944952e77cf.jpg)

![1e5e56e90418c2c75ddbec7ad09b9e4a3d79b07f8372d0694c367267c50dde4b.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/1e5e56e90418c2c75ddbec7ad09b9e4a3d79b07f8372d0694c367267c50dde4b.jpg)

![1f8d7abbfc273c856d03f65ae554ca3f0aa001dff514f779d839b04c9c09404c.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/1f8d7abbfc273c856d03f65ae554ca3f0aa001dff514f779d839b04c9c09404c.jpg)

![25a7434676d8d3ef61d70c8aa59fc8da04c914e23aa88cc267ad61703ccd0de5.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/25a7434676d8d3ef61d70c8aa59fc8da04c914e23aa88cc267ad61703ccd0de5.jpg)

![401982a23e7ad240a127bf5c159b573d56d38b05cb2e705289bc82f4fbf8c474.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/401982a23e7ad240a127bf5c159b573d56d38b05cb2e705289bc82f4fbf8c474.jpg)

![41a27078f16b875532fd5531d56e9f30b59844bdf9b5ac6e05bb8e8f9ac5f77f.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/41a27078f16b875532fd5531d56e9f30b59844bdf9b5ac6e05bb8e8f9ac5f77f.jpg)

![677614170d51b353d959f6a21b69f758c4732cf8311e946101b6b17952611430.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/677614170d51b353d959f6a21b69f758c4732cf8311e946101b6b17952611430.jpg)

![710b98ab4040c0905b010910970e735757da47e483cbeaf3ce088e8c931945e3.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/710b98ab4040c0905b010910970e735757da47e483cbeaf3ce088e8c931945e3.jpg)

![86f2dbc4320cc15d9f065da7f53a623602b6f5c1cbbe84807e552f383346d562.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/86f2dbc4320cc15d9f065da7f53a623602b6f5c1cbbe84807e552f383346d562.jpg)

![93406002c0477111b269057db4dd35380d335cb7322f44744ac8641361e7ced2.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/93406002c0477111b269057db4dd35380d335cb7322f44744ac8641361e7ced2.jpg)

![a7c71a3860ef4a78e028c8a8338091dd902728d30ec85bb71c0e8a2b2cb1b95f.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/a7c71a3860ef4a78e028c8a8338091dd902728d30ec85bb71c0e8a2b2cb1b95f.jpg)

![b7119f0f8e99f84a336734959da69e0337ceff2d49d25aa176ca36057467a47a.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/b7119f0f8e99f84a336734959da69e0337ceff2d49d25aa176ca36057467a47a.jpg)

![c1fc685821137b5f97349bbecff13868e3384697c4c7cad13dbcd454e5100dd1.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/c1fc685821137b5f97349bbecff13868e3384697c4c7cad13dbcd454e5100dd1.jpg)

![dc4ff2e6afbe9a7f8174b3e8c3fb4d3ee8c3f0da487182e522829e6fb7c62039.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/dc4ff2e6afbe9a7f8174b3e8c3fb4d3ee8c3f0da487182e522829e6fb7c62039.jpg)

![dd5f35a219d103d83ba7b339d71cb43be8eb223551b8feb413f9def91200359d.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/dd5f35a219d103d83ba7b339d71cb43be8eb223551b8feb413f9def91200359d.jpg)

![ddd83a491e335584e327d6c8abfb00e7ed4248d8618ea4b760b07c2433e37a44.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/ddd83a491e335584e327d6c8abfb00e7ed4248d8618ea4b760b07c2433e37a44.jpg)

![decff793ca258e2eb60fdaefb4fadc91bc530965eafc6a3303eb71f54f8b52bf.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/decff793ca258e2eb60fdaefb4fadc91bc530965eafc6a3303eb71f54f8b52bf.jpg)

![f842fe4ec98f3a4acd322c80aa0fbea2fafd63054468a4952a2524c1fd91e2a4.jpg](../icml_results/2423_Universal%20Reasoner_%20A%20Single%2C%20Composable%20Plug-and-Play%20Reasoner%20for%20Frozen%20LLMs/tables/f842fe4ec98f3a4acd322c80aa0fbea2fafd63054468a4952a2524c1fd91e2a4.jpg)

<a id="paper-2394"></a>

## On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length


### Images

![017bce68553ad60138cfba37ceffcc6766f2c95de52ed9cb707174f0838cb4e7.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/017bce68553ad60138cfba37ceffcc6766f2c95de52ed9cb707174f0838cb4e7.jpg)

![24162909d8770251a31be9540f277997961ccfeaac7a8508d8f9eb3705cea0e9.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/24162909d8770251a31be9540f277997961ccfeaac7a8508d8f9eb3705cea0e9.jpg)

![2e9ebf6dde54fe9aecf59a8fa7553309b9c0f43282e070e330e1e96a72e42b59.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/2e9ebf6dde54fe9aecf59a8fa7553309b9c0f43282e070e330e1e96a72e42b59.jpg)

![31c6010ad586ff27db66d959dbf497880e306dacd2431bcac92216a959e8722b.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/31c6010ad586ff27db66d959dbf497880e306dacd2431bcac92216a959e8722b.jpg)

![43c942d7d488f7992f86b419cca2dd381ea92ab68bcd94db40e6f5ac718cc07c.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/43c942d7d488f7992f86b419cca2dd381ea92ab68bcd94db40e6f5ac718cc07c.jpg)

![475b1667044d620f572d1e6321fe626f0e873d4a5a1c434b652bafbb377c32d1.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/475b1667044d620f572d1e6321fe626f0e873d4a5a1c434b652bafbb377c32d1.jpg)

![494d902e5ad88b9f81c65c994d7a2fe82b31ff41e272a6734d1435d7dbc56704.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/494d902e5ad88b9f81c65c994d7a2fe82b31ff41e272a6734d1435d7dbc56704.jpg)

![4e99d2f10d8397b1da9b1e117cdd91e808d507e28243ab1c5198066ca177700a.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/4e99d2f10d8397b1da9b1e117cdd91e808d507e28243ab1c5198066ca177700a.jpg)

![78b5f05a98c455f9250099b888cb6735eca60b07b4ebf3aef61b27d9429a4234.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/78b5f05a98c455f9250099b888cb6735eca60b07b4ebf3aef61b27d9429a4234.jpg)

![82f0d69b0684a756222f9b309da6fb4ad2daaa7d6430613432f83276db556293.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/82f0d69b0684a756222f9b309da6fb4ad2daaa7d6430613432f83276db556293.jpg)

![897e0a8385c43b7d468cb14fcfc06f79e7bbd5898b0010e0ae6668b1a9f7d628.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/897e0a8385c43b7d468cb14fcfc06f79e7bbd5898b0010e0ae6668b1a9f7d628.jpg)

![8b68d1b5d8247762562592348bf0e1ac11784085aded83a7df0782618faa74da.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/8b68d1b5d8247762562592348bf0e1ac11784085aded83a7df0782618faa74da.jpg)

![8cd97bdf0e627b8bcb59f6024bc39b796844d7099a243298e665841d20ef33cd.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/8cd97bdf0e627b8bcb59f6024bc39b796844d7099a243298e665841d20ef33cd.jpg)

![ac32b9f83d021547d3975409bb40016e7cf21e7a8f976a81f47ea8870072d4cd.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/ac32b9f83d021547d3975409bb40016e7cf21e7a8f976a81f47ea8870072d4cd.jpg)

![b836dd813c2be3de3751ce9767936ba0aa015be044070c6605994f37b5cbe709.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/b836dd813c2be3de3751ce9767936ba0aa015be044070c6605994f37b5cbe709.jpg)

![d12578d7f062dddd2cdda95aad029ada2d7120dbd0b253050e138083d510f7ef.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/d12578d7f062dddd2cdda95aad029ada2d7120dbd0b253050e138083d510f7ef.jpg)

![d1806e31eb728fc02143131c7820cd300240d1ad08d34e9dabbc5c1d09a6a60a.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/d1806e31eb728fc02143131c7820cd300240d1ad08d34e9dabbc5c1d09a6a60a.jpg)

![d195543606d1ff2b1d25a5ac01221744b754a954a0318dbb5fb9a19b0ed26363.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/d195543606d1ff2b1d25a5ac01221744b754a954a0318dbb5fb9a19b0ed26363.jpg)

![d92b2eab01b00afe2b28b7b4ef28557f70517dfe0b3b26e029ab1695a34e6a13.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/d92b2eab01b00afe2b28b7b4ef28557f70517dfe0b3b26e029ab1695a34e6a13.jpg)

![f4ff71925dd76cea1d1beaa76c3d496d587c99aedb5631234a3ae5772cf2ac01.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/f4ff71925dd76cea1d1beaa76c3d496d587c99aedb5631234a3ae5772cf2ac01.jpg)

![fcb194370bc0759790976ee5357a0de37e14fecfb9566cea51a312f4158c9f1b.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/images/fcb194370bc0759790976ee5357a0de37e14fecfb9566cea51a312f4158c9f1b.jpg)

### Tables

![0588d58be6d85f8963dc6c939292b0a26792705f20e234e33d8e0e37cf286318.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/tables/0588d58be6d85f8963dc6c939292b0a26792705f20e234e33d8e0e37cf286318.jpg)

![43050b79bacaa084dead68a1745d02ab01ae48c02421e2f401857a8a2fcf3932.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/tables/43050b79bacaa084dead68a1745d02ab01ae48c02421e2f401857a8a2fcf3932.jpg)

![7ae8f71fe71ddcad0a2a3f605f4f4d134ea4b538d091b44d2ebd8536d92b22fd.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/tables/7ae8f71fe71ddcad0a2a3f605f4f4d134ea4b538d091b44d2ebd8536d92b22fd.jpg)

![8a8d22b753fc8497bad665da0cf859d3e06a9635936e45d8e0b8e0e9dee2e112.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/tables/8a8d22b753fc8497bad665da0cf859d3e06a9635936e45d8e0b8e0e9dee2e112.jpg)

![af6430629fb17c8c4e23691186f6e64fe369b982f0f740a9e5edfc8201dfc1fb.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/tables/af6430629fb17c8c4e23691186f6e64fe369b982f0f740a9e5edfc8201dfc1fb.jpg)

![b9be89b480ff9477c775ef563190d460b4dd38a19ebd426c4e00f9678b67563a.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/tables/b9be89b480ff9477c775ef563190d460b4dd38a19ebd426c4e00f9678b67563a.jpg)

![d21001df4f0474393191ab2ebda030de8d33cbf2526ab5fea8357daa87f91223.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/tables/d21001df4f0474393191ab2ebda030de8d33cbf2526ab5fea8357daa87f91223.jpg)

![ecee52d1a908c365193a1141beb6f652798ba5eba9377158af598687aca765a7.jpg](../icml_results/2424_On%20Training%20Large%20Language%20Models%20for%20Long-Horizon%20Tasks_%20An%20Empirical%20Study%20of%20Horizon%20Length/tables/ecee52d1a908c365193a1141beb6f652798ba5eba9377158af598687aca765a7.jpg)

<a id="paper-2395"></a>

## Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models


### Images

![02409245245cc0673991a4256fd841f05ee832baa55277491c41b32e101012ea.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/02409245245cc0673991a4256fd841f05ee832baa55277491c41b32e101012ea.jpg)

![11c3561b078939eb9ee77fb4a50b6038bfc41012d0643be937ee47b659413343.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/11c3561b078939eb9ee77fb4a50b6038bfc41012d0643be937ee47b659413343.jpg)

![15261981e4484378b113fe0f013f778267cfe31d26fb64fc17dc8cb57a1d60a9.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/15261981e4484378b113fe0f013f778267cfe31d26fb64fc17dc8cb57a1d60a9.jpg)

![17cd8b3902bb86922aa1724f628b2991bd7354fbfe814bbf9d3f321bab624a6d.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/17cd8b3902bb86922aa1724f628b2991bd7354fbfe814bbf9d3f321bab624a6d.jpg)

![2692407ab4e57894956dad05175a34f68f10ed94dbf8250045f237cd7483fde3.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/2692407ab4e57894956dad05175a34f68f10ed94dbf8250045f237cd7483fde3.jpg)

![4692712a96f22ff9660df04730ecf2224977517cd22e91640e5b79501be11af9.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/4692712a96f22ff9660df04730ecf2224977517cd22e91640e5b79501be11af9.jpg)

![71b35ea000fb580b93d1ea3b1855328904bad50fbf935d1b83b76cc4cf0906c4.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/71b35ea000fb580b93d1ea3b1855328904bad50fbf935d1b83b76cc4cf0906c4.jpg)

![80368b3f7f39c6c37c4e692025e01c0ee3c8b01d4883c28422ac08cc92061035.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/80368b3f7f39c6c37c4e692025e01c0ee3c8b01d4883c28422ac08cc92061035.jpg)

![8712893f78fcf63de838561afafece62ce723756b3627246418e4b9e2c0dfd1e.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/8712893f78fcf63de838561afafece62ce723756b3627246418e4b9e2c0dfd1e.jpg)

![9389ecc87243c219201258464992b4910d1b84299eca080025a5b9ff07b14e89.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/9389ecc87243c219201258464992b4910d1b84299eca080025a5b9ff07b14e89.jpg)

![96a9d856c8933ca0736525f14a60ce693ff696da59b5fa814259d2cc4acf35de.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/96a9d856c8933ca0736525f14a60ce693ff696da59b5fa814259d2cc4acf35de.jpg)

![9846e38e8820c92e6cfe9146007c8b0660884d4e250a68a6357a5e20085b759a.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/9846e38e8820c92e6cfe9146007c8b0660884d4e250a68a6357a5e20085b759a.jpg)

![a78aacd54648c4a66a5d3db66193aa2ce1208dec187a76325e87da71256cf451.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/a78aacd54648c4a66a5d3db66193aa2ce1208dec187a76325e87da71256cf451.jpg)

![ab8a5aea348f1d73b68b4d78c1a7b902fe850339ea49a220a77f070ecdad08cf.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/ab8a5aea348f1d73b68b4d78c1a7b902fe850339ea49a220a77f070ecdad08cf.jpg)

![bb8eceaf85fb9415a2196c93db4f181a78d156b3e80e63e01d3a63b9e502c556.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/bb8eceaf85fb9415a2196c93db4f181a78d156b3e80e63e01d3a63b9e502c556.jpg)

![bdaf857c145c9ac43bd55eca46cba6d5b287bb0e9d49692db6690e497432091a.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/bdaf857c145c9ac43bd55eca46cba6d5b287bb0e9d49692db6690e497432091a.jpg)

![e0a76dab70665e16f995985c75e9cc0d60fd7784e3a038179cd43f6d45d9ebab.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/e0a76dab70665e16f995985c75e9cc0d60fd7784e3a038179cd43f6d45d9ebab.jpg)

![e8edaffecabd329ee89283e2f10f67fc4f831cae7407c9d4f00b588a291608af.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/e8edaffecabd329ee89283e2f10f67fc4f831cae7407c9d4f00b588a291608af.jpg)

![ef12a4f208b03e93f213253e9400143cd685d25a7c0829a9a5be3d61b02b7896.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/images/ef12a4f208b03e93f213253e9400143cd685d25a7c0829a9a5be3d61b02b7896.jpg)

### Tables

![24c2cda3c361bcd380c748ed10e9f07b7b2b81e53bdee2888998644e98fe7bb6.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/tables/24c2cda3c361bcd380c748ed10e9f07b7b2b81e53bdee2888998644e98fe7bb6.jpg)

![3bbdcfed3f1ebbfc83a30147951b2ea108e3d53b9ac9ca713b38581f3c7e404a.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/tables/3bbdcfed3f1ebbfc83a30147951b2ea108e3d53b9ac9ca713b38581f3c7e404a.jpg)

![5bbe91a930df81ae5a49162b09f2f927db6569ea10262b190ad3f2ffb92d15dc.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/tables/5bbe91a930df81ae5a49162b09f2f927db6569ea10262b190ad3f2ffb92d15dc.jpg)

![843a164388d87063a8d5b609937dffee74f6efe244987501ceb95d868539d2e2.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/tables/843a164388d87063a8d5b609937dffee74f6efe244987501ceb95d868539d2e2.jpg)

![a669631433f815037b9609d8b9cedfe616e6acd6b00dc5375ccd5786a26d79a4.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/tables/a669631433f815037b9609d8b9cedfe616e6acd6b00dc5375ccd5786a26d79a4.jpg)

![c4b6bea0e9fd796fde53539da262e8d52b505c56be79f07e2b43860ee30ac099.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/tables/c4b6bea0e9fd796fde53539da262e8d52b505c56be79f07e2b43860ee30ac099.jpg)

![decc89b564e8ac6b92bc226a92a112112e5fdf7bd3e26da0f059115d6958f188.jpg](../icml_results/2425_Early%20Decisions%20Matter_%20Proximity%20Bias%20and%20Initial%20Trajectory%20Shaping%20in%20Non-Autoregressive%20Diffusio/tables/decc89b564e8ac6b92bc226a92a112112e5fdf7bd3e26da0f059115d6958f188.jpg)

<a id="paper-2396"></a>

## LeakGFN: Robust Molecular Generation in Generative Flow Networks via Flow Decomposition


### Images

![1ec1b405ac1adbe3b9817b4c6351c46bec7c8ad623a57b29d913ad9d462660af.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/images/1ec1b405ac1adbe3b9817b4c6351c46bec7c8ad623a57b29d913ad9d462660af.jpg)

![6aa897cba69f8d122334490f91f4878f680f670d0722bfb1487813f19a2bb88f.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/images/6aa897cba69f8d122334490f91f4878f680f670d0722bfb1487813f19a2bb88f.jpg)

![934d1471ab7309cdba67e2f5545bc295bd1d8e6e63fbeca086f80b54ccb1ab68.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/images/934d1471ab7309cdba67e2f5545bc295bd1d8e6e63fbeca086f80b54ccb1ab68.jpg)

![9906b18e8fb6ff73d0498adcc92682ee2b0941ac11aa507ca4aaf0cd4b270c9e.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/images/9906b18e8fb6ff73d0498adcc92682ee2b0941ac11aa507ca4aaf0cd4b270c9e.jpg)

![996b83d2d08cdfb81aafa0a99bf9865f6098dd92794cc71d5cd0425eb894504e.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/images/996b83d2d08cdfb81aafa0a99bf9865f6098dd92794cc71d5cd0425eb894504e.jpg)

![e6dbc00e96b40126d047a195859792769fa78eb3f05bc4676f86130497ba47b2.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/images/e6dbc00e96b40126d047a195859792769fa78eb3f05bc4676f86130497ba47b2.jpg)

![e7dc190a7382a1f670ab705325295b675cd0b38993f8d8b4627a0503581c49a7.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/images/e7dc190a7382a1f670ab705325295b675cd0b38993f8d8b4627a0503581c49a7.jpg)

### Tables

![107005d67a115983014a2dca84432e1e3654b48a996862b6fcefeff7ee9401c8.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/tables/107005d67a115983014a2dca84432e1e3654b48a996862b6fcefeff7ee9401c8.jpg)

![11d6b252cda2b43aa8b143dba8d781cbde63bfe5c019d6cad9ebd47002c13cc4.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/tables/11d6b252cda2b43aa8b143dba8d781cbde63bfe5c019d6cad9ebd47002c13cc4.jpg)

![1c9a21b5574f440903f78b2de830632fa24b7c914f9c7736594110bbfc7cfefe.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/tables/1c9a21b5574f440903f78b2de830632fa24b7c914f9c7736594110bbfc7cfefe.jpg)

![41f7eb622b10765ca76645b53f81ae0c0e3e478ef19624a7ebee3bf54acf9d8c.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/tables/41f7eb622b10765ca76645b53f81ae0c0e3e478ef19624a7ebee3bf54acf9d8c.jpg)

![759333cafc9a37d2dcde6687956a6570964eaa4c01581c46e7b223a4ac189892.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/tables/759333cafc9a37d2dcde6687956a6570964eaa4c01581c46e7b223a4ac189892.jpg)

![7662b3a7208e7e0420b59ed1cb1489aca90a8293e3223646f0269bea4ddc5fce.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/tables/7662b3a7208e7e0420b59ed1cb1489aca90a8293e3223646f0269bea4ddc5fce.jpg)

![832d43a24241e8ae5195c00be3388efbb2cd3179b7ed4f0c1115980ee799526f.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/tables/832d43a24241e8ae5195c00be3388efbb2cd3179b7ed4f0c1115980ee799526f.jpg)

![9befe47ef0f39f4e5484ee643f132064080a4bd46b48ecf9a1d9370624448574.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/tables/9befe47ef0f39f4e5484ee643f132064080a4bd46b48ecf9a1d9370624448574.jpg)

![a5b1da7561d249a10ed5d779848ccb46a86994b9a8f5d164a5496950b9baa527.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/tables/a5b1da7561d249a10ed5d779848ccb46a86994b9a8f5d164a5496950b9baa527.jpg)

![a7764c5cbb1199b0d196db2cd7adfef97ab24a2a8c023c0a3c82ab93f2319d76.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/tables/a7764c5cbb1199b0d196db2cd7adfef97ab24a2a8c023c0a3c82ab93f2319d76.jpg)

![c6635ec35bf228f16ab93e6db28a4538e0ff082d0ed697b0ba52b94216cf3647.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/tables/c6635ec35bf228f16ab93e6db28a4538e0ff082d0ed697b0ba52b94216cf3647.jpg)

![ca596be165c3f8c4d717abd2bc9b2bd7a17a55e834d8f8674d217c28acc05e32.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/tables/ca596be165c3f8c4d717abd2bc9b2bd7a17a55e834d8f8674d217c28acc05e32.jpg)

![d192ebaa3cb05348faf3fb2ad67d52df3551f6efb721460f421f334d17606845.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/tables/d192ebaa3cb05348faf3fb2ad67d52df3551f6efb721460f421f334d17606845.jpg)

![e840bcbf079ce5de34092ea4adf872dd12703e624953bc612958109591ce8fdf.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/tables/e840bcbf079ce5de34092ea4adf872dd12703e624953bc612958109591ce8fdf.jpg)

![ef8985bfa8798d00b833b8f96a43eeeea42e3578751745081075febca28b790c.jpg](../icml_results/2426_LeakGFN_%20Robust%20Molecular%20Generation%20in%20Generative%20Flow%20Networks%20via%20Flow%20Decomposition/tables/ef8985bfa8798d00b833b8f96a43eeeea42e3578751745081075febca28b790c.jpg)

<a id="paper-2397"></a>

## Dissect and Prune: Enhancing Robustness in AI-Generated Image Detection


### Images

![0209e94e7e5e057d3151bc62a6da9fef6c9a904221c85254c001d30c4aaf28cd.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/images/0209e94e7e5e057d3151bc62a6da9fef6c9a904221c85254c001d30c4aaf28cd.jpg)

![0ae2d9149ffce1b7a90edc313a7e47ab5d597b8bf33014f2b3869a4658c6e7a0.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/images/0ae2d9149ffce1b7a90edc313a7e47ab5d597b8bf33014f2b3869a4658c6e7a0.jpg)

![1076290ab8eb9a8a2b029f19e404a86d960f784c88a330fd260370baec9f74ca.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/images/1076290ab8eb9a8a2b029f19e404a86d960f784c88a330fd260370baec9f74ca.jpg)

![185a62091de08a314f76c9b7ed11e972e68ab63e952f9507abc2f4af66c2a1cc.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/images/185a62091de08a314f76c9b7ed11e972e68ab63e952f9507abc2f4af66c2a1cc.jpg)

![596ff8bed1f03327d81e244de294b6406c67f264056c4c177488b05b3cb53c5c.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/images/596ff8bed1f03327d81e244de294b6406c67f264056c4c177488b05b3cb53c5c.jpg)

![5e34c2f63480e0a86686399dfdf6454556783b2dd2b92115b7f90e4ab311fb16.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/images/5e34c2f63480e0a86686399dfdf6454556783b2dd2b92115b7f90e4ab311fb16.jpg)

![b5ebb03017e39bcdad5481e8bfbbd48514097ca9087aa5d5c0be4a5072248d12.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/images/b5ebb03017e39bcdad5481e8bfbbd48514097ca9087aa5d5c0be4a5072248d12.jpg)

![fa0985e21a025537620959f978e3b0200b2107ad1342fe87d28130019907b7e0.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/images/fa0985e21a025537620959f978e3b0200b2107ad1342fe87d28130019907b7e0.jpg)

![ff202a0b34760b9c13b53ec48a09acda91b6dd78686a8243e006670774f138ac.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/images/ff202a0b34760b9c13b53ec48a09acda91b6dd78686a8243e006670774f138ac.jpg)

### Tables

![0ced50f39c20fae2843cb438181d8a9f9f7e1627f05fb9cc5cd8c7a32b053ee3.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/tables/0ced50f39c20fae2843cb438181d8a9f9f7e1627f05fb9cc5cd8c7a32b053ee3.jpg)

![35eb7f3ab6c88b3bca354ee11af1e76c65745c372e4cb3f88692ca4da3de49e3.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/tables/35eb7f3ab6c88b3bca354ee11af1e76c65745c372e4cb3f88692ca4da3de49e3.jpg)

![431ab72e7eb648239d37dc60c7850f0e00f852fc5d945c0be245ac7cd180661b.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/tables/431ab72e7eb648239d37dc60c7850f0e00f852fc5d945c0be245ac7cd180661b.jpg)

![83afe98bbce4ba80ff1b864e3143d7b989471ba7f5f15703eff709fc5d9569ba.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/tables/83afe98bbce4ba80ff1b864e3143d7b989471ba7f5f15703eff709fc5d9569ba.jpg)

![8987f6721ede8d2df4cb350730a191362ce15e772bb9b0592410c76d00152c1e.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/tables/8987f6721ede8d2df4cb350730a191362ce15e772bb9b0592410c76d00152c1e.jpg)

![8e6ff50c25c4ac63c28597a18b0b809223ee765b8ca441b59f8618e15bbe0989.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/tables/8e6ff50c25c4ac63c28597a18b0b809223ee765b8ca441b59f8618e15bbe0989.jpg)

![c631adf13417b4ac32ccc74f3ca113dd7382439929f4dc228c26e01bffca6e62.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/tables/c631adf13417b4ac32ccc74f3ca113dd7382439929f4dc228c26e01bffca6e62.jpg)

![d4236464cc4a56aa78d913b75f84bbd7c20d57584492a79026ed91c0cc30043c.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/tables/d4236464cc4a56aa78d913b75f84bbd7c20d57584492a79026ed91c0cc30043c.jpg)

![f8eb8572d4efb5f28ecf2454294feec5bf1a1583f8dbe3a31e62e08a367d8f77.jpg](../icml_results/2427_Dissect%20and%20Prune_%20Enhancing%20Robustness%20in%20AI-Generated%20Image%20Detection/tables/f8eb8572d4efb5f28ecf2454294feec5bf1a1583f8dbe3a31e62e08a367d8f77.jpg)

<a id="paper-2398"></a>

## Robust and Consistent Ski Rental with Distributional Advice


### Images

![adb1692664f86d783bddc0e954397903de8221d24ea3cb2c6f57b9869bedcca1.jpg](../icml_results/2428_Robust%20and%20Consistent%20Ski%20Rental%20with%20Distributional%20Advice/images/adb1692664f86d783bddc0e954397903de8221d24ea3cb2c6f57b9869bedcca1.jpg)

### Tables

![831e6243a4ccf2dd09824f33fb90c3ab01566002ec935720bc1f5ead31583b33.jpg](../icml_results/2428_Robust%20and%20Consistent%20Ski%20Rental%20with%20Distributional%20Advice/tables/831e6243a4ccf2dd09824f33fb90c3ab01566002ec935720bc1f5ead31583b33.jpg)

![c52e787926160faf1d928725206f98a3918bf485fc33a0688afd1cdca5c574c6.jpg](../icml_results/2428_Robust%20and%20Consistent%20Ski%20Rental%20with%20Distributional%20Advice/tables/c52e787926160faf1d928725206f98a3918bf485fc33a0688afd1cdca5c574c6.jpg)

<a id="paper-2399"></a>

## Stochastic Gradient Variational Inference with Price’s Gradient Estimator from Bures-Wasserstein to Parameter Space


### Images

![57c75cd20f2373aa6c0a3a0265b908b005b7b621c95061efec6eac90290a1ebd.jpg](../icml_results/2429_Stochastic%20Gradient%20Variational%20Inference%20with%20Price%E2%80%99s%20Gradient%20Estimator%20from%20Bures-Wasserstein%20to%20/images/57c75cd20f2373aa6c0a3a0265b908b005b7b621c95061efec6eac90290a1ebd.jpg)

![a411b8fe2a8ec1c2eef476932b0f8a55dbdff11a515e7c013b76979b78d9bed0.jpg](../icml_results/2429_Stochastic%20Gradient%20Variational%20Inference%20with%20Price%E2%80%99s%20Gradient%20Estimator%20from%20Bures-Wasserstein%20to%20/images/a411b8fe2a8ec1c2eef476932b0f8a55dbdff11a515e7c013b76979b78d9bed0.jpg)

![dee8ae9aa1a5ff32a7b7dbe6d68eeab3501d6d17c62609c9fd09759bd75e93a0.jpg](../icml_results/2429_Stochastic%20Gradient%20Variational%20Inference%20with%20Price%E2%80%99s%20Gradient%20Estimator%20from%20Bures-Wasserstein%20to%20/images/dee8ae9aa1a5ff32a7b7dbe6d68eeab3501d6d17c62609c9fd09759bd75e93a0.jpg)

![f06fd42b5b8ab296d91e0a3291e6428e7d028f08afe8fbc0457054257a01f087.jpg](../icml_results/2429_Stochastic%20Gradient%20Variational%20Inference%20with%20Price%E2%80%99s%20Gradient%20Estimator%20from%20Bures-Wasserstein%20to%20/images/f06fd42b5b8ab296d91e0a3291e6428e7d028f08afe8fbc0457054257a01f087.jpg)

### Tables

![a2faf8d8b947a5a7505af3b409393ce18e31fbeab61815f160ec993f6f4166b4.jpg](../icml_results/2429_Stochastic%20Gradient%20Variational%20Inference%20with%20Price%E2%80%99s%20Gradient%20Estimator%20from%20Bures-Wasserstein%20to%20/tables/a2faf8d8b947a5a7505af3b409393ce18e31fbeab61815f160ec993f6f4166b4.jpg)

![c3d7da3347d22d7fedd540769ce2965ee6ff335cbe851aa4c50fc096341edf97.jpg](../icml_results/2429_Stochastic%20Gradient%20Variational%20Inference%20with%20Price%E2%80%99s%20Gradient%20Estimator%20from%20Bures-Wasserstein%20to%20/tables/c3d7da3347d22d7fedd540769ce2965ee6ff335cbe851aa4c50fc096341edf97.jpg)

<a id="paper-2400"></a>

## Stop Training for the Worst: Progressive Unmasking Accelerates Masked Diffusion Training


### Images

![08a4db365f6cdb16dfb86470ef698615768c74e852b8811eb915d933a71f99da.jpg](../icml_results/2430_Stop%20Training%20for%20the%20Worst_%20Progressive%20Unmasking%20Accelerates%20Masked%20Diffusion%20Training/images/08a4db365f6cdb16dfb86470ef698615768c74e852b8811eb915d933a71f99da.jpg)

![09d7c6880365ea49d6c1df1f7b661b3b3001006151774f77b60443aa1415ccc2.jpg](../icml_results/2430_Stop%20Training%20for%20the%20Worst_%20Progressive%20Unmasking%20Accelerates%20Masked%20Diffusion%20Training/images/09d7c6880365ea49d6c1df1f7b661b3b3001006151774f77b60443aa1415ccc2.jpg)

![2f1d096480f2aa5efb18f6e083a3be399bc69a8ec66073bd107547fb3584e872.jpg](../icml_results/2430_Stop%20Training%20for%20the%20Worst_%20Progressive%20Unmasking%20Accelerates%20Masked%20Diffusion%20Training/images/2f1d096480f2aa5efb18f6e083a3be399bc69a8ec66073bd107547fb3584e872.jpg)

![45da194a38769b13cb82d3a7d9d31f9d516899df3f8fc32524c04ed7bab8d38b.jpg](../icml_results/2430_Stop%20Training%20for%20the%20Worst_%20Progressive%20Unmasking%20Accelerates%20Masked%20Diffusion%20Training/images/45da194a38769b13cb82d3a7d9d31f9d516899df3f8fc32524c04ed7bab8d38b.jpg)

![462f6880e645ac77524672709c9fa6500cb6fe3a9cde16b0dadb79e6bb3bb6f0.jpg](../icml_results/2430_Stop%20Training%20for%20the%20Worst_%20Progressive%20Unmasking%20Accelerates%20Masked%20Diffusion%20Training/images/462f6880e645ac77524672709c9fa6500cb6fe3a9cde16b0dadb79e6bb3bb6f0.jpg)

![4a5f5f9ef9e30e3ac8a79f592af1a5e8a39b4f6895a7e60d23690288ca0fb81b.jpg](../icml_results/2430_Stop%20Training%20for%20the%20Worst_%20Progressive%20Unmasking%20Accelerates%20Masked%20Diffusion%20Training/images/4a5f5f9ef9e30e3ac8a79f592af1a5e8a39b4f6895a7e60d23690288ca0fb81b.jpg)

![5cc1d93518421a9380833a2e87c3ac2ecc924fa5471f84503aa8f9a5f6b047d0.jpg](../icml_results/2430_Stop%20Training%20for%20the%20Worst_%20Progressive%20Unmasking%20Accelerates%20Masked%20Diffusion%20Training/images/5cc1d93518421a9380833a2e87c3ac2ecc924fa5471f84503aa8f9a5f6b047d0.jpg)

![63f1966a23e2f57c6d19bd9d89f966c9433b28436125f961f5dedc4de1830fb5.jpg](../icml_results/2430_Stop%20Training%20for%20the%20Worst_%20Progressive%20Unmasking%20Accelerates%20Masked%20Diffusion%20Training/images/63f1966a23e2f57c6d19bd9d89f966c9433b28436125f961f5dedc4de1830fb5.jpg)

![b375536518772e276ee6ad611d145de1ac42f50126fc08e76af296c5f035152c.jpg](../icml_results/2430_Stop%20Training%20for%20the%20Worst_%20Progressive%20Unmasking%20Accelerates%20Masked%20Diffusion%20Training/images/b375536518772e276ee6ad611d145de1ac42f50126fc08e76af296c5f035152c.jpg)

![e4d022a5de8842c8692d55554235d66529c0417d1207ddafa9915fb41f9c283b.jpg](../icml_results/2430_Stop%20Training%20for%20the%20Worst_%20Progressive%20Unmasking%20Accelerates%20Masked%20Diffusion%20Training/images/e4d022a5de8842c8692d55554235d66529c0417d1207ddafa9915fb41f9c283b.jpg)

![fababcf227e992f30e332eca1883380ce75fedc794af5fbba6ff0a1e45640725.jpg](../icml_results/2430_Stop%20Training%20for%20the%20Worst_%20Progressive%20Unmasking%20Accelerates%20Masked%20Diffusion%20Training/images/fababcf227e992f30e332eca1883380ce75fedc794af5fbba6ff0a1e45640725.jpg)

### Tables

![6e852f53dd8384ce2b8b5640c0b7201857b64f37186c36a986b26aa92d2622e2.jpg](../icml_results/2430_Stop%20Training%20for%20the%20Worst_%20Progressive%20Unmasking%20Accelerates%20Masked%20Diffusion%20Training/tables/6e852f53dd8384ce2b8b5640c0b7201857b64f37186c36a986b26aa92d2622e2.jpg)

![d85bacfde944782a883d2a3f328cecca097884bbacd68520ced4a0f92aecb754.jpg](../icml_results/2430_Stop%20Training%20for%20the%20Worst_%20Progressive%20Unmasking%20Accelerates%20Masked%20Diffusion%20Training/tables/d85bacfde944782a883d2a3f328cecca097884bbacd68520ced4a0f92aecb754.jpg)

<a id="paper-2401"></a>

## Equivariant Latent Alignment via Flow Matching under Group Symmetries


### Images

![059bff3741d5298f96f4719343b3a13335c89f97dbc699346fdd6e1a62315ad4.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/images/059bff3741d5298f96f4719343b3a13335c89f97dbc699346fdd6e1a62315ad4.jpg)

![2114e5aa797dbf8af8c7d13d73fef0eff3df17ff292dfdd9a41819d310f166ad.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/images/2114e5aa797dbf8af8c7d13d73fef0eff3df17ff292dfdd9a41819d310f166ad.jpg)

![2d929940c8aab88c376d3f3e84c8f19c0319cb26d58c4196304a7d682e703783.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/images/2d929940c8aab88c376d3f3e84c8f19c0319cb26d58c4196304a7d682e703783.jpg)

![4818d2c4ac0eb68325a65240917d8040cf459e9ef53e0134cbc968bb40bcbcaf.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/images/4818d2c4ac0eb68325a65240917d8040cf459e9ef53e0134cbc968bb40bcbcaf.jpg)

![4d8d96f281b91e826b4beea9f45a8f751fd994204ec2aeca4d9757d20d8ac595.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/images/4d8d96f281b91e826b4beea9f45a8f751fd994204ec2aeca4d9757d20d8ac595.jpg)

![4efb0a308c3638de56f66eeba7c2bbbec3bf68811ef28f4a11de7ef236ff59a2.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/images/4efb0a308c3638de56f66eeba7c2bbbec3bf68811ef28f4a11de7ef236ff59a2.jpg)

![70c683791b89b19a75a7b31e18be80b186d01409a3fcb21ec29fada47f60fdac.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/images/70c683791b89b19a75a7b31e18be80b186d01409a3fcb21ec29fada47f60fdac.jpg)

![840d56559a1db18a6465adbdeeb134c934cb205d8d8d695552606c8c392b4040.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/images/840d56559a1db18a6465adbdeeb134c934cb205d8d8d695552606c8c392b4040.jpg)

![8bd1dddfea33f75fb786d56e3bd3067e91ba4e2724f44ef708be7db27199231b.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/images/8bd1dddfea33f75fb786d56e3bd3067e91ba4e2724f44ef708be7db27199231b.jpg)

![be67b67ebb7003bb3050f9e678827c8147c899f3b48b858baff2a80789fd7bb6.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/images/be67b67ebb7003bb3050f9e678827c8147c899f3b48b858baff2a80789fd7bb6.jpg)

![ce13ca7f01341fb01b93dfc05cf8bceb43eea758875fb65d2b53c944da2f7d47.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/images/ce13ca7f01341fb01b93dfc05cf8bceb43eea758875fb65d2b53c944da2f7d47.jpg)

![db11bf4005cb202a12d08c3046c409a7685e0f7875d3eb4a7261b41cbfcbae3d.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/images/db11bf4005cb202a12d08c3046c409a7685e0f7875d3eb4a7261b41cbfcbae3d.jpg)

### Tables

![29b60f907de4ebf65b73d5caa671c082e867ff5b06b61ef2d6c67c82de50b47f.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/tables/29b60f907de4ebf65b73d5caa671c082e867ff5b06b61ef2d6c67c82de50b47f.jpg)

![2e144c0bd3e380e2a5867222e07de5b75b2c14b7c66ebe9c361c18f27cbc6f18.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/tables/2e144c0bd3e380e2a5867222e07de5b75b2c14b7c66ebe9c361c18f27cbc6f18.jpg)

![526b347adb1de2d832b96d5c7862af472f0e45b50ed1d3e35d1c4689a426f104.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/tables/526b347adb1de2d832b96d5c7862af472f0e45b50ed1d3e35d1c4689a426f104.jpg)

![889984a591ac0397b024dec9f3da36c6a1d8cd83d0acfd4c7883d2d0088558ec.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/tables/889984a591ac0397b024dec9f3da36c6a1d8cd83d0acfd4c7883d2d0088558ec.jpg)

![a2fcb3989026694414a4c7d62c5296dd3432a73546a95a5b39830a896eb09e9e.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/tables/a2fcb3989026694414a4c7d62c5296dd3432a73546a95a5b39830a896eb09e9e.jpg)

![adb643e0e799c19cefdf41b1f9d064937a382aa39363630a541d1e6ccb9c4b53.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/tables/adb643e0e799c19cefdf41b1f9d064937a382aa39363630a541d1e6ccb9c4b53.jpg)

![b5a3e3d7f8cf81778c21f472068e23b9d1a4e27e997542a1b54cc0d461a3298b.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/tables/b5a3e3d7f8cf81778c21f472068e23b9d1a4e27e997542a1b54cc0d461a3298b.jpg)

![e61da90b1cddce88e838842ff58b7fdd8844bda206d9b590a15d017bb4c1f996.jpg](../icml_results/2431_Equivariant%20Latent%20Alignment%20via%20Flow%20Matching%20under%20Group%20Symmetries/tables/e61da90b1cddce88e838842ff58b7fdd8844bda206d9b590a15d017bb4c1f996.jpg)

<a id="paper-2402"></a>

## Semi-Supervised Neural Super-Resolution for Mesh-Based Simulations


### Images

![02fa7448add9252bae18820ec534fd6977058a9dba77d789abafd9489e2d557b.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/02fa7448add9252bae18820ec534fd6977058a9dba77d789abafd9489e2d557b.jpg)

![0a158c8b7b53ec8c5756cf77d3175d48309652d8249b4c40bef0adb896ed45ef.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/0a158c8b7b53ec8c5756cf77d3175d48309652d8249b4c40bef0adb896ed45ef.jpg)

![210250c7e4b5f4f8e6bb921d1e29b02c53f148e491a510a5678ff4c1add0c95a.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/210250c7e4b5f4f8e6bb921d1e29b02c53f148e491a510a5678ff4c1add0c95a.jpg)

![27aa3d334c9120a6ef614f87ff920b9a9d6e419044df029e7222d41af6e3c2c1.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/27aa3d334c9120a6ef614f87ff920b9a9d6e419044df029e7222d41af6e3c2c1.jpg)

![355ba582912d0a61ccaf646964a6ba22c223c10063710de4b27cca85dd4b147c.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/355ba582912d0a61ccaf646964a6ba22c223c10063710de4b27cca85dd4b147c.jpg)

![3d86aac5f73c948e771a49318bd8a01e6adad61d11a6d3a17358c1c6d45e8ba7.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/3d86aac5f73c948e771a49318bd8a01e6adad61d11a6d3a17358c1c6d45e8ba7.jpg)

![4f787d9cca6bd0668337cb2b714bdb84c9dd58d14971e08e5b0471030d90fe16.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/4f787d9cca6bd0668337cb2b714bdb84c9dd58d14971e08e5b0471030d90fe16.jpg)

![5363f4ba6ae27477dd9fe11ce758463b1a3f89c2a7b22dbb01d7fa2a95271c3a.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/5363f4ba6ae27477dd9fe11ce758463b1a3f89c2a7b22dbb01d7fa2a95271c3a.jpg)

![550017aa04459595996204db9a03e936c90e95261b19094fa73a32b4968de946.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/550017aa04459595996204db9a03e936c90e95261b19094fa73a32b4968de946.jpg)

![5781a3c5ba1b1ee071ba9baa6c97ceb06d20360b500c8f9eb78ad670f682dc4a.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/5781a3c5ba1b1ee071ba9baa6c97ceb06d20360b500c8f9eb78ad670f682dc4a.jpg)

![5a0470a7b052727e7c4d0cc661316555e67bda9ae218520b6c1159d80cb9d27c.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/5a0470a7b052727e7c4d0cc661316555e67bda9ae218520b6c1159d80cb9d27c.jpg)

![5a3e13a4ae599c16e5dfceea5daf12be0f4e8e62f3cf7daf2235e13d794ace5e.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/5a3e13a4ae599c16e5dfceea5daf12be0f4e8e62f3cf7daf2235e13d794ace5e.jpg)

![5a8626f6576f9585e6963b1a0732f76f105b5a50eeeaa4da72eb9db0165079e4.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/5a8626f6576f9585e6963b1a0732f76f105b5a50eeeaa4da72eb9db0165079e4.jpg)

![7a497aabdb650c687e8f87de37981b94263d2fb4add2e90370624d097f8c3021.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/7a497aabdb650c687e8f87de37981b94263d2fb4add2e90370624d097f8c3021.jpg)

![7eb1e5d00eca41cf9d0ec019597e2a777001906eb9ad62d555bd1fe951bdd272.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/7eb1e5d00eca41cf9d0ec019597e2a777001906eb9ad62d555bd1fe951bdd272.jpg)

![8142831c8959ffef34dd68bcb941ac3a4206ec9e77acdc96c3787146a005e545.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/8142831c8959ffef34dd68bcb941ac3a4206ec9e77acdc96c3787146a005e545.jpg)

![876c3d5d0f774a5cb803fe0ff1873a4bf22a0a5b787e9d12c66598a14de29a4d.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/876c3d5d0f774a5cb803fe0ff1873a4bf22a0a5b787e9d12c66598a14de29a4d.jpg)

![8b851d0fb11e00b859be00610d2cd56a3cc0e392622584e4fec648571efee254.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/8b851d0fb11e00b859be00610d2cd56a3cc0e392622584e4fec648571efee254.jpg)

![9a8b4404bb8a35b6bd6514c9e12fb4d932abcd113602b40d1c658a9d8f825098.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/9a8b4404bb8a35b6bd6514c9e12fb4d932abcd113602b40d1c658a9d8f825098.jpg)

![ad8ae4196cd24c85cf3942cd0eb59c456bf35062a63ab5b0e34aa8f6543ee87c.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/ad8ae4196cd24c85cf3942cd0eb59c456bf35062a63ab5b0e34aa8f6543ee87c.jpg)

![b046c549563ebef0f3d197efeaaa5187b724583809b39ee9992dd4ae5c0c2f5c.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/b046c549563ebef0f3d197efeaaa5187b724583809b39ee9992dd4ae5c0c2f5c.jpg)

![b743d39eb75c673b48c0943cebdccf75ab766a6f1246da03fd082cf22e040f9a.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/b743d39eb75c673b48c0943cebdccf75ab766a6f1246da03fd082cf22e040f9a.jpg)

![bdb667f9d53c9ff29f208cd21000a2e9dbb5fb6e138a4766bb3539cd84a74d25.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/bdb667f9d53c9ff29f208cd21000a2e9dbb5fb6e138a4766bb3539cd84a74d25.jpg)

![c718a7dfa48847bf66702b9e0030679e58c2c6e0d81f2779130c3de395f7fada.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/c718a7dfa48847bf66702b9e0030679e58c2c6e0d81f2779130c3de395f7fada.jpg)

![ca67d30e631b44c490a391d2b8d9e1bfa7232f5b113b97c1987e5fc439b46e24.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/ca67d30e631b44c490a391d2b8d9e1bfa7232f5b113b97c1987e5fc439b46e24.jpg)

![cf687ac5d028f97b4df8fd07a22bcc31eba9990a064e7c563715064e87c7bc6f.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/cf687ac5d028f97b4df8fd07a22bcc31eba9990a064e7c563715064e87c7bc6f.jpg)

![d1e81a2bc2f6060d8a02b5955229333d9d95a79a826679e3a418c4781327d70a.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/d1e81a2bc2f6060d8a02b5955229333d9d95a79a826679e3a418c4781327d70a.jpg)

![d274935a22d9dfbc448e2302bdd21fcce16ca92dbbcf03221f13759c34b97cdc.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/d274935a22d9dfbc448e2302bdd21fcce16ca92dbbcf03221f13759c34b97cdc.jpg)

![e7931002e0261b4e544b6a54bac484e4f720c65c3118598f75a287361a628c01.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/e7931002e0261b4e544b6a54bac484e4f720c65c3118598f75a287361a628c01.jpg)

![efb41404538515675354f1cf467b0e5844794facd5caeecfd10a82d1d377add6.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/efb41404538515675354f1cf467b0e5844794facd5caeecfd10a82d1d377add6.jpg)

![f04fadd30be4f0b21f4a66467d079abcaf98cb9ecf4f4a51c6478b243be1d5bd.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/f04fadd30be4f0b21f4a66467d079abcaf98cb9ecf4f4a51c6478b243be1d5bd.jpg)

![f6a958f742ce5fc75d54ddbc8c79c0748cd1200825b622ae52b573ca775be54a.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/f6a958f742ce5fc75d54ddbc8c79c0748cd1200825b622ae52b573ca775be54a.jpg)

![faee75c8c6bb1a5a78fb957ad48ec20bb8f437ad8573bb443211d142bc17eb7c.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/images/faee75c8c6bb1a5a78fb957ad48ec20bb8f437ad8573bb443211d142bc17eb7c.jpg)

### Tables

![019f776a57619de369756096087d617a84860730ccb04f7c19c31e7605918786.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/019f776a57619de369756096087d617a84860730ccb04f7c19c31e7605918786.jpg)

![062cdaa1dce856fcf4e74359ba274b0380853b26422fa7d8672c01e760259e5a.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/062cdaa1dce856fcf4e74359ba274b0380853b26422fa7d8672c01e760259e5a.jpg)

![0bac9a0af6d1c1d8d86edcc62b3ca7821d19d5413e6c8d096e41d547bf1309ef.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/0bac9a0af6d1c1d8d86edcc62b3ca7821d19d5413e6c8d096e41d547bf1309ef.jpg)

![19b04ab9ebbf3adaab7647a20dbecf97b18e47df529d7477e293ee6ec31e6bf1.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/19b04ab9ebbf3adaab7647a20dbecf97b18e47df529d7477e293ee6ec31e6bf1.jpg)

![29aa3879f773583a47440618fdc89efea0b70d427fabf22b019ef19efe442472.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/29aa3879f773583a47440618fdc89efea0b70d427fabf22b019ef19efe442472.jpg)

![2ace84efa182f36821ca004fa45b078a9b21df1418e3236401f729354b46f763.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/2ace84efa182f36821ca004fa45b078a9b21df1418e3236401f729354b46f763.jpg)

![2b4461c44d42be4f9e2a21bb9b75126b0db7d21833d0e39a540fac5f8a39e94f.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/2b4461c44d42be4f9e2a21bb9b75126b0db7d21833d0e39a540fac5f8a39e94f.jpg)

![2b60bf56c555de37913e5d5fc56538d323298e4b099b002d8bab21af574d44e3.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/2b60bf56c555de37913e5d5fc56538d323298e4b099b002d8bab21af574d44e3.jpg)

![36a8e17fd540ba103839fc0c3febf5617c72bce1ea02adbe06752253a8836386.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/36a8e17fd540ba103839fc0c3febf5617c72bce1ea02adbe06752253a8836386.jpg)

![3d85afdc11f0ad074977ce0fa2550a7c4a546aa3db1bd6165129d64e752009f0.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/3d85afdc11f0ad074977ce0fa2550a7c4a546aa3db1bd6165129d64e752009f0.jpg)

![4019a799b9bb1bf5d3a33da183b2b36db8c6824b253d4383f7d3c18ee0ba2ef9.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/4019a799b9bb1bf5d3a33da183b2b36db8c6824b253d4383f7d3c18ee0ba2ef9.jpg)

![421a73fde70db3e40e07a683640352caf9655d5760f2e5c04907b869dacb6ef4.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/421a73fde70db3e40e07a683640352caf9655d5760f2e5c04907b869dacb6ef4.jpg)

![43e84f5702b397b002a1ab6ca73f95a18173d047cc038e8d3a6c7cde293e0af2.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/43e84f5702b397b002a1ab6ca73f95a18173d047cc038e8d3a6c7cde293e0af2.jpg)

![4c8097f517e8354f66ecb6544b8cefdb94a91915f523bfea40e6e6ecc10276aa.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/4c8097f517e8354f66ecb6544b8cefdb94a91915f523bfea40e6e6ecc10276aa.jpg)

![4f3912a7ecfdedf2a9c7aefdb004c5f6b60559e5c729ab9e458dc2f3e5a6caf1.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/4f3912a7ecfdedf2a9c7aefdb004c5f6b60559e5c729ab9e458dc2f3e5a6caf1.jpg)

![595295cefa947f7d8afa98beaf0c6d1477cfe3a6b11c84ed241bd4b67f180be6.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/595295cefa947f7d8afa98beaf0c6d1477cfe3a6b11c84ed241bd4b67f180be6.jpg)

![61668c8e9cb270d52f599c6c663de6cd340caf16f4b12261660791045fd2a1ab.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/61668c8e9cb270d52f599c6c663de6cd340caf16f4b12261660791045fd2a1ab.jpg)

![69c892b28d7fa7000662fb4b6d183773c1ba8f8ca866d5f76bfba2445c38c7c8.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/69c892b28d7fa7000662fb4b6d183773c1ba8f8ca866d5f76bfba2445c38c7c8.jpg)

![76bf01f79803e2e250054ce922063d23358590b4f4a30f7e4b067c8206ff61ec.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/76bf01f79803e2e250054ce922063d23358590b4f4a30f7e4b067c8206ff61ec.jpg)

![7f8ca4fafc0eeeb085710fb303db426f5e108e08b06a542b3e6ece4402702b0f.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/7f8ca4fafc0eeeb085710fb303db426f5e108e08b06a542b3e6ece4402702b0f.jpg)

![9cb043e33c68a9d19527d2b5538dfdf0930c347b212c7af94e2204659a7c2131.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/9cb043e33c68a9d19527d2b5538dfdf0930c347b212c7af94e2204659a7c2131.jpg)

![9f130f7ae8767958fefcc623098cf19c3f3010fdce29a3d4bcbb181601a57201.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/9f130f7ae8767958fefcc623098cf19c3f3010fdce29a3d4bcbb181601a57201.jpg)

![a2262bc23fadedd66240effd314598001d7dae54b928e08e37536c0f8bd9cdc3.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/a2262bc23fadedd66240effd314598001d7dae54b928e08e37536c0f8bd9cdc3.jpg)

![ab532878c8546af48ecf87401ef039b836dce59ac27caceb4c84f4636ad3ee31.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/ab532878c8546af48ecf87401ef039b836dce59ac27caceb4c84f4636ad3ee31.jpg)

![afdaaf141a2fa6cfbd4dd188ecddc0b6f6f166b0887d4a4edf8b346afc891ac6.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/afdaaf141a2fa6cfbd4dd188ecddc0b6f6f166b0887d4a4edf8b346afc891ac6.jpg)

![b459e81c91ec8063c8c6a19bcd23ff0a7ce3eec3b7a8cff4d30c1b0561d125f2.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/b459e81c91ec8063c8c6a19bcd23ff0a7ce3eec3b7a8cff4d30c1b0561d125f2.jpg)

![b47970cdb762ae52f54f8b9ec11d8a328310d8ccd31b328001c7acb992ba1c70.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/b47970cdb762ae52f54f8b9ec11d8a328310d8ccd31b328001c7acb992ba1c70.jpg)

![b8ecb9b04c201664c8e338adc0f7bcd4a7a3697628f9e0c9f1c823cda96fa7ea.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/b8ecb9b04c201664c8e338adc0f7bcd4a7a3697628f9e0c9f1c823cda96fa7ea.jpg)

![be036568b570a86202e26766a8e75d5b9306e7663044a868b29ed7a2b91e97a9.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/be036568b570a86202e26766a8e75d5b9306e7663044a868b29ed7a2b91e97a9.jpg)

![c8ec056bb3ea0a0f8eaec189fd2efe0e189788bbfc43f9fb90ba5c91b6b235ab.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/c8ec056bb3ea0a0f8eaec189fd2efe0e189788bbfc43f9fb90ba5c91b6b235ab.jpg)

![d62290a71540440fa90e22f88b8324b51d5c08d8d552553dce2cb9d9e2d8e8ab.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/d62290a71540440fa90e22f88b8324b51d5c08d8d552553dce2cb9d9e2d8e8ab.jpg)

![d91948ee77ba6eaefb91854e1b3d5cbfe78324c735bfdcf50d41c022caa34cc3.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/d91948ee77ba6eaefb91854e1b3d5cbfe78324c735bfdcf50d41c022caa34cc3.jpg)

![e2f4e2bf9346c24c9309049fc1a4ca5de5ec9e57df951f2abf8e358247595913.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/e2f4e2bf9346c24c9309049fc1a4ca5de5ec9e57df951f2abf8e358247595913.jpg)

![e4bdcd82dc6ba37dc9bf71866ac79e130f33397a46691155dc61d24e6a3abbee.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/e4bdcd82dc6ba37dc9bf71866ac79e130f33397a46691155dc61d24e6a3abbee.jpg)

![ec22150d3726c435c02c7a8a9ed2cc454f6ee8b2a4427f4d46e80f83a5d67230.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/ec22150d3726c435c02c7a8a9ed2cc454f6ee8b2a4427f4d46e80f83a5d67230.jpg)

![ed01c25a8b4767110825612924cb695a09f3a42a8d2370fe3111c5388673fb45.jpg](../icml_results/2432_Semi-Supervised%20Neural%20Super-Resolution%20for%20Mesh-Based%20Simulations/tables/ed01c25a8b4767110825612924cb695a09f3a42a8d2370fe3111c5388673fb45.jpg)

<a id="paper-2403"></a>

## Verifying Meta-Awareness via Predictive Rewards in Reasoning Models


### Images

![0fca63bc6eaeb17afd6d9386c2f6cdbeb4a8ac306faae535339b78c1d7801d2f.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/images/0fca63bc6eaeb17afd6d9386c2f6cdbeb4a8ac306faae535339b78c1d7801d2f.jpg)

![287f46c5a05cb734ab5f2cde559575021ac76aaf0ea8efa35e250a29fc074d40.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/images/287f46c5a05cb734ab5f2cde559575021ac76aaf0ea8efa35e250a29fc074d40.jpg)

![2fe9fd9ffd398d4fa03cd427db7a806db762dc1793379e2bfa1ce3051f0c87b8.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/images/2fe9fd9ffd398d4fa03cd427db7a806db762dc1793379e2bfa1ce3051f0c87b8.jpg)

![47e367d1c8aa78a3ab7daddef69eff660107b2ef3ae43a8f5058779093e72859.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/images/47e367d1c8aa78a3ab7daddef69eff660107b2ef3ae43a8f5058779093e72859.jpg)

![870fa9e4a3a603841c1b197433a2868e1a7b9ee239a7edd4e6d62bbc95165f94.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/images/870fa9e4a3a603841c1b197433a2868e1a7b9ee239a7edd4e6d62bbc95165f94.jpg)

![9959f2048cc4ebdf80e6fb8f759073a838d34d7bc0fc8294e5978df6d8ded830.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/images/9959f2048cc4ebdf80e6fb8f759073a838d34d7bc0fc8294e5978df6d8ded830.jpg)

![9f28a905fe0754e55f058e660b2df850ca7d30f6d72e59e40865b1ba388d0554.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/images/9f28a905fe0754e55f058e660b2df850ca7d30f6d72e59e40865b1ba388d0554.jpg)

![a78d79f8b1f82f9aa6d8282ebe14a80982510d6bf2590b260c33bcc44260a139.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/images/a78d79f8b1f82f9aa6d8282ebe14a80982510d6bf2590b260c33bcc44260a139.jpg)

![cd63a5537d7fc95719b38d23d4ea7a5df4fe1460d33c143b1e44bf03ada35a7d.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/images/cd63a5537d7fc95719b38d23d4ea7a5df4fe1460d33c143b1e44bf03ada35a7d.jpg)

![e45674baf5e91a50d8b67d0795c807345c15afebe8bf30a33bfc375434d04140.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/images/e45674baf5e91a50d8b67d0795c807345c15afebe8bf30a33bfc375434d04140.jpg)

### Tables

![1274993b02ab93d188b77b7d4b1cc6ef017fd6b94ad1cfc3d2ad7730fa03b1f7.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/tables/1274993b02ab93d188b77b7d4b1cc6ef017fd6b94ad1cfc3d2ad7730fa03b1f7.jpg)

![171e8fa62769fb1e7c2f1d5772ee2b1b0db2eb529399a37d99c5245e0f8cb220.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/tables/171e8fa62769fb1e7c2f1d5772ee2b1b0db2eb529399a37d99c5245e0f8cb220.jpg)

![22560c167de3e3751f3eeb0e8f57321823c98c9499bb5e4d21b3c67094d975c7.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/tables/22560c167de3e3751f3eeb0e8f57321823c98c9499bb5e4d21b3c67094d975c7.jpg)

![2a25093729af2b370bc29f2d1c88288d8f6150626348c5b68a96683596338118.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/tables/2a25093729af2b370bc29f2d1c88288d8f6150626348c5b68a96683596338118.jpg)

![369874c21707b83505daeeb1812383d45a70e093940cadba8750698c46991a36.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/tables/369874c21707b83505daeeb1812383d45a70e093940cadba8750698c46991a36.jpg)

![64c923a9f8d7f50407812355347e6eb397ee6a533c55b34ab0d1ad977d2a2ae8.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/tables/64c923a9f8d7f50407812355347e6eb397ee6a533c55b34ab0d1ad977d2a2ae8.jpg)

![9e5c9c555c43b622ff83331f8578c6be09cf133ad49b9413640ca2e07253c055.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/tables/9e5c9c555c43b622ff83331f8578c6be09cf133ad49b9413640ca2e07253c055.jpg)

![a9c2c709372709db8f56e2a03302d3c2983c6a0de57980bcb6e648e988cb3151.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/tables/a9c2c709372709db8f56e2a03302d3c2983c6a0de57980bcb6e648e988cb3151.jpg)

![e0663c1e30e01e97ad55eed00a0987f3fc6264cf5be8622e5399f153cf090fed.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/tables/e0663c1e30e01e97ad55eed00a0987f3fc6264cf5be8622e5399f153cf090fed.jpg)

![e53b5b2b6377357f30cb4266c909071c9b5dc40b3b6a3dca2979df5c80106344.jpg](../icml_results/2433_Verifying%20Meta-Awareness%20via%20Predictive%20Rewards%20in%20Reasoning%20Models/tables/e53b5b2b6377357f30cb4266c909071c9b5dc40b3b6a3dca2979df5c80106344.jpg)

<a id="paper-2404"></a>

## DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs


### Images

![0459b03ceb36cb5f8963ceb84549ef27ba8479a84e78b764e17f73b595d78e2c.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/0459b03ceb36cb5f8963ceb84549ef27ba8479a84e78b764e17f73b595d78e2c.jpg)

![12f87a1a87e1348b610b38cf5e6cbfc4918d831d8fababea7dbefb76b5469e71.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/12f87a1a87e1348b610b38cf5e6cbfc4918d831d8fababea7dbefb76b5469e71.jpg)

![1a746b51192aa77cbfc27c176d77371496bace668893cfe713f95871c4ee16da.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/1a746b51192aa77cbfc27c176d77371496bace668893cfe713f95871c4ee16da.jpg)

![1ecd8c1e696111630689e1bbc055ed4ff656511a8a569d37abe4c6d817c166b0.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/1ecd8c1e696111630689e1bbc055ed4ff656511a8a569d37abe4c6d817c166b0.jpg)

![29010018b25cc93c7dd3044f6b0c9c47de73e9219b5485b1dd074b14c00efe43.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/29010018b25cc93c7dd3044f6b0c9c47de73e9219b5485b1dd074b14c00efe43.jpg)

![2cd31729a25c7037a81c4b5b6ae0501989ec28d973514b586e2dcb065470c6e5.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/2cd31729a25c7037a81c4b5b6ae0501989ec28d973514b586e2dcb065470c6e5.jpg)

![3a78ae7b472d877386f95c9a3c28e93c6a158e1b87794f9bc4c42f909dfa6f18.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/3a78ae7b472d877386f95c9a3c28e93c6a158e1b87794f9bc4c42f909dfa6f18.jpg)

![73953f2d67ec973efd2f492e84a52e150e2dc319038c81dee8e9967bbd50dde9.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/73953f2d67ec973efd2f492e84a52e150e2dc319038c81dee8e9967bbd50dde9.jpg)

![85da456f0725e9b9ceb9dd626d1a494c46dd15f4f7d7267f50a99056b6269229.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/85da456f0725e9b9ceb9dd626d1a494c46dd15f4f7d7267f50a99056b6269229.jpg)

![88b0e5afd386317f87e0141b23eba95c652e198519e2a56075fda8a98ec5d481.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/88b0e5afd386317f87e0141b23eba95c652e198519e2a56075fda8a98ec5d481.jpg)

![91354e9cb1baa58af633c9c74b56b6c79e3e29369c88d6cabb7a694fe33b44ae.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/91354e9cb1baa58af633c9c74b56b6c79e3e29369c88d6cabb7a694fe33b44ae.jpg)

![934e21ded44028abcf0591ae7f9f3f061cfc2157c9ec5624eff9e5623d9263cd.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/934e21ded44028abcf0591ae7f9f3f061cfc2157c9ec5624eff9e5623d9263cd.jpg)

![b1242b4ce07824554956e32de1d698ce7e825aeac3c3c9a4b414bd80d5eb5bf0.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/b1242b4ce07824554956e32de1d698ce7e825aeac3c3c9a4b414bd80d5eb5bf0.jpg)

![c455483f7685889f1e727381b94a15a4e24d25cd814034a012464a2c9fb9ec47.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/c455483f7685889f1e727381b94a15a4e24d25cd814034a012464a2c9fb9ec47.jpg)

![ca384ca4bae659cd05239d78a1d64da47e9c4d3dfa875d896677d7c5afe3d633.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/ca384ca4bae659cd05239d78a1d64da47e9c4d3dfa875d896677d7c5afe3d633.jpg)

![e301ad72be61a6850444d7a6ae4d1bd1e7e39649d80851f9633d937568bfe77a.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/e301ad72be61a6850444d7a6ae4d1bd1e7e39649d80851f9633d937568bfe77a.jpg)

![f1b737c371d3746c3acfc0fe4718cd3e42034488dbd2bb24952d1b8b9eedc9b5.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/f1b737c371d3746c3acfc0fe4718cd3e42034488dbd2bb24952d1b8b9eedc9b5.jpg)

![f94db08c4bac6571eadb3024ea589c19538a6d9aa7c42c09d7ef2dea00d99e81.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/images/f94db08c4bac6571eadb3024ea589c19538a6d9aa7c42c09d7ef2dea00d99e81.jpg)

### Tables

![2f81a3440f2e5e55e7d3d01f28a344e42db3c4aac3d7af25b573cf2a7a14433b.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/tables/2f81a3440f2e5e55e7d3d01f28a344e42db3c4aac3d7af25b573cf2a7a14433b.jpg)

![3588421f2aaf964ec984316108f0f767ed8c53b70eb9f663c721362e960d9b52.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/tables/3588421f2aaf964ec984316108f0f767ed8c53b70eb9f663c721362e960d9b52.jpg)

![464f447f0c23253bfe5d3401b17c1250c8a02f85fc5d913033980548fef33c16.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/tables/464f447f0c23253bfe5d3401b17c1250c8a02f85fc5d913033980548fef33c16.jpg)

![515e725cd5994988d608d2256408501ecfe83cda7bb77618636a8530b2f36ff4.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/tables/515e725cd5994988d608d2256408501ecfe83cda7bb77618636a8530b2f36ff4.jpg)

![5e72e47ef0aae33c84ff2376e0ef88fb7eaad458aeee3f134f932c40beb87539.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/tables/5e72e47ef0aae33c84ff2376e0ef88fb7eaad458aeee3f134f932c40beb87539.jpg)

![6be84cf6eb64344fecc2840928667e194268963ab6bdeccdaa6e3bc65d9f1d00.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/tables/6be84cf6eb64344fecc2840928667e194268963ab6bdeccdaa6e3bc65d9f1d00.jpg)

![8cf558b9f79a1ab9e4b3fdf27110b0cfa308601e3f83635b127cb024cac723f9.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/tables/8cf558b9f79a1ab9e4b3fdf27110b0cfa308601e3f83635b127cb024cac723f9.jpg)

![8ed0e939f89d0db89099cc21893eeefe057403b7c482baa99ade60770f6002a6.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/tables/8ed0e939f89d0db89099cc21893eeefe057403b7c482baa99ade60770f6002a6.jpg)

![bea3e9a411b320851c08036c2aa9c147cae920fa0f98ace4c589b50d6ed14758.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/tables/bea3e9a411b320851c08036c2aa9c147cae920fa0f98ace4c589b50d6ed14758.jpg)

![d5c2e9e6a47d0b32562fdf23c02bd8d6379ec02bb5871d636378868a7e288bd7.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/tables/d5c2e9e6a47d0b32562fdf23c02bd8d6379ec02bb5871d636378868a7e288bd7.jpg)

![ec36fc4e4037e157469b44e6545d5c859ce8c424e6f5d3a7bf0de26e0d6ef661.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/tables/ec36fc4e4037e157469b44e6545d5c859ce8c424e6f5d3a7bf0de26e0d6ef661.jpg)

![eef29078224c6ecac2ce401d0b90da1ff52676004c4ba50a60aefcb849c3caee.jpg](../icml_results/2434_DAPD_%20Dependency-Aware%20Parallel%20Decoding%20via%20Attention%20for%20Diffusion%20LLMs/tables/eef29078224c6ecac2ce401d0b90da1ff52676004c4ba50a60aefcb849c3caee.jpg)

<a id="paper-2405"></a>

## Inverting Data Transformations via Diffusion Sampling


### Images

![08475c91a0d46e37c1109cd4f43e5deb9e858e2b7bc7d1a3b2b3d0ccf0bc3e74.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/images/08475c91a0d46e37c1109cd4f43e5deb9e858e2b7bc7d1a3b2b3d0ccf0bc3e74.jpg)

![0c03b5661fd84a4c2d6a70eef5afe249e5cc4ad3a563c9e66cc1a71463b19766.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/images/0c03b5661fd84a4c2d6a70eef5afe249e5cc4ad3a563c9e66cc1a71463b19766.jpg)

![11862e780a4b7e6275c116f840fa8e7125189d36daf87a497f3a35316d6320a9.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/images/11862e780a4b7e6275c116f840fa8e7125189d36daf87a497f3a35316d6320a9.jpg)

![2cb267d714ddc9c9eb2f71488037229f55354117a21eb663d7744a33cf9c6d8a.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/images/2cb267d714ddc9c9eb2f71488037229f55354117a21eb663d7744a33cf9c6d8a.jpg)

![35427af7217dfbde0aa617add1cb159dbac0298e08c491b3848d512a44397dbc.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/images/35427af7217dfbde0aa617add1cb159dbac0298e08c491b3848d512a44397dbc.jpg)

![a2722b1d1ce4b4db1f277a24e94160657aa64a7014e3246ac277e1701c34c176.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/images/a2722b1d1ce4b4db1f277a24e94160657aa64a7014e3246ac277e1701c34c176.jpg)

![abd2e94b19201088cc402de0669c9cce441888fbef9024ca1713623737f3d632.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/images/abd2e94b19201088cc402de0669c9cce441888fbef9024ca1713623737f3d632.jpg)

![ba2682b8599c1cf19f15ba226bc0c83f91efbfaff3bf7f1fda3653486dc5a440.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/images/ba2682b8599c1cf19f15ba226bc0c83f91efbfaff3bf7f1fda3653486dc5a440.jpg)

![c6b5fd8ce106f8570507e519c23b37a6ed2c443588a7874b45593f0a80190c80.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/images/c6b5fd8ce106f8570507e519c23b37a6ed2c443588a7874b45593f0a80190c80.jpg)

![c7b38d32791f2f470fe338ab9e2fc48886fbd56509730901588cea247b7854bd.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/images/c7b38d32791f2f470fe338ab9e2fc48886fbd56509730901588cea247b7854bd.jpg)

![d91f4128993ce3654eee958cb2f2eadbee0601775929c6dbc318e8b8f3055b96.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/images/d91f4128993ce3654eee958cb2f2eadbee0601775929c6dbc318e8b8f3055b96.jpg)

### Tables

![45433b4cc6bb69a405382adac45be6dcaa8e975480d0fdb4208bb6026c036e8d.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/tables/45433b4cc6bb69a405382adac45be6dcaa8e975480d0fdb4208bb6026c036e8d.jpg)

![6b0a0353a1cca67e61dd36fae5e8f1b2ee8fc69feca34b0cc6a645cd15c883e9.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/tables/6b0a0353a1cca67e61dd36fae5e8f1b2ee8fc69feca34b0cc6a645cd15c883e9.jpg)

![9a9eea4a11fdb9da9fc9790bed66c2cd8881bdbaa88c81341ab7246df3200884.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/tables/9a9eea4a11fdb9da9fc9790bed66c2cd8881bdbaa88c81341ab7246df3200884.jpg)

![a438eac4f83a1dd0debb4a7971d07184330a63a8d1b0725d8b61ad83db203dfd.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/tables/a438eac4f83a1dd0debb4a7971d07184330a63a8d1b0725d8b61ad83db203dfd.jpg)

![e610d9392106ceef3728ed2ab019877cf20248f494e3f70d9df7586b241efb46.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/tables/e610d9392106ceef3728ed2ab019877cf20248f494e3f70d9df7586b241efb46.jpg)

![f3d4be184f5aad6ce02c31d798aa1475dfcd4e3cea8178add29e12e34b3261af.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/tables/f3d4be184f5aad6ce02c31d798aa1475dfcd4e3cea8178add29e12e34b3261af.jpg)

![f5faca027571b6950ac03b7200a5d03b161e2eac9ee13aef4b75d3ccc7898647.jpg](../icml_results/2435_Inverting%20Data%20Transformations%20via%20Diffusion%20Sampling/tables/f5faca027571b6950ac03b7200a5d03b161e2eac9ee13aef4b75d3ccc7898647.jpg)

<a id="paper-2406"></a>

## AlienLM: Alienization of Language for API-Boundary Privacy in Black-Box LLMs


### Images

![156904397a2609bb3f23a98521156c8db308a30572cc38dd8d64344a26e9d0f6.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/images/156904397a2609bb3f23a98521156c8db308a30572cc38dd8d64344a26e9d0f6.jpg)

![194978f098176d82454bbbfb3ab452be2d51e96d1de21a3df32b83e32d0a621e.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/images/194978f098176d82454bbbfb3ab452be2d51e96d1de21a3df32b83e32d0a621e.jpg)

![4e6457d2fed4c6c2d9391534e2e254e089747cb16621290543304dc33f3bf597.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/images/4e6457d2fed4c6c2d9391534e2e254e089747cb16621290543304dc33f3bf597.jpg)

![6087ec5f900b995c4cbee1fcd2b6d9944bc02b3cbd9753346cbeece0daff2dee.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/images/6087ec5f900b995c4cbee1fcd2b6d9944bc02b3cbd9753346cbeece0daff2dee.jpg)

![cad5550de545332830b4a33b34271ed70bdb43577373b25f74b8b74734fd7dba.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/images/cad5550de545332830b4a33b34271ed70bdb43577373b25f74b8b74734fd7dba.jpg)

### Tables

![0db87e81fbd77748099b013959c9ed792a647cfa7c27b2b87ee8cadd182a4e20.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/0db87e81fbd77748099b013959c9ed792a647cfa7c27b2b87ee8cadd182a4e20.jpg)

![14e0855cad518f1db25742a9ebf62cd5fde5bcb266ac75e20e6a48dc8073cb93.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/14e0855cad518f1db25742a9ebf62cd5fde5bcb266ac75e20e6a48dc8073cb93.jpg)

![183bb2ea09286f06f54b7dc0e5af91e93e8e3feac6ce722cd74a73cd11d37eb6.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/183bb2ea09286f06f54b7dc0e5af91e93e8e3feac6ce722cd74a73cd11d37eb6.jpg)

![2501d8d407dd078d6ade21c16d999516748ae0c2372270d584fac37777cc7512.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/2501d8d407dd078d6ade21c16d999516748ae0c2372270d584fac37777cc7512.jpg)

![3834b740c56f5135f74556cee19f8e92751e1f254da61176ae758feef368958c.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/3834b740c56f5135f74556cee19f8e92751e1f254da61176ae758feef368958c.jpg)

![3d50f273db22f8314e53ce2da7e34cad81838d566dd052886b3e6d6046542652.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/3d50f273db22f8314e53ce2da7e34cad81838d566dd052886b3e6d6046542652.jpg)

![46e047423922081b18ccda7ceb3ce9ccc23d3356272437adc9ff52a20537212b.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/46e047423922081b18ccda7ceb3ce9ccc23d3356272437adc9ff52a20537212b.jpg)

![471d52155d6d530c26d85a9b81b5b210ec12e43c1378a4bbba27dff677cfefcb.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/471d52155d6d530c26d85a9b81b5b210ec12e43c1378a4bbba27dff677cfefcb.jpg)

![4af2a92b306bfc724cec8f8614baf20675fc5734de7c41c91960b6cafc33f0d0.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/4af2a92b306bfc724cec8f8614baf20675fc5734de7c41c91960b6cafc33f0d0.jpg)

![5bd831290ca8387e8aea279413741069a160c7b71a731d9960ef464de312e2f5.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/5bd831290ca8387e8aea279413741069a160c7b71a731d9960ef464de312e2f5.jpg)

![5c5df218d0eb3b4963e6d37f4f3c5b1e671d1b16dc1abfbee37990ad67cd3237.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/5c5df218d0eb3b4963e6d37f4f3c5b1e671d1b16dc1abfbee37990ad67cd3237.jpg)

![78b26f1f51d315a073a1e883bbb80ba9ff812a3203694ffff6474fb04347c719.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/78b26f1f51d315a073a1e883bbb80ba9ff812a3203694ffff6474fb04347c719.jpg)

![8c231136b7eb65c4df1336ba96245fb1ecba9e414247aeafbbd7137e0d53b65c.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/8c231136b7eb65c4df1336ba96245fb1ecba9e414247aeafbbd7137e0d53b65c.jpg)

![906997be1da97ce18ea38027b5c82759a36ae10c6860b32622532ee361acae56.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/906997be1da97ce18ea38027b5c82759a36ae10c6860b32622532ee361acae56.jpg)

![ab11be9e4489d3d715180030291f5ba9a33c4a3937122762dad0a22b604dfb80.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/ab11be9e4489d3d715180030291f5ba9a33c4a3937122762dad0a22b604dfb80.jpg)

![b320d746deec3e8f7da3d26d26d21529568e6c4fef3627416bbd779dffc95068.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/b320d746deec3e8f7da3d26d26d21529568e6c4fef3627416bbd779dffc95068.jpg)

![b8422fb676ea7cf5c4ff6dcd6171b732512fae9a7abbec56aa7ee2efd3f63761.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/b8422fb676ea7cf5c4ff6dcd6171b732512fae9a7abbec56aa7ee2efd3f63761.jpg)

![bed8a0dbce780dfa82967ccd25a209d542204211a9dbc24ae7c213699480456a.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/bed8a0dbce780dfa82967ccd25a209d542204211a9dbc24ae7c213699480456a.jpg)

![c372012e5708e4da6ec35a5941dde6a74b3b2f5bfe5d8f8498c89270f3aebd30.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/c372012e5708e4da6ec35a5941dde6a74b3b2f5bfe5d8f8498c89270f3aebd30.jpg)

![cc9755f715d38b847aaf3ef6d197c042e9675f48e7578a9d90eef6427562a911.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/cc9755f715d38b847aaf3ef6d197c042e9675f48e7578a9d90eef6427562a911.jpg)

![cfa9f0d2288853075dc58d8bab3c2c7461f34909e0a45d917ff19cbeab2fe7f9.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/cfa9f0d2288853075dc58d8bab3c2c7461f34909e0a45d917ff19cbeab2fe7f9.jpg)

![d033c78700bb9c1a9bf92e3f68d128c821fc4568fb5b2a63212d2a6175b71df0.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/d033c78700bb9c1a9bf92e3f68d128c821fc4568fb5b2a63212d2a6175b71df0.jpg)

![de589de3a2a6a72cb5ea637115e20a80d07a9788af16be2868dec38b4afa73c1.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/de589de3a2a6a72cb5ea637115e20a80d07a9788af16be2868dec38b4afa73c1.jpg)

![ebf2acffc073fc0910e43a5e8a22c94f688025a345a7c89236f3f72c5a068517.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/ebf2acffc073fc0910e43a5e8a22c94f688025a345a7c89236f3f72c5a068517.jpg)

![f1f54b44ffb5b4065e445cc9d8ac595396468b63faef1ce8a2f5980ab2d64087.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/f1f54b44ffb5b4065e445cc9d8ac595396468b63faef1ce8a2f5980ab2d64087.jpg)

![fed39ed7c9791061dfedf557d596f05811c17594f2e78f6796a542ddfcf3aaf0.jpg](../icml_results/2436_AlienLM_%20Alienization%20of%20Language%20for%20API-Boundary%20Privacy%20in%20Black-Box%20LLMs/tables/fed39ed7c9791061dfedf557d596f05811c17594f2e78f6796a542ddfcf3aaf0.jpg)

<a id="paper-2407"></a>

## Unlocking Noise-Resistant Vision: Key Architectural Secrets for Robust Models Against Gaussian Noise


### Images

![1a55a598a9bf4e3231209e7a8161dd52837f4457497e32377df5d7678e539d0e.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/images/1a55a598a9bf4e3231209e7a8161dd52837f4457497e32377df5d7678e539d0e.jpg)

![2316e2dd5e90a374369f25226fe82556316eba6df65edf277873408b205205a3.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/images/2316e2dd5e90a374369f25226fe82556316eba6df65edf277873408b205205a3.jpg)

![2ce34d0ef65b05fae82b17f8d6a62f89ab3239e3012963481bfb1239da25e4ba.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/images/2ce34d0ef65b05fae82b17f8d6a62f89ab3239e3012963481bfb1239da25e4ba.jpg)

![39fab84aa1d1e1d256df2c1290b9c6e2fe30169feed932160ef1cc149e57e1c6.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/images/39fab84aa1d1e1d256df2c1290b9c6e2fe30169feed932160ef1cc149e57e1c6.jpg)

![4ef9b982cd8c5537d521ba3ade73346e964fb836c0239c3f94a9df312a4d4e2a.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/images/4ef9b982cd8c5537d521ba3ade73346e964fb836c0239c3f94a9df312a4d4e2a.jpg)

![5b15c2f177e90e44380cd4a534bf852cd9ec78c2f20c1520c6e7e15d7b7f9952.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/images/5b15c2f177e90e44380cd4a534bf852cd9ec78c2f20c1520c6e7e15d7b7f9952.jpg)

![b9a0f79ea7362ef7f827c0c65338ddc45a96c5d2b37d574d5cdb174cc2c05aa8.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/images/b9a0f79ea7362ef7f827c0c65338ddc45a96c5d2b37d574d5cdb174cc2c05aa8.jpg)

![bd08b70d0969216fafb49770e851dd91c469896369834a8e7d1f8078124e749b.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/images/bd08b70d0969216fafb49770e851dd91c469896369834a8e7d1f8078124e749b.jpg)

![d9fa76378cd9ccc7826fe8900f25c44f570c45884183e950cc77f0fc4f7fb60c.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/images/d9fa76378cd9ccc7826fe8900f25c44f570c45884183e950cc77f0fc4f7fb60c.jpg)

### Tables

![01fc150909cbd34d24a737b93b8ec23d56c0bbf5467d8b03655ebe6ffba15f92.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/01fc150909cbd34d24a737b93b8ec23d56c0bbf5467d8b03655ebe6ffba15f92.jpg)

![083afa24e1880c8da691b6a33e033e25374fd47adf49471efa8f418927352013.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/083afa24e1880c8da691b6a33e033e25374fd47adf49471efa8f418927352013.jpg)

![11283ae97f9e4b829131eed91e7da40eb547250866d427c29d53a86847b9bb36.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/11283ae97f9e4b829131eed91e7da40eb547250866d427c29d53a86847b9bb36.jpg)

![1223a777e163ab1b8d01a63834caaa648383191878281c76e63353398f21ea01.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/1223a777e163ab1b8d01a63834caaa648383191878281c76e63353398f21ea01.jpg)

![1acbbbd03273545adf2dae3304d7ec7ba376b97336c0de43668ee1a13d19ec0d.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/1acbbbd03273545adf2dae3304d7ec7ba376b97336c0de43668ee1a13d19ec0d.jpg)

![20af846e400db42a3eff43bd931bacfaea67fa655e43f42a35b4e8d3097a67d8.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/20af846e400db42a3eff43bd931bacfaea67fa655e43f42a35b4e8d3097a67d8.jpg)

![20eb658cded9e1bc7c7539cdb8f2baee184dde105ea44b3a080cae519a697b8f.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/20eb658cded9e1bc7c7539cdb8f2baee184dde105ea44b3a080cae519a697b8f.jpg)

![261f04070736151a3f1e3cc96507b4924880ad7a7dca95759c20daf7f4d4f551.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/261f04070736151a3f1e3cc96507b4924880ad7a7dca95759c20daf7f4d4f551.jpg)

![3452bd3b67c81d9adef607c2fead96ba3c23e73db17d9adbf6ec8e0ddcd020b8.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/3452bd3b67c81d9adef607c2fead96ba3c23e73db17d9adbf6ec8e0ddcd020b8.jpg)

![50c123301ee205385298f77f83df1bb8023545882263a04ba817d3e2ff714adb.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/50c123301ee205385298f77f83df1bb8023545882263a04ba817d3e2ff714adb.jpg)

![5cab84f6c9a4f226145cd7ead3be239c6d1d02665039ddfd4ce24248350549ed.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/5cab84f6c9a4f226145cd7ead3be239c6d1d02665039ddfd4ce24248350549ed.jpg)

![5d27e0b26bb5df58f3e488dc48d2df1436e73dd4f3284970767316fbe7f87385.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/5d27e0b26bb5df58f3e488dc48d2df1436e73dd4f3284970767316fbe7f87385.jpg)

![6e97b1dce164294c88b5ca60214abb7c1ed09b2eb6f425785fbbc15680d5bd3d.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/6e97b1dce164294c88b5ca60214abb7c1ed09b2eb6f425785fbbc15680d5bd3d.jpg)

![7722fb21f4671af88ee5889e89d03e403044beff86a2c62092d6ccef4e20cfa4.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/7722fb21f4671af88ee5889e89d03e403044beff86a2c62092d6ccef4e20cfa4.jpg)

![78276615a9c9ab0446393d06fc44b643cebf4bdcb9b1ff6dd416d7365c2e00a2.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/78276615a9c9ab0446393d06fc44b643cebf4bdcb9b1ff6dd416d7365c2e00a2.jpg)

![8b645a3b7d3503eb083b2237a6e0f7fc2cd9606c3ba36df49b23dad16b5c3eeb.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/8b645a3b7d3503eb083b2237a6e0f7fc2cd9606c3ba36df49b23dad16b5c3eeb.jpg)

![97cc6f5c81733de5be241928866ddc01c76482ec4fdb52e2790e132ff9dcc7fa.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/97cc6f5c81733de5be241928866ddc01c76482ec4fdb52e2790e132ff9dcc7fa.jpg)

![b513ba2552b1e1299244eb3621bbceae5b0227593b321ffee235551a0a86981e.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/b513ba2552b1e1299244eb3621bbceae5b0227593b321ffee235551a0a86981e.jpg)

![c2aa03f33ac85c39f509de1f78ca4515bbd39a7d9756b1c1c7e2c12860d88e95.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/c2aa03f33ac85c39f509de1f78ca4515bbd39a7d9756b1c1c7e2c12860d88e95.jpg)

![c40a6b79ddc51def73d87d0615f147d7ac8b7ba41bb086c9a6d7da03f695c7c2.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/c40a6b79ddc51def73d87d0615f147d7ac8b7ba41bb086c9a6d7da03f695c7c2.jpg)

![e23a3711c23b06b720dc1920f9fa0be0534722142ce49cbf64e4eacc25adc224.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/e23a3711c23b06b720dc1920f9fa0be0534722142ce49cbf64e4eacc25adc224.jpg)

![e922b0e483b7f01a14f39313996415decfc209b046f7d0dc3bb446c191c9e06a.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/e922b0e483b7f01a14f39313996415decfc209b046f7d0dc3bb446c191c9e06a.jpg)

![ebdbf2a3fe2ba02353d15bb7726aaf6c163c9a1bc0346610b4433d5231b414aa.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/ebdbf2a3fe2ba02353d15bb7726aaf6c163c9a1bc0346610b4433d5231b414aa.jpg)

![eca92972726026102ef3434bb1744c648db186f3c12243a7d594b7c3df0dd540.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/eca92972726026102ef3434bb1744c648db186f3c12243a7d594b7c3df0dd540.jpg)

![ee286a3afe673d888ce9f01895362b348dd875775a89afb307ec530c9d555b2e.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/ee286a3afe673d888ce9f01895362b348dd875775a89afb307ec530c9d555b2e.jpg)

![f191958a5717fccb61347f917f43b05b307315528e8529db4d6ebb7ac66970eb.jpg](../icml_results/2437_Unlocking%20Noise-Resistant%20Vision_%20Key%20Architectural%20Secrets%20for%20Robust%20Models%20Against%20Gaussian%20Noise/tables/f191958a5717fccb61347f917f43b05b307315528e8529db4d6ebb7ac66970eb.jpg)

<a id="paper-2408"></a>

## Compositional Transduction with Latent Analogies for Offline Goal-Conditioned Reinforcement Learning


### Images

![1d5c3d915d53536dcd0a307a82841cf2830056c861c4e3ff9a68a54344b784b2.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/images/1d5c3d915d53536dcd0a307a82841cf2830056c861c4e3ff9a68a54344b784b2.jpg)

![5c1d68889a1215ac24ec39fb86736094aa73b76cf52d2b83a875ded6cfa4aab2.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/images/5c1d68889a1215ac24ec39fb86736094aa73b76cf52d2b83a875ded6cfa4aab2.jpg)

![7fe65f4d3d325a31dd76713616046b1431e710e2d5116a66ca04dd02c262b5c2.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/images/7fe65f4d3d325a31dd76713616046b1431e710e2d5116a66ca04dd02c262b5c2.jpg)

![977f2727a6dc0313e21e7208a08f5e5297fe49fc00f06bc8231ade3a5c879934.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/images/977f2727a6dc0313e21e7208a08f5e5297fe49fc00f06bc8231ade3a5c879934.jpg)

![99f0f3d8d375b24d2391df54a20d60d58325dd54e7ca22be9fd347b8aac77c7d.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/images/99f0f3d8d375b24d2391df54a20d60d58325dd54e7ca22be9fd347b8aac77c7d.jpg)

![9dccebf17d0e93742082791b47d06d3e1543601d2886a4e46022dba667908973.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/images/9dccebf17d0e93742082791b47d06d3e1543601d2886a4e46022dba667908973.jpg)

![bdb5ff218f20eecc99123a3047d7e5eb4486df9c212b9f820a3c7fe5754b7fd5.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/images/bdb5ff218f20eecc99123a3047d7e5eb4486df9c212b9f820a3c7fe5754b7fd5.jpg)

![e11cecfb3a67b5864c1e527fefe0e3671a56f32854b1cda9cab43c5e2aef0cdb.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/images/e11cecfb3a67b5864c1e527fefe0e3671a56f32854b1cda9cab43c5e2aef0cdb.jpg)

![e37a4e5d6ff1af2e916accff5ea061479c329a603aeeb3f5dd8a19dc101f7de2.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/images/e37a4e5d6ff1af2e916accff5ea061479c329a603aeeb3f5dd8a19dc101f7de2.jpg)

![e7196705e9cfad1e25b15c4b1092b7cb1e2dae6df741cfb246fb15fce5e92f3b.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/images/e7196705e9cfad1e25b15c4b1092b7cb1e2dae6df741cfb246fb15fce5e92f3b.jpg)

![f8d3bbcb289b604f1e79cd4836112e3098180f3bc099fbcc149e1c39b2273147.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/images/f8d3bbcb289b604f1e79cd4836112e3098180f3bc099fbcc149e1c39b2273147.jpg)

### Tables

![29bbc6ae1ad564b371a1d7b44033507b57206c0bdba56bd9d5ec00e4ff2cd2d2.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/tables/29bbc6ae1ad564b371a1d7b44033507b57206c0bdba56bd9d5ec00e4ff2cd2d2.jpg)

![4094d920da344ea1674ce166e98787f4b347010bb83cd9755f9ec4e927b08772.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/tables/4094d920da344ea1674ce166e98787f4b347010bb83cd9755f9ec4e927b08772.jpg)

![9b515e6845d10d98a9d102512aadce297fef1dc7e05765b05161f0d979dd451b.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/tables/9b515e6845d10d98a9d102512aadce297fef1dc7e05765b05161f0d979dd451b.jpg)

![a4287efaa05d6d3ff8ac9cbfc29f409e0da7219cfdf869a606af3929ae3762cd.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/tables/a4287efaa05d6d3ff8ac9cbfc29f409e0da7219cfdf869a606af3929ae3762cd.jpg)

![b5fa4ada778a2fedd26873a92e03ab27054d996fe4bc8c52d5dce051f34bb59b.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/tables/b5fa4ada778a2fedd26873a92e03ab27054d996fe4bc8c52d5dce051f34bb59b.jpg)

![c51c28abd93a53d89a11a01c33a15c1670b08af36ab8a92c2aa2ed8e366200db.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/tables/c51c28abd93a53d89a11a01c33a15c1670b08af36ab8a92c2aa2ed8e366200db.jpg)

![d589987a469f674391df73294dd5e288f75cf8e324b4a21d71c3966309ef4a82.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/tables/d589987a469f674391df73294dd5e288f75cf8e324b4a21d71c3966309ef4a82.jpg)

![e3906919d9063ba3d30d35fe23dabc059f7a519928152cd915839b05f6f392a0.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/tables/e3906919d9063ba3d30d35fe23dabc059f7a519928152cd915839b05f6f392a0.jpg)

![f00253c22c8dad82c39d1472f2502f5fa1524429c914835ca17abe695a0e7766.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/tables/f00253c22c8dad82c39d1472f2502f5fa1524429c914835ca17abe695a0e7766.jpg)

![f87addb567754e9f1ed545564000cdfa52f608c053b5907c2df8daea7a69fec6.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/tables/f87addb567754e9f1ed545564000cdfa52f608c053b5907c2df8daea7a69fec6.jpg)

![f8ba73180f12a0a501e6e66d83fab1752ad09512009437a3ac0ea32634577455.jpg](../icml_results/2438_Compositional%20Transduction%20with%20Latent%20Analogies%20for%20Offline%20Goal-Conditioned%20Reinforcement%20Learning/tables/f8ba73180f12a0a501e6e66d83fab1752ad09512009437a3ac0ea32634577455.jpg)

<a id="paper-2409"></a>

## Block-wise Codeword Embedding for Reliable Multi-bit Text Watermarking

### Images

![0d68e6c42db12b1bda658fde418886152af9ea0867729111e1d29100e962487b.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/0d68e6c42db12b1bda658fde418886152af9ea0867729111e1d29100e962487b.jpg)

![3e9bdc5ecd241ff573e24f37a63d31fb1c1776b15cb338cee3c9e265b5ff23db.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/3e9bdc5ecd241ff573e24f37a63d31fb1c1776b15cb338cee3c9e265b5ff23db.jpg)

![4b954cd98b6b2d9638f9806e7aa50aec70f40459dc76ad18c202e51e6206eb4a.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/4b954cd98b6b2d9638f9806e7aa50aec70f40459dc76ad18c202e51e6206eb4a.jpg)

![606c5a67ca5b1cb19808c71e06d0f201de66ace594624a411875d1b8fbb6d364.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/606c5a67ca5b1cb19808c71e06d0f201de66ace594624a411875d1b8fbb6d364.jpg)

![8d183b6bdd4392ab3133ae858ee6b5a613a3df9004fbdf11abfbb38ffdaa5e93.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/8d183b6bdd4392ab3133ae858ee6b5a613a3df9004fbdf11abfbb38ffdaa5e93.jpg)

![a66048060adbb6d5b00bd77178567471f26947588cfbe3d662db2a4f261014f8.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/a66048060adbb6d5b00bd77178567471f26947588cfbe3d662db2a4f261014f8.jpg)

![c482a3db423d2cd547b14320f21a97e15227a8a26da7836d79bee45ad4ce5888.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/c482a3db423d2cd547b14320f21a97e15227a8a26da7836d79bee45ad4ce5888.jpg)

![c86ef8519b65e5c8f01631ed0d60b72c55049353aadc2c4704d3c9707bf48212.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/c86ef8519b65e5c8f01631ed0d60b72c55049353aadc2c4704d3c9707bf48212.jpg)

![dd24993b03862234af04c46ad8c4dc10b8369af2afb14863868f30b68d4fdab9.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/dd24993b03862234af04c46ad8c4dc10b8369af2afb14863868f30b68d4fdab9.jpg)

![de2345c7f2ecda8a35247a3a3ffe9cb5fb996057dfa718fa52422281e5a164e7.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/de2345c7f2ecda8a35247a3a3ffe9cb5fb996057dfa718fa52422281e5a164e7.jpg)

![f88fff8b9e0ed2013ca79b629bd45db8634f4d6c8a5e57682c2af0a8087f730d.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/images/f88fff8b9e0ed2013ca79b629bd45db8634f4d6c8a5e57682c2af0a8087f730d.jpg)

### Tables

![23302c630704da62d3072c667f08c6b9a522f7afdc9cbc0610fa5c627e562d6c.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/23302c630704da62d3072c667f08c6b9a522f7afdc9cbc0610fa5c627e562d6c.jpg)

![2c02a461e9ad4f839c750fe28d8c7cb60c846437cd4ef88b898c73f26795898b.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/2c02a461e9ad4f839c750fe28d8c7cb60c846437cd4ef88b898c73f26795898b.jpg)

![2f5c563d1a325fd24af04e1ccd8678e29bc7cffe1c2710358c65f1fe87419870.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/2f5c563d1a325fd24af04e1ccd8678e29bc7cffe1c2710358c65f1fe87419870.jpg)

![308e4eb74fd168b1565d98f3328a2c6236785fe3d91e023b90e6d7c180a8deb0.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/308e4eb74fd168b1565d98f3328a2c6236785fe3d91e023b90e6d7c180a8deb0.jpg)

![436c3f80a9c794f21fdf862008b4571a1872e17e965b8233d43a0c31a599a542.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/436c3f80a9c794f21fdf862008b4571a1872e17e965b8233d43a0c31a599a542.jpg)

![530e6567d02ea872dce18537d34d70c523e7148da96fb6962274195d9229fb68.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/530e6567d02ea872dce18537d34d70c523e7148da96fb6962274195d9229fb68.jpg)

![55b4423b7d42d9c4d4f57794e4a8e4db803483ba6f6d31ac5a7cb237f0bcf4c8.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/55b4423b7d42d9c4d4f57794e4a8e4db803483ba6f6d31ac5a7cb237f0bcf4c8.jpg)

![5c460a090a90083fdfbc3f5751ba95fd9aca9cf1a14e670b00ce0e658f7acb91.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/5c460a090a90083fdfbc3f5751ba95fd9aca9cf1a14e670b00ce0e658f7acb91.jpg)

![5eb0f1948c445abed6ec083366f8f3bc3d8fb387400445a2906edaedae5b1ae4.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/5eb0f1948c445abed6ec083366f8f3bc3d8fb387400445a2906edaedae5b1ae4.jpg)

![63d989a51d66fcd6534e07c110f6fcec3ea783665af65fdf1c6ff1fac91699ba.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/63d989a51d66fcd6534e07c110f6fcec3ea783665af65fdf1c6ff1fac91699ba.jpg)

![727e8a98e33b28603f560058b8d8a575d8647e52f2223963e52b5305c630f4ee.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/727e8a98e33b28603f560058b8d8a575d8647e52f2223963e52b5305c630f4ee.jpg)

![789fd92690effc93af291d0d08697c60e805b9b3c05d7051bb32248ed75a67ce.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/789fd92690effc93af291d0d08697c60e805b9b3c05d7051bb32248ed75a67ce.jpg)

![8717170930f337075e8ce023a926776ac8a1ce91cc19c2c36ccd981ed41270cb.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/8717170930f337075e8ce023a926776ac8a1ce91cc19c2c36ccd981ed41270cb.jpg)

![8a815f2365080bb81e43267b8a01f6d517bf2e7a196dec32428674a8d839f3c4.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/8a815f2365080bb81e43267b8a01f6d517bf2e7a196dec32428674a8d839f3c4.jpg)

![8b9216d4fc9f748888c6c1f7746451c20777c9f31105031d403ad1d99b3e7f76.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/8b9216d4fc9f748888c6c1f7746451c20777c9f31105031d403ad1d99b3e7f76.jpg)

![9344a5629519c26449340378dfb1ae785f9a104aa316d32daea502d4be7202cc.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/9344a5629519c26449340378dfb1ae785f9a104aa316d32daea502d4be7202cc.jpg)

![98821807693e516fb200743cbfb94fb8794455de4aa6823cb1471c19ded69526.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/98821807693e516fb200743cbfb94fb8794455de4aa6823cb1471c19ded69526.jpg)

![a0d106f1fed2a77d79ee37148d2c44e713c3d33ac3936b3bbd478a3ac0d5b528.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/a0d106f1fed2a77d79ee37148d2c44e713c3d33ac3936b3bbd478a3ac0d5b528.jpg)

![aaf94a8b1242fc8c1e157e7ddc853020ddf212b25faca208a1569c1c424db914.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/aaf94a8b1242fc8c1e157e7ddc853020ddf212b25faca208a1569c1c424db914.jpg)

![c6a1a8552495007ba51e0402070b3c237b5ed7a7033317412c9b647e1886d12b.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/c6a1a8552495007ba51e0402070b3c237b5ed7a7033317412c9b647e1886d12b.jpg)

![c8186a01fb8e1735174b00f08477fe2cfdaef7996ada4c5e89eaccd7bb36239e.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/c8186a01fb8e1735174b00f08477fe2cfdaef7996ada4c5e89eaccd7bb36239e.jpg)

![e27d501f9cfe08e5039cf2213a31e26cec9fd42faec2e5338c39d2617957500f.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/e27d501f9cfe08e5039cf2213a31e26cec9fd42faec2e5338c39d2617957500f.jpg)

![e30449da837b2506b69515a35a53283b8cb0a2c5443b0288b385ed5f33a86238.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/e30449da837b2506b69515a35a53283b8cb0a2c5443b0288b385ed5f33a86238.jpg)

![eae153fd796ef163ad5e7c5d2f5f56d965c09984006d9924e1ab324a02c51b75.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/eae153fd796ef163ad5e7c5d2f5f56d965c09984006d9924e1ab324a02c51b75.jpg)

![ece70be24eb9d16dac694f776aa4f768ef6e43936843b09fd1e965b65641fc5e.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/ece70be24eb9d16dac694f776aa4f768ef6e43936843b09fd1e965b65641fc5e.jpg)

![f368c65ad8a8324eec20d954fbf1a8ea09a51f287f6dd60455c01d561a04b73a.jpg](../icml_results/2439_Block-wise%20Codeword%20Embedding%20for%20Reliable%20Multi-bit%20Text%20Watermarking/tables/f368c65ad8a8324eec20d954fbf1a8ea09a51f287f6dd60455c01d561a04b73a.jpg)
