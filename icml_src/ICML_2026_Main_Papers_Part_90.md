# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 90 of 197

## 目录 (Table of Contents)

1. [Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies](#paper-2938)
2. [World-Shaper: A Unified Framework for 360\textdegree Panoramic Editing](#paper-2939)
3. [PRISM: Learning Realistic Depth via Physics-Grounded Noise Disentanglement with Semantic-Geometric Collaboration](#paper-2940)
4. [Task-Aware Exploration via a Predictive Bisimulation Metric](#paper-2941)
5. [M+Adam: Low-Precision Training via Additive–Multiplicative Optimization](#paper-2942)
6. [REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations](#paper-2943)
7. [Bottleneck-Guided Spectral Subgoals For Offline Goal-Conditioned RL](#paper-2944)
8. [Implicit Action Chunking for Smooth Continuous Control](#paper-2945)
9. [Strategy Executability in Mathematical Reasoning: Leveraging Human–Model Differences for Effective Guidance](#paper-2946)
10. [Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization](#paper-2947)
11. [FOCUS: DLLMs Know How to Tame Their Compute Bound](#paper-2948)
12. [VAnim: Rendering-Aware Sparse State Modeling for Structure-Preserving Vector Animation](#paper-2949)
13. [Multilingual Safety Alignment Via Sparse Weight Editing](#paper-2950)
14. [CL-GCL: Comprehensive and Lightweight Graph Contrastive Learning](#paper-2951)
15. [SceneDirector: Bridging Explicit Geometry and Generative Priors for Unified Driving Scene Editing](#paper-2952)
16. [LineageFlow: Flow Matching for High-Fidelity Family-Aware Protein Sequence Generation](#paper-2953)
17. [UniFLoW: Universal Multi-Modal Federated LoRA Fine-Tuning Framework with Analytical Aggregation](#paper-2954)
18. [Large Language Model Teaches Visual Students: Cross-Modality Transfer of Fine-Grained Conceptual Knowledge](#paper-2955)
19. [RealisMotion: Decomposed Human Motion Control and Video Generation in the World Space](#paper-2956)
20. [SpikingLM: Towards Fully Spiking Language Model](#paper-2957)
21. [G-RANS: Generalizable Residual-Aware Neural Solvers for Sparse Systems](#paper-2958)
22. [AlphaRouter: Token-level Routing Between SLM and LLM with Reinforcement Learning and Tree Search](#paper-2959)
23. [VERA-V: Variational Inference Framework for Jailbreaking Vision-Language Models](#paper-2960)
24. [Frequency Matching in Spiking Neural Networks for mmWave Sensing](#paper-2961)
25. [Risk-Averse and Optimistic Advertiser Incentive Compatibility in Auto-bidding](#paper-2962)
26. [L-SR1: Learned Symmetric-Rank-One Preconditioning](#paper-2963)
27. [KIStego: Key-Independent Secure Image Distribution via Bipartite Structural Invariants](#paper-2964)
28. [Flow Equivariant World Models: Structured Memory for Dynamic Environments](#paper-2965)
29. [Rethinking Attention in Spiking Transformers: Overcoming Density Bias with Set Similarity](#paper-2966)
30. [QuITE: Query-Based Irregular Time Series Embedding](#paper-2967)
31. [GADA: Geometry-Aware Deformable Aggregation for Image-Based Gaussian Splatting](#paper-2968)
32. [Action Manifold Smoothing: A Lipschitz Pathway Perspective on High-Dimensional Reinforcement Learning](#paper-2969)
33. [Empirical Gaussian Processes](#paper-2970)

---


<a id="paper-2938"></a>

## Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies

### Images

![18a89e847eee25f47d9edf05dc99aedb3552614c2157d0cd512ea94bf851cf4d.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/18a89e847eee25f47d9edf05dc99aedb3552614c2157d0cd512ea94bf851cf4d.jpg)

![4554f035eb72fab14e37d3fbb96ea00a2059771cd52e1f812d5440d70098453d.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/4554f035eb72fab14e37d3fbb96ea00a2059771cd52e1f812d5440d70098453d.jpg)

![4a9caec5637a286f6f5889b5fb70799632ec590f5dbd19e7423001c279247fcb.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/4a9caec5637a286f6f5889b5fb70799632ec590f5dbd19e7423001c279247fcb.jpg)

![7ef46a1d85687f120380fc1b238adafb6c73dc664c89339b03cc2b02de818d73.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/7ef46a1d85687f120380fc1b238adafb6c73dc664c89339b03cc2b02de818d73.jpg)

![a4f1b6d00ecf025805a4a70764aad78a7a56c6663fddd92ad78119dec7586da5.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/a4f1b6d00ecf025805a4a70764aad78a7a56c6663fddd92ad78119dec7586da5.jpg)

![b6a5ac96ecff647baf8d3c24ba776186d93777582c450140b60799b436fd22b2.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/b6a5ac96ecff647baf8d3c24ba776186d93777582c450140b60799b436fd22b2.jpg)

![d1fb85295bfe7ceba4087d3363a48faa45e960c31b172cf5231274d4d57fc836.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/d1fb85295bfe7ceba4087d3363a48faa45e960c31b172cf5231274d4d57fc836.jpg)

![dd10d7de0aaf0d17b8865284a6ffc234eb7240d482c13f956b3f30abcb040ba2.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/dd10d7de0aaf0d17b8865284a6ffc234eb7240d482c13f956b3f30abcb040ba2.jpg)

![dd5ac3f96848a194716325efd15ca839b90487647f095d51ece70cc74e75bfc7.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/dd5ac3f96848a194716325efd15ca839b90487647f095d51ece70cc74e75bfc7.jpg)

![e89f874d57517b602ab34244ec78370b386304a89ce44f97cb2ccc4214912e65.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/images/e89f874d57517b602ab34244ec78370b386304a89ce44f97cb2ccc4214912e65.jpg)

### Tables

![040b0055468a5a08408050d4774ef4245a6dc398bd7ae49b81905003745244cc.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/tables/040b0055468a5a08408050d4774ef4245a6dc398bd7ae49b81905003745244cc.jpg)

![1840e5a40698e215532b7619f522bd8246c0e94a8c97aef6a6df439b7021efb3.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/tables/1840e5a40698e215532b7619f522bd8246c0e94a8c97aef6a6df439b7021efb3.jpg)

![29c69633a41dd722a17d5fbb19de837117f7465c36a6ae49b60960242127a5fb.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/tables/29c69633a41dd722a17d5fbb19de837117f7465c36a6ae49b60960242127a5fb.jpg)

![33c713ed517bdbf64f1fe932ff04769198d2d8e48f786facd48faf8fe174706e.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/tables/33c713ed517bdbf64f1fe932ff04769198d2d8e48f786facd48faf8fe174706e.jpg)

![3b088a239cad15cf990cb1e6f9751092cd215186f9303c02abf78814b2306a64.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/tables/3b088a239cad15cf990cb1e6f9751092cd215186f9303c02abf78814b2306a64.jpg)

![4ee34f024d77786d6c700f89baeef7109837d233da01d816476bf78d551b482a.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/tables/4ee34f024d77786d6c700f89baeef7109837d233da01d816476bf78d551b482a.jpg)

![5e37eab0086a0708cb64f1ff09f5727807f4babf82849667c0a17253f56be5af.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/tables/5e37eab0086a0708cb64f1ff09f5727807f4babf82849667c0a17253f56be5af.jpg)

![63d670bae28d115dfef68d20d53f13bb8b3d47e935e6e4af3287afc614175599.jpg](../icml_results/2975_AnyMod-LLVE_%20Low-Light%20Video%20Enhancement%20with%20Modality-Agnostic%20Inference/tables/63d670bae28d115dfef68d20d53f13bb8b3d47e935e6e4af3287afc614175599.jpg)

## Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies


### Images

![028c92375238e196d6eaf7e96e2cbac8eedef9bdbfc11a954756a6058787131b.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/images/028c92375238e196d6eaf7e96e2cbac8eedef9bdbfc11a954756a6058787131b.jpg)

![1afd1e4cac2a222d9754518a9e3a8957d3128a29f9b1aaf2006a149a0407cde3.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/images/1afd1e4cac2a222d9754518a9e3a8957d3128a29f9b1aaf2006a149a0407cde3.jpg)

![1def9f03b26a414ca7507f4da7fa471f3e44487b736ef1a2d8cfd8e2195e6608.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/images/1def9f03b26a414ca7507f4da7fa471f3e44487b736ef1a2d8cfd8e2195e6608.jpg)

![2cda703db716be4c0d4c265c6b1115a8ef4d8b4d0cc6672bd7b3eca064dd0a01.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/images/2cda703db716be4c0d4c265c6b1115a8ef4d8b4d0cc6672bd7b3eca064dd0a01.jpg)

![41204e8493c85d98bc759f4b98a92f2f28096cae3103ba0a81e63b7f93f8fdd9.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/images/41204e8493c85d98bc759f4b98a92f2f28096cae3103ba0a81e63b7f93f8fdd9.jpg)

![4acd2c3f0adc65b2e1172706f6029b6364ef133b937a593211e1195626e7782d.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/images/4acd2c3f0adc65b2e1172706f6029b6364ef133b937a593211e1195626e7782d.jpg)

![4bd4d566ebb19811f8c3c91ec0f2907f8ffabc899b57628086ebb52da278ba73.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/images/4bd4d566ebb19811f8c3c91ec0f2907f8ffabc899b57628086ebb52da278ba73.jpg)

![53004dfe253235123362ff9fbd17d5d695c29cc51692a0fd98c62ffc160cc731.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/images/53004dfe253235123362ff9fbd17d5d695c29cc51692a0fd98c62ffc160cc731.jpg)

![62692089535adc94a0085e984f40ad4f51380faa93e8ea6c6d53b9acfba22caa.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/images/62692089535adc94a0085e984f40ad4f51380faa93e8ea6c6d53b9acfba22caa.jpg)

![96169db30c84498cf08fd8b0a9f4d23ed89e55dc6d8965dd1b90326b7d5b646c.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/images/96169db30c84498cf08fd8b0a9f4d23ed89e55dc6d8965dd1b90326b7d5b646c.jpg)

![af171bef103ec6cfb96ddb233dd792859b8f91d7d484bb821cf1b3ac4aceeb23.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/images/af171bef103ec6cfb96ddb233dd792859b8f91d7d484bb821cf1b3ac4aceeb23.jpg)

![d9e2b5fcfd956ad79475da80069e1bb022e3effca9250ee7918632ca2162edfc.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/images/d9e2b5fcfd956ad79475da80069e1bb022e3effca9250ee7918632ca2162edfc.jpg)

![e08c554ecf3841c33b60cb36f1ff1fae808bccae2a68e906a267da10ae3024e3.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/images/e08c554ecf3841c33b60cb36f1ff1fae808bccae2a68e906a267da10ae3024e3.jpg)

![ecefc01ab2e9000928d5df63956ad3edb95eb0298eed84b7f07147646565d809.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/images/ecefc01ab2e9000928d5df63956ad3edb95eb0298eed84b7f07147646565d809.jpg)

![f0b2c433e9df569f4806fd1b476f264dac06b1d684c9b1959ead04cf2140c3ea.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/images/f0b2c433e9df569f4806fd1b476f264dac06b1d684c9b1959ead04cf2140c3ea.jpg)

### Tables

![015218097781664388a120326e7a5ebfba7f7dfd6eea38fe98a4e5df903ca747.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/tables/015218097781664388a120326e7a5ebfba7f7dfd6eea38fe98a4e5df903ca747.jpg)

![23156a478745cefcdd91e002d7b058935ed909713322e04d651954e938e1c1ac.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/tables/23156a478745cefcdd91e002d7b058935ed909713322e04d651954e938e1c1ac.jpg)

![332b0615845a995e3ce68941eedfd71d7f9694f990a577fec4645bab30109d1f.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/tables/332b0615845a995e3ce68941eedfd71d7f9694f990a577fec4645bab30109d1f.jpg)

![39ade40443d9cd6a520c899683f11bc8407b005445b66bfcd21a1ee167255fe5.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/tables/39ade40443d9cd6a520c899683f11bc8407b005445b66bfcd21a1ee167255fe5.jpg)

![3de96b0d2050e71832a95aef0b4d6d0f6121b0f8ca39451a0428b439c8802e9e.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/tables/3de96b0d2050e71832a95aef0b4d6d0f6121b0f8ca39451a0428b439c8802e9e.jpg)

![411df5e50a4744bcfcc06db82c30bb63e55d06872677fac2e82928a4ab081f3c.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/tables/411df5e50a4744bcfcc06db82c30bb63e55d06872677fac2e82928a4ab081f3c.jpg)

![49816e9cf540d1e4c4beaa2608d9aba3603b90a23c7b419d62ca39abc30ec2e3.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/tables/49816e9cf540d1e4c4beaa2608d9aba3603b90a23c7b419d62ca39abc30ec2e3.jpg)

![5507b5fa3c0d6a3ea0fa1e892534482c17b1f888b66d8687f1314faa9cc5bc6e.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/tables/5507b5fa3c0d6a3ea0fa1e892534482c17b1f888b66d8687f1314faa9cc5bc6e.jpg)

![5668177ceb4c71cd84758a4725a1ffd7884c950250d3c77eace2811ae467f116.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/tables/5668177ceb4c71cd84758a4725a1ffd7884c950250d3c77eace2811ae467f116.jpg)

![665996820d043ba400e587fa11838c23f3fa3de9849deb457ddd7bf8396140eb.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/tables/665996820d043ba400e587fa11838c23f3fa3de9849deb457ddd7bf8396140eb.jpg)

![c1f0143cc8883aae392f24d857d2e482fd037ed677cde856b1be4d16c9bd681d.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/tables/c1f0143cc8883aae392f24d857d2e482fd037ed677cde856b1be4d16c9bd681d.jpg)

![c27e515522c1561c3e81978836bf31bea908dafc136fff660cfbcf364c8228f0.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/tables/c27e515522c1561c3e81978836bf31bea908dafc136fff660cfbcf364c8228f0.jpg)

![f5ab79bf8190e128fd39decf29bda38b134439ec6e4301b42b759ce00a34ebf7.jpg](../icml_results/2976_Discrete%20Diffusion%20VLA_%20Bringing%20Discrete%20Diffusion%20to%20Action%20Decoding%20in%20Vision-Language-Action%20Pol/tables/f5ab79bf8190e128fd39decf29bda38b134439ec6e4301b42b759ce00a34ebf7.jpg)

<a id="paper-2939"></a>

## World-Shaper: A Unified Framework for 360\textdegree Panoramic Editing


### Images

![0145e0223500775cac07f7f6c92ca9c53df34213c79a803250268729060631cf.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/0145e0223500775cac07f7f6c92ca9c53df34213c79a803250268729060631cf.jpg)

![0691fb7e75b51967135d425b0d539f2ba8c95dd50286518fa322110e689ef43f.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/0691fb7e75b51967135d425b0d539f2ba8c95dd50286518fa322110e689ef43f.jpg)

![0cc0ef2e890d1e37c152f19179a9ec858a1182f9fef730edc9feca0feebf0c5a.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/0cc0ef2e890d1e37c152f19179a9ec858a1182f9fef730edc9feca0feebf0c5a.jpg)

![0f051add59287fbba02d6fad242ac67162d21426381156898625b9c91370fe95.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/0f051add59287fbba02d6fad242ac67162d21426381156898625b9c91370fe95.jpg)

![16fb4f8925d32d26dfd9128711812a026eb1b07f4d8d2528aa58b26f0f0ca0ae.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/16fb4f8925d32d26dfd9128711812a026eb1b07f4d8d2528aa58b26f0f0ca0ae.jpg)

![2232ea3b8ccaf32c4ca959e9e62346a5cb37425132abf780f39ba77c77990858.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/2232ea3b8ccaf32c4ca959e9e62346a5cb37425132abf780f39ba77c77990858.jpg)

![2bb36bfc72f8dfa34dba8ae2c30e32aeafcacf1fe45ccf863d4cffb29fecb2ae.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/2bb36bfc72f8dfa34dba8ae2c30e32aeafcacf1fe45ccf863d4cffb29fecb2ae.jpg)

![2d3d5b2e5c9ce65f7dc290414ea5334b208a8afe6ce53590a4716a06fd6a21e0.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/2d3d5b2e5c9ce65f7dc290414ea5334b208a8afe6ce53590a4716a06fd6a21e0.jpg)

![312597b9c37001350bc848abf3024199c7d33f24fc18b9a4fe8806172bed277a.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/312597b9c37001350bc848abf3024199c7d33f24fc18b9a4fe8806172bed277a.jpg)

![31f26b7903d3566459dc18f41b187f31897d9a7e8829d7bbdf6c219ece9fdfc3.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/31f26b7903d3566459dc18f41b187f31897d9a7e8829d7bbdf6c219ece9fdfc3.jpg)

![40175bde99382c3240ee6f27a6255c5cc9e0141057944465b6d6cfb52945819e.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/40175bde99382c3240ee6f27a6255c5cc9e0141057944465b6d6cfb52945819e.jpg)

![4488cc1a6c5d92910200708cf1c9ac079ec77312bd4ada0a2a41bf4dd2752fca.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/4488cc1a6c5d92910200708cf1c9ac079ec77312bd4ada0a2a41bf4dd2752fca.jpg)

![46a822fe860fd4953b5ce8f9e2cadbbd1da44071c2e181309b1be34921b15ebe.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/46a822fe860fd4953b5ce8f9e2cadbbd1da44071c2e181309b1be34921b15ebe.jpg)

![477238b139916907198b4e51af4effd598af027e4912880c47517bf6e4b7ced8.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/477238b139916907198b4e51af4effd598af027e4912880c47517bf6e4b7ced8.jpg)

![4b18004d36fa153bd97d8ba58e65c61770ddbbb7d3a26e2de03e19c3fc252edf.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/4b18004d36fa153bd97d8ba58e65c61770ddbbb7d3a26e2de03e19c3fc252edf.jpg)

![5876fb4c26e101eba3e0cfddfac49b7c7d0bff506f383e3de039831f86210af3.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/5876fb4c26e101eba3e0cfddfac49b7c7d0bff506f383e3de039831f86210af3.jpg)

![5a5af7008903f86dde0e947a92ce78e40d7c355e6abb5e47491e8f23b9e8aaa1.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/5a5af7008903f86dde0e947a92ce78e40d7c355e6abb5e47491e8f23b9e8aaa1.jpg)

![5ebfbc698ee738c25089166b04d88dee805e73449970b942e670387df5c47363.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/5ebfbc698ee738c25089166b04d88dee805e73449970b942e670387df5c47363.jpg)

![5fc20c57866738c4aef46d55b08936021c006cbb6ffe993532a97060e4393e34.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/5fc20c57866738c4aef46d55b08936021c006cbb6ffe993532a97060e4393e34.jpg)

![64488eea37999fa2d59cc16d52c020b486487f442146f177c43e1ff72b95bd20.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/64488eea37999fa2d59cc16d52c020b486487f442146f177c43e1ff72b95bd20.jpg)

![647a7c83e388966328350dd1a0924f2dab0b8ce20bab039def0ca2a0f9dac1a1.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/647a7c83e388966328350dd1a0924f2dab0b8ce20bab039def0ca2a0f9dac1a1.jpg)

![6b07efc82b28f50cad92cb69ad75960a5c42340f6650245c552a05df6bfe0ea9.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/6b07efc82b28f50cad92cb69ad75960a5c42340f6650245c552a05df6bfe0ea9.jpg)

![70503cf4cea7c2f0938767e18cb219dc31c4af900bc4fe966aa1549aa95b76b3.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/70503cf4cea7c2f0938767e18cb219dc31c4af900bc4fe966aa1549aa95b76b3.jpg)

![709c1ffecda0f2a63074ab58c78c488720fe632e61d157c945902621dabc5a37.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/709c1ffecda0f2a63074ab58c78c488720fe632e61d157c945902621dabc5a37.jpg)

![767cca1b962a6ce41e25dec014475b3d95d6f8f82e7271307def74b0c6570f64.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/767cca1b962a6ce41e25dec014475b3d95d6f8f82e7271307def74b0c6570f64.jpg)

![77451623235e51b9a22c535749df5df6a3dded6a49a572a94d1d953f69209429.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/77451623235e51b9a22c535749df5df6a3dded6a49a572a94d1d953f69209429.jpg)

![801a26355e7a8620f096fabc5b42cd75310728e9d2f0ff1c1f38c3f9401ac41f.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/801a26355e7a8620f096fabc5b42cd75310728e9d2f0ff1c1f38c3f9401ac41f.jpg)

![8e2bb6b67902acbcd72e0e453a6ef08c0fb4249f8187b341e49b7825ad36e1f5.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/8e2bb6b67902acbcd72e0e453a6ef08c0fb4249f8187b341e49b7825ad36e1f5.jpg)

![8f9dc6d1029ab9cbab83da4898a7f1ea4144700232849435d543febc5b4aed55.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/8f9dc6d1029ab9cbab83da4898a7f1ea4144700232849435d543febc5b4aed55.jpg)

![9ad0402e63ab6d937bcc03ba39ed6f751b48783835e22b89f7c33ddb26054cb2.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/9ad0402e63ab6d937bcc03ba39ed6f751b48783835e22b89f7c33ddb26054cb2.jpg)

![a22aef8be856c65b9ec8a559593ed3b93efd737bb588e0d4659a1a1d17a5ccce.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/a22aef8be856c65b9ec8a559593ed3b93efd737bb588e0d4659a1a1d17a5ccce.jpg)

![a2a018e88d9f107876128e37731bb3ed23b3294c5088d500faa9f7510e8d8a80.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/a2a018e88d9f107876128e37731bb3ed23b3294c5088d500faa9f7510e8d8a80.jpg)

![a83c92d8c4c1a890a6f6e2deeb8a2a2e91cd6755efae6fb32b2695c6d0869068.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/a83c92d8c4c1a890a6f6e2deeb8a2a2e91cd6755efae6fb32b2695c6d0869068.jpg)

![adf40b0586f192ace5ae96f2a016728ac97799c368fcf317f88dbbb72917522c.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/adf40b0586f192ace5ae96f2a016728ac97799c368fcf317f88dbbb72917522c.jpg)

![b5e4608a481dc4f4490d3a3f57bd8f746b2b329f1edd63ef7eb332c5490299ff.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/b5e4608a481dc4f4490d3a3f57bd8f746b2b329f1edd63ef7eb332c5490299ff.jpg)

![dd4e871791706d687a3e44e845e5cbd2173a264cbf63eaa82240ac2d7b210d0c.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/dd4e871791706d687a3e44e845e5cbd2173a264cbf63eaa82240ac2d7b210d0c.jpg)

![e87a7246ad27d6d0d83d1414fed8312d1fbbd3983e97cae9268dd5583061813e.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/e87a7246ad27d6d0d83d1414fed8312d1fbbd3983e97cae9268dd5583061813e.jpg)

![ed42ad2cd532e6ea8d0cd9c9c153b0271cd3c479faa3af5769f3c3cdb4b4dd53.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/ed42ad2cd532e6ea8d0cd9c9c153b0271cd3c479faa3af5769f3c3cdb4b4dd53.jpg)

![ee7702932f22763b668ff84c08f808878e82e5db5bd7f80be25d1e2d424e8942.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/ee7702932f22763b668ff84c08f808878e82e5db5bd7f80be25d1e2d424e8942.jpg)

![eef337334a41c1aac930548ce172316f1801e0c94f20a68d180d6e8b0f7ed131.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/eef337334a41c1aac930548ce172316f1801e0c94f20a68d180d6e8b0f7ed131.jpg)

![f822472d2da04c1024b2f4c9ebf3ca5165d4e2d52ec24dce18c3bc63bc301e13.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/f822472d2da04c1024b2f4c9ebf3ca5165d4e2d52ec24dce18c3bc63bc301e13.jpg)

![fab7fa0e3b6b2ec84ceb6f73c8df73fc418c90122138e6d441ad829e60c30ca3.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/images/fab7fa0e3b6b2ec84ceb6f73c8df73fc418c90122138e6d441ad829e60c30ca3.jpg)

### Tables

![042973d634ec78bf1b7b9c1851b1e5e6637b3ff5b1ada6482c052f0c04ceb4dc.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/tables/042973d634ec78bf1b7b9c1851b1e5e6637b3ff5b1ada6482c052f0c04ceb4dc.jpg)

![274f839f4c479392542308a3df966ea9f832d2c89cc67de9fc4eeb7cb648f141.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/tables/274f839f4c479392542308a3df966ea9f832d2c89cc67de9fc4eeb7cb648f141.jpg)

![387303954ad88b13b80d4004aa12e3b7052a1a79ee318f7673d71e17e9d6188f.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/tables/387303954ad88b13b80d4004aa12e3b7052a1a79ee318f7673d71e17e9d6188f.jpg)

![390ce69b89c57196c3f4863c4be7086f060304773065941dcb671939c11e5906.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/tables/390ce69b89c57196c3f4863c4be7086f060304773065941dcb671939c11e5906.jpg)

![58ab6faf20ca8cfc58d777b15fa0d56a4f02d6ddf1baa0ae7095e5eb8712169d.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/tables/58ab6faf20ca8cfc58d777b15fa0d56a4f02d6ddf1baa0ae7095e5eb8712169d.jpg)

![68837cf269c6ef8bb59181aacf07b7d7836ee63801abf38720bdc11da7453c44.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/tables/68837cf269c6ef8bb59181aacf07b7d7836ee63801abf38720bdc11da7453c44.jpg)

![a52b16a508d3a009bf2adde7fa03375167c08f236edd6d28a2ca975923b70b1a.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/tables/a52b16a508d3a009bf2adde7fa03375167c08f236edd6d28a2ca975923b70b1a.jpg)

![ba1218eba8a8dda98b75d32e30abc66fe4b6a373b1b9b976d24f3b5324f33f88.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/tables/ba1218eba8a8dda98b75d32e30abc66fe4b6a373b1b9b976d24f3b5324f33f88.jpg)

![d1b5f844738d2ba37cdfe90ed63709087afe19831f2a6e6a5254ae2c4629fbee.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/tables/d1b5f844738d2ba37cdfe90ed63709087afe19831f2a6e6a5254ae2c4629fbee.jpg)

![dc5ce12ecf6cb5d445e4d4ca6341cdd9e9f01ad438b97461785e0e40b151d1c0.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/tables/dc5ce12ecf6cb5d445e4d4ca6341cdd9e9f01ad438b97461785e0e40b151d1c0.jpg)

![e32a66c3abc45468725160bd9843ad37f5b227430ab7e3ee9d99fc266586c04d.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/tables/e32a66c3abc45468725160bd9843ad37f5b227430ab7e3ee9d99fc266586c04d.jpg)

![e46ba558c3c285509a44c4880c39b226d7c859a85d976a14386d46cd5713a76a.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/tables/e46ba558c3c285509a44c4880c39b226d7c859a85d976a14386d46cd5713a76a.jpg)

![fbfb99732177eb6714d7c939e05a0e5f60bc9fa5c88178565a38cfd8218f6029.jpg](../icml_results/2977_World-Shaper_%20A%20Unified%20Framework%20for%20360_textdegree%20Panoramic%20Editing/tables/fbfb99732177eb6714d7c939e05a0e5f60bc9fa5c88178565a38cfd8218f6029.jpg)

<a id="paper-2940"></a>

## PRISM: Learning Realistic Depth via Physics-Grounded Noise Disentanglement with Semantic-Geometric Collaboration


### Images

![0fdc971d78a4e0a8296dc3d4d6e7881e46d1558ef69a851fc34cab2082266bba.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/0fdc971d78a4e0a8296dc3d4d6e7881e46d1558ef69a851fc34cab2082266bba.jpg)

![1a2a57a35d8b44b449446ea8114fc0b0164b005d9c2b56ea55a664d44dd41fd3.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/1a2a57a35d8b44b449446ea8114fc0b0164b005d9c2b56ea55a664d44dd41fd3.jpg)

![4a5e8c87f3378bfe875d7b53db5583a6de554363d98756fc68dee83018bdde9a.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/4a5e8c87f3378bfe875d7b53db5583a6de554363d98756fc68dee83018bdde9a.jpg)

![4a897b9388aae8d926212abe28963ac378016a4ea0480696a8de549efef7df1d.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/4a897b9388aae8d926212abe28963ac378016a4ea0480696a8de549efef7df1d.jpg)

![4cdb4d363e7face1bbe6fb266d0784e472cecaa66c59c094be20b11d72e6d6a9.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/4cdb4d363e7face1bbe6fb266d0784e472cecaa66c59c094be20b11d72e6d6a9.jpg)

![508581fa36c5e0ea59c3143a5cd8b758df61d87a2a083044b2a380d36a4cacdb.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/508581fa36c5e0ea59c3143a5cd8b758df61d87a2a083044b2a380d36a4cacdb.jpg)

![5c74cd16d7612a35f0c42c5dbd8be10a9002346088a905fdc41abfd60604b788.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/5c74cd16d7612a35f0c42c5dbd8be10a9002346088a905fdc41abfd60604b788.jpg)

![71bf47a0f159cb7fef5acc1cfafddf546b0ab29ee004482c20162a06d920a6fe.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/71bf47a0f159cb7fef5acc1cfafddf546b0ab29ee004482c20162a06d920a6fe.jpg)

![75628f1b0695669a9306670f0de70c8c498d42928fc33d403fe2378aba1f3b97.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/75628f1b0695669a9306670f0de70c8c498d42928fc33d403fe2378aba1f3b97.jpg)

![7586ee51e0ed006f6e13c32adb27952835a029416c865bea6a5c8f1640fef862.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/7586ee51e0ed006f6e13c32adb27952835a029416c865bea6a5c8f1640fef862.jpg)

![7d0b08d8e7217a8ba02767e1c4ddd05b86cbc53ac6f2a067d94d0028eb85671f.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/7d0b08d8e7217a8ba02767e1c4ddd05b86cbc53ac6f2a067d94d0028eb85671f.jpg)

![95ac7685ba2375a2d89f7adc11399d8d5b3f488ee0472a41168caa368830b7f0.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/95ac7685ba2375a2d89f7adc11399d8d5b3f488ee0472a41168caa368830b7f0.jpg)

![9fb2da1e3c53ac38e652d3d461fd21da44414cfb0f853d8f6661ce6c67f5626e.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/9fb2da1e3c53ac38e652d3d461fd21da44414cfb0f853d8f6661ce6c67f5626e.jpg)

![b9aa667ccb6cfe23bc55157c2fbf3533980bbb04043b24c390e562bbba62f9e9.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/b9aa667ccb6cfe23bc55157c2fbf3533980bbb04043b24c390e562bbba62f9e9.jpg)

![bcea1ba70243faafa19d88db617051853446884d60f63c3c430af9628895915f.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/bcea1ba70243faafa19d88db617051853446884d60f63c3c430af9628895915f.jpg)

![cd3292cd2a38941a9e4a3143a0f23c4baa78af552435210f32823eff9fb696e9.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/cd3292cd2a38941a9e4a3143a0f23c4baa78af552435210f32823eff9fb696e9.jpg)

![cf4a6fbe1c7eeb0c5560522d53ff966377311da9312f6d57f19687289380c991.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/cf4a6fbe1c7eeb0c5560522d53ff966377311da9312f6d57f19687289380c991.jpg)

![de5f152c68f739a20f6d87012b1532edf56a4038c9030525086e2b5d77746050.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/de5f152c68f739a20f6d87012b1532edf56a4038c9030525086e2b5d77746050.jpg)

![fb3f5414575ca1d3a70e389ce4d0e8843bcc92cd66cbf1eaff8e5f13f6c851f6.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/fb3f5414575ca1d3a70e389ce4d0e8843bcc92cd66cbf1eaff8e5f13f6c851f6.jpg)

![ff3fcda5497f45d09659881ada40790f6587c34c20853c0296954c2dcafbf11d.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/images/ff3fcda5497f45d09659881ada40790f6587c34c20853c0296954c2dcafbf11d.jpg)

### Tables

![07d7f8ed5934d4ada7869e0b200e9059522b92cbe316600947e1f99980588c6a.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/07d7f8ed5934d4ada7869e0b200e9059522b92cbe316600947e1f99980588c6a.jpg)

![08af4c72ffff79cd8de104af6600a2e36fd75331c61f3db15e627f1a4845e2e3.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/08af4c72ffff79cd8de104af6600a2e36fd75331c61f3db15e627f1a4845e2e3.jpg)

![0c423ac0aacdc6dc9d2551c60829c40487aa2074c6ba53fc90833fc8ea054b90.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/0c423ac0aacdc6dc9d2551c60829c40487aa2074c6ba53fc90833fc8ea054b90.jpg)

![37d2821eea4cfed9e34c556356f98a71d0be1d847f925aa77ea86e2f2f966666.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/37d2821eea4cfed9e34c556356f98a71d0be1d847f925aa77ea86e2f2f966666.jpg)

![47d0c6790eae41ee745512851ec5c8b130eb2842993e703cbc288d6a17a27a58.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/47d0c6790eae41ee745512851ec5c8b130eb2842993e703cbc288d6a17a27a58.jpg)

![58a7c3897cfe6a9886e6912b9f64bb7dfaac7c2d8c5c768fe58e6e39b85bf692.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/58a7c3897cfe6a9886e6912b9f64bb7dfaac7c2d8c5c768fe58e6e39b85bf692.jpg)

![671b17ff5611055c7c2630060418cde5544c45d8d67c463e69a527d3b88b2618.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/671b17ff5611055c7c2630060418cde5544c45d8d67c463e69a527d3b88b2618.jpg)

![67ec803b8b33e449e66b698820200ee2668bb0c1c3ccd9206750789eae78681e.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/67ec803b8b33e449e66b698820200ee2668bb0c1c3ccd9206750789eae78681e.jpg)

![6b4b9d28f33126a96613f9436c469a34de94dd3139c9323d5fa5c8a907c8e516.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/6b4b9d28f33126a96613f9436c469a34de94dd3139c9323d5fa5c8a907c8e516.jpg)

![6ccf1158046f4c2d694c52d7cb42e2b93c2e02e6873236a71ae97b90c90bba84.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/6ccf1158046f4c2d694c52d7cb42e2b93c2e02e6873236a71ae97b90c90bba84.jpg)

![6eab7ed9dd3b50b2f9c49f04bcfc67316c1d8f6c32edb2ea8006e2f179312161.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/6eab7ed9dd3b50b2f9c49f04bcfc67316c1d8f6c32edb2ea8006e2f179312161.jpg)

![867cfb6c4d55fd6f472b6441a129a20ee9aff94b88124a8e2e3a73e8e921409d.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/867cfb6c4d55fd6f472b6441a129a20ee9aff94b88124a8e2e3a73e8e921409d.jpg)

![86ccc9b672345bc002bfc08997b48650dbbe9e0ff9655c32d3f63a6b9feb7fde.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/86ccc9b672345bc002bfc08997b48650dbbe9e0ff9655c32d3f63a6b9feb7fde.jpg)

![d940393e13ad6f7ea1517d54a4007b67e26c2a6c852d908810b6e3ff9fdabbfc.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/d940393e13ad6f7ea1517d54a4007b67e26c2a6c852d908810b6e3ff9fdabbfc.jpg)

![e004932fa81cc72e8c2b9afb41fe79bb4c6fa2224fb04bfc99064d46b8a0ab1b.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/e004932fa81cc72e8c2b9afb41fe79bb4c6fa2224fb04bfc99064d46b8a0ab1b.jpg)

![e7ac6b1adc9caf7fc9231f93ff85967d50cb5d1a6df35d68eb473cd766f0dd66.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/e7ac6b1adc9caf7fc9231f93ff85967d50cb5d1a6df35d68eb473cd766f0dd66.jpg)

![e8abd823583c0ce02b0e5b37a48b037fe6608adf2c22f2033f4eb2186ebb4d3c.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/e8abd823583c0ce02b0e5b37a48b037fe6608adf2c22f2033f4eb2186ebb4d3c.jpg)

![ef55f3a7fe7332a5faa6ab00649e0ea6a39874a7f5c765d74f1c02eed9036686.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/ef55f3a7fe7332a5faa6ab00649e0ea6a39874a7f5c765d74f1c02eed9036686.jpg)

![f5a1d5d59dd568143b08687c690ee972b3654ca85d7f022df04babdd57a92c56.jpg](../icml_results/2978_PRISM_%20Learning%20Realistic%20Depth%20via%20Physics-Grounded%20Noise%20Disentanglement%20with%20Semantic-Geometric%20C/tables/f5a1d5d59dd568143b08687c690ee972b3654ca85d7f022df04babdd57a92c56.jpg)

<a id="paper-2941"></a>

## Task-Aware Exploration via a Predictive Bisimulation Metric


### Images

![0fd24f102a5b731283dbe0b923c127f7da0fb357dec4796cd964f19ea69e18ce.jpg](../icml_results/2979_Task-Aware%20Exploration%20via%20a%20Predictive%20Bisimulation%20Metric/images/0fd24f102a5b731283dbe0b923c127f7da0fb357dec4796cd964f19ea69e18ce.jpg)

![19300768424c303cec0bfe9aa3e4b8314bfdfff7507322eb72b8112a5f095f71.jpg](../icml_results/2979_Task-Aware%20Exploration%20via%20a%20Predictive%20Bisimulation%20Metric/images/19300768424c303cec0bfe9aa3e4b8314bfdfff7507322eb72b8112a5f095f71.jpg)

![1e5cd01b64d09b01a5f62c4ce11a4d3b2436fa343aad3381283e981ae15815fa.jpg](../icml_results/2979_Task-Aware%20Exploration%20via%20a%20Predictive%20Bisimulation%20Metric/images/1e5cd01b64d09b01a5f62c4ce11a4d3b2436fa343aad3381283e981ae15815fa.jpg)

![1ee4327725334e701b7e635b9b9ec3b85da19f29cc644ce39ef73c16b8662fea.jpg](../icml_results/2979_Task-Aware%20Exploration%20via%20a%20Predictive%20Bisimulation%20Metric/images/1ee4327725334e701b7e635b9b9ec3b85da19f29cc644ce39ef73c16b8662fea.jpg)

![2ed19c25b02fe17859567b0e58978695618ca4aca8f1458a20e33eda3375b19b.jpg](../icml_results/2979_Task-Aware%20Exploration%20via%20a%20Predictive%20Bisimulation%20Metric/images/2ed19c25b02fe17859567b0e58978695618ca4aca8f1458a20e33eda3375b19b.jpg)

![70e36fc5417902b7d0d5da793732d5521b9708790c3d0792b3e4c7dbfaf5f9ed.jpg](../icml_results/2979_Task-Aware%20Exploration%20via%20a%20Predictive%20Bisimulation%20Metric/images/70e36fc5417902b7d0d5da793732d5521b9708790c3d0792b3e4c7dbfaf5f9ed.jpg)

![e48fb74fcce8ed9acfda7e426c35714f3cb79cbc5c83107667b74ad8fbf61915.jpg](../icml_results/2979_Task-Aware%20Exploration%20via%20a%20Predictive%20Bisimulation%20Metric/images/e48fb74fcce8ed9acfda7e426c35714f3cb79cbc5c83107667b74ad8fbf61915.jpg)

![e8b02f44142580bcb34551b879555ca46c8fcd8d6bc3b44b6aaa5fd2656ed485.jpg](../icml_results/2979_Task-Aware%20Exploration%20via%20a%20Predictive%20Bisimulation%20Metric/images/e8b02f44142580bcb34551b879555ca46c8fcd8d6bc3b44b6aaa5fd2656ed485.jpg)

### Tables

![6fe401c120185f3141bdce97139b306463d52574b19f3bf5c902001cb3136ed8.jpg](../icml_results/2979_Task-Aware%20Exploration%20via%20a%20Predictive%20Bisimulation%20Metric/tables/6fe401c120185f3141bdce97139b306463d52574b19f3bf5c902001cb3136ed8.jpg)

![7ea957b48bd0ddb84c03db35230196fab54e57f0408dd536aa3de1429b17153d.jpg](../icml_results/2979_Task-Aware%20Exploration%20via%20a%20Predictive%20Bisimulation%20Metric/tables/7ea957b48bd0ddb84c03db35230196fab54e57f0408dd536aa3de1429b17153d.jpg)

![b33a75176edf77e5a59a0a91f8f92d093599c3104875064326b446422ad61586.jpg](../icml_results/2979_Task-Aware%20Exploration%20via%20a%20Predictive%20Bisimulation%20Metric/tables/b33a75176edf77e5a59a0a91f8f92d093599c3104875064326b446422ad61586.jpg)

![f2a5c0460a73d3fc8d7baf902d29500b0d51aa368eb1a886ea013bd8f36e0d97.jpg](../icml_results/2979_Task-Aware%20Exploration%20via%20a%20Predictive%20Bisimulation%20Metric/tables/f2a5c0460a73d3fc8d7baf902d29500b0d51aa368eb1a886ea013bd8f36e0d97.jpg)

<a id="paper-2942"></a>

## M+Adam: Low-Precision Training via Additive–Multiplicative Optimization


### Images

![2520b7d36f67a601e99425dbd4c549af19e532d5619c275ffbdc60bf7adb17f5.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/images/2520b7d36f67a601e99425dbd4c549af19e532d5619c275ffbdc60bf7adb17f5.jpg)

![5673e9444d7768031c265440da7764053da5b00eddc9e242a321d95e8e5790b7.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/images/5673e9444d7768031c265440da7764053da5b00eddc9e242a321d95e8e5790b7.jpg)

![65822093fca32d0f1e5b6ff11b3510fb9598f1ac1d9e5faadbfb86cb68666f28.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/images/65822093fca32d0f1e5b6ff11b3510fb9598f1ac1d9e5faadbfb86cb68666f28.jpg)

![66c279e7e10c9ddc912a4685c484dbb72a238a64cda48f8575b66aa35fca3376.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/images/66c279e7e10c9ddc912a4685c484dbb72a238a64cda48f8575b66aa35fca3376.jpg)

![7cc5f0e32c40bc3eb1272b0f82f9363ad931deed18cd632738da81744a990bc9.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/images/7cc5f0e32c40bc3eb1272b0f82f9363ad931deed18cd632738da81744a990bc9.jpg)

![853f7ce26c4305601b05bfac4f512bccc10b8638a9166072bd257ef7e6d5419a.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/images/853f7ce26c4305601b05bfac4f512bccc10b8638a9166072bd257ef7e6d5419a.jpg)

![c1961b9544fc54f0e3eff7b92a9ea325abdf49b0bddc1c58399a80e4e91ca6e5.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/images/c1961b9544fc54f0e3eff7b92a9ea325abdf49b0bddc1c58399a80e4e91ca6e5.jpg)

![d2736045aa3b2e973a56ff99acb692b832d78150080db7c46abf61d599a5aaa1.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/images/d2736045aa3b2e973a56ff99acb692b832d78150080db7c46abf61d599a5aaa1.jpg)

![fde3b9b16482cdf7e873c1bf3b3034988eed73034d9ddf1635ce2ffb8c1b68f9.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/images/fde3b9b16482cdf7e873c1bf3b3034988eed73034d9ddf1635ce2ffb8c1b68f9.jpg)

### Tables

![023eb0ca3b35c9049ea10b0d3a459745a651d8f3a0785190285308f2c2f03a70.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/023eb0ca3b35c9049ea10b0d3a459745a651d8f3a0785190285308f2c2f03a70.jpg)

![0240cf3fc0f985e900e2e065afef4b5dd449395a1c470857bb00d5a3f0a74768.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/0240cf3fc0f985e900e2e065afef4b5dd449395a1c470857bb00d5a3f0a74768.jpg)

![0c589b6377e65dd61c08faaf2369fc4396925c61159c29cb461920f392ad8d5d.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/0c589b6377e65dd61c08faaf2369fc4396925c61159c29cb461920f392ad8d5d.jpg)

![0d34c4fe2e1871177b7841c24380ce82ad26f3ca65d08287e5d0d391da77b029.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/0d34c4fe2e1871177b7841c24380ce82ad26f3ca65d08287e5d0d391da77b029.jpg)

![101c79973c3c3392dc5f9e8ec64bca36b4d1f64cb7434abf9dfe5abb5b88377e.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/101c79973c3c3392dc5f9e8ec64bca36b4d1f64cb7434abf9dfe5abb5b88377e.jpg)

![10c17e8a228e0044b620fcfca28b9f0dbb6d2ea4994ee57372b2bb5b22ade1e3.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/10c17e8a228e0044b620fcfca28b9f0dbb6d2ea4994ee57372b2bb5b22ade1e3.jpg)

![15ca3d086c82172e295c0d80ecaad5e98f88783b302a219984210cb6a37f36b3.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/15ca3d086c82172e295c0d80ecaad5e98f88783b302a219984210cb6a37f36b3.jpg)

![1881015a295ad5ea5855aad75d99e9f3dd8cc0fef4ecfed67e5287180030ba45.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/1881015a295ad5ea5855aad75d99e9f3dd8cc0fef4ecfed67e5287180030ba45.jpg)

![19f77a3d25c2c4a05cda3c15eb5ea139931bdcffd2179f24a8fa5969ead9b68b.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/19f77a3d25c2c4a05cda3c15eb5ea139931bdcffd2179f24a8fa5969ead9b68b.jpg)

![1cf1cbf037bfa534edcfdde925613b7d086acb7037ef49f04037cb0812b2a334.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/1cf1cbf037bfa534edcfdde925613b7d086acb7037ef49f04037cb0812b2a334.jpg)

![1e52d10ed7b5cea2967ccbfb406d15212a868396da1d92414d6c46834034332e.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/1e52d10ed7b5cea2967ccbfb406d15212a868396da1d92414d6c46834034332e.jpg)

![2147dffd51d4e221f65610fd9e30856cc0c8893eabc775104b5007d9154d03e3.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/2147dffd51d4e221f65610fd9e30856cc0c8893eabc775104b5007d9154d03e3.jpg)

![223c19bb709ac6d29e9bf5aefa8ebf7d7ec186a721d5f34628348f9036392104.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/223c19bb709ac6d29e9bf5aefa8ebf7d7ec186a721d5f34628348f9036392104.jpg)

![267149a8f992ea76c41ef786a26fd5f5232dd7edc661450b4712556e34230e18.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/267149a8f992ea76c41ef786a26fd5f5232dd7edc661450b4712556e34230e18.jpg)

![27c02aa12679e808faffdb58e6f951e2436b3a5d8b25e11425e8417b8bed8f7a.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/27c02aa12679e808faffdb58e6f951e2436b3a5d8b25e11425e8417b8bed8f7a.jpg)

![28cdd7edeb193dcdada48e00d159f9495cf93f167cf4ad0b8460e7aba34c24b6.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/28cdd7edeb193dcdada48e00d159f9495cf93f167cf4ad0b8460e7aba34c24b6.jpg)

![2bcd77e2c55d2a52fea1452652eb1617cd27321b437750c645f306de677e254d.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/2bcd77e2c55d2a52fea1452652eb1617cd27321b437750c645f306de677e254d.jpg)

![31e968c3aef0097b6f904ae1d1108775d22535aad2f26935d6fc2a7e0ac49cb7.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/31e968c3aef0097b6f904ae1d1108775d22535aad2f26935d6fc2a7e0ac49cb7.jpg)

![3dd14ff85c6b7dfc335ee65fce5c6a1b552e8289bc95350be4285efde835ddac.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/3dd14ff85c6b7dfc335ee65fce5c6a1b552e8289bc95350be4285efde835ddac.jpg)

![442457445663c1c03de9614ae95d3ae6c54206953be2ee33027074b63c72da94.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/442457445663c1c03de9614ae95d3ae6c54206953be2ee33027074b63c72da94.jpg)

![4cbcd822c63c351f63401ad55c5a61b3284724a4d659cb0e60395a109e2d61f0.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/4cbcd822c63c351f63401ad55c5a61b3284724a4d659cb0e60395a109e2d61f0.jpg)

![4e63f92cf40ef7cdec55c6b44c8b3e9b11263a511096cbdb58fbaf2fbb2d6284.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/4e63f92cf40ef7cdec55c6b44c8b3e9b11263a511096cbdb58fbaf2fbb2d6284.jpg)

![4f0d7d02c444ae5dfb09c155d2608651ea18de340a59d363d1e5c8f9ea396aa6.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/4f0d7d02c444ae5dfb09c155d2608651ea18de340a59d363d1e5c8f9ea396aa6.jpg)

![4fd128da1e75268870c6696fee53082f647857468a1cc5a53287840c27aaa1f0.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/4fd128da1e75268870c6696fee53082f647857468a1cc5a53287840c27aaa1f0.jpg)

![5b4931548303e6fb9755f4a9ff70d6aa7dca9e4da322a55ac9461adb2a2a0f8f.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/5b4931548303e6fb9755f4a9ff70d6aa7dca9e4da322a55ac9461adb2a2a0f8f.jpg)

![5da908ab7952acb8239b78c49b399f8ab4314dff7f241cb712033e1fa6de420b.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/5da908ab7952acb8239b78c49b399f8ab4314dff7f241cb712033e1fa6de420b.jpg)

![5ffb94499e620b40277fa041aec2173927bea71535bb4da048484a0a9c4ea96a.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/5ffb94499e620b40277fa041aec2173927bea71535bb4da048484a0a9c4ea96a.jpg)

![6354f5aed1a0c8e5e8d9f68f9f20f98c64bc87f6ac5836c51fde8d1eebe59e4e.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/6354f5aed1a0c8e5e8d9f68f9f20f98c64bc87f6ac5836c51fde8d1eebe59e4e.jpg)

![69c82e5866e09484449d517227108f4b90cade41bd21a72068c29ce7a9e09c62.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/69c82e5866e09484449d517227108f4b90cade41bd21a72068c29ce7a9e09c62.jpg)

![6f387472ff3d087b135ffc794d665c5154a8a885fd3b4b9b26d97b17d13ec309.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/6f387472ff3d087b135ffc794d665c5154a8a885fd3b4b9b26d97b17d13ec309.jpg)

![79f0aceab78c805bd7191e861dfd9ff84b7b067c17c5129ddc8a0984cf14b4fe.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/79f0aceab78c805bd7191e861dfd9ff84b7b067c17c5129ddc8a0984cf14b4fe.jpg)

![7d1daaf0a3d10515223cc25709e133f6369b9d5dc83be452d190bc2dcfb00d62.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/7d1daaf0a3d10515223cc25709e133f6369b9d5dc83be452d190bc2dcfb00d62.jpg)

![7e3a0b1212af691a45100c9fa35c35ba2f7fe13d345216378b0991ece59e4354.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/7e3a0b1212af691a45100c9fa35c35ba2f7fe13d345216378b0991ece59e4354.jpg)

![7ebac895dc37c38c46dfbeea57b59454c473bb001b60200e9fb4c0bad97e0559.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/7ebac895dc37c38c46dfbeea57b59454c473bb001b60200e9fb4c0bad97e0559.jpg)

![7f94bd13b3d63ae4be7a2dda0c908bea5a3e890750b66e67fc8dd0bf2d3e0dd2.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/7f94bd13b3d63ae4be7a2dda0c908bea5a3e890750b66e67fc8dd0bf2d3e0dd2.jpg)

![7fa8fe55322a557dfa1fe3ee9c64b62dc93e7ba0954aa9ac06cbd3251866715c.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/7fa8fe55322a557dfa1fe3ee9c64b62dc93e7ba0954aa9ac06cbd3251866715c.jpg)

![85bb08b0dfc143f582f8057095a275d92ff5a239f6c0d5718353d034d88f98b3.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/85bb08b0dfc143f582f8057095a275d92ff5a239f6c0d5718353d034d88f98b3.jpg)

![9084ae6b121cb99d4cac07af7f84e8d30ba341f730b3332702401eebaff66fa8.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/9084ae6b121cb99d4cac07af7f84e8d30ba341f730b3332702401eebaff66fa8.jpg)

![90ff9085c76a9ab84faa5cd86fe6fd40e18c77ae39108560d4a3ca765f29e8c5.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/90ff9085c76a9ab84faa5cd86fe6fd40e18c77ae39108560d4a3ca765f29e8c5.jpg)

![911f956c8da1b5a9cfc81fa0086f190a29fa3e6688be75788021ff40cb15777f.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/911f956c8da1b5a9cfc81fa0086f190a29fa3e6688be75788021ff40cb15777f.jpg)

![93f2d4df69777e56629dd09dd9a64ea5a8f2a79b6be1123c6694148be2942380.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/93f2d4df69777e56629dd09dd9a64ea5a8f2a79b6be1123c6694148be2942380.jpg)

![a207fe8ce641f64bab0f25ae911c8c7fc2554879b13d3f4d3862c2cbb8b87d99.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/a207fe8ce641f64bab0f25ae911c8c7fc2554879b13d3f4d3862c2cbb8b87d99.jpg)

![acfc93bb368f5b584c3b3d367a3d55c97a8f209b035808560bb0a6aeb1b66365.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/acfc93bb368f5b584c3b3d367a3d55c97a8f209b035808560bb0a6aeb1b66365.jpg)

![ad19bb0fab78ab5077bc8b6e56818f1d19874f72e606140821c08c9d901094fa.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/ad19bb0fab78ab5077bc8b6e56818f1d19874f72e606140821c08c9d901094fa.jpg)

![ad5bbad9370f1e693283391b303a6e45facb0d402899f436baea3b586e5784e1.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/ad5bbad9370f1e693283391b303a6e45facb0d402899f436baea3b586e5784e1.jpg)

![b5335b0823c91c72165e0ad2267b149bf3ddf2ab33188ae9d688fe5af7a00101.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/b5335b0823c91c72165e0ad2267b149bf3ddf2ab33188ae9d688fe5af7a00101.jpg)

![b55dc41980bbbd768ad6da3bde6cc245ac228392fdbd56fdfbb3fd081970fb64.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/b55dc41980bbbd768ad6da3bde6cc245ac228392fdbd56fdfbb3fd081970fb64.jpg)

![b688679f698bd7e93331d40f1f5eaf8ba32788eb298cb1d263e7d76f5a212a43.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/b688679f698bd7e93331d40f1f5eaf8ba32788eb298cb1d263e7d76f5a212a43.jpg)

![b6cbdaf737c57179e8d364ab8db40657687e362c2e7a764c01099b21f396c384.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/b6cbdaf737c57179e8d364ab8db40657687e362c2e7a764c01099b21f396c384.jpg)

![b9244e993c1ce3d885ee68d0dca19c9273e2fb4562a02c6f966b524ca8f19e62.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/b9244e993c1ce3d885ee68d0dca19c9273e2fb4562a02c6f966b524ca8f19e62.jpg)

![bb183937aab35af3d518947a2b109a61c541e6e8f021d2aecca20ea293ff47d9.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/bb183937aab35af3d518947a2b109a61c541e6e8f021d2aecca20ea293ff47d9.jpg)

![bf679b059237d9e47ec019bdd7ff92d3894976ecde39a3e471168d0be0b57806.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/bf679b059237d9e47ec019bdd7ff92d3894976ecde39a3e471168d0be0b57806.jpg)

![c53da610b0adaf58c66299e2217f3693203c292c1cdfd47a021b9cb3288ee7b2.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/c53da610b0adaf58c66299e2217f3693203c292c1cdfd47a021b9cb3288ee7b2.jpg)

![c75b83c6bc394ac0dff51fc824884eb7c33b06dd19c0a35545e78951553b8bd3.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/c75b83c6bc394ac0dff51fc824884eb7c33b06dd19c0a35545e78951553b8bd3.jpg)

![c772a10f9b3fcb0613f8f4577600d2ff66934009742525e0e2241622f35a689f.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/c772a10f9b3fcb0613f8f4577600d2ff66934009742525e0e2241622f35a689f.jpg)

![ca786e0eaa6a3da3d6f9455dd0d5863cdc8d001fd900c03c0e6b1428af3645a6.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/ca786e0eaa6a3da3d6f9455dd0d5863cdc8d001fd900c03c0e6b1428af3645a6.jpg)

![cd54708c9797d643f1a09ca25b0d67a89b50a8f5bf534f764a3a9ef480a134b9.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/cd54708c9797d643f1a09ca25b0d67a89b50a8f5bf534f764a3a9ef480a134b9.jpg)

![cec743e6d59bff2bf37d99a142dae4f505a8fb0ab40716b7dc8b809398581c22.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/cec743e6d59bff2bf37d99a142dae4f505a8fb0ab40716b7dc8b809398581c22.jpg)

![cffa2be69aec48f7e94316a8e0d353a8520a234e9967bb625423d022c20c022c.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/cffa2be69aec48f7e94316a8e0d353a8520a234e9967bb625423d022c20c022c.jpg)

![d0f9f5c307d9e9c8cec651dc080044acdadf9e183983d01ee6a1074c763ad946.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/d0f9f5c307d9e9c8cec651dc080044acdadf9e183983d01ee6a1074c763ad946.jpg)

![d38092fd7fef696b792023f873791920470a302c206c0ef4d9be98867535d475.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/d38092fd7fef696b792023f873791920470a302c206c0ef4d9be98867535d475.jpg)

![d40245e10362e7296c7bd7c4a8b092ece695375e8da8ef9ab8de9026be083031.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/d40245e10362e7296c7bd7c4a8b092ece695375e8da8ef9ab8de9026be083031.jpg)

![d8b6daba77f0c58b5d68a0179fadbdddc23e9febb6935768e57f561e72f1ec05.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/d8b6daba77f0c58b5d68a0179fadbdddc23e9febb6935768e57f561e72f1ec05.jpg)

![d97cf56af666399b3b3e16dcabfc819c6cf5b87efc10765d998bde405a871656.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/d97cf56af666399b3b3e16dcabfc819c6cf5b87efc10765d998bde405a871656.jpg)

![dad237cb8bfb4affa8364d96fcef09988d1ba08abc110bbf1ae068b0fe798fc7.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/dad237cb8bfb4affa8364d96fcef09988d1ba08abc110bbf1ae068b0fe798fc7.jpg)

![de9ae901a2889cc4773d1b562595fb20f7ab48d3b877451473520fd2ec5d2ef8.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/de9ae901a2889cc4773d1b562595fb20f7ab48d3b877451473520fd2ec5d2ef8.jpg)

![e27aa3f63c52cce77e12f67ed5765c49fef6414dcc0e499d820a033259e39f71.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/e27aa3f63c52cce77e12f67ed5765c49fef6414dcc0e499d820a033259e39f71.jpg)

![e610a45f374b462d308fcfceab3802b5f900f00053fe2e94be286c907e68acae.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/e610a45f374b462d308fcfceab3802b5f900f00053fe2e94be286c907e68acae.jpg)

![f5028e5629aba5e20828a91293cad7ccdb422a8b849ae9d5f73cdad006e73d5f.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/f5028e5629aba5e20828a91293cad7ccdb422a8b849ae9d5f73cdad006e73d5f.jpg)

![f58a70c4f148611b7baa91aafcff631e476e1ed0dfc8ece71c116042561987c1.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/f58a70c4f148611b7baa91aafcff631e476e1ed0dfc8ece71c116042561987c1.jpg)

![febcf5ef803c6f3a9a4f4adf88e54f6388b1bb61c15900dfe1d2a447fa1682e7.jpg](../icml_results/2980_M%2BAdam_%20Low-Precision%20Training%20via%20Additive%E2%80%93Multiplicative%20Optimization/tables/febcf5ef803c6f3a9a4f4adf88e54f6388b1bb61c15900dfe1d2a447fa1682e7.jpg)

<a id="paper-2943"></a>

## REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations


### Images

![2ca17092f2d7bbe0232c299875745a430ea41902ef357dd9440a515de3c676e2.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/images/2ca17092f2d7bbe0232c299875745a430ea41902ef357dd9440a515de3c676e2.jpg)

![433fccce6de0f3cf5c78f5ef4eb1030a525121db5e58ae73064c5a8a66567d63.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/images/433fccce6de0f3cf5c78f5ef4eb1030a525121db5e58ae73064c5a8a66567d63.jpg)

![4c04f23361fa3d3d2eec1e6c43e7405dd2596d486f08aa5b947efb9553773a3b.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/images/4c04f23361fa3d3d2eec1e6c43e7405dd2596d486f08aa5b947efb9553773a3b.jpg)

![4f788b1a8ad279e45e8223a999f5a483c3361118b3c9bdbec71ed76f9787bc12.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/images/4f788b1a8ad279e45e8223a999f5a483c3361118b3c9bdbec71ed76f9787bc12.jpg)

![69933e9999431abfc0ad5c27a24ca5b3803b324c7014b4e01f1affc3e8fbd48c.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/images/69933e9999431abfc0ad5c27a24ca5b3803b324c7014b4e01f1affc3e8fbd48c.jpg)

![ba4d780e0a4976e268680a036391226476aa3f12e32ebfdffc9d3f052c8b64ae.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/images/ba4d780e0a4976e268680a036391226476aa3f12e32ebfdffc9d3f052c8b64ae.jpg)

![cf492cd17d70063a204e6e8609ef7374cf15983bea09d63861f2cd391a5b565f.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/images/cf492cd17d70063a204e6e8609ef7374cf15983bea09d63861f2cd391a5b565f.jpg)

![fbb42a55fac23126183ff24d1a14177ec0b97a0691fe4ac3e5124877f149864b.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/images/fbb42a55fac23126183ff24d1a14177ec0b97a0691fe4ac3e5124877f149864b.jpg)

### Tables

![268f157c01ff69362f7c94b40a28f7b0f6dec7c6e1a8306e45638c2af65f5fa7.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/268f157c01ff69362f7c94b40a28f7b0f6dec7c6e1a8306e45638c2af65f5fa7.jpg)

![27fbc50ba5cfb954ae0c8e673c1b1f6353f690662b15dda3d8e72bb272a047ee.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/27fbc50ba5cfb954ae0c8e673c1b1f6353f690662b15dda3d8e72bb272a047ee.jpg)

![359bff9c43c7e8f497345be41731b1d3fb2ce623cc05aa517ada2eb6b4fce49f.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/359bff9c43c7e8f497345be41731b1d3fb2ce623cc05aa517ada2eb6b4fce49f.jpg)

![35e08df05bd756c107f477af2b60b6d0763669560fe3fce7472e128c87aee09b.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/35e08df05bd756c107f477af2b60b6d0763669560fe3fce7472e128c87aee09b.jpg)

![3b45c3f2704ecb0c5d0b207de3b93c3d89f76bfe5f1014333652c7b304ec49c4.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/3b45c3f2704ecb0c5d0b207de3b93c3d89f76bfe5f1014333652c7b304ec49c4.jpg)

![42039a1f57431be0d205a44fb0a014d8ce9692fb45dbce10b1c7bb9b1d5ad02d.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/42039a1f57431be0d205a44fb0a014d8ce9692fb45dbce10b1c7bb9b1d5ad02d.jpg)

![54664d8b261f0cd064af6918550d19f07f0de36c8d47a840c862daadcbbf277b.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/54664d8b261f0cd064af6918550d19f07f0de36c8d47a840c862daadcbbf277b.jpg)

![5f6dd008718f295b1ae8e4d053f5ba32f50bf0fdc6d0b56e2eb9660ff66c9a72.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/5f6dd008718f295b1ae8e4d053f5ba32f50bf0fdc6d0b56e2eb9660ff66c9a72.jpg)

![6ac8b6af98143828ab1612161afb4e47ecf3c6cbcb42732f18978d3fb2773602.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/6ac8b6af98143828ab1612161afb4e47ecf3c6cbcb42732f18978d3fb2773602.jpg)

![849f3b8d93d815e54fe777b2e62061043f3b33cff9ab21d378388758db2415f4.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/849f3b8d93d815e54fe777b2e62061043f3b33cff9ab21d378388758db2415f4.jpg)

![8cea1601dfb531fd545121839d807ea79d8524b77418c9489a575746ad770131.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/8cea1601dfb531fd545121839d807ea79d8524b77418c9489a575746ad770131.jpg)

![90c135ad9c67e12f106aab6ee927645613e8b0e64fe88625102644e362fd49a9.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/90c135ad9c67e12f106aab6ee927645613e8b0e64fe88625102644e362fd49a9.jpg)

![99b76a9e2df642e47925f83692fbdf620bd042fbbadef0a23f6c9e21b4a86a0e.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/99b76a9e2df642e47925f83692fbdf620bd042fbbadef0a23f6c9e21b4a86a0e.jpg)

![b504596b84daf56300aa537a4ccdc69cc831aafd22e3baadbf77429038c3f2f6.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/b504596b84daf56300aa537a4ccdc69cc831aafd22e3baadbf77429038c3f2f6.jpg)

![d1b0d083be225ea83ae1763dfd7c6c6e9e53d0e157da7cb757f700e0e9074177.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/d1b0d083be225ea83ae1763dfd7c6c6e9e53d0e157da7cb757f700e0e9074177.jpg)

![d9eb33aea8cfa50b616a53c018eb08972a02f63ed318b7b67aeedb5f34c990bd.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/d9eb33aea8cfa50b616a53c018eb08972a02f63ed318b7b67aeedb5f34c990bd.jpg)

![e5ab2b3f6383862dc99a11174baedc727a4221456f7c6b2e98793864410a5abd.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/e5ab2b3f6383862dc99a11174baedc727a4221456f7c6b2e98793864410a5abd.jpg)

![fd62d960a7dd5b768273a7a2c59af6c105698fd9bb85b9af09508d9e05dcf822.jpg](../icml_results/2981_REALISTA_%20Realistic%20Latent%20Adversarial%20Attacks%20that%20Elicit%20LLM%20Hallucinations/tables/fd62d960a7dd5b768273a7a2c59af6c105698fd9bb85b9af09508d9e05dcf822.jpg)

<a id="paper-2944"></a>

## Bottleneck-Guided Spectral Subgoals For Offline Goal-Conditioned RL


### Images

![0888e3257a23263f6909d88a7a0317bc1b6bcb89c92c0bb71c48494c5cd7db11.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/images/0888e3257a23263f6909d88a7a0317bc1b6bcb89c92c0bb71c48494c5cd7db11.jpg)

![33c23ff28c623c1ec9df6fcc34869a67f31470452fbe91461afdae92a924f72d.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/images/33c23ff28c623c1ec9df6fcc34869a67f31470452fbe91461afdae92a924f72d.jpg)

![382b333f14ed07e52992202a238918e2de6e79e20b7653f10b15566af71f28b8.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/images/382b333f14ed07e52992202a238918e2de6e79e20b7653f10b15566af71f28b8.jpg)

![48b7a54d7072ea0924568c58a28eb26432e70b9a7af6101920d88f6940b4641d.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/images/48b7a54d7072ea0924568c58a28eb26432e70b9a7af6101920d88f6940b4641d.jpg)

![6ce14ef885c4674f9a7449be704839f469fb1c9e44623d6992daddd29cfedfd5.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/images/6ce14ef885c4674f9a7449be704839f469fb1c9e44623d6992daddd29cfedfd5.jpg)

![825b765f091998fb2606ac5f80076a4a1346983cfa8c2f205b8def94bcb10d12.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/images/825b765f091998fb2606ac5f80076a4a1346983cfa8c2f205b8def94bcb10d12.jpg)

![92dcd277fb911a66776ad4841d917809ddaa1b177f4ea695b1fb9c6bb1b99d74.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/images/92dcd277fb911a66776ad4841d917809ddaa1b177f4ea695b1fb9c6bb1b99d74.jpg)

![977502655396427f53a373b20c4f2fd7579ce16c1108d64f644ee363f8004e4c.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/images/977502655396427f53a373b20c4f2fd7579ce16c1108d64f644ee363f8004e4c.jpg)

![9da9aa99e4e59c6631610fc4dedca63f82be37f28c09a760f075d428f1abd51b.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/images/9da9aa99e4e59c6631610fc4dedca63f82be37f28c09a760f075d428f1abd51b.jpg)

![b3b1c56b1432dbdef18912ab4b3badaa0d151bf0691768fad776a98ebddfea63.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/images/b3b1c56b1432dbdef18912ab4b3badaa0d151bf0691768fad776a98ebddfea63.jpg)

![d1ac35f2e857424f18e905b6cb70773eb9c9abd3dcc2a77f7522a48770e02592.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/images/d1ac35f2e857424f18e905b6cb70773eb9c9abd3dcc2a77f7522a48770e02592.jpg)

![da95a8f94566e41a09610aac429fe31249043bbdac5f8e6d00aa254d9302928d.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/images/da95a8f94566e41a09610aac429fe31249043bbdac5f8e6d00aa254d9302928d.jpg)

![dbf82f313cf689068db5d243b83999f5a6d15ed457dcd8849155831d92118faf.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/images/dbf82f313cf689068db5d243b83999f5a6d15ed457dcd8849155831d92118faf.jpg)

![e3872a702e50a589b73bc7005b78e90184d32fb1f6792bf86eb246929e4d0655.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/images/e3872a702e50a589b73bc7005b78e90184d32fb1f6792bf86eb246929e4d0655.jpg)

![ebe5c14304bdb04c0523325d558eb75597164c4b1f4cc87d2893414756d0b44b.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/images/ebe5c14304bdb04c0523325d558eb75597164c4b1f4cc87d2893414756d0b44b.jpg)

![ed894018d17f213ac09f7856ce723a6a06d01a9f67e16799dcda655c87833674.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/images/ed894018d17f213ac09f7856ce723a6a06d01a9f67e16799dcda655c87833674.jpg)

### Tables

![141173225f296783864c2b781ab49f3c4b789ccfc33386aea50c799f199c5124.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/tables/141173225f296783864c2b781ab49f3c4b789ccfc33386aea50c799f199c5124.jpg)

![2d91be26d02766f46cdc360f187ea8ee446d536ca91c5643a151a3d8c7016835.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/tables/2d91be26d02766f46cdc360f187ea8ee446d536ca91c5643a151a3d8c7016835.jpg)

![60bf2d3702e6cdbadcc34873bfa76e4d83184471af41dd856aa1cf834aed002f.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/tables/60bf2d3702e6cdbadcc34873bfa76e4d83184471af41dd856aa1cf834aed002f.jpg)

![6792aac57cd93b93e054a929f30d9542163b24d3782d8fc3b8546c922d7d5f73.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/tables/6792aac57cd93b93e054a929f30d9542163b24d3782d8fc3b8546c922d7d5f73.jpg)

![84ff1dc48eadf6f0ef614e236f45f341d7f909c467ed78fcc91a3d291e5cae3b.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/tables/84ff1dc48eadf6f0ef614e236f45f341d7f909c467ed78fcc91a3d291e5cae3b.jpg)

![a1ba380b7c8c229c752f2c24e9d56106f1b9ab5016f7742deba502cffcfc25b7.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/tables/a1ba380b7c8c229c752f2c24e9d56106f1b9ab5016f7742deba502cffcfc25b7.jpg)

![ac69dc6ecfc4e8cedaf8de858194569decf87b365a5073c05a46d8bed798b9d6.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/tables/ac69dc6ecfc4e8cedaf8de858194569decf87b365a5073c05a46d8bed798b9d6.jpg)

![ebb9e8322915cc3ab9b6e0c992b57ac7f37f922d9d5bf8a2e1d883e5003b0085.jpg](../icml_results/2982_Bottleneck-Guided%20Spectral%20Subgoals%20For%20Offline%20Goal-Conditioned%20RL/tables/ebb9e8322915cc3ab9b6e0c992b57ac7f37f922d9d5bf8a2e1d883e5003b0085.jpg)

<a id="paper-2945"></a>

## Implicit Action Chunking for Smooth Continuous Control


### Images

![01c762dc335c522d918baa03b47b1a9398422ba3352fee23bc7a3f492ec31a93.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/01c762dc335c522d918baa03b47b1a9398422ba3352fee23bc7a3f492ec31a93.jpg)

![030acce31dc5274474b8924ed31687fcddf12e3e9113f0de37fb2be1c1b70003.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/030acce31dc5274474b8924ed31687fcddf12e3e9113f0de37fb2be1c1b70003.jpg)

![0449cb2c6872c4b46b5a6ded645502cbd45a5355e6b7a6bfa291e475f6321ec8.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/0449cb2c6872c4b46b5a6ded645502cbd45a5355e6b7a6bfa291e475f6321ec8.jpg)

![29dc1c2e247e188fe591427f3cc5d1f34af24f6fcf222fbddbbcaabd6612adbe.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/29dc1c2e247e188fe591427f3cc5d1f34af24f6fcf222fbddbbcaabd6612adbe.jpg)

![40cefc8afe7c43df92a7fef2e7f5e6801b458b4e434605845600ec2ff6aaafd6.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/40cefc8afe7c43df92a7fef2e7f5e6801b458b4e434605845600ec2ff6aaafd6.jpg)

![544defebb573852b37c229ad10e4d8af3b7bb9d2ca3345b469cd8c5085a96456.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/544defebb573852b37c229ad10e4d8af3b7bb9d2ca3345b469cd8c5085a96456.jpg)

![54c8d318b692585193437906ac9302357053a32ad2a1e4cb36104b036f4e37a5.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/54c8d318b692585193437906ac9302357053a32ad2a1e4cb36104b036f4e37a5.jpg)

![54d787ed16799d7459adcea4ced4d4e4e6d37324827f2468edd386ac291cb9d3.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/54d787ed16799d7459adcea4ced4d4e4e6d37324827f2468edd386ac291cb9d3.jpg)

![55bf69a702169dcd7a8a442c2eaaad5ee1033fd5482c3dc1cf27c1760373b69f.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/55bf69a702169dcd7a8a442c2eaaad5ee1033fd5482c3dc1cf27c1760373b69f.jpg)

![6db6e34f66e5c9617f10b0b2e578cbe3e276b3974480c34db30bf7fac55470f8.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/6db6e34f66e5c9617f10b0b2e578cbe3e276b3974480c34db30bf7fac55470f8.jpg)

![7d896ee7acc0d813b2f1d5ffa8c5f660e324e779cbdeb3ae42b748225e622181.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/7d896ee7acc0d813b2f1d5ffa8c5f660e324e779cbdeb3ae42b748225e622181.jpg)

![9f8356ee7b0d448fa103d311f39d754957d7118db6d11f6cc551d023ad499218.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/9f8356ee7b0d448fa103d311f39d754957d7118db6d11f6cc551d023ad499218.jpg)

![a677a480331b07d78fd376b3450b7d680eab5ffccce82320a7bf2abb8002c38a.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/a677a480331b07d78fd376b3450b7d680eab5ffccce82320a7bf2abb8002c38a.jpg)

![b7d84be34d484ba4fb8d2d47aee2f34770f373e7d34d79ff131dc7f9dac2e67b.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/b7d84be34d484ba4fb8d2d47aee2f34770f373e7d34d79ff131dc7f9dac2e67b.jpg)

![c1538ccd7ad6cbe86af353ca84ceaf20aef486c4ab515e9e05b0ab52fd689ea6.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/c1538ccd7ad6cbe86af353ca84ceaf20aef486c4ab515e9e05b0ab52fd689ea6.jpg)

![c52a2fd0012b6b3d530e1f3cf8b1c6056a61b80e8ada01be58b4b2916180df6d.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/c52a2fd0012b6b3d530e1f3cf8b1c6056a61b80e8ada01be58b4b2916180df6d.jpg)

![e7eb3d2f6d2792eb7b89b71d13489070292a7994e73e11597618301e10268506.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/e7eb3d2f6d2792eb7b89b71d13489070292a7994e73e11597618301e10268506.jpg)

![efe009e98c9558928c85e0b64d3c0cfa645bdf7e02b7ab286304433541ca91b7.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/efe009e98c9558928c85e0b64d3c0cfa645bdf7e02b7ab286304433541ca91b7.jpg)

![f177d6480d8adaa10dab17653a951e819676706904d9b944d35b7b70cfaa15c4.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/f177d6480d8adaa10dab17653a951e819676706904d9b944d35b7b70cfaa15c4.jpg)

![fb3570e6db88aee291d0453bdae3f0f2d30b807d79f1d74c0b6ea5a58166396d.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/fb3570e6db88aee291d0453bdae3f0f2d30b807d79f1d74c0b6ea5a58166396d.jpg)

![fea5a022193dabd6f005c699d35f20cd5cbc4be5842bc1507eb5b7a0aaf32ad1.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/images/fea5a022193dabd6f005c699d35f20cd5cbc4be5842bc1507eb5b7a0aaf32ad1.jpg)

### Tables

![0774afd03cd9d3c4d1e1d5ee01cd0438af3d2686e4ce22b74f44726f636b340a.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/0774afd03cd9d3c4d1e1d5ee01cd0438af3d2686e4ce22b74f44726f636b340a.jpg)

![0be93dd81c011748a7e99717f8121e7843625eb2c1fabf97c1e18bdec3b6ade1.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/0be93dd81c011748a7e99717f8121e7843625eb2c1fabf97c1e18bdec3b6ade1.jpg)

![2058857f14241a3e09daf412abf7f09ebd0d7b27f944d8f72417411a41257224.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/2058857f14241a3e09daf412abf7f09ebd0d7b27f944d8f72417411a41257224.jpg)

![2737b15002e7c9360032884481b8f3efa9a89cc66a8dce68c2febdeadfeba424.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/2737b15002e7c9360032884481b8f3efa9a89cc66a8dce68c2febdeadfeba424.jpg)

![2ab4d739bf02f4ba39600c775f262a68cf7d14694c94c72f21bf0f9299a8b26d.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/2ab4d739bf02f4ba39600c775f262a68cf7d14694c94c72f21bf0f9299a8b26d.jpg)

![2b934b9607aef174903525008937c0b49aa0755105faae0ed6329768e59d0e0b.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/2b934b9607aef174903525008937c0b49aa0755105faae0ed6329768e59d0e0b.jpg)

![2f25eae1450ae87905e78b6dd191699e89fcab2db670804533abfb86e0005ff4.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/2f25eae1450ae87905e78b6dd191699e89fcab2db670804533abfb86e0005ff4.jpg)

![32c544d8ca1560bc3cf20a590d1def970acbe3b6692039a896e4793d3fcd666c.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/32c544d8ca1560bc3cf20a590d1def970acbe3b6692039a896e4793d3fcd666c.jpg)

![394de1a4063f87b5c86859c9360c01864b6e21e9ae898dd16c3013dd0f394f24.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/394de1a4063f87b5c86859c9360c01864b6e21e9ae898dd16c3013dd0f394f24.jpg)

![39bb76b9c252667fe83e2caa0991bc8373ec8f0a5046a94901ed0345dc69692e.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/39bb76b9c252667fe83e2caa0991bc8373ec8f0a5046a94901ed0345dc69692e.jpg)

![4516b36081520afa79ea32ebafde9e1f1c1ce98259493f53308716817e1ba08f.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/4516b36081520afa79ea32ebafde9e1f1c1ce98259493f53308716817e1ba08f.jpg)

![73f0b0474fe00cb5d8b7a77ceb88f6088ade0592a21702d7b6c4f8a2c4e00e65.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/73f0b0474fe00cb5d8b7a77ceb88f6088ade0592a21702d7b6c4f8a2c4e00e65.jpg)

![95897cb7bca1a2b2e539092935be0fbf03c68414cad899bf64b32f9469ee4821.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/95897cb7bca1a2b2e539092935be0fbf03c68414cad899bf64b32f9469ee4821.jpg)

![a55a95f81eebf193286e2b376125bd195fb0a887d4f4a1fce48c02a71899997a.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/a55a95f81eebf193286e2b376125bd195fb0a887d4f4a1fce48c02a71899997a.jpg)

![ad2bfc5dbb948a26313328fee52aae2f408f3ef216d2d4f8e0b9e8d1b9a3755d.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/ad2bfc5dbb948a26313328fee52aae2f408f3ef216d2d4f8e0b9e8d1b9a3755d.jpg)

![b0cdbe235e6eceb34f90b075c80c9cb5b3486002ad0b0d6f781c9c1422e6afbc.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/b0cdbe235e6eceb34f90b075c80c9cb5b3486002ad0b0d6f781c9c1422e6afbc.jpg)

![b59fcb83106022b8ebebb330efb0febba3adf0faf298476d7e29fe69752a047d.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/b59fcb83106022b8ebebb330efb0febba3adf0faf298476d7e29fe69752a047d.jpg)

![c75acc97a8c914ec03942d9b05fa4c4f505a2f0c3c32cd4c2028d866edc9b8a8.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/c75acc97a8c914ec03942d9b05fa4c4f505a2f0c3c32cd4c2028d866edc9b8a8.jpg)

![d4ce00e801b1fc486cee9ba32070e3a55adfe1308910167b753709f73ccef825.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/d4ce00e801b1fc486cee9ba32070e3a55adfe1308910167b753709f73ccef825.jpg)

![d8516e307cf760dddb0a02b467ddf346adabf466d619ef36f8050ac8c0e9eda5.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/d8516e307cf760dddb0a02b467ddf346adabf466d619ef36f8050ac8c0e9eda5.jpg)

![dafa12cb911c794c00bc9caabf16a7abc4387a4e8c95f93c299eb8b687fa4a23.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/dafa12cb911c794c00bc9caabf16a7abc4387a4e8c95f93c299eb8b687fa4a23.jpg)

![ef0e9cabb046396d53dce512f95766998b616b44dcb9f4e02289e91727c1029e.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/ef0e9cabb046396d53dce512f95766998b616b44dcb9f4e02289e91727c1029e.jpg)

![f5402059473c9dab6861f93a4a0eef24e2c8999e80f99b468e477507921a2b02.jpg](../icml_results/2983_Implicit%20Action%20Chunking%20for%20Smooth%20Continuous%20Control/tables/f5402059473c9dab6861f93a4a0eef24e2c8999e80f99b468e477507921a2b02.jpg)

<a id="paper-2946"></a>

## Strategy Executability in Mathematical Reasoning: Leveraging Human–Model Differences for Effective Guidance


### Images

![01e8f5bed8e686f2b99ce48a6dae1d927da11265af61c8a98574deb7f1ca03c7.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/images/01e8f5bed8e686f2b99ce48a6dae1d927da11265af61c8a98574deb7f1ca03c7.jpg)

![16d7fd74c06cfa736094d39f545d32f3d6f082fbec79ab18a4d7761a2aa9a200.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/images/16d7fd74c06cfa736094d39f545d32f3d6f082fbec79ab18a4d7761a2aa9a200.jpg)

![27a54d2d89fe1b17dec041ea10b3f722f685704ae5913fecde39f30f5f9422b7.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/images/27a54d2d89fe1b17dec041ea10b3f722f685704ae5913fecde39f30f5f9422b7.jpg)

![353420d817143bffa9788978734eb0d6487cce778a64e4cabd62708d64ab4b05.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/images/353420d817143bffa9788978734eb0d6487cce778a64e4cabd62708d64ab4b05.jpg)

![43528adc18cb3d5b17914f5df8e9e002dfa8535f33b9f963f91685891d470a93.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/images/43528adc18cb3d5b17914f5df8e9e002dfa8535f33b9f963f91685891d470a93.jpg)

![61de9b9af3637061411c950df48f7402a7b99d05674e7eac12223aa0ab2c2b70.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/images/61de9b9af3637061411c950df48f7402a7b99d05674e7eac12223aa0ab2c2b70.jpg)

![9c98c53038d4c1e122fd29585e699a45e3a099d342087afcb6682674049a9551.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/images/9c98c53038d4c1e122fd29585e699a45e3a099d342087afcb6682674049a9551.jpg)

![a8b806186ac2a114bc0befe60fd63de70c57545a2973960090c03c0d9f06f3d6.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/images/a8b806186ac2a114bc0befe60fd63de70c57545a2973960090c03c0d9f06f3d6.jpg)

![b8f25d37633eceae2f882c45c4877298d14a55c428dc7f15f9c6047344fceaf2.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/images/b8f25d37633eceae2f882c45c4877298d14a55c428dc7f15f9c6047344fceaf2.jpg)

![bb4d721ff447657e3cd0d7b86cb42fd16298e0b5e52de71c4af255ee39fe9b3f.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/images/bb4d721ff447657e3cd0d7b86cb42fd16298e0b5e52de71c4af255ee39fe9b3f.jpg)

### Tables

![0625db395137140a5426017abf2990292457a7803da66020ac8938b265067301.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/tables/0625db395137140a5426017abf2990292457a7803da66020ac8938b265067301.jpg)

![09976d89ade25cdf9f06f1b9e6c654d79050cb39e13e24231b216207fed6d682.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/tables/09976d89ade25cdf9f06f1b9e6c654d79050cb39e13e24231b216207fed6d682.jpg)

![212a3f06d1ea359c6d7f2f3b5f3839ec7174077c517a748b52a7b338da17fbdb.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/tables/212a3f06d1ea359c6d7f2f3b5f3839ec7174077c517a748b52a7b338da17fbdb.jpg)

![21b2f3bb58761a4757363f150ee784f7a5552771e8fe41c0a357cc996edaa359.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/tables/21b2f3bb58761a4757363f150ee784f7a5552771e8fe41c0a357cc996edaa359.jpg)

![26070589becdeaa7333a5060573abc929780bcaed93398d94a46eb50e5529d27.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/tables/26070589becdeaa7333a5060573abc929780bcaed93398d94a46eb50e5529d27.jpg)

![306d27801c48c9c653dc5ccdc3596dda943990cc00ffe6c7b2c74aed2dacdb89.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/tables/306d27801c48c9c653dc5ccdc3596dda943990cc00ffe6c7b2c74aed2dacdb89.jpg)

![4cbac43ec0fcd7c142f8dc88e8f166c7f848dbc798d13d3b1d5a36d9382aa2a0.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/tables/4cbac43ec0fcd7c142f8dc88e8f166c7f848dbc798d13d3b1d5a36d9382aa2a0.jpg)

![4f1190ea2eecdab44e1cfc5a5610c8af1b393830ed0d7c1ebf1e6e8a0b805b33.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/tables/4f1190ea2eecdab44e1cfc5a5610c8af1b393830ed0d7c1ebf1e6e8a0b805b33.jpg)

![67a347200f9d6c212fb2c1676ce3974935ed2fa6a323f789fb0d9d37f3e46716.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/tables/67a347200f9d6c212fb2c1676ce3974935ed2fa6a323f789fb0d9d37f3e46716.jpg)

![90f357e267705a2b229259a2abbb7ac73fd70d21cde6214674b73f97a2785faf.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/tables/90f357e267705a2b229259a2abbb7ac73fd70d21cde6214674b73f97a2785faf.jpg)

![9a696209541f61a50ab2c5bf209c3d07f7d7f7f3e449676803de85632cc3727a.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/tables/9a696209541f61a50ab2c5bf209c3d07f7d7f7f3e449676803de85632cc3727a.jpg)

![a0e771ccb7e2b2b1caceeb6a0c7c9d76b749529ab847ff3566858838c6e0e6ac.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/tables/a0e771ccb7e2b2b1caceeb6a0c7c9d76b749529ab847ff3566858838c6e0e6ac.jpg)

![ac6f9c085c165bd3f97cdd13e96495eaaf7e7ffcf813c72aeda4b00d88337c51.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/tables/ac6f9c085c165bd3f97cdd13e96495eaaf7e7ffcf813c72aeda4b00d88337c51.jpg)

![d4f55f7ef465600838361f142eb64929931d497188d2b889f9df16c152497a38.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/tables/d4f55f7ef465600838361f142eb64929931d497188d2b889f9df16c152497a38.jpg)

![f82a2a2159ec984a00670b635ebd87700c95088bc94497396873595c24ffefd1.jpg](../icml_results/2984_Strategy%20Executability%20in%20Mathematical%20Reasoning_%20Leveraging%20Human%E2%80%93Model%20Differences%20for%20Effective%20G/tables/f82a2a2159ec984a00670b635ebd87700c95088bc94497396873595c24ffefd1.jpg)

<a id="paper-2947"></a>

## Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization


### Images

![0ab63bb035ad2be9df36c417b44ec4d7317ccda2cd5eb9dbdec87fc57ec79662.jpg](../icml_results/2985_Turning%20Stale%20Gradients%20into%20Stable%20Gradients_%20Coherent%20Coordinate%20Descent%20with%20Implicit%20Landscape%20S/images/0ab63bb035ad2be9df36c417b44ec4d7317ccda2cd5eb9dbdec87fc57ec79662.jpg)

![49223fd556038b7c8e804df0fa9fc222f72464bac9d0ec55839b71c04360a50d.jpg](../icml_results/2985_Turning%20Stale%20Gradients%20into%20Stable%20Gradients_%20Coherent%20Coordinate%20Descent%20with%20Implicit%20Landscape%20S/images/49223fd556038b7c8e804df0fa9fc222f72464bac9d0ec55839b71c04360a50d.jpg)

![a81dc63d8a91e488232bd8762ccec542cd9ff8128d8e40613543f9c645d23c40.jpg](../icml_results/2985_Turning%20Stale%20Gradients%20into%20Stable%20Gradients_%20Coherent%20Coordinate%20Descent%20with%20Implicit%20Landscape%20S/images/a81dc63d8a91e488232bd8762ccec542cd9ff8128d8e40613543f9c645d23c40.jpg)

![ab885a83b779e2ae4c3547669238e63534e7e3b145477a5dbe61f9d092d8a0c4.jpg](../icml_results/2985_Turning%20Stale%20Gradients%20into%20Stable%20Gradients_%20Coherent%20Coordinate%20Descent%20with%20Implicit%20Landscape%20S/images/ab885a83b779e2ae4c3547669238e63534e7e3b145477a5dbe61f9d092d8a0c4.jpg)

![bef0346e777c89e64a46e03d4096e48b998d063804fe297cd48f18b0637e5baf.jpg](../icml_results/2985_Turning%20Stale%20Gradients%20into%20Stable%20Gradients_%20Coherent%20Coordinate%20Descent%20with%20Implicit%20Landscape%20S/images/bef0346e777c89e64a46e03d4096e48b998d063804fe297cd48f18b0637e5baf.jpg)

![f0e48c54ca72b07e04a7c6ca0f82c32edd5fc7dcf36fb4c92d956e8e9a269b14.jpg](../icml_results/2985_Turning%20Stale%20Gradients%20into%20Stable%20Gradients_%20Coherent%20Coordinate%20Descent%20with%20Implicit%20Landscape%20S/images/f0e48c54ca72b07e04a7c6ca0f82c32edd5fc7dcf36fb4c92d956e8e9a269b14.jpg)

### Tables

![ba1a8e28edefa41ab617d148558979dcbd4efe73ac2ab836c4eca662fd85a70b.jpg](../icml_results/2985_Turning%20Stale%20Gradients%20into%20Stable%20Gradients_%20Coherent%20Coordinate%20Descent%20with%20Implicit%20Landscape%20S/tables/ba1a8e28edefa41ab617d148558979dcbd4efe73ac2ab836c4eca662fd85a70b.jpg)

![ec58ead908ef912ef0059db717f77fbca397ddbdaa1afa523522b0310fdb87a0.jpg](../icml_results/2985_Turning%20Stale%20Gradients%20into%20Stable%20Gradients_%20Coherent%20Coordinate%20Descent%20with%20Implicit%20Landscape%20S/tables/ec58ead908ef912ef0059db717f77fbca397ddbdaa1afa523522b0310fdb87a0.jpg)

<a id="paper-2948"></a>

## FOCUS: DLLMs Know How to Tame Their Compute Bound


### Images

![08d6381a644cdf9f5cea36ea4b61d787cd754fbbe1522d591916e96f72b58dc4.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/08d6381a644cdf9f5cea36ea4b61d787cd754fbbe1522d591916e96f72b58dc4.jpg)

![17e08b0f2da4da60e19a5779276f290d7c9ff5aeb99263efea08cb7e1c19a184.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/17e08b0f2da4da60e19a5779276f290d7c9ff5aeb99263efea08cb7e1c19a184.jpg)

![34e8d822ecd130225c3dbb57801771a40511c9550edfdcb57fe4d3d7b444a957.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/34e8d822ecd130225c3dbb57801771a40511c9550edfdcb57fe4d3d7b444a957.jpg)

![51fc08de898869f6889b1b4e61f350b1ab74afd4091e43f8b92df992045c29ee.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/51fc08de898869f6889b1b4e61f350b1ab74afd4091e43f8b92df992045c29ee.jpg)

![59584fa3fbb0c60a01ce2486856dabcbae25b7fdfa237b60e6258394e305d838.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/59584fa3fbb0c60a01ce2486856dabcbae25b7fdfa237b60e6258394e305d838.jpg)

![66884552ce9ed17ff14abd483f8706e13eb976d85abd96ae61ddd24693a535e3.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/66884552ce9ed17ff14abd483f8706e13eb976d85abd96ae61ddd24693a535e3.jpg)

![7425b8f9e646d453151ea3d707a5a67dcd26ecab31dadcd9d59e11a4a5fdc9af.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/7425b8f9e646d453151ea3d707a5a67dcd26ecab31dadcd9d59e11a4a5fdc9af.jpg)

![80115fa1f68f5616529fd649a887530dbcef6a13e05fe7e8874c1fd42ec2bf04.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/80115fa1f68f5616529fd649a887530dbcef6a13e05fe7e8874c1fd42ec2bf04.jpg)

![87eb67bb785f9ea621ef80f6cd25bfac7037b0aa076ff5b4c3c738c6900bd640.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/87eb67bb785f9ea621ef80f6cd25bfac7037b0aa076ff5b4c3c738c6900bd640.jpg)

![a5864bf03a28027aa657201d75d835ddf3c68b34511aaf64687c0672763a4d15.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/a5864bf03a28027aa657201d75d835ddf3c68b34511aaf64687c0672763a4d15.jpg)

![aa99c7a9a370a133c27f3f110f189b32a7eb5e7c1ea000a450fa41d8ec4e9f9a.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/aa99c7a9a370a133c27f3f110f189b32a7eb5e7c1ea000a450fa41d8ec4e9f9a.jpg)

![ab27085c1d32d4649cc07b590958e73a3cd6afd131a1be0ff72dcf5cf450c420.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/ab27085c1d32d4649cc07b590958e73a3cd6afd131a1be0ff72dcf5cf450c420.jpg)

![d68888f52c02d68f58b6f44856e812e35f52a2692f2a42494702a18f6472380a.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/d68888f52c02d68f58b6f44856e812e35f52a2692f2a42494702a18f6472380a.jpg)

![e531edd4e76ee7b176daca3fc3875a97a1de6c7a69dac818a0fd77a3cc8671fc.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/e531edd4e76ee7b176daca3fc3875a97a1de6c7a69dac818a0fd77a3cc8671fc.jpg)

![e637b90cc9f2a39e90a80a8fc9f34b7558b1fa14e8b91936ef6a37c202c0e907.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/e637b90cc9f2a39e90a80a8fc9f34b7558b1fa14e8b91936ef6a37c202c0e907.jpg)

![f55cb863200c483342f1c9b8fdb30ee9c6f3a658341f7f565615d384c0482351.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/f55cb863200c483342f1c9b8fdb30ee9c6f3a658341f7f565615d384c0482351.jpg)

![f586e43f8ed94b14e0e0d14547cf49a018c3710f7cae763771aec74006b63f96.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/f586e43f8ed94b14e0e0d14547cf49a018c3710f7cae763771aec74006b63f96.jpg)

![f5ceb44b0ce8d96217237d45d9d03b736927034ac658201a7ff2154b3526d534.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/f5ceb44b0ce8d96217237d45d9d03b736927034ac658201a7ff2154b3526d534.jpg)

![faf478c9724286e5b195a0022399bf17dce8b3b9bc87dae4ab71f0d98dfcbec6.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/images/faf478c9724286e5b195a0022399bf17dce8b3b9bc87dae4ab71f0d98dfcbec6.jpg)

### Tables

![06b454d0d5b0ce3763454f092fd3ea5d99d186edb0079584d1d87bede3e45ce4.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/tables/06b454d0d5b0ce3763454f092fd3ea5d99d186edb0079584d1d87bede3e45ce4.jpg)

![5bc3bae6c97e16ac714c3f2a64024cde7ec3ed6bc0d36a06ce8cf4d05197b85e.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/tables/5bc3bae6c97e16ac714c3f2a64024cde7ec3ed6bc0d36a06ce8cf4d05197b85e.jpg)

![7db0154af22e4d65b13eaa946034057458b872863895e9be8141bc456848a2b9.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/tables/7db0154af22e4d65b13eaa946034057458b872863895e9be8141bc456848a2b9.jpg)

![933e82edcbb634a712ead54f69c27c8c6cb7d60367fd1797887e58b57340421c.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/tables/933e82edcbb634a712ead54f69c27c8c6cb7d60367fd1797887e58b57340421c.jpg)

![9e7e2419d6fbd684bb92fa3dc96a44e9c89cb88737e050ac3d17e2990204a170.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/tables/9e7e2419d6fbd684bb92fa3dc96a44e9c89cb88737e050ac3d17e2990204a170.jpg)

![cf6b58a3259ddae9d652cf4f7ce4a6710ebb546c947cb36341d645734e45303a.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/tables/cf6b58a3259ddae9d652cf4f7ce4a6710ebb546c947cb36341d645734e45303a.jpg)

![f18a93327a80887baea7d817d7c0a3845ad20ede65373e284fb621288a836b9a.jpg](../icml_results/2986_FOCUS_%20DLLMs%20Know%20How%20to%20Tame%20Their%20Compute%20Bound/tables/f18a93327a80887baea7d817d7c0a3845ad20ede65373e284fb621288a836b9a.jpg)

<a id="paper-2949"></a>

## VAnim: Rendering-Aware Sparse State Modeling for Structure-Preserving Vector Animation


### Images

![5a68592cc01fa947a2adfab2ab67d8a05fdb48b5ab1313435d26d507cd1086eb.jpg](../icml_results/2987_VAnim_%20Rendering-Aware%20Sparse%20State%20Modeling%20for%20Structure-Preserving%20Vector%20Animation/images/5a68592cc01fa947a2adfab2ab67d8a05fdb48b5ab1313435d26d507cd1086eb.jpg)

![794fd841846b8a49516fb155a5dcdc29fe29b563e68f60cd643f0005edabdf01.jpg](../icml_results/2987_VAnim_%20Rendering-Aware%20Sparse%20State%20Modeling%20for%20Structure-Preserving%20Vector%20Animation/images/794fd841846b8a49516fb155a5dcdc29fe29b563e68f60cd643f0005edabdf01.jpg)

![926483540a69d3d9ebe804dd43b5d350cf9a05fe2671b437c944b866a6955705.jpg](../icml_results/2987_VAnim_%20Rendering-Aware%20Sparse%20State%20Modeling%20for%20Structure-Preserving%20Vector%20Animation/images/926483540a69d3d9ebe804dd43b5d350cf9a05fe2671b437c944b866a6955705.jpg)

![a77e75de68a98cb2963b37d90797c100bc0abbd3f0a597bb78298800f3300326.jpg](../icml_results/2987_VAnim_%20Rendering-Aware%20Sparse%20State%20Modeling%20for%20Structure-Preserving%20Vector%20Animation/images/a77e75de68a98cb2963b37d90797c100bc0abbd3f0a597bb78298800f3300326.jpg)

![acce5853863d22067643311856878077d125dff553c7a9ebe886a0728113180e.jpg](../icml_results/2987_VAnim_%20Rendering-Aware%20Sparse%20State%20Modeling%20for%20Structure-Preserving%20Vector%20Animation/images/acce5853863d22067643311856878077d125dff553c7a9ebe886a0728113180e.jpg)

![d531def133c26bd204996fcb1c4f3177b1e7918a6d54f0f3dccd873ead010966.jpg](../icml_results/2987_VAnim_%20Rendering-Aware%20Sparse%20State%20Modeling%20for%20Structure-Preserving%20Vector%20Animation/images/d531def133c26bd204996fcb1c4f3177b1e7918a6d54f0f3dccd873ead010966.jpg)

![e273d151eb68041baaabb3c04c56deac5145e6dd574567c8069e2e754ee5bbb0.jpg](../icml_results/2987_VAnim_%20Rendering-Aware%20Sparse%20State%20Modeling%20for%20Structure-Preserving%20Vector%20Animation/images/e273d151eb68041baaabb3c04c56deac5145e6dd574567c8069e2e754ee5bbb0.jpg)

![eb04b80ce35acf031844c6227d3df8cffbbb2454f284a14e3a7fd79bf96d4a23.jpg](../icml_results/2987_VAnim_%20Rendering-Aware%20Sparse%20State%20Modeling%20for%20Structure-Preserving%20Vector%20Animation/images/eb04b80ce35acf031844c6227d3df8cffbbb2454f284a14e3a7fd79bf96d4a23.jpg)

![fdc4059697fa4098673555591b732fe95828e90735380cf8ec939e6f9ae8ebb3.jpg](../icml_results/2987_VAnim_%20Rendering-Aware%20Sparse%20State%20Modeling%20for%20Structure-Preserving%20Vector%20Animation/images/fdc4059697fa4098673555591b732fe95828e90735380cf8ec939e6f9ae8ebb3.jpg)

### Tables

![0098aff3260af4479ea4ba9eee4f91f26025202338502042e8fdfad5e773e47c.jpg](../icml_results/2987_VAnim_%20Rendering-Aware%20Sparse%20State%20Modeling%20for%20Structure-Preserving%20Vector%20Animation/tables/0098aff3260af4479ea4ba9eee4f91f26025202338502042e8fdfad5e773e47c.jpg)

![24d3fcd454dc93be8717b1c375bd694f673053e9941cfff73ed3e361dfb46b45.jpg](../icml_results/2987_VAnim_%20Rendering-Aware%20Sparse%20State%20Modeling%20for%20Structure-Preserving%20Vector%20Animation/tables/24d3fcd454dc93be8717b1c375bd694f673053e9941cfff73ed3e361dfb46b45.jpg)

![6c856d5196842c0fc9de224fae3c04d22c97022a03ee5359324caa3be6fd5359.jpg](../icml_results/2987_VAnim_%20Rendering-Aware%20Sparse%20State%20Modeling%20for%20Structure-Preserving%20Vector%20Animation/tables/6c856d5196842c0fc9de224fae3c04d22c97022a03ee5359324caa3be6fd5359.jpg)

![b72d4af73d83b2bcc156c76e75b88e3f73660a62c075c4632edc5f5d4fb55697.jpg](../icml_results/2987_VAnim_%20Rendering-Aware%20Sparse%20State%20Modeling%20for%20Structure-Preserving%20Vector%20Animation/tables/b72d4af73d83b2bcc156c76e75b88e3f73660a62c075c4632edc5f5d4fb55697.jpg)

<a id="paper-2950"></a>

## Multilingual Safety Alignment Via Sparse Weight Editing


### Images

![6476fb0ccd79824b3290ced3987251a8d596ad118cfb8aa5b59c134d927f6096.jpg](../icml_results/2988_Multilingual%20Safety%20Alignment%20Via%20Sparse%20Weight%20Editing/images/6476fb0ccd79824b3290ced3987251a8d596ad118cfb8aa5b59c134d927f6096.jpg)

![f831e5d77df84be73c3b025cab992f8eaf7aeb5b295342eab0ff5a4c0ae41333.jpg](../icml_results/2988_Multilingual%20Safety%20Alignment%20Via%20Sparse%20Weight%20Editing/images/f831e5d77df84be73c3b025cab992f8eaf7aeb5b295342eab0ff5a4c0ae41333.jpg)

### Tables

![0bb180c133a973c7f1c1ae3ad166742b0da8bc09180a9ffc1067632e7a38fac9.jpg](../icml_results/2988_Multilingual%20Safety%20Alignment%20Via%20Sparse%20Weight%20Editing/tables/0bb180c133a973c7f1c1ae3ad166742b0da8bc09180a9ffc1067632e7a38fac9.jpg)

![1d6c6a9666b938c6fa56410acda9453db804aba1ef4bc3c14a72fdd915f6e3d0.jpg](../icml_results/2988_Multilingual%20Safety%20Alignment%20Via%20Sparse%20Weight%20Editing/tables/1d6c6a9666b938c6fa56410acda9453db804aba1ef4bc3c14a72fdd915f6e3d0.jpg)

![335d759d0110db8ea0e18db55cfb0ef3f87da9c0934fd6a8aece3e61995ce52b.jpg](../icml_results/2988_Multilingual%20Safety%20Alignment%20Via%20Sparse%20Weight%20Editing/tables/335d759d0110db8ea0e18db55cfb0ef3f87da9c0934fd6a8aece3e61995ce52b.jpg)

![383b126fafe7a1011f71ddf77ebd3496f0f3bfd80e9307829f253e0c7fc5dca5.jpg](../icml_results/2988_Multilingual%20Safety%20Alignment%20Via%20Sparse%20Weight%20Editing/tables/383b126fafe7a1011f71ddf77ebd3496f0f3bfd80e9307829f253e0c7fc5dca5.jpg)

![73b55876ac000002e5ae9a048c577b677349f6a0325db4b3e64a3a17343e685e.jpg](../icml_results/2988_Multilingual%20Safety%20Alignment%20Via%20Sparse%20Weight%20Editing/tables/73b55876ac000002e5ae9a048c577b677349f6a0325db4b3e64a3a17343e685e.jpg)

![9ddb0428f9d405ef8d3d1c7faa27967e5289d74227b24ee63efeca8bc2a91aa3.jpg](../icml_results/2988_Multilingual%20Safety%20Alignment%20Via%20Sparse%20Weight%20Editing/tables/9ddb0428f9d405ef8d3d1c7faa27967e5289d74227b24ee63efeca8bc2a91aa3.jpg)

![9ee8e097ee8cb21068cd4a6c4c5ee917f522a63e3c6808e66a3a18483abc8a4f.jpg](../icml_results/2988_Multilingual%20Safety%20Alignment%20Via%20Sparse%20Weight%20Editing/tables/9ee8e097ee8cb21068cd4a6c4c5ee917f522a63e3c6808e66a3a18483abc8a4f.jpg)

![a68dd0496efea8e1e017a40554e490d204b3fc8ed2605610ec48b87a0b3b33ed.jpg](../icml_results/2988_Multilingual%20Safety%20Alignment%20Via%20Sparse%20Weight%20Editing/tables/a68dd0496efea8e1e017a40554e490d204b3fc8ed2605610ec48b87a0b3b33ed.jpg)

![bb4e8d55426e89330924baeda2ee063a9d78296da7f5c5416a28dbf52e3d125e.jpg](../icml_results/2988_Multilingual%20Safety%20Alignment%20Via%20Sparse%20Weight%20Editing/tables/bb4e8d55426e89330924baeda2ee063a9d78296da7f5c5416a28dbf52e3d125e.jpg)

![c36f7413a7f56b5030d9c0185b1ef4e70b3c51c7a1947a7736d95ce7d0237b82.jpg](../icml_results/2988_Multilingual%20Safety%20Alignment%20Via%20Sparse%20Weight%20Editing/tables/c36f7413a7f56b5030d9c0185b1ef4e70b3c51c7a1947a7736d95ce7d0237b82.jpg)

![c56904e9eeb5adb83269a651d58fbaf08d6d4d847fcdb16806fd4908cc85aa19.jpg](../icml_results/2988_Multilingual%20Safety%20Alignment%20Via%20Sparse%20Weight%20Editing/tables/c56904e9eeb5adb83269a651d58fbaf08d6d4d847fcdb16806fd4908cc85aa19.jpg)

![d7778fccfebecb5c0562f90e567f3b3912592df846a8758875dd074b38d59f39.jpg](../icml_results/2988_Multilingual%20Safety%20Alignment%20Via%20Sparse%20Weight%20Editing/tables/d7778fccfebecb5c0562f90e567f3b3912592df846a8758875dd074b38d59f39.jpg)

![f2c35f979899c7b288ac87468848cbed8b5d8d3428adfe61a090403b19432030.jpg](../icml_results/2988_Multilingual%20Safety%20Alignment%20Via%20Sparse%20Weight%20Editing/tables/f2c35f979899c7b288ac87468848cbed8b5d8d3428adfe61a090403b19432030.jpg)

<a id="paper-2951"></a>

## CL-GCL: Comprehensive and Lightweight Graph Contrastive Learning


### Images

![6d7d8faf695139be9c7b5f3c5aeb458a2cd9eca31a6ba320553473c1685dcacf.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/images/6d7d8faf695139be9c7b5f3c5aeb458a2cd9eca31a6ba320553473c1685dcacf.jpg)

![ab80708a1a8938c263480ccb9ae4169a23ac167d052f566015388c9bde8a3b0c.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/images/ab80708a1a8938c263480ccb9ae4169a23ac167d052f566015388c9bde8a3b0c.jpg)

![b9c838dfb99da7851f53679821c3181437c421435af409da4611714c5056ebae.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/images/b9c838dfb99da7851f53679821c3181437c421435af409da4611714c5056ebae.jpg)

![ddd578d040a06091503a910e2b68963eb882c388ba380964ce384bb6104d8d39.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/images/ddd578d040a06091503a910e2b68963eb882c388ba380964ce384bb6104d8d39.jpg)

### Tables

![0035e75704f12aedf8a33047edec5664c4e7be4163270336ded101769885798e.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/tables/0035e75704f12aedf8a33047edec5664c4e7be4163270336ded101769885798e.jpg)

![154905fc733fd8b8a415b14817cc8fc79ac8f9784a113a2917c125ca2da3d4eb.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/tables/154905fc733fd8b8a415b14817cc8fc79ac8f9784a113a2917c125ca2da3d4eb.jpg)

![27c92959eff6bd4ce5e0376704fdb25ee7d9d7566843a94bc51621a9f9ac7556.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/tables/27c92959eff6bd4ce5e0376704fdb25ee7d9d7566843a94bc51621a9f9ac7556.jpg)

![4c41e875057c63de4c62dcac8356dbc4867b761cb42c8389f5ab92c1aaf10efc.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/tables/4c41e875057c63de4c62dcac8356dbc4867b761cb42c8389f5ab92c1aaf10efc.jpg)

![4f1e7c3e467149ae37d13290cbbfa09cd4487f6f84750bb2802c77c7bb6f793b.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/tables/4f1e7c3e467149ae37d13290cbbfa09cd4487f6f84750bb2802c77c7bb6f793b.jpg)

![5d1bf4aa12b0d489bf990e5061416d4f6a713d1f971e38702f650a30e2cfd2b8.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/tables/5d1bf4aa12b0d489bf990e5061416d4f6a713d1f971e38702f650a30e2cfd2b8.jpg)

![7c24a538380cc0753808306d228fb8922b9bd487048f734d2f0814644c9c07cc.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/tables/7c24a538380cc0753808306d228fb8922b9bd487048f734d2f0814644c9c07cc.jpg)

![92cf4c940df5840f1504fcb01d8edafc2f44024b19fafbe4110863a0f059a985.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/tables/92cf4c940df5840f1504fcb01d8edafc2f44024b19fafbe4110863a0f059a985.jpg)

![939d8d1323b5476404489f99c0b1545f554fd1118407868087be1175e8127a1f.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/tables/939d8d1323b5476404489f99c0b1545f554fd1118407868087be1175e8127a1f.jpg)

![981a99f615256cc2918b566d83e0aa9ead3350d8a3fde4ce28aa5d48fd6b6899.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/tables/981a99f615256cc2918b566d83e0aa9ead3350d8a3fde4ce28aa5d48fd6b6899.jpg)

![9afeeda118bbd38c5905090ead7f4a6e1001c0e2a99be1bf79acf770362503a5.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/tables/9afeeda118bbd38c5905090ead7f4a6e1001c0e2a99be1bf79acf770362503a5.jpg)

![b761476b618b3d0d7819542ebf0c86c2259143123244f25286aadbeda2c6e608.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/tables/b761476b618b3d0d7819542ebf0c86c2259143123244f25286aadbeda2c6e608.jpg)

![bb4192794729179076e3feb933dbfb6a263324defe0f81bb8b66442745bf4cda.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/tables/bb4192794729179076e3feb933dbfb6a263324defe0f81bb8b66442745bf4cda.jpg)

![c191bc0bfa10164ba2a3d7ae956029250d74396970274be5e5168adf92550cc7.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/tables/c191bc0bfa10164ba2a3d7ae956029250d74396970274be5e5168adf92550cc7.jpg)

![da98bda03eaaa3fcca68d6cf5263447062834d9d1ec805a111cf42030cc2c399.jpg](../icml_results/2989_CL-GCL_%20Comprehensive%20and%20Lightweight%20Graph%20Contrastive%20Learning/tables/da98bda03eaaa3fcca68d6cf5263447062834d9d1ec805a111cf42030cc2c399.jpg)

<a id="paper-2952"></a>

## SceneDirector: Bridging Explicit Geometry and Generative Priors for Unified Driving Scene Editing


### Images

![05642a6e937ccf10667f24d01bca93d1db54051c78f4b01e7a31f32b735195ca.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/images/05642a6e937ccf10667f24d01bca93d1db54051c78f4b01e7a31f32b735195ca.jpg)

![1735202508d2765a75e13dd70583e824f3e5ca79ba4943c562005d5be8cd936e.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/images/1735202508d2765a75e13dd70583e824f3e5ca79ba4943c562005d5be8cd936e.jpg)

![28c0842729a188c2b5c5b490d3af338a0cfd8b49d07028bb075458fb9e4c8b6c.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/images/28c0842729a188c2b5c5b490d3af338a0cfd8b49d07028bb075458fb9e4c8b6c.jpg)

![385fc76856c910cab5b2a6b42c581f063f5ccb6dcf0edf416d2ba06749deca2c.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/images/385fc76856c910cab5b2a6b42c581f063f5ccb6dcf0edf416d2ba06749deca2c.jpg)

![7d5422e4c4f361a6ce58eac77ddcfd0108fd1557799188d63e7d0a6a6a5b5736.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/images/7d5422e4c4f361a6ce58eac77ddcfd0108fd1557799188d63e7d0a6a6a5b5736.jpg)

![8d90473a2345b702a6aac4501464d490c59ff22f84f7ee6e057bb74b86a9b6d0.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/images/8d90473a2345b702a6aac4501464d490c59ff22f84f7ee6e057bb74b86a9b6d0.jpg)

![973e09b002347dd6b5cb026799f9533e7f97d0f6d1d8b48ec67e6d2a9f4e8285.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/images/973e09b002347dd6b5cb026799f9533e7f97d0f6d1d8b48ec67e6d2a9f4e8285.jpg)

![99d4bc389db9ef811a3ace4f0d2197a4c3f5d2cc840dd0ad94b6df5494b4bfab.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/images/99d4bc389db9ef811a3ace4f0d2197a4c3f5d2cc840dd0ad94b6df5494b4bfab.jpg)

![ee10684912310f4b08f83f370f562a5adf4a82d03508e292f7f9a56481e07116.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/images/ee10684912310f4b08f83f370f562a5adf4a82d03508e292f7f9a56481e07116.jpg)

![f6ccd56814cbfe0225c164541441c305e832664601675f27e6f0ca6b8521a70a.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/images/f6ccd56814cbfe0225c164541441c305e832664601675f27e6f0ca6b8521a70a.jpg)

### Tables

![17fe6bd9a13af6874538543f64ce0160d72817b6ac679f62d07c6149adbeda73.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/tables/17fe6bd9a13af6874538543f64ce0160d72817b6ac679f62d07c6149adbeda73.jpg)

![3d69fc8bdebff4f1d3a9ac133a72c81e9a2f19c3066ccf4b140433ce119181df.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/tables/3d69fc8bdebff4f1d3a9ac133a72c81e9a2f19c3066ccf4b140433ce119181df.jpg)

![55694c24a444e31158dfb98f1e77c8b5635249ae6dd7955f13d4b0791a73dacc.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/tables/55694c24a444e31158dfb98f1e77c8b5635249ae6dd7955f13d4b0791a73dacc.jpg)

![8c11927fb2735e22b1ba7bec399cf5f0007ab4672a3836b48bd2f6af67a7463b.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/tables/8c11927fb2735e22b1ba7bec399cf5f0007ab4672a3836b48bd2f6af67a7463b.jpg)

![adf5e344262447bdeff0eb87d6a86238c6de4698026c49167a3e61f863e9061f.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/tables/adf5e344262447bdeff0eb87d6a86238c6de4698026c49167a3e61f863e9061f.jpg)

![c83d2965e6a9616eb8e92e4d207b74d29468f2384b186d93f39eda4fa84a035e.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/tables/c83d2965e6a9616eb8e92e4d207b74d29468f2384b186d93f39eda4fa84a035e.jpg)

![ced7d6722931485d7789db70a8d477c170cef89e24818e12e287a3f2813e8ec0.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/tables/ced7d6722931485d7789db70a8d477c170cef89e24818e12e287a3f2813e8ec0.jpg)

![d07f7bff9a8581a4719871fda39516d8aa5726ace29de597751a27d1ce9619a0.jpg](../icml_results/2990_SceneDirector_%20Bridging%20Explicit%20Geometry%20and%20Generative%20Priors%20for%20Unified%20Driving%20Scene%20Editing/tables/d07f7bff9a8581a4719871fda39516d8aa5726ace29de597751a27d1ce9619a0.jpg)

<a id="paper-2953"></a>

## LineageFlow: Flow Matching for High-Fidelity Family-Aware Protein Sequence Generation


### Images

![041d58ff0edd952af5c80c7a2c643b504a9e0fbece3c9ecf7a2c2017b5d398ff.jpg](../icml_results/2991_LineageFlow_%20Flow%20Matching%20for%20High-Fidelity%20Family-Aware%20Protein%20Sequence%20Generation/images/041d58ff0edd952af5c80c7a2c643b504a9e0fbece3c9ecf7a2c2017b5d398ff.jpg)

![10aabb994bb9bebc3c041acfd08984c2cd29f935acfdd110fa5460c88b09f803.jpg](../icml_results/2991_LineageFlow_%20Flow%20Matching%20for%20High-Fidelity%20Family-Aware%20Protein%20Sequence%20Generation/images/10aabb994bb9bebc3c041acfd08984c2cd29f935acfdd110fa5460c88b09f803.jpg)

![3b480754065589f8fba1a746076efd046dc2e1640289524357e288bcbb8c8ccf.jpg](../icml_results/2991_LineageFlow_%20Flow%20Matching%20for%20High-Fidelity%20Family-Aware%20Protein%20Sequence%20Generation/images/3b480754065589f8fba1a746076efd046dc2e1640289524357e288bcbb8c8ccf.jpg)

![402014e2fdc9e5c7ee5cd579e6efa55ad3a478f4fffd2451a9bc79099fcce19f.jpg](../icml_results/2991_LineageFlow_%20Flow%20Matching%20for%20High-Fidelity%20Family-Aware%20Protein%20Sequence%20Generation/images/402014e2fdc9e5c7ee5cd579e6efa55ad3a478f4fffd2451a9bc79099fcce19f.jpg)

![49b63e19c89b0296e34e6c6d0b9d1601f0a7d0a8c2bb58ded68fb7085bda4901.jpg](../icml_results/2991_LineageFlow_%20Flow%20Matching%20for%20High-Fidelity%20Family-Aware%20Protein%20Sequence%20Generation/images/49b63e19c89b0296e34e6c6d0b9d1601f0a7d0a8c2bb58ded68fb7085bda4901.jpg)

![b960b500166a8de98c21f9d43839fe6ed39c9833dc54d8e67da0d28467278183.jpg](../icml_results/2991_LineageFlow_%20Flow%20Matching%20for%20High-Fidelity%20Family-Aware%20Protein%20Sequence%20Generation/images/b960b500166a8de98c21f9d43839fe6ed39c9833dc54d8e67da0d28467278183.jpg)

### Tables

![29060589f4de440743727e78a90ade83a88a3672638e9da4722d22c4a01dea64.jpg](../icml_results/2991_LineageFlow_%20Flow%20Matching%20for%20High-Fidelity%20Family-Aware%20Protein%20Sequence%20Generation/tables/29060589f4de440743727e78a90ade83a88a3672638e9da4722d22c4a01dea64.jpg)

![8b468ef4d34a9ed125ef338060f09c3d0b17bcc8b467a6eb9c01b987d58cec15.jpg](../icml_results/2991_LineageFlow_%20Flow%20Matching%20for%20High-Fidelity%20Family-Aware%20Protein%20Sequence%20Generation/tables/8b468ef4d34a9ed125ef338060f09c3d0b17bcc8b467a6eb9c01b987d58cec15.jpg)

<a id="paper-2954"></a>

## UniFLoW: Universal Multi-Modal Federated LoRA Fine-Tuning Framework with Analytical Aggregation


### Images

![2cc9e2842524e1e571ae049a809c170ed2fc28d0fba538d2f97c69cfee09c0f8.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/images/2cc9e2842524e1e571ae049a809c170ed2fc28d0fba538d2f97c69cfee09c0f8.jpg)

![329358d4a12a358af6ad2119786da6af64eda83fbe9b5073c86b73242a28588e.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/images/329358d4a12a358af6ad2119786da6af64eda83fbe9b5073c86b73242a28588e.jpg)

![55ea20a2cca4895e2594e02015e1485520e2c3435a16715af4b5e37ef2453815.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/images/55ea20a2cca4895e2594e02015e1485520e2c3435a16715af4b5e37ef2453815.jpg)

![aa3cd69c1f3e764eb238baff85d5b8e3d2ce86b6b82846f915cb7889c530b980.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/images/aa3cd69c1f3e764eb238baff85d5b8e3d2ce86b6b82846f915cb7889c530b980.jpg)

![dcb3f76d08a1ed479a92c7092953f4e5dba788aa17f9e8e9bced207d59f4cb48.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/images/dcb3f76d08a1ed479a92c7092953f4e5dba788aa17f9e8e9bced207d59f4cb48.jpg)

### Tables

![0f93093ae123dfbd6422551d31294ca2a98771f53b50d27b57c0c170733c6b55.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/0f93093ae123dfbd6422551d31294ca2a98771f53b50d27b57c0c170733c6b55.jpg)

![12bca6a3e09aa1fedd0adf9f4519f7e91b1b34fc60f1ee54014817cec9c48513.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/12bca6a3e09aa1fedd0adf9f4519f7e91b1b34fc60f1ee54014817cec9c48513.jpg)

![24914cfad0d0234ce74cbdcd7431d4d318d5e4e1b3c2aca0a2c575b68422231f.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/24914cfad0d0234ce74cbdcd7431d4d318d5e4e1b3c2aca0a2c575b68422231f.jpg)

![28de3a7a750f1e6e47fa0040541965b6ea1026ad35b8969cc176cee0ca0f4482.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/28de3a7a750f1e6e47fa0040541965b6ea1026ad35b8969cc176cee0ca0f4482.jpg)

![35f36abf45b88262983cef6e473910e5cf59a8b2bbec989d495d641c82297b66.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/35f36abf45b88262983cef6e473910e5cf59a8b2bbec989d495d641c82297b66.jpg)

![366fd1de9ccac515c19669d51d3edb0e15728c719e91c88fc40167a786c12c4d.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/366fd1de9ccac515c19669d51d3edb0e15728c719e91c88fc40167a786c12c4d.jpg)

![4a0652dee56f6197cb4f7f0fe0813fd4b937c24adb84419cf77c8e11ba80a637.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/4a0652dee56f6197cb4f7f0fe0813fd4b937c24adb84419cf77c8e11ba80a637.jpg)

![568abf8120201fbcf218170a7531470f88bfb0b23b3b403b6189d25cc22ee2ac.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/568abf8120201fbcf218170a7531470f88bfb0b23b3b403b6189d25cc22ee2ac.jpg)

![626b28d36acc95ccc8bda8a3f39d209f6b938e12124c38feb704f1cf1cfed341.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/626b28d36acc95ccc8bda8a3f39d209f6b938e12124c38feb704f1cf1cfed341.jpg)

![66f4e6c5cfa7dcb7516b0ef8419a3494c71ad3b2017fd9c235f23401413e4dba.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/66f4e6c5cfa7dcb7516b0ef8419a3494c71ad3b2017fd9c235f23401413e4dba.jpg)

![67a9d25f80add9fa534f5606ada6abd7b33948ef40e249bb2f084e747a7c8a9a.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/67a9d25f80add9fa534f5606ada6abd7b33948ef40e249bb2f084e747a7c8a9a.jpg)

![79375e21be77a83ef888504da37f38c8e9517bae1f7ba14c22218ada64eac890.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/79375e21be77a83ef888504da37f38c8e9517bae1f7ba14c22218ada64eac890.jpg)

![7a286451308298c43f9a5b9f01fa17c26f40a5fc78b5009fe16dcbe4df02d322.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/7a286451308298c43f9a5b9f01fa17c26f40a5fc78b5009fe16dcbe4df02d322.jpg)

![9a715b4f5bcd6b57349e101d027673750dbe045127d2118696265b44059e74e2.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/9a715b4f5bcd6b57349e101d027673750dbe045127d2118696265b44059e74e2.jpg)

![9dabd740a22cddb2e7436ba19f510cb793ae472b287047600271a08094fa8a3b.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/9dabd740a22cddb2e7436ba19f510cb793ae472b287047600271a08094fa8a3b.jpg)

![e42202bd3159e94ec501a7e62986b2a2ac9e5d2815d9e27ea451b35011bbcd3c.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/e42202bd3159e94ec501a7e62986b2a2ac9e5d2815d9e27ea451b35011bbcd3c.jpg)

![f0b1cbc5cbff5c0b2b47ca8e456a9d99f7ce3b356817fd1f29a9a6946958b7fe.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/f0b1cbc5cbff5c0b2b47ca8e456a9d99f7ce3b356817fd1f29a9a6946958b7fe.jpg)

![f2d7996d532a6dc270ba1eb1c7ee1041a48c7bf1e1b491f6574cab6ccd68d411.jpg](../icml_results/2992_UniFLoW_%20Universal%20Multi-Modal%20Federated%20LoRA%20Fine-Tuning%20Framework%20with%20Analytical%20Aggregation/tables/f2d7996d532a6dc270ba1eb1c7ee1041a48c7bf1e1b491f6574cab6ccd68d411.jpg)

<a id="paper-2955"></a>

## Large Language Model Teaches Visual Students: Cross-Modality Transfer of Fine-Grained Conceptual Knowledge


### Images

![0de77345375c9788a81372064ddc271bf1c2e4411472636b6d79e9ce94df7374.jpg](../icml_results/2993_Large%20Language%20Model%20Teaches%20Visual%20Students_%20Cross-Modality%20Transfer%20of%20Fine-Grained%20Conceptual%20Kno/images/0de77345375c9788a81372064ddc271bf1c2e4411472636b6d79e9ce94df7374.jpg)

![22950e520d18b97f0978574818b7f69b2bd687d4673c9bcd966ce1e820efe6e0.jpg](../icml_results/2993_Large%20Language%20Model%20Teaches%20Visual%20Students_%20Cross-Modality%20Transfer%20of%20Fine-Grained%20Conceptual%20Kno/images/22950e520d18b97f0978574818b7f69b2bd687d4673c9bcd966ce1e820efe6e0.jpg)

![33729c6c37118ae128baa44c04b741989dbf9052aff0df62db8970a40924be39.jpg](../icml_results/2993_Large%20Language%20Model%20Teaches%20Visual%20Students_%20Cross-Modality%20Transfer%20of%20Fine-Grained%20Conceptual%20Kno/images/33729c6c37118ae128baa44c04b741989dbf9052aff0df62db8970a40924be39.jpg)

![91476586538cdf520b8eced0a1b359e77ceb1efd96e53a83e5a12b384e20a382.jpg](../icml_results/2993_Large%20Language%20Model%20Teaches%20Visual%20Students_%20Cross-Modality%20Transfer%20of%20Fine-Grained%20Conceptual%20Kno/images/91476586538cdf520b8eced0a1b359e77ceb1efd96e53a83e5a12b384e20a382.jpg)

![9ca87fd77f3c94e0d6b352f6dc9c3581960d359c244c68e3eb5ae1e1695e1444.jpg](../icml_results/2993_Large%20Language%20Model%20Teaches%20Visual%20Students_%20Cross-Modality%20Transfer%20of%20Fine-Grained%20Conceptual%20Kno/images/9ca87fd77f3c94e0d6b352f6dc9c3581960d359c244c68e3eb5ae1e1695e1444.jpg)

![d58c371e5b6257751117c1e096949e8638e6fd87d68bba03380ea6b005ae2ce2.jpg](../icml_results/2993_Large%20Language%20Model%20Teaches%20Visual%20Students_%20Cross-Modality%20Transfer%20of%20Fine-Grained%20Conceptual%20Kno/images/d58c371e5b6257751117c1e096949e8638e6fd87d68bba03380ea6b005ae2ce2.jpg)

![e3b8c481c66efbf12f404a64a13cfa82ff1a72d55eada9fd8aeb4202abcf0c09.jpg](../icml_results/2993_Large%20Language%20Model%20Teaches%20Visual%20Students_%20Cross-Modality%20Transfer%20of%20Fine-Grained%20Conceptual%20Kno/images/e3b8c481c66efbf12f404a64a13cfa82ff1a72d55eada9fd8aeb4202abcf0c09.jpg)

![ff803153b8df5c2a6af0724c6cfcd49958d9a8237c588890cdcd446606ead026.jpg](../icml_results/2993_Large%20Language%20Model%20Teaches%20Visual%20Students_%20Cross-Modality%20Transfer%20of%20Fine-Grained%20Conceptual%20Kno/images/ff803153b8df5c2a6af0724c6cfcd49958d9a8237c588890cdcd446606ead026.jpg)

### Tables

![234cd98536def565f076e297c85cc927a87d68a8e83a2e2cc85273e72607a93a.jpg](../icml_results/2993_Large%20Language%20Model%20Teaches%20Visual%20Students_%20Cross-Modality%20Transfer%20of%20Fine-Grained%20Conceptual%20Kno/tables/234cd98536def565f076e297c85cc927a87d68a8e83a2e2cc85273e72607a93a.jpg)

![46adde867d80009be4611a1d1a73c2588a7a0f1e15bcca8e7a4e475325550464.jpg](../icml_results/2993_Large%20Language%20Model%20Teaches%20Visual%20Students_%20Cross-Modality%20Transfer%20of%20Fine-Grained%20Conceptual%20Kno/tables/46adde867d80009be4611a1d1a73c2588a7a0f1e15bcca8e7a4e475325550464.jpg)

![63b8e0e158687b15a94bcd425d1d33571a143c222bcc75c51360be4a59688d68.jpg](../icml_results/2993_Large%20Language%20Model%20Teaches%20Visual%20Students_%20Cross-Modality%20Transfer%20of%20Fine-Grained%20Conceptual%20Kno/tables/63b8e0e158687b15a94bcd425d1d33571a143c222bcc75c51360be4a59688d68.jpg)

![6a90bee19d8ed2a97f0873565680d0ebec28fd10caae5707ea52313683dd2938.jpg](../icml_results/2993_Large%20Language%20Model%20Teaches%20Visual%20Students_%20Cross-Modality%20Transfer%20of%20Fine-Grained%20Conceptual%20Kno/tables/6a90bee19d8ed2a97f0873565680d0ebec28fd10caae5707ea52313683dd2938.jpg)

![8d170b8c733c4d132a5c0653eb12203219f26e4604e7225c47f0b92b83093b41.jpg](../icml_results/2993_Large%20Language%20Model%20Teaches%20Visual%20Students_%20Cross-Modality%20Transfer%20of%20Fine-Grained%20Conceptual%20Kno/tables/8d170b8c733c4d132a5c0653eb12203219f26e4604e7225c47f0b92b83093b41.jpg)

![b634a044a4179239c0e113d4c25d8cbccfab88e36b127da7a2fa79b23ed18c8c.jpg](../icml_results/2993_Large%20Language%20Model%20Teaches%20Visual%20Students_%20Cross-Modality%20Transfer%20of%20Fine-Grained%20Conceptual%20Kno/tables/b634a044a4179239c0e113d4c25d8cbccfab88e36b127da7a2fa79b23ed18c8c.jpg)

![d6949f4b52407a808629b04845f405332c72c382752aaa9675696fdf14d26f4a.jpg](../icml_results/2993_Large%20Language%20Model%20Teaches%20Visual%20Students_%20Cross-Modality%20Transfer%20of%20Fine-Grained%20Conceptual%20Kno/tables/d6949f4b52407a808629b04845f405332c72c382752aaa9675696fdf14d26f4a.jpg)

![e57eaba0bf88238e4ebc147f8e05766ebcf9b802215a13c951088c963cf2d774.jpg](../icml_results/2993_Large%20Language%20Model%20Teaches%20Visual%20Students_%20Cross-Modality%20Transfer%20of%20Fine-Grained%20Conceptual%20Kno/tables/e57eaba0bf88238e4ebc147f8e05766ebcf9b802215a13c951088c963cf2d774.jpg)

<a id="paper-2956"></a>

## RealisMotion: Decomposed Human Motion Control and Video Generation in the World Space


### Images

![04179fcdf850a3d3bb04302f003e6c3ebceabc011a3aaa4f5052a758e1908d5e.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/images/04179fcdf850a3d3bb04302f003e6c3ebceabc011a3aaa4f5052a758e1908d5e.jpg)

![07b25fa52cfe8dabefa2ef45f53b1a5e333f83d6e6fd4ff0f50a9916e17f3793.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/images/07b25fa52cfe8dabefa2ef45f53b1a5e333f83d6e6fd4ff0f50a9916e17f3793.jpg)

![0aaf38a4dca4155956402f0413697973dad1de5a157f2c5c5a03262b5dc3c449.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/images/0aaf38a4dca4155956402f0413697973dad1de5a157f2c5c5a03262b5dc3c449.jpg)

![297a17d4ba984052b20b748636ebc6b55e16a4822634455201e83613014a72fd.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/images/297a17d4ba984052b20b748636ebc6b55e16a4822634455201e83613014a72fd.jpg)

![68e22cdb8a52df0ee624092bf3b1c7740e0dc5f7e6d78037291eb2ffbc6a2ddc.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/images/68e22cdb8a52df0ee624092bf3b1c7740e0dc5f7e6d78037291eb2ffbc6a2ddc.jpg)

![69c8f8e8bd7be6f468ab0b50b96ed1c6b21cbadd6a7ae085c5cc455cb82ac252.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/images/69c8f8e8bd7be6f468ab0b50b96ed1c6b21cbadd6a7ae085c5cc455cb82ac252.jpg)

![78dfd1300d1a914d2ac56bc0adb4563f2e47f0fa648c49b27e792576fb75da7d.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/images/78dfd1300d1a914d2ac56bc0adb4563f2e47f0fa648c49b27e792576fb75da7d.jpg)

![7969b6187bd4fde17cfe5387e5a2a98f0091f03485750e4555a04bbdce3a18b7.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/images/7969b6187bd4fde17cfe5387e5a2a98f0091f03485750e4555a04bbdce3a18b7.jpg)

![98120f4b419cfe1c33c18ef0bfc9a5f02afdc1f212a668eb400ec9a81298c55b.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/images/98120f4b419cfe1c33c18ef0bfc9a5f02afdc1f212a668eb400ec9a81298c55b.jpg)

![9a95c452354ec9247a3713a48c32c45fa984db8942bb3470be1781367a8bf7b9.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/images/9a95c452354ec9247a3713a48c32c45fa984db8942bb3470be1781367a8bf7b9.jpg)

![9ba9c1424728b372bfa5019500e7f6be180dc6aa542e5ea9c3e581860d3b2daf.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/images/9ba9c1424728b372bfa5019500e7f6be180dc6aa542e5ea9c3e581860d3b2daf.jpg)

![b46a6bdffdc741deffdc18f72554ae93a3558086114bcc7a22539510b8b0e8c4.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/images/b46a6bdffdc741deffdc18f72554ae93a3558086114bcc7a22539510b8b0e8c4.jpg)

![bdde2c709f6ef0d1bd6fa05a7d6f6d2c4fdab4abce4b7738a70283d33c1b0230.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/images/bdde2c709f6ef0d1bd6fa05a7d6f6d2c4fdab4abce4b7738a70283d33c1b0230.jpg)

![d22a41cf441369f87a61cc1a0cd41da5325029b25fdf2b2782e9da6034cb63a3.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/images/d22a41cf441369f87a61cc1a0cd41da5325029b25fdf2b2782e9da6034cb63a3.jpg)

![d4a6dfa753bae6b9ef20b12399d49db8222398ea9cfffc38a078c37dd843057f.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/images/d4a6dfa753bae6b9ef20b12399d49db8222398ea9cfffc38a078c37dd843057f.jpg)

![e24a831454d1503c04781838f755c053905811d73d8679141eb07a1956a935d3.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/images/e24a831454d1503c04781838f755c053905811d73d8679141eb07a1956a935d3.jpg)

![fdc29f31f00574559e515c8eb0a9fbf2c60399a503444cbe2e22c8f37f71d2bf.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/images/fdc29f31f00574559e515c8eb0a9fbf2c60399a503444cbe2e22c8f37f71d2bf.jpg)

### Tables

![2000b2d392d84c82a1985a8689d311d2979a87018ca437f1b23e45fe252ca75f.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/tables/2000b2d392d84c82a1985a8689d311d2979a87018ca437f1b23e45fe252ca75f.jpg)

![4f58cee335163c5aeafa0c12a1540e7b7075c4d1a407f96c2c6af1dbb5ac2eaf.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/tables/4f58cee335163c5aeafa0c12a1540e7b7075c4d1a407f96c2c6af1dbb5ac2eaf.jpg)

![521caddd5f9a334eaa0a6dbf514a87923f8492510cf99dc3784ad698d5188848.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/tables/521caddd5f9a334eaa0a6dbf514a87923f8492510cf99dc3784ad698d5188848.jpg)

![7cc3061103cbacccc4b1f7ce139bb22adfcb7302d9c86d5707d477240e08fe0a.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/tables/7cc3061103cbacccc4b1f7ce139bb22adfcb7302d9c86d5707d477240e08fe0a.jpg)

![8ccb94b0f3b0a45d3d29b7dd4fdbb0bd89f51045194875f923b9a6a960acf9c3.jpg](../icml_results/2994_RealisMotion_%20Decomposed%20Human%20Motion%20Control%20and%20Video%20Generation%20in%20the%20World%20Space/tables/8ccb94b0f3b0a45d3d29b7dd4fdbb0bd89f51045194875f923b9a6a960acf9c3.jpg)

<a id="paper-2957"></a>

## SpikingLM: Towards Fully Spiking Language Model


### Images

![040542be5a85cd3de95959bb4f3b5ec0de67bcf99b54552a0e5f028489147c5d.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/images/040542be5a85cd3de95959bb4f3b5ec0de67bcf99b54552a0e5f028489147c5d.jpg)

![10767402cb8eb6667e26197bfee9fe64246bb4a0e9a2d4b188f6d705a6e1bb52.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/images/10767402cb8eb6667e26197bfee9fe64246bb4a0e9a2d4b188f6d705a6e1bb52.jpg)

![35ad9f18c6cce9585945e3013873ce29463ce384be4c65a31a2e41a27da7cbed.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/images/35ad9f18c6cce9585945e3013873ce29463ce384be4c65a31a2e41a27da7cbed.jpg)

![36863e961aabf19c7823918ac8645bdbe5906f78a51c152fda5e0973a4894b9b.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/images/36863e961aabf19c7823918ac8645bdbe5906f78a51c152fda5e0973a4894b9b.jpg)

![3db437996f0ff4f2625b5f6f3ca02e930afecd7bfc97926b0fccc1d3d9647dee.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/images/3db437996f0ff4f2625b5f6f3ca02e930afecd7bfc97926b0fccc1d3d9647dee.jpg)

![4ff2fb0adb20dd128b061c3ea9689e0fbd316b5f22b17e7f762f7f2c76a498c0.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/images/4ff2fb0adb20dd128b061c3ea9689e0fbd316b5f22b17e7f762f7f2c76a498c0.jpg)

![76cfbe84510d44950feedf4ecf31172779003a20d13c349e23ad140f349981ff.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/images/76cfbe84510d44950feedf4ecf31172779003a20d13c349e23ad140f349981ff.jpg)

![88773b78ebc40d8eb0e1e9e202cdf46822729c2f6911e1a3d0d9bbef19e8ad14.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/images/88773b78ebc40d8eb0e1e9e202cdf46822729c2f6911e1a3d0d9bbef19e8ad14.jpg)

![c77df225eecd9d4f7fd45e9052ad1a4509294a0eae473d8110806071e290f8fc.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/images/c77df225eecd9d4f7fd45e9052ad1a4509294a0eae473d8110806071e290f8fc.jpg)

![d6ad510011f0b9d437b87d50d5cb2e819816f75ad60d6fe71478b320467e8c4e.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/images/d6ad510011f0b9d437b87d50d5cb2e819816f75ad60d6fe71478b320467e8c4e.jpg)

![f70b2f46d8d0640eff79a95eaa3818ded350cec4d7e127abecb0f3b1afe7f0e5.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/images/f70b2f46d8d0640eff79a95eaa3818ded350cec4d7e127abecb0f3b1afe7f0e5.jpg)

![fb63c4fa1a0827220d5362c06676e160baeace39a17a3368b39e040f2feb0658.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/images/fb63c4fa1a0827220d5362c06676e160baeace39a17a3368b39e040f2feb0658.jpg)

### Tables

![0056a4a5d6c8bb00167b87f0135c5fc18dcf1400185fbbe784a33c7a45bbfd31.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/tables/0056a4a5d6c8bb00167b87f0135c5fc18dcf1400185fbbe784a33c7a45bbfd31.jpg)

![0b92fb4fd077276b27ddb98c071f3285f9f9435f0ae971ea986b4db18cec2672.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/tables/0b92fb4fd077276b27ddb98c071f3285f9f9435f0ae971ea986b4db18cec2672.jpg)

![4af8c7281d42ba0bd6a175e262884b691a9400df56ab3943eef89a873a252041.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/tables/4af8c7281d42ba0bd6a175e262884b691a9400df56ab3943eef89a873a252041.jpg)

![5a1ce6cc09231eabe4aa0e6d8bb5210336899422c426a9f045d2d3dd42ee26ea.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/tables/5a1ce6cc09231eabe4aa0e6d8bb5210336899422c426a9f045d2d3dd42ee26ea.jpg)

![6a6b0c7554137f418505e4a8c71defa323b16beebcf8eab91d14350bd53d2319.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/tables/6a6b0c7554137f418505e4a8c71defa323b16beebcf8eab91d14350bd53d2319.jpg)

![dcb79422b260833e00aaa79b6359854cf802315b795464df629835e2df690e5f.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/tables/dcb79422b260833e00aaa79b6359854cf802315b795464df629835e2df690e5f.jpg)

![dde261c917e7c3e451bac3c1c0dc6dd1e12e2576fb2919931a379a5a284aa7d6.jpg](../icml_results/2995_SpikingLM_%20Towards%20Fully%20Spiking%20Language%20Model/tables/dde261c917e7c3e451bac3c1c0dc6dd1e12e2576fb2919931a379a5a284aa7d6.jpg)

<a id="paper-2958"></a>

## G-RANS: Generalizable Residual-Aware Neural Solvers for Sparse Systems


### Images

![6a467ea2aedefe734be72c1ed04656a6eba3a4a5e66242ac4accb9dadd844d5f.jpg](../icml_results/2996_G-RANS_%20Generalizable%20Residual-Aware%20Neural%20Solvers%20for%20Sparse%20Systems/images/6a467ea2aedefe734be72c1ed04656a6eba3a4a5e66242ac4accb9dadd844d5f.jpg)

![b430913c945a267e786e0a85fb326adfe11dd983b15a904a3af61da5309f2449.jpg](../icml_results/2996_G-RANS_%20Generalizable%20Residual-Aware%20Neural%20Solvers%20for%20Sparse%20Systems/images/b430913c945a267e786e0a85fb326adfe11dd983b15a904a3af61da5309f2449.jpg)

![e55d6b098460ee9a5f47b233b7e2c73b9f59c4568af90daca8407761dbc0d39c.jpg](../icml_results/2996_G-RANS_%20Generalizable%20Residual-Aware%20Neural%20Solvers%20for%20Sparse%20Systems/images/e55d6b098460ee9a5f47b233b7e2c73b9f59c4568af90daca8407761dbc0d39c.jpg)

![ec094f6ae7c0503d51d68b2d78efa07d35596b0ab56321394736631de225c2d4.jpg](../icml_results/2996_G-RANS_%20Generalizable%20Residual-Aware%20Neural%20Solvers%20for%20Sparse%20Systems/images/ec094f6ae7c0503d51d68b2d78efa07d35596b0ab56321394736631de225c2d4.jpg)

### Tables

![1a20adce8fc9dc2f8127f8168e353d6f8cad596c3cd12c15130caad5d3521a47.jpg](../icml_results/2996_G-RANS_%20Generalizable%20Residual-Aware%20Neural%20Solvers%20for%20Sparse%20Systems/tables/1a20adce8fc9dc2f8127f8168e353d6f8cad596c3cd12c15130caad5d3521a47.jpg)

![1c9e0f0e5edbd1658fd09a303ad79f11466daab14e6e9cfdaa9a469dca2c38bc.jpg](../icml_results/2996_G-RANS_%20Generalizable%20Residual-Aware%20Neural%20Solvers%20for%20Sparse%20Systems/tables/1c9e0f0e5edbd1658fd09a303ad79f11466daab14e6e9cfdaa9a469dca2c38bc.jpg)

![513cf422c9fe1176b31c484ebb9b9428e0a71cc7b8b2c5a443dada693bf8c46f.jpg](../icml_results/2996_G-RANS_%20Generalizable%20Residual-Aware%20Neural%20Solvers%20for%20Sparse%20Systems/tables/513cf422c9fe1176b31c484ebb9b9428e0a71cc7b8b2c5a443dada693bf8c46f.jpg)

![538cbf76d896ab0a282a045dff852ba19fa549198846b4136b5b1a112ecf96a2.jpg](../icml_results/2996_G-RANS_%20Generalizable%20Residual-Aware%20Neural%20Solvers%20for%20Sparse%20Systems/tables/538cbf76d896ab0a282a045dff852ba19fa549198846b4136b5b1a112ecf96a2.jpg)

![62fa3c600869a992337f2f7e9a1298c1c7db5c902760fdef78e42e3858c9ebdd.jpg](../icml_results/2996_G-RANS_%20Generalizable%20Residual-Aware%20Neural%20Solvers%20for%20Sparse%20Systems/tables/62fa3c600869a992337f2f7e9a1298c1c7db5c902760fdef78e42e3858c9ebdd.jpg)

![b0eb8172edd9b3118d9bf1cc14707077a81f72da30d1370e8fdd3d65fcb37b83.jpg](../icml_results/2996_G-RANS_%20Generalizable%20Residual-Aware%20Neural%20Solvers%20for%20Sparse%20Systems/tables/b0eb8172edd9b3118d9bf1cc14707077a81f72da30d1370e8fdd3d65fcb37b83.jpg)

![b6283e34cdfeab5cfa9955622e5739eb5cc2503c01db9afc852aeb22ede9a023.jpg](../icml_results/2996_G-RANS_%20Generalizable%20Residual-Aware%20Neural%20Solvers%20for%20Sparse%20Systems/tables/b6283e34cdfeab5cfa9955622e5739eb5cc2503c01db9afc852aeb22ede9a023.jpg)

![bdca6c3fbca5981f79b3645e4a3d23f749c4354e4766e61bfc9d1da6df60e04a.jpg](../icml_results/2996_G-RANS_%20Generalizable%20Residual-Aware%20Neural%20Solvers%20for%20Sparse%20Systems/tables/bdca6c3fbca5981f79b3645e4a3d23f749c4354e4766e61bfc9d1da6df60e04a.jpg)

![c090236c0995a45c91bd0bf360b09275c5785d58e45a78e052c0c984d5fa7f8a.jpg](../icml_results/2996_G-RANS_%20Generalizable%20Residual-Aware%20Neural%20Solvers%20for%20Sparse%20Systems/tables/c090236c0995a45c91bd0bf360b09275c5785d58e45a78e052c0c984d5fa7f8a.jpg)

![dcb3e5f9b76e75f57e34f9ed18d7d16d169668446f24efaf3b6ecff8daf3a0a2.jpg](../icml_results/2996_G-RANS_%20Generalizable%20Residual-Aware%20Neural%20Solvers%20for%20Sparse%20Systems/tables/dcb3e5f9b76e75f57e34f9ed18d7d16d169668446f24efaf3b6ecff8daf3a0a2.jpg)

<a id="paper-2959"></a>

## AlphaRouter: Token-level Routing Between SLM and LLM with Reinforcement Learning and Tree Search


### Images

![5f2d2221e6d340212c646789382b7422b210f01091f92cc55da59a04e8566609.jpg](../icml_results/2997_AlphaRouter_%20Token-level%20Routing%20Between%20SLM%20and%20LLM%20with%20Reinforcement%20Learning%20and%20Tree%20Search/images/5f2d2221e6d340212c646789382b7422b210f01091f92cc55da59a04e8566609.jpg)

![69672bdaf9977a8d768cf3e29fb4272b9666adf87109008bf3e0a1393a4053b7.jpg](../icml_results/2997_AlphaRouter_%20Token-level%20Routing%20Between%20SLM%20and%20LLM%20with%20Reinforcement%20Learning%20and%20Tree%20Search/images/69672bdaf9977a8d768cf3e29fb4272b9666adf87109008bf3e0a1393a4053b7.jpg)

![a5aab1378df829c4a3cdf84b552bfb640ea100e30a0335f7d3ed0817d9f7be06.jpg](../icml_results/2997_AlphaRouter_%20Token-level%20Routing%20Between%20SLM%20and%20LLM%20with%20Reinforcement%20Learning%20and%20Tree%20Search/images/a5aab1378df829c4a3cdf84b552bfb640ea100e30a0335f7d3ed0817d9f7be06.jpg)

### Tables

![00e356e6be356fdbe6eac0cce0b4d95f0f872575489eda5a0a835c8db3a83586.jpg](../icml_results/2997_AlphaRouter_%20Token-level%20Routing%20Between%20SLM%20and%20LLM%20with%20Reinforcement%20Learning%20and%20Tree%20Search/tables/00e356e6be356fdbe6eac0cce0b4d95f0f872575489eda5a0a835c8db3a83586.jpg)

![18943aa58eb05de85bb0e86c44df758b08d14229848e51c71bcce21620976a60.jpg](../icml_results/2997_AlphaRouter_%20Token-level%20Routing%20Between%20SLM%20and%20LLM%20with%20Reinforcement%20Learning%20and%20Tree%20Search/tables/18943aa58eb05de85bb0e86c44df758b08d14229848e51c71bcce21620976a60.jpg)

![58df051f74afe47b155dd61677918f899622f6496a25de4537c4bdc052c50ec7.jpg](../icml_results/2997_AlphaRouter_%20Token-level%20Routing%20Between%20SLM%20and%20LLM%20with%20Reinforcement%20Learning%20and%20Tree%20Search/tables/58df051f74afe47b155dd61677918f899622f6496a25de4537c4bdc052c50ec7.jpg)

![649236b5634244553034af3d07cb67ff3750d3ce5902ccd1e3e288e34a0eba60.jpg](../icml_results/2997_AlphaRouter_%20Token-level%20Routing%20Between%20SLM%20and%20LLM%20with%20Reinforcement%20Learning%20and%20Tree%20Search/tables/649236b5634244553034af3d07cb67ff3750d3ce5902ccd1e3e288e34a0eba60.jpg)

![8712abf31ca57f6d10ad2671e41409432e29896cf149eb77a5c18f8d0e0b666e.jpg](../icml_results/2997_AlphaRouter_%20Token-level%20Routing%20Between%20SLM%20and%20LLM%20with%20Reinforcement%20Learning%20and%20Tree%20Search/tables/8712abf31ca57f6d10ad2671e41409432e29896cf149eb77a5c18f8d0e0b666e.jpg)

![8e7de61feb0dfbaaa8c9b0682ae5bdb3dc46c1807b07730ee6a8010ce5221110.jpg](../icml_results/2997_AlphaRouter_%20Token-level%20Routing%20Between%20SLM%20and%20LLM%20with%20Reinforcement%20Learning%20and%20Tree%20Search/tables/8e7de61feb0dfbaaa8c9b0682ae5bdb3dc46c1807b07730ee6a8010ce5221110.jpg)

![b58382bdcade05e76ae3add393dc44df8a457d2612e12acf6ac2fa91870cd5c5.jpg](../icml_results/2997_AlphaRouter_%20Token-level%20Routing%20Between%20SLM%20and%20LLM%20with%20Reinforcement%20Learning%20and%20Tree%20Search/tables/b58382bdcade05e76ae3add393dc44df8a457d2612e12acf6ac2fa91870cd5c5.jpg)

![d829270a0b54a75eed1eb4693aa59b795cfdd39cfc55a7e40971f44e1ec6ff83.jpg](../icml_results/2997_AlphaRouter_%20Token-level%20Routing%20Between%20SLM%20and%20LLM%20with%20Reinforcement%20Learning%20and%20Tree%20Search/tables/d829270a0b54a75eed1eb4693aa59b795cfdd39cfc55a7e40971f44e1ec6ff83.jpg)

![de77d59a3deafc0029c03897a61b15ff12f34f2c8a6edebf9c3364a8212a4e5f.jpg](../icml_results/2997_AlphaRouter_%20Token-level%20Routing%20Between%20SLM%20and%20LLM%20with%20Reinforcement%20Learning%20and%20Tree%20Search/tables/de77d59a3deafc0029c03897a61b15ff12f34f2c8a6edebf9c3364a8212a4e5f.jpg)

<a id="paper-2960"></a>

## VERA-V: Variational Inference Framework for Jailbreaking Vision-Language Models


### Images

![1cd08e6df8e6f100f66cc74395410e794c8a617b13098708fdad79c6ccc59172.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/images/1cd08e6df8e6f100f66cc74395410e794c8a617b13098708fdad79c6ccc59172.jpg)

![368d9788f8ff90e9d6dc54a24d7542e083bfa8750fcad33964ae20cae4ca4500.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/images/368d9788f8ff90e9d6dc54a24d7542e083bfa8750fcad33964ae20cae4ca4500.jpg)

![3b99ee83941a8d527322c4efd074bde5fb8d05b04eb5514e95d12d17e0527473.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/images/3b99ee83941a8d527322c4efd074bde5fb8d05b04eb5514e95d12d17e0527473.jpg)

![6ef735fa42678ce6af1f263923e866cbb66b705967185131dd1137bca28274c9.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/images/6ef735fa42678ce6af1f263923e866cbb66b705967185131dd1137bca28274c9.jpg)

![7a3ebf7a01899a5a6f96634684a1e84794b5d147603fa5304b736f1ea4edde76.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/images/7a3ebf7a01899a5a6f96634684a1e84794b5d147603fa5304b736f1ea4edde76.jpg)

![8e436f9018eb37431e47a1f8c3aafe8d015b13234ca18212e42c3bc8b5c9b318.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/images/8e436f9018eb37431e47a1f8c3aafe8d015b13234ca18212e42c3bc8b5c9b318.jpg)

### Tables

![0009642b5022a64f1addc8cc35c14b37920d6e19fc1fa601d7115c270ec33ab3.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/tables/0009642b5022a64f1addc8cc35c14b37920d6e19fc1fa601d7115c270ec33ab3.jpg)

![0363df7f61d8bb0d9653e4fe0fb369d8bc7d14ceb79fde9de66699ec0280d90e.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/tables/0363df7f61d8bb0d9653e4fe0fb369d8bc7d14ceb79fde9de66699ec0280d90e.jpg)

![07d0b4b92b568adf605f603a8777f2fad4f401767781e8a1509e8a6d963d7b90.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/tables/07d0b4b92b568adf605f603a8777f2fad4f401767781e8a1509e8a6d963d7b90.jpg)

![48e09cad574d86ad2fcba0002094d600c41752217a4a2db0c7904111735ae8b7.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/tables/48e09cad574d86ad2fcba0002094d600c41752217a4a2db0c7904111735ae8b7.jpg)

![4f83b21ae109b81d8c2d3481547207f69c2fe63142a3f0d07434068c914f6c04.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/tables/4f83b21ae109b81d8c2d3481547207f69c2fe63142a3f0d07434068c914f6c04.jpg)

![4fafffcd0619eb8c3999adf2cff5f550eea6c1379f7ba43c81b2a25e3185b35a.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/tables/4fafffcd0619eb8c3999adf2cff5f550eea6c1379f7ba43c81b2a25e3185b35a.jpg)

![5072de84a08d47726e503a79f0f18765f9688ac89d9ba6ee4a24cd936b760d44.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/tables/5072de84a08d47726e503a79f0f18765f9688ac89d9ba6ee4a24cd936b760d44.jpg)

![6028be69e4794594ac38be991f813e9e7a200f11a4214674e210ff122d1aa35b.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/tables/6028be69e4794594ac38be991f813e9e7a200f11a4214674e210ff122d1aa35b.jpg)

![748d2b778bb8e1cab7418479daa09b8e89f5217e0eb93bc99e2b732989ee600a.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/tables/748d2b778bb8e1cab7418479daa09b8e89f5217e0eb93bc99e2b732989ee600a.jpg)

![9e5edd4536bc54b5bd760231f2bf78ff0f69d795f595564a5d1b94fc4127eb53.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/tables/9e5edd4536bc54b5bd760231f2bf78ff0f69d795f595564a5d1b94fc4127eb53.jpg)

![a677a8dc345e34e74e5cdbe4a7fccaa49ca6a77b680ae86ae24fd36757825bae.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/tables/a677a8dc345e34e74e5cdbe4a7fccaa49ca6a77b680ae86ae24fd36757825bae.jpg)

![c2aef5957701376166d190bffd0ed06f34061620dddccdca058ebdad4d4e12e0.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/tables/c2aef5957701376166d190bffd0ed06f34061620dddccdca058ebdad4d4e12e0.jpg)

![f5469a15469ea56f8148acd863b84e6e033af9abd8efe504ab2fd0af623af40e.jpg](../icml_results/2998_VERA-V_%20Variational%20Inference%20Framework%20for%20Jailbreaking%20Vision-Language%20Models/tables/f5469a15469ea56f8148acd863b84e6e033af9abd8efe504ab2fd0af623af40e.jpg)

<a id="paper-2961"></a>

## Frequency Matching in Spiking Neural Networks for mmWave Sensing


### Images

![1665e1442874fc9e87158afeca7597641222d6922c1aff098bda6f082cba7ead.jpg](../icml_results/2999_Frequency%20Matching%20in%20Spiking%20Neural%20Networks%20for%20mmWave%20Sensing/images/1665e1442874fc9e87158afeca7597641222d6922c1aff098bda6f082cba7ead.jpg)

![3d98f0d3c011a1b2b6ed8a00c02326f1d7fb6e1fa0866d414ea5a3bba2e377f8.jpg](../icml_results/2999_Frequency%20Matching%20in%20Spiking%20Neural%20Networks%20for%20mmWave%20Sensing/images/3d98f0d3c011a1b2b6ed8a00c02326f1d7fb6e1fa0866d414ea5a3bba2e377f8.jpg)

![6501d7f63cdcf74324a50f61e5265d03f95ac88f4aba4f1164060f5f936c331e.jpg](../icml_results/2999_Frequency%20Matching%20in%20Spiking%20Neural%20Networks%20for%20mmWave%20Sensing/images/6501d7f63cdcf74324a50f61e5265d03f95ac88f4aba4f1164060f5f936c331e.jpg)

![9448f1c58fa237835d0186a191cc5312d179ba681aa242b0604419ba3758e1bd.jpg](../icml_results/2999_Frequency%20Matching%20in%20Spiking%20Neural%20Networks%20for%20mmWave%20Sensing/images/9448f1c58fa237835d0186a191cc5312d179ba681aa242b0604419ba3758e1bd.jpg)

![acd546677e4c11c50836cfd0284826fd5c4bf5a44b47813c465c303ef6c18097.jpg](../icml_results/2999_Frequency%20Matching%20in%20Spiking%20Neural%20Networks%20for%20mmWave%20Sensing/images/acd546677e4c11c50836cfd0284826fd5c4bf5a44b47813c465c303ef6c18097.jpg)

![c02e71aace512ee0ceb6cdb08390674fa95217c38a2f962239390515938fb2e8.jpg](../icml_results/2999_Frequency%20Matching%20in%20Spiking%20Neural%20Networks%20for%20mmWave%20Sensing/images/c02e71aace512ee0ceb6cdb08390674fa95217c38a2f962239390515938fb2e8.jpg)

![c9fc777e67099da53606de1ecfd0ffa214a9ae0c789bbcbe451e97f0a77c374f.jpg](../icml_results/2999_Frequency%20Matching%20in%20Spiking%20Neural%20Networks%20for%20mmWave%20Sensing/images/c9fc777e67099da53606de1ecfd0ffa214a9ae0c789bbcbe451e97f0a77c374f.jpg)

![d89e2d68d6c13f2e107cb779c40f29dd724c28fa68c032869dc28263f999756a.jpg](../icml_results/2999_Frequency%20Matching%20in%20Spiking%20Neural%20Networks%20for%20mmWave%20Sensing/images/d89e2d68d6c13f2e107cb779c40f29dd724c28fa68c032869dc28263f999756a.jpg)

![df5d57b8c4d3f3f2f7fa20df024dc10df3474f4a0e22060b7d646ffd3368f2fb.jpg](../icml_results/2999_Frequency%20Matching%20in%20Spiking%20Neural%20Networks%20for%20mmWave%20Sensing/images/df5d57b8c4d3f3f2f7fa20df024dc10df3474f4a0e22060b7d646ffd3368f2fb.jpg)

![e321e234f4bad4f68a3a01a034f67b18722722f0ffa11d811f509e0f34958e27.jpg](../icml_results/2999_Frequency%20Matching%20in%20Spiking%20Neural%20Networks%20for%20mmWave%20Sensing/images/e321e234f4bad4f68a3a01a034f67b18722722f0ffa11d811f509e0f34958e27.jpg)

### Tables

![279f014239afea020d2ea14f40fb906031b1cb2d13a012f3aff584b0f3bdd4c4.jpg](../icml_results/2999_Frequency%20Matching%20in%20Spiking%20Neural%20Networks%20for%20mmWave%20Sensing/tables/279f014239afea020d2ea14f40fb906031b1cb2d13a012f3aff584b0f3bdd4c4.jpg)

![3c15d41452ade2753113a234aabdb14f82c7dd501510c99b8fdaf8fb9db0e6a0.jpg](../icml_results/2999_Frequency%20Matching%20in%20Spiking%20Neural%20Networks%20for%20mmWave%20Sensing/tables/3c15d41452ade2753113a234aabdb14f82c7dd501510c99b8fdaf8fb9db0e6a0.jpg)

![6785855bbb240d3fe79fe75f113eca29ae44fb5bd81baabfbe59d930cef5bdcc.jpg](../icml_results/2999_Frequency%20Matching%20in%20Spiking%20Neural%20Networks%20for%20mmWave%20Sensing/tables/6785855bbb240d3fe79fe75f113eca29ae44fb5bd81baabfbe59d930cef5bdcc.jpg)

![a17457968b93324abf368f011b9fc4f2b7a1a85e8ab226840db2c874d433a6ee.jpg](../icml_results/2999_Frequency%20Matching%20in%20Spiking%20Neural%20Networks%20for%20mmWave%20Sensing/tables/a17457968b93324abf368f011b9fc4f2b7a1a85e8ab226840db2c874d433a6ee.jpg)

![ded52f25026e5416e53c1c99c9958415f98957f6456cff9c01e58e50bcda10ce.jpg](../icml_results/2999_Frequency%20Matching%20in%20Spiking%20Neural%20Networks%20for%20mmWave%20Sensing/tables/ded52f25026e5416e53c1c99c9958415f98957f6456cff9c01e58e50bcda10ce.jpg)

![fa0d5df0057e2a91d7a622c102bf1b708bf8c6a0f5493dc62574eeeff48b3ac6.jpg](../icml_results/2999_Frequency%20Matching%20in%20Spiking%20Neural%20Networks%20for%20mmWave%20Sensing/tables/fa0d5df0057e2a91d7a622c102bf1b708bf8c6a0f5493dc62574eeeff48b3ac6.jpg)

<a id="paper-2962"></a>

## Risk-Averse and Optimistic Advertiser Incentive Compatibility in Auto-bidding


### Tables

![1102f0b825967a06334c966055690a387c8d4b33c644981def32c5f53818c1b7.jpg](../icml_results/3000_Risk-Averse%20and%20Optimistic%20Advertiser%20Incentive%20Compatibility%20in%20Auto-bidding/tables/1102f0b825967a06334c966055690a387c8d4b33c644981def32c5f53818c1b7.jpg)

<a id="paper-2963"></a>

## L-SR1: Learned Symmetric-Rank-One Preconditioning


### Images

![0c40126ca91615e43d49409e511bac231e4166a63828b4a367e220aedbb3aacd.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/images/0c40126ca91615e43d49409e511bac231e4166a63828b4a367e220aedbb3aacd.jpg)

![343420ec78b3fc6e7413176438b54f6f3a9d6a04d0fb9eaeca20b430c93a58f6.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/images/343420ec78b3fc6e7413176438b54f6f3a9d6a04d0fb9eaeca20b430c93a58f6.jpg)

![46dad823f0fec8dde0d9b31914dc381ec131ab4a455d59bd4d220d3166b4cde6.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/images/46dad823f0fec8dde0d9b31914dc381ec131ab4a455d59bd4d220d3166b4cde6.jpg)

![8cb271bdc29b41400ec3b9303f7f51fc4e0534095fc284b1cd314c56229df4e1.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/images/8cb271bdc29b41400ec3b9303f7f51fc4e0534095fc284b1cd314c56229df4e1.jpg)

![944a6e47439776f57291613fa8d60bfac0036d335cfa1e72790bb418a4d00146.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/images/944a6e47439776f57291613fa8d60bfac0036d335cfa1e72790bb418a4d00146.jpg)

![98a807af9b77329def957a130e6940543395807a9af066bba635737bf1672b45.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/images/98a807af9b77329def957a130e6940543395807a9af066bba635737bf1672b45.jpg)

![aaeef86b24c291642e923903eba8c9e1a8c1857a90c46ef53110d7eefac95c67.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/images/aaeef86b24c291642e923903eba8c9e1a8c1857a90c46ef53110d7eefac95c67.jpg)

![ac28a3119522fa1e80efed4dcfb86de313dcc30b48d83406ff5f5c8775f049d2.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/images/ac28a3119522fa1e80efed4dcfb86de313dcc30b48d83406ff5f5c8775f049d2.jpg)

![b5e884e99a2491e91f86a4606ab966482b2c572c1ca74b57d037cba4eb7e526a.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/images/b5e884e99a2491e91f86a4606ab966482b2c572c1ca74b57d037cba4eb7e526a.jpg)

![b6133b69b8882b9efa0e2dca9db8cdcce23fa35ea067ed2c481f0c639b3b8b6e.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/images/b6133b69b8882b9efa0e2dca9db8cdcce23fa35ea067ed2c481f0c639b3b8b6e.jpg)

![cf531e77081d4a11bcd5b71a113f0bd8218f11be6019efc7c2f3e44a1a88d781.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/images/cf531e77081d4a11bcd5b71a113f0bd8218f11be6019efc7c2f3e44a1a88d781.jpg)

![d0dc8d1c160bd648b53c4e0945d99429898d43ffb1d5e862569526124d677bfa.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/images/d0dc8d1c160bd648b53c4e0945d99429898d43ffb1d5e862569526124d677bfa.jpg)

![d7929cb61a2bb04c98a8ed88005bb86f704527f6a7576e96fe4562a431b870ab.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/images/d7929cb61a2bb04c98a8ed88005bb86f704527f6a7576e96fe4562a431b870ab.jpg)

![f4b831a90a2def08ad1aabc3773c577c4e1162e6ac28a9501af9523e964f4c42.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/images/f4b831a90a2def08ad1aabc3773c577c4e1162e6ac28a9501af9523e964f4c42.jpg)

![fc4a25fb00e4d4a7d9706275695476dbf3e80cd3d55b0d0585b850528d0aba6f.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/images/fc4a25fb00e4d4a7d9706275695476dbf3e80cd3d55b0d0585b850528d0aba6f.jpg)

### Tables

![122cccf2c664cddd595afbef90bc5b61da3f35b365d8593fc61392496490ce1d.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/tables/122cccf2c664cddd595afbef90bc5b61da3f35b365d8593fc61392496490ce1d.jpg)

![172d7e304ea123875b2b79453658b57a946221cf7483d34b45d85cecb2da8d58.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/tables/172d7e304ea123875b2b79453658b57a946221cf7483d34b45d85cecb2da8d58.jpg)

![68413f9443bd4706938dc2e70917980ee879bb828eedf469f4aeaae79bce943c.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/tables/68413f9443bd4706938dc2e70917980ee879bb828eedf469f4aeaae79bce943c.jpg)

![6d887cb2fe13cf072f66ef275f76776f3f91698b72aa024b20922c345918e07e.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/tables/6d887cb2fe13cf072f66ef275f76776f3f91698b72aa024b20922c345918e07e.jpg)

![9932917e8af714e3f8dccfdf257a011fe703f77dbdddc436a907e9ef30b91b8d.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/tables/9932917e8af714e3f8dccfdf257a011fe703f77dbdddc436a907e9ef30b91b8d.jpg)

![a082265e0675e683bb555d63255bda2e917774603f5b4c1553a67bfb85a039d8.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/tables/a082265e0675e683bb555d63255bda2e917774603f5b4c1553a67bfb85a039d8.jpg)

![a62796056aeb907690586745baee961c681d14b9ffcd2ca513353c1595b04b00.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/tables/a62796056aeb907690586745baee961c681d14b9ffcd2ca513353c1595b04b00.jpg)

![de37a04858f078b9b2bf43e4e42f72da45c2ff084f73fec4371f3ced0eab83c4.jpg](../icml_results/3001_L-SR1_%20Learned%20Symmetric-Rank-One%20Preconditioning/tables/de37a04858f078b9b2bf43e4e42f72da45c2ff084f73fec4371f3ced0eab83c4.jpg)

<a id="paper-2964"></a>

## KIStego: Key-Independent Secure Image Distribution via Bipartite Structural Invariants


### Images

![10cda5d7d061d0e313108803a37fc6ec2aca02db7ac9021b0e7846b9814840d5.jpg](../icml_results/3002_KIStego_%20Key-Independent%20Secure%20Image%20Distribution%20via%20Bipartite%20Structural%20Invariants/images/10cda5d7d061d0e313108803a37fc6ec2aca02db7ac9021b0e7846b9814840d5.jpg)

![3ca51161bb9024f6c1ebf3c00d0db3cbeed2349e9e098ae72e8cc160e2d9858e.jpg](../icml_results/3002_KIStego_%20Key-Independent%20Secure%20Image%20Distribution%20via%20Bipartite%20Structural%20Invariants/images/3ca51161bb9024f6c1ebf3c00d0db3cbeed2349e9e098ae72e8cc160e2d9858e.jpg)

![430612917e302b8d4ea37998d82932fa085a1e290df6cb7a752e45be69bb6940.jpg](../icml_results/3002_KIStego_%20Key-Independent%20Secure%20Image%20Distribution%20via%20Bipartite%20Structural%20Invariants/images/430612917e302b8d4ea37998d82932fa085a1e290df6cb7a752e45be69bb6940.jpg)

![4be5f1ec6f1442c3f62f11c4bebff5b042b1a66c804660cc9eb40f36bb6c330c.jpg](../icml_results/3002_KIStego_%20Key-Independent%20Secure%20Image%20Distribution%20via%20Bipartite%20Structural%20Invariants/images/4be5f1ec6f1442c3f62f11c4bebff5b042b1a66c804660cc9eb40f36bb6c330c.jpg)

![77b647a5ee1e4b65ddc667f826b7a4c7319af67013fc2d577deb3d06434d03b0.jpg](../icml_results/3002_KIStego_%20Key-Independent%20Secure%20Image%20Distribution%20via%20Bipartite%20Structural%20Invariants/images/77b647a5ee1e4b65ddc667f826b7a4c7319af67013fc2d577deb3d06434d03b0.jpg)

![9ecc4107e344347e03e85012df96d606ac73fc331b7a2e6f286ec88ec069d36d.jpg](../icml_results/3002_KIStego_%20Key-Independent%20Secure%20Image%20Distribution%20via%20Bipartite%20Structural%20Invariants/images/9ecc4107e344347e03e85012df96d606ac73fc331b7a2e6f286ec88ec069d36d.jpg)

![bb31448f3c98def5a83cae9a86169f90cc501c09d592082007fd67edc786f24c.jpg](../icml_results/3002_KIStego_%20Key-Independent%20Secure%20Image%20Distribution%20via%20Bipartite%20Structural%20Invariants/images/bb31448f3c98def5a83cae9a86169f90cc501c09d592082007fd67edc786f24c.jpg)

![d9c527f9034386feea1a15343b60197fc9b47fa2427044c161fd395c0d1b65d8.jpg](../icml_results/3002_KIStego_%20Key-Independent%20Secure%20Image%20Distribution%20via%20Bipartite%20Structural%20Invariants/images/d9c527f9034386feea1a15343b60197fc9b47fa2427044c161fd395c0d1b65d8.jpg)

### Tables

![00e705598f81066052ec6b4dcb3454a6ab0f36606a6a02e084f004e880b2bb91.jpg](../icml_results/3002_KIStego_%20Key-Independent%20Secure%20Image%20Distribution%20via%20Bipartite%20Structural%20Invariants/tables/00e705598f81066052ec6b4dcb3454a6ab0f36606a6a02e084f004e880b2bb91.jpg)

![557b366bf6c3ca73a72832368a4a2560bca5797d728ef165a2363ffffcaa5e5e.jpg](../icml_results/3002_KIStego_%20Key-Independent%20Secure%20Image%20Distribution%20via%20Bipartite%20Structural%20Invariants/tables/557b366bf6c3ca73a72832368a4a2560bca5797d728ef165a2363ffffcaa5e5e.jpg)

![ea0cae6e3f1eebe86b84e04e1ca5956cbdfe90182bcb9df65b551109e00e979f.jpg](../icml_results/3002_KIStego_%20Key-Independent%20Secure%20Image%20Distribution%20via%20Bipartite%20Structural%20Invariants/tables/ea0cae6e3f1eebe86b84e04e1ca5956cbdfe90182bcb9df65b551109e00e979f.jpg)

![ed150205dc734baa0c50026814aa8c93dd334cf74a66c7f356d7d7de1c5781e6.jpg](../icml_results/3002_KIStego_%20Key-Independent%20Secure%20Image%20Distribution%20via%20Bipartite%20Structural%20Invariants/tables/ed150205dc734baa0c50026814aa8c93dd334cf74a66c7f356d7d7de1c5781e6.jpg)

![f83989db8666671f30ef578b35f5069ed48f9127038d93414f6426733fe2bfb6.jpg](../icml_results/3002_KIStego_%20Key-Independent%20Secure%20Image%20Distribution%20via%20Bipartite%20Structural%20Invariants/tables/f83989db8666671f30ef578b35f5069ed48f9127038d93414f6426733fe2bfb6.jpg)

<a id="paper-2965"></a>

## Flow Equivariant World Models: Structured Memory for Dynamic Environments


### Images

![04359fac47a82148f42b5b19f15338805efda5e69826ca612820c7160daa30cb.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/images/04359fac47a82148f42b5b19f15338805efda5e69826ca612820c7160daa30cb.jpg)

![0e3c5a1673502989dcdc0b42a9ed89a75f8e99e26625566eeba4b2356190b8bd.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/images/0e3c5a1673502989dcdc0b42a9ed89a75f8e99e26625566eeba4b2356190b8bd.jpg)

![1a285e301b82413772a6cf50a77a90ff4218d3cbd72f22bde894503759fed7af.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/images/1a285e301b82413772a6cf50a77a90ff4218d3cbd72f22bde894503759fed7af.jpg)

![40694889cead9a159c4c468c736c895ce8bd3e866dbc27d50beb903ea28623b7.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/images/40694889cead9a159c4c468c736c895ce8bd3e866dbc27d50beb903ea28623b7.jpg)

![61789b826c15c1249b7830328dc293004745393a1aaf5b359687d80e8a6fa4fa.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/images/61789b826c15c1249b7830328dc293004745393a1aaf5b359687d80e8a6fa4fa.jpg)

![642ae8cce9141e47de12fb73fec1a6bd79fd5ae2e3cfbc85cb689668cfa26c4e.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/images/642ae8cce9141e47de12fb73fec1a6bd79fd5ae2e3cfbc85cb689668cfa26c4e.jpg)

![6e9fdc0c6ee72ab049732467df4902b1de03570ae1b82c349f30905a3262ee33.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/images/6e9fdc0c6ee72ab049732467df4902b1de03570ae1b82c349f30905a3262ee33.jpg)

![80ad71d525949fcc704ac0fa4a84946d948121418a46c3b6a61ee939f0ebb024.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/images/80ad71d525949fcc704ac0fa4a84946d948121418a46c3b6a61ee939f0ebb024.jpg)

![a9066c34ef139a3f73d444338773d4795ff240d29a9ae73ddba55d088e14a54b.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/images/a9066c34ef139a3f73d444338773d4795ff240d29a9ae73ddba55d088e14a54b.jpg)

![b85834f96383c97ed6e4b5bda500505a1e849c60f28f750a3913f68b5b03ccb0.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/images/b85834f96383c97ed6e4b5bda500505a1e849c60f28f750a3913f68b5b03ccb0.jpg)

![d0ff1c5658dc4bbd00e1308727bcf40d56e2543fde51e6f98737c3356f0c344f.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/images/d0ff1c5658dc4bbd00e1308727bcf40d56e2543fde51e6f98737c3356f0c344f.jpg)

![da06dcca84941d5d70e68943dc4c9a2daeff37a4d8257bb057c2f0f05eaeaba4.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/images/da06dcca84941d5d70e68943dc4c9a2daeff37a4d8257bb057c2f0f05eaeaba4.jpg)

![e11a7c421dd6455f9c614483fd17fe9ddf6789a8d5d1f73f25d99b63efa810ac.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/images/e11a7c421dd6455f9c614483fd17fe9ddf6789a8d5d1f73f25d99b63efa810ac.jpg)

![e41ee6dfe16c5a4e1aec23809e9a202e8fce0fc85d811bafb766ae22d328ad82.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/images/e41ee6dfe16c5a4e1aec23809e9a202e8fce0fc85d811bafb766ae22d328ad82.jpg)

![e4572d0be96fa919f2769dc031eba706d2a3ff911eeaea1473298089fad24184.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/images/e4572d0be96fa919f2769dc031eba706d2a3ff911eeaea1473298089fad24184.jpg)

### Tables

![18b7674046d3405344dddc0f4f35f374846339d6509fde188c57358a1fafe561.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/18b7674046d3405344dddc0f4f35f374846339d6509fde188c57358a1fafe561.jpg)

![19294fff17cc3663fbc12d963ef2e7cea25e98626583927c45a2a367c7663adb.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/19294fff17cc3663fbc12d963ef2e7cea25e98626583927c45a2a367c7663adb.jpg)

![2710ce77225d1f0adad2fb3ac9f6db606ed1e30476970cdf1fa2a22cb4bc56ea.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/2710ce77225d1f0adad2fb3ac9f6db606ed1e30476970cdf1fa2a22cb4bc56ea.jpg)

![3de265cfe0fef0934d39f1a6216d7697659a8e88a681297b382cf694b3fc653f.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/3de265cfe0fef0934d39f1a6216d7697659a8e88a681297b382cf694b3fc653f.jpg)

![4390ed4314aa678bef463fb7bee8a98ffb27c6a1898711cc3f5e6eb7061dc158.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/4390ed4314aa678bef463fb7bee8a98ffb27c6a1898711cc3f5e6eb7061dc158.jpg)

![499ee7a7522235b808e3bd8c342b5c1663fd706d8dd9969d7f279506b431f6dc.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/499ee7a7522235b808e3bd8c342b5c1663fd706d8dd9969d7f279506b431f6dc.jpg)

![50ce09913b17089fab31f176c951755baf5d9101dabfedb5504b00d5773fd694.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/50ce09913b17089fab31f176c951755baf5d9101dabfedb5504b00d5773fd694.jpg)

![56eee8aa8421ebbbacefb6fc6468ac3f674cfc7278430c8de55614ab7d3c84ad.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/56eee8aa8421ebbbacefb6fc6468ac3f674cfc7278430c8de55614ab7d3c84ad.jpg)

![57bcf9e9bc0db16192103edc3b3b8df34fb2e340710b9c49961426ec5f5193a2.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/57bcf9e9bc0db16192103edc3b3b8df34fb2e340710b9c49961426ec5f5193a2.jpg)

![5887ea834f79f28dfa2d9882df61efd62a3dec0153edee1ff3f9e38e6de10c96.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/5887ea834f79f28dfa2d9882df61efd62a3dec0153edee1ff3f9e38e6de10c96.jpg)

![6727e2fcca5f9d52dc509f0495f3701398430d8ecb4a7e9182b9ecd4a5bfaafd.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/6727e2fcca5f9d52dc509f0495f3701398430d8ecb4a7e9182b9ecd4a5bfaafd.jpg)

![6be93d9897e0651752bb7313a4ff67640f0713dcb94914b31a9fe2a5419bc619.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/6be93d9897e0651752bb7313a4ff67640f0713dcb94914b31a9fe2a5419bc619.jpg)

![979a190a184e78861fd251d4f4a306e72f9c50ef602269caff8de65642a010a5.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/979a190a184e78861fd251d4f4a306e72f9c50ef602269caff8de65642a010a5.jpg)

![9ddc78b9732b2089e4d7e8ba1460344ebefd498adc3d9c8b76f40cb92cbdbec2.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/9ddc78b9732b2089e4d7e8ba1460344ebefd498adc3d9c8b76f40cb92cbdbec2.jpg)

![a9e94173fff47154799e5cde55c201ab6ce6f11de47099b1498d16db6faf3eff.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/a9e94173fff47154799e5cde55c201ab6ce6f11de47099b1498d16db6faf3eff.jpg)

![ab233f27dcee658d70777e323cc77d9eff5c1dbc0634c2aa5834e9e0eb0080f8.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/ab233f27dcee658d70777e323cc77d9eff5c1dbc0634c2aa5834e9e0eb0080f8.jpg)

![abe1223a20172c830a5e960ed09ed43867ccde3f9860fa30d13bf6d495175e22.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/abe1223a20172c830a5e960ed09ed43867ccde3f9860fa30d13bf6d495175e22.jpg)

![ba01dcbbd188980a6afd3ae55565522c542e64c7dd50e252b429536f8f633b37.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/ba01dcbbd188980a6afd3ae55565522c542e64c7dd50e252b429536f8f633b37.jpg)

![bc5672e34fcfc7402f98069f6887a1650defd2243503b2fc6f438bc02e877e44.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/bc5672e34fcfc7402f98069f6887a1650defd2243503b2fc6f438bc02e877e44.jpg)

![be2c913e3947f449a5cb5ec62df327d5bd88a7fe1754de152d522d313653a7a7.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/be2c913e3947f449a5cb5ec62df327d5bd88a7fe1754de152d522d313653a7a7.jpg)

![f26f130c63bbdeabb2524c366c7cee0e0ae0b8c4683f0cab5fcbad50da03dd9c.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/f26f130c63bbdeabb2524c366c7cee0e0ae0b8c4683f0cab5fcbad50da03dd9c.jpg)

![f57191b116b9f87c988db280c5bb2591db1bbcf6ce2a81ad79c615a3d5edbe85.jpg](../icml_results/3003_Flow%20Equivariant%20World%20Models_%20Structured%20Memory%20for%20Dynamic%20Environments/tables/f57191b116b9f87c988db280c5bb2591db1bbcf6ce2a81ad79c615a3d5edbe85.jpg)

<a id="paper-2966"></a>

## Rethinking Attention in Spiking Transformers: Overcoming Density Bias with Set Similarity


### Images

![0170fb7aabe2d051a94efea5a885badbb154538a7673bdb510ee8fb3b2a7e066.jpg](../icml_results/3004_Rethinking%20Attention%20in%20Spiking%20Transformers_%20Overcoming%20Density%20Bias%20with%20Set%20Similarity/images/0170fb7aabe2d051a94efea5a885badbb154538a7673bdb510ee8fb3b2a7e066.jpg)

![bb3e7de650a6bc61932ac31803335ac83547a7c6a6f1a4081f3dd32fc435d85e.jpg](../icml_results/3004_Rethinking%20Attention%20in%20Spiking%20Transformers_%20Overcoming%20Density%20Bias%20with%20Set%20Similarity/images/bb3e7de650a6bc61932ac31803335ac83547a7c6a6f1a4081f3dd32fc435d85e.jpg)

### Tables

![059681128c7ced579e4235b6f828afa1ef47e43ee3e6f131c05fcdf301f5ae5d.jpg](../icml_results/3004_Rethinking%20Attention%20in%20Spiking%20Transformers_%20Overcoming%20Density%20Bias%20with%20Set%20Similarity/tables/059681128c7ced579e4235b6f828afa1ef47e43ee3e6f131c05fcdf301f5ae5d.jpg)

![2694d74359432fef693304a516bf8f5e812aeb7627d1179190189630ce459d99.jpg](../icml_results/3004_Rethinking%20Attention%20in%20Spiking%20Transformers_%20Overcoming%20Density%20Bias%20with%20Set%20Similarity/tables/2694d74359432fef693304a516bf8f5e812aeb7627d1179190189630ce459d99.jpg)

![2da0bb983f6154822cb12c0215a500e1e3491aacd9a9d6944896533b498a4d40.jpg](../icml_results/3004_Rethinking%20Attention%20in%20Spiking%20Transformers_%20Overcoming%20Density%20Bias%20with%20Set%20Similarity/tables/2da0bb983f6154822cb12c0215a500e1e3491aacd9a9d6944896533b498a4d40.jpg)

![2e14c59b367ba88736d9cf9fe2ed5c2d81138f9685b7297ede8e26bab9732742.jpg](../icml_results/3004_Rethinking%20Attention%20in%20Spiking%20Transformers_%20Overcoming%20Density%20Bias%20with%20Set%20Similarity/tables/2e14c59b367ba88736d9cf9fe2ed5c2d81138f9685b7297ede8e26bab9732742.jpg)

![384778cfee01c8d94e578cb94e07c83e05a02fa6d05e72efaa9e6eee92963010.jpg](../icml_results/3004_Rethinking%20Attention%20in%20Spiking%20Transformers_%20Overcoming%20Density%20Bias%20with%20Set%20Similarity/tables/384778cfee01c8d94e578cb94e07c83e05a02fa6d05e72efaa9e6eee92963010.jpg)

![47a7173579d47247b073fb330f6ae31c5e77ede22fe0e7531a8a78c35c837ccf.jpg](../icml_results/3004_Rethinking%20Attention%20in%20Spiking%20Transformers_%20Overcoming%20Density%20Bias%20with%20Set%20Similarity/tables/47a7173579d47247b073fb330f6ae31c5e77ede22fe0e7531a8a78c35c837ccf.jpg)

![4ebff2009ed40046ecf50c44520f9ce74fa61fb3251444466d77cec36ff036a5.jpg](../icml_results/3004_Rethinking%20Attention%20in%20Spiking%20Transformers_%20Overcoming%20Density%20Bias%20with%20Set%20Similarity/tables/4ebff2009ed40046ecf50c44520f9ce74fa61fb3251444466d77cec36ff036a5.jpg)

![b8a33013fb302b6ed7546c113138c2360dff004d1af6b791f2816a92634ecfe0.jpg](../icml_results/3004_Rethinking%20Attention%20in%20Spiking%20Transformers_%20Overcoming%20Density%20Bias%20with%20Set%20Similarity/tables/b8a33013fb302b6ed7546c113138c2360dff004d1af6b791f2816a92634ecfe0.jpg)

![f7b1f34e5000036cea2cb13d07cc86e1d7dfba9f800944063bc29cd3cbae285c.jpg](../icml_results/3004_Rethinking%20Attention%20in%20Spiking%20Transformers_%20Overcoming%20Density%20Bias%20with%20Set%20Similarity/tables/f7b1f34e5000036cea2cb13d07cc86e1d7dfba9f800944063bc29cd3cbae285c.jpg)

<a id="paper-2967"></a>

## QuITE: Query-Based Irregular Time Series Embedding


### Images

![1adf1d07cfba8ce5feb211cb500a5d1ae4a8aa375226a7285b1cdd53fa694291.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/1adf1d07cfba8ce5feb211cb500a5d1ae4a8aa375226a7285b1cdd53fa694291.jpg)

![1f4850dc9bc7e10f653566c3489512224ae06f1aebb0cde184597d77b780cdcf.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/1f4850dc9bc7e10f653566c3489512224ae06f1aebb0cde184597d77b780cdcf.jpg)

![31a05168e917d8ce1afea0ef8bdb815d1d005bc9c91416d39a1e988d29522468.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/31a05168e917d8ce1afea0ef8bdb815d1d005bc9c91416d39a1e988d29522468.jpg)

![3faa481be554b828d2e69241155c0ceae63efa6c842fea58455e218be27950bd.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/3faa481be554b828d2e69241155c0ceae63efa6c842fea58455e218be27950bd.jpg)

![4554bf73ace9ea9a2d7432a309d5d379dec19fb4c6b5ab67c4d7e0610c80acbc.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/4554bf73ace9ea9a2d7432a309d5d379dec19fb4c6b5ab67c4d7e0610c80acbc.jpg)

![509c09022021ace505544bb8c2782b573247e91f9d739f88895c2de1e73ef10c.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/509c09022021ace505544bb8c2782b573247e91f9d739f88895c2de1e73ef10c.jpg)

![6db820c5108bb7d430291a99a2b912602745624aa3b8b0c1ecb4206d55ea3aef.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/6db820c5108bb7d430291a99a2b912602745624aa3b8b0c1ecb4206d55ea3aef.jpg)

![6eb95e2a276370cc4503ebf8189f225f5aad45b58c9ef4ed89f955c964f79901.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/6eb95e2a276370cc4503ebf8189f225f5aad45b58c9ef4ed89f955c964f79901.jpg)

![6f79e88aa0e872db9b5cd98df124d3d60e0dc997f43e6cbdc1744ca48a52f46d.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/6f79e88aa0e872db9b5cd98df124d3d60e0dc997f43e6cbdc1744ca48a52f46d.jpg)

![7042b70dc6d5e9446eff8d700fe73a575b2b193785b3eac1421d3ad298adbc81.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/7042b70dc6d5e9446eff8d700fe73a575b2b193785b3eac1421d3ad298adbc81.jpg)

![766e407e0b425b4cae391a0d8af944c091778667d738134ca11475e61f4c952d.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/766e407e0b425b4cae391a0d8af944c091778667d738134ca11475e61f4c952d.jpg)

![91ee9a23f0f0beb59c766cd803f380066608403e301e887a6e996c929f1effa1.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/91ee9a23f0f0beb59c766cd803f380066608403e301e887a6e996c929f1effa1.jpg)

![9269d41ec5e6fa57ff1543c8afee72840d584f32e10060bde8efeec462379a43.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/9269d41ec5e6fa57ff1543c8afee72840d584f32e10060bde8efeec462379a43.jpg)

![9d285f785b2bb57fa54d8df307c7d7b373b7edd4e2319c32175466d05bdd5ce5.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/9d285f785b2bb57fa54d8df307c7d7b373b7edd4e2319c32175466d05bdd5ce5.jpg)

![d460bc5cd7ee422a5c3009788421989461736efced6de14212c3bd9935899861.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/d460bc5cd7ee422a5c3009788421989461736efced6de14212c3bd9935899861.jpg)

![df97c765d2935e8f3532035003c7d254a59ca8f244f96c7cd510f7994fcf3fa5.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/df97c765d2935e8f3532035003c7d254a59ca8f244f96c7cd510f7994fcf3fa5.jpg)

![e2cd0242f64e48eccf4b2619fd1ad7e55f49e295b079be7f2591f50170e89d3e.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/e2cd0242f64e48eccf4b2619fd1ad7e55f49e295b079be7f2591f50170e89d3e.jpg)

![e5c968093c443d64a1de5fdb0b8738ed63fa791623d688ced52222f88317face.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/e5c968093c443d64a1de5fdb0b8738ed63fa791623d688ced52222f88317face.jpg)

![ed2190f6be40fb7238d8f51997f6fe05975993833703d716853bd07f1af87e23.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/images/ed2190f6be40fb7238d8f51997f6fe05975993833703d716853bd07f1af87e23.jpg)

### Tables

![1ca5b1793725d51f55d49fca05a40ee827a0bbcd7996f66d616b60fdfb19427c.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/1ca5b1793725d51f55d49fca05a40ee827a0bbcd7996f66d616b60fdfb19427c.jpg)

![24c229cac7e1ffa47e9959ab015393aa78379a38db28392a9a6938a7c9a80d6f.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/24c229cac7e1ffa47e9959ab015393aa78379a38db28392a9a6938a7c9a80d6f.jpg)

![367c8031c09281bbac66ff56a0b4182cfd364e305ec728fb81cabce53a209557.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/367c8031c09281bbac66ff56a0b4182cfd364e305ec728fb81cabce53a209557.jpg)

![3c76f7c7484030ffbddcce9037fb90fe28cd78e50216fccc58e4be35a9aebe51.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/3c76f7c7484030ffbddcce9037fb90fe28cd78e50216fccc58e4be35a9aebe51.jpg)

![58bede2963caedffbd95a457e49c117f9f5c2b5e8dd1623cfa039b2609dfa393.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/58bede2963caedffbd95a457e49c117f9f5c2b5e8dd1623cfa039b2609dfa393.jpg)

![62bf985314ef0fa4b2a50cc09a069b1e92a03bd4ebacacceac386d8d7736030b.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/62bf985314ef0fa4b2a50cc09a069b1e92a03bd4ebacacceac386d8d7736030b.jpg)

![63e1aa482aa577a587c35cce905934814eec68f3eea1cd550c585eca0b421ab5.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/63e1aa482aa577a587c35cce905934814eec68f3eea1cd550c585eca0b421ab5.jpg)

![715904cde28d7d82312854f16847fe1fe90256f93a1b3cb6a85246461f7c68af.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/715904cde28d7d82312854f16847fe1fe90256f93a1b3cb6a85246461f7c68af.jpg)

![716ba0fc5a0ebf73d40f1b9f78f8c2dcb269a96acd421c808b277decd02ea01b.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/716ba0fc5a0ebf73d40f1b9f78f8c2dcb269a96acd421c808b277decd02ea01b.jpg)

![7863db834a33811ca287e8ffbe3fc92702c4d8c40be0c94a18d15eae1b55ab7d.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/7863db834a33811ca287e8ffbe3fc92702c4d8c40be0c94a18d15eae1b55ab7d.jpg)

![7ad5523add5d8fc7afe1cd401099f35ee07fef9c58a7f4480b07cd252e05f9ae.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/7ad5523add5d8fc7afe1cd401099f35ee07fef9c58a7f4480b07cd252e05f9ae.jpg)

![9b5d13c3cbaa550b2ebd338b200a5a0081d7af17d904ead7b2a49696ec17c518.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/9b5d13c3cbaa550b2ebd338b200a5a0081d7af17d904ead7b2a49696ec17c518.jpg)

![a30588a85ab9c9347749d70bd980068571df2e6554face56e163bfbd58295398.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/a30588a85ab9c9347749d70bd980068571df2e6554face56e163bfbd58295398.jpg)

![a594221b6a2b688d37a61b113eaedf819c8866a9db3243723b02d6a77a825cf4.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/a594221b6a2b688d37a61b113eaedf819c8866a9db3243723b02d6a77a825cf4.jpg)

![bee4d5cf1ecde0fe19b83225586ec6219a072f00664007a2196321e9cffcef59.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/bee4d5cf1ecde0fe19b83225586ec6219a072f00664007a2196321e9cffcef59.jpg)

![d71e5c69686c88bdcde5369062d2175843e0c0e5d70bbd94c66f45bf61be33d6.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/d71e5c69686c88bdcde5369062d2175843e0c0e5d70bbd94c66f45bf61be33d6.jpg)

![e1946aa016450f9ffcabc269f22f11e4e3ad6a52bd4a683d8ad6f92f9dd66ce7.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/e1946aa016450f9ffcabc269f22f11e4e3ad6a52bd4a683d8ad6f92f9dd66ce7.jpg)

![e38330aa1a8f7761976e51262e634bb941942a9ff9c5875ced89c2e9117b891d.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/e38330aa1a8f7761976e51262e634bb941942a9ff9c5875ced89c2e9117b891d.jpg)

![ebd821bc0656d20a461534997df390bee8bbb553f7446c6e27d6f5f640dffdc7.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/ebd821bc0656d20a461534997df390bee8bbb553f7446c6e27d6f5f640dffdc7.jpg)

![edbeee4e03618d27eea75c590f69bfdd73dbac406e273c581c75695ca5de6fef.jpg](../icml_results/3005_QuITE_%20Query-Based%20Irregular%20Time%20Series%20Embedding/tables/edbeee4e03618d27eea75c590f69bfdd73dbac406e273c581c75695ca5de6fef.jpg)

<a id="paper-2968"></a>

## GADA: Geometry-Aware Deformable Aggregation for Image-Based Gaussian Splatting


### Images

![3dfc4e56bb2556edfc8c3d93c90c4d4d859f212a602dfdd67c05ba43247cefda.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/images/3dfc4e56bb2556edfc8c3d93c90c4d4d859f212a602dfdd67c05ba43247cefda.jpg)

![45c8332a830d3ab9fdfe47c7a1a9939b68b49088cb2186365ab61b1f9a25b007.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/images/45c8332a830d3ab9fdfe47c7a1a9939b68b49088cb2186365ab61b1f9a25b007.jpg)

![50687ec5ff41386268ff4e559f5d436d855ad9f7b3dd772084ee2218b23287ce.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/images/50687ec5ff41386268ff4e559f5d436d855ad9f7b3dd772084ee2218b23287ce.jpg)

![59ad1ff1f66990a48a8a1f3a625b0a5264a1694ddf8f919ea3c63c506f9c9607.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/images/59ad1ff1f66990a48a8a1f3a625b0a5264a1694ddf8f919ea3c63c506f9c9607.jpg)

![5d620e61d6bb327d5d414cc20fb839bf733e459f18a9374f387ad72a177d5fd1.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/images/5d620e61d6bb327d5d414cc20fb839bf733e459f18a9374f387ad72a177d5fd1.jpg)

![5e6c9e3fdae71c80ef316a72720d711ce7cc42d4c915f126fcf0786db628c17a.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/images/5e6c9e3fdae71c80ef316a72720d711ce7cc42d4c915f126fcf0786db628c17a.jpg)

![6e0701add3c41e9ce71b1a295d20864f100cb26b012f394d084f1e83f2061b2a.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/images/6e0701add3c41e9ce71b1a295d20864f100cb26b012f394d084f1e83f2061b2a.jpg)

![74e87184d4fafc9009b06151655fb6192d9f56aa8f1cb25412b08e3b82744eb8.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/images/74e87184d4fafc9009b06151655fb6192d9f56aa8f1cb25412b08e3b82744eb8.jpg)

![7a2d9769c824863eeb740a2c4e156fa219cca73359506f7e5fd52184d47cc390.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/images/7a2d9769c824863eeb740a2c4e156fa219cca73359506f7e5fd52184d47cc390.jpg)

![90e3f66f974056351796bfe5b3282c081789ecab342658c6c7ef22ecf79d5b22.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/images/90e3f66f974056351796bfe5b3282c081789ecab342658c6c7ef22ecf79d5b22.jpg)

![d17084e3675b258dc85016787b296533e1958841c2dd595faac78deffd2d6980.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/images/d17084e3675b258dc85016787b296533e1958841c2dd595faac78deffd2d6980.jpg)

### Tables

![1ed0a2585d2e96c76706462e282daf0bf69c0eab63b65ebc8a14c1168c7c3ad4.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/1ed0a2585d2e96c76706462e282daf0bf69c0eab63b65ebc8a14c1168c7c3ad4.jpg)

![1f14a6fb7c528b9b6333aa8079475af373b783de12d06769ccfa908f715dd31f.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/1f14a6fb7c528b9b6333aa8079475af373b783de12d06769ccfa908f715dd31f.jpg)

![22da2066cfce5b2e2209deb9cde356b38df53c5f1660f3b7e1bd1891da5b8f7f.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/22da2066cfce5b2e2209deb9cde356b38df53c5f1660f3b7e1bd1891da5b8f7f.jpg)

![231846790c0585361a7910d43b9f1e6b5998718324abead151d68461a14b23ca.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/231846790c0585361a7910d43b9f1e6b5998718324abead151d68461a14b23ca.jpg)

![317713752f0f0592d56bda0a2c7501a5dd53144a22dab2bf1dcec72110933622.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/317713752f0f0592d56bda0a2c7501a5dd53144a22dab2bf1dcec72110933622.jpg)

![5eceb9e79ae65c4868cb22868671d8909c8da5140bf58e53fd62876f94999909.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/5eceb9e79ae65c4868cb22868671d8909c8da5140bf58e53fd62876f94999909.jpg)

![67a4cc583d96776067230ebc7d46dbef1bdc63f5e2acf13aef4894b34009ec73.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/67a4cc583d96776067230ebc7d46dbef1bdc63f5e2acf13aef4894b34009ec73.jpg)

![6895680247fef6624d28b08ccf507140c446e42e0a8072049845a8304fe97786.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/6895680247fef6624d28b08ccf507140c446e42e0a8072049845a8304fe97786.jpg)

![6c7437c52c6d1ea6073986cc21f28d7e4f5a9c00d5aa108b16060152ef495024.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/6c7437c52c6d1ea6073986cc21f28d7e4f5a9c00d5aa108b16060152ef495024.jpg)

![6f8bd7904de790f73cac10375c3568a4170e27705055f756fcd4ecf24489fe73.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/6f8bd7904de790f73cac10375c3568a4170e27705055f756fcd4ecf24489fe73.jpg)

![a114b34ef2522d84242452fe46802d8ef298466d5eaf33a6f5f92357ac28d2f4.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/a114b34ef2522d84242452fe46802d8ef298466d5eaf33a6f5f92357ac28d2f4.jpg)

![ae9d397df6f624061c701c283d153a96c44525d61b8b2ceda4dc9fa1d5ecedb8.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/ae9d397df6f624061c701c283d153a96c44525d61b8b2ceda4dc9fa1d5ecedb8.jpg)

![afb8c4770e188f142711726490fb7e2ce4d3e0c34da1cd0faf6c609992589994.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/afb8c4770e188f142711726490fb7e2ce4d3e0c34da1cd0faf6c609992589994.jpg)

![c098413ce3c4f2021344df8957a272ff28b67e2fc1e2ce7a62f52fdd2c213ec3.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/c098413ce3c4f2021344df8957a272ff28b67e2fc1e2ce7a62f52fdd2c213ec3.jpg)

![d201f2d716ad31c30b09f5ebeed92470743845a30fdc6606e1f84b612632e4bf.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/d201f2d716ad31c30b09f5ebeed92470743845a30fdc6606e1f84b612632e4bf.jpg)

![db6fa9c03ce2ce312d40038709fcc863eb039b5dbeec8a65faf5b0f9aad765ad.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/db6fa9c03ce2ce312d40038709fcc863eb039b5dbeec8a65faf5b0f9aad765ad.jpg)

![f7280c06e432a0894c93ac3d5f37450069de34e6ef068f22e5fbea44028056d6.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/f7280c06e432a0894c93ac3d5f37450069de34e6ef068f22e5fbea44028056d6.jpg)

![f8191768142e7dd93df927a72eb17d3279077fffa31714aec24b3a6b3cd40343.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/f8191768142e7dd93df927a72eb17d3279077fffa31714aec24b3a6b3cd40343.jpg)

![f97aa908a7eec9e6edaf5cac1bd8c174b9fabc9d364ff8b85a151e0fe9d3dfc3.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/f97aa908a7eec9e6edaf5cac1bd8c174b9fabc9d364ff8b85a151e0fe9d3dfc3.jpg)

![fb614a2e22c6868a19fc94550ed280ddcbba7017a50de2547afd42a2945b8205.jpg](../icml_results/3006_GADA_%20Geometry-Aware%20Deformable%20Aggregation%20for%20Image-Based%20Gaussian%20Splatting/tables/fb614a2e22c6868a19fc94550ed280ddcbba7017a50de2547afd42a2945b8205.jpg)

<a id="paper-2969"></a>

## Action Manifold Smoothing: A Lipschitz Pathway Perspective on High-Dimensional Reinforcement Learning


### Images

![0fe2d68197fa73ee59bf96935a8e1d4bcacdb9d685fa5528762b60c4e8a798d0.jpg](../icml_results/3007_Action%20Manifold%20Smoothing_%20A%20Lipschitz%20Pathway%20Perspective%20on%20High-Dimensional%20Reinforcement%20Learnin/images/0fe2d68197fa73ee59bf96935a8e1d4bcacdb9d685fa5528762b60c4e8a798d0.jpg)

![458665fdb8429f87b2971074dd401699bd826cdff42ce74eabd20f095b43d778.jpg](../icml_results/3007_Action%20Manifold%20Smoothing_%20A%20Lipschitz%20Pathway%20Perspective%20on%20High-Dimensional%20Reinforcement%20Learnin/images/458665fdb8429f87b2971074dd401699bd826cdff42ce74eabd20f095b43d778.jpg)

![84183f57b4c9cb1c1135a30856567cb1dd6f3b842daa1a9ceaf394aeb11bca4f.jpg](../icml_results/3007_Action%20Manifold%20Smoothing_%20A%20Lipschitz%20Pathway%20Perspective%20on%20High-Dimensional%20Reinforcement%20Learnin/images/84183f57b4c9cb1c1135a30856567cb1dd6f3b842daa1a9ceaf394aeb11bca4f.jpg)

![9e63756e68adf2cad09f6b045f536a314c3acf2106e186a8d0c7d68678880bc3.jpg](../icml_results/3007_Action%20Manifold%20Smoothing_%20A%20Lipschitz%20Pathway%20Perspective%20on%20High-Dimensional%20Reinforcement%20Learnin/images/9e63756e68adf2cad09f6b045f536a314c3acf2106e186a8d0c7d68678880bc3.jpg)

![d7e1428d888158d2c617f65366b2b28c8575c3254404f983973295117de82f5c.jpg](../icml_results/3007_Action%20Manifold%20Smoothing_%20A%20Lipschitz%20Pathway%20Perspective%20on%20High-Dimensional%20Reinforcement%20Learnin/images/d7e1428d888158d2c617f65366b2b28c8575c3254404f983973295117de82f5c.jpg)

![dc18f7bce549c309d2f913ec75c193fc499d0233bb7222e3964d2c191808574c.jpg](../icml_results/3007_Action%20Manifold%20Smoothing_%20A%20Lipschitz%20Pathway%20Perspective%20on%20High-Dimensional%20Reinforcement%20Learnin/images/dc18f7bce549c309d2f913ec75c193fc499d0233bb7222e3964d2c191808574c.jpg)

![dec5427312e58059e1b93eb951410f44c6d36e3674a397a53d56da0b233d9f03.jpg](../icml_results/3007_Action%20Manifold%20Smoothing_%20A%20Lipschitz%20Pathway%20Perspective%20on%20High-Dimensional%20Reinforcement%20Learnin/images/dec5427312e58059e1b93eb951410f44c6d36e3674a397a53d56da0b233d9f03.jpg)

### Tables

![0d5ccf918e830839a53d9669b7d98a5dc2d1f960b0e88622f4b064b288245153.jpg](../icml_results/3007_Action%20Manifold%20Smoothing_%20A%20Lipschitz%20Pathway%20Perspective%20on%20High-Dimensional%20Reinforcement%20Learnin/tables/0d5ccf918e830839a53d9669b7d98a5dc2d1f960b0e88622f4b064b288245153.jpg)

![1c7315bd7c48022085f57ae948c7d4b388a319e32679a160073ded240dd0c06b.jpg](../icml_results/3007_Action%20Manifold%20Smoothing_%20A%20Lipschitz%20Pathway%20Perspective%20on%20High-Dimensional%20Reinforcement%20Learnin/tables/1c7315bd7c48022085f57ae948c7d4b388a319e32679a160073ded240dd0c06b.jpg)

![576f9bb7f3cd2e35936d8a87df66d57f80b5f9d0eb7f606373cafb72a423c02b.jpg](../icml_results/3007_Action%20Manifold%20Smoothing_%20A%20Lipschitz%20Pathway%20Perspective%20on%20High-Dimensional%20Reinforcement%20Learnin/tables/576f9bb7f3cd2e35936d8a87df66d57f80b5f9d0eb7f606373cafb72a423c02b.jpg)

![674b39974fe965795a4cb699f70b8b83553c8ddefae18a65b63a1a485cb1bac8.jpg](../icml_results/3007_Action%20Manifold%20Smoothing_%20A%20Lipschitz%20Pathway%20Perspective%20on%20High-Dimensional%20Reinforcement%20Learnin/tables/674b39974fe965795a4cb699f70b8b83553c8ddefae18a65b63a1a485cb1bac8.jpg)

![9907322c83cf468ea9f7d97e38ff87a3a81746274b6d91e8a5334b8c36ac8728.jpg](../icml_results/3007_Action%20Manifold%20Smoothing_%20A%20Lipschitz%20Pathway%20Perspective%20on%20High-Dimensional%20Reinforcement%20Learnin/tables/9907322c83cf468ea9f7d97e38ff87a3a81746274b6d91e8a5334b8c36ac8728.jpg)

![a8ad1ca876fa417239d4b3dde63338710774e79ecd1ee2c54bf812013c75d21c.jpg](../icml_results/3007_Action%20Manifold%20Smoothing_%20A%20Lipschitz%20Pathway%20Perspective%20on%20High-Dimensional%20Reinforcement%20Learnin/tables/a8ad1ca876fa417239d4b3dde63338710774e79ecd1ee2c54bf812013c75d21c.jpg)

![e9895b08319221256cf2941a8cc87ee511526820c09de122a04166227a3879c3.jpg](../icml_results/3007_Action%20Manifold%20Smoothing_%20A%20Lipschitz%20Pathway%20Perspective%20on%20High-Dimensional%20Reinforcement%20Learnin/tables/e9895b08319221256cf2941a8cc87ee511526820c09de122a04166227a3879c3.jpg)

<a id="paper-2970"></a>

## Empirical Gaussian Processes

### Images

![1dea02bc2fe61a3b5f6d371b6e2e128bd185d8cdad714c41180814b341cc34e8.jpg](../icml_results/3008_Empirical%20Gaussian%20Processes/images/1dea02bc2fe61a3b5f6d371b6e2e128bd185d8cdad714c41180814b341cc34e8.jpg)

![24e392895f006ae84b1795c1b6d3542c683532514f13d44c4bdb02b24eb75513.jpg](../icml_results/3008_Empirical%20Gaussian%20Processes/images/24e392895f006ae84b1795c1b6d3542c683532514f13d44c4bdb02b24eb75513.jpg)

![77dc0307213c7f194163b880130021eda429d1587c554f1283027d20f8e02394.jpg](../icml_results/3008_Empirical%20Gaussian%20Processes/images/77dc0307213c7f194163b880130021eda429d1587c554f1283027d20f8e02394.jpg)

![97ea50fa2b2f5f3ae2868556422bf9f677b5ecacf9134d1a76d238b3e0919114.jpg](../icml_results/3008_Empirical%20Gaussian%20Processes/images/97ea50fa2b2f5f3ae2868556422bf9f677b5ecacf9134d1a76d238b3e0919114.jpg)

![aa2850f7a951289de5bd1302e706b27113be364f130880dc7f2d39e062083396.jpg](../icml_results/3008_Empirical%20Gaussian%20Processes/images/aa2850f7a951289de5bd1302e706b27113be364f130880dc7f2d39e062083396.jpg)

![d22d85f296f280ee40ee999d5c30e6e6a3fc46aa852be2c787c840dde160cc5b.jpg](../icml_results/3008_Empirical%20Gaussian%20Processes/images/d22d85f296f280ee40ee999d5c30e6e6a3fc46aa852be2c787c840dde160cc5b.jpg)

![dad457bf78205715a6506f28ebec97f1c0ace489eac26c7b5f46247df6cc6f0f.jpg](../icml_results/3008_Empirical%20Gaussian%20Processes/images/dad457bf78205715a6506f28ebec97f1c0ace489eac26c7b5f46247df6cc6f0f.jpg)

![f9daec78591990d50dc952cb226a46facc64706c05e55eaf59ef7b2b47503c92.jpg](../icml_results/3008_Empirical%20Gaussian%20Processes/images/f9daec78591990d50dc952cb226a46facc64706c05e55eaf59ef7b2b47503c92.jpg)

### Tables

![0029a99a442614e4575e6262b00e594dd0fc2dda768e9fbe4e88fbf127c8ff13.jpg](../icml_results/3008_Empirical%20Gaussian%20Processes/tables/0029a99a442614e4575e6262b00e594dd0fc2dda768e9fbe4e88fbf127c8ff13.jpg)

![2f8c6762209a5641ee60fc7db0a382c1cee04699c939451127869f9683b53d76.jpg](../icml_results/3008_Empirical%20Gaussian%20Processes/tables/2f8c6762209a5641ee60fc7db0a382c1cee04699c939451127869f9683b53d76.jpg)

![33d8e284a39d1810a2a4a61f535210eb245ec43682875ca5919c1e775dd13452.jpg](../icml_results/3008_Empirical%20Gaussian%20Processes/tables/33d8e284a39d1810a2a4a61f535210eb245ec43682875ca5919c1e775dd13452.jpg)

![55f3c3d67707f5885dd7e05b7feb2dfa22098466ae0e7f17966e932249f19174.jpg](../icml_results/3008_Empirical%20Gaussian%20Processes/tables/55f3c3d67707f5885dd7e05b7feb2dfa22098466ae0e7f17966e932249f19174.jpg)

![92f6ff41475ec72d2c97a487f0f932d2ba578a575fa6e10ba7901dca524dea7b.jpg](../icml_results/3008_Empirical%20Gaussian%20Processes/tables/92f6ff41475ec72d2c97a487f0f932d2ba578a575fa6e10ba7901dca524dea7b.jpg)

![9f34275dbe646acfdee561c2717eae358c34795a2bac9911d1e9f00c55a97191.jpg](../icml_results/3008_Empirical%20Gaussian%20Processes/tables/9f34275dbe646acfdee561c2717eae358c34795a2bac9911d1e9f00c55a97191.jpg)

![a6046612a56a14697793db5854dae55ea4c988ece6081385d398e159705717c6.jpg](../icml_results/3008_Empirical%20Gaussian%20Processes/tables/a6046612a56a14697793db5854dae55ea4c988ece6081385d398e159705717c6.jpg)

![bc2de4fb5f211d1343bf428f175816b58514b8a7f0c4ad4929dceaa9d2ce88a0.jpg](../icml_results/3008_Empirical%20Gaussian%20Processes/tables/bc2de4fb5f211d1343bf428f175816b58514b8a7f0c4ad4929dceaa9d2ce88a0.jpg)

![bcf7d72996b817137356fbaed7326f7c5d59904d7992b81d6fa045ff31ec1619.jpg](../icml_results/3008_Empirical%20Gaussian%20Processes/tables/bcf7d72996b817137356fbaed7326f7c5d59904d7992b81d6fa045ff31ec1619.jpg)
