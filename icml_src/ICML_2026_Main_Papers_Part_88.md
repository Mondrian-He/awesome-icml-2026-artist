# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 88 of 197

## 目录 (Table of Contents)

1. [From Historical Patches to Repair Plans: Outcome-Conditioned Reasoning for Repository-Level Program Repair](#paper-2872)
2. [MGAL: A Multilingual Granularity-Aware Long-Context Benchmark](#paper-2873)
3. [IRIS: Implicit Reward-Guided Internal Sifting for Mitigating Multimodal Hallucination](#paper-2874)
4. [Embodied Task Planning via Graph-Informed Action Generation with Large Language Models](#paper-2875)
5. [Mitigating Visual Hallucinations via Semantic Curriculum Preference Optimization in MLLMs](#paper-2876)
6. [Understanding Transfer Learning of RNA Foundation Models on Downstream Tasks](#paper-2877)
7. [GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving](#paper-2878)
8. [Being More Lightweight and Practical: Mini-sized Contrastive Learning Pre-trained Models for Fine-grained Traffic Task](#paper-2879)
9. [Reliable Thinking with Images](#paper-2880)
10. [ComplexMCP: Evaluation of LLM Agents in Dynamic, Interdependent, and Large-Scale Tool Sandbox](#paper-2881)
11. [Breaking the Block: Preserving Data Continuity to Train Superior SAEs for Instruct Models](#paper-2882)
12. [ALSO: Adversarial Online Strategy Optimization for Social Agents](#paper-2883)
13. [Video-OPD: Efficient Post-Training of Multimodal Large Language Models for Temporal Video Grounding via On-Policy Distillation](#paper-2884)
14. [Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](#paper-2885)
15. [FactGuard: Agentic Video Misinformation Detection via Reinforcement Learning](#paper-2886)
16. [Adalina: Adaptive Linear Approximation for the Shapley Value and Beyond](#paper-2887)
17. [Every Step Counts: Decoding Trajectories as Authorship Fingerprints of dLLMs](#paper-2888)
18. [Unsafer in Many Turns: Benchmarking and Defending Multi-Turn Safety Risks in Tool-Using Agents](#paper-2889)
19. [Towards Complete Multi-Agent Coordination Policy Learning via Denoising Maximum Entropy Optimization](#paper-2890)
20. [Probability of Matching for Batch Multi-Objective Bayesian Optimization](#paper-2891)
21. [DiFA: Inference-Time Forward-Process Alignment for Diffusion Models](#paper-2892)
22. [AAD-1: Asymmetric Adversarial Distillation for One-Step Autoregressive Video Generation](#paper-2893)
23. [Mitigating the Contractivity Trap in Diffusion ODEs via Stein Stabilization](#paper-2894)
24. [Eliminating Solution Bias in Differentially Private Optimization](#paper-2895)
25. [Revisiting Photometric Ambiguity for Accurate Gaussian-Splatting Surface Reconstruction](#paper-2896)
26. [Causal Flow Q-Learning for Robust Offline Reinforcement Learning](#paper-2897)
27. [Characterizing the Effect of Noise in Language Generation in the Limit](#paper-2898)
28. [StitchCUDA: An Automated Multi-Agents End-to-End GPU Programing Framework with Rubric-based Agentic Reinforcement Learning](#paper-2899)
29. [PRAC: Principal-Random Subspace for LLM Activation Compression and Memory-Efficient Training](#paper-2900)
30. [From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation](#paper-2901)
31. [LaRI: Layered Ray Intersections for Single-view 3D Geometric Reasoning](#paper-2902)
32. [Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages](#paper-2903)
33. [Modality-Decoupled Online Recursive Editing](#paper-2904)

---


<a id="paper-2872"></a>

## From Historical Patches to Repair Plans: Outcome-Conditioned Reasoning for Repository-Level Program Repair

### Images

![56fb5be429f80483cf70294ccd681f9a93f93d029e0b61db4bba185cb2c2c93b.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/56fb5be429f80483cf70294ccd681f9a93f93d029e0b61db4bba185cb2c2c93b.jpg)

![5e5bd0ac61635c3bcec12f5e056a0b53073e6977ef5cb012673bbf268cb4a93f.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/5e5bd0ac61635c3bcec12f5e056a0b53073e6977ef5cb012673bbf268cb4a93f.jpg)

![9c0efcbec9f76b514ba06c9314ff948b1281d07a240d3e647264615211540013.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/9c0efcbec9f76b514ba06c9314ff948b1281d07a240d3e647264615211540013.jpg)

![b500ee35b86a617ff13c9c52a25a6a056b194163e4eb159eea2b459f96a7d151.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/b500ee35b86a617ff13c9c52a25a6a056b194163e4eb159eea2b459f96a7d151.jpg)

![b60f72a42ac9567e6803a3f86226d1a7ab8b364c34773bfaa0255b688772902d.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/b60f72a42ac9567e6803a3f86226d1a7ab8b364c34773bfaa0255b688772902d.jpg)

![dc9bc41cf85f52ea18959a2a9b36b1c875b1525290c5c92d22007bd3578c41ae.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/dc9bc41cf85f52ea18959a2a9b36b1c875b1525290c5c92d22007bd3578c41ae.jpg)

![ec1c558bf251de27d820cd25648a57d57b40d35fdc3babd9e83d47cfeddc00fe.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/ec1c558bf251de27d820cd25648a57d57b40d35fdc3babd9e83d47cfeddc00fe.jpg)

![effaad8e4756b2710423e9660aa2111d285cdefe0d3315a84a1e89d332fcb00d.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/effaad8e4756b2710423e9660aa2111d285cdefe0d3315a84a1e89d332fcb00d.jpg)

![f7eae85d6c7ce9fbfc91dec5a2e59b7a3a8467a060cf735054e2ef2529985d81.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/images/f7eae85d6c7ce9fbfc91dec5a2e59b7a3a8467a060cf735054e2ef2529985d81.jpg)

### Tables

![052f16907d74ba267ac36af1d00abb70574d1ff5c788c758bef9efc6cce26590.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/052f16907d74ba267ac36af1d00abb70574d1ff5c788c758bef9efc6cce26590.jpg)

![0cf717f989eb030c76d98b57490d4b00665e0973a707f37ba00ede44c66fa5f7.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/0cf717f989eb030c76d98b57490d4b00665e0973a707f37ba00ede44c66fa5f7.jpg)

![190474d364d7dcc464fff43e77cefdf9fbc07ea1641234cd650118785109ce1f.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/190474d364d7dcc464fff43e77cefdf9fbc07ea1641234cd650118785109ce1f.jpg)

![2026eb985b8ff85ad0b9b24c92e13ef8d531f421a9455fed2b64017000fae9fd.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/2026eb985b8ff85ad0b9b24c92e13ef8d531f421a9455fed2b64017000fae9fd.jpg)

![52a4c5c637961c24ff2f5812ac69a69817306ebf918e58f8560ee607f5cbbf85.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/52a4c5c637961c24ff2f5812ac69a69817306ebf918e58f8560ee607f5cbbf85.jpg)

![600627ee1aa562b4013b456af008556e01f39c2739e352adfbea0ce2ee9d1ebd.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/600627ee1aa562b4013b456af008556e01f39c2739e352adfbea0ce2ee9d1ebd.jpg)

![8f4723703f85dafe505587f7a24945c76815d5666b64ef8400c9712f1bf97591.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/8f4723703f85dafe505587f7a24945c76815d5666b64ef8400c9712f1bf97591.jpg)

![92f9afe62d23d62aa44536826aeb65bd37d83225e2e8392f6df6f43b03a2dd5f.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/92f9afe62d23d62aa44536826aeb65bd37d83225e2e8392f6df6f43b03a2dd5f.jpg)

![937315f0f24888003bc9050cf1aec92189b9d01744514843c47e5028fbd4aecd.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/937315f0f24888003bc9050cf1aec92189b9d01744514843c47e5028fbd4aecd.jpg)

![a2d5517a6af14f164e5e3a61cadd08605f471dbcea3ffac40e2e11fd82ef75f4.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/a2d5517a6af14f164e5e3a61cadd08605f471dbcea3ffac40e2e11fd82ef75f4.jpg)

![d3cd9cf8e54ee69db95f0abd87bb00e2bb9884186daf5886052d25783f63223f.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/d3cd9cf8e54ee69db95f0abd87bb00e2bb9884186daf5886052d25783f63223f.jpg)

![f452cc5ec35f4f2aacdc152ee440e4264f1840e71b5ded00ebd2f18761e69fa2.jpg](../icml_results/2909_Spatially-Regularized%20Entropy%20for%20Discriminative%20Token%20Merging%20in%20Fine-Grained%20Re-Identification/tables/f452cc5ec35f4f2aacdc152ee440e4264f1840e71b5ded00ebd2f18761e69fa2.jpg)

## From Historical Patches to Repair Plans: Outcome-Conditioned Reasoning for Repository-Level Program Repair


### Images

![8bf7ffdff7f1a9b96e98dfc55926d7aae5edf3aeca2a7a8d9e0277ca229d7ecc.jpg](../icml_results/2910_From%20Historical%20Patches%20to%20Repair%20Plans_%20Outcome-Conditioned%20Reasoning%20for%20Repository-Level%20Program%20/images/8bf7ffdff7f1a9b96e98dfc55926d7aae5edf3aeca2a7a8d9e0277ca229d7ecc.jpg)

![f093963ce7e40a2d88b548dce89888319fd81a91110c7fec612072dec653072c.jpg](../icml_results/2910_From%20Historical%20Patches%20to%20Repair%20Plans_%20Outcome-Conditioned%20Reasoning%20for%20Repository-Level%20Program%20/images/f093963ce7e40a2d88b548dce89888319fd81a91110c7fec612072dec653072c.jpg)

![f9d250f167c8a0c818a75f8c957580bb3c326f699c6ef88802fc732af5bc7e22.jpg](../icml_results/2910_From%20Historical%20Patches%20to%20Repair%20Plans_%20Outcome-Conditioned%20Reasoning%20for%20Repository-Level%20Program%20/images/f9d250f167c8a0c818a75f8c957580bb3c326f699c6ef88802fc732af5bc7e22.jpg)

### Tables

![05eaafa68b69ccde2abddb41811816d69f9ca90ed40ef490ddba37f5906accc7.jpg](../icml_results/2910_From%20Historical%20Patches%20to%20Repair%20Plans_%20Outcome-Conditioned%20Reasoning%20for%20Repository-Level%20Program%20/tables/05eaafa68b69ccde2abddb41811816d69f9ca90ed40ef490ddba37f5906accc7.jpg)

![3ff5b919c07227cc21554c60cc0cca86ccbbb434204561873289bc5e7b2f386f.jpg](../icml_results/2910_From%20Historical%20Patches%20to%20Repair%20Plans_%20Outcome-Conditioned%20Reasoning%20for%20Repository-Level%20Program%20/tables/3ff5b919c07227cc21554c60cc0cca86ccbbb434204561873289bc5e7b2f386f.jpg)

![9f3b6d31bb2db655fc557d07ff00b6ebec2c2ae016edb85ca4e36854ca9b9045.jpg](../icml_results/2910_From%20Historical%20Patches%20to%20Repair%20Plans_%20Outcome-Conditioned%20Reasoning%20for%20Repository-Level%20Program%20/tables/9f3b6d31bb2db655fc557d07ff00b6ebec2c2ae016edb85ca4e36854ca9b9045.jpg)

![ad59fe291cdaf2105e593ea8c65258f26d8988f3bec3f4d1162e5fa99a5237cc.jpg](../icml_results/2910_From%20Historical%20Patches%20to%20Repair%20Plans_%20Outcome-Conditioned%20Reasoning%20for%20Repository-Level%20Program%20/tables/ad59fe291cdaf2105e593ea8c65258f26d8988f3bec3f4d1162e5fa99a5237cc.jpg)

![c291d3bbcc77c5aade9b4b26e2dbf6a2b89e7efd03d51ceb6138f4b2c8d01050.jpg](../icml_results/2910_From%20Historical%20Patches%20to%20Repair%20Plans_%20Outcome-Conditioned%20Reasoning%20for%20Repository-Level%20Program%20/tables/c291d3bbcc77c5aade9b4b26e2dbf6a2b89e7efd03d51ceb6138f4b2c8d01050.jpg)

![c30c5acd7a37e2f13cd7f971ad8b6356c33d91e4e1e335bbb703c912e772cff6.jpg](../icml_results/2910_From%20Historical%20Patches%20to%20Repair%20Plans_%20Outcome-Conditioned%20Reasoning%20for%20Repository-Level%20Program%20/tables/c30c5acd7a37e2f13cd7f971ad8b6356c33d91e4e1e335bbb703c912e772cff6.jpg)

![da4f9097b2b3e72d1cde26ed4c336d34ac98ce166cb55a55a6e47fc37a26ccb4.jpg](../icml_results/2910_From%20Historical%20Patches%20to%20Repair%20Plans_%20Outcome-Conditioned%20Reasoning%20for%20Repository-Level%20Program%20/tables/da4f9097b2b3e72d1cde26ed4c336d34ac98ce166cb55a55a6e47fc37a26ccb4.jpg)

<a id="paper-2873"></a>

## MGAL: A Multilingual Granularity-Aware Long-Context Benchmark


### Images

![0bbe20fee936b162bf0958c6e1e841ebac495f92189ffaa91b6501ee3a449d96.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/images/0bbe20fee936b162bf0958c6e1e841ebac495f92189ffaa91b6501ee3a449d96.jpg)

![0d3ade265fa7f07d364054e03e85bdf5b0c085d9539e8ac159ab84d52babaae6.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/images/0d3ade265fa7f07d364054e03e85bdf5b0c085d9539e8ac159ab84d52babaae6.jpg)

![1811e9f7145cefaa0a35cbc15adf0a828ffc2a8f8b643243bae6cb5349a39dfc.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/images/1811e9f7145cefaa0a35cbc15adf0a828ffc2a8f8b643243bae6cb5349a39dfc.jpg)

![1f32136bf06fd40a824caf0f11f56ef77b93bf2aaacfce1e613dce7c53d35989.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/images/1f32136bf06fd40a824caf0f11f56ef77b93bf2aaacfce1e613dce7c53d35989.jpg)

![3da1394270ea8165d0b679e95fde5382cf84fac2634c51b0a214639397b5f0cc.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/images/3da1394270ea8165d0b679e95fde5382cf84fac2634c51b0a214639397b5f0cc.jpg)

![6141583272c462cafc8c227913ba9578e64c2976485a4cb180acb5f32c858a30.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/images/6141583272c462cafc8c227913ba9578e64c2976485a4cb180acb5f32c858a30.jpg)

![74c1d63b525742db75c7605f9b72748b1513386f745f325eceb453b1ebc79827.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/images/74c1d63b525742db75c7605f9b72748b1513386f745f325eceb453b1ebc79827.jpg)

![8cedd52bcc894ed684ff2c95bcae97b2b7538d0e89c301582798bce14cf78210.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/images/8cedd52bcc894ed684ff2c95bcae97b2b7538d0e89c301582798bce14cf78210.jpg)

![8f25b0dce4307849f854918f40d58b9cc9e4fa4068c851a1b931a6ae2b6ff4e0.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/images/8f25b0dce4307849f854918f40d58b9cc9e4fa4068c851a1b931a6ae2b6ff4e0.jpg)

![a966296f0962d5df501e0af7a122f132c2f22784a79791170efc0ea295dcf928.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/images/a966296f0962d5df501e0af7a122f132c2f22784a79791170efc0ea295dcf928.jpg)

![ac584d20cbd8b3de391cdc34541e49a68ab9d67a1a1f2ba6bcd627abcac64fe0.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/images/ac584d20cbd8b3de391cdc34541e49a68ab9d67a1a1f2ba6bcd627abcac64fe0.jpg)

![c5be7b95ea4e6d6cf20cda108c9bb6a7cfa19d48271e8e19bed8ed4a1583d4f1.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/images/c5be7b95ea4e6d6cf20cda108c9bb6a7cfa19d48271e8e19bed8ed4a1583d4f1.jpg)

![cdea11df2300c8b91ca5d42605bfbe49968c54c64e50741621b376370e246aa9.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/images/cdea11df2300c8b91ca5d42605bfbe49968c54c64e50741621b376370e246aa9.jpg)

![da4550291cb32c9b2ae737755c731b56ee282fbe9f843cc8bbb8ee6287565cf7.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/images/da4550291cb32c9b2ae737755c731b56ee282fbe9f843cc8bbb8ee6287565cf7.jpg)

![e6adb1bd6f55e4d75c67dacdea824f39b8b4254911b4375a8943e905d7cf0242.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/images/e6adb1bd6f55e4d75c67dacdea824f39b8b4254911b4375a8943e905d7cf0242.jpg)

![f34b26ba24ec62be411fc57f16d5abd54d7060ddb000f18503c29565015d2094.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/images/f34b26ba24ec62be411fc57f16d5abd54d7060ddb000f18503c29565015d2094.jpg)

![fc1b03bab52e28f1edfdae26d2aeb38348b91937c8c658475c3c7fe9550833c5.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/images/fc1b03bab52e28f1edfdae26d2aeb38348b91937c8c658475c3c7fe9550833c5.jpg)

### Tables

![155e037e6443e6a30886a186875680a1b61feaf42bf4de28db91a275505aec0f.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/155e037e6443e6a30886a186875680a1b61feaf42bf4de28db91a275505aec0f.jpg)

![18d8bca3dbc83818cd1c18225aee38c134514a6d86d0ba8a16bb6502634ca18e.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/18d8bca3dbc83818cd1c18225aee38c134514a6d86d0ba8a16bb6502634ca18e.jpg)

![1dfd6a6f60f6e7091e49f9400ba342637e5c61d47529c9a7592484a38cdecb67.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/1dfd6a6f60f6e7091e49f9400ba342637e5c61d47529c9a7592484a38cdecb67.jpg)

![269e4a4c9a2ade0c1353ccf917a4dcd7ca514e025dcb7e6e52e2ee650de3e910.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/269e4a4c9a2ade0c1353ccf917a4dcd7ca514e025dcb7e6e52e2ee650de3e910.jpg)

![278c2e14a5825dc718289d4ed1819b29c7c161ed7fba9faa49727c7e3e7ebf8c.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/278c2e14a5825dc718289d4ed1819b29c7c161ed7fba9faa49727c7e3e7ebf8c.jpg)

![29b29ca3e9e66ba82f0e62e58ee9592695cdc030e40347d5860ab31fd59b681b.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/29b29ca3e9e66ba82f0e62e58ee9592695cdc030e40347d5860ab31fd59b681b.jpg)

![32f5885c9e60e70a0d44c4ce83a73f1b90d0f8b17d1c76eb03e5e78bab0d8554.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/32f5885c9e60e70a0d44c4ce83a73f1b90d0f8b17d1c76eb03e5e78bab0d8554.jpg)

![3e04661439daa7ca5ee7d7c62f1306feecb0edb68fc4a6d1626cd1a1b97e787b.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/3e04661439daa7ca5ee7d7c62f1306feecb0edb68fc4a6d1626cd1a1b97e787b.jpg)

![47969eae56760da693cb79e306ffd9a6d78ddae8ae011ad39d33b20cd6b3c4d2.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/47969eae56760da693cb79e306ffd9a6d78ddae8ae011ad39d33b20cd6b3c4d2.jpg)

![4a03f22dad0a6d620a97224af736a51d2a6317f43a3acf14ee99c3cfbfc1f170.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/4a03f22dad0a6d620a97224af736a51d2a6317f43a3acf14ee99c3cfbfc1f170.jpg)

![56b1b03e958afd967b93706e6e7426f182c9da818d654fa377f2ed9aeaec59f9.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/56b1b03e958afd967b93706e6e7426f182c9da818d654fa377f2ed9aeaec59f9.jpg)

![5e6ea4be1f40ea3bd678544efc4fec8f03a311e1325c9541b0ac1a5c0185dd08.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/5e6ea4be1f40ea3bd678544efc4fec8f03a311e1325c9541b0ac1a5c0185dd08.jpg)

![6cb7d173f5d2a4e64644513040059f87e212008409f79c5184d2d2d0a30af1c5.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/6cb7d173f5d2a4e64644513040059f87e212008409f79c5184d2d2d0a30af1c5.jpg)

![70f7bb7bea4dbbceac6e7c417e5ddc98362bb8ba879ff1968afbe5fa649b44a3.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/70f7bb7bea4dbbceac6e7c417e5ddc98362bb8ba879ff1968afbe5fa649b44a3.jpg)

![7c0cfbf02871157f19396afeb8e2163f23091d7bd0f42f0bfc9d56ab9e3b7805.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/7c0cfbf02871157f19396afeb8e2163f23091d7bd0f42f0bfc9d56ab9e3b7805.jpg)

![7fd788333ba03bd0cb0a357e21430ac9120312b4d41bd5a358d879456b0dd9c0.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/7fd788333ba03bd0cb0a357e21430ac9120312b4d41bd5a358d879456b0dd9c0.jpg)

![8829dfd1838259b2b11a704a476143aaa468dcecd9226f5f331339925c571368.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/8829dfd1838259b2b11a704a476143aaa468dcecd9226f5f331339925c571368.jpg)

![8bc726eb9554c25424adce2c5c5af07bd3edbea72e08840ce1e223aa98fefbb4.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/8bc726eb9554c25424adce2c5c5af07bd3edbea72e08840ce1e223aa98fefbb4.jpg)

![93eb88a549711f4aced649941ad9950b92173ea073d2461eab550773b65e903b.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/93eb88a549711f4aced649941ad9950b92173ea073d2461eab550773b65e903b.jpg)

![98a9b68b38d04ea21670a951e5fe113205d7c7418f9890b34d3ab8520d30da5f.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/98a9b68b38d04ea21670a951e5fe113205d7c7418f9890b34d3ab8520d30da5f.jpg)

![9aafe4de76dcb285faf989f8644e5fd1dd782c49965443072042dddffb2b48c6.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/9aafe4de76dcb285faf989f8644e5fd1dd782c49965443072042dddffb2b48c6.jpg)

![9b335ffff7a0ccff70d6b8fea39f8cec76625ccc4c36ee1ca1b30f9e29459baa.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/9b335ffff7a0ccff70d6b8fea39f8cec76625ccc4c36ee1ca1b30f9e29459baa.jpg)

![b59c0f3c008a48fdbab35c5a58805003bbe39be64c91919242405b520ccb30e5.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/b59c0f3c008a48fdbab35c5a58805003bbe39be64c91919242405b520ccb30e5.jpg)

![d56191ba43c13f4ddf8b5bb9a1ab19435c8ad9084762b1bb579fca30200ba0bd.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/d56191ba43c13f4ddf8b5bb9a1ab19435c8ad9084762b1bb579fca30200ba0bd.jpg)

![d9a5c317e790321ea4c02e7d28cd86f78c8e1787fd519587cd8c348ee620ebfd.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/d9a5c317e790321ea4c02e7d28cd86f78c8e1787fd519587cd8c348ee620ebfd.jpg)

![e576b4b8ee936179309bcd39276b769077aa0575fc72b736136a80ced1c8be67.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/e576b4b8ee936179309bcd39276b769077aa0575fc72b736136a80ced1c8be67.jpg)

![ea5a510b2dabaf027be5437bdc193189ad282ff60462d63434842603251f3746.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/ea5a510b2dabaf027be5437bdc193189ad282ff60462d63434842603251f3746.jpg)

![fa3e34153992f4febabb9cbebb4e0fa01f0fd5307e68af1043ac5748a66a258c.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/fa3e34153992f4febabb9cbebb4e0fa01f0fd5307e68af1043ac5748a66a258c.jpg)

![fad28caa110686f546724460025e42e11e5fdf763adef39c4ff34353d7726965.jpg](../icml_results/2911_MGAL_%20A%20Multilingual%20Granularity-Aware%20Long-Context%20Benchmark/tables/fad28caa110686f546724460025e42e11e5fdf763adef39c4ff34353d7726965.jpg)

<a id="paper-2874"></a>

## IRIS: Implicit Reward-Guided Internal Sifting for Mitigating Multimodal Hallucination


### Images

![16f034a799880fcf0a904fd396c6176fa3d3629d3051e53ae9f1eaaba30d10bd.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/images/16f034a799880fcf0a904fd396c6176fa3d3629d3051e53ae9f1eaaba30d10bd.jpg)

![26ec70fd5a408d8ba199f8925b7f213915d5cf22eeb79d8fed0738d34c6428e2.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/images/26ec70fd5a408d8ba199f8925b7f213915d5cf22eeb79d8fed0738d34c6428e2.jpg)

![3754123544a33337a6e73f53a25b60b03dec125fa688604fd95b6b65dfcb7f3e.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/images/3754123544a33337a6e73f53a25b60b03dec125fa688604fd95b6b65dfcb7f3e.jpg)

![510daca6072a0d79ce14c1a2f2a907f104f64c4056f06c28cc1879a38a35994f.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/images/510daca6072a0d79ce14c1a2f2a907f104f64c4056f06c28cc1879a38a35994f.jpg)

![78c41160447c11428e10448968dcc6ec629c0cd56225d202d934b7777c98ca36.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/images/78c41160447c11428e10448968dcc6ec629c0cd56225d202d934b7777c98ca36.jpg)

![85e63503dbe8b8ff7f2c9632172b11d7087c7098a676e9b8edf043828972423a.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/images/85e63503dbe8b8ff7f2c9632172b11d7087c7098a676e9b8edf043828972423a.jpg)

![9bd3b852bd870237d1eea97cad7c6d9c8cdfa0e83af5f9dd4e2e74c7080c3de5.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/images/9bd3b852bd870237d1eea97cad7c6d9c8cdfa0e83af5f9dd4e2e74c7080c3de5.jpg)

![e254cffa5073e2db66a3465ed93a8a96d2d5f6ed927fb1e6463a536798aa37c2.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/images/e254cffa5073e2db66a3465ed93a8a96d2d5f6ed927fb1e6463a536798aa37c2.jpg)

![e54eff28ba10cc15ee73210d5d13aeb37422b65453436d29f0747974e44c8076.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/images/e54eff28ba10cc15ee73210d5d13aeb37422b65453436d29f0747974e44c8076.jpg)

![e60078126fe97dab35cde5656332c9110846fb414267f9d25dad29451ae0e4f5.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/images/e60078126fe97dab35cde5656332c9110846fb414267f9d25dad29451ae0e4f5.jpg)

![fefff9e18ce4cce3ae53341e90bc320f2a4d54db17f2b1d7ed16f4a941199db1.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/images/fefff9e18ce4cce3ae53341e90bc320f2a4d54db17f2b1d7ed16f4a941199db1.jpg)

### Tables

![22acf0290d6e7270ec579bb6f0f3afc3825d62a4eec45fbc9ac114896d2770ae.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/tables/22acf0290d6e7270ec579bb6f0f3afc3825d62a4eec45fbc9ac114896d2770ae.jpg)

![2ef8fa6fe07050efe82328b2af5a12f69dc7c0223b6c78aa2f71ab77235c78e8.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/tables/2ef8fa6fe07050efe82328b2af5a12f69dc7c0223b6c78aa2f71ab77235c78e8.jpg)

![48574cb46d511c5b79e72ce4485ad5a06e02983042804e981db60d7b658e719f.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/tables/48574cb46d511c5b79e72ce4485ad5a06e02983042804e981db60d7b658e719f.jpg)

![626dc538c2819a33fe6aef58e079bf7b374aa8e73fb52713a1de860aae7e1bc5.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/tables/626dc538c2819a33fe6aef58e079bf7b374aa8e73fb52713a1de860aae7e1bc5.jpg)

![8188a9aa26cebfe476a077c081bece3a23f6a913cecdd559ac9573dcbc9081ec.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/tables/8188a9aa26cebfe476a077c081bece3a23f6a913cecdd559ac9573dcbc9081ec.jpg)

![828ad4a585846a230c0f4c75f05b9e6c73f69dc283a40aab6b84651384f04b10.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/tables/828ad4a585846a230c0f4c75f05b9e6c73f69dc283a40aab6b84651384f04b10.jpg)

![8e7ac8b63d8a9674194287b536bd102ea6aa183bd4180f24e903eda9b9c032cc.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/tables/8e7ac8b63d8a9674194287b536bd102ea6aa183bd4180f24e903eda9b9c032cc.jpg)

![9f743476974c3328a0878b30428a72adcaaa5112e03a7c6a9cab066c4cb376d6.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/tables/9f743476974c3328a0878b30428a72adcaaa5112e03a7c6a9cab066c4cb376d6.jpg)

![aa0368797e84c2bc9e10b908a07bdf305c35012b29edfa905fff4f9a744e28e7.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/tables/aa0368797e84c2bc9e10b908a07bdf305c35012b29edfa905fff4f9a744e28e7.jpg)

![b0fb2b7e729515b87c46c7614458b23a2692b81e93ccec36817a590021a0c058.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/tables/b0fb2b7e729515b87c46c7614458b23a2692b81e93ccec36817a590021a0c058.jpg)

![d556b41c0453b3b349422489cd75198bca709bf3fc6ec03ecdcb87101d979aba.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/tables/d556b41c0453b3b349422489cd75198bca709bf3fc6ec03ecdcb87101d979aba.jpg)

![ee6b47b7e22472628fcd8b1cb2e898c6effc17ad754e5255093385d9b0711fdf.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/tables/ee6b47b7e22472628fcd8b1cb2e898c6effc17ad754e5255093385d9b0711fdf.jpg)

![ff2a9b5c93a9baeab0b4cef9033bce9b2dd2f98d99924888e173ae7f1c72fb87.jpg](../icml_results/2912_IRIS_%20Implicit%20Reward-Guided%20Internal%20Sifting%20for%20Mitigating%20Multimodal%20Hallucination/tables/ff2a9b5c93a9baeab0b4cef9033bce9b2dd2f98d99924888e173ae7f1c72fb87.jpg)

<a id="paper-2875"></a>

## Embodied Task Planning via Graph-Informed Action Generation with Large Language Models


### Images

![34b786507b5db58c759755b77177492bbdf479ad5afa36f8214475442180c421.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/images/34b786507b5db58c759755b77177492bbdf479ad5afa36f8214475442180c421.jpg)

![3c95bcd11050cd3cf3b5499c5babe3ada31c138b1fd38cd753cc9799811a2a00.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/images/3c95bcd11050cd3cf3b5499c5babe3ada31c138b1fd38cd753cc9799811a2a00.jpg)

![4e45d94cf964fe1cf116b9d843bb405ff11af68bf3dbdce701637cc1c9af1a67.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/images/4e45d94cf964fe1cf116b9d843bb405ff11af68bf3dbdce701637cc1c9af1a67.jpg)

![6ebb925fa8f86310e9341943d6e826d957e06681ab96ae5fd2ec20c871e6fdb6.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/images/6ebb925fa8f86310e9341943d6e826d957e06681ab96ae5fd2ec20c871e6fdb6.jpg)

![8315adfe949074ac564ac3462ea9e4752bac660a2d6b74f193544e6deefd4255.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/images/8315adfe949074ac564ac3462ea9e4752bac660a2d6b74f193544e6deefd4255.jpg)

![8dbbd1cf0d758fbf7b57c0cb6791bb3d72cfefc77ffc2254ab0f30e782a4744b.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/images/8dbbd1cf0d758fbf7b57c0cb6791bb3d72cfefc77ffc2254ab0f30e782a4744b.jpg)

![8e1b47efa5478cd5e2ff119da0f77af5497609faf535856a2be6118634d6309d.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/images/8e1b47efa5478cd5e2ff119da0f77af5497609faf535856a2be6118634d6309d.jpg)

![951276bc6b2b5fd35e824e56dfdf07ad50f392ffa398e46e2993a35bce9efc7b.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/images/951276bc6b2b5fd35e824e56dfdf07ad50f392ffa398e46e2993a35bce9efc7b.jpg)

![b531c6495fcd41f5fa757c97277d7ac4ce45ff5c84efeb717604c256e9dde916.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/images/b531c6495fcd41f5fa757c97277d7ac4ce45ff5c84efeb717604c256e9dde916.jpg)

![c439bf3f5b9c2248638d30a1bc36484e891a820bda707383e5d08388a7aa1cfc.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/images/c439bf3f5b9c2248638d30a1bc36484e891a820bda707383e5d08388a7aa1cfc.jpg)

![d1ee1746ed39dc1d9945afb967a177ffa8c6223d2c02a03a5117022362121de5.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/images/d1ee1746ed39dc1d9945afb967a177ffa8c6223d2c02a03a5117022362121de5.jpg)

### Tables

![068b8c5ea562e9e6d9356333d044a98c0e5b0e8dabb6bd416125e9f1dafe0353.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/068b8c5ea562e9e6d9356333d044a98c0e5b0e8dabb6bd416125e9f1dafe0353.jpg)

![15350d69b0319238404a45a8cfac0dacf2571b250539aad6ca3b2f04dd6a733d.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/15350d69b0319238404a45a8cfac0dacf2571b250539aad6ca3b2f04dd6a733d.jpg)

![18d7840bbd7f8ccab459f5238173f70a746e4d9c1dd3d3211bc3dcb858c34e80.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/18d7840bbd7f8ccab459f5238173f70a746e4d9c1dd3d3211bc3dcb858c34e80.jpg)

![1a684e90f5b53931c3bad38819332c92f7a3758725f981a5a1d323358e4e51cb.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/1a684e90f5b53931c3bad38819332c92f7a3758725f981a5a1d323358e4e51cb.jpg)

![302bfd85fba8c9b39e04048fa672283fcc6ddd1280ea041958cfd55b35aa3e80.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/302bfd85fba8c9b39e04048fa672283fcc6ddd1280ea041958cfd55b35aa3e80.jpg)

![33f49b034d11d4d372ab8e6d6afc62002a9dc9099c36cbb6074969a3204ce1c6.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/33f49b034d11d4d372ab8e6d6afc62002a9dc9099c36cbb6074969a3204ce1c6.jpg)

![34ae994daad6b45fc5f9c52a975b862c818e7bf8ab9388ef1344b8910f9f1575.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/34ae994daad6b45fc5f9c52a975b862c818e7bf8ab9388ef1344b8910f9f1575.jpg)

![4f14401e53e605f5874ed1fdf7754c3baf3ff75cad51b3097407ad26a9a7f8f0.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/4f14401e53e605f5874ed1fdf7754c3baf3ff75cad51b3097407ad26a9a7f8f0.jpg)

![5588b81f8aca844d1418f58b119ebbfb1d438c6aa8be22a507e8f09be3c07fd3.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/5588b81f8aca844d1418f58b119ebbfb1d438c6aa8be22a507e8f09be3c07fd3.jpg)

![5fceff6f1ae487f36ff33afe87096048f32acc83be2364aca907537101b62aa8.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/5fceff6f1ae487f36ff33afe87096048f32acc83be2364aca907537101b62aa8.jpg)

![63373749275e6735f4533e8305b2a571e1afd7e06fc23b752151c36df70a5869.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/63373749275e6735f4533e8305b2a571e1afd7e06fc23b752151c36df70a5869.jpg)

![66cb1462b4cca7e87a676df94a892348872563872359286f4ddb0f948e5843a7.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/66cb1462b4cca7e87a676df94a892348872563872359286f4ddb0f948e5843a7.jpg)

![716bb98f086a6521df5c18529138a07c028c0536933540703023a617306951c7.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/716bb98f086a6521df5c18529138a07c028c0536933540703023a617306951c7.jpg)

![735729d694330826d4e6355c07964317c0ba8b5e206636f1c225abb60dfe2122.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/735729d694330826d4e6355c07964317c0ba8b5e206636f1c225abb60dfe2122.jpg)

![8efaa4f6c02289b9e95b3421c379872f0da07e0fb91b046420e30d2f8bfc20c1.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/8efaa4f6c02289b9e95b3421c379872f0da07e0fb91b046420e30d2f8bfc20c1.jpg)

![96e5153803b8f2bf54c4f75c3da82eb46b9afc203cc7d2e21f798888989d7541.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/96e5153803b8f2bf54c4f75c3da82eb46b9afc203cc7d2e21f798888989d7541.jpg)

![9bb629262c9253d2f88f3eea26011cc1dbdf2be63e08f06755322574421f9b29.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/9bb629262c9253d2f88f3eea26011cc1dbdf2be63e08f06755322574421f9b29.jpg)

![9e42fa752d18192b8636518582e7fc94608c23e5c8948f1487660d9c77db1815.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/9e42fa752d18192b8636518582e7fc94608c23e5c8948f1487660d9c77db1815.jpg)

![a2cd9ea1aad92bf83c276a54effe454f41b74f3417f940b6607f47bd488cf5cd.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/a2cd9ea1aad92bf83c276a54effe454f41b74f3417f940b6607f47bd488cf5cd.jpg)

![bc13003d091dda3656fad23362a80169f4bafd85b5f7d10dbb0e871964a8c46f.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/bc13003d091dda3656fad23362a80169f4bafd85b5f7d10dbb0e871964a8c46f.jpg)

![d68ac1e1386a03f951cc214a87d9a82be26f04e208a78571f44be04760e14ac8.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/d68ac1e1386a03f951cc214a87d9a82be26f04e208a78571f44be04760e14ac8.jpg)

![d9ee21dd601ffe566d8a1d26a4803ccdf878cdd412f8fab5e8e5cb031bab9ff6.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/d9ee21dd601ffe566d8a1d26a4803ccdf878cdd412f8fab5e8e5cb031bab9ff6.jpg)

![e9912395c664b835a408194f7cac8af3a27105749b39a0d9148e76aba5c69809.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/e9912395c664b835a408194f7cac8af3a27105749b39a0d9148e76aba5c69809.jpg)

![fbf269232637d73c1a39e4e73eeadaf1bc3e105efbcb940817c5a2859a517292.jpg](../icml_results/2913_Embodied%20Task%20Planning%20via%20Graph-Informed%20Action%20Generation%20with%20Large%20Language%20Models/tables/fbf269232637d73c1a39e4e73eeadaf1bc3e105efbcb940817c5a2859a517292.jpg)

<a id="paper-2876"></a>

## Mitigating Visual Hallucinations via Semantic Curriculum Preference Optimization in MLLMs


### Images

![24f50b8cd11ddbc1fe86d2471228d91932d2270997d2c3256aa52874112f6a6e.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/images/24f50b8cd11ddbc1fe86d2471228d91932d2270997d2c3256aa52874112f6a6e.jpg)

![5547681aaf4072c5b826641a8f063111f0d94484a46822d86679d23c807a2dc5.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/images/5547681aaf4072c5b826641a8f063111f0d94484a46822d86679d23c807a2dc5.jpg)

![5e7cf65b14decfb3f0890ad57ca2b9b83fb64d3dfd5918d79e84d2ee138f91f1.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/images/5e7cf65b14decfb3f0890ad57ca2b9b83fb64d3dfd5918d79e84d2ee138f91f1.jpg)

![66d606d224e703a88e79401eb62efb97ebcc6c11cad7899565543486a12bbbc2.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/images/66d606d224e703a88e79401eb62efb97ebcc6c11cad7899565543486a12bbbc2.jpg)

![8e167e85c6f5b98d4ccf37445297d769adea2ec736ee50758bd7dba6ea349424.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/images/8e167e85c6f5b98d4ccf37445297d769adea2ec736ee50758bd7dba6ea349424.jpg)

![a5114b6b6c9e0f885c367c127dbe5f935cf4ac77adcdf581c103feca50168e57.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/images/a5114b6b6c9e0f885c367c127dbe5f935cf4ac77adcdf581c103feca50168e57.jpg)

![a7afcefde0074500f7b61674da7035bb1cddb3476d5d7a4ae05c8a4e64d38131.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/images/a7afcefde0074500f7b61674da7035bb1cddb3476d5d7a4ae05c8a4e64d38131.jpg)

![d572c8c641048b30b3514261283f3985b9e6b47da755bf25669c3e283c2a1814.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/images/d572c8c641048b30b3514261283f3985b9e6b47da755bf25669c3e283c2a1814.jpg)

### Tables

![143061fcdc5174a7e3780eb40f4b5c00ad93dd6b300ea9a6fdb5f18e788aa5b3.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/tables/143061fcdc5174a7e3780eb40f4b5c00ad93dd6b300ea9a6fdb5f18e788aa5b3.jpg)

![2caae4ed38a761f2404537623caea916c808beb0ee1df95becccc5f88f8c692c.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/tables/2caae4ed38a761f2404537623caea916c808beb0ee1df95becccc5f88f8c692c.jpg)

![3aae40d8b75cc0f4198f973f566d40e28911ce046471874460c43554bfe11cb4.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/tables/3aae40d8b75cc0f4198f973f566d40e28911ce046471874460c43554bfe11cb4.jpg)

![408b8131bf1282349fe4e6e87d9783f915a239be31a3e4021fd9b5d21ffccb50.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/tables/408b8131bf1282349fe4e6e87d9783f915a239be31a3e4021fd9b5d21ffccb50.jpg)

![4ff8dce083e946d0ee4fda463dca1cc3bce7c08ff3fe7e567f7313fe83caed10.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/tables/4ff8dce083e946d0ee4fda463dca1cc3bce7c08ff3fe7e567f7313fe83caed10.jpg)

![88cd2ee7175bad2bea3ecca13ea6419b973df11a1710343cd09a0d2d034b9421.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/tables/88cd2ee7175bad2bea3ecca13ea6419b973df11a1710343cd09a0d2d034b9421.jpg)

![8b7dc43d1b4979fe7faefa00aa781511da58f2f2d0885aecc1ac58e47f23317a.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/tables/8b7dc43d1b4979fe7faefa00aa781511da58f2f2d0885aecc1ac58e47f23317a.jpg)

![910e5e4184531e8ca0b1fd961b9ab31c3ca00e706eb52b5cc57d2222f303358c.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/tables/910e5e4184531e8ca0b1fd961b9ab31c3ca00e706eb52b5cc57d2222f303358c.jpg)

![b874396ebb6a72730464a42eeffec02d9ca1967202e51792f97665d4b7c12756.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/tables/b874396ebb6a72730464a42eeffec02d9ca1967202e51792f97665d4b7c12756.jpg)

![bb916ca4a6011848114b798387b8ece7e701875bbb3e8475a0c997b641b6f575.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/tables/bb916ca4a6011848114b798387b8ece7e701875bbb3e8475a0c997b641b6f575.jpg)

![c834f96cf96e0c850666d9d4b5b46f1a3ba1ebe98046b008103f6b4bf860387a.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/tables/c834f96cf96e0c850666d9d4b5b46f1a3ba1ebe98046b008103f6b4bf860387a.jpg)

![c9af98e80de8e4db35d24cb022809ea0ca40b8b9a026fa315a6f0044d446f09a.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/tables/c9af98e80de8e4db35d24cb022809ea0ca40b8b9a026fa315a6f0044d446f09a.jpg)

![f2485c9ed8bcc48e268ce15f0b188be05c4801d58d2760d60825313213adc4c2.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/tables/f2485c9ed8bcc48e268ce15f0b188be05c4801d58d2760d60825313213adc4c2.jpg)

![f5b5fa77950dbded424b4186c03d099ec9fb2cd99e567c547e815d466c9957d1.jpg](../icml_results/2914_Mitigating%20Visual%20Hallucinations%20via%20Semantic%20Curriculum%20Preference%20Optimization%20in%20MLLMs/tables/f5b5fa77950dbded424b4186c03d099ec9fb2cd99e567c547e815d466c9957d1.jpg)

<a id="paper-2877"></a>

## Understanding Transfer Learning of RNA Foundation Models on Downstream Tasks


### Images

![371a5aa7d325adc46d370418df87e5583414d7e5fd5a7633957d966b9f18969c.jpg](../icml_results/2915_Understanding%20Transfer%20Learning%20of%20RNA%20Foundation%20Models%20on%20Downstream%20Tasks/images/371a5aa7d325adc46d370418df87e5583414d7e5fd5a7633957d966b9f18969c.jpg)

![5a65cde5febd078e5c84e28f23b12ad8b19894dc2c76b51c58e41dd8802399c9.jpg](../icml_results/2915_Understanding%20Transfer%20Learning%20of%20RNA%20Foundation%20Models%20on%20Downstream%20Tasks/images/5a65cde5febd078e5c84e28f23b12ad8b19894dc2c76b51c58e41dd8802399c9.jpg)

![7747067eee29e2908f1b70a3747d2c04d78c2900c9389d07604c630cdac5b443.jpg](../icml_results/2915_Understanding%20Transfer%20Learning%20of%20RNA%20Foundation%20Models%20on%20Downstream%20Tasks/images/7747067eee29e2908f1b70a3747d2c04d78c2900c9389d07604c630cdac5b443.jpg)

![8e2ba47ae42f6ee57caf8d4ca7d7cd8cd282a62bd98ff4120a86fb7ad8f6311a.jpg](../icml_results/2915_Understanding%20Transfer%20Learning%20of%20RNA%20Foundation%20Models%20on%20Downstream%20Tasks/images/8e2ba47ae42f6ee57caf8d4ca7d7cd8cd282a62bd98ff4120a86fb7ad8f6311a.jpg)

![bd9f44f6e7102be49a6a448ec32a5d0f8e2d5c05b0abe50991511cfe71a049cf.jpg](../icml_results/2915_Understanding%20Transfer%20Learning%20of%20RNA%20Foundation%20Models%20on%20Downstream%20Tasks/images/bd9f44f6e7102be49a6a448ec32a5d0f8e2d5c05b0abe50991511cfe71a049cf.jpg)

![d804c965bdf98d5ed8a1ac956c89266476a8eb11b9c8b67c3d1a9f10c3b46d2e.jpg](../icml_results/2915_Understanding%20Transfer%20Learning%20of%20RNA%20Foundation%20Models%20on%20Downstream%20Tasks/images/d804c965bdf98d5ed8a1ac956c89266476a8eb11b9c8b67c3d1a9f10c3b46d2e.jpg)

![dca1d6d8dee6b9d5725b1c728bf070311cb7eedd2bd507b9579f2653a9bb7c28.jpg](../icml_results/2915_Understanding%20Transfer%20Learning%20of%20RNA%20Foundation%20Models%20on%20Downstream%20Tasks/images/dca1d6d8dee6b9d5725b1c728bf070311cb7eedd2bd507b9579f2653a9bb7c28.jpg)

![dd9f1009bda44c47e55eba1b59b4e96c1c8967f813c44d7cfc8c885261aaa03b.jpg](../icml_results/2915_Understanding%20Transfer%20Learning%20of%20RNA%20Foundation%20Models%20on%20Downstream%20Tasks/images/dd9f1009bda44c47e55eba1b59b4e96c1c8967f813c44d7cfc8c885261aaa03b.jpg)

### Tables

![1e402f6019ca665cc9d826fc4378b08e411ee893763f548a9497e7ee56383ac3.jpg](../icml_results/2915_Understanding%20Transfer%20Learning%20of%20RNA%20Foundation%20Models%20on%20Downstream%20Tasks/tables/1e402f6019ca665cc9d826fc4378b08e411ee893763f548a9497e7ee56383ac3.jpg)

![283844f7e819f8643dd8c277c7191842e282609fbef52713c20bedc7a6f040c6.jpg](../icml_results/2915_Understanding%20Transfer%20Learning%20of%20RNA%20Foundation%20Models%20on%20Downstream%20Tasks/tables/283844f7e819f8643dd8c277c7191842e282609fbef52713c20bedc7a6f040c6.jpg)

![4a797a6ae2e4e7cd2d12d498538b38e0bcc51502190fb757f23049756c18d06f.jpg](../icml_results/2915_Understanding%20Transfer%20Learning%20of%20RNA%20Foundation%20Models%20on%20Downstream%20Tasks/tables/4a797a6ae2e4e7cd2d12d498538b38e0bcc51502190fb757f23049756c18d06f.jpg)

![6ba4861e6752df6a576e7db4bc957a3521c2c99b83fbca44c7b53e0f41e96de0.jpg](../icml_results/2915_Understanding%20Transfer%20Learning%20of%20RNA%20Foundation%20Models%20on%20Downstream%20Tasks/tables/6ba4861e6752df6a576e7db4bc957a3521c2c99b83fbca44c7b53e0f41e96de0.jpg)

![b5eb05cf125f0e8ec3140db7d54a31a61b526c68be928fc2b5ac5652a7db0ad5.jpg](../icml_results/2915_Understanding%20Transfer%20Learning%20of%20RNA%20Foundation%20Models%20on%20Downstream%20Tasks/tables/b5eb05cf125f0e8ec3140db7d54a31a61b526c68be928fc2b5ac5652a7db0ad5.jpg)

![bc4e5b47b4487de2367117cefc21900573453fd52defd8356921cc19150737b8.jpg](../icml_results/2915_Understanding%20Transfer%20Learning%20of%20RNA%20Foundation%20Models%20on%20Downstream%20Tasks/tables/bc4e5b47b4487de2367117cefc21900573453fd52defd8356921cc19150737b8.jpg)

![c3abcc42d5a4b3d9138add44a0ff8d90dd9c42725211b3bb2386e4776a4ad4fb.jpg](../icml_results/2915_Understanding%20Transfer%20Learning%20of%20RNA%20Foundation%20Models%20on%20Downstream%20Tasks/tables/c3abcc42d5a4b3d9138add44a0ff8d90dd9c42725211b3bb2386e4776a4ad4fb.jpg)

![e8a2fa2fe9408988ccff3734bd72e6acc8276191610f02544eab59cfcffd49fe.jpg](../icml_results/2915_Understanding%20Transfer%20Learning%20of%20RNA%20Foundation%20Models%20on%20Downstream%20Tasks/tables/e8a2fa2fe9408988ccff3734bd72e6acc8276191610f02544eab59cfcffd49fe.jpg)

![f9385fe2184ecaa98d1400fac98a60cf55ae719cb833369029d899f216308fc9.jpg](../icml_results/2915_Understanding%20Transfer%20Learning%20of%20RNA%20Foundation%20Models%20on%20Downstream%20Tasks/tables/f9385fe2184ecaa98d1400fac98a60cf55ae719cb833369029d899f216308fc9.jpg)

<a id="paper-2878"></a>

## GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving


### Images

![0a4372e0a54800a5fb27e63aa1dff31002fa02a5cf04b1b1905c2804e56c209b.jpg](../icml_results/2916_GraphFlow_%20A%20Graph-Based%20Workflow%20Management%20for%20Efficient%20LLM-Agent%20Serving/images/0a4372e0a54800a5fb27e63aa1dff31002fa02a5cf04b1b1905c2804e56c209b.jpg)

![3f37d2568fa01951a5dace3c9a3533c9b0115b4f8c6c8e2815231b8a046871d3.jpg](../icml_results/2916_GraphFlow_%20A%20Graph-Based%20Workflow%20Management%20for%20Efficient%20LLM-Agent%20Serving/images/3f37d2568fa01951a5dace3c9a3533c9b0115b4f8c6c8e2815231b8a046871d3.jpg)

![468b90581f7c1d7c993d3a46e17126c6e9e07a2ff38444ee559ee874cb348e40.jpg](../icml_results/2916_GraphFlow_%20A%20Graph-Based%20Workflow%20Management%20for%20Efficient%20LLM-Agent%20Serving/images/468b90581f7c1d7c993d3a46e17126c6e9e07a2ff38444ee559ee874cb348e40.jpg)

![6d1ac3ea7ffcdb27fbe0945c4efac0dff0882a28d91dcac5ab40c3d264777f11.jpg](../icml_results/2916_GraphFlow_%20A%20Graph-Based%20Workflow%20Management%20for%20Efficient%20LLM-Agent%20Serving/images/6d1ac3ea7ffcdb27fbe0945c4efac0dff0882a28d91dcac5ab40c3d264777f11.jpg)

![96acc2e984d894f15307a6e0b9cf818855297b5e8c4f9a9a93e1342e90c1dbfc.jpg](../icml_results/2916_GraphFlow_%20A%20Graph-Based%20Workflow%20Management%20for%20Efficient%20LLM-Agent%20Serving/images/96acc2e984d894f15307a6e0b9cf818855297b5e8c4f9a9a93e1342e90c1dbfc.jpg)

![a97b82fdc781d90ffd16ca30c86aafb6e9552c87064fff9ade373faaf1c0aa74.jpg](../icml_results/2916_GraphFlow_%20A%20Graph-Based%20Workflow%20Management%20for%20Efficient%20LLM-Agent%20Serving/images/a97b82fdc781d90ffd16ca30c86aafb6e9552c87064fff9ade373faaf1c0aa74.jpg)

![b2f73fb42e591979190b8e0800c0c8568c7793be931365ec32ad89c1aff6f92a.jpg](../icml_results/2916_GraphFlow_%20A%20Graph-Based%20Workflow%20Management%20for%20Efficient%20LLM-Agent%20Serving/images/b2f73fb42e591979190b8e0800c0c8568c7793be931365ec32ad89c1aff6f92a.jpg)

![ea8558e4b3c3f518d7d5ce9077fa65cc6bd885a62dd80255a49bd3762abd8b42.jpg](../icml_results/2916_GraphFlow_%20A%20Graph-Based%20Workflow%20Management%20for%20Efficient%20LLM-Agent%20Serving/images/ea8558e4b3c3f518d7d5ce9077fa65cc6bd885a62dd80255a49bd3762abd8b42.jpg)

![f780541ebb8e848ee6e2dba76cc52f1ceb3e57989159f453da78ed0c27efe507.jpg](../icml_results/2916_GraphFlow_%20A%20Graph-Based%20Workflow%20Management%20for%20Efficient%20LLM-Agent%20Serving/images/f780541ebb8e848ee6e2dba76cc52f1ceb3e57989159f453da78ed0c27efe507.jpg)

### Tables

![55011277627839aae28d40c608c0baa426b14852acf07f642d8d3ee47e58fa0f.jpg](../icml_results/2916_GraphFlow_%20A%20Graph-Based%20Workflow%20Management%20for%20Efficient%20LLM-Agent%20Serving/tables/55011277627839aae28d40c608c0baa426b14852acf07f642d8d3ee47e58fa0f.jpg)

![cddc3bbaf93a2c6b58fe8ec85f0de7ca77b62b5800fa2755408bf69bff906cbe.jpg](../icml_results/2916_GraphFlow_%20A%20Graph-Based%20Workflow%20Management%20for%20Efficient%20LLM-Agent%20Serving/tables/cddc3bbaf93a2c6b58fe8ec85f0de7ca77b62b5800fa2755408bf69bff906cbe.jpg)

<a id="paper-2879"></a>

## Being More Lightweight and Practical: Mini-sized Contrastive Learning Pre-trained Models for Fine-grained Traffic Task


### Images

![0495cb1cae42e673a71f37d3732285fb05cf280942e68e455b79c344629a9da2.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/images/0495cb1cae42e673a71f37d3732285fb05cf280942e68e455b79c344629a9da2.jpg)

![0e64bd2335cb5942563f274e8d3b7a967e33425ad9844584ae5f10980fb9d77e.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/images/0e64bd2335cb5942563f274e8d3b7a967e33425ad9844584ae5f10980fb9d77e.jpg)

![10a4a2886e6d4701484edb551e2ba1e9f182987797a47e7f4f2b63367d7695c0.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/images/10a4a2886e6d4701484edb551e2ba1e9f182987797a47e7f4f2b63367d7695c0.jpg)

![29bf53b82eade53fe502ad0987a60a7252ab087ef8bf5c9bd7853e98c42f8117.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/images/29bf53b82eade53fe502ad0987a60a7252ab087ef8bf5c9bd7853e98c42f8117.jpg)

![35a6dba1c2f5faeaa7994ad66291970cc87dbb55fc5276baca4baab6024d2e63.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/images/35a6dba1c2f5faeaa7994ad66291970cc87dbb55fc5276baca4baab6024d2e63.jpg)

![4d96ab8a806fdba311eaf4edd3dc95d32040ba5ea17d254c62ccf869ab9c8049.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/images/4d96ab8a806fdba311eaf4edd3dc95d32040ba5ea17d254c62ccf869ab9c8049.jpg)

![4ebd98de6a0283b920c20df49f247608b9b372d648c296cc6c3e82a47d7772c1.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/images/4ebd98de6a0283b920c20df49f247608b9b372d648c296cc6c3e82a47d7772c1.jpg)

![6e8953f3d5e57d0516c9ed664783b9494a7608e3920b3a6b863bd981c9bcb135.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/images/6e8953f3d5e57d0516c9ed664783b9494a7608e3920b3a6b863bd981c9bcb135.jpg)

![73c25539a307cc7d141903bf08fe22e364a6cbd48ed1890a79722ee6aa960893.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/images/73c25539a307cc7d141903bf08fe22e364a6cbd48ed1890a79722ee6aa960893.jpg)

![9dd23a27c2f0aca38e2286cde5f30b88e58ce8648c7cac8244c1d9054430f947.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/images/9dd23a27c2f0aca38e2286cde5f30b88e58ce8648c7cac8244c1d9054430f947.jpg)

![a28433b1445afc6f957f9d09ce4737185e71b0d248607b1c7754eaaca1df0045.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/images/a28433b1445afc6f957f9d09ce4737185e71b0d248607b1c7754eaaca1df0045.jpg)

![b0905f1f696c97953601888a2a9ef39c10ce08640c6c1c4d1e786c3c20b2e7de.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/images/b0905f1f696c97953601888a2a9ef39c10ce08640c6c1c4d1e786c3c20b2e7de.jpg)

![c54719be5a053f69b50dc636fa0304b5339b25f6d938d5adfea69a4d4067c4a1.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/images/c54719be5a053f69b50dc636fa0304b5339b25f6d938d5adfea69a4d4067c4a1.jpg)

![eea967430f13edb0f6c0ff066671638d734275703196c487881e1edb3c4729fa.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/images/eea967430f13edb0f6c0ff066671638d734275703196c487881e1edb3c4729fa.jpg)

### Tables

![02fe130584d7f65fd87697bf16690d9e16ea46be98215057fdbe2e52d160b1bf.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/tables/02fe130584d7f65fd87697bf16690d9e16ea46be98215057fdbe2e52d160b1bf.jpg)

![0eeb45c20bb63c4aaa6f8b0143b86a007a8a729c48d22b5bbac5d55b62a5fc7b.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/tables/0eeb45c20bb63c4aaa6f8b0143b86a007a8a729c48d22b5bbac5d55b62a5fc7b.jpg)

![2bcf3071f6eeb1c6d2a43c8748954fc1f2fb262fa02c8b9bc57b79d98f8218ee.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/tables/2bcf3071f6eeb1c6d2a43c8748954fc1f2fb262fa02c8b9bc57b79d98f8218ee.jpg)

![302584ca68c7ab79831cc1b53b96b02fa396295a2db6a28d46c41053d52d9b6f.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/tables/302584ca68c7ab79831cc1b53b96b02fa396295a2db6a28d46c41053d52d9b6f.jpg)

![42c2be602512d546eb8ba00da3e423941b4fb7120c5eddd4a66888486f73a6d6.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/tables/42c2be602512d546eb8ba00da3e423941b4fb7120c5eddd4a66888486f73a6d6.jpg)

![4c9db1b3128d2eb165cb9fd6e74613ca6d5ba6f019bb12ab54036d8b47ef7e29.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/tables/4c9db1b3128d2eb165cb9fd6e74613ca6d5ba6f019bb12ab54036d8b47ef7e29.jpg)

![71540ad565c2559ec738b8e47be90b9d18a94f632bee74f8657459d013d23df5.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/tables/71540ad565c2559ec738b8e47be90b9d18a94f632bee74f8657459d013d23df5.jpg)

![7375ada1af56aec68a6b1c043ecaf9b419b56b06c7c34d3f265a6a7442fddaa9.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/tables/7375ada1af56aec68a6b1c043ecaf9b419b56b06c7c34d3f265a6a7442fddaa9.jpg)

![96984719bfb3b874ffddcbbbc00d54d06f87f165ca34869743d6aac8e9abd32b.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/tables/96984719bfb3b874ffddcbbbc00d54d06f87f165ca34869743d6aac8e9abd32b.jpg)

![a5a65480aa68710581d5d7ff4271a15e3fe57784e6fb77d6c7c68a8e1ec357cb.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/tables/a5a65480aa68710581d5d7ff4271a15e3fe57784e6fb77d6c7c68a8e1ec357cb.jpg)

![b0ae6bd0441537f0b3d4ef019e01ede8020ebcc8b7ce0c5acac170a4c024ead0.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/tables/b0ae6bd0441537f0b3d4ef019e01ede8020ebcc8b7ce0c5acac170a4c024ead0.jpg)

![b4eea5ee40cbff0c593ee19974028833a1fd60a6188129628c44b89806965f20.jpg](../icml_results/2917_Being%20More%20Lightweight%20and%20Practical_%20Mini-sized%20Contrastive%20Learning%20Pre-trained%20Models%20for%20Fine-gr/tables/b4eea5ee40cbff0c593ee19974028833a1fd60a6188129628c44b89806965f20.jpg)

<a id="paper-2880"></a>

## Reliable Thinking with Images


### Images

![0fcc73017df12fafc363b005b455a3b7287d0ca398a6e108770194f96fec710f.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/0fcc73017df12fafc363b005b455a3b7287d0ca398a6e108770194f96fec710f.jpg)

![25110b5eeb8ca78e3255dbd6feb040a30afdb6fea27097f22347c758ead662d7.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/25110b5eeb8ca78e3255dbd6feb040a30afdb6fea27097f22347c758ead662d7.jpg)

![26e3ad4e9915cf450ca88a64840a6775d8cc976ec72a015d48fec2271bedac07.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/26e3ad4e9915cf450ca88a64840a6775d8cc976ec72a015d48fec2271bedac07.jpg)

![2b7f869501ed46a884845327c4414ad7c99428ec3e90bc8f9d40484185a660a2.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/2b7f869501ed46a884845327c4414ad7c99428ec3e90bc8f9d40484185a660a2.jpg)

![3eb245ef30d113a119e72136ae35ae724a6e5b1ccae6da14eb0ec750a9cc016a.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/3eb245ef30d113a119e72136ae35ae724a6e5b1ccae6da14eb0ec750a9cc016a.jpg)

![48d6509fd7cf696f2e9aff36d1c126934cbc38250a6d165e4c96e47cf8b8ecfd.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/48d6509fd7cf696f2e9aff36d1c126934cbc38250a6d165e4c96e47cf8b8ecfd.jpg)

![4c143b8c4daad9cfbdfd811b07ffad6e09d2f44f4a545fead2f66a22581b8e00.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/4c143b8c4daad9cfbdfd811b07ffad6e09d2f44f4a545fead2f66a22581b8e00.jpg)

![5f04051231e4a660aa5e0660441b688b87dce310c5f3f4892cd2970ee5a314af.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/5f04051231e4a660aa5e0660441b688b87dce310c5f3f4892cd2970ee5a314af.jpg)

![6392a95dc5bdbc1a0f1bc148e41d12f2b5f53cbfc330b58e8ea862b07ac5f966.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/6392a95dc5bdbc1a0f1bc148e41d12f2b5f53cbfc330b58e8ea862b07ac5f966.jpg)

![6e5586161beae5436d0ca919325defecadf06d3f6ec489bb65ca721c4e9b5845.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/6e5586161beae5436d0ca919325defecadf06d3f6ec489bb65ca721c4e9b5845.jpg)

![7250284b1fa06c1ff6e7db05f62de560ec38107d054824d81c034d32527c1fbe.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/7250284b1fa06c1ff6e7db05f62de560ec38107d054824d81c034d32527c1fbe.jpg)

![743f21926630f26623a5973921ba0861eebf5f253e1d9a6fcd41f6315216355e.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/743f21926630f26623a5973921ba0861eebf5f253e1d9a6fcd41f6315216355e.jpg)

![782c6988d3019388a5d5c93a5b96cfeb7703fc0bb2ed143ad7102ba7f55b3a25.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/782c6988d3019388a5d5c93a5b96cfeb7703fc0bb2ed143ad7102ba7f55b3a25.jpg)

![7bed3124961879a3db3719af3396b80f90cd8cc349abff92407f7f4364d3c4d7.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/7bed3124961879a3db3719af3396b80f90cd8cc349abff92407f7f4364d3c4d7.jpg)

![7e38af95d661f8dda9cd817c41cc99cf84f57309d64377c3f1ac698ee9dbe948.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/7e38af95d661f8dda9cd817c41cc99cf84f57309d64377c3f1ac698ee9dbe948.jpg)

![93ec02c0e49184cb28c1dd19016d28a9a57dcb33542da5c0b262731b7b46f280.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/93ec02c0e49184cb28c1dd19016d28a9a57dcb33542da5c0b262731b7b46f280.jpg)

![9ddac7dfa95923a4e6d681b359b6eb977ea7311d55243860829b01901bc13332.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/9ddac7dfa95923a4e6d681b359b6eb977ea7311d55243860829b01901bc13332.jpg)

![ae3b4c2b4e87d7187a8271059e9a9bded2f47b24991cf8b0932840b5076c1957.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/ae3b4c2b4e87d7187a8271059e9a9bded2f47b24991cf8b0932840b5076c1957.jpg)

![b052542bba382de1de71e31343fe735fa699333f20a8c7119a7e4946eb23db7a.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/b052542bba382de1de71e31343fe735fa699333f20a8c7119a7e4946eb23db7a.jpg)

![b231720b27d08fd3c7d0b5d962131d821ea163ff88f01d398dfb561e958ed6a8.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/b231720b27d08fd3c7d0b5d962131d821ea163ff88f01d398dfb561e958ed6a8.jpg)

![bbdf2473c6932f264dfbb69fcc7ec76da4b7cdeea6dab06a9f01dd2a24c87b89.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/bbdf2473c6932f264dfbb69fcc7ec76da4b7cdeea6dab06a9f01dd2a24c87b89.jpg)

![c05fae654bc02b99ab4aef7764fcf14b12ace6283709a28352d28391c57a89c4.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/c05fae654bc02b99ab4aef7764fcf14b12ace6283709a28352d28391c57a89c4.jpg)

![ca4ee8f7de3c96e642b28393be75d2c2af22f559c4d34378d437787677003991.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/ca4ee8f7de3c96e642b28393be75d2c2af22f559c4d34378d437787677003991.jpg)

![cc1937eff4596ea7aab9d1dfc847506bd177faa8aa27dd2cf49713bc6fff461d.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/cc1937eff4596ea7aab9d1dfc847506bd177faa8aa27dd2cf49713bc6fff461d.jpg)

![de5e49fadf7b1ac5d6a9445027c542a899186b526e92fbbd954d6f1525ccf8fc.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/de5e49fadf7b1ac5d6a9445027c542a899186b526e92fbbd954d6f1525ccf8fc.jpg)

![e1f61429bfe8e3463698cebb76c6ec903dd26fcd236efce67e06d8bd7a7adfb1.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/e1f61429bfe8e3463698cebb76c6ec903dd26fcd236efce67e06d8bd7a7adfb1.jpg)

![fcdd7f509e6709de7ac2d70b3e53dd430ea7a7ed4cd91c79a06995f58198be5a.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/images/fcdd7f509e6709de7ac2d70b3e53dd430ea7a7ed4cd91c79a06995f58198be5a.jpg)

### Tables

![55d760950a3173463fc5ab9460cd1d8d65d089d4f2a499e08e9f5e60d3c5725b.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/tables/55d760950a3173463fc5ab9460cd1d8d65d089d4f2a499e08e9f5e60d3c5725b.jpg)

![767209ce313072b410773d944858fba3f6090051628ba3cf7ec903f78d2fd3da.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/tables/767209ce313072b410773d944858fba3f6090051628ba3cf7ec903f78d2fd3da.jpg)

![7a445817a7347e49420a0985636a733bb95333f55b60930aa10e420e651689b4.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/tables/7a445817a7347e49420a0985636a733bb95333f55b60930aa10e420e651689b4.jpg)

![80fbbbae4348487ad5052263bf2a8b335e51ba09f687ff9f34b7e6d6bcd3700a.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/tables/80fbbbae4348487ad5052263bf2a8b335e51ba09f687ff9f34b7e6d6bcd3700a.jpg)

![9bf6dd0d89a71d741ec2e1bc56ff1bbda200bfb15129a0a23f7b69af4981cfef.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/tables/9bf6dd0d89a71d741ec2e1bc56ff1bbda200bfb15129a0a23f7b69af4981cfef.jpg)

![a2580ca9b9eb2f02bdf8e52f3e54f039b2c8ad01ad0e7000d17b99301f0019ab.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/tables/a2580ca9b9eb2f02bdf8e52f3e54f039b2c8ad01ad0e7000d17b99301f0019ab.jpg)

![a7e353ecfbfec81bb109e76c29e52359aefa703fea8328d3cff942fc98d95280.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/tables/a7e353ecfbfec81bb109e76c29e52359aefa703fea8328d3cff942fc98d95280.jpg)

![aef69b70e7d6f73fba461a80191588d32c3dc7ad3ee0e272ce1925c83e907776.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/tables/aef69b70e7d6f73fba461a80191588d32c3dc7ad3ee0e272ce1925c83e907776.jpg)

![c5087673ef117c29887fe9f99199b8f5f5ef6f829e7e074c29f7269317fa245c.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/tables/c5087673ef117c29887fe9f99199b8f5f5ef6f829e7e074c29f7269317fa245c.jpg)

![e5475adc2da6fec46787a4814c9ed5abdd48ec5c5fbc7106f6f92107a203d9d6.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/tables/e5475adc2da6fec46787a4814c9ed5abdd48ec5c5fbc7106f6f92107a203d9d6.jpg)

![e650ee8e2570149bbcbf97f23704d3b569dcb366bbb2e5a2cf147a865d9ce2d5.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/tables/e650ee8e2570149bbcbf97f23704d3b569dcb366bbb2e5a2cf147a865d9ce2d5.jpg)

![e974157b735bf11ddb59e403e4a4830adc32b71fa93c7f554667304e715b196b.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/tables/e974157b735bf11ddb59e403e4a4830adc32b71fa93c7f554667304e715b196b.jpg)

![ea9ebb22feaffd07a59122673ab4038c61f43c31277442fb0d7a2c7bc188064d.jpg](../icml_results/2918_Reliable%20Thinking%20with%20Images/tables/ea9ebb22feaffd07a59122673ab4038c61f43c31277442fb0d7a2c7bc188064d.jpg)

<a id="paper-2881"></a>

## ComplexMCP: Evaluation of LLM Agents in Dynamic, Interdependent, and Large-Scale Tool Sandbox


### Images

![49ed8c71c22d4f1030b7c575b1a963f00440e53783a0ea82f78f8381ad596f30.jpg](../icml_results/2919_ComplexMCP_%20Evaluation%20of%20LLM%20Agents%20in%20Dynamic%2C%20Interdependent%2C%20and%20Large-Scale%20Tool%20Sandbox/images/49ed8c71c22d4f1030b7c575b1a963f00440e53783a0ea82f78f8381ad596f30.jpg)

![539eff4369653dea6e896c14f29f37ad03ad4b11e698451e57f3a4a297961555.jpg](../icml_results/2919_ComplexMCP_%20Evaluation%20of%20LLM%20Agents%20in%20Dynamic%2C%20Interdependent%2C%20and%20Large-Scale%20Tool%20Sandbox/images/539eff4369653dea6e896c14f29f37ad03ad4b11e698451e57f3a4a297961555.jpg)

![5c63081d4bf6d1d862f6cefd46bca395a2ac0d6d925f8030a814ac9d462d1139.jpg](../icml_results/2919_ComplexMCP_%20Evaluation%20of%20LLM%20Agents%20in%20Dynamic%2C%20Interdependent%2C%20and%20Large-Scale%20Tool%20Sandbox/images/5c63081d4bf6d1d862f6cefd46bca395a2ac0d6d925f8030a814ac9d462d1139.jpg)

![5f0ca32be8519e640a21292478e3ad33a9f6e36deddd0e27a7e436f95b58b5d7.jpg](../icml_results/2919_ComplexMCP_%20Evaluation%20of%20LLM%20Agents%20in%20Dynamic%2C%20Interdependent%2C%20and%20Large-Scale%20Tool%20Sandbox/images/5f0ca32be8519e640a21292478e3ad33a9f6e36deddd0e27a7e436f95b58b5d7.jpg)

![7c69fb45f79b328efcbc86efa375724875f41247329526999805635463cd8ab5.jpg](../icml_results/2919_ComplexMCP_%20Evaluation%20of%20LLM%20Agents%20in%20Dynamic%2C%20Interdependent%2C%20and%20Large-Scale%20Tool%20Sandbox/images/7c69fb45f79b328efcbc86efa375724875f41247329526999805635463cd8ab5.jpg)

![7cc935c5b3858284edb78300bb6311f0cce64eb1e4ac345ee1be95a710e2b47c.jpg](../icml_results/2919_ComplexMCP_%20Evaluation%20of%20LLM%20Agents%20in%20Dynamic%2C%20Interdependent%2C%20and%20Large-Scale%20Tool%20Sandbox/images/7cc935c5b3858284edb78300bb6311f0cce64eb1e4ac345ee1be95a710e2b47c.jpg)

![c320ede7c6d73754a5d43338e1744e1fcba26bb028d8df7288c3efe73ee300a0.jpg](../icml_results/2919_ComplexMCP_%20Evaluation%20of%20LLM%20Agents%20in%20Dynamic%2C%20Interdependent%2C%20and%20Large-Scale%20Tool%20Sandbox/images/c320ede7c6d73754a5d43338e1744e1fcba26bb028d8df7288c3efe73ee300a0.jpg)

### Tables

![2f309bb11c964f57850d0d981b4064351f45cdd71b01b75e3180c180e8fddee3.jpg](../icml_results/2919_ComplexMCP_%20Evaluation%20of%20LLM%20Agents%20in%20Dynamic%2C%20Interdependent%2C%20and%20Large-Scale%20Tool%20Sandbox/tables/2f309bb11c964f57850d0d981b4064351f45cdd71b01b75e3180c180e8fddee3.jpg)

![4dbf976efa663bda404f58d63e35e995d073bde559d009217817c70ca2f4185a.jpg](../icml_results/2919_ComplexMCP_%20Evaluation%20of%20LLM%20Agents%20in%20Dynamic%2C%20Interdependent%2C%20and%20Large-Scale%20Tool%20Sandbox/tables/4dbf976efa663bda404f58d63e35e995d073bde559d009217817c70ca2f4185a.jpg)

![9784de2f1f3de30be6ff7ef9624b71b17e18f44f69510cbb6cfa63f6dda5988f.jpg](../icml_results/2919_ComplexMCP_%20Evaluation%20of%20LLM%20Agents%20in%20Dynamic%2C%20Interdependent%2C%20and%20Large-Scale%20Tool%20Sandbox/tables/9784de2f1f3de30be6ff7ef9624b71b17e18f44f69510cbb6cfa63f6dda5988f.jpg)

![b4af1541e2722c896542eab4ae9c02de217d8f9c08c5875ebc11fb6d5a4502e2.jpg](../icml_results/2919_ComplexMCP_%20Evaluation%20of%20LLM%20Agents%20in%20Dynamic%2C%20Interdependent%2C%20and%20Large-Scale%20Tool%20Sandbox/tables/b4af1541e2722c896542eab4ae9c02de217d8f9c08c5875ebc11fb6d5a4502e2.jpg)

![e1be8b2f45162250effbf6d852e60b71bfb2b0463b96aab0b7744d3ecd03f257.jpg](../icml_results/2919_ComplexMCP_%20Evaluation%20of%20LLM%20Agents%20in%20Dynamic%2C%20Interdependent%2C%20and%20Large-Scale%20Tool%20Sandbox/tables/e1be8b2f45162250effbf6d852e60b71bfb2b0463b96aab0b7744d3ecd03f257.jpg)

<a id="paper-2882"></a>

## Breaking the Block: Preserving Data Continuity to Train Superior SAEs for Instruct Models


### Images

![118abcf00c76be5c6873f563a310957e7da8989b0b26e3d2253055854a75b403.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/images/118abcf00c76be5c6873f563a310957e7da8989b0b26e3d2253055854a75b403.jpg)

![208fd4f6c6edbc63177d6d438a95f51d1abef2ed81f25cb9c33864a938e143bc.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/images/208fd4f6c6edbc63177d6d438a95f51d1abef2ed81f25cb9c33864a938e143bc.jpg)

![251674d7d21ec222150c7b044a8a8d79091b7e5aa965b4545faa328e9fe4163b.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/images/251674d7d21ec222150c7b044a8a8d79091b7e5aa965b4545faa328e9fe4163b.jpg)

![2bc7bed2fd12567fc96ad83480e03278feccf6766d1804467776b861b9254da7.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/images/2bc7bed2fd12567fc96ad83480e03278feccf6766d1804467776b861b9254da7.jpg)

![3e58ebf24498cd8ee6c4dc9612c92deaf874c591d9d22d08b0c7a7deb2cd5bd6.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/images/3e58ebf24498cd8ee6c4dc9612c92deaf874c591d9d22d08b0c7a7deb2cd5bd6.jpg)

![5132593d0f4029d482c96c3ef5805b11f88cbb38ae31b26af95e4d367b7568e8.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/images/5132593d0f4029d482c96c3ef5805b11f88cbb38ae31b26af95e4d367b7568e8.jpg)

![535df7445e3e97249aa64e30ab28766ac4d180cbe63f6fe5c2c7a67ade86dd10.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/images/535df7445e3e97249aa64e30ab28766ac4d180cbe63f6fe5c2c7a67ade86dd10.jpg)

![5bfafbb91fb34a3df93ab838736dbbc329984faa3198bd0e8efc7abb207f80f3.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/images/5bfafbb91fb34a3df93ab838736dbbc329984faa3198bd0e8efc7abb207f80f3.jpg)

![95e76dd54270ec7aaba1571732a746a38ee85f7d6511cc29be4fafd61046874c.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/images/95e76dd54270ec7aaba1571732a746a38ee85f7d6511cc29be4fafd61046874c.jpg)

![ac927b42a8b70070cae09e748de2bd59f5d30ccb90a37c34dffcb0270ca0f10c.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/images/ac927b42a8b70070cae09e748de2bd59f5d30ccb90a37c34dffcb0270ca0f10c.jpg)

![f2b32117a2476367758007f4f3f2bad07bc1412081c22fa359ce2133241ac846.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/images/f2b32117a2476367758007f4f3f2bad07bc1412081c22fa359ce2133241ac846.jpg)

### Tables

![04fa16e9a62b13272ecb048b20d3fd39a60344257916821bc2fb3fbb98d527ef.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/04fa16e9a62b13272ecb048b20d3fd39a60344257916821bc2fb3fbb98d527ef.jpg)

![05f59440ceb8d43cb60254506890afb80a19611f53ff4d9ba653ab70e4b46fdb.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/05f59440ceb8d43cb60254506890afb80a19611f53ff4d9ba653ab70e4b46fdb.jpg)

![0977372015b741b2242c3c8f107138de13b50f41264971d02b6ffd06a3bc10b5.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/0977372015b741b2242c3c8f107138de13b50f41264971d02b6ffd06a3bc10b5.jpg)

![097fa90361ec9c1cac1a34620d103ab2549d8d201ac186cf262fa3bfc14e7133.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/097fa90361ec9c1cac1a34620d103ab2549d8d201ac186cf262fa3bfc14e7133.jpg)

![12bc04231e307fed2dc0e6c2ee3b56d5161518df56000d9b1698d8d099467ce2.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/12bc04231e307fed2dc0e6c2ee3b56d5161518df56000d9b1698d8d099467ce2.jpg)

![1a74d49b4aa3150e66fcccab9796ae36b2d5c74e776939175a4c027ef3700df4.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/1a74d49b4aa3150e66fcccab9796ae36b2d5c74e776939175a4c027ef3700df4.jpg)

![2afc176b8bd0b0df9892ea9e16d777d472c9a3d9830cf5fba43cc4ae6066379b.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/2afc176b8bd0b0df9892ea9e16d777d472c9a3d9830cf5fba43cc4ae6066379b.jpg)

![31dafc85257938a9e802ec4e3b9741d0476eefa518fd7b5ccd32e0f692afda68.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/31dafc85257938a9e802ec4e3b9741d0476eefa518fd7b5ccd32e0f692afda68.jpg)

![34932231cdd80a1a31d3523bfc4ab045b2b2c5194abeb84955d9a50fb5cb4306.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/34932231cdd80a1a31d3523bfc4ab045b2b2c5194abeb84955d9a50fb5cb4306.jpg)

![36230ae1b4ce1ef7efad162bd5b31c969fc3bce179958d2f715165676f4dab49.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/36230ae1b4ce1ef7efad162bd5b31c969fc3bce179958d2f715165676f4dab49.jpg)

![3f2ed6479a7a0009d4f3e102faad64d8d46929139e936e7ad2558a4a2ba6a90c.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/3f2ed6479a7a0009d4f3e102faad64d8d46929139e936e7ad2558a4a2ba6a90c.jpg)

![49201d980b1f5e2a64ad730dc988d2aa7bd8de71c7246d72f4c5a1ba3e5a27d2.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/49201d980b1f5e2a64ad730dc988d2aa7bd8de71c7246d72f4c5a1ba3e5a27d2.jpg)

![57964cfac7c60377399c0083cc652fef2752db86da96871c1023b7ea9f03719f.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/57964cfac7c60377399c0083cc652fef2752db86da96871c1023b7ea9f03719f.jpg)

![59b52292e754c8380528793fb2e011d1295404365e7cda6d345aa0064386bd9b.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/59b52292e754c8380528793fb2e011d1295404365e7cda6d345aa0064386bd9b.jpg)

![5bbd7018e4384035cfe2397951a6a60b5adc39c93c50e66fdaefcfc8d0aa3cf5.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/5bbd7018e4384035cfe2397951a6a60b5adc39c93c50e66fdaefcfc8d0aa3cf5.jpg)

![5f004230c32e4c55a6c56f9600035bc1d2a8028987e687fbc7a0bd47593a796b.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/5f004230c32e4c55a6c56f9600035bc1d2a8028987e687fbc7a0bd47593a796b.jpg)

![61ea3aab964a94fecad4943b6bd7540f979d202555e37d9ef5c16d972a02b186.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/61ea3aab964a94fecad4943b6bd7540f979d202555e37d9ef5c16d972a02b186.jpg)

![7450e05abfc1a2a8c01e86752cc1d3e03258e57a021918c31cb5ba7474f23e75.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/7450e05abfc1a2a8c01e86752cc1d3e03258e57a021918c31cb5ba7474f23e75.jpg)

![89d6f09b28465c1d7d65b3d4da275401accad5d40570d32b7dd86e038f9a63e6.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/89d6f09b28465c1d7d65b3d4da275401accad5d40570d32b7dd86e038f9a63e6.jpg)

![8c5333d4aa60a2068971e8ff85a9ac209ae3b94ba7563080320adda43f391ba0.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/8c5333d4aa60a2068971e8ff85a9ac209ae3b94ba7563080320adda43f391ba0.jpg)

![93d3a13d29a5d56ceb149ec8468bf2b8840079ccd11e298183f3c20a6a38f027.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/93d3a13d29a5d56ceb149ec8468bf2b8840079ccd11e298183f3c20a6a38f027.jpg)

![a3fdf6b080d3ade03380522332da985cffd62f7e23974f0ea9741828e6b4a9fd.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/a3fdf6b080d3ade03380522332da985cffd62f7e23974f0ea9741828e6b4a9fd.jpg)

![bb63f377815990dd4b3864709ef3ae35b0461ab10ff65097125039590609967e.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/bb63f377815990dd4b3864709ef3ae35b0461ab10ff65097125039590609967e.jpg)

![c13ddb3ac9b1754ced8fb81a07ffef3737c99763d2d21d65219c108f33ef675a.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/c13ddb3ac9b1754ced8fb81a07ffef3737c99763d2d21d65219c108f33ef675a.jpg)

![c4fbf8aae046f90d1cd0f32a813a8485a7d4a1ab202e04840b8c44b4f2bd33dc.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/c4fbf8aae046f90d1cd0f32a813a8485a7d4a1ab202e04840b8c44b4f2bd33dc.jpg)

![d278adc744d555d44591e9bf4020f5ddc94b605e124a9479ecc0ac374719e18c.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/d278adc744d555d44591e9bf4020f5ddc94b605e124a9479ecc0ac374719e18c.jpg)

![d85676423f8eb67020953cfe6fcfb9027d9d9e33f278a26326eacd9a2a23ac56.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/d85676423f8eb67020953cfe6fcfb9027d9d9e33f278a26326eacd9a2a23ac56.jpg)

![e13c29f3f9d81e0e11e06db59460aaa936afbc472e3e8472fa4446dd1a6acfc1.jpg](../icml_results/2920_Breaking%20the%20Block_%20Preserving%20Data%20Continuity%20to%20Train%20Superior%20SAEs%20for%20Instruct%20Models/tables/e13c29f3f9d81e0e11e06db59460aaa936afbc472e3e8472fa4446dd1a6acfc1.jpg)

<a id="paper-2883"></a>

## ALSO: Adversarial Online Strategy Optimization for Social Agents


### Images

![32b4714b29f38f13727eb8b871cfe7aa0f055ff98ba6f5394fcbfa02a46c29cd.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/images/32b4714b29f38f13727eb8b871cfe7aa0f055ff98ba6f5394fcbfa02a46c29cd.jpg)

![7203cfcc553540a83d3ba7d1ad7101e3f44a800a32d44f3ed8f888a45f628e2a.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/images/7203cfcc553540a83d3ba7d1ad7101e3f44a800a32d44f3ed8f888a45f628e2a.jpg)

![7a3d746e0f57f4a9c9e905b6b37fe7761528de74eb4921e246e3e099009f4a82.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/images/7a3d746e0f57f4a9c9e905b6b37fe7761528de74eb4921e246e3e099009f4a82.jpg)

![906509bf5e0a9b1750c0bef6a13dfc3d4db96a703d621b12bc246dc2ed125987.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/images/906509bf5e0a9b1750c0bef6a13dfc3d4db96a703d621b12bc246dc2ed125987.jpg)

![979f9e25ce446e29e978cd32fe306b076d72d8b528929f8de5ced2ab0f836510.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/images/979f9e25ce446e29e978cd32fe306b076d72d8b528929f8de5ced2ab0f836510.jpg)

![a430d4582bad0192af94e773adf41f916062b0f04278510740289ba8ae0520a0.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/images/a430d4582bad0192af94e773adf41f916062b0f04278510740289ba8ae0520a0.jpg)

![b7a15b1cfd2eefea5d0c878295580074f23dee6bd4c9f766fbc921e0660faafa.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/images/b7a15b1cfd2eefea5d0c878295580074f23dee6bd4c9f766fbc921e0660faafa.jpg)

![ddd0907885512cd109c366f29d772c9968de1042b1f36eb8061ee591d4325a2d.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/images/ddd0907885512cd109c366f29d772c9968de1042b1f36eb8061ee591d4325a2d.jpg)

![fe0a2c93c43a3d1a4fec719d7eff3ddf3f149c4c4cf7372f5241fa6043081d9d.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/images/fe0a2c93c43a3d1a4fec719d7eff3ddf3f149c4c4cf7372f5241fa6043081d9d.jpg)

### Tables

![0538fd25823f38ba0a12f172ae6507d19ab6b43900e3ee6f2d79a7b5addec682.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/0538fd25823f38ba0a12f172ae6507d19ab6b43900e3ee6f2d79a7b5addec682.jpg)

![29a08f8a594a705c25a8452970d256a52892e3a760c5a6d2b198d391357556f8.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/29a08f8a594a705c25a8452970d256a52892e3a760c5a6d2b198d391357556f8.jpg)

![332dd42b904a27fb040fe9910e35d02fa4123296b188d72a5cd7d38a876b0bac.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/332dd42b904a27fb040fe9910e35d02fa4123296b188d72a5cd7d38a876b0bac.jpg)

![5b3bffe7382a061e09a0304d0c4654056701a0a54f811a90bf0c861821aae591.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/5b3bffe7382a061e09a0304d0c4654056701a0a54f811a90bf0c861821aae591.jpg)

![5e8ed257fa136b146bd2a6288f1a1826702edaa6c8adf6e4ca15fa7fe757330c.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/5e8ed257fa136b146bd2a6288f1a1826702edaa6c8adf6e4ca15fa7fe757330c.jpg)

![71cebe05c81200ddfc0cf64ca323d7684ccc23831916da5ce2c99bf7e2de0eae.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/71cebe05c81200ddfc0cf64ca323d7684ccc23831916da5ce2c99bf7e2de0eae.jpg)

![720397ee102d6dcd97deb79c74f5dadf994cacd1092f557892e2d12c2d156810.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/720397ee102d6dcd97deb79c74f5dadf994cacd1092f557892e2d12c2d156810.jpg)

![83d8bf9c2a841e6df6f393198a0838e0e3566cca3085834cf08c26f3c74dbe90.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/83d8bf9c2a841e6df6f393198a0838e0e3566cca3085834cf08c26f3c74dbe90.jpg)

![888f8b0e12c740448e91db9b82a2fa86295f15bec2b58439709a13f6b2b7ed4c.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/888f8b0e12c740448e91db9b82a2fa86295f15bec2b58439709a13f6b2b7ed4c.jpg)

![8e2002acccf6119c37c8fb823f7abee905ede0ad8fb28c7ba4377f16828fd843.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/8e2002acccf6119c37c8fb823f7abee905ede0ad8fb28c7ba4377f16828fd843.jpg)

![96206b7b14e0193afba626b6b63dea2a5c5855424e2d16f3152ef21698c62dc3.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/96206b7b14e0193afba626b6b63dea2a5c5855424e2d16f3152ef21698c62dc3.jpg)

![a2b87634cc24ac980951b7beb00f17678b81d67c53279cbd9c0ab4c1eacf81f3.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/a2b87634cc24ac980951b7beb00f17678b81d67c53279cbd9c0ab4c1eacf81f3.jpg)

![b35b6a6d50940632340b8553252b6da47270897409ac1383d209f1d4d73c8035.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/b35b6a6d50940632340b8553252b6da47270897409ac1383d209f1d4d73c8035.jpg)

![b4ee8d26929f6aced8fb7578fd60ffb7dbbc8b2dff4968225eadcf0e4d4ac5be.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/b4ee8d26929f6aced8fb7578fd60ffb7dbbc8b2dff4968225eadcf0e4d4ac5be.jpg)

![d5a8763e624b807f12d8086d0c1e7e8b78719dc9f7624929ceebc645d52cf2f4.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/d5a8763e624b807f12d8086d0c1e7e8b78719dc9f7624929ceebc645d52cf2f4.jpg)

![d70c3c0fc16ccdffe74d7d54a6b6cd0ec4d28545c4a3492541fac8d25570c504.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/d70c3c0fc16ccdffe74d7d54a6b6cd0ec4d28545c4a3492541fac8d25570c504.jpg)

![dea3eb048f1b34cea1d309cc051ad77598890efaabf2ab251431548d2aff4430.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/dea3eb048f1b34cea1d309cc051ad77598890efaabf2ab251431548d2aff4430.jpg)

![fe9c2cc517ebd26693a422c62a5f4a4302b9d1af706279e5c351cdec73bb2fd3.jpg](../icml_results/2921_ALSO_%20Adversarial%20Online%20Strategy%20Optimization%20for%20Social%20Agents/tables/fe9c2cc517ebd26693a422c62a5f4a4302b9d1af706279e5c351cdec73bb2fd3.jpg)

<a id="paper-2884"></a>

## Video-OPD: Efficient Post-Training of Multimodal Large Language Models for Temporal Video Grounding via On-Policy Distillation


### Images

![2170e2d56b34a64e0e5f87eeac073528b2a5917d174424b1fc096f26b1856c01.jpg](../icml_results/2922_Video-OPD_%20Efficient%20Post-Training%20of%20Multimodal%20Large%20Language%20Models%20for%20Temporal%20Video%20Grounding%20/images/2170e2d56b34a64e0e5f87eeac073528b2a5917d174424b1fc096f26b1856c01.jpg)

![4101758122a3965c3ce92431dbe8efc73fbb0bcc6d6057376cf36c9f35621021.jpg](../icml_results/2922_Video-OPD_%20Efficient%20Post-Training%20of%20Multimodal%20Large%20Language%20Models%20for%20Temporal%20Video%20Grounding%20/images/4101758122a3965c3ce92431dbe8efc73fbb0bcc6d6057376cf36c9f35621021.jpg)

![5de60fc5e68ad44813ce333f7f0d332b2447b4e5b32eb24fb0166445c8d975d7.jpg](../icml_results/2922_Video-OPD_%20Efficient%20Post-Training%20of%20Multimodal%20Large%20Language%20Models%20for%20Temporal%20Video%20Grounding%20/images/5de60fc5e68ad44813ce333f7f0d332b2447b4e5b32eb24fb0166445c8d975d7.jpg)

![a00660466aa8475e2e5829971d6f9807ea8c001b546d39d0e83c29ba263fd538.jpg](../icml_results/2922_Video-OPD_%20Efficient%20Post-Training%20of%20Multimodal%20Large%20Language%20Models%20for%20Temporal%20Video%20Grounding%20/images/a00660466aa8475e2e5829971d6f9807ea8c001b546d39d0e83c29ba263fd538.jpg)

![b3153dab916d6438445e1a5450b4019f684baf01adccdfc5d4ff2c443befaf75.jpg](../icml_results/2922_Video-OPD_%20Efficient%20Post-Training%20of%20Multimodal%20Large%20Language%20Models%20for%20Temporal%20Video%20Grounding%20/images/b3153dab916d6438445e1a5450b4019f684baf01adccdfc5d4ff2c443befaf75.jpg)

![b634820e5a26c137950c92096a3afabcd9e11723e176a52f4f035ab05a14d623.jpg](../icml_results/2922_Video-OPD_%20Efficient%20Post-Training%20of%20Multimodal%20Large%20Language%20Models%20for%20Temporal%20Video%20Grounding%20/images/b634820e5a26c137950c92096a3afabcd9e11723e176a52f4f035ab05a14d623.jpg)

### Tables

![066b34f230f96c867d2412eb4e873dfc5fbcdc376b4e24253c410d4b0e1ff2c6.jpg](../icml_results/2922_Video-OPD_%20Efficient%20Post-Training%20of%20Multimodal%20Large%20Language%20Models%20for%20Temporal%20Video%20Grounding%20/tables/066b34f230f96c867d2412eb4e873dfc5fbcdc376b4e24253c410d4b0e1ff2c6.jpg)

![0ddb76f7a8fabc672d51c1e3dbc77cd7929dde9aa3f5b944d404caabb868614f.jpg](../icml_results/2922_Video-OPD_%20Efficient%20Post-Training%20of%20Multimodal%20Large%20Language%20Models%20for%20Temporal%20Video%20Grounding%20/tables/0ddb76f7a8fabc672d51c1e3dbc77cd7929dde9aa3f5b944d404caabb868614f.jpg)

![293c5797112679b81c0114d670520ffa351ec962ef4fe4c6e000464d8f28cfb2.jpg](../icml_results/2922_Video-OPD_%20Efficient%20Post-Training%20of%20Multimodal%20Large%20Language%20Models%20for%20Temporal%20Video%20Grounding%20/tables/293c5797112679b81c0114d670520ffa351ec962ef4fe4c6e000464d8f28cfb2.jpg)

![57ab6621b1357a63ce147f29adc1b12a083b29d4d937f0d55db95949fc7fc69f.jpg](../icml_results/2922_Video-OPD_%20Efficient%20Post-Training%20of%20Multimodal%20Large%20Language%20Models%20for%20Temporal%20Video%20Grounding%20/tables/57ab6621b1357a63ce147f29adc1b12a083b29d4d937f0d55db95949fc7fc69f.jpg)

![63fad12c7a81d7ebbcb93b182a8d2287f7a069c27e9e655914b7ca847c626359.jpg](../icml_results/2922_Video-OPD_%20Efficient%20Post-Training%20of%20Multimodal%20Large%20Language%20Models%20for%20Temporal%20Video%20Grounding%20/tables/63fad12c7a81d7ebbcb93b182a8d2287f7a069c27e9e655914b7ca847c626359.jpg)

![7b41a26c63c34877a477eba76f6ddef4b19bb5ee12e2421c3dbdf2f2dbc4841f.jpg](../icml_results/2922_Video-OPD_%20Efficient%20Post-Training%20of%20Multimodal%20Large%20Language%20Models%20for%20Temporal%20Video%20Grounding%20/tables/7b41a26c63c34877a477eba76f6ddef4b19bb5ee12e2421c3dbdf2f2dbc4841f.jpg)

![aeb7175d36297f709a9261087b0c9509d015eccf05b1f2b32b8999fef4ac4096.jpg](../icml_results/2922_Video-OPD_%20Efficient%20Post-Training%20of%20Multimodal%20Large%20Language%20Models%20for%20Temporal%20Video%20Grounding%20/tables/aeb7175d36297f709a9261087b0c9509d015eccf05b1f2b32b8999fef4ac4096.jpg)

![f5f7461da55bb75b8017b4c82a65d0133ecc20d857990fbd154485b31d5d4e94.jpg](../icml_results/2922_Video-OPD_%20Efficient%20Post-Training%20of%20Multimodal%20Large%20Language%20Models%20for%20Temporal%20Video%20Grounding%20/tables/f5f7461da55bb75b8017b4c82a65d0133ecc20d857990fbd154485b31d5d4e94.jpg)

<a id="paper-2885"></a>

## Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective


### Images

![1427901f54b77622d551fde5807c09b85d2190442752d48d8a62de9b5dd0378d.jpg](../icml_results/2923_Learning%20Dynamics%20of%20Zeroth-Order%20Optimization_%20A%20Kernel%20Perspective/images/1427901f54b77622d551fde5807c09b85d2190442752d48d8a62de9b5dd0378d.jpg)

![2c862d9da070a9494ea17d65d7eb6155660641681f9ee705f91b5c29f6f61332.jpg](../icml_results/2923_Learning%20Dynamics%20of%20Zeroth-Order%20Optimization_%20A%20Kernel%20Perspective/images/2c862d9da070a9494ea17d65d7eb6155660641681f9ee705f91b5c29f6f61332.jpg)

![320e7cd5bf03e5dca71c132489186c6e27ca83a7ef043091a65085cf41a0dae6.jpg](../icml_results/2923_Learning%20Dynamics%20of%20Zeroth-Order%20Optimization_%20A%20Kernel%20Perspective/images/320e7cd5bf03e5dca71c132489186c6e27ca83a7ef043091a65085cf41a0dae6.jpg)

![41124ac1a54b39d7226a924033940395d1bcac70f63e1a31747122810662711c.jpg](../icml_results/2923_Learning%20Dynamics%20of%20Zeroth-Order%20Optimization_%20A%20Kernel%20Perspective/images/41124ac1a54b39d7226a924033940395d1bcac70f63e1a31747122810662711c.jpg)

![414b8c5dab98677c9559c1124ca454c1aacc0911a18ee57c783ab71653c35a68.jpg](../icml_results/2923_Learning%20Dynamics%20of%20Zeroth-Order%20Optimization_%20A%20Kernel%20Perspective/images/414b8c5dab98677c9559c1124ca454c1aacc0911a18ee57c783ab71653c35a68.jpg)

![5e9ba4b73bd94fdf1a4a2268c4e7100eb03ffb69bb57ef0a23ebcf9fd486f79b.jpg](../icml_results/2923_Learning%20Dynamics%20of%20Zeroth-Order%20Optimization_%20A%20Kernel%20Perspective/images/5e9ba4b73bd94fdf1a4a2268c4e7100eb03ffb69bb57ef0a23ebcf9fd486f79b.jpg)

![68d4bf4e25828b73afd37284612a09d7ac8b2d7a981c7eb05141439f7115a7a8.jpg](../icml_results/2923_Learning%20Dynamics%20of%20Zeroth-Order%20Optimization_%20A%20Kernel%20Perspective/images/68d4bf4e25828b73afd37284612a09d7ac8b2d7a981c7eb05141439f7115a7a8.jpg)

![80b29c5c97623adba8d7343a9f730a8623dd8aa89d8d3c84af8a0126f00e520b.jpg](../icml_results/2923_Learning%20Dynamics%20of%20Zeroth-Order%20Optimization_%20A%20Kernel%20Perspective/images/80b29c5c97623adba8d7343a9f730a8623dd8aa89d8d3c84af8a0126f00e520b.jpg)

![aa9004444be652339ac539095c3d219cd95b5dab7e98125ac8957e7607acf865.jpg](../icml_results/2923_Learning%20Dynamics%20of%20Zeroth-Order%20Optimization_%20A%20Kernel%20Perspective/images/aa9004444be652339ac539095c3d219cd95b5dab7e98125ac8957e7607acf865.jpg)

![ab6687461f28f818b922a9dc164ee8f81e27293a3a54b8855305b13dd559c204.jpg](../icml_results/2923_Learning%20Dynamics%20of%20Zeroth-Order%20Optimization_%20A%20Kernel%20Perspective/images/ab6687461f28f818b922a9dc164ee8f81e27293a3a54b8855305b13dd559c204.jpg)

![b34a12b1956c2016a1c07e696eee7292b427961112f3af0d9ad096c3a078d9b1.jpg](../icml_results/2923_Learning%20Dynamics%20of%20Zeroth-Order%20Optimization_%20A%20Kernel%20Perspective/images/b34a12b1956c2016a1c07e696eee7292b427961112f3af0d9ad096c3a078d9b1.jpg)

![ba31b0357e969da6ef585ed1a87fdfc71bdd56f84f881ead3ec81523ab23d583.jpg](../icml_results/2923_Learning%20Dynamics%20of%20Zeroth-Order%20Optimization_%20A%20Kernel%20Perspective/images/ba31b0357e969da6ef585ed1a87fdfc71bdd56f84f881ead3ec81523ab23d583.jpg)

![eed3fc7044e22d3ac0b77a1d1a32c8223b92752596e0e156bb629a4179b6b0c6.jpg](../icml_results/2923_Learning%20Dynamics%20of%20Zeroth-Order%20Optimization_%20A%20Kernel%20Perspective/images/eed3fc7044e22d3ac0b77a1d1a32c8223b92752596e0e156bb629a4179b6b0c6.jpg)

### Tables

![24df64448302b2a983ad571cf3f4da1c7ad36c3ed64206218a9b213c8b3c725b.jpg](../icml_results/2923_Learning%20Dynamics%20of%20Zeroth-Order%20Optimization_%20A%20Kernel%20Perspective/tables/24df64448302b2a983ad571cf3f4da1c7ad36c3ed64206218a9b213c8b3c725b.jpg)

![2dce332538a434ad683c0fcd573d108b9a0235b2ac917c16b2d7e4ae5ea496a6.jpg](../icml_results/2923_Learning%20Dynamics%20of%20Zeroth-Order%20Optimization_%20A%20Kernel%20Perspective/tables/2dce332538a434ad683c0fcd573d108b9a0235b2ac917c16b2d7e4ae5ea496a6.jpg)

<a id="paper-2886"></a>

## FactGuard: Agentic Video Misinformation Detection via Reinforcement Learning


### Images

![0f3264d4260b5f6f4baeff3d4d4abc47f234758a36cfc2021b0b15bfe56cea93.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/images/0f3264d4260b5f6f4baeff3d4d4abc47f234758a36cfc2021b0b15bfe56cea93.jpg)

![43f859a94309486e9d9b7a37eda08b719ab264290579f3e1556ed8eec2f9f977.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/images/43f859a94309486e9d9b7a37eda08b719ab264290579f3e1556ed8eec2f9f977.jpg)

![6bc3b6ebed7bd9839a7eb56d913b172e3840fe9be8e940cc059848c02b06b209.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/images/6bc3b6ebed7bd9839a7eb56d913b172e3840fe9be8e940cc059848c02b06b209.jpg)

![86b099caa80d7f3f71a90a1dd5063eab40a11c8eda61fc4b119eb965612b60c5.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/images/86b099caa80d7f3f71a90a1dd5063eab40a11c8eda61fc4b119eb965612b60c5.jpg)

![8f377be886721ac9eb70fce3982ba1e18b4a8fc0b9fa4dbdd8fe656f461a7647.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/images/8f377be886721ac9eb70fce3982ba1e18b4a8fc0b9fa4dbdd8fe656f461a7647.jpg)

![96cdb13f2c8621d3dcec11641e5efd8a25acd990c8faeeb5ae033cf4cc9653b6.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/images/96cdb13f2c8621d3dcec11641e5efd8a25acd990c8faeeb5ae033cf4cc9653b6.jpg)

![9fb87b01f664672126abe70b3ad618a0421b950ad1fa0e2aa5f20dc1e71b6547.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/images/9fb87b01f664672126abe70b3ad618a0421b950ad1fa0e2aa5f20dc1e71b6547.jpg)

![c524de624e5cd31b047ec2cb6a1c8f20643ab6bd0876886408f5d4b849c3b6a8.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/images/c524de624e5cd31b047ec2cb6a1c8f20643ab6bd0876886408f5d4b849c3b6a8.jpg)

![c90b87b8405b523dc97b680926499f691dffae5ca5e0faddcbc2da3229e86217.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/images/c90b87b8405b523dc97b680926499f691dffae5ca5e0faddcbc2da3229e86217.jpg)

![ddbb18cdedd16bbf94a4a28c6965366433ce42db52e38b1c385385b160bbe9e9.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/images/ddbb18cdedd16bbf94a4a28c6965366433ce42db52e38b1c385385b160bbe9e9.jpg)

![e15f14fc9ee4fe132f64d9078cbde549bbe9fbecd729562abc7a487bd27969a6.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/images/e15f14fc9ee4fe132f64d9078cbde549bbe9fbecd729562abc7a487bd27969a6.jpg)

![ec8674cb7468ab9b33bcf323c57ccda435fd10590d3efe8f4c0b0c0c1ddd18d3.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/images/ec8674cb7468ab9b33bcf323c57ccda435fd10590d3efe8f4c0b0c0c1ddd18d3.jpg)

### Tables

![154c04453a9231b7782e37ab08912f728cab3681e873f2d447c94b827f01df92.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/tables/154c04453a9231b7782e37ab08912f728cab3681e873f2d447c94b827f01df92.jpg)

![57000ad24c1ccbdc1a6e74f5b0b19cc0676d668443944561d5bf9b6c11c72785.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/tables/57000ad24c1ccbdc1a6e74f5b0b19cc0676d668443944561d5bf9b6c11c72785.jpg)

![7adf0f9b97ad509f97939fbc0858bf789cecad3ae25f3c85ae51a624f11e4587.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/tables/7adf0f9b97ad509f97939fbc0858bf789cecad3ae25f3c85ae51a624f11e4587.jpg)

![8395ca06d164dcaf928198c4adae43300c60229b483c87c86d5c551e15d178dd.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/tables/8395ca06d164dcaf928198c4adae43300c60229b483c87c86d5c551e15d178dd.jpg)

![85df6bd8565cca3d01f09205311938e3912390ece46a6eba7e6cfeff9cbb281e.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/tables/85df6bd8565cca3d01f09205311938e3912390ece46a6eba7e6cfeff9cbb281e.jpg)

![ace36905750b9b9d21542ed517eb3e5011bf32b2eadb195f45e039964dbfd7bf.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/tables/ace36905750b9b9d21542ed517eb3e5011bf32b2eadb195f45e039964dbfd7bf.jpg)

![c61c822c2206cb501e0752f90de71bc9eb64a19b1e3420507bd4142ac5152440.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/tables/c61c822c2206cb501e0752f90de71bc9eb64a19b1e3420507bd4142ac5152440.jpg)

![eee7d4a6d7c04834c1b41342216932aa9a1d0932b01bfdddea223f5d04dc734c.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/tables/eee7d4a6d7c04834c1b41342216932aa9a1d0932b01bfdddea223f5d04dc734c.jpg)

![f09d74f3313e206ba5f4557f715ce32498bbff119e9377d600c10c977c6867b9.jpg](../icml_results/2924_FactGuard_%20Agentic%20Video%20Misinformation%20Detection%20via%20Reinforcement%20Learning/tables/f09d74f3313e206ba5f4557f715ce32498bbff119e9377d600c10c977c6867b9.jpg)

<a id="paper-2887"></a>

## Adalina: Adaptive Linear Approximation for the Shapley Value and Beyond


### Images

![1bcfcf16e010907964374391a90a74a2ec1618d6cea10203cb1c71c54bdeac52.jpg](../icml_results/2925_Adalina_%20Adaptive%20Linear%20Approximation%20for%20the%20Shapley%20Value%20and%20Beyond/images/1bcfcf16e010907964374391a90a74a2ec1618d6cea10203cb1c71c54bdeac52.jpg)

![3f99e0289802dc10d347b15029b52e8a0482a92a981cc9cf1b8e4600f69c6ba3.jpg](../icml_results/2925_Adalina_%20Adaptive%20Linear%20Approximation%20for%20the%20Shapley%20Value%20and%20Beyond/images/3f99e0289802dc10d347b15029b52e8a0482a92a981cc9cf1b8e4600f69c6ba3.jpg)

![3fc184b2713bd6d50e30703922cf1d8c987eee5dc1ae19da8182291cf40e8130.jpg](../icml_results/2925_Adalina_%20Adaptive%20Linear%20Approximation%20for%20the%20Shapley%20Value%20and%20Beyond/images/3fc184b2713bd6d50e30703922cf1d8c987eee5dc1ae19da8182291cf40e8130.jpg)

![69c892ba1332ace3dcd874ef605b0c7d850e0e2453496ba02911a3b6cd52baa1.jpg](../icml_results/2925_Adalina_%20Adaptive%20Linear%20Approximation%20for%20the%20Shapley%20Value%20and%20Beyond/images/69c892ba1332ace3dcd874ef605b0c7d850e0e2453496ba02911a3b6cd52baa1.jpg)

![7e89c8e9332e9c8fcd2404dc4f5c76e520fe98c3ab11961444c0f6209c0451ab.jpg](../icml_results/2925_Adalina_%20Adaptive%20Linear%20Approximation%20for%20the%20Shapley%20Value%20and%20Beyond/images/7e89c8e9332e9c8fcd2404dc4f5c76e520fe98c3ab11961444c0f6209c0451ab.jpg)

![9248ec53c0fcba1b404726efb6e511b2089867f08e66e20ada80fa332b5b3c15.jpg](../icml_results/2925_Adalina_%20Adaptive%20Linear%20Approximation%20for%20the%20Shapley%20Value%20and%20Beyond/images/9248ec53c0fcba1b404726efb6e511b2089867f08e66e20ada80fa332b5b3c15.jpg)

![dc973f79fffefebf1c154d9284fa7e88f5f5a649431668beed42480fe02fba5b.jpg](../icml_results/2925_Adalina_%20Adaptive%20Linear%20Approximation%20for%20the%20Shapley%20Value%20and%20Beyond/images/dc973f79fffefebf1c154d9284fa7e88f5f5a649431668beed42480fe02fba5b.jpg)

### Tables

![823396b5132402b16097a9848ca7f5d848e1254cc41bbd657cc27abaa6f03b16.jpg](../icml_results/2925_Adalina_%20Adaptive%20Linear%20Approximation%20for%20the%20Shapley%20Value%20and%20Beyond/tables/823396b5132402b16097a9848ca7f5d848e1254cc41bbd657cc27abaa6f03b16.jpg)

<a id="paper-2888"></a>

## Every Step Counts: Decoding Trajectories as Authorship Fingerprints of dLLMs


### Images

![0ade69cfdb0c5e7b43e57f26d9c8d70752a7fcbfc50d4458f0d29da46cf80271.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/images/0ade69cfdb0c5e7b43e57f26d9c8d70752a7fcbfc50d4458f0d29da46cf80271.jpg)

![2250e0541e9c958fc779820cdd84625cdc4cd410115c54cd9262da2cc027d50c.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/images/2250e0541e9c958fc779820cdd84625cdc4cd410115c54cd9262da2cc027d50c.jpg)

![32e324dfe04803870b570d45d8bccfd082f23f019baa898c3b92049a937e791c.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/images/32e324dfe04803870b570d45d8bccfd082f23f019baa898c3b92049a937e791c.jpg)

![6022f8a657ed01a92a359ada914c6a5a8345e969631ee646d62fe11b1d957e26.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/images/6022f8a657ed01a92a359ada914c6a5a8345e969631ee646d62fe11b1d957e26.jpg)

![77a4ef73c324ccdacf00051fe9323e8a353966cb7e6f51e662dd4935c6f0df3b.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/images/77a4ef73c324ccdacf00051fe9323e8a353966cb7e6f51e662dd4935c6f0df3b.jpg)

![86d5dcb4ca27a640ad4bf087683ae6ba193b4230993cc9232e202732a3c7e647.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/images/86d5dcb4ca27a640ad4bf087683ae6ba193b4230993cc9232e202732a3c7e647.jpg)

![87336dee02aca6cfef3ac9073b03ee3d3367b1f2b531d64140202934fa9e9ef7.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/images/87336dee02aca6cfef3ac9073b03ee3d3367b1f2b531d64140202934fa9e9ef7.jpg)

![ab3d7cd2cf9296d511be8cea8e893c93c17f4d4327f650738413cbd61a5a6ec9.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/images/ab3d7cd2cf9296d511be8cea8e893c93c17f4d4327f650738413cbd61a5a6ec9.jpg)

![b0f0ff16d3a7590ca4008e4ad2ee46d1c604ce1b48c3e30bbbf4c869e68fbdd6.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/images/b0f0ff16d3a7590ca4008e4ad2ee46d1c604ce1b48c3e30bbbf4c869e68fbdd6.jpg)

![bed8093c588bb72d7b65a861d0966032c58618e5ed9b93921e65b51e8e714fc8.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/images/bed8093c588bb72d7b65a861d0966032c58618e5ed9b93921e65b51e8e714fc8.jpg)

![d0bf17b3436aaf743771b3545875c977e8e463e1464c0bac8befb3daf5490b27.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/images/d0bf17b3436aaf743771b3545875c977e8e463e1464c0bac8befb3daf5490b27.jpg)

![dd53aaf2b5942a4af5950dee8b3ceaa001cf4f3900db6db95c3618b05e996891.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/images/dd53aaf2b5942a4af5950dee8b3ceaa001cf4f3900db6db95c3618b05e996891.jpg)

![e671c7613315f834c62480d2e62619f45942f26afbf0fb73633352c2baab6b3e.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/images/e671c7613315f834c62480d2e62619f45942f26afbf0fb73633352c2baab6b3e.jpg)

![f09d5aaa4f14afaabead5ea553c7e7b330be19ad42f3f6f4a97bc4f6cc251e46.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/images/f09d5aaa4f14afaabead5ea553c7e7b330be19ad42f3f6f4a97bc4f6cc251e46.jpg)

![f25577feb0b64b9dc9892482236cb8a77400d5acbf192c2c8ec48903280f2b78.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/images/f25577feb0b64b9dc9892482236cb8a77400d5acbf192c2c8ec48903280f2b78.jpg)

![f4074b2df204cbc90800a7f0361c9d9e1308d287faa26db14fe56684d5e57360.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/images/f4074b2df204cbc90800a7f0361c9d9e1308d287faa26db14fe56684d5e57360.jpg)

### Tables

![2a22afd69f74418cf86f8483de6051f669a368af2a151d6dc53f52330f014eb8.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/tables/2a22afd69f74418cf86f8483de6051f669a368af2a151d6dc53f52330f014eb8.jpg)

![3e9d5868c5bca7f5e94d85aa0cec8dc6715f67243d32eb2af1070cc13228dc99.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/tables/3e9d5868c5bca7f5e94d85aa0cec8dc6715f67243d32eb2af1070cc13228dc99.jpg)

![579d4ace383f8f174112a059ed825efc39d544e4d079c6b833e712b648d29ae8.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/tables/579d4ace383f8f174112a059ed825efc39d544e4d079c6b833e712b648d29ae8.jpg)

![63db56478bb643c303313a0409e519c5f61bd95197103f9ac4e830614667797a.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/tables/63db56478bb643c303313a0409e519c5f61bd95197103f9ac4e830614667797a.jpg)

![67602e68088b74ce04b6bd3da2b684a1aa4fc1c8590ff825e5e3b15441aac990.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/tables/67602e68088b74ce04b6bd3da2b684a1aa4fc1c8590ff825e5e3b15441aac990.jpg)

![79837fb327c92df0d8f152fe2272e21a0038aca3d1bc0ff7aac98c239bcad4f8.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/tables/79837fb327c92df0d8f152fe2272e21a0038aca3d1bc0ff7aac98c239bcad4f8.jpg)

![d353ed15597a874443b968ff14b271ec973ae78be6114c3029d132ba245b9e36.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/tables/d353ed15597a874443b968ff14b271ec973ae78be6114c3029d132ba245b9e36.jpg)

![dfced5e63b1266964148e7304d623f13ddfd7ea3f4250ecb80520e647380fbb1.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/tables/dfced5e63b1266964148e7304d623f13ddfd7ea3f4250ecb80520e647380fbb1.jpg)

![e377d1c123fb18cea2d5999defb251ee1fc0adb23584f37095613168f24e3e31.jpg](../icml_results/2926_Every%20Step%20Counts_%20Decoding%20Trajectories%20as%20Authorship%20Fingerprints%20of%20dLLMs/tables/e377d1c123fb18cea2d5999defb251ee1fc0adb23584f37095613168f24e3e31.jpg)

<a id="paper-2889"></a>

## Unsafer in Many Turns: Benchmarking and Defending Multi-Turn Safety Risks in Tool-Using Agents


### Images

![168fbbaec078457ce397ff0ce5a3660b3a224d6970c2d7f2cd336068a48c676d.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/168fbbaec078457ce397ff0ce5a3660b3a224d6970c2d7f2cd336068a48c676d.jpg)

![1dd0222886e5336c206e1f9d0596aaa07add032a537c15619bb7f5585a68b345.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/1dd0222886e5336c206e1f9d0596aaa07add032a537c15619bb7f5585a68b345.jpg)

![1e04f054db7bd75846f156a3d58e65f998011379dc207153ef78247f45f7b382.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/1e04f054db7bd75846f156a3d58e65f998011379dc207153ef78247f45f7b382.jpg)

![1e4f525ae7c4201ef78a2c07e091d6b8048a88b94342709d2b176f1bb9952539.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/1e4f525ae7c4201ef78a2c07e091d6b8048a88b94342709d2b176f1bb9952539.jpg)

![25f8e33395d68fc85f839f544f36c7de9feaf5f71d5362fe826178d075e4a570.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/25f8e33395d68fc85f839f544f36c7de9feaf5f71d5362fe826178d075e4a570.jpg)

![262f44b6b0cbe69b6393eb0ddbb0d981fa3662759a39aa755d10b7da8f93d6a9.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/262f44b6b0cbe69b6393eb0ddbb0d981fa3662759a39aa755d10b7da8f93d6a9.jpg)

![381ab1d916b254cb3d641b6fdd0a6be8da5112fe40065ed3f50753aa3ec924f6.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/381ab1d916b254cb3d641b6fdd0a6be8da5112fe40065ed3f50753aa3ec924f6.jpg)

![396790c594e4e32c75c418f881f2cafe0a290c9f9d26cd61bae4a9df4a5368f4.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/396790c594e4e32c75c418f881f2cafe0a290c9f9d26cd61bae4a9df4a5368f4.jpg)

![433025bcffc47211cbe5f89e38f0d7956110ba99c07e20513bc0c6870ee877a1.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/433025bcffc47211cbe5f89e38f0d7956110ba99c07e20513bc0c6870ee877a1.jpg)

![4924fe31577035b8f217a465a86867e347ea90563d0b99974b8abad66d371ef5.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/4924fe31577035b8f217a465a86867e347ea90563d0b99974b8abad66d371ef5.jpg)

![51b14eb024b8c14c571fb42dc34708f128a442fe99bd9c8750a4930940f4213a.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/51b14eb024b8c14c571fb42dc34708f128a442fe99bd9c8750a4930940f4213a.jpg)

![56bb67fc8341490ac68fd0479d8a8c9dfd7e3fa46d3847f413b5907579f563c7.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/56bb67fc8341490ac68fd0479d8a8c9dfd7e3fa46d3847f413b5907579f563c7.jpg)

![57859a6eb48f424687725b9971e402a9fd036b6ac272f89d71ae9b21648c82af.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/57859a6eb48f424687725b9971e402a9fd036b6ac272f89d71ae9b21648c82af.jpg)

![57d78c08e3237cf7cb21e90c7970546e77131d72c989c0d3b71c4cb280166985.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/57d78c08e3237cf7cb21e90c7970546e77131d72c989c0d3b71c4cb280166985.jpg)

![6366c2089539b7f0c50d2176eb6c92da03c863add46ad9f0fb569e6fa689a491.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/6366c2089539b7f0c50d2176eb6c92da03c863add46ad9f0fb569e6fa689a491.jpg)

![69bc797d36d6edd24e91d5b15fbc7eedfb6f27f5ac5980bff200aabb2f2a3db5.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/69bc797d36d6edd24e91d5b15fbc7eedfb6f27f5ac5980bff200aabb2f2a3db5.jpg)

![7606a01a98d694570e995f0ec19964fbc7fc16a3977807073b52983a4abe4b9e.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/7606a01a98d694570e995f0ec19964fbc7fc16a3977807073b52983a4abe4b9e.jpg)

![7c2c4f440b063829008a27ba9944195e0bf193cb38960e99e9107dcdc79a49c9.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/7c2c4f440b063829008a27ba9944195e0bf193cb38960e99e9107dcdc79a49c9.jpg)

![837795c5f6585222633de4fc69320f94dcb09b368dca001ef29a03b7a0659a26.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/837795c5f6585222633de4fc69320f94dcb09b368dca001ef29a03b7a0659a26.jpg)

![8e604a8d4a2ff8a136d19d5bed3bc872700f36d23e087aa8de4370847cefca67.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/8e604a8d4a2ff8a136d19d5bed3bc872700f36d23e087aa8de4370847cefca67.jpg)

![99846c1d854ce2859efdd070021bc6a58365ac5c634fdab8408ea65bfff3adb5.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/99846c1d854ce2859efdd070021bc6a58365ac5c634fdab8408ea65bfff3adb5.jpg)

![9a9c558094dec93b56b3e5aec73487fda1636e07cb8077faebfa667e50bd2138.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/9a9c558094dec93b56b3e5aec73487fda1636e07cb8077faebfa667e50bd2138.jpg)

![a260cde7c25ecf216e6dea2dd07d0c9e35c9678c1055e5722269a512001239a0.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/a260cde7c25ecf216e6dea2dd07d0c9e35c9678c1055e5722269a512001239a0.jpg)

![a2f981cf4d03db420db6247d69840e5fc1ceb4f7b2cf47ac26ae05a66c0b453f.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/a2f981cf4d03db420db6247d69840e5fc1ceb4f7b2cf47ac26ae05a66c0b453f.jpg)

![ac063f2c2a8e0df4b0745bc244ab2405c321fe496a2dc0e1db0c01a1d5dd18d9.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/ac063f2c2a8e0df4b0745bc244ab2405c321fe496a2dc0e1db0c01a1d5dd18d9.jpg)

![ac1bf9cc366d84e5e032f3dcfe68e996d7d757ee4874cebb17d99b0a0fa605ca.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/ac1bf9cc366d84e5e032f3dcfe68e996d7d757ee4874cebb17d99b0a0fa605ca.jpg)

![be75e7172e6137d8fe658bafc9bdeab5ea91efc04d91e95efc442570b2755de2.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/be75e7172e6137d8fe658bafc9bdeab5ea91efc04d91e95efc442570b2755de2.jpg)

![c10795a761ce92a0aebb73e9214ce832756d5bc6633d4112f64d77dc15904658.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/c10795a761ce92a0aebb73e9214ce832756d5bc6633d4112f64d77dc15904658.jpg)

![d05aba833ca0974262a0448401f08031ef05dee728b244a61c2c4db64d1ae097.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/d05aba833ca0974262a0448401f08031ef05dee728b244a61c2c4db64d1ae097.jpg)

![d5097db037d15d5de15f68103048c8befc1b36d3c305c43bb4d8bf3110092fe1.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/d5097db037d15d5de15f68103048c8befc1b36d3c305c43bb4d8bf3110092fe1.jpg)

![d6858d3654232eb18f198598a0ecea01204261244d52f8de4dc848a0d470a396.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/d6858d3654232eb18f198598a0ecea01204261244d52f8de4dc848a0d470a396.jpg)

![e27325f8e8034f1dc8da81ad27b3ebee2ccdd248543fe17c9cd2e3c4872bc567.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/e27325f8e8034f1dc8da81ad27b3ebee2ccdd248543fe17c9cd2e3c4872bc567.jpg)

![e3107cadd7107e7c41b7074eac3debcaa6fdc5e6c8bf1c8f833cbfc2292bf9d2.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/e3107cadd7107e7c41b7074eac3debcaa6fdc5e6c8bf1c8f833cbfc2292bf9d2.jpg)

![e575381757d41a2a195c1989e59baca8476f59b75c484431288b7193e5011e5d.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/e575381757d41a2a195c1989e59baca8476f59b75c484431288b7193e5011e5d.jpg)

![e87cbdba4a704e66600e050ee5aec44a5950e7cf83d05c3125c4e1d28e77810f.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/e87cbdba4a704e66600e050ee5aec44a5950e7cf83d05c3125c4e1d28e77810f.jpg)

![f0866cb5f9613aa22e6b59a33fb61b16c0d84f964138642cd208c4eaa497d13d.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/images/f0866cb5f9613aa22e6b59a33fb61b16c0d84f964138642cd208c4eaa497d13d.jpg)

### Tables

![053cce67a6512ca0cd20606a2defbe5a134fa4751d65287d4c1271d41b49df61.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/053cce67a6512ca0cd20606a2defbe5a134fa4751d65287d4c1271d41b49df61.jpg)

![0bf83d0336eb9f91d937365ac2bd24a31932a5a5cbcadd20637c872b0efef812.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/0bf83d0336eb9f91d937365ac2bd24a31932a5a5cbcadd20637c872b0efef812.jpg)

![0fb3e065f8e44fd160db0c78d5d367caad61b9ff978bc039a5974a11c59a8195.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/0fb3e065f8e44fd160db0c78d5d367caad61b9ff978bc039a5974a11c59a8195.jpg)

![2a4eb5d68b577c9434d4b3ae55d8e0bbd4773799545c81469d98caf17dba1f95.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/2a4eb5d68b577c9434d4b3ae55d8e0bbd4773799545c81469d98caf17dba1f95.jpg)

![398d243392af0c559e021fa1b8a2a2f9ba7e896da8dafdadec9821251a2857dd.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/398d243392af0c559e021fa1b8a2a2f9ba7e896da8dafdadec9821251a2857dd.jpg)

![3af30222a1ced97b504245c07b0942e1c3018d95831087af693eac6fc473c5c9.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/3af30222a1ced97b504245c07b0942e1c3018d95831087af693eac6fc473c5c9.jpg)

![478f047ac498ac9b98f6a598360a5075ce79d325d11416aea374e7d1c15171ee.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/478f047ac498ac9b98f6a598360a5075ce79d325d11416aea374e7d1c15171ee.jpg)

![645dda1ab9d3aebd8c5ae2d45810e84a29b21a44bdd92ed08c10b6c50c8ed9a1.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/645dda1ab9d3aebd8c5ae2d45810e84a29b21a44bdd92ed08c10b6c50c8ed9a1.jpg)

![70f9cbfe9d6f0ba30959cf045510a7afaf757c37911d28297cfeb254fe57e66c.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/70f9cbfe9d6f0ba30959cf045510a7afaf757c37911d28297cfeb254fe57e66c.jpg)

![7a3714cadaabd03f65515a3c25e76b5f0650949716d45180ab5a1d625bf0a376.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/7a3714cadaabd03f65515a3c25e76b5f0650949716d45180ab5a1d625bf0a376.jpg)

![7aac8187d507ad096ac81e729d6b649976e1fb4267e00c1df2df611b19074174.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/7aac8187d507ad096ac81e729d6b649976e1fb4267e00c1df2df611b19074174.jpg)

![850e4c6f5a5f84351c779f253b27dad895f29138fb8baa874f1b67d319416476.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/850e4c6f5a5f84351c779f253b27dad895f29138fb8baa874f1b67d319416476.jpg)

![95522ffa1013d4ca324e1ef23e02889f20475f591d3b9b0e5fa32faadbc75c72.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/95522ffa1013d4ca324e1ef23e02889f20475f591d3b9b0e5fa32faadbc75c72.jpg)

![b8272a217ff205d375ace8140b4166765108b4335705023958d8837b93390076.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/b8272a217ff205d375ace8140b4166765108b4335705023958d8837b93390076.jpg)

![d72bc1d459b928eae34ca2bcfd550a036a5bb736e5979c9b83f80421c3bb4dbc.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/d72bc1d459b928eae34ca2bcfd550a036a5bb736e5979c9b83f80421c3bb4dbc.jpg)

![e5c23240557ce988cc26a2b76a3d5c191bae722f07c641fba8f7eecb8bf07ff8.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/e5c23240557ce988cc26a2b76a3d5c191bae722f07c641fba8f7eecb8bf07ff8.jpg)

![f68682986b081d150e6c58598b95ce1835d02de86843f1af0b8bec8f45904bdc.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/f68682986b081d150e6c58598b95ce1835d02de86843f1af0b8bec8f45904bdc.jpg)

![fad5c6507c53bb3915461a9efd0c08462c79227cf0cfea209b54db4f0ea2d4e5.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/fad5c6507c53bb3915461a9efd0c08462c79227cf0cfea209b54db4f0ea2d4e5.jpg)

![fe980298358185373cacc2f2029ee72657b9281ae46610c81b92a62351b2efe0.jpg](../icml_results/2927_Unsafer%20in%20Many%20Turns_%20Benchmarking%20and%20Defending%20Multi-Turn%20Safety%20Risks%20in%20Tool-Using%20Agents/tables/fe980298358185373cacc2f2029ee72657b9281ae46610c81b92a62351b2efe0.jpg)

<a id="paper-2890"></a>

## Towards Complete Multi-Agent Coordination Policy Learning via Denoising Maximum Entropy Optimization


### Images

![3cc0d2bcd7b5b6946e8ed99051ff524f74c06b02662021b420461fea0459b091.jpg](../icml_results/2928_Towards%20Complete%20Multi-Agent%20Coordination%20Policy%20Learning%20via%20Denoising%20Maximum%20Entropy%20Optimization/images/3cc0d2bcd7b5b6946e8ed99051ff524f74c06b02662021b420461fea0459b091.jpg)

![71c2004909a53b2413085bb842953a8cc7ffbd725d454ae7685aa47b4003c5fe.jpg](../icml_results/2928_Towards%20Complete%20Multi-Agent%20Coordination%20Policy%20Learning%20via%20Denoising%20Maximum%20Entropy%20Optimization/images/71c2004909a53b2413085bb842953a8cc7ffbd725d454ae7685aa47b4003c5fe.jpg)

![75a1dbcdb0af4deb01e9f21897bc96f193d20e5008ec3bddfe4e68ac3422a844.jpg](../icml_results/2928_Towards%20Complete%20Multi-Agent%20Coordination%20Policy%20Learning%20via%20Denoising%20Maximum%20Entropy%20Optimization/images/75a1dbcdb0af4deb01e9f21897bc96f193d20e5008ec3bddfe4e68ac3422a844.jpg)

![78b72a3f248109ba94fec0bb4fd0a128ab32fd889ad080d8b6b85142be0051e4.jpg](../icml_results/2928_Towards%20Complete%20Multi-Agent%20Coordination%20Policy%20Learning%20via%20Denoising%20Maximum%20Entropy%20Optimization/images/78b72a3f248109ba94fec0bb4fd0a128ab32fd889ad080d8b6b85142be0051e4.jpg)

![861610842e9c579956ee9a51132b878a04ecaf0a83b0958faccae853df6768ea.jpg](../icml_results/2928_Towards%20Complete%20Multi-Agent%20Coordination%20Policy%20Learning%20via%20Denoising%20Maximum%20Entropy%20Optimization/images/861610842e9c579956ee9a51132b878a04ecaf0a83b0958faccae853df6768ea.jpg)

![8e9a65945dd2c3df0c45d7277c2946202e6ff9d1b252487c0ac1b1aaad060056.jpg](../icml_results/2928_Towards%20Complete%20Multi-Agent%20Coordination%20Policy%20Learning%20via%20Denoising%20Maximum%20Entropy%20Optimization/images/8e9a65945dd2c3df0c45d7277c2946202e6ff9d1b252487c0ac1b1aaad060056.jpg)

![bb1e6ac3ac677bbfd52ee922017890fff7972091d6fdac0bb9abc7bc0f6b2f7b.jpg](../icml_results/2928_Towards%20Complete%20Multi-Agent%20Coordination%20Policy%20Learning%20via%20Denoising%20Maximum%20Entropy%20Optimization/images/bb1e6ac3ac677bbfd52ee922017890fff7972091d6fdac0bb9abc7bc0f6b2f7b.jpg)

![bc1d7808fb5567c1be10bf1b81c8347cddc62a9879d893e19095f2e16453bc5d.jpg](../icml_results/2928_Towards%20Complete%20Multi-Agent%20Coordination%20Policy%20Learning%20via%20Denoising%20Maximum%20Entropy%20Optimization/images/bc1d7808fb5567c1be10bf1b81c8347cddc62a9879d893e19095f2e16453bc5d.jpg)

![d7854cf6055c5173fa062336846e9778be28a54c16bd6657395159b99afe3912.jpg](../icml_results/2928_Towards%20Complete%20Multi-Agent%20Coordination%20Policy%20Learning%20via%20Denoising%20Maximum%20Entropy%20Optimization/images/d7854cf6055c5173fa062336846e9778be28a54c16bd6657395159b99afe3912.jpg)

### Tables

![40d5be270c3dbc0ffb127e583226b10779ab620695b5e15cad4f90560b849905.jpg](../icml_results/2928_Towards%20Complete%20Multi-Agent%20Coordination%20Policy%20Learning%20via%20Denoising%20Maximum%20Entropy%20Optimization/tables/40d5be270c3dbc0ffb127e583226b10779ab620695b5e15cad4f90560b849905.jpg)

![66a5045dac27578da5736598a64e67ad33536b4c4417c617f66e1b5ed3d6a2d7.jpg](../icml_results/2928_Towards%20Complete%20Multi-Agent%20Coordination%20Policy%20Learning%20via%20Denoising%20Maximum%20Entropy%20Optimization/tables/66a5045dac27578da5736598a64e67ad33536b4c4417c617f66e1b5ed3d6a2d7.jpg)

![6d9c811d2cf1b2921104485092f25221c444de87c618373bec55203029cc45d0.jpg](../icml_results/2928_Towards%20Complete%20Multi-Agent%20Coordination%20Policy%20Learning%20via%20Denoising%20Maximum%20Entropy%20Optimization/tables/6d9c811d2cf1b2921104485092f25221c444de87c618373bec55203029cc45d0.jpg)

![9eb0494454ed78789a4c4fd7e4f65c7bb7b6c6cda1221d10b39de390c93fbc0b.jpg](../icml_results/2928_Towards%20Complete%20Multi-Agent%20Coordination%20Policy%20Learning%20via%20Denoising%20Maximum%20Entropy%20Optimization/tables/9eb0494454ed78789a4c4fd7e4f65c7bb7b6c6cda1221d10b39de390c93fbc0b.jpg)

![b989186be8fe145d9de9846ba8488d2bd8329ce25235825251dd7320c3007e08.jpg](../icml_results/2928_Towards%20Complete%20Multi-Agent%20Coordination%20Policy%20Learning%20via%20Denoising%20Maximum%20Entropy%20Optimization/tables/b989186be8fe145d9de9846ba8488d2bd8329ce25235825251dd7320c3007e08.jpg)

![c59988e034fdf97b3323b7f9a7d028ce045282f904708d4948409d2d1d99ed25.jpg](../icml_results/2928_Towards%20Complete%20Multi-Agent%20Coordination%20Policy%20Learning%20via%20Denoising%20Maximum%20Entropy%20Optimization/tables/c59988e034fdf97b3323b7f9a7d028ce045282f904708d4948409d2d1d99ed25.jpg)

![ecb07e2b04c9577c4a8d82077e8b706699bb949571fa255dbb4ec06b5e384758.jpg](../icml_results/2928_Towards%20Complete%20Multi-Agent%20Coordination%20Policy%20Learning%20via%20Denoising%20Maximum%20Entropy%20Optimization/tables/ecb07e2b04c9577c4a8d82077e8b706699bb949571fa255dbb4ec06b5e384758.jpg)

<a id="paper-2891"></a>

## Probability of Matching for Batch Multi-Objective Bayesian Optimization


### Images

![062899c8ef95cc65877cb2416c2018140546f6471672c18db13c77d04ea1d4ad.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/images/062899c8ef95cc65877cb2416c2018140546f6471672c18db13c77d04ea1d4ad.jpg)

![72a7775267aaf631faf2675948cc5e257aac64769dcd1b18bbd8988a56b08c28.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/images/72a7775267aaf631faf2675948cc5e257aac64769dcd1b18bbd8988a56b08c28.jpg)

![8ceda99355311b1c784cb17e41a63e501ca532614c9ad31ae717773e703dcff5.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/images/8ceda99355311b1c784cb17e41a63e501ca532614c9ad31ae717773e703dcff5.jpg)

![999c6342431288a6c19686cb399b0c23e731891d78053cabe33acbdb2b463d1a.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/images/999c6342431288a6c19686cb399b0c23e731891d78053cabe33acbdb2b463d1a.jpg)

![9b09568228cfdeeebb96b3edc6a7bc21ad64cc51182f543c93ddee91b3332077.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/images/9b09568228cfdeeebb96b3edc6a7bc21ad64cc51182f543c93ddee91b3332077.jpg)

![d096a66ff867fc69eccca2117feb6815a8816d3bbabf2716438ed3a575af0b27.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/images/d096a66ff867fc69eccca2117feb6815a8816d3bbabf2716438ed3a575af0b27.jpg)

![d74486e8dc36ae1c3a902afa8ff9ad177d4562b3382b962a51e456c776e4a39e.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/images/d74486e8dc36ae1c3a902afa8ff9ad177d4562b3382b962a51e456c776e4a39e.jpg)

![ec2f1c1d359b6ad969c02bd2f8abeae1ddbbc3602b76bd98aa90bf1333661267.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/images/ec2f1c1d359b6ad969c02bd2f8abeae1ddbbc3602b76bd98aa90bf1333661267.jpg)

### Tables

![0343bea7a810f93887faa97555629d56a19095c098bffdf0012ce497fb68e245.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/tables/0343bea7a810f93887faa97555629d56a19095c098bffdf0012ce497fb68e245.jpg)

![11d5ce42e32bbd98807225b62e9acf8b3788413d5270164831be2c12230f16bb.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/tables/11d5ce42e32bbd98807225b62e9acf8b3788413d5270164831be2c12230f16bb.jpg)

![1dbafe168c43875dbee98acdf57faff73ca57a715145ee124149be88fae05a72.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/tables/1dbafe168c43875dbee98acdf57faff73ca57a715145ee124149be88fae05a72.jpg)

![2751c5d270a23147a1d08729c6ea9076442a378348dd501161da3c1397eb33f7.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/tables/2751c5d270a23147a1d08729c6ea9076442a378348dd501161da3c1397eb33f7.jpg)

![2ca2f3f7a33be17d47a44f9713b497785c04011c652089e8f59c3161c1a58449.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/tables/2ca2f3f7a33be17d47a44f9713b497785c04011c652089e8f59c3161c1a58449.jpg)

![3012f77719e27cf91e2cad44a922b40873019a830db2c2016fd9060ca724ec35.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/tables/3012f77719e27cf91e2cad44a922b40873019a830db2c2016fd9060ca724ec35.jpg)

![30e4c518259310c02c6eac483f43536663eca4c0312c134e2de40ba659295e15.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/tables/30e4c518259310c02c6eac483f43536663eca4c0312c134e2de40ba659295e15.jpg)

![4b524ac1fb163829f3dad80af7d57cf75b3215b7b3d5d8f91aa190686cb095a4.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/tables/4b524ac1fb163829f3dad80af7d57cf75b3215b7b3d5d8f91aa190686cb095a4.jpg)

![96c6726babb86943108bfe82b88804792e3833cd3f68daf3c087623ecde36463.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/tables/96c6726babb86943108bfe82b88804792e3833cd3f68daf3c087623ecde36463.jpg)

![c60ca5fef46d279fac43c3e66a085f206eaaa8c6a5935e3121456bf8f47a8d72.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/tables/c60ca5fef46d279fac43c3e66a085f206eaaa8c6a5935e3121456bf8f47a8d72.jpg)

![cb8aa4c5c67af6bbdb8a2fbebfbfb80201868038a712fe1654121b87ae242ccd.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/tables/cb8aa4c5c67af6bbdb8a2fbebfbfb80201868038a712fe1654121b87ae242ccd.jpg)

![d6b5e38e57b4d7a53ccfdf202d5e8c0e1f577aa864031790fd381ee7b0377d4f.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/tables/d6b5e38e57b4d7a53ccfdf202d5e8c0e1f577aa864031790fd381ee7b0377d4f.jpg)

![ef48eb8819a7c3d689893b21ac6fd60c1666a68c79cc6474e2e41c37559d02a2.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/tables/ef48eb8819a7c3d689893b21ac6fd60c1666a68c79cc6474e2e41c37559d02a2.jpg)

![fa293a19fb74cc0ec316735956ffd857ca4f10e2ad7e1bb278cf98ffb09cebd2.jpg](../icml_results/2929_Probability%20of%20Matching%20for%20Batch%20Multi-Objective%20Bayesian%20Optimization/tables/fa293a19fb74cc0ec316735956ffd857ca4f10e2ad7e1bb278cf98ffb09cebd2.jpg)

<a id="paper-2892"></a>

## DiFA: Inference-Time Forward-Process Alignment for Diffusion Models


### Images

![226e544c264aa0add653568bf8282ebdf9c547f689352ae536277cbb4ad3ef98.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/images/226e544c264aa0add653568bf8282ebdf9c547f689352ae536277cbb4ad3ef98.jpg)

![2513915bed73685ca43f3c0431b7e7ed5cb8c7ecd435386ee914d6b4f1381e94.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/images/2513915bed73685ca43f3c0431b7e7ed5cb8c7ecd435386ee914d6b4f1381e94.jpg)

![5b343c902f75d3f0ab99a69244ac2ca9b09c61e6e24142170952d56d02a29c08.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/images/5b343c902f75d3f0ab99a69244ac2ca9b09c61e6e24142170952d56d02a29c08.jpg)

![65cece658d4b7e40c3f623093f0e24b5276fafd54e85de7ccf029fbb779a841e.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/images/65cece658d4b7e40c3f623093f0e24b5276fafd54e85de7ccf029fbb779a841e.jpg)

![6e83ebd1cbda3e98de9f04af4ad3739e2e6c0fc7869704d2dfce5c189eef0674.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/images/6e83ebd1cbda3e98de9f04af4ad3739e2e6c0fc7869704d2dfce5c189eef0674.jpg)

![747a55f515e5e9d47efdfeca3d99c48f98fd781ca91ff8c3c74e170e51697e0a.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/images/747a55f515e5e9d47efdfeca3d99c48f98fd781ca91ff8c3c74e170e51697e0a.jpg)

![de77e144d6fd773328fb5229e7683cbcf3daec586b82ebf0f5f26ac0bee89b64.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/images/de77e144d6fd773328fb5229e7683cbcf3daec586b82ebf0f5f26ac0bee89b64.jpg)

### Tables

![0e04f19a39570856952847318d0b61db139ce7aa52bb94c0e3049eddd4684865.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/tables/0e04f19a39570856952847318d0b61db139ce7aa52bb94c0e3049eddd4684865.jpg)

![1750db15fa3dbb8d54e1609c74009f855365db79c7ceefb460eb112eb607f03a.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/tables/1750db15fa3dbb8d54e1609c74009f855365db79c7ceefb460eb112eb607f03a.jpg)

![377edd3fcbd05cb977270ffd546871b25ceab50b40b53f24091a65a6db4038ee.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/tables/377edd3fcbd05cb977270ffd546871b25ceab50b40b53f24091a65a6db4038ee.jpg)

![504e9e9a8369698bcfcf835b7ad68a6fcfd84a3eab879c50f9d13d6a7d2de89e.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/tables/504e9e9a8369698bcfcf835b7ad68a6fcfd84a3eab879c50f9d13d6a7d2de89e.jpg)

![52a6c3c4c7d25e81a138df9fdbe0815c4fe74ff9a4a3179207e263f3350e5580.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/tables/52a6c3c4c7d25e81a138df9fdbe0815c4fe74ff9a4a3179207e263f3350e5580.jpg)

![618859387b0e4007907d750f8a6cde65e95ab03bd5d86fee491384dff7820bc5.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/tables/618859387b0e4007907d750f8a6cde65e95ab03bd5d86fee491384dff7820bc5.jpg)

![67f4309b3dd29a2f4c075ce0fd63052f5fb2514ecf9b35f493876f2174cdcd49.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/tables/67f4309b3dd29a2f4c075ce0fd63052f5fb2514ecf9b35f493876f2174cdcd49.jpg)

![7c5f06f32bb5cd958256436f5c7781e4ee469fb69e3ad282e0ae1a5cb72dc6bf.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/tables/7c5f06f32bb5cd958256436f5c7781e4ee469fb69e3ad282e0ae1a5cb72dc6bf.jpg)

![89c1da72247fa7982cb9f39d5f1f209d0ad327ec7ce0e44fb1fe4c86ff233caa.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/tables/89c1da72247fa7982cb9f39d5f1f209d0ad327ec7ce0e44fb1fe4c86ff233caa.jpg)

![9718e7ad2f5442fde3d8227f9b25c8b7aa27f7beb67bf04ad57e3c37c62e2217.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/tables/9718e7ad2f5442fde3d8227f9b25c8b7aa27f7beb67bf04ad57e3c37c62e2217.jpg)

![c20c7bfe90d2664933784d0278badc38d5653559165e5e2c847a367e45b6a27c.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/tables/c20c7bfe90d2664933784d0278badc38d5653559165e5e2c847a367e45b6a27c.jpg)

![c46e9744dcb43d644bd956f211f6733c0d98b24078426d6e662a7fff57cdeceb.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/tables/c46e9744dcb43d644bd956f211f6733c0d98b24078426d6e662a7fff57cdeceb.jpg)

![d0c621bfffe14e71b6e5ab6a3942b45e6623031b55d023b3021f202f3d18ee38.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/tables/d0c621bfffe14e71b6e5ab6a3942b45e6623031b55d023b3021f202f3d18ee38.jpg)

![e7c88fe8d8a7c71c57718a8a38e7249b9c0f3f332b8582ec980429f55dd185f7.jpg](../icml_results/2930_DiFA_%20Inference-Time%20Forward-Process%20Alignment%20for%20Diffusion%20Models/tables/e7c88fe8d8a7c71c57718a8a38e7249b9c0f3f332b8582ec980429f55dd185f7.jpg)

<a id="paper-2893"></a>

## AAD-1: Asymmetric Adversarial Distillation for One-Step Autoregressive Video Generation


### Images

![3b3bccf095407a43df1eb7dd3491bbf217a298702f57b365d3c908d6864fbd0f.jpg](../icml_results/2931_AAD-1_%20Asymmetric%20Adversarial%20Distillation%20for%20One-Step%20Autoregressive%20Video%20Generation/images/3b3bccf095407a43df1eb7dd3491bbf217a298702f57b365d3c908d6864fbd0f.jpg)

![400c4d2411ea37b5f7fbb32f93560cc11efeeec723066e16a8904d452540eae0.jpg](../icml_results/2931_AAD-1_%20Asymmetric%20Adversarial%20Distillation%20for%20One-Step%20Autoregressive%20Video%20Generation/images/400c4d2411ea37b5f7fbb32f93560cc11efeeec723066e16a8904d452540eae0.jpg)

![4ed456681a479d196c58dfd8a0c4efbda661557ce792a4252027ab8d6d5ef563.jpg](../icml_results/2931_AAD-1_%20Asymmetric%20Adversarial%20Distillation%20for%20One-Step%20Autoregressive%20Video%20Generation/images/4ed456681a479d196c58dfd8a0c4efbda661557ce792a4252027ab8d6d5ef563.jpg)

![5a2fcac832210d5a4191c733befc2dbecc132219fe352bbe60f8cdb7fb92d025.jpg](../icml_results/2931_AAD-1_%20Asymmetric%20Adversarial%20Distillation%20for%20One-Step%20Autoregressive%20Video%20Generation/images/5a2fcac832210d5a4191c733befc2dbecc132219fe352bbe60f8cdb7fb92d025.jpg)

![7fef3a55cc1bcb6cc40d71b469ea891819b5be03098baf253a0d6969a2b3d1e8.jpg](../icml_results/2931_AAD-1_%20Asymmetric%20Adversarial%20Distillation%20for%20One-Step%20Autoregressive%20Video%20Generation/images/7fef3a55cc1bcb6cc40d71b469ea891819b5be03098baf253a0d6969a2b3d1e8.jpg)

![93eb75257cf3761beeb8bb3d3ac2c4738da1efb8b430f8047654b679cfd0cdda.jpg](../icml_results/2931_AAD-1_%20Asymmetric%20Adversarial%20Distillation%20for%20One-Step%20Autoregressive%20Video%20Generation/images/93eb75257cf3761beeb8bb3d3ac2c4738da1efb8b430f8047654b679cfd0cdda.jpg)

![b109bc722edd2d2dbfe06da049654cbe35834040cff2975c9cf02b170fda80df.jpg](../icml_results/2931_AAD-1_%20Asymmetric%20Adversarial%20Distillation%20for%20One-Step%20Autoregressive%20Video%20Generation/images/b109bc722edd2d2dbfe06da049654cbe35834040cff2975c9cf02b170fda80df.jpg)

![c4cbf9f72a5031f386eaf7bbdc4da0b06bf485cdcd086476007aad8ada3d0d49.jpg](../icml_results/2931_AAD-1_%20Asymmetric%20Adversarial%20Distillation%20for%20One-Step%20Autoregressive%20Video%20Generation/images/c4cbf9f72a5031f386eaf7bbdc4da0b06bf485cdcd086476007aad8ada3d0d49.jpg)

![f415dfcf84e76b2125bf47b4f6bfc5d965534237b070b782d4ffaafd12280132.jpg](../icml_results/2931_AAD-1_%20Asymmetric%20Adversarial%20Distillation%20for%20One-Step%20Autoregressive%20Video%20Generation/images/f415dfcf84e76b2125bf47b4f6bfc5d965534237b070b782d4ffaafd12280132.jpg)

![fa9015206f1ff3113bc07fd49ca251b87228392c827766628b2f91f1e666be07.jpg](../icml_results/2931_AAD-1_%20Asymmetric%20Adversarial%20Distillation%20for%20One-Step%20Autoregressive%20Video%20Generation/images/fa9015206f1ff3113bc07fd49ca251b87228392c827766628b2f91f1e666be07.jpg)

### Tables

![07129937f7e7c74b0fbe53238f0ad0ce002e45984adbc5fb310a4075d3f636a9.jpg](../icml_results/2931_AAD-1_%20Asymmetric%20Adversarial%20Distillation%20for%20One-Step%20Autoregressive%20Video%20Generation/tables/07129937f7e7c74b0fbe53238f0ad0ce002e45984adbc5fb310a4075d3f636a9.jpg)

![833bb822ca2f2dffaf18effa3c66ef8f572ba8e6ea08968c10da443cc666c2b5.jpg](../icml_results/2931_AAD-1_%20Asymmetric%20Adversarial%20Distillation%20for%20One-Step%20Autoregressive%20Video%20Generation/tables/833bb822ca2f2dffaf18effa3c66ef8f572ba8e6ea08968c10da443cc666c2b5.jpg)

![c83a5dd96ecc28ae21f4bf472746300be73b9397415e17aa6339014e894e4035.jpg](../icml_results/2931_AAD-1_%20Asymmetric%20Adversarial%20Distillation%20for%20One-Step%20Autoregressive%20Video%20Generation/tables/c83a5dd96ecc28ae21f4bf472746300be73b9397415e17aa6339014e894e4035.jpg)

![cb7d9db598d9ae97cb8f9e8f05ba73813d5a829666bcfac573ddf2be3978ef2d.jpg](../icml_results/2931_AAD-1_%20Asymmetric%20Adversarial%20Distillation%20for%20One-Step%20Autoregressive%20Video%20Generation/tables/cb7d9db598d9ae97cb8f9e8f05ba73813d5a829666bcfac573ddf2be3978ef2d.jpg)

![de5bb24c1ca1ae3522b2f22507b9d6d946f9c2dabdab868752c390dbe60c44b4.jpg](../icml_results/2931_AAD-1_%20Asymmetric%20Adversarial%20Distillation%20for%20One-Step%20Autoregressive%20Video%20Generation/tables/de5bb24c1ca1ae3522b2f22507b9d6d946f9c2dabdab868752c390dbe60c44b4.jpg)

<a id="paper-2894"></a>

## Mitigating the Contractivity Trap in Diffusion ODEs via Stein Stabilization


### Images

![1edcfd36f81ac30c56728303963c1ca8ffccacd98e30d2dbb95ae099cb95bc43.jpg](../icml_results/2932_Mitigating%20the%20Contractivity%20Trap%20in%20Diffusion%20ODEs%20via%20Stein%20Stabilization/images/1edcfd36f81ac30c56728303963c1ca8ffccacd98e30d2dbb95ae099cb95bc43.jpg)

![2e04c7f10006dd14b3ecfc833fc465b0279f9220f7fdf83e1a0cab26a47392a7.jpg](../icml_results/2932_Mitigating%20the%20Contractivity%20Trap%20in%20Diffusion%20ODEs%20via%20Stein%20Stabilization/images/2e04c7f10006dd14b3ecfc833fc465b0279f9220f7fdf83e1a0cab26a47392a7.jpg)

![3323ad6e2acf2e74cc162efb168bf4cceae5fc834f210761eb17d1d6e55dc3da.jpg](../icml_results/2932_Mitigating%20the%20Contractivity%20Trap%20in%20Diffusion%20ODEs%20via%20Stein%20Stabilization/images/3323ad6e2acf2e74cc162efb168bf4cceae5fc834f210761eb17d1d6e55dc3da.jpg)

![4ebf1deaf89dffc63251fe5b642f7e06f5db4c43ad934681de7066db69549793.jpg](../icml_results/2932_Mitigating%20the%20Contractivity%20Trap%20in%20Diffusion%20ODEs%20via%20Stein%20Stabilization/images/4ebf1deaf89dffc63251fe5b642f7e06f5db4c43ad934681de7066db69549793.jpg)

![54176dce166a320c6e9fdb5df18274c560794bfaa1ad1e3d4bc9c694914b5961.jpg](../icml_results/2932_Mitigating%20the%20Contractivity%20Trap%20in%20Diffusion%20ODEs%20via%20Stein%20Stabilization/images/54176dce166a320c6e9fdb5df18274c560794bfaa1ad1e3d4bc9c694914b5961.jpg)

![5adbefc348072557efee1936ec3974b8e3799ca0b3f163f39c34fc771e82125c.jpg](../icml_results/2932_Mitigating%20the%20Contractivity%20Trap%20in%20Diffusion%20ODEs%20via%20Stein%20Stabilization/images/5adbefc348072557efee1936ec3974b8e3799ca0b3f163f39c34fc771e82125c.jpg)

![7265b1f24a82dd63f058e313c5cd3ab339a2ddb4a384088f755401e76e646408.jpg](../icml_results/2932_Mitigating%20the%20Contractivity%20Trap%20in%20Diffusion%20ODEs%20via%20Stein%20Stabilization/images/7265b1f24a82dd63f058e313c5cd3ab339a2ddb4a384088f755401e76e646408.jpg)

![7c059ac882d2bddb89f4fcbbea26642a33e05c1f6d468821b6b10060d93c95da.jpg](../icml_results/2932_Mitigating%20the%20Contractivity%20Trap%20in%20Diffusion%20ODEs%20via%20Stein%20Stabilization/images/7c059ac882d2bddb89f4fcbbea26642a33e05c1f6d468821b6b10060d93c95da.jpg)

![7eb173801af55f00c654f8f4c01b1a6728f9cde22c27bcc7ca0d6cedf1a2f91c.jpg](../icml_results/2932_Mitigating%20the%20Contractivity%20Trap%20in%20Diffusion%20ODEs%20via%20Stein%20Stabilization/images/7eb173801af55f00c654f8f4c01b1a6728f9cde22c27bcc7ca0d6cedf1a2f91c.jpg)

![838e155f285d8e3248e5718f128353a48a98c252a6620d599d43ea02e7f7a33c.jpg](../icml_results/2932_Mitigating%20the%20Contractivity%20Trap%20in%20Diffusion%20ODEs%20via%20Stein%20Stabilization/images/838e155f285d8e3248e5718f128353a48a98c252a6620d599d43ea02e7f7a33c.jpg)

![beb40fd0ff6cccbdffc779ea3a32ab11010bce86f5fdaa5d91e28d2b6e65addd.jpg](../icml_results/2932_Mitigating%20the%20Contractivity%20Trap%20in%20Diffusion%20ODEs%20via%20Stein%20Stabilization/images/beb40fd0ff6cccbdffc779ea3a32ab11010bce86f5fdaa5d91e28d2b6e65addd.jpg)

![ec71fefff40faa6a7513aae6a743fad73a66df7b58e52e298a92df3e0a4a7e7b.jpg](../icml_results/2932_Mitigating%20the%20Contractivity%20Trap%20in%20Diffusion%20ODEs%20via%20Stein%20Stabilization/images/ec71fefff40faa6a7513aae6a743fad73a66df7b58e52e298a92df3e0a4a7e7b.jpg)

### Tables

![9068381391cecae4a301c17ac371b21a40cd0e88a90c48a7e4df35b49e295df4.jpg](../icml_results/2932_Mitigating%20the%20Contractivity%20Trap%20in%20Diffusion%20ODEs%20via%20Stein%20Stabilization/tables/9068381391cecae4a301c17ac371b21a40cd0e88a90c48a7e4df35b49e295df4.jpg)

![a73598f5aafded7a2c836cf363d35ba8e38782583e708fdcfb9e65ff6d2f7a7d.jpg](../icml_results/2932_Mitigating%20the%20Contractivity%20Trap%20in%20Diffusion%20ODEs%20via%20Stein%20Stabilization/tables/a73598f5aafded7a2c836cf363d35ba8e38782583e708fdcfb9e65ff6d2f7a7d.jpg)

![c30d82c2300835ca134c21668229084859802e44514a6b2e63b1bff4e71e8a34.jpg](../icml_results/2932_Mitigating%20the%20Contractivity%20Trap%20in%20Diffusion%20ODEs%20via%20Stein%20Stabilization/tables/c30d82c2300835ca134c21668229084859802e44514a6b2e63b1bff4e71e8a34.jpg)

<a id="paper-2895"></a>

## Eliminating Solution Bias in Differentially Private Optimization


### Images

![7ac7fd4aeeb9fb9fc2f090824fd23b325c1061a77525e324185ae7cadee50367.jpg](../icml_results/2933_Eliminating%20Solution%20Bias%20in%20Differentially%20Private%20Optimization/images/7ac7fd4aeeb9fb9fc2f090824fd23b325c1061a77525e324185ae7cadee50367.jpg)

![b443821aa65e3e2c73c6710855c5505dbf727ad14d76cdf33f9f5c1355f13678.jpg](../icml_results/2933_Eliminating%20Solution%20Bias%20in%20Differentially%20Private%20Optimization/images/b443821aa65e3e2c73c6710855c5505dbf727ad14d76cdf33f9f5c1355f13678.jpg)

![f26d65b732aa3619854ea117975dfd27cadb9dd176dd65e27a706ef2cd659751.jpg](../icml_results/2933_Eliminating%20Solution%20Bias%20in%20Differentially%20Private%20Optimization/images/f26d65b732aa3619854ea117975dfd27cadb9dd176dd65e27a706ef2cd659751.jpg)

![f80afd235c69fc1e31bb012cbffc47c235c38da98898f41f318ea9b3e9b7e4b4.jpg](../icml_results/2933_Eliminating%20Solution%20Bias%20in%20Differentially%20Private%20Optimization/images/f80afd235c69fc1e31bb012cbffc47c235c38da98898f41f318ea9b3e9b7e4b4.jpg)

### Tables

![33328736dbe92da99d336835947921d05b0a35743657387a4a59ad3a51b9cf9e.jpg](../icml_results/2933_Eliminating%20Solution%20Bias%20in%20Differentially%20Private%20Optimization/tables/33328736dbe92da99d336835947921d05b0a35743657387a4a59ad3a51b9cf9e.jpg)

![6a3adb68d9321d2620d9d35f7359e32d2f467a084fb97a9aeb7f052358e4002e.jpg](../icml_results/2933_Eliminating%20Solution%20Bias%20in%20Differentially%20Private%20Optimization/tables/6a3adb68d9321d2620d9d35f7359e32d2f467a084fb97a9aeb7f052358e4002e.jpg)

![6c611507613bf1c328eb589cde860e799f1123d923a9f7c63984f442d967731b.jpg](../icml_results/2933_Eliminating%20Solution%20Bias%20in%20Differentially%20Private%20Optimization/tables/6c611507613bf1c328eb589cde860e799f1123d923a9f7c63984f442d967731b.jpg)

![6cbca0034618bb7ea9ad6fdbe1398358d4ed7c57fdfa513bd6436421d82352a0.jpg](../icml_results/2933_Eliminating%20Solution%20Bias%20in%20Differentially%20Private%20Optimization/tables/6cbca0034618bb7ea9ad6fdbe1398358d4ed7c57fdfa513bd6436421d82352a0.jpg)

![7d0048f9d5da86ad3e21d78ac42c0898bfe22c2b148d632cbc778a852b13b071.jpg](../icml_results/2933_Eliminating%20Solution%20Bias%20in%20Differentially%20Private%20Optimization/tables/7d0048f9d5da86ad3e21d78ac42c0898bfe22c2b148d632cbc778a852b13b071.jpg)

![86c6c08e14ba764fc031756ac63fe6e97a009cfd41e24cd81cbd47ed06bf1c3e.jpg](../icml_results/2933_Eliminating%20Solution%20Bias%20in%20Differentially%20Private%20Optimization/tables/86c6c08e14ba764fc031756ac63fe6e97a009cfd41e24cd81cbd47ed06bf1c3e.jpg)

![a32217285f45c0e850bfbfbf91e562f978067d2dd050d45b041cfba65ef103af.jpg](../icml_results/2933_Eliminating%20Solution%20Bias%20in%20Differentially%20Private%20Optimization/tables/a32217285f45c0e850bfbfbf91e562f978067d2dd050d45b041cfba65ef103af.jpg)

![b5ea2d634483e3e43e6b2990c6663ea7bf29444e6dc612164c2a84514c1d43e0.jpg](../icml_results/2933_Eliminating%20Solution%20Bias%20in%20Differentially%20Private%20Optimization/tables/b5ea2d634483e3e43e6b2990c6663ea7bf29444e6dc612164c2a84514c1d43e0.jpg)

![ca8737610de2f2d6e8c310997dba011a88fc53f8ff6c8e147041f83ba20fa4bd.jpg](../icml_results/2933_Eliminating%20Solution%20Bias%20in%20Differentially%20Private%20Optimization/tables/ca8737610de2f2d6e8c310997dba011a88fc53f8ff6c8e147041f83ba20fa4bd.jpg)

![ef58d9875e0d5e43040fe15d809b047f4ffec74f8d68ca6113cfe9cdf66bd341.jpg](../icml_results/2933_Eliminating%20Solution%20Bias%20in%20Differentially%20Private%20Optimization/tables/ef58d9875e0d5e43040fe15d809b047f4ffec74f8d68ca6113cfe9cdf66bd341.jpg)

<a id="paper-2896"></a>

## Revisiting Photometric Ambiguity for Accurate Gaussian-Splatting Surface Reconstruction


### Images

![0ae23c2771d11bd968923b13dfd533df81ec3dba92beb590df9c8e27883caa09.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/0ae23c2771d11bd968923b13dfd533df81ec3dba92beb590df9c8e27883caa09.jpg)

![0f6e0203e2e737e4110f21ca84edca1e62fa894fe6fe145344a5b9b99e31530d.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/0f6e0203e2e737e4110f21ca84edca1e62fa894fe6fe145344a5b9b99e31530d.jpg)

![1bd43797f5ce3259df321a0f90615af9f80f8f90c9f503f69a0d179b46579e4d.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/1bd43797f5ce3259df321a0f90615af9f80f8f90c9f503f69a0d179b46579e4d.jpg)

![1d758530db08d1a55887b76415bc362a3662b5326d420c838c7202a4a9a6b2c3.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/1d758530db08d1a55887b76415bc362a3662b5326d420c838c7202a4a9a6b2c3.jpg)

![3d57b197110208a4af4527769a3faf7a2495a872a3f43fa73404f79e2175fb89.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/3d57b197110208a4af4527769a3faf7a2495a872a3f43fa73404f79e2175fb89.jpg)

![501bff59dbde703bdca30c1aba0fd4584caa522fd813a3b62b69bf55eef526bd.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/501bff59dbde703bdca30c1aba0fd4584caa522fd813a3b62b69bf55eef526bd.jpg)

![50d985090ecb816f1718a1dc3481d7446a53a2a537980d4df1f3b8b922d2043e.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/50d985090ecb816f1718a1dc3481d7446a53a2a537980d4df1f3b8b922d2043e.jpg)

![5335902806bf33b9ca727d74d4e048c7a70395e2cdb1a437ac0317dc47e55c0d.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/5335902806bf33b9ca727d74d4e048c7a70395e2cdb1a437ac0317dc47e55c0d.jpg)

![5e88a9fa99b1c880b7ec623a3d0bb59754bfae0bb2f64537fcb5312c0c68cc53.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/5e88a9fa99b1c880b7ec623a3d0bb59754bfae0bb2f64537fcb5312c0c68cc53.jpg)

![8cc7a760d021960c75fbdc9735d677cdbc786c5dc3546e32e9dd7f9dd851471c.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/8cc7a760d021960c75fbdc9735d677cdbc786c5dc3546e32e9dd7f9dd851471c.jpg)

![8fc2e8d102b386316dbf43e2b352ad9cdc98d2f58d1cae225a08dbc0edfd42d5.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/8fc2e8d102b386316dbf43e2b352ad9cdc98d2f58d1cae225a08dbc0edfd42d5.jpg)

![98379f222329536ebd6fddf73b396f4cb2edfe5a87531c4868fde47aebf45b5a.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/98379f222329536ebd6fddf73b396f4cb2edfe5a87531c4868fde47aebf45b5a.jpg)

![9afd7c65c4b35abb73fa82c4e996c12edcff31df69d48e353036fa8d5105cbcd.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/9afd7c65c4b35abb73fa82c4e996c12edcff31df69d48e353036fa8d5105cbcd.jpg)

![ad714dc4715ae2e44ce6d14b5b949c2843bf4c4e233464a368a4313974ae3866.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/ad714dc4715ae2e44ce6d14b5b949c2843bf4c4e233464a368a4313974ae3866.jpg)

![c117686476462bdb491933a02c005f4bc279ed39814319e29a699f374c4a375b.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/c117686476462bdb491933a02c005f4bc279ed39814319e29a699f374c4a375b.jpg)

![c3911fc259a2b2140aa0c2916757a75eb58b1919bc47e1625f8f9b7ef1c921e1.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/c3911fc259a2b2140aa0c2916757a75eb58b1919bc47e1625f8f9b7ef1c921e1.jpg)

![c9302caab8cadf476ba160531f2bca8d273e4cedae385b683adc51c8f6e47ee8.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/c9302caab8cadf476ba160531f2bca8d273e4cedae385b683adc51c8f6e47ee8.jpg)

![e33a82863a58e14a3512685a9ae7d17a14ee888e78daf62795a02f4ceb447102.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/e33a82863a58e14a3512685a9ae7d17a14ee888e78daf62795a02f4ceb447102.jpg)

![e3cf40ce9d6b04e523a9dead1523f8d276057e37dde9905d15ade70ea8722e7b.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/images/e3cf40ce9d6b04e523a9dead1523f8d276057e37dde9905d15ade70ea8722e7b.jpg)

### Tables

![0c5a64248dd39506a691263418e1c85a6138a339ddc92dcc978e6ba82f498051.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/tables/0c5a64248dd39506a691263418e1c85a6138a339ddc92dcc978e6ba82f498051.jpg)

![20f0e448dfc5086fa122922f062a4a71d6cd04f47a115320efebb3fb146be5a5.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/tables/20f0e448dfc5086fa122922f062a4a71d6cd04f47a115320efebb3fb146be5a5.jpg)

![21144441bff623850036460bcc671663f5ce590fb378fb8965fe79a9630117be.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/tables/21144441bff623850036460bcc671663f5ce590fb378fb8965fe79a9630117be.jpg)

![3d58b994f1894c3e303a279b795652e4f8a7700d8494b9dc7262d5ef7b3851a3.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/tables/3d58b994f1894c3e303a279b795652e4f8a7700d8494b9dc7262d5ef7b3851a3.jpg)

![45f70222b7038c88b561b4e8d40114dabb7a8054cdd2ade3af0523f59f0ae517.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/tables/45f70222b7038c88b561b4e8d40114dabb7a8054cdd2ade3af0523f59f0ae517.jpg)

![6aa4edb0dd0876b35de0fd040b89f397c86687b1b55186161deb0d8d7e85da9c.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/tables/6aa4edb0dd0876b35de0fd040b89f397c86687b1b55186161deb0d8d7e85da9c.jpg)

![77e674d718a4265ec68e5d1277df8164af9f543ce79cc9c45eca0ed3df0c8d50.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/tables/77e674d718a4265ec68e5d1277df8164af9f543ce79cc9c45eca0ed3df0c8d50.jpg)

![8c7f6194143738c2477ec296f6bf7071c875fc4bae7513a1c948ceb179d376c0.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/tables/8c7f6194143738c2477ec296f6bf7071c875fc4bae7513a1c948ceb179d376c0.jpg)

![d86cda39769ac86689fbfddbb0dc65c20b1a15ce07e1a99a42aae133375c81ad.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/tables/d86cda39769ac86689fbfddbb0dc65c20b1a15ce07e1a99a42aae133375c81ad.jpg)

![f2dae43284be1df5329bfcbfd249b26bfaa6f95f82114ef6f530486c4937a8cc.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/tables/f2dae43284be1df5329bfcbfd249b26bfaa6f95f82114ef6f530486c4937a8cc.jpg)

![f8478584bbd5358fd6f0066a640d809f8ede7152d310e5de2504df1284dd9d2a.jpg](../icml_results/2934_Revisiting%20Photometric%20Ambiguity%20for%20Accurate%20Gaussian-Splatting%20Surface%20Reconstruction/tables/f8478584bbd5358fd6f0066a640d809f8ede7152d310e5de2504df1284dd9d2a.jpg)

<a id="paper-2897"></a>

## Causal Flow Q-Learning for Robust Offline Reinforcement Learning


### Images

![0edee3eb3b99cab061ed6fe5b1dda0b45293f2f92d1f3cf13d5cf368329bb373.jpg](../icml_results/2935_Causal%20Flow%20Q-Learning%20for%20Robust%20Offline%20Reinforcement%20Learning/images/0edee3eb3b99cab061ed6fe5b1dda0b45293f2f92d1f3cf13d5cf368329bb373.jpg)

![0f908cacf40e724dc701a2ad62c356dbbd1462462d9474f5eb76b29764866070.jpg](../icml_results/2935_Causal%20Flow%20Q-Learning%20for%20Robust%20Offline%20Reinforcement%20Learning/images/0f908cacf40e724dc701a2ad62c356dbbd1462462d9474f5eb76b29764866070.jpg)

![368cc294c391683d42411ba568404daca007c9a0a0a4f3930a54600204f57e2e.jpg](../icml_results/2935_Causal%20Flow%20Q-Learning%20for%20Robust%20Offline%20Reinforcement%20Learning/images/368cc294c391683d42411ba568404daca007c9a0a0a4f3930a54600204f57e2e.jpg)

![60062713463d0e6f00fcbf200862cfe27746b6e5831251bf15f4d679d39a12e9.jpg](../icml_results/2935_Causal%20Flow%20Q-Learning%20for%20Robust%20Offline%20Reinforcement%20Learning/images/60062713463d0e6f00fcbf200862cfe27746b6e5831251bf15f4d679d39a12e9.jpg)

![bb2ada2532ef42b4e9af870af3509601f3af5a020aeb59bfa07c14a8f62fdce5.jpg](../icml_results/2935_Causal%20Flow%20Q-Learning%20for%20Robust%20Offline%20Reinforcement%20Learning/images/bb2ada2532ef42b4e9af870af3509601f3af5a020aeb59bfa07c14a8f62fdce5.jpg)

![bd6a1c050666ccca5b6d9fd67103cb443f5cb02c47fdf2e6698fbb1b1a64c1d7.jpg](../icml_results/2935_Causal%20Flow%20Q-Learning%20for%20Robust%20Offline%20Reinforcement%20Learning/images/bd6a1c050666ccca5b6d9fd67103cb443f5cb02c47fdf2e6698fbb1b1a64c1d7.jpg)

![e32426d1ca4bd61253ddee6fc04a172d85da5bafabcb8395920c7c52e811d13c.jpg](../icml_results/2935_Causal%20Flow%20Q-Learning%20for%20Robust%20Offline%20Reinforcement%20Learning/images/e32426d1ca4bd61253ddee6fc04a172d85da5bafabcb8395920c7c52e811d13c.jpg)

### Tables

![2fdc0bd7a3bcab12ff6ce8c0a11daaf6162991ccbd9d2594d71bf44532b0c48e.jpg](../icml_results/2935_Causal%20Flow%20Q-Learning%20for%20Robust%20Offline%20Reinforcement%20Learning/tables/2fdc0bd7a3bcab12ff6ce8c0a11daaf6162991ccbd9d2594d71bf44532b0c48e.jpg)

![4910b787641d99159a146be1ccc2fe7ba385e3e20d3035eb4197690c06dd3bdb.jpg](../icml_results/2935_Causal%20Flow%20Q-Learning%20for%20Robust%20Offline%20Reinforcement%20Learning/tables/4910b787641d99159a146be1ccc2fe7ba385e3e20d3035eb4197690c06dd3bdb.jpg)

![779d4e21602e7231af970559f061f32f76778c2bd1d4ad25e88a152c435985fb.jpg](../icml_results/2935_Causal%20Flow%20Q-Learning%20for%20Robust%20Offline%20Reinforcement%20Learning/tables/779d4e21602e7231af970559f061f32f76778c2bd1d4ad25e88a152c435985fb.jpg)

<a id="paper-2898"></a>

## Characterizing the Effect of Noise in Language Generation in the Limit


### Tables

![e62b3ee33a866da46bb2a6b917dab1a0b9e017b7d575a815abd743dd7c8d4110.jpg](../icml_results/2936_Characterizing%20the%20Effect%20of%20Noise%20in%20Language%20Generation%20in%20the%20Limit/tables/e62b3ee33a866da46bb2a6b917dab1a0b9e017b7d575a815abd743dd7c8d4110.jpg)

<a id="paper-2899"></a>

## StitchCUDA: An Automated Multi-Agents End-to-End GPU Programing Framework with Rubric-based Agentic Reinforcement Learning


### Images

![0e713989d1c0510ad37dffb33d3ec5790a66a10413c501fbe870264be497414d.jpg](../icml_results/2937_StitchCUDA_%20An%20Automated%20Multi-Agents%20End-to-End%20GPU%20Programing%20Framework%20with%20Rubric-based%20Agentic%20/images/0e713989d1c0510ad37dffb33d3ec5790a66a10413c501fbe870264be497414d.jpg)

![35a8521813269e19b2723c97c37d55d5b6cea639d77548480425858eb1a3bb59.jpg](../icml_results/2937_StitchCUDA_%20An%20Automated%20Multi-Agents%20End-to-End%20GPU%20Programing%20Framework%20with%20Rubric-based%20Agentic%20/images/35a8521813269e19b2723c97c37d55d5b6cea639d77548480425858eb1a3bb59.jpg)

![4747cdc736799e24a790a5fe9fa9c63cd719cc6c31a13f54002b39a9a8bb44a7.jpg](../icml_results/2937_StitchCUDA_%20An%20Automated%20Multi-Agents%20End-to-End%20GPU%20Programing%20Framework%20with%20Rubric-based%20Agentic%20/images/4747cdc736799e24a790a5fe9fa9c63cd719cc6c31a13f54002b39a9a8bb44a7.jpg)

![5c3bcc699109289c08af7b4ff7e2ebce25660c942df6bed53615d5626269bb64.jpg](../icml_results/2937_StitchCUDA_%20An%20Automated%20Multi-Agents%20End-to-End%20GPU%20Programing%20Framework%20with%20Rubric-based%20Agentic%20/images/5c3bcc699109289c08af7b4ff7e2ebce25660c942df6bed53615d5626269bb64.jpg)

![60d93fa96af1095737f5df6bb4860ff7f23b151fb6964681d1a05555fa839251.jpg](../icml_results/2937_StitchCUDA_%20An%20Automated%20Multi-Agents%20End-to-End%20GPU%20Programing%20Framework%20with%20Rubric-based%20Agentic%20/images/60d93fa96af1095737f5df6bb4860ff7f23b151fb6964681d1a05555fa839251.jpg)

![709885ecf435af035f40cbb88ad4d766cc09718375e998309bd35eeb35ecced5.jpg](../icml_results/2937_StitchCUDA_%20An%20Automated%20Multi-Agents%20End-to-End%20GPU%20Programing%20Framework%20with%20Rubric-based%20Agentic%20/images/709885ecf435af035f40cbb88ad4d766cc09718375e998309bd35eeb35ecced5.jpg)

![ad2e999a10e2946915c4a83bafdc578e4076bae1a14f8e56d5c926e8b61f885c.jpg](../icml_results/2937_StitchCUDA_%20An%20Automated%20Multi-Agents%20End-to-End%20GPU%20Programing%20Framework%20with%20Rubric-based%20Agentic%20/images/ad2e999a10e2946915c4a83bafdc578e4076bae1a14f8e56d5c926e8b61f885c.jpg)

![c17419b65feeebd5b3ad558ab11334cb20b27bd249c2eef9769302954a031acb.jpg](../icml_results/2937_StitchCUDA_%20An%20Automated%20Multi-Agents%20End-to-End%20GPU%20Programing%20Framework%20with%20Rubric-based%20Agentic%20/images/c17419b65feeebd5b3ad558ab11334cb20b27bd249c2eef9769302954a031acb.jpg)

![d6809f32cf72abff60be3761a0e542a79d0fcc97bf3d17abc293789e4919eb1a.jpg](../icml_results/2937_StitchCUDA_%20An%20Automated%20Multi-Agents%20End-to-End%20GPU%20Programing%20Framework%20with%20Rubric-based%20Agentic%20/images/d6809f32cf72abff60be3761a0e542a79d0fcc97bf3d17abc293789e4919eb1a.jpg)

### Tables

![361c58c85953c4ee8e9d5d654f05aef42a2c29a47985e99020ec2863790c70bd.jpg](../icml_results/2937_StitchCUDA_%20An%20Automated%20Multi-Agents%20End-to-End%20GPU%20Programing%20Framework%20with%20Rubric-based%20Agentic%20/tables/361c58c85953c4ee8e9d5d654f05aef42a2c29a47985e99020ec2863790c70bd.jpg)

![4e448a41b80a95699dfbd3b9cb79dfb9c9d1516e67233bd69de37de9864ce46e.jpg](../icml_results/2937_StitchCUDA_%20An%20Automated%20Multi-Agents%20End-to-End%20GPU%20Programing%20Framework%20with%20Rubric-based%20Agentic%20/tables/4e448a41b80a95699dfbd3b9cb79dfb9c9d1516e67233bd69de37de9864ce46e.jpg)

![5f35419e3794af2e702e2e55820a51eb793f07cec058aeeaef2fcd353b0ac32f.jpg](../icml_results/2937_StitchCUDA_%20An%20Automated%20Multi-Agents%20End-to-End%20GPU%20Programing%20Framework%20with%20Rubric-based%20Agentic%20/tables/5f35419e3794af2e702e2e55820a51eb793f07cec058aeeaef2fcd353b0ac32f.jpg)

![a6b6b17add3a4adb2fd3d95b556018c088b40ad30e7a141cfe34c773c1ff2a9e.jpg](../icml_results/2937_StitchCUDA_%20An%20Automated%20Multi-Agents%20End-to-End%20GPU%20Programing%20Framework%20with%20Rubric-based%20Agentic%20/tables/a6b6b17add3a4adb2fd3d95b556018c088b40ad30e7a141cfe34c773c1ff2a9e.jpg)

![be6c308d10b51c73da8ea919ac5ad1b8375a05ffe1ef1f5bd09468a0982132bb.jpg](../icml_results/2937_StitchCUDA_%20An%20Automated%20Multi-Agents%20End-to-End%20GPU%20Programing%20Framework%20with%20Rubric-based%20Agentic%20/tables/be6c308d10b51c73da8ea919ac5ad1b8375a05ffe1ef1f5bd09468a0982132bb.jpg)

![e23ba5f85b7c8a2fd11463950678dc0dd6a361d331e12963cfeab4adc8fb8ec4.jpg](../icml_results/2937_StitchCUDA_%20An%20Automated%20Multi-Agents%20End-to-End%20GPU%20Programing%20Framework%20with%20Rubric-based%20Agentic%20/tables/e23ba5f85b7c8a2fd11463950678dc0dd6a361d331e12963cfeab4adc8fb8ec4.jpg)

<a id="paper-2900"></a>

## PRAC: Principal-Random Subspace for LLM Activation Compression and Memory-Efficient Training


### Images

![4432eb79b6e75c21047ee73bbb5d59ff4c29e3e37e557e2dd0a9091ac8630ab7.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/images/4432eb79b6e75c21047ee73bbb5d59ff4c29e3e37e557e2dd0a9091ac8630ab7.jpg)

![73a68c77097a95e12723d6db3abb078f0e5877ff77fe3e930a311f8500d7542a.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/images/73a68c77097a95e12723d6db3abb078f0e5877ff77fe3e930a311f8500d7542a.jpg)

![75969b124aa7feb9285d1273640456d385084527359d233fdb6762027f99788f.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/images/75969b124aa7feb9285d1273640456d385084527359d233fdb6762027f99788f.jpg)

![78febe08f7f1c0c94e9e7a8703e1396f6eea3250e7142f2470798aa5ed7ed667.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/images/78febe08f7f1c0c94e9e7a8703e1396f6eea3250e7142f2470798aa5ed7ed667.jpg)

![7b3fc476f4c7381c5dc4b914b22ca9890281d02f6b68975ecb5e336e8cd884c2.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/images/7b3fc476f4c7381c5dc4b914b22ca9890281d02f6b68975ecb5e336e8cd884c2.jpg)

![884fb258f83f22e45287593eec237ed7262b0f9f148d533193f2f393f897f183.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/images/884fb258f83f22e45287593eec237ed7262b0f9f148d533193f2f393f897f183.jpg)

![9bde1420514265fbb6b3c3668d5d4a1efba81b3601e6e115bae881d676a9fcb3.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/images/9bde1420514265fbb6b3c3668d5d4a1efba81b3601e6e115bae881d676a9fcb3.jpg)

![c412412384874fe63a84b87dc0795bdc2faf43c24992af70c9acce25f5762697.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/images/c412412384874fe63a84b87dc0795bdc2faf43c24992af70c9acce25f5762697.jpg)

![cff0d5875d3e09e45ad67a061ade11a935efacc1f8b733b3e129eb5d87ed2590.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/images/cff0d5875d3e09e45ad67a061ade11a935efacc1f8b733b3e129eb5d87ed2590.jpg)

### Tables

![2104470d510c427ab20d1db9855b8547155106045826cec818f21781dd08e061.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/tables/2104470d510c427ab20d1db9855b8547155106045826cec818f21781dd08e061.jpg)

![2c1466b10291ce45906aec3f23cfaeecb0ef3786f46c492318737df63f87ae11.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/tables/2c1466b10291ce45906aec3f23cfaeecb0ef3786f46c492318737df63f87ae11.jpg)

![595d2ce126bad581a68f9a4e91e9290be3bc0c38d84e1d34c7555ff19907eb73.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/tables/595d2ce126bad581a68f9a4e91e9290be3bc0c38d84e1d34c7555ff19907eb73.jpg)

![5b1cebb8730695c7d3a57fae0c51728e540e76c22aa87dee89b07ffb5892b008.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/tables/5b1cebb8730695c7d3a57fae0c51728e540e76c22aa87dee89b07ffb5892b008.jpg)

![5ba1ce1a9b5448090b035f033cc524a6d3487d309fd61a5093cd4040e92a8da9.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/tables/5ba1ce1a9b5448090b035f033cc524a6d3487d309fd61a5093cd4040e92a8da9.jpg)

![63bbc264d78735d78ee321b0733c36edac528b0ef381fe8afa675630a3cf2094.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/tables/63bbc264d78735d78ee321b0733c36edac528b0ef381fe8afa675630a3cf2094.jpg)

![66e4fb5884847627cef8e1f9db79426b8e2759ef16bb3a5cfcce4cda35420cd2.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/tables/66e4fb5884847627cef8e1f9db79426b8e2759ef16bb3a5cfcce4cda35420cd2.jpg)

![8bfea889a5969813bc470e50b2d6c54371016044a1f3785a9a844b054770ed41.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/tables/8bfea889a5969813bc470e50b2d6c54371016044a1f3785a9a844b054770ed41.jpg)

![90c44225481e69b1158bf1417bdee907f0934ada3ce7c11776d7a8e53301a912.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/tables/90c44225481e69b1158bf1417bdee907f0934ada3ce7c11776d7a8e53301a912.jpg)

![b3d33194e60f08db93da10c7ff1aef891e6f2d385fda54f0935b8aab77122854.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/tables/b3d33194e60f08db93da10c7ff1aef891e6f2d385fda54f0935b8aab77122854.jpg)

![d77cc5fc8334593be6aad4a1bdd2c622c3afaf2b8fdb6c8cd86840bc5b6eb11d.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/tables/d77cc5fc8334593be6aad4a1bdd2c622c3afaf2b8fdb6c8cd86840bc5b6eb11d.jpg)

![e5d15586f8a1879530b570c2b9e42b90e3b0df307cf482bb2e0337c268ebb11e.jpg](../icml_results/2938_PRAC_%20Principal-Random%20Subspace%20for%20LLM%20Activation%20Compression%20and%20Memory-Efficient%20Training/tables/e5d15586f8a1879530b570c2b9e42b90e3b0df307cf482bb2e0337c268ebb11e.jpg)

<a id="paper-2901"></a>

## From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation


### Images

![12f9b79ec74a099930af4198d1b5ec435da8f389e82b8481ef3edb75108af360.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/images/12f9b79ec74a099930af4198d1b5ec435da8f389e82b8481ef3edb75108af360.jpg)

![139123dbf9a216f9da45d8f3b1d363337acba53ad9095f692769aec93d8a4bdb.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/images/139123dbf9a216f9da45d8f3b1d363337acba53ad9095f692769aec93d8a4bdb.jpg)

![18595cf07308e918d990a4c23edcf0067a1cd5e21d75a6af7a201b0d912bae7d.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/images/18595cf07308e918d990a4c23edcf0067a1cd5e21d75a6af7a201b0d912bae7d.jpg)

![259be8f09c09fda65495ecae3d02673d9a8d66cfbc3d574edd368603940ef1f4.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/images/259be8f09c09fda65495ecae3d02673d9a8d66cfbc3d574edd368603940ef1f4.jpg)

![2d94b4f340c9b2c8f95246ab75244c6b5a8070041e392389e29e7afaa4ae6809.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/images/2d94b4f340c9b2c8f95246ab75244c6b5a8070041e392389e29e7afaa4ae6809.jpg)

![3d1faf9e3320141087b921e600eeb8650d4c636d2fc7734df357528df4e8709c.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/images/3d1faf9e3320141087b921e600eeb8650d4c636d2fc7734df357528df4e8709c.jpg)

![486aca880b6ec90df446ebc236780b5a7838e6a67a8f91f00094785ad5e50614.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/images/486aca880b6ec90df446ebc236780b5a7838e6a67a8f91f00094785ad5e50614.jpg)

![50f6dee20f0cd543b24e9b50708f3cbf7ca11dacc62d378cd7918bd0382aa7b6.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/images/50f6dee20f0cd543b24e9b50708f3cbf7ca11dacc62d378cd7918bd0382aa7b6.jpg)

![5afc753386bdcd046d3f8b0660dc6378b2552bc3f17ab1f57676a5b59a6e1a57.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/images/5afc753386bdcd046d3f8b0660dc6378b2552bc3f17ab1f57676a5b59a6e1a57.jpg)

![6a40ad297dca33af5b870582f9e34c9ec6c32e1ca9c5d7ae2deb2e7325fb789b.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/images/6a40ad297dca33af5b870582f9e34c9ec6c32e1ca9c5d7ae2deb2e7325fb789b.jpg)

![7ca51128e239884cea8fece8b249a633a9b4bec1b728a8328a3eab49fc5da110.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/images/7ca51128e239884cea8fece8b249a633a9b4bec1b728a8328a3eab49fc5da110.jpg)

![b255def2055f4535b9a0b9156176443793809d133f407c22d2d935c07c89bd74.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/images/b255def2055f4535b9a0b9156176443793809d133f407c22d2d935c07c89bd74.jpg)

![c68f15ffe1dd3ce2cf05305f9af335411783e446473a4136c06ccb3a6b7bd23f.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/images/c68f15ffe1dd3ce2cf05305f9af335411783e446473a4136c06ccb3a6b7bd23f.jpg)

![e3ebd0b03357d685440c6e3e7a1d5baf9ea5391786ae7b7377d78ce2e42850fa.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/images/e3ebd0b03357d685440c6e3e7a1d5baf9ea5391786ae7b7377d78ce2e42850fa.jpg)

![e7a702a1bbf288ab315bd88ddde5df01efa17795880333e7ac224ee9831d8d22.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/images/e7a702a1bbf288ab315bd88ddde5df01efa17795880333e7ac224ee9831d8d22.jpg)

![ed6196fb4392651b4619aa6b55d6f4b2b22e58bb58272173355db6af9eb34cd5.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/images/ed6196fb4392651b4619aa6b55d6f4b2b22e58bb58272173355db6af9eb34cd5.jpg)

![f4f7014981610122f6be087ab2f39edbff028e7f0b7f8967962f169d8ed4bc91.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/images/f4f7014981610122f6be087ab2f39edbff028e7f0b7f8967962f169d8ed4bc91.jpg)

### Tables

![155eb1afbe4f59a640f67f441098c52de2d7a5f06a0ea5a4f13fba82b53ff070.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/tables/155eb1afbe4f59a640f67f441098c52de2d7a5f06a0ea5a4f13fba82b53ff070.jpg)

![35679475ecca21d8951f96d090ceafe39f1acf1e4256b374cda570e37cb40206.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/tables/35679475ecca21d8951f96d090ceafe39f1acf1e4256b374cda570e37cb40206.jpg)

![3fe7fe3b6cf4e7326fe68a36533aa63f680efca7841ae3618784911e1e3396fa.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/tables/3fe7fe3b6cf4e7326fe68a36533aa63f680efca7841ae3618784911e1e3396fa.jpg)

![4f0a198d900b93ba701a89434e52a054765ed486b4e3f12fb3e774d76f3a8cc2.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/tables/4f0a198d900b93ba701a89434e52a054765ed486b4e3f12fb3e774d76f3a8cc2.jpg)

![7aab819c7ee334625090949d0586e4cb5e63829f1240d802cbde119e8ebeb967.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/tables/7aab819c7ee334625090949d0586e4cb5e63829f1240d802cbde119e8ebeb967.jpg)

![7b584c3531c9d7997f9f90fb8056c4c50f69410eaf2497bc993380df545665d0.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/tables/7b584c3531c9d7997f9f90fb8056c4c50f69410eaf2497bc993380df545665d0.jpg)

![9327b54bd73447076f8ca59814ece9fafb2b22c3f722b9a06feb8665192b21f6.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/tables/9327b54bd73447076f8ca59814ece9fafb2b22c3f722b9a06feb8665192b21f6.jpg)

![9794fd807879fbb9348c6ee130c1858541a8bb790b32f671aa29300463014d20.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/tables/9794fd807879fbb9348c6ee130c1858541a8bb790b32f671aa29300463014d20.jpg)

![9de498a83798193d9c757c057c7e6d9c85add4d2e75898ccea20f9818c8adff6.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/tables/9de498a83798193d9c757c057c7e6d9c85add4d2e75898ccea20f9818c8adff6.jpg)

![c506f982e3d21f26d5fd66df9a832da94eb618f146019d407f8081641c6aa2dd.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/tables/c506f982e3d21f26d5fd66df9a832da94eb618f146019d407f8081641c6aa2dd.jpg)

![cc809b69172c55b76d88ec0a393aed04fb5fb93c79299222da58f3972d9a477e.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/tables/cc809b69172c55b76d88ec0a393aed04fb5fb93c79299222da58f3972d9a477e.jpg)

![d6bab38a441ebb5c9ff75782b3399920d33a5cc98ffd3dd57552b183660f4d34.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/tables/d6bab38a441ebb5c9ff75782b3399920d33a5cc98ffd3dd57552b183660f4d34.jpg)

![d6d6e86176e646079db17cdf453f179bb1b4140180c97afd0c2a9135562b819c.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/tables/d6d6e86176e646079db17cdf453f179bb1b4140180c97afd0c2a9135562b819c.jpg)

![db57ec6f236c92ba85c7c2f27e05c674bfe182ce7c29820c71e7737c40b09c21.jpg](../icml_results/2939_From%20Imagined%20Futures%20to%20Executable%20Actions_%20Mixture%20of%20Latent%20Actions%20for%20Robot%20Manipulation/tables/db57ec6f236c92ba85c7c2f27e05c674bfe182ce7c29820c71e7737c40b09c21.jpg)

<a id="paper-2902"></a>

## LaRI: Layered Ray Intersections for Single-view 3D Geometric Reasoning


### Images

![4eef02671cff4bce5a390b7754812d60403ce25ecc0d8c40df3cfb55f04dd642.jpg](../icml_results/2940_LaRI_%20Layered%20Ray%20Intersections%20for%20Single-view%203D%20Geometric%20Reasoning/images/4eef02671cff4bce5a390b7754812d60403ce25ecc0d8c40df3cfb55f04dd642.jpg)

![5e8d8a3f1a5ad8fa9ae2831027750f59e9665c9ae79dd0f4f6cfdbc7aafd2bfe.jpg](../icml_results/2940_LaRI_%20Layered%20Ray%20Intersections%20for%20Single-view%203D%20Geometric%20Reasoning/images/5e8d8a3f1a5ad8fa9ae2831027750f59e9665c9ae79dd0f4f6cfdbc7aafd2bfe.jpg)

![68a1083657f78ee6207003d3c1513d6ea806bdf97e39bb191df9c17ffb213fe3.jpg](../icml_results/2940_LaRI_%20Layered%20Ray%20Intersections%20for%20Single-view%203D%20Geometric%20Reasoning/images/68a1083657f78ee6207003d3c1513d6ea806bdf97e39bb191df9c17ffb213fe3.jpg)

![7fa3e5fdc084d3dde888e72a44012917f70b163891c6e8ab9778fa046650dac9.jpg](../icml_results/2940_LaRI_%20Layered%20Ray%20Intersections%20for%20Single-view%203D%20Geometric%20Reasoning/images/7fa3e5fdc084d3dde888e72a44012917f70b163891c6e8ab9778fa046650dac9.jpg)

![8afdaa983989ca437808697bde184ca84e74f41ed60fdfbf9a829ef92f25a657.jpg](../icml_results/2940_LaRI_%20Layered%20Ray%20Intersections%20for%20Single-view%203D%20Geometric%20Reasoning/images/8afdaa983989ca437808697bde184ca84e74f41ed60fdfbf9a829ef92f25a657.jpg)

![ba2e057d9cba1f6158e790ba77c40f15aa27ba5a0ef934082fd6ee968f9b72c5.jpg](../icml_results/2940_LaRI_%20Layered%20Ray%20Intersections%20for%20Single-view%203D%20Geometric%20Reasoning/images/ba2e057d9cba1f6158e790ba77c40f15aa27ba5a0ef934082fd6ee968f9b72c5.jpg)

![d6b2cab9c097563488eb7a45740199c397dbdb2ef62a4ffe23a3df493064d7e9.jpg](../icml_results/2940_LaRI_%20Layered%20Ray%20Intersections%20for%20Single-view%203D%20Geometric%20Reasoning/images/d6b2cab9c097563488eb7a45740199c397dbdb2ef62a4ffe23a3df493064d7e9.jpg)

![f86d69bd9eb2240f4f771e6b9496c6ffadb7c90edc091bb586e244573b83e935.jpg](../icml_results/2940_LaRI_%20Layered%20Ray%20Intersections%20for%20Single-view%203D%20Geometric%20Reasoning/images/f86d69bd9eb2240f4f771e6b9496c6ffadb7c90edc091bb586e244573b83e935.jpg)

### Tables

![1b3fb466aad944ff1be7b66a53fe562bd47afc4732f7ae9537578120e86ee0c8.jpg](../icml_results/2940_LaRI_%20Layered%20Ray%20Intersections%20for%20Single-view%203D%20Geometric%20Reasoning/tables/1b3fb466aad944ff1be7b66a53fe562bd47afc4732f7ae9537578120e86ee0c8.jpg)

![598c8b194c356f3c0695581e06ffcbc892033cf0311a5d1cfb91046eb79b8a78.jpg](../icml_results/2940_LaRI_%20Layered%20Ray%20Intersections%20for%20Single-view%203D%20Geometric%20Reasoning/tables/598c8b194c356f3c0695581e06ffcbc892033cf0311a5d1cfb91046eb79b8a78.jpg)

![7bc09e754cadeab76196ee6f484bafe0e4387636c1fff40f401ff4838def1295.jpg](../icml_results/2940_LaRI_%20Layered%20Ray%20Intersections%20for%20Single-view%203D%20Geometric%20Reasoning/tables/7bc09e754cadeab76196ee6f484bafe0e4387636c1fff40f401ff4838def1295.jpg)

![8b3cca4d958693e293b738983fe6990c83891b683bc05e409023fe97dd8037e0.jpg](../icml_results/2940_LaRI_%20Layered%20Ray%20Intersections%20for%20Single-view%203D%20Geometric%20Reasoning/tables/8b3cca4d958693e293b738983fe6990c83891b683bc05e409023fe97dd8037e0.jpg)

![9067112c7eef8494b43262f94842f77ab9d4defa732336bc0e836337da10dd52.jpg](../icml_results/2940_LaRI_%20Layered%20Ray%20Intersections%20for%20Single-view%203D%20Geometric%20Reasoning/tables/9067112c7eef8494b43262f94842f77ab9d4defa732336bc0e836337da10dd52.jpg)

![b4036f3997246ada81110301b0ccd93f35d3a21207198e4578d901b4e0713496.jpg](../icml_results/2940_LaRI_%20Layered%20Ray%20Intersections%20for%20Single-view%203D%20Geometric%20Reasoning/tables/b4036f3997246ada81110301b0ccd93f35d3a21207198e4578d901b4e0713496.jpg)

![b49b59ff110e1acb3fb56c2d0f91930b52034b2080efd3f727fdfb1f51a0a4cd.jpg](../icml_results/2940_LaRI_%20Layered%20Ray%20Intersections%20for%20Single-view%203D%20Geometric%20Reasoning/tables/b49b59ff110e1acb3fb56c2d0f91930b52034b2080efd3f727fdfb1f51a0a4cd.jpg)

![e62cb1caad7a359d5fb433bb5d9d3f723bcffa5d33b9d5fc6e2339a1c425ca91.jpg](../icml_results/2940_LaRI_%20Layered%20Ray%20Intersections%20for%20Single-view%203D%20Geometric%20Reasoning/tables/e62cb1caad7a359d5fb433bb5d9d3f723bcffa5d33b9d5fc6e2339a1c425ca91.jpg)

![fa2b50aa3258774d65b6af5489a25abe2304ac845d14ee1840a0bc02a3c7d5c9.jpg](../icml_results/2940_LaRI_%20Layered%20Ray%20Intersections%20for%20Single-view%203D%20Geometric%20Reasoning/tables/fa2b50aa3258774d65b6af5489a25abe2304ac845d14ee1840a0bc02a3c7d5c9.jpg)

<a id="paper-2903"></a>

## Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages


### Images

![020348be8e1495a93730e31e6985316287f45316c59ccc41a61857e7ff474d16.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/images/020348be8e1495a93730e31e6985316287f45316c59ccc41a61857e7ff474d16.jpg)

![43243a88ed963907c1f2455b51717a79a9c0307cd1b0c6b9a38efcb62528056f.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/images/43243a88ed963907c1f2455b51717a79a9c0307cd1b0c6b9a38efcb62528056f.jpg)

![4695c4ec382771c03125bfb0180e0947f5256dbea0f45240f11a95b49589179b.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/images/4695c4ec382771c03125bfb0180e0947f5256dbea0f45240f11a95b49589179b.jpg)

![5db2fbc01d8f5853339972051fe862b24c0b3591489914bd63025d68e1b6ea8c.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/images/5db2fbc01d8f5853339972051fe862b24c0b3591489914bd63025d68e1b6ea8c.jpg)

![6ac0029407386be043ac9d3e394019d03f93f8157177f70f1fe1b11b22da703f.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/images/6ac0029407386be043ac9d3e394019d03f93f8157177f70f1fe1b11b22da703f.jpg)

![ae93c2ffe52d7d70718e92f0cb2cc1472b40a7504ca1358b1253e754039023a7.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/images/ae93c2ffe52d7d70718e92f0cb2cc1472b40a7504ca1358b1253e754039023a7.jpg)

![be9318f2ad05a37b223ac2915b57982dc44ef43ef56cf62c7c835694e3b145d6.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/images/be9318f2ad05a37b223ac2915b57982dc44ef43ef56cf62c7c835694e3b145d6.jpg)

![d91a5574d327277efe95993f7cac692ed907d6902d9ed2608a6584ddcd72c2b5.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/images/d91a5574d327277efe95993f7cac692ed907d6902d9ed2608a6584ddcd72c2b5.jpg)

![f269468e3ae9427c07e40ef8f2584a5886e970e0ce6c99d54a3efda806f2749e.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/images/f269468e3ae9427c07e40ef8f2584a5886e970e0ce6c99d54a3efda806f2749e.jpg)

### Tables

![0ba1c3330a835cb34e0a28ad1403b5b9f27ef0b310bfd10416ac9c5b9d8acf9e.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/tables/0ba1c3330a835cb34e0a28ad1403b5b9f27ef0b310bfd10416ac9c5b9d8acf9e.jpg)

![3c1a64ecd4da70ce257cddb1b68f19ff6b1abce21f54d3795e67af73b33d9621.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/tables/3c1a64ecd4da70ce257cddb1b68f19ff6b1abce21f54d3795e67af73b33d9621.jpg)

![3e2a7846091d3a911c072258521aa2ee76f3da3b0b7f02343a945a41e48a9e61.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/tables/3e2a7846091d3a911c072258521aa2ee76f3da3b0b7f02343a945a41e48a9e61.jpg)

![42138a36ef93f4206127876d89b2163b1b7057ae3bd649e6cb7233eac1208a14.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/tables/42138a36ef93f4206127876d89b2163b1b7057ae3bd649e6cb7233eac1208a14.jpg)

![46144a4c06748e4f6a5b2b0124606534dbc27f245ed4ce9daac90e30bd176d7d.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/tables/46144a4c06748e4f6a5b2b0124606534dbc27f245ed4ce9daac90e30bd176d7d.jpg)

![5b1d9019f69eda1043fb104129b3488866f6c82100fc73b6069b55f56b08a233.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/tables/5b1d9019f69eda1043fb104129b3488866f6c82100fc73b6069b55f56b08a233.jpg)

![63d4e714ec41922c315d984833e075934cc7facacf9712ad04c2f2d0c19e70d2.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/tables/63d4e714ec41922c315d984833e075934cc7facacf9712ad04c2f2d0c19e70d2.jpg)

![64a1699dcc6bd9ca3ac99291e77897360af67036c0be48c90d39b849c7f2e81a.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/tables/64a1699dcc6bd9ca3ac99291e77897360af67036c0be48c90d39b849c7f2e81a.jpg)

![886433165f72bd02ff83804a0b86b60e050e7220d16168d6888f273ab68bec9f.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/tables/886433165f72bd02ff83804a0b86b60e050e7220d16168d6888f273ab68bec9f.jpg)

![955cf3e4389af39ee29773c222035e1bad356deb53e41426857af0d4c1946c42.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/tables/955cf3e4389af39ee29773c222035e1bad356deb53e41426857af0d4c1946c42.jpg)

![ae9fe1614a2a1bcf392da8977e63c60c0d6601d1f3ca1d4cf2fe74ecf9e07a44.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/tables/ae9fe1614a2a1bcf392da8977e63c60c0d6601d1f3ca1d4cf2fe74ecf9e07a44.jpg)

![b73a7e7fc560490b16851488981a928a094ee909ffa5d4a8d8fe7b5fe5f3dca0.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/tables/b73a7e7fc560490b16851488981a928a094ee909ffa5d4a8d8fe7b5fe5f3dca0.jpg)

![bad75edbe28a755068b1d88647d57063f8eaac008c5176fda800acb9409a2361.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/tables/bad75edbe28a755068b1d88647d57063f8eaac008c5176fda800acb9409a2361.jpg)

![cb2160a9540b6695542c7f4a2258ce08e99033474dc2b5e0088078fc7bdcde5a.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/tables/cb2160a9540b6695542c7f4a2258ce08e99033474dc2b5e0088078fc7bdcde5a.jpg)

![dd1924d4a9ac1ac9fd0aa0fed09cd6859156e9b054858c343d6c8424827ed429.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/tables/dd1924d4a9ac1ac9fd0aa0fed09cd6859156e9b054858c343d6c8424827ed429.jpg)

![deb561dda6c90ddaada50da8a90f23773da08622b03b0fc70fd69a9c992fd553.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/tables/deb561dda6c90ddaada50da8a90f23773da08622b03b0fc70fd69a9c992fd553.jpg)

![fdaec22f3efef5076c2da9dc26e0a08a8d7b17adde4e1e9ce7383ddad75adbbf.jpg](../icml_results/2941_Toward%20Robust%20Multilingual%20Adaptation%20of%20LLMs%20for%20Low-Resource%20Languages/tables/fdaec22f3efef5076c2da9dc26e0a08a8d7b17adde4e1e9ce7383ddad75adbbf.jpg)

<a id="paper-2904"></a>

## Modality-Decoupled Online Recursive Editing

### Images

![0a65cacfbe810dafea088dd2de9742cf63e9072b6a32b8d5b8004c1d0147269f.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/0a65cacfbe810dafea088dd2de9742cf63e9072b6a32b8d5b8004c1d0147269f.jpg)

![0b145398a9700f46f368c949d088322ed38c41058aca0a8655ab5037028264fc.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/0b145398a9700f46f368c949d088322ed38c41058aca0a8655ab5037028264fc.jpg)

![19b16d790108e86419781d59dd2286e7098a47be90c035ba2edc45b74a9c4622.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/19b16d790108e86419781d59dd2286e7098a47be90c035ba2edc45b74a9c4622.jpg)

![1bf3c65439deea9ba8cc51859edbd7538faf5331745f15f353da3e5340608be2.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/1bf3c65439deea9ba8cc51859edbd7538faf5331745f15f353da3e5340608be2.jpg)

![21ea9a43d5863e6423800174cde7db95b63a3011223dce04161c19689e37c26b.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/21ea9a43d5863e6423800174cde7db95b63a3011223dce04161c19689e37c26b.jpg)

![22b66033065414598c974caeb426dfe90eb73d03bc3fd23c25b0c6149f0ba15e.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/22b66033065414598c974caeb426dfe90eb73d03bc3fd23c25b0c6149f0ba15e.jpg)

![2a302a61ec7ee303a9bd867751b89f1d9227a08500ec29425c3cd1f2c7d6380b.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/2a302a61ec7ee303a9bd867751b89f1d9227a08500ec29425c3cd1f2c7d6380b.jpg)

![30cfd2717f8f50bb0a4e13132bd756bc572f3a2bec02ca2978cb87bd1344b5b5.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/30cfd2717f8f50bb0a4e13132bd756bc572f3a2bec02ca2978cb87bd1344b5b5.jpg)

![310770226fa899fc6e9936fc06724186b2cdccfe6a0b5a0110298a439f45b22c.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/310770226fa899fc6e9936fc06724186b2cdccfe6a0b5a0110298a439f45b22c.jpg)

![3fe3e0fc7be9dd7bcfdd0da551df4ece3d5b20356590ecce5eac80d58dd8d084.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/3fe3e0fc7be9dd7bcfdd0da551df4ece3d5b20356590ecce5eac80d58dd8d084.jpg)

![416a1bc506f512373c52df72b179331111f3c3523b8c395e69f220f3100d2fc9.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/416a1bc506f512373c52df72b179331111f3c3523b8c395e69f220f3100d2fc9.jpg)

![45d73e060e5eee39e97463c2da24bb55b9dc627e89f92ae26ec0834671aac75b.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/45d73e060e5eee39e97463c2da24bb55b9dc627e89f92ae26ec0834671aac75b.jpg)

![4f13bba6edc4101c7e14a5eab55110e79f424e81f164476a02931b673e6ac217.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/4f13bba6edc4101c7e14a5eab55110e79f424e81f164476a02931b673e6ac217.jpg)

![676c0bebeb2b990867936fd616edfb87e82058797745fc237a1b2e37d257ed99.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/676c0bebeb2b990867936fd616edfb87e82058797745fc237a1b2e37d257ed99.jpg)

![7bcbd8eb335900b053b556d294b94688693739e92da1a61e589dc3e35cc82205.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/7bcbd8eb335900b053b556d294b94688693739e92da1a61e589dc3e35cc82205.jpg)

![8bd42222f3f9070d6dc9ceda9bfc3efbb7b8a7073b36564feec27544f0e2ec94.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/8bd42222f3f9070d6dc9ceda9bfc3efbb7b8a7073b36564feec27544f0e2ec94.jpg)

![9cea405a267194fdb2a9b4271825d68786e987abd725cde6a3a82157f50b1dbe.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/9cea405a267194fdb2a9b4271825d68786e987abd725cde6a3a82157f50b1dbe.jpg)

![9ee8e97ca37b86e1a557e8c82932118e1687fc6898b86bf514108aab26cae6ec.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/9ee8e97ca37b86e1a557e8c82932118e1687fc6898b86bf514108aab26cae6ec.jpg)

![a0f21583385f60ed742fd897c6e4c580cd2130d0e474ec60107d88720bb84642.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/a0f21583385f60ed742fd897c6e4c580cd2130d0e474ec60107d88720bb84642.jpg)

![ade7011d4f299e9a9c7df013abb1a5a2293aee515d3d8ee08b771965a5043390.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/ade7011d4f299e9a9c7df013abb1a5a2293aee515d3d8ee08b771965a5043390.jpg)

![b84db6bb267c7c83053260d0319d8b336b4c2f3e2b7e0ed9bf93fa29b3ce4d83.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/b84db6bb267c7c83053260d0319d8b336b4c2f3e2b7e0ed9bf93fa29b3ce4d83.jpg)

![bcdc601fd40659e1bde12c5c38549d115afbfd59e5bc8ffcab00a14c3facff60.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/bcdc601fd40659e1bde12c5c38549d115afbfd59e5bc8ffcab00a14c3facff60.jpg)

![bfe1115ce1f3ace40d4188c052b7e5ad73e16df13f205a875f93b7a9e80eb230.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/bfe1115ce1f3ace40d4188c052b7e5ad73e16df13f205a875f93b7a9e80eb230.jpg)

![c1d9283a92b4b95cc8aa4ccfe96a34d092ca102f13a0bb2d1db56cfc3e314cdd.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/c1d9283a92b4b95cc8aa4ccfe96a34d092ca102f13a0bb2d1db56cfc3e314cdd.jpg)

![e7f0b58f6ac7ce3b080a7b9942579ea04145bffb2702770eb3550437bdb7eafa.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/e7f0b58f6ac7ce3b080a7b9942579ea04145bffb2702770eb3550437bdb7eafa.jpg)

![f41a909060f8e7572280a74399062ae533938103c4698200b4c6fb0625bb38cf.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/f41a909060f8e7572280a74399062ae533938103c4698200b4c6fb0625bb38cf.jpg)

![f7f0843092b30b952c5d4783dce625e6b905fef746dba6b107cf622431810596.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/f7f0843092b30b952c5d4783dce625e6b905fef746dba6b107cf622431810596.jpg)

![fda1858328273a12036d681706b402975ff89064ddbb35f9c716aff83e6be30d.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/images/fda1858328273a12036d681706b402975ff89064ddbb35f9c716aff83e6be30d.jpg)

### Tables

![1fcb3cb2ba8e9731c138465ae1c812df7cf48aab7996e0d7ac4c31c2b71caa71.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/1fcb3cb2ba8e9731c138465ae1c812df7cf48aab7996e0d7ac4c31c2b71caa71.jpg)

![38d641fe928cac9f8af0887ecf4d2b527087858bff2a5d5c2bc776831a3c7775.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/38d641fe928cac9f8af0887ecf4d2b527087858bff2a5d5c2bc776831a3c7775.jpg)

![486ff668b1c6857b1b8bff4da32f34dd30cb7b9f6eebb5a329a57679de256084.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/486ff668b1c6857b1b8bff4da32f34dd30cb7b9f6eebb5a329a57679de256084.jpg)

![48bec6965c87fdf66a5c904aa52a05d24d34abbdf1adc2d72c6b472547bd5d97.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/48bec6965c87fdf66a5c904aa52a05d24d34abbdf1adc2d72c6b472547bd5d97.jpg)

![71029f4dff94651f5b5c5989eb31703ac7816e7fd9b218510344b08e61195eb6.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/71029f4dff94651f5b5c5989eb31703ac7816e7fd9b218510344b08e61195eb6.jpg)

![775a815e0e5b3c5a1f45dc5001cdb2b4cb37136a998723c314096ca36d345e15.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/775a815e0e5b3c5a1f45dc5001cdb2b4cb37136a998723c314096ca36d345e15.jpg)

![a430b279dc34d53265d98c3fc4afb162b37775c08cdb8e4b9169f990d662989f.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/a430b279dc34d53265d98c3fc4afb162b37775c08cdb8e4b9169f990d662989f.jpg)

![c8db1896a42022866dbd5a592ba49c7226b821e0b892e7be89e8f91585576613.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/c8db1896a42022866dbd5a592ba49c7226b821e0b892e7be89e8f91585576613.jpg)

![daec1b8ca174fee7d7119e4758f836699128dc4e602fa1a0f43adce63972f350.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/daec1b8ca174fee7d7119e4758f836699128dc4e602fa1a0f43adce63972f350.jpg)

![e3957c68a3ab04094bbd675c93a78b3b9b86e45c6f62ae4162bb563186023895.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/e3957c68a3ab04094bbd675c93a78b3b9b86e45c6f62ae4162bb563186023895.jpg)

![e74c09e10f4b6ec8345949321d804921f1f782b3b39caf41392feacbaefa1b86.jpg](../icml_results/2942_Modality-Decoupled%20Online%20Recursive%20Editing/tables/e74c09e10f4b6ec8345949321d804921f1f782b3b39caf41392feacbaefa1b86.jpg)
