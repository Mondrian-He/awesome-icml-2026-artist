# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 174 of 197

## 目录 (Table of Contents)

1. [Hi-Time: Hierarchical Latent Prediction for Multivariate Time Series Classification](#paper-5710)
2. [ERAlign: Energy-based Representation Alignment of GNNs and LLMs on Text-attributed Graphs](#paper-5711)
3. [HiCI: Hierarchical Construction–Integration for Long-Context Attention](#paper-5712)
4. [On Testing Conditional Mean Independence for Manifold-Valued Data](#paper-5713)
5. [Fix Before Search: Benchmarking Agentic Visual Query Pre-processing in Multimodal Retrieval-augmented Generation](#paper-5714)
6. [ReJump: A Tree-Jump Representation for Analyzing and Improving LLM Reasoning](#paper-5715)
7. [Video-o3: Native Interleaved Clue Seeking for Long Video Multi-Hop Reasoning](#paper-5716)
8. [Shrinking the Variance: Shrinkage Baselines for Reinforcement Learning with Verifiable Rewards](#paper-5717)
9. [Discrete Diffusion with Physical Mass Constraints for De Novo Peptide Sequencing](#paper-5718)
10. [Physiology as Language: Translating Respiration to EEG during Sleep](#paper-5719)
11. [Graph is a Natural Regularization: Revisiting Vector Quantization for Graph Representation Learning](#paper-5720)
12. [CORE: Context-Robust Remasking for Diffusion Language Models](#paper-5721)
13. [Distilling Task-Level Coordination Policies for Generalizable Multi-Agent Cooperation](#paper-5722)
14. [SCOPE: Evolving Symbolic World for Planning in Open-Ended Environments](#paper-5723)
15. [Simple Policy Gradients for Reasoning with Diffusion Language Models](#paper-5724)
16. [Mind Your Entropy: From Maximum Entropy to Trajectory Entropy-Constrained RL](#paper-5725)
17. [Harmonized Dual Policy Improvement for Modelic Reinforcement Learning](#paper-5726)
18. [Guidance: Sentence-Level Citation Enforcement via Prefix-Tail Guidance during LLM Decoding](#paper-5727)
19. [Decomposition-Based Modular Conformal Prediction for Two-Stage Modeling](#paper-5728)
20. [BALLAST: Bayesian Active Learning with Look-ahead Amendment for Sea-drifter Trajectories under Spatio-Temporal Vector Fields](#paper-5729)
21. [SENDAI: A Hierarchical Sparse-measurement, EfficieNt Data AssImilation Framework](#paper-5730)
22. [Weights to Code: Extracting Interpretable Algorithms from the Discrete Transformer](#paper-5731)
23. [Local MAP Sampling for Diffusion Models](#paper-5732)
24. [Prediction-Powered Risk Monitoring of Deployed Models for Detecting Harmful Distribution Shifts](#paper-5733)
25. [RiboSphere: Learning Unified and Efficient Representations of RNA Structures](#paper-5734)
26. [Transformers Efficiently Perform In-Context Logistic Regression via Normalized Gradient Descent](#paper-5735)
27. [FUSE: Quantifying Uncertainty in Vision-Language Models by Bayesian Fusing Epistemic and Aleatoric Uncertainty](#paper-5736)
28. [Mitigating Translationese Bias in Multilingual LLM-as-a-Judge via Disentangled Information Bottleneck](#paper-5737)
29. [Forgetting Whenever You Want: A Decentralized Continual Learning Framework with On-Demand Unlearning](#paper-5738)
30. [Agent Learning via Early Experience](#paper-5739)
31. [Internalizing Safety Understanding in Large Reasoning Models via Verification](#paper-5740)
32. [ReVSI: Rebuilding Visual Spatial Intelligence Evaluation for Accurate Assessment of VLM 3D Reasoning](#paper-5741)
33. [L-Drive: Beyond a Single Mapping—Latent Context Drives Time Series Forecasting](#paper-5742)

---


<a id="paper-5710"></a>

## Hi-Time: Hierarchical Latent Prediction for Multivariate Time Series Classification

### Images

![073830bd6636fa74de2307a5dd84cc4936a15e218af009517c023371b336aecf.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/073830bd6636fa74de2307a5dd84cc4936a15e218af009517c023371b336aecf.jpg)

![0ed275c2782d436f72b3ad799c00a48f8724c2b563d469be0e57784fc28ebee6.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/0ed275c2782d436f72b3ad799c00a48f8724c2b563d469be0e57784fc28ebee6.jpg)

![207199890ae8b75748e3df0b9b0f2d6e06fcf2148b850ba7aae7064e734f354c.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/207199890ae8b75748e3df0b9b0f2d6e06fcf2148b850ba7aae7064e734f354c.jpg)

![338c7b59546c37528f04a1e22c24ef6026340afbda7b57c2dabd95102da842a0.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/338c7b59546c37528f04a1e22c24ef6026340afbda7b57c2dabd95102da842a0.jpg)

![4d390c9cbe303d304436c94cdadb9172af87cee7e3730c7c80c35ab3d72bdd39.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/4d390c9cbe303d304436c94cdadb9172af87cee7e3730c7c80c35ab3d72bdd39.jpg)

![567b836c723512d53197982a9296d375ab59f25b02c06738a5e9dea52472fb3c.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/567b836c723512d53197982a9296d375ab59f25b02c06738a5e9dea52472fb3c.jpg)

![75c68ed947d825f00ccec6ad2bce419a61e1ceb9658eba071b2d5ba45c8ae8d2.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/75c68ed947d825f00ccec6ad2bce419a61e1ceb9658eba071b2d5ba45c8ae8d2.jpg)

![7879b48ab52dec32238e7743beb7c32306a7b5a22024c92fe1b0457b92aa0f1e.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/7879b48ab52dec32238e7743beb7c32306a7b5a22024c92fe1b0457b92aa0f1e.jpg)

![8ba4b0f74892bbd37fde5409b17bade2b73ae71fc9f600d1aeb93d87d6bafb66.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/8ba4b0f74892bbd37fde5409b17bade2b73ae71fc9f600d1aeb93d87d6bafb66.jpg)

![8c93bb7bd636384e69de5603a4d68bc0a218abbb337fb3bd0e4312db9c62c751.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/8c93bb7bd636384e69de5603a4d68bc0a218abbb337fb3bd0e4312db9c62c751.jpg)

![8f6cbfb81659ccbe1474fa04c7ab735481c37a2aa34801a0f57140f75794e0d0.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/8f6cbfb81659ccbe1474fa04c7ab735481c37a2aa34801a0f57140f75794e0d0.jpg)

![bbec98d5889f250066b792e4d043def982fab0496ff01b15f8221e5cecebc0ff.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/bbec98d5889f250066b792e4d043def982fab0496ff01b15f8221e5cecebc0ff.jpg)

![f2d08441ec64541391405bd385b9e347ac99d7098775d665e84175c1874dd3e4.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/f2d08441ec64541391405bd385b9e347ac99d7098775d665e84175c1874dd3e4.jpg)

![fc68999691ffe18f9366eac47b55f2897300858b114c1f80cd32ad93048d1aa3.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/images/fc68999691ffe18f9366eac47b55f2897300858b114c1f80cd32ad93048d1aa3.jpg)

### Tables

![13902792d787e3bf74f3ca06cf735b105a1444f6ac2f38d8e33858b1a551f002.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/13902792d787e3bf74f3ca06cf735b105a1444f6ac2f38d8e33858b1a551f002.jpg)

![1b604ffa9f84c76f7e3dddd96c4ab90d53ddcdf44c72df4a401d75dcb94251c7.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/1b604ffa9f84c76f7e3dddd96c4ab90d53ddcdf44c72df4a401d75dcb94251c7.jpg)

![1e81e24d52fbda8ba2dac488937fb21fdab53925164051d43ae9aaebfaedcc56.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/1e81e24d52fbda8ba2dac488937fb21fdab53925164051d43ae9aaebfaedcc56.jpg)

![234d4bfdbe5d4ac4f8c3f499eb69c70de73baf60101482bcc49ffb961aac93d4.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/234d4bfdbe5d4ac4f8c3f499eb69c70de73baf60101482bcc49ffb961aac93d4.jpg)

![30dc4cd67aaefe0dd649f07ddc2e6074720ef8951a60d5b8773c5a481a3aad97.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/30dc4cd67aaefe0dd649f07ddc2e6074720ef8951a60d5b8773c5a481a3aad97.jpg)

![46c5b3347d84d6c5382b81e82206f93402e037f3d9f5030e3bcea483f32307bc.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/46c5b3347d84d6c5382b81e82206f93402e037f3d9f5030e3bcea483f32307bc.jpg)

![5eeb089b9db585c1f35bb9cdf59b7bd53a039b17d8f5cf75d4a8fb2bca1ac8dd.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/5eeb089b9db585c1f35bb9cdf59b7bd53a039b17d8f5cf75d4a8fb2bca1ac8dd.jpg)

![81426a7b3e766082c81421716d3b83dc280b87972b29b134261dacc0031adc79.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/81426a7b3e766082c81421716d3b83dc280b87972b29b134261dacc0031adc79.jpg)

![89899bb885df214ec1c66e796b7365d856363417e58eb2b8791dea43967705f5.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/89899bb885df214ec1c66e796b7365d856363417e58eb2b8791dea43967705f5.jpg)

![a98710a731a520cf3580213f8d6d8323f01df3dbc4735a7ec1177f497b5e167d.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/a98710a731a520cf3580213f8d6d8323f01df3dbc4735a7ec1177f497b5e167d.jpg)

![b723436b59de61998ebd3fca6ac2b92a934fe0f0058aada0a158be2bade97a97.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/b723436b59de61998ebd3fca6ac2b92a934fe0f0058aada0a158be2bade97a97.jpg)

![c55d10b9b8b30dd35cf1fbc705ff21944faf247f6482371a067a9880b26ba3f0.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/c55d10b9b8b30dd35cf1fbc705ff21944faf247f6482371a067a9880b26ba3f0.jpg)

![d299bc48ecd816ff82ee495f88cfe6dff6193d35ad4eaa469026deba3254ab14.jpg](../icml_results/5758_MTNL_%20A%20Unified%20Modeling%20Perspective%20for%20Enhancing%20Tensor%20Network%20Learning/tables/d299bc48ecd816ff82ee495f88cfe6dff6193d35ad4eaa469026deba3254ab14.jpg)

## Hi-Time: Hierarchical Latent Prediction for Multivariate Time Series Classification


### Images

![2eaf28a96223aaac27f7a69721836e20faae615c519d1bd54053ba39c7f3d3d0.jpg](../icml_results/5759_Hi-Time_%20Hierarchical%20Latent%20Prediction%20for%20Multivariate%20Time%20Series%20Classification/images/2eaf28a96223aaac27f7a69721836e20faae615c519d1bd54053ba39c7f3d3d0.jpg)

![aa7a0e3259aeb40f31beed66ff93d3b058689dbb2550e1b5c733bdecfdb9faa0.jpg](../icml_results/5759_Hi-Time_%20Hierarchical%20Latent%20Prediction%20for%20Multivariate%20Time%20Series%20Classification/images/aa7a0e3259aeb40f31beed66ff93d3b058689dbb2550e1b5c733bdecfdb9faa0.jpg)

![efd63b7bb86ff97392db138024bfd3d30bb1654f4224fbf051aea5cccf6980b0.jpg](../icml_results/5759_Hi-Time_%20Hierarchical%20Latent%20Prediction%20for%20Multivariate%20Time%20Series%20Classification/images/efd63b7bb86ff97392db138024bfd3d30bb1654f4224fbf051aea5cccf6980b0.jpg)

![f36da20dad433e657f2345ddcaf8d0a9399eb3afc2969fdf0b2e9f9928473743.jpg](../icml_results/5759_Hi-Time_%20Hierarchical%20Latent%20Prediction%20for%20Multivariate%20Time%20Series%20Classification/images/f36da20dad433e657f2345ddcaf8d0a9399eb3afc2969fdf0b2e9f9928473743.jpg)

### Tables

![6756ab0746fa35e83a6f3cdc682c8a7e2f8912b50b53651288de980365095b0b.jpg](../icml_results/5759_Hi-Time_%20Hierarchical%20Latent%20Prediction%20for%20Multivariate%20Time%20Series%20Classification/tables/6756ab0746fa35e83a6f3cdc682c8a7e2f8912b50b53651288de980365095b0b.jpg)

![a6929a3fc626f1aa16035faf8eb87fa2865dd4bbd133f9b5e03fa7d49c74caec.jpg](../icml_results/5759_Hi-Time_%20Hierarchical%20Latent%20Prediction%20for%20Multivariate%20Time%20Series%20Classification/tables/a6929a3fc626f1aa16035faf8eb87fa2865dd4bbd133f9b5e03fa7d49c74caec.jpg)

<a id="paper-5711"></a>

## ERAlign: Energy-based Representation Alignment of GNNs and LLMs on Text-attributed Graphs


### Images

![0e551ffd555168bd40ccd3df4cf5f7bb112df0d91ccfd15076bd81f4d00095b2.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/images/0e551ffd555168bd40ccd3df4cf5f7bb112df0d91ccfd15076bd81f4d00095b2.jpg)

![81b78dec645520db630f534c79f231c2b277e7efc40df9a5bd8edabcff93be79.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/images/81b78dec645520db630f534c79f231c2b277e7efc40df9a5bd8edabcff93be79.jpg)

![8f9df109813ba05d0ec6c5b88ba160a35ff21ca1f7311a8af3ef3f26aa738cfa.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/images/8f9df109813ba05d0ec6c5b88ba160a35ff21ca1f7311a8af3ef3f26aa738cfa.jpg)

![c658481be90f1168528d48658f2bf7ec6f39c8d197a35e4595164d3077a0f5be.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/images/c658481be90f1168528d48658f2bf7ec6f39c8d197a35e4595164d3077a0f5be.jpg)

![ec70b4def77b17995994ec840cc2250d7e04d3ccb4377972a6f816fbb8eec2b7.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/images/ec70b4def77b17995994ec840cc2250d7e04d3ccb4377972a6f816fbb8eec2b7.jpg)

### Tables

![00bb08d7f8503d501149699ca0817008a96d98b5100149a1351f52c136fe2dbf.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/tables/00bb08d7f8503d501149699ca0817008a96d98b5100149a1351f52c136fe2dbf.jpg)

![016b06624ac2e34df76eda9429ce38a04eeb4b8584dae8f57ec1229d4a272da2.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/tables/016b06624ac2e34df76eda9429ce38a04eeb4b8584dae8f57ec1229d4a272da2.jpg)

![2c8689d2c370bf1b6f6995aae9f7c6c94fbfdf4f6bb859fc0ff6e216afda0927.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/tables/2c8689d2c370bf1b6f6995aae9f7c6c94fbfdf4f6bb859fc0ff6e216afda0927.jpg)

![2e1f000a8404799c0236aaf8f1c832da7d9caa2ac4c63d75983cc3443c5c362b.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/tables/2e1f000a8404799c0236aaf8f1c832da7d9caa2ac4c63d75983cc3443c5c362b.jpg)

![32aeba469213da3ce8f52bd1ff15d9d103314f57a06092a513fd1acf36f3805e.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/tables/32aeba469213da3ce8f52bd1ff15d9d103314f57a06092a513fd1acf36f3805e.jpg)

![36d077eb300825669fe841eacb13adc85b586ad8d1e9c9ac4068704bda0c1d96.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/tables/36d077eb300825669fe841eacb13adc85b586ad8d1e9c9ac4068704bda0c1d96.jpg)

![3acfa540f215cb1470b22c714e99db3774980ed52b6878f56c004c17a84ded6f.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/tables/3acfa540f215cb1470b22c714e99db3774980ed52b6878f56c004c17a84ded6f.jpg)

![6a18e483d41346d05a813a4e444cac66c2d99a2f04c5661c5eebdbad7573384e.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/tables/6a18e483d41346d05a813a4e444cac66c2d99a2f04c5661c5eebdbad7573384e.jpg)

![6d106db0bbae4cad4cd73a53cead0c0d322cf1c33e35593058840b37450aea96.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/tables/6d106db0bbae4cad4cd73a53cead0c0d322cf1c33e35593058840b37450aea96.jpg)

![6d50f65ed835cbd0567d070461940e09daa6ef20ec76a62e9a8497daa64505db.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/tables/6d50f65ed835cbd0567d070461940e09daa6ef20ec76a62e9a8497daa64505db.jpg)

![a93301cb919ccdbf7260cd55a5db7b9df884de209cbfd482c443d2afdf369141.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/tables/a93301cb919ccdbf7260cd55a5db7b9df884de209cbfd482c443d2afdf369141.jpg)

![d9aedaee86c0fcfb2937a08bda79a683e7d3b6d85f591927e6690da1db69f206.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/tables/d9aedaee86c0fcfb2937a08bda79a683e7d3b6d85f591927e6690da1db69f206.jpg)

![e420374350c094e65eb358c7e4940485a1c39fe00d6bc22dae84be13453d914b.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/tables/e420374350c094e65eb358c7e4940485a1c39fe00d6bc22dae84be13453d914b.jpg)

![e8f346d4944ac0bf7854e7dd71b0fe8e303e7fb47bcda660ac535f66b9635c98.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/tables/e8f346d4944ac0bf7854e7dd71b0fe8e303e7fb47bcda660ac535f66b9635c98.jpg)

![ef03805d09eac76ce62d690e0f5f87fba67c1319e57e4afbc08f818359e2543b.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/tables/ef03805d09eac76ce62d690e0f5f87fba67c1319e57e4afbc08f818359e2543b.jpg)

![fff1ed37a49b6185fe5225c30a7bc588ad4000b46aae363d96a03d5711ed8ea0.jpg](../icml_results/5760_ERAlign_%20Energy-based%20Representation%20Alignment%20of%20GNNs%20and%20LLMs%20on%20Text-attributed%20Graphs/tables/fff1ed37a49b6185fe5225c30a7bc588ad4000b46aae363d96a03d5711ed8ea0.jpg)

<a id="paper-5712"></a>

## HiCI: Hierarchical Construction–Integration for Long-Context Attention


### Images

![38db6b96220d2eed14572de2f7a32bc797fc3d53d38120b974e5bc1261c61883.jpg](../icml_results/5761_HiCI_%20Hierarchical%20Construction%E2%80%93Integration%20for%20Long-Context%20Attention/images/38db6b96220d2eed14572de2f7a32bc797fc3d53d38120b974e5bc1261c61883.jpg)

![396e3f4f0affae010ab70bfff37db543fde5c25f4e88116b0102a9af2912d838.jpg](../icml_results/5761_HiCI_%20Hierarchical%20Construction%E2%80%93Integration%20for%20Long-Context%20Attention/images/396e3f4f0affae010ab70bfff37db543fde5c25f4e88116b0102a9af2912d838.jpg)

![840cc774e0b9c725642482654a9c7e476b01c0fd1d766a8be4359f381df8ca59.jpg](../icml_results/5761_HiCI_%20Hierarchical%20Construction%E2%80%93Integration%20for%20Long-Context%20Attention/images/840cc774e0b9c725642482654a9c7e476b01c0fd1d766a8be4359f381df8ca59.jpg)

![b226db4db6073ecb598639745031cf32fdeeff047940bcec8003f2353a6fa894.jpg](../icml_results/5761_HiCI_%20Hierarchical%20Construction%E2%80%93Integration%20for%20Long-Context%20Attention/images/b226db4db6073ecb598639745031cf32fdeeff047940bcec8003f2353a6fa894.jpg)

![f6daa65005c8d7702b7ad55716dc13aad2fa7db0d7ed0cae1ed085d37352a2ea.jpg](../icml_results/5761_HiCI_%20Hierarchical%20Construction%E2%80%93Integration%20for%20Long-Context%20Attention/images/f6daa65005c8d7702b7ad55716dc13aad2fa7db0d7ed0cae1ed085d37352a2ea.jpg)

### Tables

![21446a143c3f95df398d1aa690c45b8378961851974d650765117c40d10310bc.jpg](../icml_results/5761_HiCI_%20Hierarchical%20Construction%E2%80%93Integration%20for%20Long-Context%20Attention/tables/21446a143c3f95df398d1aa690c45b8378961851974d650765117c40d10310bc.jpg)

![32be41a22813602eb8a6fb9ae96b45a2b0634c51a5b2fe19c7492e62a3fe1e11.jpg](../icml_results/5761_HiCI_%20Hierarchical%20Construction%E2%80%93Integration%20for%20Long-Context%20Attention/tables/32be41a22813602eb8a6fb9ae96b45a2b0634c51a5b2fe19c7492e62a3fe1e11.jpg)

![6d0fa692df7acbc6c1ea6ab4099d39f5ca72fb07548f878ed5d0ebbd9ee9e863.jpg](../icml_results/5761_HiCI_%20Hierarchical%20Construction%E2%80%93Integration%20for%20Long-Context%20Attention/tables/6d0fa692df7acbc6c1ea6ab4099d39f5ca72fb07548f878ed5d0ebbd9ee9e863.jpg)

![714bf3e3798dcd939f363a72a4e1ce35498d2a6331186073ba7bbaa774644c53.jpg](../icml_results/5761_HiCI_%20Hierarchical%20Construction%E2%80%93Integration%20for%20Long-Context%20Attention/tables/714bf3e3798dcd939f363a72a4e1ce35498d2a6331186073ba7bbaa774644c53.jpg)

![7ef723a7f5064268f14004623eb8cda428960e6899770fb714f6039b03727cf3.jpg](../icml_results/5761_HiCI_%20Hierarchical%20Construction%E2%80%93Integration%20for%20Long-Context%20Attention/tables/7ef723a7f5064268f14004623eb8cda428960e6899770fb714f6039b03727cf3.jpg)

![84ef425284b3e00588c388941b1a33d4a38df8988d583bb27393bf2020ee2958.jpg](../icml_results/5761_HiCI_%20Hierarchical%20Construction%E2%80%93Integration%20for%20Long-Context%20Attention/tables/84ef425284b3e00588c388941b1a33d4a38df8988d583bb27393bf2020ee2958.jpg)

![94ffeb50d647efa925731f646d621c357fc0397401aa59cd82da4771d9490e76.jpg](../icml_results/5761_HiCI_%20Hierarchical%20Construction%E2%80%93Integration%20for%20Long-Context%20Attention/tables/94ffeb50d647efa925731f646d621c357fc0397401aa59cd82da4771d9490e76.jpg)

![ac005c8827b50ab27a7978f6461f6d84ebb14290567e1fa1bc3c7369b864477d.jpg](../icml_results/5761_HiCI_%20Hierarchical%20Construction%E2%80%93Integration%20for%20Long-Context%20Attention/tables/ac005c8827b50ab27a7978f6461f6d84ebb14290567e1fa1bc3c7369b864477d.jpg)

![dd5c342739af6344b34e9034c7abc6a3f3464ba1496349be16ecfd126d013176.jpg](../icml_results/5761_HiCI_%20Hierarchical%20Construction%E2%80%93Integration%20for%20Long-Context%20Attention/tables/dd5c342739af6344b34e9034c7abc6a3f3464ba1496349be16ecfd126d013176.jpg)

![ea0034573fe0f4e13d2b1b60fbb47d0924503745e408934690c5cedf0497336a.jpg](../icml_results/5761_HiCI_%20Hierarchical%20Construction%E2%80%93Integration%20for%20Long-Context%20Attention/tables/ea0034573fe0f4e13d2b1b60fbb47d0924503745e408934690c5cedf0497336a.jpg)

![f794fe7d546ddc4c0bb5b2e5aba287b71941dd2309b24e9de496888931080d43.jpg](../icml_results/5761_HiCI_%20Hierarchical%20Construction%E2%80%93Integration%20for%20Long-Context%20Attention/tables/f794fe7d546ddc4c0bb5b2e5aba287b71941dd2309b24e9de496888931080d43.jpg)

<a id="paper-5713"></a>

## On Testing Conditional Mean Independence for Manifold-Valued Data


### Images

![673d353f97e9fc7a5888086f5757a55f8a211e5b639750957cc5e0935ddc4c0b.jpg](../icml_results/5762_On%20Testing%20Conditional%20Mean%20Independence%20for%20Manifold-Valued%20Data/images/673d353f97e9fc7a5888086f5757a55f8a211e5b639750957cc5e0935ddc4c0b.jpg)

![ca999fc3d7da72253823bba14d6a838c10a7345a1238065419a186119e829499.jpg](../icml_results/5762_On%20Testing%20Conditional%20Mean%20Independence%20for%20Manifold-Valued%20Data/images/ca999fc3d7da72253823bba14d6a838c10a7345a1238065419a186119e829499.jpg)

### Tables

![31d4891c14919809f55147413870936029294ef1f00c46eb5d37325f4cdbdab0.jpg](../icml_results/5762_On%20Testing%20Conditional%20Mean%20Independence%20for%20Manifold-Valued%20Data/tables/31d4891c14919809f55147413870936029294ef1f00c46eb5d37325f4cdbdab0.jpg)

![391f417629840d0fa94dac3cf4c98d9ad569e3cdd3c330062d441bff6699250d.jpg](../icml_results/5762_On%20Testing%20Conditional%20Mean%20Independence%20for%20Manifold-Valued%20Data/tables/391f417629840d0fa94dac3cf4c98d9ad569e3cdd3c330062d441bff6699250d.jpg)

![4a830296f1f8c058039029f7488124c83bc18c2faadf1b2b4942498c22fad18f.jpg](../icml_results/5762_On%20Testing%20Conditional%20Mean%20Independence%20for%20Manifold-Valued%20Data/tables/4a830296f1f8c058039029f7488124c83bc18c2faadf1b2b4942498c22fad18f.jpg)

![6cfa1ec1df61104e07099af23a7691593d761aa0a19a451c9dc805491d990f90.jpg](../icml_results/5762_On%20Testing%20Conditional%20Mean%20Independence%20for%20Manifold-Valued%20Data/tables/6cfa1ec1df61104e07099af23a7691593d761aa0a19a451c9dc805491d990f90.jpg)

![9df87ba814f501f83e5635995f509af836607fd33858ff04d73d21ac9cb26b95.jpg](../icml_results/5762_On%20Testing%20Conditional%20Mean%20Independence%20for%20Manifold-Valued%20Data/tables/9df87ba814f501f83e5635995f509af836607fd33858ff04d73d21ac9cb26b95.jpg)

![e2fb97a232faf6612eb56d17ba2896479662947239f9ca155613c800064547a3.jpg](../icml_results/5762_On%20Testing%20Conditional%20Mean%20Independence%20for%20Manifold-Valued%20Data/tables/e2fb97a232faf6612eb56d17ba2896479662947239f9ca155613c800064547a3.jpg)

<a id="paper-5714"></a>

## Fix Before Search: Benchmarking Agentic Visual Query Pre-processing in Multimodal Retrieval-augmented Generation


### Images

![05991c75e1bde46c6a56555e2f7f2d57d38cc1e3e4feeec39a29c6370164fd00.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/05991c75e1bde46c6a56555e2f7f2d57d38cc1e3e4feeec39a29c6370164fd00.jpg)

![0855f0b60410e65bb08f434ed4371f3533b07422084d00ffb259a91ea1684e5f.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/0855f0b60410e65bb08f434ed4371f3533b07422084d00ffb259a91ea1684e5f.jpg)

![2f5c728355c034017d3c19a58b49c07fa0d0278bd25e433086fc410b9e9e23df.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/2f5c728355c034017d3c19a58b49c07fa0d0278bd25e433086fc410b9e9e23df.jpg)

![44d265bc85e62044aa8a956cda49d59ce5cc9102ee24add3064e0059c4f56279.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/44d265bc85e62044aa8a956cda49d59ce5cc9102ee24add3064e0059c4f56279.jpg)

![6157f2ba446d7f8ab5c190cd6d76e6e385521b318579d107c58819dcaf352876.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/6157f2ba446d7f8ab5c190cd6d76e6e385521b318579d107c58819dcaf352876.jpg)

![75fbfefef5b179225e36d77ca5ada51931c9c1d6317cd8a6b28104407909c6b8.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/75fbfefef5b179225e36d77ca5ada51931c9c1d6317cd8a6b28104407909c6b8.jpg)

![a2af674b76940b719e07c84e124ae2d989421168a32491df133d82614607f810.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/a2af674b76940b719e07c84e124ae2d989421168a32491df133d82614607f810.jpg)

![a6527e30590c16f6d2a69a8807872f5be9dc751ea1b5106282a2cb89d18a9875.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/a6527e30590c16f6d2a69a8807872f5be9dc751ea1b5106282a2cb89d18a9875.jpg)

![a9ce1ddadcc343d469e4c9be80bb1bd7b893e29c001617c1348d48faf02af21d.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/a9ce1ddadcc343d469e4c9be80bb1bd7b893e29c001617c1348d48faf02af21d.jpg)

![c9dda598933c2890a700d61b6a46c610346e9901e5fc6176d796ed773af717d7.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/c9dda598933c2890a700d61b6a46c610346e9901e5fc6176d796ed773af717d7.jpg)

![cef15266dd5378962d9f49c8e6adf71c35634577094a178b4db02eda891dcebd.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/cef15266dd5378962d9f49c8e6adf71c35634577094a178b4db02eda891dcebd.jpg)

![d53356646efc460bb33e6bcf82e5f8f411cc11feadb2432d44929883fd652913.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/d53356646efc460bb33e6bcf82e5f8f411cc11feadb2432d44929883fd652913.jpg)

![d58beb7c3bb0b2851d83f70c5e5099db72d586da71f85042beb7dcf870d34ac9.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/d58beb7c3bb0b2851d83f70c5e5099db72d586da71f85042beb7dcf870d34ac9.jpg)

![d90801587eb7b2f267445b584a8c215011c48359646dd435868a7a8a027b874d.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/d90801587eb7b2f267445b584a8c215011c48359646dd435868a7a8a027b874d.jpg)

![d9e87014ffadabec21168de06ec5805f529d43af1ae31e638446ad19bc526774.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/d9e87014ffadabec21168de06ec5805f529d43af1ae31e638446ad19bc526774.jpg)

![e46785406739dfa7f9699904b874bcc18ec6a1ad8fc78e1f3e7dbaf831e86fd6.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/e46785406739dfa7f9699904b874bcc18ec6a1ad8fc78e1f3e7dbaf831e86fd6.jpg)

![ed5b4600e4868532e474688305c774b2eef2ac8c0e9f8135cf4c84da5cb40d32.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/ed5b4600e4868532e474688305c774b2eef2ac8c0e9f8135cf4c84da5cb40d32.jpg)

![f065ef6c9cf191cbd4f0caa8beb29a0f50a124c47367134887cbd352b2392c50.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/f065ef6c9cf191cbd4f0caa8beb29a0f50a124c47367134887cbd352b2392c50.jpg)

![f2898746a338509e1dec73b8df5305c9c12aaaefa81b46d4ea05bbad42408120.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/f2898746a338509e1dec73b8df5305c9c12aaaefa81b46d4ea05bbad42408120.jpg)

![fe82f437e7f761358501071ffdc16b3d7adc18b07601eaf5e4fcfd83205380b7.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/images/fe82f437e7f761358501071ffdc16b3d7adc18b07601eaf5e4fcfd83205380b7.jpg)

### Tables

![08f002f2a220223ef35e0a15e06eb6f85b0cc1caa190f80566feaba259487644.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/tables/08f002f2a220223ef35e0a15e06eb6f85b0cc1caa190f80566feaba259487644.jpg)

![1aa925f90afcc1cd0c78c1c00c9f58e3187aa67d22e0bda901f9743c6e7b3dfd.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/tables/1aa925f90afcc1cd0c78c1c00c9f58e3187aa67d22e0bda901f9743c6e7b3dfd.jpg)

![1ea034dc3e32bde5995e17639a246edf9b96aac37524b05f2442623db2cdea01.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/tables/1ea034dc3e32bde5995e17639a246edf9b96aac37524b05f2442623db2cdea01.jpg)

![2186cc0233f6974fd1fb6414ad06e2527aac3c31ed5279808b2f4828d87302a8.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/tables/2186cc0233f6974fd1fb6414ad06e2527aac3c31ed5279808b2f4828d87302a8.jpg)

![2fec850503f0698880eb568cad320bc2399a30611329939f0290900c7974c86a.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/tables/2fec850503f0698880eb568cad320bc2399a30611329939f0290900c7974c86a.jpg)

![34b9732456e47ac9e97d8e786b6934a1b8ee67075661407a65c730241f456a25.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/tables/34b9732456e47ac9e97d8e786b6934a1b8ee67075661407a65c730241f456a25.jpg)

![4c4ab405174ca723dd59f2e6c0d23cda45ff4faa74e5589bc3abb023c33d8c9e.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/tables/4c4ab405174ca723dd59f2e6c0d23cda45ff4faa74e5589bc3abb023c33d8c9e.jpg)

![6f4912ec99922545ef70eb1f46ac44273f936bc61798d044f13e8d7a71c7e974.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/tables/6f4912ec99922545ef70eb1f46ac44273f936bc61798d044f13e8d7a71c7e974.jpg)

![6f879134841aa60cd463b4d5de2e82cbcf40934d88de71ea9b75d91c4efe2e0c.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/tables/6f879134841aa60cd463b4d5de2e82cbcf40934d88de71ea9b75d91c4efe2e0c.jpg)

![ad562e86f3ce26888b77421688e103a020f9ae8bc57f3b013d2a1237a3500e33.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/tables/ad562e86f3ce26888b77421688e103a020f9ae8bc57f3b013d2a1237a3500e33.jpg)

![cee9127581f9a754a788f73d9c5d968e93b768ba0091d47f85610cf3dcaf9c66.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/tables/cee9127581f9a754a788f73d9c5d968e93b768ba0091d47f85610cf3dcaf9c66.jpg)

![d9cca0f0864a752068dc117a0a903aed63ca0f87be12a1993c75126c8f6c457d.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/tables/d9cca0f0864a752068dc117a0a903aed63ca0f87be12a1993c75126c8f6c457d.jpg)

![ef019855085d1e6017ddd8e8fa722849db10d82073b1fec1d64763d451f35764.jpg](../icml_results/5763_Fix%20Before%20Search_%20Benchmarking%20Agentic%20Visual%20Query%20Pre-processing%20in%20Multimodal%20Retrieval-augmente/tables/ef019855085d1e6017ddd8e8fa722849db10d82073b1fec1d64763d451f35764.jpg)

<a id="paper-5715"></a>

## ReJump: A Tree-Jump Representation for Analyzing and Improving LLM Reasoning


### Images

![016478593fc7ae770974314ab80a56082ee5064d05413e8e98466eb602c3235f.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/016478593fc7ae770974314ab80a56082ee5064d05413e8e98466eb602c3235f.jpg)

![07967a719d28697dadcb63f3883edc441f42bdb2761e2665a89172b09f41f206.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/07967a719d28697dadcb63f3883edc441f42bdb2761e2665a89172b09f41f206.jpg)

![19a047e7ed8ae2e70f53a2a4b1afb6b115a9aa024bc2b1eedd4de3089d782384.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/19a047e7ed8ae2e70f53a2a4b1afb6b115a9aa024bc2b1eedd4de3089d782384.jpg)

![32284f264ae4d66cfd3de54eadc5cf27a57436c329e82df33e523295e3b01890.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/32284f264ae4d66cfd3de54eadc5cf27a57436c329e82df33e523295e3b01890.jpg)

![4bcb4fb2bfb8437e5d157cfd7e0d2015fc8c69a029f5fc59cd35d8823cca76cf.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/4bcb4fb2bfb8437e5d157cfd7e0d2015fc8c69a029f5fc59cd35d8823cca76cf.jpg)

![58366a3252edd73115d2dade9e9c44f29645f15d989c64a8680ad56d67063d27.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/58366a3252edd73115d2dade9e9c44f29645f15d989c64a8680ad56d67063d27.jpg)

![58f5abdaeef238fca18327daa7f154df86795024965c46d70fdf85372049b723.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/58f5abdaeef238fca18327daa7f154df86795024965c46d70fdf85372049b723.jpg)

![62c28fbb44dcce285e96adbcfa2fc81956ad6642678feb19b2e07507a6a6d50d.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/62c28fbb44dcce285e96adbcfa2fc81956ad6642678feb19b2e07507a6a6d50d.jpg)

![664bc54f6ecb99e7bb9ce444e1a8dc62e420a95d0a9f20d2ff22bccb9ca1b9c4.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/664bc54f6ecb99e7bb9ce444e1a8dc62e420a95d0a9f20d2ff22bccb9ca1b9c4.jpg)

![6ce68e0890996c7bd5c8093f898066c544d11331b5938655f6ce8b6435518488.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/6ce68e0890996c7bd5c8093f898066c544d11331b5938655f6ce8b6435518488.jpg)

![6e411d5d130e866cc04ccbd348a85c8fb23cdb38367939edf4bed18de9493258.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/6e411d5d130e866cc04ccbd348a85c8fb23cdb38367939edf4bed18de9493258.jpg)

![82fb106a62360c1e93d814b1eecb5ff01719489e4c1c8ed93b78a529701aa850.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/82fb106a62360c1e93d814b1eecb5ff01719489e4c1c8ed93b78a529701aa850.jpg)

![87d4d8b07db65e5a4f2e490fd398d39b41f69583f3dd63f6872f100ce8746469.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/87d4d8b07db65e5a4f2e490fd398d39b41f69583f3dd63f6872f100ce8746469.jpg)

![8f683c2b27759065872f03ad6b6b5419e393479e9db9004ca2ba382670b2f63c.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/8f683c2b27759065872f03ad6b6b5419e393479e9db9004ca2ba382670b2f63c.jpg)

![94179923d683ac3ae56089cfc7aa2dc41dd773e2e466dc34c9d7a7e686c4bd7a.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/94179923d683ac3ae56089cfc7aa2dc41dd773e2e466dc34c9d7a7e686c4bd7a.jpg)

![94d9686d610b4fd1cda02fc9c4c65d6f44a2139b79912299f119fd6da8b93558.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/94d9686d610b4fd1cda02fc9c4c65d6f44a2139b79912299f119fd6da8b93558.jpg)

![95d3ce51f003727584f8a3c2ba03cc2b67c1873ee237885d927116362c128e94.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/95d3ce51f003727584f8a3c2ba03cc2b67c1873ee237885d927116362c128e94.jpg)

![9716fe5687c719ba8f6ea928f06712debc7ae1d1fdf42d3f4744baf2cbc59b56.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/9716fe5687c719ba8f6ea928f06712debc7ae1d1fdf42d3f4744baf2cbc59b56.jpg)

![a091c0e97897dfae622df92052a23cad1d21a42c98bc279fd2679744b865b02a.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/a091c0e97897dfae622df92052a23cad1d21a42c98bc279fd2679744b865b02a.jpg)

![b4be970ba9dcc7930b93195a44610816c1d070166b3e80933b6cdc7c8fda0ed0.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/b4be970ba9dcc7930b93195a44610816c1d070166b3e80933b6cdc7c8fda0ed0.jpg)

![bc9da89068edf3afe817850af005309ab435b778b8694b0d5627f3af8fa9dc8a.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/bc9da89068edf3afe817850af005309ab435b778b8694b0d5627f3af8fa9dc8a.jpg)

![ca46e2cdf725e16f38b5f83172f0b5483386b78a1d88f2024c7c776cded4b9e5.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/ca46e2cdf725e16f38b5f83172f0b5483386b78a1d88f2024c7c776cded4b9e5.jpg)

![d3638f1b6c319026e2c08ac1a7626f0ae21b2253ee0c47736c310815190f7f15.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/images/d3638f1b6c319026e2c08ac1a7626f0ae21b2253ee0c47736c310815190f7f15.jpg)

### Tables

![04285a6f3a16a7185978a442fd83cda093ca4aa1ef280cde5ba61e456c965fa8.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/tables/04285a6f3a16a7185978a442fd83cda093ca4aa1ef280cde5ba61e456c965fa8.jpg)

![0c7dcce95dc454cd01101a18fe9f0f7fbd9e183223d591e507deb5956bd65a1d.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/tables/0c7dcce95dc454cd01101a18fe9f0f7fbd9e183223d591e507deb5956bd65a1d.jpg)

![1dc12904c0fd2505a26b3972588ad795f61fc83851f1b2312ae1d51865a6dbdd.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/tables/1dc12904c0fd2505a26b3972588ad795f61fc83851f1b2312ae1d51865a6dbdd.jpg)

![20c1a4702600796050a73a74065fcc9edc58ecea4745e807be8cd5201e7b6939.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/tables/20c1a4702600796050a73a74065fcc9edc58ecea4745e807be8cd5201e7b6939.jpg)

![23b684ba3d887dabd11c2bb750ff058c5d933837a548ba3fc0b16de564a905c8.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/tables/23b684ba3d887dabd11c2bb750ff058c5d933837a548ba3fc0b16de564a905c8.jpg)

![80a79db8e1f9bc99cbb1a3e83e63055018b02557cc9abe57b775d88960ce533c.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/tables/80a79db8e1f9bc99cbb1a3e83e63055018b02557cc9abe57b775d88960ce533c.jpg)

![8b279ba55f4f634a0ac8896d2ed9a0cfac2631b89b2046b6f4c7ead9b2c1f1f9.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/tables/8b279ba55f4f634a0ac8896d2ed9a0cfac2631b89b2046b6f4c7ead9b2c1f1f9.jpg)

![985165d81d327ef5aa48f8cc77b0fff61eb2e21869db85a121de146de575af0b.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/tables/985165d81d327ef5aa48f8cc77b0fff61eb2e21869db85a121de146de575af0b.jpg)

![9b334ce044a5842b804ac818718b3b22baeefa0c2f5248593abc7094bd47b8fb.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/tables/9b334ce044a5842b804ac818718b3b22baeefa0c2f5248593abc7094bd47b8fb.jpg)

![9f7a8dac57dc87a3cbcb887dfa1cdfc9da61028fcbcd97d1bda937a8bde2d3cc.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/tables/9f7a8dac57dc87a3cbcb887dfa1cdfc9da61028fcbcd97d1bda937a8bde2d3cc.jpg)

![b9e24d428b009fe86f07c3e2d9a7eba19db87db45e370e55ab3516ced4b3cc3e.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/tables/b9e24d428b009fe86f07c3e2d9a7eba19db87db45e370e55ab3516ced4b3cc3e.jpg)

![c79ed9565fb4f33b231a4f4800c705c0951468028c86a07ce621cb1779894bd9.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/tables/c79ed9565fb4f33b231a4f4800c705c0951468028c86a07ce621cb1779894bd9.jpg)

![d7e0c055e97fbd0e3464963645742575e07e1d63947fa8fe804e45e2413e397d.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/tables/d7e0c055e97fbd0e3464963645742575e07e1d63947fa8fe804e45e2413e397d.jpg)

![ed87450595987106f310457ccd6549a7e37b33c38da407b665305abdf840070a.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/tables/ed87450595987106f310457ccd6549a7e37b33c38da407b665305abdf840070a.jpg)

![f0f109f7370ad15e234f414a1c47d23c7d3ba458b2c1278a47d0715b11988df7.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/tables/f0f109f7370ad15e234f414a1c47d23c7d3ba458b2c1278a47d0715b11988df7.jpg)

![f9f5e3dbece6c734035716097f3bf1a89470231f9e3a824b597bacab961c2ca7.jpg](../icml_results/5764_ReJump_%20A%20Tree-Jump%20Representation%20for%20Analyzing%20and%20Improving%20LLM%20Reasoning/tables/f9f5e3dbece6c734035716097f3bf1a89470231f9e3a824b597bacab961c2ca7.jpg)

<a id="paper-5716"></a>

## Video-o3: Native Interleaved Clue Seeking for Long Video Multi-Hop Reasoning


### Images

![06522b0c906a6632e420940a8635201b73ca0c0bde439613fc884aa1c9f93094.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/06522b0c906a6632e420940a8635201b73ca0c0bde439613fc884aa1c9f93094.jpg)

![07c6ec88799ef16229c8fccc67f10449ea17bca1d08cb37d086984e2ee7eaa55.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/07c6ec88799ef16229c8fccc67f10449ea17bca1d08cb37d086984e2ee7eaa55.jpg)

![2205f669010318f9f2bdeefe3a53595f017837ea4b776e639cc543dd15c948f4.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/2205f669010318f9f2bdeefe3a53595f017837ea4b776e639cc543dd15c948f4.jpg)

![22e17435857ac8c8ae786b4e418a93453dcb7f63cb2ca118e2de0f46a72442eb.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/22e17435857ac8c8ae786b4e418a93453dcb7f63cb2ca118e2de0f46a72442eb.jpg)

![250d60436372d1e03c71433ac5ad71092058752bf994bf6272aa82bbb631161b.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/250d60436372d1e03c71433ac5ad71092058752bf994bf6272aa82bbb631161b.jpg)

![32efb2f17f30300287ea6a82062e4e7592c9c0f121e287bffad1f0e1d15ea896.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/32efb2f17f30300287ea6a82062e4e7592c9c0f121e287bffad1f0e1d15ea896.jpg)

![36377b88761e044061b1a32097651d1c8d897f5bbd0845bbe02821b7f8e4eb03.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/36377b88761e044061b1a32097651d1c8d897f5bbd0845bbe02821b7f8e4eb03.jpg)

![48b3fb02aa9a02d00f07c24b3622008b1b4a34e64eb84952748b0bb8e7e1b176.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/48b3fb02aa9a02d00f07c24b3622008b1b4a34e64eb84952748b0bb8e7e1b176.jpg)

![4b8a3f47fe971c0ceeb3cda1bcc411bd05bc5d10f38b796a3193be18df4ac71a.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/4b8a3f47fe971c0ceeb3cda1bcc411bd05bc5d10f38b796a3193be18df4ac71a.jpg)

![51a48692b223463c51792650cbf3b8b836b0f864c283df9b436787cdbf6be37b.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/51a48692b223463c51792650cbf3b8b836b0f864c283df9b436787cdbf6be37b.jpg)

![64f992dcb918e4381f86b1db9da2e80cb4a10b9f236f08034c21d0da15c46d84.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/64f992dcb918e4381f86b1db9da2e80cb4a10b9f236f08034c21d0da15c46d84.jpg)

![68bacba9aff42b831a28696b2b2a5c09138082087faddf76f7e18686ba1f1ad6.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/68bacba9aff42b831a28696b2b2a5c09138082087faddf76f7e18686ba1f1ad6.jpg)

![77885cce9c06e31dab9ecad18457595e3cf089d1538f58a328c1b248ca6ea7ba.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/77885cce9c06e31dab9ecad18457595e3cf089d1538f58a328c1b248ca6ea7ba.jpg)

![8e8aebe5f2f322dd1328008f40f82cc7dd5a8062b780c1cd438b3bdea7ecb6b5.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/8e8aebe5f2f322dd1328008f40f82cc7dd5a8062b780c1cd438b3bdea7ecb6b5.jpg)

![905c3b2761cd684a26282d395da5e50130a57fe2575aa552e919b3f00c3e6244.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/905c3b2761cd684a26282d395da5e50130a57fe2575aa552e919b3f00c3e6244.jpg)

![9e930e6a15d04d57aee417de8668fbe0888185a5c686635be45a68f65e74fe9a.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/9e930e6a15d04d57aee417de8668fbe0888185a5c686635be45a68f65e74fe9a.jpg)

![9fc3416c567d1684dc40db5bc1ffeeef15643868aee81e24350c0befa138bfa1.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/9fc3416c567d1684dc40db5bc1ffeeef15643868aee81e24350c0befa138bfa1.jpg)

![a06f1a5593f06d2d1c60c8e50f6356c5c54d413ce95a543053db1a283bb41db3.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/a06f1a5593f06d2d1c60c8e50f6356c5c54d413ce95a543053db1a283bb41db3.jpg)

![bd95ac867186f801073e929190d8ba52d573128939d0aec2b0a1bd0896373f45.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/bd95ac867186f801073e929190d8ba52d573128939d0aec2b0a1bd0896373f45.jpg)

![c2e3f0ad349208082def8fec2c473f656cfd76d355f09c7202a78addd4cf1d2b.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/c2e3f0ad349208082def8fec2c473f656cfd76d355f09c7202a78addd4cf1d2b.jpg)

![c3904c1e1a89b9626eab8b4ce6e822317fc97f9257d8a6e35d96c29bfe7779a9.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/c3904c1e1a89b9626eab8b4ce6e822317fc97f9257d8a6e35d96c29bfe7779a9.jpg)

![c76fb02863cde0c547f825be3f6338e1dc2a68f63c76d0fb660d90899eb82176.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/c76fb02863cde0c547f825be3f6338e1dc2a68f63c76d0fb660d90899eb82176.jpg)

![c988373fc9dbf7c0cb99d91cd6b2cc8d16114cec74cdcf43d436bd3a5c0a6776.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/c988373fc9dbf7c0cb99d91cd6b2cc8d16114cec74cdcf43d436bd3a5c0a6776.jpg)

![ccab7eee5ccea5e5593f47cad4bca49ccc4f4e288831434eb50a158419647230.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/ccab7eee5ccea5e5593f47cad4bca49ccc4f4e288831434eb50a158419647230.jpg)

![d4019a09008df9d24fe91dc1cef27ec4c1a9075189de388baf91498ad7145179.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/d4019a09008df9d24fe91dc1cef27ec4c1a9075189de388baf91498ad7145179.jpg)

![d4f6a22c9fecab2caaf11c3a632b7f52f387328a47cf82d58bf7072efb4b2e02.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/d4f6a22c9fecab2caaf11c3a632b7f52f387328a47cf82d58bf7072efb4b2e02.jpg)

![e77f97c34797ca8c0d0e4b2a7322e0e3f175941949ea3322d3f22586f69be18e.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/e77f97c34797ca8c0d0e4b2a7322e0e3f175941949ea3322d3f22586f69be18e.jpg)

![ecf2c72fbc2cf263f4a441bef76c4144be51c940dcabe5f91d6908948d08d777.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/ecf2c72fbc2cf263f4a441bef76c4144be51c940dcabe5f91d6908948d08d777.jpg)

![ee5f91633033cec9f9846a75ebc2a8a596fd8bf0ce0f6cf4d664dc19cac0ae84.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/ee5f91633033cec9f9846a75ebc2a8a596fd8bf0ce0f6cf4d664dc19cac0ae84.jpg)

![f3e443ce104a47149df50e496386ab95d3b81d80aeeae8385fd8871b54cc5e79.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/images/f3e443ce104a47149df50e496386ab95d3b81d80aeeae8385fd8871b54cc5e79.jpg)

### Tables

![15017e02967b7a25a7c00eacd7bcda971a678bcc4d8a36c476799d7b82d0b06b.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/tables/15017e02967b7a25a7c00eacd7bcda971a678bcc4d8a36c476799d7b82d0b06b.jpg)

![1eb7ddfb5c5fb2834ad5c840bf878d449591cd2567fb5e3683bafc378ee727d1.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/tables/1eb7ddfb5c5fb2834ad5c840bf878d449591cd2567fb5e3683bafc378ee727d1.jpg)

![2dfb8397abf435761ca61b02ae459210e4263e1963d72bbb5cb3c3d5bcc9f49c.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/tables/2dfb8397abf435761ca61b02ae459210e4263e1963d72bbb5cb3c3d5bcc9f49c.jpg)

![3ef9a8a5cdde1a1a87c00a802abd7ff5c7f513cae2e9a4d3c1bda1b659f9895a.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/tables/3ef9a8a5cdde1a1a87c00a802abd7ff5c7f513cae2e9a4d3c1bda1b659f9895a.jpg)

![3fa04e92c1e3a2614254d7b4b7e63072acd9fe2af8c900eea9e8e9b29bad4177.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/tables/3fa04e92c1e3a2614254d7b4b7e63072acd9fe2af8c900eea9e8e9b29bad4177.jpg)

![5966656b0ea827e88a402691284a9cd324a0058495384784f1f207b59133c073.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/tables/5966656b0ea827e88a402691284a9cd324a0058495384784f1f207b59133c073.jpg)

![701dd50d01667511e0d57a0f48c1bc9251cb92bafe883be317e11a14494763c1.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/tables/701dd50d01667511e0d57a0f48c1bc9251cb92bafe883be317e11a14494763c1.jpg)

![716c03be4d50024cf633aeeed30198272868b612a6c11457cb29d6dfdfcfb66c.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/tables/716c03be4d50024cf633aeeed30198272868b612a6c11457cb29d6dfdfcfb66c.jpg)

![77325589b461ad67ce17d4c6986617144b0b84f243be3e01e46fb8117f8e6494.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/tables/77325589b461ad67ce17d4c6986617144b0b84f243be3e01e46fb8117f8e6494.jpg)

![7824c6c0ac8c03619602fdf28e6905d39d68a12e97fca7d4195aea4885e9baa4.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/tables/7824c6c0ac8c03619602fdf28e6905d39d68a12e97fca7d4195aea4885e9baa4.jpg)

![894713089be33685f40a1d96ed31154ca337acde62b45bcc07f64635198fafec.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/tables/894713089be33685f40a1d96ed31154ca337acde62b45bcc07f64635198fafec.jpg)

![a327174588d11f09411816a68deb10d378ce462e27a9fe5fb7b820d2aec5cd9d.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/tables/a327174588d11f09411816a68deb10d378ce462e27a9fe5fb7b820d2aec5cd9d.jpg)

![c19827379c57ee9abf9e9279a039ad7ddaea4ba8b362b4893d7817817fc2bef5.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/tables/c19827379c57ee9abf9e9279a039ad7ddaea4ba8b362b4893d7817817fc2bef5.jpg)

![c367de5a5fc8e079b75319501905c9f0f4d8fd3143879d4a53bc411156797a0e.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/tables/c367de5a5fc8e079b75319501905c9f0f4d8fd3143879d4a53bc411156797a0e.jpg)

![dd0e25874b00d98cda6832132e95327af4f6f648b6e90e6b03863bfa07781979.jpg](../icml_results/5765_Video-o3_%20Native%20Interleaved%20Clue%20Seeking%20for%20Long%20Video%20Multi-Hop%20Reasoning/tables/dd0e25874b00d98cda6832132e95327af4f6f648b6e90e6b03863bfa07781979.jpg)

<a id="paper-5717"></a>

## Shrinking the Variance: Shrinkage Baselines for Reinforcement Learning with Verifiable Rewards


### Images

![0907ca93629fa57ddbe1fed0f7ab7486f3d4d3a7bdd54207238d178cb23cc3ae.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/0907ca93629fa57ddbe1fed0f7ab7486f3d4d3a7bdd54207238d178cb23cc3ae.jpg)

![2dcc2e79e30342c7edcd99cfe82742c3b2a221c8325b8787911017c3da0df7e0.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/2dcc2e79e30342c7edcd99cfe82742c3b2a221c8325b8787911017c3da0df7e0.jpg)

![2e849c7af9038a2b41393b2d5b8f432ec1e4cad4a4457853035c73e9f2e5133f.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/2e849c7af9038a2b41393b2d5b8f432ec1e4cad4a4457853035c73e9f2e5133f.jpg)

![32ffdaa54d51970e75c7827dc3e46e9464a7ada9f33df3202ef088f93f3c87f4.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/32ffdaa54d51970e75c7827dc3e46e9464a7ada9f33df3202ef088f93f3c87f4.jpg)

![3361a5075faefa9c0754d3d85f37f8f0a6991d5a8fb9b21798d36c2db9aa029a.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/3361a5075faefa9c0754d3d85f37f8f0a6991d5a8fb9b21798d36c2db9aa029a.jpg)

![3b37c13d11c3bdfed9f361e56cc1369e8ecc63e621c6a6d5b132fb2fc877963b.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/3b37c13d11c3bdfed9f361e56cc1369e8ecc63e621c6a6d5b132fb2fc877963b.jpg)

![51f900c40550024663047c8922043e701f032cfcb75aacbe9f8837ca98d7af35.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/51f900c40550024663047c8922043e701f032cfcb75aacbe9f8837ca98d7af35.jpg)

![5604f3b6c15341c93d621535fa145eedc3173b6d0bfd5ed9989f8345c4cfd989.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/5604f3b6c15341c93d621535fa145eedc3173b6d0bfd5ed9989f8345c4cfd989.jpg)

![63fa1da3c10bd048ddc7844553230d8f5657b21cfbfc24d0c829637db50ee95a.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/63fa1da3c10bd048ddc7844553230d8f5657b21cfbfc24d0c829637db50ee95a.jpg)

![7f1f6cef1f2c55f37d6440f712afbc4a54cd7c58ea158cb02dc3a324f44f64ee.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/7f1f6cef1f2c55f37d6440f712afbc4a54cd7c58ea158cb02dc3a324f44f64ee.jpg)

![83b4b4bfa6cbb07b44d1b9f1abdceb73dd53522f0f030180bbc818c17424fa23.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/83b4b4bfa6cbb07b44d1b9f1abdceb73dd53522f0f030180bbc818c17424fa23.jpg)

![a2dab4a2a333151f92e6d7bce47bc3ef81ecbfbafef623ff9d841e1d654106c8.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/a2dab4a2a333151f92e6d7bce47bc3ef81ecbfbafef623ff9d841e1d654106c8.jpg)

![e461a1ed1b56b30906449beaade3c989fb7b10be506e7b25886f2de6636a172f.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/e461a1ed1b56b30906449beaade3c989fb7b10be506e7b25886f2de6636a172f.jpg)

![e5c7f3d2960c548b3066fa43453cc63ee98980e9f15661a83f59234a73016d6e.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/e5c7f3d2960c548b3066fa43453cc63ee98980e9f15661a83f59234a73016d6e.jpg)

![fc60faef7fdf7ff1870221bb916fb433a8a09e108a91666c2e11036a8f578605.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/fc60faef7fdf7ff1870221bb916fb433a8a09e108a91666c2e11036a8f578605.jpg)

![fe6c5cc351796dab0aae99b927c0230223851c56b30691f2adff83d5aac23c99.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/images/fe6c5cc351796dab0aae99b927c0230223851c56b30691f2adff83d5aac23c99.jpg)

### Tables

![01c9b0ac7bb517e789fbb201acc65ae6f306734919942f75c8b23af36b7cf858.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/01c9b0ac7bb517e789fbb201acc65ae6f306734919942f75c8b23af36b7cf858.jpg)

![03d44a2cf17e608e155de8a2c3e7846ed33fa96e3f4ef017056a404c3ea2e1a6.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/03d44a2cf17e608e155de8a2c3e7846ed33fa96e3f4ef017056a404c3ea2e1a6.jpg)

![0c26762b78c213e490d885c1fed0644620822e1a72016e89076961e40592b94b.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/0c26762b78c213e490d885c1fed0644620822e1a72016e89076961e40592b94b.jpg)

![0f5db8c6ee87f7b29d54b2a2e2cf56ae8e1bb40acb2f3f8ea4d9c60c9c98c544.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/0f5db8c6ee87f7b29d54b2a2e2cf56ae8e1bb40acb2f3f8ea4d9c60c9c98c544.jpg)

![108edd624166f2829d86d5ec073e2610c3f3599345c5ceda755f9b17f0d43c6c.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/108edd624166f2829d86d5ec073e2610c3f3599345c5ceda755f9b17f0d43c6c.jpg)

![14d53772ab0c2d62fbd89473c434761848efcc4d991ae2dc89f52155f1af4940.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/14d53772ab0c2d62fbd89473c434761848efcc4d991ae2dc89f52155f1af4940.jpg)

![1d4e01e6752a52d30b2c09156911423f28bc388de073a4eb50010280124ef41a.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/1d4e01e6752a52d30b2c09156911423f28bc388de073a4eb50010280124ef41a.jpg)

![1dc733f4c47c211f2bb61ad72babe72d62c82d21e90de8f7ca316847f0559f74.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/1dc733f4c47c211f2bb61ad72babe72d62c82d21e90de8f7ca316847f0559f74.jpg)

![3aa3663c6cdf82e2129416bcf91ee754d37ab89d05aa58b3cef42242ecdb554e.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/3aa3663c6cdf82e2129416bcf91ee754d37ab89d05aa58b3cef42242ecdb554e.jpg)

![4c7f1aab316ebcfa9e7fc58d68c9ce9d0739b68538401983efb01cb7be0184b3.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/4c7f1aab316ebcfa9e7fc58d68c9ce9d0739b68538401983efb01cb7be0184b3.jpg)

![4e69a4387c2436e0d37136a0eb1526016a10752aa1d45b89f77db380bf13ca39.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/4e69a4387c2436e0d37136a0eb1526016a10752aa1d45b89f77db380bf13ca39.jpg)

![5665bcdf0642748cf27022a09732c3e76cdda272379b13efb7dedc7d8191be5a.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/5665bcdf0642748cf27022a09732c3e76cdda272379b13efb7dedc7d8191be5a.jpg)

![802c1823d9b7dd56171c2a34e5c06f6d0fd673a3537c943dba616ae9675c4a6c.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/802c1823d9b7dd56171c2a34e5c06f6d0fd673a3537c943dba616ae9675c4a6c.jpg)

![845c15bdf2708754f0c0f3573300f107e64ad0dab1c8f26cd25471992a47caab.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/845c15bdf2708754f0c0f3573300f107e64ad0dab1c8f26cd25471992a47caab.jpg)

![85cb884b433aa849b2363d285af81fbc8cd98bc1c962828bc974dd55985234f3.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/85cb884b433aa849b2363d285af81fbc8cd98bc1c962828bc974dd55985234f3.jpg)

![9ee292846f5c7ca57124a903c2429fbe200d9984841e8420ad375760cdba5d4e.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/9ee292846f5c7ca57124a903c2429fbe200d9984841e8420ad375760cdba5d4e.jpg)

![a3584cb49adeba53d5df568fec9bfd5332520bf03d229a15dcff02c1de0ea560.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/a3584cb49adeba53d5df568fec9bfd5332520bf03d229a15dcff02c1de0ea560.jpg)

![b68f94973d3838eb54b0da98a9266efadc1669deb200960fb5d439d92e4dd4ce.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/b68f94973d3838eb54b0da98a9266efadc1669deb200960fb5d439d92e4dd4ce.jpg)

![bd0206c9f0d8c865d754906383937d486950c1f825bbec72a11785b503663cc9.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/bd0206c9f0d8c865d754906383937d486950c1f825bbec72a11785b503663cc9.jpg)

![dca5b3f99e25941c338cdb5f44e66e6967657370e7fae3eca3343a3a22b03299.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/dca5b3f99e25941c338cdb5f44e66e6967657370e7fae3eca3343a3a22b03299.jpg)

![e19f95cc69f500a41e8efa9cc3c3cb4acfe3f9a4c4b9243a7c10d50ea99627db.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/e19f95cc69f500a41e8efa9cc3c3cb4acfe3f9a4c4b9243a7c10d50ea99627db.jpg)

![f64526c804d68f6582e1cd18e039576fbc3db5f6b70a52f8ace12eca79aa4358.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/f64526c804d68f6582e1cd18e039576fbc3db5f6b70a52f8ace12eca79aa4358.jpg)

![faef7da6f26b7fa7a6745bcefb5b98b5afa41f9deb03add5d94fd5f040bff536.jpg](../icml_results/5766_Shrinking%20the%20Variance_%20Shrinkage%20Baselines%20for%20Reinforcement%20Learning%20with%20Verifiable%20Rewards/tables/faef7da6f26b7fa7a6745bcefb5b98b5afa41f9deb03add5d94fd5f040bff536.jpg)

<a id="paper-5718"></a>

## Discrete Diffusion with Physical Mass Constraints for De Novo Peptide Sequencing


### Images

![190fdc427a03df681b55279b40525bc41211cc158614d949378a56ef2a48b9a4.jpg](../icml_results/5767_Discrete%20Diffusion%20with%20Physical%20Mass%20Constraints%20for%20De%20Novo%20Peptide%20Sequencing/images/190fdc427a03df681b55279b40525bc41211cc158614d949378a56ef2a48b9a4.jpg)

![95db890f5cd6dd136cb6f5357ab48605979f049b218dcd37162168fb58703d79.jpg](../icml_results/5767_Discrete%20Diffusion%20with%20Physical%20Mass%20Constraints%20for%20De%20Novo%20Peptide%20Sequencing/images/95db890f5cd6dd136cb6f5357ab48605979f049b218dcd37162168fb58703d79.jpg)

![dc5654e6f675da0f3ce549290e451d6153bc63250c04f5a56d60872bc4f90898.jpg](../icml_results/5767_Discrete%20Diffusion%20with%20Physical%20Mass%20Constraints%20for%20De%20Novo%20Peptide%20Sequencing/images/dc5654e6f675da0f3ce549290e451d6153bc63250c04f5a56d60872bc4f90898.jpg)

![eecdd622fdd2f3bd31ad55d8031083f9bf3d42c41c8ac4f9a9798b3dc1cf49ef.jpg](../icml_results/5767_Discrete%20Diffusion%20with%20Physical%20Mass%20Constraints%20for%20De%20Novo%20Peptide%20Sequencing/images/eecdd622fdd2f3bd31ad55d8031083f9bf3d42c41c8ac4f9a9798b3dc1cf49ef.jpg)

![fb00f5ccff91b710290d456099d028c4d0f91f73230973cbec4311e8f4773c8b.jpg](../icml_results/5767_Discrete%20Diffusion%20with%20Physical%20Mass%20Constraints%20for%20De%20Novo%20Peptide%20Sequencing/images/fb00f5ccff91b710290d456099d028c4d0f91f73230973cbec4311e8f4773c8b.jpg)

### Tables

![04a0611a1c4d17e3e9d1e33a0166ebe6cb0f311fe27c0224a5d57a279eebd071.jpg](../icml_results/5767_Discrete%20Diffusion%20with%20Physical%20Mass%20Constraints%20for%20De%20Novo%20Peptide%20Sequencing/tables/04a0611a1c4d17e3e9d1e33a0166ebe6cb0f311fe27c0224a5d57a279eebd071.jpg)

![15f0d3adcdf9924ce8cfd9551d90daca2614c437eafc9ec0d966c0af81f685cf.jpg](../icml_results/5767_Discrete%20Diffusion%20with%20Physical%20Mass%20Constraints%20for%20De%20Novo%20Peptide%20Sequencing/tables/15f0d3adcdf9924ce8cfd9551d90daca2614c437eafc9ec0d966c0af81f685cf.jpg)

![17fae2e25f838ec769aa309b4b101fd86b17b8026fd8748b332229bf39a9fb6f.jpg](../icml_results/5767_Discrete%20Diffusion%20with%20Physical%20Mass%20Constraints%20for%20De%20Novo%20Peptide%20Sequencing/tables/17fae2e25f838ec769aa309b4b101fd86b17b8026fd8748b332229bf39a9fb6f.jpg)

![32c3e2da10cd79d325a7d6181c29543828644fb08f837a676a26b570f686026f.jpg](../icml_results/5767_Discrete%20Diffusion%20with%20Physical%20Mass%20Constraints%20for%20De%20Novo%20Peptide%20Sequencing/tables/32c3e2da10cd79d325a7d6181c29543828644fb08f837a676a26b570f686026f.jpg)

![3e99edc8fc4864fdfbcffe359c348cc7b0e468a3be88f9c4f16a6a057a42a479.jpg](../icml_results/5767_Discrete%20Diffusion%20with%20Physical%20Mass%20Constraints%20for%20De%20Novo%20Peptide%20Sequencing/tables/3e99edc8fc4864fdfbcffe359c348cc7b0e468a3be88f9c4f16a6a057a42a479.jpg)

![5035bc20b8a0176aac91ab8d9dd09a376e67e237486ae41238e15d1ab57cb54f.jpg](../icml_results/5767_Discrete%20Diffusion%20with%20Physical%20Mass%20Constraints%20for%20De%20Novo%20Peptide%20Sequencing/tables/5035bc20b8a0176aac91ab8d9dd09a376e67e237486ae41238e15d1ab57cb54f.jpg)

![56f069dfa792961df127b5f2c7654b90f00b7785e1281dbc6d99cb4a2dca37c7.jpg](../icml_results/5767_Discrete%20Diffusion%20with%20Physical%20Mass%20Constraints%20for%20De%20Novo%20Peptide%20Sequencing/tables/56f069dfa792961df127b5f2c7654b90f00b7785e1281dbc6d99cb4a2dca37c7.jpg)

![98e689f1ea6e59c41089d403209369def13afd4f419c58cda2a60d9cdea6a18f.jpg](../icml_results/5767_Discrete%20Diffusion%20with%20Physical%20Mass%20Constraints%20for%20De%20Novo%20Peptide%20Sequencing/tables/98e689f1ea6e59c41089d403209369def13afd4f419c58cda2a60d9cdea6a18f.jpg)

![bb99e3eee0829bb6d216e706187b3a7392b02e00bfa63c9e78c86b43ec37791a.jpg](../icml_results/5767_Discrete%20Diffusion%20with%20Physical%20Mass%20Constraints%20for%20De%20Novo%20Peptide%20Sequencing/tables/bb99e3eee0829bb6d216e706187b3a7392b02e00bfa63c9e78c86b43ec37791a.jpg)

![d191f9c17549112d90400e4e26c0d6d405ca4ec6d3c59ab2a735d37919e3e816.jpg](../icml_results/5767_Discrete%20Diffusion%20with%20Physical%20Mass%20Constraints%20for%20De%20Novo%20Peptide%20Sequencing/tables/d191f9c17549112d90400e4e26c0d6d405ca4ec6d3c59ab2a735d37919e3e816.jpg)

<a id="paper-5719"></a>

## Physiology as Language: Translating Respiration to EEG during Sleep


### Images

![35b1ff3b5873fe2013ebc4ce1c63747901beb3ab6b0c3d1f5689e8b5e5120170.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/images/35b1ff3b5873fe2013ebc4ce1c63747901beb3ab6b0c3d1f5689e8b5e5120170.jpg)

![759ae70a0bc2314d05a90c392b5742a4305911753f2d29eec206b375635b8c75.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/images/759ae70a0bc2314d05a90c392b5742a4305911753f2d29eec206b375635b8c75.jpg)

![78f662c9faab99f562b69b2ce081a6d9a4f00c9720b100628f9cfa0a0ec48ad8.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/images/78f662c9faab99f562b69b2ce081a6d9a4f00c9720b100628f9cfa0a0ec48ad8.jpg)

![a3611f73cc476c2672c90765a03afdbcb7e67be70533251cfd7a226768236111.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/images/a3611f73cc476c2672c90765a03afdbcb7e67be70533251cfd7a226768236111.jpg)

![cdb3053d253fa81a116b6eab7e534b8127b3c10e73ad28c50c7b2c8bf2f2caeb.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/images/cdb3053d253fa81a116b6eab7e534b8127b3c10e73ad28c50c7b2c8bf2f2caeb.jpg)

![f97c2bac663c841d53104554badcdade7701e40b675727608df1a7c53e7db164.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/images/f97c2bac663c841d53104554badcdade7701e40b675727608df1a7c53e7db164.jpg)

### Tables

![00cfbcb2d38dd0f49a1fc85e922838db1c3a8af7b28aee79ba5b256af32010e2.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/tables/00cfbcb2d38dd0f49a1fc85e922838db1c3a8af7b28aee79ba5b256af32010e2.jpg)

![01a1b05945f3d1411c067c403f2c48666c27e5dd95c2978cc81f01cdf5cf10d5.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/tables/01a1b05945f3d1411c067c403f2c48666c27e5dd95c2978cc81f01cdf5cf10d5.jpg)

![0a455f18979247526d2603770ac22c1e466c62d0a2174c48519a31eed2515824.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/tables/0a455f18979247526d2603770ac22c1e466c62d0a2174c48519a31eed2515824.jpg)

![1661b443824e3c77dd6a73198eb10dcb898da8480555f5a7b4ff96eb550f48d0.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/tables/1661b443824e3c77dd6a73198eb10dcb898da8480555f5a7b4ff96eb550f48d0.jpg)

![4351fdf091f171811c4dbc20ec8cda3b47412afc27b196a60ac2b9b290c9b592.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/tables/4351fdf091f171811c4dbc20ec8cda3b47412afc27b196a60ac2b9b290c9b592.jpg)

![4de6cd95e86089d960419ce38dbe5c09114153840a2b5121753817eba947058a.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/tables/4de6cd95e86089d960419ce38dbe5c09114153840a2b5121753817eba947058a.jpg)

![660aed212abf2d37a3bc8234cc42de65c10e96cc245783ddd903ec6f4911da88.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/tables/660aed212abf2d37a3bc8234cc42de65c10e96cc245783ddd903ec6f4911da88.jpg)

![79cf4d9fc5c3cc88b1bc4a9886c652c8edbfe6e81d0cbc5dc3bb207d6197c993.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/tables/79cf4d9fc5c3cc88b1bc4a9886c652c8edbfe6e81d0cbc5dc3bb207d6197c993.jpg)

![ae2bbf1f2b923323a6a0ecb0d3225a0cbb3f358a0aeab80581ee760dc5514ad3.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/tables/ae2bbf1f2b923323a6a0ecb0d3225a0cbb3f358a0aeab80581ee760dc5514ad3.jpg)

![baa894b1ea8a93cdc273181c3944b7d0e29ff329e076ba6a25500ac69b7a7a35.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/tables/baa894b1ea8a93cdc273181c3944b7d0e29ff329e076ba6a25500ac69b7a7a35.jpg)

![c16687ddf6eae119a0af8d9179707998eb654f2465707cb7517de851ff12c5ab.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/tables/c16687ddf6eae119a0af8d9179707998eb654f2465707cb7517de851ff12c5ab.jpg)

![c98cf8ee01d38137ad62fbcc1cd58a3a2038c4f4af17f5a617ba8f3e813847db.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/tables/c98cf8ee01d38137ad62fbcc1cd58a3a2038c4f4af17f5a617ba8f3e813847db.jpg)

![d6496a794ab62012a378515648111d6cc4ad7e4aa19d589b05c856d927586ad3.jpg](../icml_results/5768_Physiology%20as%20Language_%20Translating%20Respiration%20to%20EEG%20during%20Sleep/tables/d6496a794ab62012a378515648111d6cc4ad7e4aa19d589b05c856d927586ad3.jpg)

<a id="paper-5720"></a>

## Graph is a Natural Regularization: Revisiting Vector Quantization for Graph Representation Learning


### Images

![2f22c239bb9e9221956124dd41500b8883d60bb16d1b98cfedc555eb99ad107f.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/images/2f22c239bb9e9221956124dd41500b8883d60bb16d1b98cfedc555eb99ad107f.jpg)

![5b575325863992d432d3fd8527a8451873f64c735261fe4dce826bc90094f610.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/images/5b575325863992d432d3fd8527a8451873f64c735261fe4dce826bc90094f610.jpg)

![66c24c6fd3c8805f3560411aab5e0e4ac5de0eae3326e2f13569855f3e6d310e.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/images/66c24c6fd3c8805f3560411aab5e0e4ac5de0eae3326e2f13569855f3e6d310e.jpg)

![74c585734a08168236160690908fd1422aa4d6c0e32731566599f152dafefe06.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/images/74c585734a08168236160690908fd1422aa4d6c0e32731566599f152dafefe06.jpg)

![d018bb5a0050ca33ecafa7ddf453ec579ebaecab93cca33ccbf8432a42e87458.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/images/d018bb5a0050ca33ecafa7ddf453ec579ebaecab93cca33ccbf8432a42e87458.jpg)

![da84879aa67db9a0da401754971436459aee7d315341a7780a681b02ab631cb3.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/images/da84879aa67db9a0da401754971436459aee7d315341a7780a681b02ab631cb3.jpg)

![e7a60d7a65649cc7a3051d986c5696bb522e7e94fe4d45e0b2fd17a48f171067.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/images/e7a60d7a65649cc7a3051d986c5696bb522e7e94fe4d45e0b2fd17a48f171067.jpg)

### Tables

![03657e54376ce4a05cb0a98abc311588f33fd55f48a994697f76c983b06179b7.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/tables/03657e54376ce4a05cb0a98abc311588f33fd55f48a994697f76c983b06179b7.jpg)

![0b5d6ac39895692cd41bcf660f16aa0d674e5d5197a6056e945606a01875c7a4.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/tables/0b5d6ac39895692cd41bcf660f16aa0d674e5d5197a6056e945606a01875c7a4.jpg)

![0dba4b516c26f32f230a8172449e7ce7af1175d4b7dd3d210ad97e37cc9c5fd1.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/tables/0dba4b516c26f32f230a8172449e7ce7af1175d4b7dd3d210ad97e37cc9c5fd1.jpg)

![75b44294b487f97ae813af1a3a433f60fa2ba4293591c389798d876b783089a1.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/tables/75b44294b487f97ae813af1a3a433f60fa2ba4293591c389798d876b783089a1.jpg)

![7a893871c3e0130682b096511ce05cefc753522b464f9176bfc8dd0bf2a659f1.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/tables/7a893871c3e0130682b096511ce05cefc753522b464f9176bfc8dd0bf2a659f1.jpg)

![7b6e232e9889c320775e16dc3abcdba68bc45b36ba12bc33029ab4ab12a1abee.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/tables/7b6e232e9889c320775e16dc3abcdba68bc45b36ba12bc33029ab4ab12a1abee.jpg)

![7f002625944ffa3576387a97f43abc8751d8822bd84ca9f7a50677fa92a1e716.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/tables/7f002625944ffa3576387a97f43abc8751d8822bd84ca9f7a50677fa92a1e716.jpg)

![857b8553ce928d54eed7e46f9ccb06af991facf6d368fea166d0461474f42405.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/tables/857b8553ce928d54eed7e46f9ccb06af991facf6d368fea166d0461474f42405.jpg)

![a8468feafa45cd167406491b72c21f012879d9bf189ede9ac1c6227d980308da.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/tables/a8468feafa45cd167406491b72c21f012879d9bf189ede9ac1c6227d980308da.jpg)

![db7f3ad507c9e60869fa3e786cf7b01cc95c5f06433b83f1da0cb3614c1e6da6.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/tables/db7f3ad507c9e60869fa3e786cf7b01cc95c5f06433b83f1da0cb3614c1e6da6.jpg)

![df8556ba0841d8f1e88e4bcd050f8dfe5b53f0bd3afc246cbf211ff2748fd9f2.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/tables/df8556ba0841d8f1e88e4bcd050f8dfe5b53f0bd3afc246cbf211ff2748fd9f2.jpg)

![ee93cc31b2da9d0d78650f6a45f1272dda596df95dd0f95b8b9c4faa2ad1194a.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/tables/ee93cc31b2da9d0d78650f6a45f1272dda596df95dd0f95b8b9c4faa2ad1194a.jpg)

![f11992f55eef26ede6c05be05a3c1e7621b4339e2c6e5a56ffba89a89976b3f6.jpg](../icml_results/5769_Graph%20is%20a%20Natural%20Regularization_%20Revisiting%20Vector%20Quantization%20for%20Graph%20Representation%20Learning/tables/f11992f55eef26ede6c05be05a3c1e7621b4339e2c6e5a56ffba89a89976b3f6.jpg)

<a id="paper-5721"></a>

## CORE: Context-Robust Remasking for Diffusion Language Models


### Images

![69d42dd6315b35e908dc4fc468ad4992cae96d112aac2b76a60e7ed6d529b4ae.jpg](../icml_results/5770_CORE_%20Context-Robust%20Remasking%20for%20Diffusion%20Language%20Models/images/69d42dd6315b35e908dc4fc468ad4992cae96d112aac2b76a60e7ed6d529b4ae.jpg)

![6cb6a4f327e92c4dd8aea39474a428dd623f3fca604476757e1535e6beb7ff32.jpg](../icml_results/5770_CORE_%20Context-Robust%20Remasking%20for%20Diffusion%20Language%20Models/images/6cb6a4f327e92c4dd8aea39474a428dd623f3fca604476757e1535e6beb7ff32.jpg)

![f57155d5e92863103692db152da1b337bf0ff62e8f094979e6b48371e9c4687e.jpg](../icml_results/5770_CORE_%20Context-Robust%20Remasking%20for%20Diffusion%20Language%20Models/images/f57155d5e92863103692db152da1b337bf0ff62e8f094979e6b48371e9c4687e.jpg)

### Tables

![334a9262a9d8db2bb5ee9b3765986782ac51a7b7688c1ba5c582715a4adf0722.jpg](../icml_results/5770_CORE_%20Context-Robust%20Remasking%20for%20Diffusion%20Language%20Models/tables/334a9262a9d8db2bb5ee9b3765986782ac51a7b7688c1ba5c582715a4adf0722.jpg)

![60a6a6bc9408f8e833be5ceebbb35cbedd6e17c9713b7516fe573c41f00cf9e4.jpg](../icml_results/5770_CORE_%20Context-Robust%20Remasking%20for%20Diffusion%20Language%20Models/tables/60a6a6bc9408f8e833be5ceebbb35cbedd6e17c9713b7516fe573c41f00cf9e4.jpg)

![6eaf5ff418cffe6cbead0cb430abf04edd63693115b702725357ec20185befc9.jpg](../icml_results/5770_CORE_%20Context-Robust%20Remasking%20for%20Diffusion%20Language%20Models/tables/6eaf5ff418cffe6cbead0cb430abf04edd63693115b702725357ec20185befc9.jpg)

![76a6ba5e16bcdeb45c0cbe846bcd1b4033ea7f01a9f3d1e3fc9c115c12942d54.jpg](../icml_results/5770_CORE_%20Context-Robust%20Remasking%20for%20Diffusion%20Language%20Models/tables/76a6ba5e16bcdeb45c0cbe846bcd1b4033ea7f01a9f3d1e3fc9c115c12942d54.jpg)

![8c9547230aa051405ff54c5b4aa49f81088a55103a41d52974a2a327d1b79522.jpg](../icml_results/5770_CORE_%20Context-Robust%20Remasking%20for%20Diffusion%20Language%20Models/tables/8c9547230aa051405ff54c5b4aa49f81088a55103a41d52974a2a327d1b79522.jpg)

![c6d7a9f74b8900c1ff531ddcdd99774ad31a5c1f387a547595c68a7bec7d02c2.jpg](../icml_results/5770_CORE_%20Context-Robust%20Remasking%20for%20Diffusion%20Language%20Models/tables/c6d7a9f74b8900c1ff531ddcdd99774ad31a5c1f387a547595c68a7bec7d02c2.jpg)

![d15a04f028fd9dfdff789ac4ad092045796cbeb76d85edaa0f5f7b4172e29918.jpg](../icml_results/5770_CORE_%20Context-Robust%20Remasking%20for%20Diffusion%20Language%20Models/tables/d15a04f028fd9dfdff789ac4ad092045796cbeb76d85edaa0f5f7b4172e29918.jpg)

![e093c80d913a107cf4bb4f8a06ba63729a774a9012b8c378a05440be4fe5c830.jpg](../icml_results/5770_CORE_%20Context-Robust%20Remasking%20for%20Diffusion%20Language%20Models/tables/e093c80d913a107cf4bb4f8a06ba63729a774a9012b8c378a05440be4fe5c830.jpg)

<a id="paper-5722"></a>

## Distilling Task-Level Coordination Policies for Generalizable Multi-Agent Cooperation


### Images

![2783ef1bfe77a8a4023d42d57c9f042b45ffc9753f64ac42c1321581d5c3b6e6.jpg](../icml_results/5771_Distilling%20Task-Level%20Coordination%20Policies%20for%20Generalizable%20Multi-Agent%20Cooperation/images/2783ef1bfe77a8a4023d42d57c9f042b45ffc9753f64ac42c1321581d5c3b6e6.jpg)

![32d250ddf7550da3194dfd3a141c67bb48b2a35bbe68fb5ee04fdbb0ed2659ba.jpg](../icml_results/5771_Distilling%20Task-Level%20Coordination%20Policies%20for%20Generalizable%20Multi-Agent%20Cooperation/images/32d250ddf7550da3194dfd3a141c67bb48b2a35bbe68fb5ee04fdbb0ed2659ba.jpg)

![7404d1fa65178a782c311a68acf5b244b5f89e8c85f2ea50d3b0de50157e953d.jpg](../icml_results/5771_Distilling%20Task-Level%20Coordination%20Policies%20for%20Generalizable%20Multi-Agent%20Cooperation/images/7404d1fa65178a782c311a68acf5b244b5f89e8c85f2ea50d3b0de50157e953d.jpg)

![8b2bb06c937c9ae1abd9fdff5eb27f62e589bb545dfd86c1780f21dcdccf473c.jpg](../icml_results/5771_Distilling%20Task-Level%20Coordination%20Policies%20for%20Generalizable%20Multi-Agent%20Cooperation/images/8b2bb06c937c9ae1abd9fdff5eb27f62e589bb545dfd86c1780f21dcdccf473c.jpg)

![be7075658c54b09c7c9f7118fe4a8d5e35fc29410dc9b123cb2619cb9ce806d7.jpg](../icml_results/5771_Distilling%20Task-Level%20Coordination%20Policies%20for%20Generalizable%20Multi-Agent%20Cooperation/images/be7075658c54b09c7c9f7118fe4a8d5e35fc29410dc9b123cb2619cb9ce806d7.jpg)

![d56d4d6a65cc034e2f8cf0003adf0141c58f736f5e5c1dc0435db2eac01eb161.jpg](../icml_results/5771_Distilling%20Task-Level%20Coordination%20Policies%20for%20Generalizable%20Multi-Agent%20Cooperation/images/d56d4d6a65cc034e2f8cf0003adf0141c58f736f5e5c1dc0435db2eac01eb161.jpg)

### Tables

![3a33565725a7b49a63be543e69448594dc1ab76b1380d1a8babf33d0efc17d4e.jpg](../icml_results/5771_Distilling%20Task-Level%20Coordination%20Policies%20for%20Generalizable%20Multi-Agent%20Cooperation/tables/3a33565725a7b49a63be543e69448594dc1ab76b1380d1a8babf33d0efc17d4e.jpg)

![3cb3f19898ad86ba6136850c2e3e644b8db83c398e15df648f0273dd9d451d4f.jpg](../icml_results/5771_Distilling%20Task-Level%20Coordination%20Policies%20for%20Generalizable%20Multi-Agent%20Cooperation/tables/3cb3f19898ad86ba6136850c2e3e644b8db83c398e15df648f0273dd9d451d4f.jpg)

![6420f67ffb7a3e72be930d14a19fb5cc1ff639ec1e97b7f57b1d972c05417cb3.jpg](../icml_results/5771_Distilling%20Task-Level%20Coordination%20Policies%20for%20Generalizable%20Multi-Agent%20Cooperation/tables/6420f67ffb7a3e72be930d14a19fb5cc1ff639ec1e97b7f57b1d972c05417cb3.jpg)

![6be9f042e485c5677ae523abe6c803788d6d9f3efc3ab5ec430e886bd13c3151.jpg](../icml_results/5771_Distilling%20Task-Level%20Coordination%20Policies%20for%20Generalizable%20Multi-Agent%20Cooperation/tables/6be9f042e485c5677ae523abe6c803788d6d9f3efc3ab5ec430e886bd13c3151.jpg)

![7760091a48041347029eda03a233cf065103c117fbd4180b69ffdf3190c74fe3.jpg](../icml_results/5771_Distilling%20Task-Level%20Coordination%20Policies%20for%20Generalizable%20Multi-Agent%20Cooperation/tables/7760091a48041347029eda03a233cf065103c117fbd4180b69ffdf3190c74fe3.jpg)

![8724158e0d2aba79ede7de191cd8cd3ec2754e7521fc4cf03febfa5cc8820989.jpg](../icml_results/5771_Distilling%20Task-Level%20Coordination%20Policies%20for%20Generalizable%20Multi-Agent%20Cooperation/tables/8724158e0d2aba79ede7de191cd8cd3ec2754e7521fc4cf03febfa5cc8820989.jpg)

![8797e029ad1392aa4c3caff644a059c63d39d6a6d4c36016183eeca0f06ea680.jpg](../icml_results/5771_Distilling%20Task-Level%20Coordination%20Policies%20for%20Generalizable%20Multi-Agent%20Cooperation/tables/8797e029ad1392aa4c3caff644a059c63d39d6a6d4c36016183eeca0f06ea680.jpg)

![94e97a05b654213aa568db5cad0a97b0c753b83880e9324c5babd6f66610794f.jpg](../icml_results/5771_Distilling%20Task-Level%20Coordination%20Policies%20for%20Generalizable%20Multi-Agent%20Cooperation/tables/94e97a05b654213aa568db5cad0a97b0c753b83880e9324c5babd6f66610794f.jpg)

<a id="paper-5723"></a>

## SCOPE: Evolving Symbolic World for Planning in Open-Ended Environments


### Images

![057dea32e62d3da33f33cfe45ab71d73643afafcf809bc679619dd3a053c10b6.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/images/057dea32e62d3da33f33cfe45ab71d73643afafcf809bc679619dd3a053c10b6.jpg)

![4bd4bca2ceffac84f8f666f9038890bdd29f56bdfafb5bcf9c0641fed57b40c4.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/images/4bd4bca2ceffac84f8f666f9038890bdd29f56bdfafb5bcf9c0641fed57b40c4.jpg)

![4e589bfa03bcb9d5c0f795f90bcc3e01f5e08a602ab1bc692253c024762ffdb0.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/images/4e589bfa03bcb9d5c0f795f90bcc3e01f5e08a602ab1bc692253c024762ffdb0.jpg)

![540737ba6afb3d1d852ab8eef29751653466f6123a3c7c71fa6f1b6184767ce6.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/images/540737ba6afb3d1d852ab8eef29751653466f6123a3c7c71fa6f1b6184767ce6.jpg)

![5f7df5518a0ef23096e03f462bfe7a42c8beb1e91a868cf523705f496ed50da8.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/images/5f7df5518a0ef23096e03f462bfe7a42c8beb1e91a868cf523705f496ed50da8.jpg)

![6d94ec609aeff19d85e8c3902e370371c4d10ddb0c44b689eef6955035e5ceca.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/images/6d94ec609aeff19d85e8c3902e370371c4d10ddb0c44b689eef6955035e5ceca.jpg)

![833291e07f7ce679dc97fc7550f13e93240d65c6aa8b8a0dbb10bd4cafd01bd9.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/images/833291e07f7ce679dc97fc7550f13e93240d65c6aa8b8a0dbb10bd4cafd01bd9.jpg)

![a3cd9f6488c2373b53046e30311b11e4ba54049bbc7b897ba2cca8075ffdb11a.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/images/a3cd9f6488c2373b53046e30311b11e4ba54049bbc7b897ba2cca8075ffdb11a.jpg)

### Tables

![1108d77c152ab877acecbdc6ec6fbb15e17626f400ba894e4bb2869576ebc79d.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/tables/1108d77c152ab877acecbdc6ec6fbb15e17626f400ba894e4bb2869576ebc79d.jpg)

![2adaf261d5098ed7096de3add02c1eaf4bd8e7c10675610acee0a4ff31351737.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/tables/2adaf261d5098ed7096de3add02c1eaf4bd8e7c10675610acee0a4ff31351737.jpg)

![459834cfa896f92d8478accd3506dc337b5ef6b55f60d14d3ef8b70bf8c479ef.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/tables/459834cfa896f92d8478accd3506dc337b5ef6b55f60d14d3ef8b70bf8c479ef.jpg)

![60b041571044dea628eb13329374653c41861621f29cea942d953a38a8c21120.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/tables/60b041571044dea628eb13329374653c41861621f29cea942d953a38a8c21120.jpg)

![6c522ad50b98d24618384512a8bdca5f44896b8c1a97084f4181740df31af6aa.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/tables/6c522ad50b98d24618384512a8bdca5f44896b8c1a97084f4181740df31af6aa.jpg)

![88a9ab5e4646a61a4eeae2f18bcfa63ff29643d3c48fdf444a51e7bc4771066c.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/tables/88a9ab5e4646a61a4eeae2f18bcfa63ff29643d3c48fdf444a51e7bc4771066c.jpg)

![aa1859dd03ba41eba5ea1c01a23449d76b6c0eb56ed726334a3bb45332652aa4.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/tables/aa1859dd03ba41eba5ea1c01a23449d76b6c0eb56ed726334a3bb45332652aa4.jpg)

![afb056bf79eab25b236b65fc86805817b82f70706e6fc5fc8820d1a1fe4f0f52.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/tables/afb056bf79eab25b236b65fc86805817b82f70706e6fc5fc8820d1a1fe4f0f52.jpg)

![cb981c5fb44bd100f0a79892318a04269dc8ff4b63a3738d641ba6f3ad5853d3.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/tables/cb981c5fb44bd100f0a79892318a04269dc8ff4b63a3738d641ba6f3ad5853d3.jpg)

![e722ccea651e23038d7cf227e9a50af2f825898e53e74a490808c4e01d3a0678.jpg](../icml_results/5772_SCOPE_%20Evolving%20Symbolic%20World%20for%20Planning%20in%20Open-Ended%20Environments/tables/e722ccea651e23038d7cf227e9a50af2f825898e53e74a490808c4e01d3a0678.jpg)

<a id="paper-5724"></a>

## Simple Policy Gradients for Reasoning with Diffusion Language Models


### Images

![02b8e65f1f602762147736eb21ab8ccfb486c0b90b9c3b913c5f7bce9fd3ca97.jpg](../icml_results/5773_Simple%20Policy%20Gradients%20for%20Reasoning%20with%20Diffusion%20Language%20Models/images/02b8e65f1f602762147736eb21ab8ccfb486c0b90b9c3b913c5f7bce9fd3ca97.jpg)

![678b6d13ad4833a9d75228a732fc55550a74d050f6fe6142d6fea9b459bcade5.jpg](../icml_results/5773_Simple%20Policy%20Gradients%20for%20Reasoning%20with%20Diffusion%20Language%20Models/images/678b6d13ad4833a9d75228a732fc55550a74d050f6fe6142d6fea9b459bcade5.jpg)

![caf0b1c63774d4d472df3ed7d7b25b7e05270613d8d1d3a39c3932b924c6a05e.jpg](../icml_results/5773_Simple%20Policy%20Gradients%20for%20Reasoning%20with%20Diffusion%20Language%20Models/images/caf0b1c63774d4d472df3ed7d7b25b7e05270613d8d1d3a39c3932b924c6a05e.jpg)

![d329e22bfd555b0067f19cdb9a0b40062945cc59f0bbb20c5e7e798187bd7ebe.jpg](../icml_results/5773_Simple%20Policy%20Gradients%20for%20Reasoning%20with%20Diffusion%20Language%20Models/images/d329e22bfd555b0067f19cdb9a0b40062945cc59f0bbb20c5e7e798187bd7ebe.jpg)

### Tables

![004ed18f1e763e3bad7aca05e1d386563b93b92f62e6281303a68c292074e697.jpg](../icml_results/5773_Simple%20Policy%20Gradients%20for%20Reasoning%20with%20Diffusion%20Language%20Models/tables/004ed18f1e763e3bad7aca05e1d386563b93b92f62e6281303a68c292074e697.jpg)

![38051808c404a17c135041771c39937493ecb16ee8d5426b85a337727f5adf54.jpg](../icml_results/5773_Simple%20Policy%20Gradients%20for%20Reasoning%20with%20Diffusion%20Language%20Models/tables/38051808c404a17c135041771c39937493ecb16ee8d5426b85a337727f5adf54.jpg)

![6004c7dee92bbff62b3915ead435be0d3f3cdeb256a92b97414a9b6d3a271660.jpg](../icml_results/5773_Simple%20Policy%20Gradients%20for%20Reasoning%20with%20Diffusion%20Language%20Models/tables/6004c7dee92bbff62b3915ead435be0d3f3cdeb256a92b97414a9b6d3a271660.jpg)

![9797c550d54a710335e01a58f38c2c79d4300ae24a41e9a9153c7a0bc510eeab.jpg](../icml_results/5773_Simple%20Policy%20Gradients%20for%20Reasoning%20with%20Diffusion%20Language%20Models/tables/9797c550d54a710335e01a58f38c2c79d4300ae24a41e9a9153c7a0bc510eeab.jpg)

![9b399e5474f61b19b4b001a77b148eda2706e9449e1c6775572b56486a8d0abd.jpg](../icml_results/5773_Simple%20Policy%20Gradients%20for%20Reasoning%20with%20Diffusion%20Language%20Models/tables/9b399e5474f61b19b4b001a77b148eda2706e9449e1c6775572b56486a8d0abd.jpg)

![b67e59a534ac71fc07042c7d9624e29aacd4cbfe88b6040168a95aa9e9c05934.jpg](../icml_results/5773_Simple%20Policy%20Gradients%20for%20Reasoning%20with%20Diffusion%20Language%20Models/tables/b67e59a534ac71fc07042c7d9624e29aacd4cbfe88b6040168a95aa9e9c05934.jpg)

![ce150f7a7c30231f455f1b5c728913c5065b4148ffb497d73cbb9bbfbc7886c5.jpg](../icml_results/5773_Simple%20Policy%20Gradients%20for%20Reasoning%20with%20Diffusion%20Language%20Models/tables/ce150f7a7c30231f455f1b5c728913c5065b4148ffb497d73cbb9bbfbc7886c5.jpg)

![ce51f5fa441fdf77ad49f4fec212cafd96e1c3d7eaa6dc737a02c2ed00d3aca2.jpg](../icml_results/5773_Simple%20Policy%20Gradients%20for%20Reasoning%20with%20Diffusion%20Language%20Models/tables/ce51f5fa441fdf77ad49f4fec212cafd96e1c3d7eaa6dc737a02c2ed00d3aca2.jpg)

<a id="paper-5725"></a>

## Mind Your Entropy: From Maximum Entropy to Trajectory Entropy-Constrained RL


### Images

![158f01666371688de71c8a92e2323228c844382d12cb27805bdf1f8b1dc68d18.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/images/158f01666371688de71c8a92e2323228c844382d12cb27805bdf1f8b1dc68d18.jpg)

![15d8d6f38ccdb4cdf77852a2aa7fc5a716599a8ed400a0cf324d79d9cef444f8.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/images/15d8d6f38ccdb4cdf77852a2aa7fc5a716599a8ed400a0cf324d79d9cef444f8.jpg)

![1742b982764587ebb0e8e664af32a89d0b99d0ddc0e3169eb13fe53529466b88.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/images/1742b982764587ebb0e8e664af32a89d0b99d0ddc0e3169eb13fe53529466b88.jpg)

![2928d90484006ed31d0f0c3df0e556c91a8b8af24b903784ce7a0ba8035158cf.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/images/2928d90484006ed31d0f0c3df0e556c91a8b8af24b903784ce7a0ba8035158cf.jpg)

![39ef9da1da9407dbb4c0a38baa762e12cc816a054ed4296ff163d5402a803648.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/images/39ef9da1da9407dbb4c0a38baa762e12cc816a054ed4296ff163d5402a803648.jpg)

![4330c6bef5ae3892b460cd4e04212f3e597280e9da366917785d95f9da45406c.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/images/4330c6bef5ae3892b460cd4e04212f3e597280e9da366917785d95f9da45406c.jpg)

![75d289e10d3fb530e06ef815452685946ed1bc27f153e1f4c4e77315b76c11a7.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/images/75d289e10d3fb530e06ef815452685946ed1bc27f153e1f4c4e77315b76c11a7.jpg)

![76ab8fdbb106d3fa14ff190295fc0a6aa76504ee539499799167f5d884d126fd.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/images/76ab8fdbb106d3fa14ff190295fc0a6aa76504ee539499799167f5d884d126fd.jpg)

![9fb8c9e37d30fd38fe450bb2dd0b0284a8a442fda83861ed22f282fec0722b1d.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/images/9fb8c9e37d30fd38fe450bb2dd0b0284a8a442fda83861ed22f282fec0722b1d.jpg)

![a9bde00a6e80ffa48dad48afbb885ca58c73e0b97965dc49918c836029df47ad.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/images/a9bde00a6e80ffa48dad48afbb885ca58c73e0b97965dc49918c836029df47ad.jpg)

![dfacddff3eedcddd92f5d96d8f434883335c5c8613a72727e25aaf41fec25d5e.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/images/dfacddff3eedcddd92f5d96d8f434883335c5c8613a72727e25aaf41fec25d5e.jpg)

![f78ae27cd52a4416f7787e099a7da140337b896bbc33b9c0124459901b98ba10.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/images/f78ae27cd52a4416f7787e099a7da140337b896bbc33b9c0124459901b98ba10.jpg)

### Tables

![02fbdf31df7149374428e869b8beee7f3ffe525e97772d5e4a52013ebff79757.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/tables/02fbdf31df7149374428e869b8beee7f3ffe525e97772d5e4a52013ebff79757.jpg)

![43b723c7d5c8938dc2872c50f091a2dbd1db8371a59f27a94b7d8f1abc61bf34.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/tables/43b723c7d5c8938dc2872c50f091a2dbd1db8371a59f27a94b7d8f1abc61bf34.jpg)

![74c764f3cfe183062fb50d566ed065700dd0d8bbb2149d6cdcb0bd13961ccd5a.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/tables/74c764f3cfe183062fb50d566ed065700dd0d8bbb2149d6cdcb0bd13961ccd5a.jpg)

![9e06bb859ffeeaac845b410165cd2ec0b4dac41a21b96c7a74867cbbadc790bc.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/tables/9e06bb859ffeeaac845b410165cd2ec0b4dac41a21b96c7a74867cbbadc790bc.jpg)

![b4e0757da7dd1c68b8e12938a68f1871ffd2726d70658ece2cdb7790bf9ba1bd.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/tables/b4e0757da7dd1c68b8e12938a68f1871ffd2726d70658ece2cdb7790bf9ba1bd.jpg)

![cd8e0c5a56037a68110b2837f1e7fa98555931e219e1b32b9981bc8fdab68fb1.jpg](../icml_results/5774_Mind%20Your%20Entropy_%20From%20Maximum%20Entropy%20to%20Trajectory%20Entropy-Constrained%20RL/tables/cd8e0c5a56037a68110b2837f1e7fa98555931e219e1b32b9981bc8fdab68fb1.jpg)

<a id="paper-5726"></a>

## Harmonized Dual Policy Improvement for Modelic Reinforcement Learning


### Images

![02072bcfb18baa78516fd5481db65d20c82767d9e293a431fc5d108f97c6cdbb.jpg](../icml_results/5775_Harmonized%20Dual%20Policy%20Improvement%20for%20Modelic%20Reinforcement%20Learning/images/02072bcfb18baa78516fd5481db65d20c82767d9e293a431fc5d108f97c6cdbb.jpg)

![250f16a41440bb4f7b7572be4c69e53e8ebb38b60a91d04dc7303d2c8d5ab243.jpg](../icml_results/5775_Harmonized%20Dual%20Policy%20Improvement%20for%20Modelic%20Reinforcement%20Learning/images/250f16a41440bb4f7b7572be4c69e53e8ebb38b60a91d04dc7303d2c8d5ab243.jpg)

![41e0eeda176c3d9925bdfeab83f0a1e26015c2486d90d2234b14b05cfd8156d7.jpg](../icml_results/5775_Harmonized%20Dual%20Policy%20Improvement%20for%20Modelic%20Reinforcement%20Learning/images/41e0eeda176c3d9925bdfeab83f0a1e26015c2486d90d2234b14b05cfd8156d7.jpg)

![5cf1388ec30db493cc41948f15cd5b83da2560ab9b8fcc2a8e8c471ec5bd048e.jpg](../icml_results/5775_Harmonized%20Dual%20Policy%20Improvement%20for%20Modelic%20Reinforcement%20Learning/images/5cf1388ec30db493cc41948f15cd5b83da2560ab9b8fcc2a8e8c471ec5bd048e.jpg)

![5e0211be8ddcba0be0f72029656446f7880bc0d0ecf8a63e65013675884f6378.jpg](../icml_results/5775_Harmonized%20Dual%20Policy%20Improvement%20for%20Modelic%20Reinforcement%20Learning/images/5e0211be8ddcba0be0f72029656446f7880bc0d0ecf8a63e65013675884f6378.jpg)

![8d421c80dfba64a0a0be24df5147dbafacd35fbdabd89e7432579821f9ac979c.jpg](../icml_results/5775_Harmonized%20Dual%20Policy%20Improvement%20for%20Modelic%20Reinforcement%20Learning/images/8d421c80dfba64a0a0be24df5147dbafacd35fbdabd89e7432579821f9ac979c.jpg)

![8ede07de2de8cc047552987a4c282a2785c0e6abd5d454b216c9360efad811fc.jpg](../icml_results/5775_Harmonized%20Dual%20Policy%20Improvement%20for%20Modelic%20Reinforcement%20Learning/images/8ede07de2de8cc047552987a4c282a2785c0e6abd5d454b216c9360efad811fc.jpg)

![97cf1e9471afc9e4ea0fd48ccee0b7e61ee6fbf739b08244d84430351d2b928d.jpg](../icml_results/5775_Harmonized%20Dual%20Policy%20Improvement%20for%20Modelic%20Reinforcement%20Learning/images/97cf1e9471afc9e4ea0fd48ccee0b7e61ee6fbf739b08244d84430351d2b928d.jpg)

![9f065ce66cb30be48051359fd3ae39c5c9816a4c9d8e205e144503098ed9c50f.jpg](../icml_results/5775_Harmonized%20Dual%20Policy%20Improvement%20for%20Modelic%20Reinforcement%20Learning/images/9f065ce66cb30be48051359fd3ae39c5c9816a4c9d8e205e144503098ed9c50f.jpg)

![a2dbd05e88f815e7bc2b5c8e227eaf5f0680231c5f3f50830e6d6ca48088e017.jpg](../icml_results/5775_Harmonized%20Dual%20Policy%20Improvement%20for%20Modelic%20Reinforcement%20Learning/images/a2dbd05e88f815e7bc2b5c8e227eaf5f0680231c5f3f50830e6d6ca48088e017.jpg)

![a61bdbd4f82ab10dfe69fdd2f9362ded5e1d94865fa7203ba23b6eabda93b144.jpg](../icml_results/5775_Harmonized%20Dual%20Policy%20Improvement%20for%20Modelic%20Reinforcement%20Learning/images/a61bdbd4f82ab10dfe69fdd2f9362ded5e1d94865fa7203ba23b6eabda93b144.jpg)

![af81e28f4cfa5d78dd57878e75bd2f609e44368797cf9cecc1504f00ba9fab90.jpg](../icml_results/5775_Harmonized%20Dual%20Policy%20Improvement%20for%20Modelic%20Reinforcement%20Learning/images/af81e28f4cfa5d78dd57878e75bd2f609e44368797cf9cecc1504f00ba9fab90.jpg)

![f26f1dec14f3b835c45ae483161b3175278b69252a0de0fccf2a0eb368da9334.jpg](../icml_results/5775_Harmonized%20Dual%20Policy%20Improvement%20for%20Modelic%20Reinforcement%20Learning/images/f26f1dec14f3b835c45ae483161b3175278b69252a0de0fccf2a0eb368da9334.jpg)

![f98ddd80f008c28dc88e28b06545308d77925c266ce56ac4818c6b38944df01c.jpg](../icml_results/5775_Harmonized%20Dual%20Policy%20Improvement%20for%20Modelic%20Reinforcement%20Learning/images/f98ddd80f008c28dc88e28b06545308d77925c266ce56ac4818c6b38944df01c.jpg)

### Tables

![0d06b360433446e88ab0ba6c80091ea83df760ff57856ebce57c34229f3ce22a.jpg](../icml_results/5775_Harmonized%20Dual%20Policy%20Improvement%20for%20Modelic%20Reinforcement%20Learning/tables/0d06b360433446e88ab0ba6c80091ea83df760ff57856ebce57c34229f3ce22a.jpg)

![b6b09da45a07532622bce0a08aa6fe2535d67eb69071ec98f980410834bb63e5.jpg](../icml_results/5775_Harmonized%20Dual%20Policy%20Improvement%20for%20Modelic%20Reinforcement%20Learning/tables/b6b09da45a07532622bce0a08aa6fe2535d67eb69071ec98f980410834bb63e5.jpg)

![e4ab8a5d977c6a30b3d92cdadd97896bcfb62c0d66b346e9fa0492a64771bad4.jpg](../icml_results/5775_Harmonized%20Dual%20Policy%20Improvement%20for%20Modelic%20Reinforcement%20Learning/tables/e4ab8a5d977c6a30b3d92cdadd97896bcfb62c0d66b346e9fa0492a64771bad4.jpg)

<a id="paper-5727"></a>

## Guidance: Sentence-Level Citation Enforcement via Prefix-Tail Guidance during LLM Decoding


### Images

![5be7b89d8b2e9f7fd72cff182c4c6679f96d6d3cc088ee8fc158a1242e16e4a5.jpg](../icml_results/5776_Guidance_%20Sentence-Level%20Citation%20Enforcement%20via%20Prefix-Tail%20Guidance%20during%20LLM%20Decoding/images/5be7b89d8b2e9f7fd72cff182c4c6679f96d6d3cc088ee8fc158a1242e16e4a5.jpg)

![9d270232b43a84f3ab0ec76cdf403a83bc153458511de13ac83e5792d92eb06a.jpg](../icml_results/5776_Guidance_%20Sentence-Level%20Citation%20Enforcement%20via%20Prefix-Tail%20Guidance%20during%20LLM%20Decoding/images/9d270232b43a84f3ab0ec76cdf403a83bc153458511de13ac83e5792d92eb06a.jpg)

![ffe27a172101e13b7e5968f6cf0c0510f53a95ce3e296a6a8877aa828ad28636.jpg](../icml_results/5776_Guidance_%20Sentence-Level%20Citation%20Enforcement%20via%20Prefix-Tail%20Guidance%20during%20LLM%20Decoding/images/ffe27a172101e13b7e5968f6cf0c0510f53a95ce3e296a6a8877aa828ad28636.jpg)

### Tables

![0aaa04c4a95baaa6a47d463400c5a671a5e9939be3cfabede6dbd57af6a2f131.jpg](../icml_results/5776_Guidance_%20Sentence-Level%20Citation%20Enforcement%20via%20Prefix-Tail%20Guidance%20during%20LLM%20Decoding/tables/0aaa04c4a95baaa6a47d463400c5a671a5e9939be3cfabede6dbd57af6a2f131.jpg)

![0f50180f3b4c9a00b87ac00f793d8669ccb1a768019beca99004e05f61a2422c.jpg](../icml_results/5776_Guidance_%20Sentence-Level%20Citation%20Enforcement%20via%20Prefix-Tail%20Guidance%20during%20LLM%20Decoding/tables/0f50180f3b4c9a00b87ac00f793d8669ccb1a768019beca99004e05f61a2422c.jpg)

![1f0fad4b68a53d8b1f8b54a9aac4811920eb5941eb3f0e8e9e0a80a243e232bf.jpg](../icml_results/5776_Guidance_%20Sentence-Level%20Citation%20Enforcement%20via%20Prefix-Tail%20Guidance%20during%20LLM%20Decoding/tables/1f0fad4b68a53d8b1f8b54a9aac4811920eb5941eb3f0e8e9e0a80a243e232bf.jpg)

![1fd75233b41f71ba337ae6bde21a150281d117e6c91cb93e32a299e78e3d9df7.jpg](../icml_results/5776_Guidance_%20Sentence-Level%20Citation%20Enforcement%20via%20Prefix-Tail%20Guidance%20during%20LLM%20Decoding/tables/1fd75233b41f71ba337ae6bde21a150281d117e6c91cb93e32a299e78e3d9df7.jpg)

![23ee7b5e1f3cccc1758b1f78a919ffd647a64b3b5f25c1c5e2e7b7397ee2e51e.jpg](../icml_results/5776_Guidance_%20Sentence-Level%20Citation%20Enforcement%20via%20Prefix-Tail%20Guidance%20during%20LLM%20Decoding/tables/23ee7b5e1f3cccc1758b1f78a919ffd647a64b3b5f25c1c5e2e7b7397ee2e51e.jpg)

![544a51b97bfcfe42585fcaf3710a9cfea2e23f925da80704e46d7611b7d47362.jpg](../icml_results/5776_Guidance_%20Sentence-Level%20Citation%20Enforcement%20via%20Prefix-Tail%20Guidance%20during%20LLM%20Decoding/tables/544a51b97bfcfe42585fcaf3710a9cfea2e23f925da80704e46d7611b7d47362.jpg)

![57fbcc7d40f6cbdb397a15f4825d51cc566f02e395da97c7e10ab1666a09dc93.jpg](../icml_results/5776_Guidance_%20Sentence-Level%20Citation%20Enforcement%20via%20Prefix-Tail%20Guidance%20during%20LLM%20Decoding/tables/57fbcc7d40f6cbdb397a15f4825d51cc566f02e395da97c7e10ab1666a09dc93.jpg)

![6243ace441afea1f022b19ffffdb4a30c902b09ce3c02868232ab1388af4ced4.jpg](../icml_results/5776_Guidance_%20Sentence-Level%20Citation%20Enforcement%20via%20Prefix-Tail%20Guidance%20during%20LLM%20Decoding/tables/6243ace441afea1f022b19ffffdb4a30c902b09ce3c02868232ab1388af4ced4.jpg)

![635005ce97fcbddbf961abc6597de37852d6704a9a09c864211248cbcd50bcea.jpg](../icml_results/5776_Guidance_%20Sentence-Level%20Citation%20Enforcement%20via%20Prefix-Tail%20Guidance%20during%20LLM%20Decoding/tables/635005ce97fcbddbf961abc6597de37852d6704a9a09c864211248cbcd50bcea.jpg)

![7d5558d6d715323c4f3137bfec06a8fb70e063a427ee72d118074baa5205b14e.jpg](../icml_results/5776_Guidance_%20Sentence-Level%20Citation%20Enforcement%20via%20Prefix-Tail%20Guidance%20during%20LLM%20Decoding/tables/7d5558d6d715323c4f3137bfec06a8fb70e063a427ee72d118074baa5205b14e.jpg)

![c4a70faf303f2dbdda4f38f5632b5c788fc858cddf7138efe5b2657ec25f835e.jpg](../icml_results/5776_Guidance_%20Sentence-Level%20Citation%20Enforcement%20via%20Prefix-Tail%20Guidance%20during%20LLM%20Decoding/tables/c4a70faf303f2dbdda4f38f5632b5c788fc858cddf7138efe5b2657ec25f835e.jpg)

![cbc457e93ae01e866af6c173423bf8bd3dd87d3cef8ff249e0cf873ee27f11e0.jpg](../icml_results/5776_Guidance_%20Sentence-Level%20Citation%20Enforcement%20via%20Prefix-Tail%20Guidance%20during%20LLM%20Decoding/tables/cbc457e93ae01e866af6c173423bf8bd3dd87d3cef8ff249e0cf873ee27f11e0.jpg)

![d9c75fedcaeec136dc64cfb9a242b152741675854bf7bcf426d687db95dda0b9.jpg](../icml_results/5776_Guidance_%20Sentence-Level%20Citation%20Enforcement%20via%20Prefix-Tail%20Guidance%20during%20LLM%20Decoding/tables/d9c75fedcaeec136dc64cfb9a242b152741675854bf7bcf426d687db95dda0b9.jpg)

<a id="paper-5728"></a>

## Decomposition-Based Modular Conformal Prediction for Two-Stage Modeling


### Images

![0930ddf431cce01e827a370e6f62ee2d7e55cc2a6fa136077dc16bf551412893.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/0930ddf431cce01e827a370e6f62ee2d7e55cc2a6fa136077dc16bf551412893.jpg)

![14bb14f802381b25f6c1b5861f8268e3d0e7c05a3fbfeea88baa9cb91b316fb2.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/14bb14f802381b25f6c1b5861f8268e3d0e7c05a3fbfeea88baa9cb91b316fb2.jpg)

![1bb8885642bacd7e42cd479be8c43c2a7168b0a1a0625859aab929d16362a61a.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/1bb8885642bacd7e42cd479be8c43c2a7168b0a1a0625859aab929d16362a61a.jpg)

![1c5de66d0e23a6a7fc971f717cc2422891653f62b8e12dfa0aaf055c00f6fd6e.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/1c5de66d0e23a6a7fc971f717cc2422891653f62b8e12dfa0aaf055c00f6fd6e.jpg)

![270dc6bf6575caad9d11692aee34e17dcda0b8403203ac87132ff30a024fca7a.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/270dc6bf6575caad9d11692aee34e17dcda0b8403203ac87132ff30a024fca7a.jpg)

![2d855a3fd0b63583a071aace1f7a5c154cb05a10c34eaeb091d6b8276dbde1c7.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/2d855a3fd0b63583a071aace1f7a5c154cb05a10c34eaeb091d6b8276dbde1c7.jpg)

![3e8d94aca018ad78df0ccc122a90cdb20d131d48c5353f6ded138be4c8b6e85d.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/3e8d94aca018ad78df0ccc122a90cdb20d131d48c5353f6ded138be4c8b6e85d.jpg)

![48289e4880ea6eb5ae722598ec4983da57f94f45bf8d70cae5d0c1b94808ceb6.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/48289e4880ea6eb5ae722598ec4983da57f94f45bf8d70cae5d0c1b94808ceb6.jpg)

![59a7ee240427975f56e70aeb4e001950e69473e689463d9dff4b978117bbfe97.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/59a7ee240427975f56e70aeb4e001950e69473e689463d9dff4b978117bbfe97.jpg)

![6301c6543b219ce78bb9b120e140346314fa5a9acfd4c41049d33028cc38c935.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/6301c6543b219ce78bb9b120e140346314fa5a9acfd4c41049d33028cc38c935.jpg)

![76ad53319f2f1b32a822eb2e89bdbbf7c3b05649bbe184536fdce6eef35c88af.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/76ad53319f2f1b32a822eb2e89bdbbf7c3b05649bbe184536fdce6eef35c88af.jpg)

![7b2840933c408d7ab0dc2e0016c59e5fd6011e646cef9d5175bee2558d947278.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/7b2840933c408d7ab0dc2e0016c59e5fd6011e646cef9d5175bee2558d947278.jpg)

![844a6e822bbb77499419c5322a98e7d56288a0c4a09c4d8b0aac5a833c3834f6.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/844a6e822bbb77499419c5322a98e7d56288a0c4a09c4d8b0aac5a833c3834f6.jpg)

![84c9bc162140cb8ebde8880eac48a7e54865e51146dbf2f57233ff67386b53c4.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/84c9bc162140cb8ebde8880eac48a7e54865e51146dbf2f57233ff67386b53c4.jpg)

![8627d81db1ebc152bbe35e67169a9acfff7620fa5f1d45bed2334eb696374c37.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/8627d81db1ebc152bbe35e67169a9acfff7620fa5f1d45bed2334eb696374c37.jpg)

![90b04f41654b6d16bbd94b2dfeab6fd58fcb3e7f47443e838a9a7ae741f1bfe8.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/90b04f41654b6d16bbd94b2dfeab6fd58fcb3e7f47443e838a9a7ae741f1bfe8.jpg)

![9ab07907d4e09f5844e05114622a964260b01681cb14109d8ec0f86cd2690ceb.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/9ab07907d4e09f5844e05114622a964260b01681cb14109d8ec0f86cd2690ceb.jpg)

![a2249b9367291fd30dae700b9c662e1346d33daff4f00fa05f5f855b452f6c6a.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/a2249b9367291fd30dae700b9c662e1346d33daff4f00fa05f5f855b452f6c6a.jpg)

![a2837363f2a26643567cd7f3fc304093715ac2b779eb919d150fa1dd2fe5abee.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/a2837363f2a26643567cd7f3fc304093715ac2b779eb919d150fa1dd2fe5abee.jpg)

![af3a9d66e3179ab5cb6f045798b2429936427d957a551c749b5d71599b86fb72.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/af3a9d66e3179ab5cb6f045798b2429936427d957a551c749b5d71599b86fb72.jpg)

![b4776c5c9d12310500c845652ec38d73a38cf1a795a2c090dbf3827ac41213df.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/b4776c5c9d12310500c845652ec38d73a38cf1a795a2c090dbf3827ac41213df.jpg)

![bfab03fe43383196b6635429734654aa8e7e0686a086605d5412bd0da93629f5.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/bfab03fe43383196b6635429734654aa8e7e0686a086605d5412bd0da93629f5.jpg)

![c2341d11ac6e91e843c4158dfe52c0d83de2e4b167fe1bf4d6b8e2ff03a123a0.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/c2341d11ac6e91e843c4158dfe52c0d83de2e4b167fe1bf4d6b8e2ff03a123a0.jpg)

![ca8f33f5b2af66d74d364cd99e1f9bd5bf73f51f4af43af1c2757405d0ce9736.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/ca8f33f5b2af66d74d364cd99e1f9bd5bf73f51f4af43af1c2757405d0ce9736.jpg)

![d4f6a2be267971e6b722d12d983daa0393395508d28f95f7bb7f4ed345bcbfdc.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/d4f6a2be267971e6b722d12d983daa0393395508d28f95f7bb7f4ed345bcbfdc.jpg)

![eb5ced7e0ad4221fe47d5c4e5112eff51169dc2c37885373b42b3da5d261efb3.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/eb5ced7e0ad4221fe47d5c4e5112eff51169dc2c37885373b42b3da5d261efb3.jpg)

![ecb44ad4833ed5fb208c82728308f3b607a5a57736b16ab71e834be7448cbb41.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/ecb44ad4833ed5fb208c82728308f3b607a5a57736b16ab71e834be7448cbb41.jpg)

![ee5e564c35272c09e94bd7413200a2e54cf7736ac095f76419d8a7c688ca85cc.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/ee5e564c35272c09e94bd7413200a2e54cf7736ac095f76419d8a7c688ca85cc.jpg)

![ef0d54812bcbbd191c9fde79cc3d379cba456186c67b33dd0336261ebe0f4f98.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/images/ef0d54812bcbbd191c9fde79cc3d379cba456186c67b33dd0336261ebe0f4f98.jpg)

### Tables

![095da07a558573b116f55da8a90f65a228bd8ff4dbf074af1ed0001d1f710978.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/tables/095da07a558573b116f55da8a90f65a228bd8ff4dbf074af1ed0001d1f710978.jpg)

![2e74c4adb7893a92b84792182631f8b925ddf79e416d407a23b118eade39085b.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/tables/2e74c4adb7893a92b84792182631f8b925ddf79e416d407a23b118eade39085b.jpg)

![2f3281ec36cdec7a32b7da13bc0fe30aa5506cf9f9d5ba58c86455f28c13e979.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/tables/2f3281ec36cdec7a32b7da13bc0fe30aa5506cf9f9d5ba58c86455f28c13e979.jpg)

![4d13459a0390d3441ddb6e143fc8de687d10ffbb3e4303e1b65c8aba520d7164.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/tables/4d13459a0390d3441ddb6e143fc8de687d10ffbb3e4303e1b65c8aba520d7164.jpg)

![567ea921914afe9d0dad8f44c141c2fdfc28fe9bf9f2cb52b3466ed90a53b4d7.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/tables/567ea921914afe9d0dad8f44c141c2fdfc28fe9bf9f2cb52b3466ed90a53b4d7.jpg)

![5c3fcb1965b282f2766dfe5ac94fabaf0e15120f3c94f13b144a94e60f2b88de.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/tables/5c3fcb1965b282f2766dfe5ac94fabaf0e15120f3c94f13b144a94e60f2b88de.jpg)

![613583e580d1770e51389cb0adbf009669a9dd2fe2c59dcb07c6006839f81b8a.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/tables/613583e580d1770e51389cb0adbf009669a9dd2fe2c59dcb07c6006839f81b8a.jpg)

![7faddb0a5948c79ecbfc8a6c23457cda189450a538da79f0768fd11e478621c7.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/tables/7faddb0a5948c79ecbfc8a6c23457cda189450a538da79f0768fd11e478621c7.jpg)

![8c6d3042fcbee83885fd2fa6c2902a4ea2603a25911863e766bc45962793b96e.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/tables/8c6d3042fcbee83885fd2fa6c2902a4ea2603a25911863e766bc45962793b96e.jpg)

![d100e51b4ff5d8ae35bbe93a2c09cbbf48e6f30bb96b472ded62b0c68a04de95.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/tables/d100e51b4ff5d8ae35bbe93a2c09cbbf48e6f30bb96b472ded62b0c68a04de95.jpg)

![d64e786c3b70ff9c0857e52b93f98279fe31fa29d0cffedc9e051fe079b21d61.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/tables/d64e786c3b70ff9c0857e52b93f98279fe31fa29d0cffedc9e051fe079b21d61.jpg)

![f4d5e4f67584566e2a3b56f5cc4943220879131329c25cc5f8718e3f5bf50675.jpg](../icml_results/5777_Decomposition-Based%20Modular%20Conformal%20Prediction%20for%20Two-Stage%20Modeling/tables/f4d5e4f67584566e2a3b56f5cc4943220879131329c25cc5f8718e3f5bf50675.jpg)

<a id="paper-5729"></a>

## BALLAST: Bayesian Active Learning with Look-ahead Amendment for Sea-drifter Trajectories under Spatio-Temporal Vector Fields


### Images

![046b57d954ae7c9ba57942a33f5ace830eca3f6ffa7e88f8276a1ed8f7baedde.jpg](../icml_results/5778_BALLAST_%20Bayesian%20Active%20Learning%20with%20Look-ahead%20Amendment%20for%20Sea-drifter%20Trajectories%20under%20Spati/images/046b57d954ae7c9ba57942a33f5ace830eca3f6ffa7e88f8276a1ed8f7baedde.jpg)

![0bdf957bc46dfb2fe17190fc17aad445b3286df25a32a48ee46970cec435a760.jpg](../icml_results/5778_BALLAST_%20Bayesian%20Active%20Learning%20with%20Look-ahead%20Amendment%20for%20Sea-drifter%20Trajectories%20under%20Spati/images/0bdf957bc46dfb2fe17190fc17aad445b3286df25a32a48ee46970cec435a760.jpg)

![2d47fa2cf538000772d63a7e20b4cfd9a0f9a5098b93ee88a70e44445d2e0db4.jpg](../icml_results/5778_BALLAST_%20Bayesian%20Active%20Learning%20with%20Look-ahead%20Amendment%20for%20Sea-drifter%20Trajectories%20under%20Spati/images/2d47fa2cf538000772d63a7e20b4cfd9a0f9a5098b93ee88a70e44445d2e0db4.jpg)

![36cb8eb06ceeb010975f2a980e7d5354155da57c26c8ed20a7be94a04f80e4e3.jpg](../icml_results/5778_BALLAST_%20Bayesian%20Active%20Learning%20with%20Look-ahead%20Amendment%20for%20Sea-drifter%20Trajectories%20under%20Spati/images/36cb8eb06ceeb010975f2a980e7d5354155da57c26c8ed20a7be94a04f80e4e3.jpg)

![6b5ebb91b676385a0e09f99e0a650ac0f9b0b92c69f53d9468377c757594ba85.jpg](../icml_results/5778_BALLAST_%20Bayesian%20Active%20Learning%20with%20Look-ahead%20Amendment%20for%20Sea-drifter%20Trajectories%20under%20Spati/images/6b5ebb91b676385a0e09f99e0a650ac0f9b0b92c69f53d9468377c757594ba85.jpg)

![7502acfebbb4f5ed734eb8780bbae4c5bd683b4186464e3deee33c9509ae96da.jpg](../icml_results/5778_BALLAST_%20Bayesian%20Active%20Learning%20with%20Look-ahead%20Amendment%20for%20Sea-drifter%20Trajectories%20under%20Spati/images/7502acfebbb4f5ed734eb8780bbae4c5bd683b4186464e3deee33c9509ae96da.jpg)

![b6e9b6f12b45c55c0781d2e76df0127c443075cb342995373cf29e4e5492cf2d.jpg](../icml_results/5778_BALLAST_%20Bayesian%20Active%20Learning%20with%20Look-ahead%20Amendment%20for%20Sea-drifter%20Trajectories%20under%20Spati/images/b6e9b6f12b45c55c0781d2e76df0127c443075cb342995373cf29e4e5492cf2d.jpg)

![d7e85a881060e83b0356c88b086bcf7216a4619007726455f21841d49cb07cdf.jpg](../icml_results/5778_BALLAST_%20Bayesian%20Active%20Learning%20with%20Look-ahead%20Amendment%20for%20Sea-drifter%20Trajectories%20under%20Spati/images/d7e85a881060e83b0356c88b086bcf7216a4619007726455f21841d49cb07cdf.jpg)

![e75d6e9c4e2788073d1aec90663d3e4c6a001a2a866d151f14d14b493133c610.jpg](../icml_results/5778_BALLAST_%20Bayesian%20Active%20Learning%20with%20Look-ahead%20Amendment%20for%20Sea-drifter%20Trajectories%20under%20Spati/images/e75d6e9c4e2788073d1aec90663d3e4c6a001a2a866d151f14d14b493133c610.jpg)

![ef49fd811d61b6b7703c65cac366624abf551758baa8e7c6699bd687eb8b0dde.jpg](../icml_results/5778_BALLAST_%20Bayesian%20Active%20Learning%20with%20Look-ahead%20Amendment%20for%20Sea-drifter%20Trajectories%20under%20Spati/images/ef49fd811d61b6b7703c65cac366624abf551758baa8e7c6699bd687eb8b0dde.jpg)

### Tables

![2045fd7cccb928a49c94e5d4fb17b645d4cffb98fff1ca78c162618a131a2b26.jpg](../icml_results/5778_BALLAST_%20Bayesian%20Active%20Learning%20with%20Look-ahead%20Amendment%20for%20Sea-drifter%20Trajectories%20under%20Spati/tables/2045fd7cccb928a49c94e5d4fb17b645d4cffb98fff1ca78c162618a131a2b26.jpg)

![8d86889779f5190e61898ec1af7fa19173c65abc4e4d2d1763920825ffbd12b1.jpg](../icml_results/5778_BALLAST_%20Bayesian%20Active%20Learning%20with%20Look-ahead%20Amendment%20for%20Sea-drifter%20Trajectories%20under%20Spati/tables/8d86889779f5190e61898ec1af7fa19173c65abc4e4d2d1763920825ffbd12b1.jpg)

![b800bfb936088a3dc1c43fdcd45f2a8e58057a8d4f4e6db3df20f555ed6eb8b8.jpg](../icml_results/5778_BALLAST_%20Bayesian%20Active%20Learning%20with%20Look-ahead%20Amendment%20for%20Sea-drifter%20Trajectories%20under%20Spati/tables/b800bfb936088a3dc1c43fdcd45f2a8e58057a8d4f4e6db3df20f555ed6eb8b8.jpg)

![bf1dbdd1c11c125fc6d86aa1751173fdaf5ffa0052393a157a6bf895219efdbe.jpg](../icml_results/5778_BALLAST_%20Bayesian%20Active%20Learning%20with%20Look-ahead%20Amendment%20for%20Sea-drifter%20Trajectories%20under%20Spati/tables/bf1dbdd1c11c125fc6d86aa1751173fdaf5ffa0052393a157a6bf895219efdbe.jpg)

<a id="paper-5730"></a>

## SENDAI: A Hierarchical Sparse-measurement, EfficieNt Data AssImilation Framework


### Images

![01479dfc1a9918991430cf8c0b2fd73ad4249b83daac76358e98bbb9d9dd6c19.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/01479dfc1a9918991430cf8c0b2fd73ad4249b83daac76358e98bbb9d9dd6c19.jpg)

![03e0fde4da29c8d755c2ab26d7785faa33a034703794ff0112e85cdad33bb703.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/03e0fde4da29c8d755c2ab26d7785faa33a034703794ff0112e85cdad33bb703.jpg)

![0449e51d78b2791f0bacbf492a6c76af86c79532af1f3496b3765a28f80450b2.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/0449e51d78b2791f0bacbf492a6c76af86c79532af1f3496b3765a28f80450b2.jpg)

![0bca03e69db7cfc65c20166b946d006ee4864dccaead4cbc74a23df66071bda1.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/0bca03e69db7cfc65c20166b946d006ee4864dccaead4cbc74a23df66071bda1.jpg)

![19ba799782aea4bf0f66e82f18f9ac1c295276cf687a471aa3d1e2336bbf3eed.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/19ba799782aea4bf0f66e82f18f9ac1c295276cf687a471aa3d1e2336bbf3eed.jpg)

![19d98430563a89c5afd80a35f3e5828c6405d64793b4af8b1ccaf6282d8770df.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/19d98430563a89c5afd80a35f3e5828c6405d64793b4af8b1ccaf6282d8770df.jpg)

![216490a859e33910eb37679bad1df904bdcfa146d3baee7163a08374c2e86458.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/216490a859e33910eb37679bad1df904bdcfa146d3baee7163a08374c2e86458.jpg)

![2e34f5f81ace7dcc7c1d57799ca29a14d119ca053acdce3acc461b9cd7f83cad.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/2e34f5f81ace7dcc7c1d57799ca29a14d119ca053acdce3acc461b9cd7f83cad.jpg)

![345ff1eeaec4b303f10f1f5ded8b76603ccf2148fbad3d98cc7766a23e2fc706.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/345ff1eeaec4b303f10f1f5ded8b76603ccf2148fbad3d98cc7766a23e2fc706.jpg)

![37f567b9a2c479ebff8eec38b62dcc477db00b4b33f1b48dbd4aa5b8164dfe3a.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/37f567b9a2c479ebff8eec38b62dcc477db00b4b33f1b48dbd4aa5b8164dfe3a.jpg)

![3cafd15935451e5434b643734fbcd09668df90fdfcd809a234e547e6f3ba2ecf.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/3cafd15935451e5434b643734fbcd09668df90fdfcd809a234e547e6f3ba2ecf.jpg)

![3d0fe6ea19a18e32cc8ce5536c7de401c69bbe1b0b0aaf1e72859d27fa783728.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/3d0fe6ea19a18e32cc8ce5536c7de401c69bbe1b0b0aaf1e72859d27fa783728.jpg)

![43b24e458aefca817c52ff088b84d3cf0a4e618a95cfbcc98ec1204246f450e8.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/43b24e458aefca817c52ff088b84d3cf0a4e618a95cfbcc98ec1204246f450e8.jpg)

![4ec0295524a0478eff8d252bce444248406265d95657fcb07bff38982b5f93b2.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/4ec0295524a0478eff8d252bce444248406265d95657fcb07bff38982b5f93b2.jpg)

![56ec685285ddc6238a7565139c70356e39b18f130e55239d07b70202e7750f27.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/56ec685285ddc6238a7565139c70356e39b18f130e55239d07b70202e7750f27.jpg)

![5ca038e89f66500073d030384231fd01540b860e0baff0c0e246d5601bacc48c.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/5ca038e89f66500073d030384231fd01540b860e0baff0c0e246d5601bacc48c.jpg)

![643134a6af4f05d84244304ebc92e890c2490ae4fb28655197c670c9fc63b202.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/643134a6af4f05d84244304ebc92e890c2490ae4fb28655197c670c9fc63b202.jpg)

![6997a40082e245d9494f52fd8476de47dbd85d3c94fa571f154b871de84a2cb7.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/6997a40082e245d9494f52fd8476de47dbd85d3c94fa571f154b871de84a2cb7.jpg)

![6d647f8b1ed46aae504bd04ab976ca050adc214bae3bb5fcec536929ac874577.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/6d647f8b1ed46aae504bd04ab976ca050adc214bae3bb5fcec536929ac874577.jpg)

![78f2214a8d5638d993f8deb6c8fd12203435e8fac7db9686722e90528bde4a79.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/78f2214a8d5638d993f8deb6c8fd12203435e8fac7db9686722e90528bde4a79.jpg)

![7d2ada3112923af621c50c1e15d204eee5929b2e487f942715cf8f6a2ddd5aac.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/7d2ada3112923af621c50c1e15d204eee5929b2e487f942715cf8f6a2ddd5aac.jpg)

![92517a0e85ce4ace6fc1cb92ecb2737fa65044a4fff5b0a90e9138f5334d6393.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/92517a0e85ce4ace6fc1cb92ecb2737fa65044a4fff5b0a90e9138f5334d6393.jpg)

![a1ef914e550fc11f236a204737b3221c57977fa6204b14ade5bc8dce1b887839.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/a1ef914e550fc11f236a204737b3221c57977fa6204b14ade5bc8dce1b887839.jpg)

![a2c2340da6cd13fdbdfb897d79df9e3e1d34c5f725320761862ca39221eb6aec.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/a2c2340da6cd13fdbdfb897d79df9e3e1d34c5f725320761862ca39221eb6aec.jpg)

![a3d68e40ec544c5f8e8e0f6e58a03ed468a70b6f01aeb3a0c82045e8e195a3a7.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/a3d68e40ec544c5f8e8e0f6e58a03ed468a70b6f01aeb3a0c82045e8e195a3a7.jpg)

![a86074b66459834af822a111644a449854f8d30643d1923678ab6ec9e81c2726.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/a86074b66459834af822a111644a449854f8d30643d1923678ab6ec9e81c2726.jpg)

![a955724d3817d20762c8ae9dd45f8bc7b1e2cf23062f5aa3316d86443ef628cb.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/a955724d3817d20762c8ae9dd45f8bc7b1e2cf23062f5aa3316d86443ef628cb.jpg)

![afae9f741e0f5e07f6fb03356a7a63ed1f6e648556e361974b686dbaa11bc2ad.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/afae9f741e0f5e07f6fb03356a7a63ed1f6e648556e361974b686dbaa11bc2ad.jpg)

![bb6f590a74c901f0fa4057f93832a7b3a9f1aa9d022fe7570e71f2613cb59f1c.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/bb6f590a74c901f0fa4057f93832a7b3a9f1aa9d022fe7570e71f2613cb59f1c.jpg)

![c316a30dfba0c584e342f892844ddd1ee12d94776d8fe7a4d24b820ac8c88e31.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/c316a30dfba0c584e342f892844ddd1ee12d94776d8fe7a4d24b820ac8c88e31.jpg)

![cd5103aa3c5b1eebfa9565ccc061048dfb6d51343cdfafbcf72645a575f1dd0c.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/cd5103aa3c5b1eebfa9565ccc061048dfb6d51343cdfafbcf72645a575f1dd0c.jpg)

![cf72e802584fe2c31d04523f7db3d3849efe0f5a61a0e6d9ec7641a8d6cdc08d.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/cf72e802584fe2c31d04523f7db3d3849efe0f5a61a0e6d9ec7641a8d6cdc08d.jpg)

![d66d8d4944195fe0f05a2f4d064fd9001c029108a519dfe6cc823a8bdfc2f91f.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/d66d8d4944195fe0f05a2f4d064fd9001c029108a519dfe6cc823a8bdfc2f91f.jpg)

![d764eba000e4b0aff5638eee50bab914c9d30148260ea9429c55a1e2e36e9c20.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/d764eba000e4b0aff5638eee50bab914c9d30148260ea9429c55a1e2e36e9c20.jpg)

![e170a0559bdc09a3a728a5ec56ae41f94a4653e5a4cced0a78b82ed4d5093601.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/e170a0559bdc09a3a728a5ec56ae41f94a4653e5a4cced0a78b82ed4d5093601.jpg)

![ed7ec7804eb97ede3a89e2041506530f852248dd3b23d20e752652f2b33aacbe.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/images/ed7ec7804eb97ede3a89e2041506530f852248dd3b23d20e752652f2b33aacbe.jpg)

### Tables

![0de3f153917085820bfafeb97c261d89d90a0076f38323e5fba481cff79a6b2e.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/tables/0de3f153917085820bfafeb97c261d89d90a0076f38323e5fba481cff79a6b2e.jpg)

![17552cfdfad5848a501d3013efafbbc41a5b39b60461d6b38adbccb77d8aee23.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/tables/17552cfdfad5848a501d3013efafbbc41a5b39b60461d6b38adbccb77d8aee23.jpg)

![439e3fe64ded5aeb66030594b26409c45ea75af4cf1aabe7828baba8eb69b1d1.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/tables/439e3fe64ded5aeb66030594b26409c45ea75af4cf1aabe7828baba8eb69b1d1.jpg)

![8400c5e5f83fc05a053ba95061c8a5a6cbda8deebf70d267d79479bd53423c0b.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/tables/8400c5e5f83fc05a053ba95061c8a5a6cbda8deebf70d267d79479bd53423c0b.jpg)

![842c0bd62f82bac525f17308073a770074c7d5ab46989b78d51016c971bad9a6.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/tables/842c0bd62f82bac525f17308073a770074c7d5ab46989b78d51016c971bad9a6.jpg)

![90bdb7ba1e604b3d4d23ed4ff64f9df33dac130e8504f28e2eb8d34765ef8fbe.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/tables/90bdb7ba1e604b3d4d23ed4ff64f9df33dac130e8504f28e2eb8d34765ef8fbe.jpg)

![975dde7ffa535db0cb36bb62491d08b790a48554a8af21cd6de4e7a1d4f8e95c.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/tables/975dde7ffa535db0cb36bb62491d08b790a48554a8af21cd6de4e7a1d4f8e95c.jpg)

![995ee7ec77613093fffc7e0f87ea605b46659f688ee7e652ae342a6ea35622a5.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/tables/995ee7ec77613093fffc7e0f87ea605b46659f688ee7e652ae342a6ea35622a5.jpg)

![a45627f6d999163a8594e114403a86d71b2f14750145d940fcf66312e85749cb.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/tables/a45627f6d999163a8594e114403a86d71b2f14750145d940fcf66312e85749cb.jpg)

![c09984f9f1b94a8a54a80c19e42915babbb6d5947ecadf2ba72c161f0810e80f.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/tables/c09984f9f1b94a8a54a80c19e42915babbb6d5947ecadf2ba72c161f0810e80f.jpg)

![d7b99809ba48046cca68db44bc3c76d89b53a45317f7df4fce0197f7cc1ca6e3.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/tables/d7b99809ba48046cca68db44bc3c76d89b53a45317f7df4fce0197f7cc1ca6e3.jpg)

![ec5665679fc6780f78e5881aabf206541e228435fdbb624dc289ac00d81a8c1a.jpg](../icml_results/5779_SENDAI_%20A%20Hierarchical%20Sparse-measurement%2C%20EfficieNt%20Data%20AssImilation%20Framework/tables/ec5665679fc6780f78e5881aabf206541e228435fdbb624dc289ac00d81a8c1a.jpg)

<a id="paper-5731"></a>

## Weights to Code: Extracting Interpretable Algorithms from the Discrete Transformer


### Images

![3f040cbe32f3f6f1bb9cfeb26cf14d2b3737326ffb9362f642ea92a471b0946a.jpg](../icml_results/5780_Weights%20to%20Code_%20Extracting%20Interpretable%20Algorithms%20from%20the%20Discrete%20Transformer/images/3f040cbe32f3f6f1bb9cfeb26cf14d2b3737326ffb9362f642ea92a471b0946a.jpg)

![8a921635e8be7a00350319a8fbeb7250baa2f03fe7211ffae8a5af23973ac34f.jpg](../icml_results/5780_Weights%20to%20Code_%20Extracting%20Interpretable%20Algorithms%20from%20the%20Discrete%20Transformer/images/8a921635e8be7a00350319a8fbeb7250baa2f03fe7211ffae8a5af23973ac34f.jpg)

![c24cb0649917e8c62c3402aeacc34891af0cc6ea1786b7dd9504a966f1ed8a29.jpg](../icml_results/5780_Weights%20to%20Code_%20Extracting%20Interpretable%20Algorithms%20from%20the%20Discrete%20Transformer/images/c24cb0649917e8c62c3402aeacc34891af0cc6ea1786b7dd9504a966f1ed8a29.jpg)

![d83467a69d4d8631260eadd65402e720b4942b9e8f1ae597690bca53106c4dbd.jpg](../icml_results/5780_Weights%20to%20Code_%20Extracting%20Interpretable%20Algorithms%20from%20the%20Discrete%20Transformer/images/d83467a69d4d8631260eadd65402e720b4942b9e8f1ae597690bca53106c4dbd.jpg)

### Tables

![137dae99f4d5ab84366a5b9db2e98f554aaca47e67f52dedb05974a42b69eb13.jpg](../icml_results/5780_Weights%20to%20Code_%20Extracting%20Interpretable%20Algorithms%20from%20the%20Discrete%20Transformer/tables/137dae99f4d5ab84366a5b9db2e98f554aaca47e67f52dedb05974a42b69eb13.jpg)

![5c6fe0f26173627194b96e81e0669649f44f0b47ac1b31efe4270f92c563c3a8.jpg](../icml_results/5780_Weights%20to%20Code_%20Extracting%20Interpretable%20Algorithms%20from%20the%20Discrete%20Transformer/tables/5c6fe0f26173627194b96e81e0669649f44f0b47ac1b31efe4270f92c563c3a8.jpg)

![61b5351403f62358d4d3c4f3e06c650a8d5828293fadd4aa7e9078dccb5edb43.jpg](../icml_results/5780_Weights%20to%20Code_%20Extracting%20Interpretable%20Algorithms%20from%20the%20Discrete%20Transformer/tables/61b5351403f62358d4d3c4f3e06c650a8d5828293fadd4aa7e9078dccb5edb43.jpg)

![6e4f194e566f8448953015b249c63324e634f2c155801df21cb7b557976e626b.jpg](../icml_results/5780_Weights%20to%20Code_%20Extracting%20Interpretable%20Algorithms%20from%20the%20Discrete%20Transformer/tables/6e4f194e566f8448953015b249c63324e634f2c155801df21cb7b557976e626b.jpg)

![7ae674a6b276925185ccbb6b882461a12f43d7d48499a7cf859eec58b092ccf1.jpg](../icml_results/5780_Weights%20to%20Code_%20Extracting%20Interpretable%20Algorithms%20from%20the%20Discrete%20Transformer/tables/7ae674a6b276925185ccbb6b882461a12f43d7d48499a7cf859eec58b092ccf1.jpg)

![c338168af6378950cd69a2325ce082c1f24819c717dacb7954742193a6f6b99a.jpg](../icml_results/5780_Weights%20to%20Code_%20Extracting%20Interpretable%20Algorithms%20from%20the%20Discrete%20Transformer/tables/c338168af6378950cd69a2325ce082c1f24819c717dacb7954742193a6f6b99a.jpg)

<a id="paper-5732"></a>

## Local MAP Sampling for Diffusion Models


### Images

![015e414cd701b74f8f9a51d5e6c4c91b61018552b67ef6b2d2e9a23f05d18357.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/images/015e414cd701b74f8f9a51d5e6c4c91b61018552b67ef6b2d2e9a23f05d18357.jpg)

![07522334e460965f90a1f17972c30c65b554ecad132214df37d9a94d700140ac.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/images/07522334e460965f90a1f17972c30c65b554ecad132214df37d9a94d700140ac.jpg)

![1a065617086882b06889025a56d56b7c89d61d1eab220930f949fc45b0ec1807.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/images/1a065617086882b06889025a56d56b7c89d61d1eab220930f949fc45b0ec1807.jpg)

![2befb3b111d8c9946dae42069a849ec4be746b2c75c92109488df4c3da0615d4.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/images/2befb3b111d8c9946dae42069a849ec4be746b2c75c92109488df4c3da0615d4.jpg)

![36a55f625191fd92ef17cd30cc30bd15a00fa2fb8104c15cf74b958e8aee9690.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/images/36a55f625191fd92ef17cd30cc30bd15a00fa2fb8104c15cf74b958e8aee9690.jpg)

![6331d02a1d0f3930649ed90d4a8e3756a98c9472a859ab6dfce1a905e9c84ff1.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/images/6331d02a1d0f3930649ed90d4a8e3756a98c9472a859ab6dfce1a905e9c84ff1.jpg)

![636095b45f332f001ff993aae61d9d9c03d0337d85df3143ecf444091173f853.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/images/636095b45f332f001ff993aae61d9d9c03d0337d85df3143ecf444091173f853.jpg)

![7e576ae498c1f0aa7ac9b3bc619542ae586d37bfa284416765a11249f00740d9.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/images/7e576ae498c1f0aa7ac9b3bc619542ae586d37bfa284416765a11249f00740d9.jpg)

![a856170031506d79c7692b43965461b5a2897cca3a4ac1a8534c8b8941ea2b26.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/images/a856170031506d79c7692b43965461b5a2897cca3a4ac1a8534c8b8941ea2b26.jpg)

![bbc5d639e9a7cf3e20dd86ac97f8a90f1b10f6880df9aa45b57102aa8f4843d8.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/images/bbc5d639e9a7cf3e20dd86ac97f8a90f1b10f6880df9aa45b57102aa8f4843d8.jpg)

![c6ef5b8d76ed39f9f102b199e19dbaf6e858ddf60845621fc8f3ef0f322e78ca.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/images/c6ef5b8d76ed39f9f102b199e19dbaf6e858ddf60845621fc8f3ef0f322e78ca.jpg)

![f06750c5566b4316a69211381c8235d9d14d57881165fd6d6d99f8f6c25eae1d.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/images/f06750c5566b4316a69211381c8235d9d14d57881165fd6d6d99f8f6c25eae1d.jpg)

![f4048609a267ac90682ececbd8d809417e7f3f55b04a505cdaee2d6043f19888.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/images/f4048609a267ac90682ececbd8d809417e7f3f55b04a505cdaee2d6043f19888.jpg)

![fcb03b3c59345726a207e060f170832d9b228ed349f45acdd3ef7ac05b46e2e0.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/images/fcb03b3c59345726a207e060f170832d9b228ed349f45acdd3ef7ac05b46e2e0.jpg)

### Tables

![03d20018eee3888a1fb979520e4d5b92701b326f1273a88b993082ac76a263e4.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/tables/03d20018eee3888a1fb979520e4d5b92701b326f1273a88b993082ac76a263e4.jpg)

![1ef89fab62f69599e9dc4723ce6ba44d0bbc06421f5f12daa70e07be3a89acaf.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/tables/1ef89fab62f69599e9dc4723ce6ba44d0bbc06421f5f12daa70e07be3a89acaf.jpg)

![251c6cf687d75a74e03d703a73729c76c92bfb1fa379f8a7bf9713da249c9b8e.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/tables/251c6cf687d75a74e03d703a73729c76c92bfb1fa379f8a7bf9713da249c9b8e.jpg)

![30b73f68fbad1d8a2430863a6a7bd8827939dfc1feef294b90e7f38e3a1097bf.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/tables/30b73f68fbad1d8a2430863a6a7bd8827939dfc1feef294b90e7f38e3a1097bf.jpg)

![521b7575f3f6c6cec1a4cbfceb495dfa732192c364a46ed7305aa5ed3f4f0203.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/tables/521b7575f3f6c6cec1a4cbfceb495dfa732192c364a46ed7305aa5ed3f4f0203.jpg)

![6648e4d3238270ce8e4a5b7609cbdffcb19d9b8dc7e3b3e7b10b8149ecdb82fc.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/tables/6648e4d3238270ce8e4a5b7609cbdffcb19d9b8dc7e3b3e7b10b8149ecdb82fc.jpg)

![73c7103f412425b1e3f290c30c9a21054b57eaaae35f0e55964558927845cdab.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/tables/73c7103f412425b1e3f290c30c9a21054b57eaaae35f0e55964558927845cdab.jpg)

![b97dfe41d94e3e1076e2b5c81ff75774f5ec2fccb5125ecb9e56d34aa3a52727.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/tables/b97dfe41d94e3e1076e2b5c81ff75774f5ec2fccb5125ecb9e56d34aa3a52727.jpg)

![d7bd0f1d23f3680ad54ccfed4d5435be4feaea413aade27bf5e86558ab7664f5.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/tables/d7bd0f1d23f3680ad54ccfed4d5435be4feaea413aade27bf5e86558ab7664f5.jpg)

![db85dae6ba05ffe6511bb57d0d0d130700665930bf09e8a149a1b433982a17b1.jpg](../icml_results/5781_Local%20MAP%20Sampling%20for%20Diffusion%20Models/tables/db85dae6ba05ffe6511bb57d0d0d130700665930bf09e8a149a1b433982a17b1.jpg)

<a id="paper-5733"></a>

## Prediction-Powered Risk Monitoring of Deployed Models for Detecting Harmful Distribution Shifts


### Images

![1a495da46cb31f78fd1fbb79362f672ae670a1d092f6963a45d486432b421be5.jpg](../icml_results/5782_Prediction-Powered%20Risk%20Monitoring%20of%20Deployed%20Models%20for%20Detecting%20Harmful%20Distribution%20Shifts/images/1a495da46cb31f78fd1fbb79362f672ae670a1d092f6963a45d486432b421be5.jpg)

![32c60a188467fd4b5da7c21b661ccb4d00905636f7564cd17f82c6ab3791d0c1.jpg](../icml_results/5782_Prediction-Powered%20Risk%20Monitoring%20of%20Deployed%20Models%20for%20Detecting%20Harmful%20Distribution%20Shifts/images/32c60a188467fd4b5da7c21b661ccb4d00905636f7564cd17f82c6ab3791d0c1.jpg)

![4df9d757511c45cd5f6f900bafa4bc9b8b27156679cd5cd84f4bd4459461c9bd.jpg](../icml_results/5782_Prediction-Powered%20Risk%20Monitoring%20of%20Deployed%20Models%20for%20Detecting%20Harmful%20Distribution%20Shifts/images/4df9d757511c45cd5f6f900bafa4bc9b8b27156679cd5cd84f4bd4459461c9bd.jpg)

![4f1bdc2400ab0aaddec7c44fd6da50469686d346c42e9e18d4fd60371782336d.jpg](../icml_results/5782_Prediction-Powered%20Risk%20Monitoring%20of%20Deployed%20Models%20for%20Detecting%20Harmful%20Distribution%20Shifts/images/4f1bdc2400ab0aaddec7c44fd6da50469686d346c42e9e18d4fd60371782336d.jpg)

![5891441680db408fd4e44ba89b028e9dd99ecf336d83dd3f9ea5ae6a23e66834.jpg](../icml_results/5782_Prediction-Powered%20Risk%20Monitoring%20of%20Deployed%20Models%20for%20Detecting%20Harmful%20Distribution%20Shifts/images/5891441680db408fd4e44ba89b028e9dd99ecf336d83dd3f9ea5ae6a23e66834.jpg)

![6fc90ca327723811cd5df7b4c92f70ecea375c577fa1abd0ca4818f0c73943ad.jpg](../icml_results/5782_Prediction-Powered%20Risk%20Monitoring%20of%20Deployed%20Models%20for%20Detecting%20Harmful%20Distribution%20Shifts/images/6fc90ca327723811cd5df7b4c92f70ecea375c577fa1abd0ca4818f0c73943ad.jpg)

![76b1bf025fbbe41b8ffb6c2fa3fc77f646d6d9afb51d079b25a253995931d603.jpg](../icml_results/5782_Prediction-Powered%20Risk%20Monitoring%20of%20Deployed%20Models%20for%20Detecting%20Harmful%20Distribution%20Shifts/images/76b1bf025fbbe41b8ffb6c2fa3fc77f646d6d9afb51d079b25a253995931d603.jpg)

![81864b478e179dcf2eb5feef84f50fae4176b71563859628fa3923c59ee004da.jpg](../icml_results/5782_Prediction-Powered%20Risk%20Monitoring%20of%20Deployed%20Models%20for%20Detecting%20Harmful%20Distribution%20Shifts/images/81864b478e179dcf2eb5feef84f50fae4176b71563859628fa3923c59ee004da.jpg)

![8558b372afac57c910b86e49786f2889b14b4f8ded397d4f5f1cccaee922457b.jpg](../icml_results/5782_Prediction-Powered%20Risk%20Monitoring%20of%20Deployed%20Models%20for%20Detecting%20Harmful%20Distribution%20Shifts/images/8558b372afac57c910b86e49786f2889b14b4f8ded397d4f5f1cccaee922457b.jpg)

![901b595b7b2aa2eb9d119204b9da27b6bba1b91c4d0e06599cc834ee6d175be7.jpg](../icml_results/5782_Prediction-Powered%20Risk%20Monitoring%20of%20Deployed%20Models%20for%20Detecting%20Harmful%20Distribution%20Shifts/images/901b595b7b2aa2eb9d119204b9da27b6bba1b91c4d0e06599cc834ee6d175be7.jpg)

![94f03b6476f33f04f8f333182134536dd2aa66d41f5de2b21c465d72b9a78068.jpg](../icml_results/5782_Prediction-Powered%20Risk%20Monitoring%20of%20Deployed%20Models%20for%20Detecting%20Harmful%20Distribution%20Shifts/images/94f03b6476f33f04f8f333182134536dd2aa66d41f5de2b21c465d72b9a78068.jpg)

![a237f530a8bd5e7080b119ae933e8f62f6eccadfa2b5317cb4a7fd0b63cd7dbd.jpg](../icml_results/5782_Prediction-Powered%20Risk%20Monitoring%20of%20Deployed%20Models%20for%20Detecting%20Harmful%20Distribution%20Shifts/images/a237f530a8bd5e7080b119ae933e8f62f6eccadfa2b5317cb4a7fd0b63cd7dbd.jpg)

![be51700ba5f3add2a82e05aa86a9c05724a6e149b66a26bc8bc36f407bad3ca0.jpg](../icml_results/5782_Prediction-Powered%20Risk%20Monitoring%20of%20Deployed%20Models%20for%20Detecting%20Harmful%20Distribution%20Shifts/images/be51700ba5f3add2a82e05aa86a9c05724a6e149b66a26bc8bc36f407bad3ca0.jpg)

![d371e6c98a7a980c7c6c48bace0cb6b0e05cac7d0fdfe0df6638fb001abdc1cb.jpg](../icml_results/5782_Prediction-Powered%20Risk%20Monitoring%20of%20Deployed%20Models%20for%20Detecting%20Harmful%20Distribution%20Shifts/images/d371e6c98a7a980c7c6c48bace0cb6b0e05cac7d0fdfe0df6638fb001abdc1cb.jpg)

### Tables

![7a33f8f9856f8d4a26f11488d1d55c4b599f76f6965f2b573c4a9b0e2befcfcc.jpg](../icml_results/5782_Prediction-Powered%20Risk%20Monitoring%20of%20Deployed%20Models%20for%20Detecting%20Harmful%20Distribution%20Shifts/tables/7a33f8f9856f8d4a26f11488d1d55c4b599f76f6965f2b573c4a9b0e2befcfcc.jpg)

<a id="paper-5734"></a>

## RiboSphere: Learning Unified and Efficient Representations of RNA Structures


### Images

![1d44561cefb3406a5705d36029c3818e3ade3b6268cee0cb7eea4e0c9ec80fa7.jpg](../icml_results/5783_RiboSphere_%20Learning%20Unified%20and%20Efficient%20Representations%20of%20RNA%20Structures/images/1d44561cefb3406a5705d36029c3818e3ade3b6268cee0cb7eea4e0c9ec80fa7.jpg)

![3bcbde8fb431d19c545f90d206fe9f29289557a7bb6f3527ec51c07186ac0673.jpg](../icml_results/5783_RiboSphere_%20Learning%20Unified%20and%20Efficient%20Representations%20of%20RNA%20Structures/images/3bcbde8fb431d19c545f90d206fe9f29289557a7bb6f3527ec51c07186ac0673.jpg)

![87304aa1a82c08889c353e935649b8bc65c4b354d74dc4e8437baba4081107fd.jpg](../icml_results/5783_RiboSphere_%20Learning%20Unified%20and%20Efficient%20Representations%20of%20RNA%20Structures/images/87304aa1a82c08889c353e935649b8bc65c4b354d74dc4e8437baba4081107fd.jpg)

![baeee1d072edcdb95b96d0b5b957a0f39ee1ed94b6ef7be355087f3e7fe63619.jpg](../icml_results/5783_RiboSphere_%20Learning%20Unified%20and%20Efficient%20Representations%20of%20RNA%20Structures/images/baeee1d072edcdb95b96d0b5b957a0f39ee1ed94b6ef7be355087f3e7fe63619.jpg)

![e9868df3263f104dcc4ca1c9c17367ed8d7f0978bed8f063697dc0d5e99c8b03.jpg](../icml_results/5783_RiboSphere_%20Learning%20Unified%20and%20Efficient%20Representations%20of%20RNA%20Structures/images/e9868df3263f104dcc4ca1c9c17367ed8d7f0978bed8f063697dc0d5e99c8b03.jpg)

### Tables

![1f4d801728004188b60e9b176988d40d977b8533f0c9a5c0585cedacae924644.jpg](../icml_results/5783_RiboSphere_%20Learning%20Unified%20and%20Efficient%20Representations%20of%20RNA%20Structures/tables/1f4d801728004188b60e9b176988d40d977b8533f0c9a5c0585cedacae924644.jpg)

![3c4f63155c03db640a0ab06a69fb3899fcaa2afa44e62d36adfee31570d0d8bb.jpg](../icml_results/5783_RiboSphere_%20Learning%20Unified%20and%20Efficient%20Representations%20of%20RNA%20Structures/tables/3c4f63155c03db640a0ab06a69fb3899fcaa2afa44e62d36adfee31570d0d8bb.jpg)

![4faad5d7113f7369355b8b218aadbeb1a20008cbbb1e8e637f2c19bd283c7ac6.jpg](../icml_results/5783_RiboSphere_%20Learning%20Unified%20and%20Efficient%20Representations%20of%20RNA%20Structures/tables/4faad5d7113f7369355b8b218aadbeb1a20008cbbb1e8e637f2c19bd283c7ac6.jpg)

![56dedcc02545971e4224cdc8a29e93afa341bc78c3df23b56f0b3ca4d19f676b.jpg](../icml_results/5783_RiboSphere_%20Learning%20Unified%20and%20Efficient%20Representations%20of%20RNA%20Structures/tables/56dedcc02545971e4224cdc8a29e93afa341bc78c3df23b56f0b3ca4d19f676b.jpg)

![dc1caff9d5c1e82fafb6ab53fc6aac32209cf4c8136969502d70749a290c4382.jpg](../icml_results/5783_RiboSphere_%20Learning%20Unified%20and%20Efficient%20Representations%20of%20RNA%20Structures/tables/dc1caff9d5c1e82fafb6ab53fc6aac32209cf4c8136969502d70749a290c4382.jpg)

<a id="paper-5735"></a>

## Transformers Efficiently Perform In-Context Logistic Regression via Normalized Gradient Descent


### Images

![082221150146c73c722834bad1526bb3fe1affedd0392e53518da18a2dbbe328.jpg](../icml_results/5784_Transformers%20Efficiently%20Perform%20In-Context%20Logistic%20Regression%20via%20Normalized%20Gradient%20Descent/images/082221150146c73c722834bad1526bb3fe1affedd0392e53518da18a2dbbe328.jpg)

![30bf202c339c2aee39c9e99c6949b9a06dd790fdc7189b3b7ab7c0c4a06d77d2.jpg](../icml_results/5784_Transformers%20Efficiently%20Perform%20In-Context%20Logistic%20Regression%20via%20Normalized%20Gradient%20Descent/images/30bf202c339c2aee39c9e99c6949b9a06dd790fdc7189b3b7ab7c0c4a06d77d2.jpg)

![4673839685353b83c93a35dabc2a7e9f9e8b0facfbd596f767e1ef0c7366fc97.jpg](../icml_results/5784_Transformers%20Efficiently%20Perform%20In-Context%20Logistic%20Regression%20via%20Normalized%20Gradient%20Descent/images/4673839685353b83c93a35dabc2a7e9f9e8b0facfbd596f767e1ef0c7366fc97.jpg)

![59f46479900675fd9d0c09c45180ace4ba22344db432bf8f2bb97daadd3eb915.jpg](../icml_results/5784_Transformers%20Efficiently%20Perform%20In-Context%20Logistic%20Regression%20via%20Normalized%20Gradient%20Descent/images/59f46479900675fd9d0c09c45180ace4ba22344db432bf8f2bb97daadd3eb915.jpg)

![8d9eced7d02d2fb8faa2c4745f9e62a04b3619142ffb8d343614b67d606e78a6.jpg](../icml_results/5784_Transformers%20Efficiently%20Perform%20In-Context%20Logistic%20Regression%20via%20Normalized%20Gradient%20Descent/images/8d9eced7d02d2fb8faa2c4745f9e62a04b3619142ffb8d343614b67d606e78a6.jpg)

![a8a6cd7b5fe12ae2054424e34d7dace00d924a05f25bccf583f956cc6ccc6444.jpg](../icml_results/5784_Transformers%20Efficiently%20Perform%20In-Context%20Logistic%20Regression%20via%20Normalized%20Gradient%20Descent/images/a8a6cd7b5fe12ae2054424e34d7dace00d924a05f25bccf583f956cc6ccc6444.jpg)

![c2930cbf6f197e1f2dd7920e62d44f0c874344a0044062d65326facad1fbf686.jpg](../icml_results/5784_Transformers%20Efficiently%20Perform%20In-Context%20Logistic%20Regression%20via%20Normalized%20Gradient%20Descent/images/c2930cbf6f197e1f2dd7920e62d44f0c874344a0044062d65326facad1fbf686.jpg)

![e6d8ad5eed5bbf8867741916807e0563df0b1b446695261527120bfdacebbf46.jpg](../icml_results/5784_Transformers%20Efficiently%20Perform%20In-Context%20Logistic%20Regression%20via%20Normalized%20Gradient%20Descent/images/e6d8ad5eed5bbf8867741916807e0563df0b1b446695261527120bfdacebbf46.jpg)

![e9eb1b984e9427dbc674ad7c41facfcb6f16cb4bcc9d640bd0eee5e43f91752c.jpg](../icml_results/5784_Transformers%20Efficiently%20Perform%20In-Context%20Logistic%20Regression%20via%20Normalized%20Gradient%20Descent/images/e9eb1b984e9427dbc674ad7c41facfcb6f16cb4bcc9d640bd0eee5e43f91752c.jpg)

![ef4a2d5aeaf145c3e6f1697ebc7791acc4fa6e1953e5886d497ca3c880445389.jpg](../icml_results/5784_Transformers%20Efficiently%20Perform%20In-Context%20Logistic%20Regression%20via%20Normalized%20Gradient%20Descent/images/ef4a2d5aeaf145c3e6f1697ebc7791acc4fa6e1953e5886d497ca3c880445389.jpg)

<a id="paper-5736"></a>

## FUSE: Quantifying Uncertainty in Vision-Language Models by Bayesian Fusing Epistemic and Aleatoric Uncertainty


### Images

![00b451856f98db0b551b191c82adc39ef7b87417d803232928e7bcb436c06084.jpg](../icml_results/5785_FUSE_%20Quantifying%20Uncertainty%20in%20Vision-Language%20Models%20by%20Bayesian%20Fusing%20Epistemic%20and%20Aleatoric%20U/images/00b451856f98db0b551b191c82adc39ef7b87417d803232928e7bcb436c06084.jpg)

![0fd2a614da3adb0308d91244cec9b555625caec1b58f2d95deca801c9d2c13e2.jpg](../icml_results/5785_FUSE_%20Quantifying%20Uncertainty%20in%20Vision-Language%20Models%20by%20Bayesian%20Fusing%20Epistemic%20and%20Aleatoric%20U/images/0fd2a614da3adb0308d91244cec9b555625caec1b58f2d95deca801c9d2c13e2.jpg)

![1cf6431ea896d60fef04113c59b938e1866c3d8004ccc3001da0d9e2b96c7491.jpg](../icml_results/5785_FUSE_%20Quantifying%20Uncertainty%20in%20Vision-Language%20Models%20by%20Bayesian%20Fusing%20Epistemic%20and%20Aleatoric%20U/images/1cf6431ea896d60fef04113c59b938e1866c3d8004ccc3001da0d9e2b96c7491.jpg)

![1d1a1c709448c72fc60a0613d8abc6dce21b15290ccc33f9c9e43c073c7db105.jpg](../icml_results/5785_FUSE_%20Quantifying%20Uncertainty%20in%20Vision-Language%20Models%20by%20Bayesian%20Fusing%20Epistemic%20and%20Aleatoric%20U/images/1d1a1c709448c72fc60a0613d8abc6dce21b15290ccc33f9c9e43c073c7db105.jpg)

![409d204fedfadb38770f5cd2bc9624cd14257af24061e98aa3a9c4d22a06b92f.jpg](../icml_results/5785_FUSE_%20Quantifying%20Uncertainty%20in%20Vision-Language%20Models%20by%20Bayesian%20Fusing%20Epistemic%20and%20Aleatoric%20U/images/409d204fedfadb38770f5cd2bc9624cd14257af24061e98aa3a9c4d22a06b92f.jpg)

![43b893562c2fa9c1581a8b7d3db34a8f6d91e87639ae7061116b19ee7fab1aa8.jpg](../icml_results/5785_FUSE_%20Quantifying%20Uncertainty%20in%20Vision-Language%20Models%20by%20Bayesian%20Fusing%20Epistemic%20and%20Aleatoric%20U/images/43b893562c2fa9c1581a8b7d3db34a8f6d91e87639ae7061116b19ee7fab1aa8.jpg)

![6bd0948d9440c12a69898073c3163ddbb5586fc00c31cf034bd64991b39c202c.jpg](../icml_results/5785_FUSE_%20Quantifying%20Uncertainty%20in%20Vision-Language%20Models%20by%20Bayesian%20Fusing%20Epistemic%20and%20Aleatoric%20U/images/6bd0948d9440c12a69898073c3163ddbb5586fc00c31cf034bd64991b39c202c.jpg)

![80c9d11312ac12ea024a5bfc22dd14e5b6dde67281045888a1e37ca04376f666.jpg](../icml_results/5785_FUSE_%20Quantifying%20Uncertainty%20in%20Vision-Language%20Models%20by%20Bayesian%20Fusing%20Epistemic%20and%20Aleatoric%20U/images/80c9d11312ac12ea024a5bfc22dd14e5b6dde67281045888a1e37ca04376f666.jpg)

![9200ce210894aa131f52a966f00cef23a5d1afab25a8be0415b4f4a8e113a0ce.jpg](../icml_results/5785_FUSE_%20Quantifying%20Uncertainty%20in%20Vision-Language%20Models%20by%20Bayesian%20Fusing%20Epistemic%20and%20Aleatoric%20U/images/9200ce210894aa131f52a966f00cef23a5d1afab25a8be0415b4f4a8e113a0ce.jpg)

![b1c1b87a36d9281b70d532085e516a0add0005a9642ddbd0d84139ac1222382a.jpg](../icml_results/5785_FUSE_%20Quantifying%20Uncertainty%20in%20Vision-Language%20Models%20by%20Bayesian%20Fusing%20Epistemic%20and%20Aleatoric%20U/images/b1c1b87a36d9281b70d532085e516a0add0005a9642ddbd0d84139ac1222382a.jpg)

![d0417c9dfef2dd124865f59a1116db8b8649aff9661bee92595c14c98bec34e8.jpg](../icml_results/5785_FUSE_%20Quantifying%20Uncertainty%20in%20Vision-Language%20Models%20by%20Bayesian%20Fusing%20Epistemic%20and%20Aleatoric%20U/images/d0417c9dfef2dd124865f59a1116db8b8649aff9661bee92595c14c98bec34e8.jpg)

![d7ff58aea5dde14d16f9d2c17cf9dac2773763066c04bb7bd67de5402423475c.jpg](../icml_results/5785_FUSE_%20Quantifying%20Uncertainty%20in%20Vision-Language%20Models%20by%20Bayesian%20Fusing%20Epistemic%20and%20Aleatoric%20U/images/d7ff58aea5dde14d16f9d2c17cf9dac2773763066c04bb7bd67de5402423475c.jpg)

![dac1aab35ab88f5067b62fb5821832b58ae3c879b508d1b12a3c884edecb4a20.jpg](../icml_results/5785_FUSE_%20Quantifying%20Uncertainty%20in%20Vision-Language%20Models%20by%20Bayesian%20Fusing%20Epistemic%20and%20Aleatoric%20U/images/dac1aab35ab88f5067b62fb5821832b58ae3c879b508d1b12a3c884edecb4a20.jpg)

### Tables

![5435fab54fa43a999dac997eb63277a73028caaf9ac1fb9535f428ff5c40c8f0.jpg](../icml_results/5785_FUSE_%20Quantifying%20Uncertainty%20in%20Vision-Language%20Models%20by%20Bayesian%20Fusing%20Epistemic%20and%20Aleatoric%20U/tables/5435fab54fa43a999dac997eb63277a73028caaf9ac1fb9535f428ff5c40c8f0.jpg)

![8dafd71bc9f16acf0fc069c3fc874fe78b26ed76a10ad13a0f9617f4d5e45edb.jpg](../icml_results/5785_FUSE_%20Quantifying%20Uncertainty%20in%20Vision-Language%20Models%20by%20Bayesian%20Fusing%20Epistemic%20and%20Aleatoric%20U/tables/8dafd71bc9f16acf0fc069c3fc874fe78b26ed76a10ad13a0f9617f4d5e45edb.jpg)

<a id="paper-5737"></a>

## Mitigating Translationese Bias in Multilingual LLM-as-a-Judge via Disentangled Information Bottleneck


### Images

![06a9fec3d2dae1871f3f10e2612f5b5ec269ec1f30443da3c62786307d2b8b2f.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/images/06a9fec3d2dae1871f3f10e2612f5b5ec269ec1f30443da3c62786307d2b8b2f.jpg)

![21166de38513d4eda19529e165d041b41b063bb00ab2b79c68108525b730f383.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/images/21166de38513d4eda19529e165d041b41b063bb00ab2b79c68108525b730f383.jpg)

![6a429ea07fa31c0bd9245b15731207940aa8004c89f3f7d4ba2ad2360c768d9b.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/images/6a429ea07fa31c0bd9245b15731207940aa8004c89f3f7d4ba2ad2360c768d9b.jpg)

![80070d37d2c5f20efb6b00528e9e922788243a2c546970c52c772649994c1eef.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/images/80070d37d2c5f20efb6b00528e9e922788243a2c546970c52c772649994c1eef.jpg)

![acf397ceddeef3d516766a4250edc61286d9c390e5350f381669d9754818f9ca.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/images/acf397ceddeef3d516766a4250edc61286d9c390e5350f381669d9754818f9ca.jpg)

![d99beca6eb6894d44cb4b3555223fc20db5e4b07bc128a7cbfb26a8829b66df1.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/images/d99beca6eb6894d44cb4b3555223fc20db5e4b07bc128a7cbfb26a8829b66df1.jpg)

![db2ecb82282fc83eb69706a75884093655a0e7be70b733778458824aa8c49c6b.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/images/db2ecb82282fc83eb69706a75884093655a0e7be70b733778458824aa8c49c6b.jpg)

![f7d6aa15668b1cbfc462aaf8ede66036b94c1c8b0fb6a26afa4ad896e4cb3f8b.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/images/f7d6aa15668b1cbfc462aaf8ede66036b94c1c8b0fb6a26afa4ad896e4cb3f8b.jpg)

### Tables

![0306e801d3a4c761f6021687d867eb8472ec586ef821fcb18d7a8412a3554201.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/0306e801d3a4c761f6021687d867eb8472ec586ef821fcb18d7a8412a3554201.jpg)

![14a1d318e40cde1d180e94898b51aa82c54f9a7c74b5992f0456db08a3147148.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/14a1d318e40cde1d180e94898b51aa82c54f9a7c74b5992f0456db08a3147148.jpg)

![1a8b484946ba9127431f3e9546066a586f6727b90e7a77dfec2c54ed572a33a5.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/1a8b484946ba9127431f3e9546066a586f6727b90e7a77dfec2c54ed572a33a5.jpg)

![3b804cb9ffc3f39bffb90f941a7db166a9f5a421c886f58e3f5333f769816890.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/3b804cb9ffc3f39bffb90f941a7db166a9f5a421c886f58e3f5333f769816890.jpg)

![473394e021f174c6838223f58d3527e90d72f891b5ff320763d20dd712a84bb8.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/473394e021f174c6838223f58d3527e90d72f891b5ff320763d20dd712a84bb8.jpg)

![4823f30d260ff014d4d3ba876cc023833180d07d34ac135dd0fb8f753d80b6d0.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/4823f30d260ff014d4d3ba876cc023833180d07d34ac135dd0fb8f753d80b6d0.jpg)

![4b0a23c9c082a103c9bb779bb8057d4185aab59bec96ba33d7b04fbde0261257.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/4b0a23c9c082a103c9bb779bb8057d4185aab59bec96ba33d7b04fbde0261257.jpg)

![58d12b075dd5040984e0e85dcbee81c73e54292754b596e7af3dac48e38ff10c.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/58d12b075dd5040984e0e85dcbee81c73e54292754b596e7af3dac48e38ff10c.jpg)

![5b12aaf86fe773e794aa1dc35af90bc1374ccec95248f8cbc67b456715ec69e4.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/5b12aaf86fe773e794aa1dc35af90bc1374ccec95248f8cbc67b456715ec69e4.jpg)

![66c3e06988cac03fc03c4e039ab7139680c1806848472bc488be143c262084c7.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/66c3e06988cac03fc03c4e039ab7139680c1806848472bc488be143c262084c7.jpg)

![84ef1613b03e4d12ae2118e47c9a040faa3935359b8bdb3c65d1eab4dd159b6a.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/84ef1613b03e4d12ae2118e47c9a040faa3935359b8bdb3c65d1eab4dd159b6a.jpg)

![864fe8e9170cfaa31cf63f679b73968c8ecfee1d5c2d66ec318605e769067923.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/864fe8e9170cfaa31cf63f679b73968c8ecfee1d5c2d66ec318605e769067923.jpg)

![8891de7c031a8ec0fa12a8bfda0da877c4ee3173ecc6b449465b64e0eea60503.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/8891de7c031a8ec0fa12a8bfda0da877c4ee3173ecc6b449465b64e0eea60503.jpg)

![8a2364d971caf6a22040117b117c895a4e7ad192f3ffcc73889eabd7ff6b3482.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/8a2364d971caf6a22040117b117c895a4e7ad192f3ffcc73889eabd7ff6b3482.jpg)

![8e3d42873ede56b1032ef83454cae864838a96eb73fa9c4d17e148232267b810.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/8e3d42873ede56b1032ef83454cae864838a96eb73fa9c4d17e148232267b810.jpg)

![936d2ba44b0e44f812c50c6b9b78d87152a4e39b1ce76a78312f501d5bd6be2d.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/936d2ba44b0e44f812c50c6b9b78d87152a4e39b1ce76a78312f501d5bd6be2d.jpg)

![986f02cc892ebf3db5cb1e398635e46f75539975c9bf8cec99a961f94ff362a3.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/986f02cc892ebf3db5cb1e398635e46f75539975c9bf8cec99a961f94ff362a3.jpg)

![a3d1ce81ffc1f6252592ae70ee3de304c135be4155e170fbd3fb99abb23d34c3.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/a3d1ce81ffc1f6252592ae70ee3de304c135be4155e170fbd3fb99abb23d34c3.jpg)

![c57202e141b14b7869ad01c441f0316448861b97ce8383212d832af4544f922e.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/c57202e141b14b7869ad01c441f0316448861b97ce8383212d832af4544f922e.jpg)

![d041a89b32c3642848534876f836202307f165a68b4d03aeee37518c12d9d65d.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/d041a89b32c3642848534876f836202307f165a68b4d03aeee37518c12d9d65d.jpg)

![e72a6b3b165b55b3756f5fd5f48120e03cd721d4542bf76096aa8fc806f1bdea.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/e72a6b3b165b55b3756f5fd5f48120e03cd721d4542bf76096aa8fc806f1bdea.jpg)

![e9f5ab098e006654ef83cfda74a72229fb384bb409c98a6bdaef959a0ecc4f0a.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/e9f5ab098e006654ef83cfda74a72229fb384bb409c98a6bdaef959a0ecc4f0a.jpg)

![eb6a97ec8716755dde1bb82297cd481c521524b8e430b321a1dc11ef8c969e2e.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/eb6a97ec8716755dde1bb82297cd481c521524b8e430b321a1dc11ef8c969e2e.jpg)

![f65d8ab6060c4e0a7c2f01ec57f1350204c5a993299f623e95dbf93a42b09a4a.jpg](../icml_results/5786_Mitigating%20Translationese%20Bias%20in%20Multilingual%20LLM-as-a-Judge%20via%20Disentangled%20Information%20Bottlenec/tables/f65d8ab6060c4e0a7c2f01ec57f1350204c5a993299f623e95dbf93a42b09a4a.jpg)

<a id="paper-5738"></a>

## Forgetting Whenever You Want: A Decentralized Continual Learning Framework with On-Demand Unlearning


### Images

![31ff2766a610560d96fdf9d211da0d4474c30f4b7a5db695296cfab306ead35e.jpg](../icml_results/5787_Forgetting%20Whenever%20You%20Want_%20A%20Decentralized%20Continual%20Learning%20Framework%20with%20On-Demand%20Unlearning/images/31ff2766a610560d96fdf9d211da0d4474c30f4b7a5db695296cfab306ead35e.jpg)

![98fd5744eb7891c79d656a0385c679cb8c053c7e7dbe4f56f5eac4f0445ea082.jpg](../icml_results/5787_Forgetting%20Whenever%20You%20Want_%20A%20Decentralized%20Continual%20Learning%20Framework%20with%20On-Demand%20Unlearning/images/98fd5744eb7891c79d656a0385c679cb8c053c7e7dbe4f56f5eac4f0445ea082.jpg)

![a1c47b14df85c07ccd996eecc77c24582ee078339c15379b94f33b52178a584e.jpg](../icml_results/5787_Forgetting%20Whenever%20You%20Want_%20A%20Decentralized%20Continual%20Learning%20Framework%20with%20On-Demand%20Unlearning/images/a1c47b14df85c07ccd996eecc77c24582ee078339c15379b94f33b52178a584e.jpg)

![d0875395c783ca93608cbc6d7d5376e70f9ed5889948ac6daec5942a52481eca.jpg](../icml_results/5787_Forgetting%20Whenever%20You%20Want_%20A%20Decentralized%20Continual%20Learning%20Framework%20with%20On-Demand%20Unlearning/images/d0875395c783ca93608cbc6d7d5376e70f9ed5889948ac6daec5942a52481eca.jpg)

### Tables

![09fa92eb2c4dfc58e6962cdf53a885eabdc57783cc15cb5fc2b5e5b1d751ff90.jpg](../icml_results/5787_Forgetting%20Whenever%20You%20Want_%20A%20Decentralized%20Continual%20Learning%20Framework%20with%20On-Demand%20Unlearning/tables/09fa92eb2c4dfc58e6962cdf53a885eabdc57783cc15cb5fc2b5e5b1d751ff90.jpg)

![6389dc598c74cfbe69f932de15c747a5e563a2a9a651299cdded0a39853975d1.jpg](../icml_results/5787_Forgetting%20Whenever%20You%20Want_%20A%20Decentralized%20Continual%20Learning%20Framework%20with%20On-Demand%20Unlearning/tables/6389dc598c74cfbe69f932de15c747a5e563a2a9a651299cdded0a39853975d1.jpg)

![6f3a60017920c316897cad69c65b9525f84de8b184824a2cf607a3686e7e93cc.jpg](../icml_results/5787_Forgetting%20Whenever%20You%20Want_%20A%20Decentralized%20Continual%20Learning%20Framework%20with%20On-Demand%20Unlearning/tables/6f3a60017920c316897cad69c65b9525f84de8b184824a2cf607a3686e7e93cc.jpg)

![76cc0904be47fb18b75a57339930654bf35dbb564cb35d006b401ad6339c41ff.jpg](../icml_results/5787_Forgetting%20Whenever%20You%20Want_%20A%20Decentralized%20Continual%20Learning%20Framework%20with%20On-Demand%20Unlearning/tables/76cc0904be47fb18b75a57339930654bf35dbb564cb35d006b401ad6339c41ff.jpg)

![a343b4716cc850cf3b73d2900f305978dd8d7c4bbc5e9441b5d1f7f9bf50ae97.jpg](../icml_results/5787_Forgetting%20Whenever%20You%20Want_%20A%20Decentralized%20Continual%20Learning%20Framework%20with%20On-Demand%20Unlearning/tables/a343b4716cc850cf3b73d2900f305978dd8d7c4bbc5e9441b5d1f7f9bf50ae97.jpg)

<a id="paper-5739"></a>

## Agent Learning via Early Experience


### Images

![05ee4e6f96da66c24942520c87097cd496a7d313e685cd28449d820956ba13bd.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/images/05ee4e6f96da66c24942520c87097cd496a7d313e685cd28449d820956ba13bd.jpg)

![50401d1e785bfddb9e3c1ecc8a27dcf54c140d19fafbff5a278747ae4847765a.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/images/50401d1e785bfddb9e3c1ecc8a27dcf54c140d19fafbff5a278747ae4847765a.jpg)

![5e9272ec88c72c60915c62d690d66b7f78380ec60c23a07bfb663f3c6a4e32f5.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/images/5e9272ec88c72c60915c62d690d66b7f78380ec60c23a07bfb663f3c6a4e32f5.jpg)

![a5a1ae90fa1ec0fa9d235891799e2532292b261be33798485e3dbcef6aadf7f7.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/images/a5a1ae90fa1ec0fa9d235891799e2532292b261be33798485e3dbcef6aadf7f7.jpg)

![eb57d8df919696b4aba6cb8e3800e5f8027614093146cdb3fd8cc333028fba81.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/images/eb57d8df919696b4aba6cb8e3800e5f8027614093146cdb3fd8cc333028fba81.jpg)

### Tables

![1c5dae64e33f161c191536ab2ff8bb06a4b6568c4f4455091c5b5b4b678b9d2d.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/tables/1c5dae64e33f161c191536ab2ff8bb06a4b6568c4f4455091c5b5b4b678b9d2d.jpg)

![40388667d555d450c156246e6355c37a3b0048247c7e8b4998de5e8551d17074.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/tables/40388667d555d450c156246e6355c37a3b0048247c7e8b4998de5e8551d17074.jpg)

![41a3db1a9d96a22199edf3730fe1ac6d045a8cd87c60b4fec8f5d25f25608f9f.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/tables/41a3db1a9d96a22199edf3730fe1ac6d045a8cd87c60b4fec8f5d25f25608f9f.jpg)

![46145cbe4f139c150d2bfcbfb223a067ce1e0161b1b3ffafd812018a7738a550.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/tables/46145cbe4f139c150d2bfcbfb223a067ce1e0161b1b3ffafd812018a7738a550.jpg)

![596020ab2318e72fb806d6356a81757b22f94f57bed5fa92edd7132f0de464e4.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/tables/596020ab2318e72fb806d6356a81757b22f94f57bed5fa92edd7132f0de464e4.jpg)

![5de8ce2ca4de7e08ca90c3d20ff62ec27ddb9f987a78d2bc3a5216eb646afa90.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/tables/5de8ce2ca4de7e08ca90c3d20ff62ec27ddb9f987a78d2bc3a5216eb646afa90.jpg)

![760af0024ba21110cb02186d97d21d98dae25a5d40c7768336921ee46382ed6c.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/tables/760af0024ba21110cb02186d97d21d98dae25a5d40c7768336921ee46382ed6c.jpg)

![7d9b0a9747ab17970e1c8928bc62583140816d6ffd8be5fb507b0ab9a3b5e52f.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/tables/7d9b0a9747ab17970e1c8928bc62583140816d6ffd8be5fb507b0ab9a3b5e52f.jpg)

![7e441c5b6867a3beed273b75af33296254174c418ac094c1a41d05ec0680b096.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/tables/7e441c5b6867a3beed273b75af33296254174c418ac094c1a41d05ec0680b096.jpg)

![8097776f9254b3371f7b3716a8643a5f7c03bd729c56ceaf6d6341bdabfd1b78.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/tables/8097776f9254b3371f7b3716a8643a5f7c03bd729c56ceaf6d6341bdabfd1b78.jpg)

![87b511fc378f1be20a86624f05085314c600c427f7d8d7f5ed40742d78653d8e.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/tables/87b511fc378f1be20a86624f05085314c600c427f7d8d7f5ed40742d78653d8e.jpg)

![929d527e5fcafe6b6983ab81abe31ac7c7cdc6b5c02eb5add0e95daeaaf1a234.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/tables/929d527e5fcafe6b6983ab81abe31ac7c7cdc6b5c02eb5add0e95daeaaf1a234.jpg)

![b7ba2990fc229085c22066b3dde489e18536a23208be9675a17c46740a0aaa00.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/tables/b7ba2990fc229085c22066b3dde489e18536a23208be9675a17c46740a0aaa00.jpg)

![d1794b753a47f91d6163003814581c5cac1ad100bb3ec1582a735e60fe5d4f4c.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/tables/d1794b753a47f91d6163003814581c5cac1ad100bb3ec1582a735e60fe5d4f4c.jpg)

![db8ee2682be7728eb58119727dfd3f5d3569f58907e2c5c022048fc4c4eb7056.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/tables/db8ee2682be7728eb58119727dfd3f5d3569f58907e2c5c022048fc4c4eb7056.jpg)

![e229e4c866533b23a99d33ce5eb700ec390bf80d492f4cdd018c284548d5c4dc.jpg](../icml_results/5788_Agent%20Learning%20via%20Early%20Experience/tables/e229e4c866533b23a99d33ce5eb700ec390bf80d492f4cdd018c284548d5c4dc.jpg)

<a id="paper-5740"></a>

## Internalizing Safety Understanding in Large Reasoning Models via Verification


### Images

![478f7bb8068b0f502ff17fb084b651f0d889652d37eacb44306fe97ec0a43781.jpg](../icml_results/5789_Internalizing%20Safety%20Understanding%20in%20Large%20Reasoning%20Models%20via%20Verification/images/478f7bb8068b0f502ff17fb084b651f0d889652d37eacb44306fe97ec0a43781.jpg)

![57a3783073b71624449d39ddad5ff9c40a02339e295769dbbb773a37de2d6829.jpg](../icml_results/5789_Internalizing%20Safety%20Understanding%20in%20Large%20Reasoning%20Models%20via%20Verification/images/57a3783073b71624449d39ddad5ff9c40a02339e295769dbbb773a37de2d6829.jpg)

![6b2bbf69d9e45328e0b26f46f108ff104d57ef3416c2c083ad8c6b00027f40ad.jpg](../icml_results/5789_Internalizing%20Safety%20Understanding%20in%20Large%20Reasoning%20Models%20via%20Verification/images/6b2bbf69d9e45328e0b26f46f108ff104d57ef3416c2c083ad8c6b00027f40ad.jpg)

![97fd50d8fba016c595a76ed50bab00ef26ac35a13e54bdafc2ee44034990591a.jpg](../icml_results/5789_Internalizing%20Safety%20Understanding%20in%20Large%20Reasoning%20Models%20via%20Verification/images/97fd50d8fba016c595a76ed50bab00ef26ac35a13e54bdafc2ee44034990591a.jpg)

![9cba265850e1cd97dc1df30239c4c639085eb504c6f5044f4943cbf8b3140063.jpg](../icml_results/5789_Internalizing%20Safety%20Understanding%20in%20Large%20Reasoning%20Models%20via%20Verification/images/9cba265850e1cd97dc1df30239c4c639085eb504c6f5044f4943cbf8b3140063.jpg)

![a37b6c8701e3407b80784c7720348bc955d3a5c45c5c433dd4cb59d268207bc5.jpg](../icml_results/5789_Internalizing%20Safety%20Understanding%20in%20Large%20Reasoning%20Models%20via%20Verification/images/a37b6c8701e3407b80784c7720348bc955d3a5c45c5c433dd4cb59d268207bc5.jpg)

![bcdff2953fc8355d55cfc4f977c6bbdda66966e818ad17b07d23a5c2bf9e5312.jpg](../icml_results/5789_Internalizing%20Safety%20Understanding%20in%20Large%20Reasoning%20Models%20via%20Verification/images/bcdff2953fc8355d55cfc4f977c6bbdda66966e818ad17b07d23a5c2bf9e5312.jpg)

### Tables

![04f9e881924c222c07c7026939615c02b9d85d803a72ccafb4fb5c4fa9bf383e.jpg](../icml_results/5789_Internalizing%20Safety%20Understanding%20in%20Large%20Reasoning%20Models%20via%20Verification/tables/04f9e881924c222c07c7026939615c02b9d85d803a72ccafb4fb5c4fa9bf383e.jpg)

![155d64d3cd3d14be9b3b5564756c386755e25edb4a87c703f4303a13105d558f.jpg](../icml_results/5789_Internalizing%20Safety%20Understanding%20in%20Large%20Reasoning%20Models%20via%20Verification/tables/155d64d3cd3d14be9b3b5564756c386755e25edb4a87c703f4303a13105d558f.jpg)

![2583be51c962a62e70d86f98c7d02f4b66bbfc00ae74ad53afaaa1aaec2aed14.jpg](../icml_results/5789_Internalizing%20Safety%20Understanding%20in%20Large%20Reasoning%20Models%20via%20Verification/tables/2583be51c962a62e70d86f98c7d02f4b66bbfc00ae74ad53afaaa1aaec2aed14.jpg)

![4b66a2f2d8d3840cedcba10f1343470ddb49086c971d8083a934bd9069986133.jpg](../icml_results/5789_Internalizing%20Safety%20Understanding%20in%20Large%20Reasoning%20Models%20via%20Verification/tables/4b66a2f2d8d3840cedcba10f1343470ddb49086c971d8083a934bd9069986133.jpg)

![585f21f98b52096a5d876121a497bc4d0c3f1f14a5c93bf05d761d790daacb1f.jpg](../icml_results/5789_Internalizing%20Safety%20Understanding%20in%20Large%20Reasoning%20Models%20via%20Verification/tables/585f21f98b52096a5d876121a497bc4d0c3f1f14a5c93bf05d761d790daacb1f.jpg)

![b31d7105e238513d6939998811d8c4a8d7e34737f62cc08006b27002e50c63f8.jpg](../icml_results/5789_Internalizing%20Safety%20Understanding%20in%20Large%20Reasoning%20Models%20via%20Verification/tables/b31d7105e238513d6939998811d8c4a8d7e34737f62cc08006b27002e50c63f8.jpg)

![bcb8fe8a48336320917cc2ea2feacc3e0f56fad1d9a86d6bdd82f73988ad0730.jpg](../icml_results/5789_Internalizing%20Safety%20Understanding%20in%20Large%20Reasoning%20Models%20via%20Verification/tables/bcb8fe8a48336320917cc2ea2feacc3e0f56fad1d9a86d6bdd82f73988ad0730.jpg)

![e075c6ef33d599277d96b53bcaa1e637ea68e889acbab7bec54af38b83143f99.jpg](../icml_results/5789_Internalizing%20Safety%20Understanding%20in%20Large%20Reasoning%20Models%20via%20Verification/tables/e075c6ef33d599277d96b53bcaa1e637ea68e889acbab7bec54af38b83143f99.jpg)

<a id="paper-5741"></a>

## ReVSI: Rebuilding Visual Spatial Intelligence Evaluation for Accurate Assessment of VLM 3D Reasoning


### Images

![036ca79a28ed22180df4895912724fa18b327c0ff9b22728303bbf6350390142.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/images/036ca79a28ed22180df4895912724fa18b327c0ff9b22728303bbf6350390142.jpg)

![26df9bfecf2dae6e71940d841dc181419ac329a5a481398b136e4c0eaa2f486d.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/images/26df9bfecf2dae6e71940d841dc181419ac329a5a481398b136e4c0eaa2f486d.jpg)

![33aaaa9c2e8c5a05f41af0182ba2d29b6f5a57984fdb66e9473f454321f36ae6.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/images/33aaaa9c2e8c5a05f41af0182ba2d29b6f5a57984fdb66e9473f454321f36ae6.jpg)

![36c322bb4dc491a1bdf8b2924c190b2defd8e6195e5c9c7f0ea3631a69485317.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/images/36c322bb4dc491a1bdf8b2924c190b2defd8e6195e5c9c7f0ea3631a69485317.jpg)

![3cc5695f2090e6c63ff607346b333b48f6df4bc9295bf5c6edaf5575c1d95a63.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/images/3cc5695f2090e6c63ff607346b333b48f6df4bc9295bf5c6edaf5575c1d95a63.jpg)

![518d5776541b61110ff5f6ce120e0af5abade136716cd68ad6351f4a05afb9ef.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/images/518d5776541b61110ff5f6ce120e0af5abade136716cd68ad6351f4a05afb9ef.jpg)

![593fce43d4736e9c7299a12924fcf10b7404b27b538632a64bc6c6657536a803.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/images/593fce43d4736e9c7299a12924fcf10b7404b27b538632a64bc6c6657536a803.jpg)

![666afb7d68fa5e837a7bb5c63142973c3613e417ee41cb014e2d0d199bc2819d.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/images/666afb7d68fa5e837a7bb5c63142973c3613e417ee41cb014e2d0d199bc2819d.jpg)

![84bcd3d394b14216b12c2d50f8d6f872001c92832122d2750f79a28202d12bd8.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/images/84bcd3d394b14216b12c2d50f8d6f872001c92832122d2750f79a28202d12bd8.jpg)

![9920db2b9c21d5ea346954ea33abe06249108c00700479478fc7ba945fdc0efb.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/images/9920db2b9c21d5ea346954ea33abe06249108c00700479478fc7ba945fdc0efb.jpg)

![c2508b731c82f4402d51e6f61707f181e918cb8b63d5d9e588fc6df78dafbd2d.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/images/c2508b731c82f4402d51e6f61707f181e918cb8b63d5d9e588fc6df78dafbd2d.jpg)

![d1eea9b8c1a8a16102b90359a98e95fa55b1d612e863ea1c6c70f0c22bd16517.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/images/d1eea9b8c1a8a16102b90359a98e95fa55b1d612e863ea1c6c70f0c22bd16517.jpg)

![e197283f8c5c7571a617c0b401e2187b7dd26df0057554dc0b566809aad9634b.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/images/e197283f8c5c7571a617c0b401e2187b7dd26df0057554dc0b566809aad9634b.jpg)

![ecef0860d602abea57f983216a04a790c05667c5dd6267a94db43bc9840e624f.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/images/ecef0860d602abea57f983216a04a790c05667c5dd6267a94db43bc9840e624f.jpg)

![f85bf069e0ba23aa78810254716c56af019a3fb2685edb523bb718350ca8748f.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/images/f85bf069e0ba23aa78810254716c56af019a3fb2685edb523bb718350ca8748f.jpg)

![f88bf5f98c21f2422579732d3eaa119f33f5990ea90cb220ac1ca980b5fdfbf6.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/images/f88bf5f98c21f2422579732d3eaa119f33f5990ea90cb220ac1ca980b5fdfbf6.jpg)

![fbad85fab1f9cd1180abc009e71be607b39c90c4ad91bd155ba74642821d3f19.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/images/fbad85fab1f9cd1180abc009e71be607b39c90c4ad91bd155ba74642821d3f19.jpg)

### Tables

![003b1ee3e2d1850195ec3657f94dc31fdea7399119ef1163ff27d96222ac3fb7.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/tables/003b1ee3e2d1850195ec3657f94dc31fdea7399119ef1163ff27d96222ac3fb7.jpg)

![07e197f76aa4209a4495456bbf6377d6805721a8413be1649ac1025dfbaf175a.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/tables/07e197f76aa4209a4495456bbf6377d6805721a8413be1649ac1025dfbaf175a.jpg)

![07e3b4b7f15ffa1422e828ceb101e0c707f5612ade08e52336412471a938dee5.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/tables/07e3b4b7f15ffa1422e828ceb101e0c707f5612ade08e52336412471a938dee5.jpg)

![1859f9cf0cc4d41db452bb49452090cd4f43b0a9c1e70194f20476f924be139f.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/tables/1859f9cf0cc4d41db452bb49452090cd4f43b0a9c1e70194f20476f924be139f.jpg)

![3774a3378606996e25a65f407839b9db1ca5d3a13071afbbdf16bf0286067dfa.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/tables/3774a3378606996e25a65f407839b9db1ca5d3a13071afbbdf16bf0286067dfa.jpg)

![3e646ca72e63857ec9a2f96add9f3149eae1942b3913749f7331a74f667cfcc9.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/tables/3e646ca72e63857ec9a2f96add9f3149eae1942b3913749f7331a74f667cfcc9.jpg)

![71d704e8b4cfd7b098dcb1669951c05ae2534c766e664a2dcadd537c8dd30baa.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/tables/71d704e8b4cfd7b098dcb1669951c05ae2534c766e664a2dcadd537c8dd30baa.jpg)

![b8d7c357fe13292575248341d154e48aa9649ac4fe2bc2ab313f21234013d36c.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/tables/b8d7c357fe13292575248341d154e48aa9649ac4fe2bc2ab313f21234013d36c.jpg)

![bda5af580b742249d6b3a4d709aaaeddf93d52e90c6f5b9cec0a0bd97a5cc297.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/tables/bda5af580b742249d6b3a4d709aaaeddf93d52e90c6f5b9cec0a0bd97a5cc297.jpg)

![bfddf2b601e31cd2348a2f05cf82acb5306940d1b4aaec2ca03c7da640d65b0a.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/tables/bfddf2b601e31cd2348a2f05cf82acb5306940d1b4aaec2ca03c7da640d65b0a.jpg)

![c04f7a9173af9ed92fb69efb2ed531903911cdc2d8127fc11300675b9f5c3bae.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/tables/c04f7a9173af9ed92fb69efb2ed531903911cdc2d8127fc11300675b9f5c3bae.jpg)

![c84f83c1d35c2928eb8b8e9d5dc734d7cb652a4be61c5fa510f6e0b0d96f817d.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/tables/c84f83c1d35c2928eb8b8e9d5dc734d7cb652a4be61c5fa510f6e0b0d96f817d.jpg)

![eb57ae7b35c8b982deaf9e84b0abd3cb5aa9f985641205be398f4ce64d4b5efd.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/tables/eb57ae7b35c8b982deaf9e84b0abd3cb5aa9f985641205be398f4ce64d4b5efd.jpg)

![eddeeb7adb8d6dd39556d21f547ca7b1054df2e58532135ca68636bb69bf8094.jpg](../icml_results/5790_ReVSI_%20Rebuilding%20Visual%20Spatial%20Intelligence%20Evaluation%20for%20Accurate%20Assessment%20of%20VLM%203D%20Reasoning/tables/eddeeb7adb8d6dd39556d21f547ca7b1054df2e58532135ca68636bb69bf8094.jpg)

<a id="paper-5742"></a>

## L-Drive: Beyond a Single Mapping—Latent Context Drives Time Series Forecasting

### Images

![11e7746d558bd0f0eee40295a19764c92c1da2d73f98eb4061e32573bbb2a071.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/images/11e7746d558bd0f0eee40295a19764c92c1da2d73f98eb4061e32573bbb2a071.jpg)

![3138daf48abdd2260f8075d18bbf22628771b9e26e9e56a6ed310d5409e3f2a8.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/images/3138daf48abdd2260f8075d18bbf22628771b9e26e9e56a6ed310d5409e3f2a8.jpg)

![5dedc7ac9e2e90e6d32f82e6496f26d74d893c7d3607b6e3310173998b9956ea.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/images/5dedc7ac9e2e90e6d32f82e6496f26d74d893c7d3607b6e3310173998b9956ea.jpg)

![61a58adfa9d4922d2fe175e9b25096bbd43a25d46ee696c0b7eb6b65a65dc326.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/images/61a58adfa9d4922d2fe175e9b25096bbd43a25d46ee696c0b7eb6b65a65dc326.jpg)

![b08b6ba1c21c48b128cec3d886cc743ef11de9a947699a056c7a7558c99503dd.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/images/b08b6ba1c21c48b128cec3d886cc743ef11de9a947699a056c7a7558c99503dd.jpg)

### Tables

![0ea6cbcfbced72bfdcdd862fef7a878599a8c53816355545bd1d278876bb6968.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/0ea6cbcfbced72bfdcdd862fef7a878599a8c53816355545bd1d278876bb6968.jpg)

![165a28f0370f70f9a462fb00f8ac24796e50d30c8c908a7d09b2df8954b121a5.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/165a28f0370f70f9a462fb00f8ac24796e50d30c8c908a7d09b2df8954b121a5.jpg)

![30b364418f895fa46a32cbcb645d14b9656324b35e9cd44680ad04dbd4e6804f.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/30b364418f895fa46a32cbcb645d14b9656324b35e9cd44680ad04dbd4e6804f.jpg)

![41f72596d90387d2c7d7fb45ed4aad8bd5ae1ae1c3aeac25e2ed73463713a461.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/41f72596d90387d2c7d7fb45ed4aad8bd5ae1ae1c3aeac25e2ed73463713a461.jpg)

![45fdf22952a35cb02b411c2eccebbc9fb9188226087cd93298d350aa0cd58b8e.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/45fdf22952a35cb02b411c2eccebbc9fb9188226087cd93298d350aa0cd58b8e.jpg)

![547ce36e3afe12ce359965879e4b61354fa872da9cdecfbb3f876428b04a786b.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/547ce36e3afe12ce359965879e4b61354fa872da9cdecfbb3f876428b04a786b.jpg)

![77a675a8ca9f32eca1af8fbe926582e8ce3004e6429143ac58bfac6b571ada51.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/77a675a8ca9f32eca1af8fbe926582e8ce3004e6429143ac58bfac6b571ada51.jpg)

![85f8f869a0184d5367e15053b1f7dcb64dd80dc6807844b36eb8eb5a15a894c9.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/85f8f869a0184d5367e15053b1f7dcb64dd80dc6807844b36eb8eb5a15a894c9.jpg)

![ae35fc794a291e04de7e4495d5c01b8e4639708da5bc09fc4a89863b90e86469.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/ae35fc794a291e04de7e4495d5c01b8e4639708da5bc09fc4a89863b90e86469.jpg)

![c4123fc395f6091a0719780be8c27cec72f3f860acc59ee0439f07dea7e6f7f6.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/c4123fc395f6091a0719780be8c27cec72f3f860acc59ee0439f07dea7e6f7f6.jpg)

![c6957bc8d72105cf8b855777254da99f098f3d84b3ad6e1e472c9d8645489b95.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/c6957bc8d72105cf8b855777254da99f098f3d84b3ad6e1e472c9d8645489b95.jpg)

![cbbcdda193c9975226c820536b193abb488117d666e0f9e1977dd022970f2e92.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/cbbcdda193c9975226c820536b193abb488117d666e0f9e1977dd022970f2e92.jpg)

![d76c1b6a0ffa958c866113f2a91e83b1d1307d104027c5c05148824ca3f2017c.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/d76c1b6a0ffa958c866113f2a91e83b1d1307d104027c5c05148824ca3f2017c.jpg)

![e33b98c97a12e50cb42302ee27587c21b2f8cd5ee6a6ee180b4149d485eb7ff9.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/e33b98c97a12e50cb42302ee27587c21b2f8cd5ee6a6ee180b4149d485eb7ff9.jpg)

![e3c06791c5a792f9d2c07aa8bdccc4e0d64a9b1e611879ef030eac75b099629a.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/e3c06791c5a792f9d2c07aa8bdccc4e0d64a9b1e611879ef030eac75b099629a.jpg)

![ea505811799570acbd6d8be74c82b273757451de95f74fa665065ff52b32f52f.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/ea505811799570acbd6d8be74c82b273757451de95f74fa665065ff52b32f52f.jpg)

![f26b97c1c885d1f4939a0d04981a07894acf92c6e4eb3be825b1c90bd047946f.jpg](../icml_results/5791_L-Drive_%20Beyond%20a%20Single%20Mapping%E2%80%94Latent%20Context%20Drives%20Time%20Series%20Forecasting/tables/f26b97c1c885d1f4939a0d04981a07894acf92c6e4eb3be825b1c90bd047946f.jpg)
