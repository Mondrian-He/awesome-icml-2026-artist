# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 154 of 197

## 目录 (Table of Contents)

1. [Diffusion Models Are Statistically Optimal for Learning Low-Dimensional Multi-Modal Distributions](#paper-5050)
2. [Single-Rollout Hidden-State Dynamics for Training-Free RLVR Data Selection](#paper-5051)
3. [Improved Scaling Laws via Weak-to-Strong Generalization in Random Feature Ridge Regression](#paper-5052)
4. [From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models](#paper-5053)
5. [LSGQuant: Layer-Sensitivity Guided Quantization for One-Step Diffusion Real-World Video Super-Resolution](#paper-5054)
6. [GCIB: Graph Contrastive Information Bottleneck for Multi-Behavior Recommendation](#paper-5055)
7. [Efficient Diffusion Models via Time Step Optimization with Consistent Training and Inference Constraints](#paper-5056)
8. [Multi-scale Explainer for Graph Neural Networks](#paper-5057)
9. [HumanLM: Simulating Users with State Alignment Beats Response Imitation](#paper-5058)
10. [TimeSeed: Effective Time Series Forecasting with Sparse Endogenous Variables](#paper-5059)
11. [VLANeXt: Recipes for Building Strong VLA Models](#paper-5060)
12. [PyHealth 2.0: A Comprehensive Open-Source Toolkit for Accessible and Reproducible Clinical Deep Learning](#paper-5061)
13. [EasyBalance: Cross-Layer Load Balancing in Distributed MoE Inference](#paper-5062)
14. [OpenDeception: Learning Deception and Trust in Human–AI Interaction via Multi-Agent Simulation](#paper-5063)
15. [SGMD: Score Gradient Matching Distillation for Few-Step Video Diffusion Distillation](#paper-5064)
16. [Linear Ensembles Wash Away Watermarks: On the Fragility of Distributional Perturbations in LLMs](#paper-5065)
17. [GeoPT: Scaling Physics Simulation via Lifted Geometric Pre-Training](#paper-5066)
18. [Selecting Samples on Graphs: A Unified Dataset Pruning Framework for Lossless Training Acceleration](#paper-5067)
19. [An Empirical Study on the Resilience of Partial Merging to Model Clone Attacks](#paper-5068)
20. [Infinite-World: Scaling Interactive World Models to 1000-Frame Horizons via Pose-Free Hierarchical Memory](#paper-5069)
21. [Towards Seed-Robust Safety Alignment in Text-to-Image Models](#paper-5070)
22. [Hyperbolic neural population geometry benefits computation](#paper-5071)
23. [KODA: Contrastive Representation Comparison and Alignment for Vision-Language Foundation Models](#paper-5072)
24. [Trustworthy Federated Label Distribution Learning under Annotation Quality Disparity](#paper-5073)
25. [Adaptively Grouped Contextual Bandits for Heterogeneous Human-AI Decision Making with Conformal Prediction Sets](#paper-5074)
26. [Scalable Topology-Preserving Graph Coarsening: Concepts and Algorithms](#paper-5075)
27. [Frequency-Aware Perceptual Optimization for Low-Complexity Implicit Image Compression](#paper-5076)
28. [Learning Generalized Label Distributions](#paper-5077)
29. [Diversity-Preserved Distribution Matching Distillation for Fast Visual Synthesis](#paper-5078)
30. [Native Parallel Reasoner: Reasoning in Parallelism via Self-Distilled Reinforcement Learning](#paper-5079)
31. [Matrix-Free GPU Semidefinite Programming for Quantum Ordered Search at the k=6 Frontier](#paper-5080)
32. [Flow for Future: Geometric SE(3)-Equivariant Flow Matching for 3D Trajectory Prediction](#paper-5081)
33. [Can Agents Generalize to the Open World? Unveiling the Fragility of Static Training in Tool Use](#paper-5082)

---


<a id="paper-5050"></a>

## Diffusion Models Are Statistically Optimal for Learning Low-Dimensional Multi-Modal Distributions

### Images

![106512e83ae285d24fc27c5741e792c1697868a475ff58bb119cc0ea5184954c.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/images/106512e83ae285d24fc27c5741e792c1697868a475ff58bb119cc0ea5184954c.jpg)

![12706cae5f3e3d00ef178950f7f2b3aae2516582eb366b59151e4359c5cb487b.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/images/12706cae5f3e3d00ef178950f7f2b3aae2516582eb366b59151e4359c5cb487b.jpg)

![139fc01d9c128dc6e31da65bf08f7016a8123ec0ae338fb65370bede5e530cfd.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/images/139fc01d9c128dc6e31da65bf08f7016a8123ec0ae338fb65370bede5e530cfd.jpg)

![255aee27b1c265d914efc5b0940d1c70ad711bf606cd1b744ac321489df7526c.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/images/255aee27b1c265d914efc5b0940d1c70ad711bf606cd1b744ac321489df7526c.jpg)

![25f99864035bb975ba945e17435f3e64eeab450f7443aef27a5dcb3b220f4c14.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/images/25f99864035bb975ba945e17435f3e64eeab450f7443aef27a5dcb3b220f4c14.jpg)

![2891f7ae9a65ffc125cf26928e8d9f0e16d5415c26c17a48588ddfc0d78f713f.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/images/2891f7ae9a65ffc125cf26928e8d9f0e16d5415c26c17a48588ddfc0d78f713f.jpg)

![2e074079466cf95450a5ea298f4400fdbcbe9d96f2711a629cc0b3a34366a088.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/images/2e074079466cf95450a5ea298f4400fdbcbe9d96f2711a629cc0b3a34366a088.jpg)

![766c0ccbdb4c6fa91941f29ff5ae2649401b9393ac01969a89ec6564d97224cc.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/images/766c0ccbdb4c6fa91941f29ff5ae2649401b9393ac01969a89ec6564d97224cc.jpg)

![8371c74c660625b725894fc702c79608eb38d326f942819a011b0ae0724a87e6.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/images/8371c74c660625b725894fc702c79608eb38d326f942819a011b0ae0724a87e6.jpg)

![9f7db5d62f3b15e4794c1e560d745b192143fedabeb9212e4a98cb2b0c1321ca.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/images/9f7db5d62f3b15e4794c1e560d745b192143fedabeb9212e4a98cb2b0c1321ca.jpg)

![a60591aa991817918e204c0aaf54b5753724066df9735dcc78ac6f0c8ebc5fb4.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/images/a60591aa991817918e204c0aaf54b5753724066df9735dcc78ac6f0c8ebc5fb4.jpg)

![b285bf16df06bf842394713b9c40c00b6717787c1ccafffb29222aee57c92d7e.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/images/b285bf16df06bf842394713b9c40c00b6717787c1ccafffb29222aee57c92d7e.jpg)

![c46bdcd17d4d1866674684a32165d05e1a50ae3590b15ee1b829c734f20e5cab.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/images/c46bdcd17d4d1866674684a32165d05e1a50ae3590b15ee1b829c734f20e5cab.jpg)

![e04876836133adcea0f98a4639661a7c7443443d06b429516d420dac8f847b84.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/images/e04876836133adcea0f98a4639661a7c7443443d06b429516d420dac8f847b84.jpg)

![f66c3c2a904ac15a80b9e4d422f8bfee7c4d7659fc0037aeacaa82d7aad9a41c.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/images/f66c3c2a904ac15a80b9e4d422f8bfee7c4d7659fc0037aeacaa82d7aad9a41c.jpg)

### Tables

![0bebf266b4dc863d6c54ab205705d7291da2cb59dd05c4ea4f30bad826e3aad0.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/tables/0bebf266b4dc863d6c54ab205705d7291da2cb59dd05c4ea4f30bad826e3aad0.jpg)

![5d7872c6670b602e39f756b10bd2544f90946e4f231b33b7c75e7495d718db2b.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/tables/5d7872c6670b602e39f756b10bd2544f90946e4f231b33b7c75e7495d718db2b.jpg)

![5e24edafe9d08bb43e14e6bf2f719a41edddecc239487bdf7dd32b632dbaaba7.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/tables/5e24edafe9d08bb43e14e6bf2f719a41edddecc239487bdf7dd32b632dbaaba7.jpg)

![7352678fdfa9ef9a4a877bcd1f0dd003d4cc3e7387ef56d9df71299ec6b37ede.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/tables/7352678fdfa9ef9a4a877bcd1f0dd003d4cc3e7387ef56d9df71299ec6b37ede.jpg)

![d3b4f66cd60f55b8d92329a9a59910e270c34196674a96e19d4408976805ee81.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/tables/d3b4f66cd60f55b8d92329a9a59910e270c34196674a96e19d4408976805ee81.jpg)

![e1529939ffd65b328da55ac4e21d8934a09a0242a0038b7f222446973af1f1e2.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/tables/e1529939ffd65b328da55ac4e21d8934a09a0242a0038b7f222446973af1f1e2.jpg)

![f4214ed2d6e79727af4d5b8c6e5fe47ed77e35f05b7d08d2fa889e32c654c7c9.jpg](../icml_results/5097_Optimal%20Stopping%20in%20Latent%20Diffusion%20Models/tables/f4214ed2d6e79727af4d5b8c6e5fe47ed77e35f05b7d08d2fa889e32c654c7c9.jpg)

## Diffusion Models Are Statistically Optimal for Learning Low-Dimensional Multi-Modal Distributions


### Images

![7e2cadc264376a4ec3c74da94ace6662dec06fd079b746df8f944d8f04dde023.jpg](../icml_results/5098_Diffusion%20Models%20Are%20Statistically%20Optimal%20for%20Learning%20Low-Dimensional%20Multi-Modal%20Distributions/images/7e2cadc264376a4ec3c74da94ace6662dec06fd079b746df8f944d8f04dde023.jpg)

<a id="paper-5051"></a>

## Single-Rollout Hidden-State Dynamics for Training-Free RLVR Data Selection


### Images

![029052c5e1d05cd08bae57cac7d6e3df144bbc63e630108861dc9df21e9b0bf3.jpg](../icml_results/5099_Single-Rollout%20Hidden-State%20Dynamics%20for%20Training-Free%20RLVR%20Data%20Selection/images/029052c5e1d05cd08bae57cac7d6e3df144bbc63e630108861dc9df21e9b0bf3.jpg)

![293783666344348730cf0ee7210b73cf4bb1765a9bb9982e9000f3648068b205.jpg](../icml_results/5099_Single-Rollout%20Hidden-State%20Dynamics%20for%20Training-Free%20RLVR%20Data%20Selection/images/293783666344348730cf0ee7210b73cf4bb1765a9bb9982e9000f3648068b205.jpg)

### Tables

![29459a5128c3827ade84fc0369cc29cfb5341763180160c2786dc673e188d2f8.jpg](../icml_results/5099_Single-Rollout%20Hidden-State%20Dynamics%20for%20Training-Free%20RLVR%20Data%20Selection/tables/29459a5128c3827ade84fc0369cc29cfb5341763180160c2786dc673e188d2f8.jpg)

![847ba68133df74f726226cd9aa51b050f88954b22c259b736b756d77a339015e.jpg](../icml_results/5099_Single-Rollout%20Hidden-State%20Dynamics%20for%20Training-Free%20RLVR%20Data%20Selection/tables/847ba68133df74f726226cd9aa51b050f88954b22c259b736b756d77a339015e.jpg)

![9a36518203a04ac70d93406cbf9f70b4fb21ec33004fd426c85c109836e6c007.jpg](../icml_results/5099_Single-Rollout%20Hidden-State%20Dynamics%20for%20Training-Free%20RLVR%20Data%20Selection/tables/9a36518203a04ac70d93406cbf9f70b4fb21ec33004fd426c85c109836e6c007.jpg)

![cfa61df26c015b523d171a7c1b9626cfce918ffffb638da1d75567f657689020.jpg](../icml_results/5099_Single-Rollout%20Hidden-State%20Dynamics%20for%20Training-Free%20RLVR%20Data%20Selection/tables/cfa61df26c015b523d171a7c1b9626cfce918ffffb638da1d75567f657689020.jpg)

![d9e1c1b78835dc1e23b0442be8e179a64343137d551d1016c9984ef213f73852.jpg](../icml_results/5099_Single-Rollout%20Hidden-State%20Dynamics%20for%20Training-Free%20RLVR%20Data%20Selection/tables/d9e1c1b78835dc1e23b0442be8e179a64343137d551d1016c9984ef213f73852.jpg)

<a id="paper-5052"></a>

## Improved Scaling Laws via Weak-to-Strong Generalization in Random Feature Ridge Regression


### Images

![0f22877f882d998210654d498ab804013f5e742335d02b0bf2c4b756b596c0f8.jpg](../icml_results/5100_Improved%20Scaling%20Laws%20via%20Weak-to-Strong%20Generalization%20in%20Random%20Feature%20Ridge%20Regression/images/0f22877f882d998210654d498ab804013f5e742335d02b0bf2c4b756b596c0f8.jpg)

![9183bf6a937cacef243a8217078ff4523c55f7047c348471ceefafeb16da0747.jpg](../icml_results/5100_Improved%20Scaling%20Laws%20via%20Weak-to-Strong%20Generalization%20in%20Random%20Feature%20Ridge%20Regression/images/9183bf6a937cacef243a8217078ff4523c55f7047c348471ceefafeb16da0747.jpg)

<a id="paper-5053"></a>

## From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models


### Images

![2b084b976aa7ae19e5f79911236f3e79b892b83e12e34bc81592ad945e5e0cd2.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/images/2b084b976aa7ae19e5f79911236f3e79b892b83e12e34bc81592ad945e5e0cd2.jpg)

![383d743a252ca2d209864bb4e61041b5fdcb51d17351b687d876c1043586276e.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/images/383d743a252ca2d209864bb4e61041b5fdcb51d17351b687d876c1043586276e.jpg)

![42f8abd0706973dba92857aed232048c1dd48ba40095a2d1fe02968e6b451a0f.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/images/42f8abd0706973dba92857aed232048c1dd48ba40095a2d1fe02968e6b451a0f.jpg)

![5a0fec423e05ebeeaee17f3d446033fbfe62476f3e0c42e2296d32aca95817e8.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/images/5a0fec423e05ebeeaee17f3d446033fbfe62476f3e0c42e2296d32aca95817e8.jpg)

![833f43179bfb6aca856d34c7b448677464f0778e1e0bc564256c839544b733b1.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/images/833f43179bfb6aca856d34c7b448677464f0778e1e0bc564256c839544b733b1.jpg)

![84d0ccac2c863efb6dcd0c4453f2a273c06fe96157081c62ce07a74938325fee.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/images/84d0ccac2c863efb6dcd0c4453f2a273c06fe96157081c62ce07a74938325fee.jpg)

![91e855db050bb73fbe1fe450bc93b4a028aa8d9d67df017ce1295c132df364ea.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/images/91e855db050bb73fbe1fe450bc93b4a028aa8d9d67df017ce1295c132df364ea.jpg)

![a433553f8604b8538708e3722ef0e50b652ae8b983273146a2fc9e34de024f1a.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/images/a433553f8604b8538708e3722ef0e50b652ae8b983273146a2fc9e34de024f1a.jpg)

![b65e03e1c3f6fd0051a18908eecbf353ba9590690c5bc61d597c56b77f89b18f.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/images/b65e03e1c3f6fd0051a18908eecbf353ba9590690c5bc61d597c56b77f89b18f.jpg)

![f50b2fb9bddb24036c13a25534ac8633f01db62836b36a9ecef636f3697d7923.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/images/f50b2fb9bddb24036c13a25534ac8633f01db62836b36a9ecef636f3697d7923.jpg)

### Tables

![04f9bee721db171ef1cc3524e7f4fd4eb2059bffa063a86d0d1599738a0b190d.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/tables/04f9bee721db171ef1cc3524e7f4fd4eb2059bffa063a86d0d1599738a0b190d.jpg)

![0814767a19504979dae9f6acd9974c4e135cab92d44757cede3a9686aa5627d6.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/tables/0814767a19504979dae9f6acd9974c4e135cab92d44757cede3a9686aa5627d6.jpg)

![2fda83a3db8deace58b8310c7355e0f4ecf7cc0ffd3efdf55bc08f6ba90240d4.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/tables/2fda83a3db8deace58b8310c7355e0f4ecf7cc0ffd3efdf55bc08f6ba90240d4.jpg)

![90577439d1532268aca2feee74c246dafb9a6d01687f10c3e950fe276d143662.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/tables/90577439d1532268aca2feee74c246dafb9a6d01687f10c3e950fe276d143662.jpg)

![9d2ade8816c8ea9e630b293376767ca45545c96ec3fe69c1121ca1f4b8dadb6a.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/tables/9d2ade8816c8ea9e630b293376767ca45545c96ec3fe69c1121ca1f4b8dadb6a.jpg)

![b6a9ff3c0764bb2caadafd98b412e029e4bb302d16c151ff9eb326243c9bee62.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/tables/b6a9ff3c0764bb2caadafd98b412e029e4bb302d16c151ff9eb326243c9bee62.jpg)

![bff17a5bc8febddae788d75f5cffbde9d859df9f1edfc742f9a486c96cf2ee92.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/tables/bff17a5bc8febddae788d75f5cffbde9d859df9f1edfc742f9a486c96cf2ee92.jpg)

![cd73fa80740375c3d13f6e51f68c11228be9dfec15d7c88381559f90420a18e4.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/tables/cd73fa80740375c3d13f6e51f68c11228be9dfec15d7c88381559f90420a18e4.jpg)

![f634ab6a63c221300d4a26d086037671deff1a0a0d617444488b5781e5189259.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/tables/f634ab6a63c221300d4a26d086037671deff1a0a0d617444488b5781e5189259.jpg)

![f6a91b9ed803177a8d4a1c8033dbb6885b6b3bd8f65e078a20923e1ae6845ab4.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/tables/f6a91b9ed803177a8d4a1c8033dbb6885b6b3bd8f65e078a20923e1ae6845ab4.jpg)

![f71c17e7d1cb99c26b63eda574fdc97c4c52e03a1b4bca08bed34e9ddf39ad90.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/tables/f71c17e7d1cb99c26b63eda574fdc97c4c52e03a1b4bca08bed34e9ddf39ad90.jpg)

![fad1e0340d1dd6796b17f9f800112ee7bd70ef44277f748c3a0389ef96a0c447.jpg](../icml_results/5101_From%20Seeing%20to%20Thinking_%20Decoupling%20Perception%20and%20Reasoning%20Improves%20Post-Training%20of%20Vision-Langua/tables/fad1e0340d1dd6796b17f9f800112ee7bd70ef44277f748c3a0389ef96a0c447.jpg)

<a id="paper-5054"></a>

## LSGQuant: Layer-Sensitivity Guided Quantization for One-Step Diffusion Real-World Video Super-Resolution


### Images

![01627b30115ee8fa103cb9ad62fc7af615bc4897a0e9a15bde7ad0238faa8820.jpg](../icml_results/5102_LSGQuant_%20Layer-Sensitivity%20Guided%20Quantization%20for%20One-Step%20Diffusion%20Real-World%20Video%20Super-Resolu/images/01627b30115ee8fa103cb9ad62fc7af615bc4897a0e9a15bde7ad0238faa8820.jpg)

![134590feda08d0ec8c37857bbd6700982e6e6c83d9f42abdd978b5d54e76f51c.jpg](../icml_results/5102_LSGQuant_%20Layer-Sensitivity%20Guided%20Quantization%20for%20One-Step%20Diffusion%20Real-World%20Video%20Super-Resolu/images/134590feda08d0ec8c37857bbd6700982e6e6c83d9f42abdd978b5d54e76f51c.jpg)

![1359fd6ced3c1715ef13105fea5a5b8f0d2a6a8b698e4c34ce67b4f51923407e.jpg](../icml_results/5102_LSGQuant_%20Layer-Sensitivity%20Guided%20Quantization%20for%20One-Step%20Diffusion%20Real-World%20Video%20Super-Resolu/images/1359fd6ced3c1715ef13105fea5a5b8f0d2a6a8b698e4c34ce67b4f51923407e.jpg)

![6d977ed3ec37b573d26625b2d11dd775b49214615d9e05f054542773a4f5407e.jpg](../icml_results/5102_LSGQuant_%20Layer-Sensitivity%20Guided%20Quantization%20for%20One-Step%20Diffusion%20Real-World%20Video%20Super-Resolu/images/6d977ed3ec37b573d26625b2d11dd775b49214615d9e05f054542773a4f5407e.jpg)

![c40840c056b424b96e91606fcc723de9dfc427c9ced4c517ab420a0eed28f4aa.jpg](../icml_results/5102_LSGQuant_%20Layer-Sensitivity%20Guided%20Quantization%20for%20One-Step%20Diffusion%20Real-World%20Video%20Super-Resolu/images/c40840c056b424b96e91606fcc723de9dfc427c9ced4c517ab420a0eed28f4aa.jpg)

![ffebb5956a19388f495f144461232c899fc67dbdd4af9d6bacaa9f2ab761922e.jpg](../icml_results/5102_LSGQuant_%20Layer-Sensitivity%20Guided%20Quantization%20for%20One-Step%20Diffusion%20Real-World%20Video%20Super-Resolu/images/ffebb5956a19388f495f144461232c899fc67dbdd4af9d6bacaa9f2ab761922e.jpg)

### Tables

![2edbf6899a7c2ae0d566b02cb2171a42e38ce8277dea5d7847ac3bef200c9bd8.jpg](../icml_results/5102_LSGQuant_%20Layer-Sensitivity%20Guided%20Quantization%20for%20One-Step%20Diffusion%20Real-World%20Video%20Super-Resolu/tables/2edbf6899a7c2ae0d566b02cb2171a42e38ce8277dea5d7847ac3bef200c9bd8.jpg)

![34e565922ce582c0aea512b740f4351a71a61f72ed34d6b4f5a345cacc359818.jpg](../icml_results/5102_LSGQuant_%20Layer-Sensitivity%20Guided%20Quantization%20for%20One-Step%20Diffusion%20Real-World%20Video%20Super-Resolu/tables/34e565922ce582c0aea512b740f4351a71a61f72ed34d6b4f5a345cacc359818.jpg)

![3fc87f3c072060f19ba4191d73f9f0d4dcf14f5c545fbee8b695faa64e90afa3.jpg](../icml_results/5102_LSGQuant_%20Layer-Sensitivity%20Guided%20Quantization%20for%20One-Step%20Diffusion%20Real-World%20Video%20Super-Resolu/tables/3fc87f3c072060f19ba4191d73f9f0d4dcf14f5c545fbee8b695faa64e90afa3.jpg)

![65c3dd323dacb738cf753cf96ccc1e2eadec6aa4c0bbcde936aadb9e38ed4c84.jpg](../icml_results/5102_LSGQuant_%20Layer-Sensitivity%20Guided%20Quantization%20for%20One-Step%20Diffusion%20Real-World%20Video%20Super-Resolu/tables/65c3dd323dacb738cf753cf96ccc1e2eadec6aa4c0bbcde936aadb9e38ed4c84.jpg)

![8ce096f9cf6a8e6e1081d9478fe0c4048d4a423bc7731c9c35a4ba6638bc99d1.jpg](../icml_results/5102_LSGQuant_%20Layer-Sensitivity%20Guided%20Quantization%20for%20One-Step%20Diffusion%20Real-World%20Video%20Super-Resolu/tables/8ce096f9cf6a8e6e1081d9478fe0c4048d4a423bc7731c9c35a4ba6638bc99d1.jpg)

![cf3e199a36b5e1cfeb9caf974acc84b2396db9a9c14a86ba29e77e46af5f5e03.jpg](../icml_results/5102_LSGQuant_%20Layer-Sensitivity%20Guided%20Quantization%20for%20One-Step%20Diffusion%20Real-World%20Video%20Super-Resolu/tables/cf3e199a36b5e1cfeb9caf974acc84b2396db9a9c14a86ba29e77e46af5f5e03.jpg)

![e5d78b8d9fa430a170c5d51fe2abfdb07699571f89a24b77342d6687dad07a3b.jpg](../icml_results/5102_LSGQuant_%20Layer-Sensitivity%20Guided%20Quantization%20for%20One-Step%20Diffusion%20Real-World%20Video%20Super-Resolu/tables/e5d78b8d9fa430a170c5d51fe2abfdb07699571f89a24b77342d6687dad07a3b.jpg)

<a id="paper-5055"></a>

## GCIB: Graph Contrastive Information Bottleneck for Multi-Behavior Recommendation


### Images

![16c25b34f8ebd1ca9ef62897bd06ebc6038b06d31a5f6d7eb5c9403417f6faaf.jpg](../icml_results/5103_GCIB_%20Graph%20Contrastive%20Information%20Bottleneck%20for%20Multi-Behavior%20Recommendation/images/16c25b34f8ebd1ca9ef62897bd06ebc6038b06d31a5f6d7eb5c9403417f6faaf.jpg)

![1e1f32576465a59a056e1fb43785ea5485af0b0154f3a25a4b5bcbaebfe44f36.jpg](../icml_results/5103_GCIB_%20Graph%20Contrastive%20Information%20Bottleneck%20for%20Multi-Behavior%20Recommendation/images/1e1f32576465a59a056e1fb43785ea5485af0b0154f3a25a4b5bcbaebfe44f36.jpg)

![43e9099c659d5921b75bed49b96bd20dd303d8249fa975e467086884835c4e98.jpg](../icml_results/5103_GCIB_%20Graph%20Contrastive%20Information%20Bottleneck%20for%20Multi-Behavior%20Recommendation/images/43e9099c659d5921b75bed49b96bd20dd303d8249fa975e467086884835c4e98.jpg)

![53649970df37173cb06922fd5a65042e76e372aa7e7aafe985bce1d7cb7bb5c6.jpg](../icml_results/5103_GCIB_%20Graph%20Contrastive%20Information%20Bottleneck%20for%20Multi-Behavior%20Recommendation/images/53649970df37173cb06922fd5a65042e76e372aa7e7aafe985bce1d7cb7bb5c6.jpg)

![8f1efbd961f928e4f70b99594c281f7b19954ab50b9e345861d133f78a48cbb1.jpg](../icml_results/5103_GCIB_%20Graph%20Contrastive%20Information%20Bottleneck%20for%20Multi-Behavior%20Recommendation/images/8f1efbd961f928e4f70b99594c281f7b19954ab50b9e345861d133f78a48cbb1.jpg)

![abcf9fcab6be7ac669b22b4d28fb8a7148f48d6975a824aafefce649d005c69d.jpg](../icml_results/5103_GCIB_%20Graph%20Contrastive%20Information%20Bottleneck%20for%20Multi-Behavior%20Recommendation/images/abcf9fcab6be7ac669b22b4d28fb8a7148f48d6975a824aafefce649d005c69d.jpg)

### Tables

![196856f8ad7cecfd534bb10e86bdf8f523102fad069d7cfaa9c3ab505bdbd852.jpg](../icml_results/5103_GCIB_%20Graph%20Contrastive%20Information%20Bottleneck%20for%20Multi-Behavior%20Recommendation/tables/196856f8ad7cecfd534bb10e86bdf8f523102fad069d7cfaa9c3ab505bdbd852.jpg)

![21b48129763dccd27702a8e0d61f02435129d36e041884c1bb3cc3c11217ac76.jpg](../icml_results/5103_GCIB_%20Graph%20Contrastive%20Information%20Bottleneck%20for%20Multi-Behavior%20Recommendation/tables/21b48129763dccd27702a8e0d61f02435129d36e041884c1bb3cc3c11217ac76.jpg)

![4fce01d6327572d5d4b0c152f0c7cece2d624ec532396c469ec767f0514c5cbc.jpg](../icml_results/5103_GCIB_%20Graph%20Contrastive%20Information%20Bottleneck%20for%20Multi-Behavior%20Recommendation/tables/4fce01d6327572d5d4b0c152f0c7cece2d624ec532396c469ec767f0514c5cbc.jpg)

![56763ff54ede17a36c8c45ee26d15249f85f7ccc904bd457ba40883974761a0a.jpg](../icml_results/5103_GCIB_%20Graph%20Contrastive%20Information%20Bottleneck%20for%20Multi-Behavior%20Recommendation/tables/56763ff54ede17a36c8c45ee26d15249f85f7ccc904bd457ba40883974761a0a.jpg)

![aeedaa5260804828a2d29270116e4ee2698a4a59f8044f0ef5ac3dda70e0ad3e.jpg](../icml_results/5103_GCIB_%20Graph%20Contrastive%20Information%20Bottleneck%20for%20Multi-Behavior%20Recommendation/tables/aeedaa5260804828a2d29270116e4ee2698a4a59f8044f0ef5ac3dda70e0ad3e.jpg)

![c3d0adc8f74ccc6ee5a623532ec07e5af621a776206bff412e62754ded75be87.jpg](../icml_results/5103_GCIB_%20Graph%20Contrastive%20Information%20Bottleneck%20for%20Multi-Behavior%20Recommendation/tables/c3d0adc8f74ccc6ee5a623532ec07e5af621a776206bff412e62754ded75be87.jpg)

<a id="paper-5056"></a>

## Efficient Diffusion Models via Time Step Optimization with Consistent Training and Inference Constraints


### Images

![38daf13d37390ce3831ece1bc880b3d1ad6c2093e662ceb0bae55f439b66ec86.jpg](../icml_results/5104_Efficient%20Diffusion%20Models%20via%20Time%20Step%20Optimization%20with%20Consistent%20Training%20and%20Inference%20Constra/images/38daf13d37390ce3831ece1bc880b3d1ad6c2093e662ceb0bae55f439b66ec86.jpg)

### Tables

![1bd2163fc7278edd5be63da080d0518662ddd183ec076dd7d54bea1aa0247492.jpg](../icml_results/5104_Efficient%20Diffusion%20Models%20via%20Time%20Step%20Optimization%20with%20Consistent%20Training%20and%20Inference%20Constra/tables/1bd2163fc7278edd5be63da080d0518662ddd183ec076dd7d54bea1aa0247492.jpg)

![3a88e29f16937edc2a5de98e5d2c9f7025a9be54016fe468a8febf06a2d4ad50.jpg](../icml_results/5104_Efficient%20Diffusion%20Models%20via%20Time%20Step%20Optimization%20with%20Consistent%20Training%20and%20Inference%20Constra/tables/3a88e29f16937edc2a5de98e5d2c9f7025a9be54016fe468a8febf06a2d4ad50.jpg)

![68ccb47c9c3d7fde14123480351365db41ae02d3ccdc007e11c8bc80d51d17e4.jpg](../icml_results/5104_Efficient%20Diffusion%20Models%20via%20Time%20Step%20Optimization%20with%20Consistent%20Training%20and%20Inference%20Constra/tables/68ccb47c9c3d7fde14123480351365db41ae02d3ccdc007e11c8bc80d51d17e4.jpg)

![a90d3423dad69ba80b3dbe978315e2bae2bc05e60f28abb06e96438e8c6b63af.jpg](../icml_results/5104_Efficient%20Diffusion%20Models%20via%20Time%20Step%20Optimization%20with%20Consistent%20Training%20and%20Inference%20Constra/tables/a90d3423dad69ba80b3dbe978315e2bae2bc05e60f28abb06e96438e8c6b63af.jpg)

![af7e2bc788b68b261ce5463f6735b59fa59d857e6c310a11027880f628279224.jpg](../icml_results/5104_Efficient%20Diffusion%20Models%20via%20Time%20Step%20Optimization%20with%20Consistent%20Training%20and%20Inference%20Constra/tables/af7e2bc788b68b261ce5463f6735b59fa59d857e6c310a11027880f628279224.jpg)

![e6bb51202757c9b836697df2926fc01960d76992a56b5d9d47a35439596ba50a.jpg](../icml_results/5104_Efficient%20Diffusion%20Models%20via%20Time%20Step%20Optimization%20with%20Consistent%20Training%20and%20Inference%20Constra/tables/e6bb51202757c9b836697df2926fc01960d76992a56b5d9d47a35439596ba50a.jpg)

<a id="paper-5057"></a>

## Multi-scale Explainer for Graph Neural Networks


### Images

![096f0fdcb7f57d788f13229b3045e0b2bcf1449016929acd7dccaa60595dac2d.jpg](../icml_results/5105_Multi-scale%20Explainer%20for%20Graph%20Neural%20Networks/images/096f0fdcb7f57d788f13229b3045e0b2bcf1449016929acd7dccaa60595dac2d.jpg)

![16b7c3af84d1ceed31dfbcc75ddf5b25460f5d001a134507f311179e6cae4823.jpg](../icml_results/5105_Multi-scale%20Explainer%20for%20Graph%20Neural%20Networks/images/16b7c3af84d1ceed31dfbcc75ddf5b25460f5d001a134507f311179e6cae4823.jpg)

![1c27bdd025f4563c8d048303f44413f648ba362286e920df08dec8895391806f.jpg](../icml_results/5105_Multi-scale%20Explainer%20for%20Graph%20Neural%20Networks/images/1c27bdd025f4563c8d048303f44413f648ba362286e920df08dec8895391806f.jpg)

![5c550f0af00d3f32a9e10251ffba116d97a448c737c6e77a09f39c23a72e8d2b.jpg](../icml_results/5105_Multi-scale%20Explainer%20for%20Graph%20Neural%20Networks/images/5c550f0af00d3f32a9e10251ffba116d97a448c737c6e77a09f39c23a72e8d2b.jpg)

![a05e3fe31968360af7f8b1eba02708566838d950cdf61928517f4a6ae703b8f3.jpg](../icml_results/5105_Multi-scale%20Explainer%20for%20Graph%20Neural%20Networks/images/a05e3fe31968360af7f8b1eba02708566838d950cdf61928517f4a6ae703b8f3.jpg)

![d62845edadb4c432d4167035a7ab90c385a13b53335f85470dfcc412db9812b7.jpg](../icml_results/5105_Multi-scale%20Explainer%20for%20Graph%20Neural%20Networks/images/d62845edadb4c432d4167035a7ab90c385a13b53335f85470dfcc412db9812b7.jpg)

![d83fc659f03acc4ed48902b84bbe15492b91aa559bb2ad20741075ef8a9e306a.jpg](../icml_results/5105_Multi-scale%20Explainer%20for%20Graph%20Neural%20Networks/images/d83fc659f03acc4ed48902b84bbe15492b91aa559bb2ad20741075ef8a9e306a.jpg)

### Tables

![55ca5d837315adc5e4d7d2cb367fafd50d3eada75058c4f57f3c6bae13b4921a.jpg](../icml_results/5105_Multi-scale%20Explainer%20for%20Graph%20Neural%20Networks/tables/55ca5d837315adc5e4d7d2cb367fafd50d3eada75058c4f57f3c6bae13b4921a.jpg)

![7d01034218f7d0fc941300b50e3a03c5240044f210f2742acc2c86047c3d09c2.jpg](../icml_results/5105_Multi-scale%20Explainer%20for%20Graph%20Neural%20Networks/tables/7d01034218f7d0fc941300b50e3a03c5240044f210f2742acc2c86047c3d09c2.jpg)

![e983f243f2e45a56334dafb10c4112200571f630afe3a965ce0ce12bc2a885bc.jpg](../icml_results/5105_Multi-scale%20Explainer%20for%20Graph%20Neural%20Networks/tables/e983f243f2e45a56334dafb10c4112200571f630afe3a965ce0ce12bc2a885bc.jpg)

<a id="paper-5058"></a>

## HumanLM: Simulating Users with State Alignment Beats Response Imitation


### Images

![07563eb97ea051c250f8c89f39f16ad26fe8c619048eeab971d18c6759a0ea83.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/07563eb97ea051c250f8c89f39f16ad26fe8c619048eeab971d18c6759a0ea83.jpg)

![15990dea1cc45b181780fb9b9c936f359da0e0da642b09bc5820e74e5f5e5eea.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/15990dea1cc45b181780fb9b9c936f359da0e0da642b09bc5820e74e5f5e5eea.jpg)

![3291473a0765f1d65864bd99d53448e02df968d62bf3f57228069d01511f2ae1.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/3291473a0765f1d65864bd99d53448e02df968d62bf3f57228069d01511f2ae1.jpg)

![46fb9d6868cffab162021976090bafaef643c5674b0cc02e6005f9f5df5279fd.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/46fb9d6868cffab162021976090bafaef643c5674b0cc02e6005f9f5df5279fd.jpg)

![5362ce873f311d8d834e69e1f2d9bf3f05ab91ab421936b66683fbae1a47868f.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/5362ce873f311d8d834e69e1f2d9bf3f05ab91ab421936b66683fbae1a47868f.jpg)

![6b8cf31b1a0faaffc3c63f997b257da1fdcd15aa764d2e4758e885a63318236a.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/6b8cf31b1a0faaffc3c63f997b257da1fdcd15aa764d2e4758e885a63318236a.jpg)

![76748a5a9f458fa5b2cdf797e1964851a7a01d37dc882b530521eeaf2057d8a8.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/76748a5a9f458fa5b2cdf797e1964851a7a01d37dc882b530521eeaf2057d8a8.jpg)

![773b5dc90cc41a43d639aab3a159bfd99b12b8d715fc71b26b8622635c7bb4cc.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/773b5dc90cc41a43d639aab3a159bfd99b12b8d715fc71b26b8622635c7bb4cc.jpg)

![9bbb585b87660011c1efd09c235142ee0393f90fac1dc4fc8ec43eb7a19b6bd8.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/9bbb585b87660011c1efd09c235142ee0393f90fac1dc4fc8ec43eb7a19b6bd8.jpg)

![9c1b94c475dd956284bf51468df21eb3f119ed131265492e3d4ec0c2408417de.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/9c1b94c475dd956284bf51468df21eb3f119ed131265492e3d4ec0c2408417de.jpg)

![a4e79a91c62712d148944ec598071fc2719a0c8f38b20f7677db11d569e1a119.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/a4e79a91c62712d148944ec598071fc2719a0c8f38b20f7677db11d569e1a119.jpg)

![a596342c935de3bbb5800e5166cc1f09987e8f1917e78ec4756de23ff8dcca22.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/a596342c935de3bbb5800e5166cc1f09987e8f1917e78ec4756de23ff8dcca22.jpg)

![af5a8622e05cde722f6ad13f8f0123ca65d1fc19d5337f1bccc1a41c5bc00a2b.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/af5a8622e05cde722f6ad13f8f0123ca65d1fc19d5337f1bccc1a41c5bc00a2b.jpg)

![b7004a2b53d345b246521f1a380fe2aa7fc8e0c12dc487a18c48b8d6942395fd.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/b7004a2b53d345b246521f1a380fe2aa7fc8e0c12dc487a18c48b8d6942395fd.jpg)

![c2af7fedf459cd7bdae194f2de1360dea5efed1f6c27cf93460b83c6ab7b8e09.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/c2af7fedf459cd7bdae194f2de1360dea5efed1f6c27cf93460b83c6ab7b8e09.jpg)

![c3c3faddcd4b8307eeb1725e9fcae0ca5e8757edf544bbc912dae04df11fffdf.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/c3c3faddcd4b8307eeb1725e9fcae0ca5e8757edf544bbc912dae04df11fffdf.jpg)

![f5bba37337f89dc18b025b166a46230ee7caf783dccde7460a56e16f9b69dd6c.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/f5bba37337f89dc18b025b166a46230ee7caf783dccde7460a56e16f9b69dd6c.jpg)

![fd8f8a537ae60816c1fab2032620e66c8c196827ed9bf70f55352f6b902eac24.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/images/fd8f8a537ae60816c1fab2032620e66c8c196827ed9bf70f55352f6b902eac24.jpg)

### Tables

![000df979d2ace78f37d1ddd5634f746714e034408a554dcb02d3f7474dd627f2.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/tables/000df979d2ace78f37d1ddd5634f746714e034408a554dcb02d3f7474dd627f2.jpg)

![626f1881ea8ba201d815039aeb71e061b544f17c62411723215bfb72be12eacc.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/tables/626f1881ea8ba201d815039aeb71e061b544f17c62411723215bfb72be12eacc.jpg)

![67473b14c8204cabfb7f54105d1920914f80ae363e98c7acd3ee76e9c83b5d10.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/tables/67473b14c8204cabfb7f54105d1920914f80ae363e98c7acd3ee76e9c83b5d10.jpg)

![839e636fed67c2dd165dc6647a040b1f98bb80444ba08de447c0ef5e06a352a8.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/tables/839e636fed67c2dd165dc6647a040b1f98bb80444ba08de447c0ef5e06a352a8.jpg)

![85d00e8079078d8da729201cf93b7cec17c308c516966c3867c7139db9a2bff3.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/tables/85d00e8079078d8da729201cf93b7cec17c308c516966c3867c7139db9a2bff3.jpg)

![abc992bb396027b72a00e0173a5542932ba89f661b9913daf4207a0513354c82.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/tables/abc992bb396027b72a00e0173a5542932ba89f661b9913daf4207a0513354c82.jpg)

![b2513476bb76345357fddc02e0f01935f26286e481a0f247d06c1857da05fc88.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/tables/b2513476bb76345357fddc02e0f01935f26286e481a0f247d06c1857da05fc88.jpg)

![b41fef6907576e4b8a0a08fb2bc874b5f38e686e6f37031265109d4d9a8c75ae.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/tables/b41fef6907576e4b8a0a08fb2bc874b5f38e686e6f37031265109d4d9a8c75ae.jpg)

![d80cd6f16c3d470ddb0767171d00997272c3bdce0d0ad5d9b2ca8cade7a949f3.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/tables/d80cd6f16c3d470ddb0767171d00997272c3bdce0d0ad5d9b2ca8cade7a949f3.jpg)

![ee45b40adf7f93b100f7c156b84b39f6574262926a80e07ecd36aba7d9aec779.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/tables/ee45b40adf7f93b100f7c156b84b39f6574262926a80e07ecd36aba7d9aec779.jpg)

![f45f8b1c781ea76f736a7aae21c2086debf6c413bd24eb30c099048711a72039.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/tables/f45f8b1c781ea76f736a7aae21c2086debf6c413bd24eb30c099048711a72039.jpg)

![fe7ff25f55f97d70f8a35d5626587171251eca9003eca262f08812803a37370c.jpg](../icml_results/5106_HumanLM_%20Simulating%20Users%20with%20State%20Alignment%20Beats%20Response%20Imitation/tables/fe7ff25f55f97d70f8a35d5626587171251eca9003eca262f08812803a37370c.jpg)

<a id="paper-5059"></a>

## TimeSeed: Effective Time Series Forecasting with Sparse Endogenous Variables


### Images

![04123c2a51c93b68d00705cae0076385de8b93f86d8d41ba2263609dcd15c6bb.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/images/04123c2a51c93b68d00705cae0076385de8b93f86d8d41ba2263609dcd15c6bb.jpg)

![051036f2d5d083953756cca2ebf963370d8c9f8730a13d42d8626a48f5d48552.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/images/051036f2d5d083953756cca2ebf963370d8c9f8730a13d42d8626a48f5d48552.jpg)

![092dd16e82452dc3aed2a48da37e4d0589516e8ece21b3914bb6fa0c11ca388f.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/images/092dd16e82452dc3aed2a48da37e4d0589516e8ece21b3914bb6fa0c11ca388f.jpg)

![22bc46988c5fd6a2e7c553375eee84a975678da35146ea1976d0702b27fe1f93.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/images/22bc46988c5fd6a2e7c553375eee84a975678da35146ea1976d0702b27fe1f93.jpg)

![569c26a7263dbf2713a939ef55ba699e1403b7463b9ca62a900ac99ddffc3b3a.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/images/569c26a7263dbf2713a939ef55ba699e1403b7463b9ca62a900ac99ddffc3b3a.jpg)

![677152015c82f425f263df84d27666cc88c41a641f8de72b203aa842f32d0a54.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/images/677152015c82f425f263df84d27666cc88c41a641f8de72b203aa842f32d0a54.jpg)

![6941c3484062197b8160cae8835afb691187d95ccac1c343b848ef377c61c2da.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/images/6941c3484062197b8160cae8835afb691187d95ccac1c343b848ef377c61c2da.jpg)

![6bcb69ef216f2ca3f43a9fe92dc4c4f083ad4b008efbcd80b6953c4fd93fe676.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/images/6bcb69ef216f2ca3f43a9fe92dc4c4f083ad4b008efbcd80b6953c4fd93fe676.jpg)

![79552e39f1f4db9ee73038e7f31f68a1f2cf0b2048181185da3e41b9df8e9928.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/images/79552e39f1f4db9ee73038e7f31f68a1f2cf0b2048181185da3e41b9df8e9928.jpg)

![8ec4163adbe3d88d881df25e4a64be431ef86f0a7f43080dcad16c053d7a3432.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/images/8ec4163adbe3d88d881df25e4a64be431ef86f0a7f43080dcad16c053d7a3432.jpg)

![a0ccd116bf3089688ac2fc003df39eb47573fbcb0f0f6bd5e9a6c700b9071433.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/images/a0ccd116bf3089688ac2fc003df39eb47573fbcb0f0f6bd5e9a6c700b9071433.jpg)

![afc06b63ca8ab1017aabc96ba8b95277d868258493f2502abbe1faffbb86c31a.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/images/afc06b63ca8ab1017aabc96ba8b95277d868258493f2502abbe1faffbb86c31a.jpg)

![b652a243001de0878d74b68befdc78bbacde0186db986373433df77d5532c21d.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/images/b652a243001de0878d74b68befdc78bbacde0186db986373433df77d5532c21d.jpg)

![b9625fcf1f1c866f896114344fafdff0b51e13dc68ee27a1869a0761206fa7a8.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/images/b9625fcf1f1c866f896114344fafdff0b51e13dc68ee27a1869a0761206fa7a8.jpg)

![d2a325bfcf7c1c193631cc25c531707e4884788bd64f3835a31d362f8235c6dd.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/images/d2a325bfcf7c1c193631cc25c531707e4884788bd64f3835a31d362f8235c6dd.jpg)

![eeada49a3b64d6862ecb4c0992d5c6d2acd061ea0402623904ba890bd3afb179.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/images/eeada49a3b64d6862ecb4c0992d5c6d2acd061ea0402623904ba890bd3afb179.jpg)

### Tables

![0c5c401670d268e9bf83f8b8c37ed0426eaa09f5543af4051c18571c110a5050.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/0c5c401670d268e9bf83f8b8c37ed0426eaa09f5543af4051c18571c110a5050.jpg)

![217a346bc27227874d62d1ea8a774783aef679a0e577d93f4892e83ee9e6f501.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/217a346bc27227874d62d1ea8a774783aef679a0e577d93f4892e83ee9e6f501.jpg)

![2b95bb614c9f948bcc7fdbaefbc2ab0419de4cc4deda6ab035d4f20741e55a45.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/2b95bb614c9f948bcc7fdbaefbc2ab0419de4cc4deda6ab035d4f20741e55a45.jpg)

![333ce8881e86a4ed1793527de0a9fc6727f84803218a552cd98105531fd6b4bb.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/333ce8881e86a4ed1793527de0a9fc6727f84803218a552cd98105531fd6b4bb.jpg)

![4c25bbfd83a89688c94171099ad1f24b37bbd535823c07cafbb27423383d9df7.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/4c25bbfd83a89688c94171099ad1f24b37bbd535823c07cafbb27423383d9df7.jpg)

![52352ebe838753a953325ba1461907bb2e7dea799b88a4c6fd3a5299830e8ba1.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/52352ebe838753a953325ba1461907bb2e7dea799b88a4c6fd3a5299830e8ba1.jpg)

![5319081c53588ada602048b27c90493af992d3eec83e32d56bfb8cd5389893b7.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/5319081c53588ada602048b27c90493af992d3eec83e32d56bfb8cd5389893b7.jpg)

![5ae409c48e483748c8211410019ac1965683cf79ffd2050da5abf3b574264556.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/5ae409c48e483748c8211410019ac1965683cf79ffd2050da5abf3b574264556.jpg)

![5d09d3b3e45267b476773ebc025cc87cd79ffada1142f4e5aab0becb04651f2b.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/5d09d3b3e45267b476773ebc025cc87cd79ffada1142f4e5aab0becb04651f2b.jpg)

![6a5606a7d1327032d029023f904521d659d2344865b266fa87979229f9670025.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/6a5606a7d1327032d029023f904521d659d2344865b266fa87979229f9670025.jpg)

![6c89d402f119de6cc3cb98a32bafcdf050e8e1b54c0818b23f3cb4ce945d82d5.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/6c89d402f119de6cc3cb98a32bafcdf050e8e1b54c0818b23f3cb4ce945d82d5.jpg)

![8724069cec5c2d5310e3c130ea8efc4574682d9f6cd45fa85fb02b213f71712d.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/8724069cec5c2d5310e3c130ea8efc4574682d9f6cd45fa85fb02b213f71712d.jpg)

![993f801ad0163e4f35bef7b9a4e9b8187756d68fa9c0a45e756b1b0800eb0fc0.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/993f801ad0163e4f35bef7b9a4e9b8187756d68fa9c0a45e756b1b0800eb0fc0.jpg)

![b201454a8bb706adce6c6606854d04ed959826c6c7e2891795a6d95a28524283.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/b201454a8bb706adce6c6606854d04ed959826c6c7e2891795a6d95a28524283.jpg)

![b3072ff4964db02899d7c81f75b362043967dbfc5ecde9eebfffe7d0d7608d9c.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/b3072ff4964db02899d7c81f75b362043967dbfc5ecde9eebfffe7d0d7608d9c.jpg)

![b5cfffb4139e9b541434822d8baf90a1aff6af45b74cfaf1939fbcfafee43a27.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/b5cfffb4139e9b541434822d8baf90a1aff6af45b74cfaf1939fbcfafee43a27.jpg)

![d8b7b5ec488cbbf3555f8887816f3e8ecb33901deb2b9ef7d0bbbf88773e4011.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/d8b7b5ec488cbbf3555f8887816f3e8ecb33901deb2b9ef7d0bbbf88773e4011.jpg)

![ec221a1087ba2b89cd4a0ba6e91061d16df3e10c41d7056ecbedcb7f58ba5348.jpg](../icml_results/5107_TimeSeed_%20Effective%20Time%20Series%20Forecasting%20with%20Sparse%20Endogenous%20Variables/tables/ec221a1087ba2b89cd4a0ba6e91061d16df3e10c41d7056ecbedcb7f58ba5348.jpg)

<a id="paper-5060"></a>

## VLANeXt: Recipes for Building Strong VLA Models


### Images

![03ee81d9991c9c69d0beee31a02dbaa1aa71bd5eb4e416e1733b04c95a6b5295.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/images/03ee81d9991c9c69d0beee31a02dbaa1aa71bd5eb4e416e1733b04c95a6b5295.jpg)

![3641596162215b2d9fa0fe9e8439533818172e191c5c87ed459281b8d60fc09a.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/images/3641596162215b2d9fa0fe9e8439533818172e191c5c87ed459281b8d60fc09a.jpg)

![3bff4a7e6b44551651eb267b08a3b15511488df1408e1148e6d900cbc2c5de51.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/images/3bff4a7e6b44551651eb267b08a3b15511488df1408e1148e6d900cbc2c5de51.jpg)

![4d9c7de14a83e468ee0ad808316dcc1519f5f9f4653f1f3b59b52782961c9aa5.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/images/4d9c7de14a83e468ee0ad808316dcc1519f5f9f4653f1f3b59b52782961c9aa5.jpg)

![7d2499b8cbd72fa3ec10369a5fdd7f24a8fe3dd8ac713c40a2152992a203005f.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/images/7d2499b8cbd72fa3ec10369a5fdd7f24a8fe3dd8ac713c40a2152992a203005f.jpg)

![9e41ccdba23f9c30198980d61638412d10e5e0b7a7bfd5ed898bc58d9fb5c9f1.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/images/9e41ccdba23f9c30198980d61638412d10e5e0b7a7bfd5ed898bc58d9fb5c9f1.jpg)

![9f93e1a74fdee9eabffde9c24c5de583cd098da6a8976cd09bc53cfee0024d20.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/images/9f93e1a74fdee9eabffde9c24c5de583cd098da6a8976cd09bc53cfee0024d20.jpg)

![aa08eb3191d3f9aef55b5d4a7f30eea6a4cb0e2aef928a2265e5d27fafe9e1c3.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/images/aa08eb3191d3f9aef55b5d4a7f30eea6a4cb0e2aef928a2265e5d27fafe9e1c3.jpg)

![c50918a45c9231a661c15a93813acedf8d0f2f37a563f6746f38079b972ad5ea.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/images/c50918a45c9231a661c15a93813acedf8d0f2f37a563f6746f38079b972ad5ea.jpg)

![da9440d77fc701666607833c18572f6abc7746608c205d0a977329220b2cf830.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/images/da9440d77fc701666607833c18572f6abc7746608c205d0a977329220b2cf830.jpg)

![eb88c744b67644caca0042bf2587f771addf01ce0757e8b9f5e7a2451bf27c5d.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/images/eb88c744b67644caca0042bf2587f771addf01ce0757e8b9f5e7a2451bf27c5d.jpg)

### Tables

![1226886b73828af351fe6c59e32ae29de6ab049e0cff543c2e8a39f91f82e556.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/tables/1226886b73828af351fe6c59e32ae29de6ab049e0cff543c2e8a39f91f82e556.jpg)

![13e401190918ad49fc5e5401152d9db82aa843375608b0f5b53d87c36575c403.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/tables/13e401190918ad49fc5e5401152d9db82aa843375608b0f5b53d87c36575c403.jpg)

![3a6dd5440fb3b98230f17c15d182520754138c30f7e66992ecd94e9404807d1e.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/tables/3a6dd5440fb3b98230f17c15d182520754138c30f7e66992ecd94e9404807d1e.jpg)

![b576277c05fb9f99eb0a4fe1c223058382ff5c2f09368cc494e7fe03fc0b6dc3.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/tables/b576277c05fb9f99eb0a4fe1c223058382ff5c2f09368cc494e7fe03fc0b6dc3.jpg)

![bd666d050538119539a5a81db7614fa367ca706743dc93209d058815ed6a8fb6.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/tables/bd666d050538119539a5a81db7614fa367ca706743dc93209d058815ed6a8fb6.jpg)

![ec344447071cf9472c494120452060dd1212e34b0f322e06747a148a2556ed98.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/tables/ec344447071cf9472c494120452060dd1212e34b0f322e06747a148a2556ed98.jpg)

![fe83d46429a3e0183e83c4753683a33a7a24d4eab69e1e545e984653952a8d47.jpg](../icml_results/5108_VLANeXt_%20Recipes%20for%20Building%20Strong%20VLA%20Models/tables/fe83d46429a3e0183e83c4753683a33a7a24d4eab69e1e545e984653952a8d47.jpg)

<a id="paper-5061"></a>

## PyHealth 2.0: A Comprehensive Open-Source Toolkit for Accessible and Reproducible Clinical Deep Learning


### Images

![054a072d0a482d0f337f1710cc4d094b5ea38415aa1f8d7bbb5604c951f137d2.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/images/054a072d0a482d0f337f1710cc4d094b5ea38415aa1f8d7bbb5604c951f137d2.jpg)

![2518d95eeb71288688cc86cfaffc33545d76f7c979ba50c0fb2e0f1c5023f653.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/images/2518d95eeb71288688cc86cfaffc33545d76f7c979ba50c0fb2e0f1c5023f653.jpg)

![3e1e83f0780c8d06d64459b629c5828fcfdbf9ef00af110ecac5c58d6ec34fc0.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/images/3e1e83f0780c8d06d64459b629c5828fcfdbf9ef00af110ecac5c58d6ec34fc0.jpg)

![866d000feb604b50e7bcf587770029f2a03c3608654741801435b251070e839c.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/images/866d000feb604b50e7bcf587770029f2a03c3608654741801435b251070e839c.jpg)

![9febc4aeefb020f1c37f2f17fbd2e1df862c13defd1ef1331d5566987bf531b1.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/images/9febc4aeefb020f1c37f2f17fbd2e1df862c13defd1ef1331d5566987bf531b1.jpg)

![b503e1b5fef05c7123af030bdd7fb4a81c8e208ea6e6de8346d0f328f7c0029d.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/images/b503e1b5fef05c7123af030bdd7fb4a81c8e208ea6e6de8346d0f328f7c0029d.jpg)

### Tables

![00df74e42039e3c53503b787d18f8d399a5144be5c66b54cb25e471830049b2c.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/00df74e42039e3c53503b787d18f8d399a5144be5c66b54cb25e471830049b2c.jpg)

![049a3821457d50f435d6978f375996b03aa69cb127ffa78ce564febae489ea57.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/049a3821457d50f435d6978f375996b03aa69cb127ffa78ce564febae489ea57.jpg)

![2219f26b8860943ee5b8f6b17fa5372b35aff5d61158da8d3637a43ec28c114e.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/2219f26b8860943ee5b8f6b17fa5372b35aff5d61158da8d3637a43ec28c114e.jpg)

![223f5207481f1e38559b4f6993fbf64239db0e15e90f3d86bf11152ef603e2af.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/223f5207481f1e38559b4f6993fbf64239db0e15e90f3d86bf11152ef603e2af.jpg)

![232304585d96ca1047a36ec5b46a0e935c903d39058915e59a30a6df7954c28a.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/232304585d96ca1047a36ec5b46a0e935c903d39058915e59a30a6df7954c28a.jpg)

![265e37f23433316978782caa6dfdd5c1f392840e54b9a7a5d1ef5285775d18fb.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/265e37f23433316978782caa6dfdd5c1f392840e54b9a7a5d1ef5285775d18fb.jpg)

![2e0e86d41b064902b5d66ef9dee46569e5322cb4f9b93678217a78ae2c171926.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/2e0e86d41b064902b5d66ef9dee46569e5322cb4f9b93678217a78ae2c171926.jpg)

![2eb9854282d99f1c7fcd3259c9efc7fad9ca36bce688ec375f68cd5126757135.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/2eb9854282d99f1c7fcd3259c9efc7fad9ca36bce688ec375f68cd5126757135.jpg)

![3164f4b8166191c073769c9aac583715b6b835e12ad37a2ae86aee931d43a7dd.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/3164f4b8166191c073769c9aac583715b6b835e12ad37a2ae86aee931d43a7dd.jpg)

![37c162563cd79e0ad5f31b2d428ddbfc9c2ad2dde811c284923b1ca0e66a9916.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/37c162563cd79e0ad5f31b2d428ddbfc9c2ad2dde811c284923b1ca0e66a9916.jpg)

![44ef8e3a93a9801b0383b5579149a8ce942d62bad4df6f94764f2f2bb64cf879.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/44ef8e3a93a9801b0383b5579149a8ce942d62bad4df6f94764f2f2bb64cf879.jpg)

![4dab33d6d18d2b711251a0fd3ff19f750f9379b4d4a9914bc934ea13b94138cc.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/4dab33d6d18d2b711251a0fd3ff19f750f9379b4d4a9914bc934ea13b94138cc.jpg)

![4ddcc442e70b142ca0ddf6fd6edc775ae0fcdb974095e76c1db96e0800b17320.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/4ddcc442e70b142ca0ddf6fd6edc775ae0fcdb974095e76c1db96e0800b17320.jpg)

![51fb54378ff356b558ecebbc3ac8708247af0ae05beea5cc2deaaf903d754a5a.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/51fb54378ff356b558ecebbc3ac8708247af0ae05beea5cc2deaaf903d754a5a.jpg)

![587d2700e6e0203be188e2d9f1f68c0f6c5b8d89f3df1fe4c3932b6db27af9d8.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/587d2700e6e0203be188e2d9f1f68c0f6c5b8d89f3df1fe4c3932b6db27af9d8.jpg)

![5a064540bd273de63820badc7fe4d2d34ff9dfc5203c441db22cdf3cce034ca0.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/5a064540bd273de63820badc7fe4d2d34ff9dfc5203c441db22cdf3cce034ca0.jpg)

![5ae86028c9a5c8bfb6444f3e24e212e5ad03078531895bb257f34a8ecd4ceb9d.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/5ae86028c9a5c8bfb6444f3e24e212e5ad03078531895bb257f34a8ecd4ceb9d.jpg)

![5cabd3e38745e6cbbfe2dbf3d99094bad3bfc532966a27cf03a463f774ecc84a.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/5cabd3e38745e6cbbfe2dbf3d99094bad3bfc532966a27cf03a463f774ecc84a.jpg)

![6a0efdbbbdd520424b5ce774f40fa46df7e67dd8c192258cdb651e41b51c46df.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/6a0efdbbbdd520424b5ce774f40fa46df7e67dd8c192258cdb651e41b51c46df.jpg)

![6d5a6e9cf20962208579291a3aaa9add773d28aa1ad3612d52f4c3a38d17a35a.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/6d5a6e9cf20962208579291a3aaa9add773d28aa1ad3612d52f4c3a38d17a35a.jpg)

![72ac6f6066b724346b5a261cc9565086f8f1428361be93650a79d417a71d6b72.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/72ac6f6066b724346b5a261cc9565086f8f1428361be93650a79d417a71d6b72.jpg)

![761f5f751223094bffa798794e0f71767141bec8842b4537459084c61de397fc.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/761f5f751223094bffa798794e0f71767141bec8842b4537459084c61de397fc.jpg)

![79b40200c01b2626be8831fb98f40343c441b713cc8e90b856318e66c11e1641.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/79b40200c01b2626be8831fb98f40343c441b713cc8e90b856318e66c11e1641.jpg)

![7c4ae79d9df92308ead9cd2c9823a354afb3eff4400e8de68b70b1928920455e.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/7c4ae79d9df92308ead9cd2c9823a354afb3eff4400e8de68b70b1928920455e.jpg)

![8269172432229c16eb863874fe0e0927bc5baede5ee613dc8b56274897fa9e6e.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/8269172432229c16eb863874fe0e0927bc5baede5ee613dc8b56274897fa9e6e.jpg)

![84c3962aa2253656bdbde1cc4b2c0dfbc66e4af2b78fb70dbd656d26ff9a2c62.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/84c3962aa2253656bdbde1cc4b2c0dfbc66e4af2b78fb70dbd656d26ff9a2c62.jpg)

![8bbc737eca547ca81b80f2628bb82d4c59d3efbedf649febefe354443b399d11.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/8bbc737eca547ca81b80f2628bb82d4c59d3efbedf649febefe354443b399d11.jpg)

![8fb6d89abd067fd39006e82a573df8d17f975c12fa1ce058f69ea36923a983a7.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/8fb6d89abd067fd39006e82a573df8d17f975c12fa1ce058f69ea36923a983a7.jpg)

![949f38c9b50d87ad3aa5850ddc399ef36852c07068568c2206aa826f0d9e0b19.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/949f38c9b50d87ad3aa5850ddc399ef36852c07068568c2206aa826f0d9e0b19.jpg)

![9bd51c98777cc1bbee46cd267e3a40007cced1c84dd6f5877a2c73de76553ed1.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/9bd51c98777cc1bbee46cd267e3a40007cced1c84dd6f5877a2c73de76553ed1.jpg)

![9e8ace1c9219a1c5e59981bf4b6f9e10ff1ca1090de6ca4a07d04243eb01b9e1.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/9e8ace1c9219a1c5e59981bf4b6f9e10ff1ca1090de6ca4a07d04243eb01b9e1.jpg)

![a07b99e0f57090a292189f8d0e753c01cbffe02c16f247962d27d33ce1b3cf50.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/a07b99e0f57090a292189f8d0e753c01cbffe02c16f247962d27d33ce1b3cf50.jpg)

![a84655cd69e67c7a9304aa7c7a88c48d03d31d4dc92b6bc8724411569f5bab1f.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/a84655cd69e67c7a9304aa7c7a88c48d03d31d4dc92b6bc8724411569f5bab1f.jpg)

![aa3748fd22c537982c2ce553e3d0e6996f5dc52ec320b243f1f17b12433e93e3.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/aa3748fd22c537982c2ce553e3d0e6996f5dc52ec320b243f1f17b12433e93e3.jpg)

![abba73b46e667b2594560e679b9fee8d9298860b49616915da6211c11c1972fc.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/abba73b46e667b2594560e679b9fee8d9298860b49616915da6211c11c1972fc.jpg)

![abbb2a8ba943e8c19037792e1f1072ab2afa188a68e96f3113d2e21986605373.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/abbb2a8ba943e8c19037792e1f1072ab2afa188a68e96f3113d2e21986605373.jpg)

![b0f1b29444d5332d6aa700b29bb8af71851fbc9dbdf17c83d88fd5a9e41e6a3a.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/b0f1b29444d5332d6aa700b29bb8af71851fbc9dbdf17c83d88fd5a9e41e6a3a.jpg)

![b48e0be422e09db4573e39a92c6142ba19e73508d4f6aa739928fb1967f78a48.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/b48e0be422e09db4573e39a92c6142ba19e73508d4f6aa739928fb1967f78a48.jpg)

![b7e54b75f7cf6af4093d4efeb83635ca5aea66fab4494abd5f6f2b00ef886b2c.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/b7e54b75f7cf6af4093d4efeb83635ca5aea66fab4494abd5f6f2b00ef886b2c.jpg)

![bb79bff364b417bba913ae15998aad64ddd889822ebe0b638181fccfd10030d0.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/bb79bff364b417bba913ae15998aad64ddd889822ebe0b638181fccfd10030d0.jpg)

![bc9b79cded04ec564f21e30ec317de949170ee5f71345b53f4aecabbe3b2e0c5.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/bc9b79cded04ec564f21e30ec317de949170ee5f71345b53f4aecabbe3b2e0c5.jpg)

![bcf6e7b9ee5e906ba31b1550eaf88bedb7b31460cbcb784b59ac71cb1e4ba19d.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/bcf6e7b9ee5e906ba31b1550eaf88bedb7b31460cbcb784b59ac71cb1e4ba19d.jpg)

![c1dc17f09fc50f36558748df4e393ccc75097ba68e8fa2dc1144cdc57d4af3c5.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/c1dc17f09fc50f36558748df4e393ccc75097ba68e8fa2dc1144cdc57d4af3c5.jpg)

![c86c0081d4a2353455e69e184f839a6a8ed3865e3f77b9b1013f0c3356c5c82d.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/c86c0081d4a2353455e69e184f839a6a8ed3865e3f77b9b1013f0c3356c5c82d.jpg)

![c9bde3f3aed5bd0364054f2cfffc3611b16d7ab980eee7bab98bb8cd88353ff7.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/c9bde3f3aed5bd0364054f2cfffc3611b16d7ab980eee7bab98bb8cd88353ff7.jpg)

![ca3d410a6ea1d5475ca1ebbba09b7820999d5fc7502df407f59c392c9626fefa.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/ca3d410a6ea1d5475ca1ebbba09b7820999d5fc7502df407f59c392c9626fefa.jpg)

![cb236c077320c747f00da3f01ac77ba283de6a37877ef531c36dedc4e91d77d8.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/cb236c077320c747f00da3f01ac77ba283de6a37877ef531c36dedc4e91d77d8.jpg)

![d27929929af2591bba61fcbbf5bc5e14ae9e4d30e9400c3ee481daef567528d0.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/d27929929af2591bba61fcbbf5bc5e14ae9e4d30e9400c3ee481daef567528d0.jpg)

![d2b0934e237744177ba0f08cc81bc509a895a82f3e762ea68835df4600fca9af.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/d2b0934e237744177ba0f08cc81bc509a895a82f3e762ea68835df4600fca9af.jpg)

![da09170b4806b958f2ebbc83156153c9d782e84ff56bdfb8d4ea16be50c5a009.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/da09170b4806b958f2ebbc83156153c9d782e84ff56bdfb8d4ea16be50c5a009.jpg)

![dd3a091fa52b2d4349cb6f34a451328c3dd9f0dc84d1a99df6abc8a53a4c14da.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/dd3a091fa52b2d4349cb6f34a451328c3dd9f0dc84d1a99df6abc8a53a4c14da.jpg)

![e482f58894490bcafe982181f52dc21f66adaf52acd1c7b587a9672f2c33c19b.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/e482f58894490bcafe982181f52dc21f66adaf52acd1c7b587a9672f2c33c19b.jpg)

![eec682e540ba5b600b438530c21405391e5e65ed2c890e3521541d1f923d1449.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/eec682e540ba5b600b438530c21405391e5e65ed2c890e3521541d1f923d1449.jpg)

![f02eb1d42d45f06a39f910a058a5df223727a407dfa851504d145ef0a3ed0c5d.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/f02eb1d42d45f06a39f910a058a5df223727a407dfa851504d145ef0a3ed0c5d.jpg)

![f4f4f38d49304ba02484ff82137e3981249377761a0270a7f04c821bcc849c2a.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/f4f4f38d49304ba02484ff82137e3981249377761a0270a7f04c821bcc849c2a.jpg)

![f5e0206584910c1614bd02de0bc6475792c2a6ce76aa2e33b2be0399a08f6803.jpg](../icml_results/5109_PyHealth%202.0_%20A%20Comprehensive%20Open-Source%20Toolkit%20for%20Accessible%20and%20Reproducible%20Clinical%20Deep%20Lear/tables/f5e0206584910c1614bd02de0bc6475792c2a6ce76aa2e33b2be0399a08f6803.jpg)

<a id="paper-5062"></a>

## EasyBalance: Cross-Layer Load Balancing in Distributed MoE Inference


### Images

![22ae4ebe621719c7628350c856e61561ad46718997fef5ec50f707d94c8cad15.jpg](../icml_results/5110_EasyBalance_%20Cross-Layer%20Load%20Balancing%20in%20Distributed%20MoE%20Inference/images/22ae4ebe621719c7628350c856e61561ad46718997fef5ec50f707d94c8cad15.jpg)

![30471bcf3c6aa402d82cd5c1faab7b4d7110fbc327a3728713c90dba752a960d.jpg](../icml_results/5110_EasyBalance_%20Cross-Layer%20Load%20Balancing%20in%20Distributed%20MoE%20Inference/images/30471bcf3c6aa402d82cd5c1faab7b4d7110fbc327a3728713c90dba752a960d.jpg)

![55de7733b70a7f5bcb2be701c63604772a235db523f48548e574dcc42d55e223.jpg](../icml_results/5110_EasyBalance_%20Cross-Layer%20Load%20Balancing%20in%20Distributed%20MoE%20Inference/images/55de7733b70a7f5bcb2be701c63604772a235db523f48548e574dcc42d55e223.jpg)

![5f51428babb5b96bf487f8a7bccf1f212435cacc77ef690702f6e05a9bd98146.jpg](../icml_results/5110_EasyBalance_%20Cross-Layer%20Load%20Balancing%20in%20Distributed%20MoE%20Inference/images/5f51428babb5b96bf487f8a7bccf1f212435cacc77ef690702f6e05a9bd98146.jpg)

![6ccab1de7ed20a32ca10ccbfc0047f87c52da088aaaef1fbb9cba72684d09ea8.jpg](../icml_results/5110_EasyBalance_%20Cross-Layer%20Load%20Balancing%20in%20Distributed%20MoE%20Inference/images/6ccab1de7ed20a32ca10ccbfc0047f87c52da088aaaef1fbb9cba72684d09ea8.jpg)

![6da2f2c3ea35bd444a651c249683e762c0c546d9b23de219d0a8f7341a526cbd.jpg](../icml_results/5110_EasyBalance_%20Cross-Layer%20Load%20Balancing%20in%20Distributed%20MoE%20Inference/images/6da2f2c3ea35bd444a651c249683e762c0c546d9b23de219d0a8f7341a526cbd.jpg)

![7dc8f49dffeb2aee93b199ad2b6ee461396e2deec0d58fa8674cd2a01f1be43e.jpg](../icml_results/5110_EasyBalance_%20Cross-Layer%20Load%20Balancing%20in%20Distributed%20MoE%20Inference/images/7dc8f49dffeb2aee93b199ad2b6ee461396e2deec0d58fa8674cd2a01f1be43e.jpg)

![c97c47b116b837049d45068a9252cf8c44e89002e2a10b7e78ad0701b60ac770.jpg](../icml_results/5110_EasyBalance_%20Cross-Layer%20Load%20Balancing%20in%20Distributed%20MoE%20Inference/images/c97c47b116b837049d45068a9252cf8c44e89002e2a10b7e78ad0701b60ac770.jpg)

![f7f302e30d42afb1ed9ad75e22f21658cbf558453bbfd2fc255bf9fec27fc275.jpg](../icml_results/5110_EasyBalance_%20Cross-Layer%20Load%20Balancing%20in%20Distributed%20MoE%20Inference/images/f7f302e30d42afb1ed9ad75e22f21658cbf558453bbfd2fc255bf9fec27fc275.jpg)

### Tables

![2c4abf3149a20f6e4be580c7b1d6a4b24d450c44d690ff87ae8d2a41a4ccea19.jpg](../icml_results/5110_EasyBalance_%20Cross-Layer%20Load%20Balancing%20in%20Distributed%20MoE%20Inference/tables/2c4abf3149a20f6e4be580c7b1d6a4b24d450c44d690ff87ae8d2a41a4ccea19.jpg)

![6011825238db01a970b17a2d2102880a142561d13fce787333a39343ed41dc33.jpg](../icml_results/5110_EasyBalance_%20Cross-Layer%20Load%20Balancing%20in%20Distributed%20MoE%20Inference/tables/6011825238db01a970b17a2d2102880a142561d13fce787333a39343ed41dc33.jpg)

![7e811e88979fa7f0a776c866ae4ea96bd0061e43b48b8d09b84481300b4c4b0e.jpg](../icml_results/5110_EasyBalance_%20Cross-Layer%20Load%20Balancing%20in%20Distributed%20MoE%20Inference/tables/7e811e88979fa7f0a776c866ae4ea96bd0061e43b48b8d09b84481300b4c4b0e.jpg)

![a9b8460462fdc916b7f9fe4812d3ea1745110649df852ab05538bf20975582c1.jpg](../icml_results/5110_EasyBalance_%20Cross-Layer%20Load%20Balancing%20in%20Distributed%20MoE%20Inference/tables/a9b8460462fdc916b7f9fe4812d3ea1745110649df852ab05538bf20975582c1.jpg)

![f8d027d7d72332ca3629c6aac839f1d51c919e7b2f24f4fb47de89fed9be558a.jpg](../icml_results/5110_EasyBalance_%20Cross-Layer%20Load%20Balancing%20in%20Distributed%20MoE%20Inference/tables/f8d027d7d72332ca3629c6aac839f1d51c919e7b2f24f4fb47de89fed9be558a.jpg)

<a id="paper-5063"></a>

## OpenDeception: Learning Deception and Trust in Human–AI Interaction via Multi-Agent Simulation


### Images

![0230618c69c7176c43f1740ad798de60d0292b9f0e2ed762e1592082ea6136fa.jpg](../icml_results/5111_OpenDeception_%20Learning%20Deception%20and%20Trust%20in%20Human%E2%80%93AI%20Interaction%20via%20Multi-Agent%20Simulation/images/0230618c69c7176c43f1740ad798de60d0292b9f0e2ed762e1592082ea6136fa.jpg)

![8f23183ebb141ad98d1839ebd89a55fa54f702b2d70fb1e7d3c7d3e2d1a303ac.jpg](../icml_results/5111_OpenDeception_%20Learning%20Deception%20and%20Trust%20in%20Human%E2%80%93AI%20Interaction%20via%20Multi-Agent%20Simulation/images/8f23183ebb141ad98d1839ebd89a55fa54f702b2d70fb1e7d3c7d3e2d1a303ac.jpg)

![99e51c6619e39f65e1db245e8d29d5896c3f7892b9c4924c21f4bfcc76d4eced.jpg](../icml_results/5111_OpenDeception_%20Learning%20Deception%20and%20Trust%20in%20Human%E2%80%93AI%20Interaction%20via%20Multi-Agent%20Simulation/images/99e51c6619e39f65e1db245e8d29d5896c3f7892b9c4924c21f4bfcc76d4eced.jpg)

![bc18b6c54959869827069bd168b3cce848c93f7289f01869c4e35601a22c7f2f.jpg](../icml_results/5111_OpenDeception_%20Learning%20Deception%20and%20Trust%20in%20Human%E2%80%93AI%20Interaction%20via%20Multi-Agent%20Simulation/images/bc18b6c54959869827069bd168b3cce848c93f7289f01869c4e35601a22c7f2f.jpg)

![c5807b858359ab35191b3e8fa6e47ba90757c08b331d0303eb71522792192162.jpg](../icml_results/5111_OpenDeception_%20Learning%20Deception%20and%20Trust%20in%20Human%E2%80%93AI%20Interaction%20via%20Multi-Agent%20Simulation/images/c5807b858359ab35191b3e8fa6e47ba90757c08b331d0303eb71522792192162.jpg)

![d1ab416d9eefe38556bac7ce5292e3392d7d272b386e71245d9a6d7e43730430.jpg](../icml_results/5111_OpenDeception_%20Learning%20Deception%20and%20Trust%20in%20Human%E2%80%93AI%20Interaction%20via%20Multi-Agent%20Simulation/images/d1ab416d9eefe38556bac7ce5292e3392d7d272b386e71245d9a6d7e43730430.jpg)

![d8e3ca6c22278f27ab62595d155413d316782eba3b45fde332f5dd85cd9bdc00.jpg](../icml_results/5111_OpenDeception_%20Learning%20Deception%20and%20Trust%20in%20Human%E2%80%93AI%20Interaction%20via%20Multi-Agent%20Simulation/images/d8e3ca6c22278f27ab62595d155413d316782eba3b45fde332f5dd85cd9bdc00.jpg)

![f736cc78139a9a644751d62f88131afe680dd54d01d2c96a09b91d38d82be38a.jpg](../icml_results/5111_OpenDeception_%20Learning%20Deception%20and%20Trust%20in%20Human%E2%80%93AI%20Interaction%20via%20Multi-Agent%20Simulation/images/f736cc78139a9a644751d62f88131afe680dd54d01d2c96a09b91d38d82be38a.jpg)

### Tables

![207ba4c2a5aa32a2b103170e4d40c2b6c5b4de74c244a5237b7d68d78d41919d.jpg](../icml_results/5111_OpenDeception_%20Learning%20Deception%20and%20Trust%20in%20Human%E2%80%93AI%20Interaction%20via%20Multi-Agent%20Simulation/tables/207ba4c2a5aa32a2b103170e4d40c2b6c5b4de74c244a5237b7d68d78d41919d.jpg)

![2cc3bf0afa9a4a883f623b1f91575eb73e729e161199d41f13b64a95bc30e786.jpg](../icml_results/5111_OpenDeception_%20Learning%20Deception%20and%20Trust%20in%20Human%E2%80%93AI%20Interaction%20via%20Multi-Agent%20Simulation/tables/2cc3bf0afa9a4a883f623b1f91575eb73e729e161199d41f13b64a95bc30e786.jpg)

![45cec73f54ce8aeb5d2feb824298ece4093b623fa56bbd75a4463d3b51c9fe3e.jpg](../icml_results/5111_OpenDeception_%20Learning%20Deception%20and%20Trust%20in%20Human%E2%80%93AI%20Interaction%20via%20Multi-Agent%20Simulation/tables/45cec73f54ce8aeb5d2feb824298ece4093b623fa56bbd75a4463d3b51c9fe3e.jpg)

![6566caec48200cafd617441f212b81faf7d0dc9ed7f6a237ff96fdf6fa8ae7ee.jpg](../icml_results/5111_OpenDeception_%20Learning%20Deception%20and%20Trust%20in%20Human%E2%80%93AI%20Interaction%20via%20Multi-Agent%20Simulation/tables/6566caec48200cafd617441f212b81faf7d0dc9ed7f6a237ff96fdf6fa8ae7ee.jpg)

![901b14101b11e79639d545ab2d0c808950db82f39595a4faec4f01e11e5d742e.jpg](../icml_results/5111_OpenDeception_%20Learning%20Deception%20and%20Trust%20in%20Human%E2%80%93AI%20Interaction%20via%20Multi-Agent%20Simulation/tables/901b14101b11e79639d545ab2d0c808950db82f39595a4faec4f01e11e5d742e.jpg)

![dcb47c79a2bf43f2397bcde29a9b9ef86d83359947971a0de33baae4decc49e1.jpg](../icml_results/5111_OpenDeception_%20Learning%20Deception%20and%20Trust%20in%20Human%E2%80%93AI%20Interaction%20via%20Multi-Agent%20Simulation/tables/dcb47c79a2bf43f2397bcde29a9b9ef86d83359947971a0de33baae4decc49e1.jpg)

![e14b388bb0c2c9e92a226bbd3ebb900944835d206351d1ae7c65516a012f0f40.jpg](../icml_results/5111_OpenDeception_%20Learning%20Deception%20and%20Trust%20in%20Human%E2%80%93AI%20Interaction%20via%20Multi-Agent%20Simulation/tables/e14b388bb0c2c9e92a226bbd3ebb900944835d206351d1ae7c65516a012f0f40.jpg)

![e8c2760e0a8fccac6f2f0a4a251a0b1a4c1f8d4a21c1879af392ba30066c856c.jpg](../icml_results/5111_OpenDeception_%20Learning%20Deception%20and%20Trust%20in%20Human%E2%80%93AI%20Interaction%20via%20Multi-Agent%20Simulation/tables/e8c2760e0a8fccac6f2f0a4a251a0b1a4c1f8d4a21c1879af392ba30066c856c.jpg)

<a id="paper-5064"></a>

## SGMD: Score Gradient Matching Distillation for Few-Step Video Diffusion Distillation


### Images

![1bb8aae83d4bca571c02cafa3a0af219470f30ead7c908e9ca994dba87d2fe18.jpg](../icml_results/5112_SGMD_%20Score%20Gradient%20Matching%20Distillation%20for%20Few-Step%20Video%20Diffusion%20Distillation/images/1bb8aae83d4bca571c02cafa3a0af219470f30ead7c908e9ca994dba87d2fe18.jpg)

![4f76b3135bc91c655d50201b21d8ceb45722006135db72c5b0ca0a0b3cb2f73f.jpg](../icml_results/5112_SGMD_%20Score%20Gradient%20Matching%20Distillation%20for%20Few-Step%20Video%20Diffusion%20Distillation/images/4f76b3135bc91c655d50201b21d8ceb45722006135db72c5b0ca0a0b3cb2f73f.jpg)

![705e90e0c7b3d653ef0a36711a307126cc5f6dd9be3f2092c977eccfcdacc00e.jpg](../icml_results/5112_SGMD_%20Score%20Gradient%20Matching%20Distillation%20for%20Few-Step%20Video%20Diffusion%20Distillation/images/705e90e0c7b3d653ef0a36711a307126cc5f6dd9be3f2092c977eccfcdacc00e.jpg)

![a54750e214345f9899bf731bfeb62cc0ca400de7b1f4ba93b09a0018b2a5cb5b.jpg](../icml_results/5112_SGMD_%20Score%20Gradient%20Matching%20Distillation%20for%20Few-Step%20Video%20Diffusion%20Distillation/images/a54750e214345f9899bf731bfeb62cc0ca400de7b1f4ba93b09a0018b2a5cb5b.jpg)

![d9f6e9eeda55f75404f08db2fae0e6fce3dc2b93a8c9e0ba5ed89f7a925aafb7.jpg](../icml_results/5112_SGMD_%20Score%20Gradient%20Matching%20Distillation%20for%20Few-Step%20Video%20Diffusion%20Distillation/images/d9f6e9eeda55f75404f08db2fae0e6fce3dc2b93a8c9e0ba5ed89f7a925aafb7.jpg)

### Tables

![2fd82dbd7d5ff054a7e550c65479741b508e03e38241a5d4c865a316595dd1e4.jpg](../icml_results/5112_SGMD_%20Score%20Gradient%20Matching%20Distillation%20for%20Few-Step%20Video%20Diffusion%20Distillation/tables/2fd82dbd7d5ff054a7e550c65479741b508e03e38241a5d4c865a316595dd1e4.jpg)

![34277c49937b294df91d8ead0a122aed0df837b6eb81d01a0c74d824246c3fd5.jpg](../icml_results/5112_SGMD_%20Score%20Gradient%20Matching%20Distillation%20for%20Few-Step%20Video%20Diffusion%20Distillation/tables/34277c49937b294df91d8ead0a122aed0df837b6eb81d01a0c74d824246c3fd5.jpg)

![4c9cc80b233f1cc41e3cbf088201f70319f2269bd27f816793a206482c40f7d2.jpg](../icml_results/5112_SGMD_%20Score%20Gradient%20Matching%20Distillation%20for%20Few-Step%20Video%20Diffusion%20Distillation/tables/4c9cc80b233f1cc41e3cbf088201f70319f2269bd27f816793a206482c40f7d2.jpg)

![95b6b45ab51bebe46bccf972b8430f72703955f288bb4981bac088e2105cff98.jpg](../icml_results/5112_SGMD_%20Score%20Gradient%20Matching%20Distillation%20for%20Few-Step%20Video%20Diffusion%20Distillation/tables/95b6b45ab51bebe46bccf972b8430f72703955f288bb4981bac088e2105cff98.jpg)

![9954d217f9357a0449250003282e0abbf66ec34196f320a4e21cebc6946c8d61.jpg](../icml_results/5112_SGMD_%20Score%20Gradient%20Matching%20Distillation%20for%20Few-Step%20Video%20Diffusion%20Distillation/tables/9954d217f9357a0449250003282e0abbf66ec34196f320a4e21cebc6946c8d61.jpg)

![bd23890eb3455c09b5b9a5757fa962787d8387962f6f358ec769c1bc4b27535d.jpg](../icml_results/5112_SGMD_%20Score%20Gradient%20Matching%20Distillation%20for%20Few-Step%20Video%20Diffusion%20Distillation/tables/bd23890eb3455c09b5b9a5757fa962787d8387962f6f358ec769c1bc4b27535d.jpg)

<a id="paper-5065"></a>

## Linear Ensembles Wash Away Watermarks: On the Fragility of Distributional Perturbations in LLMs


### Images

![07c9b26176bf2ce3c14a80559fd7b82002f829aba6dcc8b2ef05d77b7184b8cd.jpg](../icml_results/5113_Linear%20Ensembles%20Wash%20Away%20Watermarks_%20On%20the%20Fragility%20of%20Distributional%20Perturbations%20in%20LLMs/images/07c9b26176bf2ce3c14a80559fd7b82002f829aba6dcc8b2ef05d77b7184b8cd.jpg)

![47185222298fa3ee6b1124a39f4a2f021158ce211d304bcb704bc991ddfdfecf.jpg](../icml_results/5113_Linear%20Ensembles%20Wash%20Away%20Watermarks_%20On%20the%20Fragility%20of%20Distributional%20Perturbations%20in%20LLMs/images/47185222298fa3ee6b1124a39f4a2f021158ce211d304bcb704bc991ddfdfecf.jpg)

![5cadb489a91005418b9d70ebbdcbde2d7d9e09816ad3d6e55ba97fd1d1d86ed1.jpg](../icml_results/5113_Linear%20Ensembles%20Wash%20Away%20Watermarks_%20On%20the%20Fragility%20of%20Distributional%20Perturbations%20in%20LLMs/images/5cadb489a91005418b9d70ebbdcbde2d7d9e09816ad3d6e55ba97fd1d1d86ed1.jpg)

### Tables

![07c2ef0bcde0b8ea0e04ab15954bac90cfd7e04078a1d99fde7b822c131cb706.jpg](../icml_results/5113_Linear%20Ensembles%20Wash%20Away%20Watermarks_%20On%20the%20Fragility%20of%20Distributional%20Perturbations%20in%20LLMs/tables/07c2ef0bcde0b8ea0e04ab15954bac90cfd7e04078a1d99fde7b822c131cb706.jpg)

![0fd517a6a44bad9a79d6ccb82c88546ad8b01da95fba502b6657c73f9e50b43d.jpg](../icml_results/5113_Linear%20Ensembles%20Wash%20Away%20Watermarks_%20On%20the%20Fragility%20of%20Distributional%20Perturbations%20in%20LLMs/tables/0fd517a6a44bad9a79d6ccb82c88546ad8b01da95fba502b6657c73f9e50b43d.jpg)

![317293711746a9503d055b3ba5a68ab990e943ca740dba45542fd64ac82ea5ae.jpg](../icml_results/5113_Linear%20Ensembles%20Wash%20Away%20Watermarks_%20On%20the%20Fragility%20of%20Distributional%20Perturbations%20in%20LLMs/tables/317293711746a9503d055b3ba5a68ab990e943ca740dba45542fd64ac82ea5ae.jpg)

![43acc97ec772560f91641e7441ccce1dac20d51550f2d43ade40e363480524ea.jpg](../icml_results/5113_Linear%20Ensembles%20Wash%20Away%20Watermarks_%20On%20the%20Fragility%20of%20Distributional%20Perturbations%20in%20LLMs/tables/43acc97ec772560f91641e7441ccce1dac20d51550f2d43ade40e363480524ea.jpg)

![5ee9e3576034225a09e0f01e80bcc03a12fa806f762562c9ce4b463980f46d4e.jpg](../icml_results/5113_Linear%20Ensembles%20Wash%20Away%20Watermarks_%20On%20the%20Fragility%20of%20Distributional%20Perturbations%20in%20LLMs/tables/5ee9e3576034225a09e0f01e80bcc03a12fa806f762562c9ce4b463980f46d4e.jpg)

![a66c35c6851c68f8761fcae5ae4ca68bd97a5d6896343f5abab837b9b4add444.jpg](../icml_results/5113_Linear%20Ensembles%20Wash%20Away%20Watermarks_%20On%20the%20Fragility%20of%20Distributional%20Perturbations%20in%20LLMs/tables/a66c35c6851c68f8761fcae5ae4ca68bd97a5d6896343f5abab837b9b4add444.jpg)

![c520d9ce5ed81f79f002bbc56e7a09f01824c0565276553a20e0a9653b75ab78.jpg](../icml_results/5113_Linear%20Ensembles%20Wash%20Away%20Watermarks_%20On%20the%20Fragility%20of%20Distributional%20Perturbations%20in%20LLMs/tables/c520d9ce5ed81f79f002bbc56e7a09f01824c0565276553a20e0a9653b75ab78.jpg)

![eb35ab23ce33a2e617e3f0ed2338405a2ac83d0f3e7280f17158521fb84166a9.jpg](../icml_results/5113_Linear%20Ensembles%20Wash%20Away%20Watermarks_%20On%20the%20Fragility%20of%20Distributional%20Perturbations%20in%20LLMs/tables/eb35ab23ce33a2e617e3f0ed2338405a2ac83d0f3e7280f17158521fb84166a9.jpg)

![f6433c8a0da9a57cbf14146758ff1e4498fb737894f072c315e6b8477412e90a.jpg](../icml_results/5113_Linear%20Ensembles%20Wash%20Away%20Watermarks_%20On%20the%20Fragility%20of%20Distributional%20Perturbations%20in%20LLMs/tables/f6433c8a0da9a57cbf14146758ff1e4498fb737894f072c315e6b8477412e90a.jpg)

<a id="paper-5066"></a>

## GeoPT: Scaling Physics Simulation via Lifted Geometric Pre-Training


### Images

![02290de6ec4a3a4aadcdba1d1a41c17666579e27711e3364a92d995047294dc9.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/02290de6ec4a3a4aadcdba1d1a41c17666579e27711e3364a92d995047294dc9.jpg)

![085a4387c2ceaff4b57921c8a311744028eceb1eb0e910ab0635499d568c9048.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/085a4387c2ceaff4b57921c8a311744028eceb1eb0e910ab0635499d568c9048.jpg)

![0a16e7328565ff2ec0e57ce0f663a42c94caa39751090b1733a53ac61fbe9a54.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/0a16e7328565ff2ec0e57ce0f663a42c94caa39751090b1733a53ac61fbe9a54.jpg)

![161e4b330831da9ddc68dfeb2f0eaf03789ea48f32ea090ea1416f73c94282f8.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/161e4b330831da9ddc68dfeb2f0eaf03789ea48f32ea090ea1416f73c94282f8.jpg)

![27fb7186dd197d1a7733ea62516869095b054563a3f40b12f3bf1482d740aa8b.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/27fb7186dd197d1a7733ea62516869095b054563a3f40b12f3bf1482d740aa8b.jpg)

![30e90132e56f37b8bc1e4dc4e58c3b9e6c350133a188ccfee4d8f3e1ae539efb.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/30e90132e56f37b8bc1e4dc4e58c3b9e6c350133a188ccfee4d8f3e1ae539efb.jpg)

![35a417174e09446629f973a9f2968b335f7ec572faa32317a3f114e28e603dbc.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/35a417174e09446629f973a9f2968b335f7ec572faa32317a3f114e28e603dbc.jpg)

![3c96603049cbc12e5569bdd417f5dc7bf1ed8ee863c9ccad23762d8b8580b609.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/3c96603049cbc12e5569bdd417f5dc7bf1ed8ee863c9ccad23762d8b8580b609.jpg)

![42a641a2d0194c773a83d04a92cfe99664a38d38db9ee5e8b939a551191313b7.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/42a641a2d0194c773a83d04a92cfe99664a38d38db9ee5e8b939a551191313b7.jpg)

![6c0464857db88df598dad8c63af99d1d230d9852c6b55a7818dccf19ffb4e1b4.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/6c0464857db88df598dad8c63af99d1d230d9852c6b55a7818dccf19ffb4e1b4.jpg)

![7307b71943a677e01971e4e5703bdc4050046743186c8f31f984e6139ba2e1ce.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/7307b71943a677e01971e4e5703bdc4050046743186c8f31f984e6139ba2e1ce.jpg)

![742539121c85038b1938a586cd0f21458eb601c1cf461c0e7892f3d29bd895ea.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/742539121c85038b1938a586cd0f21458eb601c1cf461c0e7892f3d29bd895ea.jpg)

![835b5ac7bdf3d00304a4d5390366fb78a92522f50943bb425626f00f8b900aa8.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/835b5ac7bdf3d00304a4d5390366fb78a92522f50943bb425626f00f8b900aa8.jpg)

![8650b3948765fa025e96d6b892e9056947a1696621315079dfade125217171e8.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/8650b3948765fa025e96d6b892e9056947a1696621315079dfade125217171e8.jpg)

![8e165048065e266a7b47f981f1c94dcb3c8ea5b4450d7a65608e93211c543470.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/8e165048065e266a7b47f981f1c94dcb3c8ea5b4450d7a65608e93211c543470.jpg)

![9bb3bc9f84b170dcd8d76d632d07e8eb54e7812b9da37b8b790c02447446d005.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/9bb3bc9f84b170dcd8d76d632d07e8eb54e7812b9da37b8b790c02447446d005.jpg)

![9cf3b6716f437e7aa02f3a1f03207332456c856b4e83140f07ef381ce56fc3eb.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/9cf3b6716f437e7aa02f3a1f03207332456c856b4e83140f07ef381ce56fc3eb.jpg)

![a3ab4f90fe504f6e714abfab67c723669bcf665f54cbe565812757517f2bc582.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/a3ab4f90fe504f6e714abfab67c723669bcf665f54cbe565812757517f2bc582.jpg)

![b5b323506a7ca98e06120df06b1811460f8c3ef836fb543e1dee3b2863ef99b8.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/b5b323506a7ca98e06120df06b1811460f8c3ef836fb543e1dee3b2863ef99b8.jpg)

![ceef1d071495c217f449453d7b57194c732448238498dc76b42bfdfe5ec04cca.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/ceef1d071495c217f449453d7b57194c732448238498dc76b42bfdfe5ec04cca.jpg)

![de26af929d68065013b14d81ad54b824d26d511731e2227d762df8f7637542d9.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/de26af929d68065013b14d81ad54b824d26d511731e2227d762df8f7637542d9.jpg)

![e100df255789492417be4335098bb48cdd1ce6925b49a90adfd283a437ad4272.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/e100df255789492417be4335098bb48cdd1ce6925b49a90adfd283a437ad4272.jpg)

![e4f567dd643edc82dd1d5cc8d24380893e05651e0399c81619b13e37390401cc.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/e4f567dd643edc82dd1d5cc8d24380893e05651e0399c81619b13e37390401cc.jpg)

![e9d190cd43b31440b8ac72e430e1c49f918a1dc45a7d37e1ecbf92b33b997d73.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/e9d190cd43b31440b8ac72e430e1c49f918a1dc45a7d37e1ecbf92b33b997d73.jpg)

![f0d252ed76a3edc2c12bdf79f86f1a2c30e3cdc1b7004d6e72218aaea75d074a.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/f0d252ed76a3edc2c12bdf79f86f1a2c30e3cdc1b7004d6e72218aaea75d074a.jpg)

![f1e59cc23669d16d315c6774db2e14452ede2cad2312446b7078c3c87d01eb98.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/f1e59cc23669d16d315c6774db2e14452ede2cad2312446b7078c3c87d01eb98.jpg)

![fdfd54ed05450c9fd4cdb19d5c97e169aeb670cc57673e7aa962ea9c641a6c2f.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/images/fdfd54ed05450c9fd4cdb19d5c97e169aeb670cc57673e7aa962ea9c641a6c2f.jpg)

### Tables

![22c6175436a5bf42fb7b3167b540a16d7b907244aea1617d32c641d2c2dbf2d2.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/22c6175436a5bf42fb7b3167b540a16d7b907244aea1617d32c641d2c2dbf2d2.jpg)

![2ed7f13e3920e12ae4898b1f231b4c09aa4f230d174754f598e8c3bad5e1759b.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/2ed7f13e3920e12ae4898b1f231b4c09aa4f230d174754f598e8c3bad5e1759b.jpg)

![335d8b1b8a68a44508e3961c753e41274d4cb5b3b0b4ff408fe70d44e839bf90.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/335d8b1b8a68a44508e3961c753e41274d4cb5b3b0b4ff408fe70d44e839bf90.jpg)

![3a13f5997268b88ce8fb6613a9e1d505074cee6f95dd2200e340cd58c249cdd1.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/3a13f5997268b88ce8fb6613a9e1d505074cee6f95dd2200e340cd58c249cdd1.jpg)

![40753eb5e3b920f706696d778f6e8fa5a781a04a02d5af95a4fd79f21b280c5f.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/40753eb5e3b920f706696d778f6e8fa5a781a04a02d5af95a4fd79f21b280c5f.jpg)

![46af219895723a2892f58dfe9dad514be1bdb1fc59735ab365084858952b6f8c.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/46af219895723a2892f58dfe9dad514be1bdb1fc59735ab365084858952b6f8c.jpg)

![47dcf3ac123cfaf4a3b6a9b370faad7b74be803384197777b9ad05beeefbd15e.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/47dcf3ac123cfaf4a3b6a9b370faad7b74be803384197777b9ad05beeefbd15e.jpg)

![482a3b18d4f6804cbc7d923fd5306e422552fc5719bd723b283c38cd1332973b.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/482a3b18d4f6804cbc7d923fd5306e422552fc5719bd723b283c38cd1332973b.jpg)

![49bd9b02de1302d963b840e7c227bb7dbc505f375009438d1edbd5a90f4b4cac.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/49bd9b02de1302d963b840e7c227bb7dbc505f375009438d1edbd5a90f4b4cac.jpg)

![4d9bf636e13c6e583b883cfa993a5a2be12e5c22dc699dd71404c76649f0ea2c.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/4d9bf636e13c6e583b883cfa993a5a2be12e5c22dc699dd71404c76649f0ea2c.jpg)

![4f7b294ff407b3e6bb7d31adab56399e328d569532ca085fb2dd291582167548.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/4f7b294ff407b3e6bb7d31adab56399e328d569532ca085fb2dd291582167548.jpg)

![59c83c6a5e3968b2b7b67f1a02430aacfa56cbc1c79d1b4a8c372e9aa95c7758.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/59c83c6a5e3968b2b7b67f1a02430aacfa56cbc1c79d1b4a8c372e9aa95c7758.jpg)

![634dab405b73a900967729e6d598d7952eceb414433988fd01ad81041ae55737.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/634dab405b73a900967729e6d598d7952eceb414433988fd01ad81041ae55737.jpg)

![64821be5d88dd0f34400ea4f84f2f739cf44d73ef24895c4e9662ef67786ec76.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/64821be5d88dd0f34400ea4f84f2f739cf44d73ef24895c4e9662ef67786ec76.jpg)

![6dd0f3010f0f7500e4962298608e16634eabd7ff33337176e9c55b0355734c88.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/6dd0f3010f0f7500e4962298608e16634eabd7ff33337176e9c55b0355734c88.jpg)

![76e4bf365260954a13f653cb4b256ce8e4d08c731d85f1845a0336048b7edbf8.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/76e4bf365260954a13f653cb4b256ce8e4d08c731d85f1845a0336048b7edbf8.jpg)

![79ad7f80d14597e3dad38a98c51c48084f3c91540c1dce6dad4cdd46c246da58.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/79ad7f80d14597e3dad38a98c51c48084f3c91540c1dce6dad4cdd46c246da58.jpg)

![7a5c267fff6b3494a5454df7f42d7a77e81b9921ef41901311246b22331669e9.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/7a5c267fff6b3494a5454df7f42d7a77e81b9921ef41901311246b22331669e9.jpg)

![9886a198d51bc58a59ed4cd523ee101629abf443440e12edbb4cf31e9c3537b8.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/9886a198d51bc58a59ed4cd523ee101629abf443440e12edbb4cf31e9c3537b8.jpg)

![9e34b7c3a0a768bcbc3bfc53fa96cbdef0055978d1b8c8e723d05192de365381.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/9e34b7c3a0a768bcbc3bfc53fa96cbdef0055978d1b8c8e723d05192de365381.jpg)

![af53db4027f2db44cb7d342b95a0487b0d1696de20508e9a9e7fdc24e87eb71d.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/af53db4027f2db44cb7d342b95a0487b0d1696de20508e9a9e7fdc24e87eb71d.jpg)

![b834b24d631f44f3a39e624f9cdae35908fe6ded1d00fb477e3942d1c4480459.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/b834b24d631f44f3a39e624f9cdae35908fe6ded1d00fb477e3942d1c4480459.jpg)

![cee41830cf42c6be1b1abe623f8c139507af2b00dc90e77c587163750448544d.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/cee41830cf42c6be1b1abe623f8c139507af2b00dc90e77c587163750448544d.jpg)

![d081205296b343fee142a55bb80e00ef468d7b9cf3935714d42cdcf4ee7b5501.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/d081205296b343fee142a55bb80e00ef468d7b9cf3935714d42cdcf4ee7b5501.jpg)

![d9f33107e4e6350a8e4c4475477df20abd322e3324dacd93bb2e0d5dbdf883d8.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/d9f33107e4e6350a8e4c4475477df20abd322e3324dacd93bb2e0d5dbdf883d8.jpg)

![da978f0445db8f5d1423594121b8fa17852cad91e8e23b3da17023e55ea5f14e.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/da978f0445db8f5d1423594121b8fa17852cad91e8e23b3da17023e55ea5f14e.jpg)

![e0286d5b4de9db7dde27a3fdd4bfc3db2edff2704d771986662609a02efd331c.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/e0286d5b4de9db7dde27a3fdd4bfc3db2edff2704d771986662609a02efd331c.jpg)

![e245c3eebbf71cc50e76a641704502a6c7c442478b046e2e18200731b2724734.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/e245c3eebbf71cc50e76a641704502a6c7c442478b046e2e18200731b2724734.jpg)

![ec6e033fffbaa335518d177203701531bb19151ebdbba44ea56b2b4ab97e5b53.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/ec6e033fffbaa335518d177203701531bb19151ebdbba44ea56b2b4ab97e5b53.jpg)

![ee1b7f76ad51ddc4f235227957ef513b6b2480511d3623b5e108fab70d03ad3a.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/ee1b7f76ad51ddc4f235227957ef513b6b2480511d3623b5e108fab70d03ad3a.jpg)

![fa707edc672172ebe187809c7f78f8c1215dbe637f21c42af1cada8d33edc8db.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/fa707edc672172ebe187809c7f78f8c1215dbe637f21c42af1cada8d33edc8db.jpg)

![fe6e1d665396c9c4fc95f9ee77afb80d7be4c25e9478e115f05d1760796e0d3d.jpg](../icml_results/5114_GeoPT_%20Scaling%20Physics%20Simulation%20via%20Lifted%20Geometric%20Pre-Training/tables/fe6e1d665396c9c4fc95f9ee77afb80d7be4c25e9478e115f05d1760796e0d3d.jpg)

<a id="paper-5067"></a>

## Selecting Samples on Graphs: A Unified Dataset Pruning Framework for Lossless Training Acceleration


### Images

![889d29ceea111ccdc91beb0dd2f92f4c1032c8e143180c79f8b5a1be7f29bff8.jpg](../icml_results/5115_Selecting%20Samples%20on%20Graphs_%20A%20Unified%20Dataset%20Pruning%20Framework%20for%20Lossless%20Training%20Acceleration/images/889d29ceea111ccdc91beb0dd2f92f4c1032c8e143180c79f8b5a1be7f29bff8.jpg)

![d3c8a10294a2ac2ff46f2c262c6d68c4abe22e11aa0b35fc7c08a4d06e86aa3f.jpg](../icml_results/5115_Selecting%20Samples%20on%20Graphs_%20A%20Unified%20Dataset%20Pruning%20Framework%20for%20Lossless%20Training%20Acceleration/images/d3c8a10294a2ac2ff46f2c262c6d68c4abe22e11aa0b35fc7c08a4d06e86aa3f.jpg)

![e8767bc4624700ceee82081dbfa391f94a335704afbe240bac7828a2a705bd93.jpg](../icml_results/5115_Selecting%20Samples%20on%20Graphs_%20A%20Unified%20Dataset%20Pruning%20Framework%20for%20Lossless%20Training%20Acceleration/images/e8767bc4624700ceee82081dbfa391f94a335704afbe240bac7828a2a705bd93.jpg)

![fa320847b7c56b99f09d9e2e15bf776c45365eb6f8f4732f64313b806271c109.jpg](../icml_results/5115_Selecting%20Samples%20on%20Graphs_%20A%20Unified%20Dataset%20Pruning%20Framework%20for%20Lossless%20Training%20Acceleration/images/fa320847b7c56b99f09d9e2e15bf776c45365eb6f8f4732f64313b806271c109.jpg)

### Tables

![26443395b062274be197cb415550b4633dc10685dd973976c10408c2141661ce.jpg](../icml_results/5115_Selecting%20Samples%20on%20Graphs_%20A%20Unified%20Dataset%20Pruning%20Framework%20for%20Lossless%20Training%20Acceleration/tables/26443395b062274be197cb415550b4633dc10685dd973976c10408c2141661ce.jpg)

![2af15d0b2fd3e72da16a5d96e372335b23ac60c0df69d11dd12d8fe540c56843.jpg](../icml_results/5115_Selecting%20Samples%20on%20Graphs_%20A%20Unified%20Dataset%20Pruning%20Framework%20for%20Lossless%20Training%20Acceleration/tables/2af15d0b2fd3e72da16a5d96e372335b23ac60c0df69d11dd12d8fe540c56843.jpg)

![51de5d2c7893bd4b8ba6e56985abe73bad518e822a3dc0081596d0cc9efa6088.jpg](../icml_results/5115_Selecting%20Samples%20on%20Graphs_%20A%20Unified%20Dataset%20Pruning%20Framework%20for%20Lossless%20Training%20Acceleration/tables/51de5d2c7893bd4b8ba6e56985abe73bad518e822a3dc0081596d0cc9efa6088.jpg)

![627e2dc0a447b6c28233ba9053954ba6e545b337e9c30521815e0bcf4d661d25.jpg](../icml_results/5115_Selecting%20Samples%20on%20Graphs_%20A%20Unified%20Dataset%20Pruning%20Framework%20for%20Lossless%20Training%20Acceleration/tables/627e2dc0a447b6c28233ba9053954ba6e545b337e9c30521815e0bcf4d661d25.jpg)

![71abefe5be690cf7b462554f93cfda43e3b047274e8645eb05473155e8637c4b.jpg](../icml_results/5115_Selecting%20Samples%20on%20Graphs_%20A%20Unified%20Dataset%20Pruning%20Framework%20for%20Lossless%20Training%20Acceleration/tables/71abefe5be690cf7b462554f93cfda43e3b047274e8645eb05473155e8637c4b.jpg)

![8024ac5b7eae9b0532e2f0396c2c8f50fee90ee1e68d4169db2f81e4cd354d54.jpg](../icml_results/5115_Selecting%20Samples%20on%20Graphs_%20A%20Unified%20Dataset%20Pruning%20Framework%20for%20Lossless%20Training%20Acceleration/tables/8024ac5b7eae9b0532e2f0396c2c8f50fee90ee1e68d4169db2f81e4cd354d54.jpg)

![a234d984e3f8a9519b4718f4aec0d4f2356888206158e921bdd619ab73c63a83.jpg](../icml_results/5115_Selecting%20Samples%20on%20Graphs_%20A%20Unified%20Dataset%20Pruning%20Framework%20for%20Lossless%20Training%20Acceleration/tables/a234d984e3f8a9519b4718f4aec0d4f2356888206158e921bdd619ab73c63a83.jpg)

![dd3d6c084a256f0decb4581c9d28631408de88324418609636ff0f76427f0d6e.jpg](../icml_results/5115_Selecting%20Samples%20on%20Graphs_%20A%20Unified%20Dataset%20Pruning%20Framework%20for%20Lossless%20Training%20Acceleration/tables/dd3d6c084a256f0decb4581c9d28631408de88324418609636ff0f76427f0d6e.jpg)

![f4c431c4c0716acd1b9d6a304cc0ecc7e113ef0656065ff087e98b65a13e35e8.jpg](../icml_results/5115_Selecting%20Samples%20on%20Graphs_%20A%20Unified%20Dataset%20Pruning%20Framework%20for%20Lossless%20Training%20Acceleration/tables/f4c431c4c0716acd1b9d6a304cc0ecc7e113ef0656065ff087e98b65a13e35e8.jpg)

![f8e91667484b4fe15174d34480dee27196e22b19315816cb64e0a350c157e94a.jpg](../icml_results/5115_Selecting%20Samples%20on%20Graphs_%20A%20Unified%20Dataset%20Pruning%20Framework%20for%20Lossless%20Training%20Acceleration/tables/f8e91667484b4fe15174d34480dee27196e22b19315816cb64e0a350c157e94a.jpg)

![faa7640ddce09b249abb58075bb41facae95a685bfc9c17bb52d8a369a1586a2.jpg](../icml_results/5115_Selecting%20Samples%20on%20Graphs_%20A%20Unified%20Dataset%20Pruning%20Framework%20for%20Lossless%20Training%20Acceleration/tables/faa7640ddce09b249abb58075bb41facae95a685bfc9c17bb52d8a369a1586a2.jpg)

<a id="paper-5068"></a>

## An Empirical Study on the Resilience of Partial Merging to Model Clone Attacks


### Images

![1b64d558cda581da5f5fc2e385f131ed84a5c9c932972f89b1b62984dc5a1a9d.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/images/1b64d558cda581da5f5fc2e385f131ed84a5c9c932972f89b1b62984dc5a1a9d.jpg)

![303d5837c060397a4bbc69ec44939cc4f5c6cb53567f867f0ac46549e4ffccc3.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/images/303d5837c060397a4bbc69ec44939cc4f5c6cb53567f867f0ac46549e4ffccc3.jpg)

![315a7a96338085b97a99dde36d2298185e1ca5aea4ce87fd982555661dcead4d.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/images/315a7a96338085b97a99dde36d2298185e1ca5aea4ce87fd982555661dcead4d.jpg)

![40e3478dab9c15e420fa388504ff51c1583066e74d38424fb78c86e00ebab998.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/images/40e3478dab9c15e420fa388504ff51c1583066e74d38424fb78c86e00ebab998.jpg)

![4315d947691afba91f9c8dd728132b0a067e56f07afe56230a4209408ceba653.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/images/4315d947691afba91f9c8dd728132b0a067e56f07afe56230a4209408ceba653.jpg)

![48dc973c92babfbdb6709c70d75ff1ae6c42879d6a7e9d3aac9ce4f9d5c93adf.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/images/48dc973c92babfbdb6709c70d75ff1ae6c42879d6a7e9d3aac9ce4f9d5c93adf.jpg)

![7224b0025542bb2f1f45014f25ab9dde5cd6aaf3a6f93c7020fa4d50b71c56ba.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/images/7224b0025542bb2f1f45014f25ab9dde5cd6aaf3a6f93c7020fa4d50b71c56ba.jpg)

![77429e7e8e0d4979306e86d651bebdff45bc53d758604ce1137f837c9c8eb753.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/images/77429e7e8e0d4979306e86d651bebdff45bc53d758604ce1137f837c9c8eb753.jpg)

![788588e5c4504e4f117af1184245d0a177ff306162a178e584b4d2e1840eba36.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/images/788588e5c4504e4f117af1184245d0a177ff306162a178e584b4d2e1840eba36.jpg)

![b5cc140962e7cceb76028bb8e423685adba193f9eba6ece7d06d105778d9a942.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/images/b5cc140962e7cceb76028bb8e423685adba193f9eba6ece7d06d105778d9a942.jpg)

![bfe74c005b7c5fad28e0ad0b005a1c5ee7b921f3d01f441e39ad33c275201b35.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/images/bfe74c005b7c5fad28e0ad0b005a1c5ee7b921f3d01f441e39ad33c275201b35.jpg)

![e0db4f4bde52d467d44a195c86d2eb641da45ae0aa5bf51be3705eb59a2ac57e.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/images/e0db4f4bde52d467d44a195c86d2eb641da45ae0aa5bf51be3705eb59a2ac57e.jpg)

![fd983024c230ef601cdf1de2faf25bcb594bf1defc6e0592b757eabfcf675f23.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/images/fd983024c230ef601cdf1de2faf25bcb594bf1defc6e0592b757eabfcf675f23.jpg)

### Tables

![263cbebd205dc1f979cc3cdfafdbe8b8a88e6484ee8db957964dacf0e2167d2b.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/tables/263cbebd205dc1f979cc3cdfafdbe8b8a88e6484ee8db957964dacf0e2167d2b.jpg)

![85482ccf48b3210652235e00491655ffb7ec842e8d70d58a851544d151d58768.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/tables/85482ccf48b3210652235e00491655ffb7ec842e8d70d58a851544d151d58768.jpg)

![999416baca6a0fa8f6ff92c31fe20fd94da6b87157fc6daf1f9dc1e3a8a549a4.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/tables/999416baca6a0fa8f6ff92c31fe20fd94da6b87157fc6daf1f9dc1e3a8a549a4.jpg)

![d9b9f1a1964f4542deb3d06a6bbabc15072e0bfdedee9d97aa9f18eb53fd751b.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/tables/d9b9f1a1964f4542deb3d06a6bbabc15072e0bfdedee9d97aa9f18eb53fd751b.jpg)

![db9384be8b165370abde4e5c0b5939adc97977e596072d81a9994b193a918e26.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/tables/db9384be8b165370abde4e5c0b5939adc97977e596072d81a9994b193a918e26.jpg)

![f3e6b276e46fe3c76c7399c28f1915b208ce66a956072d1cb5d57eeb41eadb1a.jpg](../icml_results/5116_An%20Empirical%20Study%20on%20the%20Resilience%20of%20Partial%20Merging%20to%20Model%20Clone%20Attacks/tables/f3e6b276e46fe3c76c7399c28f1915b208ce66a956072d1cb5d57eeb41eadb1a.jpg)

<a id="paper-5069"></a>

## Infinite-World: Scaling Interactive World Models to 1000-Frame Horizons via Pose-Free Hierarchical Memory


### Images

![0fbb7dcadd3d1bd976b530eabdc2be92881660ff7e3b6dba982f805d751e7387.jpg](../icml_results/5117_Infinite-World_%20Scaling%20Interactive%20World%20Models%20to%201000-Frame%20Horizons%20via%20Pose-Free%20Hierarchical%20M/images/0fbb7dcadd3d1bd976b530eabdc2be92881660ff7e3b6dba982f805d751e7387.jpg)

![16870bf6a715c5ade3a03e52a261e6e48007cb1c3a3aba3152f12a7d578d9999.jpg](../icml_results/5117_Infinite-World_%20Scaling%20Interactive%20World%20Models%20to%201000-Frame%20Horizons%20via%20Pose-Free%20Hierarchical%20M/images/16870bf6a715c5ade3a03e52a261e6e48007cb1c3a3aba3152f12a7d578d9999.jpg)

![3da455b0f8045d472353c29019091ddce99e989d1a9f00b635415c90d913029d.jpg](../icml_results/5117_Infinite-World_%20Scaling%20Interactive%20World%20Models%20to%201000-Frame%20Horizons%20via%20Pose-Free%20Hierarchical%20M/images/3da455b0f8045d472353c29019091ddce99e989d1a9f00b635415c90d913029d.jpg)

![46930b712839a57a0e01c337adc0beb0576c41543696345135465e9ba4b950de.jpg](../icml_results/5117_Infinite-World_%20Scaling%20Interactive%20World%20Models%20to%201000-Frame%20Horizons%20via%20Pose-Free%20Hierarchical%20M/images/46930b712839a57a0e01c337adc0beb0576c41543696345135465e9ba4b950de.jpg)

![4c7a7ab7fab9192c8c8255f34604fa85c2f290eecdf56d624d066df2dc3cf0a0.jpg](../icml_results/5117_Infinite-World_%20Scaling%20Interactive%20World%20Models%20to%201000-Frame%20Horizons%20via%20Pose-Free%20Hierarchical%20M/images/4c7a7ab7fab9192c8c8255f34604fa85c2f290eecdf56d624d066df2dc3cf0a0.jpg)

![4f488460cfa22db445faeb638186883c399533365b312a54a9d866a0d0ccf948.jpg](../icml_results/5117_Infinite-World_%20Scaling%20Interactive%20World%20Models%20to%201000-Frame%20Horizons%20via%20Pose-Free%20Hierarchical%20M/images/4f488460cfa22db445faeb638186883c399533365b312a54a9d866a0d0ccf948.jpg)

![8bd0c968de8ef4333737362c740dfb982a4f72fa7a6b2344f7732013eed04a6b.jpg](../icml_results/5117_Infinite-World_%20Scaling%20Interactive%20World%20Models%20to%201000-Frame%20Horizons%20via%20Pose-Free%20Hierarchical%20M/images/8bd0c968de8ef4333737362c740dfb982a4f72fa7a6b2344f7732013eed04a6b.jpg)

![ccb8cef76e80d225d0269ade269e3f90ff87eb00b0b73bf1e1cc5542a36abf8a.jpg](../icml_results/5117_Infinite-World_%20Scaling%20Interactive%20World%20Models%20to%201000-Frame%20Horizons%20via%20Pose-Free%20Hierarchical%20M/images/ccb8cef76e80d225d0269ade269e3f90ff87eb00b0b73bf1e1cc5542a36abf8a.jpg)

### Tables

![3eed29bec35c55f85b8dc0f98d21d05d11e0e846b74523b98435e9d7a2cf66f5.jpg](../icml_results/5117_Infinite-World_%20Scaling%20Interactive%20World%20Models%20to%201000-Frame%20Horizons%20via%20Pose-Free%20Hierarchical%20M/tables/3eed29bec35c55f85b8dc0f98d21d05d11e0e846b74523b98435e9d7a2cf66f5.jpg)

![72280263715d959bb1b9e26b1bb89c712ef4f2caa6544aadc2caedb0847f26da.jpg](../icml_results/5117_Infinite-World_%20Scaling%20Interactive%20World%20Models%20to%201000-Frame%20Horizons%20via%20Pose-Free%20Hierarchical%20M/tables/72280263715d959bb1b9e26b1bb89c712ef4f2caa6544aadc2caedb0847f26da.jpg)

<a id="paper-5070"></a>

## Towards Seed-Robust Safety Alignment in Text-to-Image Models


### Images

![4edd508766c0b66485eda2a22e86e105d874b3e29fc41e9f92cc13771eabf01b.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/images/4edd508766c0b66485eda2a22e86e105d874b3e29fc41e9f92cc13771eabf01b.jpg)

![59543e3cb3230b3583ffc521cf2cce2fce3bb3d99a116caab1c7546a901a9823.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/images/59543e3cb3230b3583ffc521cf2cce2fce3bb3d99a116caab1c7546a901a9823.jpg)

![776b2215a3e01bc609f373ab81dc407f5508cd881978c2b604792915f8b93608.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/images/776b2215a3e01bc609f373ab81dc407f5508cd881978c2b604792915f8b93608.jpg)

![849a5d9f5e56640aa34d9aee5337271912b5a3cacb848b14eace7171a6a9baf5.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/images/849a5d9f5e56640aa34d9aee5337271912b5a3cacb848b14eace7171a6a9baf5.jpg)

![8caefa73aa3f0515b7f3c41d401cc2e6808a98f6fce78dd51cc8d594dc358659.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/images/8caefa73aa3f0515b7f3c41d401cc2e6808a98f6fce78dd51cc8d594dc358659.jpg)

![9acf7e33a36fcd80d1cab375591456a1f168be26167aeaeb7db22ead2a238e05.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/images/9acf7e33a36fcd80d1cab375591456a1f168be26167aeaeb7db22ead2a238e05.jpg)

![9e194fd4abb39de351c405cb30aa049b3a899572fc47c4a4131a31833b72f6e8.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/images/9e194fd4abb39de351c405cb30aa049b3a899572fc47c4a4131a31833b72f6e8.jpg)

![b69df3efdf1b988fdefec8884111bf1ce2bb61cac7b79300ffe1436706a2eb24.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/images/b69df3efdf1b988fdefec8884111bf1ce2bb61cac7b79300ffe1436706a2eb24.jpg)

![d8d8d9edce036048e6518b44f9d7f70a63efaff21ad6e02714053032d210adfb.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/images/d8d8d9edce036048e6518b44f9d7f70a63efaff21ad6e02714053032d210adfb.jpg)

![df4d01e450a37cde00f726f97eb6e43c02704f57472a86f9b4d14630ff6f8ba9.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/images/df4d01e450a37cde00f726f97eb6e43c02704f57472a86f9b4d14630ff6f8ba9.jpg)

![e05a6d77fbe946992d90a5eb3db85fd67156fca7987a48bc41937fb84ec2c2c9.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/images/e05a6d77fbe946992d90a5eb3db85fd67156fca7987a48bc41937fb84ec2c2c9.jpg)

![e977ae7b166918fdfc2b6034b2c2d2f6954f75f487533e11f63741fc02af5507.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/images/e977ae7b166918fdfc2b6034b2c2d2f6954f75f487533e11f63741fc02af5507.jpg)

![fcc2ea9183f1687bf8cd98ec827c30c7edf13f4ce09d7f81190c308a78283345.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/images/fcc2ea9183f1687bf8cd98ec827c30c7edf13f4ce09d7f81190c308a78283345.jpg)

### Tables

![1702b640a97768283d6e93c92ef5be127e0324b9c627bc31673e15d9ce2434df.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/tables/1702b640a97768283d6e93c92ef5be127e0324b9c627bc31673e15d9ce2434df.jpg)

![5d1328d13f0a1b793d314893323244f7b82cd89854515ea015f6de782f78ce85.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/tables/5d1328d13f0a1b793d314893323244f7b82cd89854515ea015f6de782f78ce85.jpg)

![65bf8ca4b322646275a11607a8377314bccea806f25f59aa9ab3d12b5194cf08.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/tables/65bf8ca4b322646275a11607a8377314bccea806f25f59aa9ab3d12b5194cf08.jpg)

![68083e9b3411f573a97717027556b77c2b4719d7c00ef0cdcee32d0cdbce4a39.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/tables/68083e9b3411f573a97717027556b77c2b4719d7c00ef0cdcee32d0cdbce4a39.jpg)

![80cfe4e1c97ee49b1cbee63b5defeaf0bf5f4fde709d52cf4fa735020d5f129e.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/tables/80cfe4e1c97ee49b1cbee63b5defeaf0bf5f4fde709d52cf4fa735020d5f129e.jpg)

![c9882434f6d487841c094e3c6b2db0995a5db8f2bd19c9840ccb62423b62017a.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/tables/c9882434f6d487841c094e3c6b2db0995a5db8f2bd19c9840ccb62423b62017a.jpg)

![cf313abac59d63a53475d9d8218458cee0d1729773acbd8ba201c73df8a7c692.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/tables/cf313abac59d63a53475d9d8218458cee0d1729773acbd8ba201c73df8a7c692.jpg)

![e99180dfc5591accfd41129d899e158dd4b26d757e69c3a72e9e0f19f5728fb1.jpg](../icml_results/5118_Towards%20Seed-Robust%20Safety%20Alignment%20in%20Text-to-Image%20Models/tables/e99180dfc5591accfd41129d899e158dd4b26d757e69c3a72e9e0f19f5728fb1.jpg)

<a id="paper-5071"></a>

## Hyperbolic neural population geometry benefits computation


### Images

![290aa5b3e6d9c28b5b89a4fd475d06939c5e591a6408954cbb8ea205a6f9f544.jpg](../icml_results/5119_Hyperbolic%20neural%20population%20geometry%20benefits%20computation/images/290aa5b3e6d9c28b5b89a4fd475d06939c5e591a6408954cbb8ea205a6f9f544.jpg)

![61bf3833efb6052d4012e8c37c1f7fca168188cbd0846461f4850486ec9fa93d.jpg](../icml_results/5119_Hyperbolic%20neural%20population%20geometry%20benefits%20computation/images/61bf3833efb6052d4012e8c37c1f7fca168188cbd0846461f4850486ec9fa93d.jpg)

![a1e4db9b4c7a7930661b0a8d56cdc79a67a30e07a78a780f63460aaabe88b221.jpg](../icml_results/5119_Hyperbolic%20neural%20population%20geometry%20benefits%20computation/images/a1e4db9b4c7a7930661b0a8d56cdc79a67a30e07a78a780f63460aaabe88b221.jpg)

![aebb7979a2f4db15e0426c8c2e9226737e7b18f75f04900cfa6cfddf07c4a784.jpg](../icml_results/5119_Hyperbolic%20neural%20population%20geometry%20benefits%20computation/images/aebb7979a2f4db15e0426c8c2e9226737e7b18f75f04900cfa6cfddf07c4a784.jpg)

![d59b989d74f74a18a56782a899538c868d94d27538bd9aaf1bff3a9aa3d09eda.jpg](../icml_results/5119_Hyperbolic%20neural%20population%20geometry%20benefits%20computation/images/d59b989d74f74a18a56782a899538c868d94d27538bd9aaf1bff3a9aa3d09eda.jpg)

### Tables

![03d846c440d3e393270746d3a703f5d86ec7688d03ddf144280db102d6916221.jpg](../icml_results/5119_Hyperbolic%20neural%20population%20geometry%20benefits%20computation/tables/03d846c440d3e393270746d3a703f5d86ec7688d03ddf144280db102d6916221.jpg)

![0a4f8f8b413cc47c568096dd1d6d86274a5289610e73cac5cc0b875c3e9256b1.jpg](../icml_results/5119_Hyperbolic%20neural%20population%20geometry%20benefits%20computation/tables/0a4f8f8b413cc47c568096dd1d6d86274a5289610e73cac5cc0b875c3e9256b1.jpg)

![5db1ee97ba9662cebfac2446df4b74e5da3f4c7596ca8189f2d9041759676fdc.jpg](../icml_results/5119_Hyperbolic%20neural%20population%20geometry%20benefits%20computation/tables/5db1ee97ba9662cebfac2446df4b74e5da3f4c7596ca8189f2d9041759676fdc.jpg)

![65ca0af2dda4d9fec1e42cef031b9c94ed522f32534ca9cf81b9a0f630fab584.jpg](../icml_results/5119_Hyperbolic%20neural%20population%20geometry%20benefits%20computation/tables/65ca0af2dda4d9fec1e42cef031b9c94ed522f32534ca9cf81b9a0f630fab584.jpg)

![fc41d4f1b16bd11780bf1459af6e35ef0f9b26347e3e04147892230d71d03e09.jpg](../icml_results/5119_Hyperbolic%20neural%20population%20geometry%20benefits%20computation/tables/fc41d4f1b16bd11780bf1459af6e35ef0f9b26347e3e04147892230d71d03e09.jpg)

![ff05e465b7c1259e05e54edaa67ae38ab54f1ee77ea717cc36e7a0ce9c535151.jpg](../icml_results/5119_Hyperbolic%20neural%20population%20geometry%20benefits%20computation/tables/ff05e465b7c1259e05e54edaa67ae38ab54f1ee77ea717cc36e7a0ce9c535151.jpg)

<a id="paper-5072"></a>

## KODA: Contrastive Representation Comparison and Alignment for Vision-Language Foundation Models


### Images

![04087a40f742189a12365c2001a9132ac0270d52fcb9ca5fc1f829e97fffb6cd.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/04087a40f742189a12365c2001a9132ac0270d52fcb9ca5fc1f829e97fffb6cd.jpg)

![058a81962967c1defdfaa647babfb05933e38e802486d25350052c24d5c12746.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/058a81962967c1defdfaa647babfb05933e38e802486d25350052c24d5c12746.jpg)

![0da9385ea139bb0d8dec0afd1e0ee39976314cc92a2ab37c0fb1ed24820f0ab9.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/0da9385ea139bb0d8dec0afd1e0ee39976314cc92a2ab37c0fb1ed24820f0ab9.jpg)

![0e0f56a151d6cadc2be8258c1b6146dbc99104e8011e23a87703ba4de448aeaf.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/0e0f56a151d6cadc2be8258c1b6146dbc99104e8011e23a87703ba4de448aeaf.jpg)

![0efbc2208f8afd3e07af27b592ee595da7959cd1804e87d88addd826d7eaeef3.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/0efbc2208f8afd3e07af27b592ee595da7959cd1804e87d88addd826d7eaeef3.jpg)

![22f7d6c5f442420995941c063a594046b4e60936c779ef4e3aa78e44388f1549.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/22f7d6c5f442420995941c063a594046b4e60936c779ef4e3aa78e44388f1549.jpg)

![24c796545898bf95adb3cfd96fa2c2b786f577cc54ee05590ca19dc37e18bfd1.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/24c796545898bf95adb3cfd96fa2c2b786f577cc54ee05590ca19dc37e18bfd1.jpg)

![2d4e6da81879d273ddf17c2156727e9b8aa5d55ef56d3e9906dc071daefe2f3d.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/2d4e6da81879d273ddf17c2156727e9b8aa5d55ef56d3e9906dc071daefe2f3d.jpg)

![3119bc28e5c5f7bf14fff11ce6d3186ae668365b35fdcd92a3d2d86f12db0f01.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/3119bc28e5c5f7bf14fff11ce6d3186ae668365b35fdcd92a3d2d86f12db0f01.jpg)

![35ac64ae579a5ded93735b8560a94d1d8c7617aae6b77758d4c8b6a447e2d9d9.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/35ac64ae579a5ded93735b8560a94d1d8c7617aae6b77758d4c8b6a447e2d9d9.jpg)

![3d9c73efdc1d765d70faa0ff2e9d5ece0cb2cde3d9fae48f89c3a896a59fb891.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/3d9c73efdc1d765d70faa0ff2e9d5ece0cb2cde3d9fae48f89c3a896a59fb891.jpg)

![3fcce67eb95b3f9380dee63638dc754be816ed26ec19c6d57e7a762c2116ca9d.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/3fcce67eb95b3f9380dee63638dc754be816ed26ec19c6d57e7a762c2116ca9d.jpg)

![40086630398edce392fbe0b206128919bae0b2e60e255ba4c402f969e592345f.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/40086630398edce392fbe0b206128919bae0b2e60e255ba4c402f969e592345f.jpg)

![43b8b7652d821ae262095bae7e25ea6ccd7f125a74bc93118d4ce8cea3f8856b.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/43b8b7652d821ae262095bae7e25ea6ccd7f125a74bc93118d4ce8cea3f8856b.jpg)

![4958a8e3b3753242256dd79b6868399b90d5052688e5d2953e3b6cf17fe1cbc3.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/4958a8e3b3753242256dd79b6868399b90d5052688e5d2953e3b6cf17fe1cbc3.jpg)

![49d279cc4e7f16671dab65d47957eba05539d52b2e4261c94f91650e6d5f52b0.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/49d279cc4e7f16671dab65d47957eba05539d52b2e4261c94f91650e6d5f52b0.jpg)

![54ea4835e35b6661d9d9e9f191c3ed6bc7967bf3b6767f90c497b651a68d5a4f.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/54ea4835e35b6661d9d9e9f191c3ed6bc7967bf3b6767f90c497b651a68d5a4f.jpg)

![573f0494f50cf6a94d8aa2508e9e457de577c6e762649cb11c485bbd359cfb9c.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/573f0494f50cf6a94d8aa2508e9e457de577c6e762649cb11c485bbd359cfb9c.jpg)

![642c5cfffe2d46463825d228a50139e2dc18852fc370f17038a16095f7f1b6ce.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/642c5cfffe2d46463825d228a50139e2dc18852fc370f17038a16095f7f1b6ce.jpg)

![9357d1dc2053da3eec0f545f2511e9d4603588c4cd90581034741c12918b87d2.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/9357d1dc2053da3eec0f545f2511e9d4603588c4cd90581034741c12918b87d2.jpg)

![9c077de0ce1578db88d4a0d0c298c9a8aee6fdbf3353ea3f7829cdb920d3e285.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/9c077de0ce1578db88d4a0d0c298c9a8aee6fdbf3353ea3f7829cdb920d3e285.jpg)

![9da114ee2bfb66a61cc89e59df8c2f64ffbb5638af0ea6f1686755b586ce76ad.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/9da114ee2bfb66a61cc89e59df8c2f64ffbb5638af0ea6f1686755b586ce76ad.jpg)

![a1f1a4fd3cd5a06e75fd019433cbfa7cc127ad71da8668297be89570712700a9.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/a1f1a4fd3cd5a06e75fd019433cbfa7cc127ad71da8668297be89570712700a9.jpg)

![ac05bd92073c730b0d743cc988365a0275108924e35454508219eeaf49d7992b.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/ac05bd92073c730b0d743cc988365a0275108924e35454508219eeaf49d7992b.jpg)

![adb153725caebdb4dd6b327e9dd68c09e38876981c000918128aa1c8669b431c.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/adb153725caebdb4dd6b327e9dd68c09e38876981c000918128aa1c8669b431c.jpg)

![adbcd6f0ec5a960eacdc52d7a7fb589519ba01f1b981b3ee4e4b154cdbb1d762.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/adbcd6f0ec5a960eacdc52d7a7fb589519ba01f1b981b3ee4e4b154cdbb1d762.jpg)

![b4a3913f5ade0a34bf217ebfc37043b7aca7df3561ea2de3d7a2a5c9a9690741.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/b4a3913f5ade0a34bf217ebfc37043b7aca7df3561ea2de3d7a2a5c9a9690741.jpg)

![b60c6d37b9abbbd37bf63ee8171ac3418a63c86d77cab9c0b6a0ef08f4d1b5b8.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/b60c6d37b9abbbd37bf63ee8171ac3418a63c86d77cab9c0b6a0ef08f4d1b5b8.jpg)

![b7154e56de0a2a120a85b96124d2e1c8586c18e4afcbca9daf7e7c07e377f3df.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/b7154e56de0a2a120a85b96124d2e1c8586c18e4afcbca9daf7e7c07e377f3df.jpg)

![bb5a3a13abf5d32929cd7cbfe0d0d43e547ddcb7b69b7ccab3ce76a47d1e1b0f.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/bb5a3a13abf5d32929cd7cbfe0d0d43e547ddcb7b69b7ccab3ce76a47d1e1b0f.jpg)

![c4892e739a7201f5924e5300e5559df983bc2a0d56f27a3172ce5d6c7a9a24d2.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/c4892e739a7201f5924e5300e5559df983bc2a0d56f27a3172ce5d6c7a9a24d2.jpg)

![c65b6d01bf007a8b28a38a66c68329a82a594fe42f6f75fff57d2bff8cd71de8.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/c65b6d01bf007a8b28a38a66c68329a82a594fe42f6f75fff57d2bff8cd71de8.jpg)

![c9348f9a0092e4a8bc7ab2e32c3c5ad1df0402407421725b74d5d488343eddae.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/c9348f9a0092e4a8bc7ab2e32c3c5ad1df0402407421725b74d5d488343eddae.jpg)

![d4ffff8bee663506b1023edb93deb15ec5ff31268895bdd464e9a0c58cf735ae.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/d4ffff8bee663506b1023edb93deb15ec5ff31268895bdd464e9a0c58cf735ae.jpg)

![da224e34625a587fbc6a1e4ea56b0c79dfce873b1f76026ded41fd08a3e67aa2.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/da224e34625a587fbc6a1e4ea56b0c79dfce873b1f76026ded41fd08a3e67aa2.jpg)

![daf8a1db63099f27e3b7475cc1491ad991c63329c52ad0ddf439414970642216.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/daf8a1db63099f27e3b7475cc1491ad991c63329c52ad0ddf439414970642216.jpg)

![dbee6665f129ad5a0bc3ff01f0b0d82bc0451ffc0091ea486f7440d10f86ab6f.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/dbee6665f129ad5a0bc3ff01f0b0d82bc0451ffc0091ea486f7440d10f86ab6f.jpg)

![e300c67716006b1e797da6d5ca62a81e04743018b0a2a0fad7f9ba683da50aae.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/e300c67716006b1e797da6d5ca62a81e04743018b0a2a0fad7f9ba683da50aae.jpg)

![e9e697b1145993d4b95fe41888372d75ffb7fa9e52fd55ee0da909ce69c3443a.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/e9e697b1145993d4b95fe41888372d75ffb7fa9e52fd55ee0da909ce69c3443a.jpg)

![ebbab969004f0355d37016b72a1af7531782590ee9de9a94d60173dc13dd2cb2.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/ebbab969004f0355d37016b72a1af7531782590ee9de9a94d60173dc13dd2cb2.jpg)

![f341a5c5e6b739eae85f2ff36811a0c4c2657cf66fa3598ac3f5865360cd6fb3.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/f341a5c5e6b739eae85f2ff36811a0c4c2657cf66fa3598ac3f5865360cd6fb3.jpg)

![facdee16347b3c1aec94edc77a3d584f599e6d246a89e00a11e875e1eb0543ae.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/images/facdee16347b3c1aec94edc77a3d584f599e6d246a89e00a11e875e1eb0543ae.jpg)

### Tables

![7ce386da6c52d4ccdaee1b6ab60c070913779b93ee3e49dc2c88f93cfe01f2a1.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/tables/7ce386da6c52d4ccdaee1b6ab60c070913779b93ee3e49dc2c88f93cfe01f2a1.jpg)

![a7ae91c892030d162fed4f1914b97d4b54eba3fe51e5351448ad544182b29715.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/tables/a7ae91c892030d162fed4f1914b97d4b54eba3fe51e5351448ad544182b29715.jpg)

![f0c9c3857ed8539e9fb774f7ca8b2cbedfc236b51103042f1d3bf9dfc3ed5e05.jpg](../icml_results/5120_KODA_%20Contrastive%20Representation%20Comparison%20and%20Alignment%20for%20Vision-Language%20Foundation%20Models/tables/f0c9c3857ed8539e9fb774f7ca8b2cbedfc236b51103042f1d3bf9dfc3ed5e05.jpg)

<a id="paper-5073"></a>

## Trustworthy Federated Label Distribution Learning under Annotation Quality Disparity


### Images

![30c1ea8dbe0eb2e335995c1be356395c6d89ee97a2dd9861adfc17ba58f30862.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/images/30c1ea8dbe0eb2e335995c1be356395c6d89ee97a2dd9861adfc17ba58f30862.jpg)

![313874ad3ce48a9ab8aed1e001f310e7469d064c10af3802a33e837dfa9606d2.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/images/313874ad3ce48a9ab8aed1e001f310e7469d064c10af3802a33e837dfa9606d2.jpg)

![31435cae39823c902bf66540575bea0d1a526868e6738fc9353073b31df39f83.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/images/31435cae39823c902bf66540575bea0d1a526868e6738fc9353073b31df39f83.jpg)

![4f3c3263eefa371f0a5bb49417fe4e1f949a6ebb6d7a5a70da31a7c020a12338.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/images/4f3c3263eefa371f0a5bb49417fe4e1f949a6ebb6d7a5a70da31a7c020a12338.jpg)

![6658d60672d71a4bb90840ad066db3adf16cc3b0d0cc913f1d26f56c434000c5.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/images/6658d60672d71a4bb90840ad066db3adf16cc3b0d0cc913f1d26f56c434000c5.jpg)

![67c6ad2953fe81a6eb25e40baed3f4c2c491cd5a67472cc418a540018c4ce327.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/images/67c6ad2953fe81a6eb25e40baed3f4c2c491cd5a67472cc418a540018c4ce327.jpg)

![7032bd80ae2db1a3457510ab872465676558bda5a88fb4a7f2b81bdd8cd299cf.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/images/7032bd80ae2db1a3457510ab872465676558bda5a88fb4a7f2b81bdd8cd299cf.jpg)

![74bb2833d9c55e4411ae1bfb6ebc36d59834507c2d5ef80046e399c49ec87ce3.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/images/74bb2833d9c55e4411ae1bfb6ebc36d59834507c2d5ef80046e399c49ec87ce3.jpg)

![799036e0a1ac32a1914789955d8fe0fd224557b1e101af3051f939ddab9ee22d.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/images/799036e0a1ac32a1914789955d8fe0fd224557b1e101af3051f939ddab9ee22d.jpg)

![8ed58a75c4f733588f1fe262ceed545f84228cddbc4daec01eaccd6bab30d61b.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/images/8ed58a75c4f733588f1fe262ceed545f84228cddbc4daec01eaccd6bab30d61b.jpg)

![9725d2cb94355d7aabd7be03eaf64f50dbdadd58f11f467420932df77a3aad2d.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/images/9725d2cb94355d7aabd7be03eaf64f50dbdadd58f11f467420932df77a3aad2d.jpg)

![ce9e47aeeb373694633507dadc9aed48322ade1ba4336f1664200eb20b3624ed.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/images/ce9e47aeeb373694633507dadc9aed48322ade1ba4336f1664200eb20b3624ed.jpg)

![f3967c314a77b2a4d5e7d1c066e00b0df341c149dfb0350d38c7183607162fb4.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/images/f3967c314a77b2a4d5e7d1c066e00b0df341c149dfb0350d38c7183607162fb4.jpg)

![f5ae2c4c526236651e6f832f5493973d73e941fc5de78f6a6987a7e8cb889ca2.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/images/f5ae2c4c526236651e6f832f5493973d73e941fc5de78f6a6987a7e8cb889ca2.jpg)

### Tables

![10f0f8759497e82fa26b5fa26c0cc1e6f2183cdd6d459451acb2b88c4e9cb917.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/tables/10f0f8759497e82fa26b5fa26c0cc1e6f2183cdd6d459451acb2b88c4e9cb917.jpg)

![6195ae846a606d1f1225c7f090a70ff3e19a8e3249111229de6a12c48e50f16a.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/tables/6195ae846a606d1f1225c7f090a70ff3e19a8e3249111229de6a12c48e50f16a.jpg)

![65cadc9fd6e6562f7d8883d16f074aeb4b732ada03c6a1fc875e0bb37b50b660.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/tables/65cadc9fd6e6562f7d8883d16f074aeb4b732ada03c6a1fc875e0bb37b50b660.jpg)

![92d82f9dcdeaf75a44687fafd1eb3d15ce374a6de5d44c4def546e53831b07c9.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/tables/92d82f9dcdeaf75a44687fafd1eb3d15ce374a6de5d44c4def546e53831b07c9.jpg)

![ba6d08ef539df771abe634707e8ff8631942a67a523016bab2461ba0baa61ae1.jpg](../icml_results/5121_Trustworthy%20Federated%20Label%20Distribution%20Learning%20under%20Annotation%20Quality%20Disparity/tables/ba6d08ef539df771abe634707e8ff8631942a67a523016bab2461ba0baa61ae1.jpg)

<a id="paper-5074"></a>

## Adaptively Grouped Contextual Bandits for Heterogeneous Human-AI Decision Making with Conformal Prediction Sets


### Images

![657365760b73aa4741f19918ec95350247137d2c03851172f3ade662c5d64f09.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/images/657365760b73aa4741f19918ec95350247137d2c03851172f3ade662c5d64f09.jpg)

![beec8ab83bd3f20b3c026335e62bd71beea6eb585621fd069c3e0cb351bf3dcb.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/images/beec8ab83bd3f20b3c026335e62bd71beea6eb585621fd069c3e0cb351bf3dcb.jpg)

![f4e1e344d5ddaf56909fca3b97a17e2ae0d7fe8a801e29170c546837640eb2da.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/images/f4e1e344d5ddaf56909fca3b97a17e2ae0d7fe8a801e29170c546837640eb2da.jpg)

### Tables

![223cc8b9f2258e11daa9e30c38199150ab9a5932fce1926af90aa53a6d6a18f3.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/tables/223cc8b9f2258e11daa9e30c38199150ab9a5932fce1926af90aa53a6d6a18f3.jpg)

![40e105bf55123a7a6800cf7ccbd682d2835e01cb8ce1324fc75bb1f66f980b2c.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/tables/40e105bf55123a7a6800cf7ccbd682d2835e01cb8ce1324fc75bb1f66f980b2c.jpg)

![4b7cc8803705835a66ee7268dc05a391fb1532dc27a02454e37dd8d9996ed5e1.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/tables/4b7cc8803705835a66ee7268dc05a391fb1532dc27a02454e37dd8d9996ed5e1.jpg)

![6b32fd6ddcdd84939fc73d8f469c5a5b6bdb582777c24610a17c998819c572d6.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/tables/6b32fd6ddcdd84939fc73d8f469c5a5b6bdb582777c24610a17c998819c572d6.jpg)

![6b432bc4cc06b3f2ce13014954472e8775103b957ee8b0292cf24fef014caf9e.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/tables/6b432bc4cc06b3f2ce13014954472e8775103b957ee8b0292cf24fef014caf9e.jpg)

![7b73f0bffd4b7402153664b43daad5a83ba8110bee162a0f1cde243edc6e9706.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/tables/7b73f0bffd4b7402153664b43daad5a83ba8110bee162a0f1cde243edc6e9706.jpg)

![7ed92e4e27ce30d585d6e720eb8e5e21506e06affca8b9e177c432c1cd885eef.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/tables/7ed92e4e27ce30d585d6e720eb8e5e21506e06affca8b9e177c432c1cd885eef.jpg)

![96d895ea9cc7ad417734552dbb3ef76d23033bc26013df8916f06f3f7d3548a2.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/tables/96d895ea9cc7ad417734552dbb3ef76d23033bc26013df8916f06f3f7d3548a2.jpg)

![9b5d16979f1c92fa125a80101eb6e9a342940b5c2544bf29811e96daac55d570.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/tables/9b5d16979f1c92fa125a80101eb6e9a342940b5c2544bf29811e96daac55d570.jpg)

![a71ed50c8a28795afe0ab442ea132dc6206c01779e1ed68b0c673253fe0c3561.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/tables/a71ed50c8a28795afe0ab442ea132dc6206c01779e1ed68b0c673253fe0c3561.jpg)

![b24d30984bbcf1167de84951beb1ebf40478ec1ab1efe584016e4011e617f2b1.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/tables/b24d30984bbcf1167de84951beb1ebf40478ec1ab1efe584016e4011e617f2b1.jpg)

![b9b4581bfb24c4d53cbb7305ff965ae3167e889f743d521ed03e464d3f388220.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/tables/b9b4581bfb24c4d53cbb7305ff965ae3167e889f743d521ed03e464d3f388220.jpg)

![c3d7e3aa52a9a22b014581f3c423c8b2e95eea047a7563e515ec6279734a1957.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/tables/c3d7e3aa52a9a22b014581f3c423c8b2e95eea047a7563e515ec6279734a1957.jpg)

![d204414329ea3fd7035965e1b691562a21e39c5917fd9decdc62d311c16ab2d8.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/tables/d204414329ea3fd7035965e1b691562a21e39c5917fd9decdc62d311c16ab2d8.jpg)

![f720f7e415ee44682c7a0b4067b142f693cccc158a5f9ddfc36a9aea69fd0ba2.jpg](../icml_results/5122_Adaptively%20Grouped%20Contextual%20Bandits%20for%20Heterogeneous%20Human-AI%20Decision%20Making%20with%20Conformal%20Pred/tables/f720f7e415ee44682c7a0b4067b142f693cccc158a5f9ddfc36a9aea69fd0ba2.jpg)

<a id="paper-5075"></a>

## Scalable Topology-Preserving Graph Coarsening: Concepts and Algorithms


### Images

![27c4418c2e2cade8ee86af5a7b5a77306f4125bf5b18e43cc01c60da688a09da.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/images/27c4418c2e2cade8ee86af5a7b5a77306f4125bf5b18e43cc01c60da688a09da.jpg)

![281193b74a74c3c14b163cb4d025920ce08a34a43bbf9d34bb9bc654fcba89fe.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/images/281193b74a74c3c14b163cb4d025920ce08a34a43bbf9d34bb9bc654fcba89fe.jpg)

![3572c6dd046e49fe8ae7e3adbba9eab259cdafda08cf8564b3b56c6baf180033.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/images/3572c6dd046e49fe8ae7e3adbba9eab259cdafda08cf8564b3b56c6baf180033.jpg)

![6fe76d8023f5221b5dfdd831ceafcc7c3c0dbed416edd6e25ae2bd3bb2d3ce61.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/images/6fe76d8023f5221b5dfdd831ceafcc7c3c0dbed416edd6e25ae2bd3bb2d3ce61.jpg)

![7d740fb40fdb8e5bf0d6a56b0b78a8444b592157f19e8f8a371d98e9fac4c83f.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/images/7d740fb40fdb8e5bf0d6a56b0b78a8444b592157f19e8f8a371d98e9fac4c83f.jpg)

![9b479d1ec05e887176532102372225ac296947c36b483d0c70f7d93669be1733.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/images/9b479d1ec05e887176532102372225ac296947c36b483d0c70f7d93669be1733.jpg)

![9de480b52166536ddf34f4d108b978ff2cb1fd55847bd3ca26469353365a7f6f.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/images/9de480b52166536ddf34f4d108b978ff2cb1fd55847bd3ca26469353365a7f6f.jpg)

![beea5fa0f2a1c077f697e11d07819c5fb228ea043c7c30a3541f784b2a28ffd0.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/images/beea5fa0f2a1c077f697e11d07819c5fb228ea043c7c30a3541f784b2a28ffd0.jpg)

![c63820d131f97a351dfcdb9f71c9918d076e0982e064d7ed59ef731bda67f598.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/images/c63820d131f97a351dfcdb9f71c9918d076e0982e064d7ed59ef731bda67f598.jpg)

![db67184a007b8a3a8d147c8c94843279da317712fd98cf0f8bb9d66c870bb299.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/images/db67184a007b8a3a8d147c8c94843279da317712fd98cf0f8bb9d66c870bb299.jpg)

![e5262df22c1a047ea0647afd72b53f5f71b69792dfe0e3118d3a84a2790f4d26.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/images/e5262df22c1a047ea0647afd72b53f5f71b69792dfe0e3118d3a84a2790f4d26.jpg)

![eeb478c42d392476a52c39da936d141477efd9716dc8c583e5406223d87a39c9.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/images/eeb478c42d392476a52c39da936d141477efd9716dc8c583e5406223d87a39c9.jpg)

![fa4863fcd66e9df7b04d0e81ade51fa6948c6f6a118314c85fb5dc7a41438d15.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/images/fa4863fcd66e9df7b04d0e81ade51fa6948c6f6a118314c85fb5dc7a41438d15.jpg)

![fcd7d2fb297c53dcfe01abd806e565e73e1eeaa1a3593f30b574a8614498764b.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/images/fcd7d2fb297c53dcfe01abd806e565e73e1eeaa1a3593f30b574a8614498764b.jpg)

### Tables

![343ab10af8403e4daec1f63ca3fb9024494ebc3585629fc4efcedb8d12c19f24.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/tables/343ab10af8403e4daec1f63ca3fb9024494ebc3585629fc4efcedb8d12c19f24.jpg)

![6ab592cd882a3b7a7f71550ee3e9a711d109da335a604911360debb90f520ada.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/tables/6ab592cd882a3b7a7f71550ee3e9a711d109da335a604911360debb90f520ada.jpg)

![6f1225326d95475def87750034ea5da85387462bf16693d829d1283b5d18aac5.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/tables/6f1225326d95475def87750034ea5da85387462bf16693d829d1283b5d18aac5.jpg)

![923c98e1993fc6af6422dc6a6f6bb6e1e7b2c603769f2bf5c2c72cbbabe60e62.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/tables/923c98e1993fc6af6422dc6a6f6bb6e1e7b2c603769f2bf5c2c72cbbabe60e62.jpg)

![aa5dff87e241b29f1549dd9777c593a3b1ce80ec0af74d7d21743b993653729b.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/tables/aa5dff87e241b29f1549dd9777c593a3b1ce80ec0af74d7d21743b993653729b.jpg)

![c08ec04415d56357c719d0496e7e0493e8813b888b2ab2afbe2187cb129b6642.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/tables/c08ec04415d56357c719d0496e7e0493e8813b888b2ab2afbe2187cb129b6642.jpg)

![d8db52d7b3f29c69b4cc2f9cf2dadfab0fc7dfd64602c4cc2a035ecfd76e318c.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/tables/d8db52d7b3f29c69b4cc2f9cf2dadfab0fc7dfd64602c4cc2a035ecfd76e318c.jpg)

![dfe5c664304d6d5a98c23bae69fb878b8e7780e4aa16b0ad9fc7b6a2dfe50a7c.jpg](../icml_results/5123_Scalable%20Topology-Preserving%20Graph%20Coarsening_%20Concepts%20and%20Algorithms/tables/dfe5c664304d6d5a98c23bae69fb878b8e7780e4aa16b0ad9fc7b6a2dfe50a7c.jpg)

<a id="paper-5076"></a>

## Frequency-Aware Perceptual Optimization for Low-Complexity Implicit Image Compression


### Images

![1a946dc8bc980d44170a107bdd630d8100b509ccd33c18dc3703611f025475d2.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/1a946dc8bc980d44170a107bdd630d8100b509ccd33c18dc3703611f025475d2.jpg)

![1febe56ec36fc9b62da9f4fd09f4993d0dbba6a43c0c177d8b346ae62521eea9.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/1febe56ec36fc9b62da9f4fd09f4993d0dbba6a43c0c177d8b346ae62521eea9.jpg)

![22c912ab261db966f67dfceeb3fca1589aeb6e5d6b0f7cb55c80452eb6a77f0c.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/22c912ab261db966f67dfceeb3fca1589aeb6e5d6b0f7cb55c80452eb6a77f0c.jpg)

![253a9d0b3052d638b5d9e1e0df507e9e7b421b700528badbd0e164d33e4bcb23.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/253a9d0b3052d638b5d9e1e0df507e9e7b421b700528badbd0e164d33e4bcb23.jpg)

![254a869e46ea3458b15ce11f826b2dab55df2c3d4eacb5e5cb9be775e5ef8cbd.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/254a869e46ea3458b15ce11f826b2dab55df2c3d4eacb5e5cb9be775e5ef8cbd.jpg)

![255c3e1d7e7979344f1ec60c23a0dafe84f57f7cb3a4dcbc286ab66b57f53288.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/255c3e1d7e7979344f1ec60c23a0dafe84f57f7cb3a4dcbc286ab66b57f53288.jpg)

![336c25a402bea5adf2192898ebc5cba350feaa82f9907e75ef76a28d99df28a7.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/336c25a402bea5adf2192898ebc5cba350feaa82f9907e75ef76a28d99df28a7.jpg)

![36d62e870833aa0d150f2da2a31fe8a42eb2d388939f0ffe0d26651114fa7475.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/36d62e870833aa0d150f2da2a31fe8a42eb2d388939f0ffe0d26651114fa7475.jpg)

![3e756b6a3b85ce2e46dba6da66a1a73259d3b373f3004aad0475ea48a9c2191e.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/3e756b6a3b85ce2e46dba6da66a1a73259d3b373f3004aad0475ea48a9c2191e.jpg)

![42e8dbd170ca4fac0932958b2084fd6b4f8ed469c28605881745c98e1bee01df.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/42e8dbd170ca4fac0932958b2084fd6b4f8ed469c28605881745c98e1bee01df.jpg)

![5910036fa3e731638266c141f48b239477f3a328826ba2a2f91a7118ecbaad85.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/5910036fa3e731638266c141f48b239477f3a328826ba2a2f91a7118ecbaad85.jpg)

![5a5e275cf384789bc5d7d16888c97cd8ac6f70dbab812558ae5042d6f66efa56.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/5a5e275cf384789bc5d7d16888c97cd8ac6f70dbab812558ae5042d6f66efa56.jpg)

![6cf62c82212ca83d3e70833c233fd53ebdeaf09eee339ad30a3f331ba5330533.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/6cf62c82212ca83d3e70833c233fd53ebdeaf09eee339ad30a3f331ba5330533.jpg)

![6df4989cca2c1b7f1ef1b701617d3ae10476fdf3d3f1a5aa711e3f83bbdbec0a.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/6df4989cca2c1b7f1ef1b701617d3ae10476fdf3d3f1a5aa711e3f83bbdbec0a.jpg)

![7326fee781a2c680f8326ad7b4e9e48685f9fde8572c1dec003a551d62b18f1f.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/7326fee781a2c680f8326ad7b4e9e48685f9fde8572c1dec003a551d62b18f1f.jpg)

![7790f6aca1c873db872623ca7933fb71e5640163960461783bb74d4edb090d80.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/7790f6aca1c873db872623ca7933fb71e5640163960461783bb74d4edb090d80.jpg)

![7801dfad356ca5c1b44cb64a0851df492850578011b6899e06424308bd5139f5.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/7801dfad356ca5c1b44cb64a0851df492850578011b6899e06424308bd5139f5.jpg)

![8eab41dfbf2b8dcb39e14f4ff28fcea21ffa6dcf1066b1378c9a0b1fc87aa686.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/8eab41dfbf2b8dcb39e14f4ff28fcea21ffa6dcf1066b1378c9a0b1fc87aa686.jpg)

![90c6545ba342caf515e1d3aa56539cc07f2966588088bb4ebfe9cfe5eba85951.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/90c6545ba342caf515e1d3aa56539cc07f2966588088bb4ebfe9cfe5eba85951.jpg)

![9411b288b001b6da08e9ce76de4333bcaed86838a67c3152990ddb33a2410119.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/9411b288b001b6da08e9ce76de4333bcaed86838a67c3152990ddb33a2410119.jpg)

![a14b3470f6838c4780070ed88e5c83b82bd57d0c17bc86bf6c990c116450eff5.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/a14b3470f6838c4780070ed88e5c83b82bd57d0c17bc86bf6c990c116450eff5.jpg)

![b6bb4312b37e613867e7bc19ab1fd2f400e4dbd5a59c5c9363d951d7c698ac2e.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/b6bb4312b37e613867e7bc19ab1fd2f400e4dbd5a59c5c9363d951d7c698ac2e.jpg)

![bbb64cd788d571696bc8ca16f123533787286cdf70c10794ed0534b0546c0c97.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/bbb64cd788d571696bc8ca16f123533787286cdf70c10794ed0534b0546c0c97.jpg)

![bbc9f71155d85e6089b23a350faa2da5a409182a94c092b64e04fd9290eeece1.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/bbc9f71155d85e6089b23a350faa2da5a409182a94c092b64e04fd9290eeece1.jpg)

![c4599d4ef494f1bb01a955cb4b0c297db530c63f2209b1ce5081cfdc512b7fa4.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/c4599d4ef494f1bb01a955cb4b0c297db530c63f2209b1ce5081cfdc512b7fa4.jpg)

![c92c4841e0077af3e8385097ae2a8d5c63d847fa712a4dd9be95628e9d17a9ce.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/c92c4841e0077af3e8385097ae2a8d5c63d847fa712a4dd9be95628e9d17a9ce.jpg)

![ca5e2e9743b4d713482a8c040c2aecb9955247fe77a2ff0323b946066e3a6b7b.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/ca5e2e9743b4d713482a8c040c2aecb9955247fe77a2ff0323b946066e3a6b7b.jpg)

![db743e0895139f362d961362ef07f8886f0d9c7a439f9804d3078d6395e391a0.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/db743e0895139f362d961362ef07f8886f0d9c7a439f9804d3078d6395e391a0.jpg)

![dfbda88ef16709109f5440c3fdfb67138f2ce7f8ef1f6766696995a828ccaf18.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/dfbda88ef16709109f5440c3fdfb67138f2ce7f8ef1f6766696995a828ccaf18.jpg)

![f7fb2fc72f71b86550c79f01e529722ded69fe71e2e036a3aa57939a37848b9d.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/f7fb2fc72f71b86550c79f01e529722ded69fe71e2e036a3aa57939a37848b9d.jpg)

![f9962d6884295e239eb5266f0f4a2be0564b0e038d8b3c090afc667c5bab7b13.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/f9962d6884295e239eb5266f0f4a2be0564b0e038d8b3c090afc667c5bab7b13.jpg)

![ff60a967c219bfba4592a8f4a5560ec6405daa9c92c944ac27f620b58c4bb994.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/images/ff60a967c219bfba4592a8f4a5560ec6405daa9c92c944ac27f620b58c4bb994.jpg)

### Tables

![0b3765a6aec0f29c04aa854d205033007aa2edab7d1e59aea10103f9497563d8.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/tables/0b3765a6aec0f29c04aa854d205033007aa2edab7d1e59aea10103f9497563d8.jpg)

![0b9dca303b97b644ef9fa0c06fd31f0d30b916cb476b6c059c1e11a4df4405d1.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/tables/0b9dca303b97b644ef9fa0c06fd31f0d30b916cb476b6c059c1e11a4df4405d1.jpg)

![314b0c14f5da085aed59622bef9ea832abff01fac29534729ef2fa39bfe3c75a.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/tables/314b0c14f5da085aed59622bef9ea832abff01fac29534729ef2fa39bfe3c75a.jpg)

![3cb149ced4aa41ff983b206e729aeef7a43cfab9e458aae6834ff27064ce6927.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/tables/3cb149ced4aa41ff983b206e729aeef7a43cfab9e458aae6834ff27064ce6927.jpg)

![49dfcf3d1c62df7e7c655c2707746b19cfe3ceeede9bde5d01bf91660e0e145f.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/tables/49dfcf3d1c62df7e7c655c2707746b19cfe3ceeede9bde5d01bf91660e0e145f.jpg)

![4eb4420acb2002eb1ff57cef91a400e6f6d70bd5acc0279d447df4c00eed4627.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/tables/4eb4420acb2002eb1ff57cef91a400e6f6d70bd5acc0279d447df4c00eed4627.jpg)

![6e77f55f8dfdf614aa767be72b63ebceb5f6e7ec1c5bb349702a8b7325012825.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/tables/6e77f55f8dfdf614aa767be72b63ebceb5f6e7ec1c5bb349702a8b7325012825.jpg)

![847d6e0438f2bb3caaf442238f63d0b429edcef052cc496113e3564251a9a130.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/tables/847d6e0438f2bb3caaf442238f63d0b429edcef052cc496113e3564251a9a130.jpg)

![8f358ecbb580dd19df5375945369106d1150a88d63547357f5944551ba7f7bfc.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/tables/8f358ecbb580dd19df5375945369106d1150a88d63547357f5944551ba7f7bfc.jpg)

![a155d07113b378a550dd5913aee0ea73d18094f0971c80c674bc2c35e4a0ff9b.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/tables/a155d07113b378a550dd5913aee0ea73d18094f0971c80c674bc2c35e4a0ff9b.jpg)

![b06b7fcd82c93f652e76cb14cc0254e3c7bbc2a044ef5aa6097ebda9084440ec.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/tables/b06b7fcd82c93f652e76cb14cc0254e3c7bbc2a044ef5aa6097ebda9084440ec.jpg)

![b9efd908c68bb03d388e05d7a57d1bc7c8dedca2037b3404ee9ca6271042f7a8.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/tables/b9efd908c68bb03d388e05d7a57d1bc7c8dedca2037b3404ee9ca6271042f7a8.jpg)

![c33f4e58b5e2502e458538d7af906e60ea86ae981477f6c3671703167817f9a8.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/tables/c33f4e58b5e2502e458538d7af906e60ea86ae981477f6c3671703167817f9a8.jpg)

![df35c048b95740f20283f6552d51d51f8d13906c972ae9a869518448ea503a5b.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/tables/df35c048b95740f20283f6552d51d51f8d13906c972ae9a869518448ea503a5b.jpg)

![ef884912356bd524724a7d0027176d85715ee4bb820185c46fccdfbffbc10f0f.jpg](../icml_results/5124_Frequency-Aware%20Perceptual%20Optimization%20for%20Low-Complexity%20Implicit%20Image%20Compression/tables/ef884912356bd524724a7d0027176d85715ee4bb820185c46fccdfbffbc10f0f.jpg)

<a id="paper-5077"></a>

## Learning Generalized Label Distributions


### Images

![2fed9619c4aeec363296567f6a60428bc58215774e414e638eec0973c7ec91d6.jpg](../icml_results/5125_Learning%20Generalized%20Label%20Distributions/images/2fed9619c4aeec363296567f6a60428bc58215774e414e638eec0973c7ec91d6.jpg)

![44f711b0fbcdfc587c47452afee198ebcf00966ff3d82a2df6c54d1cabe1bf99.jpg](../icml_results/5125_Learning%20Generalized%20Label%20Distributions/images/44f711b0fbcdfc587c47452afee198ebcf00966ff3d82a2df6c54d1cabe1bf99.jpg)

![e3ea02892d0d835dd1fb76d971e411d67954e2dfa1c9d98b7e4c1b2dbe4972cf.jpg](../icml_results/5125_Learning%20Generalized%20Label%20Distributions/images/e3ea02892d0d835dd1fb76d971e411d67954e2dfa1c9d98b7e4c1b2dbe4972cf.jpg)

### Tables

![0bc945639a80ce947a77e6815331fed481f9b541169b5aae389ce051a0485579.jpg](../icml_results/5125_Learning%20Generalized%20Label%20Distributions/tables/0bc945639a80ce947a77e6815331fed481f9b541169b5aae389ce051a0485579.jpg)

![19e53b78ab6d021960e8148d1f5adf7011983fcaf33b9bdf0c6505c6f3e739da.jpg](../icml_results/5125_Learning%20Generalized%20Label%20Distributions/tables/19e53b78ab6d021960e8148d1f5adf7011983fcaf33b9bdf0c6505c6f3e739da.jpg)

![26006fd85e14ac34a1a175bdbbe7651f8579da419ec97e792d14d2524dd2fe1d.jpg](../icml_results/5125_Learning%20Generalized%20Label%20Distributions/tables/26006fd85e14ac34a1a175bdbbe7651f8579da419ec97e792d14d2524dd2fe1d.jpg)

![553faf642825cb3ecb61dc3a4bdcf919d3610f18d426ccd428028823cd9446a6.jpg](../icml_results/5125_Learning%20Generalized%20Label%20Distributions/tables/553faf642825cb3ecb61dc3a4bdcf919d3610f18d426ccd428028823cd9446a6.jpg)

![706cbbbfdfa7edb8d6e7ea462d73848cb79f2cc6986e671c332368d5a791a041.jpg](../icml_results/5125_Learning%20Generalized%20Label%20Distributions/tables/706cbbbfdfa7edb8d6e7ea462d73848cb79f2cc6986e671c332368d5a791a041.jpg)

![8c13a151c9681e2b05363575387ffcca70ed0fb87490795a3a99bf42765ba6f0.jpg](../icml_results/5125_Learning%20Generalized%20Label%20Distributions/tables/8c13a151c9681e2b05363575387ffcca70ed0fb87490795a3a99bf42765ba6f0.jpg)

![90471a46c1cbb7692b37f9b28c7874032ee5fc198b842115381ddf4e6ddc030b.jpg](../icml_results/5125_Learning%20Generalized%20Label%20Distributions/tables/90471a46c1cbb7692b37f9b28c7874032ee5fc198b842115381ddf4e6ddc030b.jpg)

![edc35a6316d108f93f0cf66c0c0732f1c301d66e8349e7ac18cf6a04038db1ec.jpg](../icml_results/5125_Learning%20Generalized%20Label%20Distributions/tables/edc35a6316d108f93f0cf66c0c0732f1c301d66e8349e7ac18cf6a04038db1ec.jpg)

<a id="paper-5078"></a>

## Diversity-Preserved Distribution Matching Distillation for Fast Visual Synthesis


### Images

![20c3f5e2f5b7f29fe4469a8244976b8efb8cd0b194b6edfa3793cd3e96898baf.jpg](../icml_results/5126_Diversity-Preserved%20Distribution%20Matching%20Distillation%20for%20Fast%20Visual%20Synthesis/images/20c3f5e2f5b7f29fe4469a8244976b8efb8cd0b194b6edfa3793cd3e96898baf.jpg)

![240774baa54f30bc212b6c4ddc138817f86f766864f08a8f1094bccff8942a5f.jpg](../icml_results/5126_Diversity-Preserved%20Distribution%20Matching%20Distillation%20for%20Fast%20Visual%20Synthesis/images/240774baa54f30bc212b6c4ddc138817f86f766864f08a8f1094bccff8942a5f.jpg)

![5d2112e8f5049d64547a66137c4d2fe8441ff17eb8a8326a8f20bef942db9b97.jpg](../icml_results/5126_Diversity-Preserved%20Distribution%20Matching%20Distillation%20for%20Fast%20Visual%20Synthesis/images/5d2112e8f5049d64547a66137c4d2fe8441ff17eb8a8326a8f20bef942db9b97.jpg)

![8880d7bcbb456556d14c18fefebf9506eaad6bb47e6c2c78ebb630e70bcd72aa.jpg](../icml_results/5126_Diversity-Preserved%20Distribution%20Matching%20Distillation%20for%20Fast%20Visual%20Synthesis/images/8880d7bcbb456556d14c18fefebf9506eaad6bb47e6c2c78ebb630e70bcd72aa.jpg)

![8b6096c00268ed265b4d34be973479fa74f2e6efb91b3abcdeeeac95208ca52b.jpg](../icml_results/5126_Diversity-Preserved%20Distribution%20Matching%20Distillation%20for%20Fast%20Visual%20Synthesis/images/8b6096c00268ed265b4d34be973479fa74f2e6efb91b3abcdeeeac95208ca52b.jpg)

![942fdfa249cb83d0fb2a316f9274fd232aab9504648704071f2ddb3d8bd44aec.jpg](../icml_results/5126_Diversity-Preserved%20Distribution%20Matching%20Distillation%20for%20Fast%20Visual%20Synthesis/images/942fdfa249cb83d0fb2a316f9274fd232aab9504648704071f2ddb3d8bd44aec.jpg)

![a3e50d8f414c3e7d52488fa7ca05d17761f11683909f5ed714d2e564ea38b1aa.jpg](../icml_results/5126_Diversity-Preserved%20Distribution%20Matching%20Distillation%20for%20Fast%20Visual%20Synthesis/images/a3e50d8f414c3e7d52488fa7ca05d17761f11683909f5ed714d2e564ea38b1aa.jpg)

![ae39163e45b423c2e3516838b828700937a527f9bd9d26af08d2b268dd693d59.jpg](../icml_results/5126_Diversity-Preserved%20Distribution%20Matching%20Distillation%20for%20Fast%20Visual%20Synthesis/images/ae39163e45b423c2e3516838b828700937a527f9bd9d26af08d2b268dd693d59.jpg)

![d71ee50dd76fda68722ea01c7421c648b18ac039ca6712416d86b1303cea3f02.jpg](../icml_results/5126_Diversity-Preserved%20Distribution%20Matching%20Distillation%20for%20Fast%20Visual%20Synthesis/images/d71ee50dd76fda68722ea01c7421c648b18ac039ca6712416d86b1303cea3f02.jpg)

### Tables

![45c696f8f299274deba9d40ff734bbb4665a0c124e77cdbfd1a855084f68b8b3.jpg](../icml_results/5126_Diversity-Preserved%20Distribution%20Matching%20Distillation%20for%20Fast%20Visual%20Synthesis/tables/45c696f8f299274deba9d40ff734bbb4665a0c124e77cdbfd1a855084f68b8b3.jpg)

![a30a519857970f4d02edd29e77264880422d163a353cd37ccd4ef604857641e9.jpg](../icml_results/5126_Diversity-Preserved%20Distribution%20Matching%20Distillation%20for%20Fast%20Visual%20Synthesis/tables/a30a519857970f4d02edd29e77264880422d163a353cd37ccd4ef604857641e9.jpg)

![efb90c02648967a3b19032f39591386e2232b5bc8d7b5e06b601f1747eedeccc.jpg](../icml_results/5126_Diversity-Preserved%20Distribution%20Matching%20Distillation%20for%20Fast%20Visual%20Synthesis/tables/efb90c02648967a3b19032f39591386e2232b5bc8d7b5e06b601f1747eedeccc.jpg)

![fc10e9b98e63b255fff3574612cc258ec183fa201fdd5c8474dd727faea4339e.jpg](../icml_results/5126_Diversity-Preserved%20Distribution%20Matching%20Distillation%20for%20Fast%20Visual%20Synthesis/tables/fc10e9b98e63b255fff3574612cc258ec183fa201fdd5c8474dd727faea4339e.jpg)

![fc9f214557ecbea4053aedd73ee749b1885afc7acc13fc9c35434b39c8078de5.jpg](../icml_results/5126_Diversity-Preserved%20Distribution%20Matching%20Distillation%20for%20Fast%20Visual%20Synthesis/tables/fc9f214557ecbea4053aedd73ee749b1885afc7acc13fc9c35434b39c8078de5.jpg)

<a id="paper-5079"></a>

## Native Parallel Reasoner: Reasoning in Parallelism via Self-Distilled Reinforcement Learning


### Images

![69621bae177e98951ef699528410ba27a01b2f329a12eb6755d817afe192dcf1.jpg](../icml_results/5127_Native%20Parallel%20Reasoner_%20Reasoning%20in%20Parallelism%20via%20Self-Distilled%20Reinforcement%20Learning/images/69621bae177e98951ef699528410ba27a01b2f329a12eb6755d817afe192dcf1.jpg)

![6a3c90852632157f9531c2c6cada14c8a84805d29856edf1587d6b7527ddd485.jpg](../icml_results/5127_Native%20Parallel%20Reasoner_%20Reasoning%20in%20Parallelism%20via%20Self-Distilled%20Reinforcement%20Learning/images/6a3c90852632157f9531c2c6cada14c8a84805d29856edf1587d6b7527ddd485.jpg)

![933657ab79725e97592e57ea3badf5160d7447ed19ecca36e8e2dbd709d28076.jpg](../icml_results/5127_Native%20Parallel%20Reasoner_%20Reasoning%20in%20Parallelism%20via%20Self-Distilled%20Reinforcement%20Learning/images/933657ab79725e97592e57ea3badf5160d7447ed19ecca36e8e2dbd709d28076.jpg)

![b1b7aac30a5358897f60b36d19c5b8c2afc8db4dfd2678d201ed8599bb714bcc.jpg](../icml_results/5127_Native%20Parallel%20Reasoner_%20Reasoning%20in%20Parallelism%20via%20Self-Distilled%20Reinforcement%20Learning/images/b1b7aac30a5358897f60b36d19c5b8c2afc8db4dfd2678d201ed8599bb714bcc.jpg)

### Tables

![0924d40689d41e7b566531a9c63d522f037daa6fe94ddf9f461cd20d454e69e5.jpg](../icml_results/5127_Native%20Parallel%20Reasoner_%20Reasoning%20in%20Parallelism%20via%20Self-Distilled%20Reinforcement%20Learning/tables/0924d40689d41e7b566531a9c63d522f037daa6fe94ddf9f461cd20d454e69e5.jpg)

![0cadc25c4b88541c0709c4bfc2eeb839e32a3aecf02cf26a3293d48fa9005563.jpg](../icml_results/5127_Native%20Parallel%20Reasoner_%20Reasoning%20in%20Parallelism%20via%20Self-Distilled%20Reinforcement%20Learning/tables/0cadc25c4b88541c0709c4bfc2eeb839e32a3aecf02cf26a3293d48fa9005563.jpg)

![1b08f9c393817b0a2dda20ab93e4e92696e8916ef7109ad33605347b21ddd60e.jpg](../icml_results/5127_Native%20Parallel%20Reasoner_%20Reasoning%20in%20Parallelism%20via%20Self-Distilled%20Reinforcement%20Learning/tables/1b08f9c393817b0a2dda20ab93e4e92696e8916ef7109ad33605347b21ddd60e.jpg)

![c701c94d03e8325bd32253fa108df8f08c32166fd28353988ff041e7540e369f.jpg](../icml_results/5127_Native%20Parallel%20Reasoner_%20Reasoning%20in%20Parallelism%20via%20Self-Distilled%20Reinforcement%20Learning/tables/c701c94d03e8325bd32253fa108df8f08c32166fd28353988ff041e7540e369f.jpg)

![e5a1ff021c1ffdf9ba8146af6699cb5c713282202890af5b918cb0e2d1241574.jpg](../icml_results/5127_Native%20Parallel%20Reasoner_%20Reasoning%20in%20Parallelism%20via%20Self-Distilled%20Reinforcement%20Learning/tables/e5a1ff021c1ffdf9ba8146af6699cb5c713282202890af5b918cb0e2d1241574.jpg)

![ed73ab87b3e881712874cf713cad410c74155b83a8a9de825a03cb3355942743.jpg](../icml_results/5127_Native%20Parallel%20Reasoner_%20Reasoning%20in%20Parallelism%20via%20Self-Distilled%20Reinforcement%20Learning/tables/ed73ab87b3e881712874cf713cad410c74155b83a8a9de825a03cb3355942743.jpg)

<a id="paper-5080"></a>

## Matrix-Free GPU Semidefinite Programming for Quantum Ordered Search at the k=6 Frontier


### Images

![0ce46ee2f040555dd59e45b1cf41fb1153360084ca759a653705ce503897ce04.jpg](../icml_results/5128_Matrix-Free%20GPU%20Semidefinite%20Programming%20for%20Quantum%20Ordered%20Search%20at%20the%20k%3D6%20Frontier/images/0ce46ee2f040555dd59e45b1cf41fb1153360084ca759a653705ce503897ce04.jpg)

![5ef881099bfb744642c2fb0f00e05d37cead232c204c1ef5161637b4055c862a.jpg](../icml_results/5128_Matrix-Free%20GPU%20Semidefinite%20Programming%20for%20Quantum%20Ordered%20Search%20at%20the%20k%3D6%20Frontier/images/5ef881099bfb744642c2fb0f00e05d37cead232c204c1ef5161637b4055c862a.jpg)

![6486a61d3943584be91874db4587ec703e1ac2598927016fc97bd96621dde190.jpg](../icml_results/5128_Matrix-Free%20GPU%20Semidefinite%20Programming%20for%20Quantum%20Ordered%20Search%20at%20the%20k%3D6%20Frontier/images/6486a61d3943584be91874db4587ec703e1ac2598927016fc97bd96621dde190.jpg)

![817f3b4fdd16fbfdc4bc8264c35882e15c551d36a6c31e008fcfdb7e62deb8ae.jpg](../icml_results/5128_Matrix-Free%20GPU%20Semidefinite%20Programming%20for%20Quantum%20Ordered%20Search%20at%20the%20k%3D6%20Frontier/images/817f3b4fdd16fbfdc4bc8264c35882e15c551d36a6c31e008fcfdb7e62deb8ae.jpg)

![9273a7aeab01066816c7ea315157b5587ff784ac2a07b1c0ce1bb44296713ebc.jpg](../icml_results/5128_Matrix-Free%20GPU%20Semidefinite%20Programming%20for%20Quantum%20Ordered%20Search%20at%20the%20k%3D6%20Frontier/images/9273a7aeab01066816c7ea315157b5587ff784ac2a07b1c0ce1bb44296713ebc.jpg)

![c9bb688966651830fafa5385bf251b1ddc654010cee32d81f641c70cc83ac032.jpg](../icml_results/5128_Matrix-Free%20GPU%20Semidefinite%20Programming%20for%20Quantum%20Ordered%20Search%20at%20the%20k%3D6%20Frontier/images/c9bb688966651830fafa5385bf251b1ddc654010cee32d81f641c70cc83ac032.jpg)

![d3e3a3cbc4ac45ca1f12683b035599680adc8fc79400fadacfe696f0096780fc.jpg](../icml_results/5128_Matrix-Free%20GPU%20Semidefinite%20Programming%20for%20Quantum%20Ordered%20Search%20at%20the%20k%3D6%20Frontier/images/d3e3a3cbc4ac45ca1f12683b035599680adc8fc79400fadacfe696f0096780fc.jpg)

![dfa76f94ba92e3cafec60ea7a33c9a9d19e105e3dc9250c1c0fc8b982af33973.jpg](../icml_results/5128_Matrix-Free%20GPU%20Semidefinite%20Programming%20for%20Quantum%20Ordered%20Search%20at%20the%20k%3D6%20Frontier/images/dfa76f94ba92e3cafec60ea7a33c9a9d19e105e3dc9250c1c0fc8b982af33973.jpg)

### Tables

![42deb2dfbef5006f1866acaa5136aec0c1f8ed36ea6a5298f81a93bd889e7bf7.jpg](../icml_results/5128_Matrix-Free%20GPU%20Semidefinite%20Programming%20for%20Quantum%20Ordered%20Search%20at%20the%20k%3D6%20Frontier/tables/42deb2dfbef5006f1866acaa5136aec0c1f8ed36ea6a5298f81a93bd889e7bf7.jpg)

![acd7de76b5585984b061c967232f74a81bc2a90e6655e2ebca1a6814aead8ba3.jpg](../icml_results/5128_Matrix-Free%20GPU%20Semidefinite%20Programming%20for%20Quantum%20Ordered%20Search%20at%20the%20k%3D6%20Frontier/tables/acd7de76b5585984b061c967232f74a81bc2a90e6655e2ebca1a6814aead8ba3.jpg)

<a id="paper-5081"></a>

## Flow for Future: Geometric SE(3)-Equivariant Flow Matching for 3D Trajectory Prediction


### Images

![03ddad41060fe29daeacfb39d5a9b02c6b6cfd691fe0987a58b11b0f9676ad74.jpg](../icml_results/5129_Flow%20for%20Future_%20Geometric%20SE%283%29-Equivariant%20Flow%20Matching%20for%203D%20Trajectory%20Prediction/images/03ddad41060fe29daeacfb39d5a9b02c6b6cfd691fe0987a58b11b0f9676ad74.jpg)

![08851e3c265a4950fe55b232f0b336dd070b2a9b0dfb3a86b23a990a072c6674.jpg](../icml_results/5129_Flow%20for%20Future_%20Geometric%20SE%283%29-Equivariant%20Flow%20Matching%20for%203D%20Trajectory%20Prediction/images/08851e3c265a4950fe55b232f0b336dd070b2a9b0dfb3a86b23a990a072c6674.jpg)

![08fb3768bd7310b9bfb69a5ce5b4ee2e218ea754174221ef7dbb57d1c2429fd7.jpg](../icml_results/5129_Flow%20for%20Future_%20Geometric%20SE%283%29-Equivariant%20Flow%20Matching%20for%203D%20Trajectory%20Prediction/images/08fb3768bd7310b9bfb69a5ce5b4ee2e218ea754174221ef7dbb57d1c2429fd7.jpg)

![daf58836e12f5ec16518b857512f74bd447b7c3edf5c5c54cf8d3953b9d3fa9f.jpg](../icml_results/5129_Flow%20for%20Future_%20Geometric%20SE%283%29-Equivariant%20Flow%20Matching%20for%203D%20Trajectory%20Prediction/images/daf58836e12f5ec16518b857512f74bd447b7c3edf5c5c54cf8d3953b9d3fa9f.jpg)

### Tables

![15bad39aa76e9374de558a0cd62a96d5a69834fd44227054b9117e581509b99c.jpg](../icml_results/5129_Flow%20for%20Future_%20Geometric%20SE%283%29-Equivariant%20Flow%20Matching%20for%203D%20Trajectory%20Prediction/tables/15bad39aa76e9374de558a0cd62a96d5a69834fd44227054b9117e581509b99c.jpg)

![2b743c3c6a52bde7e882df9c3b14a4486dbb3e691f11dca29e13b242d1c749cf.jpg](../icml_results/5129_Flow%20for%20Future_%20Geometric%20SE%283%29-Equivariant%20Flow%20Matching%20for%203D%20Trajectory%20Prediction/tables/2b743c3c6a52bde7e882df9c3b14a4486dbb3e691f11dca29e13b242d1c749cf.jpg)

![4191b5eeea695b12fb39ff623cbeebd31dbd3fb8fda1dc11ce051aa57cf6aca5.jpg](../icml_results/5129_Flow%20for%20Future_%20Geometric%20SE%283%29-Equivariant%20Flow%20Matching%20for%203D%20Trajectory%20Prediction/tables/4191b5eeea695b12fb39ff623cbeebd31dbd3fb8fda1dc11ce051aa57cf6aca5.jpg)

![56993d7d3cf8bd33ac168dfbc055a5479cb475d98eb9e990b959689d564de83b.jpg](../icml_results/5129_Flow%20for%20Future_%20Geometric%20SE%283%29-Equivariant%20Flow%20Matching%20for%203D%20Trajectory%20Prediction/tables/56993d7d3cf8bd33ac168dfbc055a5479cb475d98eb9e990b959689d564de83b.jpg)

![689eff33b907d1511d4e113b3ce80414ede2ea3a1646a3cfd74d88234c4d938a.jpg](../icml_results/5129_Flow%20for%20Future_%20Geometric%20SE%283%29-Equivariant%20Flow%20Matching%20for%203D%20Trajectory%20Prediction/tables/689eff33b907d1511d4e113b3ce80414ede2ea3a1646a3cfd74d88234c4d938a.jpg)

![ac22f0fcf980a0e53166ddd8697a6d828086f64dd5a2cc8d43af888b0b2f5b5d.jpg](../icml_results/5129_Flow%20for%20Future_%20Geometric%20SE%283%29-Equivariant%20Flow%20Matching%20for%203D%20Trajectory%20Prediction/tables/ac22f0fcf980a0e53166ddd8697a6d828086f64dd5a2cc8d43af888b0b2f5b5d.jpg)

![f00287219942d6c27c10302e43921c5a848db2833c4c9f46151eac195dbad19d.jpg](../icml_results/5129_Flow%20for%20Future_%20Geometric%20SE%283%29-Equivariant%20Flow%20Matching%20for%203D%20Trajectory%20Prediction/tables/f00287219942d6c27c10302e43921c5a848db2833c4c9f46151eac195dbad19d.jpg)

![f1b622779687e40a600b606bdc7babf42918af008f76b3ef57972e9c110a248b.jpg](../icml_results/5129_Flow%20for%20Future_%20Geometric%20SE%283%29-Equivariant%20Flow%20Matching%20for%203D%20Trajectory%20Prediction/tables/f1b622779687e40a600b606bdc7babf42918af008f76b3ef57972e9c110a248b.jpg)

<a id="paper-5082"></a>

## Can Agents Generalize to the Open World? Unveiling the Fragility of Static Training in Tool Use

### Images

![1ba7e3afdad77d3ab06e15e5189b16c2faa95bf97f923d344efbec3d620ba817.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/1ba7e3afdad77d3ab06e15e5189b16c2faa95bf97f923d344efbec3d620ba817.jpg)

![1e47ec9cd05d2f908c6a2ca234e1f2bfcba1e1cf8128c1cf17f08dca1e6cf0f7.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/1e47ec9cd05d2f908c6a2ca234e1f2bfcba1e1cf8128c1cf17f08dca1e6cf0f7.jpg)

![1e850a2a1edfc07559aa30f47f56866c78fcc095adc3eadf1eddeb261783db91.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/1e850a2a1edfc07559aa30f47f56866c78fcc095adc3eadf1eddeb261783db91.jpg)

![46637e3bd21ff14f04e6d1a545a508b6d771bc2b2d838c69c176009c3174ec8f.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/46637e3bd21ff14f04e6d1a545a508b6d771bc2b2d838c69c176009c3174ec8f.jpg)

![622d617993a8c197d839804da9d5d87e8fbfefc67285aa68527ebc9363f8e072.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/622d617993a8c197d839804da9d5d87e8fbfefc67285aa68527ebc9363f8e072.jpg)

![93df79d4d02b000fdae1ab61e3fc7bf95d7bac878fdd918df06e4f9218f97cf9.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/93df79d4d02b000fdae1ab61e3fc7bf95d7bac878fdd918df06e4f9218f97cf9.jpg)

![9c0832577e55c5dd13eb7d5f1d826e6633b8b5db595d8c6accdee269f0324b06.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/9c0832577e55c5dd13eb7d5f1d826e6633b8b5db595d8c6accdee269f0324b06.jpg)

![ca53372b5251e7bbc24345d9a371f1e5bff4d6bbc01462b5d657a6e0433f1d7e.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/ca53372b5251e7bbc24345d9a371f1e5bff4d6bbc01462b5d657a6e0433f1d7e.jpg)

![dc317663e5cd9e9bf2f5d132325a21298f57e2e7db466e627b1520a04e39ee7b.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/dc317663e5cd9e9bf2f5d132325a21298f57e2e7db466e627b1520a04e39ee7b.jpg)

![f71044d0fa600fe6ce2bef91ff4571eba4095e44e0b9a7e22c7c410bc56e9a94.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/f71044d0fa600fe6ce2bef91ff4571eba4095e44e0b9a7e22c7c410bc56e9a94.jpg)

![fb83694a5905fe0d4362db07ccec6ae4caf23f45c6b72a1043d599a3fa7c7f74.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/fb83694a5905fe0d4362db07ccec6ae4caf23f45c6b72a1043d599a3fa7c7f74.jpg)

### Tables

![0ef5034e025f44ec92b0aa818c5419c83dcd2f15743cde216ceb6af0a6e18732.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/0ef5034e025f44ec92b0aa818c5419c83dcd2f15743cde216ceb6af0a6e18732.jpg)

![112570d5e5f0dc26f4f40f9c2be4118c8e949121e8adf35acd861bcc0427a01b.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/112570d5e5f0dc26f4f40f9c2be4118c8e949121e8adf35acd861bcc0427a01b.jpg)

![12a755395589903037d38e049e5f4afa8206efb10ed5bdda168bdb9da1fcd7a0.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/12a755395589903037d38e049e5f4afa8206efb10ed5bdda168bdb9da1fcd7a0.jpg)

![1842af13b4483316610d41cedc443f9c63393b8f7b1d58fa53df3cecf729e8b0.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/1842af13b4483316610d41cedc443f9c63393b8f7b1d58fa53df3cecf729e8b0.jpg)

![1daf72c66e2da4ee6b5d867f0c859a05a224bf78954f95aa7e0a4c91510d4af7.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/1daf72c66e2da4ee6b5d867f0c859a05a224bf78954f95aa7e0a4c91510d4af7.jpg)

![1db5c06c93d6f8b73fc5126c6a45885d90293499729f62473e387b598297f1b3.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/1db5c06c93d6f8b73fc5126c6a45885d90293499729f62473e387b598297f1b3.jpg)

![1e4224db95cfe16e1b066febdbbe5cb9a3c508d3b08a6fc484868eb265a8f82e.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/1e4224db95cfe16e1b066febdbbe5cb9a3c508d3b08a6fc484868eb265a8f82e.jpg)

![2247ab4469d30bab4fa317ed81bdf21a24f08ca063d97c3f2f40191e929e984b.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/2247ab4469d30bab4fa317ed81bdf21a24f08ca063d97c3f2f40191e929e984b.jpg)

![3189b826b1b17418431fca87d3a38381815584318c669f9a9a289556958beadf.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/3189b826b1b17418431fca87d3a38381815584318c669f9a9a289556958beadf.jpg)

![35319cdf66b60d3ec3d5b897e0d9e511f365e042e120b9c5e476178fac3e69e9.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/35319cdf66b60d3ec3d5b897e0d9e511f365e042e120b9c5e476178fac3e69e9.jpg)

![3607c48743215ae51ae1f110887b52d628bc1386eb0c9fc75191c4d656d00029.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/3607c48743215ae51ae1f110887b52d628bc1386eb0c9fc75191c4d656d00029.jpg)

![3d0805fc4a167cd7114098c257a0e73eed2e1e3f47d7f9847d68200492c8f555.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/3d0805fc4a167cd7114098c257a0e73eed2e1e3f47d7f9847d68200492c8f555.jpg)

![3d93b56fa808924ad157790c8c2b1e294e4fe5824ad02d782e10bea4089cbe4a.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/3d93b56fa808924ad157790c8c2b1e294e4fe5824ad02d782e10bea4089cbe4a.jpg)

![40bed2a50b2b111e2096a72fe670137405f49548b9162b30e9871e84c1d5bb76.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/40bed2a50b2b111e2096a72fe670137405f49548b9162b30e9871e84c1d5bb76.jpg)

![4cd3c71297254711ccfcc759fe99030dbd5298144e5a736ac9aab0f34808b070.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/4cd3c71297254711ccfcc759fe99030dbd5298144e5a736ac9aab0f34808b070.jpg)

![527ad707fcf8f6aeaaa36dc226d682e3c97f234be69e41e0da67163ed27020c9.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/527ad707fcf8f6aeaaa36dc226d682e3c97f234be69e41e0da67163ed27020c9.jpg)

![5535caf649563b1d87cd282707a62f1e05f9b201c5ca7f8e08b563070e1f7327.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/5535caf649563b1d87cd282707a62f1e05f9b201c5ca7f8e08b563070e1f7327.jpg)

![5ec4853be9678ebd6435fa9e165b980a5752e9525ee95c12bb11ca63a94740ea.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/5ec4853be9678ebd6435fa9e165b980a5752e9525ee95c12bb11ca63a94740ea.jpg)

![6457235ffc417eb00f0e5ac87848685eb8185a9967cc321327ee1431c49c6fbd.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/6457235ffc417eb00f0e5ac87848685eb8185a9967cc321327ee1431c49c6fbd.jpg)

![68003786df293f66a4d222bf8ebb74f2c2f042884074c0f2cf33b3c527b6d276.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/68003786df293f66a4d222bf8ebb74f2c2f042884074c0f2cf33b3c527b6d276.jpg)

![6917dfaafcc602245c359719b73af2372dce0512d37a52f69109d77732db7281.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/6917dfaafcc602245c359719b73af2372dce0512d37a52f69109d77732db7281.jpg)

![712d34f8c51305a6a8a9c6d8f29bbbd584d5de5b3a5d050c1bb48ec3d257397b.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/712d34f8c51305a6a8a9c6d8f29bbbd584d5de5b3a5d050c1bb48ec3d257397b.jpg)

![7f35523cc6de5c11bd7ff86b6a4bbe5db477a94cfaca72d3a07e170969f0c556.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/7f35523cc6de5c11bd7ff86b6a4bbe5db477a94cfaca72d3a07e170969f0c556.jpg)

![80f9fe2c2920affb266e31e78a1d764f073ef3b3b1e26ca1b6ef22a6216c2394.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/80f9fe2c2920affb266e31e78a1d764f073ef3b3b1e26ca1b6ef22a6216c2394.jpg)

![9aa1ef29de4521386aa1c3a7a4b702c9db64e161b985bc7d7bc6cba4880969f0.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/9aa1ef29de4521386aa1c3a7a4b702c9db64e161b985bc7d7bc6cba4880969f0.jpg)

![a5ac836ff87c5e582feafe0eab16567b6e468a95b33c4de1ca5744edf0cf3461.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/a5ac836ff87c5e582feafe0eab16567b6e468a95b33c4de1ca5744edf0cf3461.jpg)

![b876e3c17102c8cfcdfbcb36e6db8080210b39a16113c9a00e5a6fe0c4b8faef.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/b876e3c17102c8cfcdfbcb36e6db8080210b39a16113c9a00e5a6fe0c4b8faef.jpg)

![d085c86a0114d456e3ae23aab28bf442551e4f220d3c2e37e218a55addef3906.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/d085c86a0114d456e3ae23aab28bf442551e4f220d3c2e37e218a55addef3906.jpg)

![d2940905ed1bb5ee590605673af16e980651a85554aae0336dadb8ad3ba4c12d.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/d2940905ed1bb5ee590605673af16e980651a85554aae0336dadb8ad3ba4c12d.jpg)

![dc90c42308cf808c96264ac64d489215b4e883b988d1c6d7e08e31d4a2c492bd.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/dc90c42308cf808c96264ac64d489215b4e883b988d1c6d7e08e31d4a2c492bd.jpg)

![de21235675fb37bf26ec2196ee9ea2b771d08c966948b7c3ec8e6b3735543273.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/de21235675fb37bf26ec2196ee9ea2b771d08c966948b7c3ec8e6b3735543273.jpg)

![f2ab140710684ade97b0ebd2162f4abe87c85cce57721e9255a1824958323f9f.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/f2ab140710684ade97b0ebd2162f4abe87c85cce57721e9255a1824958323f9f.jpg)

![f3169daf095bf0f57428b1a8136230d384d2f394cd24b8e23ea8608ec255f3b8.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/f3169daf095bf0f57428b1a8136230d384d2f394cd24b8e23ea8608ec255f3b8.jpg)
