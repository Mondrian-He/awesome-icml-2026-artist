# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 121 of 197

## 目录 (Table of Contents)

1. [MCCE: A Framework for Multi-LLM Collaborative Search in Discrete Spaces with Similarity-Filtered Preference Learning](#paper-3961)
2. [KernelBand: Steering LLM-based Kernel Optimization via Hardware-Aware Multi-Armed Bandits](#paper-3962)
3. [Differentially Private Cross-Silo Recommendation from Implicit Feedback](#paper-3963)
4. [Pair2Scene: Learning Local Object Relations for Procedural Scene Generation](#paper-3964)
5. [Learning Discrete Diffusion on Graphs via Free-Energy Gradient Flows](#paper-3965)
6. [PostTrainBench: Can LLM Agents Automate LLM Post-Training?](#paper-3966)
7. [From Lyapunov Analysis to Algorithm Design in two-sided PL Minimax Optimization](#paper-3967)
8. [Localized, High-resolution Geographic Representations with Slepian Functions](#paper-3968)
9. [A Regime-Aware Trajectory Prediction Framework for 1000+ Systems Biology Models](#paper-3969)
10. [On the Generalization in Topology Optimization via Sensitivity-Conditioned Bernoulli Flow Matching](#paper-3970)
11. [Factored Value Functions for Graph-Based Multi-Agent Reinforcement Learning](#paper-3971)
12. [No More, No Less: Least-Privilege Language Models](#paper-3972)
13. [$τ$-Voice: Benchmarking Full-Duplex Voice Agents on Real-World Domains](#paper-3973)
14. [AutoNumerics-Zero: Automated Discovery of State-of-the-Art Mathematical Functions](#paper-3974)
15. [Learning to Watermark in the Latent Space of Generative Models](#paper-3975)
16. [The Hidden Cost of Structured Generation in LLMs: Draft-Conditioned Constrained Decoding](#paper-3976)
17. [AdaSCALE: Adaptive Scaling for OOD Detection](#paper-3977)
18. [Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization](#paper-3978)
19. [Leak@$k$: Unlearning Does Not Make LLMs Forget Under Probabilistic Decoding](#paper-3979)
20. [Alterbute: Editing Intrinsic Attributes of Objects in Images](#paper-3980)
21. [Correct Looks Better: Pairwise Comparisons Reveal Accuracy Rankings](#paper-3981)
22. [Fisher-Preserving Guidance: Training-Free Manifold Constraints for Safe Diffusion Control](#paper-3982)
23. [iGRPO: Fast Online RL for Flow Matching Model with Instant Reward](#paper-3983)
24. [Unbiased Principles, Robust Rewards](#paper-3984)
25. [Towards Reliable Marking and Verification of AI-Generated Text via Geometry-aware Sentence-level Watermarking](#paper-3985)
26. [Streaming Covariate Balancing via Discrepancy-Based Feature Coresets](#paper-3986)
27. [Evaluating and Rewarding LALMs for Expressive Role-Play TTS via Mean Continuation Log-Probability](#paper-3987)
28. [Low-dimensional topology of deep neural networks](#paper-3988)
29. [Powerful and Theoretically Guaranteed Independence Testing on Heterogeneous Federated Clients](#paper-3989)
30. [MIND: Decoupling Model-Induced Label Noise via Latent Manifold Disentanglement](#paper-3990)
31. [Transfer Learning in Nonparametric Regression with Deep ReLU Networks](#paper-3991)
32. [POLCA: Stochastic Generative Optimization with LLM](#paper-3992)
33. [Riemannian Dueling Optimization](#paper-3993)

---


<a id="paper-3961"></a>

## MCCE: A Framework for Multi-LLM Collaborative Search in Discrete Spaces with Similarity-Filtered Preference Learning

### Images

![0dc83b2afeb144e605e4d45704132a56249f76632a9c9d2f192ff677ddad47d9.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/0dc83b2afeb144e605e4d45704132a56249f76632a9c9d2f192ff677ddad47d9.jpg)

![0ee86dd44f9c8a484b31d6e58ab56b0df8d07469aaa54462dc0a60da800c3933.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/0ee86dd44f9c8a484b31d6e58ab56b0df8d07469aaa54462dc0a60da800c3933.jpg)

![1f1ff3d3ed1e5590f7ec2eaad10653a0f434c48932fd179b067d62b82ce24b3e.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/1f1ff3d3ed1e5590f7ec2eaad10653a0f434c48932fd179b067d62b82ce24b3e.jpg)

![226fce7e228797116dc8253be7c5dba0d32fa5f37f0eaad55f4ab1a761c0e3fe.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/226fce7e228797116dc8253be7c5dba0d32fa5f37f0eaad55f4ab1a761c0e3fe.jpg)

![4ca22b12fe9b62c8e1328b7cf4a97bf312c919a52701583b0b062c020af537df.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/4ca22b12fe9b62c8e1328b7cf4a97bf312c919a52701583b0b062c020af537df.jpg)

![59102f031a1bbb018276b8f90a69700bf8803fc767606c62db046783572882ae.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/59102f031a1bbb018276b8f90a69700bf8803fc767606c62db046783572882ae.jpg)

![5ffbd3073667bdaa397a8d3a3da05a8ce17954e0a71ab7de1959a3b4536896af.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/5ffbd3073667bdaa397a8d3a3da05a8ce17954e0a71ab7de1959a3b4536896af.jpg)

![8cf9cd7e371ffe6ed133f2c05b4ff363120cffa26ca04164180c175f9089abce.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/8cf9cd7e371ffe6ed133f2c05b4ff363120cffa26ca04164180c175f9089abce.jpg)

![932609f5223a9447c14f85ca57cf0ddc2183d6279728e7b0032d2b3819719830.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/932609f5223a9447c14f85ca57cf0ddc2183d6279728e7b0032d2b3819719830.jpg)

![995b1917b8bda035a2f33c49ea3d6e6ff2aec9b4d361c06a8e4e9d85d891e1fa.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/995b1917b8bda035a2f33c49ea3d6e6ff2aec9b4d361c06a8e4e9d85d891e1fa.jpg)

![b10541d01cc46c19dbd453179d2b07f9390bd549fae1804910a6afc92164c6c3.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/b10541d01cc46c19dbd453179d2b07f9390bd549fae1804910a6afc92164c6c3.jpg)

![b78ae16402c84a2d134d6bd4b3d7194d49ccd3a96652341bf570ed3da78b8dc7.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/b78ae16402c84a2d134d6bd4b3d7194d49ccd3a96652341bf570ed3da78b8dc7.jpg)

![ccc34371768e3303bad0c4ccad8108427d4a9cdae3dcc51c34a17a87285b30e2.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/images/ccc34371768e3303bad0c4ccad8108427d4a9cdae3dcc51c34a17a87285b30e2.jpg)

### Tables

![00b45f687a79b0a7ab4abdcfefe656fe95def6020a8215f5411f115e7b0f11ac.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/00b45f687a79b0a7ab4abdcfefe656fe95def6020a8215f5411f115e7b0f11ac.jpg)

![33f59155efd9caab58ec5b8992fa9030736b5bf9c70fae856014f624c2027282.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/33f59155efd9caab58ec5b8992fa9030736b5bf9c70fae856014f624c2027282.jpg)

![498368de8a67050458f54e1a70ef52b2350a30481b7d9f1e48adcb3f49535469.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/498368de8a67050458f54e1a70ef52b2350a30481b7d9f1e48adcb3f49535469.jpg)

![4bf87f59281172b7b145c691662d55ac03275dfa869174bd9a6ddf779e432028.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/4bf87f59281172b7b145c691662d55ac03275dfa869174bd9a6ddf779e432028.jpg)

![5166ee480e2b17a4d0f5ab50cc3e5271721aabf3eea448276006efb961cd1744.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/5166ee480e2b17a4d0f5ab50cc3e5271721aabf3eea448276006efb961cd1744.jpg)

![593fb21dbaa575b57ad3a172e3c3da2c8dd6631b88d4c50ffaa827cfca55ae07.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/593fb21dbaa575b57ad3a172e3c3da2c8dd6631b88d4c50ffaa827cfca55ae07.jpg)

![6131b45c3ab690279f2d45e7721b908165c41fdf0767c994422df0ca7cfdf1bf.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/6131b45c3ab690279f2d45e7721b908165c41fdf0767c994422df0ca7cfdf1bf.jpg)

![6980baf3b9ebe5cc04ac16e9361c7a3b36437c96b9229ef8a2152fd6a5b517d0.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/6980baf3b9ebe5cc04ac16e9361c7a3b36437c96b9229ef8a2152fd6a5b517d0.jpg)

![863f36f23bb2b5fd7e195f921af63dd14a60137f39d0bbfe8cbba8dff03ebefa.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/863f36f23bb2b5fd7e195f921af63dd14a60137f39d0bbfe8cbba8dff03ebefa.jpg)

![882a9274c1b6662b24ae6b363c98c5f40c7dbf9b3e9aa308e781ad91a50fbc37.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/882a9274c1b6662b24ae6b363c98c5f40c7dbf9b3e9aa308e781ad91a50fbc37.jpg)

![8989ce7b30fa2554a8d888fce9696bba80d4b4279cbbebce3b8b5b80e3751ff3.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/8989ce7b30fa2554a8d888fce9696bba80d4b4279cbbebce3b8b5b80e3751ff3.jpg)

![8a20104c47c9f4b4317dcdd3dd3bcd82eff587bfbd8952b5cc99d0be2f9c3f03.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/8a20104c47c9f4b4317dcdd3dd3bcd82eff587bfbd8952b5cc99d0be2f9c3f03.jpg)

![9fbeb20d3fc66a7a603fd41c55f244e4bd77c8b14a8763054e7c04af8c883c1a.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/9fbeb20d3fc66a7a603fd41c55f244e4bd77c8b14a8763054e7c04af8c883c1a.jpg)

![ad277dac398a749085e4cb45695ccfc26612dc6d7e07109ee6f66e822cd05332.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/ad277dac398a749085e4cb45695ccfc26612dc6d7e07109ee6f66e822cd05332.jpg)

![b6cf10935ffebe8cd8909c562f88cee0be5d4964f42ab45d6884731c5cf5b051.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/b6cf10935ffebe8cd8909c562f88cee0be5d4964f42ab45d6884731c5cf5b051.jpg)

![b7c83770df3b7559d0671d5ab6b9c3dd817530d395eb4a93b925a6bd2d441374.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/b7c83770df3b7559d0671d5ab6b9c3dd817530d395eb4a93b925a6bd2d441374.jpg)

![bf51826c4574641135ac0b0177bd6fc4b61e321067dad91ef7bc6c3328dfe994.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/bf51826c4574641135ac0b0177bd6fc4b61e321067dad91ef7bc6c3328dfe994.jpg)

![d93226d7ed1fcad660f2e85eacae4c898cc732f6e34b55c150b3725d77d0fd96.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/d93226d7ed1fcad660f2e85eacae4c898cc732f6e34b55c150b3725d77d0fd96.jpg)

![f43bdb56a0186c4173756058755087a1f692306413056b8df3e1e5bca9885bb3.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/f43bdb56a0186c4173756058755087a1f692306413056b8df3e1e5bca9885bb3.jpg)

![fe827174a65c1322a26ee35f49f50d37eafa77ad0522c75fc4c6feca1c54efa0.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/fe827174a65c1322a26ee35f49f50d37eafa77ad0522c75fc4c6feca1c54efa0.jpg)

![ffb21fbafb1038a4a702489b983670d4b243d63434746faaf8e70c733b063203.jpg](../icml_results/4001_TriForces_%20Augmenting%20Atomistic%20GNNs%20for%20Transferable%20Representations/tables/ffb21fbafb1038a4a702489b983670d4b243d63434746faaf8e70c733b063203.jpg)

## MCCE: A Framework for Multi-LLM Collaborative Search in Discrete Spaces with Similarity-Filtered Preference Learning


### Images

![50e02f435fcab050a68a3213289cce0fa2ab5a0f47ad78e6e805d2152e07c857.jpg](../icml_results/4002_MCCE_%20A%20Framework%20for%20Multi-LLM%20Collaborative%20Search%20in%20Discrete%20Spaces%20with%20Similarity-Filtered%20Pre/images/50e02f435fcab050a68a3213289cce0fa2ab5a0f47ad78e6e805d2152e07c857.jpg)

![744eaec9aea0d44ec39b093730e9112068482a18c18e87dcb5dce408f5683637.jpg](../icml_results/4002_MCCE_%20A%20Framework%20for%20Multi-LLM%20Collaborative%20Search%20in%20Discrete%20Spaces%20with%20Similarity-Filtered%20Pre/images/744eaec9aea0d44ec39b093730e9112068482a18c18e87dcb5dce408f5683637.jpg)

![a2c5bebb706e54cddc0f7dfab96a9fa807fe61d0fd0557fbdd72f75bec9c35ed.jpg](../icml_results/4002_MCCE_%20A%20Framework%20for%20Multi-LLM%20Collaborative%20Search%20in%20Discrete%20Spaces%20with%20Similarity-Filtered%20Pre/images/a2c5bebb706e54cddc0f7dfab96a9fa807fe61d0fd0557fbdd72f75bec9c35ed.jpg)

![ac66b8be3399400004b6156252c8a9b69a9a690eb690f37c0759c45764ef8620.jpg](../icml_results/4002_MCCE_%20A%20Framework%20for%20Multi-LLM%20Collaborative%20Search%20in%20Discrete%20Spaces%20with%20Similarity-Filtered%20Pre/images/ac66b8be3399400004b6156252c8a9b69a9a690eb690f37c0759c45764ef8620.jpg)

![d1cd3487223ed3255babd97c97e3519a821fe9cc48126c28610984c7bd5c2084.jpg](../icml_results/4002_MCCE_%20A%20Framework%20for%20Multi-LLM%20Collaborative%20Search%20in%20Discrete%20Spaces%20with%20Similarity-Filtered%20Pre/images/d1cd3487223ed3255babd97c97e3519a821fe9cc48126c28610984c7bd5c2084.jpg)

![fa05ad205684f01945c9d79acc70c99fa9cc156900f5ffbda5d64d336cc13751.jpg](../icml_results/4002_MCCE_%20A%20Framework%20for%20Multi-LLM%20Collaborative%20Search%20in%20Discrete%20Spaces%20with%20Similarity-Filtered%20Pre/images/fa05ad205684f01945c9d79acc70c99fa9cc156900f5ffbda5d64d336cc13751.jpg)

### Tables

![23e7d73034db8457a9144e9129b33de081684692cb31c4d20ce3ccba34aa9a63.jpg](../icml_results/4002_MCCE_%20A%20Framework%20for%20Multi-LLM%20Collaborative%20Search%20in%20Discrete%20Spaces%20with%20Similarity-Filtered%20Pre/tables/23e7d73034db8457a9144e9129b33de081684692cb31c4d20ce3ccba34aa9a63.jpg)

![26ddcecef937e4a6ebde476552c8c88c2a1503e96c8ecc1b88dd19d2d74c4d97.jpg](../icml_results/4002_MCCE_%20A%20Framework%20for%20Multi-LLM%20Collaborative%20Search%20in%20Discrete%20Spaces%20with%20Similarity-Filtered%20Pre/tables/26ddcecef937e4a6ebde476552c8c88c2a1503e96c8ecc1b88dd19d2d74c4d97.jpg)

![33799c6ac664941eb20060a29ea53c67b73e0e90b413e58aee800f490a2f8ecb.jpg](../icml_results/4002_MCCE_%20A%20Framework%20for%20Multi-LLM%20Collaborative%20Search%20in%20Discrete%20Spaces%20with%20Similarity-Filtered%20Pre/tables/33799c6ac664941eb20060a29ea53c67b73e0e90b413e58aee800f490a2f8ecb.jpg)

![51b47f70401cc468be861deba3e6c357034d0479bb42d4a7bfdf6e87bc29736a.jpg](../icml_results/4002_MCCE_%20A%20Framework%20for%20Multi-LLM%20Collaborative%20Search%20in%20Discrete%20Spaces%20with%20Similarity-Filtered%20Pre/tables/51b47f70401cc468be861deba3e6c357034d0479bb42d4a7bfdf6e87bc29736a.jpg)

![9f2c7ae3ef3f64720dc4e788d95bf5cc172ceb49e0ad7f3889eec4831002ca5f.jpg](../icml_results/4002_MCCE_%20A%20Framework%20for%20Multi-LLM%20Collaborative%20Search%20in%20Discrete%20Spaces%20with%20Similarity-Filtered%20Pre/tables/9f2c7ae3ef3f64720dc4e788d95bf5cc172ceb49e0ad7f3889eec4831002ca5f.jpg)

![abb70cd93a430878f786fd6b343f86a52dc9e1a17ea3db1fb9510edd465bb774.jpg](../icml_results/4002_MCCE_%20A%20Framework%20for%20Multi-LLM%20Collaborative%20Search%20in%20Discrete%20Spaces%20with%20Similarity-Filtered%20Pre/tables/abb70cd93a430878f786fd6b343f86a52dc9e1a17ea3db1fb9510edd465bb774.jpg)

![e2da4c2b9ae8400ba72d802040de257b854d49673ed4296e69c29676a6b38c54.jpg](../icml_results/4002_MCCE_%20A%20Framework%20for%20Multi-LLM%20Collaborative%20Search%20in%20Discrete%20Spaces%20with%20Similarity-Filtered%20Pre/tables/e2da4c2b9ae8400ba72d802040de257b854d49673ed4296e69c29676a6b38c54.jpg)

![fbf02d5ecdfa3feed9f59ebd861b559bd5e0dfc69e9e0185152dd0eba6a872ad.jpg](../icml_results/4002_MCCE_%20A%20Framework%20for%20Multi-LLM%20Collaborative%20Search%20in%20Discrete%20Spaces%20with%20Similarity-Filtered%20Pre/tables/fbf02d5ecdfa3feed9f59ebd861b559bd5e0dfc69e9e0185152dd0eba6a872ad.jpg)

![fea159fdba99599f6719b9141cd8f1445bd172aa11e095894effbe93b8e63ab9.jpg](../icml_results/4002_MCCE_%20A%20Framework%20for%20Multi-LLM%20Collaborative%20Search%20in%20Discrete%20Spaces%20with%20Similarity-Filtered%20Pre/tables/fea159fdba99599f6719b9141cd8f1445bd172aa11e095894effbe93b8e63ab9.jpg)

<a id="paper-3962"></a>

## KernelBand: Steering LLM-based Kernel Optimization via Hardware-Aware Multi-Armed Bandits


### Images

![3f45636b16d287a972b9fa0da24d01e303060ab113295f951d1203e53fa5cb67.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/images/3f45636b16d287a972b9fa0da24d01e303060ab113295f951d1203e53fa5cb67.jpg)

![67c3ed21a0b620942fde887fbe07599a6f9b7cd6febfb606cba7d607d067002f.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/images/67c3ed21a0b620942fde887fbe07599a6f9b7cd6febfb606cba7d607d067002f.jpg)

![cb89f0f42a7992d2b084fb979dcb51e2d0fa5107192e5c0c7457f12be9a33d8e.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/images/cb89f0f42a7992d2b084fb979dcb51e2d0fa5107192e5c0c7457f12be9a33d8e.jpg)

![da0c8eb21f5a0c7c274b1aeb03271abebbf988515af23b0645e63100f51f7176.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/images/da0c8eb21f5a0c7c274b1aeb03271abebbf988515af23b0645e63100f51f7176.jpg)

### Tables

![165de274d4976151051ed35cf5b5e13679b75c8ae9383d909b636a7f65942914.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/tables/165de274d4976151051ed35cf5b5e13679b75c8ae9383d909b636a7f65942914.jpg)

![1d1f89f888acbc4889ee5fe645ef8a46c24d9463fbb18ab48964b12b1b52da1b.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/tables/1d1f89f888acbc4889ee5fe645ef8a46c24d9463fbb18ab48964b12b1b52da1b.jpg)

![2d1626b938d2ffd59278bc026abbdfe198ec0ebf11406603b3c730ce271d3e64.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/tables/2d1626b938d2ffd59278bc026abbdfe198ec0ebf11406603b3c730ce271d3e64.jpg)

![37db42f191687763f376db402fe3ee1598f655cc304af38754807c72d25d8b4e.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/tables/37db42f191687763f376db402fe3ee1598f655cc304af38754807c72d25d8b4e.jpg)

![3977268f88df6adfa5cf76acdd31f4a055ac29a80c351ffd04b3c3e4642ccd96.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/tables/3977268f88df6adfa5cf76acdd31f4a055ac29a80c351ffd04b3c3e4642ccd96.jpg)

![3b489488df0c2e6c7d0e071fa657dc5dc00c193910372554000514dbdd695420.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/tables/3b489488df0c2e6c7d0e071fa657dc5dc00c193910372554000514dbdd695420.jpg)

![517a3fc312793f401b3dde4e2782d3d73e9cbfa47fde25e36cd1527ccc765ec6.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/tables/517a3fc312793f401b3dde4e2782d3d73e9cbfa47fde25e36cd1527ccc765ec6.jpg)

![79edc4df9092c6d51097904ede614ea1abaa8d9b0672a46e0355aa3b1eb2dd06.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/tables/79edc4df9092c6d51097904ede614ea1abaa8d9b0672a46e0355aa3b1eb2dd06.jpg)

![a1e5d84bd71e6037d4ffc1267b277963e2a1603646c9253a2ebf4f70cefab530.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/tables/a1e5d84bd71e6037d4ffc1267b277963e2a1603646c9253a2ebf4f70cefab530.jpg)

![b47a503a265a01a9bc46e4d28b58160e9b10127f49f7002538cced303ed14c1f.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/tables/b47a503a265a01a9bc46e4d28b58160e9b10127f49f7002538cced303ed14c1f.jpg)

![c3fc2d5d5996fcf539597630b430a5d344c08a635cdbeeea2dfe3ad60071997f.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/tables/c3fc2d5d5996fcf539597630b430a5d344c08a635cdbeeea2dfe3ad60071997f.jpg)

![d07d5c75a7746f998aec0928e2c25a4f29e33de9e8191fa15d3963a775b8143a.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/tables/d07d5c75a7746f998aec0928e2c25a4f29e33de9e8191fa15d3963a775b8143a.jpg)

![d4d622c439cd48cf92a9b8a122b1aa194b615a2e3dc84a661187a6522603e05e.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/tables/d4d622c439cd48cf92a9b8a122b1aa194b615a2e3dc84a661187a6522603e05e.jpg)

![dd3b6a384968e14524af8972d4945d045880a27f53289bc49e98a52bca80ee96.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/tables/dd3b6a384968e14524af8972d4945d045880a27f53289bc49e98a52bca80ee96.jpg)

![eb6c57c02c4a7ba2deb40cd2f00b290ccaa7cee3738f87c2e28bcf7eb605400e.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/tables/eb6c57c02c4a7ba2deb40cd2f00b290ccaa7cee3738f87c2e28bcf7eb605400e.jpg)

![ef546824643fa5467baa6ef708296e6f359029acd949178b909ad3ed633084a5.jpg](../icml_results/4003_KernelBand_%20Steering%20LLM-based%20Kernel%20Optimization%20via%20Hardware-Aware%20Multi-Armed%20Bandits/tables/ef546824643fa5467baa6ef708296e6f359029acd949178b909ad3ed633084a5.jpg)

<a id="paper-3963"></a>

## Differentially Private Cross-Silo Recommendation from Implicit Feedback


### Images

![1218f4b3c25146d078dd7c37f39f12609d4895e969e3d6b95f4ec3a7aa6fcd96.jpg](../icml_results/4004_Differentially%20Private%20Cross-Silo%20Recommendation%20from%20Implicit%20Feedback/images/1218f4b3c25146d078dd7c37f39f12609d4895e969e3d6b95f4ec3a7aa6fcd96.jpg)

![189e4acf8eac0c73a85714107137ae7f7ea75f523ec84a501508d0993b42353b.jpg](../icml_results/4004_Differentially%20Private%20Cross-Silo%20Recommendation%20from%20Implicit%20Feedback/images/189e4acf8eac0c73a85714107137ae7f7ea75f523ec84a501508d0993b42353b.jpg)

![207e8d79a54b85b3ec5ae5779ea731ba565a5db3a3ac16c4e4fc46e0973478f3.jpg](../icml_results/4004_Differentially%20Private%20Cross-Silo%20Recommendation%20from%20Implicit%20Feedback/images/207e8d79a54b85b3ec5ae5779ea731ba565a5db3a3ac16c4e4fc46e0973478f3.jpg)

![58387eaca401d382139e9516f867a82c389efec7f291ade261befcdbdb07f25a.jpg](../icml_results/4004_Differentially%20Private%20Cross-Silo%20Recommendation%20from%20Implicit%20Feedback/images/58387eaca401d382139e9516f867a82c389efec7f291ade261befcdbdb07f25a.jpg)

![6fb7acc3da3ed009c273512facd28933003326c73558682c5ee19f85fc4328e1.jpg](../icml_results/4004_Differentially%20Private%20Cross-Silo%20Recommendation%20from%20Implicit%20Feedback/images/6fb7acc3da3ed009c273512facd28933003326c73558682c5ee19f85fc4328e1.jpg)

![72faecc056c4a8bfcf8ce4cde3754e93569182fd6b2416767d722e1297ae4ae8.jpg](../icml_results/4004_Differentially%20Private%20Cross-Silo%20Recommendation%20from%20Implicit%20Feedback/images/72faecc056c4a8bfcf8ce4cde3754e93569182fd6b2416767d722e1297ae4ae8.jpg)

![8503489f605981074fe42f0fafd213d3375b9ff484a40cc0ef1ebaf4e65b40e9.jpg](../icml_results/4004_Differentially%20Private%20Cross-Silo%20Recommendation%20from%20Implicit%20Feedback/images/8503489f605981074fe42f0fafd213d3375b9ff484a40cc0ef1ebaf4e65b40e9.jpg)

![85beab074fced41ef89220db5ebd49467b8da34e5fae1304fc964e2519ddc646.jpg](../icml_results/4004_Differentially%20Private%20Cross-Silo%20Recommendation%20from%20Implicit%20Feedback/images/85beab074fced41ef89220db5ebd49467b8da34e5fae1304fc964e2519ddc646.jpg)

![e368610dd79e1ebfeca97c6d96db4fb660b3abf5d28660724e6c547219fa1440.jpg](../icml_results/4004_Differentially%20Private%20Cross-Silo%20Recommendation%20from%20Implicit%20Feedback/images/e368610dd79e1ebfeca97c6d96db4fb660b3abf5d28660724e6c547219fa1440.jpg)

![faa6750d06bf203cb60636baa2f00455436a2c1f5bd801985f6879625c2b599b.jpg](../icml_results/4004_Differentially%20Private%20Cross-Silo%20Recommendation%20from%20Implicit%20Feedback/images/faa6750d06bf203cb60636baa2f00455436a2c1f5bd801985f6879625c2b599b.jpg)

### Tables

![06c9cc4816e98eb08e3118612e4bd0697a500125d32ad79ba2eeceb2ac1a7b0d.jpg](../icml_results/4004_Differentially%20Private%20Cross-Silo%20Recommendation%20from%20Implicit%20Feedback/tables/06c9cc4816e98eb08e3118612e4bd0697a500125d32ad79ba2eeceb2ac1a7b0d.jpg)

![f0e1402f13fa8691e7dfedf6d21c2a2fc816dafae7cf8a2d19cec0d881a87884.jpg](../icml_results/4004_Differentially%20Private%20Cross-Silo%20Recommendation%20from%20Implicit%20Feedback/tables/f0e1402f13fa8691e7dfedf6d21c2a2fc816dafae7cf8a2d19cec0d881a87884.jpg)

![f317b5751159af9af5979aed0ee2785d17c49db3c74fc0460a86377af3d73cb0.jpg](../icml_results/4004_Differentially%20Private%20Cross-Silo%20Recommendation%20from%20Implicit%20Feedback/tables/f317b5751159af9af5979aed0ee2785d17c49db3c74fc0460a86377af3d73cb0.jpg)

<a id="paper-3964"></a>

## Pair2Scene: Learning Local Object Relations for Procedural Scene Generation


### Images

![05d4045a0f98bbc772d1705276fce92b31964ed68ce57a2a73483cb8a16b61b5.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/05d4045a0f98bbc772d1705276fce92b31964ed68ce57a2a73483cb8a16b61b5.jpg)

![06f5c805d10a693b3398cec7aece1db8ac8efcacdfd3aa46f84da1d8b819d250.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/06f5c805d10a693b3398cec7aece1db8ac8efcacdfd3aa46f84da1d8b819d250.jpg)

![1409e7fbdae9612001dac22a78f4a82f7f405cfc3b73c87519164b47f05fbb08.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/1409e7fbdae9612001dac22a78f4a82f7f405cfc3b73c87519164b47f05fbb08.jpg)

![1a0678f8d8920afe9ba4089fdb954ad1dc4ec3930430bbc4758f8234f014d506.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/1a0678f8d8920afe9ba4089fdb954ad1dc4ec3930430bbc4758f8234f014d506.jpg)

![1aa2ac691c4ab007f7e099bb54a5e9f3456477f7c18976d2bb44620bda1c3d44.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/1aa2ac691c4ab007f7e099bb54a5e9f3456477f7c18976d2bb44620bda1c3d44.jpg)

![410ca4e256abbb13bb8edc7e3ce299d8e4775d53a7a4f6de760126cb8349a1e5.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/410ca4e256abbb13bb8edc7e3ce299d8e4775d53a7a4f6de760126cb8349a1e5.jpg)

![410ebbd19d4ab08ecb487bcd414d01febaa2052d9ed57cdd12346f0c00fb35a7.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/410ebbd19d4ab08ecb487bcd414d01febaa2052d9ed57cdd12346f0c00fb35a7.jpg)

![649c27f87f3b9b4a1d7b8aca2a67bee34afda9bce25959db54c062975ca96a7b.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/649c27f87f3b9b4a1d7b8aca2a67bee34afda9bce25959db54c062975ca96a7b.jpg)

![6a533c03e612dde6eaaa658f839c6950cd0f7151dd4bffa6cf600968be33c6d4.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/6a533c03e612dde6eaaa658f839c6950cd0f7151dd4bffa6cf600968be33c6d4.jpg)

![6cfdb77bc3183a08635d5864cafe32940444593c33f5f24ed0d511ae912d77e0.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/6cfdb77bc3183a08635d5864cafe32940444593c33f5f24ed0d511ae912d77e0.jpg)

![74825feba721aabd915291a3dda07b00c2dc055fc8e5a76174d758e942345c2c.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/74825feba721aabd915291a3dda07b00c2dc055fc8e5a76174d758e942345c2c.jpg)

![7f887e62e0a1f6f742738c72e9309b2cb730c690bc77e20aaa6549516eec3cad.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/7f887e62e0a1f6f742738c72e9309b2cb730c690bc77e20aaa6549516eec3cad.jpg)

![81e014a4046b0d2215b4941ca013a6405156f2f806798e8dd90c7f82f79a8b6f.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/81e014a4046b0d2215b4941ca013a6405156f2f806798e8dd90c7f82f79a8b6f.jpg)

![a6f5934c7aa7e6aab6f829a9317a33f6867ea73fc953ad2c3e484581952e8d4b.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/a6f5934c7aa7e6aab6f829a9317a33f6867ea73fc953ad2c3e484581952e8d4b.jpg)

![b79e6d7c618ab7c329fa398f495ddcf0bf86a44623efb724b1edd9c22d8c3c87.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/b79e6d7c618ab7c329fa398f495ddcf0bf86a44623efb724b1edd9c22d8c3c87.jpg)

![c6fea5c060a4dd532ee2cd50558143b58238a47ae649d27d514744171f5d700e.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/c6fea5c060a4dd532ee2cd50558143b58238a47ae649d27d514744171f5d700e.jpg)

![d9dc3bf99ed3a9541f827a05d2048a2f698f3be80e192e0be04c61b0e27e649a.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/d9dc3bf99ed3a9541f827a05d2048a2f698f3be80e192e0be04c61b0e27e649a.jpg)

![f609a494fcfca325d295751c82e57028144fc3d0160694f5bdeb26dba2e9828b.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/images/f609a494fcfca325d295751c82e57028144fc3d0160694f5bdeb26dba2e9828b.jpg)

### Tables

![0178cfdbaa040c9a43fef2f1c0a164152f17ba3f83b92a2d162d5730aa4d5ac8.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/tables/0178cfdbaa040c9a43fef2f1c0a164152f17ba3f83b92a2d162d5730aa4d5ac8.jpg)

![471fb83959a0029991ecfc39477be8e8258d7275d539e87d61bfbab219920603.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/tables/471fb83959a0029991ecfc39477be8e8258d7275d539e87d61bfbab219920603.jpg)

![83639645dd8b35f54c9e23a2cc4d16ce77115d17cc78eb4577dbea46ec1b99c4.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/tables/83639645dd8b35f54c9e23a2cc4d16ce77115d17cc78eb4577dbea46ec1b99c4.jpg)

![9a47702ef231b0cd2c6e21905b625cb3bfbe65305d8914f83ed46d4a1bfe727a.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/tables/9a47702ef231b0cd2c6e21905b625cb3bfbe65305d8914f83ed46d4a1bfe727a.jpg)

![b08e54496dc3dbe7b1d24f0c20e27cbf4c4afc7eb33ee94c29966038e48da877.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/tables/b08e54496dc3dbe7b1d24f0c20e27cbf4c4afc7eb33ee94c29966038e48da877.jpg)

![b17b47f925b9fee8d088d51277c070a540b444b32e40274973b7b1d9615f4d36.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/tables/b17b47f925b9fee8d088d51277c070a540b444b32e40274973b7b1d9615f4d36.jpg)

![b797644def0d8bc7b16e29e393918c374ddd62c781b9350c6458ab391731e6ca.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/tables/b797644def0d8bc7b16e29e393918c374ddd62c781b9350c6458ab391731e6ca.jpg)

![bbbec5f4b781a5a7da5f71abf9dedd757f29847ab1af1e6b60bf8ea8daed278d.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/tables/bbbec5f4b781a5a7da5f71abf9dedd757f29847ab1af1e6b60bf8ea8daed278d.jpg)

![cdf8770344dbd27bc07d312153db78a74acf3158c97d0379892a15be0816e41f.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/tables/cdf8770344dbd27bc07d312153db78a74acf3158c97d0379892a15be0816e41f.jpg)

![dfdba0d3da0b6ad4af7799bdcc7a739cf5291cdf86ca6bfe14dce96ae533e3a0.jpg](../icml_results/4005_Pair2Scene_%20Learning%20Local%20Object%20Relations%20for%20Procedural%20Scene%20Generation/tables/dfdba0d3da0b6ad4af7799bdcc7a739cf5291cdf86ca6bfe14dce96ae533e3a0.jpg)

<a id="paper-3965"></a>

## Learning Discrete Diffusion on Graphs via Free-Energy Gradient Flows


### Images

![2102c163bcc9ebc940a6062a6e91f23383fac51bbb1d04080e922fc07ee71fa3.jpg](../icml_results/4006_Learning%20Discrete%20Diffusion%20on%20Graphs%20via%20Free-Energy%20Gradient%20Flows/images/2102c163bcc9ebc940a6062a6e91f23383fac51bbb1d04080e922fc07ee71fa3.jpg)

![40432098b70b40622125124274ee5e63306120f446f45ef8ffb38cc4231c0f8a.jpg](../icml_results/4006_Learning%20Discrete%20Diffusion%20on%20Graphs%20via%20Free-Energy%20Gradient%20Flows/images/40432098b70b40622125124274ee5e63306120f446f45ef8ffb38cc4231c0f8a.jpg)

![422ae4b1de913e2a4aad6a2c9ff7fd07c299e0bb23a5babd0bedaecae2dcf5f5.jpg](../icml_results/4006_Learning%20Discrete%20Diffusion%20on%20Graphs%20via%20Free-Energy%20Gradient%20Flows/images/422ae4b1de913e2a4aad6a2c9ff7fd07c299e0bb23a5babd0bedaecae2dcf5f5.jpg)

![c74f99085cf31479af52aec8fcadc08189f618ef7b3b48236e4d53d3a8a98249.jpg](../icml_results/4006_Learning%20Discrete%20Diffusion%20on%20Graphs%20via%20Free-Energy%20Gradient%20Flows/images/c74f99085cf31479af52aec8fcadc08189f618ef7b3b48236e4d53d3a8a98249.jpg)

![c7c28bc4ccdfaf6aeee285a83337990a0924c5105f8088ad7b251d4750b96369.jpg](../icml_results/4006_Learning%20Discrete%20Diffusion%20on%20Graphs%20via%20Free-Energy%20Gradient%20Flows/images/c7c28bc4ccdfaf6aeee285a83337990a0924c5105f8088ad7b251d4750b96369.jpg)

![dcfa95fa6b7815b0cc484b501290a1e796840a9e36ba43ca858cdb023b182f47.jpg](../icml_results/4006_Learning%20Discrete%20Diffusion%20on%20Graphs%20via%20Free-Energy%20Gradient%20Flows/images/dcfa95fa6b7815b0cc484b501290a1e796840a9e36ba43ca858cdb023b182f47.jpg)

### Tables

![261a99673672a0814b0d336b1f630247b6cb17d8b8956dddbb29a169f9e9c22a.jpg](../icml_results/4006_Learning%20Discrete%20Diffusion%20on%20Graphs%20via%20Free-Energy%20Gradient%20Flows/tables/261a99673672a0814b0d336b1f630247b6cb17d8b8956dddbb29a169f9e9c22a.jpg)

![694b8095370cf7f367934ce0ac1aff2cd47adf17224e70967acb4ee127ea1e8a.jpg](../icml_results/4006_Learning%20Discrete%20Diffusion%20on%20Graphs%20via%20Free-Energy%20Gradient%20Flows/tables/694b8095370cf7f367934ce0ac1aff2cd47adf17224e70967acb4ee127ea1e8a.jpg)

<a id="paper-3966"></a>

## PostTrainBench: Can LLM Agents Automate LLM Post-Training?


### Images

![41c1f49f39b0d438d7330272e0a9decba7baead598a2c431ebaeb89901289f14.jpg](../icml_results/4007_PostTrainBench_%20Can%20LLM%20Agents%20Automate%20LLM%20Post-Training_/images/41c1f49f39b0d438d7330272e0a9decba7baead598a2c431ebaeb89901289f14.jpg)

![acc60adf886f9ad57204a43fc2d50ba432567dc2bc812bcee123d8c84a843f66.jpg](../icml_results/4007_PostTrainBench_%20Can%20LLM%20Agents%20Automate%20LLM%20Post-Training_/images/acc60adf886f9ad57204a43fc2d50ba432567dc2bc812bcee123d8c84a843f66.jpg)

![bfafedc263adee5fe328c9ba047597002333ff189c25c0f0e6e5d417f0147555.jpg](../icml_results/4007_PostTrainBench_%20Can%20LLM%20Agents%20Automate%20LLM%20Post-Training_/images/bfafedc263adee5fe328c9ba047597002333ff189c25c0f0e6e5d417f0147555.jpg)

![ce32b167284c04858664193b037d989774b49c8468499b9d49f966fe364ce084.jpg](../icml_results/4007_PostTrainBench_%20Can%20LLM%20Agents%20Automate%20LLM%20Post-Training_/images/ce32b167284c04858664193b037d989774b49c8468499b9d49f966fe364ce084.jpg)

![d6003e78fa2c7f24048acf667540ed58165ddd48887b3cfb63b7ef5439b57459.jpg](../icml_results/4007_PostTrainBench_%20Can%20LLM%20Agents%20Automate%20LLM%20Post-Training_/images/d6003e78fa2c7f24048acf667540ed58165ddd48887b3cfb63b7ef5439b57459.jpg)

![d8fb512698897849a91863034ffc76f52813dee93f322ed196d838d767fb2857.jpg](../icml_results/4007_PostTrainBench_%20Can%20LLM%20Agents%20Automate%20LLM%20Post-Training_/images/d8fb512698897849a91863034ffc76f52813dee93f322ed196d838d767fb2857.jpg)

### Tables

![1ff762ad96d348d561e1cc1a7ddafcd4c7795e4aa3c6834275e08143a01320d0.jpg](../icml_results/4007_PostTrainBench_%20Can%20LLM%20Agents%20Automate%20LLM%20Post-Training_/tables/1ff762ad96d348d561e1cc1a7ddafcd4c7795e4aa3c6834275e08143a01320d0.jpg)

![b07f1a7f873749896565e778bcbd955153f5d9f352fb3bc681c28756da22c6f2.jpg](../icml_results/4007_PostTrainBench_%20Can%20LLM%20Agents%20Automate%20LLM%20Post-Training_/tables/b07f1a7f873749896565e778bcbd955153f5d9f352fb3bc681c28756da22c6f2.jpg)

![ca3deeef602b26260e4e3cebb8ee49ce54dce832bbafdf24ad73ef0b9b3dcbc5.jpg](../icml_results/4007_PostTrainBench_%20Can%20LLM%20Agents%20Automate%20LLM%20Post-Training_/tables/ca3deeef602b26260e4e3cebb8ee49ce54dce832bbafdf24ad73ef0b9b3dcbc5.jpg)

![d11b228b9e2844503badc7b21e1464deb9664534d6c528455db97f98a887cb0a.jpg](../icml_results/4007_PostTrainBench_%20Can%20LLM%20Agents%20Automate%20LLM%20Post-Training_/tables/d11b228b9e2844503badc7b21e1464deb9664534d6c528455db97f98a887cb0a.jpg)

![d9776100a576df7c84ec7ec2e41e2625d76f0f86f1d33348fcfefb4b7cc3713e.jpg](../icml_results/4007_PostTrainBench_%20Can%20LLM%20Agents%20Automate%20LLM%20Post-Training_/tables/d9776100a576df7c84ec7ec2e41e2625d76f0f86f1d33348fcfefb4b7cc3713e.jpg)

![e3f60f29a8e1d4882ebeda4c8b0b12b1f3ddc63d5293567d8050c987e6ca9ed8.jpg](../icml_results/4007_PostTrainBench_%20Can%20LLM%20Agents%20Automate%20LLM%20Post-Training_/tables/e3f60f29a8e1d4882ebeda4c8b0b12b1f3ddc63d5293567d8050c987e6ca9ed8.jpg)

![e97424944b19b5d3fef5f46f2484b436ea8a98661093911673ddb9a9852a40eb.jpg](../icml_results/4007_PostTrainBench_%20Can%20LLM%20Agents%20Automate%20LLM%20Post-Training_/tables/e97424944b19b5d3fef5f46f2484b436ea8a98661093911673ddb9a9852a40eb.jpg)

<a id="paper-3967"></a>

## From Lyapunov Analysis to Algorithm Design in two-sided PL Minimax Optimization


### Images

![bcea11d8659483b62e1974f3f893c27d3828e1a0e6a2f39d84ab8cab16c7f433.jpg](../icml_results/4008_From%20Lyapunov%20Analysis%20to%20Algorithm%20Design%20in%20two-sided%20PL%20Minimax%20Optimization/images/bcea11d8659483b62e1974f3f893c27d3828e1a0e6a2f39d84ab8cab16c7f433.jpg)

![e4e6774289734f1db4b6c8c8c5470b4ae24a59a5d301b01cd45bd9bab21b41f2.jpg](../icml_results/4008_From%20Lyapunov%20Analysis%20to%20Algorithm%20Design%20in%20two-sided%20PL%20Minimax%20Optimization/images/e4e6774289734f1db4b6c8c8c5470b4ae24a59a5d301b01cd45bd9bab21b41f2.jpg)

### Tables

![ba56af00d5674066e60e53b12551d0d5e4c7924b2ca9669fb85623795fd7d82d.jpg](../icml_results/4008_From%20Lyapunov%20Analysis%20to%20Algorithm%20Design%20in%20two-sided%20PL%20Minimax%20Optimization/tables/ba56af00d5674066e60e53b12551d0d5e4c7924b2ca9669fb85623795fd7d82d.jpg)

![d4a083b89bc47aa76dfb44fc9d72c2dc03080dce8a5ea6509f306aa349b4253d.jpg](../icml_results/4008_From%20Lyapunov%20Analysis%20to%20Algorithm%20Design%20in%20two-sided%20PL%20Minimax%20Optimization/tables/d4a083b89bc47aa76dfb44fc9d72c2dc03080dce8a5ea6509f306aa349b4253d.jpg)

<a id="paper-3968"></a>

## Localized, High-resolution Geographic Representations with Slepian Functions


### Images

![0fbf1020509f4be24f8e354cc8dda37c36ab3f17ba30d16fd5b5fdc1f1a3f2a2.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/images/0fbf1020509f4be24f8e354cc8dda37c36ab3f17ba30d16fd5b5fdc1f1a3f2a2.jpg)

![261d8756df34f054c567dddfc3421c9d404017b934bc2365c217a79ce9d50480.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/images/261d8756df34f054c567dddfc3421c9d404017b934bc2365c217a79ce9d50480.jpg)

![3d2542b77cd447c42035024c3feff0739bb70aab9d52a866f6a30f60a60dddf9.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/images/3d2542b77cd447c42035024c3feff0739bb70aab9d52a866f6a30f60a60dddf9.jpg)

![50a64e3a60b95ae352121fcf0fab0c4d83d0ab8950bf47294cc7ebe0ae64a043.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/images/50a64e3a60b95ae352121fcf0fab0c4d83d0ab8950bf47294cc7ebe0ae64a043.jpg)

![57b953a78672c092c99582f543cc05a0368f959f21213dcd51ca7026cef6e08a.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/images/57b953a78672c092c99582f543cc05a0368f959f21213dcd51ca7026cef6e08a.jpg)

![6cb7a94f16a7bdcc8fcaf2d3c7d3c4afe53adf65d77e276761a155bfad67198e.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/images/6cb7a94f16a7bdcc8fcaf2d3c7d3c4afe53adf65d77e276761a155bfad67198e.jpg)

![70551b4ed7bccf5d99d0ffb3ddc693f83dd8fad20d26a6ce76a3a86d72f186ee.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/images/70551b4ed7bccf5d99d0ffb3ddc693f83dd8fad20d26a6ce76a3a86d72f186ee.jpg)

![a20a803d54b852b8937deed9e87508087b04e6c3d2de801dbab1464e85794fa2.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/images/a20a803d54b852b8937deed9e87508087b04e6c3d2de801dbab1464e85794fa2.jpg)

![b2f294d8c8b63b608383c1728c887b2f66f6328bb713d99ae22cbf5a92d40441.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/images/b2f294d8c8b63b608383c1728c887b2f66f6328bb713d99ae22cbf5a92d40441.jpg)

![be5fdd2a3eb126e11a6f4c51f89972792df85ccacbc25acfd6fc9c348e3e33e7.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/images/be5fdd2a3eb126e11a6f4c51f89972792df85ccacbc25acfd6fc9c348e3e33e7.jpg)

![d9ebe0b7ffef117c27db5e06b43ef52c8333151aa62048b6441c7e3d455c63fc.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/images/d9ebe0b7ffef117c27db5e06b43ef52c8333151aa62048b6441c7e3d455c63fc.jpg)

![ea7928a4fdcb0fa7279f85e96e4ed05b489e2a94cc9c885044b81817f4ac1ac4.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/images/ea7928a4fdcb0fa7279f85e96e4ed05b489e2a94cc9c885044b81817f4ac1ac4.jpg)

### Tables

![0724fde8e813aa30bb8df01c5029469213cd5b45fffad2904fd06ec8053e2258.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/tables/0724fde8e813aa30bb8df01c5029469213cd5b45fffad2904fd06ec8053e2258.jpg)

![0dd184f745d2fd76dea1c241ad0f7a73aa241879d88ba1437ad4d44a26bf93df.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/tables/0dd184f745d2fd76dea1c241ad0f7a73aa241879d88ba1437ad4d44a26bf93df.jpg)

![54316be3b65ceb7187ccc9e326819a59c7e0a76fd82355c4ad161a858a7e7d17.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/tables/54316be3b65ceb7187ccc9e326819a59c7e0a76fd82355c4ad161a858a7e7d17.jpg)

![60cbc6c6d9972e13cfe7578c2f178f0d546184f7dcfd8c6a1715dccf405738b3.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/tables/60cbc6c6d9972e13cfe7578c2f178f0d546184f7dcfd8c6a1715dccf405738b3.jpg)

![68e8cb5e046608ddbc5ec2f57ae73f0b5243dacca79199cf09ee0b714800c292.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/tables/68e8cb5e046608ddbc5ec2f57ae73f0b5243dacca79199cf09ee0b714800c292.jpg)

![80e02fbb6035f2cfa5845bb4f07bef9bd250781aba3bc289e5b0bf4777a68e8f.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/tables/80e02fbb6035f2cfa5845bb4f07bef9bd250781aba3bc289e5b0bf4777a68e8f.jpg)

![b93087ca94427bd990a433b58685fc096b1fce9a5da9be7ef46785a70e699d75.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/tables/b93087ca94427bd990a433b58685fc096b1fce9a5da9be7ef46785a70e699d75.jpg)

![bac86f3fa6a3b87afe4dc5422239451fa691912192e2d90ba71c8f8ad6dd6568.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/tables/bac86f3fa6a3b87afe4dc5422239451fa691912192e2d90ba71c8f8ad6dd6568.jpg)

![ef162c51bb1662fde6fad3f2791fbe5d58d4a8baaf649c96a6f41f5c9996818f.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/tables/ef162c51bb1662fde6fad3f2791fbe5d58d4a8baaf649c96a6f41f5c9996818f.jpg)

![ff363132d9146c44643d9d9355bdb21051f8e3c9e2dbd557784691800610b81f.jpg](../icml_results/4009_Localized%2C%20High-resolution%20Geographic%20Representations%20with%20Slepian%20Functions/tables/ff363132d9146c44643d9d9355bdb21051f8e3c9e2dbd557784691800610b81f.jpg)

<a id="paper-3969"></a>

## A Regime-Aware Trajectory Prediction Framework for 1000+ Systems Biology Models


### Images

![0a691d7857ec84dd60526d27afa361913158ef2d714227365681f5e5c6652b25.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/images/0a691d7857ec84dd60526d27afa361913158ef2d714227365681f5e5c6652b25.jpg)

![230714dfe6c6601231a69f7b81d4a64bc2e84ce12d835603d42c0f4f10596bc1.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/images/230714dfe6c6601231a69f7b81d4a64bc2e84ce12d835603d42c0f4f10596bc1.jpg)

![2e5633499676d8c7d4ddf67cd50ea1c1f514c6da872b04ec6fed509f865f2892.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/images/2e5633499676d8c7d4ddf67cd50ea1c1f514c6da872b04ec6fed509f865f2892.jpg)

![386ca164f4114b9c7873e3349789cf8e628549d3bc66121d97ff42722c70557c.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/images/386ca164f4114b9c7873e3349789cf8e628549d3bc66121d97ff42722c70557c.jpg)

![62ca175405ae70f9ad4e1a43ea87aec63b5c5e61c6ebe9b89bd8245928973da8.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/images/62ca175405ae70f9ad4e1a43ea87aec63b5c5e61c6ebe9b89bd8245928973da8.jpg)

![72a0758a20401ddd09da8a89b5900dea1ac6b680755f4cc7b51be66d85d97d2f.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/images/72a0758a20401ddd09da8a89b5900dea1ac6b680755f4cc7b51be66d85d97d2f.jpg)

![75a205aeee07654c8474baff2d12345b416ad83d8ec1eb2ce18bb8ac21b60220.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/images/75a205aeee07654c8474baff2d12345b416ad83d8ec1eb2ce18bb8ac21b60220.jpg)

![7a093a915b291b028fb5d27d479e62581f831676a94bc884f2bfe9639dab176f.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/images/7a093a915b291b028fb5d27d479e62581f831676a94bc884f2bfe9639dab176f.jpg)

![8c845f29cff08a829cf98bd972ef53416fcf8ebb096079beeab9c7ea902b975d.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/images/8c845f29cff08a829cf98bd972ef53416fcf8ebb096079beeab9c7ea902b975d.jpg)

![ad3893baa3974f0fcf7f03b3f5cf42f4a49348fa8b247ac08c0ea6f5f3001c4f.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/images/ad3893baa3974f0fcf7f03b3f5cf42f4a49348fa8b247ac08c0ea6f5f3001c4f.jpg)

![b60f3cf0fc3991885cae374cfd2d31275919132a9c8741cce8003ef9cd78d3d3.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/images/b60f3cf0fc3991885cae374cfd2d31275919132a9c8741cce8003ef9cd78d3d3.jpg)

![bb6ecaf6ff8e250d1b018ec3995df9f65466fba6a98c50911ad5ea86fc8e4983.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/images/bb6ecaf6ff8e250d1b018ec3995df9f65466fba6a98c50911ad5ea86fc8e4983.jpg)

![d3977a2b02da46f1a04d5864b8bd927557c327caa1e38c888501bc5353fbdda2.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/images/d3977a2b02da46f1a04d5864b8bd927557c327caa1e38c888501bc5353fbdda2.jpg)

![d6246deec663be04108fcca7c5a65a9f2dcf05796a9fa64cf2fee0ada7b9bfbf.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/images/d6246deec663be04108fcca7c5a65a9f2dcf05796a9fa64cf2fee0ada7b9bfbf.jpg)

![ed3a4d552dfaae84e9ae34ca7a8bc7ee1ee5fe0ce333528814628b24b926d019.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/images/ed3a4d552dfaae84e9ae34ca7a8bc7ee1ee5fe0ce333528814628b24b926d019.jpg)

### Tables

![107e6b790434f24c2e106de437d6204ab9ce0551b210effdc7929b76ae37b2c5.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/tables/107e6b790434f24c2e106de437d6204ab9ce0551b210effdc7929b76ae37b2c5.jpg)

![136137b012d2a8452b4e0598489d85a8c435a121c440198c7d819b163c0bf49f.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/tables/136137b012d2a8452b4e0598489d85a8c435a121c440198c7d819b163c0bf49f.jpg)

![1951888c16f2b187338091cc361da8fdbba70f2d42f231d8a4a4aef7a8bed0f5.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/tables/1951888c16f2b187338091cc361da8fdbba70f2d42f231d8a4a4aef7a8bed0f5.jpg)

![56f646722f1b18f46972acb64c52ce2fc1b9bb71c99bc91eb16b56947d187010.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/tables/56f646722f1b18f46972acb64c52ce2fc1b9bb71c99bc91eb16b56947d187010.jpg)

![b4ac897d33281a75694c1607a77c4d8a6c43a0bd6b2b8c00a994608bf3acb28d.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/tables/b4ac897d33281a75694c1607a77c4d8a6c43a0bd6b2b8c00a994608bf3acb28d.jpg)

![bb2914958de35b16d1eb8637494ffa98f305a9feb412fa088f5369679fef1baf.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/tables/bb2914958de35b16d1eb8637494ffa98f305a9feb412fa088f5369679fef1baf.jpg)

![c064f6047e56ef188e3a9c0823bd7635b49c0ce1d707456902f704dc5aa81fbc.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/tables/c064f6047e56ef188e3a9c0823bd7635b49c0ce1d707456902f704dc5aa81fbc.jpg)

![c0e5981a87e6d4b4db47dd712ae4fae76574c7fed2ce6316bc2218a753be54e7.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/tables/c0e5981a87e6d4b4db47dd712ae4fae76574c7fed2ce6316bc2218a753be54e7.jpg)

![d1a8ef7628b014251069b32071aa81008a8e79a8551f3a6ea950cd6e23d4aea7.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/tables/d1a8ef7628b014251069b32071aa81008a8e79a8551f3a6ea950cd6e23d4aea7.jpg)

![df9e839206194abde0de45500e9c17ed02378e3d0715399faaf1c76b35b26b99.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/tables/df9e839206194abde0de45500e9c17ed02378e3d0715399faaf1c76b35b26b99.jpg)

![e08643ccc7d16290e715acd8a2a2957f0e886f733bdefe79cd392893e0161aad.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/tables/e08643ccc7d16290e715acd8a2a2957f0e886f733bdefe79cd392893e0161aad.jpg)

![edc14b61d0b83335b5a0f85e3b439d088fff4e581172655937074da24240ebcb.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/tables/edc14b61d0b83335b5a0f85e3b439d088fff4e581172655937074da24240ebcb.jpg)

![f375038f2058e7bb5d4dae9d17d57b5f02019d2cbf8ebf9ac392443d7f875f54.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/tables/f375038f2058e7bb5d4dae9d17d57b5f02019d2cbf8ebf9ac392443d7f875f54.jpg)

![f7878f5ba2b2cd13fa2bb7896c143a8e8c0ec00e0a21a6a3c99f209490fc1b9e.jpg](../icml_results/4010_A%20Regime-Aware%20Trajectory%20Prediction%20Framework%20for%201000%2B%20Systems%20Biology%20Models/tables/f7878f5ba2b2cd13fa2bb7896c143a8e8c0ec00e0a21a6a3c99f209490fc1b9e.jpg)

<a id="paper-3970"></a>

## On the Generalization in Topology Optimization via Sensitivity-Conditioned Bernoulli Flow Matching


### Images

![08c21f6de0521544b6f04e30bed7eafa5e1288fec04f489ffc2f78f8e03f3d1e.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/images/08c21f6de0521544b6f04e30bed7eafa5e1288fec04f489ffc2f78f8e03f3d1e.jpg)

![11395a24747d67fa52f61923c479c06efedfcbf86715c05cd371941ff6e4876a.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/images/11395a24747d67fa52f61923c479c06efedfcbf86715c05cd371941ff6e4876a.jpg)

![13e020a27d0d2a3f96b38c1c7b10b45d29be5a39b38cac9945903cd6f4decfcd.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/images/13e020a27d0d2a3f96b38c1c7b10b45d29be5a39b38cac9945903cd6f4decfcd.jpg)

![37aa20931efc886ba3a9aebe5bba1d88ea94266ef0650baca9426241965e1c34.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/images/37aa20931efc886ba3a9aebe5bba1d88ea94266ef0650baca9426241965e1c34.jpg)

![53d774c029b5800d853ecc6bbed866649c5fe900e2c931cd652dd56309bd575b.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/images/53d774c029b5800d853ecc6bbed866649c5fe900e2c931cd652dd56309bd575b.jpg)

![554beb5398057fa984284547547f457946721b2690d46f9997fb6637aad1b8d2.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/images/554beb5398057fa984284547547f457946721b2690d46f9997fb6637aad1b8d2.jpg)

![562f7c2b3a918d344b55f45d98e25138753b34c12ef8a995e9b4cf5ee072b2e7.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/images/562f7c2b3a918d344b55f45d98e25138753b34c12ef8a995e9b4cf5ee072b2e7.jpg)

![6c4a8db767d9127af49424d17a96a27bc4d29ad07d9a602a4a0701be6bcb0799.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/images/6c4a8db767d9127af49424d17a96a27bc4d29ad07d9a602a4a0701be6bcb0799.jpg)

![87e3ae2186d61e957cdb634ed03a3a21339575e1237ebd5f511073a33abc6abf.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/images/87e3ae2186d61e957cdb634ed03a3a21339575e1237ebd5f511073a33abc6abf.jpg)

![8c25b8da66336ac54ba0addec61aef2dd5363c8c5d19ba6066eb279738fa7923.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/images/8c25b8da66336ac54ba0addec61aef2dd5363c8c5d19ba6066eb279738fa7923.jpg)

![a75ff46d7ac55d4fcc31b4dc311cde1fbd3b163e53b44f11c725e02997184a99.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/images/a75ff46d7ac55d4fcc31b4dc311cde1fbd3b163e53b44f11c725e02997184a99.jpg)

![b2413919c7c26cbc9da2faab4bec36a278b2466a2ed662a80c5fe9bb881d4552.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/images/b2413919c7c26cbc9da2faab4bec36a278b2466a2ed662a80c5fe9bb881d4552.jpg)

![c849a57f9aa34752d044e2ed8f588078406b8d598e25c896780288893488fe0e.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/images/c849a57f9aa34752d044e2ed8f588078406b8d598e25c896780288893488fe0e.jpg)

![ecc62f280ede58ea18b418c615ba3e151d86dd0dd0a7728eccef9d46075584ff.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/images/ecc62f280ede58ea18b418c615ba3e151d86dd0dd0a7728eccef9d46075584ff.jpg)

![f2dfbfda4534f90aa3702ccd3fc942cff631bdf6602926049839a8b8d13df6c9.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/images/f2dfbfda4534f90aa3702ccd3fc942cff631bdf6602926049839a8b8d13df6c9.jpg)

![fdcb0595e33795d6b02490a65325a4710c50cf37b3f28c0f0cdc450165175830.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/images/fdcb0595e33795d6b02490a65325a4710c50cf37b3f28c0f0cdc450165175830.jpg)

### Tables

![031a0d828685cce1fe6d5182a360ca69e18f5a4a3134c8e63d165c86d83ed43b.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/tables/031a0d828685cce1fe6d5182a360ca69e18f5a4a3134c8e63d165c86d83ed43b.jpg)

![032b7fa2d201be1e8d459dd27c60efaaaa6ebff914f51c02ef2c111131f9d06a.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/tables/032b7fa2d201be1e8d459dd27c60efaaaa6ebff914f51c02ef2c111131f9d06a.jpg)

![2e00a52c41431eeff023eefa90ad6b23f357a0dba5e475cf5b60ea63c287a780.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/tables/2e00a52c41431eeff023eefa90ad6b23f357a0dba5e475cf5b60ea63c287a780.jpg)

![34db302c9860004daaeae931c6be1e43fab331f89f539b30f02251a7cf2d03e2.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/tables/34db302c9860004daaeae931c6be1e43fab331f89f539b30f02251a7cf2d03e2.jpg)

![542f600cc05f47777355db66c35d5f54720c6247388b9a02436a2728790997ab.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/tables/542f600cc05f47777355db66c35d5f54720c6247388b9a02436a2728790997ab.jpg)

![567f09c6c7ba0f36ab80b7bd397feba09dd47e68ca3ec2c6d01cc26df5d3574e.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/tables/567f09c6c7ba0f36ab80b7bd397feba09dd47e68ca3ec2c6d01cc26df5d3574e.jpg)

![684ab84a3d74a00bdc644294e94d11e81c69d21c2b475ac93f4face086769410.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/tables/684ab84a3d74a00bdc644294e94d11e81c69d21c2b475ac93f4face086769410.jpg)

![94928cc79d3d05b9bd22db0c6393b3e7661db90513e62e96b9f23888a7b52643.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/tables/94928cc79d3d05b9bd22db0c6393b3e7661db90513e62e96b9f23888a7b52643.jpg)

![b04d150dc68649ce5b8b790a10de8c4d026f901d106b30635eaff99f5dd665c5.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/tables/b04d150dc68649ce5b8b790a10de8c4d026f901d106b30635eaff99f5dd665c5.jpg)

![ddc780b1d05c6dc84ca687d4e7a3c4f5db1777f90b2010ab21777a0053603819.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/tables/ddc780b1d05c6dc84ca687d4e7a3c4f5db1777f90b2010ab21777a0053603819.jpg)

![e7246998f72e6a84984e1d0a410b4790d60b80134ad7c0f7fcfc8b0f20c17433.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/tables/e7246998f72e6a84984e1d0a410b4790d60b80134ad7c0f7fcfc8b0f20c17433.jpg)

![f1a502755cf0c3332dd8bca6a03c73f8b6788da89e88b7a497e2ddfd0214e5ea.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/tables/f1a502755cf0c3332dd8bca6a03c73f8b6788da89e88b7a497e2ddfd0214e5ea.jpg)

![ff6a41e091ac5ca808aee39b96098ecb4437730ed1ec75a7f6c51729b461ec72.jpg](../icml_results/4011_On%20the%20Generalization%20in%20Topology%20Optimization%20via%20Sensitivity-Conditioned%20Bernoulli%20Flow%20Matching/tables/ff6a41e091ac5ca808aee39b96098ecb4437730ed1ec75a7f6c51729b461ec72.jpg)

<a id="paper-3971"></a>

## Factored Value Functions for Graph-Based Multi-Agent Reinforcement Learning


### Images

![0fa000e8d4bd32aa650dd367ef4f86ca63e86790f20f4493f65b61de2b8b7847.jpg](../icml_results/4012_Factored%20Value%20Functions%20for%20Graph-Based%20Multi-Agent%20Reinforcement%20Learning/images/0fa000e8d4bd32aa650dd367ef4f86ca63e86790f20f4493f65b61de2b8b7847.jpg)

![3ed1e91dba52cfede26349a94de7479dd38a4969bde5d465af246cf24cc4d728.jpg](../icml_results/4012_Factored%20Value%20Functions%20for%20Graph-Based%20Multi-Agent%20Reinforcement%20Learning/images/3ed1e91dba52cfede26349a94de7479dd38a4969bde5d465af246cf24cc4d728.jpg)

![7318fd884f7bace35d5d69d50d32caf17253508c45e093420af5171c958ac09c.jpg](../icml_results/4012_Factored%20Value%20Functions%20for%20Graph-Based%20Multi-Agent%20Reinforcement%20Learning/images/7318fd884f7bace35d5d69d50d32caf17253508c45e093420af5171c958ac09c.jpg)

![774b557e28d31eef4c562b1baf8661ad4e670c9f74bc99146d41d1998d15ae2c.jpg](../icml_results/4012_Factored%20Value%20Functions%20for%20Graph-Based%20Multi-Agent%20Reinforcement%20Learning/images/774b557e28d31eef4c562b1baf8661ad4e670c9f74bc99146d41d1998d15ae2c.jpg)

![94bf72b96fb3604d33622d218b41e0f55ef5e8fc8856bfd9e880550718c0add1.jpg](../icml_results/4012_Factored%20Value%20Functions%20for%20Graph-Based%20Multi-Agent%20Reinforcement%20Learning/images/94bf72b96fb3604d33622d218b41e0f55ef5e8fc8856bfd9e880550718c0add1.jpg)

![a09474ed2c8e9cca2c2c4936ab06f5706a625db4ec69138d95a3518084186359.jpg](../icml_results/4012_Factored%20Value%20Functions%20for%20Graph-Based%20Multi-Agent%20Reinforcement%20Learning/images/a09474ed2c8e9cca2c2c4936ab06f5706a625db4ec69138d95a3518084186359.jpg)

![ae236221ca9aa72dc28e02d6608f30c254b7a07ab51cffe042a9922247fce2db.jpg](../icml_results/4012_Factored%20Value%20Functions%20for%20Graph-Based%20Multi-Agent%20Reinforcement%20Learning/images/ae236221ca9aa72dc28e02d6608f30c254b7a07ab51cffe042a9922247fce2db.jpg)

![b1d02116f22f0c2e8be51f626849a4aecea6c5a53cac9f2d2196370dcd35a29b.jpg](../icml_results/4012_Factored%20Value%20Functions%20for%20Graph-Based%20Multi-Agent%20Reinforcement%20Learning/images/b1d02116f22f0c2e8be51f626849a4aecea6c5a53cac9f2d2196370dcd35a29b.jpg)

![fcac21b83292f8db172b14fe71133ed8217b226ac5b840672e42d9dd6ebc021f.jpg](../icml_results/4012_Factored%20Value%20Functions%20for%20Graph-Based%20Multi-Agent%20Reinforcement%20Learning/images/fcac21b83292f8db172b14fe71133ed8217b226ac5b840672e42d9dd6ebc021f.jpg)

### Tables

![28d71a3bf51ba23b328cbe0c27d44a25e7d574c3b30fb7ece4170d517154c134.jpg](../icml_results/4012_Factored%20Value%20Functions%20for%20Graph-Based%20Multi-Agent%20Reinforcement%20Learning/tables/28d71a3bf51ba23b328cbe0c27d44a25e7d574c3b30fb7ece4170d517154c134.jpg)

![698e33b5439dfe10907190e9cc0761a46ded6abdfbaafdda6acc09ce11a614ec.jpg](../icml_results/4012_Factored%20Value%20Functions%20for%20Graph-Based%20Multi-Agent%20Reinforcement%20Learning/tables/698e33b5439dfe10907190e9cc0761a46ded6abdfbaafdda6acc09ce11a614ec.jpg)

![6fe41f61be38db64f108aa84a8ee606f90fda492d8d967b139d87dfc7c878ec3.jpg](../icml_results/4012_Factored%20Value%20Functions%20for%20Graph-Based%20Multi-Agent%20Reinforcement%20Learning/tables/6fe41f61be38db64f108aa84a8ee606f90fda492d8d967b139d87dfc7c878ec3.jpg)

![a3a653553a3591642751ce0199a8dc2102ac5b3f08819256a347ce349fe8ad8b.jpg](../icml_results/4012_Factored%20Value%20Functions%20for%20Graph-Based%20Multi-Agent%20Reinforcement%20Learning/tables/a3a653553a3591642751ce0199a8dc2102ac5b3f08819256a347ce349fe8ad8b.jpg)

<a id="paper-3972"></a>

## No More, No Less: Least-Privilege Language Models


### Images

![013208e0ae33bab92aaa84eca6160c53379b7db03d04831b57886aa5a7332130.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/013208e0ae33bab92aaa84eca6160c53379b7db03d04831b57886aa5a7332130.jpg)

![07a7bbb0a913dd4f9dc98de1769a64f047046f9142a02eb68c1960d3e4025277.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/07a7bbb0a913dd4f9dc98de1769a64f047046f9142a02eb68c1960d3e4025277.jpg)

![1b9885bf1910191e2263524d92c5287abd183a2ec499ef3856e2ddaaa3283120.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/1b9885bf1910191e2263524d92c5287abd183a2ec499ef3856e2ddaaa3283120.jpg)

![1ca38e4f3beb115c9a3227b1fb82a7705901ef9284a73621fe7827936adeb07e.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/1ca38e4f3beb115c9a3227b1fb82a7705901ef9284a73621fe7827936adeb07e.jpg)

![2ad3e91acf107bcb8fd323c6cfb6fd409a87e0ef07353c341732e64fde3abc81.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/2ad3e91acf107bcb8fd323c6cfb6fd409a87e0ef07353c341732e64fde3abc81.jpg)

![3bfa633c2c941a0173530adb0bea957ec82df6e4743ba0037aaa7158301301be.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/3bfa633c2c941a0173530adb0bea957ec82df6e4743ba0037aaa7158301301be.jpg)

![44d08bca14435902a0259522e94b9523620e788a3f52d3a1062c2218009240bf.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/44d08bca14435902a0259522e94b9523620e788a3f52d3a1062c2218009240bf.jpg)

![4a47c96b32c6a86f213ce41bfd4f7d7cddd2db152c73365b6e56eec0ca809d34.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/4a47c96b32c6a86f213ce41bfd4f7d7cddd2db152c73365b6e56eec0ca809d34.jpg)

![4b1afd4928a34a8efb84ca69acd843e18409e9cc03f66b3e4f34a2fbe57eda25.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/4b1afd4928a34a8efb84ca69acd843e18409e9cc03f66b3e4f34a2fbe57eda25.jpg)

![4de4aaefa3a49344e4eb56705bf7c288416f7aac8b0ecb55b8517fc4cdbd3b7b.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/4de4aaefa3a49344e4eb56705bf7c288416f7aac8b0ecb55b8517fc4cdbd3b7b.jpg)

![51ac4032792fa215c4689729a8c3f14c01196cc87ec3f43b2c22773ebae2682d.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/51ac4032792fa215c4689729a8c3f14c01196cc87ec3f43b2c22773ebae2682d.jpg)

![52a92bf3a361a061eed798cac29eb10ffbf4606b19c065fa961196632062e245.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/52a92bf3a361a061eed798cac29eb10ffbf4606b19c065fa961196632062e245.jpg)

![53862678770b6a0ceb37fe225c6fc8ef7ef07534863c548fe395260de00f633c.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/53862678770b6a0ceb37fe225c6fc8ef7ef07534863c548fe395260de00f633c.jpg)

![5cd8360dd21b6a96e80572ee5f0cccf0498f3daf80cf2f3915863894eb3f2601.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/5cd8360dd21b6a96e80572ee5f0cccf0498f3daf80cf2f3915863894eb3f2601.jpg)

![5f5fb749bf5c02b1e7926a688ed3ee6bb209524b0a9cf7c2836af737f80830ee.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/5f5fb749bf5c02b1e7926a688ed3ee6bb209524b0a9cf7c2836af737f80830ee.jpg)

![5fd617e7860e3c8431476a0b30e8b21b452d601caba549e1db5811b7786bbbb2.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/5fd617e7860e3c8431476a0b30e8b21b452d601caba549e1db5811b7786bbbb2.jpg)

![61b0b55cf36a1f17341da7891ea62c8cc8fd4f11f40d378ef2d5a36b0f769bce.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/61b0b55cf36a1f17341da7891ea62c8cc8fd4f11f40d378ef2d5a36b0f769bce.jpg)

![66634d78a48727c095acbdda2cd632a55d7035ea57ddeddbb8baa0ac58bdaea1.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/66634d78a48727c095acbdda2cd632a55d7035ea57ddeddbb8baa0ac58bdaea1.jpg)

![86ac145415c67093d742aa77875c26c7ad52332447b8a25f32b9e616994da995.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/86ac145415c67093d742aa77875c26c7ad52332447b8a25f32b9e616994da995.jpg)

![86ff81cae412cb71ba906e182324fa9d301e87fd678497e238bb698369e47dc2.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/86ff81cae412cb71ba906e182324fa9d301e87fd678497e238bb698369e47dc2.jpg)

![8a332c905c637ba4e580d76ca0827b1ee4c674ca3fa4151c804cb9465d0eb783.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/8a332c905c637ba4e580d76ca0827b1ee4c674ca3fa4151c804cb9465d0eb783.jpg)

![987f22b3a6708440a8b3622420f8b2d2c6c4ca1358959d85ba699c92bbf59c32.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/987f22b3a6708440a8b3622420f8b2d2c6c4ca1358959d85ba699c92bbf59c32.jpg)

![a13cd9248fd95f796b86026d882908329495cc6e7a138ba2fb31240d95e084f7.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/a13cd9248fd95f796b86026d882908329495cc6e7a138ba2fb31240d95e084f7.jpg)

![a15675e695a927c6ad3d89318b06db008b235f2227b8b5defc094a054ae6a9cd.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/a15675e695a927c6ad3d89318b06db008b235f2227b8b5defc094a054ae6a9cd.jpg)

![ad576296e8faf62ce1465bc347cd9d3075ebe4f25b142efe1ed37ad4fa564b98.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/ad576296e8faf62ce1465bc347cd9d3075ebe4f25b142efe1ed37ad4fa564b98.jpg)

![b241deec9525a4750fbf7e8e06d83af5e6387116c1535f3ac619e3af792c652b.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/b241deec9525a4750fbf7e8e06d83af5e6387116c1535f3ac619e3af792c652b.jpg)

![d5bded21f176c43f5e6ad67c0bc323acec01ecaba2e2c04081f33b569c8c7370.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/images/d5bded21f176c43f5e6ad67c0bc323acec01ecaba2e2c04081f33b569c8c7370.jpg)

### Tables

![109b95340fe55fd10f74742899394296b628fd26cae4616ee758fd0deedefd6f.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/tables/109b95340fe55fd10f74742899394296b628fd26cae4616ee758fd0deedefd6f.jpg)

![362e1dfdf3c79c3065952344b0851958336dff7aa088eab2b51ac04601eaef1a.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/tables/362e1dfdf3c79c3065952344b0851958336dff7aa088eab2b51ac04601eaef1a.jpg)

![36b7634a49658647dd9273fd3363d8288e071c72abac21ae15be07279b4771df.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/tables/36b7634a49658647dd9273fd3363d8288e071c72abac21ae15be07279b4771df.jpg)

![36eca06a7fb9a051ed614be271797b52e38cea98fd46b1dab9a1744dcf557c24.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/tables/36eca06a7fb9a051ed614be271797b52e38cea98fd46b1dab9a1744dcf557c24.jpg)

![458d44f4c0b458d2876fea5c76a476cb3b06878b6b15577059ac76767a86596b.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/tables/458d44f4c0b458d2876fea5c76a476cb3b06878b6b15577059ac76767a86596b.jpg)

![55281c965dad50fa77cbd13a5408f4066a38f85f43364d19f826eacc8476fa1d.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/tables/55281c965dad50fa77cbd13a5408f4066a38f85f43364d19f826eacc8476fa1d.jpg)

![5539a7301d626ac21d242c1efd341e1ddb5b05775505c2dd1483125657e97188.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/tables/5539a7301d626ac21d242c1efd341e1ddb5b05775505c2dd1483125657e97188.jpg)

![903994ee2e6757bc147ced406819758ca475871ee71c1690d3c8714d4f88ae0a.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/tables/903994ee2e6757bc147ced406819758ca475871ee71c1690d3c8714d4f88ae0a.jpg)

![a3b16d88c71ee5374624a8db747b3feac16967a9bef36f97fefc02a53b44d528.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/tables/a3b16d88c71ee5374624a8db747b3feac16967a9bef36f97fefc02a53b44d528.jpg)

![b63ab431665a9677c9ad43095c6f92cc53dd23c4c31c46e8f67fc829bece983d.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/tables/b63ab431665a9677c9ad43095c6f92cc53dd23c4c31c46e8f67fc829bece983d.jpg)

![d1fa8b925804b0ea796ed6b3a7780920c9aaa64fc4361200c8b3d6cb62ee6205.jpg](../icml_results/4013_No%20More%2C%20No%20Less_%20Least-Privilege%20Language%20Models/tables/d1fa8b925804b0ea796ed6b3a7780920c9aaa64fc4361200c8b3d6cb62ee6205.jpg)

<a id="paper-3973"></a>

## $τ$-Voice: Benchmarking Full-Duplex Voice Agents on Real-World Domains


### Images

![3b1b650a2a31e70f9b329418e473b4fd3e1111331c76d0d59f570dc05db88473.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/images/3b1b650a2a31e70f9b329418e473b4fd3e1111331c76d0d59f570dc05db88473.jpg)

![6634c53ab7242ee1a1395dc73ec7860151134d6d7c49a38a29c243a099d29497.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/images/6634c53ab7242ee1a1395dc73ec7860151134d6d7c49a38a29c243a099d29497.jpg)

![86c0eb6d7c6b59c52b090e05a6187893c9a6a46a984de2e84029549169853204.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/images/86c0eb6d7c6b59c52b090e05a6187893c9a6a46a984de2e84029549169853204.jpg)

![9dd1186f2a6a01b40d75eaf754a17852e4d4824339be1392d9542ead7b1b9f4e.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/images/9dd1186f2a6a01b40d75eaf754a17852e4d4824339be1392d9542ead7b1b9f4e.jpg)

![d66a318a27174fb0cb49b3c977d032a75db967d43a03b8ed302beb6fb1578d8f.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/images/d66a318a27174fb0cb49b3c977d032a75db967d43a03b8ed302beb6fb1578d8f.jpg)

### Tables

![08cbde74c5324369165820c03d2e73c4993a910739f0f4b8b7d48acd87a89405.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/08cbde74c5324369165820c03d2e73c4993a910739f0f4b8b7d48acd87a89405.jpg)

![0de9633238b63b48fcc1e9a7806a34d0799e380dead7bc21e6e93e855cf912d7.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/0de9633238b63b48fcc1e9a7806a34d0799e380dead7bc21e6e93e855cf912d7.jpg)

![0f8e973d40ecf044f3d03b6822924ada13765414d683ff05d02b87d945a23509.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/0f8e973d40ecf044f3d03b6822924ada13765414d683ff05d02b87d945a23509.jpg)

![18a3773c7a42e78424f61482c3997208b6c8691b16b990fd47ae5841dd7fa40b.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/18a3773c7a42e78424f61482c3997208b6c8691b16b990fd47ae5841dd7fa40b.jpg)

![3364dbd2826d7a844653045c202e62bed42aef67b05fdd541a304ec11ff09731.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/3364dbd2826d7a844653045c202e62bed42aef67b05fdd541a304ec11ff09731.jpg)

![3aff1e2a8bd8596e6809538fd470ada235169dc09d2b6c5add837006cf5df4b9.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/3aff1e2a8bd8596e6809538fd470ada235169dc09d2b6c5add837006cf5df4b9.jpg)

![3d5048940289014806b653c52065ef7dfa2d64b64fffc5b87f6a72c7cb85bf23.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/3d5048940289014806b653c52065ef7dfa2d64b64fffc5b87f6a72c7cb85bf23.jpg)

![48ea9753ce025b1f930341cf6628a8f8dacfbeb540e0346adbb8cbf10403bf0d.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/48ea9753ce025b1f930341cf6628a8f8dacfbeb540e0346adbb8cbf10403bf0d.jpg)

![5136a28da7397dd37238f15edd98d90dce841c924fd254b503755d45adeaddd8.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/5136a28da7397dd37238f15edd98d90dce841c924fd254b503755d45adeaddd8.jpg)

![54d1c0855cb9ad769220eb430b654f48f7614d1a36ae54795cd1c4c7ce2d42a8.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/54d1c0855cb9ad769220eb430b654f48f7614d1a36ae54795cd1c4c7ce2d42a8.jpg)

![6b88ce0a5889259251f8960641292706e59c0920335dfc6054dd6609819625f3.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/6b88ce0a5889259251f8960641292706e59c0920335dfc6054dd6609819625f3.jpg)

![712e88edad580b266036544ee40b87bbd6c18596147fd6f8baded41220f69ad5.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/712e88edad580b266036544ee40b87bbd6c18596147fd6f8baded41220f69ad5.jpg)

![750a5470cdb8f04642e173e2636d93c2908712f74cc6496beca81e631a112aef.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/750a5470cdb8f04642e173e2636d93c2908712f74cc6496beca81e631a112aef.jpg)

![7806f6237a57676079f3fc64f108cb9dfd24091a46a37ee327071dbc3d0a9f9d.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/7806f6237a57676079f3fc64f108cb9dfd24091a46a37ee327071dbc3d0a9f9d.jpg)

![7deb9d4f81654547de9d1548b4ceaa76635a98551076a72a7fa6fcb8a80411aa.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/7deb9d4f81654547de9d1548b4ceaa76635a98551076a72a7fa6fcb8a80411aa.jpg)

![80e273a08424045df7da85bd2e49233eda0be212313a10d80472a072d74aece7.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/80e273a08424045df7da85bd2e49233eda0be212313a10d80472a072d74aece7.jpg)

![81e6ce743572b3911fa22c5fabf4bf5580242026e2d0db2d2214eb1290a09f83.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/81e6ce743572b3911fa22c5fabf4bf5580242026e2d0db2d2214eb1290a09f83.jpg)

![967d2078afb4b1f6f324519387aecf6748ccf10f0f93e216827ea4e89d6ca380.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/967d2078afb4b1f6f324519387aecf6748ccf10f0f93e216827ea4e89d6ca380.jpg)

![989c8581255a95a3bf5a1b13da93ed43670d5644d8bd8713b93922ee4e6ecaff.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/989c8581255a95a3bf5a1b13da93ed43670d5644d8bd8713b93922ee4e6ecaff.jpg)

![a82a2202a4dfdb00b37ffe0944fd6b7887db9c00ff47c97803f76a7bfdb94859.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/a82a2202a4dfdb00b37ffe0944fd6b7887db9c00ff47c97803f76a7bfdb94859.jpg)

![b493439c79ce8990cd57f343dd950d81a93943c3154e6072401b28250db0d2dd.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/b493439c79ce8990cd57f343dd950d81a93943c3154e6072401b28250db0d2dd.jpg)

![b827a342e5099feb565742914163b0e7864f7ac154a50b5191baad2c4591d027.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/b827a342e5099feb565742914163b0e7864f7ac154a50b5191baad2c4591d027.jpg)

![c80ae93d57d6ce004ef03322c4f55085c78ae187cd4407d29061334a8782bd93.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/c80ae93d57d6ce004ef03322c4f55085c78ae187cd4407d29061334a8782bd93.jpg)

![d11882b7909b2dc482905f0650201000dad4654dfe279913d02935c58fb89b08.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/d11882b7909b2dc482905f0650201000dad4654dfe279913d02935c58fb89b08.jpg)

![d4b91e42039dfbf8854ca3e9f35255809a3be60659117e478d804cad1b6d98d5.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/d4b91e42039dfbf8854ca3e9f35255809a3be60659117e478d804cad1b6d98d5.jpg)

![dc91202d6215778af6f243e2fdb7c056d20bdebbb49f797975a6b8c7aadc49cb.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/dc91202d6215778af6f243e2fdb7c056d20bdebbb49f797975a6b8c7aadc49cb.jpg)

![e21ddc375058374d237f0e4504cfeef623edf5be44ad3c8e892003d746d26e7c.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/e21ddc375058374d237f0e4504cfeef623edf5be44ad3c8e892003d746d26e7c.jpg)

![e477ea076d775c7eca5ea20943a5717a85524ec0d14c139eeee21d5ce9969157.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/e477ea076d775c7eca5ea20943a5717a85524ec0d14c139eeee21d5ce9969157.jpg)

![eb358a69f4e42a7d9e1bf492dacaf896256871e2e9788551bf14f7222f3bd5ce.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/eb358a69f4e42a7d9e1bf492dacaf896256871e2e9788551bf14f7222f3bd5ce.jpg)

![edc2211a71c5cd10068ee3ba5235ee64ffa15bd3b14734a4bd4ee5b0e109f423.jpg](../icml_results/4014_%24%CF%84%24-Voice_%20Benchmarking%20Full-Duplex%20Voice%20Agents%20on%20Real-World%20Domains/tables/edc2211a71c5cd10068ee3ba5235ee64ffa15bd3b14734a4bd4ee5b0e109f423.jpg)

<a id="paper-3974"></a>

## AutoNumerics-Zero: Automated Discovery of State-of-the-Art Mathematical Functions


### Images

![04f49d7de8fbfd662977c024bd8800b9f87f91d72c9e76c249666e5cc5241cf8.jpg](../icml_results/4015_AutoNumerics-Zero_%20Automated%20Discovery%20of%20State-of-the-Art%20Mathematical%20Functions/images/04f49d7de8fbfd662977c024bd8800b9f87f91d72c9e76c249666e5cc5241cf8.jpg)

![0a41d8d41a8d0513eef90f0264082e88d50e1bb350c30055820aa713f4606261.jpg](../icml_results/4015_AutoNumerics-Zero_%20Automated%20Discovery%20of%20State-of-the-Art%20Mathematical%20Functions/images/0a41d8d41a8d0513eef90f0264082e88d50e1bb350c30055820aa713f4606261.jpg)

![1be29b11854e5bf4ba3cd9f467e1350682ea684029b6a0ed233125a30e0d1035.jpg](../icml_results/4015_AutoNumerics-Zero_%20Automated%20Discovery%20of%20State-of-the-Art%20Mathematical%20Functions/images/1be29b11854e5bf4ba3cd9f467e1350682ea684029b6a0ed233125a30e0d1035.jpg)

![26734cd86e181475ce4009445e1aefd366ad5b50244cabe41a131f2214ea4a50.jpg](../icml_results/4015_AutoNumerics-Zero_%20Automated%20Discovery%20of%20State-of-the-Art%20Mathematical%20Functions/images/26734cd86e181475ce4009445e1aefd366ad5b50244cabe41a131f2214ea4a50.jpg)

![3794c8ae8502ba0f30176c7bc84383bcca9479d2d20ab6327d97526700660cb0.jpg](../icml_results/4015_AutoNumerics-Zero_%20Automated%20Discovery%20of%20State-of-the-Art%20Mathematical%20Functions/images/3794c8ae8502ba0f30176c7bc84383bcca9479d2d20ab6327d97526700660cb0.jpg)

![5f9e6e2b6580bb3c9ed7289583ff3519916947779c197792ca3f8fc128fbbae3.jpg](../icml_results/4015_AutoNumerics-Zero_%20Automated%20Discovery%20of%20State-of-the-Art%20Mathematical%20Functions/images/5f9e6e2b6580bb3c9ed7289583ff3519916947779c197792ca3f8fc128fbbae3.jpg)

![6cfc68a92b5e1addee162ce662a35866920eef965ec2b41f606d9fc2458f537f.jpg](../icml_results/4015_AutoNumerics-Zero_%20Automated%20Discovery%20of%20State-of-the-Art%20Mathematical%20Functions/images/6cfc68a92b5e1addee162ce662a35866920eef965ec2b41f606d9fc2458f537f.jpg)

![771e335343cccda9ac647b5e59ca0a77286912f77eef097266418114ac572a70.jpg](../icml_results/4015_AutoNumerics-Zero_%20Automated%20Discovery%20of%20State-of-the-Art%20Mathematical%20Functions/images/771e335343cccda9ac647b5e59ca0a77286912f77eef097266418114ac572a70.jpg)

![912ebf452839ffa52098cc562b38bf0d6c0babc009bd38f41d69c265777702a3.jpg](../icml_results/4015_AutoNumerics-Zero_%20Automated%20Discovery%20of%20State-of-the-Art%20Mathematical%20Functions/images/912ebf452839ffa52098cc562b38bf0d6c0babc009bd38f41d69c265777702a3.jpg)

![91bd72bd9787a14a1a558b95ac807bf354ea39ec48a4ae94a18ed43499c70572.jpg](../icml_results/4015_AutoNumerics-Zero_%20Automated%20Discovery%20of%20State-of-the-Art%20Mathematical%20Functions/images/91bd72bd9787a14a1a558b95ac807bf354ea39ec48a4ae94a18ed43499c70572.jpg)

![9a9b2c56436bdf1bce01c2579ab2e66cacaecc8ab9013eee4e29470633b6dcbe.jpg](../icml_results/4015_AutoNumerics-Zero_%20Automated%20Discovery%20of%20State-of-the-Art%20Mathematical%20Functions/images/9a9b2c56436bdf1bce01c2579ab2e66cacaecc8ab9013eee4e29470633b6dcbe.jpg)

![a7d1e729ce2c42570f2be1c759ceb7beb19287100e38da093da3742d3f093f5d.jpg](../icml_results/4015_AutoNumerics-Zero_%20Automated%20Discovery%20of%20State-of-the-Art%20Mathematical%20Functions/images/a7d1e729ce2c42570f2be1c759ceb7beb19287100e38da093da3742d3f093f5d.jpg)

### Tables

![f93d4fa2e9e0ab1a15a9775b29debcfd1364f92d08848fefc59e8d6a70bb4add.jpg](../icml_results/4015_AutoNumerics-Zero_%20Automated%20Discovery%20of%20State-of-the-Art%20Mathematical%20Functions/tables/f93d4fa2e9e0ab1a15a9775b29debcfd1364f92d08848fefc59e8d6a70bb4add.jpg)

<a id="paper-3975"></a>

## Learning to Watermark in the Latent Space of Generative Models


### Images

![056e81a906da039c808dac2f4a2cecbc26b5d10eac1a8be70b3251f47817cd4a.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/056e81a906da039c808dac2f4a2cecbc26b5d10eac1a8be70b3251f47817cd4a.jpg)

![08e18bcc4f9643a88704d8108941824827b79acbbb14b1366aa7b26e6ef582cc.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/08e18bcc4f9643a88704d8108941824827b79acbbb14b1366aa7b26e6ef582cc.jpg)

![1785055793b2936998e5ed929422e00ed270cf28593bbd5a0484f0f022db8b07.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/1785055793b2936998e5ed929422e00ed270cf28593bbd5a0484f0f022db8b07.jpg)

![17fbde9e6ed90db3048daffb2b88fe617eb412b51753d7b4bc96ee88fba64ede.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/17fbde9e6ed90db3048daffb2b88fe617eb412b51753d7b4bc96ee88fba64ede.jpg)

![189d320202611ea652bf49804f75af2a793f4a9b2bcb3508bde79c1284f9ea52.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/189d320202611ea652bf49804f75af2a793f4a9b2bcb3508bde79c1284f9ea52.jpg)

![1cb6ba4f5ef0717f7e85bc9093a592b5c4a89ade5b8a5a23f3c39f40607d4655.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/1cb6ba4f5ef0717f7e85bc9093a592b5c4a89ade5b8a5a23f3c39f40607d4655.jpg)

![3015dd3fdf6f26c10b480c960eb2e4493607784dc884f5625a470d30edeea1e2.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/3015dd3fdf6f26c10b480c960eb2e4493607784dc884f5625a470d30edeea1e2.jpg)

![3599bf7dd29ed15b8caa34a5c4f4f7a9e07d3eade37513e289db1751fbe5461f.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/3599bf7dd29ed15b8caa34a5c4f4f7a9e07d3eade37513e289db1751fbe5461f.jpg)

![3b889de0bb5b0f61f42cc489c5e1f26f13b66939a3d6d39af5bd44e39753b7b4.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/3b889de0bb5b0f61f42cc489c5e1f26f13b66939a3d6d39af5bd44e39753b7b4.jpg)

![44d3ffd13431a03407e28f9830694f4d7721cfc499399737a28e3b3eb21fcf02.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/44d3ffd13431a03407e28f9830694f4d7721cfc499399737a28e3b3eb21fcf02.jpg)

![4ec4e00a07b4db1a10af6f32e9e8db61071f91cc5ddbd958b685c98ceaf38dfc.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/4ec4e00a07b4db1a10af6f32e9e8db61071f91cc5ddbd958b685c98ceaf38dfc.jpg)

![65ea7f780fa8b56c453aa2822caaa8109fa7bbd657ce51b611d8bbd5613d39c5.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/65ea7f780fa8b56c453aa2822caaa8109fa7bbd657ce51b611d8bbd5613d39c5.jpg)

![691f907f353389292779ab47e8a46df0c58410e9602cf5115ec8e30f0ced6e36.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/691f907f353389292779ab47e8a46df0c58410e9602cf5115ec8e30f0ced6e36.jpg)

![6c8e2f36dc36bc9f248426aa45095a6f51d6cce55cfc2566bd643d4d5ff5b8ba.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/6c8e2f36dc36bc9f248426aa45095a6f51d6cce55cfc2566bd643d4d5ff5b8ba.jpg)

![a38f47cc8d59fd59bf43f27a6d9e2d0e03af72d3f9255ed97aacbe98b228defc.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/a38f47cc8d59fd59bf43f27a6d9e2d0e03af72d3f9255ed97aacbe98b228defc.jpg)

![b28cc353f8ec08baa0047e0ba7544b04b5dbeb74099a93179dfbaa68a7ced158.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/b28cc353f8ec08baa0047e0ba7544b04b5dbeb74099a93179dfbaa68a7ced158.jpg)

![d85ecd2049a1d755156370525b74eda083c03a9da7574d4628a59e4d844293c9.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/d85ecd2049a1d755156370525b74eda083c03a9da7574d4628a59e4d844293c9.jpg)

![e8568590a40674dd943020234a2e42823ef834dd7b4b9332dd10a3b3fdc5eebb.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/e8568590a40674dd943020234a2e42823ef834dd7b4b9332dd10a3b3fdc5eebb.jpg)

![ee577be7a953f89ebb351b6c47b58c781b5f2931253cfc2c0b37ee4ae9997832.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/images/ee577be7a953f89ebb351b6c47b58c781b5f2931253cfc2c0b37ee4ae9997832.jpg)

### Tables

![05ff2a740d86553c7323770f3918c2ddedb9d9e4c4c00e5b313d30ec55dc802b.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/05ff2a740d86553c7323770f3918c2ddedb9d9e4c4c00e5b313d30ec55dc802b.jpg)

![3633b8b60240ec66bdda7cfb4857b48e538f690a1976b108b3f1679f015eb571.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/3633b8b60240ec66bdda7cfb4857b48e538f690a1976b108b3f1679f015eb571.jpg)

![3e25df52773fb5dcfeef56ad7771969eba4c7c3ce5ef774f189b8caf51df36fd.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/3e25df52773fb5dcfeef56ad7771969eba4c7c3ce5ef774f189b8caf51df36fd.jpg)

![3f290e665f494a01c5bc707756149d0b466c8aa69ccd1cf78987e67b4b22e553.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/3f290e665f494a01c5bc707756149d0b466c8aa69ccd1cf78987e67b4b22e553.jpg)

![460e083b12db54def7550fbc27a8befbd76c6a81ed1b40d9413cca0026c6e86b.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/460e083b12db54def7550fbc27a8befbd76c6a81ed1b40d9413cca0026c6e86b.jpg)

![5e7d3c85de80364848b4410d92b74fcb74dabe8f09ea558138f020c0f0666ae1.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/5e7d3c85de80364848b4410d92b74fcb74dabe8f09ea558138f020c0f0666ae1.jpg)

![6ee588db9149ff4faaf5c3a4621d0e93d63d534ed83fd9335603dc369746bb70.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/6ee588db9149ff4faaf5c3a4621d0e93d63d534ed83fd9335603dc369746bb70.jpg)

![719da15bd9f7d074d89a39c021f4bdc91d9df3ef090be31a208a5ea9e19a4faf.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/719da15bd9f7d074d89a39c021f4bdc91d9df3ef090be31a208a5ea9e19a4faf.jpg)

![756ce94ae056793061425a892b113903d75aa32ff99afa5f5be561a0183ecfc0.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/756ce94ae056793061425a892b113903d75aa32ff99afa5f5be561a0183ecfc0.jpg)

![878181e7ad7e712969fdb7de3aff8f5444a4191f2040bca03e603b1a91120ce1.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/878181e7ad7e712969fdb7de3aff8f5444a4191f2040bca03e603b1a91120ce1.jpg)

![88cf81d928cbbab595809c08e2028884058cefc4690463c4ea6e44c90d775445.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/88cf81d928cbbab595809c08e2028884058cefc4690463c4ea6e44c90d775445.jpg)

![8ea86cf9daf2c99e9e2a4fa7e43d6bb6f17eeb4287b64a4c79a54ebfd90e4bc9.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/8ea86cf9daf2c99e9e2a4fa7e43d6bb6f17eeb4287b64a4c79a54ebfd90e4bc9.jpg)

![94848450e8fbfdfe354e215839d090e041d2869f38032476b879c6481b51ebd6.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/94848450e8fbfdfe354e215839d090e041d2869f38032476b879c6481b51ebd6.jpg)

![a72d3863bd8fa06e22c15c028b02034ea07b66b79a8fa0d71ce7a2d299dc51cf.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/a72d3863bd8fa06e22c15c028b02034ea07b66b79a8fa0d71ce7a2d299dc51cf.jpg)

![a75cfc7d6f2dfd80b03fa56f0b6a7380d93bb276be69546c856fcc987ebde90e.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/a75cfc7d6f2dfd80b03fa56f0b6a7380d93bb276be69546c856fcc987ebde90e.jpg)

![bfc2bc041f53243bb6976c718701bf42cd6d135606052c4f831e6ad730982254.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/bfc2bc041f53243bb6976c718701bf42cd6d135606052c4f831e6ad730982254.jpg)

![dcd5d2de1fd78b177e7a905329a9a44b5b645b489b25b4e5fa4b79d4616f5495.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/dcd5d2de1fd78b177e7a905329a9a44b5b645b489b25b4e5fa4b79d4616f5495.jpg)

![e0ecc2838aad35e59468fc9a0e73358afd8597865d4d0a148de153738d02c140.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/e0ecc2838aad35e59468fc9a0e73358afd8597865d4d0a148de153738d02c140.jpg)

![f0d44c71d937509fea02f99f0d62b160e8f393e51725a26df5994834c8f9dff4.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/f0d44c71d937509fea02f99f0d62b160e8f393e51725a26df5994834c8f9dff4.jpg)

![fe53b76b3d3d1b1249156012f1464230fbd729e53ecc1b67b936cc9141fcaf2e.jpg](../icml_results/4016_Learning%20to%20Watermark%20in%20the%20Latent%20Space%20of%20Generative%20Models/tables/fe53b76b3d3d1b1249156012f1464230fbd729e53ecc1b67b936cc9141fcaf2e.jpg)

<a id="paper-3976"></a>

## The Hidden Cost of Structured Generation in LLMs: Draft-Conditioned Constrained Decoding


### Images

![1babc108df9bd73d6cd428355546506fb7f1bb34632a376e55ef68c948844197.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/images/1babc108df9bd73d6cd428355546506fb7f1bb34632a376e55ef68c948844197.jpg)

![203ad4cbbba875625531ee4b2943b5986713cc90c64873308770b01295a92099.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/images/203ad4cbbba875625531ee4b2943b5986713cc90c64873308770b01295a92099.jpg)

![27692574cc630ed75dc938a3b33082f2f6f63276d636ecc9f61ddac09581c28f.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/images/27692574cc630ed75dc938a3b33082f2f6f63276d636ecc9f61ddac09581c28f.jpg)

![2d42f4f1983752f3a74bd92222352939533cf93edbf3aae66dfe24c532d83a02.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/images/2d42f4f1983752f3a74bd92222352939533cf93edbf3aae66dfe24c532d83a02.jpg)

![2f3e7360806c361b03295354c4c680b2f06c2980136487f9d262767493f29d57.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/images/2f3e7360806c361b03295354c4c680b2f06c2980136487f9d262767493f29d57.jpg)

![399b2a27e666711c5711a5a99da60414826215ea73d6126a7d8ca5eb3ed347b0.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/images/399b2a27e666711c5711a5a99da60414826215ea73d6126a7d8ca5eb3ed347b0.jpg)

![4882326aa4216657f68d6b783fbbb631bac2d72836b7a2ad99324b8edb10e0b8.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/images/4882326aa4216657f68d6b783fbbb631bac2d72836b7a2ad99324b8edb10e0b8.jpg)

![4d2e3fad81d3357962da14689310cae1b704a019c41fab4f3ac1f9597cac30fe.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/images/4d2e3fad81d3357962da14689310cae1b704a019c41fab4f3ac1f9597cac30fe.jpg)

![65d1df5eb64fd5addba9bc8324b83f2583eb97404da0a800e0020ec75b0def11.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/images/65d1df5eb64fd5addba9bc8324b83f2583eb97404da0a800e0020ec75b0def11.jpg)

![6c67dcce87708ca614695f250dabe110388b08f126aa840e19338b15804c3513.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/images/6c67dcce87708ca614695f250dabe110388b08f126aa840e19338b15804c3513.jpg)

![a4a34b90dcbf866a9e951bf2ccd962d6e785f1e6fe310163931592718f1d2361.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/images/a4a34b90dcbf866a9e951bf2ccd962d6e785f1e6fe310163931592718f1d2361.jpg)

![b5b18ed1baf8c540703cb3099771a48ed9b1a6211266ca64d5a5f17879765d8c.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/images/b5b18ed1baf8c540703cb3099771a48ed9b1a6211266ca64d5a5f17879765d8c.jpg)

![e18c51da30f691244cbfa0eb3bc31ced11770cb2bd251bd2ba1a0dd8ddcaf6d7.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/images/e18c51da30f691244cbfa0eb3bc31ced11770cb2bd251bd2ba1a0dd8ddcaf6d7.jpg)

![f27858b743f69e8e08582406f74dc3d7ea9b66d4cfd399db52b595c742f3801e.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/images/f27858b743f69e8e08582406f74dc3d7ea9b66d4cfd399db52b595c742f3801e.jpg)

![fa8dc8a47aec393fb28fc5527162573607e4a2672490cdf47da61a3f64ba609a.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/images/fa8dc8a47aec393fb28fc5527162573607e4a2672490cdf47da61a3f64ba609a.jpg)

![feeb486b8da109685406a2758cd38cf5fbf076e32b4b4b93165355fa755c5470.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/images/feeb486b8da109685406a2758cd38cf5fbf076e32b4b4b93165355fa755c5470.jpg)

### Tables

![000941534ef4c44380f3b6ad4d6c9cab72607e52a19d7cc2a46e874c9cba00ce.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/tables/000941534ef4c44380f3b6ad4d6c9cab72607e52a19d7cc2a46e874c9cba00ce.jpg)

![0cfa286fde794ceab83a849085ec21433a24902f39c8ec59a3b8e70df845adba.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/tables/0cfa286fde794ceab83a849085ec21433a24902f39c8ec59a3b8e70df845adba.jpg)

![1f45ba6040fd27625761c4f5ec1bc0a5580fb8777fc2230f2b51e4c56d2d39bc.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/tables/1f45ba6040fd27625761c4f5ec1bc0a5580fb8777fc2230f2b51e4c56d2d39bc.jpg)

![22c1596c6fc2db786f162a992255228339bb0cdc55c5a9db179399af752a0ce9.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/tables/22c1596c6fc2db786f162a992255228339bb0cdc55c5a9db179399af752a0ce9.jpg)

![2dd74606ae0cfe05bb39352eb04c66f9058251573de976c808ded281546f5987.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/tables/2dd74606ae0cfe05bb39352eb04c66f9058251573de976c808ded281546f5987.jpg)

![38954f336e2619029f066edb4bb6b3805bcfadebe112762db52d6440182332bf.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/tables/38954f336e2619029f066edb4bb6b3805bcfadebe112762db52d6440182332bf.jpg)

![51fa16aa2c2ff38f518071e694230107be043b537fa8078a8a759afe279d60db.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/tables/51fa16aa2c2ff38f518071e694230107be043b537fa8078a8a759afe279d60db.jpg)

![5411dd47989a8c210049c68c943db5c33453b02b5192000318843ace0809b0fd.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/tables/5411dd47989a8c210049c68c943db5c33453b02b5192000318843ace0809b0fd.jpg)

![70caf722bc0e9c5ecdf59e6c2cc327d83d0be7f06a4e1ae281e1430ef304cdcb.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/tables/70caf722bc0e9c5ecdf59e6c2cc327d83d0be7f06a4e1ae281e1430ef304cdcb.jpg)

![79a8da4c7d6684fecfe75171d6690bc7fa2fee451e8baa7dce8570fd79a52f59.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/tables/79a8da4c7d6684fecfe75171d6690bc7fa2fee451e8baa7dce8570fd79a52f59.jpg)

![7db084e094f7c7413f6407f121f75fc3885c3586a1104a3550b6f9831714c31c.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/tables/7db084e094f7c7413f6407f121f75fc3885c3586a1104a3550b6f9831714c31c.jpg)

![c0aa2e7afd8d26559e16fff5d1391c1ccfaa7ca3b23fefc4f32fa36e22a02312.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/tables/c0aa2e7afd8d26559e16fff5d1391c1ccfaa7ca3b23fefc4f32fa36e22a02312.jpg)

![c3a4feccdc5fdc951b2e787bff9f653f2fb273178a56e25a52302705e3245981.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/tables/c3a4feccdc5fdc951b2e787bff9f653f2fb273178a56e25a52302705e3245981.jpg)

![cd080a6ba98206dd2cbf02656cd1de2f4aaae9e94255fd3992b06f2e0c933553.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/tables/cd080a6ba98206dd2cbf02656cd1de2f4aaae9e94255fd3992b06f2e0c933553.jpg)

![cdbe3fa5a1fd3514c5c5c6ceafe847d051c727e97463405aac2ffeaab932315c.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/tables/cdbe3fa5a1fd3514c5c5c6ceafe847d051c727e97463405aac2ffeaab932315c.jpg)

![d404180d0e0b612299d74ca6798e4554b748f151608eaa6a91db0fd7966764f9.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/tables/d404180d0e0b612299d74ca6798e4554b748f151608eaa6a91db0fd7966764f9.jpg)

![d6fdfe422b17df59979345cacfd0a67fbc6ca9a547da38d8bbfd35fea7e7b975.jpg](../icml_results/4017_The%20Hidden%20Cost%20of%20Structured%20Generation%20in%20LLMs_%20Draft-Conditioned%20Constrained%20Decoding/tables/d6fdfe422b17df59979345cacfd0a67fbc6ca9a547da38d8bbfd35fea7e7b975.jpg)

<a id="paper-3977"></a>

## AdaSCALE: Adaptive Scaling for OOD Detection


### Images

![052daf71c05b85c9977219e9ac50175a966dd6fdaab3b672a8aaa91ac741c098.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/images/052daf71c05b85c9977219e9ac50175a966dd6fdaab3b672a8aaa91ac741c098.jpg)

![2332b423df977c9b465a4220e18db2dd98b0fb1dcb089f79e6faaadd1eec1f79.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/images/2332b423df977c9b465a4220e18db2dd98b0fb1dcb089f79e6faaadd1eec1f79.jpg)

![49a23719a597b117cd7d1ec48180568f8d8b6555782acecc9a757f76179a2b29.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/images/49a23719a597b117cd7d1ec48180568f8d8b6555782acecc9a757f76179a2b29.jpg)

![743a3cbd670477546bab8c5f19f4dd8e49d62aba23afca900d5e12e6f5708f94.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/images/743a3cbd670477546bab8c5f19f4dd8e49d62aba23afca900d5e12e6f5708f94.jpg)

![832e79b205dd3965399c599c23ee851a041c4ef2573739ed233c6b0700607ebe.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/images/832e79b205dd3965399c599c23ee851a041c4ef2573739ed233c6b0700607ebe.jpg)

![ce1fe20ded831d05f88d1f6f3f48a5f792c34d0a38b34107f108e2d9bd01dc70.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/images/ce1fe20ded831d05f88d1f6f3f48a5f792c34d0a38b34107f108e2d9bd01dc70.jpg)

### Tables

![095c975c287d18d1af4b85438b4ab04fc155b821f7cfe94b7c0cd7e408a490f8.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/095c975c287d18d1af4b85438b4ab04fc155b821f7cfe94b7c0cd7e408a490f8.jpg)

![0c91fd423149dfbbf928e90b544a025453f15bd0cdd844ee7fadf7c6dc36d5a5.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/0c91fd423149dfbbf928e90b544a025453f15bd0cdd844ee7fadf7c6dc36d5a5.jpg)

![0fa2e1d4a7bbe703eda60a05ff39f8c0e9b147a0146518014453faa6c3eb080c.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/0fa2e1d4a7bbe703eda60a05ff39f8c0e9b147a0146518014453faa6c3eb080c.jpg)

![197fccf633f4b9ade65a341ae4298bcf586c8443717cb3f864d5a5cd4a37a89e.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/197fccf633f4b9ade65a341ae4298bcf586c8443717cb3f864d5a5cd4a37a89e.jpg)

![1ac0f98c3f6f44d4d4843b936e7f999ae65e14868160af2928da2c6a0743fa53.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/1ac0f98c3f6f44d4d4843b936e7f999ae65e14868160af2928da2c6a0743fa53.jpg)

![1c5925b2d94049243d06c594b4268061a3db0653b780de2ac6f741179ffa9aee.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/1c5925b2d94049243d06c594b4268061a3db0653b780de2ac6f741179ffa9aee.jpg)

![23251839537765c8e8f20d58bd1790927e46052ebc48cc380908a47365c9a02d.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/23251839537765c8e8f20d58bd1790927e46052ebc48cc380908a47365c9a02d.jpg)

![23a26b7fb66c10a13f4869dafafb1f92eec528b6611c9ba8821545041d43bff0.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/23a26b7fb66c10a13f4869dafafb1f92eec528b6611c9ba8821545041d43bff0.jpg)

![2be6eb79085f0779c709fb60d404854d14a67d2cf430805a82636ad2e654f460.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/2be6eb79085f0779c709fb60d404854d14a67d2cf430805a82636ad2e654f460.jpg)

![2d3ec0f4e93a7f29e5f97c52f23105bb80ead311ac3f5b9a34663af7db733702.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/2d3ec0f4e93a7f29e5f97c52f23105bb80ead311ac3f5b9a34663af7db733702.jpg)

![3e1b884fcd1a772267f78ab34f05d58d22c4ecf21f148c1137dc7b1170ffc915.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/3e1b884fcd1a772267f78ab34f05d58d22c4ecf21f148c1137dc7b1170ffc915.jpg)

![419197f0083f620682dc43273f25f49063b3cb0d6ac63e9701693de4822b5269.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/419197f0083f620682dc43273f25f49063b3cb0d6ac63e9701693de4822b5269.jpg)

![42b0db83b53ab74b556b97d14aa869a0370a5a1078ab72c25643aab2613b24e9.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/42b0db83b53ab74b556b97d14aa869a0370a5a1078ab72c25643aab2613b24e9.jpg)

![452dcbab0a6f38d6947408cf4f557dd5288490964dec022a1c7219622631ea03.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/452dcbab0a6f38d6947408cf4f557dd5288490964dec022a1c7219622631ea03.jpg)

![4916cdf3b6732348bade28669afe36487097533703c126b4b407d2b6618d2f01.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/4916cdf3b6732348bade28669afe36487097533703c126b4b407d2b6618d2f01.jpg)

![499926fd5f63847607f94f89ef7b68e57ba021751ef4762629663f4fa2d2ff15.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/499926fd5f63847607f94f89ef7b68e57ba021751ef4762629663f4fa2d2ff15.jpg)

![49b19bc6f73b611aaec260df5c95f7c663d5e9c71577c8929bcd903cc603fe29.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/49b19bc6f73b611aaec260df5c95f7c663d5e9c71577c8929bcd903cc603fe29.jpg)

![4ceb742d4b4e21355ebd23db50ad2d2f6257b25835229c5f8005591cd553c302.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/4ceb742d4b4e21355ebd23db50ad2d2f6257b25835229c5f8005591cd553c302.jpg)

![4d1798154cc5e2266ed2799b46bc15d8bfb4c823e4a7b5e9b8244a683f10c271.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/4d1798154cc5e2266ed2799b46bc15d8bfb4c823e4a7b5e9b8244a683f10c271.jpg)

![4e1aa909a925ac93398875f9a58a6ef8a73970a9436dbea48a28e0a4ec75e1ac.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/4e1aa909a925ac93398875f9a58a6ef8a73970a9436dbea48a28e0a4ec75e1ac.jpg)

![5131a09a2583cac1c280b0e8e20c5a63c4fe1d4652516bd3e390690e8910b15d.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/5131a09a2583cac1c280b0e8e20c5a63c4fe1d4652516bd3e390690e8910b15d.jpg)

![52251c06e8305068734c57cf22d4c096f0fe8ba2298985f8fecbf41536d6b594.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/52251c06e8305068734c57cf22d4c096f0fe8ba2298985f8fecbf41536d6b594.jpg)

![55edc0f448cc10e831c9065af6ded63e2a0e62f216a0f6f90e309ae847c614bf.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/55edc0f448cc10e831c9065af6ded63e2a0e62f216a0f6f90e309ae847c614bf.jpg)

![567c475f7b5fd47e67785eca48ef93a4835178153eada5565d8a1715e295f0ee.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/567c475f7b5fd47e67785eca48ef93a4835178153eada5565d8a1715e295f0ee.jpg)

![5a70654bffb604f4b63b5ad8e98dec1c5236754568658d89c38cfe920a8b6cdd.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/5a70654bffb604f4b63b5ad8e98dec1c5236754568658d89c38cfe920a8b6cdd.jpg)

![5b2abdeafa0d473bc218deee93be9b4fca244c2fa608e1b622124e4512e988a0.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/5b2abdeafa0d473bc218deee93be9b4fca244c2fa608e1b622124e4512e988a0.jpg)

![5b7a66c72da341a6b47e4ce937b6f8a5f0a27d240aa50649323988676de41cf7.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/5b7a66c72da341a6b47e4ce937b6f8a5f0a27d240aa50649323988676de41cf7.jpg)

![5e7df43ef0494e994c66a5986045dbcd20a92c74aa58c53609b26f3556a7ff53.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/5e7df43ef0494e994c66a5986045dbcd20a92c74aa58c53609b26f3556a7ff53.jpg)

![60dfde91a05ee2dcb8936c151145b1e58feee8eb6e8ec6ad7984525ea7e425c5.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/60dfde91a05ee2dcb8936c151145b1e58feee8eb6e8ec6ad7984525ea7e425c5.jpg)

![672bb28a8730851bdd452a263e498f3761d4bff46d2a6b6c75be3a2976d49faa.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/672bb28a8730851bdd452a263e498f3761d4bff46d2a6b6c75be3a2976d49faa.jpg)

![67a8565a9801b2e0f833a2dbf05292b51d99680d459aa07db7aff5ab76e7edac.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/67a8565a9801b2e0f833a2dbf05292b51d99680d459aa07db7aff5ab76e7edac.jpg)

![69023e2b3f518050a9ac80b4122e3cc1565298a678d3e445ee00adc9283bdf92.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/69023e2b3f518050a9ac80b4122e3cc1565298a678d3e445ee00adc9283bdf92.jpg)

![6f374be6dea92c769985b6192714a60ae9f5ad46e7172d4626855353b20039ef.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/6f374be6dea92c769985b6192714a60ae9f5ad46e7172d4626855353b20039ef.jpg)

![71b28613621e1f5789332dfca68bb3a9205ee55c21a762d72dc3b2171387331f.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/71b28613621e1f5789332dfca68bb3a9205ee55c21a762d72dc3b2171387331f.jpg)

![743cdf59b0196b4dd6ee43ab93a071d88b691bc1e62c1bfb2bbf13aa49c5648d.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/743cdf59b0196b4dd6ee43ab93a071d88b691bc1e62c1bfb2bbf13aa49c5648d.jpg)

![757027a09f0db9fffbd11fd1de6c8dc1cdcb6603dd5ba3dc638371b40e54647a.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/757027a09f0db9fffbd11fd1de6c8dc1cdcb6603dd5ba3dc638371b40e54647a.jpg)

![8310bc89ce5dd67f0e45e2cdd5c63444cde84e337fdbf9f8859a9ba0a23b83a4.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/8310bc89ce5dd67f0e45e2cdd5c63444cde84e337fdbf9f8859a9ba0a23b83a4.jpg)

![83590b40b5a31fe1f13d1a9987c6a260e67cfe76ca520ed72282865ea76031db.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/83590b40b5a31fe1f13d1a9987c6a260e67cfe76ca520ed72282865ea76031db.jpg)

![88a1f48ec5d02d61aaefdce9bfdd26cad004077f243241ea064266a28e3210d7.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/88a1f48ec5d02d61aaefdce9bfdd26cad004077f243241ea064266a28e3210d7.jpg)

![8b988220bb8cdce138534e314c77480fc60c9bd2cc85dcfca175705fe0ae2825.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/8b988220bb8cdce138534e314c77480fc60c9bd2cc85dcfca175705fe0ae2825.jpg)

![8d1d1ca06709485391fb48762f4e7c90c1f446978ff0b0353f4ad3c600aa5230.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/8d1d1ca06709485391fb48762f4e7c90c1f446978ff0b0353f4ad3c600aa5230.jpg)

![92f281c0d558cc30f6cb494b233e7c8da21d3cce2d95d2d23c41740e81c53a8d.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/92f281c0d558cc30f6cb494b233e7c8da21d3cce2d95d2d23c41740e81c53a8d.jpg)

![9582124fd5488e3b5c33a7cef65b72f4d3096d96ce1c24f98c44c3080bc81950.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/9582124fd5488e3b5c33a7cef65b72f4d3096d96ce1c24f98c44c3080bc81950.jpg)

![96a1841d751019029bff9ad737abc4ea73fb22272becb3fe3e3a7816b3f7e9cb.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/96a1841d751019029bff9ad737abc4ea73fb22272becb3fe3e3a7816b3f7e9cb.jpg)

![a746436686d763185bfcdead9e78d697fbd4aafbbe323fbc719ce6739978efda.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/a746436686d763185bfcdead9e78d697fbd4aafbbe323fbc719ce6739978efda.jpg)

![a82b19d12310029b5f426b09aacffdc3b37881a5ef89c50ab1eefb5f6f6542a2.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/a82b19d12310029b5f426b09aacffdc3b37881a5ef89c50ab1eefb5f6f6542a2.jpg)

![a96284c1f803b858f74c23c4d1513eff964b313e47fd8290898c532710eba138.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/a96284c1f803b858f74c23c4d1513eff964b313e47fd8290898c532710eba138.jpg)

![ab220a35a1656614753423166f0d9080346b62bf96f13490558abd09dad01025.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/ab220a35a1656614753423166f0d9080346b62bf96f13490558abd09dad01025.jpg)

![aef8a682e68d718e1d25d5f9e8d1a68675ffb2ae93adb9deb589f03978291822.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/aef8a682e68d718e1d25d5f9e8d1a68675ffb2ae93adb9deb589f03978291822.jpg)

![b59fe9fde4e00cc55b977ccd4bd2b931925e53acd37fd23e9aae050dfeba2c98.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/b59fe9fde4e00cc55b977ccd4bd2b931925e53acd37fd23e9aae050dfeba2c98.jpg)

![b5a226004a7cb0f2417558308a94031079e5580a28340664e78f2b57468c2e83.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/b5a226004a7cb0f2417558308a94031079e5580a28340664e78f2b57468c2e83.jpg)

![b5ab2f3ad83cc8377ec4a33d541f4de1cff3d65019a2442503df0a8433c347b0.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/b5ab2f3ad83cc8377ec4a33d541f4de1cff3d65019a2442503df0a8433c347b0.jpg)

![b72e61507fef5644f4ebd84bfbd6a406cfdd75dc651767e23860805b84293181.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/b72e61507fef5644f4ebd84bfbd6a406cfdd75dc651767e23860805b84293181.jpg)

![c8295b422be47e3276f78e7b83e51044d24dfbc3d07c64cf96f594e42cc1bfb7.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/c8295b422be47e3276f78e7b83e51044d24dfbc3d07c64cf96f594e42cc1bfb7.jpg)

![cbb77d24338bbc8bef61d85676fd3859f11adca27b29f2b129ad39d89a63ffc5.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/cbb77d24338bbc8bef61d85676fd3859f11adca27b29f2b129ad39d89a63ffc5.jpg)

![cc83ee37d859aafb5a751afe408f5dd3ac10cb6f12b59b36cc16ffb77d671d5a.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/cc83ee37d859aafb5a751afe408f5dd3ac10cb6f12b59b36cc16ffb77d671d5a.jpg)

![cf5ae9ee74e3423d9aba166ba9b6e4aff5dca661f5979d4c29e71a4e40f9b729.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/cf5ae9ee74e3423d9aba166ba9b6e4aff5dca661f5979d4c29e71a4e40f9b729.jpg)

![d5ed71fc76d32ae00370654f3d01545afcbd713b1c0d5dd8ad5a0a0336f745ca.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/d5ed71fc76d32ae00370654f3d01545afcbd713b1c0d5dd8ad5a0a0336f745ca.jpg)

![d923d04699eb1244258055c05d6dc70df4829f962c069b7d591042351419ea8c.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/d923d04699eb1244258055c05d6dc70df4829f962c069b7d591042351419ea8c.jpg)

![da8eed6cb4d9d598211c88e8e087f048a79bf12b33c8f7787390e6a95dfc2185.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/da8eed6cb4d9d598211c88e8e087f048a79bf12b33c8f7787390e6a95dfc2185.jpg)

![db31c90606336f5862d845c222b653fbb7cb0b83bb1dd5fec09f9871f073c273.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/db31c90606336f5862d845c222b653fbb7cb0b83bb1dd5fec09f9871f073c273.jpg)

![e2f7b4617c30c3626aac576f2dc251c3eee5b777f13e84763ee0cffa7d059836.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/e2f7b4617c30c3626aac576f2dc251c3eee5b777f13e84763ee0cffa7d059836.jpg)

![e30c2fd2bde94292ef31a17a987bc2c4a69d1ac2d9876f5d4b53f2c33d554f06.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/e30c2fd2bde94292ef31a17a987bc2c4a69d1ac2d9876f5d4b53f2c33d554f06.jpg)

![e4edff2082253dfd8bc8f3ea137d3e1c085f4666492bef4a0af0b15e3d42a692.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/e4edff2082253dfd8bc8f3ea137d3e1c085f4666492bef4a0af0b15e3d42a692.jpg)

![e57254e59739defba645f8341a03270eac338f61b0c880564bf8a3a6f228bd47.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/e57254e59739defba645f8341a03270eac338f61b0c880564bf8a3a6f228bd47.jpg)

![e752209925c5def9a191dc5812089bf906559723c637a009ce120963651907f7.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/e752209925c5def9a191dc5812089bf906559723c637a009ce120963651907f7.jpg)

![e8f5ac2ab5134888289fb1ac342ce4765437de5d4f1117f4d53b1f02f171c0b3.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/e8f5ac2ab5134888289fb1ac342ce4765437de5d4f1117f4d53b1f02f171c0b3.jpg)

![eb6c974aab8639917be150d29e7299ede64afd74f0109373710c05f2e0f28936.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/eb6c974aab8639917be150d29e7299ede64afd74f0109373710c05f2e0f28936.jpg)

![f88530700f903326c7aeae35ae3e394fb65634fc566d0646b951f4018eea417b.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/f88530700f903326c7aeae35ae3e394fb65634fc566d0646b951f4018eea417b.jpg)

![facd7bb4005f080262f5fb189fc57b2efdd1146b77b23991902de5023d93df47.jpg](../icml_results/4018_AdaSCALE_%20Adaptive%20Scaling%20for%20OOD%20Detection/tables/facd7bb4005f080262f5fb189fc57b2efdd1146b77b23991902de5023d93df47.jpg)

<a id="paper-3978"></a>

## Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization


### Images

![075064e0c658fe9f90d4cd2c1eceaef8a4002206f15bb2d8bbfd2affd66e5255.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/images/075064e0c658fe9f90d4cd2c1eceaef8a4002206f15bb2d8bbfd2affd66e5255.jpg)

![0caa352d124e2a6ecc7ab01e57ed90cbcaf9f911d568aeeca75a8454e3ee4b74.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/images/0caa352d124e2a6ecc7ab01e57ed90cbcaf9f911d568aeeca75a8454e3ee4b74.jpg)

![2b43f2660995072d5337f65493643a05f1d288c830e1958b4160c5a484094055.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/images/2b43f2660995072d5337f65493643a05f1d288c830e1958b4160c5a484094055.jpg)

![4cffd6c9b84812d4037f67b8d0eea3e362e40007b783bfc019e36da17e9f24ca.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/images/4cffd6c9b84812d4037f67b8d0eea3e362e40007b783bfc019e36da17e9f24ca.jpg)

![562d1d8a9a18683955f2afd6ce249ab4357911b7e16d5278fce210587b9c14ef.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/images/562d1d8a9a18683955f2afd6ce249ab4357911b7e16d5278fce210587b9c14ef.jpg)

![5df04daa7ae81a1c2846f533f686b873baaf749c09ccca786e13611a046753b7.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/images/5df04daa7ae81a1c2846f533f686b873baaf749c09ccca786e13611a046753b7.jpg)

![7331587391b093ef37578993b01ffe8a8628bba307c57ca00a494a0bd1eba3a5.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/images/7331587391b093ef37578993b01ffe8a8628bba307c57ca00a494a0bd1eba3a5.jpg)

![80ea52378a16433a47efe202403818e653de5c3f6ab615a325b6a4766deb08dc.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/images/80ea52378a16433a47efe202403818e653de5c3f6ab615a325b6a4766deb08dc.jpg)

![b07ffce8a55c17c5c2fb1320bdf6ac2f28d973dae92412441bf3d902f30cc0a9.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/images/b07ffce8a55c17c5c2fb1320bdf6ac2f28d973dae92412441bf3d902f30cc0a9.jpg)

![bcaf4e0ac8b30128d11c450b9181eff14cdec722e3d506fd5aac6e27c6555d51.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/images/bcaf4e0ac8b30128d11c450b9181eff14cdec722e3d506fd5aac6e27c6555d51.jpg)

![c4f538fc0bcc52f88af7cc9d61c51cd1d7f8260a703199f72a8e73c855867220.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/images/c4f538fc0bcc52f88af7cc9d61c51cd1d7f8260a703199f72a8e73c855867220.jpg)

![fba91d202d548be11b2fc9b267e777fbf61f3d52212d7d596e39a0f387222dd8.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/images/fba91d202d548be11b2fc9b267e777fbf61f3d52212d7d596e39a0f387222dd8.jpg)

### Tables

![0cfc1d1c9b76917dda2c18bc49c55884b9649cb10fa9a8e02a0e5b58a06991d2.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/tables/0cfc1d1c9b76917dda2c18bc49c55884b9649cb10fa9a8e02a0e5b58a06991d2.jpg)

![33fc11d1f509e8cc296ff7a4408540c00e4478b280518f310bb89a9c45a38920.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/tables/33fc11d1f509e8cc296ff7a4408540c00e4478b280518f310bb89a9c45a38920.jpg)

![42f4b4b878341b355955e2055f7b671b168e331fd3cef636b027fdca9c3fc4c4.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/tables/42f4b4b878341b355955e2055f7b671b168e331fd3cef636b027fdca9c3fc4c4.jpg)

![44789d60915cd1ab05a7a0308e437f4943e4cc43d6feda49b1234f3b65353db4.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/tables/44789d60915cd1ab05a7a0308e437f4943e4cc43d6feda49b1234f3b65353db4.jpg)

![4e431988dc0cf0fa8f375957de8310f6dca0b47ec8e230455b3f4909611cb12f.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/tables/4e431988dc0cf0fa8f375957de8310f6dca0b47ec8e230455b3f4909611cb12f.jpg)

![5c348b495f121d582511af31bd3df3c89933085a7bb97c8488a8b752c1080c77.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/tables/5c348b495f121d582511af31bd3df3c89933085a7bb97c8488a8b752c1080c77.jpg)

![679467f970e8bf71b5486f65dfe6f6c11acd142484d2561ccbeff916b119bcd2.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/tables/679467f970e8bf71b5486f65dfe6f6c11acd142484d2561ccbeff916b119bcd2.jpg)

![6e070c66bea34d9f8678c634d286513e37536f0a9beb49d8723e5b7cda5874c0.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/tables/6e070c66bea34d9f8678c634d286513e37536f0a9beb49d8723e5b7cda5874c0.jpg)

![9f1b9cf0f99c07389cd1f157c0dc15d284322b08cf706c5a06afe00f0ed49ddb.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/tables/9f1b9cf0f99c07389cd1f157c0dc15d284322b08cf706c5a06afe00f0ed49ddb.jpg)

![e0ee25e0ad43e12c15a89547c84b719438c2edbe7234c91935d684fae99744fc.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/tables/e0ee25e0ad43e12c15a89547c84b719438c2edbe7234c91935d684fae99744fc.jpg)

![e6d2903c22509d9d77647da726c5cf5a192d573751059584ecd12f3e86eeace5.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/tables/e6d2903c22509d9d77647da726c5cf5a192d573751059584ecd12f3e86eeace5.jpg)

![fa7b653fa3f5261cbdc2621fd7857999a8e5d2a3ca07891bbaa4035e98fc4213.jpg](../icml_results/4019_Scaling%20Unsupervised%20Multi-Source%20Federated%20Domain%20Adaptation%20through%20Group-Wise%20Discrepancy%20Minimiz/tables/fa7b653fa3f5261cbdc2621fd7857999a8e5d2a3ca07891bbaa4035e98fc4213.jpg)

<a id="paper-3979"></a>

## Leak@$k$: Unlearning Does Not Make LLMs Forget Under Probabilistic Decoding


### Images

![13c1506cc78ad117f90a5c293a8550ea249b2013fa5e48887aee4cc5e0fa8045.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/images/13c1506cc78ad117f90a5c293a8550ea249b2013fa5e48887aee4cc5e0fa8045.jpg)

![207a083d843cca53f83723dab6f7f78fbac08929a55bcdc2a10b9ca7f9b831d1.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/images/207a083d843cca53f83723dab6f7f78fbac08929a55bcdc2a10b9ca7f9b831d1.jpg)

![2116f406120ff7dd6b5c13d5773a08aeffc877608a4b970455e1e7d937b27e8f.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/images/2116f406120ff7dd6b5c13d5773a08aeffc877608a4b970455e1e7d937b27e8f.jpg)

![4cd1f0a4a263f0332a674401787d5be0b4bb4dd44fa3e7af1b3dc64b464a32d5.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/images/4cd1f0a4a263f0332a674401787d5be0b4bb4dd44fa3e7af1b3dc64b464a32d5.jpg)

![513900b48fb69357d95d88c0e79786fb857609142a101eb7aef4671f7fe4b772.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/images/513900b48fb69357d95d88c0e79786fb857609142a101eb7aef4671f7fe4b772.jpg)

![5dcc9b2c7024be4e2f67a33324fbe733cfa3a682add22a81cdd7c293075dc01f.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/images/5dcc9b2c7024be4e2f67a33324fbe733cfa3a682add22a81cdd7c293075dc01f.jpg)

![85eb344223edee073a081791ac061b06fbfc7235ab7c7310071704d872df1307.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/images/85eb344223edee073a081791ac061b06fbfc7235ab7c7310071704d872df1307.jpg)

![909558ecfe097104600c4c8370332bf29a45b0c91090ece614f72c4948217723.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/images/909558ecfe097104600c4c8370332bf29a45b0c91090ece614f72c4948217723.jpg)

![ac7802f2aa6215f51737d6ba5b5b77b25742ecee49fa6d3f64dfdbc938a914f4.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/images/ac7802f2aa6215f51737d6ba5b5b77b25742ecee49fa6d3f64dfdbc938a914f4.jpg)

![bac5516cf438c76ac32b88b2f1f5b08cd34f970c2e98167d9a28e6c306dcd706.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/images/bac5516cf438c76ac32b88b2f1f5b08cd34f970c2e98167d9a28e6c306dcd706.jpg)

![ce4059b94629313feeb2dbe3d9cdb6144eb3d1e0046f6dd1d5d9e4224d6884fa.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/images/ce4059b94629313feeb2dbe3d9cdb6144eb3d1e0046f6dd1d5d9e4224d6884fa.jpg)

![de6a00bc0f60885d6d6437de039c48af4ad385c64faf668dbf286325ec959d7d.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/images/de6a00bc0f60885d6d6437de039c48af4ad385c64faf668dbf286325ec959d7d.jpg)

![e44c91ef17a55c232ef5a6c0a6f6c3e013449f1e153c5d92e4e978df6b09122c.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/images/e44c91ef17a55c232ef5a6c0a6f6c3e013449f1e153c5d92e4e978df6b09122c.jpg)

![fc09fa096609268f99183c7cf67945d58a20256a92a3bbbadd8a0eb6d07879fe.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/images/fc09fa096609268f99183c7cf67945d58a20256a92a3bbbadd8a0eb6d07879fe.jpg)

### Tables

![134c5d57a54bdbdd17303e67dd43c4d943a09fedc46ce9a0ac3fe33d3456923d.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/134c5d57a54bdbdd17303e67dd43c4d943a09fedc46ce9a0ac3fe33d3456923d.jpg)

![1528b69e71bf1a7285cf96b8aaa6bd3957671822dfe09f612f6e27fbc2f0f590.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/1528b69e71bf1a7285cf96b8aaa6bd3957671822dfe09f612f6e27fbc2f0f590.jpg)

![16f2abcc8a645e0f749aa9b7ddcb27669bd0cc702abe4ab7ef61b75768cdee24.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/16f2abcc8a645e0f749aa9b7ddcb27669bd0cc702abe4ab7ef61b75768cdee24.jpg)

![36efbae2fa1fcfdccb90951e48d8b5f35979943f042046811e6c33f05fc53372.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/36efbae2fa1fcfdccb90951e48d8b5f35979943f042046811e6c33f05fc53372.jpg)

![3c5b5cb4aa5de7501ce8b47c4e4ad73894c2f21dff1342bb35eaac1780fb1382.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/3c5b5cb4aa5de7501ce8b47c4e4ad73894c2f21dff1342bb35eaac1780fb1382.jpg)

![3e40a04b1d85df684d8dd2b11ec293e8087845c0343f81c3e3c2bd957b9391dc.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/3e40a04b1d85df684d8dd2b11ec293e8087845c0343f81c3e3c2bd957b9391dc.jpg)

![4aec33dadc1bfbe4dbc887ab6f4c04e1d86c2262dc485c0014ffbdc28b2b35e8.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/4aec33dadc1bfbe4dbc887ab6f4c04e1d86c2262dc485c0014ffbdc28b2b35e8.jpg)

![5d06589f69f9a20899f64fd91d6e0f822872081fd3b8ac11355be2eb11c0c56b.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/5d06589f69f9a20899f64fd91d6e0f822872081fd3b8ac11355be2eb11c0c56b.jpg)

![63cd12682feb974eb1f271474a3aaa9fff91a215e4601c50edd594a1adf5f059.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/63cd12682feb974eb1f271474a3aaa9fff91a215e4601c50edd594a1adf5f059.jpg)

![6a0ee973cc3a8850aa2c8ae887d1dea859e4940b184a3ff34bd003ea0d2acf71.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/6a0ee973cc3a8850aa2c8ae887d1dea859e4940b184a3ff34bd003ea0d2acf71.jpg)

![6a993b548754e61b797f9df0d6289e5730f0f679f06c68dfe6aebd313a423f89.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/6a993b548754e61b797f9df0d6289e5730f0f679f06c68dfe6aebd313a423f89.jpg)

![793e097fa1448b8f9b4b42af39f2cfee6a995f4a9c516769cf55838bd952d4ec.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/793e097fa1448b8f9b4b42af39f2cfee6a995f4a9c516769cf55838bd952d4ec.jpg)

![807e68c18214fd1a1bdcf136e6fcc921814c05ccc59ad6c525a9e9cbe27a62cc.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/807e68c18214fd1a1bdcf136e6fcc921814c05ccc59ad6c525a9e9cbe27a62cc.jpg)

![83b3816b94f3bcc0f09bac0625a1151408c1554523d64ea7299d3b657ea4b6e1.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/83b3816b94f3bcc0f09bac0625a1151408c1554523d64ea7299d3b657ea4b6e1.jpg)

![9509e77b553fb311171d69c291c8a485355b1ad0f3291a856d2139bb57327461.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/9509e77b553fb311171d69c291c8a485355b1ad0f3291a856d2139bb57327461.jpg)

![b20200bc9be205faa2fb4239931ab4626d48df288fe9bb10aaba9c29dbd494b1.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/b20200bc9be205faa2fb4239931ab4626d48df288fe9bb10aaba9c29dbd494b1.jpg)

![b864af7fba0d6de2023431c58138dae5e28a63297a161a6a4aff9608aa7c8bef.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/b864af7fba0d6de2023431c58138dae5e28a63297a161a6a4aff9608aa7c8bef.jpg)

![d304f88b00cc85e5aa178e400fbccf510555921f970cd9645ca9eb2ac932d68e.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/d304f88b00cc85e5aa178e400fbccf510555921f970cd9645ca9eb2ac932d68e.jpg)

![d3a664eae75cdb2fdafb26301d9d073c102d022e6096193e749ad46ae4de293c.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/d3a664eae75cdb2fdafb26301d9d073c102d022e6096193e749ad46ae4de293c.jpg)

![d77e289044ff3bd1f3b12dd24764b89209005f3e2b65c1f0adbe7a5aaf45e35e.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/d77e289044ff3bd1f3b12dd24764b89209005f3e2b65c1f0adbe7a5aaf45e35e.jpg)

![e8c80f81c1b25d79737ec15f5b32d9528db5a4c175afeccd646f5c0514dd1798.jpg](../icml_results/4020_Leak%40%24k%24_%20Unlearning%20Does%20Not%20Make%20LLMs%20Forget%20Under%20Probabilistic%20Decoding/tables/e8c80f81c1b25d79737ec15f5b32d9528db5a4c175afeccd646f5c0514dd1798.jpg)

<a id="paper-3980"></a>

## Alterbute: Editing Intrinsic Attributes of Objects in Images


### Images

![056e6d9bc2547bf5d6aa85e4270174c2a9aa6fee0649ec651f4261f4b0071239.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/056e6d9bc2547bf5d6aa85e4270174c2a9aa6fee0649ec651f4261f4b0071239.jpg)

![0d9f0f23462db1acd1fd96e55436ca2a013a7f010f8256a2803c18f3ac0f2adc.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/0d9f0f23462db1acd1fd96e55436ca2a013a7f010f8256a2803c18f3ac0f2adc.jpg)

![11380c8fec95ca01db8140f0a2b0ec4698d7067bdf03e6226d7623bcb397cc86.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/11380c8fec95ca01db8140f0a2b0ec4698d7067bdf03e6226d7623bcb397cc86.jpg)

![1c253ff80027bc410083edec7d61396b43749328f016fd856a57558ce2700ecd.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/1c253ff80027bc410083edec7d61396b43749328f016fd856a57558ce2700ecd.jpg)

![1c2b6283931d6a52a20c2d96b38256e68450e65154e537c76e9fd4ad306409da.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/1c2b6283931d6a52a20c2d96b38256e68450e65154e537c76e9fd4ad306409da.jpg)

![230783fa34ff34d75ce411d7082614a70fcc3f87559191e7c321d28ba4aa9114.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/230783fa34ff34d75ce411d7082614a70fcc3f87559191e7c321d28ba4aa9114.jpg)

![3872e9560d6b4821c38533e9aff58eac3ef5a2421227fbedf70af17efb4b405c.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/3872e9560d6b4821c38533e9aff58eac3ef5a2421227fbedf70af17efb4b405c.jpg)

![388a0317a7984cfc2b4abee4b23aa0d5731966ce25ec966425adcabff0a91e91.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/388a0317a7984cfc2b4abee4b23aa0d5731966ce25ec966425adcabff0a91e91.jpg)

![49e7feac00e08f13f2523f63e2eeb99a33c8ea10b683a02b75ac60188397042d.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/49e7feac00e08f13f2523f63e2eeb99a33c8ea10b683a02b75ac60188397042d.jpg)

![4cdcd6c3d3ebde7ba23ddfab5c3c0499194474b0775ea1f10c132988e78a1628.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/4cdcd6c3d3ebde7ba23ddfab5c3c0499194474b0775ea1f10c132988e78a1628.jpg)

![6e931aa573bada167467c9a061c702742cf30f74a73c160300f34a1a3f169aa9.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/6e931aa573bada167467c9a061c702742cf30f74a73c160300f34a1a3f169aa9.jpg)

![87ca46b83f2a18c2917f90e61d27aafe06a2f9d3d1421a9896ae1a789f5cf396.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/87ca46b83f2a18c2917f90e61d27aafe06a2f9d3d1421a9896ae1a789f5cf396.jpg)

![8a0a2221a4fa86107ecb6e3f797f60b2250d2f82a5d201ecb65b5dfb3a77ccf0.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/8a0a2221a4fa86107ecb6e3f797f60b2250d2f82a5d201ecb65b5dfb3a77ccf0.jpg)

![8ad47ce96d6169e1a3200363ce82d24b304d7b9b8306a8c1ab5ad8113f9a821b.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/8ad47ce96d6169e1a3200363ce82d24b304d7b9b8306a8c1ab5ad8113f9a821b.jpg)

![bc85bc3cdf253e5ea4eceb2cbd5bfe1d84ae5410cfeb1b7408f38ff48a10efd0.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/bc85bc3cdf253e5ea4eceb2cbd5bfe1d84ae5410cfeb1b7408f38ff48a10efd0.jpg)

![bfa01d4dbb526301b0ee037229b17c9f5b22179e515e66831315c286a20f8491.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/bfa01d4dbb526301b0ee037229b17c9f5b22179e515e66831315c286a20f8491.jpg)

![c368bfff46d429abdcdd1415cc730d006828935b35a0b2ed8a229e9a72bbf694.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/c368bfff46d429abdcdd1415cc730d006828935b35a0b2ed8a229e9a72bbf694.jpg)

![ca5d7657d83093ba447b86064fafea89327e883309b0cbd0ef1485af5f52754c.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/ca5d7657d83093ba447b86064fafea89327e883309b0cbd0ef1485af5f52754c.jpg)

![cf0702991a5b7f7b2259a2ec9d4569295dae9c5ed0074de7d17e904593ce1d05.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/cf0702991a5b7f7b2259a2ec9d4569295dae9c5ed0074de7d17e904593ce1d05.jpg)

![e32a0d2e1c69808491cf56238286322cf3e1797f8701999f35da0d1beeac9ea2.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/e32a0d2e1c69808491cf56238286322cf3e1797f8701999f35da0d1beeac9ea2.jpg)

![ecc64af16741373c8fa0950112d40cad8541366894e5397c4471520e46ea56a5.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/ecc64af16741373c8fa0950112d40cad8541366894e5397c4471520e46ea56a5.jpg)

![f4aaeb808ee67c08ce0e08dda19d61ad94e8f377277ab20a5bcb1290c00366a4.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/f4aaeb808ee67c08ce0e08dda19d61ad94e8f377277ab20a5bcb1290c00366a4.jpg)

![f921b17b164dfdf1428ac4983da7ee4aa0a96ff1e8cd47ccc6b7528493dbbf0f.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/images/f921b17b164dfdf1428ac4983da7ee4aa0a96ff1e8cd47ccc6b7528493dbbf0f.jpg)

### Tables

![4bed0282e294663acba5ce4c5983c0f87b3428ffbf8b29fb64098a0e7404f66f.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/tables/4bed0282e294663acba5ce4c5983c0f87b3428ffbf8b29fb64098a0e7404f66f.jpg)

![65e8362319744b2616332e11efdb67d306b7507fe4038665d157a5c89ff4c817.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/tables/65e8362319744b2616332e11efdb67d306b7507fe4038665d157a5c89ff4c817.jpg)

![97a6ace10d111a3f6ae64def5abffe2bb83d594a35ec3dff83ba4fced45fbb90.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/tables/97a6ace10d111a3f6ae64def5abffe2bb83d594a35ec3dff83ba4fced45fbb90.jpg)

![9a3e7fda15059d2e65a61ceefed9233a8fb2ba78d304107d0a0edfdea946ea62.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/tables/9a3e7fda15059d2e65a61ceefed9233a8fb2ba78d304107d0a0edfdea946ea62.jpg)

![b8d16a33129dee105cafcf9193b8a27a68e7fc18617b079f9d84a62a54f5be4c.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/tables/b8d16a33129dee105cafcf9193b8a27a68e7fc18617b079f9d84a62a54f5be4c.jpg)

![c58bb747286ee1134b26b6bdeebe532f02d50efddebc9dbb53cad3bdd8b0fd2f.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/tables/c58bb747286ee1134b26b6bdeebe532f02d50efddebc9dbb53cad3bdd8b0fd2f.jpg)

![d69a30e4d57382e5650ea843e7854bff2666c84aca1e56fbbf47e0b00b1879ac.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/tables/d69a30e4d57382e5650ea843e7854bff2666c84aca1e56fbbf47e0b00b1879ac.jpg)

![dfd7600697b2cbb422f1911afc6335125a2940a584adcc1d10b510f54c11e6a3.jpg](../icml_results/4021_Alterbute_%20Editing%20Intrinsic%20Attributes%20of%20Objects%20in%20Images/tables/dfd7600697b2cbb422f1911afc6335125a2940a584adcc1d10b510f54c11e6a3.jpg)

<a id="paper-3981"></a>

## Correct Looks Better: Pairwise Comparisons Reveal Accuracy Rankings


### Images

![066287f37cfb26d1eaab531d4fec5a1987e9c57cdcecb3033556c5e4be232397.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/images/066287f37cfb26d1eaab531d4fec5a1987e9c57cdcecb3033556c5e4be232397.jpg)

![12f46a410183c5291ee47c3836b99df1d029ff1cf96ec576a638bbbd7228457c.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/images/12f46a410183c5291ee47c3836b99df1d029ff1cf96ec576a638bbbd7228457c.jpg)

![4ab510596f84647955a89fbab1aa168067d094c917fc4c56244e4af63550e362.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/images/4ab510596f84647955a89fbab1aa168067d094c917fc4c56244e4af63550e362.jpg)

![4d1c249cd5cc1d255bb8cb69a51c3203cc59c59d0beb36e5e12ad504e7f98418.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/images/4d1c249cd5cc1d255bb8cb69a51c3203cc59c59d0beb36e5e12ad504e7f98418.jpg)

![51a85d9781b656f21ecea328a15798b78b6c4756388ebdb47dee75b58602148d.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/images/51a85d9781b656f21ecea328a15798b78b6c4756388ebdb47dee75b58602148d.jpg)

![6a246e50b65c547f7c2bf57eb86c37d600f64f55bb9a22aef674d6d699441be0.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/images/6a246e50b65c547f7c2bf57eb86c37d600f64f55bb9a22aef674d6d699441be0.jpg)

![6c18b29107ffe53c36874274a81b036168d1d1904c92a5bc8a667db452a0dd48.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/images/6c18b29107ffe53c36874274a81b036168d1d1904c92a5bc8a667db452a0dd48.jpg)

![7b4b13f6bc6f91b6cf341cf39b88c7cd6de7752326943fe401db09c480acdce6.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/images/7b4b13f6bc6f91b6cf341cf39b88c7cd6de7752326943fe401db09c480acdce6.jpg)

![ba5cbf59c18ec93fc6e065bedac928edbbf3c8ba9ec95115a8fd09e7c417285c.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/images/ba5cbf59c18ec93fc6e065bedac928edbbf3c8ba9ec95115a8fd09e7c417285c.jpg)

![bf78a7234f1efda522eddff873d63ac3c55a2dfd2610063ff217b7eea2c96556.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/images/bf78a7234f1efda522eddff873d63ac3c55a2dfd2610063ff217b7eea2c96556.jpg)

![c437eb3b0b4a0fbdb93cfcdbbfecc729846d749bffcdac6b7b1cff3e26034716.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/images/c437eb3b0b4a0fbdb93cfcdbbfecc729846d749bffcdac6b7b1cff3e26034716.jpg)

![cff757365f33c2b7db93205acf73d8f0ce59d5c29e0db86af91da6ff765fc9dc.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/images/cff757365f33c2b7db93205acf73d8f0ce59d5c29e0db86af91da6ff765fc9dc.jpg)

### Tables

![005451ed8c5ff114768dc17961bc43f746129ccdce6924ce5f20f00e94126be3.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/tables/005451ed8c5ff114768dc17961bc43f746129ccdce6924ce5f20f00e94126be3.jpg)

![0f40aa92da6334f7e3d19b98151931e671ba0a4e2b7675b6a8d50bcdef91e2d2.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/tables/0f40aa92da6334f7e3d19b98151931e671ba0a4e2b7675b6a8d50bcdef91e2d2.jpg)

![348b1814212c0b99831d265e3d1993b858c6641e019389e19ae65f99bb7db3fd.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/tables/348b1814212c0b99831d265e3d1993b858c6641e019389e19ae65f99bb7db3fd.jpg)

![5ad4be8d5eb6ddc5b926fe9bbb1184edc47041fb501522e43b7595df980a81dc.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/tables/5ad4be8d5eb6ddc5b926fe9bbb1184edc47041fb501522e43b7595df980a81dc.jpg)

![7e9ef45ccbce86bfad4bf893fca0f163b75281b1670a597853ecc7be5834f200.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/tables/7e9ef45ccbce86bfad4bf893fca0f163b75281b1670a597853ecc7be5834f200.jpg)

![dac16fdb4d164fb4d7cba50e55a0bfe7a84fcbd5017cde49c0217a04e01ada81.jpg](../icml_results/4022_Correct%20Looks%20Better_%20Pairwise%20Comparisons%20Reveal%20Accuracy%20Rankings/tables/dac16fdb4d164fb4d7cba50e55a0bfe7a84fcbd5017cde49c0217a04e01ada81.jpg)

<a id="paper-3982"></a>

## Fisher-Preserving Guidance: Training-Free Manifold Constraints for Safe Diffusion Control


### Images

![2847d0837571b63b68904087d2e89fc5a3fdb4efe6eab49d4aba6720c763830f.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/images/2847d0837571b63b68904087d2e89fc5a3fdb4efe6eab49d4aba6720c763830f.jpg)

![2e47713bb36e48ba3c541d4545462e89bdd294ab511f2c5f3643ae35fede1d86.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/images/2e47713bb36e48ba3c541d4545462e89bdd294ab511f2c5f3643ae35fede1d86.jpg)

![30d987ee8bca53b08306eac4a58cc605867a037e663c5eb66b3a6c65e54e1f99.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/images/30d987ee8bca53b08306eac4a58cc605867a037e663c5eb66b3a6c65e54e1f99.jpg)

![4177a2ad222a5552b45fb5f62c9e249ea384aeb857ab8e6e5f6bdc1f0b17592f.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/images/4177a2ad222a5552b45fb5f62c9e249ea384aeb857ab8e6e5f6bdc1f0b17592f.jpg)

![422434cf0da6b879bc5fe54cf74f05e166ba32c1e9e332598ea074a6a8e1cd68.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/images/422434cf0da6b879bc5fe54cf74f05e166ba32c1e9e332598ea074a6a8e1cd68.jpg)

![50f0c6d214234bfecc54ac62c2e41fdd0b2fe39b71de8498c4a733857a90b0ec.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/images/50f0c6d214234bfecc54ac62c2e41fdd0b2fe39b71de8498c4a733857a90b0ec.jpg)

![738960d30ed81e1dd5a539cf179f8ff50bfe3566dd6255777caaa5bdd0035232.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/images/738960d30ed81e1dd5a539cf179f8ff50bfe3566dd6255777caaa5bdd0035232.jpg)

![7765177c9a7c1f5d699e106d427b08b8ea4238168a7fc1756d50fbb2e027734a.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/images/7765177c9a7c1f5d699e106d427b08b8ea4238168a7fc1756d50fbb2e027734a.jpg)

![91020664b2f1c5e67e70e66a6b1f130c23b714e3442ec1fbd7048214a20b5f6a.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/images/91020664b2f1c5e67e70e66a6b1f130c23b714e3442ec1fbd7048214a20b5f6a.jpg)

![ad271297d295fbb85125308fc4659269f2a8f0526826558d62f978ecd0dd5038.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/images/ad271297d295fbb85125308fc4659269f2a8f0526826558d62f978ecd0dd5038.jpg)

![ea45b5b27b0a1095955fdad53f60e3117d1b6c8fe4ed267a53255461ddf60b25.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/images/ea45b5b27b0a1095955fdad53f60e3117d1b6c8fe4ed267a53255461ddf60b25.jpg)

### Tables

![08c4e3f3cc52c5b307adde4c6500ffcb55a3a326cdb78201ea7caf94e6227630.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/tables/08c4e3f3cc52c5b307adde4c6500ffcb55a3a326cdb78201ea7caf94e6227630.jpg)

![46e24bca3b202f13210d22afd6febe3ff967fc3c1faa6998b55778b70a7adccb.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/tables/46e24bca3b202f13210d22afd6febe3ff967fc3c1faa6998b55778b70a7adccb.jpg)

![692e782f4ec29681cb71e17b907078d1cea412306fbf95b868a756d26a68b904.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/tables/692e782f4ec29681cb71e17b907078d1cea412306fbf95b868a756d26a68b904.jpg)

![860087118cea753119c3766d834b3450fb29fd1ce64fbb832ac2e7a56d27ced4.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/tables/860087118cea753119c3766d834b3450fb29fd1ce64fbb832ac2e7a56d27ced4.jpg)

![8dd92333f6b2f693ada5f79d5b48c3e4519c3c034ffc4d769b1b91c82c2efe4d.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/tables/8dd92333f6b2f693ada5f79d5b48c3e4519c3c034ffc4d769b1b91c82c2efe4d.jpg)

![9b63da2f73985172e8132349bd8b603e8ceac653270ef37aa1e45123ed4b3c06.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/tables/9b63da2f73985172e8132349bd8b603e8ceac653270ef37aa1e45123ed4b3c06.jpg)

![bdf4ec3a7a684b439592cb6556df58c567621a969b73d014f7356affa88998e0.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/tables/bdf4ec3a7a684b439592cb6556df58c567621a969b73d014f7356affa88998e0.jpg)

![db5680fae13bce4fc801ed70708d053e1ad5d088e1e7f4216b5f2bce6a2b494c.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/tables/db5680fae13bce4fc801ed70708d053e1ad5d088e1e7f4216b5f2bce6a2b494c.jpg)

![e30f78eb02b1174200a100135825c40b75041ddb9e78fefa4d6d1c765969c35e.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/tables/e30f78eb02b1174200a100135825c40b75041ddb9e78fefa4d6d1c765969c35e.jpg)

![ff33672e9285e9b6033ea50a387b54b672d6dc5c053a6127cb93e00265239ed2.jpg](../icml_results/4023_Fisher-Preserving%20Guidance_%20Training-Free%20Manifold%20Constraints%20for%20Safe%20Diffusion%20Control/tables/ff33672e9285e9b6033ea50a387b54b672d6dc5c053a6127cb93e00265239ed2.jpg)

<a id="paper-3983"></a>

## iGRPO: Fast Online RL for Flow Matching Model with Instant Reward


### Images

![1e48685c1d6ed4d0546f32064ac4eb2338a478982e65465cf1c0fb67aef1652c.jpg](../icml_results/4024_iGRPO_%20Fast%20Online%20RL%20for%20Flow%20Matching%20Model%20with%20Instant%20Reward/images/1e48685c1d6ed4d0546f32064ac4eb2338a478982e65465cf1c0fb67aef1652c.jpg)

![4fd3db684c073cb6d293e0be224abd861aeb02b01df8d71c3beb7905fff86295.jpg](../icml_results/4024_iGRPO_%20Fast%20Online%20RL%20for%20Flow%20Matching%20Model%20with%20Instant%20Reward/images/4fd3db684c073cb6d293e0be224abd861aeb02b01df8d71c3beb7905fff86295.jpg)

![5e605489e11570da7d2c3580dd9f89027236c7eac501b806296feec3570bc329.jpg](../icml_results/4024_iGRPO_%20Fast%20Online%20RL%20for%20Flow%20Matching%20Model%20with%20Instant%20Reward/images/5e605489e11570da7d2c3580dd9f89027236c7eac501b806296feec3570bc329.jpg)

![bd8f12039ca64edc407b3fe89a2f94607e9fb7e19cf1912af3c09d2340fc51e9.jpg](../icml_results/4024_iGRPO_%20Fast%20Online%20RL%20for%20Flow%20Matching%20Model%20with%20Instant%20Reward/images/bd8f12039ca64edc407b3fe89a2f94607e9fb7e19cf1912af3c09d2340fc51e9.jpg)

![eb00a7b3d32339f2469ce1a72921065537571fdf9d758ca3dc7d1019d6a47e53.jpg](../icml_results/4024_iGRPO_%20Fast%20Online%20RL%20for%20Flow%20Matching%20Model%20with%20Instant%20Reward/images/eb00a7b3d32339f2469ce1a72921065537571fdf9d758ca3dc7d1019d6a47e53.jpg)

![f3ddb223faf26f53384e17620ff14d6b1f805346ec417c169679bf79ab25ee88.jpg](../icml_results/4024_iGRPO_%20Fast%20Online%20RL%20for%20Flow%20Matching%20Model%20with%20Instant%20Reward/images/f3ddb223faf26f53384e17620ff14d6b1f805346ec417c169679bf79ab25ee88.jpg)

### Tables

![170fdf49a88b7113532833d54507510e902718b7cada2c8af030644cef2a7e38.jpg](../icml_results/4024_iGRPO_%20Fast%20Online%20RL%20for%20Flow%20Matching%20Model%20with%20Instant%20Reward/tables/170fdf49a88b7113532833d54507510e902718b7cada2c8af030644cef2a7e38.jpg)

![54ac4e3d0f397d179dbbee7637020aa32defee0fa30d8bb4b04b9c93926ff233.jpg](../icml_results/4024_iGRPO_%20Fast%20Online%20RL%20for%20Flow%20Matching%20Model%20with%20Instant%20Reward/tables/54ac4e3d0f397d179dbbee7637020aa32defee0fa30d8bb4b04b9c93926ff233.jpg)

![a41fa0c5733cd4b2781a918ee8cf805087f2f3979c2f0d2a6ee1ea14393ef2fe.jpg](../icml_results/4024_iGRPO_%20Fast%20Online%20RL%20for%20Flow%20Matching%20Model%20with%20Instant%20Reward/tables/a41fa0c5733cd4b2781a918ee8cf805087f2f3979c2f0d2a6ee1ea14393ef2fe.jpg)

![cefa6f95bb5382dbadd7fb0c83b98e94bfa8ca18a67e2ffb7642e3a18447a422.jpg](../icml_results/4024_iGRPO_%20Fast%20Online%20RL%20for%20Flow%20Matching%20Model%20with%20Instant%20Reward/tables/cefa6f95bb5382dbadd7fb0c83b98e94bfa8ca18a67e2ffb7642e3a18447a422.jpg)

![f5e821f9133a6b5533bca4e83df2dce60a9c9cde2d0a18b3f97b9e681bacb981.jpg](../icml_results/4024_iGRPO_%20Fast%20Online%20RL%20for%20Flow%20Matching%20Model%20with%20Instant%20Reward/tables/f5e821f9133a6b5533bca4e83df2dce60a9c9cde2d0a18b3f97b9e681bacb981.jpg)

<a id="paper-3984"></a>

## Unbiased Principles, Robust Rewards


### Images

![05c3aa21537a18bbaac7f0278cfcd0b5dd50d291c6885c8bdcfc8953bbc66661.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/images/05c3aa21537a18bbaac7f0278cfcd0b5dd50d291c6885c8bdcfc8953bbc66661.jpg)

![5310f9b33e7ad1528eea622d057a3ee712110b1362afeb558be4bbce29d2033d.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/images/5310f9b33e7ad1528eea622d057a3ee712110b1362afeb558be4bbce29d2033d.jpg)

![bab5409e699525a88ee4fe35439cce1e86652dcd14d495ea2d6d516127b5da1c.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/images/bab5409e699525a88ee4fe35439cce1e86652dcd14d495ea2d6d516127b5da1c.jpg)

![c039094797e17d0147f02bd427e099042dac9bdf4673cb0412ceeaa274eebac5.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/images/c039094797e17d0147f02bd427e099042dac9bdf4673cb0412ceeaa274eebac5.jpg)

![cec01dc75b8559becf0efc2807cea9c89f16c052e9774a43db6ebb5a2677083d.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/images/cec01dc75b8559becf0efc2807cea9c89f16c052e9774a43db6ebb5a2677083d.jpg)

![e51d9b350a85b4e926775029c4f4582e03feadb760ba22fe6f905a3d6407e933.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/images/e51d9b350a85b4e926775029c4f4582e03feadb760ba22fe6f905a3d6407e933.jpg)

### Tables

![0268ede2ad81471b31cb22004e2237ed53b13f515214695b940974491aa43783.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/0268ede2ad81471b31cb22004e2237ed53b13f515214695b940974491aa43783.jpg)

![15f651a5bce7f63d60d73e5e1fd4ffb795d303480dc71d40858b7d5ba0f2d6bc.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/15f651a5bce7f63d60d73e5e1fd4ffb795d303480dc71d40858b7d5ba0f2d6bc.jpg)

![3989aa2aaea48280a7ff8578d5459d2a0fa970ddf8a5b387e96b596c0b0883c8.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/3989aa2aaea48280a7ff8578d5459d2a0fa970ddf8a5b387e96b596c0b0883c8.jpg)

![44edc4305ea25a5ed4dce3cabd952bcbcb767bc3195a102213f2841c3dc32455.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/44edc4305ea25a5ed4dce3cabd952bcbcb767bc3195a102213f2841c3dc32455.jpg)

![60df872d984bb12a7bc5653323b561956df4834b6c38fdd52ded9b99846bbbc5.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/60df872d984bb12a7bc5653323b561956df4834b6c38fdd52ded9b99846bbbc5.jpg)

![626d4bde8137d6f956e7be44fd658e94ebe68f795f59aee616435aebd48700d8.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/626d4bde8137d6f956e7be44fd658e94ebe68f795f59aee616435aebd48700d8.jpg)

![650e4f8af32706ac262d851fb10629d3aa0b8c5c1c42c0177333a7bb3725fe9d.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/650e4f8af32706ac262d851fb10629d3aa0b8c5c1c42c0177333a7bb3725fe9d.jpg)

![6859250a0eff67f236a8d38005477b15a8724fa37e5cd71c16d7891c79ccddf9.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/6859250a0eff67f236a8d38005477b15a8724fa37e5cd71c16d7891c79ccddf9.jpg)

![6884d37f656d5355ccf4eb96369b0a778d2b8dd7047b460d814403cf08405ce2.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/6884d37f656d5355ccf4eb96369b0a778d2b8dd7047b460d814403cf08405ce2.jpg)

![75cda75e57c616d2c63e7ca11c748af53b0753c1bcc29f227a9a6384c339e850.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/75cda75e57c616d2c63e7ca11c748af53b0753c1bcc29f227a9a6384c339e850.jpg)

![83a08bc07a30918270aad02b77b5409c7b32073cdad85a1ee09fe9c1044dd992.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/83a08bc07a30918270aad02b77b5409c7b32073cdad85a1ee09fe9c1044dd992.jpg)

![a87951de21e9db1e51ba2737c5f9d28ae42d4be29fcb1b8680fc75f55212c0ec.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/a87951de21e9db1e51ba2737c5f9d28ae42d4be29fcb1b8680fc75f55212c0ec.jpg)

![a8f2248936b489d0593467101d5ef1adccd5c7c286c669714cdde47d63600f61.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/a8f2248936b489d0593467101d5ef1adccd5c7c286c669714cdde47d63600f61.jpg)

![c741ef510179fa9d947a8e6eccee8ba57fa2405611fccac97dcb32c1f6701aee.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/c741ef510179fa9d947a8e6eccee8ba57fa2405611fccac97dcb32c1f6701aee.jpg)

![d8381a5090740946192d873b382fc7aa46dc9754da4b0f3d5a942ac39c8a7b2c.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/d8381a5090740946192d873b382fc7aa46dc9754da4b0f3d5a942ac39c8a7b2c.jpg)

![de9f26bda2f03bd9d78662dd927ca7871bd7a9e97573708f2fe38fb711849ce4.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/de9f26bda2f03bd9d78662dd927ca7871bd7a9e97573708f2fe38fb711849ce4.jpg)

![e0c0a88c82bd27eab4fa5ad4f33fd307031ffeb5430ce4b4f4d1e2a73a790ad6.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/e0c0a88c82bd27eab4fa5ad4f33fd307031ffeb5430ce4b4f4d1e2a73a790ad6.jpg)

![e4bc3eb4ff78784b0ec5f60e728bcc8e773f36a138f34d7ee5bfec7cc10b163f.jpg](../icml_results/4025_Unbiased%20Principles%2C%20Robust%20Rewards/tables/e4bc3eb4ff78784b0ec5f60e728bcc8e773f36a138f34d7ee5bfec7cc10b163f.jpg)

<a id="paper-3985"></a>

## Towards Reliable Marking and Verification of AI-Generated Text via Geometry-aware Sentence-level Watermarking


### Images

![4681852fad0e332efca5f7014bec9344a1271f894be82ad2708bd66e9d24bf66.jpg](../icml_results/4026_Towards%20Reliable%20Marking%20and%20Verification%20of%20AI-Generated%20Text%20via%20Geometry-aware%20Sentence-level%20Wat/images/4681852fad0e332efca5f7014bec9344a1271f894be82ad2708bd66e9d24bf66.jpg)

![788fbcdd613c9e81512085faa7997623a1791b47817117bc45a4332b8a2eb9b7.jpg](../icml_results/4026_Towards%20Reliable%20Marking%20and%20Verification%20of%20AI-Generated%20Text%20via%20Geometry-aware%20Sentence-level%20Wat/images/788fbcdd613c9e81512085faa7997623a1791b47817117bc45a4332b8a2eb9b7.jpg)

![82a272ce9610a20a37b479dfca0301c4c1df8e07f220bd4cbca79241c0436452.jpg](../icml_results/4026_Towards%20Reliable%20Marking%20and%20Verification%20of%20AI-Generated%20Text%20via%20Geometry-aware%20Sentence-level%20Wat/images/82a272ce9610a20a37b479dfca0301c4c1df8e07f220bd4cbca79241c0436452.jpg)

![896d6dfb0ae50781d12011b2e6a0262b68031a697aa2eb76035b147f2bd3e6ff.jpg](../icml_results/4026_Towards%20Reliable%20Marking%20and%20Verification%20of%20AI-Generated%20Text%20via%20Geometry-aware%20Sentence-level%20Wat/images/896d6dfb0ae50781d12011b2e6a0262b68031a697aa2eb76035b147f2bd3e6ff.jpg)

### Tables

![062a0cf1115b1aa0f2c01e82b74c2ae223c8d96c73b1246cf9ed7b3c4dc37b5b.jpg](../icml_results/4026_Towards%20Reliable%20Marking%20and%20Verification%20of%20AI-Generated%20Text%20via%20Geometry-aware%20Sentence-level%20Wat/tables/062a0cf1115b1aa0f2c01e82b74c2ae223c8d96c73b1246cf9ed7b3c4dc37b5b.jpg)

![9361629798ea0828740cfa0234349bd2d0567785999a732237ab130fffe8fb49.jpg](../icml_results/4026_Towards%20Reliable%20Marking%20and%20Verification%20of%20AI-Generated%20Text%20via%20Geometry-aware%20Sentence-level%20Wat/tables/9361629798ea0828740cfa0234349bd2d0567785999a732237ab130fffe8fb49.jpg)

![ab7ade91f7a526fba42d0d0e825b89bdf40d0983ee92903540b820d6d9f3d9be.jpg](../icml_results/4026_Towards%20Reliable%20Marking%20and%20Verification%20of%20AI-Generated%20Text%20via%20Geometry-aware%20Sentence-level%20Wat/tables/ab7ade91f7a526fba42d0d0e825b89bdf40d0983ee92903540b820d6d9f3d9be.jpg)

![dd02b40c8f1666c5237279fc98f7101cff264ad68d5379fa2ddf76e1b3a06605.jpg](../icml_results/4026_Towards%20Reliable%20Marking%20and%20Verification%20of%20AI-Generated%20Text%20via%20Geometry-aware%20Sentence-level%20Wat/tables/dd02b40c8f1666c5237279fc98f7101cff264ad68d5379fa2ddf76e1b3a06605.jpg)

<a id="paper-3986"></a>

## Streaming Covariate Balancing via Discrepancy-Based Feature Coresets


### Images

![23f0aacfc8edcc5770483a4911f52645c2c42fd1190f501e9e71f0d3bd763c91.jpg](../icml_results/4027_Streaming%20Covariate%20Balancing%20via%20Discrepancy-Based%20Feature%20Coresets/images/23f0aacfc8edcc5770483a4911f52645c2c42fd1190f501e9e71f0d3bd763c91.jpg)

![4e2425ef8eaf69d6208d89205d996e136be08cf0403641027794ba1532c5068e.jpg](../icml_results/4027_Streaming%20Covariate%20Balancing%20via%20Discrepancy-Based%20Feature%20Coresets/images/4e2425ef8eaf69d6208d89205d996e136be08cf0403641027794ba1532c5068e.jpg)

![5df53adf624c44ef1752a2d9c52e4ebb4079261808e947e35943da28a3981f58.jpg](../icml_results/4027_Streaming%20Covariate%20Balancing%20via%20Discrepancy-Based%20Feature%20Coresets/images/5df53adf624c44ef1752a2d9c52e4ebb4079261808e947e35943da28a3981f58.jpg)

![77ca19b8429e9c6c62d4365023cdca15ac83539adfc1ff4bb46e10ab8f8cc6aa.jpg](../icml_results/4027_Streaming%20Covariate%20Balancing%20via%20Discrepancy-Based%20Feature%20Coresets/images/77ca19b8429e9c6c62d4365023cdca15ac83539adfc1ff4bb46e10ab8f8cc6aa.jpg)

![914b3914ab82bbe216bebc032bcea851a2a031198dfcbe3177d1bbf0c3d7de60.jpg](../icml_results/4027_Streaming%20Covariate%20Balancing%20via%20Discrepancy-Based%20Feature%20Coresets/images/914b3914ab82bbe216bebc032bcea851a2a031198dfcbe3177d1bbf0c3d7de60.jpg)

![a5ec00c682ddf1d36022a9e92969af6e9c669717719e8409cc1745d130eecf35.jpg](../icml_results/4027_Streaming%20Covariate%20Balancing%20via%20Discrepancy-Based%20Feature%20Coresets/images/a5ec00c682ddf1d36022a9e92969af6e9c669717719e8409cc1745d130eecf35.jpg)

![fe67d2ef0d0cff12fcaa627a6dfee1e0f99561144b47806d12f1146c798e2ab4.jpg](../icml_results/4027_Streaming%20Covariate%20Balancing%20via%20Discrepancy-Based%20Feature%20Coresets/images/fe67d2ef0d0cff12fcaa627a6dfee1e0f99561144b47806d12f1146c798e2ab4.jpg)

### Tables

![10663b06b425bac46318df71bdfe7af69f4137af492463992427db0af2ebc5a9.jpg](../icml_results/4027_Streaming%20Covariate%20Balancing%20via%20Discrepancy-Based%20Feature%20Coresets/tables/10663b06b425bac46318df71bdfe7af69f4137af492463992427db0af2ebc5a9.jpg)

![12856099886660a94fa300eeefd32d198530dc17ed0f2f28274bfcae077e2acf.jpg](../icml_results/4027_Streaming%20Covariate%20Balancing%20via%20Discrepancy-Based%20Feature%20Coresets/tables/12856099886660a94fa300eeefd32d198530dc17ed0f2f28274bfcae077e2acf.jpg)

![63ab2e55089636b7225edf10139cb6a7e37fdb8765b28ffe68e275f37d1d6e6b.jpg](../icml_results/4027_Streaming%20Covariate%20Balancing%20via%20Discrepancy-Based%20Feature%20Coresets/tables/63ab2e55089636b7225edf10139cb6a7e37fdb8765b28ffe68e275f37d1d6e6b.jpg)

![98a0e6d63958f10bad43f654173229d3cee4bae3fdf57122c954c1eb9f0cf2c1.jpg](../icml_results/4027_Streaming%20Covariate%20Balancing%20via%20Discrepancy-Based%20Feature%20Coresets/tables/98a0e6d63958f10bad43f654173229d3cee4bae3fdf57122c954c1eb9f0cf2c1.jpg)

![ca7efe9ded8b2b291c1f53893448d09873b730e0b20cb19153d6dad20692a525.jpg](../icml_results/4027_Streaming%20Covariate%20Balancing%20via%20Discrepancy-Based%20Feature%20Coresets/tables/ca7efe9ded8b2b291c1f53893448d09873b730e0b20cb19153d6dad20692a525.jpg)

<a id="paper-3987"></a>

## Evaluating and Rewarding LALMs for Expressive Role-Play TTS via Mean Continuation Log-Probability


### Images

![13746476b00f76b19f44c90e884421719895c2595044abfd07234453d7e101e4.jpg](../icml_results/4028_Evaluating%20and%20Rewarding%20LALMs%20for%20Expressive%20Role-Play%20TTS%20via%20Mean%20Continuation%20Log-Probability/images/13746476b00f76b19f44c90e884421719895c2595044abfd07234453d7e101e4.jpg)

![1c3cf9e42bc038a806d3618b90a7acc7b3c32bccc8ef3995986a41b32229d868.jpg](../icml_results/4028_Evaluating%20and%20Rewarding%20LALMs%20for%20Expressive%20Role-Play%20TTS%20via%20Mean%20Continuation%20Log-Probability/images/1c3cf9e42bc038a806d3618b90a7acc7b3c32bccc8ef3995986a41b32229d868.jpg)

![4e54515188a232cf6c8de95cc719e67f63c272a01d715411aa7cc94a5427e0a1.jpg](../icml_results/4028_Evaluating%20and%20Rewarding%20LALMs%20for%20Expressive%20Role-Play%20TTS%20via%20Mean%20Continuation%20Log-Probability/images/4e54515188a232cf6c8de95cc719e67f63c272a01d715411aa7cc94a5427e0a1.jpg)

![7f1c7f97e850c631aa75558cbb6f376bbbbe00b8cb854cf1a2ca8b823347334b.jpg](../icml_results/4028_Evaluating%20and%20Rewarding%20LALMs%20for%20Expressive%20Role-Play%20TTS%20via%20Mean%20Continuation%20Log-Probability/images/7f1c7f97e850c631aa75558cbb6f376bbbbe00b8cb854cf1a2ca8b823347334b.jpg)

![b82fdb502834af9ada7feaec9d8e9081764322b95d94f020b8a53661d305baa2.jpg](../icml_results/4028_Evaluating%20and%20Rewarding%20LALMs%20for%20Expressive%20Role-Play%20TTS%20via%20Mean%20Continuation%20Log-Probability/images/b82fdb502834af9ada7feaec9d8e9081764322b95d94f020b8a53661d305baa2.jpg)

### Tables

![30a3d6ed9007cf75a3575acee643acc478a168e628ce69c6cf8ff56ed75c4f99.jpg](../icml_results/4028_Evaluating%20and%20Rewarding%20LALMs%20for%20Expressive%20Role-Play%20TTS%20via%20Mean%20Continuation%20Log-Probability/tables/30a3d6ed9007cf75a3575acee643acc478a168e628ce69c6cf8ff56ed75c4f99.jpg)

![3494a2af3ddf2d917db393de30f54a56e55d0c430b621a6d7a08cb9d702ce6fd.jpg](../icml_results/4028_Evaluating%20and%20Rewarding%20LALMs%20for%20Expressive%20Role-Play%20TTS%20via%20Mean%20Continuation%20Log-Probability/tables/3494a2af3ddf2d917db393de30f54a56e55d0c430b621a6d7a08cb9d702ce6fd.jpg)

![4106a7252cbe658c6da2aac82c88bae445ae4e55c9a62dfb2cd00c9c20cb8d06.jpg](../icml_results/4028_Evaluating%20and%20Rewarding%20LALMs%20for%20Expressive%20Role-Play%20TTS%20via%20Mean%20Continuation%20Log-Probability/tables/4106a7252cbe658c6da2aac82c88bae445ae4e55c9a62dfb2cd00c9c20cb8d06.jpg)

![487e9395e418b1e526e6312a1e7e234e6566e1ddf684012fe7de119fda030fea.jpg](../icml_results/4028_Evaluating%20and%20Rewarding%20LALMs%20for%20Expressive%20Role-Play%20TTS%20via%20Mean%20Continuation%20Log-Probability/tables/487e9395e418b1e526e6312a1e7e234e6566e1ddf684012fe7de119fda030fea.jpg)

![63f06155bdddca6d334eea1c25004cf70b1e6dd04c8e75526abc1484935bc11b.jpg](../icml_results/4028_Evaluating%20and%20Rewarding%20LALMs%20for%20Expressive%20Role-Play%20TTS%20via%20Mean%20Continuation%20Log-Probability/tables/63f06155bdddca6d334eea1c25004cf70b1e6dd04c8e75526abc1484935bc11b.jpg)

![8dce459b88c8de5721502f56045ce037aca07c6e75ddc34e429dca23c6eee58b.jpg](../icml_results/4028_Evaluating%20and%20Rewarding%20LALMs%20for%20Expressive%20Role-Play%20TTS%20via%20Mean%20Continuation%20Log-Probability/tables/8dce459b88c8de5721502f56045ce037aca07c6e75ddc34e429dca23c6eee58b.jpg)

![a12bec4b487367ffd602dff7a0f8fdff2742f5dc3b98c410b09fa773aa58c61e.jpg](../icml_results/4028_Evaluating%20and%20Rewarding%20LALMs%20for%20Expressive%20Role-Play%20TTS%20via%20Mean%20Continuation%20Log-Probability/tables/a12bec4b487367ffd602dff7a0f8fdff2742f5dc3b98c410b09fa773aa58c61e.jpg)

![dc2a67235f3bb79c66e075255e10c3cb48ba0ce9ebd7d104a3034afd37c58ec9.jpg](../icml_results/4028_Evaluating%20and%20Rewarding%20LALMs%20for%20Expressive%20Role-Play%20TTS%20via%20Mean%20Continuation%20Log-Probability/tables/dc2a67235f3bb79c66e075255e10c3cb48ba0ce9ebd7d104a3034afd37c58ec9.jpg)

![ec607217b9a04c34fcb151437714fab272382d79c4b8746d2e3e6883c2ce1ef0.jpg](../icml_results/4028_Evaluating%20and%20Rewarding%20LALMs%20for%20Expressive%20Role-Play%20TTS%20via%20Mean%20Continuation%20Log-Probability/tables/ec607217b9a04c34fcb151437714fab272382d79c4b8746d2e3e6883c2ce1ef0.jpg)

<a id="paper-3988"></a>

## Low-dimensional topology of deep neural networks


### Images

![1225f2b0c59fdb7e21d27b209958ea7fcfd2645f093fa7ed22d4bbbf380e7ce7.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/images/1225f2b0c59fdb7e21d27b209958ea7fcfd2645f093fa7ed22d4bbbf380e7ce7.jpg)

![29583eafdd8aeb1dc3188c384d0067f113e98383f2cac8d84273078088533479.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/images/29583eafdd8aeb1dc3188c384d0067f113e98383f2cac8d84273078088533479.jpg)

![2b0a3cb86363d08059388afe8560292f189be0285b0c55d0f5998fdba2e632b8.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/images/2b0a3cb86363d08059388afe8560292f189be0285b0c55d0f5998fdba2e632b8.jpg)

![3aea003daa07babe41ec6b693d7e16903709d800b0237d0f33566aed96ac118f.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/images/3aea003daa07babe41ec6b693d7e16903709d800b0237d0f33566aed96ac118f.jpg)

![4ba6971fc1a5e20126e6c88cdd650f6b0d4115a092639e655e6ade9204095e38.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/images/4ba6971fc1a5e20126e6c88cdd650f6b0d4115a092639e655e6ade9204095e38.jpg)

![596a831d2ed099bb64629b953af5c6598c5ec30769cdecfab92ea06d759c1d88.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/images/596a831d2ed099bb64629b953af5c6598c5ec30769cdecfab92ea06d759c1d88.jpg)

![5c134f64955f9ee7ac48ded9c300f0eb50d1020706742a7fef69b493ffb9c5b5.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/images/5c134f64955f9ee7ac48ded9c300f0eb50d1020706742a7fef69b493ffb9c5b5.jpg)

![670a9e83d3cf38165b5ce03e7e0dd2116f6d2ca8ac0e24d782c99cb9a855a4de.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/images/670a9e83d3cf38165b5ce03e7e0dd2116f6d2ca8ac0e24d782c99cb9a855a4de.jpg)

![7cf01f2130b4ab373738ac764c887364b423e5be1352e32ba5bccd1621402d43.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/images/7cf01f2130b4ab373738ac764c887364b423e5be1352e32ba5bccd1621402d43.jpg)

![b4dcb7989f135b469ba4be354439104a41ce4ddec5b6d6323319fc78d08d4120.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/images/b4dcb7989f135b469ba4be354439104a41ce4ddec5b6d6323319fc78d08d4120.jpg)

![c7b095c7bdda7f89f5b2c0d469bf7a93a696583f9883f4fb042eb1153c58285b.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/images/c7b095c7bdda7f89f5b2c0d469bf7a93a696583f9883f4fb042eb1153c58285b.jpg)

### Tables

![038424e5787c966a72ba92164b5f806559d7487c30f7fc4024b6662433cd2ab4.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/tables/038424e5787c966a72ba92164b5f806559d7487c30f7fc4024b6662433cd2ab4.jpg)

![278460333d6625f09ef59653d1d75aea3c70e2e67e0a97ae966b90aa0a649a50.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/tables/278460333d6625f09ef59653d1d75aea3c70e2e67e0a97ae966b90aa0a649a50.jpg)

![32af7f108717ee9944e08900ecf7ef9cf2b2aa75b495d6ca9d444e48808ef6e2.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/tables/32af7f108717ee9944e08900ecf7ef9cf2b2aa75b495d6ca9d444e48808ef6e2.jpg)

![44a33426398dc8faa8705388614b5964846b56ba5d0987cdc6c5dacc49130b96.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/tables/44a33426398dc8faa8705388614b5964846b56ba5d0987cdc6c5dacc49130b96.jpg)

![4b9dd94e9e60ac29c284b0b5ffa6b648fd8e3d4fdc222e4e929d360c20a00428.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/tables/4b9dd94e9e60ac29c284b0b5ffa6b648fd8e3d4fdc222e4e929d360c20a00428.jpg)

![8e832deebdaa2b596f5b23fe5c0936027dc805871803e025f8dfe3a00b95fd61.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/tables/8e832deebdaa2b596f5b23fe5c0936027dc805871803e025f8dfe3a00b95fd61.jpg)

![c588cf6eccbb60d742d3f8720d275e0a90c73386679b8e01f90e4d654f45a5cd.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/tables/c588cf6eccbb60d742d3f8720d275e0a90c73386679b8e01f90e4d654f45a5cd.jpg)

![d081a706f8c8025b711944e37aad3230bb05a9bace34d3a3c6ecc323f41eaccb.jpg](../icml_results/4029_Low-dimensional%20topology%20of%20deep%20neural%20networks/tables/d081a706f8c8025b711944e37aad3230bb05a9bace34d3a3c6ecc323f41eaccb.jpg)

<a id="paper-3989"></a>

## Powerful and Theoretically Guaranteed Independence Testing on Heterogeneous Federated Clients


### Images

![10640962b5873a5ca3481e1bfa05947c42cfd3d1f5099d95e9201b2a724bc80f.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/images/10640962b5873a5ca3481e1bfa05947c42cfd3d1f5099d95e9201b2a724bc80f.jpg)

![40138fba96e030f1c2b4dea2e22248597852de84a96ea19f3d6b45c2d4e8db85.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/images/40138fba96e030f1c2b4dea2e22248597852de84a96ea19f3d6b45c2d4e8db85.jpg)

![5895bfe3510c860c97156f17613264fb3b72e77c6094a8ea8cfd1ccce1c7afc2.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/images/5895bfe3510c860c97156f17613264fb3b72e77c6094a8ea8cfd1ccce1c7afc2.jpg)

![6cc3b30d7ec03e8d603cc25729bddbef319e4e94e4205cc0f2a172f5fb43eab6.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/images/6cc3b30d7ec03e8d603cc25729bddbef319e4e94e4205cc0f2a172f5fb43eab6.jpg)

![7ec013e8b1fdbab9df500ed5c321a035b64cd3421cc9b8aacfd72e746803c30b.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/images/7ec013e8b1fdbab9df500ed5c321a035b64cd3421cc9b8aacfd72e746803c30b.jpg)

![89699afd537f4fd6be1e9ada6d2664331fb480552fba4ccf79a153a2c1161c28.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/images/89699afd537f4fd6be1e9ada6d2664331fb480552fba4ccf79a153a2c1161c28.jpg)

![937a62d43c5695e135d7f1980baa668daaed19d0f12af66834c10bd126c0e18f.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/images/937a62d43c5695e135d7f1980baa668daaed19d0f12af66834c10bd126c0e18f.jpg)

![98cb9e9cb632a863fdcb59d2155f240f972621414bfe95ae9f482fdcd72da4bf.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/images/98cb9e9cb632a863fdcb59d2155f240f972621414bfe95ae9f482fdcd72da4bf.jpg)

![e2fc5da180b4c77748087d2c035947a72d5672174183d2038f530826aeea5d76.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/images/e2fc5da180b4c77748087d2c035947a72d5672174183d2038f530826aeea5d76.jpg)

### Tables

![06311f15ab33a9adecae06df718808a22f86581b25ab1806965d97c0d662d98d.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/tables/06311f15ab33a9adecae06df718808a22f86581b25ab1806965d97c0d662d98d.jpg)

![114ecdd80f88d52b10d4cddf94c1b68f3e2497d7ae3b8cfb3e652b3bdeb023f0.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/tables/114ecdd80f88d52b10d4cddf94c1b68f3e2497d7ae3b8cfb3e652b3bdeb023f0.jpg)

![196c30fdf3a8f2b26b41715d6219d0c4c9bab689aee44d8a091b13b151a94862.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/tables/196c30fdf3a8f2b26b41715d6219d0c4c9bab689aee44d8a091b13b151a94862.jpg)

![28a55992c9cfa124e83f2ea91f53fd4fead369841863b27f2bbf4a5f4debd562.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/tables/28a55992c9cfa124e83f2ea91f53fd4fead369841863b27f2bbf4a5f4debd562.jpg)

![31f248c58510f79749665ae8b117df0ef9ab78e4d0af05f0559bfb03d6f4f88c.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/tables/31f248c58510f79749665ae8b117df0ef9ab78e4d0af05f0559bfb03d6f4f88c.jpg)

![36b403bb3f152682e085e7cfdd1e617964da622ecb4f293ea6688a97bab5dd14.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/tables/36b403bb3f152682e085e7cfdd1e617964da622ecb4f293ea6688a97bab5dd14.jpg)

![38ba3767d3a657106d172ed6543844a6b82347c7a410631030cf15b74b6defe6.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/tables/38ba3767d3a657106d172ed6543844a6b82347c7a410631030cf15b74b6defe6.jpg)

![d85f666235c67844da311bc3d43951accd1a43dbf0a37b560b3bfa876ea0870f.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/tables/d85f666235c67844da311bc3d43951accd1a43dbf0a37b560b3bfa876ea0870f.jpg)

![ff831568c69b3dd925f7be97541877a793e7f6a5ae17d4f2e6819ec6b54e0958.jpg](../icml_results/4030_Powerful%20and%20Theoretically%20Guaranteed%20Independence%20Testing%20on%20Heterogeneous%20Federated%20Clients/tables/ff831568c69b3dd925f7be97541877a793e7f6a5ae17d4f2e6819ec6b54e0958.jpg)

<a id="paper-3990"></a>

## MIND: Decoupling Model-Induced Label Noise via Latent Manifold Disentanglement


### Images

![0ac25ae5c1a49cb530311b6e7bc666af6d9369a78ee85ddc681569aff2e9c550.jpg](../icml_results/4031_MIND_%20Decoupling%20Model-Induced%20Label%20Noise%20via%20Latent%20Manifold%20Disentanglement/images/0ac25ae5c1a49cb530311b6e7bc666af6d9369a78ee85ddc681569aff2e9c550.jpg)

![12c4b1121a6069ae56358d12ddb42dab9e753e479b9a7cf480bcdabd04448e60.jpg](../icml_results/4031_MIND_%20Decoupling%20Model-Induced%20Label%20Noise%20via%20Latent%20Manifold%20Disentanglement/images/12c4b1121a6069ae56358d12ddb42dab9e753e479b9a7cf480bcdabd04448e60.jpg)

![83f09e2e0ff636c83435cc89f483a34b33b998e1b67090de0b83557d900abacb.jpg](../icml_results/4031_MIND_%20Decoupling%20Model-Induced%20Label%20Noise%20via%20Latent%20Manifold%20Disentanglement/images/83f09e2e0ff636c83435cc89f483a34b33b998e1b67090de0b83557d900abacb.jpg)

![86bad80c711cd71307f92481735eb20b8b1ec62d0fb105957d3de1040f80b8ed.jpg](../icml_results/4031_MIND_%20Decoupling%20Model-Induced%20Label%20Noise%20via%20Latent%20Manifold%20Disentanglement/images/86bad80c711cd71307f92481735eb20b8b1ec62d0fb105957d3de1040f80b8ed.jpg)

![8f6af1a19bf8cda1030bf2e5d11f096c173241c6cfed9949c1aab467c43304fc.jpg](../icml_results/4031_MIND_%20Decoupling%20Model-Induced%20Label%20Noise%20via%20Latent%20Manifold%20Disentanglement/images/8f6af1a19bf8cda1030bf2e5d11f096c173241c6cfed9949c1aab467c43304fc.jpg)

![decea4f6f1d00b79b5305c1b324f9b88c1191715774c4e40b5ec75ae8c07bcf9.jpg](../icml_results/4031_MIND_%20Decoupling%20Model-Induced%20Label%20Noise%20via%20Latent%20Manifold%20Disentanglement/images/decea4f6f1d00b79b5305c1b324f9b88c1191715774c4e40b5ec75ae8c07bcf9.jpg)

![ff433ca7777ecc4ae2a1076b0740fe069315c5d1a7272a3f48471169836dd0cd.jpg](../icml_results/4031_MIND_%20Decoupling%20Model-Induced%20Label%20Noise%20via%20Latent%20Manifold%20Disentanglement/images/ff433ca7777ecc4ae2a1076b0740fe069315c5d1a7272a3f48471169836dd0cd.jpg)

### Tables

![0aae5a007d9f5b4708d0d85c0f2f745fa2dfb50d29e4f9460e28996705ed3a3a.jpg](../icml_results/4031_MIND_%20Decoupling%20Model-Induced%20Label%20Noise%20via%20Latent%20Manifold%20Disentanglement/tables/0aae5a007d9f5b4708d0d85c0f2f745fa2dfb50d29e4f9460e28996705ed3a3a.jpg)

![13fee6f822ac41ff3fd16586ed35efe6e016257cb0b14eac18cf5ded2de2bab3.jpg](../icml_results/4031_MIND_%20Decoupling%20Model-Induced%20Label%20Noise%20via%20Latent%20Manifold%20Disentanglement/tables/13fee6f822ac41ff3fd16586ed35efe6e016257cb0b14eac18cf5ded2de2bab3.jpg)

![51b7fd050b58feb08484c01c33c846c879a8b7ca71c3edef381e6d2c62a09109.jpg](../icml_results/4031_MIND_%20Decoupling%20Model-Induced%20Label%20Noise%20via%20Latent%20Manifold%20Disentanglement/tables/51b7fd050b58feb08484c01c33c846c879a8b7ca71c3edef381e6d2c62a09109.jpg)

![6b762d560e45664123791c466a5582fbbefa4f9aa2b28cdb5e21392f41eb1b9c.jpg](../icml_results/4031_MIND_%20Decoupling%20Model-Induced%20Label%20Noise%20via%20Latent%20Manifold%20Disentanglement/tables/6b762d560e45664123791c466a5582fbbefa4f9aa2b28cdb5e21392f41eb1b9c.jpg)

![b38170f2b17b56836390d5de072cba6e0af6fe028545e65e9f2cf0b538ff77ca.jpg](../icml_results/4031_MIND_%20Decoupling%20Model-Induced%20Label%20Noise%20via%20Latent%20Manifold%20Disentanglement/tables/b38170f2b17b56836390d5de072cba6e0af6fe028545e65e9f2cf0b538ff77ca.jpg)

<a id="paper-3991"></a>

## Transfer Learning in Nonparametric Regression with Deep ReLU Networks


### Images

![2ba07e4779014959a6563a5f4abfada9cbb4547ff815a56131bf8aedc5fce704.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/images/2ba07e4779014959a6563a5f4abfada9cbb4547ff815a56131bf8aedc5fce704.jpg)

![9f2aee71fc19548ffb427c37e76975b4845d5ec8f079f8ac25d7a0fa2d79e19a.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/images/9f2aee71fc19548ffb427c37e76975b4845d5ec8f079f8ac25d7a0fa2d79e19a.jpg)

![ac28837619b6c76b50b8fd80e71a629a3b6d6fe416850c07808a8653052de152.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/images/ac28837619b6c76b50b8fd80e71a629a3b6d6fe416850c07808a8653052de152.jpg)

### Tables

![07ad3310000739cb90fa7821316dffbf4126214c3e608c42f5e83832629d462b.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/07ad3310000739cb90fa7821316dffbf4126214c3e608c42f5e83832629d462b.jpg)

![0ab22c3ecdfdb0feee03eb15f2214974074c1cb7829710ed9289f5ba156c2177.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/0ab22c3ecdfdb0feee03eb15f2214974074c1cb7829710ed9289f5ba156c2177.jpg)

![0fb1e3e3a26caa2ef7aa2ba263e0ccc4a9a899c32deae978b73d7e22bff06d45.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/0fb1e3e3a26caa2ef7aa2ba263e0ccc4a9a899c32deae978b73d7e22bff06d45.jpg)

![10ef5cbf319705b99fa1eeac92ea75f30a7107ef6f3c4bbef97698bbc9f4fe82.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/10ef5cbf319705b99fa1eeac92ea75f30a7107ef6f3c4bbef97698bbc9f4fe82.jpg)

![1b46f93be031c8c96adc1a5bc7dcd05e7f7817ca8aa8242d06f874cde52925db.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/1b46f93be031c8c96adc1a5bc7dcd05e7f7817ca8aa8242d06f874cde52925db.jpg)

![31b8fa21f2e8bbafc69691ad00424fd491e86969dbd095f4c51db915e41d17ef.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/31b8fa21f2e8bbafc69691ad00424fd491e86969dbd095f4c51db915e41d17ef.jpg)

![31cce9b7eccb6425b77e5b507bdf23379a0c7d16bac01555de699ee8dcb0139d.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/31cce9b7eccb6425b77e5b507bdf23379a0c7d16bac01555de699ee8dcb0139d.jpg)

![3bd4170ad17128a4a3c749f5f035ea872e5d4a928c515303f4cc3c8afc18fadc.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/3bd4170ad17128a4a3c749f5f035ea872e5d4a928c515303f4cc3c8afc18fadc.jpg)

![3e347f063b9eaff04109711fb2ed4d813a24a13e5b7582c99c0837ac911f98c7.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/3e347f063b9eaff04109711fb2ed4d813a24a13e5b7582c99c0837ac911f98c7.jpg)

![3f6729c41f8a30eb0c868b6fdf91eb7c3628b5591dacd8a98dc2c26aa0ed480d.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/3f6729c41f8a30eb0c868b6fdf91eb7c3628b5591dacd8a98dc2c26aa0ed480d.jpg)

![54d632e6f76f4d13cff65c6e21300bcb3bb701e123af57b39ee8df8d1de32849.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/54d632e6f76f4d13cff65c6e21300bcb3bb701e123af57b39ee8df8d1de32849.jpg)

![5667136a5ac66f497fa3f812a4599ccbad843cd84e89db2700074d2cbc08c93e.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/5667136a5ac66f497fa3f812a4599ccbad843cd84e89db2700074d2cbc08c93e.jpg)

![6360e5752a23188b8cf594eda7bdfdc75e0ed68e2aa61785456099e96ef177a0.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/6360e5752a23188b8cf594eda7bdfdc75e0ed68e2aa61785456099e96ef177a0.jpg)

![66a86e476319fdba4ae3de7f99b7eec732b282274a8844d60afdf1c9c24a633f.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/66a86e476319fdba4ae3de7f99b7eec732b282274a8844d60afdf1c9c24a633f.jpg)

![6eb6a235e0ee112dfacceda2ca260f13707e29471bdfd0dc4c223a9c0ccd8ca2.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/6eb6a235e0ee112dfacceda2ca260f13707e29471bdfd0dc4c223a9c0ccd8ca2.jpg)

![7e0f12e1dd8518f3ba08dc7a9abaf2b66cbd6c08f8a71b6692f5243601816b73.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/7e0f12e1dd8518f3ba08dc7a9abaf2b66cbd6c08f8a71b6692f5243601816b73.jpg)

![84602b4c18cb857d1cade88cd0dd34f49a5be2c1993c39da54019c64194ee2ca.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/84602b4c18cb857d1cade88cd0dd34f49a5be2c1993c39da54019c64194ee2ca.jpg)

![8b1ecd49e659e1a3e7c2b220b33047800831d6755951077f65c9843ddc0025e4.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/8b1ecd49e659e1a3e7c2b220b33047800831d6755951077f65c9843ddc0025e4.jpg)

![9db80398dde8be627e7d411c4cd49b4024c926ee12c0f5a9ab54cd4ae0f6cb25.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/9db80398dde8be627e7d411c4cd49b4024c926ee12c0f5a9ab54cd4ae0f6cb25.jpg)

![b58bf77a41e1efb235db8d499bf009e2ef6719325989e5e45a6b8d8017e2d22d.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/b58bf77a41e1efb235db8d499bf009e2ef6719325989e5e45a6b8d8017e2d22d.jpg)

![bc2aa982c4b23aef937be2eb4f1187139bba887e3d46aff70cc2591e89f60821.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/bc2aa982c4b23aef937be2eb4f1187139bba887e3d46aff70cc2591e89f60821.jpg)

![c8ac962c3ef9bb2c38acd543ff70c0963ccedace0bb11ac56f7ae583f9126ce7.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/c8ac962c3ef9bb2c38acd543ff70c0963ccedace0bb11ac56f7ae583f9126ce7.jpg)

![edf911fdc7d099df1e8197a0bdfc22b77d8a59442927955fc7e7bdea8f917f15.jpg](../icml_results/4032_Transfer%20Learning%20in%20Nonparametric%20Regression%20with%20Deep%20ReLU%20Networks/tables/edf911fdc7d099df1e8197a0bdfc22b77d8a59442927955fc7e7bdea8f917f15.jpg)

<a id="paper-3992"></a>

## POLCA: Stochastic Generative Optimization with LLM


### Images

![0b24abdbd5e1c05ee12e5c01c85017cd4f61c00e5d44f96176fa09f2dd1f7700.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/0b24abdbd5e1c05ee12e5c01c85017cd4f61c00e5d44f96176fa09f2dd1f7700.jpg)

![0ca984b5b7c8aba80ab5002e78451f3edbe2ea66ba9e6a9ab56a22f2830f2b38.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/0ca984b5b7c8aba80ab5002e78451f3edbe2ea66ba9e6a9ab56a22f2830f2b38.jpg)

![12b687125649793e675b051b01315c145d13ffcc7895da754cd2d98d22959bb3.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/12b687125649793e675b051b01315c145d13ffcc7895da754cd2d98d22959bb3.jpg)

![192bc542b04085dcca99b05edab0acafa22731b3d15f4465eb8dc79aee88a9df.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/192bc542b04085dcca99b05edab0acafa22731b3d15f4465eb8dc79aee88a9df.jpg)

![1a36da936b500c096c7bc64dfb59f1cb9fb7c0847c59e2b8bbd3d6db5c3af461.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/1a36da936b500c096c7bc64dfb59f1cb9fb7c0847c59e2b8bbd3d6db5c3af461.jpg)

![2191ad73dbdf9103042e3d4bbd5262f2bd5f630060ad6a6d34ef7067946d9bd4.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/2191ad73dbdf9103042e3d4bbd5262f2bd5f630060ad6a6d34ef7067946d9bd4.jpg)

![3503af1adadfa94bba007110f56e4b64883c20bd813e32e1e014ec4ff6c56c9e.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/3503af1adadfa94bba007110f56e4b64883c20bd813e32e1e014ec4ff6c56c9e.jpg)

![3ca7f810dcdad608d196a48c88f903e8406abbab7a409d6100a4c463bf5ec695.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/3ca7f810dcdad608d196a48c88f903e8406abbab7a409d6100a4c463bf5ec695.jpg)

![5cef4fa70ee0c2488177655891b6857cc70b0b10c82735b3d6d5182f0efb0987.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/5cef4fa70ee0c2488177655891b6857cc70b0b10c82735b3d6d5182f0efb0987.jpg)

![7e3df3be2aaacc2ea88aeb8d8234e245a8c972f6a9dc3cd6ae0e76ef0576a77f.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/7e3df3be2aaacc2ea88aeb8d8234e245a8c972f6a9dc3cd6ae0e76ef0576a77f.jpg)

![7f85bb757b16dbbba15c9e0997f4c87be242e1ad5322809db49143b94427cb94.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/7f85bb757b16dbbba15c9e0997f4c87be242e1ad5322809db49143b94427cb94.jpg)

![827807303da26b52e92bfe085c02fc39be9e443b40d1ae7f0c4d54865e415a7c.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/827807303da26b52e92bfe085c02fc39be9e443b40d1ae7f0c4d54865e415a7c.jpg)

![82b865a0d8045d67f0280b075675af7d3fb0d4442cc91ae23d1fdce0ee008678.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/82b865a0d8045d67f0280b075675af7d3fb0d4442cc91ae23d1fdce0ee008678.jpg)

![850ec73c7f882eec977662440c8497c4df5558c31e4843e7996ff4f16c56593f.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/850ec73c7f882eec977662440c8497c4df5558c31e4843e7996ff4f16c56593f.jpg)

![87f18da4af0bad8df39efe904f7526e0aebe4b7b952bde87ce123b144cfa3226.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/87f18da4af0bad8df39efe904f7526e0aebe4b7b952bde87ce123b144cfa3226.jpg)

![9e96b1568d3c9bb331efe5a77ff91fcb82b59ca4b4e2abcd336869acc21e329f.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/9e96b1568d3c9bb331efe5a77ff91fcb82b59ca4b4e2abcd336869acc21e329f.jpg)

![cfeb2a8598694c89ebf98427f4edbbec79c77d8ea92641c9be8814455832d5d5.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/cfeb2a8598694c89ebf98427f4edbbec79c77d8ea92641c9be8814455832d5d5.jpg)

![eca6f354630986f3acb04918106a375c7d2cc04277a35027e8d09bbd0018514a.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/eca6f354630986f3acb04918106a375c7d2cc04277a35027e8d09bbd0018514a.jpg)

![f6ec7109cfed2a5279a5af9b07aa5a0526f1ad77f2ca2f213c8deaac8815618f.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/f6ec7109cfed2a5279a5af9b07aa5a0526f1ad77f2ca2f213c8deaac8815618f.jpg)

![fcc8f178fc53552503606d79de03f3f6d28577558a820c721b5dcdacc579d0d1.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/images/fcc8f178fc53552503606d79de03f3f6d28577558a820c721b5dcdacc579d0d1.jpg)

### Tables

![0013c022329c427cf30cbf071aa7a53a771935729d226e21601fddf10d5bb79f.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/tables/0013c022329c427cf30cbf071aa7a53a771935729d226e21601fddf10d5bb79f.jpg)

![4a95bf3c7706dd22a4c54d0f73d8bc651973b3bc6458cd606df3da697224edaf.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/tables/4a95bf3c7706dd22a4c54d0f73d8bc651973b3bc6458cd606df3da697224edaf.jpg)

![b95e1089dfbd3cd77c5f57fdabb0e5df42ab2b2ed67ffc99c22ca3ba5cf661fb.jpg](../icml_results/4033_POLCA_%20Stochastic%20Generative%20Optimization%20with%20LLM/tables/b95e1089dfbd3cd77c5f57fdabb0e5df42ab2b2ed67ffc99c22ca3ba5cf661fb.jpg)

<a id="paper-3993"></a>

## Riemannian Dueling Optimization

### Images

![0145d4d5a6b0591a63292830b12df9963827694174bdcc1f67f6608e2220eccc.jpg](../icml_results/4034_Riemannian%20Dueling%20Optimization/images/0145d4d5a6b0591a63292830b12df9963827694174bdcc1f67f6608e2220eccc.jpg)

![1046d955653d9d2d41688e9f4db3e8215500fb877763340567c6a8fcc182d9d2.jpg](../icml_results/4034_Riemannian%20Dueling%20Optimization/images/1046d955653d9d2d41688e9f4db3e8215500fb877763340567c6a8fcc182d9d2.jpg)

![1d91965f24a277df8c71fb8f1bb55503e7e8e9df852e49ffa1cf75b2be7e9ad8.jpg](../icml_results/4034_Riemannian%20Dueling%20Optimization/images/1d91965f24a277df8c71fb8f1bb55503e7e8e9df852e49ffa1cf75b2be7e9ad8.jpg)

![33185b15a3718669246dbfec5c335f3eeb3cfee55ad710fcebd240ef45465eb7.jpg](../icml_results/4034_Riemannian%20Dueling%20Optimization/images/33185b15a3718669246dbfec5c335f3eeb3cfee55ad710fcebd240ef45465eb7.jpg)

![3ed3482e1c779d75f20aaaa89d4677a359e76f726d952ee6df12f56adcb2aaed.jpg](../icml_results/4034_Riemannian%20Dueling%20Optimization/images/3ed3482e1c779d75f20aaaa89d4677a359e76f726d952ee6df12f56adcb2aaed.jpg)

![41629c848f14abfd56dc03d933f8008257b661dd40984ba09393744f169c72f8.jpg](../icml_results/4034_Riemannian%20Dueling%20Optimization/images/41629c848f14abfd56dc03d933f8008257b661dd40984ba09393744f169c72f8.jpg)

![50600b147e9c011990dbc8d80f1e0fd41928656f25e09e5a12cee18c94a2a638.jpg](../icml_results/4034_Riemannian%20Dueling%20Optimization/images/50600b147e9c011990dbc8d80f1e0fd41928656f25e09e5a12cee18c94a2a638.jpg)

![761685395ab66e08cf75cbb30af389270c246d76c76f575fefd735d45ca58c89.jpg](../icml_results/4034_Riemannian%20Dueling%20Optimization/images/761685395ab66e08cf75cbb30af389270c246d76c76f575fefd735d45ca58c89.jpg)

![87edade42bc0f3e1fc67bb9e0febcac3d283598a672b4b75aafaa34905fb7774.jpg](../icml_results/4034_Riemannian%20Dueling%20Optimization/images/87edade42bc0f3e1fc67bb9e0febcac3d283598a672b4b75aafaa34905fb7774.jpg)

![91590bf5cf11a5034c20428651cf606fa617c1807ae133eef1b45bd7c6732877.jpg](../icml_results/4034_Riemannian%20Dueling%20Optimization/images/91590bf5cf11a5034c20428651cf606fa617c1807ae133eef1b45bd7c6732877.jpg)

![a6a346fe7205b56a2fa7aae8fa01867b12fd1143d493e5eb186986d59944e4db.jpg](../icml_results/4034_Riemannian%20Dueling%20Optimization/images/a6a346fe7205b56a2fa7aae8fa01867b12fd1143d493e5eb186986d59944e4db.jpg)

![be8f5ecb7776f5ac7d29a8c0baf11cff42fe2053501ddc858e5b1661a9fe2169.jpg](../icml_results/4034_Riemannian%20Dueling%20Optimization/images/be8f5ecb7776f5ac7d29a8c0baf11cff42fe2053501ddc858e5b1661a9fe2169.jpg)

![bee8adf13ba6e5c58c011902399a69a05bbd06342dd728a4768e9eac37a1f49e.jpg](../icml_results/4034_Riemannian%20Dueling%20Optimization/images/bee8adf13ba6e5c58c011902399a69a05bbd06342dd728a4768e9eac37a1f49e.jpg)

![c7be5f77868b30fae8c16f2665ec470d1111ec3626bbcb56a76e7e3c178791eb.jpg](../icml_results/4034_Riemannian%20Dueling%20Optimization/images/c7be5f77868b30fae8c16f2665ec470d1111ec3626bbcb56a76e7e3c178791eb.jpg)

![cb19ad66caf51d97a36334658aded22a3e01794d25fcd585ef4738bd20ec9cd5.jpg](../icml_results/4034_Riemannian%20Dueling%20Optimization/images/cb19ad66caf51d97a36334658aded22a3e01794d25fcd585ef4738bd20ec9cd5.jpg)

### Tables

![38b00941c6324c0fddd7e434635189c9e0cf05cbd79970dece2cc57b85fb3915.jpg](../icml_results/4034_Riemannian%20Dueling%20Optimization/tables/38b00941c6324c0fddd7e434635189c9e0cf05cbd79970dece2cc57b85fb3915.jpg)

![3eee45f04b63edd02debe081bbaa584a3f4c8dc33950b841767d370a929f0356.jpg](../icml_results/4034_Riemannian%20Dueling%20Optimization/tables/3eee45f04b63edd02debe081bbaa584a3f4c8dc33950b841767d370a929f0356.jpg)
