# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 140 of 197

## 目录 (Table of Contents)

1. [ScaleEnv: Scaling Environment Synthesis from Scratch for Generalist Interactive Tool-Use Agent Training](#paper-4588)
2. [Rethinking 1-bit Optimization Leveraging Pre-trained Large Language Models](#paper-4589)
3. [DyCon: Dynamic Reasoning Control via Evolving Difficulty Modeling](#paper-4590)
4. [Unifying Heterogeneous Degradations: Uncertainty-Aware Diffusion Bridge Model for All-in-One Image Restoration](#paper-4591)
5. [TPGDiff : Hierarchical Triple-Prior Guided Diffusion for Image Restoration](#paper-4592)
6. [Structure-Induced Information for Rerooting Levin Tree Search](#paper-4593)
7. [Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences](#paper-4594)
8. [SparseSSM: Efficient Selective Structured State Space Models Can Be Pruned in One-Shot](#paper-4595)
9. [Unfolding Generative Flows with Koopman Operators: Trajectory-Preserving Linearization](#paper-4596)
10. [Greedy Coordinate Diffusion: Effective and Semantically Coherent Adversarial Attacks via Diffusion Guidance](#paper-4597)
11. [SCOUT: Cyclic Causal Discovery Under Soft Interventions with Unknown Targets](#paper-4598)
12. [Quantifying the Uncertainty of Foundation Models with Singular Value Ensembles](#paper-4599)
13. [Gradient Descent as a Perceptron Algorithm: Understanding Dynamics and Implicit Acceleration](#paper-4600)
14. [Near-Optimal Convergence of Accelerated Gradient Methods under Generalized and $(L_0,L_1)$-Smoothness](#paper-4601)
15. [A Geometric Lens on Physics-Aligned Data Compression](#paper-4602)
16. [Efficient DP-SGD for LLMs with Randomized Clipping](#paper-4603)
17. [HASTE: Hardware-Aware Dynamic Sparse Training for Large Output Spaces](#paper-4604)
18. [VeRO: A Harness for Agents to Optimize Agents](#paper-4605)
19. [The Art of Interrogation: Consistency Amplifies Factuality in Spatial Reasoning](#paper-4606)
20. [BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](#paper-4607)
21. [Trust Functions: Near-Lossless Weak-to-Strong Generalization by Learning When to Trust the Weak Teacher](#paper-4608)
22. [RAIGen: Rare Attribute Identification in Text-to-Image Generative Models](#paper-4609)
23. [Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?](#paper-4610)
24. [Convergence Rate of the Last Iterate of Stochastic Proximal Algorithms](#paper-4611)
25. [Long-term Fairness with Selective Labels](#paper-4612)
26. [DB-KSVD: Scalable Alternating Optimization for Disentangling High-Dimensional Embedding Spaces](#paper-4613)
27. [Optimal Transport Group Counterfactual Explanations](#paper-4614)
28. [FPTQuant: Function-Preserving Transforms for LLM Quantization](#paper-4615)
29. [DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning](#paper-4616)
30. [Neural Low-Discrepancy Sequences](#paper-4617)
31. [Stein Diffusion Guidance: Training-Free Posterior Correction for Sampling Beyond High-Density Regions](#paper-4618)
32. [Variance-Reduced $(\varepsilon, δ)-$Unlearning using Forget Set Gradients](#paper-4619)
33. [Symbal: Detecting Systematic Misalignments in Model-Generated Captions](#paper-4620)

---


<a id="paper-4588"></a>

## ScaleEnv: Scaling Environment Synthesis from Scratch for Generalist Interactive Tool-Use Agent Training

### Images

![0670715f5a510f0b94e26b7b0daf23ea6d46d0c6b7c95bef57b99860fd4ef9c1.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/0670715f5a510f0b94e26b7b0daf23ea6d46d0c6b7c95bef57b99860fd4ef9c1.jpg)

![0bc23acdaf3bffe21758dc1c9a8245bc87588123b060a556776af7c5828a992a.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/0bc23acdaf3bffe21758dc1c9a8245bc87588123b060a556776af7c5828a992a.jpg)

![0ed5560589b0ddd08863ef6ffae8d081f606bb3c7935942293a41f55b2b78db4.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/0ed5560589b0ddd08863ef6ffae8d081f606bb3c7935942293a41f55b2b78db4.jpg)

![142c3b22f6630bb9a231897cd7e395f0ac7f5f007832f17f7c915636ecec3286.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/142c3b22f6630bb9a231897cd7e395f0ac7f5f007832f17f7c915636ecec3286.jpg)

![231744735515e97230d185d0ac544e5e5fb520142fb545569f00ea4526ecf913.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/231744735515e97230d185d0ac544e5e5fb520142fb545569f00ea4526ecf913.jpg)

![26ce346038ea9ead68d9ff63941511e7b548c6e32fbf71eccd6969d2cd9e09b0.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/26ce346038ea9ead68d9ff63941511e7b548c6e32fbf71eccd6969d2cd9e09b0.jpg)

![2fa5342da47b5b731e7d90b3de1923c649b9f3a01f10ad5393ac6e6579bc11b2.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/2fa5342da47b5b731e7d90b3de1923c649b9f3a01f10ad5393ac6e6579bc11b2.jpg)

![3c0fbb8dad746e2f1ad171d9f7e09a2ea6d648f712d5a61e50b64f8659bd5fa3.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/3c0fbb8dad746e2f1ad171d9f7e09a2ea6d648f712d5a61e50b64f8659bd5fa3.jpg)

![42af4ad6aa9fdc9f2a1c0fc7a30e785510b831d70bb46b687766b0d5a35eb956.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/42af4ad6aa9fdc9f2a1c0fc7a30e785510b831d70bb46b687766b0d5a35eb956.jpg)

![44553a95bfa4d8233a3405ef4d29985398513bbaa0f5ef723ea8c1fe4e11c90a.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/44553a95bfa4d8233a3405ef4d29985398513bbaa0f5ef723ea8c1fe4e11c90a.jpg)

![6f6f84e201e2381d4dbf6423174e7eea5495ec08f5c68f0d14a185fa11e250e3.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/6f6f84e201e2381d4dbf6423174e7eea5495ec08f5c68f0d14a185fa11e250e3.jpg)

![728816faf85836625d584b959ea901a673a95bb5ed378109f7f091a73d82ac1c.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/728816faf85836625d584b959ea901a673a95bb5ed378109f7f091a73d82ac1c.jpg)

![888ecb8f72a86524382c311771f50ed87bed1b4046f089e1298f00a083feaa90.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/888ecb8f72a86524382c311771f50ed87bed1b4046f089e1298f00a083feaa90.jpg)

![976b7dd7db454a56d8519ad8a97b7ed500b14efbfe41f336d5a50e217a1bf3c4.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/976b7dd7db454a56d8519ad8a97b7ed500b14efbfe41f336d5a50e217a1bf3c4.jpg)

![a321ba489245a642ca59076e53ee1e61ca9b1f354bab697d3cd2a4186887ee3c.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/a321ba489245a642ca59076e53ee1e61ca9b1f354bab697d3cd2a4186887ee3c.jpg)

![cdb6065df10e77fe44f560f7d481dbc3f7af65fb51e0b4f8e32865362847f000.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/cdb6065df10e77fe44f560f7d481dbc3f7af65fb51e0b4f8e32865362847f000.jpg)

![cf402613c958d03d23dfc4d8d23fa85ad4c23375f8c789b7aa27f302613b0834.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/cf402613c958d03d23dfc4d8d23fa85ad4c23375f8c789b7aa27f302613b0834.jpg)

![d7e95d70e780c7d201c0962d1bb139965568dafd81e4386cb633fc82e4b0bd98.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/images/d7e95d70e780c7d201c0962d1bb139965568dafd81e4386cb633fc82e4b0bd98.jpg)

### Tables

![15f8ccd38c7c18b7d8895de196a130864c9baf12722715e627fe39722e8045e3.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/15f8ccd38c7c18b7d8895de196a130864c9baf12722715e627fe39722e8045e3.jpg)

![2d7b06918c3ab3d31e0e61901114b93459458a5a18dc3ba536f8f05b9d0f1c29.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/2d7b06918c3ab3d31e0e61901114b93459458a5a18dc3ba536f8f05b9d0f1c29.jpg)

![3eca65b7a92975729bc824c7e0b743be84628059c6d605041383c19103516ba5.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/3eca65b7a92975729bc824c7e0b743be84628059c6d605041383c19103516ba5.jpg)

![802ceaad152fba5b089eacd48e46d5580e26d27cf8a1c416ffaaf346a95bbb88.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/802ceaad152fba5b089eacd48e46d5580e26d27cf8a1c416ffaaf346a95bbb88.jpg)

![83cb89d419a369a462fba9b62cfadac55d5de4588920be5e94e573aa3315daea.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/83cb89d419a369a462fba9b62cfadac55d5de4588920be5e94e573aa3315daea.jpg)

![8777fb9fd0fe863f73fbfa8a3180bd3fb44eef885bc14618c553e657429d1f1e.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/8777fb9fd0fe863f73fbfa8a3180bd3fb44eef885bc14618c553e657429d1f1e.jpg)

![9a5c22af11b813c1055baa452913e8e095b5181981ef34d8f1ba06a8025a4e84.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/9a5c22af11b813c1055baa452913e8e095b5181981ef34d8f1ba06a8025a4e84.jpg)

![a839af2b474953ed50a654251a7fee786c5328fc68222047061487d1bec92c11.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/a839af2b474953ed50a654251a7fee786c5328fc68222047061487d1bec92c11.jpg)

![ac8733cd1d641fe4f432b6d22faaa7b2f883c56185dd263e5314c7375e79eddb.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/ac8733cd1d641fe4f432b6d22faaa7b2f883c56185dd263e5314c7375e79eddb.jpg)

![e3eb3d17ddac136d0f8b3ec9c13a03c435b952af4feab9ddd432b48dd0b925f8.jpg](../icml_results/4634_Protein%20Circuit%20Tracing%20via%20Cross-layer%20Transcoders/tables/e3eb3d17ddac136d0f8b3ec9c13a03c435b952af4feab9ddd432b48dd0b925f8.jpg)

## ScaleEnv: Scaling Environment Synthesis from Scratch for Generalist Interactive Tool-Use Agent Training


### Images

![33f48b07f0011c6e7e87e36c7742598152870f361bd6e079ac78140d4028360d.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/images/33f48b07f0011c6e7e87e36c7742598152870f361bd6e079ac78140d4028360d.jpg)

![8356e1b0fec6c87b3121fcab83edd6f3d7130332d7c42b43189e03eb444b461a.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/images/8356e1b0fec6c87b3121fcab83edd6f3d7130332d7c42b43189e03eb444b461a.jpg)

![bb98bb0cebad855d4f0bd8a0f8951f5a3adcecbe411bf67d761ad7d28ebedb78.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/images/bb98bb0cebad855d4f0bd8a0f8951f5a3adcecbe411bf67d761ad7d28ebedb78.jpg)

![ca9a259d9d3632a32c1791e2aad632a521088ae8b2d15f474310c353141a869e.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/images/ca9a259d9d3632a32c1791e2aad632a521088ae8b2d15f474310c353141a869e.jpg)

![d1272da1f10157e7c2dc4157ad2e945603df9e5dcd9737dc3fdf3c44da30d213.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/images/d1272da1f10157e7c2dc4157ad2e945603df9e5dcd9737dc3fdf3c44da30d213.jpg)

![d46dbad2df99596283bad27331051312e2f205a0053fca17de8c9d7ef1035735.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/images/d46dbad2df99596283bad27331051312e2f205a0053fca17de8c9d7ef1035735.jpg)

![f8db85ccdbf186608b9f15dc8cad404b747ae7398ef1144f5ff5a0fdd1437f52.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/images/f8db85ccdbf186608b9f15dc8cad404b747ae7398ef1144f5ff5a0fdd1437f52.jpg)

### Tables

![0033a68028a5d5298d5b795bc158e194af6f47dbc6d9487d7bbdc8c6e7af9e33.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/tables/0033a68028a5d5298d5b795bc158e194af6f47dbc6d9487d7bbdc8c6e7af9e33.jpg)

![0fbf9bfb4bf892353c5935e1e95e6859ff470d689daa76af173891e4fcf7ebe1.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/tables/0fbf9bfb4bf892353c5935e1e95e6859ff470d689daa76af173891e4fcf7ebe1.jpg)

![20fdc029f013a00ba0820c7e70af5f7a4299a86f1b4f8e6ea177e72eb2934a6b.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/tables/20fdc029f013a00ba0820c7e70af5f7a4299a86f1b4f8e6ea177e72eb2934a6b.jpg)

![4fee4ec04f1866f40dd12643c39277843ac57af7f88ffc3ee9da90a95986db16.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/tables/4fee4ec04f1866f40dd12643c39277843ac57af7f88ffc3ee9da90a95986db16.jpg)

![77ea876100752def70f5af6fd07f4a7965c6fcbaaf0243d887ba63e88baa4f90.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/tables/77ea876100752def70f5af6fd07f4a7965c6fcbaaf0243d887ba63e88baa4f90.jpg)

![784893ded5a86db1924786052ce4456a3b7af48eafcaee65e2559a38940f48b8.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/tables/784893ded5a86db1924786052ce4456a3b7af48eafcaee65e2559a38940f48b8.jpg)

![7b6435d2ac59d1099f8874341de0d7f2377cf324f2e8b1940f3f7d4045803ee7.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/tables/7b6435d2ac59d1099f8874341de0d7f2377cf324f2e8b1940f3f7d4045803ee7.jpg)

![80cc0285e3d3b9ddb8fa4a5df23ef621557ca9f4825a1930178f6d4ba8bad1ff.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/tables/80cc0285e3d3b9ddb8fa4a5df23ef621557ca9f4825a1930178f6d4ba8bad1ff.jpg)

![985e98dab45c13d99e61d5bd32152eeaa7c8afe22ceabc30b12a213fd776cfdd.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/tables/985e98dab45c13d99e61d5bd32152eeaa7c8afe22ceabc30b12a213fd776cfdd.jpg)

![99ddc6ed660024c60450c4dc76fe493b453012c7fa628a72acc5ccd29538ba3f.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/tables/99ddc6ed660024c60450c4dc76fe493b453012c7fa628a72acc5ccd29538ba3f.jpg)

![a9046ce0671396a073a958ec520e798b40aeca99fff38f219374a7a925ee9ceb.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/tables/a9046ce0671396a073a958ec520e798b40aeca99fff38f219374a7a925ee9ceb.jpg)

![bfa2039361d11b2c662048f3bc6d17f7f1c5afc3b31e4849df7b8f96706b2589.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/tables/bfa2039361d11b2c662048f3bc6d17f7f1c5afc3b31e4849df7b8f96706b2589.jpg)

![c03a6192603cd1417215024d7703b8b2e906255b27672c2e58d2a7d2a3d519cd.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/tables/c03a6192603cd1417215024d7703b8b2e906255b27672c2e58d2a7d2a3d519cd.jpg)

![dd1c1f32cd2fa08912ad953048c19ace7184ebf28530e70ffc72090e1eae8fcd.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/tables/dd1c1f32cd2fa08912ad953048c19ace7184ebf28530e70ffc72090e1eae8fcd.jpg)

![e61fa54f0681f5bae82578cb770cf21d9cfd21cb8cdb8e9ba6c673f156f3157c.jpg](../icml_results/4635_ScaleEnv_%20Scaling%20Environment%20Synthesis%20from%20Scratch%20for%20Generalist%20Interactive%20Tool-Use%20Agent%20Train/tables/e61fa54f0681f5bae82578cb770cf21d9cfd21cb8cdb8e9ba6c673f156f3157c.jpg)

<a id="paper-4589"></a>

## Rethinking 1-bit Optimization Leveraging Pre-trained Large Language Models


### Images

![09bca7a2c1ef0eb61016ae2c52df0479cc6f6e6b2e25342e9dde01cd344f8e38.jpg](../icml_results/4636_Rethinking%201-bit%20Optimization%20Leveraging%20Pre-trained%20Large%20Language%20Models/images/09bca7a2c1ef0eb61016ae2c52df0479cc6f6e6b2e25342e9dde01cd344f8e38.jpg)

![296a31ffaa18ea6dfdfdaa595b099ef72fe00a1aa4b8fa54600b62de82843103.jpg](../icml_results/4636_Rethinking%201-bit%20Optimization%20Leveraging%20Pre-trained%20Large%20Language%20Models/images/296a31ffaa18ea6dfdfdaa595b099ef72fe00a1aa4b8fa54600b62de82843103.jpg)

![ae39636385ed0f7f3362106db0632b398cf9e9a57ca693ac90870604099b2469.jpg](../icml_results/4636_Rethinking%201-bit%20Optimization%20Leveraging%20Pre-trained%20Large%20Language%20Models/images/ae39636385ed0f7f3362106db0632b398cf9e9a57ca693ac90870604099b2469.jpg)

![c416ba6daac500a253820874eba7a958efdb6e8baebd3dc101827836bad1dcb4.jpg](../icml_results/4636_Rethinking%201-bit%20Optimization%20Leveraging%20Pre-trained%20Large%20Language%20Models/images/c416ba6daac500a253820874eba7a958efdb6e8baebd3dc101827836bad1dcb4.jpg)

![cce81ebb3f6bd05b6148cd52a5ae11dfdc4dcc17c3246ec933b93fb6714e2bd5.jpg](../icml_results/4636_Rethinking%201-bit%20Optimization%20Leveraging%20Pre-trained%20Large%20Language%20Models/images/cce81ebb3f6bd05b6148cd52a5ae11dfdc4dcc17c3246ec933b93fb6714e2bd5.jpg)

![e04108642db4a33457929407c1c85c4ff6b8885a58a3524b9a6153252a948c3c.jpg](../icml_results/4636_Rethinking%201-bit%20Optimization%20Leveraging%20Pre-trained%20Large%20Language%20Models/images/e04108642db4a33457929407c1c85c4ff6b8885a58a3524b9a6153252a948c3c.jpg)

![e38bd0675484d2a9653a4412c45d44660d6a3383dd355e7e750d38e4571fca60.jpg](../icml_results/4636_Rethinking%201-bit%20Optimization%20Leveraging%20Pre-trained%20Large%20Language%20Models/images/e38bd0675484d2a9653a4412c45d44660d6a3383dd355e7e750d38e4571fca60.jpg)

### Tables

![25bba3de9a6a728aa7d192a5f33b1e29eba6cdf92531838dbfbf1e085aded5aa.jpg](../icml_results/4636_Rethinking%201-bit%20Optimization%20Leveraging%20Pre-trained%20Large%20Language%20Models/tables/25bba3de9a6a728aa7d192a5f33b1e29eba6cdf92531838dbfbf1e085aded5aa.jpg)

![6ade6d82695e5277904aa43cfe924df30eb0851c6cea5dc40b9f7f9c3a670def.jpg](../icml_results/4636_Rethinking%201-bit%20Optimization%20Leveraging%20Pre-trained%20Large%20Language%20Models/tables/6ade6d82695e5277904aa43cfe924df30eb0851c6cea5dc40b9f7f9c3a670def.jpg)

![82385fcaab90969c8ff3115d2f18165d4532e4ad08c856c2fc7cf7b3b679c35c.jpg](../icml_results/4636_Rethinking%201-bit%20Optimization%20Leveraging%20Pre-trained%20Large%20Language%20Models/tables/82385fcaab90969c8ff3115d2f18165d4532e4ad08c856c2fc7cf7b3b679c35c.jpg)

![a2d315373107fb4fa45beaab33e317e0de04948a70b392a463c505a477605cf2.jpg](../icml_results/4636_Rethinking%201-bit%20Optimization%20Leveraging%20Pre-trained%20Large%20Language%20Models/tables/a2d315373107fb4fa45beaab33e317e0de04948a70b392a463c505a477605cf2.jpg)

![a347557b928ed7bc627d113739460e6003a5788bd9fb4d6906c279fedeb24f2b.jpg](../icml_results/4636_Rethinking%201-bit%20Optimization%20Leveraging%20Pre-trained%20Large%20Language%20Models/tables/a347557b928ed7bc627d113739460e6003a5788bd9fb4d6906c279fedeb24f2b.jpg)

![bc5b6034f0f1bfe4ad62c889d73561e052a0269c80d17dc121758dace5c455fc.jpg](../icml_results/4636_Rethinking%201-bit%20Optimization%20Leveraging%20Pre-trained%20Large%20Language%20Models/tables/bc5b6034f0f1bfe4ad62c889d73561e052a0269c80d17dc121758dace5c455fc.jpg)

![cea54352022e3d29caa8c3c7103447bdc2cab4805f2c2b7c3f988e133679737c.jpg](../icml_results/4636_Rethinking%201-bit%20Optimization%20Leveraging%20Pre-trained%20Large%20Language%20Models/tables/cea54352022e3d29caa8c3c7103447bdc2cab4805f2c2b7c3f988e133679737c.jpg)

![da9fabce31b34e1141a24acd139ef621d66e352ac1d8af2bb6f79660c0669e7a.jpg](../icml_results/4636_Rethinking%201-bit%20Optimization%20Leveraging%20Pre-trained%20Large%20Language%20Models/tables/da9fabce31b34e1141a24acd139ef621d66e352ac1d8af2bb6f79660c0669e7a.jpg)

![e4a79aacf7b4f207e0aeef30ce6ccd9f2d224b4bdca39e0c3921379922bee2d3.jpg](../icml_results/4636_Rethinking%201-bit%20Optimization%20Leveraging%20Pre-trained%20Large%20Language%20Models/tables/e4a79aacf7b4f207e0aeef30ce6ccd9f2d224b4bdca39e0c3921379922bee2d3.jpg)

<a id="paper-4590"></a>

## DyCon: Dynamic Reasoning Control via Evolving Difficulty Modeling


### Images

![06b87d83ea99838ab686a85c62377977ff9ffb3765f1836e53be8e1796ee80a1.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/images/06b87d83ea99838ab686a85c62377977ff9ffb3765f1836e53be8e1796ee80a1.jpg)

![15ea6d11ee1fe5dae0f6761c68930412a83c0451b1e20e909ba3113744b4b1c2.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/images/15ea6d11ee1fe5dae0f6761c68930412a83c0451b1e20e909ba3113744b4b1c2.jpg)

![19cdd5e94a213df5a807585183a889a1273c86ca7b6a584cddf8946f451adb82.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/images/19cdd5e94a213df5a807585183a889a1273c86ca7b6a584cddf8946f451adb82.jpg)

![4015b237558763a3791ef10916c5311155b518c9f9ebd6fef966ce2fabd1b2c8.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/images/4015b237558763a3791ef10916c5311155b518c9f9ebd6fef966ce2fabd1b2c8.jpg)

![484ff9ae886fbf547df20d437927771d4d751f879dfb80645af6dd4cf5e940fb.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/images/484ff9ae886fbf547df20d437927771d4d751f879dfb80645af6dd4cf5e940fb.jpg)

![496907a035711832d77d65fd44ee44eb6117858ad593c2bf27aac9dd47007d1d.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/images/496907a035711832d77d65fd44ee44eb6117858ad593c2bf27aac9dd47007d1d.jpg)

![5a5566dab2f7c39d30fd958d233bb7ecba54ac35ff45c97a996b7083f165c356.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/images/5a5566dab2f7c39d30fd958d233bb7ecba54ac35ff45c97a996b7083f165c356.jpg)

![6298cf5938b99b09be3b77864af3e881b09fded070fb6438c182e7870f508e49.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/images/6298cf5938b99b09be3b77864af3e881b09fded070fb6438c182e7870f508e49.jpg)

![9cc6107d048e0bc9b736ec0eb1c4437497f89ae5001594358380ae0eb5ab73b1.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/images/9cc6107d048e0bc9b736ec0eb1c4437497f89ae5001594358380ae0eb5ab73b1.jpg)

![bf29160c7619c265dbd808d983006f4959e6ac37f3697c807a5b808af3520f41.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/images/bf29160c7619c265dbd808d983006f4959e6ac37f3697c807a5b808af3520f41.jpg)

![c1cc0105fa2ff8a5e8d18ea024d536ea2604f6206e55a57633e808288b34da28.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/images/c1cc0105fa2ff8a5e8d18ea024d536ea2604f6206e55a57633e808288b34da28.jpg)

![c8190d0ba5f52419b8351389e249aa5a6053f1128b751ea1bb5407e6564f21ed.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/images/c8190d0ba5f52419b8351389e249aa5a6053f1128b751ea1bb5407e6564f21ed.jpg)

![d1c82b63137a25b2bacb4945436781a370a25907241b2e82e18699604511671d.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/images/d1c82b63137a25b2bacb4945436781a370a25907241b2e82e18699604511671d.jpg)

![f91ce1b634f67e53d02cd5d14c4f0dccf3dddbe5c39cd5ceaaeba76623331c74.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/images/f91ce1b634f67e53d02cd5d14c4f0dccf3dddbe5c39cd5ceaaeba76623331c74.jpg)

![f971842bcc49a3153dca8c2bafd15204adf4b9a519528963908e07923e7bfed5.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/images/f971842bcc49a3153dca8c2bafd15204adf4b9a519528963908e07923e7bfed5.jpg)

![fa2c8c0b91ad80465021c6dbf22b407388bdc309c6137d37b78745392cc55b29.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/images/fa2c8c0b91ad80465021c6dbf22b407388bdc309c6137d37b78745392cc55b29.jpg)

### Tables

![019374dffec789fdc67f6cc7f52423f1d2f7739e9c48bbd2bf3676d52e853195.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/019374dffec789fdc67f6cc7f52423f1d2f7739e9c48bbd2bf3676d52e853195.jpg)

![156f3ce8667dd5b945b736a31def72ab67d6ca29fb2abf7d13fa0db11fc3f9f4.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/156f3ce8667dd5b945b736a31def72ab67d6ca29fb2abf7d13fa0db11fc3f9f4.jpg)

![2115633310c4f30e4162ce4c5090a6f1a52f83ecdb062e6b164d39420949b7be.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/2115633310c4f30e4162ce4c5090a6f1a52f83ecdb062e6b164d39420949b7be.jpg)

![23607b3e4e09ee64beb61eb1fb7fc2e91ec163c91c50959007686c4c90bda255.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/23607b3e4e09ee64beb61eb1fb7fc2e91ec163c91c50959007686c4c90bda255.jpg)

![2c868c373618655daececd76d972ab7c0deb61138b2099a67f906c842b756e91.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/2c868c373618655daececd76d972ab7c0deb61138b2099a67f906c842b756e91.jpg)

![48f0195025553521ccf70b6c4209ec5f523408dfa29d8921f4270116f89e56a9.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/48f0195025553521ccf70b6c4209ec5f523408dfa29d8921f4270116f89e56a9.jpg)

![4923daa7744cdf39266ef370e853cb852b91196be0e03ea7ab22a1031ef54e62.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/4923daa7744cdf39266ef370e853cb852b91196be0e03ea7ab22a1031ef54e62.jpg)

![5f8d64540b86efaa2bf703c0b59070ab64de6d94fafa3c7e906910171b433163.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/5f8d64540b86efaa2bf703c0b59070ab64de6d94fafa3c7e906910171b433163.jpg)

![6096a01081bef9f0ce2d9d7fdd4fe8a74a1529cb523f23ff09fc2b8ccafc9825.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/6096a01081bef9f0ce2d9d7fdd4fe8a74a1529cb523f23ff09fc2b8ccafc9825.jpg)

![663e6b8c869dd79106b3d2c1d4eacfd505d579f9fb607051aa0c189ce72ecf2e.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/663e6b8c869dd79106b3d2c1d4eacfd505d579f9fb607051aa0c189ce72ecf2e.jpg)

![66e8d387257b97a2bdc40ed7d29bc24cbff41516abe881404609a2316f08f5bf.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/66e8d387257b97a2bdc40ed7d29bc24cbff41516abe881404609a2316f08f5bf.jpg)

![6ce5fc06a05708a757e6d66b3e61342d1a4fb3e951408891beae4d23b0844496.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/6ce5fc06a05708a757e6d66b3e61342d1a4fb3e951408891beae4d23b0844496.jpg)

![75eaf1c36d9de1a12e410fa9961962b88f96ec56e325205b1089eb62101f4f80.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/75eaf1c36d9de1a12e410fa9961962b88f96ec56e325205b1089eb62101f4f80.jpg)

![84f57473dde5fcf35edbf20a4e89893ed87f33216ba8cb911115722af6c3f230.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/84f57473dde5fcf35edbf20a4e89893ed87f33216ba8cb911115722af6c3f230.jpg)

![8b60ba8571bbf34f9438eac9e2be047b013f1e847401e4162c801e693f660775.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/8b60ba8571bbf34f9438eac9e2be047b013f1e847401e4162c801e693f660775.jpg)

![8d8437dd22a73d59f22c9c2506fe868fe5160c6fdf3328f2713584218066cfa9.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/8d8437dd22a73d59f22c9c2506fe868fe5160c6fdf3328f2713584218066cfa9.jpg)

![978653d27e0cf8a16d832ed49b2da24d30bac3e33d21978859eb4229569eeb2d.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/978653d27e0cf8a16d832ed49b2da24d30bac3e33d21978859eb4229569eeb2d.jpg)

![99c3f2824e17c8c8647237c8eeab62ba8df573c048c54360731eb3926126b118.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/99c3f2824e17c8c8647237c8eeab62ba8df573c048c54360731eb3926126b118.jpg)

![9b7e6ad61202c1402fc6d08dd01b8c119536d3ce510894a2b5ce6b56c61b9060.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/9b7e6ad61202c1402fc6d08dd01b8c119536d3ce510894a2b5ce6b56c61b9060.jpg)

![9c82da89a956c357b9400a2f9698c6a394852cc2064e63dd481ac5664655c848.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/9c82da89a956c357b9400a2f9698c6a394852cc2064e63dd481ac5664655c848.jpg)

![a22197432a0d108d2f574dd98163741ca47be9b9f908201ae9a6147bf4af7a53.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/a22197432a0d108d2f574dd98163741ca47be9b9f908201ae9a6147bf4af7a53.jpg)

![af4bae3d6fde3ac4e38898fffbee827a173bacc33022806c5a0de1e264dfb449.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/af4bae3d6fde3ac4e38898fffbee827a173bacc33022806c5a0de1e264dfb449.jpg)

![b25445ce676fff1fed2049cfc58bcb6ef1c26d696a04fac6367790f11f7e990f.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/b25445ce676fff1fed2049cfc58bcb6ef1c26d696a04fac6367790f11f7e990f.jpg)

![c59749f242c0eb7029a90dd49379353055030ee3bec2e7e17e464c4f9e7370c8.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/c59749f242c0eb7029a90dd49379353055030ee3bec2e7e17e464c4f9e7370c8.jpg)

![c921dae22097d6fb9b918fafd57cd702d1ce5be5cfb9a9d7a88560c0416b200d.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/c921dae22097d6fb9b918fafd57cd702d1ce5be5cfb9a9d7a88560c0416b200d.jpg)

![d35025a382bfa291f30f33aa529c6dd4411063b7833a9d5c96a2f96b80dd756e.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/d35025a382bfa291f30f33aa529c6dd4411063b7833a9d5c96a2f96b80dd756e.jpg)

![d5e0fd2b6249b06b6ff7bd78922e22d4377cd85c610dc9f97b533510d076dfaf.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/d5e0fd2b6249b06b6ff7bd78922e22d4377cd85c610dc9f97b533510d076dfaf.jpg)

![d7d37c8f6cc77db927ac1ce0e5abab9374e643b1ad4da5508ffcce970f248ab8.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/d7d37c8f6cc77db927ac1ce0e5abab9374e643b1ad4da5508ffcce970f248ab8.jpg)

![d7f1aab1e4377e0b7a00e55a5feb7049b8910e1b2b441f065b72cea2dbe6b2a1.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/d7f1aab1e4377e0b7a00e55a5feb7049b8910e1b2b441f065b72cea2dbe6b2a1.jpg)

![da6576bb3af542a4c5c4d165e325b0ae290aa74536472c519a298f28ca0618c2.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/da6576bb3af542a4c5c4d165e325b0ae290aa74536472c519a298f28ca0618c2.jpg)

![dd2a111d9c7e6e4be053973f4e192ca5bf09a1d0c488a03c60b6b43f977c841a.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/dd2a111d9c7e6e4be053973f4e192ca5bf09a1d0c488a03c60b6b43f977c841a.jpg)

![e1717277cebf0ce40cdaf53d5d53f97f2bd292166473ab780652e5aeeea1808d.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/e1717277cebf0ce40cdaf53d5d53f97f2bd292166473ab780652e5aeeea1808d.jpg)

![e59d0d2233ca9f6ab9c8bb48a0fb8ed184a4ffd37fb521314799ebab17a43cf8.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/e59d0d2233ca9f6ab9c8bb48a0fb8ed184a4ffd37fb521314799ebab17a43cf8.jpg)

![ed3f6e6c11188da2ab6f813ce1c1a98af95ee1d239fc8dc1704827d29a53cf6c.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/ed3f6e6c11188da2ab6f813ce1c1a98af95ee1d239fc8dc1704827d29a53cf6c.jpg)

![f3c05099035f9a66e34ea4d0db8264d094b97617edb0bd47cb846c6547bdfe7d.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/f3c05099035f9a66e34ea4d0db8264d094b97617edb0bd47cb846c6547bdfe7d.jpg)

![f5af0e3054910550b1e211fba197ea865379b02f1d01fbb88f2a9775e44fbcb7.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/f5af0e3054910550b1e211fba197ea865379b02f1d01fbb88f2a9775e44fbcb7.jpg)

![f5e04eb0d9e0be73160f955ee584a13aa4dba34488d9b05c951dab08d4ee18fd.jpg](../icml_results/4637_DyCon_%20Dynamic%20Reasoning%20Control%20via%20Evolving%20Difficulty%20Modeling/tables/f5e04eb0d9e0be73160f955ee584a13aa4dba34488d9b05c951dab08d4ee18fd.jpg)

<a id="paper-4591"></a>

## Unifying Heterogeneous Degradations: Uncertainty-Aware Diffusion Bridge Model for All-in-One Image Restoration


### Images

![0271eee119915ed06d723e93c7d695ae48ab8cc9b1b35b4f8a37de9278c52885.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/0271eee119915ed06d723e93c7d695ae48ab8cc9b1b35b4f8a37de9278c52885.jpg)

![139b2032f7e25224d881b188b36e36f4d4764100e47589752bbd6ab457777f55.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/139b2032f7e25224d881b188b36e36f4d4764100e47589752bbd6ab457777f55.jpg)

![30937793233e820b9c2000f65274286c763debc4336f85375215e2e8877df0c9.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/30937793233e820b9c2000f65274286c763debc4336f85375215e2e8877df0c9.jpg)

![3ffea511b942b4aafbef097715ed9ac6866722a7e945cd21e1b505295aff5602.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/3ffea511b942b4aafbef097715ed9ac6866722a7e945cd21e1b505295aff5602.jpg)

![5658334bd5261ddddbe8ec752d70602f52e23a379e6a692c03692559b4809bbb.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/5658334bd5261ddddbe8ec752d70602f52e23a379e6a692c03692559b4809bbb.jpg)

![5b4ab69a7f13c23a3206331a7c95cee8cf6152fa1b0ea70883c77f28d02cb570.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/5b4ab69a7f13c23a3206331a7c95cee8cf6152fa1b0ea70883c77f28d02cb570.jpg)

![666c608ed1fb55848dbc1137f373f4d311d48aa5adebe830e6d02b6db2a3fe8e.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/666c608ed1fb55848dbc1137f373f4d311d48aa5adebe830e6d02b6db2a3fe8e.jpg)

![6bbb0651534fe0529b95effa480c0ebb876fdec62eafdb515846c71693fb9b78.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/6bbb0651534fe0529b95effa480c0ebb876fdec62eafdb515846c71693fb9b78.jpg)

![6f86c8c3d3dba4abfa7256c461823d986ed3aef888cba48ebe762efaaeb1616a.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/6f86c8c3d3dba4abfa7256c461823d986ed3aef888cba48ebe762efaaeb1616a.jpg)

![850b8e05f64d1e15c9efd8974f345fde58a602304b78dbbb218394295818752d.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/850b8e05f64d1e15c9efd8974f345fde58a602304b78dbbb218394295818752d.jpg)

![86453706bd9c9bfc4afe529fa6c99e35c399efc92aee784c29cb84cd800bd68f.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/86453706bd9c9bfc4afe529fa6c99e35c399efc92aee784c29cb84cd800bd68f.jpg)

![92e3170a1c6fa53e5266b5399b5cb5863c6044d0cfb61738027e75c75be63ad4.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/92e3170a1c6fa53e5266b5399b5cb5863c6044d0cfb61738027e75c75be63ad4.jpg)

![9636d39e0465a6639968b2e445701ec512b2a4ecb2cad6bb73fbef617b8a47c7.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/9636d39e0465a6639968b2e445701ec512b2a4ecb2cad6bb73fbef617b8a47c7.jpg)

![98958f2d5af4b5bc648de40c3c606f59b379b187b082c5dcc0c6f8c8b767f9ab.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/98958f2d5af4b5bc648de40c3c606f59b379b187b082c5dcc0c6f8c8b767f9ab.jpg)

![a435004b015186135b88bda871f08260f26952a842ae2749d254e3b6b6cdc527.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/a435004b015186135b88bda871f08260f26952a842ae2749d254e3b6b6cdc527.jpg)

![bb15bb0776358a7b41c72fb4d184d58e4d48b2823859c73c9899517d0b77f021.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/bb15bb0776358a7b41c72fb4d184d58e4d48b2823859c73c9899517d0b77f021.jpg)

![d05a739dfa6dc76e9e9d6bf68ae99e87009a2e8951893d2f7b8bf0d3ebe7849d.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/d05a739dfa6dc76e9e9d6bf68ae99e87009a2e8951893d2f7b8bf0d3ebe7849d.jpg)

![e8ad4165304fa598c32d206c85738e7230a797e37ad69f3fdac89cc50350f215.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/images/e8ad4165304fa598c32d206c85738e7230a797e37ad69f3fdac89cc50350f215.jpg)

### Tables

![042126ccaf6b21f21f4c4812204a2470cafa3ea6c6ba53648d66a83d97320430.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/042126ccaf6b21f21f4c4812204a2470cafa3ea6c6ba53648d66a83d97320430.jpg)

![0a60984198defcf379e389d96377be47f3afbe4f65e92286236dae4db830323b.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/0a60984198defcf379e389d96377be47f3afbe4f65e92286236dae4db830323b.jpg)

![19b8e1d1d330685235da2dd6ba18925b67b13d4fc2e399f01a68a5f569564553.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/19b8e1d1d330685235da2dd6ba18925b67b13d4fc2e399f01a68a5f569564553.jpg)

![20895be24242a3f619172a0bd2654b45dcb17c59cad1d1f4138d89020613cbfe.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/20895be24242a3f619172a0bd2654b45dcb17c59cad1d1f4138d89020613cbfe.jpg)

![3826ed4a2857f919ef137d8ee4673b83dc912e3206e8623533c1e26fad9c7d64.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/3826ed4a2857f919ef137d8ee4673b83dc912e3206e8623533c1e26fad9c7d64.jpg)

![63d323d94f4f78c67c704d3d52fb07e53b33c7f64e5db14a72fa9f07c6b96a8a.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/63d323d94f4f78c67c704d3d52fb07e53b33c7f64e5db14a72fa9f07c6b96a8a.jpg)

![6746db36efb58c4a6e6e95ad92e28e5d5fb87729801837e0e2c4dbda0a2dfc76.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/6746db36efb58c4a6e6e95ad92e28e5d5fb87729801837e0e2c4dbda0a2dfc76.jpg)

![7b58b5fe1f480347c2516564e031d0b43988e1d06c2bc69bb1f0edcdff7ffe4c.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/7b58b5fe1f480347c2516564e031d0b43988e1d06c2bc69bb1f0edcdff7ffe4c.jpg)

![7fe88e088e9432cc28275e44f798afe25286280c6824bf5da0b1f9b994d858b6.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/7fe88e088e9432cc28275e44f798afe25286280c6824bf5da0b1f9b994d858b6.jpg)

![901c74a88cc0074eb54dcbaaf770b43c63371401200dbc29c4f30db6500d8bca.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/901c74a88cc0074eb54dcbaaf770b43c63371401200dbc29c4f30db6500d8bca.jpg)

![9037880d4752effc6ad465d24e4e516e550afce8b047e0e4911b2be024794ae4.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/9037880d4752effc6ad465d24e4e516e550afce8b047e0e4911b2be024794ae4.jpg)

![99155f60f67353755e1be669f1b39420c5c6b6f72b4b5864d844eae583bb82ae.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/99155f60f67353755e1be669f1b39420c5c6b6f72b4b5864d844eae583bb82ae.jpg)

![b2b0f9436e1a534eeeba3475190d36db817e158ec45e0851326ffc7db6c5d738.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/b2b0f9436e1a534eeeba3475190d36db817e158ec45e0851326ffc7db6c5d738.jpg)

![c1c505efe0c8badb5402da22b18649f6a0a63e0c6e56305148dd44a5b8282791.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/c1c505efe0c8badb5402da22b18649f6a0a63e0c6e56305148dd44a5b8282791.jpg)

![e19f6e83bec2ee235d7e972e120b3b6e3415736c5715ddb0392c5751d99d82f9.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/e19f6e83bec2ee235d7e972e120b3b6e3415736c5715ddb0392c5751d99d82f9.jpg)

![e81c28ca22b75a73f164a2884ab3b242e06a4abfca6713e9328dadbf28ac6ca4.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/e81c28ca22b75a73f164a2884ab3b242e06a4abfca6713e9328dadbf28ac6ca4.jpg)

![f00e8fe0b5d6f61d9a9f9e7b6d9ac6dfc89c46a586d1bf4949b5e12d19dfa644.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/f00e8fe0b5d6f61d9a9f9e7b6d9ac6dfc89c46a586d1bf4949b5e12d19dfa644.jpg)

![f1844fd43ccb745941a48e764aae6f4ca2f96370b4e607de61a3df443dbb7556.jpg](../icml_results/4638_Unifying%20Heterogeneous%20Degradations_%20Uncertainty-Aware%20Diffusion%20Bridge%20Model%20for%20All-in-One%20Image%20R/tables/f1844fd43ccb745941a48e764aae6f4ca2f96370b4e607de61a3df443dbb7556.jpg)

<a id="paper-4592"></a>

## TPGDiff : Hierarchical Triple-Prior Guided Diffusion for Image Restoration


### Images

![0a60e0bf9364a7cb02867aa673a8bfb1186e7b6406fab0f9685c9350df5dc326.jpg](../icml_results/4639_TPGDiff%20_%20Hierarchical%20Triple-Prior%20Guided%20Diffusion%20for%20Image%20Restoration/images/0a60e0bf9364a7cb02867aa673a8bfb1186e7b6406fab0f9685c9350df5dc326.jpg)

![0ad66c730b28ca103d916ca3ee061bd881c1fd34f2bca672e70b7bc8397ae652.jpg](../icml_results/4639_TPGDiff%20_%20Hierarchical%20Triple-Prior%20Guided%20Diffusion%20for%20Image%20Restoration/images/0ad66c730b28ca103d916ca3ee061bd881c1fd34f2bca672e70b7bc8397ae652.jpg)

![3a01cd9f700522df2540d1c6ad0e4416372d6583ed920cdf46ff35032eb53e09.jpg](../icml_results/4639_TPGDiff%20_%20Hierarchical%20Triple-Prior%20Guided%20Diffusion%20for%20Image%20Restoration/images/3a01cd9f700522df2540d1c6ad0e4416372d6583ed920cdf46ff35032eb53e09.jpg)

![863f32348bfd7aad2ba5e80ed54f3d6a8fa122edababf28b67f8df5128db2c8b.jpg](../icml_results/4639_TPGDiff%20_%20Hierarchical%20Triple-Prior%20Guided%20Diffusion%20for%20Image%20Restoration/images/863f32348bfd7aad2ba5e80ed54f3d6a8fa122edababf28b67f8df5128db2c8b.jpg)

![b5c54766f205e316321cb39c0248250f464b683c27c350a0ff64817f21c4cff9.jpg](../icml_results/4639_TPGDiff%20_%20Hierarchical%20Triple-Prior%20Guided%20Diffusion%20for%20Image%20Restoration/images/b5c54766f205e316321cb39c0248250f464b683c27c350a0ff64817f21c4cff9.jpg)

![d0cd2ba5b8ff7882d3129eab9259d9921b2ea10ce42e1e32ddcd8fc17cc24d3c.jpg](../icml_results/4639_TPGDiff%20_%20Hierarchical%20Triple-Prior%20Guided%20Diffusion%20for%20Image%20Restoration/images/d0cd2ba5b8ff7882d3129eab9259d9921b2ea10ce42e1e32ddcd8fc17cc24d3c.jpg)

### Tables

![1eaf447e9c28d8c3d9980cc5e2c0d918012159075259af43d7564ed20e1ded34.jpg](../icml_results/4639_TPGDiff%20_%20Hierarchical%20Triple-Prior%20Guided%20Diffusion%20for%20Image%20Restoration/tables/1eaf447e9c28d8c3d9980cc5e2c0d918012159075259af43d7564ed20e1ded34.jpg)

![71ed7dd851f78775d764c18cab1dd2d2921d77f30b38a9831a780e4936e0effa.jpg](../icml_results/4639_TPGDiff%20_%20Hierarchical%20Triple-Prior%20Guided%20Diffusion%20for%20Image%20Restoration/tables/71ed7dd851f78775d764c18cab1dd2d2921d77f30b38a9831a780e4936e0effa.jpg)

![9ed7598d330f19b9c2eedf603f6c31e2ab44d5fecca4fa04134a28bf11063f5e.jpg](../icml_results/4639_TPGDiff%20_%20Hierarchical%20Triple-Prior%20Guided%20Diffusion%20for%20Image%20Restoration/tables/9ed7598d330f19b9c2eedf603f6c31e2ab44d5fecca4fa04134a28bf11063f5e.jpg)

![b4e9b643d8dc2f4d5a48ffe9a92d150b23c7a662710ad046ae0f866e22c18e4d.jpg](../icml_results/4639_TPGDiff%20_%20Hierarchical%20Triple-Prior%20Guided%20Diffusion%20for%20Image%20Restoration/tables/b4e9b643d8dc2f4d5a48ffe9a92d150b23c7a662710ad046ae0f866e22c18e4d.jpg)

![bcdf27370c0f123140dc61d5dd841ba1ac9941bcd7e7be2527acf99320b4d9d4.jpg](../icml_results/4639_TPGDiff%20_%20Hierarchical%20Triple-Prior%20Guided%20Diffusion%20for%20Image%20Restoration/tables/bcdf27370c0f123140dc61d5dd841ba1ac9941bcd7e7be2527acf99320b4d9d4.jpg)

![c5f3a5fd809cd460aacb4b32e5077dc8ba9d3054f785a0e1624edfb06a432afb.jpg](../icml_results/4639_TPGDiff%20_%20Hierarchical%20Triple-Prior%20Guided%20Diffusion%20for%20Image%20Restoration/tables/c5f3a5fd809cd460aacb4b32e5077dc8ba9d3054f785a0e1624edfb06a432afb.jpg)

![d25f4a9c25e5fecd68177dbdcd778ec698f922fb9dbe14fbea5ffcf956d6e519.jpg](../icml_results/4639_TPGDiff%20_%20Hierarchical%20Triple-Prior%20Guided%20Diffusion%20for%20Image%20Restoration/tables/d25f4a9c25e5fecd68177dbdcd778ec698f922fb9dbe14fbea5ffcf956d6e519.jpg)

<a id="paper-4593"></a>

## Structure-Induced Information for Rerooting Levin Tree Search


### Images

![2c0152dd774331687b969dcc7cb48961b8e8c3152831e336b88095a94e1224be.jpg](../icml_results/4640_Structure-Induced%20Information%20for%20Rerooting%20Levin%20Tree%20Search/images/2c0152dd774331687b969dcc7cb48961b8e8c3152831e336b88095a94e1224be.jpg)

![3fea4f3252aefa6a56ca1a5991a25a41a611b1f655089d6106003de4babaf3bb.jpg](../icml_results/4640_Structure-Induced%20Information%20for%20Rerooting%20Levin%20Tree%20Search/images/3fea4f3252aefa6a56ca1a5991a25a41a611b1f655089d6106003de4babaf3bb.jpg)

### Tables

![24cf8752a89356e29c06a5b26e9c801fa16b087334d0c2a9e13a4ba77fd093b7.jpg](../icml_results/4640_Structure-Induced%20Information%20for%20Rerooting%20Levin%20Tree%20Search/tables/24cf8752a89356e29c06a5b26e9c801fa16b087334d0c2a9e13a4ba77fd093b7.jpg)

![2d0f11fecdde5cadd1891915123acbb235893beadcf0ddb5ca4402e760d5a397.jpg](../icml_results/4640_Structure-Induced%20Information%20for%20Rerooting%20Levin%20Tree%20Search/tables/2d0f11fecdde5cadd1891915123acbb235893beadcf0ddb5ca4402e760d5a397.jpg)

![3c0db3eec773afe3b0ead1f4569799fdcb38dc729e43c2f3cbd04c20416c4f0a.jpg](../icml_results/4640_Structure-Induced%20Information%20for%20Rerooting%20Levin%20Tree%20Search/tables/3c0db3eec773afe3b0ead1f4569799fdcb38dc729e43c2f3cbd04c20416c4f0a.jpg)

![4e741e3379f52b139b0b34016201c07b1f799b279d591ae61922051bc95e5435.jpg](../icml_results/4640_Structure-Induced%20Information%20for%20Rerooting%20Levin%20Tree%20Search/tables/4e741e3379f52b139b0b34016201c07b1f799b279d591ae61922051bc95e5435.jpg)

![5e5533ce5c5ef93657e1ee8312190fdde7cd37ed2fa34375b6e8c773935ae806.jpg](../icml_results/4640_Structure-Induced%20Information%20for%20Rerooting%20Levin%20Tree%20Search/tables/5e5533ce5c5ef93657e1ee8312190fdde7cd37ed2fa34375b6e8c773935ae806.jpg)

![92c2a5c286973e10adf9fe25e84faf45f5ef9f0f5ab9ebb0b605388d10b6f40a.jpg](../icml_results/4640_Structure-Induced%20Information%20for%20Rerooting%20Levin%20Tree%20Search/tables/92c2a5c286973e10adf9fe25e84faf45f5ef9f0f5ab9ebb0b605388d10b6f40a.jpg)

![c0977f5f6f25ab21e99dc0bbd495117aed48086b804886ae1625e477863d490e.jpg](../icml_results/4640_Structure-Induced%20Information%20for%20Rerooting%20Levin%20Tree%20Search/tables/c0977f5f6f25ab21e99dc0bbd495117aed48086b804886ae1625e477863d490e.jpg)

![c68f4a75f9ff1ceedef91126310903cfb377828c6b992f522a0f66919e7122b0.jpg](../icml_results/4640_Structure-Induced%20Information%20for%20Rerooting%20Levin%20Tree%20Search/tables/c68f4a75f9ff1ceedef91126310903cfb377828c6b992f522a0f66919e7122b0.jpg)

<a id="paper-4594"></a>

## Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences


### Images

![0fcca575cb988ce355113a5811c0fbd68c7ff8d51642c96ecc21fe8bdab64d48.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/images/0fcca575cb988ce355113a5811c0fbd68c7ff8d51642c96ecc21fe8bdab64d48.jpg)

![11fbdb9d39476e11cd9cf69698c9ce256ce898d06d5e46d25018e2c333ea40a4.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/images/11fbdb9d39476e11cd9cf69698c9ce256ce898d06d5e46d25018e2c333ea40a4.jpg)

![39b8ed7461a56d5d6f418d6c4da28b4f91900fa81dc7cd08b9a4557523823e02.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/images/39b8ed7461a56d5d6f418d6c4da28b4f91900fa81dc7cd08b9a4557523823e02.jpg)

![39e77c60c5084882aa6de799ba91a7d3f7ed729cf2bcffa231b96b91171c742e.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/images/39e77c60c5084882aa6de799ba91a7d3f7ed729cf2bcffa231b96b91171c742e.jpg)

![3c2b4ac6485243c8dd6e34bc99c5b00c272afa48804100444571a4d99fc13162.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/images/3c2b4ac6485243c8dd6e34bc99c5b00c272afa48804100444571a4d99fc13162.jpg)

![4b56295d4fb387b2c03e2cb4240ae7c4fa017a61d3d6787692c911e65e6707d2.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/images/4b56295d4fb387b2c03e2cb4240ae7c4fa017a61d3d6787692c911e65e6707d2.jpg)

![54b7e3089b5fc3c99e9cb11713c0c67d5a29f4d83cacc20874320cebb4da8893.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/images/54b7e3089b5fc3c99e9cb11713c0c67d5a29f4d83cacc20874320cebb4da8893.jpg)

![55745f0e098bb67b4b7050c44af1ec6b1831b45e0ce3a96d141233e59ff9420e.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/images/55745f0e098bb67b4b7050c44af1ec6b1831b45e0ce3a96d141233e59ff9420e.jpg)

![7a6b51d91070cc96753b17664bb2a12a5e2b807c013204028adef0405dfdb083.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/images/7a6b51d91070cc96753b17664bb2a12a5e2b807c013204028adef0405dfdb083.jpg)

![9abb1276457d11d925a943a0d1d19ed15f0cc5b757dce8f35eebcc0528040547.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/images/9abb1276457d11d925a943a0d1d19ed15f0cc5b757dce8f35eebcc0528040547.jpg)

![ac1badb2406de1cec65a0cfe9c1b32bb5e6a7dd2d047d1530168c811a2ccabb2.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/images/ac1badb2406de1cec65a0cfe9c1b32bb5e6a7dd2d047d1530168c811a2ccabb2.jpg)

![b7168e8fe9f7c7463081579d0b2db94747cafc638c9f6cb61c598249e5e17462.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/images/b7168e8fe9f7c7463081579d0b2db94747cafc638c9f6cb61c598249e5e17462.jpg)

![ba5a6ea3aa93ccfebb6570c40b5f3c446a620dc84acb9541f94d7f5e26cc8210.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/images/ba5a6ea3aa93ccfebb6570c40b5f3c446a620dc84acb9541f94d7f5e26cc8210.jpg)

![d21fe9394626b1441999154ad3177346e2739027cea5ca1e53cd6a56023a5dbf.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/images/d21fe9394626b1441999154ad3177346e2739027cea5ca1e53cd6a56023a5dbf.jpg)

![f263df2104ab8079afce14df33ca799a7d6155ba50c16faa0e335c519055bdab.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/images/f263df2104ab8079afce14df33ca799a7d6155ba50c16faa0e335c519055bdab.jpg)

![f773d4bd0cccbc39d11342fcd3f1e3da1857075431596dc23d2d16a0cc827b7a.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/images/f773d4bd0cccbc39d11342fcd3f1e3da1857075431596dc23d2d16a0cc827b7a.jpg)

### Tables

![4cac5821d6496aa4d5a3acca64b4b27090d99a1ba3c9503472700d5e09b1a566.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/tables/4cac5821d6496aa4d5a3acca64b4b27090d99a1ba3c9503472700d5e09b1a566.jpg)

![64864d61886f7e8a226678c34fc714d9f80339b4815595f751a9b05447522367.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/tables/64864d61886f7e8a226678c34fc714d9f80339b4815595f751a9b05447522367.jpg)

![71039ace78089864a08b0f3f346c09f293870affbe20d5bac002fdf95ce72a84.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/tables/71039ace78089864a08b0f3f346c09f293870affbe20d5bac002fdf95ce72a84.jpg)

![72d01d7c91ba8fae84c047002fb2403631534ad898f529ceb97f92bfd536f0a7.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/tables/72d01d7c91ba8fae84c047002fb2403631534ad898f529ceb97f92bfd536f0a7.jpg)

![8777e29fcd4719484c41bfc67a0be7511614d1dcb4df9f9923bbf2dbcf95662d.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/tables/8777e29fcd4719484c41bfc67a0be7511614d1dcb4df9f9923bbf2dbcf95662d.jpg)

![95e83619bc1ed4c480e0c2b52e4de56b8bcd16aa79d247c0f2a80017c9c5921a.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/tables/95e83619bc1ed4c480e0c2b52e4de56b8bcd16aa79d247c0f2a80017c9c5921a.jpg)

![9dfce6050d2abcb939528cf345ec0dedf84789753ab1e98a968769280df71c7d.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/tables/9dfce6050d2abcb939528cf345ec0dedf84789753ab1e98a968769280df71c7d.jpg)

![a0f6fcd0e163de436e10a80b531f4fe6e64521b4ae3f23b3087345eed65145d5.jpg](../icml_results/4641_Preconditioned%20DeltaNet_%20Curvature-aware%20Sequence%20Modeling%20for%20Linear%20Recurrences/tables/a0f6fcd0e163de436e10a80b531f4fe6e64521b4ae3f23b3087345eed65145d5.jpg)

<a id="paper-4595"></a>

## SparseSSM: Efficient Selective Structured State Space Models Can Be Pruned in One-Shot


### Images

![27550a794beeb0479c69a753af5c5aa4858789fc237781061578a97c817934c6.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/images/27550a794beeb0479c69a753af5c5aa4858789fc237781061578a97c817934c6.jpg)

![5e763cb82486a09055a869251d80b69cd60a5cf80321a19cda2993e236d1f349.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/images/5e763cb82486a09055a869251d80b69cd60a5cf80321a19cda2993e236d1f349.jpg)

![80ee0f88fcb6866d5a4f365830d36f8d9b01877d977461e3318bf25b8ba99d73.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/images/80ee0f88fcb6866d5a4f365830d36f8d9b01877d977461e3318bf25b8ba99d73.jpg)

![949ddcf3e9d31a78bc5ef507eed7a30f7e75490149b79471236b49994fb8955b.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/images/949ddcf3e9d31a78bc5ef507eed7a30f7e75490149b79471236b49994fb8955b.jpg)

![e1feda2d928b0b397ac09c8ed20dd79464ce67e352cbc7b18c16d1f277a75d92.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/images/e1feda2d928b0b397ac09c8ed20dd79464ce67e352cbc7b18c16d1f277a75d92.jpg)

### Tables

![089da5a69bdf9da4aac5cfce3f1c8dd3070cd361804015205c65f9198585a31e.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/089da5a69bdf9da4aac5cfce3f1c8dd3070cd361804015205c65f9198585a31e.jpg)

![199942e5c287b9ee73764808fff51466982f98e6e7dc757b31f3df4765c391b6.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/199942e5c287b9ee73764808fff51466982f98e6e7dc757b31f3df4765c391b6.jpg)

![33c0fd53a6151393f2c20e16976ec17f7324ce804cde2f37fa336f49b8a63ba3.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/33c0fd53a6151393f2c20e16976ec17f7324ce804cde2f37fa336f49b8a63ba3.jpg)

![38340e4de92d9d584b05fac851bf07501644e5060eb620906ee1e20ecf4d56e0.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/38340e4de92d9d584b05fac851bf07501644e5060eb620906ee1e20ecf4d56e0.jpg)

![419f166d5a7324d9c98a4e0f401b323351f8c3e757829953190cf631207213b4.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/419f166d5a7324d9c98a4e0f401b323351f8c3e757829953190cf631207213b4.jpg)

![432023ea85076dc42ffb91fc4e8ce979c03983e148d2f18ba4770373122e1f18.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/432023ea85076dc42ffb91fc4e8ce979c03983e148d2f18ba4770373122e1f18.jpg)

![43d2409c086f86faae954851b612954af186d0755fed582f9c25b40e708dc4bb.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/43d2409c086f86faae954851b612954af186d0755fed582f9c25b40e708dc4bb.jpg)

![4baa37ca33f5ac349f5dc5607e4fbae8050495a892a4b4a8d15155e124811003.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/4baa37ca33f5ac349f5dc5607e4fbae8050495a892a4b4a8d15155e124811003.jpg)

![5d05c7051e33ae4f2d42a0bef0c2ec7182707ec6028b321a40422b282f96781d.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/5d05c7051e33ae4f2d42a0bef0c2ec7182707ec6028b321a40422b282f96781d.jpg)

![6b4cbc9e373677b37235a53cc5d1c244f26d65d74ae3affdadd8235148b4e40d.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/6b4cbc9e373677b37235a53cc5d1c244f26d65d74ae3affdadd8235148b4e40d.jpg)

![93439c49170a1cb3d8b2ccba0c54a14b6baee176d7d5ab4237d4fe258bcb8e9a.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/93439c49170a1cb3d8b2ccba0c54a14b6baee176d7d5ab4237d4fe258bcb8e9a.jpg)

![9a10f6f35569b7dada4fa651866857031af119056f4e5395738000373442be65.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/9a10f6f35569b7dada4fa651866857031af119056f4e5395738000373442be65.jpg)

![a92de13ded418c0e48e67819429c347c0eee9aa8dc657605c39aa285af155e02.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/a92de13ded418c0e48e67819429c347c0eee9aa8dc657605c39aa285af155e02.jpg)

![b17b542aa9e8a6016a263264cb872c0e191d36b8381aa9a2b36234cf55ace89b.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/b17b542aa9e8a6016a263264cb872c0e191d36b8381aa9a2b36234cf55ace89b.jpg)

![b9bbb6a66bbd5aedf74d1eafe46302169a579438e1caeed6b666fedf8bbd391a.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/b9bbb6a66bbd5aedf74d1eafe46302169a579438e1caeed6b666fedf8bbd391a.jpg)

![d0d30ef9093922ae871938ae5801cfebc14b9a47661b2e88ef23bd3b2ceba252.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/d0d30ef9093922ae871938ae5801cfebc14b9a47661b2e88ef23bd3b2ceba252.jpg)

![d91b9c68a7cc4f07a453e5fdd981ee751aae148bd7f5ba6c2496f95f2391276f.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/d91b9c68a7cc4f07a453e5fdd981ee751aae148bd7f5ba6c2496f95f2391276f.jpg)

![dbffc8a29bf33456270ab5827c8b7fb708c3d8eae05c4aa86f90e1172b9b1e93.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/dbffc8a29bf33456270ab5827c8b7fb708c3d8eae05c4aa86f90e1172b9b1e93.jpg)

![e1e09e0fced125f59fca4135fd23306881b25b74465c2d7678cdc597689393ed.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/e1e09e0fced125f59fca4135fd23306881b25b74465c2d7678cdc597689393ed.jpg)

![e32d8142cf89a414af64a695330e13de6e523af9d247c4fced36779441f921a6.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/e32d8142cf89a414af64a695330e13de6e523af9d247c4fced36779441f921a6.jpg)

![ecddf0f1896f1f9ea4d98ae7569a60ea5ccb785e5ad0966467f1a8a60c1d3ecf.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/ecddf0f1896f1f9ea4d98ae7569a60ea5ccb785e5ad0966467f1a8a60c1d3ecf.jpg)

![f4dd60f5b8e7346809f8dde62dfd6e766825f78c9e2e3ece528de444bac68f18.jpg](../icml_results/4642_SparseSSM_%20Efficient%20Selective%20Structured%20State%20Space%20Models%20Can%20Be%20Pruned%20in%20One-Shot/tables/f4dd60f5b8e7346809f8dde62dfd6e766825f78c9e2e3ece528de444bac68f18.jpg)

<a id="paper-4596"></a>

## Unfolding Generative Flows with Koopman Operators: Trajectory-Preserving Linearization


### Images

![0a9b3e7ba0228393c007266fa1814ef43e0e2ed6a037aa05f729f4c7080baa21.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/0a9b3e7ba0228393c007266fa1814ef43e0e2ed6a037aa05f729f4c7080baa21.jpg)

![15372676e23e66c1dae001b440ae9980d63346065f57192cb207acfcfe9a252b.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/15372676e23e66c1dae001b440ae9980d63346065f57192cb207acfcfe9a252b.jpg)

![1554228eab8dfff32d2a49a7ccbc5c3fcb1f6a863981ca209fd681cba84d3140.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/1554228eab8dfff32d2a49a7ccbc5c3fcb1f6a863981ca209fd681cba84d3140.jpg)

![1c12ad0fc30d583c60c26388468c229d73e530ae7072a224f895f3a48e4edeb0.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/1c12ad0fc30d583c60c26388468c229d73e530ae7072a224f895f3a48e4edeb0.jpg)

![204983ab8e605e79a592d2c412611184385be5f69b843962c4025a76ca34e5b5.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/204983ab8e605e79a592d2c412611184385be5f69b843962c4025a76ca34e5b5.jpg)

![2cc4b37bf5c1f770798b93dc8b78b913a538e36fc4a52ff7d0495ebaafb06e5a.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/2cc4b37bf5c1f770798b93dc8b78b913a538e36fc4a52ff7d0495ebaafb06e5a.jpg)

![402c53884e05e690e7b5503971cd32c312e76ac928094efe35175ef23d7bfbe1.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/402c53884e05e690e7b5503971cd32c312e76ac928094efe35175ef23d7bfbe1.jpg)

![5615103d1bff24fb25f256f355f46eea584814627ead5d680b50116aa9c9fbf2.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/5615103d1bff24fb25f256f355f46eea584814627ead5d680b50116aa9c9fbf2.jpg)

![5a8cd448f5b66b2f533baf070a2ac62b77c5be3e44bdee84c342081713cfa7e8.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/5a8cd448f5b66b2f533baf070a2ac62b77c5be3e44bdee84c342081713cfa7e8.jpg)

![636f7d615918e0c1f300a8f3975e5b526d361cf89c98782a61d76aaf3374e32a.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/636f7d615918e0c1f300a8f3975e5b526d361cf89c98782a61d76aaf3374e32a.jpg)

![6bfdf69a2d220e8748d736798f02c88651d6dbd077d0b4ac1df75127cb6e3c58.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/6bfdf69a2d220e8748d736798f02c88651d6dbd077d0b4ac1df75127cb6e3c58.jpg)

![7c4baa8319dc71253c52f65880643d9e014efc7091b6c6c72f69ab6dc0816ee5.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/7c4baa8319dc71253c52f65880643d9e014efc7091b6c6c72f69ab6dc0816ee5.jpg)

![7e0a5176336c2be61fa59951b74a40cd6bcad5aae7a92440d76d1bf3cf2d302f.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/7e0a5176336c2be61fa59951b74a40cd6bcad5aae7a92440d76d1bf3cf2d302f.jpg)

![96ba7a3181b7a7abe4f45fc30bdca6d60444fc94030e491b97282dce1143630f.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/96ba7a3181b7a7abe4f45fc30bdca6d60444fc94030e491b97282dce1143630f.jpg)

![99cd0208c8da16f53a3c0d23dd10ce8552d6aed6951ff2566afe545417eb8fcf.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/99cd0208c8da16f53a3c0d23dd10ce8552d6aed6951ff2566afe545417eb8fcf.jpg)

![9ca1a39477eff2837996226877b02b6dcd70a9c9cd01c8ced8868ca1aa164abb.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/9ca1a39477eff2837996226877b02b6dcd70a9c9cd01c8ced8868ca1aa164abb.jpg)

![ad148eb3ff6788cfaadd0a44a6626e9d7451396496486ef00d0e04f21618f4d8.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/ad148eb3ff6788cfaadd0a44a6626e9d7451396496486ef00d0e04f21618f4d8.jpg)

![ca9a801bd0e4e1c31347f75f676c98f35425286606f4cf31cea4e2e92ad6b8c9.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/ca9a801bd0e4e1c31347f75f676c98f35425286606f4cf31cea4e2e92ad6b8c9.jpg)

![d23eb6d3e14af422c320b9a864391197f6a006a4e27d957ded074bec6518800a.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/d23eb6d3e14af422c320b9a864391197f6a006a4e27d957ded074bec6518800a.jpg)

![deaab20346f5438d4eafdc8cff27bd10255d09fb9b544de9aaba3be88b0d3330.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/deaab20346f5438d4eafdc8cff27bd10255d09fb9b544de9aaba3be88b0d3330.jpg)

![ea5964c35686ed8bbe72d49990cb2a280470a7532cf7a0386f421c15935dda7b.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/ea5964c35686ed8bbe72d49990cb2a280470a7532cf7a0386f421c15935dda7b.jpg)

![ef4a96b44067c6a3819c41d7e175239a88b9317cb734b90c7a8ccadfcf80a006.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/ef4a96b44067c6a3819c41d7e175239a88b9317cb734b90c7a8ccadfcf80a006.jpg)

![ef8a97b2d9a3097dc4980f7f3f49707d7075bd1a6b22d81976e8df35c09aaf9a.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/ef8a97b2d9a3097dc4980f7f3f49707d7075bd1a6b22d81976e8df35c09aaf9a.jpg)

![fe4e38793a362cb13bf7c863fe20f46193e02e4567bec99f4a7b5ef38f8313a9.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/images/fe4e38793a362cb13bf7c863fe20f46193e02e4567bec99f4a7b5ef38f8313a9.jpg)

### Tables

![175c667646cd77f5b8e289596bc4fe813b3c520f711f1963d573ddfa4d6c8961.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/tables/175c667646cd77f5b8e289596bc4fe813b3c520f711f1963d573ddfa4d6c8961.jpg)

![25d1b1458355678f03ea1f355c8b0f17a8d9072fd88705133d735e270f014e77.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/tables/25d1b1458355678f03ea1f355c8b0f17a8d9072fd88705133d735e270f014e77.jpg)

![2cb3c22f2413d28763af67940073eee8cc77ae3efd4b1413195f6096adaca612.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/tables/2cb3c22f2413d28763af67940073eee8cc77ae3efd4b1413195f6096adaca612.jpg)

![2d4db7e57ae8d94f031017b1d8d06283129efac1a7a1013ef694f427ec3c47b2.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/tables/2d4db7e57ae8d94f031017b1d8d06283129efac1a7a1013ef694f427ec3c47b2.jpg)

![4bf99844a74a5c61bd2958a3e1469e02510a9138403ef38f91dd6272dd1fd23a.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/tables/4bf99844a74a5c61bd2958a3e1469e02510a9138403ef38f91dd6272dd1fd23a.jpg)

![4c780df664cf981094d17c41daf1786fe622d8ae11d55045adb77200de71b64e.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/tables/4c780df664cf981094d17c41daf1786fe622d8ae11d55045adb77200de71b64e.jpg)

![99e2438fbad718477a1fa2942863f17097851b20da42e8e242946b61840d8297.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/tables/99e2438fbad718477a1fa2942863f17097851b20da42e8e242946b61840d8297.jpg)

![ad1a40d0c3cd1fb860c61501a2e244154ff076108cfa0a62162dd6ecd5437f87.jpg](../icml_results/4643_Unfolding%20Generative%20Flows%20with%20Koopman%20Operators_%20Trajectory-Preserving%20Linearization/tables/ad1a40d0c3cd1fb860c61501a2e244154ff076108cfa0a62162dd6ecd5437f87.jpg)

<a id="paper-4597"></a>

## Greedy Coordinate Diffusion: Effective and Semantically Coherent Adversarial Attacks via Diffusion Guidance


### Images

![4c53264d5debc213bcc343c1b330a88a29197130ad61f4819ca32ff2a4c9dbac.jpg](../icml_results/4644_Greedy%20Coordinate%20Diffusion_%20Effective%20and%20Semantically%20Coherent%20Adversarial%20Attacks%20via%20Diffusion%20G/images/4c53264d5debc213bcc343c1b330a88a29197130ad61f4819ca32ff2a4c9dbac.jpg)

### Tables

![5c796f054e8ea5fec6943869335f8d783f86400677fef1eae0b7ee7ae2e3d34f.jpg](../icml_results/4644_Greedy%20Coordinate%20Diffusion_%20Effective%20and%20Semantically%20Coherent%20Adversarial%20Attacks%20via%20Diffusion%20G/tables/5c796f054e8ea5fec6943869335f8d783f86400677fef1eae0b7ee7ae2e3d34f.jpg)

![72e86102133bfea5a62587a93b1748301088590438d0f6010886fa93a85c273b.jpg](../icml_results/4644_Greedy%20Coordinate%20Diffusion_%20Effective%20and%20Semantically%20Coherent%20Adversarial%20Attacks%20via%20Diffusion%20G/tables/72e86102133bfea5a62587a93b1748301088590438d0f6010886fa93a85c273b.jpg)

<a id="paper-4598"></a>

## SCOUT: Cyclic Causal Discovery Under Soft Interventions with Unknown Targets


### Images

![0988600509501570c136a385cf6c25dd34b30cae5007b6ab5bd421488676be91.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/0988600509501570c136a385cf6c25dd34b30cae5007b6ab5bd421488676be91.jpg)

![28d0a14f062fb1078bb5e6d1c04acb2940f3ea7ab0b8c0f43cb48e9bb6ee66e0.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/28d0a14f062fb1078bb5e6d1c04acb2940f3ea7ab0b8c0f43cb48e9bb6ee66e0.jpg)

![3c0ef6b186fa2ad1d5ef6fd2c55378ab792308fcc61e7d10067db60e1e029b74.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/3c0ef6b186fa2ad1d5ef6fd2c55378ab792308fcc61e7d10067db60e1e029b74.jpg)

![60ff4f7095293483024c9442a5c9969d44d89392a82c8556e1313008478312b9.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/60ff4f7095293483024c9442a5c9969d44d89392a82c8556e1313008478312b9.jpg)

![647ac6b3a15edba53eb4c5dc9cf9ea7ad39169b02eac49bd402d88d973533057.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/647ac6b3a15edba53eb4c5dc9cf9ea7ad39169b02eac49bd402d88d973533057.jpg)

![6b8c5754fd7020a8235e9b481cc3618f7921eba89158138c0ba97fed0bd2ccb9.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/6b8c5754fd7020a8235e9b481cc3618f7921eba89158138c0ba97fed0bd2ccb9.jpg)

![6bc5e1ff04ec2edaae84c7c83532b1042443e62da3539903afa271af49bc89c9.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/6bc5e1ff04ec2edaae84c7c83532b1042443e62da3539903afa271af49bc89c9.jpg)

![6fefe93743967a2be5f0b127a7b790c336e128bcb40c87681eac8df398e7f2e4.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/6fefe93743967a2be5f0b127a7b790c336e128bcb40c87681eac8df398e7f2e4.jpg)

![89d08c1b60f4234e91c0384e8f1340fe97797de4b229c989a2aae4210972e947.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/89d08c1b60f4234e91c0384e8f1340fe97797de4b229c989a2aae4210972e947.jpg)

![9b9aab80bbbd7b940bb52a86a3343c13d44826e90c26ccdbac947bea20648d22.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/9b9aab80bbbd7b940bb52a86a3343c13d44826e90c26ccdbac947bea20648d22.jpg)

![9cac2962da97114456a2436ea182a8a045cd286d3fde12e92fc08fd2052b67cb.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/9cac2962da97114456a2436ea182a8a045cd286d3fde12e92fc08fd2052b67cb.jpg)

![a2df9dcf9cc0e600831a9fd10d2bdc2c44e8275fc26e471c3450270187802bcb.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/a2df9dcf9cc0e600831a9fd10d2bdc2c44e8275fc26e471c3450270187802bcb.jpg)

![a52d502e378028e4000730dcd998352f6b41a4c1f642d5da37fc5f81d1fa1912.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/a52d502e378028e4000730dcd998352f6b41a4c1f642d5da37fc5f81d1fa1912.jpg)

![c6d949ebba9c9bb7ecefb2a2d231d983af736666f4217a9028ecfcf5ed7a427d.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/c6d949ebba9c9bb7ecefb2a2d231d983af736666f4217a9028ecfcf5ed7a427d.jpg)

![cebff6b27a832d156f3f5184b79325f9c7a2e78f45585e419d1371d64cb7e6e8.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/cebff6b27a832d156f3f5184b79325f9c7a2e78f45585e419d1371d64cb7e6e8.jpg)

![d28d9bae3f19b991ced76fd9920373af4ab913f09de7d150448388e903dcd21b.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/d28d9bae3f19b991ced76fd9920373af4ab913f09de7d150448388e903dcd21b.jpg)

![d5b9d1626317524e53201dd168b248c6892071116ef7cd008c92ed904a25e3e2.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/d5b9d1626317524e53201dd168b248c6892071116ef7cd008c92ed904a25e3e2.jpg)

![df0806ef91df3e873d6bb5fdfd98d8121ce756988afbcff5e6ba960a56006d68.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/df0806ef91df3e873d6bb5fdfd98d8121ce756988afbcff5e6ba960a56006d68.jpg)

![f30d51961c2c91a0a4062587ee9c9c538c8f66750bbac915d01c0cceb62c7b0c.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/f30d51961c2c91a0a4062587ee9c9c538c8f66750bbac915d01c0cceb62c7b0c.jpg)

![f90ee3bb5e834fdf4cb65bb0e3ec9ff3622c425dee26612c8df436de88689f92.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/f90ee3bb5e834fdf4cb65bb0e3ec9ff3622c425dee26612c8df436de88689f92.jpg)

![fbd2c9ea921013b32975310ee36b14dd9df7c9031382ed481113e05ca60dc9a4.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/fbd2c9ea921013b32975310ee36b14dd9df7c9031382ed481113e05ca60dc9a4.jpg)

![fd8777c56d56aaebaebd5da0fab615e764a429acf89b4d26250717145f6ac44b.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/fd8777c56d56aaebaebd5da0fab615e764a429acf89b4d26250717145f6ac44b.jpg)

![fe636a23b28504961468278d034e2c95c8f726648fcc8093169c3953144fca9d.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/fe636a23b28504961468278d034e2c95c8f726648fcc8093169c3953144fca9d.jpg)

![fff0e7d9591e562c4f846d24db643c9deb4097558f4b2c130db9acb0e5eadb3c.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/images/fff0e7d9591e562c4f846d24db643c9deb4097558f4b2c130db9acb0e5eadb3c.jpg)

### Tables

![009bfbac6bdd058d9250be7fbec24546a5840bfc65eb0a5ea21831f0c77f572c.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/tables/009bfbac6bdd058d9250be7fbec24546a5840bfc65eb0a5ea21831f0c77f572c.jpg)

![07178751fb8151e073de6a30ec3b2aebc0182f5ba7579669cc2ee07747202d5b.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/tables/07178751fb8151e073de6a30ec3b2aebc0182f5ba7579669cc2ee07747202d5b.jpg)

![10ccd3eafcee0adfa5bc93186fcff58cf9dd519150a4e2da5b1e53e646cd636c.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/tables/10ccd3eafcee0adfa5bc93186fcff58cf9dd519150a4e2da5b1e53e646cd636c.jpg)

![1192080a44c6d57b8cc6fa03a47e4ea1232ee0e1a44edc999380cfcfbffe0a11.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/tables/1192080a44c6d57b8cc6fa03a47e4ea1232ee0e1a44edc999380cfcfbffe0a11.jpg)

![17eac7fd811fe0139440e65c4205692d0a0a8509568b6eed035cd11d3ced88bc.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/tables/17eac7fd811fe0139440e65c4205692d0a0a8509568b6eed035cd11d3ced88bc.jpg)

![26764449dcfea41f9eda2b0e2b1133a4244cc51ade613c6f30ca71b96fbef085.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/tables/26764449dcfea41f9eda2b0e2b1133a4244cc51ade613c6f30ca71b96fbef085.jpg)

![3aa7cc592ae7995aed7d04fe080de2faf2d146f250350f5e907edc848158551d.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/tables/3aa7cc592ae7995aed7d04fe080de2faf2d146f250350f5e907edc848158551d.jpg)

![3cea3ed6e86298b47ce949d7871fc6b1f74a33156c71d9fc438a33da23863b56.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/tables/3cea3ed6e86298b47ce949d7871fc6b1f74a33156c71d9fc438a33da23863b56.jpg)

![4c882e76fdb9cfe202f99d17ae0218a0275bd799c0a19188ee38bf6fd096e417.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/tables/4c882e76fdb9cfe202f99d17ae0218a0275bd799c0a19188ee38bf6fd096e417.jpg)

![6fabb399f673587aaa38a1502000ca33f6b5bfb6328cc7e5e2f028c5d742ef46.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/tables/6fabb399f673587aaa38a1502000ca33f6b5bfb6328cc7e5e2f028c5d742ef46.jpg)

![94b1de25561192eb2b808cb727ba45c68e6b60b8d7e8bd30b90a0b6723056206.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/tables/94b1de25561192eb2b808cb727ba45c68e6b60b8d7e8bd30b90a0b6723056206.jpg)

![9dd99eaf9d54d4f15564488061c291d2583d2d6cc9f736f75524fc0db1da2de0.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/tables/9dd99eaf9d54d4f15564488061c291d2583d2d6cc9f736f75524fc0db1da2de0.jpg)

![a21ca621589e2293ea2761c4b7bf9e11586e289a62ef34e97f40b44b74146083.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/tables/a21ca621589e2293ea2761c4b7bf9e11586e289a62ef34e97f40b44b74146083.jpg)

![c011b7b0fcc25c5ac0e33fe64e981715f9b043426376de16ac17ada7b83a83e9.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/tables/c011b7b0fcc25c5ac0e33fe64e981715f9b043426376de16ac17ada7b83a83e9.jpg)

![ee9dedb21afa5a5b8feb8b3ee8e91dc093f5e4889f7c077241cb95c7ed82125b.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/tables/ee9dedb21afa5a5b8feb8b3ee8e91dc093f5e4889f7c077241cb95c7ed82125b.jpg)

![f8e1712142aa5b7859e42829fa994857daf4cbccdf7b15f454934f1111156708.jpg](../icml_results/4645_SCOUT_%20Cyclic%20Causal%20Discovery%20Under%20Soft%20Interventions%20with%20Unknown%20Targets/tables/f8e1712142aa5b7859e42829fa994857daf4cbccdf7b15f454934f1111156708.jpg)

<a id="paper-4599"></a>

## Quantifying the Uncertainty of Foundation Models with Singular Value Ensembles


### Images

![220a53132dfa0fc9d7ab02601fd178d6a98cd16ff67fbaac84ec8aa5200f06de.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/images/220a53132dfa0fc9d7ab02601fd178d6a98cd16ff67fbaac84ec8aa5200f06de.jpg)

![259681d4792070123955acd4c38a133b80aa92c84533d63b97fb64967173c341.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/images/259681d4792070123955acd4c38a133b80aa92c84533d63b97fb64967173c341.jpg)

![d049e3a7e302f6b75bb5290676993eda6a519e4eb0859320197d119190f0d71e.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/images/d049e3a7e302f6b75bb5290676993eda6a519e4eb0859320197d119190f0d71e.jpg)

![e666eca5b0919ad3564522a0787dc62b422d877e4b4bab1084a02411c057a5e7.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/images/e666eca5b0919ad3564522a0787dc62b422d877e4b4bab1084a02411c057a5e7.jpg)

![e682c9b49774e873b9bf175e384cf2b53566247acc0172a75867ad9ef5ea54d5.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/images/e682c9b49774e873b9bf175e384cf2b53566247acc0172a75867ad9ef5ea54d5.jpg)

![e7dfd3e6def6949186e866c0c68df6e35216c5d6710fad73489abb93391356e6.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/images/e7dfd3e6def6949186e866c0c68df6e35216c5d6710fad73489abb93391356e6.jpg)

### Tables

![1a0af41ef27990fa9ebac839f1df42639e495bb1052752fb5c32a9f91ad59fa8.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/tables/1a0af41ef27990fa9ebac839f1df42639e495bb1052752fb5c32a9f91ad59fa8.jpg)

![31f4f540d6a615c399f1e5a575afcb03ed52e69ae3a6dae907af91598609a878.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/tables/31f4f540d6a615c399f1e5a575afcb03ed52e69ae3a6dae907af91598609a878.jpg)

![6db6689cae29cec1ce8c7bf7bda01c83c882947c60edb10fdad80dedb921175f.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/tables/6db6689cae29cec1ce8c7bf7bda01c83c882947c60edb10fdad80dedb921175f.jpg)

![74334f9e014e792034c79b1e88445ceba18447478b4f28d883bfe17081c79074.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/tables/74334f9e014e792034c79b1e88445ceba18447478b4f28d883bfe17081c79074.jpg)

![777ab0b626ec95fa8b0cc8c80435822953ae271d150764bc501e1a44663c71b7.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/tables/777ab0b626ec95fa8b0cc8c80435822953ae271d150764bc501e1a44663c71b7.jpg)

![90bb62ea55266b5d2215a4a51e89b48fa1b2b532a54e9fdb0c22786e2d841f6c.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/tables/90bb62ea55266b5d2215a4a51e89b48fa1b2b532a54e9fdb0c22786e2d841f6c.jpg)

![bc717cc845751de8e8d24b154c802a4c02d628af635ad22ee47520c6ab748014.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/tables/bc717cc845751de8e8d24b154c802a4c02d628af635ad22ee47520c6ab748014.jpg)

![cc69fbb7d585733a7d08cd52fd1b9b1eba56c3b2c3b4e1cc401f8053320be57e.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/tables/cc69fbb7d585733a7d08cd52fd1b9b1eba56c3b2c3b4e1cc401f8053320be57e.jpg)

![d428d43110b81b8dd61d3a73ca84613cef3e61fa84d4cc54a43f14d0f7c3487b.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/tables/d428d43110b81b8dd61d3a73ca84613cef3e61fa84d4cc54a43f14d0f7c3487b.jpg)

![d9c2c26d4c0c0cd6d74b7a3ced3e7f34bef2f3124b86c1ec5af92038b9038a38.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/tables/d9c2c26d4c0c0cd6d74b7a3ced3e7f34bef2f3124b86c1ec5af92038b9038a38.jpg)

![ed5818a2af20c4bd910583efb56031c1c8a33f7d9fa20666eefb8c4da2915077.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/tables/ed5818a2af20c4bd910583efb56031c1c8a33f7d9fa20666eefb8c4da2915077.jpg)

![f067344e20391b844c0101cce6495d1e136a4d2821bfaad107372217b9d455c7.jpg](../icml_results/4646_Quantifying%20the%20Uncertainty%20of%20Foundation%20Models%20with%20Singular%20Value%20Ensembles/tables/f067344e20391b844c0101cce6495d1e136a4d2821bfaad107372217b9d455c7.jpg)

<a id="paper-4600"></a>

## Gradient Descent as a Perceptron Algorithm: Understanding Dynamics and Implicit Acceleration


### Images

![0601ef6bed668216ede426ca98b2e634c5b046b3c07aa54ee4e90b4635393c75.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/0601ef6bed668216ede426ca98b2e634c5b046b3c07aa54ee4e90b4635393c75.jpg)

![1829b2dfa9ba2ea0fab500e1580c893d664aefb7e757d16bfb2b71a2eef37f1c.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/1829b2dfa9ba2ea0fab500e1580c893d664aefb7e757d16bfb2b71a2eef37f1c.jpg)

![25e00926d9244a526356d788e03ac26f7af1c436915ccecc6e67cb7eb89ceb33.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/25e00926d9244a526356d788e03ac26f7af1c436915ccecc6e67cb7eb89ceb33.jpg)

![2d694549d70137d8f208405954fe9245b276707f711f7d28f19c162e900de181.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/2d694549d70137d8f208405954fe9245b276707f711f7d28f19c162e900de181.jpg)

![387922e78787d226c6457c5f76e0505dd28961f29f2b001cf99a7bad7d692d55.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/387922e78787d226c6457c5f76e0505dd28961f29f2b001cf99a7bad7d692d55.jpg)

![46bdf33ee6ac405df63786bf3b87e02bd8644c21cd30483863c7813988a72c12.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/46bdf33ee6ac405df63786bf3b87e02bd8644c21cd30483863c7813988a72c12.jpg)

![50af631ac0fce2c68785a346e17a5afa1e370c4b2b56e66be9e175c66aa9d8a0.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/50af631ac0fce2c68785a346e17a5afa1e370c4b2b56e66be9e175c66aa9d8a0.jpg)

![5b88862848565c3c3f3f76e7ea9e5cfe2eb29e0f5da4da09ef3058ceab94ec23.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/5b88862848565c3c3f3f76e7ea9e5cfe2eb29e0f5da4da09ef3058ceab94ec23.jpg)

![64334988e54a1ac2e6b3b7f6bb228d0619a2b52a297efbe8c8e67fd193353075.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/64334988e54a1ac2e6b3b7f6bb228d0619a2b52a297efbe8c8e67fd193353075.jpg)

![6de8a616266c4b84feda7c49c81e1c1337488c7af62d2da47af1f004ef2d536a.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/6de8a616266c4b84feda7c49c81e1c1337488c7af62d2da47af1f004ef2d536a.jpg)

![6e4958189ebd48ee380c56c7d9d97fd4b728c0db1eedf27dfd968ffbac896b4d.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/6e4958189ebd48ee380c56c7d9d97fd4b728c0db1eedf27dfd968ffbac896b4d.jpg)

![72a6b553cd696a06ecb7ff890fd8c750efbbceef28f9d2b973b9f574f73de9dc.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/72a6b553cd696a06ecb7ff890fd8c750efbbceef28f9d2b973b9f574f73de9dc.jpg)

![8216aba8969426e0b385e2ebd4063b570a132c140e9d881a91d0012ac39198a7.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/8216aba8969426e0b385e2ebd4063b570a132c140e9d881a91d0012ac39198a7.jpg)

![82df502bd923f17cac6ca42c5963decc273f75c349b10e9ccbd7b6f45538b078.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/82df502bd923f17cac6ca42c5963decc273f75c349b10e9ccbd7b6f45538b078.jpg)

![833e8782fab6a5e723a8155a943063947d1344514e8afa3a9b6c5a892a826faa.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/833e8782fab6a5e723a8155a943063947d1344514e8afa3a9b6c5a892a826faa.jpg)

![852265eebc98b7b6dc935d4f89feb9390dacba7f52dde26e28305aac3def1ddf.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/852265eebc98b7b6dc935d4f89feb9390dacba7f52dde26e28305aac3def1ddf.jpg)

![87b44bb2bfe534fc75e9a31b13561d9d4b6988a29280c0862ad7c68eef5e5f66.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/87b44bb2bfe534fc75e9a31b13561d9d4b6988a29280c0862ad7c68eef5e5f66.jpg)

![881a140cca4691375abdaefc481d61d27b243df99bd47705472114304a262419.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/881a140cca4691375abdaefc481d61d27b243df99bd47705472114304a262419.jpg)

![99ce9f0e3a935703fbdee772aad2f3db05fae8ad19296e547e8d6f605e44a3f3.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/99ce9f0e3a935703fbdee772aad2f3db05fae8ad19296e547e8d6f605e44a3f3.jpg)

![9bfce065ed66dc481f59f11cf01b7b1b6e332a3999b5ece90d17c2c1c1914b95.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/9bfce065ed66dc481f59f11cf01b7b1b6e332a3999b5ece90d17c2c1c1914b95.jpg)

![bad4196598d266a01141083e7881d648438973a6082e459480597a26d50a5233.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/bad4196598d266a01141083e7881d648438973a6082e459480597a26d50a5233.jpg)

![c9c5741f555aab1b3779291292b7d8133aec536ab427c42793f6facc095584f4.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/c9c5741f555aab1b3779291292b7d8133aec536ab427c42793f6facc095584f4.jpg)

![cb121706860aba7095de637aa199e472ca1e07e1f83bdc2512a7685fa710d510.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/cb121706860aba7095de637aa199e472ca1e07e1f83bdc2512a7685fa710d510.jpg)

![cb170533f571c9a04a9ee099ab8e7ba1b94058368fde17fa0947771ec2f34197.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/cb170533f571c9a04a9ee099ab8e7ba1b94058368fde17fa0947771ec2f34197.jpg)

![cc5b96326cb8186c26fcaffde24a786d87f4984dbbb249874ab1c45f74307429.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/cc5b96326cb8186c26fcaffde24a786d87f4984dbbb249874ab1c45f74307429.jpg)

![ceed7092a3c55f68e27f3b04c3a6545cb9a9fa1fc7f38e6c8b7b030af2069264.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/ceed7092a3c55f68e27f3b04c3a6545cb9a9fa1fc7f38e6c8b7b030af2069264.jpg)

![d583bae92b4bd8f035edb52e396481f882ba2288fb5aa573fea0b10e971563a1.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/d583bae92b4bd8f035edb52e396481f882ba2288fb5aa573fea0b10e971563a1.jpg)

![e276c60f42bb3abed119f6dea2fdf732b98461c3c697969b0d72e08a3ec59800.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/e276c60f42bb3abed119f6dea2fdf732b98461c3c697969b0d72e08a3ec59800.jpg)

![e4d378dba578c0f9eff2751fbeedb185c933dfabd27e0c80eb13d448509f0247.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/e4d378dba578c0f9eff2751fbeedb185c933dfabd27e0c80eb13d448509f0247.jpg)

![efc53be57150306bebfa2105f7b63918c5f449381ae7ed84af6649de12ed3cf3.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/efc53be57150306bebfa2105f7b63918c5f449381ae7ed84af6649de12ed3cf3.jpg)

![f4b7370e5e1bd59a0f82fc307c840e0f7423e62cba3270de3506a7663b272276.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/images/f4b7370e5e1bd59a0f82fc307c840e0f7423e62cba3270de3506a7663b272276.jpg)

### Tables

![6b5a41eaac6cac93c6c8d5e07782b580a0a1a5ab121524fdc594d51a42300c60.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/tables/6b5a41eaac6cac93c6c8d5e07782b580a0a1a5ab121524fdc594d51a42300c60.jpg)

![dc5fed0c8574b88cc8e9e4f1d5b1f496e9940888e8ca7d256f4df279d2daacbe.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/tables/dc5fed0c8574b88cc8e9e4f1d5b1f496e9940888e8ca7d256f4df279d2daacbe.jpg)

![f87ddc4090462543b9029c906ee6d003cd751f9f62f0685abaa24a80622d90fe.jpg](../icml_results/4647_Gradient%20Descent%20as%20a%20Perceptron%20Algorithm_%20Understanding%20Dynamics%20and%20Implicit%20Acceleration/tables/f87ddc4090462543b9029c906ee6d003cd751f9f62f0685abaa24a80622d90fe.jpg)

<a id="paper-4601"></a>

## Near-Optimal Convergence of Accelerated Gradient Methods under Generalized and $(L_0,L_1)$-Smoothness


### Images

![162b3d85f0332a5cf7d00c1130aa9a3e5b8aa02868c17d2a49775702ff643989.jpg](../icml_results/4648_Near-Optimal%20Convergence%20of%20Accelerated%20Gradient%20Methods%20under%20Generalized%20and%20%24%28L_0%2CL_1%29%24-Smoothnes/images/162b3d85f0332a5cf7d00c1130aa9a3e5b8aa02868c17d2a49775702ff643989.jpg)

![462063d15c43ce4de70312d757e4c1d4ae368449a3555fa3fec2953fef2d9b81.jpg](../icml_results/4648_Near-Optimal%20Convergence%20of%20Accelerated%20Gradient%20Methods%20under%20Generalized%20and%20%24%28L_0%2CL_1%29%24-Smoothnes/images/462063d15c43ce4de70312d757e4c1d4ae368449a3555fa3fec2953fef2d9b81.jpg)

![6a9c400dd3cb986d4ca92c61e2374a4a5d35f262a41668974eb48f50380a21bb.jpg](../icml_results/4648_Near-Optimal%20Convergence%20of%20Accelerated%20Gradient%20Methods%20under%20Generalized%20and%20%24%28L_0%2CL_1%29%24-Smoothnes/images/6a9c400dd3cb986d4ca92c61e2374a4a5d35f262a41668974eb48f50380a21bb.jpg)

![97ab54d020ba45286ea5c4dd983ef17e3febac209181e229931012d0da84a85b.jpg](../icml_results/4648_Near-Optimal%20Convergence%20of%20Accelerated%20Gradient%20Methods%20under%20Generalized%20and%20%24%28L_0%2CL_1%29%24-Smoothnes/images/97ab54d020ba45286ea5c4dd983ef17e3febac209181e229931012d0da84a85b.jpg)

![dd962a070219dff1dae09b8e2b0af5e6f8de3bc8ca96801e997d94580482d3ed.jpg](../icml_results/4648_Near-Optimal%20Convergence%20of%20Accelerated%20Gradient%20Methods%20under%20Generalized%20and%20%24%28L_0%2CL_1%29%24-Smoothnes/images/dd962a070219dff1dae09b8e2b0af5e6f8de3bc8ca96801e997d94580482d3ed.jpg)

![eedd3a4ae09a782aa3181ee0ca25c5e711a9b8222ee0209f0a0ea49de31acfcc.jpg](../icml_results/4648_Near-Optimal%20Convergence%20of%20Accelerated%20Gradient%20Methods%20under%20Generalized%20and%20%24%28L_0%2CL_1%29%24-Smoothnes/images/eedd3a4ae09a782aa3181ee0ca25c5e711a9b8222ee0209f0a0ea49de31acfcc.jpg)

### Tables

![a5b498a23b6c41a2f93802bb9ef7fd6a7c2819bb1ad63cec7704a6db5fb5db0f.jpg](../icml_results/4648_Near-Optimal%20Convergence%20of%20Accelerated%20Gradient%20Methods%20under%20Generalized%20and%20%24%28L_0%2CL_1%29%24-Smoothnes/tables/a5b498a23b6c41a2f93802bb9ef7fd6a7c2819bb1ad63cec7704a6db5fb5db0f.jpg)

<a id="paper-4602"></a>

## A Geometric Lens on Physics-Aligned Data Compression


### Images

![2b59c56fb922c24d299ccf61fefbe754dd47c821e3d639696d046d0a2671ef91.jpg](../icml_results/4649_A%20Geometric%20Lens%20on%20Physics-Aligned%20Data%20Compression/images/2b59c56fb922c24d299ccf61fefbe754dd47c821e3d639696d046d0a2671ef91.jpg)

![2e84ba891019eea7db69a0dd43ef7f969e287f1cdaef26627692ae884f0014eb.jpg](../icml_results/4649_A%20Geometric%20Lens%20on%20Physics-Aligned%20Data%20Compression/images/2e84ba891019eea7db69a0dd43ef7f969e287f1cdaef26627692ae884f0014eb.jpg)

![375e95def020382fe4f2b8b6c61c1e871b92af4380fb02ea15c79b44fd476a67.jpg](../icml_results/4649_A%20Geometric%20Lens%20on%20Physics-Aligned%20Data%20Compression/images/375e95def020382fe4f2b8b6c61c1e871b92af4380fb02ea15c79b44fd476a67.jpg)

![4fa973db7f99ab72913040a9f97951c31e3ccfb45e00639bd9693f60f98dee7b.jpg](../icml_results/4649_A%20Geometric%20Lens%20on%20Physics-Aligned%20Data%20Compression/images/4fa973db7f99ab72913040a9f97951c31e3ccfb45e00639bd9693f60f98dee7b.jpg)

![71a4516171f0292d3a94f3441c0b7d44c10d91379d0b222da35acbe03ffcb91d.jpg](../icml_results/4649_A%20Geometric%20Lens%20on%20Physics-Aligned%20Data%20Compression/images/71a4516171f0292d3a94f3441c0b7d44c10d91379d0b222da35acbe03ffcb91d.jpg)

![bcdc98473e117fa4a48df2d4392ae2e58d6d096b006c44842478a7fe7a20e54a.jpg](../icml_results/4649_A%20Geometric%20Lens%20on%20Physics-Aligned%20Data%20Compression/images/bcdc98473e117fa4a48df2d4392ae2e58d6d096b006c44842478a7fe7a20e54a.jpg)

![eedbec8cb9b1a20f0a6d40ce769ef249e35dbd243237c0a3a74893f909187a6d.jpg](../icml_results/4649_A%20Geometric%20Lens%20on%20Physics-Aligned%20Data%20Compression/images/eedbec8cb9b1a20f0a6d40ce769ef249e35dbd243237c0a3a74893f909187a6d.jpg)

![eee689525793c730141ea00fba56c501d41ca1a9bc97c5146b2ebc55b869a1f5.jpg](../icml_results/4649_A%20Geometric%20Lens%20on%20Physics-Aligned%20Data%20Compression/images/eee689525793c730141ea00fba56c501d41ca1a9bc97c5146b2ebc55b869a1f5.jpg)

![f5a9084028c0c87ff020a0be1ad81725f05b57dc0656c26d0e54ff150e331491.jpg](../icml_results/4649_A%20Geometric%20Lens%20on%20Physics-Aligned%20Data%20Compression/images/f5a9084028c0c87ff020a0be1ad81725f05b57dc0656c26d0e54ff150e331491.jpg)

![f61aac29b04fbadaf52772a5ea79817918c90b2ff2145cc7808ec2e3f0947145.jpg](../icml_results/4649_A%20Geometric%20Lens%20on%20Physics-Aligned%20Data%20Compression/images/f61aac29b04fbadaf52772a5ea79817918c90b2ff2145cc7808ec2e3f0947145.jpg)

![f758fd59223663ce09eb9cdadc05e98ae53ed8b9c4109d6edaf5af4c27b6b008.jpg](../icml_results/4649_A%20Geometric%20Lens%20on%20Physics-Aligned%20Data%20Compression/images/f758fd59223663ce09eb9cdadc05e98ae53ed8b9c4109d6edaf5af4c27b6b008.jpg)

![f9d864ba18b18ffb41a697ca375cb9c3798e982f476145f6fb4895ee882b03c9.jpg](../icml_results/4649_A%20Geometric%20Lens%20on%20Physics-Aligned%20Data%20Compression/images/f9d864ba18b18ffb41a697ca375cb9c3798e982f476145f6fb4895ee882b03c9.jpg)

### Tables

![53892fcd42e4713fd95291feb2869e450f887f8a84946520c4fc1acab95ca133.jpg](../icml_results/4649_A%20Geometric%20Lens%20on%20Physics-Aligned%20Data%20Compression/tables/53892fcd42e4713fd95291feb2869e450f887f8a84946520c4fc1acab95ca133.jpg)

![96a7af51c3f3e9fcb668f1f586203de2ccb49e4cdb854caca48acff2205a134c.jpg](../icml_results/4649_A%20Geometric%20Lens%20on%20Physics-Aligned%20Data%20Compression/tables/96a7af51c3f3e9fcb668f1f586203de2ccb49e4cdb854caca48acff2205a134c.jpg)

<a id="paper-4603"></a>

## Efficient DP-SGD for LLMs with Randomized Clipping


### Images

![052346a3a5769f00281f18c23408d0288c7b6f75f74aa63b808b71e61a7b0e39.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/images/052346a3a5769f00281f18c23408d0288c7b6f75f74aa63b808b71e61a7b0e39.jpg)

![2ef18efe59ea4b9c2b11e452384f2346db40477ed60a1c580166f21dce467994.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/images/2ef18efe59ea4b9c2b11e452384f2346db40477ed60a1c580166f21dce467994.jpg)

![36426d4c2290f73cd6d73ab61865be6eb054deda9f2760a1002383fe388ca701.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/images/36426d4c2290f73cd6d73ab61865be6eb054deda9f2760a1002383fe388ca701.jpg)

![4c986c237f26fb0ca7b6c9593a9472f932f6d118d3855ced483ceab5a3b279d8.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/images/4c986c237f26fb0ca7b6c9593a9472f932f6d118d3855ced483ceab5a3b279d8.jpg)

![527191a6e452c02b8dd79d9df077c05958ab02dc9f3e5b71bfc9d172728d0a8b.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/images/527191a6e452c02b8dd79d9df077c05958ab02dc9f3e5b71bfc9d172728d0a8b.jpg)

![63b3b346e20cfbf83ed5a9a25f4569ce88c015aab0ffb0941b5b51ee48c13410.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/images/63b3b346e20cfbf83ed5a9a25f4569ce88c015aab0ffb0941b5b51ee48c13410.jpg)

![6897a4f5021b9f11b4c8f0d66bf4bca38114e360a83770e7256a3522304dbc18.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/images/6897a4f5021b9f11b4c8f0d66bf4bca38114e360a83770e7256a3522304dbc18.jpg)

![92e5452527cffa2a9a3daaed5973b5e61c3cf98fd31d8e78f6fbcde22e9fdac2.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/images/92e5452527cffa2a9a3daaed5973b5e61c3cf98fd31d8e78f6fbcde22e9fdac2.jpg)

![b14e8e4448cb850ca3642170210c58c89592c849547e1c8aa2b7fa2a60de04a6.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/images/b14e8e4448cb850ca3642170210c58c89592c849547e1c8aa2b7fa2a60de04a6.jpg)

![ba199b9fe54acad8afda4cd36a6bee8051ecc50e394fb954d6c65c441863b06b.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/images/ba199b9fe54acad8afda4cd36a6bee8051ecc50e394fb954d6c65c441863b06b.jpg)

![bc5e842a8fb9632084ba4fe129333e00b83aa7ff52429e4f5c312774fc6a9e76.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/images/bc5e842a8fb9632084ba4fe129333e00b83aa7ff52429e4f5c312774fc6a9e76.jpg)

![c5f9a5da01352201d65f866154cefa9e4fdea05cf252a354727249fefed1c542.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/images/c5f9a5da01352201d65f866154cefa9e4fdea05cf252a354727249fefed1c542.jpg)

![dfa80baf760550a4369e9717d7857fc2432d2d3c7a730d1bc3f5ffc3210c6aa2.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/images/dfa80baf760550a4369e9717d7857fc2432d2d3c7a730d1bc3f5ffc3210c6aa2.jpg)

![e295b9ec86a556ff89602c5fde5e5490accfae861ff114eb8816175fc9e83c12.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/images/e295b9ec86a556ff89602c5fde5e5490accfae861ff114eb8816175fc9e83c12.jpg)

### Tables

![23aca3f42dd8138633f023633cdcafc62cc3b965ce20fc5d1437bbf63dbbb405.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/tables/23aca3f42dd8138633f023633cdcafc62cc3b965ce20fc5d1437bbf63dbbb405.jpg)

![3d5f14501fee9aad2be1cf66effda8d2e2fa7ca975969e32133439c9b32aaf3c.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/tables/3d5f14501fee9aad2be1cf66effda8d2e2fa7ca975969e32133439c9b32aaf3c.jpg)

![4060cd8a594827e6b4b6a1acd463ed3d2cd0e6834d85fd2160b2c0d227c3da76.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/tables/4060cd8a594827e6b4b6a1acd463ed3d2cd0e6834d85fd2160b2c0d227c3da76.jpg)

![423d3e52defe470a680ed4a2232b3bbacd7c0b02d46df9c894955a0bbe2040c1.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/tables/423d3e52defe470a680ed4a2232b3bbacd7c0b02d46df9c894955a0bbe2040c1.jpg)

![59838d4a4fba04b4df3805c13324e70c5509215fa73a08bbb6fa99977f40e7a8.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/tables/59838d4a4fba04b4df3805c13324e70c5509215fa73a08bbb6fa99977f40e7a8.jpg)

![60cfeb0524a78f344c7b9ea2a29dfec94c9a334b34de83e21af17fe9b2f10354.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/tables/60cfeb0524a78f344c7b9ea2a29dfec94c9a334b34de83e21af17fe9b2f10354.jpg)

![6441a262d01d73b182422f1428dc2ca428d5338dfed28858c491778711fd8384.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/tables/6441a262d01d73b182422f1428dc2ca428d5338dfed28858c491778711fd8384.jpg)

![76e518a92d16b37e10e0040fb9e9dda69576780a5a91ac69514297dab45c2fad.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/tables/76e518a92d16b37e10e0040fb9e9dda69576780a5a91ac69514297dab45c2fad.jpg)

![7ff6e16811a1fac0e7aff931e4a3226575d9dedc34ac2091a4acf5de8abd7803.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/tables/7ff6e16811a1fac0e7aff931e4a3226575d9dedc34ac2091a4acf5de8abd7803.jpg)

![e64a3a7ba035acfe01337fbd628cf8704d8dffd739cadb018509385051980d94.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/tables/e64a3a7ba035acfe01337fbd628cf8704d8dffd739cadb018509385051980d94.jpg)

![f0d0da97367692c98e155e39dc81e99d1ae10446caad47ac06b9e2135c1e25d9.jpg](../icml_results/4650_Efficient%20DP-SGD%20for%20LLMs%20with%20Randomized%20Clipping/tables/f0d0da97367692c98e155e39dc81e99d1ae10446caad47ac06b9e2135c1e25d9.jpg)

<a id="paper-4604"></a>

## HASTE: Hardware-Aware Dynamic Sparse Training for Large Output Spaces


### Images

![24f16cc5037975b4f0d7f319d3be627f5535451774aa15a1154cf5f8c8c6fd9d.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/images/24f16cc5037975b4f0d7f319d3be627f5535451774aa15a1154cf5f8c8c6fd9d.jpg)

![31cead98b543f03493f8126df940cdfc21eeb4cb26619cff32523cb3111d90ab.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/images/31cead98b543f03493f8126df940cdfc21eeb4cb26619cff32523cb3111d90ab.jpg)

![84de53f7b4ee56b594b3fd79afa828d1df60b833120e70d0121e08a9381e7fce.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/images/84de53f7b4ee56b594b3fd79afa828d1df60b833120e70d0121e08a9381e7fce.jpg)

![9dae0304e4c3347e9b0eced741898059ad2e5dd4c1a3c511f1fe17f45f4e16bc.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/images/9dae0304e4c3347e9b0eced741898059ad2e5dd4c1a3c511f1fe17f45f4e16bc.jpg)

![dd8a53dee7913c3c700246c5336bea1bef253da7f62fdfbf3fff823306135367.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/images/dd8a53dee7913c3c700246c5336bea1bef253da7f62fdfbf3fff823306135367.jpg)

### Tables

![0e228fb04e0094dd5e313c64fe92adcad1f0a07a40bdb3e306ac163052b738f9.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/tables/0e228fb04e0094dd5e313c64fe92adcad1f0a07a40bdb3e306ac163052b738f9.jpg)

![100996c24b8c729cf7c3e12b29d2a529f5ed3de125bb097631711c5352b40129.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/tables/100996c24b8c729cf7c3e12b29d2a529f5ed3de125bb097631711c5352b40129.jpg)

![11c638f2891842ea4c2b4425f9e4e394f5bf98509ff013cf2e0353c2be745284.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/tables/11c638f2891842ea4c2b4425f9e4e394f5bf98509ff013cf2e0353c2be745284.jpg)

![129e4d77176f0b104404baf7e88990d3f0ddf8948a3b813acee39c725a412d06.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/tables/129e4d77176f0b104404baf7e88990d3f0ddf8948a3b813acee39c725a412d06.jpg)

![1afbbcbcab45ae03384548458d5e2d38669beb39aa41d3023b485ded7adf5409.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/tables/1afbbcbcab45ae03384548458d5e2d38669beb39aa41d3023b485ded7adf5409.jpg)

![2030709ee7d2435fb18c28608d33465c4c7f81390d744e706412eef3795a7acb.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/tables/2030709ee7d2435fb18c28608d33465c4c7f81390d744e706412eef3795a7acb.jpg)

![3dcbf52a031fb6c6d70728484f0910319ad01c5fb915b02370bbae5f35b8995a.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/tables/3dcbf52a031fb6c6d70728484f0910319ad01c5fb915b02370bbae5f35b8995a.jpg)

![46068a090bfb39ac3f046150231f68ecbabdbaf60c55e411bd942b494007b240.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/tables/46068a090bfb39ac3f046150231f68ecbabdbaf60c55e411bd942b494007b240.jpg)

![464319cb05a1e2111dea644861b48e50316c01a170867f8dad59d360c2a7f848.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/tables/464319cb05a1e2111dea644861b48e50316c01a170867f8dad59d360c2a7f848.jpg)

![65e5cdbe11d593812f2474a3a6ff008e3473be7b1ce14237226a45798e0fa169.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/tables/65e5cdbe11d593812f2474a3a6ff008e3473be7b1ce14237226a45798e0fa169.jpg)

![73a12223f8055b556e0589df32f89bfeb5800b526c5138112a3161d18a916b30.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/tables/73a12223f8055b556e0589df32f89bfeb5800b526c5138112a3161d18a916b30.jpg)

![9201809578e31e4a5dcbf813f7efb4422d43065549d501e8ffc39ea2c067145e.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/tables/9201809578e31e4a5dcbf813f7efb4422d43065549d501e8ffc39ea2c067145e.jpg)

![aad9344cc3eed92af3ada6f126c4f8da8d60c243bee4feafe4499096e5e25896.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/tables/aad9344cc3eed92af3ada6f126c4f8da8d60c243bee4feafe4499096e5e25896.jpg)

![ce76c227c769dc3a68a4d59360272faeb0b15f144caef75b9e0a73e7edf5a2c7.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/tables/ce76c227c769dc3a68a4d59360272faeb0b15f144caef75b9e0a73e7edf5a2c7.jpg)

![cecb4c1f2ff41ff4a422aef5054af21b6d7ab278e026fdee77d42072a8886863.jpg](../icml_results/4651_HASTE_%20Hardware-Aware%20Dynamic%20Sparse%20Training%20for%20Large%20Output%20Spaces/tables/cecb4c1f2ff41ff4a422aef5054af21b6d7ab278e026fdee77d42072a8886863.jpg)

<a id="paper-4605"></a>

## VeRO: A Harness for Agents to Optimize Agents


### Images

![0b0fc07c57dc0f68cdea175dea933c9f4d340550d8f003d441940df445481d73.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/images/0b0fc07c57dc0f68cdea175dea933c9f4d340550d8f003d441940df445481d73.jpg)

![2cbc23382bc7d2a478aa1046c941726ffc7471170e2870abe1c87882c74fb0a9.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/images/2cbc23382bc7d2a478aa1046c941726ffc7471170e2870abe1c87882c74fb0a9.jpg)

![2f2a2d938ffbd04fed3ab3704b6945ff6772c356bba46410bca34fb8a8df5274.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/images/2f2a2d938ffbd04fed3ab3704b6945ff6772c356bba46410bca34fb8a8df5274.jpg)

![33bbc5f43c64ae165bd6741650d7f16d4c4f10616b60fbb2dc5fd8591787fbd9.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/images/33bbc5f43c64ae165bd6741650d7f16d4c4f10616b60fbb2dc5fd8591787fbd9.jpg)

![36edcfbb96afd885e17d4be3e4e0ccae453f5373be51efb9a96513685eef51ba.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/images/36edcfbb96afd885e17d4be3e4e0ccae453f5373be51efb9a96513685eef51ba.jpg)

![3cc94b0d18159e4b75d5730b971d31d82499a7e4a73bd2c604ac22c9c77bf332.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/images/3cc94b0d18159e4b75d5730b971d31d82499a7e4a73bd2c604ac22c9c77bf332.jpg)

![575a0b81b2b556aae4625a6a9edce906aa0640ae99365011641c6d71fd6a8f75.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/images/575a0b81b2b556aae4625a6a9edce906aa0640ae99365011641c6d71fd6a8f75.jpg)

![6384d9c5b358ec338b190b984049df136d34cb50dbb4763a809703cd8323320f.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/images/6384d9c5b358ec338b190b984049df136d34cb50dbb4763a809703cd8323320f.jpg)

![6642973282db2a4ab824cc14905c30ada37af0239479cfb848bc28681affea6c.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/images/6642973282db2a4ab824cc14905c30ada37af0239479cfb848bc28681affea6c.jpg)

![727775bff4c462c0c4a5eba8971eafe96bf9b77c42fb2877b7240f3e9b5aafdc.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/images/727775bff4c462c0c4a5eba8971eafe96bf9b77c42fb2877b7240f3e9b5aafdc.jpg)

![77c2b4838efaa5c902118ed510ef33f7a6d9af152b1b1f945d383ddae82dbc2b.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/images/77c2b4838efaa5c902118ed510ef33f7a6d9af152b1b1f945d383ddae82dbc2b.jpg)

![a76564c71e5396641b9c9bbb08d90f30e01324ebf41ea578b1b71e4b511595d5.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/images/a76564c71e5396641b9c9bbb08d90f30e01324ebf41ea578b1b71e4b511595d5.jpg)

![accf269924084e933374148d975313d2fa30d3fc2ff44ef2e09367d5498ed2c4.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/images/accf269924084e933374148d975313d2fa30d3fc2ff44ef2e09367d5498ed2c4.jpg)

![c0cf189876c1d31bf4dff237d9d05eee912b9971cec84e6d0d95eef1dc3bf60e.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/images/c0cf189876c1d31bf4dff237d9d05eee912b9971cec84e6d0d95eef1dc3bf60e.jpg)

![d005c44bec8e3fb1115df567a212d2fa97282279183bf722c53f9c62c1400e28.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/images/d005c44bec8e3fb1115df567a212d2fa97282279183bf722c53f9c62c1400e28.jpg)

![d5f693afdaa48db1a1665d6f02e464721924ff9dc859b96c487912b863e65fdd.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/images/d5f693afdaa48db1a1665d6f02e464721924ff9dc859b96c487912b863e65fdd.jpg)

### Tables

![33f97425c8bca4081697ae12bee7884738fc13796f2a7127a8457ae48fb8ad46.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/33f97425c8bca4081697ae12bee7884738fc13796f2a7127a8457ae48fb8ad46.jpg)

![3a291f1065930aa8e9ffe5e4dc9e21773cfd74820f5a6dd0a5ad881a18f872a6.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/3a291f1065930aa8e9ffe5e4dc9e21773cfd74820f5a6dd0a5ad881a18f872a6.jpg)

![45062ec1d123870492cea0ea6982b5efabc275f19a6db548026b7a401c88c5ba.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/45062ec1d123870492cea0ea6982b5efabc275f19a6db548026b7a401c88c5ba.jpg)

![4621c55057e383efedf65211a7f93dc2d16646874f11443ba8964fa30179db0f.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/4621c55057e383efedf65211a7f93dc2d16646874f11443ba8964fa30179db0f.jpg)

![469d177f77b459096cafb119d8376e8681f00f514884a83110bc121bfc259878.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/469d177f77b459096cafb119d8376e8681f00f514884a83110bc121bfc259878.jpg)

![4e9d255ac3e7a211252438b80570f458cbbe38ead6b14296fbc00fca2ff6ca31.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/4e9d255ac3e7a211252438b80570f458cbbe38ead6b14296fbc00fca2ff6ca31.jpg)

![539788b3fb7e24816342f46199c0493717a599ae7042d471d2051cea4cacac78.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/539788b3fb7e24816342f46199c0493717a599ae7042d471d2051cea4cacac78.jpg)

![5e0c25be29445db5c3d454a47a7376ec44d107a4def8b89823a9878ad7bd8909.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/5e0c25be29445db5c3d454a47a7376ec44d107a4def8b89823a9878ad7bd8909.jpg)

![73edd5bf8c0faaf3ea4046ea3b0767dc71e921e9977ee70d08a3da5af96d511d.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/73edd5bf8c0faaf3ea4046ea3b0767dc71e921e9977ee70d08a3da5af96d511d.jpg)

![7ee097c8d44eddb456d969108eb338f5c2581a36aebf562d1922ed648d9747b3.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/7ee097c8d44eddb456d969108eb338f5c2581a36aebf562d1922ed648d9747b3.jpg)

![9e71c61b343332ca29bfe1900c2874fd2643fa922d044657d212f3c7dc7b439d.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/9e71c61b343332ca29bfe1900c2874fd2643fa922d044657d212f3c7dc7b439d.jpg)

![a09d61a664dc48df48b7baec21ffed37e45b517c0cdc07b1ebb252496afc3555.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/a09d61a664dc48df48b7baec21ffed37e45b517c0cdc07b1ebb252496afc3555.jpg)

![bab382cbc07c29a0fccad7bba39900e5c67d71a17ff77f5c9e0270b9bd1a7198.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/bab382cbc07c29a0fccad7bba39900e5c67d71a17ff77f5c9e0270b9bd1a7198.jpg)

![e727a80c092d5136ace7c0e0c14bcf92752d7a8a67a631335463ab7bc04b2bbb.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/e727a80c092d5136ace7c0e0c14bcf92752d7a8a67a631335463ab7bc04b2bbb.jpg)

![e96a43e1f9fa9e8dacbb5ba3f11a43da990de004fdf3b89e6f903f81135811e1.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/e96a43e1f9fa9e8dacbb5ba3f11a43da990de004fdf3b89e6f903f81135811e1.jpg)

![f452c2a6c393228adcdecc88b9813ebe0ed75631cd205724137ba969b1e2b3e9.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/f452c2a6c393228adcdecc88b9813ebe0ed75631cd205724137ba969b1e2b3e9.jpg)

![fa53ca798782d2368d1799d5e8b9d121ceb2b93bb26fff199429b0c07ba6d342.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/fa53ca798782d2368d1799d5e8b9d121ceb2b93bb26fff199429b0c07ba6d342.jpg)

![ff8b227ec80cc0e96b305470e275c66d52dd9fd8375520f0f8c855a885294cf0.jpg](../icml_results/4652_VeRO_%20A%20Harness%20for%20Agents%20to%20Optimize%20Agents/tables/ff8b227ec80cc0e96b305470e275c66d52dd9fd8375520f0f8c855a885294cf0.jpg)

<a id="paper-4606"></a>

## The Art of Interrogation: Consistency Amplifies Factuality in Spatial Reasoning


### Images

![01eeeb066bf506cd37fca1c91ea37e02903d2361d7b4f84346860c97035f5ab2.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/01eeeb066bf506cd37fca1c91ea37e02903d2361d7b4f84346860c97035f5ab2.jpg)

![09bc3c3be3e325a6111868381df612c5203774cc6070335b7abe281e9148bd6e.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/09bc3c3be3e325a6111868381df612c5203774cc6070335b7abe281e9148bd6e.jpg)

![16be73c73910407d83194402f565b13dc74dd918a2b03ed16a1e0ae83184ff4c.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/16be73c73910407d83194402f565b13dc74dd918a2b03ed16a1e0ae83184ff4c.jpg)

![1c02a99ecb59405842a87335921d59a28da441849bae5b53bb4505151edac305.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/1c02a99ecb59405842a87335921d59a28da441849bae5b53bb4505151edac305.jpg)

![1c51f2c1a63113c4b2c99e358e38c0c80e81c7c90dbac38bffb0ce7ac5a4477d.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/1c51f2c1a63113c4b2c99e358e38c0c80e81c7c90dbac38bffb0ce7ac5a4477d.jpg)

![22729c9b7c695fc60a1ff546ce2ee4c9fe3ac7ae1e3c0feae75467861198c79c.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/22729c9b7c695fc60a1ff546ce2ee4c9fe3ac7ae1e3c0feae75467861198c79c.jpg)

![2aa85ce4333f9dbc08e115400891ece1eea0be6bee8234d4bd7d85da251e5caf.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/2aa85ce4333f9dbc08e115400891ece1eea0be6bee8234d4bd7d85da251e5caf.jpg)

![2b48c7d50f0b968d21b1e0efafcb6066b01e13b3ff52a3d90f3141249bc3b6b9.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/2b48c7d50f0b968d21b1e0efafcb6066b01e13b3ff52a3d90f3141249bc3b6b9.jpg)

![2c25ee1e62c70824174895a372167b64d7c2384a7427791400b8c0367eade87c.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/2c25ee1e62c70824174895a372167b64d7c2384a7427791400b8c0367eade87c.jpg)

![3fa5e3ec5027c70de94833e33464a33a82f707bf00294781565a37670dd4f224.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/3fa5e3ec5027c70de94833e33464a33a82f707bf00294781565a37670dd4f224.jpg)

![4361145bd176de21fec1676bca8412fb06031c733c08ce487d3efd8178929bd0.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/4361145bd176de21fec1676bca8412fb06031c733c08ce487d3efd8178929bd0.jpg)

![4504281ff2c50e45995f655d80ab221c65f4a2492f34b280e91e886c77ecc225.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/4504281ff2c50e45995f655d80ab221c65f4a2492f34b280e91e886c77ecc225.jpg)

![45715f7d154376779d26e1a9e5960630406985a358eb2069078173ea8d0f55c5.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/45715f7d154376779d26e1a9e5960630406985a358eb2069078173ea8d0f55c5.jpg)

![5eb3baaa4b4e260518128334345779d70097dbe7497819c113ea022b7216c60b.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/5eb3baaa4b4e260518128334345779d70097dbe7497819c113ea022b7216c60b.jpg)

![5fa68a10d8d950486340391095457c71fa610fede86f86e6899d73cf7d3dee24.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/5fa68a10d8d950486340391095457c71fa610fede86f86e6899d73cf7d3dee24.jpg)

![60787d4972dc25659600c3787017406c9b75c6b38e341b5ff0ac4fc9e70e5786.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/60787d4972dc25659600c3787017406c9b75c6b38e341b5ff0ac4fc9e70e5786.jpg)

![7687d8d7439c2351716f3053cc414b69d927c895bdd905c44510d17c3fc6a63f.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/7687d8d7439c2351716f3053cc414b69d927c895bdd905c44510d17c3fc6a63f.jpg)

![7e4c87c2df0d5d46cfbbd602c7661496eec5d7e043d135baffe81368ffdefa0c.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/7e4c87c2df0d5d46cfbbd602c7661496eec5d7e043d135baffe81368ffdefa0c.jpg)

![7e51b13eaa8ba85ff1f36709fbb11cc6be0a55426ec64c27d40863f854a5b2ef.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/7e51b13eaa8ba85ff1f36709fbb11cc6be0a55426ec64c27d40863f854a5b2ef.jpg)

![a29c180ae2a0f23d11c95ce66f047756d6446aac0cec2646280e2093b565f706.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/a29c180ae2a0f23d11c95ce66f047756d6446aac0cec2646280e2093b565f706.jpg)

![a8b65cb6ee7029d21643defc8f9ef166e1f46b71dfdc2c5e60ac07594784a642.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/a8b65cb6ee7029d21643defc8f9ef166e1f46b71dfdc2c5e60ac07594784a642.jpg)

![a8c1d08ce07cfa4c0f6668cf11d77f613ef4f225ec9bdb28564b39edf621311c.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/a8c1d08ce07cfa4c0f6668cf11d77f613ef4f225ec9bdb28564b39edf621311c.jpg)

![aac51d53f4f71436bde148fdf568b6724ba5f5100427455b8f43a674df90cfdc.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/aac51d53f4f71436bde148fdf568b6724ba5f5100427455b8f43a674df90cfdc.jpg)

![b2beff67892a381059c5cbeb08a7bdb1c310440068b3874355b39be1b2c5a24a.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/b2beff67892a381059c5cbeb08a7bdb1c310440068b3874355b39be1b2c5a24a.jpg)

![b31e017f715a7e80b8d62e4c044d6eca86f2d90b124f56a48aea981097580d89.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/b31e017f715a7e80b8d62e4c044d6eca86f2d90b124f56a48aea981097580d89.jpg)

![b5c90f737dbf535f2cee07bcef7f35a11b3a43f62d9ecd3d1f905de633b41b90.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/b5c90f737dbf535f2cee07bcef7f35a11b3a43f62d9ecd3d1f905de633b41b90.jpg)

![c677f2f9cde4ea207e30836a2d5d7765b4616370d0c334521a7b0d08097d01bb.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/c677f2f9cde4ea207e30836a2d5d7765b4616370d0c334521a7b0d08097d01bb.jpg)

![d896b3433680c643cb1b1fc81217bb0193e0eb156173278f4b99184acee31218.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/d896b3433680c643cb1b1fc81217bb0193e0eb156173278f4b99184acee31218.jpg)

![e09c220796a4ce0f3379b0373e263646ea2417d070b7902d1603173c638bfdec.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/e09c220796a4ce0f3379b0373e263646ea2417d070b7902d1603173c638bfdec.jpg)

![e82a61c7a7891aa9f78e992ec7d408bd676d1f3f8e6f254a3b8c295e23f9d9ad.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/e82a61c7a7891aa9f78e992ec7d408bd676d1f3f8e6f254a3b8c295e23f9d9ad.jpg)

![eb124f952d10faeb6115c1dbb3c3fa7874ee2752116631ba2c5d48b6be7bbcec.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/images/eb124f952d10faeb6115c1dbb3c3fa7874ee2752116631ba2c5d48b6be7bbcec.jpg)

### Tables

![2418450d395c2500faa79978760d86709ae77a763d59dade77ea69e251aeecbe.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/tables/2418450d395c2500faa79978760d86709ae77a763d59dade77ea69e251aeecbe.jpg)

![2a421d57c9d593988b4f3dcdedb95d5c5bcbdb771f261a9a683dbf233c410eef.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/tables/2a421d57c9d593988b4f3dcdedb95d5c5bcbdb771f261a9a683dbf233c410eef.jpg)

![2a67a6efdda6dce057ddb520be4f5c2ef6040567425e308ca4ac37703ac4fa29.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/tables/2a67a6efdda6dce057ddb520be4f5c2ef6040567425e308ca4ac37703ac4fa29.jpg)

![75a5d9dc7220382d4bd92d64d135c1dda42876a2b5264cf97007fd86210d6e3b.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/tables/75a5d9dc7220382d4bd92d64d135c1dda42876a2b5264cf97007fd86210d6e3b.jpg)

![aece426381f3767279a54b5847d02ac07ef9970dbfc9c4da9e4a920a2ecf38a0.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/tables/aece426381f3767279a54b5847d02ac07ef9970dbfc9c4da9e4a920a2ecf38a0.jpg)

![c6f1437e45dc298dd99944e05f8aa1dfd3dcf9ea25ccf16ee964ecc3c0a28b5c.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/tables/c6f1437e45dc298dd99944e05f8aa1dfd3dcf9ea25ccf16ee964ecc3c0a28b5c.jpg)

![de38592c48b0fd267f5eab4c475db1a55d9fd7405db92f0e6e3a68de44ee72e6.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/tables/de38592c48b0fd267f5eab4c475db1a55d9fd7405db92f0e6e3a68de44ee72e6.jpg)

![e4fa6c78d18116c2cb7541abb95a8fb99a6f7a12d8d78fbc7ecd95cb28d08b7e.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/tables/e4fa6c78d18116c2cb7541abb95a8fb99a6f7a12d8d78fbc7ecd95cb28d08b7e.jpg)

![ecf0fc18c223b94850aafbc4b99a66a878bf4922b82b71c01de91fb318fca0fb.jpg](../icml_results/4653_The%20Art%20of%20Interrogation_%20Consistency%20Amplifies%20Factuality%20in%20Spatial%20Reasoning/tables/ecf0fc18c223b94850aafbc4b99a66a878bf4922b82b71c01de91fb318fca0fb.jpg)

<a id="paper-4607"></a>

## BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking


### Images

![03ed56d6c1e8f161070e1d08a23b2fa0597d06045923c6ca4c19103eb8df5afe.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/03ed56d6c1e8f161070e1d08a23b2fa0597d06045923c6ca4c19103eb8df5afe.jpg)

![05baa05f1d768c0cc4181d7541afe650ebcbd15b682379223aabddbddc40da19.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/05baa05f1d768c0cc4181d7541afe650ebcbd15b682379223aabddbddc40da19.jpg)

![07a6b60c0f4afb635fd094d889351f3ad87ad04facc99a68752d4ce27dc4f155.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/07a6b60c0f4afb635fd094d889351f3ad87ad04facc99a68752d4ce27dc4f155.jpg)

![07e6222804e5de86653c8da3694f1f399063d7e13524eefa6240df23079f4f91.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/07e6222804e5de86653c8da3694f1f399063d7e13524eefa6240df23079f4f91.jpg)

![08554dbeb840b80425ec502410ce8f33da3583a1829877acef66919aea111207.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/08554dbeb840b80425ec502410ce8f33da3583a1829877acef66919aea111207.jpg)

![1d37060c260ff393152dbd180e2c30a9428c1d885a2a8f3a2115e021365333a0.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/1d37060c260ff393152dbd180e2c30a9428c1d885a2a8f3a2115e021365333a0.jpg)

![20120eef88128d39766fa370bcc516a8494d9b20e828f12eb54c1393392998ec.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/20120eef88128d39766fa370bcc516a8494d9b20e828f12eb54c1393392998ec.jpg)

![2996c5c6de79155c0e4995a2c90e70ddf45f5d4a7dcde3cafe11f31cccd2e5c7.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/2996c5c6de79155c0e4995a2c90e70ddf45f5d4a7dcde3cafe11f31cccd2e5c7.jpg)

![2c6f14a7429a5fc1457d664e76c4d523902787803bec9414ab8b3e3a58c3c685.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/2c6f14a7429a5fc1457d664e76c4d523902787803bec9414ab8b3e3a58c3c685.jpg)

![33251a3ceaf2058ee4a6017606ad38aae902ffb5aa17f90ea20bf73f464edb22.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/33251a3ceaf2058ee4a6017606ad38aae902ffb5aa17f90ea20bf73f464edb22.jpg)

![41710d1cddaeb09435d61db06f7d6776ab7d66c9886d9fb06178936191603bc8.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/41710d1cddaeb09435d61db06f7d6776ab7d66c9886d9fb06178936191603bc8.jpg)

![4c8ab68c1b919661ebd8a864937a2e3251d27889f4e094afc70be857bc4e451c.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/4c8ab68c1b919661ebd8a864937a2e3251d27889f4e094afc70be857bc4e451c.jpg)

![5dc2c89f08114777ec085f9251d99e42293ad6e74f483f796019f70f6db7ffab.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/5dc2c89f08114777ec085f9251d99e42293ad6e74f483f796019f70f6db7ffab.jpg)

![74763f50ab9a3070186ab7428c10aec55bfd9a9cca8675cdd485bfba7fced955.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/74763f50ab9a3070186ab7428c10aec55bfd9a9cca8675cdd485bfba7fced955.jpg)

![74bd346cd04d1d3f2d0fcb4fda4e06890c035582f83b9e73cf90c6d4d767e925.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/74bd346cd04d1d3f2d0fcb4fda4e06890c035582f83b9e73cf90c6d4d767e925.jpg)

![753a406e4e1bb35f60601aa09c6d200161507f77c2c0894eb2530c4f0ac4a881.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/753a406e4e1bb35f60601aa09c6d200161507f77c2c0894eb2530c4f0ac4a881.jpg)

![76ed4d2f070cb29af1d111ae120afa0555ebdb6552578334ba3391a81e8bf994.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/76ed4d2f070cb29af1d111ae120afa0555ebdb6552578334ba3391a81e8bf994.jpg)

![7b52400aff0f8f5681c9fdd609176ae3232ebbc1c6d5b2cb6471aed5e6a7d13f.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/7b52400aff0f8f5681c9fdd609176ae3232ebbc1c6d5b2cb6471aed5e6a7d13f.jpg)

![84beab15279ff268f0f6eb7abeca549cf6673a42c98c5e856ae6f13aa4741e88.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/84beab15279ff268f0f6eb7abeca549cf6673a42c98c5e856ae6f13aa4741e88.jpg)

![8a841dbf00d155344547bcdf97d00872e8cb6cac9652823a9ba0e63f57d42731.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/8a841dbf00d155344547bcdf97d00872e8cb6cac9652823a9ba0e63f57d42731.jpg)

![8fa6c3279c96c2c546ab00f3f6050a21028095809b5aac92d6be7fa0bfff1855.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/8fa6c3279c96c2c546ab00f3f6050a21028095809b5aac92d6be7fa0bfff1855.jpg)

![9503003da7fbc53f51d4e4fa0d9980f97bc72931a2467549387dffd96e24b870.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/9503003da7fbc53f51d4e4fa0d9980f97bc72931a2467549387dffd96e24b870.jpg)

![a1cf28914b9ac243af9a7ae27206aa32327f15e0ef82fe8ca34108a54aa1fad9.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/a1cf28914b9ac243af9a7ae27206aa32327f15e0ef82fe8ca34108a54aa1fad9.jpg)

![ac321d493c932312ac44ecc0b8042801e4bdba4b03dcb2dd59dea5612039e774.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/ac321d493c932312ac44ecc0b8042801e4bdba4b03dcb2dd59dea5612039e774.jpg)

![ba603f07845755a881f8315873d7c62ee61bffd88486273af835fed01150b34d.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/ba603f07845755a881f8315873d7c62ee61bffd88486273af835fed01150b34d.jpg)

![ba74cd49006159afdf0daf7d9c56adf9da26099a9990654fd8d6021f75a1dda7.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/ba74cd49006159afdf0daf7d9c56adf9da26099a9990654fd8d6021f75a1dda7.jpg)

![baaf8f4adbf16828f275036ffa1e966d74f210a9d4645e1cd458cc60c1752f21.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/baaf8f4adbf16828f275036ffa1e966d74f210a9d4645e1cd458cc60c1752f21.jpg)

![bb59b03b8ff1c24dee773847164568071c7607bb5583cf778db63bfb32250964.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/bb59b03b8ff1c24dee773847164568071c7607bb5583cf778db63bfb32250964.jpg)

![bdd546a9e3a318d248a3414c1d63e6a9b7a2ea01a1c7b8773d6dae66bf93ea92.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/bdd546a9e3a318d248a3414c1d63e6a9b7a2ea01a1c7b8773d6dae66bf93ea92.jpg)

![c483dd074ef69aae471a11697b4372bf0aeb61ff99d8d2277973e30ed10f2747.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/c483dd074ef69aae471a11697b4372bf0aeb61ff99d8d2277973e30ed10f2747.jpg)

![c6aa51aed95ccece6e663d48f62a51e5ff86d0e5ebdee39983d9e3375a83e635.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/c6aa51aed95ccece6e663d48f62a51e5ff86d0e5ebdee39983d9e3375a83e635.jpg)

![c9854a067a6de284e3256cd9dea8a035d8a8ce182eedf4c4f6bd400ae64b9f63.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/c9854a067a6de284e3256cd9dea8a035d8a8ce182eedf4c4f6bd400ae64b9f63.jpg)

![cfbac95e3713fcfbf755f51e11afcb9854766ea3a6580d441da599f6e3a4d5a8.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/cfbac95e3713fcfbf755f51e11afcb9854766ea3a6580d441da599f6e3a4d5a8.jpg)

![ee2c85613c2cfc2dc887193ec713ff64d8f321b9c870b508fcebe973e08def09.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/ee2c85613c2cfc2dc887193ec713ff64d8f321b9c870b508fcebe973e08def09.jpg)

![fe13d80009fc2c1ac0075bf41db082e8a013767f8c9e2f7b4fdf6b91aa255c56.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/images/fe13d80009fc2c1ac0075bf41db082e8a013767f8c9e2f7b4fdf6b91aa255c56.jpg)

### Tables

![4577850f3b0328c1306502b58ff8fa01b08454d6e0b905360d1beb7d39f86cb1.jpg](../icml_results/4654_BLOCK-EM_%20Preventing%20Emergent%20Misalignment%20via%20Latent%20Blocking/tables/4577850f3b0328c1306502b58ff8fa01b08454d6e0b905360d1beb7d39f86cb1.jpg)

<a id="paper-4608"></a>

## Trust Functions: Near-Lossless Weak-to-Strong Generalization by Learning When to Trust the Weak Teacher


### Images

![10e957d9cd6cb7bb13c557410e1d33a686e2a84e1e76c940cbe6789db3dacc47.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/images/10e957d9cd6cb7bb13c557410e1d33a686e2a84e1e76c940cbe6789db3dacc47.jpg)

![1c2bb16affd136db998174d2fcaa4c812bdf66f7964a797aa543d1acfd90dd8f.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/images/1c2bb16affd136db998174d2fcaa4c812bdf66f7964a797aa543d1acfd90dd8f.jpg)

![49ab843af200a0eee9f7b39efcf0f0496c97867b17f5c2335002d1af98573087.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/images/49ab843af200a0eee9f7b39efcf0f0496c97867b17f5c2335002d1af98573087.jpg)

![72cc6e20fcee5c7236b3c5e5bf6b671c95c3f230ffa4b2c0fe736111fb0ce2ea.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/images/72cc6e20fcee5c7236b3c5e5bf6b671c95c3f230ffa4b2c0fe736111fb0ce2ea.jpg)

![a2add0b85442f2b61f7bd87aa227f3ae69764a89055ab0af534a41aec631c7e0.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/images/a2add0b85442f2b61f7bd87aa227f3ae69764a89055ab0af534a41aec631c7e0.jpg)

![fe0ddb407c21bc62c5e33b7431f624ae6fc96ea3c8f44b483d5aa149670c0ced.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/images/fe0ddb407c21bc62c5e33b7431f624ae6fc96ea3c8f44b483d5aa149670c0ced.jpg)

### Tables

![09f68b5dd5550d3e499852f2488458b20fd69f5d24c9e224ad9fb3a619ff7ff0.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/09f68b5dd5550d3e499852f2488458b20fd69f5d24c9e224ad9fb3a619ff7ff0.jpg)

![1308fdbafee9e15b03b03b0d7678c34275da23261f30285e52bc93be2640bfbf.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/1308fdbafee9e15b03b03b0d7678c34275da23261f30285e52bc93be2640bfbf.jpg)

![2959b3b26cdd00fe829726dd24910dbf5008ced0c0931d21452bf6c5fbd21c2d.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/2959b3b26cdd00fe829726dd24910dbf5008ced0c0931d21452bf6c5fbd21c2d.jpg)

![2ace173c6da46f2d19d02211bb67deb9bba5f4cb82609562ea3f92b134f621a8.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/2ace173c6da46f2d19d02211bb67deb9bba5f4cb82609562ea3f92b134f621a8.jpg)

![3489d3be70d58f126037cff8763c8824c7e225b0b293a2245459ea4299e259bb.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/3489d3be70d58f126037cff8763c8824c7e225b0b293a2245459ea4299e259bb.jpg)

![45bb17252c47942acf20c6275bd7284a0525867e79c10c6342e0b67cf24aeef3.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/45bb17252c47942acf20c6275bd7284a0525867e79c10c6342e0b67cf24aeef3.jpg)

![45dad8147c46497c9a40e9c49e6849e11f26d5a79474d880fdca529f9f0d7fb7.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/45dad8147c46497c9a40e9c49e6849e11f26d5a79474d880fdca529f9f0d7fb7.jpg)

![4e3272b23591a1602790d9fde69965f94f9a524b3a4bd6df80a726835ec44016.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/4e3272b23591a1602790d9fde69965f94f9a524b3a4bd6df80a726835ec44016.jpg)

![6d82688a7d4476dd8e7fc1e7f0cf9a5efbcb129dbfbcddd72468f105897975cf.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/6d82688a7d4476dd8e7fc1e7f0cf9a5efbcb129dbfbcddd72468f105897975cf.jpg)

![72533a5034360c296851c7a5fe8daeb414618afc15b1fe1422b87204aa664470.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/72533a5034360c296851c7a5fe8daeb414618afc15b1fe1422b87204aa664470.jpg)

![7c4589b994c00348c8893b6b224d5d828e4d2e6608e262a9ebc0d3d6d9388e0e.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/7c4589b994c00348c8893b6b224d5d828e4d2e6608e262a9ebc0d3d6d9388e0e.jpg)

![88f81d24de50d988e38196b52096e76101af88f7565aa227eafb00911f32e17d.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/88f81d24de50d988e38196b52096e76101af88f7565aa227eafb00911f32e17d.jpg)

![8a5d4e2cfed3104c886d1957c0036f8f8f6d52f55b14c58fad849da89dd0173f.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/8a5d4e2cfed3104c886d1957c0036f8f8f6d52f55b14c58fad849da89dd0173f.jpg)

![8a720acb04ba4b737e3a7602141ac8b1a953275a5b885037eec00ff6d3faed8e.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/8a720acb04ba4b737e3a7602141ac8b1a953275a5b885037eec00ff6d3faed8e.jpg)

![8a99dabc34864f812fadd6d434c7fbbc087b62e1a7f2840637acb8a45634589c.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/8a99dabc34864f812fadd6d434c7fbbc087b62e1a7f2840637acb8a45634589c.jpg)

![90c8e10ab9d08190c2e31d55078b6248b4e239639302d2db5270935297631662.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/90c8e10ab9d08190c2e31d55078b6248b4e239639302d2db5270935297631662.jpg)

![9bbf7e98b2e131048dded68bd70cfb422b62e86b5bf35e2ab965b469c4bb3fe0.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/9bbf7e98b2e131048dded68bd70cfb422b62e86b5bf35e2ab965b469c4bb3fe0.jpg)

![9be591b596b4e2fac0b424f75e6413f7a7ca1bb21318f4a6f5e0fc3f5f70110b.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/9be591b596b4e2fac0b424f75e6413f7a7ca1bb21318f4a6f5e0fc3f5f70110b.jpg)

![a3834197c6ee40193288124f9507d0a5a24371ff633245748f71aba4dbdca8f3.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/a3834197c6ee40193288124f9507d0a5a24371ff633245748f71aba4dbdca8f3.jpg)

![a5ffc4dc8bfb0e5aee142b3b43152177aca7b1ed48a443814c7f326a2d339fc6.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/a5ffc4dc8bfb0e5aee142b3b43152177aca7b1ed48a443814c7f326a2d339fc6.jpg)

![a730973b8dae0c74f66228530cdbf2616e78187e47a9741b4c59ab26f9f7f832.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/a730973b8dae0c74f66228530cdbf2616e78187e47a9741b4c59ab26f9f7f832.jpg)

![b45c48ac249e20253969ab52d9b0adfce16ad8e0ec6f6045da63591df71d7a91.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/b45c48ac249e20253969ab52d9b0adfce16ad8e0ec6f6045da63591df71d7a91.jpg)

![c31274d2b4462fb8dab08af56d2bcdae900dfeef1eedea47be4336d7c2f2a222.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/c31274d2b4462fb8dab08af56d2bcdae900dfeef1eedea47be4336d7c2f2a222.jpg)

![da16519389a5f95ccba03b5cf167795e8ef04c3203b15adc0543cdd7767a0d74.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/da16519389a5f95ccba03b5cf167795e8ef04c3203b15adc0543cdd7767a0d74.jpg)

![de253cbcb41e7b14b462ac220ee170148fb439667620ae245cda37c653cd8f40.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/de253cbcb41e7b14b462ac220ee170148fb439667620ae245cda37c653cd8f40.jpg)

![f51fa515606da56badb67dc36f0fcdabe8fb69b49f3a7e16c9d7ef2dccd752f0.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/f51fa515606da56badb67dc36f0fcdabe8fb69b49f3a7e16c9d7ef2dccd752f0.jpg)

![f6274393fc96515731a1af0bcf62ffbde1d6a8c000d742befa10669ff11ae4ba.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/f6274393fc96515731a1af0bcf62ffbde1d6a8c000d742befa10669ff11ae4ba.jpg)

![fa7a69e0d4007421d5035cc2b32817e825367b54afadbd36bc21f44589851ea1.jpg](../icml_results/4655_Trust%20Functions_%20Near-Lossless%20Weak-to-Strong%20Generalization%20by%20Learning%20When%20to%20Trust%20the%20Weak%20Teac/tables/fa7a69e0d4007421d5035cc2b32817e825367b54afadbd36bc21f44589851ea1.jpg)

<a id="paper-4609"></a>

## RAIGen: Rare Attribute Identification in Text-to-Image Generative Models


### Images

![0311c9c878585a28c985fc29411c5152d13796315ed3ffe34ee78062065f8be0.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/0311c9c878585a28c985fc29411c5152d13796315ed3ffe34ee78062065f8be0.jpg)

![0bb452cb24706bdbc704c3dc5aeaf98241ea1cfb3ae0fdc4ddb9b1bebd0563ef.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/0bb452cb24706bdbc704c3dc5aeaf98241ea1cfb3ae0fdc4ddb9b1bebd0563ef.jpg)

![0f0bf9936b865efa4782b20d2e852dc0731c3c6788400e630ac330150ff6e0c1.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/0f0bf9936b865efa4782b20d2e852dc0731c3c6788400e630ac330150ff6e0c1.jpg)

![220ae1aaa54b875e84685b86b74f5550cdfc5530fc6a952c3a31af3079405432.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/220ae1aaa54b875e84685b86b74f5550cdfc5530fc6a952c3a31af3079405432.jpg)

![2f387bf74eedbf1b0c721d05210d853ccddd8aa32a892c900b40455b67093794.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/2f387bf74eedbf1b0c721d05210d853ccddd8aa32a892c900b40455b67093794.jpg)

![3003fcf272fb8379eba426b770d269a1413dbb76819b7619c012471f554480a1.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/3003fcf272fb8379eba426b770d269a1413dbb76819b7619c012471f554480a1.jpg)

![46ac6f756bc5d19b45c5d852364a2fc87d284f91e95f199dcd4ae04df0ecf4f5.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/46ac6f756bc5d19b45c5d852364a2fc87d284f91e95f199dcd4ae04df0ecf4f5.jpg)

![5d08fea6a1fff00c9d449c8a3ec3fe8853ed96de7d3a1544b6ba844408434668.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/5d08fea6a1fff00c9d449c8a3ec3fe8853ed96de7d3a1544b6ba844408434668.jpg)

![5ede691968186b113eb580c01960c9069f5d5052e0929e2c4988d48b6f3d53bc.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/5ede691968186b113eb580c01960c9069f5d5052e0929e2c4988d48b6f3d53bc.jpg)

![7f9eaae2b0c884814906084a28aae2100e55ea3c06c68381951cc92cda12d6ed.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/7f9eaae2b0c884814906084a28aae2100e55ea3c06c68381951cc92cda12d6ed.jpg)

![a258eb389df715d3261aff65489cff1bce1fc0446674b2532913613eb7b97240.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/a258eb389df715d3261aff65489cff1bce1fc0446674b2532913613eb7b97240.jpg)

![aa6e8af4139fb252657e5d73e34bd6056ac1b858d64b2ae9bdd2d6beabac6a29.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/aa6e8af4139fb252657e5d73e34bd6056ac1b858d64b2ae9bdd2d6beabac6a29.jpg)

![ab60d9ddb0eef3410b924cbbeb280130305c4fb1bb765c6483a70d321e3f099c.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/ab60d9ddb0eef3410b924cbbeb280130305c4fb1bb765c6483a70d321e3f099c.jpg)

![ae812e0a4b8b850270f2ebe9045c21586e62a4e9357febdab731ded253704d26.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/ae812e0a4b8b850270f2ebe9045c21586e62a4e9357febdab731ded253704d26.jpg)

![b8bbd88161b2da38264b60bef5b9ce174e086da33a8e4433be4c28ce176cacca.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/b8bbd88161b2da38264b60bef5b9ce174e086da33a8e4433be4c28ce176cacca.jpg)

![b9ddc4fb9e5e0a5d78eeb49cb76d7e2cd1478e35e59d5d3466b9c173774f7771.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/b9ddc4fb9e5e0a5d78eeb49cb76d7e2cd1478e35e59d5d3466b9c173774f7771.jpg)

![bb34a8bb647fc07b45f4af178805c9f7806a8351055ca25c9d69584a10313c8c.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/bb34a8bb647fc07b45f4af178805c9f7806a8351055ca25c9d69584a10313c8c.jpg)

![bedb0dc6c41add5b5354333ef42245a43ae24c1ef90bea7205896ce5324e4ccb.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/bedb0dc6c41add5b5354333ef42245a43ae24c1ef90bea7205896ce5324e4ccb.jpg)

![c023402bd32cf99db2c5c54f5f381e3d574777585afde137ba88596625fc502e.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/c023402bd32cf99db2c5c54f5f381e3d574777585afde137ba88596625fc502e.jpg)

![c18a1b4f47282baf4cdb663a750cad2d58c7c7cd56af15a90a3e189826d97975.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/c18a1b4f47282baf4cdb663a750cad2d58c7c7cd56af15a90a3e189826d97975.jpg)

![d98a59210e5f34ee344ace1f7f5baa5ff82c656792fa4f088d26c4ecdf248f93.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/d98a59210e5f34ee344ace1f7f5baa5ff82c656792fa4f088d26c4ecdf248f93.jpg)

![e84b996d20d98105eaab3f01bb7d9c7345ca6bc772f13a8fed3e50c7cfa26d49.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/e84b996d20d98105eaab3f01bb7d9c7345ca6bc772f13a8fed3e50c7cfa26d49.jpg)

![eb6ae7e30d58436ce4dfe10dece7e70f62ff9ce4226669b30bfb63ca08cf595c.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/eb6ae7e30d58436ce4dfe10dece7e70f62ff9ce4226669b30bfb63ca08cf595c.jpg)

![f0cb1ca81b7fcbc5d233508a2c8b9139e4558e0d74432845114efc0c6ab061b5.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/f0cb1ca81b7fcbc5d233508a2c8b9139e4558e0d74432845114efc0c6ab061b5.jpg)

![fb3e9ce3392bae685e889d619a58e457b279c0851325c039d61a693e2ad19550.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/images/fb3e9ce3392bae685e889d619a58e457b279c0851325c039d61a693e2ad19550.jpg)

### Tables

![1877c2daa74b6073ee2da8f13a9ee807648d2d7f0de7eed5b88a175ae841e799.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/tables/1877c2daa74b6073ee2da8f13a9ee807648d2d7f0de7eed5b88a175ae841e799.jpg)

![43945893dfb516482885cf2a3d73f979da9be32720850fdf069dc4f7f3e5aea7.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/tables/43945893dfb516482885cf2a3d73f979da9be32720850fdf069dc4f7f3e5aea7.jpg)

![56e2e0f5696f674abb54d7b6effafb0fda6bbc30e9baa7d68c2d4e923e2dd982.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/tables/56e2e0f5696f674abb54d7b6effafb0fda6bbc30e9baa7d68c2d4e923e2dd982.jpg)

![5fb2b351ae94cef4bbc0ecd669ded63c0b160b1ff126a23e279b4b09f037cd61.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/tables/5fb2b351ae94cef4bbc0ecd669ded63c0b160b1ff126a23e279b4b09f037cd61.jpg)

![7ed3a713db996c1f553171478be039499efe55e02ef5fb1a4366c1c1a97125ae.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/tables/7ed3a713db996c1f553171478be039499efe55e02ef5fb1a4366c1c1a97125ae.jpg)

![844e1d06e30dd948422ad44e7ebc6686adf3d2b5e63f074791f5d6e502380c6c.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/tables/844e1d06e30dd948422ad44e7ebc6686adf3d2b5e63f074791f5d6e502380c6c.jpg)

![90724f2dd8d6340e9e6dcf4c4fe732cdc9e04ec83dd70b20b30154ea2359d420.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/tables/90724f2dd8d6340e9e6dcf4c4fe732cdc9e04ec83dd70b20b30154ea2359d420.jpg)

![aa899006cea9444367a48233810f4bbde3654c1e49f74b1c5944834e3ef30510.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/tables/aa899006cea9444367a48233810f4bbde3654c1e49f74b1c5944834e3ef30510.jpg)

![ab1d3e8bbb20e81f226f94c3ff219b32cf56af1ca238b43e57fa6ac8857c99f2.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/tables/ab1d3e8bbb20e81f226f94c3ff219b32cf56af1ca238b43e57fa6ac8857c99f2.jpg)

![bbf982db11c4e1749b7ad93e27f2d6f94e0ff69566783e7f98e0bd6a8545be8a.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/tables/bbf982db11c4e1749b7ad93e27f2d6f94e0ff69566783e7f98e0bd6a8545be8a.jpg)

![d39511a910649f987b8a7f887bc0384759524dbf20caeca4575039c70a820356.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/tables/d39511a910649f987b8a7f887bc0384759524dbf20caeca4575039c70a820356.jpg)

![fc15cbc3af9f05fd4dd318bcb8528855bd1939bea8dea171a6a525427ae87563.jpg](../icml_results/4656_RAIGen_%20Rare%20Attribute%20Identification%20in%20Text-to-Image%20Generative%20Models/tables/fc15cbc3af9f05fd4dd318bcb8528855bd1939bea8dea171a6a525427ae87563.jpg)

<a id="paper-4610"></a>

## Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?


### Images

![32b960eba0ed1cbaaa2231e9089fdf124a77a8ea7b6e5f180c7cf38ae0301e9d.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/images/32b960eba0ed1cbaaa2231e9089fdf124a77a8ea7b6e5f180c7cf38ae0301e9d.jpg)

![4bed1bff58674cda611f88b333eb767bcfe7b67dbb19b4e0306076ba3f573628.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/images/4bed1bff58674cda611f88b333eb767bcfe7b67dbb19b4e0306076ba3f573628.jpg)

![5cbc0e328b8f564703b045a4164d0501037c0a0857a33ea2f5738696b39e085d.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/images/5cbc0e328b8f564703b045a4164d0501037c0a0857a33ea2f5738696b39e085d.jpg)

![5d368d68827b314f97ded1857acbfe4a32e39b996c4a6b6ed8664906b0302e7c.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/images/5d368d68827b314f97ded1857acbfe4a32e39b996c4a6b6ed8664906b0302e7c.jpg)

![953b69f458c82443b308f2c0229819fbdfad1c61be478a32395623154a22ba6a.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/images/953b69f458c82443b308f2c0229819fbdfad1c61be478a32395623154a22ba6a.jpg)

![ae4355b1c4468409d8a54674ac97867cbff26a35a6711bca7b137157648c9159.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/images/ae4355b1c4468409d8a54674ac97867cbff26a35a6711bca7b137157648c9159.jpg)

![b401b27934152fd38a4a1131c509f071ac676b33b5f622250890b20fd505de38.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/images/b401b27934152fd38a4a1131c509f071ac676b33b5f622250890b20fd505de38.jpg)

![e7e5bb2cdf9ffdff1dac7d7079a17818c82568eb3a45b726f02a48509575cc02.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/images/e7e5bb2cdf9ffdff1dac7d7079a17818c82568eb3a45b726f02a48509575cc02.jpg)

![eb92e823c6eb19f85ebab15275807c60c0379a7268a656edf7b42e1a52f89192.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/images/eb92e823c6eb19f85ebab15275807c60c0379a7268a656edf7b42e1a52f89192.jpg)

![fea44213cebbcef50f9318e2a75e73f69779ca847f105c3c4c2058bae96b737d.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/images/fea44213cebbcef50f9318e2a75e73f69779ca847f105c3c4c2058bae96b737d.jpg)

### Tables

![253edbddc9b6063cc391e872a0b009d0f42c2ca4ab665a06481028b2c118a600.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/tables/253edbddc9b6063cc391e872a0b009d0f42c2ca4ab665a06481028b2c118a600.jpg)

![327fb61772344ae1f32ed27be9f4577bc4addddb68f84c7548a6a3f803c57dff.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/tables/327fb61772344ae1f32ed27be9f4577bc4addddb68f84c7548a6a3f803c57dff.jpg)

![5bb4a473adde416b101909c15880ba5d31cb4399b3711f4fbbe62117c812b8e0.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/tables/5bb4a473adde416b101909c15880ba5d31cb4399b3711f4fbbe62117c812b8e0.jpg)

![77effd735037b081293caba2ac0eb2e3e5cd66816f3bae301da4940f34769733.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/tables/77effd735037b081293caba2ac0eb2e3e5cd66816f3bae301da4940f34769733.jpg)

![ab6f85bd39f764723d41eb52fad17beccadcebf8bc8298423bd7c8c72bb4510f.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/tables/ab6f85bd39f764723d41eb52fad17beccadcebf8bc8298423bd7c8c72bb4510f.jpg)

![b0aee38e1d2ee3472b6bbf7b7d78db2daf3691a7f17773a192d5edf46c0ae0f3.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/tables/b0aee38e1d2ee3472b6bbf7b7d78db2daf3691a7f17773a192d5edf46c0ae0f3.jpg)

![c979a80cab8f8128149dbcb65a2c0e60b57ca57bf6f79a03db3ea3dfd8d701e9.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/tables/c979a80cab8f8128149dbcb65a2c0e60b57ca57bf6f79a03db3ea3dfd8d701e9.jpg)

![cc9a7d3f91718cc82cc02e74d0e23e47658c30f930dec4ad7ba198365f27941e.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/tables/cc9a7d3f91718cc82cc02e74d0e23e47658c30f930dec4ad7ba198365f27941e.jpg)

![e19660ec98f1fcbfb435acfbb932b88a0d1330f65a7f3ede6ad4d47e81b2b194.jpg](../icml_results/4657_Shapley%20Neuron%20Values%20for%20Continual%20Learning_%20Which%20Neurons%20Matter%20Most_/tables/e19660ec98f1fcbfb435acfbb932b88a0d1330f65a7f3ede6ad4d47e81b2b194.jpg)

<a id="paper-4611"></a>

## Convergence Rate of the Last Iterate of Stochastic Proximal Algorithms


### Images

![51f77a393688a2878851d1b2294c3a15a135f32ef67f533480756869ff7da408.jpg](../icml_results/4658_Convergence%20Rate%20of%20the%20Last%20Iterate%20of%20Stochastic%20Proximal%20Algorithms/images/51f77a393688a2878851d1b2294c3a15a135f32ef67f533480756869ff7da408.jpg)

![a656e47aa1bf3469accc004eee03640c84ed3502252a6ed73536f3411f62af68.jpg](../icml_results/4658_Convergence%20Rate%20of%20the%20Last%20Iterate%20of%20Stochastic%20Proximal%20Algorithms/images/a656e47aa1bf3469accc004eee03640c84ed3502252a6ed73536f3411f62af68.jpg)

![ddf223052a427b9ea42af516ac02877a4dfde03073e66d59e76b62e2c2d9b212.jpg](../icml_results/4658_Convergence%20Rate%20of%20the%20Last%20Iterate%20of%20Stochastic%20Proximal%20Algorithms/images/ddf223052a427b9ea42af516ac02877a4dfde03073e66d59e76b62e2c2d9b212.jpg)

### Tables

![ab84ce73b2746e2d30c318351c448138301c1b492043cafa05cffa58a63a30c1.jpg](../icml_results/4658_Convergence%20Rate%20of%20the%20Last%20Iterate%20of%20Stochastic%20Proximal%20Algorithms/tables/ab84ce73b2746e2d30c318351c448138301c1b492043cafa05cffa58a63a30c1.jpg)

<a id="paper-4612"></a>

## Long-term Fairness with Selective Labels


### Images

![0c7a624ca7f6c19d7486423fbd941825d9356f93fdfd357b07ccfd31540897e4.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/0c7a624ca7f6c19d7486423fbd941825d9356f93fdfd357b07ccfd31540897e4.jpg)

![13c5297f57d11eec0e567d8d461edcd86d5cfb2402a9006ddff5b3ade1c072cb.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/13c5297f57d11eec0e567d8d461edcd86d5cfb2402a9006ddff5b3ade1c072cb.jpg)

![186d2318585a0d65f619bdd6aa05fd4fb98aec34310eabd6cbbffa0f41a41d1b.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/186d2318585a0d65f619bdd6aa05fd4fb98aec34310eabd6cbbffa0f41a41d1b.jpg)

![1aeda980359da9016f7945d0be8e05aaf0afae1515444ecb2aff77effdb640b4.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/1aeda980359da9016f7945d0be8e05aaf0afae1515444ecb2aff77effdb640b4.jpg)

![1cc2b87fc37821f0399c9bb15ac4351d55d449453db64f7f30b54a4038a63a79.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/1cc2b87fc37821f0399c9bb15ac4351d55d449453db64f7f30b54a4038a63a79.jpg)

![21dc39c83db2c6c0201e55980809730514878ee1f44a3383063736d258cf3de6.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/21dc39c83db2c6c0201e55980809730514878ee1f44a3383063736d258cf3de6.jpg)

![2dd8b34ecee3ba44070aee95eefd05d11ffefc75acafca9c13196f8e3df6ba79.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/2dd8b34ecee3ba44070aee95eefd05d11ffefc75acafca9c13196f8e3df6ba79.jpg)

![35aaf94da1384a4b6d69dae6ae082244252ab68ca7703fa3086861c1b5b8a3be.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/35aaf94da1384a4b6d69dae6ae082244252ab68ca7703fa3086861c1b5b8a3be.jpg)

![3893af01f38ff6d4dfef3a45845116112d3bbbbfc7979704fda839b4a357e005.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/3893af01f38ff6d4dfef3a45845116112d3bbbbfc7979704fda839b4a357e005.jpg)

![3a676901a9e74dfebf2b7ae23963388149e1fa039d5e54d64c382fb0d7bd8adf.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/3a676901a9e74dfebf2b7ae23963388149e1fa039d5e54d64c382fb0d7bd8adf.jpg)

![41852f6ab180c2137bdebe22aea97e1eddd9feef4c8f61c07b208e0171032749.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/41852f6ab180c2137bdebe22aea97e1eddd9feef4c8f61c07b208e0171032749.jpg)

![882845b8ec481293af11f58551aafbe07e24e75b8d204cb6ea8f818b128e4004.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/882845b8ec481293af11f58551aafbe07e24e75b8d204cb6ea8f818b128e4004.jpg)

![886ae01d0c9c6be44f403d68d8db1c35c3630bae8846faad2faa6b225c1d245e.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/886ae01d0c9c6be44f403d68d8db1c35c3630bae8846faad2faa6b225c1d245e.jpg)

![9d79207338f92cc32ea82dd1ad2bd446d624e38af3cd8edb81f21795daa36a55.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/9d79207338f92cc32ea82dd1ad2bd446d624e38af3cd8edb81f21795daa36a55.jpg)

![9f90ca66de19b572317da99e2b4d23746e887655025bba5e4203c67fe230c535.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/9f90ca66de19b572317da99e2b4d23746e887655025bba5e4203c67fe230c535.jpg)

![c21dea44712faac9f76df912477317340169a90c0c0626271281e15ddaa775a1.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/c21dea44712faac9f76df912477317340169a90c0c0626271281e15ddaa775a1.jpg)

![db46b04b722022e5c5f7f28e5c39dba91afc99cd6972906978ec9f77941cf3bb.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/db46b04b722022e5c5f7f28e5c39dba91afc99cd6972906978ec9f77941cf3bb.jpg)

![db5864d1155099618621d3e999674c82f7cd1a6e21c69369d270bbfe7e6e9954.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/db5864d1155099618621d3e999674c82f7cd1a6e21c69369d270bbfe7e6e9954.jpg)

![ff865641efa8a660118efcd474a1c215414cda14d3378e6f4a0c2254d7f5695f.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/images/ff865641efa8a660118efcd474a1c215414cda14d3378e6f4a0c2254d7f5695f.jpg)

### Tables

![10ade296747353e6752850d618f422c307c04c3e2c425fd3aa786e68dbe8a5c7.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/tables/10ade296747353e6752850d618f422c307c04c3e2c425fd3aa786e68dbe8a5c7.jpg)

![466f9499fc2140c968982e720c274b54b508fc42cb7f41949c09b66ef16f5d35.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/tables/466f9499fc2140c968982e720c274b54b508fc42cb7f41949c09b66ef16f5d35.jpg)

![b914bd4f6cc0642d8a518e530708cb2aafc6551e768ecb4cee31edb4d6f5a4b4.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/tables/b914bd4f6cc0642d8a518e530708cb2aafc6551e768ecb4cee31edb4d6f5a4b4.jpg)

![db8441f53f31a88771874b1aa8bfd5e2a4a51109bc3b9772ff51e10bfa60884c.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/tables/db8441f53f31a88771874b1aa8bfd5e2a4a51109bc3b9772ff51e10bfa60884c.jpg)

![fb1bf87e23e693f6b3715ef898667c55318b0298a0fd580ceb7c0a273d8db79e.jpg](../icml_results/4659_Long-term%20Fairness%20with%20Selective%20Labels/tables/fb1bf87e23e693f6b3715ef898667c55318b0298a0fd580ceb7c0a273d8db79e.jpg)

<a id="paper-4613"></a>

## DB-KSVD: Scalable Alternating Optimization for Disentangling High-Dimensional Embedding Spaces


### Images

![2b611f88a71e90e2c1321d481e35faf19bbc519f3392f04f0d0ec99b1b58b43d.jpg](../icml_results/4660_DB-KSVD_%20Scalable%20Alternating%20Optimization%20for%20Disentangling%20High-Dimensional%20Embedding%20Spaces/images/2b611f88a71e90e2c1321d481e35faf19bbc519f3392f04f0d0ec99b1b58b43d.jpg)

![529046ab716d45a061e7eedf2ce7c0d07e62cefba2e328f97e82477ac11508b3.jpg](../icml_results/4660_DB-KSVD_%20Scalable%20Alternating%20Optimization%20for%20Disentangling%20High-Dimensional%20Embedding%20Spaces/images/529046ab716d45a061e7eedf2ce7c0d07e62cefba2e328f97e82477ac11508b3.jpg)

![624476a3a763b3c794de3442482a27d02d14821b196ada98aff9977aeaedbb5f.jpg](../icml_results/4660_DB-KSVD_%20Scalable%20Alternating%20Optimization%20for%20Disentangling%20High-Dimensional%20Embedding%20Spaces/images/624476a3a763b3c794de3442482a27d02d14821b196ada98aff9977aeaedbb5f.jpg)

![9196e882967468742d2ffb538608d79e415fa396862b11aa87f3f3e09edd88ff.jpg](../icml_results/4660_DB-KSVD_%20Scalable%20Alternating%20Optimization%20for%20Disentangling%20High-Dimensional%20Embedding%20Spaces/images/9196e882967468742d2ffb538608d79e415fa396862b11aa87f3f3e09edd88ff.jpg)

![9313185284f92770c424355a7ff664a655d5f39efea157bf0925d55ec51d1894.jpg](../icml_results/4660_DB-KSVD_%20Scalable%20Alternating%20Optimization%20for%20Disentangling%20High-Dimensional%20Embedding%20Spaces/images/9313185284f92770c424355a7ff664a655d5f39efea157bf0925d55ec51d1894.jpg)

![b58c0008e8cf067c428219f08716f14fd1cf92fd11f7a3bb9b32882b762578d7.jpg](../icml_results/4660_DB-KSVD_%20Scalable%20Alternating%20Optimization%20for%20Disentangling%20High-Dimensional%20Embedding%20Spaces/images/b58c0008e8cf067c428219f08716f14fd1cf92fd11f7a3bb9b32882b762578d7.jpg)

![bd47645f1b139d4776662127d44c396708e55ec526674f95e6c84a3e5a46dcc1.jpg](../icml_results/4660_DB-KSVD_%20Scalable%20Alternating%20Optimization%20for%20Disentangling%20High-Dimensional%20Embedding%20Spaces/images/bd47645f1b139d4776662127d44c396708e55ec526674f95e6c84a3e5a46dcc1.jpg)

![c8914b6a7bc6d8704b70b10ed36b61851feebade724747698a343d148e9d44d7.jpg](../icml_results/4660_DB-KSVD_%20Scalable%20Alternating%20Optimization%20for%20Disentangling%20High-Dimensional%20Embedding%20Spaces/images/c8914b6a7bc6d8704b70b10ed36b61851feebade724747698a343d148e9d44d7.jpg)

![e4467636b5000afba0e8b7da433c21e8d094fc5ffd4ffe324182978f191a7ac6.jpg](../icml_results/4660_DB-KSVD_%20Scalable%20Alternating%20Optimization%20for%20Disentangling%20High-Dimensional%20Embedding%20Spaces/images/e4467636b5000afba0e8b7da433c21e8d094fc5ffd4ffe324182978f191a7ac6.jpg)

### Tables

![03e000725e1baa61153690c0055dc842652fe670e2513b3aa010fa237c2ff3fa.jpg](../icml_results/4660_DB-KSVD_%20Scalable%20Alternating%20Optimization%20for%20Disentangling%20High-Dimensional%20Embedding%20Spaces/tables/03e000725e1baa61153690c0055dc842652fe670e2513b3aa010fa237c2ff3fa.jpg)

![094beab0990a25827202043bed0541ae2621e79a04a34c500a5e130cd362aebc.jpg](../icml_results/4660_DB-KSVD_%20Scalable%20Alternating%20Optimization%20for%20Disentangling%20High-Dimensional%20Embedding%20Spaces/tables/094beab0990a25827202043bed0541ae2621e79a04a34c500a5e130cd362aebc.jpg)

![51122cb3a7320973622286071a12c31ff77d9344cbe182bd66dabf3790e97c42.jpg](../icml_results/4660_DB-KSVD_%20Scalable%20Alternating%20Optimization%20for%20Disentangling%20High-Dimensional%20Embedding%20Spaces/tables/51122cb3a7320973622286071a12c31ff77d9344cbe182bd66dabf3790e97c42.jpg)

![a109a59a1ec871aadf5397fc8ac3493e857849c50869578757b298bcf550dc55.jpg](../icml_results/4660_DB-KSVD_%20Scalable%20Alternating%20Optimization%20for%20Disentangling%20High-Dimensional%20Embedding%20Spaces/tables/a109a59a1ec871aadf5397fc8ac3493e857849c50869578757b298bcf550dc55.jpg)

![e63ecdbeb8c6dfa7d2396d09aacb7c53be71e56f10f081d2a15e6605a3f5b46e.jpg](../icml_results/4660_DB-KSVD_%20Scalable%20Alternating%20Optimization%20for%20Disentangling%20High-Dimensional%20Embedding%20Spaces/tables/e63ecdbeb8c6dfa7d2396d09aacb7c53be71e56f10f081d2a15e6605a3f5b46e.jpg)

<a id="paper-4614"></a>

## Optimal Transport Group Counterfactual Explanations


### Images

![01bccf9f8356f9e1870c9d6153d469b40e34c2b569b29e309faed2bc95896e40.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/01bccf9f8356f9e1870c9d6153d469b40e34c2b569b29e309faed2bc95896e40.jpg)

![0566413a56efa6571adae2846ec03c1d77c36c58c9056c9dd136102d97a4782e.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/0566413a56efa6571adae2846ec03c1d77c36c58c9056c9dd136102d97a4782e.jpg)

![09d75d73a6f18a858ac769e482f55bb2bc0519e05ce9a90173d04f5b513ac0be.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/09d75d73a6f18a858ac769e482f55bb2bc0519e05ce9a90173d04f5b513ac0be.jpg)

![123d02f6e7cf757d7e777286022608a4356dbdb9d50b2fef66f996ca4354525d.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/123d02f6e7cf757d7e777286022608a4356dbdb9d50b2fef66f996ca4354525d.jpg)

![3df21b3b3bb4b3658896984f47b8035d6dabff51ae33391afbc7da30d9985eb5.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/3df21b3b3bb4b3658896984f47b8035d6dabff51ae33391afbc7da30d9985eb5.jpg)

![40347eb02432fa42eff5c49f8db7b28b560c4802d815e3ec28066338fa2f5905.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/40347eb02432fa42eff5c49f8db7b28b560c4802d815e3ec28066338fa2f5905.jpg)

![4c01f1fa94f229381df3a2e82fb2e22ba489f25027df50eac996321a091f6c5c.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/4c01f1fa94f229381df3a2e82fb2e22ba489f25027df50eac996321a091f6c5c.jpg)

![6291fb37c8ff17330f338dc377ab695c34b4eaa760e99c2c72c1be4f304317e7.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/6291fb37c8ff17330f338dc377ab695c34b4eaa760e99c2c72c1be4f304317e7.jpg)

![63128522298ebb753dcc97603beb6341fc47ae9bd3cac29a00bb5acb98e67aed.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/63128522298ebb753dcc97603beb6341fc47ae9bd3cac29a00bb5acb98e67aed.jpg)

![72e0dff82dad49e2d876b88c88cafc52dd9b4d945e125ad92d7becc56ef9fdbb.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/72e0dff82dad49e2d876b88c88cafc52dd9b4d945e125ad92d7becc56ef9fdbb.jpg)

![753b46f2711a6de5119c39355ab3e7eef7c8971cedd57a856860d911454b09e0.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/753b46f2711a6de5119c39355ab3e7eef7c8971cedd57a856860d911454b09e0.jpg)

![761babe17696c7878ff0ca96ec9a64b984e40ac18d10c992f4ac6d10eb75237b.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/761babe17696c7878ff0ca96ec9a64b984e40ac18d10c992f4ac6d10eb75237b.jpg)

![82d07e36f4df5819fb87365433dd8b6fe0d0b0075b0cf12941ceaeda71c4fea6.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/82d07e36f4df5819fb87365433dd8b6fe0d0b0075b0cf12941ceaeda71c4fea6.jpg)

![8868a56ef91b9be1906c44600ee2ec6da5cd2ccb36431b865c624773ea2860bc.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/8868a56ef91b9be1906c44600ee2ec6da5cd2ccb36431b865c624773ea2860bc.jpg)

![90723304b3c4265b958b2cf94348cc446a8a0fc22559edcb9e422c0b8e6b2f35.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/90723304b3c4265b958b2cf94348cc446a8a0fc22559edcb9e422c0b8e6b2f35.jpg)

![91fe9cfc0c7e12a4890296d3c3a21461b424f7dbb6bb0385039600e86f855ba4.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/91fe9cfc0c7e12a4890296d3c3a21461b424f7dbb6bb0385039600e86f855ba4.jpg)

![923772f047279e4515290818e34df822440c0cebdb68ea717cced5827faa406c.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/923772f047279e4515290818e34df822440c0cebdb68ea717cced5827faa406c.jpg)

![b70afa8b5ef37bd26cfa4b967fa626365c2476d93ad8dd969990bcc54e9bed7a.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/b70afa8b5ef37bd26cfa4b967fa626365c2476d93ad8dd969990bcc54e9bed7a.jpg)

![c91d8d4d01c9b1ab6f9c018a334cea5dc2204e2d3f355193ea54cdf5309a59d7.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/c91d8d4d01c9b1ab6f9c018a334cea5dc2204e2d3f355193ea54cdf5309a59d7.jpg)

![cf0c33136b77f70bd0b583caa357fe386ecaf7f9245c1f2590958c0a09d6d79f.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/cf0c33136b77f70bd0b583caa357fe386ecaf7f9245c1f2590958c0a09d6d79f.jpg)

![d00c0af55097312b9d919520c6085f5fbd61c575c764b26e12a855565c8f4148.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/d00c0af55097312b9d919520c6085f5fbd61c575c764b26e12a855565c8f4148.jpg)

![d441cf6097141380941ebf76d83c0199ff7c351de1e2e1ef16b94dfb8f05bdf0.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/d441cf6097141380941ebf76d83c0199ff7c351de1e2e1ef16b94dfb8f05bdf0.jpg)

![ed1e829c5dd066d371c86e96cde4d82d89b04a4478919d33ba76468b83a735d2.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/images/ed1e829c5dd066d371c86e96cde4d82d89b04a4478919d33ba76468b83a735d2.jpg)

### Tables

![581a884827363c1153868bc9675dd661465d2d5b788cce7d4a86f87a66049e06.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/tables/581a884827363c1153868bc9675dd661465d2d5b788cce7d4a86f87a66049e06.jpg)

![60a981ac3926c1e98e0c3aa23d41c07e83ff23b4af5aca2759757dc9ee5751dc.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/tables/60a981ac3926c1e98e0c3aa23d41c07e83ff23b4af5aca2759757dc9ee5751dc.jpg)

![b7d55a0ef5c4c3aa7225b441b71886effd2791eaa1ce2d00bd27483595673e5f.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/tables/b7d55a0ef5c4c3aa7225b441b71886effd2791eaa1ce2d00bd27483595673e5f.jpg)

![c74ec93d78f8b74d16312859ac2843afce44a17f9fe282054d0ab8b47932fd18.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/tables/c74ec93d78f8b74d16312859ac2843afce44a17f9fe282054d0ab8b47932fd18.jpg)

![ef306e2d990cb16a7ae8d5afd43ed7e259bd937ce45c9bbe8b7486cea283c194.jpg](../icml_results/4661_Optimal%20Transport%20Group%20Counterfactual%20Explanations/tables/ef306e2d990cb16a7ae8d5afd43ed7e259bd937ce45c9bbe8b7486cea283c194.jpg)

<a id="paper-4615"></a>

## FPTQuant: Function-Preserving Transforms for LLM Quantization


### Images

![207c688f2837dcfdd7f507791b18fb91eaf6beef7e9fb1283615369c908e9297.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/images/207c688f2837dcfdd7f507791b18fb91eaf6beef7e9fb1283615369c908e9297.jpg)

![73a8fb5ddb5bf17256d9e7fb1abf0b8edc73f79a24ecebff881888af61d99b03.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/images/73a8fb5ddb5bf17256d9e7fb1abf0b8edc73f79a24ecebff881888af61d99b03.jpg)

![77f4080a3dc47cb2ad87490413310807f336fd03b8cae2470ea0258d22e2aad4.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/images/77f4080a3dc47cb2ad87490413310807f336fd03b8cae2470ea0258d22e2aad4.jpg)

![925be26135b5f94e3b43d362f2adedf74342879467d535c4dc09e44253398687.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/images/925be26135b5f94e3b43d362f2adedf74342879467d535c4dc09e44253398687.jpg)

![9e054a4adbb045f95c215d9d1db9d307b71db64dcb979102a93fc287bc09657b.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/images/9e054a4adbb045f95c215d9d1db9d307b71db64dcb979102a93fc287bc09657b.jpg)

![9fbc13c9594c2265bd916c21edccdf09088564d3b9a9bde9c0fdcde815a7c848.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/images/9fbc13c9594c2265bd916c21edccdf09088564d3b9a9bde9c0fdcde815a7c848.jpg)

![b552dd3cd4d37ead9b73a925a3203abb72f6920f454a9a3cd875e5f81c7df6a5.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/images/b552dd3cd4d37ead9b73a925a3203abb72f6920f454a9a3cd875e5f81c7df6a5.jpg)

### Tables

![1b2ab6080be300663f7a12298e70412b2cc6d1e21766e9c89dc32a2d25fb3d85.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/1b2ab6080be300663f7a12298e70412b2cc6d1e21766e9c89dc32a2d25fb3d85.jpg)

![253a9064fdfcf9cfff27dc4a95431111db051f9d1f00bbebd12e95d1dd68107f.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/253a9064fdfcf9cfff27dc4a95431111db051f9d1f00bbebd12e95d1dd68107f.jpg)

![28decf36cb25ca405dcbcc3221f39fe264aa3305f82436d7320d0cf05740e3f5.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/28decf36cb25ca405dcbcc3221f39fe264aa3305f82436d7320d0cf05740e3f5.jpg)

![31fc9ce9458144ce921b426e1eafa40ec75382543ba23e4d5ec2cfd431998442.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/31fc9ce9458144ce921b426e1eafa40ec75382543ba23e4d5ec2cfd431998442.jpg)

![3fd21a40ae5f8c0aea78176ed7c01dba5b0c3b5860e5f6d2663f8689db3c734b.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/3fd21a40ae5f8c0aea78176ed7c01dba5b0c3b5860e5f6d2663f8689db3c734b.jpg)

![52efa07e027e60a84f2a30d057ec3d54b3dc27a630a2931fed1c3afc4c371e74.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/52efa07e027e60a84f2a30d057ec3d54b3dc27a630a2931fed1c3afc4c371e74.jpg)

![59bdd4c4466d57777fd05f33c8deb3e734dfca788ad04c063dd01d24184f45ee.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/59bdd4c4466d57777fd05f33c8deb3e734dfca788ad04c063dd01d24184f45ee.jpg)

![5d8e705082522547723a5758650cb5a41477a50a3a0d0b6a183f6afec93dea03.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/5d8e705082522547723a5758650cb5a41477a50a3a0d0b6a183f6afec93dea03.jpg)

![7b271bfdb50b9968540008e98cd3ee08ca767e5559bb43fbf59ab418f8d8d348.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/7b271bfdb50b9968540008e98cd3ee08ca767e5559bb43fbf59ab418f8d8d348.jpg)

![7de15c1b99a1a4d8763a9fe672d1b1649da200578972ac3aca816981726bb96e.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/7de15c1b99a1a4d8763a9fe672d1b1649da200578972ac3aca816981726bb96e.jpg)

![825464fb808667b24f42013d37fac996b5d532ec29216cca853ce03570928185.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/825464fb808667b24f42013d37fac996b5d532ec29216cca853ce03570928185.jpg)

![88384c9089352a8bed89fdf31af11de2a2f745f1d4764aabbb47fa7e878a20ef.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/88384c9089352a8bed89fdf31af11de2a2f745f1d4764aabbb47fa7e878a20ef.jpg)

![976c22ab5ddaa41e28a0488277afc80e99a90100a2ae7dddf7ab0f098239f1b3.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/976c22ab5ddaa41e28a0488277afc80e99a90100a2ae7dddf7ab0f098239f1b3.jpg)

![a7c4d2222d96906435e8bb737967e675dbfbde91b65ac108515d95b4d6a6aa68.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/a7c4d2222d96906435e8bb737967e675dbfbde91b65ac108515d95b4d6a6aa68.jpg)

![b026c5638fbe1d02eb6de30635c129325a77590745a8032320ed4d30de4463be.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/b026c5638fbe1d02eb6de30635c129325a77590745a8032320ed4d30de4463be.jpg)

![c61cebee23fea2ffa02c8f49b5ee192ded700ed28a49f3cc6279051138764b48.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/c61cebee23fea2ffa02c8f49b5ee192ded700ed28a49f3cc6279051138764b48.jpg)

![ddc928f3ee1fadd2917e114846fc3ae937510be335818ffa04443cf9d37cc679.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/ddc928f3ee1fadd2917e114846fc3ae937510be335818ffa04443cf9d37cc679.jpg)

![e405db3889c365ec395bb3bee2d99aa72df4eb54f95cb8dbbd0d221531ba7484.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/e405db3889c365ec395bb3bee2d99aa72df4eb54f95cb8dbbd0d221531ba7484.jpg)

![e839565f683cb2ce5e377a1851898316993645bdcfe459a2511833bee15d49bf.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/e839565f683cb2ce5e377a1851898316993645bdcfe459a2511833bee15d49bf.jpg)

![fd25f4621595f8512a1e932822284cd4db2dd3a2945b5cde58e5934cc77d2dd5.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/fd25f4621595f8512a1e932822284cd4db2dd3a2945b5cde58e5934cc77d2dd5.jpg)

![fdec76c704247f09594c789e4c59e1e32be79812120b04603c6aec719fc39223.jpg](../icml_results/4662_FPTQuant_%20Function-Preserving%20Transforms%20for%20LLM%20Quantization/tables/fdec76c704247f09594c789e4c59e1e32be79812120b04603c6aec719fc39223.jpg)

<a id="paper-4616"></a>

## DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning


### Images

![171f8634efd5a10295a3c77472086794b9f311c4118a26c8a9cf5d6bda65e019.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/images/171f8634efd5a10295a3c77472086794b9f311c4118a26c8a9cf5d6bda65e019.jpg)

![28d6347485994f159795df3e59670a08e89e17fa4cb768f16b81b08ad071061a.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/images/28d6347485994f159795df3e59670a08e89e17fa4cb768f16b81b08ad071061a.jpg)

![3258c6c108e3a9f1eece16957161c0bcad4d7d6362dd3b49ec4e8f548341630c.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/images/3258c6c108e3a9f1eece16957161c0bcad4d7d6362dd3b49ec4e8f548341630c.jpg)

![63912da7ac9349f4d8b90a544c82226bdbd1fcb5454fe531b2dce90b2affcb76.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/images/63912da7ac9349f4d8b90a544c82226bdbd1fcb5454fe531b2dce90b2affcb76.jpg)

![652d7033a2c2b5c54b8a4739b86c6a5904275ff4addbe68857c81242f5e42aba.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/images/652d7033a2c2b5c54b8a4739b86c6a5904275ff4addbe68857c81242f5e42aba.jpg)

![9157b05fbd898484c996fdd0eebec7dbb0267fd6626d5867fc4ec2629305c42b.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/images/9157b05fbd898484c996fdd0eebec7dbb0267fd6626d5867fc4ec2629305c42b.jpg)

![b52734d91ce745bf033fd50f893359da8640951ed3c1e3e364560329b33d2008.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/images/b52734d91ce745bf033fd50f893359da8640951ed3c1e3e364560329b33d2008.jpg)

![b5acddc217b1e10498efd92998168cd9346b842ca46604a40cd6097ce792890d.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/images/b5acddc217b1e10498efd92998168cd9346b842ca46604a40cd6097ce792890d.jpg)

![cc9badf621e7f68f5ed0e7690d066a62d49c220889fcb910d21a59cdb07a3342.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/images/cc9badf621e7f68f5ed0e7690d066a62d49c220889fcb910d21a59cdb07a3342.jpg)

![de5f5a334baf037b9f6e52c52c7a4ce9d67505ce4633c05285d05d9ec2c78444.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/images/de5f5a334baf037b9f6e52c52c7a4ce9d67505ce4633c05285d05d9ec2c78444.jpg)

![efd32c23b63fae62afc754d1cb0ec662691f49a37232171a12f5da8b76b459fd.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/images/efd32c23b63fae62afc754d1cb0ec662691f49a37232171a12f5da8b76b459fd.jpg)

### Tables

![090c6f600a33c2bdd43c6034715d597bdc88bce33e7d4f0382610f0318b56894.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/tables/090c6f600a33c2bdd43c6034715d597bdc88bce33e7d4f0382610f0318b56894.jpg)

![184c8a20f56e294c244438768e8ba80151ee2c5b2841cde29a91f33df425d8f3.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/tables/184c8a20f56e294c244438768e8ba80151ee2c5b2841cde29a91f33df425d8f3.jpg)

![1f0da6c8142ad51e8c135da1a14ffe3dad139411f0f09d1d5483c79c52c1da03.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/tables/1f0da6c8142ad51e8c135da1a14ffe3dad139411f0f09d1d5483c79c52c1da03.jpg)

![36f9ef12c03f966f1e793fbaea0ba407ad86718acd5e3f1cda722e1d54c3d7f6.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/tables/36f9ef12c03f966f1e793fbaea0ba407ad86718acd5e3f1cda722e1d54c3d7f6.jpg)

![452b82609f5fca2bbbb9a193572f3c60175038e4f4ea61e24e38c19987756754.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/tables/452b82609f5fca2bbbb9a193572f3c60175038e4f4ea61e24e38c19987756754.jpg)

![477c41c0a242ef0aca9a9912abc10da09ab5277fc49c5f41b54f10b57e6a3b91.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/tables/477c41c0a242ef0aca9a9912abc10da09ab5277fc49c5f41b54f10b57e6a3b91.jpg)

![c7599e0b05c894a8b2e3b18abd11b9dd2262fce3bf427897f2ee2f3d2cfb8095.jpg](../icml_results/4663_DP-KFC_%20Data-Free%20Preconditioning%20for%20Privacy-Preserving%20Deep%20Learning/tables/c7599e0b05c894a8b2e3b18abd11b9dd2262fce3bf427897f2ee2f3d2cfb8095.jpg)

<a id="paper-4617"></a>

## Neural Low-Discrepancy Sequences


### Images

![15e8dcc294f3562fea735ac718c9d61ec154c626159825f0a099a145c893c9bc.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/images/15e8dcc294f3562fea735ac718c9d61ec154c626159825f0a099a145c893c9bc.jpg)

![25e78f3e0040cb7cb3e078888135f8228aee190a7b6ce09e3e2cfc1335e2b13f.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/images/25e78f3e0040cb7cb3e078888135f8228aee190a7b6ce09e3e2cfc1335e2b13f.jpg)

![3b09ad089a38cf29c0bf8bf4d1089546527c63ddb02f1cb7c05eece00d9246e2.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/images/3b09ad089a38cf29c0bf8bf4d1089546527c63ddb02f1cb7c05eece00d9246e2.jpg)

![5adc4b776269066e95eddcff30155f098fe5281f0d02298f27479b73b34d260a.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/images/5adc4b776269066e95eddcff30155f098fe5281f0d02298f27479b73b34d260a.jpg)

![6926bff96fc39f6deccec5268c4ee4ae33c696c430ec1c3866c1beffb05219c1.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/images/6926bff96fc39f6deccec5268c4ee4ae33c696c430ec1c3866c1beffb05219c1.jpg)

![9e45e120b1e5ade9fe08dad20befdc059a6f694528e75ff7653c25f6958ce09e.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/images/9e45e120b1e5ade9fe08dad20befdc059a6f694528e75ff7653c25f6958ce09e.jpg)

![a2f6fc6ce17487634c985c3c095020d0fe5800b731fbef9a23eb0d7c159c4560.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/images/a2f6fc6ce17487634c985c3c095020d0fe5800b731fbef9a23eb0d7c159c4560.jpg)

![a50c8b65d7c1e2db5075bdd506a339ef56963585073749d7046ceeafed51b905.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/images/a50c8b65d7c1e2db5075bdd506a339ef56963585073749d7046ceeafed51b905.jpg)

![b4c9c0f17f79e2f7bce75466d1ef037ce60d49738e712b95f4a3c26106f2d4e8.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/images/b4c9c0f17f79e2f7bce75466d1ef037ce60d49738e712b95f4a3c26106f2d4e8.jpg)

![bb05c692d44b560d6bff7352582e2878c3c7c5a40f342e7b07945d99cd0c9b9b.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/images/bb05c692d44b560d6bff7352582e2878c3c7c5a40f342e7b07945d99cd0c9b9b.jpg)

![efa64291f3ee7f504f503cca86d2cb336b4cf8d5e00a8f45ccf22e32bf6b16c5.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/images/efa64291f3ee7f504f503cca86d2cb336b4cf8d5e00a8f45ccf22e32bf6b16c5.jpg)

![f41d8024b5619c8f19050556727cd99ccc820dd08082f356e06ac1ba74b656ad.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/images/f41d8024b5619c8f19050556727cd99ccc820dd08082f356e06ac1ba74b656ad.jpg)

### Tables

![1e536ca0a8db7f0cc843c66a023e81e9a725536c388dfaeb5f77508077015c59.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/tables/1e536ca0a8db7f0cc843c66a023e81e9a725536c388dfaeb5f77508077015c59.jpg)

![29a63ec8ca9ab30719765b1dbc54dcf15e50a5b8e9e1bb79bfdc1bcb7cb2dac5.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/tables/29a63ec8ca9ab30719765b1dbc54dcf15e50a5b8e9e1bb79bfdc1bcb7cb2dac5.jpg)

![4dad09330db5a2f8da539cb111c8192d0c906cc3ca50ed003595b48309d49f0d.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/tables/4dad09330db5a2f8da539cb111c8192d0c906cc3ca50ed003595b48309d49f0d.jpg)

![6a3a5cd2e609ae17db819dfee0b30c4758e3e620f4372596613af2f6263211c3.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/tables/6a3a5cd2e609ae17db819dfee0b30c4758e3e620f4372596613af2f6263211c3.jpg)

![c8b94f41acacd7f70a6426654381d69ed0a67dd64ce2cd13fc306d91df5eae61.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/tables/c8b94f41acacd7f70a6426654381d69ed0a67dd64ce2cd13fc306d91df5eae61.jpg)

![ea2582c5136a7683caac733923e876e6f3a5d847f37736da6f3b3f24b28cf8ec.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/tables/ea2582c5136a7683caac733923e876e6f3a5d847f37736da6f3b3f24b28cf8ec.jpg)

![f2a55c358c8a91f055d42c85dc44cf3c98ef85b6704eb1991910dffee3f007e3.jpg](../icml_results/4664_Neural%20Low-Discrepancy%20Sequences/tables/f2a55c358c8a91f055d42c85dc44cf3c98ef85b6704eb1991910dffee3f007e3.jpg)

<a id="paper-4618"></a>

## Stein Diffusion Guidance: Training-Free Posterior Correction for Sampling Beyond High-Density Regions


### Images

![04255d565707b60de8e6d0acf39a430faf2489cc3f711bafaff689e56933d65a.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/04255d565707b60de8e6d0acf39a430faf2489cc3f711bafaff689e56933d65a.jpg)

![12b903b479e8e3ce16e1dad72a900ba632a72d2dc2ca0dfb06e884c86b6959b5.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/12b903b479e8e3ce16e1dad72a900ba632a72d2dc2ca0dfb06e884c86b6959b5.jpg)

![194debd73c0bd90d5894fa2ebd13d0ea6ac52b1f6650600cf558587702386dff.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/194debd73c0bd90d5894fa2ebd13d0ea6ac52b1f6650600cf558587702386dff.jpg)

![1e3fedf9d2fe129fda4b5208b5ed71d142187e6927eef562870e0b0007e6e4a7.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/1e3fedf9d2fe129fda4b5208b5ed71d142187e6927eef562870e0b0007e6e4a7.jpg)

![1fa98fdc612184c8ff248b99e646206da4758685e195f65588412821ea9449a4.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/1fa98fdc612184c8ff248b99e646206da4758685e195f65588412821ea9449a4.jpg)

![30ad7a8edd6b0ac14623d1a6ef4dba717114854946043fc976e8950f6fbe5a48.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/30ad7a8edd6b0ac14623d1a6ef4dba717114854946043fc976e8950f6fbe5a48.jpg)

![3f32e28469f157c3d33ac244dbd6158a21efed79ce076aac8379ebcc2b6f9e20.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/3f32e28469f157c3d33ac244dbd6158a21efed79ce076aac8379ebcc2b6f9e20.jpg)

![4b9a89ea402b1b62c2bd652647afaf8505cd6e1890ec0d09f0f725a61c9bf8e0.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/4b9a89ea402b1b62c2bd652647afaf8505cd6e1890ec0d09f0f725a61c9bf8e0.jpg)

![5e69b30fcf5f0bcb2c5eafabbb5fbd3949528881d04720ad478078da84b0bc7c.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/5e69b30fcf5f0bcb2c5eafabbb5fbd3949528881d04720ad478078da84b0bc7c.jpg)

![65015e8807518751400d4889be8c8c7349418ac828bc7c60b0719b7940c27df2.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/65015e8807518751400d4889be8c8c7349418ac828bc7c60b0719b7940c27df2.jpg)

![665f32588b36a012a29854431218e19b2c50cfb522e0668cf089fa89bfcccff9.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/665f32588b36a012a29854431218e19b2c50cfb522e0668cf089fa89bfcccff9.jpg)

![69e8ffadc63971557ce51d0a2f9bcc64e46e6952aa021693694ecab632345089.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/69e8ffadc63971557ce51d0a2f9bcc64e46e6952aa021693694ecab632345089.jpg)

![718ae9fdb664c01b6dee90fd627393a5e259353192bc425819065747aad79791.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/718ae9fdb664c01b6dee90fd627393a5e259353192bc425819065747aad79791.jpg)

![83a0c8348ae2147dca50520ed4235f6e8e0bf96579e8192c54a38736218bd84c.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/83a0c8348ae2147dca50520ed4235f6e8e0bf96579e8192c54a38736218bd84c.jpg)

![a2245141ae07b9592609bcb2f10ccb5eb97beb00f95c52864b994dc1bdd30601.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/a2245141ae07b9592609bcb2f10ccb5eb97beb00f95c52864b994dc1bdd30601.jpg)

![a395d4ecab2c8b519e9b5e3cf542d40267bb58066328f5194b8e6a3e53e09120.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/a395d4ecab2c8b519e9b5e3cf542d40267bb58066328f5194b8e6a3e53e09120.jpg)

![a4d7572ed1952342152e75d1263e98575a962e64e31dfdda6a7703f96650c455.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/a4d7572ed1952342152e75d1263e98575a962e64e31dfdda6a7703f96650c455.jpg)

![b50bbe21d111f4cfcc54c11f02f22d82219aa42e19cbe7ad85873dce706a7477.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/b50bbe21d111f4cfcc54c11f02f22d82219aa42e19cbe7ad85873dce706a7477.jpg)

![bdd48bb179fdc6c2e7e4adc4862937e59af8b207b63a4c70164e7b6f565e9e45.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/bdd48bb179fdc6c2e7e4adc4862937e59af8b207b63a4c70164e7b6f565e9e45.jpg)

![ca4f712ab43546b7ad64cdf37a0cc799522ac0dcdbac35aee83d3d35b384c22e.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/ca4f712ab43546b7ad64cdf37a0cc799522ac0dcdbac35aee83d3d35b384c22e.jpg)

![e4c0d6cb04951dea696e37905045b14238203da5be08b98ed429097983ca0ed1.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/e4c0d6cb04951dea696e37905045b14238203da5be08b98ed429097983ca0ed1.jpg)

![f47713cb0a5e268f3d22a694dfa52b27ae4e4a0a8426c10ce2c853c952c273ae.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/f47713cb0a5e268f3d22a694dfa52b27ae4e4a0a8426c10ce2c853c952c273ae.jpg)

![f6413e33b6a01ee5d127e9fd2b4f471dce78a791c98e3baf725c73e1d5e372f4.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/images/f6413e33b6a01ee5d127e9fd2b4f471dce78a791c98e3baf725c73e1d5e372f4.jpg)

### Tables

![04187ebc895eb017d5610f168a9d682c32dd48ee991e2e0246761cd8c6e921fb.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/tables/04187ebc895eb017d5610f168a9d682c32dd48ee991e2e0246761cd8c6e921fb.jpg)

![2bf9f7433e074387677ae27902c1d2f3e59be54646722c943d10b1f280aa637e.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/tables/2bf9f7433e074387677ae27902c1d2f3e59be54646722c943d10b1f280aa637e.jpg)

![52f6adb800542762e0615acae77a9065186996761af8115189b983c179f87b19.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/tables/52f6adb800542762e0615acae77a9065186996761af8115189b983c179f87b19.jpg)

![66c7800038ba0af5ab8bd2930ecdc38b11569e4ca5f9e172a9c5e5fed40ca0ab.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/tables/66c7800038ba0af5ab8bd2930ecdc38b11569e4ca5f9e172a9c5e5fed40ca0ab.jpg)

![77a1de9eec8fb7597a165318baf07fbc7ffa8bf374b90d7361e051c7b7750463.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/tables/77a1de9eec8fb7597a165318baf07fbc7ffa8bf374b90d7361e051c7b7750463.jpg)

![8af987457bba3fe6e7b8968c834cf2567d8823d39d5d21987b9edb637a222ca7.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/tables/8af987457bba3fe6e7b8968c834cf2567d8823d39d5d21987b9edb637a222ca7.jpg)

![95f3eeaf2d799c9b8411b4db5635d6e3f33d4ce4650a5cc4bf113df1f899f24e.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/tables/95f3eeaf2d799c9b8411b4db5635d6e3f33d4ce4650a5cc4bf113df1f899f24e.jpg)

![eb80d88d4630ed7ea1b2225406a9913206dc26eeb86445125e6c1c5e2c902632.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/tables/eb80d88d4630ed7ea1b2225406a9913206dc26eeb86445125e6c1c5e2c902632.jpg)

![f8233675abed0cf918e1ffaa2c03320dca30cf61879e1f7d89afc97e4d31f9cc.jpg](../icml_results/4665_Stein%20Diffusion%20Guidance_%20Training-Free%20Posterior%20Correction%20for%20Sampling%20Beyond%20High-Density%20Region/tables/f8233675abed0cf918e1ffaa2c03320dca30cf61879e1f7d89afc97e4d31f9cc.jpg)

<a id="paper-4619"></a>

## Variance-Reduced $(\varepsilon, δ)-$Unlearning using Forget Set Gradients


### Images

![47850595ea077d182cd2cb1ee8c4a2b6fc6927d9066a90564a1b442cd221da1d.jpg](../icml_results/4666_Variance-Reduced%20%24%28_varepsilon%2C%20%CE%B4%29-%24Unlearning%20using%20Forget%20Set%20Gradients/images/47850595ea077d182cd2cb1ee8c4a2b6fc6927d9066a90564a1b442cd221da1d.jpg)

![8fe76dcc5e9727f01922a61ee15568cec603ef34943f4280c03eeba5872d8ec5.jpg](../icml_results/4666_Variance-Reduced%20%24%28_varepsilon%2C%20%CE%B4%29-%24Unlearning%20using%20Forget%20Set%20Gradients/images/8fe76dcc5e9727f01922a61ee15568cec603ef34943f4280c03eeba5872d8ec5.jpg)

![9b55ae00f8f7d36cc3d3776e0f53b2b9d1281522e3996ebdbc8858ddd1effef7.jpg](../icml_results/4666_Variance-Reduced%20%24%28_varepsilon%2C%20%CE%B4%29-%24Unlearning%20using%20Forget%20Set%20Gradients/images/9b55ae00f8f7d36cc3d3776e0f53b2b9d1281522e3996ebdbc8858ddd1effef7.jpg)

![fe608672636e238e5fa21ae2103cc5dfc3a09baca112f60922593575435c313c.jpg](../icml_results/4666_Variance-Reduced%20%24%28_varepsilon%2C%20%CE%B4%29-%24Unlearning%20using%20Forget%20Set%20Gradients/images/fe608672636e238e5fa21ae2103cc5dfc3a09baca112f60922593575435c313c.jpg)

### Tables

![147eb6a7507cdbd58088d5bf5e9b1f973b345963d688265d6604849158ed4b0d.jpg](../icml_results/4666_Variance-Reduced%20%24%28_varepsilon%2C%20%CE%B4%29-%24Unlearning%20using%20Forget%20Set%20Gradients/tables/147eb6a7507cdbd58088d5bf5e9b1f973b345963d688265d6604849158ed4b0d.jpg)

![d03b851cf8e5deb30d86df466e9d7fcb60eb0cd934bcbdce6c0271047021da6c.jpg](../icml_results/4666_Variance-Reduced%20%24%28_varepsilon%2C%20%CE%B4%29-%24Unlearning%20using%20Forget%20Set%20Gradients/tables/d03b851cf8e5deb30d86df466e9d7fcb60eb0cd934bcbdce6c0271047021da6c.jpg)

![f7f9ce8918d288135522bd7e91deb03dd8c558114c5f0964ee1049f56b5c7f9a.jpg](../icml_results/4666_Variance-Reduced%20%24%28_varepsilon%2C%20%CE%B4%29-%24Unlearning%20using%20Forget%20Set%20Gradients/tables/f7f9ce8918d288135522bd7e91deb03dd8c558114c5f0964ee1049f56b5c7f9a.jpg)

<a id="paper-4620"></a>

## Symbal: Detecting Systematic Misalignments in Model-Generated Captions

### Images

![05e268276297b792ce725b30142410bc5ea3b136b09061f3b1743a0938951251.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/05e268276297b792ce725b30142410bc5ea3b136b09061f3b1743a0938951251.jpg)

![116e2463d147b36a75f7ae9bf05956f1b283f34cc65def45b4c63b6da337803b.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/116e2463d147b36a75f7ae9bf05956f1b283f34cc65def45b4c63b6da337803b.jpg)

![14817e186ed180ac9458b217a197e207870bfd952e7a39c390f2ab822e5f35a2.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/14817e186ed180ac9458b217a197e207870bfd952e7a39c390f2ab822e5f35a2.jpg)

![14b2239d14e31cdb0c12090c58577e5adcb495b6152ad76164404dba2824837e.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/14b2239d14e31cdb0c12090c58577e5adcb495b6152ad76164404dba2824837e.jpg)

![1819b78e5307a4ab2b927257498d1f57c6f89951f3313c51f6f798e94dd7e449.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/1819b78e5307a4ab2b927257498d1f57c6f89951f3313c51f6f798e94dd7e449.jpg)

![182ea1d2e77c850dc6afc84893ed7144b814f40bd7ff942c9eb740eb55103e0e.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/182ea1d2e77c850dc6afc84893ed7144b814f40bd7ff942c9eb740eb55103e0e.jpg)

![1efb28c91541af639208b68c5cb97f3653d1b8da3dc247a7c6d6486b1f368a5e.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/1efb28c91541af639208b68c5cb97f3653d1b8da3dc247a7c6d6486b1f368a5e.jpg)

![207e2baeb8c53294ba8a2ae1fc335f513d7eef1e95150045c821f5029b9c5a09.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/207e2baeb8c53294ba8a2ae1fc335f513d7eef1e95150045c821f5029b9c5a09.jpg)

![21c51ea0eadecda6557527d7c6601231f9fb75ab86fd28e6bc05856c7efc45ae.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/21c51ea0eadecda6557527d7c6601231f9fb75ab86fd28e6bc05856c7efc45ae.jpg)

![28d9143fd8f4667e6e9aa3889b0b776a56f41bc48afccb69f5e9745017d1f44b.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/28d9143fd8f4667e6e9aa3889b0b776a56f41bc48afccb69f5e9745017d1f44b.jpg)

![3116089fe7121db001403d8fea2f5c01fc79dc764de40d29b8e6c72b18edb7bc.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/3116089fe7121db001403d8fea2f5c01fc79dc764de40d29b8e6c72b18edb7bc.jpg)

![386b4ce2d24805afce0c2d543b3a9d74aa66161c0d6d08665569a5c0f458284f.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/386b4ce2d24805afce0c2d543b3a9d74aa66161c0d6d08665569a5c0f458284f.jpg)

![412eb21a0443acb1277ed26915eb3d21ddd03434601e630f637779447725193f.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/412eb21a0443acb1277ed26915eb3d21ddd03434601e630f637779447725193f.jpg)

![434bb4d7f8c5f22989c2df0c2aea8d62a1f3386fcd84e73f5bce01270767f82e.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/434bb4d7f8c5f22989c2df0c2aea8d62a1f3386fcd84e73f5bce01270767f82e.jpg)

![47948776d2a690c515c8337fc10ad2532e43308cee1f4788fcc6b5066fea2fd7.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/47948776d2a690c515c8337fc10ad2532e43308cee1f4788fcc6b5066fea2fd7.jpg)

![47daee695571ca256f7be87df8386f080033da04bf401552340bfce8a2888d7c.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/47daee695571ca256f7be87df8386f080033da04bf401552340bfce8a2888d7c.jpg)

![4a25baadf32eb3859553dff17ec7c566780223ac29fd4fcdad8a44e42495a0dc.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/4a25baadf32eb3859553dff17ec7c566780223ac29fd4fcdad8a44e42495a0dc.jpg)

![517cb5336fdc093ec507b983f803ec6a8fd9f34d5dfeef62d87b78e4a3fcfa8b.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/517cb5336fdc093ec507b983f803ec6a8fd9f34d5dfeef62d87b78e4a3fcfa8b.jpg)

![58a6ecb914e234370481f3ab932cd74c873ab6f1ffc530447fd700f644ea5cab.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/58a6ecb914e234370481f3ab932cd74c873ab6f1ffc530447fd700f644ea5cab.jpg)

![58cf14cc6f7915901c06604eaf28cea0b3f55a1e1f842d33907f1d9d1d43769c.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/58cf14cc6f7915901c06604eaf28cea0b3f55a1e1f842d33907f1d9d1d43769c.jpg)

![779db65c27984745b01f1eaa9d5f91f4cc6e1bffff30a1071b1f4fb9b226bb78.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/779db65c27984745b01f1eaa9d5f91f4cc6e1bffff30a1071b1f4fb9b226bb78.jpg)

![79862ed3397b316a640d4aebcc0d915f2d5680d540872ace815cb42d989ba948.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/79862ed3397b316a640d4aebcc0d915f2d5680d540872ace815cb42d989ba948.jpg)

![83906a08e04fbe82319aba8d0636add8f6c0c6b2ab71479bfa4b2dffdc2c4150.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/83906a08e04fbe82319aba8d0636add8f6c0c6b2ab71479bfa4b2dffdc2c4150.jpg)

![883e35ae95d0c17f392edfcc1e891b0f5a0293f0c7fe2ed2c0efcc0915f7c6e1.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/883e35ae95d0c17f392edfcc1e891b0f5a0293f0c7fe2ed2c0efcc0915f7c6e1.jpg)

![89f58526011ef1cc257e54ff7bb3911773a0676b463e1bb1572764df07c29aad.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/89f58526011ef1cc257e54ff7bb3911773a0676b463e1bb1572764df07c29aad.jpg)

![8b7a9943d035d4e03e98eb0f2ca632970207d7634b313f43d3c6e18031bea613.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/8b7a9943d035d4e03e98eb0f2ca632970207d7634b313f43d3c6e18031bea613.jpg)

![8f95364aa7894ec1e64bc707e8fcb41014712e3f4d6a342afacfcfb0032fc763.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/8f95364aa7894ec1e64bc707e8fcb41014712e3f4d6a342afacfcfb0032fc763.jpg)

![9528e2c22674677335079b955e718020dd9880afa417574bed0499ed502476da.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/9528e2c22674677335079b955e718020dd9880afa417574bed0499ed502476da.jpg)

![958c08eb955cf23a1e58b05dfd19b0e3cddcba8f0e2b2b911a91285537e6effe.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/958c08eb955cf23a1e58b05dfd19b0e3cddcba8f0e2b2b911a91285537e6effe.jpg)

![97cdd508ef3ae742bf77585550dd08920e906201e0f97fffb3a1e61a047a6c98.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/97cdd508ef3ae742bf77585550dd08920e906201e0f97fffb3a1e61a047a6c98.jpg)

![9bebb6cb423d6c38c4d57cff6494d18ad0d0e02349f46a8b2512b5a21b533071.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/9bebb6cb423d6c38c4d57cff6494d18ad0d0e02349f46a8b2512b5a21b533071.jpg)

![9dd6da29a53832136e06e53ba54c32dae73ac4c49d67395b950a123832502e56.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/9dd6da29a53832136e06e53ba54c32dae73ac4c49d67395b950a123832502e56.jpg)

![9efd768c9735ad9c7d1cf97363c63fc2308a081857b2419ebb6d9c4712829a67.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/9efd768c9735ad9c7d1cf97363c63fc2308a081857b2419ebb6d9c4712829a67.jpg)

![9f7ada97dff6c9103dd0e7032e9a877dc29d82e83699cc74c9ffb30cb05ee251.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/9f7ada97dff6c9103dd0e7032e9a877dc29d82e83699cc74c9ffb30cb05ee251.jpg)

![a5cbe8333fd2eada57d7c4a5b9eb9cbd9b1fbbcee27f7dbe73a739ec1a3a3645.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/a5cbe8333fd2eada57d7c4a5b9eb9cbd9b1fbbcee27f7dbe73a739ec1a3a3645.jpg)

![b2507b852c080e1c1747092063b50fb7a96afd50daed8751d69292cdb809ae7b.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/b2507b852c080e1c1747092063b50fb7a96afd50daed8751d69292cdb809ae7b.jpg)

![b609a60363a1fbe867954deb8690671ae868c4a503e93ebbd2d53fa41676e257.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/b609a60363a1fbe867954deb8690671ae868c4a503e93ebbd2d53fa41676e257.jpg)

![b69019b07d97935fe9cd076b9a399ade7dc867b722420add06b2e53cc6315f82.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/b69019b07d97935fe9cd076b9a399ade7dc867b722420add06b2e53cc6315f82.jpg)

![b7d07f12294e9cb74d7babb63e4f11814f4c86b3ddead8d6c983f46c2cfab1e1.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/b7d07f12294e9cb74d7babb63e4f11814f4c86b3ddead8d6c983f46c2cfab1e1.jpg)

![be6db83a23199499cd2fd93a9d76fff32f1988e020557ffcda1cca26d3f11c97.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/be6db83a23199499cd2fd93a9d76fff32f1988e020557ffcda1cca26d3f11c97.jpg)

![becb36f67a113027336022658f11b2878b41a984a1380ce23cd632b5d77cb2bb.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/becb36f67a113027336022658f11b2878b41a984a1380ce23cd632b5d77cb2bb.jpg)

![c2685697d3c65e92214721bdd124b3e80a1d80c9e2758ad760a0c8d6ea6cb813.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/c2685697d3c65e92214721bdd124b3e80a1d80c9e2758ad760a0c8d6ea6cb813.jpg)

![c277ee387b862764bd9035be1fcd84700583cecdda4b85ed323615669798bead.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/c277ee387b862764bd9035be1fcd84700583cecdda4b85ed323615669798bead.jpg)

![d137776b45c910422fbc4079473dbcc7631d75485605ebaa2422d3e81eddb3bc.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/d137776b45c910422fbc4079473dbcc7631d75485605ebaa2422d3e81eddb3bc.jpg)

![ec6d960479c36b17fbff246bc73c8c6a9badcb682150213eb7585ffe066464ec.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/ec6d960479c36b17fbff246bc73c8c6a9badcb682150213eb7585ffe066464ec.jpg)

![ee140d8a5e0a958fc5267556e0a12c7279205898bf2826c21a52a0c14b10b862.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/ee140d8a5e0a958fc5267556e0a12c7279205898bf2826c21a52a0c14b10b862.jpg)

![eebabbc75b8f0b5369bee17eace88dab2e6bc9a4884a99d49b04117fe8baed10.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/eebabbc75b8f0b5369bee17eace88dab2e6bc9a4884a99d49b04117fe8baed10.jpg)

![f09d547b747fb8280d3d2938258693891272548c665b0fb70ba4c03a293b9035.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/f09d547b747fb8280d3d2938258693891272548c665b0fb70ba4c03a293b9035.jpg)

![f643d73425c24d0a62a52860bab968b7c7aa4608dac42ed95bab89b9427aafeb.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/f643d73425c24d0a62a52860bab968b7c7aa4608dac42ed95bab89b9427aafeb.jpg)

![fae2d8c729751d9572bbeb469bc757154f5bf422945f2c6de3ade931c2ffd300.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/fae2d8c729751d9572bbeb469bc757154f5bf422945f2c6de3ade931c2ffd300.jpg)

![fbb04f6ae3c90a8665dae09fc753a92a91d9539220cbab4a56e1f176bb23762f.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/fbb04f6ae3c90a8665dae09fc753a92a91d9539220cbab4a56e1f176bb23762f.jpg)

### Tables

![3309f2f62ec5bb8a0dcd968049cf9858efd05cb4c8804ec147f6e294c51bdd9c.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/tables/3309f2f62ec5bb8a0dcd968049cf9858efd05cb4c8804ec147f6e294c51bdd9c.jpg)

![4cdeceb90ffa9bef340b5e7a9a84bdf326bfc960790c05eaff4c3635c5ffce9e.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/tables/4cdeceb90ffa9bef340b5e7a9a84bdf326bfc960790c05eaff4c3635c5ffce9e.jpg)

![6257229a1eac9aba3149438cc5fd9d8642c8a3c7248af1d1a3d4028d7d4612a2.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/tables/6257229a1eac9aba3149438cc5fd9d8642c8a3c7248af1d1a3d4028d7d4612a2.jpg)

![951b5e3e5568fce43933f8eb621a1944a34030b9da50bbe56f34df3b741e0849.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/tables/951b5e3e5568fce43933f8eb621a1944a34030b9da50bbe56f34df3b741e0849.jpg)

![d01f4f65d7124fa3a1ffb42fa8579c139fb2d388788f4f9c0a6398a696466ea5.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/tables/d01f4f65d7124fa3a1ffb42fa8579c139fb2d388788f4f9c0a6398a696466ea5.jpg)

![fbe888bd3e476401231187d7fbe1a5c7270f2030f6cb8b641d731f28857123c2.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/tables/fbe888bd3e476401231187d7fbe1a5c7270f2030f6cb8b641d731f28857123c2.jpg)
