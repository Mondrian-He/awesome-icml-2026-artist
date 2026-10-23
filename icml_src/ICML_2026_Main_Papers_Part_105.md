# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 105 of 197

## 目录 (Table of Contents)

1. [CLAA: Cross-Layer Attention Aggregation for Accelerating LLM Prefill](#paper-3433)
2. [Optimizing Rank for High-Fidelity Implicit Neural Representations](#paper-3434)
3. [Endogenous Resistance to Activation Steering in Language Models](#paper-3435)
4. [Positive Distribution Shift as a Framework for Understanding Tractable Learning](#paper-3436)
5. [BioToken and BioFM – Biologically-Informed Tokenization Enables Accurate and Efficient Genomic Foundation Models](#paper-3437)
6. [Fractional is Better: Learnable Derivative Orders in Neural Operator Learning](#paper-3438)
7. [Improving Adversarial Robustness of Attribution via Implicit Regularization](#paper-3439)
8. [Head-in-Head in Linear Attention](#paper-3440)
9. [What Language is This? Ask Your Tokenizer.](#paper-3441)
10. [Needles in the Haystack: Addressing Signal Dilution Improves scRNA-seq Perturbation Response Modeling and Evaluation](#paper-3442)
11. [SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering](#paper-3443)
12. [ActiveUltraFeedback: Efficient Preference Data Generation using Active Learning](#paper-3444)
13. [Frequentist Consistency of Prior-Data Fitted Networks for Causal Inference](#paper-3445)
14. [Learning to Emulate Chaos: Adversarial Optimal Transport Regularization](#paper-3446)
15. [Mechanistic Interpretability as Statistical Estimation: A Variance Analysis](#paper-3447)
16. [FunCQNet: A Functional Censored Quantile Neural Network for Predicting Long-Term Post-Transplant Kidney Survival](#paper-3448)
17. [DistMatch: Adaptive Binning via Distribution Matching for Robust Sequential Conformal Prediction](#paper-3449)
18. [A Solvable High-Dimensional Model Where Nonlinear Autoencoders Learn Structure Invisible to PCA While Test Loss Misaligns With Generalization](#paper-3450)
19. [Making Expert Reasoning Learnable with Self-Distillation](#paper-3451)
20. [Supervised Classification Heads as Semantic Prototypes: Unlocking Vision-Language Alignment via Weight Recycling](#paper-3452)
21. [GHOST: Unmasking Phantom States in Mamba2 via Grouped Hidden-state Output-aware Selection & Truncation](#paper-3453)
22. [Group Cognition Learning: Making Everything Better Through Governed Two-Stage Agents Collaboration](#paper-3454)
23. [Watermarking LLM Agent Trajectories](#paper-3455)
24. [Fair-FedMOE: Group-Fair One-Shot Federated Learning via Prototype-Guided Experts for Medical Imaging Analysis](#paper-3456)
25. [DV-World: Benchmarking Data Visualization Agents in Real-World Scenarios](#paper-3457)
26. [Open-o3-Video: Grounded Video Reasoning with Explicit Spatio-Temporal Evidence](#paper-3458)
27. [Norm$\times$Direction: Restoring the Missing Query Norm in Vision Linear Attention](#paper-3459)
28. [WhisperSplat: Lossless Steganography in 3D Gaussian Splatting](#paper-3460)
29. [Deep Coupling Learning for Solving PDEs](#paper-3461)
30. [Gradient Smoothing: Coupling Layer-wise Updates for Improved Optimization](#paper-3462)
31. [Benchmarking LLM-Assisted Blue Teaming via Standardized Threat Hunting](#paper-3463)
32. [Small Agent Group is the Future of Digital Health](#paper-3464)
33. [PoMtVRS: Preference-Optimized Multi-Task Vehicle Routing Solver with Preference Gating](#paper-3465)

---


<a id="paper-3433"></a>

## CLAA: Cross-Layer Attention Aggregation for Accelerating LLM Prefill

### Images

![48e4ee2dfe9ed491a4364d64bbdc20712ca126354f2388492b11197d02be29dc.jpg](../icml_results/3472_BARRED_%20Synthetic%20Training%20of%20Custom%20Policy%20Guardrails%20via%20Asymmetric%20Debate/images/48e4ee2dfe9ed491a4364d64bbdc20712ca126354f2388492b11197d02be29dc.jpg)

![8ef9ef6ac15635052850a740a00d27d29ab6aee82b52aa7dc79abb72569594a8.jpg](../icml_results/3472_BARRED_%20Synthetic%20Training%20of%20Custom%20Policy%20Guardrails%20via%20Asymmetric%20Debate/images/8ef9ef6ac15635052850a740a00d27d29ab6aee82b52aa7dc79abb72569594a8.jpg)

![a66294bc817eb79d1c395384da31b4d984b86fdc357eed6145349a0de7bacb52.jpg](../icml_results/3472_BARRED_%20Synthetic%20Training%20of%20Custom%20Policy%20Guardrails%20via%20Asymmetric%20Debate/images/a66294bc817eb79d1c395384da31b4d984b86fdc357eed6145349a0de7bacb52.jpg)

![b7ad56de4005c864601612a51a92750ffb46d00562f8add3f3161c67048a4c50.jpg](../icml_results/3472_BARRED_%20Synthetic%20Training%20of%20Custom%20Policy%20Guardrails%20via%20Asymmetric%20Debate/images/b7ad56de4005c864601612a51a92750ffb46d00562f8add3f3161c67048a4c50.jpg)

![e35fb8622f791a5f957d0b1f0e142cc5b32eb66ec0bce1879f7cc6ed87ca87d6.jpg](../icml_results/3472_BARRED_%20Synthetic%20Training%20of%20Custom%20Policy%20Guardrails%20via%20Asymmetric%20Debate/images/e35fb8622f791a5f957d0b1f0e142cc5b32eb66ec0bce1879f7cc6ed87ca87d6.jpg)

### Tables

![4b1d92b14f7bdb02e334a5e2cde1979b5fdc3cfdfdde62e9b98b69f8cb0da839.jpg](../icml_results/3472_BARRED_%20Synthetic%20Training%20of%20Custom%20Policy%20Guardrails%20via%20Asymmetric%20Debate/tables/4b1d92b14f7bdb02e334a5e2cde1979b5fdc3cfdfdde62e9b98b69f8cb0da839.jpg)

![4ee3c0e5624453946e5541d08315ade0a27b05e34d22d497c66b8f2ccdb7794c.jpg](../icml_results/3472_BARRED_%20Synthetic%20Training%20of%20Custom%20Policy%20Guardrails%20via%20Asymmetric%20Debate/tables/4ee3c0e5624453946e5541d08315ade0a27b05e34d22d497c66b8f2ccdb7794c.jpg)

![5a74e79a5aad19a8c62db1be281cb40d72770f3ac17ed0fc3d50006c2525192c.jpg](../icml_results/3472_BARRED_%20Synthetic%20Training%20of%20Custom%20Policy%20Guardrails%20via%20Asymmetric%20Debate/tables/5a74e79a5aad19a8c62db1be281cb40d72770f3ac17ed0fc3d50006c2525192c.jpg)

![923e366bb9b150574b59855cc542d427c5a4eac1a55c547d5e2f362e6a779432.jpg](../icml_results/3472_BARRED_%20Synthetic%20Training%20of%20Custom%20Policy%20Guardrails%20via%20Asymmetric%20Debate/tables/923e366bb9b150574b59855cc542d427c5a4eac1a55c547d5e2f362e6a779432.jpg)

![a3040cf4b6655c06843e9f865dcfa5eafb029025c0d7f5714ad152c40d399944.jpg](../icml_results/3472_BARRED_%20Synthetic%20Training%20of%20Custom%20Policy%20Guardrails%20via%20Asymmetric%20Debate/tables/a3040cf4b6655c06843e9f865dcfa5eafb029025c0d7f5714ad152c40d399944.jpg)

![ce2b74802b94d3be096809b866a18cc756ab5f88036a669b73d9bf4b2001a38d.jpg](../icml_results/3472_BARRED_%20Synthetic%20Training%20of%20Custom%20Policy%20Guardrails%20via%20Asymmetric%20Debate/tables/ce2b74802b94d3be096809b866a18cc756ab5f88036a669b73d9bf4b2001a38d.jpg)

![e78e0163d7a273fa00ae78e9d3e7b58dc50094e2a622475404b7e1394eacd361.jpg](../icml_results/3472_BARRED_%20Synthetic%20Training%20of%20Custom%20Policy%20Guardrails%20via%20Asymmetric%20Debate/tables/e78e0163d7a273fa00ae78e9d3e7b58dc50094e2a622475404b7e1394eacd361.jpg)

## CLAA: Cross-Layer Attention Aggregation for Accelerating LLM Prefill


### Images

![3a70e1267c20616c2291d06f374f8ab52e68cdf3d9d468199d24900d9da261c3.jpg](../icml_results/3473_CLAA_%20Cross-Layer%20Attention%20Aggregation%20for%20Accelerating%20LLM%20Prefill/images/3a70e1267c20616c2291d06f374f8ab52e68cdf3d9d468199d24900d9da261c3.jpg)

![58417ce88bcb6ccdf3fbb99703d9351123e0643402c5b0507db25f2460026f6d.jpg](../icml_results/3473_CLAA_%20Cross-Layer%20Attention%20Aggregation%20for%20Accelerating%20LLM%20Prefill/images/58417ce88bcb6ccdf3fbb99703d9351123e0643402c5b0507db25f2460026f6d.jpg)

![65c3b6789637e72bae93e5a7e0384436cf82936c878a0efde0b441c539b4c0c6.jpg](../icml_results/3473_CLAA_%20Cross-Layer%20Attention%20Aggregation%20for%20Accelerating%20LLM%20Prefill/images/65c3b6789637e72bae93e5a7e0384436cf82936c878a0efde0b441c539b4c0c6.jpg)

![6d1204cfc15a0e8f1d0b98c7ae21af5263ed25248cad4d2cd21242bf7b7ee3b5.jpg](../icml_results/3473_CLAA_%20Cross-Layer%20Attention%20Aggregation%20for%20Accelerating%20LLM%20Prefill/images/6d1204cfc15a0e8f1d0b98c7ae21af5263ed25248cad4d2cd21242bf7b7ee3b5.jpg)

![6da0be8d43b29d411ed634e20e99da136eb68805f602fe0cfcf69f6aae93a7c0.jpg](../icml_results/3473_CLAA_%20Cross-Layer%20Attention%20Aggregation%20for%20Accelerating%20LLM%20Prefill/images/6da0be8d43b29d411ed634e20e99da136eb68805f602fe0cfcf69f6aae93a7c0.jpg)

![c407ccdb6a4715194ce548730d317ac194b4acc376e654a4a45bf1e3b51b354d.jpg](../icml_results/3473_CLAA_%20Cross-Layer%20Attention%20Aggregation%20for%20Accelerating%20LLM%20Prefill/images/c407ccdb6a4715194ce548730d317ac194b4acc376e654a4a45bf1e3b51b354d.jpg)

![c79ad0ef24d8555582b657874e4c4a49aa8c725540806b951828f77ac2384422.jpg](../icml_results/3473_CLAA_%20Cross-Layer%20Attention%20Aggregation%20for%20Accelerating%20LLM%20Prefill/images/c79ad0ef24d8555582b657874e4c4a49aa8c725540806b951828f77ac2384422.jpg)

![ccd857bd46031eb36e5d29ab4831b754c1907379d57847fa2fc1de9adc9ec393.jpg](../icml_results/3473_CLAA_%20Cross-Layer%20Attention%20Aggregation%20for%20Accelerating%20LLM%20Prefill/images/ccd857bd46031eb36e5d29ab4831b754c1907379d57847fa2fc1de9adc9ec393.jpg)

### Tables

![11737c3a4dc85ae330e393eec6053c6d9f82efcafc7fb33a96aea9d7aa79c34b.jpg](../icml_results/3473_CLAA_%20Cross-Layer%20Attention%20Aggregation%20for%20Accelerating%20LLM%20Prefill/tables/11737c3a4dc85ae330e393eec6053c6d9f82efcafc7fb33a96aea9d7aa79c34b.jpg)

![621859df087598609692a7eabb4ec8bfc056b3ea03f5604817e3a2b840f50373.jpg](../icml_results/3473_CLAA_%20Cross-Layer%20Attention%20Aggregation%20for%20Accelerating%20LLM%20Prefill/tables/621859df087598609692a7eabb4ec8bfc056b3ea03f5604817e3a2b840f50373.jpg)

![93348500b7bb2a433563b0cbd7c06c1ce786619d2a714b2d3b9f9e2bb4f0a598.jpg](../icml_results/3473_CLAA_%20Cross-Layer%20Attention%20Aggregation%20for%20Accelerating%20LLM%20Prefill/tables/93348500b7bb2a433563b0cbd7c06c1ce786619d2a714b2d3b9f9e2bb4f0a598.jpg)

![c337c431539180b008594ec18687c1bf004fad6bf72c16e56e9e55f87d239798.jpg](../icml_results/3473_CLAA_%20Cross-Layer%20Attention%20Aggregation%20for%20Accelerating%20LLM%20Prefill/tables/c337c431539180b008594ec18687c1bf004fad6bf72c16e56e9e55f87d239798.jpg)

![f19b6b9702f2fa5c9f60371acf1e8568021b3e550df5778ac9664f958d923245.jpg](../icml_results/3473_CLAA_%20Cross-Layer%20Attention%20Aggregation%20for%20Accelerating%20LLM%20Prefill/tables/f19b6b9702f2fa5c9f60371acf1e8568021b3e550df5778ac9664f958d923245.jpg)

![ffb974b63694faa0d68ee1869616e4696604e2903ba6a5788f2bbfce0a227046.jpg](../icml_results/3473_CLAA_%20Cross-Layer%20Attention%20Aggregation%20for%20Accelerating%20LLM%20Prefill/tables/ffb974b63694faa0d68ee1869616e4696604e2903ba6a5788f2bbfce0a227046.jpg)

<a id="paper-3434"></a>

## Optimizing Rank for High-Fidelity Implicit Neural Representations


### Images

![05f1feb3b2c096d5d8fc85be3211ce3afba262a0c2b71b91afe8e8b432c02864.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/images/05f1feb3b2c096d5d8fc85be3211ce3afba262a0c2b71b91afe8e8b432c02864.jpg)

![0bd186b3d3c1818c1947b5d084c626f4c3da282ed77df801612a74e5ea1e61a1.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/images/0bd186b3d3c1818c1947b5d084c626f4c3da282ed77df801612a74e5ea1e61a1.jpg)

![36a5d9d5ee0935dca6662a17e5e8d31ad499e58fdf4f8f562dbb3fe4381997e8.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/images/36a5d9d5ee0935dca6662a17e5e8d31ad499e58fdf4f8f562dbb3fe4381997e8.jpg)

![3b53cdce5fc4551d3b87a42716e3d400f839ddaa2c349b68f1c85199215632c4.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/images/3b53cdce5fc4551d3b87a42716e3d400f839ddaa2c349b68f1c85199215632c4.jpg)

![4178458e35ed9cb1e0b950ad3d1d5d77de9a1fbb69254d225f273be30efd297e.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/images/4178458e35ed9cb1e0b950ad3d1d5d77de9a1fbb69254d225f273be30efd297e.jpg)

![77b84ffb66889af8ff550a0390c90f50e1019e5fdc28837201481311a84e749c.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/images/77b84ffb66889af8ff550a0390c90f50e1019e5fdc28837201481311a84e749c.jpg)

![792f49ef8afec35bd6da5e94cd652a9c2dc44a57e1c50c97208cdf9036fec925.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/images/792f49ef8afec35bd6da5e94cd652a9c2dc44a57e1c50c97208cdf9036fec925.jpg)

![9dd1a4e35b76ee5cb0ad0c21f4edc57d8929a21fd6d1436f125ab8f2714895b5.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/images/9dd1a4e35b76ee5cb0ad0c21f4edc57d8929a21fd6d1436f125ab8f2714895b5.jpg)

![ab748f4821effd0361f3d621ea528ae6aa860b51d0318a49184b8548e0d75d21.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/images/ab748f4821effd0361f3d621ea528ae6aa860b51d0318a49184b8548e0d75d21.jpg)

![bf836c7dabdffa9e8af0ed1fd67ee8709854caecf9c102e7d53773e4e4077edd.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/images/bf836c7dabdffa9e8af0ed1fd67ee8709854caecf9c102e7d53773e4e4077edd.jpg)

![c22415107710f1de2c0e303790cd4abac88606710ed665143cb2849adeedef57.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/images/c22415107710f1de2c0e303790cd4abac88606710ed665143cb2849adeedef57.jpg)

![d81beedadc37d2f657d5d8f54649624c289b01c3bdb62097c489a5a36825a19f.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/images/d81beedadc37d2f657d5d8f54649624c289b01c3bdb62097c489a5a36825a19f.jpg)

![eab457ce5b1625888d5f657fb1787e908fdb434437c4cdea42fc9970f2f6163a.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/images/eab457ce5b1625888d5f657fb1787e908fdb434437c4cdea42fc9970f2f6163a.jpg)

### Tables

![14c025bf797dcb941aaa22cc525d38f1fd31afa5f6289cf317e7ef50febc984f.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/14c025bf797dcb941aaa22cc525d38f1fd31afa5f6289cf317e7ef50febc984f.jpg)

![2258e3e7042a9545262b19de82f92efebab2818b9ad2d55d601e108cc758fe2c.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/2258e3e7042a9545262b19de82f92efebab2818b9ad2d55d601e108cc758fe2c.jpg)

![365881dd5ef8ee23b6ad098fcdd1e8b79f06d6663ab570a8c118428f60b3a087.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/365881dd5ef8ee23b6ad098fcdd1e8b79f06d6663ab570a8c118428f60b3a087.jpg)

![3e04afd8bf9c4af129d5c90971abefc7266f0420305a5c5a825e87abf5a99245.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/3e04afd8bf9c4af129d5c90971abefc7266f0420305a5c5a825e87abf5a99245.jpg)

![44b2ae5bb69c94fa4f9ca95d414167fc329ec65511dfe716ae2ee32efc2fc344.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/44b2ae5bb69c94fa4f9ca95d414167fc329ec65511dfe716ae2ee32efc2fc344.jpg)

![46480824f8846bc18f0680d3ea0abe4455e234b6093aca94721e861b00dc39be.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/46480824f8846bc18f0680d3ea0abe4455e234b6093aca94721e861b00dc39be.jpg)

![4e2fca878f4a3d89f0f1f85f36da3b152dffa1fc314bf281f84244071aff5572.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/4e2fca878f4a3d89f0f1f85f36da3b152dffa1fc314bf281f84244071aff5572.jpg)

![5e44ecef1e6a5dc87202f945f92a11e6441d32e6bbb251df2bd91139c639a19e.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/5e44ecef1e6a5dc87202f945f92a11e6441d32e6bbb251df2bd91139c639a19e.jpg)

![6a4cc78b3cf1264fb6adaf00f90f8ff94f605830be719f9633de724aed674563.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/6a4cc78b3cf1264fb6adaf00f90f8ff94f605830be719f9633de724aed674563.jpg)

![6ff9d983557372de605c1fcb04c74de25cf3c771860ed379569defb3eaf01f2b.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/6ff9d983557372de605c1fcb04c74de25cf3c771860ed379569defb3eaf01f2b.jpg)

![73bd4b493727d57d30d95798a4e93f0820a58bd8baa9c1e866112d8eef8272e1.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/73bd4b493727d57d30d95798a4e93f0820a58bd8baa9c1e866112d8eef8272e1.jpg)

![7a768acd2c73bae23c5611a4af103515989b27b265bb50e9c1f0416645c13c6d.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/7a768acd2c73bae23c5611a4af103515989b27b265bb50e9c1f0416645c13c6d.jpg)

![7ded1c9cb19602ac5f2c1abd6df5f75d1fb1ac3fc2b8574096b40a6ebea05c40.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/7ded1c9cb19602ac5f2c1abd6df5f75d1fb1ac3fc2b8574096b40a6ebea05c40.jpg)

![9580ca39f90d9d4de3dc6062de7ae33e01a26fc0f1afccb62b5e7409ab700a1c.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/9580ca39f90d9d4de3dc6062de7ae33e01a26fc0f1afccb62b5e7409ab700a1c.jpg)

![97c8a612cb119f88435038d271e29e2a0c3cd1f4916be0b99ab41194fafc6657.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/97c8a612cb119f88435038d271e29e2a0c3cd1f4916be0b99ab41194fafc6657.jpg)

![9c0a578485236cdffee905047e1876db63eb5c4b93ee40a30d57a09a2775f251.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/9c0a578485236cdffee905047e1876db63eb5c4b93ee40a30d57a09a2775f251.jpg)

![b8226dcba19605f4cb07400710bae0999f825027d18dba18c34d51f87cab9717.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/b8226dcba19605f4cb07400710bae0999f825027d18dba18c34d51f87cab9717.jpg)

![be82e8f49e39fbc4a05b22a4cfe7d36c2d4fc7010f3e83e8fe7bfd09ede17875.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/be82e8f49e39fbc4a05b22a4cfe7d36c2d4fc7010f3e83e8fe7bfd09ede17875.jpg)

![c78006137c213f5662d94e61aa6a50dfa2849582bb4811f555b51483847bf58a.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/c78006137c213f5662d94e61aa6a50dfa2849582bb4811f555b51483847bf58a.jpg)

![e0f0d16838e2fdc59ce2b0f75729c03e64a635091f7512a456e37a092d9153cf.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/e0f0d16838e2fdc59ce2b0f75729c03e64a635091f7512a456e37a092d9153cf.jpg)

![ecd5d1294ab64cc472438963e3cf01ddad253d96daa4e78645c76808aeb4463a.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/ecd5d1294ab64cc472438963e3cf01ddad253d96daa4e78645c76808aeb4463a.jpg)

![effb87e2aaf80bdef96bc5901250a34a56e670af09dcc29e28916f757b133f69.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/effb87e2aaf80bdef96bc5901250a34a56e670af09dcc29e28916f757b133f69.jpg)

![f509a30c7d7d8f649c70d2de72929d8c84ddbca8cf848ec89f4d9c91cbec0372.jpg](../icml_results/3474_Optimizing%20Rank%20for%20High-Fidelity%20Implicit%20Neural%20Representations/tables/f509a30c7d7d8f649c70d2de72929d8c84ddbca8cf848ec89f4d9c91cbec0372.jpg)

<a id="paper-3435"></a>

## Endogenous Resistance to Activation Steering in Language Models


### Images

![169db21fa8ef932a00bdce29c92f3257987d182439e353dec37c34c146159fcc.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/169db21fa8ef932a00bdce29c92f3257987d182439e353dec37c34c146159fcc.jpg)

![1a5cddc4fb352f0ab152f57ec98423f4a53b6a6272db8fb327a8f0edf7f16a22.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/1a5cddc4fb352f0ab152f57ec98423f4a53b6a6272db8fb327a8f0edf7f16a22.jpg)

![201a0251ca8ff226c455d0987182b37224e98760d2e79dddde1084cb5e75859a.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/201a0251ca8ff226c455d0987182b37224e98760d2e79dddde1084cb5e75859a.jpg)

![24c8ab1f3ef9a3e2b06843b28e9a931ff177f271f8a8d970701762ac01041f20.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/24c8ab1f3ef9a3e2b06843b28e9a931ff177f271f8a8d970701762ac01041f20.jpg)

![349334bfaa0fc3fcdf18ce3453936306aff346d119032d399553b4d8e78522a6.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/349334bfaa0fc3fcdf18ce3453936306aff346d119032d399553b4d8e78522a6.jpg)

![54f07ef36fbe7304487be1d9e87676e13ad35a71051cc79930514aaa037fea0f.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/54f07ef36fbe7304487be1d9e87676e13ad35a71051cc79930514aaa037fea0f.jpg)

![65a10f9590c70e32a5128fa0ca264a3b6d16f64144bbe65ebff4dd7f919ce7f7.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/65a10f9590c70e32a5128fa0ca264a3b6d16f64144bbe65ebff4dd7f919ce7f7.jpg)

![813d3fcd5bfc76bebc05f174148df29b332413b23810d722e3ceecbe2574acb4.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/813d3fcd5bfc76bebc05f174148df29b332413b23810d722e3ceecbe2574acb4.jpg)

![859d6cd734fce1449b7c08517149178fd7c988c51fc4f8827a2627dbda9c6647.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/859d6cd734fce1449b7c08517149178fd7c988c51fc4f8827a2627dbda9c6647.jpg)

![97e3132da5adc2609496da35eaf11509a1667702c40fd69813cb5be77d8b59b6.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/97e3132da5adc2609496da35eaf11509a1667702c40fd69813cb5be77d8b59b6.jpg)

![a12733b5c97fe8c1a49ec281984696eb399dd6db3d9034017afab793d965e7d8.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/a12733b5c97fe8c1a49ec281984696eb399dd6db3d9034017afab793d965e7d8.jpg)

![a25d08867bad6724cbc5eb41f4b468d143e885504c8f41d1c0d46810cf4e5497.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/a25d08867bad6724cbc5eb41f4b468d143e885504c8f41d1c0d46810cf4e5497.jpg)

![a62f9e2c836aa1793ae52f008a99ec813c8b0f220adf37c2b373b7deb5869402.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/a62f9e2c836aa1793ae52f008a99ec813c8b0f220adf37c2b373b7deb5869402.jpg)

![b237f7ebdad8a941ad6037ad9804c1a222b7a030ced33b60772b7fbf8cc91cd6.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/b237f7ebdad8a941ad6037ad9804c1a222b7a030ced33b60772b7fbf8cc91cd6.jpg)

![bcb9230464ea24420d665c90df05e037d2e053312984c6b10b410f9f7589abfa.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/bcb9230464ea24420d665c90df05e037d2e053312984c6b10b410f9f7589abfa.jpg)

![bdbc470656d771974ce0fd3637a3f9881f51f2cd55c25c806aa10dcc6b6dba17.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/bdbc470656d771974ce0fd3637a3f9881f51f2cd55c25c806aa10dcc6b6dba17.jpg)

![c53b32ef26e7ab396a3f5575c345657b316047a280bc5ccbbaca454750a70f67.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/c53b32ef26e7ab396a3f5575c345657b316047a280bc5ccbbaca454750a70f67.jpg)

![c59547b67fd34bf5b4d7d3a139fbb8a685d5f2e2593e05f914b696e7bdc31326.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/c59547b67fd34bf5b4d7d3a139fbb8a685d5f2e2593e05f914b696e7bdc31326.jpg)

![ca5b268a27feef33dcc50abb307ac1e8aad7622cacee3811af3987edb97d5f84.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/ca5b268a27feef33dcc50abb307ac1e8aad7622cacee3811af3987edb97d5f84.jpg)

![d9a3aeb5721d1f843d5abbf63627052c278c9fb5b46f605da6b6f17edc669b70.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/d9a3aeb5721d1f843d5abbf63627052c278c9fb5b46f605da6b6f17edc669b70.jpg)

![e450f6b1aedaf58dcf0575fe265cc01f1e0cbc87ac74a8f32319909cc4de0c8e.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/e450f6b1aedaf58dcf0575fe265cc01f1e0cbc87ac74a8f32319909cc4de0c8e.jpg)

![e50abe5f1970acf41971a90ca2a8822a17a88168535b978291465ecfe565afa6.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/e50abe5f1970acf41971a90ca2a8822a17a88168535b978291465ecfe565afa6.jpg)

![fecd142b782b3db55140e8c018ff0ed47ce807e34b6ae9990ad71c3b113b3efd.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/images/fecd142b782b3db55140e8c018ff0ed47ce807e34b6ae9990ad71c3b113b3efd.jpg)

### Tables

![08e1c8e66c953245671822741ee1c3d4c7f60c3166889819158404ca1a02f910.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/tables/08e1c8e66c953245671822741ee1c3d4c7f60c3166889819158404ca1a02f910.jpg)

![472810ce4f294b59fcd1b5627fc972f012d4a53168f25cfa2af8cd1e3074dbec.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/tables/472810ce4f294b59fcd1b5627fc972f012d4a53168f25cfa2af8cd1e3074dbec.jpg)

![7596f09f60f1c5dbf79468d629120cd3f4087e2f6e36f1f32a1406665ab54cc9.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/tables/7596f09f60f1c5dbf79468d629120cd3f4087e2f6e36f1f32a1406665ab54cc9.jpg)

![b2492c3d26b598740cdee66c9d658c07116fa73d54dc5c21f4ce1875aec22f94.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/tables/b2492c3d26b598740cdee66c9d658c07116fa73d54dc5c21f4ce1875aec22f94.jpg)

![b26b81f13b9d16a99e083f7b70f7caa2b59570e4fe12968ce1a71c367934ca20.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/tables/b26b81f13b9d16a99e083f7b70f7caa2b59570e4fe12968ce1a71c367934ca20.jpg)

![b2d90ee82c68b7a27fc70e2732f5ac8b44c9884fd14f7cbee10d351a91c54f59.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/tables/b2d90ee82c68b7a27fc70e2732f5ac8b44c9884fd14f7cbee10d351a91c54f59.jpg)

![bdba3dbd0b9bd3c1a13034c64b3cd4a0a7848643259e46f31858a496d60f0f30.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/tables/bdba3dbd0b9bd3c1a13034c64b3cd4a0a7848643259e46f31858a496d60f0f30.jpg)

![d35c171a6398f1c4ba252ed73c2129ac0d6922063f42c68b66c71e6d4acbbf8f.jpg](../icml_results/3475_Endogenous%20Resistance%20to%20Activation%20Steering%20in%20Language%20Models/tables/d35c171a6398f1c4ba252ed73c2129ac0d6922063f42c68b66c71e6d4acbbf8f.jpg)

<a id="paper-3436"></a>

## Positive Distribution Shift as a Framework for Understanding Tractable Learning


### Images

![5dcd53c24917d3a0dbc4bca63d61bd8100f6f8213538d6187e63fdb043f8352e.jpg](../icml_results/3476_Positive%20Distribution%20Shift%20as%20a%20Framework%20for%20Understanding%20Tractable%20Learning/images/5dcd53c24917d3a0dbc4bca63d61bd8100f6f8213538d6187e63fdb043f8352e.jpg)

![8520909a56322bcbde62f6c1966b46715073f7d05b2e5c6a08dc5473dc7c5aa7.jpg](../icml_results/3476_Positive%20Distribution%20Shift%20as%20a%20Framework%20for%20Understanding%20Tractable%20Learning/images/8520909a56322bcbde62f6c1966b46715073f7d05b2e5c6a08dc5473dc7c5aa7.jpg)

![96d284f7ee64cdde184e50e10976834f7424a9747736a4d9cc851de9023c53ae.jpg](../icml_results/3476_Positive%20Distribution%20Shift%20as%20a%20Framework%20for%20Understanding%20Tractable%20Learning/images/96d284f7ee64cdde184e50e10976834f7424a9747736a4d9cc851de9023c53ae.jpg)

![c8364368c3bf9a16721251c3adbf03a4d38fe0240a6d39c7f0fe974618c6e766.jpg](../icml_results/3476_Positive%20Distribution%20Shift%20as%20a%20Framework%20for%20Understanding%20Tractable%20Learning/images/c8364368c3bf9a16721251c3adbf03a4d38fe0240a6d39c7f0fe974618c6e766.jpg)

### Tables

![d176d9bef66c04a65c30e1648f0eb8c85b14e8911a5b1d2db9e6cd315d57504f.jpg](../icml_results/3476_Positive%20Distribution%20Shift%20as%20a%20Framework%20for%20Understanding%20Tractable%20Learning/tables/d176d9bef66c04a65c30e1648f0eb8c85b14e8911a5b1d2db9e6cd315d57504f.jpg)

<a id="paper-3437"></a>

## BioToken and BioFM – Biologically-Informed Tokenization Enables Accurate and Efficient Genomic Foundation Models


### Images

![0324430ca0c09c1fc53f38ca0494bf9fdce9bb632289b39670f058b99242951c.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/images/0324430ca0c09c1fc53f38ca0494bf9fdce9bb632289b39670f058b99242951c.jpg)

![294a78dad1b0812b7ee0bebcdf5d9e71c3541ce48de54aee2f1c94b661d2b4a4.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/images/294a78dad1b0812b7ee0bebcdf5d9e71c3541ce48de54aee2f1c94b661d2b4a4.jpg)

![4d88697de51c4f9037df44ea4e3df4b2f7acb24e6731a2e2c360d0ddd6e75207.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/images/4d88697de51c4f9037df44ea4e3df4b2f7acb24e6731a2e2c360d0ddd6e75207.jpg)

![af490ed69ab392521f8502808479486c63527962f06799f979fc42fdc7ca5f96.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/images/af490ed69ab392521f8502808479486c63527962f06799f979fc42fdc7ca5f96.jpg)

![b907d2a989376aa6bfdb4ea248e83004cba9587756480396cca22c454e59013c.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/images/b907d2a989376aa6bfdb4ea248e83004cba9587756480396cca22c454e59013c.jpg)

![b92692ce04efd492611e54330df4164371aaaa0cb2de61384d3ed564641f3157.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/images/b92692ce04efd492611e54330df4164371aaaa0cb2de61384d3ed564641f3157.jpg)

![bad967294f7172e5fe2ee9d8de61c8003c0ff8ee8ee7a0e5766589ec74b15feb.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/images/bad967294f7172e5fe2ee9d8de61c8003c0ff8ee8ee7a0e5766589ec74b15feb.jpg)

![c66b53c1b2a8ee149edad26a565aaa416b74f83175474e6c4c5c13cc59a281e5.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/images/c66b53c1b2a8ee149edad26a565aaa416b74f83175474e6c4c5c13cc59a281e5.jpg)

![d40075bdd685bd5e8658a990ccd8aa219ce18f50808bf8cf99c3674e2f466440.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/images/d40075bdd685bd5e8658a990ccd8aa219ce18f50808bf8cf99c3674e2f466440.jpg)

![df6a02dc17859a87493560bb59565ca4550b0fdc6dea534efafae5caf1aca773.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/images/df6a02dc17859a87493560bb59565ca4550b0fdc6dea534efafae5caf1aca773.jpg)

### Tables

![43bfdec5b5ab4b9a2786eee3e08c9e7cef8dc8a677fd59536024c788c0410503.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/tables/43bfdec5b5ab4b9a2786eee3e08c9e7cef8dc8a677fd59536024c788c0410503.jpg)

![457ea4dc777294777da6124c12d49e3214ee628f900c8120dfb16eb2ba060b01.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/tables/457ea4dc777294777da6124c12d49e3214ee628f900c8120dfb16eb2ba060b01.jpg)

![5544cad4f0895699931bc9612816036b09c8d6a385e4afa16bd2fdf650b0e1d5.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/tables/5544cad4f0895699931bc9612816036b09c8d6a385e4afa16bd2fdf650b0e1d5.jpg)

![5bcd519c68b77e3fdeae73e23380a61caecc33ad96c91e70d202e9923f85ac1f.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/tables/5bcd519c68b77e3fdeae73e23380a61caecc33ad96c91e70d202e9923f85ac1f.jpg)

![698d38b40f04c74dea67a8d4aeea4063eaa19751079af01890ade2f85ad3be0b.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/tables/698d38b40f04c74dea67a8d4aeea4063eaa19751079af01890ade2f85ad3be0b.jpg)

![727980570dbfe92586710ccb9b247ba449334b3bb9292c5b55cc51aed9d26bb7.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/tables/727980570dbfe92586710ccb9b247ba449334b3bb9292c5b55cc51aed9d26bb7.jpg)

![73e1e1e5a4cc9487a92df95e985a9f943cd7022727cc69945d6fb67a57f0e56d.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/tables/73e1e1e5a4cc9487a92df95e985a9f943cd7022727cc69945d6fb67a57f0e56d.jpg)

![932b53eb7fa8cba4d715c70a6a0545c0b9f2704d932945368b00bc336e3f66d3.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/tables/932b53eb7fa8cba4d715c70a6a0545c0b9f2704d932945368b00bc336e3f66d3.jpg)

![e7d3cccb66d259bfc5b094d9258d6ae42880e9247845e04b60e4323ea9978932.jpg](../icml_results/3477_BioToken%20and%20BioFM%20%E2%80%93%20Biologically-Informed%20Tokenization%20Enables%20Accurate%20and%20Efficient%20Genomic%20Found/tables/e7d3cccb66d259bfc5b094d9258d6ae42880e9247845e04b60e4323ea9978932.jpg)

<a id="paper-3438"></a>

## Fractional is Better: Learnable Derivative Orders in Neural Operator Learning


### Images

![2572448be02f03efff60d9dde7af281ceb9fd8ec2a0807e64796782d77585503.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/images/2572448be02f03efff60d9dde7af281ceb9fd8ec2a0807e64796782d77585503.jpg)

![40abe918c1909f6635ac7762382a66ad4a7c8f09f8991729f7fa49fedf0212e8.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/images/40abe918c1909f6635ac7762382a66ad4a7c8f09f8991729f7fa49fedf0212e8.jpg)

![5d90cf7144691a30f7688a456842dd59cf9cf4579c0e9d1e967d2fa64a75e27c.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/images/5d90cf7144691a30f7688a456842dd59cf9cf4579c0e9d1e967d2fa64a75e27c.jpg)

![9069df2526440fd2ec1d17ad0804ed523ddcfc9f0c8f821243f4160e4af94898.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/images/9069df2526440fd2ec1d17ad0804ed523ddcfc9f0c8f821243f4160e4af94898.jpg)

![92752e299cc2965a40ac667f56a62316bad22d749d2beea121f06c452873c9ea.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/images/92752e299cc2965a40ac667f56a62316bad22d749d2beea121f06c452873c9ea.jpg)

![a25d58c75719dbd3983a6c3c458690c9af2147dedda042ff97a37633d6ecffb9.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/images/a25d58c75719dbd3983a6c3c458690c9af2147dedda042ff97a37633d6ecffb9.jpg)

![b028712a1497356709540780227bcce215e357205f0ecb46aee2f25f26a9667d.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/images/b028712a1497356709540780227bcce215e357205f0ecb46aee2f25f26a9667d.jpg)

![b03adcdbda3cb1dc5fa2c88f79217fa253eaf20c756b40bb10069d6749581cfa.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/images/b03adcdbda3cb1dc5fa2c88f79217fa253eaf20c756b40bb10069d6749581cfa.jpg)

![d4460d1bbf1182e131c2212c5f8a1581df41d9d3a7eb01a745830778202d3b2d.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/images/d4460d1bbf1182e131c2212c5f8a1581df41d9d3a7eb01a745830778202d3b2d.jpg)

![dd947386fd87784de4f1272428e826d55c46143a696842f62dc4d9a0b56785d7.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/images/dd947386fd87784de4f1272428e826d55c46143a696842f62dc4d9a0b56785d7.jpg)

![f1bbfe7fec9bf213003653b4e2f0021e3ea0ac584d672007f0197e5cb3b6ad77.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/images/f1bbfe7fec9bf213003653b4e2f0021e3ea0ac584d672007f0197e5cb3b6ad77.jpg)

### Tables

![0baa56457a6b48d34dde0e1b389106c139e9a901cf1c3d32fe8da2b4f8e4bcee.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/tables/0baa56457a6b48d34dde0e1b389106c139e9a901cf1c3d32fe8da2b4f8e4bcee.jpg)

![21e15caadaed14b0d35ca2730ea6b05a27ef75b75b14bce046daad38f29d7d81.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/tables/21e15caadaed14b0d35ca2730ea6b05a27ef75b75b14bce046daad38f29d7d81.jpg)

![486642c6491b7d9ef6ee8991187943961a28007f2c46dbc4c7dc4964605c4bc7.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/tables/486642c6491b7d9ef6ee8991187943961a28007f2c46dbc4c7dc4964605c4bc7.jpg)

![5e9c134ddf88abc00c988ed7c6d6ebbe97956521b8945123c1b2f5d464cf57f6.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/tables/5e9c134ddf88abc00c988ed7c6d6ebbe97956521b8945123c1b2f5d464cf57f6.jpg)

![68d393a4bfe426eb28670497e2322f5cd43adadd0592f8bd402ae5f02859d4cb.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/tables/68d393a4bfe426eb28670497e2322f5cd43adadd0592f8bd402ae5f02859d4cb.jpg)

![92247847e446458b120d19221f94d869af191963369620191f2c704b79f096d0.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/tables/92247847e446458b120d19221f94d869af191963369620191f2c704b79f096d0.jpg)

![95f808c96cbc555c55aac80b69970958652a7e990022dda3a0189a113df6bed6.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/tables/95f808c96cbc555c55aac80b69970958652a7e990022dda3a0189a113df6bed6.jpg)

![a0cb3cd5cc114b2c42732da53740328feda5e3b852f62f4bf4ec010521029359.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/tables/a0cb3cd5cc114b2c42732da53740328feda5e3b852f62f4bf4ec010521029359.jpg)

![a7430e0a70f2f7ea3c421b70522bdeb26119f71a3469dbd33362b2a757f7f90b.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/tables/a7430e0a70f2f7ea3c421b70522bdeb26119f71a3469dbd33362b2a757f7f90b.jpg)

![d8fd3db3dc8a8e0d83e4d4cc152a239baa951be963688deca1b0dee61e103bb3.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/tables/d8fd3db3dc8a8e0d83e4d4cc152a239baa951be963688deca1b0dee61e103bb3.jpg)

![e72329df4d93190456a37808f106c7d73a80cd9cf49fcc41942d13b8b91fd814.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/tables/e72329df4d93190456a37808f106c7d73a80cd9cf49fcc41942d13b8b91fd814.jpg)

![ed87bcbcba99f33fa5d56b11dcb79c9814dee1c4f994bf6e563ca7ca5bfa6e33.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/tables/ed87bcbcba99f33fa5d56b11dcb79c9814dee1c4f994bf6e563ca7ca5bfa6e33.jpg)

![fa77f8225b5bb6788761c315761e41a3bf0272eef52465627ee148c1881ad5f8.jpg](../icml_results/3478_Fractional%20is%20Better_%20Learnable%20Derivative%20Orders%20in%20Neural%20Operator%20Learning/tables/fa77f8225b5bb6788761c315761e41a3bf0272eef52465627ee148c1881ad5f8.jpg)

<a id="paper-3439"></a>

## Improving Adversarial Robustness of Attribution via Implicit Regularization


### Images

![08c699d5865e96800417e497960cfc96ec3c5ca7cb76813a77b482121f692c13.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/08c699d5865e96800417e497960cfc96ec3c5ca7cb76813a77b482121f692c13.jpg)

![2803ad6ca9241bb06bad8b2251b16fe79914dcb1dc0e390d2c6d4f1704124b84.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/2803ad6ca9241bb06bad8b2251b16fe79914dcb1dc0e390d2c6d4f1704124b84.jpg)

![33fd6b90655ac2198c788d536f7b3aa9740304a77e0433426c7b0a2d50b65ec7.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/33fd6b90655ac2198c788d536f7b3aa9740304a77e0433426c7b0a2d50b65ec7.jpg)

![352efb168008813318719ca6a397869d8701425d16dfbc6b305208e7269c1d90.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/352efb168008813318719ca6a397869d8701425d16dfbc6b305208e7269c1d90.jpg)

![3d6140ee6ec69ac64cd16b7b405c9779770ff7709d1a5fbd44f29845025fd79e.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/3d6140ee6ec69ac64cd16b7b405c9779770ff7709d1a5fbd44f29845025fd79e.jpg)

![3e87b668069149633d4ce26c78f7b627854de72e46159187f62faf42543d256a.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/3e87b668069149633d4ce26c78f7b627854de72e46159187f62faf42543d256a.jpg)

![455809099cbb2924ffeece4dbacabf11c2018aed4d9ca53dc820308b296b4bc2.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/455809099cbb2924ffeece4dbacabf11c2018aed4d9ca53dc820308b296b4bc2.jpg)

![45e618541177a6e12bd11163e25037027d7ddfcfc76d9d69fbb6819477cf6a99.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/45e618541177a6e12bd11163e25037027d7ddfcfc76d9d69fbb6819477cf6a99.jpg)

![5636dee2e14d859e29bf330e2c2c385ed99f668834ee7c89588b3000cad06fc1.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/5636dee2e14d859e29bf330e2c2c385ed99f668834ee7c89588b3000cad06fc1.jpg)

![5ae30cff243328063994d4f02cf1ea05cd73065a288fa67de7bead79955483e0.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/5ae30cff243328063994d4f02cf1ea05cd73065a288fa67de7bead79955483e0.jpg)

![71ecd7b8e6d5c1ad82b837af9340447b0ffda9e5b7f5d64e298a3827712c416f.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/71ecd7b8e6d5c1ad82b837af9340447b0ffda9e5b7f5d64e298a3827712c416f.jpg)

![97e61080f14313cc018c544d8502cc4c36fbb5b89e269e26d05f6fb59b04d1a7.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/97e61080f14313cc018c544d8502cc4c36fbb5b89e269e26d05f6fb59b04d1a7.jpg)

![a23432f2ede68067542a1ba64db8b8384d14be6a20e30bd2a054addb4b41c1b5.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/a23432f2ede68067542a1ba64db8b8384d14be6a20e30bd2a054addb4b41c1b5.jpg)

![aca7b5aab813fd58f56fc76f72a9aaf3431e3360df727ceb204f6b8b5211d782.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/aca7b5aab813fd58f56fc76f72a9aaf3431e3360df727ceb204f6b8b5211d782.jpg)

![cbe0351f94420880601b230b9733b2db95406bcf2ea45ff769ce19b3cfc3a9c8.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/cbe0351f94420880601b230b9733b2db95406bcf2ea45ff769ce19b3cfc3a9c8.jpg)

![d92bba0313368153f3384bf4032f75736d2683a3526f7eb9ccf2b806e9c35968.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/d92bba0313368153f3384bf4032f75736d2683a3526f7eb9ccf2b806e9c35968.jpg)

![eb7aa8abe73a58418feb2db3e899eeaff1dae01102b336a39a65990dbb5fac49.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/eb7aa8abe73a58418feb2db3e899eeaff1dae01102b336a39a65990dbb5fac49.jpg)

![f1be0a37a74764b4045f16ad4e80694f0383b7601af90aa68106ed765a23a0ec.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/f1be0a37a74764b4045f16ad4e80694f0383b7601af90aa68106ed765a23a0ec.jpg)

![f77300c3ef238b59247f6562f171e025dd08df4cab6fd10f9cdfb8f3c47ed2f6.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/f77300c3ef238b59247f6562f171e025dd08df4cab6fd10f9cdfb8f3c47ed2f6.jpg)

![f7878d6bf301cb1dc3272fa633ea5974a6ecd806f227878fee63a1b119cf4a7c.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/f7878d6bf301cb1dc3272fa633ea5974a6ecd806f227878fee63a1b119cf4a7c.jpg)

![f7c514a1b8577588b478d32a88689f21f782736635c86258fcaa041e49f50092.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/f7c514a1b8577588b478d32a88689f21f782736635c86258fcaa041e49f50092.jpg)

![f916370840a7f82f4096bb25c58634d353033deb9e09dc4e5805e605cfeca266.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/images/f916370840a7f82f4096bb25c58634d353033deb9e09dc4e5805e605cfeca266.jpg)

### Tables

![1440b85c01de6b5d4f2cb5491633ec22425d8c7b2537918a644899b6725f761d.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/tables/1440b85c01de6b5d4f2cb5491633ec22425d8c7b2537918a644899b6725f761d.jpg)

![1516c715c90d3f91affc852f15ae9433c4768680d0ed6a4cf783337bf365b329.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/tables/1516c715c90d3f91affc852f15ae9433c4768680d0ed6a4cf783337bf365b329.jpg)

![1edc237603d4a4b6aeb069e1b793f3885376469ce12e9cd13a5508df618eb2e1.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/tables/1edc237603d4a4b6aeb069e1b793f3885376469ce12e9cd13a5508df618eb2e1.jpg)

![21708d355d745c6c923dee8e0af0955cc17bd021cc1b856c63c5f172a6633672.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/tables/21708d355d745c6c923dee8e0af0955cc17bd021cc1b856c63c5f172a6633672.jpg)

![568df718647e8044f9085f3fa8c4d0062f19b6040b3e1be8e9ab18c885dedeb1.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/tables/568df718647e8044f9085f3fa8c4d0062f19b6040b3e1be8e9ab18c885dedeb1.jpg)

![65b76ce3d050ffbc7b900a0c8635cf42a2b4d3933d6f4af13c410601b3c950a9.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/tables/65b76ce3d050ffbc7b900a0c8635cf42a2b4d3933d6f4af13c410601b3c950a9.jpg)

![6c585b0db82f6d406683ee794d7a4d3701adf0136115248b671d4555ede5a4e9.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/tables/6c585b0db82f6d406683ee794d7a4d3701adf0136115248b671d4555ede5a4e9.jpg)

![94b1ac7a2f2f6f46b9168a2f7b621fa79073f8b25ec5ea2e7947542c95067c3f.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/tables/94b1ac7a2f2f6f46b9168a2f7b621fa79073f8b25ec5ea2e7947542c95067c3f.jpg)

![b022e9b48f9694bc635d75287b968fe7a2a29c54e8cf58f5b6e1cac26294e00b.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/tables/b022e9b48f9694bc635d75287b968fe7a2a29c54e8cf58f5b6e1cac26294e00b.jpg)

![c52712eda1a64c352d0adf1cb98d6623f52d0f601b8ed456ea661d559304aac0.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/tables/c52712eda1a64c352d0adf1cb98d6623f52d0f601b8ed456ea661d559304aac0.jpg)

![c770b470ad86bb14d622d1550519e244e6576c813cccc1b169158c193ad71de4.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/tables/c770b470ad86bb14d622d1550519e244e6576c813cccc1b169158c193ad71de4.jpg)

![f6cb3c021ec12c0c4acd08931990d243cb8665039c6c54abf3792af097b94941.jpg](../icml_results/3479_Improving%20Adversarial%20Robustness%20of%20Attribution%20via%20Implicit%20Regularization/tables/f6cb3c021ec12c0c4acd08931990d243cb8665039c6c54abf3792af097b94941.jpg)

<a id="paper-3440"></a>

## Head-in-Head in Linear Attention


### Images

![2ae449e6fae55ac1ef0f0398e62e2b8b199d5ce2c57d71b359bb3c4beda08768.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/images/2ae449e6fae55ac1ef0f0398e62e2b8b199d5ce2c57d71b359bb3c4beda08768.jpg)

![59dc03e41eb2aa92f3fad5d6566043193e886bb0fe0850d3120659d87e7e7600.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/images/59dc03e41eb2aa92f3fad5d6566043193e886bb0fe0850d3120659d87e7e7600.jpg)

![614c8d2429d6da81ed517094d5e2001d2698cdcaeccd96ff4b8a63ffe2f0e4db.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/images/614c8d2429d6da81ed517094d5e2001d2698cdcaeccd96ff4b8a63ffe2f0e4db.jpg)

![91475abeffc8ac0731a8b6035e4bca03227281d485b5db58bd6dd1a98dd356e1.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/images/91475abeffc8ac0731a8b6035e4bca03227281d485b5db58bd6dd1a98dd356e1.jpg)

![acfa7d5cd46a5063639eceb1904d23eed78362346cbb3c24683dbee1a94b7b7c.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/images/acfa7d5cd46a5063639eceb1904d23eed78362346cbb3c24683dbee1a94b7b7c.jpg)

![b5fad28da7b98f4db581342e8e94dec20e0f3a7859cb3270bbee3da21f608095.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/images/b5fad28da7b98f4db581342e8e94dec20e0f3a7859cb3270bbee3da21f608095.jpg)

![dbd6bf4a87923255ec4644a3cee8e4aaa373fb6c8e6476b48ff1bdaa95a6a6ae.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/images/dbd6bf4a87923255ec4644a3cee8e4aaa373fb6c8e6476b48ff1bdaa95a6a6ae.jpg)

![fba2f0dde2f71842945ee2040142bf92a8f691194f46cbf74e113b4adaa58353.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/images/fba2f0dde2f71842945ee2040142bf92a8f691194f46cbf74e113b4adaa58353.jpg)

### Tables

![030f91b64c78775656e070cf1cf9c88d487a9802c14ef3feebba1858359a1cbc.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/tables/030f91b64c78775656e070cf1cf9c88d487a9802c14ef3feebba1858359a1cbc.jpg)

![109c62f3482caddb81786557e427edba509c8c2afdb75370f8a5f27c4900a390.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/tables/109c62f3482caddb81786557e427edba509c8c2afdb75370f8a5f27c4900a390.jpg)

![139fd1384a8030051880ffdd0ef6845381eef679305adb4efbc3712090b84999.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/tables/139fd1384a8030051880ffdd0ef6845381eef679305adb4efbc3712090b84999.jpg)

![14f6cd4898c588684b3d96852d8e916da6e7cb53a7c0e0c8db6695757c6b3df9.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/tables/14f6cd4898c588684b3d96852d8e916da6e7cb53a7c0e0c8db6695757c6b3df9.jpg)

![3d7a320cff10548c9cb01d1e82fba51db1638064822fe685d4aff0da0dad0320.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/tables/3d7a320cff10548c9cb01d1e82fba51db1638064822fe685d4aff0da0dad0320.jpg)

![42ad26a14659f2e3469c7e986bb713590cdac278f65b9e127ff21dbac13665c6.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/tables/42ad26a14659f2e3469c7e986bb713590cdac278f65b9e127ff21dbac13665c6.jpg)

![42b07e161d35dc982778d3eef2a6ece3fd3ffef9b29a40a0322606017fab3513.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/tables/42b07e161d35dc982778d3eef2a6ece3fd3ffef9b29a40a0322606017fab3513.jpg)

![48263b2a33b169e9baefeed0e315476cbcce96727160524ff15e93f46fc587ec.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/tables/48263b2a33b169e9baefeed0e315476cbcce96727160524ff15e93f46fc587ec.jpg)

![5aa3bcc44fab9703f39111d3ab43455b98836a073c3d42789b9e2e2daa4bb6d8.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/tables/5aa3bcc44fab9703f39111d3ab43455b98836a073c3d42789b9e2e2daa4bb6d8.jpg)

![681d496a74a462d61dc3064cd125af41d42b02c7c7c81fb634edf38f777b4b6d.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/tables/681d496a74a462d61dc3064cd125af41d42b02c7c7c81fb634edf38f777b4b6d.jpg)

![73360664757c01930c6744ea6bc081f314fba5344216aeac27a38bffc4bf36d5.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/tables/73360664757c01930c6744ea6bc081f314fba5344216aeac27a38bffc4bf36d5.jpg)

![9853fba69a9ada3f7337674241466ea5e9e5b80c130cf697d1dba05193258e2d.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/tables/9853fba69a9ada3f7337674241466ea5e9e5b80c130cf697d1dba05193258e2d.jpg)

![d89c9c026d97f531217745c97b5f7f0f96fd3760f29bb4f3fafc1259e4cbce1d.jpg](../icml_results/3480_Head-in-Head%20in%20Linear%20Attention/tables/d89c9c026d97f531217745c97b5f7f0f96fd3760f29bb4f3fafc1259e4cbce1d.jpg)

<a id="paper-3441"></a>

## What Language is This? Ask Your Tokenizer.


### Images

![23fe7fb24671d84284c5d84c10af6b34d00822b04435745862f372e17655c5da.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./images/23fe7fb24671d84284c5d84c10af6b34d00822b04435745862f372e17655c5da.jpg)

### Tables

![0389ae3c1cbfcb243c69d5363f6433de895555b4572fc09428b7d56c284463dd.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/0389ae3c1cbfcb243c69d5363f6433de895555b4572fc09428b7d56c284463dd.jpg)

![0fd21f0ecbced1fc18b6e20a3818a73dc54e29f6b227dba72639fbe6664e5035.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/0fd21f0ecbced1fc18b6e20a3818a73dc54e29f6b227dba72639fbe6664e5035.jpg)

![1f05aef4098a7fede6a0c52c243c69fa624cfc988f5227a3d6155da41feb9165.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/1f05aef4098a7fede6a0c52c243c69fa624cfc988f5227a3d6155da41feb9165.jpg)

![3487ff9f78a69fa3686edf9cbbc69441fa1d9d76c0d0de6a04c1ac6a2232d990.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/3487ff9f78a69fa3686edf9cbbc69441fa1d9d76c0d0de6a04c1ac6a2232d990.jpg)

![38d6475f3206e70d17f5c1f64912222b9cd460e4f4d0fdf579cb60db23922345.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/38d6475f3206e70d17f5c1f64912222b9cd460e4f4d0fdf579cb60db23922345.jpg)

![4a8a162aac063be22be47085db9ecc77c7a4e86eed0fa9cbedc1a7c76e9d03d7.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/4a8a162aac063be22be47085db9ecc77c7a4e86eed0fa9cbedc1a7c76e9d03d7.jpg)

![7fadd5f90cbbc172922f64d8c92dd941d2b6e97cbb41437d1ceb665b96a09296.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/7fadd5f90cbbc172922f64d8c92dd941d2b6e97cbb41437d1ceb665b96a09296.jpg)

![805e8362a9d8ff5b06fc62d309d0bfbf3fc8cb6c15fdbd6d209983c338227a78.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/805e8362a9d8ff5b06fc62d309d0bfbf3fc8cb6c15fdbd6d209983c338227a78.jpg)

![99b32bf040dec51dc6aaed95915adc67e78e81d4696387caec57f275d2df0129.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/99b32bf040dec51dc6aaed95915adc67e78e81d4696387caec57f275d2df0129.jpg)

![99fbeaea2cf9e80d4a3b5b1b33fc3c68d25bc5f199b29864575b19c733608f1c.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/99fbeaea2cf9e80d4a3b5b1b33fc3c68d25bc5f199b29864575b19c733608f1c.jpg)

![9bc7ca69706e200ad58dd485d5d0aff35db6755e681d1eb83ae8340065483066.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/9bc7ca69706e200ad58dd485d5d0aff35db6755e681d1eb83ae8340065483066.jpg)

![ba53a93f63d61991f0de3bbdd8dd23061ed8d78e5191cc93ca28cee5c21f15a9.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/ba53a93f63d61991f0de3bbdd8dd23061ed8d78e5191cc93ca28cee5c21f15a9.jpg)

![bb252d924f609061ecf7d1ae840f6f8e85edefbc429b29b2053fd913fe461e12.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/bb252d924f609061ecf7d1ae840f6f8e85edefbc429b29b2053fd913fe461e12.jpg)

![bff8b9ae845d71eb7bce7db81d3d97746bea942662a7faaa90fbfe82913f0ad5.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/bff8b9ae845d71eb7bce7db81d3d97746bea942662a7faaa90fbfe82913f0ad5.jpg)

![c1b85d109f84a38deb70b4632e51f0a09e45eab5f17d18593866bf28e9acb00b.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/c1b85d109f84a38deb70b4632e51f0a09e45eab5f17d18593866bf28e9acb00b.jpg)

![c8bfef215e7fa7a7186ae4a029dd1421118ed2679d45d84c9971472284b85fae.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/c8bfef215e7fa7a7186ae4a029dd1421118ed2679d45d84c9971472284b85fae.jpg)

![dec5e26df48f93f34ba26a7aa41b93fde269dc2e7e4c89a5d0f1969b2f1b5ccc.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/dec5e26df48f93f34ba26a7aa41b93fde269dc2e7e4c89a5d0f1969b2f1b5ccc.jpg)

![f69de167076c205eb5f067294fa66f6a7e6dbdb0201e7ce03857de32c3501371.jpg](../icml_results/3481_What%20Language%20is%20This_%20Ask%20Your%20Tokenizer./tables/f69de167076c205eb5f067294fa66f6a7e6dbdb0201e7ce03857de32c3501371.jpg)

<a id="paper-3442"></a>

## Needles in the Haystack: Addressing Signal Dilution Improves scRNA-seq Perturbation Response Modeling and Evaluation


### Images

![00f33720b6e5e77f2334ae499b371c2cd979a65c64b087adf369f4c0c0a5fbda.jpg](../icml_results/3482_Needles%20in%20the%20Haystack_%20Addressing%20Signal%20Dilution%20Improves%20scRNA-seq%20Perturbation%20Response%20Modelin/images/00f33720b6e5e77f2334ae499b371c2cd979a65c64b087adf369f4c0c0a5fbda.jpg)

![216301be59f83ee46e4a41ce6c9834f1ed2ddab90feba5bc60ae304bff7805cb.jpg](../icml_results/3482_Needles%20in%20the%20Haystack_%20Addressing%20Signal%20Dilution%20Improves%20scRNA-seq%20Perturbation%20Response%20Modelin/images/216301be59f83ee46e4a41ce6c9834f1ed2ddab90feba5bc60ae304bff7805cb.jpg)

![2c9538856b9e7366f8f9a784824fa265dc81f42aa2ca1ce1ae907934dc351933.jpg](../icml_results/3482_Needles%20in%20the%20Haystack_%20Addressing%20Signal%20Dilution%20Improves%20scRNA-seq%20Perturbation%20Response%20Modelin/images/2c9538856b9e7366f8f9a784824fa265dc81f42aa2ca1ce1ae907934dc351933.jpg)

![543747d642c31248b9f35b8f759246a6d6c24e402d861e3d99e75af0761ffe33.jpg](../icml_results/3482_Needles%20in%20the%20Haystack_%20Addressing%20Signal%20Dilution%20Improves%20scRNA-seq%20Perturbation%20Response%20Modelin/images/543747d642c31248b9f35b8f759246a6d6c24e402d861e3d99e75af0761ffe33.jpg)

![7c743ce214e4c2d5ee3cea401609b4ee1aade35c6db0f4df65e50235eda8865b.jpg](../icml_results/3482_Needles%20in%20the%20Haystack_%20Addressing%20Signal%20Dilution%20Improves%20scRNA-seq%20Perturbation%20Response%20Modelin/images/7c743ce214e4c2d5ee3cea401609b4ee1aade35c6db0f4df65e50235eda8865b.jpg)

![81db3b784d3db2f962c616ae3b9360eae563cb2c700536f3b324001a3d5ceb65.jpg](../icml_results/3482_Needles%20in%20the%20Haystack_%20Addressing%20Signal%20Dilution%20Improves%20scRNA-seq%20Perturbation%20Response%20Modelin/images/81db3b784d3db2f962c616ae3b9360eae563cb2c700536f3b324001a3d5ceb65.jpg)

![9f5d1581a74e748df8f8975105acb4f74f241de227e78ec09548504f4477958e.jpg](../icml_results/3482_Needles%20in%20the%20Haystack_%20Addressing%20Signal%20Dilution%20Improves%20scRNA-seq%20Perturbation%20Response%20Modelin/images/9f5d1581a74e748df8f8975105acb4f74f241de227e78ec09548504f4477958e.jpg)

![ab7be574689f375db221faa8ab436c828fb27f5b75de60335f38125c32eeb789.jpg](../icml_results/3482_Needles%20in%20the%20Haystack_%20Addressing%20Signal%20Dilution%20Improves%20scRNA-seq%20Perturbation%20Response%20Modelin/images/ab7be574689f375db221faa8ab436c828fb27f5b75de60335f38125c32eeb789.jpg)

![b07cffe12860c29dc6a3239a89c08ab6126901a1009d444f57c25fc96a595c5d.jpg](../icml_results/3482_Needles%20in%20the%20Haystack_%20Addressing%20Signal%20Dilution%20Improves%20scRNA-seq%20Perturbation%20Response%20Modelin/images/b07cffe12860c29dc6a3239a89c08ab6126901a1009d444f57c25fc96a595c5d.jpg)

![cfe58e8c431258afe15a6a2c5fca08c429e5b80727494497fad1c0b4cd051ccc.jpg](../icml_results/3482_Needles%20in%20the%20Haystack_%20Addressing%20Signal%20Dilution%20Improves%20scRNA-seq%20Perturbation%20Response%20Modelin/images/cfe58e8c431258afe15a6a2c5fca08c429e5b80727494497fad1c0b4cd051ccc.jpg)

![e149000555cd63c6a8c16fc10c935c01a02e61f1d6a558261d6c1b2729b5b64e.jpg](../icml_results/3482_Needles%20in%20the%20Haystack_%20Addressing%20Signal%20Dilution%20Improves%20scRNA-seq%20Perturbation%20Response%20Modelin/images/e149000555cd63c6a8c16fc10c935c01a02e61f1d6a558261d6c1b2729b5b64e.jpg)

### Tables

![0accd3815bda6f2514c0e5de7797f5e30fb049d336d6f4606131d9177ec529fb.jpg](../icml_results/3482_Needles%20in%20the%20Haystack_%20Addressing%20Signal%20Dilution%20Improves%20scRNA-seq%20Perturbation%20Response%20Modelin/tables/0accd3815bda6f2514c0e5de7797f5e30fb049d336d6f4606131d9177ec529fb.jpg)

![1a78bd4ad1f026cf523fff42b7ca437d52fbfb290276f871af549b7df2b1fc69.jpg](../icml_results/3482_Needles%20in%20the%20Haystack_%20Addressing%20Signal%20Dilution%20Improves%20scRNA-seq%20Perturbation%20Response%20Modelin/tables/1a78bd4ad1f026cf523fff42b7ca437d52fbfb290276f871af549b7df2b1fc69.jpg)

![5c39f908dd0f3c8a27d52a2997e0c1f752e97674189272d1cf9845dc58af1f7d.jpg](../icml_results/3482_Needles%20in%20the%20Haystack_%20Addressing%20Signal%20Dilution%20Improves%20scRNA-seq%20Perturbation%20Response%20Modelin/tables/5c39f908dd0f3c8a27d52a2997e0c1f752e97674189272d1cf9845dc58af1f7d.jpg)

![dacbd81afc74954ecc7024ba19591ee3b3d3dac4c28ad7f5b585cf1d3ab348a9.jpg](../icml_results/3482_Needles%20in%20the%20Haystack_%20Addressing%20Signal%20Dilution%20Improves%20scRNA-seq%20Perturbation%20Response%20Modelin/tables/dacbd81afc74954ecc7024ba19591ee3b3d3dac4c28ad7f5b585cf1d3ab348a9.jpg)

![f05769202aec55a765182dea4b46e1b04461b916c9283cc662c483001f073d3a.jpg](../icml_results/3482_Needles%20in%20the%20Haystack_%20Addressing%20Signal%20Dilution%20Improves%20scRNA-seq%20Perturbation%20Response%20Modelin/tables/f05769202aec55a765182dea4b46e1b04461b916c9283cc662c483001f073d3a.jpg)

<a id="paper-3443"></a>

## SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering


### Images

![2c2cdce42810449baaaa0745a4a9e598e22321aa47e98f80a28dad30f10b641a.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/images/2c2cdce42810449baaaa0745a4a9e598e22321aa47e98f80a28dad30f10b641a.jpg)

![2efb1597fa70ab5669f7b446e34128f6bbfa10c04a473f8f0dcc5c726d819d5a.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/images/2efb1597fa70ab5669f7b446e34128f6bbfa10c04a473f8f0dcc5c726d819d5a.jpg)

![32a7b28a26cc910ac9f5800d819ff7b9bd02f63bb1b38df75e49e9b21a0d4c64.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/images/32a7b28a26cc910ac9f5800d819ff7b9bd02f63bb1b38df75e49e9b21a0d4c64.jpg)

![32e047db8079d0fe324f3c0eeb81411a6ac26a23653ba5c23b6f91b4f0b94bc7.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/images/32e047db8079d0fe324f3c0eeb81411a6ac26a23653ba5c23b6f91b4f0b94bc7.jpg)

![4fdd5f3d420b41803e7c652aae43116e7c4987a36de2ff0202215052a73dda7d.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/images/4fdd5f3d420b41803e7c652aae43116e7c4987a36de2ff0202215052a73dda7d.jpg)

![50a050fb7c561ec28ca23624389a5fb5f6081855d1886aea55988ca6b9fb3cd9.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/images/50a050fb7c561ec28ca23624389a5fb5f6081855d1886aea55988ca6b9fb3cd9.jpg)

![56a4e070dcaa152dd662a15b96400aa5c539591420cb6604797a938109606cf5.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/images/56a4e070dcaa152dd662a15b96400aa5c539591420cb6604797a938109606cf5.jpg)

![5c515071406ed271569306b15aba0da88db1d442991814dbd38a7b9b9594bf0a.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/images/5c515071406ed271569306b15aba0da88db1d442991814dbd38a7b9b9594bf0a.jpg)

![7ec892978fe5d6e55b01cde9a295dce9ac47190bf32a08fd590aecd7defcc533.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/images/7ec892978fe5d6e55b01cde9a295dce9ac47190bf32a08fd590aecd7defcc533.jpg)

![8a3d9966b00d67088b8ebafa03440229dd22499880e7a16af29c0cf3e0928706.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/images/8a3d9966b00d67088b8ebafa03440229dd22499880e7a16af29c0cf3e0928706.jpg)

![8b143fe6158d1471851802c160bc28f0c1fa75c82472cb7bda547dc35e5f9388.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/images/8b143fe6158d1471851802c160bc28f0c1fa75c82472cb7bda547dc35e5f9388.jpg)

![c1e1ee5c592102bfdfa23855c3ce11e571402a9fea689b26212b02adeb03d03a.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/images/c1e1ee5c592102bfdfa23855c3ce11e571402a9fea689b26212b02adeb03d03a.jpg)

![e7aa17cd7b062ef8a1606f0044bd615e2767f2c79a710fa84d4ad27adac7849b.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/images/e7aa17cd7b062ef8a1606f0044bd615e2767f2c79a710fa84d4ad27adac7849b.jpg)

![f597b1f6e263d884805b9f7d7998903bd0f2108d45fc037b643f22cc005f68ce.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/images/f597b1f6e263d884805b9f7d7998903bd0f2108d45fc037b643f22cc005f68ce.jpg)

![f942bab06c2c45e2f97684c0618f097ef2e18687b703a1a13b4583dfc041f499.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/images/f942bab06c2c45e2f97684c0618f097ef2e18687b703a1a13b4583dfc041f499.jpg)

![fa278f0c946b1791f1790ab7004cffd7280c4aced928497a8c39a59af42a028d.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/images/fa278f0c946b1791f1790ab7004cffd7280c4aced928497a8c39a59af42a028d.jpg)

### Tables

![239c0774a0f6cf333a95675f9ae58630f4f4f5858d6ad64f601b818cb8d54f4b.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/tables/239c0774a0f6cf333a95675f9ae58630f4f4f5858d6ad64f601b818cb8d54f4b.jpg)

![46420f99043d64879ec232598abcb47757366e8507402992d9c4a29a20df2340.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/tables/46420f99043d64879ec232598abcb47757366e8507402992d9c4a29a20df2340.jpg)

![48c1961091292063355d9e329b8c3df6d5e5be7b8daf83d705a14c87f4582ea1.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/tables/48c1961091292063355d9e329b8c3df6d5e5be7b8daf83d705a14c87f4582ea1.jpg)

![585f3bdd5ad9dd4b630159497c55888f55476909d33077b974e7ed6f5dc4cdd7.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/tables/585f3bdd5ad9dd4b630159497c55888f55476909d33077b974e7ed6f5dc4cdd7.jpg)

![72ac19591790f4c5022cea427e9f0d6bdf1acee9880b20c27310a1b3742ad58b.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/tables/72ac19591790f4c5022cea427e9f0d6bdf1acee9880b20c27310a1b3742ad58b.jpg)

![7629e0afbf95f3f216fb149dff1c0436193cacbd7701b70779073858a9cba22c.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/tables/7629e0afbf95f3f216fb149dff1c0436193cacbd7701b70779073858a9cba22c.jpg)

![8470881bfd25753574de96b31efd1a8e854394282a1412e1e67b289b34ad4eb3.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/tables/8470881bfd25753574de96b31efd1a8e854394282a1412e1e67b289b34ad4eb3.jpg)

![8f32b23116269a680c776c61a0f28debd74a9ecce061723ab5f5f4ec57056a9c.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/tables/8f32b23116269a680c776c61a0f28debd74a9ecce061723ab5f5f4ec57056a9c.jpg)

![97f8beb8c0dc7de462e7f696be9dbdc51abed55df96c156838a5a1afe7188bc8.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/tables/97f8beb8c0dc7de462e7f696be9dbdc51abed55df96c156838a5a1afe7188bc8.jpg)

![af97e28570ac2c90186529be72c2d7f5409be66c8c1dcd602e5b3ebc7c0d493b.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/tables/af97e28570ac2c90186529be72c2d7f5409be66c8c1dcd602e5b3ebc7c0d493b.jpg)

![d8d600927bc2a4ebf20a0744078f3dd15d6644c82b455af7fe6819ebb2d45ca4.jpg](../icml_results/3483_SonicMaster_%20Towards%20Controllable%20All-in-One%20Music%20Restoration%20and%20Mastering/tables/d8d600927bc2a4ebf20a0744078f3dd15d6644c82b455af7fe6819ebb2d45ca4.jpg)

<a id="paper-3444"></a>

## ActiveUltraFeedback: Efficient Preference Data Generation using Active Learning


### Images

![0408a31f60c2c199a8e839db2b175a8e5b3b16929c3acd23fdb25443a5fa8a4a.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/images/0408a31f60c2c199a8e839db2b175a8e5b3b16929c3acd23fdb25443a5fa8a4a.jpg)

![0f5fc072cdb84509896e8db565debd4cc29f23f91bab7865e67f7eac622456e8.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/images/0f5fc072cdb84509896e8db565debd4cc29f23f91bab7865e67f7eac622456e8.jpg)

![13e9be3c5150c3700126482f3cfae1cbe573a11bb74b7b3ff3e3fe87c225bbd1.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/images/13e9be3c5150c3700126482f3cfae1cbe573a11bb74b7b3ff3e3fe87c225bbd1.jpg)

![39a86c7ea917b04dcd9b5bf73790e1ebbb585e25e459565cf14a6731970657ef.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/images/39a86c7ea917b04dcd9b5bf73790e1ebbb585e25e459565cf14a6731970657ef.jpg)

![5859ed3824003056e4db469ac7c61f69d0a4ae0d738db05783d74bcc507a7f39.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/images/5859ed3824003056e4db469ac7c61f69d0a4ae0d738db05783d74bcc507a7f39.jpg)

![69cbdca3dd89f46045c8c8bb7241fd4c8bc7af4c10ac2c1a3de73fd95d069c8a.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/images/69cbdca3dd89f46045c8c8bb7241fd4c8bc7af4c10ac2c1a3de73fd95d069c8a.jpg)

![69d0fe11a5d345e21585f651010ff76894a86c6ff60cce69ea5565e2134039ce.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/images/69d0fe11a5d345e21585f651010ff76894a86c6ff60cce69ea5565e2134039ce.jpg)

![7027db2fc5d2f1fd9d126bb9c6036cf07608128b579c8193b407e58b23c6c5ba.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/images/7027db2fc5d2f1fd9d126bb9c6036cf07608128b579c8193b407e58b23c6c5ba.jpg)

![7078e1104bca9b764cb3c512a69f5160a405956168216a2ba97ef2b27fb4bace.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/images/7078e1104bca9b764cb3c512a69f5160a405956168216a2ba97ef2b27fb4bace.jpg)

![959739f03d5b24d88319b74baf20b60ca6108a9ce3955cf85646dd69a7ac09f0.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/images/959739f03d5b24d88319b74baf20b60ca6108a9ce3955cf85646dd69a7ac09f0.jpg)

![a08c27bee950cce1d6d34d4f00246f630e4e5e8c02059448ee6e03c3aadcab35.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/images/a08c27bee950cce1d6d34d4f00246f630e4e5e8c02059448ee6e03c3aadcab35.jpg)

![b8fb00ee284082f940da0a50d6bc0d59ccb9e8b1818e291d6cfd7446a14f2d99.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/images/b8fb00ee284082f940da0a50d6bc0d59ccb9e8b1818e291d6cfd7446a14f2d99.jpg)

![c0c358efc1863a586e690b534be19d76750b307ec21c3557373c10cca7e53fa0.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/images/c0c358efc1863a586e690b534be19d76750b307ec21c3557373c10cca7e53fa0.jpg)

![c9b1cbd91e82b991b4f4b79c307ff82c16db18cd38ed952148860fa3ee846f75.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/images/c9b1cbd91e82b991b4f4b79c307ff82c16db18cd38ed952148860fa3ee846f75.jpg)

![dd0767e11f726ff8aff74584327526298588e1bf77e66b6edc4a7254af589907.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/images/dd0767e11f726ff8aff74584327526298588e1bf77e66b6edc4a7254af589907.jpg)

![f75e9f7d8181265b82f1fc9fd86d979c79564d86e0903af2964abbc815b6d980.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/images/f75e9f7d8181265b82f1fc9fd86d979c79564d86e0903af2964abbc815b6d980.jpg)

### Tables

![10b7caf8bdf9bc0a5530cca3f77fc293ad1588138078908aa4a1beb68c758b1b.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/10b7caf8bdf9bc0a5530cca3f77fc293ad1588138078908aa4a1beb68c758b1b.jpg)

![16d93dbd0f772e69ad52b94e060558f5d67e6361dcb4ae26905530dff53a8dbb.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/16d93dbd0f772e69ad52b94e060558f5d67e6361dcb4ae26905530dff53a8dbb.jpg)

![1cb9f7eb4c1704b7a880022a3a6eda5e14b9bfe4852cb7c086f5c195698a646c.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/1cb9f7eb4c1704b7a880022a3a6eda5e14b9bfe4852cb7c086f5c195698a646c.jpg)

![1f3a31aa0fc16213d8e204b794617fcd20a42bec1323c3ddd1251ddfa5ad4240.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/1f3a31aa0fc16213d8e204b794617fcd20a42bec1323c3ddd1251ddfa5ad4240.jpg)

![2bdef2663b28b4178a9bc03bebd3dd52063087fb8c005ce6bafea7894c13284c.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/2bdef2663b28b4178a9bc03bebd3dd52063087fb8c005ce6bafea7894c13284c.jpg)

![2cb88ed3ef4152d76e5d20659e9f993131667717cd04f2c2c4da15fd534848d3.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/2cb88ed3ef4152d76e5d20659e9f993131667717cd04f2c2c4da15fd534848d3.jpg)

![2e86e635db668d88019ef3e31f0564a8302a915f2bd53232e3aa3923756c716e.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/2e86e635db668d88019ef3e31f0564a8302a915f2bd53232e3aa3923756c716e.jpg)

![3a8b154d251787a9f219200e862ac0bcf470f6e343d3f2023605adcfb43815db.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/3a8b154d251787a9f219200e862ac0bcf470f6e343d3f2023605adcfb43815db.jpg)

![46165fb57576227961bf76aa78bb6d220da481cdfdc743c667a931f68de10343.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/46165fb57576227961bf76aa78bb6d220da481cdfdc743c667a931f68de10343.jpg)

![4fe4386746942a2c656f92bbaaf721825ec5bd8debc6cfc7dfe8f9604c2c6d17.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/4fe4386746942a2c656f92bbaaf721825ec5bd8debc6cfc7dfe8f9604c2c6d17.jpg)

![53f1e8e108870ac73f4829b6a2e88e176ebc54a694f09749dcb35261cd2b868a.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/53f1e8e108870ac73f4829b6a2e88e176ebc54a694f09749dcb35261cd2b868a.jpg)

![56e2b8955547ef87bdb931b3d5a7be233a12f303427ab45f5424ad151ba08bf8.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/56e2b8955547ef87bdb931b3d5a7be233a12f303427ab45f5424ad151ba08bf8.jpg)

![579b89631b59ee925bebc7e83dabbf4857df0adc07336b2504e327f9dbdaf35c.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/579b89631b59ee925bebc7e83dabbf4857df0adc07336b2504e327f9dbdaf35c.jpg)

![5973af10a67f25641c75ffaaa065c7fe4fb3a1e99c0ea667f232ffaa08087cf1.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/5973af10a67f25641c75ffaaa065c7fe4fb3a1e99c0ea667f232ffaa08087cf1.jpg)

![6484601348202eb87ebc29147677a0440fc483c7c8c1b38cae656e66d90e1f9f.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/6484601348202eb87ebc29147677a0440fc483c7c8c1b38cae656e66d90e1f9f.jpg)

![6e402f37a2ebb1f6f1f06515d32f581fe531da47ad198a0532c9a3f6e993a4dc.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/6e402f37a2ebb1f6f1f06515d32f581fe531da47ad198a0532c9a3f6e993a4dc.jpg)

![815bd5cfbb2b123fffc1be1e01b4c7bcdfff7f845525225ada3c00237225b698.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/815bd5cfbb2b123fffc1be1e01b4c7bcdfff7f845525225ada3c00237225b698.jpg)

![91bc3f500ea65beab8a16a2c0860d27f7b93c4690175ee119471a259bb99c5c0.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/91bc3f500ea65beab8a16a2c0860d27f7b93c4690175ee119471a259bb99c5c0.jpg)

![9268a27d966f9b20473285e8e00b1e4b7429942da212c1875d0eec4cc4b9a89f.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/9268a27d966f9b20473285e8e00b1e4b7429942da212c1875d0eec4cc4b9a89f.jpg)

![a1ef2565301d91ba8da7d886f49f4790283c87c2161f410eeb389fcb2dbfa89c.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/a1ef2565301d91ba8da7d886f49f4790283c87c2161f410eeb389fcb2dbfa89c.jpg)

![a44c13d1e55fa92a33f51e751835876938845649f0a46b7438bb03a92e482ff5.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/a44c13d1e55fa92a33f51e751835876938845649f0a46b7438bb03a92e482ff5.jpg)

![a57e8d7cf6fe91a5206ad76d4a6dae6f47e221341c6e035db018ba8e0b360493.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/a57e8d7cf6fe91a5206ad76d4a6dae6f47e221341c6e035db018ba8e0b360493.jpg)

![b99e06f2930c8e11ed8a2b854caa1f761066c64bdfc463b784671b58cab1f01f.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/b99e06f2930c8e11ed8a2b854caa1f761066c64bdfc463b784671b58cab1f01f.jpg)

![bd30545ec2ba0ec1ef65693577624095975f3eabb2ae63791557bc571fab75f3.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/bd30545ec2ba0ec1ef65693577624095975f3eabb2ae63791557bc571fab75f3.jpg)

![bf95b87c9b8e20d07eaf6b1c3292f8ac60bb7084291de38f4f3c7297c2a3ef5b.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/bf95b87c9b8e20d07eaf6b1c3292f8ac60bb7084291de38f4f3c7297c2a3ef5b.jpg)

![cca216bd38139aac9e62172ddc787c44dbcc448392389252c6263648ecdcbc9e.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/cca216bd38139aac9e62172ddc787c44dbcc448392389252c6263648ecdcbc9e.jpg)

![d58e8fceada5f1746ddf08a83d394dbdf22bd07d58c14cd753fdd91939431b6f.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/d58e8fceada5f1746ddf08a83d394dbdf22bd07d58c14cd753fdd91939431b6f.jpg)

![e8dd95754db4e50974c651a4c1150983c01cac99b734da757b6385aee47e9960.jpg](../icml_results/3484_ActiveUltraFeedback_%20Efficient%20Preference%20Data%20Generation%20using%20Active%20Learning/tables/e8dd95754db4e50974c651a4c1150983c01cac99b734da757b6385aee47e9960.jpg)

<a id="paper-3445"></a>

## Frequentist Consistency of Prior-Data Fitted Networks for Causal Inference


### Images

![040c9d48e60bc27e41ae7212c71b845b32396663b7eb3cd9d5bb00fda63c6024.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/images/040c9d48e60bc27e41ae7212c71b845b32396663b7eb3cd9d5bb00fda63c6024.jpg)

![44a1e3fb4023fb87ce2f6aa793ef42bee1c109c751577a588926bf69c9aac3c7.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/images/44a1e3fb4023fb87ce2f6aa793ef42bee1c109c751577a588926bf69c9aac3c7.jpg)

![56b28a4a02632de5c36b5ea6ce3126ea803c0d4d220de91cc7c4e5a4f4788e79.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/images/56b28a4a02632de5c36b5ea6ce3126ea803c0d4d220de91cc7c4e5a4f4788e79.jpg)

![57ea5df22cf3a0f79574910cf413c1bc17c59f507e645406c9f1b83d6c7c0afb.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/images/57ea5df22cf3a0f79574910cf413c1bc17c59f507e645406c9f1b83d6c7c0afb.jpg)

![5acf08a77b4626eb7f449edc8aed17d0fb3e69302a0168c937c5eb2c6a41cb92.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/images/5acf08a77b4626eb7f449edc8aed17d0fb3e69302a0168c937c5eb2c6a41cb92.jpg)

![5ede55ae3dfb1cbfd1be8fcfa5322449da3ae1471143ee5c6ab091673494672a.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/images/5ede55ae3dfb1cbfd1be8fcfa5322449da3ae1471143ee5c6ab091673494672a.jpg)

![65992bc33b0ca3d090ede2c56479b08be0f2dce294b86943e6d18f4401e06213.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/images/65992bc33b0ca3d090ede2c56479b08be0f2dce294b86943e6d18f4401e06213.jpg)

![66ff049730347c9ebee5bd1b0f1ee8e3391544b9863b0d5342dbcf0f86f61ea1.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/images/66ff049730347c9ebee5bd1b0f1ee8e3391544b9863b0d5342dbcf0f86f61ea1.jpg)

![801174c00054488c22ca58773be50bad4529e43629342fa67dff017b0732e790.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/images/801174c00054488c22ca58773be50bad4529e43629342fa67dff017b0732e790.jpg)

![84b4379839b80e258e54c6f27f1ca63fbc9b94b5831d1cc310d33566ecbf5924.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/images/84b4379839b80e258e54c6f27f1ca63fbc9b94b5831d1cc310d33566ecbf5924.jpg)

![883a0a92352528ccb93c4d2b2b67d218089837995c8cc1edd9bc24962364fab5.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/images/883a0a92352528ccb93c4d2b2b67d218089837995c8cc1edd9bc24962364fab5.jpg)

![8844dd251ad69f5fae4a450bac971c2fac62b7ec44bb26bb9bed8bd32b5e6e81.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/images/8844dd251ad69f5fae4a450bac971c2fac62b7ec44bb26bb9bed8bd32b5e6e81.jpg)

![bf429031dc569f8de48a0e4a2a6b86ed326b08b7335d51d55a9a25c4e900f60b.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/images/bf429031dc569f8de48a0e4a2a6b86ed326b08b7335d51d55a9a25c4e900f60b.jpg)

![d3009ea6a1babee1b899f0eb4413ecd41aa51f2a929b45a6d40b3f3ba9957f46.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/images/d3009ea6a1babee1b899f0eb4413ecd41aa51f2a929b45a6d40b3f3ba9957f46.jpg)

![f04c0fb7c300c1164e770628cb033b7aa13d2eebf5285f7c336100db666fce40.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/images/f04c0fb7c300c1164e770628cb033b7aa13d2eebf5285f7c336100db666fce40.jpg)

![ff9cc29fcf3cb4748bfe0d8eb279e21aa27f6cbcfc892e0748f3607c824d853a.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/images/ff9cc29fcf3cb4748bfe0d8eb279e21aa27f6cbcfc892e0748f3607c824d853a.jpg)

### Tables

![30bd84c014dc07da72175508ff1507cbff6a06294cc0a048e5cb74b030ba909e.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/tables/30bd84c014dc07da72175508ff1507cbff6a06294cc0a048e5cb74b030ba909e.jpg)

![38f97b06ab2c27a276f275342c1147de92300f210d8f1974483d8147a80ae21a.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/tables/38f97b06ab2c27a276f275342c1147de92300f210d8f1974483d8147a80ae21a.jpg)

![4f68310f559df25869d6f32239eae5b2e782c2cbf35e0480352ea66213fcf305.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/tables/4f68310f559df25869d6f32239eae5b2e782c2cbf35e0480352ea66213fcf305.jpg)

![8cc6801ec032fdb6163cbb67e7f4554e3d5a21b87414199bd57d69cdc5ebcbc5.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/tables/8cc6801ec032fdb6163cbb67e7f4554e3d5a21b87414199bd57d69cdc5ebcbc5.jpg)

![f2675078fa358546af319d677fcd8275ed88f0c3ee3769fc981631c3b625f527.jpg](../icml_results/3485_Frequentist%20Consistency%20of%20Prior-Data%20Fitted%20Networks%20for%20Causal%20Inference/tables/f2675078fa358546af319d677fcd8275ed88f0c3ee3769fc981631c3b625f527.jpg)

<a id="paper-3446"></a>

## Learning to Emulate Chaos: Adversarial Optimal Transport Regularization


### Images

![01cba8387422771653bee20bcbc6ea4d182a09aa8264b973ee6630e81d947e62.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/images/01cba8387422771653bee20bcbc6ea4d182a09aa8264b973ee6630e81d947e62.jpg)

![21a6af15f4f7da21a8491a69b1592d663800afaa93be39bdf8c0c2d8ff31f577.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/images/21a6af15f4f7da21a8491a69b1592d663800afaa93be39bdf8c0c2d8ff31f577.jpg)

![4a5b3d95874e354ee4d8ae2799b791449e0a1ccc81e9f6aac1e7a0a21565310b.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/images/4a5b3d95874e354ee4d8ae2799b791449e0a1ccc81e9f6aac1e7a0a21565310b.jpg)

![541978f12af117222a8dd9c60925424bc3587a23f4fa050dc24a08135cee3b02.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/images/541978f12af117222a8dd9c60925424bc3587a23f4fa050dc24a08135cee3b02.jpg)

![83a9d4ef0b8a582dafbe97c8976d88d70ab941000897fb078c7de11011b9dc7a.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/images/83a9d4ef0b8a582dafbe97c8976d88d70ab941000897fb078c7de11011b9dc7a.jpg)

![854773771729732ec2837b8805c6cdab01257aca277d967d8c199cb9b4cca519.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/images/854773771729732ec2837b8805c6cdab01257aca277d967d8c199cb9b4cca519.jpg)

![95284b72b260645e23edccd10c5dfbcbf00c865ce5da100364508fe1b12bac8d.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/images/95284b72b260645e23edccd10c5dfbcbf00c865ce5da100364508fe1b12bac8d.jpg)

![bc696b4db9848894b84eb9c4f7c142fd8101e14de43d1702f07b23f1b937c437.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/images/bc696b4db9848894b84eb9c4f7c142fd8101e14de43d1702f07b23f1b937c437.jpg)

![c6bc3e05abb4277bfdbe17c4dd636162de43b38fa36503b531bbd254508e183f.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/images/c6bc3e05abb4277bfdbe17c4dd636162de43b38fa36503b531bbd254508e183f.jpg)

![d2b843e2f28c1d5f55e6228b3f86cb831df31dfe238889814210908d3b3d6588.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/images/d2b843e2f28c1d5f55e6228b3f86cb831df31dfe238889814210908d3b3d6588.jpg)

![e70db8c632968dabfce2aef74043089398dd31e30d38997ca8457ea0e8d5bdc5.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/images/e70db8c632968dabfce2aef74043089398dd31e30d38997ca8457ea0e8d5bdc5.jpg)

### Tables

![04f21f580e970113a021a2e904cdec54ec94ba5b88e58cf180fbaba3fc858780.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/tables/04f21f580e970113a021a2e904cdec54ec94ba5b88e58cf180fbaba3fc858780.jpg)

![400925740e7c85faa6c9ec7f729443e75ffceb80abd164f648fead02b76dd431.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/tables/400925740e7c85faa6c9ec7f729443e75ffceb80abd164f648fead02b76dd431.jpg)

![45b599415462d78d7c7253bdf5f97bb0f8c9667fea21f8f4a4642f081ddf3c83.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/tables/45b599415462d78d7c7253bdf5f97bb0f8c9667fea21f8f4a4642f081ddf3c83.jpg)

![462ac55007100f74d40700b0139a7b4e41628f20fed9da22100a8128e68236d0.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/tables/462ac55007100f74d40700b0139a7b4e41628f20fed9da22100a8128e68236d0.jpg)

![6258287b29a431fbe4a182c389dffaf370a101fb7df80ca654235b3c9b094fe5.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/tables/6258287b29a431fbe4a182c389dffaf370a101fb7df80ca654235b3c9b094fe5.jpg)

![64f82a649db33a554a70e98d39d3117d9eaf78e50cf8ee5714cd938f69093b0e.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/tables/64f82a649db33a554a70e98d39d3117d9eaf78e50cf8ee5714cd938f69093b0e.jpg)

![a1797e572b55493b4891341381830d97145a4432add0e24a931f789166211cf3.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/tables/a1797e572b55493b4891341381830d97145a4432add0e24a931f789166211cf3.jpg)

![bc64434d9817b408e28439154084a4393e7f4c6a34f87c8024e35b92159ac2af.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/tables/bc64434d9817b408e28439154084a4393e7f4c6a34f87c8024e35b92159ac2af.jpg)

![ca43a1aefc8c8571c209453f764cb615f15b92dbc4f2ba03e6e7ce799e366223.jpg](../icml_results/3486_Learning%20to%20Emulate%20Chaos_%20Adversarial%20Optimal%20Transport%20Regularization/tables/ca43a1aefc8c8571c209453f764cb615f15b92dbc4f2ba03e6e7ce799e366223.jpg)

<a id="paper-3447"></a>

## Mechanistic Interpretability as Statistical Estimation: A Variance Analysis


### Images

![1ce730a247d79eb9ea6cc694eb955f76ede8bb9a3a3e71388b01643bb418711c.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/images/1ce730a247d79eb9ea6cc694eb955f76ede8bb9a3a3e71388b01643bb418711c.jpg)

![364edf2c19ad1d3b14af1cc6987522d862898dda75da6ce909f8519281bf166a.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/images/364edf2c19ad1d3b14af1cc6987522d862898dda75da6ce909f8519281bf166a.jpg)

![397c4caf81b22d856f22a786405f6168e0e37e8931dcb9688c0ebddd23473fa8.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/images/397c4caf81b22d856f22a786405f6168e0e37e8931dcb9688c0ebddd23473fa8.jpg)

![9a4358e4590981fc7814848535b6f25542ef48b487a34a6b00c848ea83be951f.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/images/9a4358e4590981fc7814848535b6f25542ef48b487a34a6b00c848ea83be951f.jpg)

![b27697d62f028ac42f5a483458319d1034c734cfa0d1b9c2bd85f7f730d29a0b.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/images/b27697d62f028ac42f5a483458319d1034c734cfa0d1b9c2bd85f7f730d29a0b.jpg)

![c683ad3d6bccd01c8c8524dcb70df39673e38df0447644416c10395c240ef8e9.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/images/c683ad3d6bccd01c8c8524dcb70df39673e38df0447644416c10395c240ef8e9.jpg)

![e53b0f7b6cecbe50e77d63813c09e5088ebe25529b7693922e4dc519161fa6f4.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/images/e53b0f7b6cecbe50e77d63813c09e5088ebe25529b7693922e4dc519161fa6f4.jpg)

![e7630b3a98b8bf5491e5b9553961fa206029caa7c3b08a655613fb0761b22a87.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/images/e7630b3a98b8bf5491e5b9553961fa206029caa7c3b08a655613fb0761b22a87.jpg)

![f28f0002889d70b0feaa8d40817e89cdeb76026a36c15dca25debeac89298022.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/images/f28f0002889d70b0feaa8d40817e89cdeb76026a36c15dca25debeac89298022.jpg)

### Tables

![223672fe5f96b7e9c99fd6540acd5c706c0ed090feba24e4f5ce9aae5c4be73a.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/tables/223672fe5f96b7e9c99fd6540acd5c706c0ed090feba24e4f5ce9aae5c4be73a.jpg)

![521e4d356287cd125bcdb8be4dfda49621da62240c5973ec2643fd6a9c94386d.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/tables/521e4d356287cd125bcdb8be4dfda49621da62240c5973ec2643fd6a9c94386d.jpg)

![58133a6d39b32acc3e51432e7f71d1fbc0a051e2c0df0aced949f9da6e7a9e88.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/tables/58133a6d39b32acc3e51432e7f71d1fbc0a051e2c0df0aced949f9da6e7a9e88.jpg)

![771792fbcf0bd1097782a5e86b6e92fc0be717839264f5cab266b55ee1069efd.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/tables/771792fbcf0bd1097782a5e86b6e92fc0be717839264f5cab266b55ee1069efd.jpg)

![7df02d1f1b14328f7ec6b485dee2f1a5eb6b3879c319233779534fae05a122c7.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/tables/7df02d1f1b14328f7ec6b485dee2f1a5eb6b3879c319233779534fae05a122c7.jpg)

![8b6bdc74c8e056eaa3fcc08fe1fec8f20df6c0e4f6e4bfc56106bfb4294f9f65.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/tables/8b6bdc74c8e056eaa3fcc08fe1fec8f20df6c0e4f6e4bfc56106bfb4294f9f65.jpg)

![8e23f0baa2f208bf01f7b5789dd854691d750f180ed843f3bfe841b4a6ccc55c.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/tables/8e23f0baa2f208bf01f7b5789dd854691d750f180ed843f3bfe841b4a6ccc55c.jpg)

![d94da634d0f02dfd43bda746abfc843cf3acd4d09f8599bcf4791afceec9c677.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/tables/d94da634d0f02dfd43bda746abfc843cf3acd4d09f8599bcf4791afceec9c677.jpg)

![f20cfd81112972948cd5dfc632da5cde0137eb3d16962a10d524b01d6fe2371d.jpg](../icml_results/3487_Mechanistic%20Interpretability%20as%20Statistical%20Estimation_%20A%20Variance%20Analysis/tables/f20cfd81112972948cd5dfc632da5cde0137eb3d16962a10d524b01d6fe2371d.jpg)

<a id="paper-3448"></a>

## FunCQNet: A Functional Censored Quantile Neural Network for Predicting Long-Term Post-Transplant Kidney Survival


### Images

![00960e3e18f71c8fdd582dbfd0a5e66c70e164946d11092ef52a3b7b0066443b.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/images/00960e3e18f71c8fdd582dbfd0a5e66c70e164946d11092ef52a3b7b0066443b.jpg)

![36ffd3adf40cafa1c46f738284427240e481dafad1900d16db8d526fd0139e40.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/images/36ffd3adf40cafa1c46f738284427240e481dafad1900d16db8d526fd0139e40.jpg)

![4425431c8f0691055c80e968f9c1224164c8ff05fcc92d4e5baac1fe9ccb51eb.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/images/4425431c8f0691055c80e968f9c1224164c8ff05fcc92d4e5baac1fe9ccb51eb.jpg)

![55574107a638f29b658ce22ec9f519f8fffc05c905d5e97eee2bac84c19387b1.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/images/55574107a638f29b658ce22ec9f519f8fffc05c905d5e97eee2bac84c19387b1.jpg)

![666d6a0853caa672dee2b17852b0767af3885416576296baeb48c9aeded6ee10.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/images/666d6a0853caa672dee2b17852b0767af3885416576296baeb48c9aeded6ee10.jpg)

![6763711a22abb3d821437dbf76d833589bf2ab6669ae4adae8ec5467830803dc.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/images/6763711a22abb3d821437dbf76d833589bf2ab6669ae4adae8ec5467830803dc.jpg)

![77bc0b4c045da2f95b46551c5ff7583e527d913eb86a85dea946126000c9e7cd.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/images/77bc0b4c045da2f95b46551c5ff7583e527d913eb86a85dea946126000c9e7cd.jpg)

![7cac717873e08f0de7a20d1415bb72fd0ec02c99610b8e224cd41a0cf73a12ee.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/images/7cac717873e08f0de7a20d1415bb72fd0ec02c99610b8e224cd41a0cf73a12ee.jpg)

![8d88bec0baee5edfd74397908607bc2f0ec434bf6ccc3180b3219d88e0dd0f3e.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/images/8d88bec0baee5edfd74397908607bc2f0ec434bf6ccc3180b3219d88e0dd0f3e.jpg)

![9de13bb26fb97c88b1e3958a690ac9ef95e72b7db22305e1a81bffa3e5745b1d.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/images/9de13bb26fb97c88b1e3958a690ac9ef95e72b7db22305e1a81bffa3e5745b1d.jpg)

![b8b71d094c8cc7b8cdb9fa3e23d95dca59c1b9776fd3032b1c3c64a270bf3819.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/images/b8b71d094c8cc7b8cdb9fa3e23d95dca59c1b9776fd3032b1c3c64a270bf3819.jpg)

![bb2c7223a54658a7390f0c9918cc9c2372806689ded2d83457d72f143960382f.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/images/bb2c7223a54658a7390f0c9918cc9c2372806689ded2d83457d72f143960382f.jpg)

![ca28e1dafa6be5d17b441f372e6c71efb51a27f924803159ca2c26be22330848.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/images/ca28e1dafa6be5d17b441f372e6c71efb51a27f924803159ca2c26be22330848.jpg)

![e18b4a9357e0473646609cd8818cfb764fad88b9be032f0d430f7acc0394cb03.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/images/e18b4a9357e0473646609cd8818cfb764fad88b9be032f0d430f7acc0394cb03.jpg)

![eeff5fca9588a897aba0908945e0b1e542a3219ac5b1ebb8044aee15dbd67ab3.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/images/eeff5fca9588a897aba0908945e0b1e542a3219ac5b1ebb8044aee15dbd67ab3.jpg)

![ef138ffb4c4cf0b91629b292ed41e9d8a7a19caf15cc5c0d27873675f55635c5.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/images/ef138ffb4c4cf0b91629b292ed41e9d8a7a19caf15cc5c0d27873675f55635c5.jpg)

### Tables

![1bfe13f0d3900c1490a705dd1794e0fbb70732e6107f96279a3b6b07eda2e9d9.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/tables/1bfe13f0d3900c1490a705dd1794e0fbb70732e6107f96279a3b6b07eda2e9d9.jpg)

![20bb2b24f5b77fbef714aa91eccdc4121939566057a5d3462401ec0f454e3199.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/tables/20bb2b24f5b77fbef714aa91eccdc4121939566057a5d3462401ec0f454e3199.jpg)

![298904b3ff8ad952d14f37f62d51b083892d99e6efdcf91e2e1e2493153b145d.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/tables/298904b3ff8ad952d14f37f62d51b083892d99e6efdcf91e2e1e2493153b145d.jpg)

![37d4f2a33057ef29732c3b38178c90f99ee68ff1387239979fab2f3b9725435a.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/tables/37d4f2a33057ef29732c3b38178c90f99ee68ff1387239979fab2f3b9725435a.jpg)

![3b669c1fd4c8b02d4eea9abad04c14c4ca317c3728d106b2e40a09b158597747.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/tables/3b669c1fd4c8b02d4eea9abad04c14c4ca317c3728d106b2e40a09b158597747.jpg)

![3ea34a86a523f3a30863c567b15406ecc26f85cc814e3df21e390e483bab1eda.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/tables/3ea34a86a523f3a30863c567b15406ecc26f85cc814e3df21e390e483bab1eda.jpg)

![430679797bf2d4ee633dcbc229b1c64257eabcc98a72b5b3f093ab3ca1731a32.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/tables/430679797bf2d4ee633dcbc229b1c64257eabcc98a72b5b3f093ab3ca1731a32.jpg)

![58e3f18cf3b52a25736188b179b297b515abddc3c8c8f8d6bdcf819edb215d2d.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/tables/58e3f18cf3b52a25736188b179b297b515abddc3c8c8f8d6bdcf819edb215d2d.jpg)

![5ba759ccb53d90d6ae3f418befc99a94f97bc33297071471f3c0fe0aac5e0626.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/tables/5ba759ccb53d90d6ae3f418befc99a94f97bc33297071471f3c0fe0aac5e0626.jpg)

![79feb7e9e008b86c779ffcd1ad0d312b6546363430b6784f0d3d0f1839ebad74.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/tables/79feb7e9e008b86c779ffcd1ad0d312b6546363430b6784f0d3d0f1839ebad74.jpg)

![847cdd16205e43962eabcbb14cf81db2ecfa45c54e68e01f0d551b36a6406ee5.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/tables/847cdd16205e43962eabcbb14cf81db2ecfa45c54e68e01f0d551b36a6406ee5.jpg)

![91bac3b988f055a31be5bb238e1c2cc3a14e20116d8e9170622366ecb4898fe8.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/tables/91bac3b988f055a31be5bb238e1c2cc3a14e20116d8e9170622366ecb4898fe8.jpg)

![994eddb843b2e727b9d1132c429b61c7664b18c7f777a22319acc57cedf46c9e.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/tables/994eddb843b2e727b9d1132c429b61c7664b18c7f777a22319acc57cedf46c9e.jpg)

![ab44a61c483573767b41373734aad939e837a5421ac5fd14818207e4d4cb6ad1.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/tables/ab44a61c483573767b41373734aad939e837a5421ac5fd14818207e4d4cb6ad1.jpg)

![be0f8acbfb4098015f56eeffa0dccd8b6f62654cd3c54a1fa3f22e07e99f8b06.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/tables/be0f8acbfb4098015f56eeffa0dccd8b6f62654cd3c54a1fa3f22e07e99f8b06.jpg)

![c8d3e4dc0277a818aa2d6e88fe1f4af5fcd55962ca944d2cc7fb09c57aec564b.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/tables/c8d3e4dc0277a818aa2d6e88fe1f4af5fcd55962ca944d2cc7fb09c57aec564b.jpg)

![ef275d19b0c50526bc27705d1c8fffa32bd4999a3c70e34594d109667c885b41.jpg](../icml_results/3488_FunCQNet_%20A%20Functional%20Censored%20Quantile%20Neural%20Network%20for%20Predicting%20Long-Term%20Post-Transplant%20Kid/tables/ef275d19b0c50526bc27705d1c8fffa32bd4999a3c70e34594d109667c885b41.jpg)

<a id="paper-3449"></a>

## DistMatch: Adaptive Binning via Distribution Matching for Robust Sequential Conformal Prediction


### Images

![139e04322c496902ed3348ea280ccc2ac9dd9c7e590d45f2427fb79c9f79a09b.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/images/139e04322c496902ed3348ea280ccc2ac9dd9c7e590d45f2427fb79c9f79a09b.jpg)

![199344068037e06c9ede307d18864972d9ff1b898458077e0b6ac4723540551c.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/images/199344068037e06c9ede307d18864972d9ff1b898458077e0b6ac4723540551c.jpg)

![30427065edc71e02a4b7951da01573ea2b784024a19f86c5beb58703393f4526.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/images/30427065edc71e02a4b7951da01573ea2b784024a19f86c5beb58703393f4526.jpg)

![343cb7ebd6f7150a866ebe032f228aebc3a33510d1ea2b2d1d6679fefa967697.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/images/343cb7ebd6f7150a866ebe032f228aebc3a33510d1ea2b2d1d6679fefa967697.jpg)

![3ba58bcfeb358e397c5b9fc8bec45f4f01f8c0f951bfa400585ab93e8c396cd8.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/images/3ba58bcfeb358e397c5b9fc8bec45f4f01f8c0f951bfa400585ab93e8c396cd8.jpg)

![57fd1b2cca29b52bb47b181d9c391296dc88d3e4cfc2a2a83305804c9304cf0a.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/images/57fd1b2cca29b52bb47b181d9c391296dc88d3e4cfc2a2a83305804c9304cf0a.jpg)

![8714d0ad014efcf43b7e9ec16079704151164f530b19f23fcde6e17b12b28895.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/images/8714d0ad014efcf43b7e9ec16079704151164f530b19f23fcde6e17b12b28895.jpg)

![abc1ac92f182b00fa6358bbb32ec0a3aa9c336ffd8bab0d5b069ff6a59d4bcad.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/images/abc1ac92f182b00fa6358bbb32ec0a3aa9c336ffd8bab0d5b069ff6a59d4bcad.jpg)

![adaf10586db0514b079f742c3d2f35a814a7d9a3305153a8d48baa4108c42a1d.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/images/adaf10586db0514b079f742c3d2f35a814a7d9a3305153a8d48baa4108c42a1d.jpg)

![cb7cd1fe9cba0b1cd00ac413b73d155d6344314142ee9ae4c004dd00837fb1f5.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/images/cb7cd1fe9cba0b1cd00ac413b73d155d6344314142ee9ae4c004dd00837fb1f5.jpg)

![ce1f815a7aa3eb13af88bc0b07c46cf2784f8bbdc9e81ca7aa43aebb8ac64caf.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/images/ce1f815a7aa3eb13af88bc0b07c46cf2784f8bbdc9e81ca7aa43aebb8ac64caf.jpg)

![cf0179bbb33022fce104af166a551152fbac1f2f7effb55232ce616b886ffb3a.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/images/cf0179bbb33022fce104af166a551152fbac1f2f7effb55232ce616b886ffb3a.jpg)

![d816e808d6df331661e82bbd5c2f814924f878546111a9be8e31482d5bdea2b9.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/images/d816e808d6df331661e82bbd5c2f814924f878546111a9be8e31482d5bdea2b9.jpg)

![f51320c2a9882e8c0d74df6c95398d3d391ec0da4651eb33beec057964d7d472.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/images/f51320c2a9882e8c0d74df6c95398d3d391ec0da4651eb33beec057964d7d472.jpg)

### Tables

![105a77e8e3e1b9f0d499f05ad24db8b852732223ace02ac25247202d60eaa5f5.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/tables/105a77e8e3e1b9f0d499f05ad24db8b852732223ace02ac25247202d60eaa5f5.jpg)

![2481014e87b0c947911578b5c3be79c6869296f5d2db3ec02bfbf9879e46e9c7.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/tables/2481014e87b0c947911578b5c3be79c6869296f5d2db3ec02bfbf9879e46e9c7.jpg)

![35ecd20a5aa498ee45becc616bcca0279f466d7b4c67624abd085b16702f8905.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/tables/35ecd20a5aa498ee45becc616bcca0279f466d7b4c67624abd085b16702f8905.jpg)

![3ae0703f52191ed09672910327421b152c85602bad995e4be28b6f7f063972dd.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/tables/3ae0703f52191ed09672910327421b152c85602bad995e4be28b6f7f063972dd.jpg)

![446fb34759963f653ddfbec42bb4f144fb118f4ab77cdec6c373cc856bab21e1.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/tables/446fb34759963f653ddfbec42bb4f144fb118f4ab77cdec6c373cc856bab21e1.jpg)

![6e462db7ee16bca1c8cd132aa40b9f6edc6862886f46550a351b6d7377866185.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/tables/6e462db7ee16bca1c8cd132aa40b9f6edc6862886f46550a351b6d7377866185.jpg)

![86a639dc9a1890c77ac10bd77cde9cf561b0a09910cbf77236b3b78b6adfd172.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/tables/86a639dc9a1890c77ac10bd77cde9cf561b0a09910cbf77236b3b78b6adfd172.jpg)

![8e967d813915b1ff752ab5fe2432abf17d65007449e8f89211502b9746fff477.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/tables/8e967d813915b1ff752ab5fe2432abf17d65007449e8f89211502b9746fff477.jpg)

![8f80cafeb7fdbcb99176b77fd4e8bd258614d7bec5802e5bfcf2b0ae6c6b3ec3.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/tables/8f80cafeb7fdbcb99176b77fd4e8bd258614d7bec5802e5bfcf2b0ae6c6b3ec3.jpg)

![91ab7ed2f642178bdb8dfedee1e2b7d4f70666846e9d5234ab866b5ff6058c05.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/tables/91ab7ed2f642178bdb8dfedee1e2b7d4f70666846e9d5234ab866b5ff6058c05.jpg)

![932014567bb87372dc96fbe4d39aca1a4ef5f33a4e2692e41e2ba33c43e61856.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/tables/932014567bb87372dc96fbe4d39aca1a4ef5f33a4e2692e41e2ba33c43e61856.jpg)

![987bb751020bfe35aa6db68899ae6d55759b61104222943ca4e943136950ddef.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/tables/987bb751020bfe35aa6db68899ae6d55759b61104222943ca4e943136950ddef.jpg)

![a9d1795c96ffa64356904552190a9a06d6d678b327489080278aa8623505733c.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/tables/a9d1795c96ffa64356904552190a9a06d6d678b327489080278aa8623505733c.jpg)

![ae6cb54bf55640a8af55aec783f221c5321718c754f450e8ed8b704cadd4fb85.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/tables/ae6cb54bf55640a8af55aec783f221c5321718c754f450e8ed8b704cadd4fb85.jpg)

![dfa05715347e6d829323407aa93f3e00819ebf811b68f7b007e840ef3336d7c9.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/tables/dfa05715347e6d829323407aa93f3e00819ebf811b68f7b007e840ef3336d7c9.jpg)

![fcbd4a593dc4a926312bf0669ddb888abdd5230a029a8c1a5322fbf7c85ab956.jpg](../icml_results/3489_DistMatch_%20Adaptive%20Binning%20via%20Distribution%20Matching%20for%20Robust%20Sequential%20Conformal%20Prediction/tables/fcbd4a593dc4a926312bf0669ddb888abdd5230a029a8c1a5322fbf7c85ab956.jpg)

<a id="paper-3450"></a>

## A Solvable High-Dimensional Model Where Nonlinear Autoencoders Learn Structure Invisible to PCA While Test Loss Misaligns With Generalization


### Images

![1eebc7ee9b8444aa8e1ee3abfb674186d08f1b92d81182afb1b996c811ccdd2c.jpg](../icml_results/3490_A%20Solvable%20High-Dimensional%20Model%20Where%20Nonlinear%20Autoencoders%20Learn%20Structure%20Invisible%20to%20PCA%20Whil/images/1eebc7ee9b8444aa8e1ee3abfb674186d08f1b92d81182afb1b996c811ccdd2c.jpg)

![2aa2f8544022a37e0bbbd9f88d74a72610ef50d1714dab4f67f65d7ecac09818.jpg](../icml_results/3490_A%20Solvable%20High-Dimensional%20Model%20Where%20Nonlinear%20Autoencoders%20Learn%20Structure%20Invisible%20to%20PCA%20Whil/images/2aa2f8544022a37e0bbbd9f88d74a72610ef50d1714dab4f67f65d7ecac09818.jpg)

![476429e033a0da33551112aec18b89cb04c7345f6f5b263af9ec9886cdff48ce.jpg](../icml_results/3490_A%20Solvable%20High-Dimensional%20Model%20Where%20Nonlinear%20Autoencoders%20Learn%20Structure%20Invisible%20to%20PCA%20Whil/images/476429e033a0da33551112aec18b89cb04c7345f6f5b263af9ec9886cdff48ce.jpg)

![5668b74b20daa09daf51d7adaf7729bb322e547a5cc7d1b1dd4372c7f1b10207.jpg](../icml_results/3490_A%20Solvable%20High-Dimensional%20Model%20Where%20Nonlinear%20Autoencoders%20Learn%20Structure%20Invisible%20to%20PCA%20Whil/images/5668b74b20daa09daf51d7adaf7729bb322e547a5cc7d1b1dd4372c7f1b10207.jpg)

![6f700e72200c0f1195068c7337bd1506fda71cb23216756dc5fc0d7d021e0606.jpg](../icml_results/3490_A%20Solvable%20High-Dimensional%20Model%20Where%20Nonlinear%20Autoencoders%20Learn%20Structure%20Invisible%20to%20PCA%20Whil/images/6f700e72200c0f1195068c7337bd1506fda71cb23216756dc5fc0d7d021e0606.jpg)

![8634d74d980acf1ee0475676dd712d406b5bcc64683dae4e99888a7905af5ef7.jpg](../icml_results/3490_A%20Solvable%20High-Dimensional%20Model%20Where%20Nonlinear%20Autoencoders%20Learn%20Structure%20Invisible%20to%20PCA%20Whil/images/8634d74d980acf1ee0475676dd712d406b5bcc64683dae4e99888a7905af5ef7.jpg)

![adb00f5c9d2cf9b09c9e944774565e1f7cca54b5c043657e15d40e530f7e9668.jpg](../icml_results/3490_A%20Solvable%20High-Dimensional%20Model%20Where%20Nonlinear%20Autoencoders%20Learn%20Structure%20Invisible%20to%20PCA%20Whil/images/adb00f5c9d2cf9b09c9e944774565e1f7cca54b5c043657e15d40e530f7e9668.jpg)

![c44389a6ecf97ba4486e336634089b0da3e45d38c82a543f575da4747ab85ee4.jpg](../icml_results/3490_A%20Solvable%20High-Dimensional%20Model%20Where%20Nonlinear%20Autoencoders%20Learn%20Structure%20Invisible%20to%20PCA%20Whil/images/c44389a6ecf97ba4486e336634089b0da3e45d38c82a543f575da4747ab85ee4.jpg)

![cd4ee37551bdfaadd1550fd9fa0b834ee9fa79f493ac850f13084c2e5d940143.jpg](../icml_results/3490_A%20Solvable%20High-Dimensional%20Model%20Where%20Nonlinear%20Autoencoders%20Learn%20Structure%20Invisible%20to%20PCA%20Whil/images/cd4ee37551bdfaadd1550fd9fa0b834ee9fa79f493ac850f13084c2e5d940143.jpg)

![d0e67e1ccf7f3e73ae94bc31738dc7bd342198405ea946cd1501a8db67235e19.jpg](../icml_results/3490_A%20Solvable%20High-Dimensional%20Model%20Where%20Nonlinear%20Autoencoders%20Learn%20Structure%20Invisible%20to%20PCA%20Whil/images/d0e67e1ccf7f3e73ae94bc31738dc7bd342198405ea946cd1501a8db67235e19.jpg)

![ec1a9c49448c09750cdbd750a3de9ebb38a4f2f5dfc7d919b0fb18a02b2e31b8.jpg](../icml_results/3490_A%20Solvable%20High-Dimensional%20Model%20Where%20Nonlinear%20Autoencoders%20Learn%20Structure%20Invisible%20to%20PCA%20Whil/images/ec1a9c49448c09750cdbd750a3de9ebb38a4f2f5dfc7d919b0fb18a02b2e31b8.jpg)

### Tables

![6c83995c83aa5bb82bd892072ee09a471f733959d68f6cbbae41251fd38c89bd.jpg](../icml_results/3490_A%20Solvable%20High-Dimensional%20Model%20Where%20Nonlinear%20Autoencoders%20Learn%20Structure%20Invisible%20to%20PCA%20Whil/tables/6c83995c83aa5bb82bd892072ee09a471f733959d68f6cbbae41251fd38c89bd.jpg)

<a id="paper-3451"></a>

## Making Expert Reasoning Learnable with Self-Distillation


### Images

![0af9322f7b95b55c862b416dc0c2e34e199d25fbe8d26dc20cd5d9872136b358.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/images/0af9322f7b95b55c862b416dc0c2e34e199d25fbe8d26dc20cd5d9872136b358.jpg)

![28a2d49d7c76f4e0aec7f37535a40f6f827d95f5929700cf5dd4d7537bbd226e.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/images/28a2d49d7c76f4e0aec7f37535a40f6f827d95f5929700cf5dd4d7537bbd226e.jpg)

![5bf34e78cd2dcaf5e4998a5d19fd77ab1a3d9078f780a37c5e8f8d0a2ae9a67d.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/images/5bf34e78cd2dcaf5e4998a5d19fd77ab1a3d9078f780a37c5e8f8d0a2ae9a67d.jpg)

![5da1aa944f9a604b594bc73a54f98ddc80f45512970cf643e2a1ae2cd142949a.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/images/5da1aa944f9a604b594bc73a54f98ddc80f45512970cf643e2a1ae2cd142949a.jpg)

![679be8a3ca9f26a3becdac6b8cb88b5cc394995d803cea72478b1b26f7da6157.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/images/679be8a3ca9f26a3becdac6b8cb88b5cc394995d803cea72478b1b26f7da6157.jpg)

![8873ed99dbdde73e23a78044441bcd74428805737dfab3cefff85a57bc445c43.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/images/8873ed99dbdde73e23a78044441bcd74428805737dfab3cefff85a57bc445c43.jpg)

![9664a13a15226f33f40d9a85be86c5c3b4d7cc90345c63e53f00e8f0d6b0b4fd.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/images/9664a13a15226f33f40d9a85be86c5c3b4d7cc90345c63e53f00e8f0d6b0b4fd.jpg)

![9dd689841b0be34224ae63a9f33199908c91975bd4ba41e1ec316f6b13cc87db.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/images/9dd689841b0be34224ae63a9f33199908c91975bd4ba41e1ec316f6b13cc87db.jpg)

![bd343649a132c2d914f025e5564b90bf62352a0d3df91b376a93b74a2a91b9d4.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/images/bd343649a132c2d914f025e5564b90bf62352a0d3df91b376a93b74a2a91b9d4.jpg)

![c17a6a2bf5fc35e4bccac3cce42241e8ee760c339a3e64ae6bfa13d3a7aaff67.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/images/c17a6a2bf5fc35e4bccac3cce42241e8ee760c339a3e64ae6bfa13d3a7aaff67.jpg)

![cc6524bf2fdda4ef8c756fb9d20f5ca5759a8198d8dccc133b2a9a3ca04e32d9.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/images/cc6524bf2fdda4ef8c756fb9d20f5ca5759a8198d8dccc133b2a9a3ca04e32d9.jpg)

![e4e34ff40b9f63214605568333bd98928a97c162105447420672a555eb9835c4.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/images/e4e34ff40b9f63214605568333bd98928a97c162105447420672a555eb9835c4.jpg)

![ff84d69a3c57d98bc799e1277602e543d61fc4679ae4076a077d2ae52d5e86f0.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/images/ff84d69a3c57d98bc799e1277602e543d61fc4679ae4076a077d2ae52d5e86f0.jpg)

### Tables

![0d4e9848627de3beb8939ad329406de89526a9bd3c30544d4f7ff72eb7295402.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/tables/0d4e9848627de3beb8939ad329406de89526a9bd3c30544d4f7ff72eb7295402.jpg)

![1bfdde1baec6e725cc8424441c5f54f305d7f27e98e5249099a4d64968d5c3eb.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/tables/1bfdde1baec6e725cc8424441c5f54f305d7f27e98e5249099a4d64968d5c3eb.jpg)

![29ba1167e36dca89bdf623240ea463595b5863ab8b7baf62f10e6e2982cb329c.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/tables/29ba1167e36dca89bdf623240ea463595b5863ab8b7baf62f10e6e2982cb329c.jpg)

![41ad2d8820341c6af33283ff069aaa29175190853c538c16bde994d1be5fa266.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/tables/41ad2d8820341c6af33283ff069aaa29175190853c538c16bde994d1be5fa266.jpg)

![737ad22c5e47b6fa460f531d90c8dcbb48c0e04cc9b37ea4c35afd57ab891d48.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/tables/737ad22c5e47b6fa460f531d90c8dcbb48c0e04cc9b37ea4c35afd57ab891d48.jpg)

![9266c8a7a383e4d286725da50ea2a3872bdc1c8b6368d40a2d4259b3bbc9ba2d.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/tables/9266c8a7a383e4d286725da50ea2a3872bdc1c8b6368d40a2d4259b3bbc9ba2d.jpg)

![aaef945b6517db4141bc9a984f008535e39c764e11201ccee3e173fd7f099dd0.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/tables/aaef945b6517db4141bc9a984f008535e39c764e11201ccee3e173fd7f099dd0.jpg)

![caf18112c6a5a0eb0b0839aae0f12117c218c692c3b6e959517fac0881bf60db.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/tables/caf18112c6a5a0eb0b0839aae0f12117c218c692c3b6e959517fac0881bf60db.jpg)

![e2c5906f18a727c41aabbf5630e7c2f4012f9e459759e6b320da348c03036ccf.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/tables/e2c5906f18a727c41aabbf5630e7c2f4012f9e459759e6b320da348c03036ccf.jpg)

![ec83e425c1c54a22345418809fa4dd68b1343cb5fb00b9af461841c319f116be.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/tables/ec83e425c1c54a22345418809fa4dd68b1343cb5fb00b9af461841c319f116be.jpg)

![f2eb03e790848d800fd37cf9855374521134adc8fddd050f3119aafe79f04ea9.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/tables/f2eb03e790848d800fd37cf9855374521134adc8fddd050f3119aafe79f04ea9.jpg)

![fe3786c6648c2b9d2ebe0d378f2f93b7bedb0a89efc4122758cb35d728fdd7b7.jpg](../icml_results/3491_Making%20Expert%20Reasoning%20Learnable%20with%20Self-Distillation/tables/fe3786c6648c2b9d2ebe0d378f2f93b7bedb0a89efc4122758cb35d728fdd7b7.jpg)

<a id="paper-3452"></a>

## Supervised Classification Heads as Semantic Prototypes: Unlocking Vision-Language Alignment via Weight Recycling


### Images

![1db5e67eeed3ba79772bd9d2d90c893f23b30b3c1bfa133f1b8ec8da4c7ff1e9.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/images/1db5e67eeed3ba79772bd9d2d90c893f23b30b3c1bfa133f1b8ec8da4c7ff1e9.jpg)

![576d874cd2c9d0164865bcd1c6bbadcd9e7828c46dff61864b8f6bd775fca2b1.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/images/576d874cd2c9d0164865bcd1c6bbadcd9e7828c46dff61864b8f6bd775fca2b1.jpg)

![74f00a179dc22d4aba0968b292bdc9c854ce924c1f87e5bf782472bb84e10e7a.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/images/74f00a179dc22d4aba0968b292bdc9c854ce924c1f87e5bf782472bb84e10e7a.jpg)

![af67485b030c3f417b33be59f3c3b51975670b1c267aead5bde92cd0aff9389b.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/images/af67485b030c3f417b33be59f3c3b51975670b1c267aead5bde92cd0aff9389b.jpg)

![c213669726ed25a72fd8948fa6811f78f9ed4e43f544b353db30fb7c3edc8d54.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/images/c213669726ed25a72fd8948fa6811f78f9ed4e43f544b353db30fb7c3edc8d54.jpg)

![d3e88b9d4f2399d5e8817a1cf74f6d459c6c15d868f690a5bebe28c9230de955.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/images/d3e88b9d4f2399d5e8817a1cf74f6d459c6c15d868f690a5bebe28c9230de955.jpg)

![f17693b299620d1e88e41ac540592c6720d692812d5d37db41de9316daa4ef95.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/images/f17693b299620d1e88e41ac540592c6720d692812d5d37db41de9316daa4ef95.jpg)

![fe85c8761447b6869b5adcd9390f1bb8c7b8c174464f4911dd6cf1188ebc2e3e.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/images/fe85c8761447b6869b5adcd9390f1bb8c7b8c174464f4911dd6cf1188ebc2e3e.jpg)

### Tables

![1aea9aaff42ded13873e8cbc58bcda79bf2f5568fcc542a6b81e391e7c5e396c.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/tables/1aea9aaff42ded13873e8cbc58bcda79bf2f5568fcc542a6b81e391e7c5e396c.jpg)

![1de3ec80cbf120f7ef17d453f205939bf72e8dc6fed000f6f956e51072b89ee8.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/tables/1de3ec80cbf120f7ef17d453f205939bf72e8dc6fed000f6f956e51072b89ee8.jpg)

![20e48003e97baaf33255e5531879f09a2ddea101e490db910ad4a261fc5133dc.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/tables/20e48003e97baaf33255e5531879f09a2ddea101e490db910ad4a261fc5133dc.jpg)

![2fcac936e514b03cabf94fb6b8d18f556b8351833ad903943e481be01dfa2d6c.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/tables/2fcac936e514b03cabf94fb6b8d18f556b8351833ad903943e481be01dfa2d6c.jpg)

![4fee3f2728b785f4263087a3385cc9b813f067c3d2ca93598cb58734f0aa4078.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/tables/4fee3f2728b785f4263087a3385cc9b813f067c3d2ca93598cb58734f0aa4078.jpg)

![89f97f3180cb7a44bbf296f9b42598888f61d5b7e6394533ba5f8045a41a7948.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/tables/89f97f3180cb7a44bbf296f9b42598888f61d5b7e6394533ba5f8045a41a7948.jpg)

![95eff21369ed158fd5e0a3eeb34fb215287ab25e0fea9a28429e61f229e98ade.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/tables/95eff21369ed158fd5e0a3eeb34fb215287ab25e0fea9a28429e61f229e98ade.jpg)

![a05ad61b7c2b2f2171ed70223c49f52228223bf30a7e79dc771686a054813c2c.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/tables/a05ad61b7c2b2f2171ed70223c49f52228223bf30a7e79dc771686a054813c2c.jpg)

![b2ba698b81eb7cd32d685972db0ad1a8fc41f94d7eeea1947dbdbf04847990de.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/tables/b2ba698b81eb7cd32d685972db0ad1a8fc41f94d7eeea1947dbdbf04847990de.jpg)

![bac54f8a9e6e0fdbcff8f1840f2b88b6a43f25ed9ad4dc27ce30872cb420b7ec.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/tables/bac54f8a9e6e0fdbcff8f1840f2b88b6a43f25ed9ad4dc27ce30872cb420b7ec.jpg)

![d0a76ccf73be47fad5ca8c93ff679fb075223b257fd621ad1cf60c952b44014f.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/tables/d0a76ccf73be47fad5ca8c93ff679fb075223b257fd621ad1cf60c952b44014f.jpg)

![d410c6e2cca387dd4b4be1c92c927b26249d4653700adf92470303b625aa018d.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/tables/d410c6e2cca387dd4b4be1c92c927b26249d4653700adf92470303b625aa018d.jpg)

![dacfe3692dff673614fd6b68bbef84f5a3446c2fce7c6cbe8c0c864e8c8e4bf0.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/tables/dacfe3692dff673614fd6b68bbef84f5a3446c2fce7c6cbe8c0c864e8c8e4bf0.jpg)

![df34cc6005b4a8e70aefb3fef0b3e9521a9ad6636048d920aa6a5acffa1ba48e.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/tables/df34cc6005b4a8e70aefb3fef0b3e9521a9ad6636048d920aa6a5acffa1ba48e.jpg)

![e9db99ee3bc72308dc332c3dd90ccc9b71e4b8499b630306df002aa3954de2de.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/tables/e9db99ee3bc72308dc332c3dd90ccc9b71e4b8499b630306df002aa3954de2de.jpg)

![f4678489bf814ed31df236404566fb7d97ff1d99fc1d4a3a3b1a2b0ea16c6d0b.jpg](../icml_results/3492_Supervised%20Classification%20Heads%20as%20Semantic%20Prototypes_%20Unlocking%20Vision-Language%20Alignment%20via%20Weig/tables/f4678489bf814ed31df236404566fb7d97ff1d99fc1d4a3a3b1a2b0ea16c6d0b.jpg)

<a id="paper-3453"></a>

## GHOST: Unmasking Phantom States in Mamba2 via Grouped Hidden-state Output-aware Selection & Truncation


### Images

![1bb02524faf068955c835e32ef0fb1ea73cea4be78f2128e55e4bf40742326c8.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/images/1bb02524faf068955c835e32ef0fb1ea73cea4be78f2128e55e4bf40742326c8.jpg)

![2eb4352fa2ab023fb8cc0347a94dcdd01d4ec6fd0af8e477da5d6432228b3f7e.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/images/2eb4352fa2ab023fb8cc0347a94dcdd01d4ec6fd0af8e477da5d6432228b3f7e.jpg)

![81573f42d602c9d3cd9acd3520dc8de5f11b02dd8f1c1db55b8b7723025322d7.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/images/81573f42d602c9d3cd9acd3520dc8de5f11b02dd8f1c1db55b8b7723025322d7.jpg)

![9e3b99626218db54be893500192db66848c70287b8ffc8864632cea6c619e5f7.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/images/9e3b99626218db54be893500192db66848c70287b8ffc8864632cea6c619e5f7.jpg)

![aa67f1dd786ab7a11583d06d706b4f546ed45d27469a6b109d8830af3c08a835.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/images/aa67f1dd786ab7a11583d06d706b4f546ed45d27469a6b109d8830af3c08a835.jpg)

![b435a6a7fb49387d8913ee06e3675c94d148f61f543700a499c03633aa0c0d96.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/images/b435a6a7fb49387d8913ee06e3675c94d148f61f543700a499c03633aa0c0d96.jpg)

![ba4334892fbe36f5e94c4e8937907de0711b81e5adddc556c909c61fc002d094.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/images/ba4334892fbe36f5e94c4e8937907de0711b81e5adddc556c909c61fc002d094.jpg)

![c5b35ed5d61b12e445aa3ea861aa419b80b34c8baec1a59663ce5cd398146825.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/images/c5b35ed5d61b12e445aa3ea861aa419b80b34c8baec1a59663ce5cd398146825.jpg)

![e37050e3596b20a3d23c19686219eadadc4b34a1453310f0ce75f45d381d7289.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/images/e37050e3596b20a3d23c19686219eadadc4b34a1453310f0ce75f45d381d7289.jpg)

![f24d9a0ebd9cdd3d5c2f826390817660d7e9ead30512e247eb2d567addca4044.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/images/f24d9a0ebd9cdd3d5c2f826390817660d7e9ead30512e247eb2d567addca4044.jpg)

### Tables

![0ed05706cdb7d43867104aef2778e717c6723344219595cd85b4de5294e110d0.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/tables/0ed05706cdb7d43867104aef2778e717c6723344219595cd85b4de5294e110d0.jpg)

![2bb41edef610f8d4e7c6c994437e0e653b53fdae4c8a675100fb7f7d817a596d.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/tables/2bb41edef610f8d4e7c6c994437e0e653b53fdae4c8a675100fb7f7d817a596d.jpg)

![2fbe1fcc613abd19191978ef559159a26caff994e31ab01cec6170838780c81c.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/tables/2fbe1fcc613abd19191978ef559159a26caff994e31ab01cec6170838780c81c.jpg)

![379a9fad7f6294ac9b5a04dbd3e368c93e5690a69710f593c6215a1c97bf9bbf.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/tables/379a9fad7f6294ac9b5a04dbd3e368c93e5690a69710f593c6215a1c97bf9bbf.jpg)

![4793c02f7a39173c555b4fc985e77b3843599849434491caa4a17a172bbfa21a.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/tables/4793c02f7a39173c555b4fc985e77b3843599849434491caa4a17a172bbfa21a.jpg)

![6207b35d6ea105005390e0f78fd847940c73cf463762253e410535724accc364.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/tables/6207b35d6ea105005390e0f78fd847940c73cf463762253e410535724accc364.jpg)

![647bfa97ca3fcc326332142856aee69876d07f4103cca05cd4581e7c2fd7e539.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/tables/647bfa97ca3fcc326332142856aee69876d07f4103cca05cd4581e7c2fd7e539.jpg)

![6ae5162d99d4d7a56690cf89ce0ac8c345b65afae66ae0ad4e8ac062f1faaf22.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/tables/6ae5162d99d4d7a56690cf89ce0ac8c345b65afae66ae0ad4e8ac062f1faaf22.jpg)

![6eb0b37316d5d1f3f104ed461f315c2339947280c13f83286d4decd8284da83c.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/tables/6eb0b37316d5d1f3f104ed461f315c2339947280c13f83286d4decd8284da83c.jpg)

![882b08e3961b91d8e8538b9a79fbabbd8edb8d49199ed433a503ac70aa7d5ce1.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/tables/882b08e3961b91d8e8538b9a79fbabbd8edb8d49199ed433a503ac70aa7d5ce1.jpg)

![95a8db55eeaef116c72d73baa051c9b4ccadcc49a1a7d39470ce98feb17315d5.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/tables/95a8db55eeaef116c72d73baa051c9b4ccadcc49a1a7d39470ce98feb17315d5.jpg)

![a0b7946bc933376125f5f7444e6caf07752730c2e87a7faa8aad5c00664c11a4.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/tables/a0b7946bc933376125f5f7444e6caf07752730c2e87a7faa8aad5c00664c11a4.jpg)

![b37b3f3dc7a98affd625b885745601bc243ccdae3101d4aa2b3be452d4544b0e.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/tables/b37b3f3dc7a98affd625b885745601bc243ccdae3101d4aa2b3be452d4544b0e.jpg)

![e02e7bd3ddd18c928d45590b5a338507629a20d08ec68af029fd4f09c0e52673.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/tables/e02e7bd3ddd18c928d45590b5a338507629a20d08ec68af029fd4f09c0e52673.jpg)

![e11fab336431fbe0ac5ae1a6aebc37aacfbed91cbf0009cfc898148e7fb6c51c.jpg](../icml_results/3493_GHOST_%20Unmasking%20Phantom%20States%20in%20Mamba2%20via%20Grouped%20Hidden-state%20Output-aware%20Selection%20%26%20Truncati/tables/e11fab336431fbe0ac5ae1a6aebc37aacfbed91cbf0009cfc898148e7fb6c51c.jpg)

<a id="paper-3454"></a>

## Group Cognition Learning: Making Everything Better Through Governed Two-Stage Agents Collaboration


### Images

![3e90efa56e917991dd8330121e79cf65041cf0552bf4a57a330d91e79348f0df.jpg](../icml_results/3494_Group%20Cognition%20Learning_%20Making%20Everything%20Better%20Through%20Governed%20Two-Stage%20Agents%20Collaboration/images/3e90efa56e917991dd8330121e79cf65041cf0552bf4a57a330d91e79348f0df.jpg)

![5a80debdf29c31ab8d323ac8c74336e387166e6d0743c6c8d8977de2487f64d8.jpg](../icml_results/3494_Group%20Cognition%20Learning_%20Making%20Everything%20Better%20Through%20Governed%20Two-Stage%20Agents%20Collaboration/images/5a80debdf29c31ab8d323ac8c74336e387166e6d0743c6c8d8977de2487f64d8.jpg)

![5f1888e1b150d349eb28ff0aa490dca097f5265a06ac8bfa63aabd6a009d3590.jpg](../icml_results/3494_Group%20Cognition%20Learning_%20Making%20Everything%20Better%20Through%20Governed%20Two-Stage%20Agents%20Collaboration/images/5f1888e1b150d349eb28ff0aa490dca097f5265a06ac8bfa63aabd6a009d3590.jpg)

![6df829d3e9a26d1e3b3a07eaac62a0f194187cbe7a3b7076cf24f71b9c1daa91.jpg](../icml_results/3494_Group%20Cognition%20Learning_%20Making%20Everything%20Better%20Through%20Governed%20Two-Stage%20Agents%20Collaboration/images/6df829d3e9a26d1e3b3a07eaac62a0f194187cbe7a3b7076cf24f71b9c1daa91.jpg)

![7dcd94d0056fb73b9a6908a8e1e75fd0a279165b0a1b89213cfa2e4a8cb812a7.jpg](../icml_results/3494_Group%20Cognition%20Learning_%20Making%20Everything%20Better%20Through%20Governed%20Two-Stage%20Agents%20Collaboration/images/7dcd94d0056fb73b9a6908a8e1e75fd0a279165b0a1b89213cfa2e4a8cb812a7.jpg)

![c14da6aff51a85a6ace1837475cec4b2c094271e6e2c4138481abcdf5aed17a9.jpg](../icml_results/3494_Group%20Cognition%20Learning_%20Making%20Everything%20Better%20Through%20Governed%20Two-Stage%20Agents%20Collaboration/images/c14da6aff51a85a6ace1837475cec4b2c094271e6e2c4138481abcdf5aed17a9.jpg)

### Tables

![2f95d7338078f511205b518e758eff39ba4f59ed85eb2dfe798f63c8ac15d7c0.jpg](../icml_results/3494_Group%20Cognition%20Learning_%20Making%20Everything%20Better%20Through%20Governed%20Two-Stage%20Agents%20Collaboration/tables/2f95d7338078f511205b518e758eff39ba4f59ed85eb2dfe798f63c8ac15d7c0.jpg)

![43fdc2b01f03469c4523d212f900c8ee0e9685e80d8d94d58de4b2d4e026fe39.jpg](../icml_results/3494_Group%20Cognition%20Learning_%20Making%20Everything%20Better%20Through%20Governed%20Two-Stage%20Agents%20Collaboration/tables/43fdc2b01f03469c4523d212f900c8ee0e9685e80d8d94d58de4b2d4e026fe39.jpg)

![54cfe7d94e125b2a9d7b7843cb4f154219c55d07ce62663246308cd05949c94a.jpg](../icml_results/3494_Group%20Cognition%20Learning_%20Making%20Everything%20Better%20Through%20Governed%20Two-Stage%20Agents%20Collaboration/tables/54cfe7d94e125b2a9d7b7843cb4f154219c55d07ce62663246308cd05949c94a.jpg)

![98b0eb580208b3175035a4cbec911ec9d111754de3eb71275a2c869c2b2e6166.jpg](../icml_results/3494_Group%20Cognition%20Learning_%20Making%20Everything%20Better%20Through%20Governed%20Two-Stage%20Agents%20Collaboration/tables/98b0eb580208b3175035a4cbec911ec9d111754de3eb71275a2c869c2b2e6166.jpg)

![b443f003222ac4ec38bdb18ef673eaf3c1a21538e9b5f4828f6680e62bba335e.jpg](../icml_results/3494_Group%20Cognition%20Learning_%20Making%20Everything%20Better%20Through%20Governed%20Two-Stage%20Agents%20Collaboration/tables/b443f003222ac4ec38bdb18ef673eaf3c1a21538e9b5f4828f6680e62bba335e.jpg)

![e3dbffab4b6c5cfd90fd10c31b4487c1c6ccd488c58ac82059a459caccc6f7f0.jpg](../icml_results/3494_Group%20Cognition%20Learning_%20Making%20Everything%20Better%20Through%20Governed%20Two-Stage%20Agents%20Collaboration/tables/e3dbffab4b6c5cfd90fd10c31b4487c1c6ccd488c58ac82059a459caccc6f7f0.jpg)

![f38ea03efb5394bb7bfa9e8b31d0fe1cbe91a58d010be2b45391b455f9495409.jpg](../icml_results/3494_Group%20Cognition%20Learning_%20Making%20Everything%20Better%20Through%20Governed%20Two-Stage%20Agents%20Collaboration/tables/f38ea03efb5394bb7bfa9e8b31d0fe1cbe91a58d010be2b45391b455f9495409.jpg)

<a id="paper-3455"></a>

## Watermarking LLM Agent Trajectories


### Images

![1189272f4e1a72a39b21951b8062510547e876ea6ee6e2abe8a091043ea3c5d7.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/images/1189272f4e1a72a39b21951b8062510547e876ea6ee6e2abe8a091043ea3c5d7.jpg)

![2373974d603adab1ce99aa1523c737b637fcc61821204a4cc56bc6efa579b7b6.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/images/2373974d603adab1ce99aa1523c737b637fcc61821204a4cc56bc6efa579b7b6.jpg)

![5a81f275b82318772ddb9e0a83e6b26acc03a5082f0db1f6e67c27d16e8318d9.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/images/5a81f275b82318772ddb9e0a83e6b26acc03a5082f0db1f6e67c27d16e8318d9.jpg)

![7ef89f381c5229687072eb5d42f6824f3386e6e890efe0b4cc0d1c9f5ca872e9.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/images/7ef89f381c5229687072eb5d42f6824f3386e6e890efe0b4cc0d1c9f5ca872e9.jpg)

![a090d1f8230b4dcf153ddee5e4752c21a857ccb25538130ed01fb4a6633dec56.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/images/a090d1f8230b4dcf153ddee5e4752c21a857ccb25538130ed01fb4a6633dec56.jpg)

![b2abba41ec2e0aa90cdd5ab3835d2c6b02ccbbde9b00f49835fd738acc71b04a.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/images/b2abba41ec2e0aa90cdd5ab3835d2c6b02ccbbde9b00f49835fd738acc71b04a.jpg)

![b5f7f75d186bc2f11a32854fe7d4d69bb8cb627d5c6713a4b18cfc4add427fd0.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/images/b5f7f75d186bc2f11a32854fe7d4d69bb8cb627d5c6713a4b18cfc4add427fd0.jpg)

![bc542d76908dd444c1ffd58186d977c23041b598827711390395b596cd311baa.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/images/bc542d76908dd444c1ffd58186d977c23041b598827711390395b596cd311baa.jpg)

![d8af3ae8111ae0e09cfd30550f261f96376f30cafa7ddf2713bbf92544fc5bac.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/images/d8af3ae8111ae0e09cfd30550f261f96376f30cafa7ddf2713bbf92544fc5bac.jpg)

![e2ded0e3019eca568d60e6044dbb218fe47aea142f9f7c4ef7d812fa763f9fef.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/images/e2ded0e3019eca568d60e6044dbb218fe47aea142f9f7c4ef7d812fa763f9fef.jpg)

### Tables

![0dc93b374348e9cf3ea7c1517e804212755d6826ae49f24627d6ab25b24fde74.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/tables/0dc93b374348e9cf3ea7c1517e804212755d6826ae49f24627d6ab25b24fde74.jpg)

![1f667729e3cf3da8ef88a23a66430548c0ee651b9512b742169b5d7e8043f4af.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/tables/1f667729e3cf3da8ef88a23a66430548c0ee651b9512b742169b5d7e8043f4af.jpg)

![3915366441ddb811f31f1acc7f5c53051fb4be30a521e8e32b9c510bbc6a4f66.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/tables/3915366441ddb811f31f1acc7f5c53051fb4be30a521e8e32b9c510bbc6a4f66.jpg)

![432a030aacb6ee551346f5ca687b855adaeaaca883b229317053e1c813742be2.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/tables/432a030aacb6ee551346f5ca687b855adaeaaca883b229317053e1c813742be2.jpg)

![6604f3771275af3ac0cb81716bd3faafb11cf45bebdbfe16e08531786c1a78d9.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/tables/6604f3771275af3ac0cb81716bd3faafb11cf45bebdbfe16e08531786c1a78d9.jpg)

![9ee23cd4f8b8768039848767859faf94f79c3a1639066b83a9e1a17a454366a1.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/tables/9ee23cd4f8b8768039848767859faf94f79c3a1639066b83a9e1a17a454366a1.jpg)

![bb5eb408437d08857c51abc9d29e031beba78bf14b5a534c445368ead19983dd.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/tables/bb5eb408437d08857c51abc9d29e031beba78bf14b5a534c445368ead19983dd.jpg)

![ea9618226fea73a4862ed460ce99a0b514d6c869112425c3fd9b3fb9c11ec528.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/tables/ea9618226fea73a4862ed460ce99a0b514d6c869112425c3fd9b3fb9c11ec528.jpg)

![fd3ec2cb077e93649812b731d1dae680a1a73d85cc2f7560bf0cb8d11d7605af.jpg](../icml_results/3495_Watermarking%20LLM%20Agent%20Trajectories/tables/fd3ec2cb077e93649812b731d1dae680a1a73d85cc2f7560bf0cb8d11d7605af.jpg)

<a id="paper-3456"></a>

## Fair-FedMOE: Group-Fair One-Shot Federated Learning via Prototype-Guided Experts for Medical Imaging Analysis


### Images

![158ecc70ef4e1140067a497f2459782a05b85e494ceac900045c90bd81f0b601.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/images/158ecc70ef4e1140067a497f2459782a05b85e494ceac900045c90bd81f0b601.jpg)

![50c385fc7b62c767d946d19c2065cca83fc2cf81689d204609993b1f5ea516d8.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/images/50c385fc7b62c767d946d19c2065cca83fc2cf81689d204609993b1f5ea516d8.jpg)

![6800413b4f9640cec46b2b8c36869f65f1a5d885de0ee2f13a8bb0c63898ec42.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/images/6800413b4f9640cec46b2b8c36869f65f1a5d885de0ee2f13a8bb0c63898ec42.jpg)

![73e05c6c48425bd0c5b4579a26e8958c8438993ca7c9f7fd3f7b3eb0fe6c27a5.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/images/73e05c6c48425bd0c5b4579a26e8958c8438993ca7c9f7fd3f7b3eb0fe6c27a5.jpg)

![8a1ad0be29db05361904cc748ed55493ee97ecfab01a0e369923ae8edd8c3337.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/images/8a1ad0be29db05361904cc748ed55493ee97ecfab01a0e369923ae8edd8c3337.jpg)

![aaab791e5035c26c1bd04c3403b128540df66a16046b4bba225a76a7df4b1fa0.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/images/aaab791e5035c26c1bd04c3403b128540df66a16046b4bba225a76a7df4b1fa0.jpg)

![bcccfa9e97998dfa667f57983ac8c0a8b56ff905e02be1a20a8e999cf6e92071.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/images/bcccfa9e97998dfa667f57983ac8c0a8b56ff905e02be1a20a8e999cf6e92071.jpg)

![d2217c400f7bffe8f01d1b8bb2eb51bebe627d7aab88ded0ae8579a812216a82.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/images/d2217c400f7bffe8f01d1b8bb2eb51bebe627d7aab88ded0ae8579a812216a82.jpg)

![e32fab468541e9c5742a99a374a806daa90ccb56b79a17e3e1ebb98250823ab5.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/images/e32fab468541e9c5742a99a374a806daa90ccb56b79a17e3e1ebb98250823ab5.jpg)

![ef76b351cf6b43d5f4248a175fc01986bfbdfa373c00e6aa26979e4974631348.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/images/ef76b351cf6b43d5f4248a175fc01986bfbdfa373c00e6aa26979e4974631348.jpg)

![f2773ec01d2e06e46a69d5bcb401d141a9443f2f895c85dc8c701dcef3ea759a.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/images/f2773ec01d2e06e46a69d5bcb401d141a9443f2f895c85dc8c701dcef3ea759a.jpg)

### Tables

![00273c45fb64c14bf7e246b84c46ed983f58d6a14ec81dbfe7c04173f1f95565.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/tables/00273c45fb64c14bf7e246b84c46ed983f58d6a14ec81dbfe7c04173f1f95565.jpg)

![072d37aa4b46dda54d5d7b7798c470ce43e5ac915c8cc911bbb70b2ffc67559a.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/tables/072d37aa4b46dda54d5d7b7798c470ce43e5ac915c8cc911bbb70b2ffc67559a.jpg)

![1717ace7e664b5bdee53ba16dc54e71eea90b4e074ed67db143a1aee083e7c73.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/tables/1717ace7e664b5bdee53ba16dc54e71eea90b4e074ed67db143a1aee083e7c73.jpg)

![37d62943b5769f00ae06c6227903b0c84b2033a78af56ffc04f02417f5233d71.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/tables/37d62943b5769f00ae06c6227903b0c84b2033a78af56ffc04f02417f5233d71.jpg)

![3dda088cb8561a767298014c403c39d712a7bb5f4ecb62a603d329c9426a4c82.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/tables/3dda088cb8561a767298014c403c39d712a7bb5f4ecb62a603d329c9426a4c82.jpg)

![4dd655ae7fe69212c19cecd86551b209b19087739047299b9c89846f83be59da.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/tables/4dd655ae7fe69212c19cecd86551b209b19087739047299b9c89846f83be59da.jpg)

![575f6583b7fd640e24c92f3e78d648970b22169ce279256b81147693bc129c48.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/tables/575f6583b7fd640e24c92f3e78d648970b22169ce279256b81147693bc129c48.jpg)

![83f66813c8cdf1b1456c3187f1dc71568532bf206c8b82ded2b96950c5f43fb8.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/tables/83f66813c8cdf1b1456c3187f1dc71568532bf206c8b82ded2b96950c5f43fb8.jpg)

![8c151f5d544e1b0e5d98e50008446679cdff38d93eef22bff6473c9bcb71171d.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/tables/8c151f5d544e1b0e5d98e50008446679cdff38d93eef22bff6473c9bcb71171d.jpg)

![962d7da2f5ff26d02a875fb5a1d5b0c83d5639b886a3ef2a685e1af74ab1c4ec.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/tables/962d7da2f5ff26d02a875fb5a1d5b0c83d5639b886a3ef2a685e1af74ab1c4ec.jpg)

![b8aa3b070a6c450304e775d9b66ffead942f284a49a1b7fc81a145e7a812202a.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/tables/b8aa3b070a6c450304e775d9b66ffead942f284a49a1b7fc81a145e7a812202a.jpg)

![bb28db4e16d80c58215753a84ea25b1a3ecf0880caba2c5c76d695a2d80fe23e.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/tables/bb28db4e16d80c58215753a84ea25b1a3ecf0880caba2c5c76d695a2d80fe23e.jpg)

![bd842c03edf75e80f32632fb1a41a569c1c35dc2b5a7e6940730ea0d0ce7c1a6.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/tables/bd842c03edf75e80f32632fb1a41a569c1c35dc2b5a7e6940730ea0d0ce7c1a6.jpg)

![cc04ca0424b89fdd58a35eabea869237a3b3dc837d2f6b42d351f91e2e0fba8d.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/tables/cc04ca0424b89fdd58a35eabea869237a3b3dc837d2f6b42d351f91e2e0fba8d.jpg)

![d647aa8f1d1c88692768c0edf50315de58d883cb2b03f9e6a1bcb7999b68922f.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/tables/d647aa8f1d1c88692768c0edf50315de58d883cb2b03f9e6a1bcb7999b68922f.jpg)

![eca9204dfbda21ba3add4b5790b9842402d42e547255b3083f26690d7e28b609.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/tables/eca9204dfbda21ba3add4b5790b9842402d42e547255b3083f26690d7e28b609.jpg)

![f764e65303f58319f62b4d3f0ab00e33d5fc0d521a05728ff61b0251b64ecd71.jpg](../icml_results/3496_Fair-FedMOE_%20Group-Fair%20One-Shot%20Federated%20Learning%20via%20Prototype-Guided%20Experts%20for%20Medical%20Imaging/tables/f764e65303f58319f62b4d3f0ab00e33d5fc0d521a05728ff61b0251b64ecd71.jpg)

<a id="paper-3457"></a>

## DV-World: Benchmarking Data Visualization Agents in Real-World Scenarios


### Images

![00c8fa4d0bbfffee757b1d3c1e349bfb7573669c6f9ea33fa134ac848abff13d.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/00c8fa4d0bbfffee757b1d3c1e349bfb7573669c6f9ea33fa134ac848abff13d.jpg)

![0c8dde56642633cf98c162fa6a61bd9245671b88d61c8e0b2398e736c60ae7fe.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/0c8dde56642633cf98c162fa6a61bd9245671b88d61c8e0b2398e736c60ae7fe.jpg)

![0f52d98fee864ca50c166ad7e663b29ca174febc9d5ad6b6298332a23bb19aa4.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/0f52d98fee864ca50c166ad7e663b29ca174febc9d5ad6b6298332a23bb19aa4.jpg)

![104a8b5c32b79373f66ad494f356facf360eb92fc144e6de9748f972fe864db2.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/104a8b5c32b79373f66ad494f356facf360eb92fc144e6de9748f972fe864db2.jpg)

![10c6c949bcbe9f77c71b70ed2fa4f29eb0e4b8593d9726bc525ee55b76f094e9.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/10c6c949bcbe9f77c71b70ed2fa4f29eb0e4b8593d9726bc525ee55b76f094e9.jpg)

![10fbac03a3e813b1768dfa83499a9b3bf3a4a5fc4c697b235f778541b0670b76.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/10fbac03a3e813b1768dfa83499a9b3bf3a4a5fc4c697b235f778541b0670b76.jpg)

![11caa0b0ab9b5beea886422b388803d394a0a5a1a3922d4b4684fbacb9f54596.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/11caa0b0ab9b5beea886422b388803d394a0a5a1a3922d4b4684fbacb9f54596.jpg)

![15b7b9b2c1b16d7e84217325a67313782b690a5861d31e29144578ab1a31c0fc.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/15b7b9b2c1b16d7e84217325a67313782b690a5861d31e29144578ab1a31c0fc.jpg)

![1624f2dc2c4f27855340d62f198b12697479f372db835844bc44f6c4d082d6a6.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/1624f2dc2c4f27855340d62f198b12697479f372db835844bc44f6c4d082d6a6.jpg)

![18e3ccba03026c2f9fc5109b00adaef19fdf2005731a3bcee632d4c9c9c549a0.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/18e3ccba03026c2f9fc5109b00adaef19fdf2005731a3bcee632d4c9c9c549a0.jpg)

![19dbbbe482080253014d754c7db703ea2bdcde3ca5a73b9bc2d308ccd5de4ac8.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/19dbbbe482080253014d754c7db703ea2bdcde3ca5a73b9bc2d308ccd5de4ac8.jpg)

![1b03a23fe2c3cbeeeb393b5b60efce9a2ee65b8aa8a703db0589c0c7aadfa527.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/1b03a23fe2c3cbeeeb393b5b60efce9a2ee65b8aa8a703db0589c0c7aadfa527.jpg)

![1e00febdbfc743e287c5f4b4d480d447a894c2317768909d987be64d8c6cb86f.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/1e00febdbfc743e287c5f4b4d480d447a894c2317768909d987be64d8c6cb86f.jpg)

![1ff0ead2bba5d9466093fabdd160ddc876231bc60e724cb3223606fc576cb37f.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/1ff0ead2bba5d9466093fabdd160ddc876231bc60e724cb3223606fc576cb37f.jpg)

![222947644bdcd46323106f5ddbd0eafc9e23d6c73bee2568deb033b9f59ca304.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/222947644bdcd46323106f5ddbd0eafc9e23d6c73bee2568deb033b9f59ca304.jpg)

![2b689ac8268021a3a203cd70da149e182a8979669d2e64a46ee4e0f36ac32357.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/2b689ac8268021a3a203cd70da149e182a8979669d2e64a46ee4e0f36ac32357.jpg)

![2c2461f5d2b4886f057cc6afb44c2a232c1fc7631a6d6aa1f56625a1606c7af9.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/2c2461f5d2b4886f057cc6afb44c2a232c1fc7631a6d6aa1f56625a1606c7af9.jpg)

![2f41770e061446a2529d70bf92bab7b6ebf94c9a85b0f07f35b6edfa4abe7b0d.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/2f41770e061446a2529d70bf92bab7b6ebf94c9a85b0f07f35b6edfa4abe7b0d.jpg)

![345ed4ee1591c71c8bc35e9d6b1d1ef9e5e970110f220830d210849516d73c26.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/345ed4ee1591c71c8bc35e9d6b1d1ef9e5e970110f220830d210849516d73c26.jpg)

![35d48bfdb0296dcb2dd8f0af522f4f8510842cb742e7de8f04e166b7d29b6fb2.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/35d48bfdb0296dcb2dd8f0af522f4f8510842cb742e7de8f04e166b7d29b6fb2.jpg)

![3960f22c5a64f82fd312c5f0ffe94bf3ccc6b5bb26761ca87d94291911057d81.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/3960f22c5a64f82fd312c5f0ffe94bf3ccc6b5bb26761ca87d94291911057d81.jpg)

![3ced2b38e516dbbb4008c1dbf9ac7dd3d3f959d119a577a6a40603a30b366079.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/3ced2b38e516dbbb4008c1dbf9ac7dd3d3f959d119a577a6a40603a30b366079.jpg)

![420da9c9e617a316d9230030be544a746f134ffb0d5303f96dc516026c47dc37.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/420da9c9e617a316d9230030be544a746f134ffb0d5303f96dc516026c47dc37.jpg)

![44e12a95d2fd70c33ff4d30316da7ed0419ca41b2c640093ad7c2fd4e012ec25.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/44e12a95d2fd70c33ff4d30316da7ed0419ca41b2c640093ad7c2fd4e012ec25.jpg)

![46195cc8f6ca6d0103ba67a63e99a5954377d5c95b65ab9d5e8b563b5c2559e0.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/46195cc8f6ca6d0103ba67a63e99a5954377d5c95b65ab9d5e8b563b5c2559e0.jpg)

![4efc1e67435e335e06839850403cf558958a1cbcfef0efd11bce6164de683cbd.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/4efc1e67435e335e06839850403cf558958a1cbcfef0efd11bce6164de683cbd.jpg)

![568883eecd0175dbe71244294c355b55a43b735008b7e3a41ed30c525d1e9efb.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/568883eecd0175dbe71244294c355b55a43b735008b7e3a41ed30c525d1e9efb.jpg)

![58b385f7b94f7fe6c96b5ae08372c5ce312508b6959c941ca721cb808a3d9a8d.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/58b385f7b94f7fe6c96b5ae08372c5ce312508b6959c941ca721cb808a3d9a8d.jpg)

![5a0aa419ee2a2e7ef8efb8af1e2132855caec5e2088e0fc6eb292d0d441d8205.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/5a0aa419ee2a2e7ef8efb8af1e2132855caec5e2088e0fc6eb292d0d441d8205.jpg)

![66adf84d6f5f104507f11128fbca6f5349c2970379010cd427478d6279a19e6a.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/66adf84d6f5f104507f11128fbca6f5349c2970379010cd427478d6279a19e6a.jpg)

![66f5bf7348c1ec13c3fee3c48c3a112384c71af9fac26d5f9d19e5855fac0179.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/66f5bf7348c1ec13c3fee3c48c3a112384c71af9fac26d5f9d19e5855fac0179.jpg)

![6c1ae4b8787bebc4dcbef49ef1372135fe1fa892a3f6e9fc9dad825e8bbb9178.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/6c1ae4b8787bebc4dcbef49ef1372135fe1fa892a3f6e9fc9dad825e8bbb9178.jpg)

![71d630f103d958798ec870823f4c33b0823b0002d6ce5b73066db97601cd03e0.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/71d630f103d958798ec870823f4c33b0823b0002d6ce5b73066db97601cd03e0.jpg)

![75566c1db765e71fc1d48b0b27f5e581b34bae4a15b13f0e1b90fef77a8f830b.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/75566c1db765e71fc1d48b0b27f5e581b34bae4a15b13f0e1b90fef77a8f830b.jpg)

![76a68b754b1d4a1755a759c4a44d979a0a488d4cf66c82becdec23c6c7df40c0.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/76a68b754b1d4a1755a759c4a44d979a0a488d4cf66c82becdec23c6c7df40c0.jpg)

![7d838a9285bb20a003731513dac2ff0ee9f301556c238bce9bdb19ac4d4ccaec.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/7d838a9285bb20a003731513dac2ff0ee9f301556c238bce9bdb19ac4d4ccaec.jpg)

![883fe1cf58d0dbcbaea8fdd056c034a6ec428c7ce5847314f9dd6211b47c5e37.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/883fe1cf58d0dbcbaea8fdd056c034a6ec428c7ce5847314f9dd6211b47c5e37.jpg)

![8af3af40ff85808f7a7cfa00e9013c7731297477e5f20adaf551c8909817022f.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/8af3af40ff85808f7a7cfa00e9013c7731297477e5f20adaf551c8909817022f.jpg)

![9073a4bd47e444995b5a9bdf6a2d3c5020585467b8fd05a3127613421ccb8683.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/9073a4bd47e444995b5a9bdf6a2d3c5020585467b8fd05a3127613421ccb8683.jpg)

![90a6489ef1ee94dbc662438618efde8e48719869b5ba0dc6349b1ced17eace0d.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/90a6489ef1ee94dbc662438618efde8e48719869b5ba0dc6349b1ced17eace0d.jpg)

![91116a2b8f67a18936a6de930c441fd3dae9f93ccc20cc15b62b347197c023b7.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/91116a2b8f67a18936a6de930c441fd3dae9f93ccc20cc15b62b347197c023b7.jpg)

![9621d36dc49556bd73c965cedc42fc11ef44f10e0789daa46b14fc8c01157b62.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/9621d36dc49556bd73c965cedc42fc11ef44f10e0789daa46b14fc8c01157b62.jpg)

![9a6c335a8c78b5b814f33de03b22a983e0f13684b338542900fdc062529bd85e.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/9a6c335a8c78b5b814f33de03b22a983e0f13684b338542900fdc062529bd85e.jpg)

![9b3a2a5428884f32d66d5ecdfe4a41d32d3412add8d0d95087339b8da52b21f2.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/9b3a2a5428884f32d66d5ecdfe4a41d32d3412add8d0d95087339b8da52b21f2.jpg)

![9ef74e0834cfa7ea2ba6008b1d05f537c7ab38467a2e8d6d133c8868a626144f.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/9ef74e0834cfa7ea2ba6008b1d05f537c7ab38467a2e8d6d133c8868a626144f.jpg)

![a2366b19ab53eae0a5476f1e07d0db6782acf8b64de8c3c78c08a71dea7031bb.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/a2366b19ab53eae0a5476f1e07d0db6782acf8b64de8c3c78c08a71dea7031bb.jpg)

![a9c3c238642060d76fade50fda6c26f74c71cca7680eaa84c22b7580af96a5be.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/a9c3c238642060d76fade50fda6c26f74c71cca7680eaa84c22b7580af96a5be.jpg)

![ab05d33be5411a207f6729960f281f5c0f09504a10237a04a7b4537592944d6f.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/ab05d33be5411a207f6729960f281f5c0f09504a10237a04a7b4537592944d6f.jpg)

![b004af4d39bcb362d4a2253c250163ade8f8fd4d3706c5d6e04e27a4cfa64ee6.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/b004af4d39bcb362d4a2253c250163ade8f8fd4d3706c5d6e04e27a4cfa64ee6.jpg)

![b4360a796618a05d428d32fb8638df178c599b563d3bf9a4be355ea933226a4f.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/b4360a796618a05d428d32fb8638df178c599b563d3bf9a4be355ea933226a4f.jpg)

![b4883fc5e3100e25aa234521a2d3ad75ed7ffccc85eef9a7d3e02c074f0f9288.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/b4883fc5e3100e25aa234521a2d3ad75ed7ffccc85eef9a7d3e02c074f0f9288.jpg)

![b707fa6406757e7134b15fa5fd0de8c08b18808081e85f5dd6114e1d7bf77f91.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/b707fa6406757e7134b15fa5fd0de8c08b18808081e85f5dd6114e1d7bf77f91.jpg)

![b9fbed07271ad3bbaeaff32dcd106b22ace2b5198629f7a41ad9660d80a58f22.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/b9fbed07271ad3bbaeaff32dcd106b22ace2b5198629f7a41ad9660d80a58f22.jpg)

![bd8ee2985b23ee03202f9b3e4a3173259940dc898d3cf9578cc7ed296f36bae7.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/bd8ee2985b23ee03202f9b3e4a3173259940dc898d3cf9578cc7ed296f36bae7.jpg)

![be24aa574d042b24b96293c6e9047e59e0a6155487e134c08e0855567e200695.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/be24aa574d042b24b96293c6e9047e59e0a6155487e134c08e0855567e200695.jpg)

![bf97ff846e48d1adfa7c5b2d4df80656b7db5e7eb0e6aec762fab0aa42b0a437.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/bf97ff846e48d1adfa7c5b2d4df80656b7db5e7eb0e6aec762fab0aa42b0a437.jpg)

![c8e2d85930ed2c761ba8b6aa8e147792da5e23f753166009260ce4d2e6cb924c.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/c8e2d85930ed2c761ba8b6aa8e147792da5e23f753166009260ce4d2e6cb924c.jpg)

![d0691aad4d25d29cdd650e771b61efcd2074e6df2544fdcbb94a44d68d254e95.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/d0691aad4d25d29cdd650e771b61efcd2074e6df2544fdcbb94a44d68d254e95.jpg)

![d5bc24cadef4fbb9fcbeca595f15807ee2a4648c2a60bb2e95edb57569be12e8.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/d5bc24cadef4fbb9fcbeca595f15807ee2a4648c2a60bb2e95edb57569be12e8.jpg)

![d8b6670944461071d0f291a45ed5833cca49ac19e91e318ba030503529773256.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/d8b6670944461071d0f291a45ed5833cca49ac19e91e318ba030503529773256.jpg)

![d969e8a680b24888e2510992031be739817a26140060ebce46df1222c6b1cfb7.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/d969e8a680b24888e2510992031be739817a26140060ebce46df1222c6b1cfb7.jpg)

![daaaf2aa0c41b20216017a1f464de0f2f0ff20080cf7f54024492a2415cbecc9.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/daaaf2aa0c41b20216017a1f464de0f2f0ff20080cf7f54024492a2415cbecc9.jpg)

![df374098cbd14dd21bf77d5e23d9494eaeeeae09ddd0e7ed641a7e6382aa3dd8.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/df374098cbd14dd21bf77d5e23d9494eaeeeae09ddd0e7ed641a7e6382aa3dd8.jpg)

![e31e18a69619b4d1bc25b6ce2315a062212b40266e8c1f2953b40fa43450833e.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/e31e18a69619b4d1bc25b6ce2315a062212b40266e8c1f2953b40fa43450833e.jpg)

![e60b813861344e1bbffaa39acae4355c1eaaded162238796849d66e14efc145a.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/e60b813861344e1bbffaa39acae4355c1eaaded162238796849d66e14efc145a.jpg)

![ee5cf3d632229c1688736bf2bfc7f76f19b024a2bf9d0c0aabe774ce97bed986.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/ee5cf3d632229c1688736bf2bfc7f76f19b024a2bf9d0c0aabe774ce97bed986.jpg)

![ef910c00a7c01d022a4bfb113cb7c17bb84c21a38f63b7ffc38ab22cfbbb8f0c.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/ef910c00a7c01d022a4bfb113cb7c17bb84c21a38f63b7ffc38ab22cfbbb8f0c.jpg)

![f5115a0eaa61e345fce6d92902410833b345f249ee8a0c14dc4fd9a483470a8d.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/f5115a0eaa61e345fce6d92902410833b345f249ee8a0c14dc4fd9a483470a8d.jpg)

![f5fab379433a94d7c5ecb323f70d0ed2ff773f8989ad91c3015755e5051b324e.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/f5fab379433a94d7c5ecb323f70d0ed2ff773f8989ad91c3015755e5051b324e.jpg)

![f889b4588be06fd533654c8b5e05c0aa90357e6c4edfbab9f36686cd8715a6d5.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/f889b4588be06fd533654c8b5e05c0aa90357e6c4edfbab9f36686cd8715a6d5.jpg)

![f991b797672281f192ba596b97ec305c8b869ac8b5784e43ad93ac2067673130.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/images/f991b797672281f192ba596b97ec305c8b869ac8b5784e43ad93ac2067673130.jpg)

### Tables

![05cdd99b8b8ed1941fef4d2c76cca3f701bb8d6a14b7bb0a2bc046ab97488153.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/05cdd99b8b8ed1941fef4d2c76cca3f701bb8d6a14b7bb0a2bc046ab97488153.jpg)

![08981d03ec04d89e6def8aff72839fcdeacb85a36f0565a91b4c5f32241d2159.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/08981d03ec04d89e6def8aff72839fcdeacb85a36f0565a91b4c5f32241d2159.jpg)

![0a04257ad0c9b19b83570df4c4c5264420ca63eb40ff57ec4fd345333986eee6.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/0a04257ad0c9b19b83570df4c4c5264420ca63eb40ff57ec4fd345333986eee6.jpg)

![181c5b4d895f70058b274ee45a70dde338b7d34ce59ef3bc02c2f0e4e5defa70.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/181c5b4d895f70058b274ee45a70dde338b7d34ce59ef3bc02c2f0e4e5defa70.jpg)

![1cfff056183d0572b212c596a9c18b37fd1f901ff89de75d69d6c2fb31f0d517.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/1cfff056183d0572b212c596a9c18b37fd1f901ff89de75d69d6c2fb31f0d517.jpg)

![316fa35fe776399f729e10f63c0dcec94990720a7e8b4134c138454d48efe982.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/316fa35fe776399f729e10f63c0dcec94990720a7e8b4134c138454d48efe982.jpg)

![3e5b55fff613edae9104b6249e6f56e6b23145356c69d29ca60268b61d2b588a.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/3e5b55fff613edae9104b6249e6f56e6b23145356c69d29ca60268b61d2b588a.jpg)

![442e306d15f85074a3fe91648224f826082c8bf4f19535a8dedd5d51ea435a3b.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/442e306d15f85074a3fe91648224f826082c8bf4f19535a8dedd5d51ea435a3b.jpg)

![5735f0a8aa5816769ce41445decfc86b5ac73711fa7529bb0426428ec48d340c.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/5735f0a8aa5816769ce41445decfc86b5ac73711fa7529bb0426428ec48d340c.jpg)

![62515f6a0cdd24042c904be7b35ad4985b78a32737e056dd6deed5181879f9b0.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/62515f6a0cdd24042c904be7b35ad4985b78a32737e056dd6deed5181879f9b0.jpg)

![6bf48ccafb04e82b64657e71b63341f897c8a8bc73a646a5e053e3158f984e09.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/6bf48ccafb04e82b64657e71b63341f897c8a8bc73a646a5e053e3158f984e09.jpg)

![6e4ea787b8811c5e224ebf8e334344d5a38546cd4acc27f72630dda75d1287c1.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/6e4ea787b8811c5e224ebf8e334344d5a38546cd4acc27f72630dda75d1287c1.jpg)

![74d190bc7f681e9cfbb85d745b00877e1c08d83279b30b5492c0f099d4aac41b.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/74d190bc7f681e9cfbb85d745b00877e1c08d83279b30b5492c0f099d4aac41b.jpg)

![7c2076bbb2a076065cdb9dfc3451bdcf26cc09b1e230886cad6c1ed9626a95c1.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/7c2076bbb2a076065cdb9dfc3451bdcf26cc09b1e230886cad6c1ed9626a95c1.jpg)

![8a48a21db9f1ac31406bd1853c53904904cab626942c170ad7815d0cc4cb6a22.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/8a48a21db9f1ac31406bd1853c53904904cab626942c170ad7815d0cc4cb6a22.jpg)

![8ca0e076171fb329b8869d3300cbb0459544ed671d723cabd7724cd7331b85eb.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/8ca0e076171fb329b8869d3300cbb0459544ed671d723cabd7724cd7331b85eb.jpg)

![948da832bc271be95049bfeb5736ad7936fe1af202d8e938cf329fa7613b3316.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/948da832bc271be95049bfeb5736ad7936fe1af202d8e938cf329fa7613b3316.jpg)

![97702b0da754befb9e6a5ecd5daa26877fb5edd2835e0837f08ad4d5769ec71f.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/97702b0da754befb9e6a5ecd5daa26877fb5edd2835e0837f08ad4d5769ec71f.jpg)

![9d70e8d7c2a66473ec58f17256e987356800aca095d99caee85d70ee17b7b71c.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/9d70e8d7c2a66473ec58f17256e987356800aca095d99caee85d70ee17b7b71c.jpg)

![a7faef4452c3d7f9861713dbb39880e14ffb00a2aeecdf2fdf6fc6320f8c9808.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/a7faef4452c3d7f9861713dbb39880e14ffb00a2aeecdf2fdf6fc6320f8c9808.jpg)

![a84be1d7b0af6d7fd76ca1d45cdf11fc96daba8da8ddb41a90db0361533c05b6.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/a84be1d7b0af6d7fd76ca1d45cdf11fc96daba8da8ddb41a90db0361533c05b6.jpg)

![ad2604ecedf9a9ba1006dab49ceeea790649d42d6bb6028fbc2211dbddbbcc01.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/ad2604ecedf9a9ba1006dab49ceeea790649d42d6bb6028fbc2211dbddbbcc01.jpg)

![ae4f46c210b86538eded5a97c8d85fe9b76990f19c7dc626b900757dcb78f119.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/ae4f46c210b86538eded5a97c8d85fe9b76990f19c7dc626b900757dcb78f119.jpg)

![afee8cbd5f7b3f8d179c090e073b92fb1c6371527a03239be70b0f56c28a14d3.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/afee8cbd5f7b3f8d179c090e073b92fb1c6371527a03239be70b0f56c28a14d3.jpg)

![bec020409093fbf3805686756efab79f07e6dd3e8ea46eddb1bd8125b0142503.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/bec020409093fbf3805686756efab79f07e6dd3e8ea46eddb1bd8125b0142503.jpg)

![bfef6618b9cf9fb01b4706f007cac852274ef87e09c9e390262697192feffb8b.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/bfef6618b9cf9fb01b4706f007cac852274ef87e09c9e390262697192feffb8b.jpg)

![d770af78e5fa469196702543e222662be442545da05b4ece77b148d42d95cb78.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/d770af78e5fa469196702543e222662be442545da05b4ece77b148d42d95cb78.jpg)

![d7f7b0d2cc7f05ef22f4b608ba7371550bb4be3885ea2c3bafb0f013d43dbe8a.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/d7f7b0d2cc7f05ef22f4b608ba7371550bb4be3885ea2c3bafb0f013d43dbe8a.jpg)

![d943296aca2540cda5011f7dddce49e9af7a988beb8e5bd8fd79559c0d4769ac.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/d943296aca2540cda5011f7dddce49e9af7a988beb8e5bd8fd79559c0d4769ac.jpg)

![e04f05aa5e15e7a0157dd97c6dd725c4b91ace6efae39e607c3abc35d676e544.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/e04f05aa5e15e7a0157dd97c6dd725c4b91ace6efae39e607c3abc35d676e544.jpg)

![e6d74acf1aab9c06b00f086ec193f0651fbaf773a6f8001027d25dda9ed13cbb.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/e6d74acf1aab9c06b00f086ec193f0651fbaf773a6f8001027d25dda9ed13cbb.jpg)

![ecd9cc38656c79e92e75d2c630d9be9cb0a02e47daf658c3ea53c86f81496121.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/ecd9cc38656c79e92e75d2c630d9be9cb0a02e47daf658c3ea53c86f81496121.jpg)

![f7c1764a6b9b5a3ce9dbdc06508316394f07d82a3f118fe4d1edd6f92b0214a0.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/f7c1764a6b9b5a3ce9dbdc06508316394f07d82a3f118fe4d1edd6f92b0214a0.jpg)

![fb714186468ee0800afc913eb108e639e87805ff9944b5fb1f919af5f9ff5a81.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/fb714186468ee0800afc913eb108e639e87805ff9944b5fb1f919af5f9ff5a81.jpg)

![fcb3b90534dad057872fa2e1c91cb84018008b2f84dd0698dca1a3fdc4ad27a1.jpg](../icml_results/3497_DV-World_%20Benchmarking%20Data%20Visualization%20Agents%20in%20Real-World%20Scenarios/tables/fcb3b90534dad057872fa2e1c91cb84018008b2f84dd0698dca1a3fdc4ad27a1.jpg)

<a id="paper-3458"></a>

## Open-o3-Video: Grounded Video Reasoning with Explicit Spatio-Temporal Evidence


### Images

![1289160e50c6d8f52ec2c918cee75b1ed54e82614cc0e5121e10d917727afdaa.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/images/1289160e50c6d8f52ec2c918cee75b1ed54e82614cc0e5121e10d917727afdaa.jpg)

![12cee4b18adba4e0daeed0d60418b3e222ff7c101cef47724fce393ce6fc61f1.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/images/12cee4b18adba4e0daeed0d60418b3e222ff7c101cef47724fce393ce6fc61f1.jpg)

![1b61d20ead7d6d5a6ea5953ac1fbe0b4761f525737a54ab9c4f00adfc2909b12.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/images/1b61d20ead7d6d5a6ea5953ac1fbe0b4761f525737a54ab9c4f00adfc2909b12.jpg)

![26c83716f16aa5fd7131c770fdefa73b19c06b0779de15ccacf1a17ce7f1c42c.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/images/26c83716f16aa5fd7131c770fdefa73b19c06b0779de15ccacf1a17ce7f1c42c.jpg)

![2dcef178184c83d24e6f044259ea8c0d5e80db07e9eb9137dc684338958ad58f.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/images/2dcef178184c83d24e6f044259ea8c0d5e80db07e9eb9137dc684338958ad58f.jpg)

![5bf31a0200f9ccae9e939f96777c1512b2a6e76c23ef289451ebaf4afadef63b.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/images/5bf31a0200f9ccae9e939f96777c1512b2a6e76c23ef289451ebaf4afadef63b.jpg)

![66c199d21004024505bc95e4eda839d260bf66f8f5536fc3e0c0c17f26a3c9e0.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/images/66c199d21004024505bc95e4eda839d260bf66f8f5536fc3e0c0c17f26a3c9e0.jpg)

![6e30497ca526db03a275bff0b7ffd10d6c512690a4c1b3689b4660f2da65e210.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/images/6e30497ca526db03a275bff0b7ffd10d6c512690a4c1b3689b4660f2da65e210.jpg)

![80df5baf30f6a5cfb66b1b92572f4ca0006ea4979df5fc54b12151290465347a.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/images/80df5baf30f6a5cfb66b1b92572f4ca0006ea4979df5fc54b12151290465347a.jpg)

![a7c3ca29e916e018485f0d67bf149f1756e4f02e3508aef93bdfa5082c4cb5b1.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/images/a7c3ca29e916e018485f0d67bf149f1756e4f02e3508aef93bdfa5082c4cb5b1.jpg)

![aa7122473baa208fe60271c8f12a73b94575f98ba8d5b63548590808e086e8db.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/images/aa7122473baa208fe60271c8f12a73b94575f98ba8d5b63548590808e086e8db.jpg)

![c1f946de752df55f647ad0a94978424e28ecbc86aa95988cfe07aa18e23a4354.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/images/c1f946de752df55f647ad0a94978424e28ecbc86aa95988cfe07aa18e23a4354.jpg)

![c287124fdc7c89da0310f607a8340df736347d8a42bafdd865cd5ecfdf9fb9a0.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/images/c287124fdc7c89da0310f607a8340df736347d8a42bafdd865cd5ecfdf9fb9a0.jpg)

![e53000c081dfb5c64fb3c2deb91f0d8674709712fb9505e350a63d9663c56de4.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/images/e53000c081dfb5c64fb3c2deb91f0d8674709712fb9505e350a63d9663c56de4.jpg)

![e91faa86000b6cc85090079c922e8c650ce9ed569234d30941086ff444fa9314.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/images/e91faa86000b6cc85090079c922e8c650ce9ed569234d30941086ff444fa9314.jpg)

### Tables

![17284a14965d882b2a262c6ab614505f517f2b7f32a26f5ca04a8ec0a2180b8b.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/tables/17284a14965d882b2a262c6ab614505f517f2b7f32a26f5ca04a8ec0a2180b8b.jpg)

![2bea1ee4159ba6d8d59a09c73a83a67949824c4cf0f48c670de73887b506b84b.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/tables/2bea1ee4159ba6d8d59a09c73a83a67949824c4cf0f48c670de73887b506b84b.jpg)

![43af2c93e76c3cc35aee3b6d45bee9dd715f1172d5e28a4f58448b5031539d2d.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/tables/43af2c93e76c3cc35aee3b6d45bee9dd715f1172d5e28a4f58448b5031539d2d.jpg)

![4d46ef8fbe135ca05b186038dca6983c4fefa224de1fd4f0d7ff29e72631d372.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/tables/4d46ef8fbe135ca05b186038dca6983c4fefa224de1fd4f0d7ff29e72631d372.jpg)

![59ab523be1eb3d0d115a564b21a91f898a30fab190fc723d354fdde790f0f5d4.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/tables/59ab523be1eb3d0d115a564b21a91f898a30fab190fc723d354fdde790f0f5d4.jpg)

![5ec577c706b346250ca8d860dd9b58741b5d4a111394821ae78fbfbeb510a64a.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/tables/5ec577c706b346250ca8d860dd9b58741b5d4a111394821ae78fbfbeb510a64a.jpg)

![6155f2b75bb7fab55aec38d596d68499ee688e5d9449c276e933c099d75f1288.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/tables/6155f2b75bb7fab55aec38d596d68499ee688e5d9449c276e933c099d75f1288.jpg)

![675c8668efb6aecf9bf391a280c40636bf0d9374b88227aa156403b1366c156d.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/tables/675c8668efb6aecf9bf391a280c40636bf0d9374b88227aa156403b1366c156d.jpg)

![8e413142bbd77621f970ff32439f4bb86e8380d915ae5283fe89153a79e5ef10.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/tables/8e413142bbd77621f970ff32439f4bb86e8380d915ae5283fe89153a79e5ef10.jpg)

![9368439f15cd4af5f8b6a3baa90895050177feec95d70a0f6e040573c1806e88.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/tables/9368439f15cd4af5f8b6a3baa90895050177feec95d70a0f6e040573c1806e88.jpg)

![af79e19be1642560d75047693745090c937d6e64d92e16a4308cc98c5f8af012.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/tables/af79e19be1642560d75047693745090c937d6e64d92e16a4308cc98c5f8af012.jpg)

![c8114edf3af8f0c6a674a314052f83a9c21534f5d70bdf537c56db4101a68e72.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/tables/c8114edf3af8f0c6a674a314052f83a9c21534f5d70bdf537c56db4101a68e72.jpg)

![ccefa801ab6f6ecf7ba99fe039e528a533785cd09ee679e33df401c3574d7058.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/tables/ccefa801ab6f6ecf7ba99fe039e528a533785cd09ee679e33df401c3574d7058.jpg)

![d90cb069b6bfc69b26a320770e87969ac80ff45d3f63fbf2506a55e2a425baf8.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/tables/d90cb069b6bfc69b26a320770e87969ac80ff45d3f63fbf2506a55e2a425baf8.jpg)

![fc2f280c2ce502923d1399b242dfa4ad5be59e579bdbf8d020e7fedea82b44f8.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/tables/fc2f280c2ce502923d1399b242dfa4ad5be59e579bdbf8d020e7fedea82b44f8.jpg)

![ffafb35992b12ff98c0e4820e27cea7c738867fbd79c52dc2721e256979a0278.jpg](../icml_results/3498_Open-o3-Video_%20Grounded%20Video%20Reasoning%20with%20Explicit%20Spatio-Temporal%20Evidence/tables/ffafb35992b12ff98c0e4820e27cea7c738867fbd79c52dc2721e256979a0278.jpg)

<a id="paper-3459"></a>

## Norm$\times$Direction: Restoring the Missing Query Norm in Vision Linear Attention


### Images

![0c21578c0bc39a599531832703c36aa4169949cd982de763c69292f3dd77f92c.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/images/0c21578c0bc39a599531832703c36aa4169949cd982de763c69292f3dd77f92c.jpg)

![1161bff2704c9e1f9497fead768afd1f86b14d1537d43490f54b177874f7a3b3.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/images/1161bff2704c9e1f9497fead768afd1f86b14d1537d43490f54b177874f7a3b3.jpg)

![5f6f2e81f2c41e1cbdabc474ca6eb42bdef8702f4f9240cd362846917f2d8d24.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/images/5f6f2e81f2c41e1cbdabc474ca6eb42bdef8702f4f9240cd362846917f2d8d24.jpg)

![60f1b56aeb7dd52058d4e48f0171f8818e37ef06d845702cb4cd25eb99e91bd2.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/images/60f1b56aeb7dd52058d4e48f0171f8818e37ef06d845702cb4cd25eb99e91bd2.jpg)

![6b640e198078f08563d0b67b862747e5cf5f0ca101e2bdeb040dc0fc47d57a83.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/images/6b640e198078f08563d0b67b862747e5cf5f0ca101e2bdeb040dc0fc47d57a83.jpg)

![8cc47ccf373b2bfe8f8a5747ed82c7cd34085daec2875daa0090626ffb37d342.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/images/8cc47ccf373b2bfe8f8a5747ed82c7cd34085daec2875daa0090626ffb37d342.jpg)

![a8e4437ef445e4a91dcecd9b46ff328b55a80aef8032ea38ea8b48289d273756.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/images/a8e4437ef445e4a91dcecd9b46ff328b55a80aef8032ea38ea8b48289d273756.jpg)

![e8af0461c2746bf3775198cfd3b13d500cc14d8264da605726c80ce09d31c0a0.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/images/e8af0461c2746bf3775198cfd3b13d500cc14d8264da605726c80ce09d31c0a0.jpg)

![ee2ce9559059012b3fd667063f39d453c5f3911122a796be325f3d49e1480e4d.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/images/ee2ce9559059012b3fd667063f39d453c5f3911122a796be325f3d49e1480e4d.jpg)

![f1f3a9b7f736b31373f738a989165fa73489688ba9b89e5913624b7060f2ac29.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/images/f1f3a9b7f736b31373f738a989165fa73489688ba9b89e5913624b7060f2ac29.jpg)

### Tables

![06a7baeb98a603ecf9d9284d521e6bec403a6d557f556f52e988f00fd1bc7498.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/06a7baeb98a603ecf9d9284d521e6bec403a6d557f556f52e988f00fd1bc7498.jpg)

![0cad7bd3018b0d81c81bd5762c0b7431e494e01de3c1d29a3651825ae3fca4eb.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/0cad7bd3018b0d81c81bd5762c0b7431e494e01de3c1d29a3651825ae3fca4eb.jpg)

![1796570bca770c3adcbae31e6712fa1e41bf81c0dbcda254a6d79a3e5489b275.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/1796570bca770c3adcbae31e6712fa1e41bf81c0dbcda254a6d79a3e5489b275.jpg)

![17c84e12b94009185bc8f2412281fe7249854c992f363c1d77dbc6350944f578.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/17c84e12b94009185bc8f2412281fe7249854c992f363c1d77dbc6350944f578.jpg)

![26a8a10491d435e9b8dd55e6d413f27f9b68bfd1222ebdc8591dd062410f7821.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/26a8a10491d435e9b8dd55e6d413f27f9b68bfd1222ebdc8591dd062410f7821.jpg)

![35611c03d76047373b06f8ce06daa693512c99c504b1eb18b57d733dcab8b091.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/35611c03d76047373b06f8ce06daa693512c99c504b1eb18b57d733dcab8b091.jpg)

![3e7a300532e935e0db3428e5a74dfbb34dc07e718a5f524d160b12dbeed393cc.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/3e7a300532e935e0db3428e5a74dfbb34dc07e718a5f524d160b12dbeed393cc.jpg)

![457e08cacd9e96d343ddaa5922f824ed98151dac7c4aebb90c3f35976876dda4.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/457e08cacd9e96d343ddaa5922f824ed98151dac7c4aebb90c3f35976876dda4.jpg)

![654198f7da0881fb538597b83da56d65b22987b2288c7f22d7672801ca38841e.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/654198f7da0881fb538597b83da56d65b22987b2288c7f22d7672801ca38841e.jpg)

![6b3fcc6636df042ac962ee7bec8e90dff0ac25efb574355f619570d8c5c0101c.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/6b3fcc6636df042ac962ee7bec8e90dff0ac25efb574355f619570d8c5c0101c.jpg)

![79fefed875bedfcb4d37f522dc2b8efe2593d164586de17d0d4b282de41a03e3.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/79fefed875bedfcb4d37f522dc2b8efe2593d164586de17d0d4b282de41a03e3.jpg)

![7e616770d0c2dd1e30ef36174c8eecb4e70d93072e23b4f3f20c69669fca6e82.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/7e616770d0c2dd1e30ef36174c8eecb4e70d93072e23b4f3f20c69669fca6e82.jpg)

![85c5d0f3b1cbc8fb245a13a36207ff0dd84a5c0f8a412bab6180760c856c916c.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/85c5d0f3b1cbc8fb245a13a36207ff0dd84a5c0f8a412bab6180760c856c916c.jpg)

![a0b4dde9e2038aa2152804d8ec84cc7be3f6dcad96bfbaefb0b94273d3a395af.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/a0b4dde9e2038aa2152804d8ec84cc7be3f6dcad96bfbaefb0b94273d3a395af.jpg)

![b4e207c653e1cdd58dbe843e48faa5eb0060a25c5a3e1eae26c7efee14b2719d.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/b4e207c653e1cdd58dbe843e48faa5eb0060a25c5a3e1eae26c7efee14b2719d.jpg)

![ba0527db79ac1e03e9a5d291f252513e4bc451073044119f36275bd7db193415.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/ba0527db79ac1e03e9a5d291f252513e4bc451073044119f36275bd7db193415.jpg)

![cc10ca7b4f4ead89ad2b1537810d10a577a6ce6fe04492364a38618fb34b0ebd.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/cc10ca7b4f4ead89ad2b1537810d10a577a6ce6fe04492364a38618fb34b0ebd.jpg)

![cca3bfa5f69721cf1fcd9a1afd253d032b4af32c8bbc1ff71ce9aa81b45dbbd6.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/cca3bfa5f69721cf1fcd9a1afd253d032b4af32c8bbc1ff71ce9aa81b45dbbd6.jpg)

![ddf43147037973b0d621c387119129f359080cc3732bd4a8d507894fe6a8b6aa.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/ddf43147037973b0d621c387119129f359080cc3732bd4a8d507894fe6a8b6aa.jpg)

![e1f71bc7c227a2c1db19cd81cced4c09cbf078627bfb9c0b231f114c89b23a12.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/e1f71bc7c227a2c1db19cd81cced4c09cbf078627bfb9c0b231f114c89b23a12.jpg)

![e6c9880f4218277956b8874a6116b6157bc6a95b9a5b55a463c23537bc6d2061.jpg](../icml_results/3499_Norm%24_times%24Direction_%20Restoring%20the%20Missing%20Query%20Norm%20in%20Vision%20Linear%20Attention/tables/e6c9880f4218277956b8874a6116b6157bc6a95b9a5b55a463c23537bc6d2061.jpg)

<a id="paper-3460"></a>

## WhisperSplat: Lossless Steganography in 3D Gaussian Splatting


### Images

![0cdde76648aa9f5f2ac278e40df94abc82760646014728460430488a4669b141.jpg](../icml_results/3500_WhisperSplat_%20Lossless%20Steganography%20in%203D%20Gaussian%20Splatting/images/0cdde76648aa9f5f2ac278e40df94abc82760646014728460430488a4669b141.jpg)

![122dd8505be43c3e4d46ded213faf560be1c7e98dc9d2f62fef9dce0625e2728.jpg](../icml_results/3500_WhisperSplat_%20Lossless%20Steganography%20in%203D%20Gaussian%20Splatting/images/122dd8505be43c3e4d46ded213faf560be1c7e98dc9d2f62fef9dce0625e2728.jpg)

![5f5bc0ac174d94678e2a4bb898a3cb5dc382427f0f04b431b259e6b342d47b00.jpg](../icml_results/3500_WhisperSplat_%20Lossless%20Steganography%20in%203D%20Gaussian%20Splatting/images/5f5bc0ac174d94678e2a4bb898a3cb5dc382427f0f04b431b259e6b342d47b00.jpg)

![80246fcc7524750df27a69ccc0d2516c7425ff9a4171f9df336a868547489458.jpg](../icml_results/3500_WhisperSplat_%20Lossless%20Steganography%20in%203D%20Gaussian%20Splatting/images/80246fcc7524750df27a69ccc0d2516c7425ff9a4171f9df336a868547489458.jpg)

### Tables

![044e0eec651851df7006fb07469875b1c8c0022a50ff7379f0892b29c182430b.jpg](../icml_results/3500_WhisperSplat_%20Lossless%20Steganography%20in%203D%20Gaussian%20Splatting/tables/044e0eec651851df7006fb07469875b1c8c0022a50ff7379f0892b29c182430b.jpg)

![34a9e23e2b7216ac8abf415cef3fb2d7a37378c69ec510e444b50880f24b3515.jpg](../icml_results/3500_WhisperSplat_%20Lossless%20Steganography%20in%203D%20Gaussian%20Splatting/tables/34a9e23e2b7216ac8abf415cef3fb2d7a37378c69ec510e444b50880f24b3515.jpg)

![437d553f6329e76fda79d0c0b510ee9f20d101efcb4610d353a3eec05be20977.jpg](../icml_results/3500_WhisperSplat_%20Lossless%20Steganography%20in%203D%20Gaussian%20Splatting/tables/437d553f6329e76fda79d0c0b510ee9f20d101efcb4610d353a3eec05be20977.jpg)

![4daff8a97004845522da102d3ff8782d4296ad60a9f50e515a60d97971d1e3f6.jpg](../icml_results/3500_WhisperSplat_%20Lossless%20Steganography%20in%203D%20Gaussian%20Splatting/tables/4daff8a97004845522da102d3ff8782d4296ad60a9f50e515a60d97971d1e3f6.jpg)

![5643d25506aca87800ae2c23a8b4e83a62eaf394054f24a726e61d5e255e910f.jpg](../icml_results/3500_WhisperSplat_%20Lossless%20Steganography%20in%203D%20Gaussian%20Splatting/tables/5643d25506aca87800ae2c23a8b4e83a62eaf394054f24a726e61d5e255e910f.jpg)

![6aa1b7abb43b9edc89855ddfeb384c9a1338f4295c0a184fcd4153d7a00fb9c7.jpg](../icml_results/3500_WhisperSplat_%20Lossless%20Steganography%20in%203D%20Gaussian%20Splatting/tables/6aa1b7abb43b9edc89855ddfeb384c9a1338f4295c0a184fcd4153d7a00fb9c7.jpg)

![6f32f85eed71ce3660183ac05fba17019fb8c76a4413af72dd65f24d7eded5f6.jpg](../icml_results/3500_WhisperSplat_%20Lossless%20Steganography%20in%203D%20Gaussian%20Splatting/tables/6f32f85eed71ce3660183ac05fba17019fb8c76a4413af72dd65f24d7eded5f6.jpg)

![7300d658a85f06b49f98ebe34c524d297421a712ac89baf398f93cf3e49194b3.jpg](../icml_results/3500_WhisperSplat_%20Lossless%20Steganography%20in%203D%20Gaussian%20Splatting/tables/7300d658a85f06b49f98ebe34c524d297421a712ac89baf398f93cf3e49194b3.jpg)

![a5282fb59936dc92122222afb2a49d60be294e303e273bb5d936dc97a068073e.jpg](../icml_results/3500_WhisperSplat_%20Lossless%20Steganography%20in%203D%20Gaussian%20Splatting/tables/a5282fb59936dc92122222afb2a49d60be294e303e273bb5d936dc97a068073e.jpg)

![a8d3bc72aa149766ca314a1f2952cf6cfb7a603b73a6732300a0f88460d1c9e0.jpg](../icml_results/3500_WhisperSplat_%20Lossless%20Steganography%20in%203D%20Gaussian%20Splatting/tables/a8d3bc72aa149766ca314a1f2952cf6cfb7a603b73a6732300a0f88460d1c9e0.jpg)

![b0c50a6bc9ff8b5c7bb782e417c7d91eb843ebc4e61086ddac324f078eeae6c8.jpg](../icml_results/3500_WhisperSplat_%20Lossless%20Steganography%20in%203D%20Gaussian%20Splatting/tables/b0c50a6bc9ff8b5c7bb782e417c7d91eb843ebc4e61086ddac324f078eeae6c8.jpg)

![d3daa5a3d8a37ece40cbba024847404551045e63db8dd979375aab75ae9db020.jpg](../icml_results/3500_WhisperSplat_%20Lossless%20Steganography%20in%203D%20Gaussian%20Splatting/tables/d3daa5a3d8a37ece40cbba024847404551045e63db8dd979375aab75ae9db020.jpg)

<a id="paper-3461"></a>

## Deep Coupling Learning for Solving PDEs


### Images

![253c72e571c91b77457791fd7db540fb0e7229b1b101221d180dbadbbc7c0ce0.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/images/253c72e571c91b77457791fd7db540fb0e7229b1b101221d180dbadbbc7c0ce0.jpg)

![44d50899051da8372633b03e4b3c647003fccf82352aa63cf0d5581f561f1f8e.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/images/44d50899051da8372633b03e4b3c647003fccf82352aa63cf0d5581f561f1f8e.jpg)

![5181a4fb2a996ccbcfe0a4771661b820813cb747779f79a9a9ee1acdcb1080a9.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/images/5181a4fb2a996ccbcfe0a4771661b820813cb747779f79a9a9ee1acdcb1080a9.jpg)

![61277f5bd722e2225000856f5315a2ea35ef88b949a19ed6a31e74b1c8dd232e.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/images/61277f5bd722e2225000856f5315a2ea35ef88b949a19ed6a31e74b1c8dd232e.jpg)

![63403fe58f1139f5c00ac3d73aed8760bb29d9bb9450f18554be9e2ef134b577.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/images/63403fe58f1139f5c00ac3d73aed8760bb29d9bb9450f18554be9e2ef134b577.jpg)

![745cc68aa3298402faf4e8c6aadef2237763e2e73da0c91f316ef13c2906dbf1.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/images/745cc68aa3298402faf4e8c6aadef2237763e2e73da0c91f316ef13c2906dbf1.jpg)

![8c6b761c110a8cb76bc9683ac9b3c6a55ceea9015e91c7321e484dd5c0a26be5.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/images/8c6b761c110a8cb76bc9683ac9b3c6a55ceea9015e91c7321e484dd5c0a26be5.jpg)

![adbe29c3a9935c7f0cc2bae4b2adfefa8c42dba848441e063e818a8e9c58b095.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/images/adbe29c3a9935c7f0cc2bae4b2adfefa8c42dba848441e063e818a8e9c58b095.jpg)

![b0f836d4f2fd97f793cb40f1e8c31d692e7a1cac65730621511cfd22e702b3bc.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/images/b0f836d4f2fd97f793cb40f1e8c31d692e7a1cac65730621511cfd22e702b3bc.jpg)

![c71a11844c665cbff648dbc36e2c105d54b97e578f7645ad24bdf35dfb86e9b3.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/images/c71a11844c665cbff648dbc36e2c105d54b97e578f7645ad24bdf35dfb86e9b3.jpg)

![d0216b8ad6ee92d898b8fdf598a64224c11082044e9534f1a853fe78a8ba3869.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/images/d0216b8ad6ee92d898b8fdf598a64224c11082044e9534f1a853fe78a8ba3869.jpg)

![d4e31944bed4812af67c34f3b6485242656c81ac6b1ca621b38b3731c954c9ad.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/images/d4e31944bed4812af67c34f3b6485242656c81ac6b1ca621b38b3731c954c9ad.jpg)

![d881364aa95421231c0eb5ab29ef2c57638b135e100c56e93eb01cd889294503.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/images/d881364aa95421231c0eb5ab29ef2c57638b135e100c56e93eb01cd889294503.jpg)

![e404bbb472b88faf7f418a9336e84f572c290164d39c0723874b59de376d3a86.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/images/e404bbb472b88faf7f418a9336e84f572c290164d39c0723874b59de376d3a86.jpg)

![e5984acef574cc597bbea56fb3ace1792ae30c34c247a1a73b9a5d37ccf3398c.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/images/e5984acef574cc597bbea56fb3ace1792ae30c34c247a1a73b9a5d37ccf3398c.jpg)

![e8ccb82149119584a0ee4e0ef98465bc2f203f82cc422cb7ca4b49032f37c1d3.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/images/e8ccb82149119584a0ee4e0ef98465bc2f203f82cc422cb7ca4b49032f37c1d3.jpg)

![eab6c54d0752905815d08aff187cd8c241a490e29c0154a9504cbd2286067226.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/images/eab6c54d0752905815d08aff187cd8c241a490e29c0154a9504cbd2286067226.jpg)

### Tables

![03342a6dab9b3395f312a518d7e6638618841c650f11f6efa30c26cf9a094fe2.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/tables/03342a6dab9b3395f312a518d7e6638618841c650f11f6efa30c26cf9a094fe2.jpg)

![11010f597addbbb69f72bfc81ef5fa05da9a82c8138767bdd6efa047b41a0c5c.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/tables/11010f597addbbb69f72bfc81ef5fa05da9a82c8138767bdd6efa047b41a0c5c.jpg)

![1e35542dba58e144b507712faa8082fa056fe8a3a510312ca50a1d9e5f24080c.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/tables/1e35542dba58e144b507712faa8082fa056fe8a3a510312ca50a1d9e5f24080c.jpg)

![4e94a1976ea98c0aa955cdff6116bece43b561f023d664afa2a8e8665bfb69f7.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/tables/4e94a1976ea98c0aa955cdff6116bece43b561f023d664afa2a8e8665bfb69f7.jpg)

![52c4dbeffdd81fb2633e3bf75677965b097b5f119a9fe7de250d2a1f01dcdc1b.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/tables/52c4dbeffdd81fb2633e3bf75677965b097b5f119a9fe7de250d2a1f01dcdc1b.jpg)

![5b2c094d48f4adead09d92541dbd2482ea407cb97108d2fe2410d124b6eb0786.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/tables/5b2c094d48f4adead09d92541dbd2482ea407cb97108d2fe2410d124b6eb0786.jpg)

![9237cbde270e9391b3838140a01b43058eb500c684788a181ad0085bd0577ec2.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/tables/9237cbde270e9391b3838140a01b43058eb500c684788a181ad0085bd0577ec2.jpg)

![97b5accc5f7c9d5169d36fd2148b6eebf420a170076862601e6fdaffd6a6d003.jpg](../icml_results/3501_Deep%20Coupling%20Learning%20for%20Solving%20PDEs/tables/97b5accc5f7c9d5169d36fd2148b6eebf420a170076862601e6fdaffd6a6d003.jpg)

<a id="paper-3462"></a>

## Gradient Smoothing: Coupling Layer-wise Updates for Improved Optimization


### Images

![12e6afa9c7208adbac85da8c087117d0d008f86e4cce49c18d828af9d9b73560.jpg](../icml_results/3502_Gradient%20Smoothing_%20Coupling%20Layer-wise%20Updates%20for%20Improved%20Optimization/images/12e6afa9c7208adbac85da8c087117d0d008f86e4cce49c18d828af9d9b73560.jpg)

![2326fd217385c48e5d2deccc3d024102405e97bf3f061ced90d4abfd37ae46e6.jpg](../icml_results/3502_Gradient%20Smoothing_%20Coupling%20Layer-wise%20Updates%20for%20Improved%20Optimization/images/2326fd217385c48e5d2deccc3d024102405e97bf3f061ced90d4abfd37ae46e6.jpg)

![4dcc5adc11f9ff583e05b67a70f21c9ee3f15af5a0ad336b2c4420f760962dc3.jpg](../icml_results/3502_Gradient%20Smoothing_%20Coupling%20Layer-wise%20Updates%20for%20Improved%20Optimization/images/4dcc5adc11f9ff583e05b67a70f21c9ee3f15af5a0ad336b2c4420f760962dc3.jpg)

![60a0c1a8d3dea70f6e86ba21664a93ab4befde3953347ee4973e43abd72ca47e.jpg](../icml_results/3502_Gradient%20Smoothing_%20Coupling%20Layer-wise%20Updates%20for%20Improved%20Optimization/images/60a0c1a8d3dea70f6e86ba21664a93ab4befde3953347ee4973e43abd72ca47e.jpg)

![85b052f938101b8d5ef684cfc4f9467e6f019366bfcc21773d6b3d701b365d97.jpg](../icml_results/3502_Gradient%20Smoothing_%20Coupling%20Layer-wise%20Updates%20for%20Improved%20Optimization/images/85b052f938101b8d5ef684cfc4f9467e6f019366bfcc21773d6b3d701b365d97.jpg)

![c1673900fc2cb2553d25f93e2c1778820ff4978bf01dfabfec08fb544079805d.jpg](../icml_results/3502_Gradient%20Smoothing_%20Coupling%20Layer-wise%20Updates%20for%20Improved%20Optimization/images/c1673900fc2cb2553d25f93e2c1778820ff4978bf01dfabfec08fb544079805d.jpg)

![d1771dde28e0a3721c9f9abd6c4f719281502c8152fa7d0075de69b6f517e371.jpg](../icml_results/3502_Gradient%20Smoothing_%20Coupling%20Layer-wise%20Updates%20for%20Improved%20Optimization/images/d1771dde28e0a3721c9f9abd6c4f719281502c8152fa7d0075de69b6f517e371.jpg)

![e7678139a689ece7f8cff574c921fd01c7b045eb7fe10d0f3a40c42114b4d36c.jpg](../icml_results/3502_Gradient%20Smoothing_%20Coupling%20Layer-wise%20Updates%20for%20Improved%20Optimization/images/e7678139a689ece7f8cff574c921fd01c7b045eb7fe10d0f3a40c42114b4d36c.jpg)

### Tables

![341977c9b889da8f9685bde87eef6cd8b2b6c9e6aa15642d134ff969b31a5088.jpg](../icml_results/3502_Gradient%20Smoothing_%20Coupling%20Layer-wise%20Updates%20for%20Improved%20Optimization/tables/341977c9b889da8f9685bde87eef6cd8b2b6c9e6aa15642d134ff969b31a5088.jpg)

![9c9bbfd22ec7698e245f798fdf05ccf8ce30028ab82bf9b54c8e23d2658d3b10.jpg](../icml_results/3502_Gradient%20Smoothing_%20Coupling%20Layer-wise%20Updates%20for%20Improved%20Optimization/tables/9c9bbfd22ec7698e245f798fdf05ccf8ce30028ab82bf9b54c8e23d2658d3b10.jpg)

![b303bc2bb8cf9e5b593237b60389babf2249c622fb948299c6f8c3382c7c70af.jpg](../icml_results/3502_Gradient%20Smoothing_%20Coupling%20Layer-wise%20Updates%20for%20Improved%20Optimization/tables/b303bc2bb8cf9e5b593237b60389babf2249c622fb948299c6f8c3382c7c70af.jpg)

<a id="paper-3463"></a>

## Benchmarking LLM-Assisted Blue Teaming via Standardized Threat Hunting


### Images

![21c46203eddf30611a7b4791fc64c83706bc16b16a20622aa4f1b558444e2ae6.jpg](../icml_results/3503_Benchmarking%20LLM-Assisted%20Blue%20Teaming%20via%20Standardized%20Threat%20Hunting/images/21c46203eddf30611a7b4791fc64c83706bc16b16a20622aa4f1b558444e2ae6.jpg)

![3ec73d6984ea4bc71e8523150965b639527d62474d22a0a2eb3da50f59d7c6b6.jpg](../icml_results/3503_Benchmarking%20LLM-Assisted%20Blue%20Teaming%20via%20Standardized%20Threat%20Hunting/images/3ec73d6984ea4bc71e8523150965b639527d62474d22a0a2eb3da50f59d7c6b6.jpg)

![55e76aa1a6082debf3c3a99eebab11226f97e3e698d9796855c018fa3741edfb.jpg](../icml_results/3503_Benchmarking%20LLM-Assisted%20Blue%20Teaming%20via%20Standardized%20Threat%20Hunting/images/55e76aa1a6082debf3c3a99eebab11226f97e3e698d9796855c018fa3741edfb.jpg)

![ae0d5a06de14af749d4bbcc8befdcbf3bf768035a1b89893dc4940f0b38540c5.jpg](../icml_results/3503_Benchmarking%20LLM-Assisted%20Blue%20Teaming%20via%20Standardized%20Threat%20Hunting/images/ae0d5a06de14af749d4bbcc8befdcbf3bf768035a1b89893dc4940f0b38540c5.jpg)

![ba12dc0bae43048f63d6f4dc48e934e8eca7fff59ce222c8be2dfc43da8e03d0.jpg](../icml_results/3503_Benchmarking%20LLM-Assisted%20Blue%20Teaming%20via%20Standardized%20Threat%20Hunting/images/ba12dc0bae43048f63d6f4dc48e934e8eca7fff59ce222c8be2dfc43da8e03d0.jpg)

![bb70ae2afcec80d7cd2b26906d9c7769f355722434475a146ba55a6bc6d3ab9f.jpg](../icml_results/3503_Benchmarking%20LLM-Assisted%20Blue%20Teaming%20via%20Standardized%20Threat%20Hunting/images/bb70ae2afcec80d7cd2b26906d9c7769f355722434475a146ba55a6bc6d3ab9f.jpg)

![d620e351ea96c1faff78cf8657e680d6f3a6951ec06a746b4cf816f721dc32b2.jpg](../icml_results/3503_Benchmarking%20LLM-Assisted%20Blue%20Teaming%20via%20Standardized%20Threat%20Hunting/images/d620e351ea96c1faff78cf8657e680d6f3a6951ec06a746b4cf816f721dc32b2.jpg)

![f6b9d09554c52c1b3144e1e1f7e06f0da442cd19bbb07511d7ab6b0634c55552.jpg](../icml_results/3503_Benchmarking%20LLM-Assisted%20Blue%20Teaming%20via%20Standardized%20Threat%20Hunting/images/f6b9d09554c52c1b3144e1e1f7e06f0da442cd19bbb07511d7ab6b0634c55552.jpg)

### Tables

![095b7f8c9c04041237b9340a51440af61df5dcae73523e92d46181ad3ccc7621.jpg](../icml_results/3503_Benchmarking%20LLM-Assisted%20Blue%20Teaming%20via%20Standardized%20Threat%20Hunting/tables/095b7f8c9c04041237b9340a51440af61df5dcae73523e92d46181ad3ccc7621.jpg)

![174246342c6ff4785fd846a3a35ff32b30fae0e0ad5ef80bc8a743d9dcab138c.jpg](../icml_results/3503_Benchmarking%20LLM-Assisted%20Blue%20Teaming%20via%20Standardized%20Threat%20Hunting/tables/174246342c6ff4785fd846a3a35ff32b30fae0e0ad5ef80bc8a743d9dcab138c.jpg)

![3ae47e6301e2bf113ec7f45970848fefb3fa70dedb5e489ff20ca88cd81a59d0.jpg](../icml_results/3503_Benchmarking%20LLM-Assisted%20Blue%20Teaming%20via%20Standardized%20Threat%20Hunting/tables/3ae47e6301e2bf113ec7f45970848fefb3fa70dedb5e489ff20ca88cd81a59d0.jpg)

![531ecbc4605ca2b42455beb860101ed58123abddfe4a35cd2ed66d1ba95fdec7.jpg](../icml_results/3503_Benchmarking%20LLM-Assisted%20Blue%20Teaming%20via%20Standardized%20Threat%20Hunting/tables/531ecbc4605ca2b42455beb860101ed58123abddfe4a35cd2ed66d1ba95fdec7.jpg)

![7fff08db939e2fd110f81a48b6f8332bdc2be9e8a97c0a2f2f0a446a94cfd388.jpg](../icml_results/3503_Benchmarking%20LLM-Assisted%20Blue%20Teaming%20via%20Standardized%20Threat%20Hunting/tables/7fff08db939e2fd110f81a48b6f8332bdc2be9e8a97c0a2f2f0a446a94cfd388.jpg)

![a139759edfc62e204347d1415b58e33ec278ce7e87d5d06ce7a1d001507fa31f.jpg](../icml_results/3503_Benchmarking%20LLM-Assisted%20Blue%20Teaming%20via%20Standardized%20Threat%20Hunting/tables/a139759edfc62e204347d1415b58e33ec278ce7e87d5d06ce7a1d001507fa31f.jpg)

![aeec4a0f72e061213c6a0448df717bbe040105dc5597885e41e555f1ee6b57a9.jpg](../icml_results/3503_Benchmarking%20LLM-Assisted%20Blue%20Teaming%20via%20Standardized%20Threat%20Hunting/tables/aeec4a0f72e061213c6a0448df717bbe040105dc5597885e41e555f1ee6b57a9.jpg)

![f1927c0d27e5ca133ca97d7c304581f1041fed465c3655f9fa2d3fae4672c691.jpg](../icml_results/3503_Benchmarking%20LLM-Assisted%20Blue%20Teaming%20via%20Standardized%20Threat%20Hunting/tables/f1927c0d27e5ca133ca97d7c304581f1041fed465c3655f9fa2d3fae4672c691.jpg)

![fcb4d9ef02a91402980d6cc786c7c80a75021bb521666feaf61cda1c3bceacf7.jpg](../icml_results/3503_Benchmarking%20LLM-Assisted%20Blue%20Teaming%20via%20Standardized%20Threat%20Hunting/tables/fcb4d9ef02a91402980d6cc786c7c80a75021bb521666feaf61cda1c3bceacf7.jpg)

<a id="paper-3464"></a>

## Small Agent Group is the Future of Digital Health


### Images

![058d78ac761ae4591ed652bf2383d31d2b26aa47db56826e2ca21e5b4e682d94.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/images/058d78ac761ae4591ed652bf2383d31d2b26aa47db56826e2ca21e5b4e682d94.jpg)

![091cd698b14b429a508ee205a79e6b8975de4a8a4c182e69b2c8595afbc38716.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/images/091cd698b14b429a508ee205a79e6b8975de4a8a4c182e69b2c8595afbc38716.jpg)

![650b18d8bdc32c4644c9e5ba7540dd10acc7cb4fafde500c2fea4653511e09ee.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/images/650b18d8bdc32c4644c9e5ba7540dd10acc7cb4fafde500c2fea4653511e09ee.jpg)

![6cae9b4c9d40a55a8057e23c0654ca02f6f2d1d97edf9c13074d316f25ec1d71.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/images/6cae9b4c9d40a55a8057e23c0654ca02f6f2d1d97edf9c13074d316f25ec1d71.jpg)

![7536a53f5c0629f07b37f2966ecf2b57dad3be9f1fc1eebe5d4c9fc5addc8c25.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/images/7536a53f5c0629f07b37f2966ecf2b57dad3be9f1fc1eebe5d4c9fc5addc8c25.jpg)

![f51b6403a056a8328ddb1712345d9cd2fec937ea1be409b667309086684be8cd.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/images/f51b6403a056a8328ddb1712345d9cd2fec937ea1be409b667309086684be8cd.jpg)

![fd10a3b026a5fc61b39713cc9e4b6af87e412094ebf6c7d3b230ecbcc61a6e5d.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/images/fd10a3b026a5fc61b39713cc9e4b6af87e412094ebf6c7d3b230ecbcc61a6e5d.jpg)

### Tables

![02114aa437551efc1cbdd922f4ede811405bc468ca41d2e77dd7a011a07fecb8.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/tables/02114aa437551efc1cbdd922f4ede811405bc468ca41d2e77dd7a011a07fecb8.jpg)

![10eae9708e21532a9ceec78f3866c3e00ef1bde3081a922aa8be32348409e271.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/tables/10eae9708e21532a9ceec78f3866c3e00ef1bde3081a922aa8be32348409e271.jpg)

![2c4041e79afb3fecd667d94f1dec815a5e27acd54ef1e13dfaf4ea9a3877b5b5.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/tables/2c4041e79afb3fecd667d94f1dec815a5e27acd54ef1e13dfaf4ea9a3877b5b5.jpg)

![2f10d7c846ac43cf735fa625e1b9d29f548335f014cbcc6ac8fd4d663a92c698.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/tables/2f10d7c846ac43cf735fa625e1b9d29f548335f014cbcc6ac8fd4d663a92c698.jpg)

![761f21caf1bf38d5c456d0241da14bec0ad4f8d45f2575ed4d0676f3cac84adf.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/tables/761f21caf1bf38d5c456d0241da14bec0ad4f8d45f2575ed4d0676f3cac84adf.jpg)

![a7f917df9cc9d526c78da3b8f6d8b2d2f02522a85b363f2c71e6d4a4b82231e5.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/tables/a7f917df9cc9d526c78da3b8f6d8b2d2f02522a85b363f2c71e6d4a4b82231e5.jpg)

![aef0a283730492a30af1e42fae751f694d45b42955b5c1b7b0295a3e287c7d24.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/tables/aef0a283730492a30af1e42fae751f694d45b42955b5c1b7b0295a3e287c7d24.jpg)

![c1679a8e335d9ffd02ff7ca996b4727f9417351d9aa10521bd449ff309627921.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/tables/c1679a8e335d9ffd02ff7ca996b4727f9417351d9aa10521bd449ff309627921.jpg)

![d30128e48de087059cc5400c3f9e68a6646eb8e504691ff9570894c976f93370.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/tables/d30128e48de087059cc5400c3f9e68a6646eb8e504691ff9570894c976f93370.jpg)

![d4372640fce393aee112d2754b73a9eb9f008c7ba06e34c90aa81386dda66dbb.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/tables/d4372640fce393aee112d2754b73a9eb9f008c7ba06e34c90aa81386dda66dbb.jpg)

![d54efd60ad747930aab01d3049e1fa2fac6dc6c5393485af2e86668d631a10e3.jpg](../icml_results/3504_Small%20Agent%20Group%20is%20the%20Future%20of%20Digital%20Health/tables/d54efd60ad747930aab01d3049e1fa2fac6dc6c5393485af2e86668d631a10e3.jpg)

<a id="paper-3465"></a>

## PoMtVRS: Preference-Optimized Multi-Task Vehicle Routing Solver with Preference Gating

### Images

![5e6702535d088b0fafc71c9e4ce88eb6e1441a3b6f5eff094c47aeae0b036ff0.jpg](../icml_results/3505_PoMtVRS_%20Preference-Optimized%20Multi-Task%20Vehicle%20Routing%20Solver%20with%20Preference%20Gating/images/5e6702535d088b0fafc71c9e4ce88eb6e1441a3b6f5eff094c47aeae0b036ff0.jpg)

![ecb08d1338a051bf692ec1296c192145036b501e978f2728ccaaddc53035c6eb.jpg](../icml_results/3505_PoMtVRS_%20Preference-Optimized%20Multi-Task%20Vehicle%20Routing%20Solver%20with%20Preference%20Gating/images/ecb08d1338a051bf692ec1296c192145036b501e978f2728ccaaddc53035c6eb.jpg)

### Tables

![0eab02cc1e0da20f61376e34eaf590ce37fdd900af8b7ac0ce2f3c92d57dce82.jpg](../icml_results/3505_PoMtVRS_%20Preference-Optimized%20Multi-Task%20Vehicle%20Routing%20Solver%20with%20Preference%20Gating/tables/0eab02cc1e0da20f61376e34eaf590ce37fdd900af8b7ac0ce2f3c92d57dce82.jpg)

![16b5ec0587eba8ff19e52918d1812b16cc6780037815f0a6613198222914686e.jpg](../icml_results/3505_PoMtVRS_%20Preference-Optimized%20Multi-Task%20Vehicle%20Routing%20Solver%20with%20Preference%20Gating/tables/16b5ec0587eba8ff19e52918d1812b16cc6780037815f0a6613198222914686e.jpg)

![35435155c65ab0f0fc503e3ea1584ab6560f71632348c4f89133fdce749027ec.jpg](../icml_results/3505_PoMtVRS_%20Preference-Optimized%20Multi-Task%20Vehicle%20Routing%20Solver%20with%20Preference%20Gating/tables/35435155c65ab0f0fc503e3ea1584ab6560f71632348c4f89133fdce749027ec.jpg)

![5b73bce1110fb473dc802239cdb709252a6e9afd7f7f5c57fd6faa3898cbfe89.jpg](../icml_results/3505_PoMtVRS_%20Preference-Optimized%20Multi-Task%20Vehicle%20Routing%20Solver%20with%20Preference%20Gating/tables/5b73bce1110fb473dc802239cdb709252a6e9afd7f7f5c57fd6faa3898cbfe89.jpg)

![75e585d0913b196af3258507b6a65a050850a5386d89ef8d37ae27c19915c0e3.jpg](../icml_results/3505_PoMtVRS_%20Preference-Optimized%20Multi-Task%20Vehicle%20Routing%20Solver%20with%20Preference%20Gating/tables/75e585d0913b196af3258507b6a65a050850a5386d89ef8d37ae27c19915c0e3.jpg)

![7eaee14b00c7cd6c09cd3469a844daf2fa045a3adcb9f40fbd3432b0dd471712.jpg](../icml_results/3505_PoMtVRS_%20Preference-Optimized%20Multi-Task%20Vehicle%20Routing%20Solver%20with%20Preference%20Gating/tables/7eaee14b00c7cd6c09cd3469a844daf2fa045a3adcb9f40fbd3432b0dd471712.jpg)

![91061dcb62afad05060af41aefd04e8f2a41f532befa3b23bd1c89b3cdaa5c8a.jpg](../icml_results/3505_PoMtVRS_%20Preference-Optimized%20Multi-Task%20Vehicle%20Routing%20Solver%20with%20Preference%20Gating/tables/91061dcb62afad05060af41aefd04e8f2a41f532befa3b23bd1c89b3cdaa5c8a.jpg)

![cee27b522d9c8c931ed0684da943ae23e88dc7d638fb58f88c374364e5d4a0bd.jpg](../icml_results/3505_PoMtVRS_%20Preference-Optimized%20Multi-Task%20Vehicle%20Routing%20Solver%20with%20Preference%20Gating/tables/cee27b522d9c8c931ed0684da943ae23e88dc7d638fb58f88c374364e5d4a0bd.jpg)

![ec410617036f5def275e2ff1c134832002dd64fd3ef006b9b919df8f31d1c7ba.jpg](../icml_results/3505_PoMtVRS_%20Preference-Optimized%20Multi-Task%20Vehicle%20Routing%20Solver%20with%20Preference%20Gating/tables/ec410617036f5def275e2ff1c134832002dd64fd3ef006b9b919df8f31d1c7ba.jpg)

![f79aad999ab8a58b9c65ad6ad9435780ca5539d09b0371e4a5eea186711c642e.jpg](../icml_results/3505_PoMtVRS_%20Preference-Optimized%20Multi-Task%20Vehicle%20Routing%20Solver%20with%20Preference%20Gating/tables/f79aad999ab8a58b9c65ad6ad9435780ca5539d09b0371e4a5eea186711c642e.jpg)
