# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 101 of 197

## 目录 (Table of Contents)

1. [Mitigating Gradient Pathology in PINNs through Aligned Constraint](#paper-3301)
2. [Variance Driven Exploration: A Provable and Efficient Methodology for Pure Exploration in Highly Stochastic Environments](#paper-3302)
3. [The Decrypto Benchmark for Multi-Agent Reasoning and Theory of Mind](#paper-3303)
4. [Directional Neural Collapse Explains Few-Shot Transfer in Self-Supervised Learning](#paper-3304)
5. [DC-LA: Difference-of-Convex Langevin Algorithm](#paper-3305)
6. [Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models](#paper-3306)
7. [The Hidden Link between RLHF and Contrastive Learning](#paper-3307)
8. [AtomWorld: A Benchmark for Evaluating Spatial Reasoning in Large Language Models on Material Structures](#paper-3308)
9. [SCalDA: Semantics-Calibrated and Diffusion-Enhanced Data Augmentation](#paper-3309)
10. [FLAC: Maximum Entropy RL via Kinetic Energy Regularized Bridge Matching](#paper-3310)
11. [Spatial-Aware Reduction Framework: Towards Efficient and Faithful Visual State Space Models](#paper-3311)
12. [Beyond Rational Illusion: Behaviorally Realistic Strategic Classification](#paper-3312)
13. [When Tabular Foundation Models Meet Strategic Tabular Data: A Prior Alignment Approach](#paper-3313)
14. [Hierarchical ODE: Learning Continuous-Time Physical Prototypes for Early Link Failure Detection](#paper-3314)
15. [Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](#paper-3315)
16. [SHARP-Q: Spectral Hessian Alignment and Rectification for Post-training Quantization](#paper-3316)
17. [Knowing the Unknown: Interpretable Open-World Object Detection via Concept Decomposition Model](#paper-3317)
18. [When Labelers Stay Silent: The Power of Ties in Cost-Effective Preference Learning](#paper-3318)
19. [NaviCache: Test-Time Self-Calibration Caching for Video Generation](#paper-3319)
20. [General Covariant Action Modeling: Constructing Generalized Manifolds via Spatio-Temporal Decoupling](#paper-3320)
21. [LAST: Bridging Vision-Language and Action Manifolds via Gromov-Wasserstein Alignment](#paper-3321)
22. [QiMeng-ChipV-RTL: Exploiting Information Locality for IP-level Verilog Generation](#paper-3322)
23. [Transformers with RL or SFT Provably Learn Sparse Boolean Functions, But Differently](#paper-3323)
24. [Efficiently Solving Discounted MDPs via Predictions with Unknown Prediction Errors](#paper-3324)
25. [Debiased Model-based Representations for Sample-efficient Continuous Control](#paper-3325)
26. [On the Intrinsic Limits of Transformer Image Embeddings in Non-Solvable Spatial Reasoning](#paper-3326)
27. [Cutting LLM Evaluation Costs with SySRs: A Bandit Algorithm That Provably Exploits Model Similarity](#paper-3327)
28. [Explaining Concept Shift with Interpretable Feature Attribution](#paper-3328)
29. [Student-Centered Distillation Narrows the Agentic Gap Between Small and Large LLMs](#paper-3329)
30. [The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?](#paper-3330)
31. [SALAAD: Sparse And Low-Rank Adaptation via ADMM for Large Language Model Inference](#paper-3331)
32. [$μ$pscaling small models: Principled warm starts and hyperparameter transfer](#paper-3332)
33. [Convolutional Learnable-Group Weightless Neural Network](#paper-3333)

---


<a id="paper-3301"></a>

## Mitigating Gradient Pathology in PINNs through Aligned Constraint

### Images

![0b6825c7ae81b49de8d6aac7e6f452fa1bc8c75d0e55ef5aa4d36465495fe509.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/images/0b6825c7ae81b49de8d6aac7e6f452fa1bc8c75d0e55ef5aa4d36465495fe509.jpg)

![3fd49ad55a165bd69d70398fae14d7eddeef926d5d082deeb445d3edaeb6f21a.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/images/3fd49ad55a165bd69d70398fae14d7eddeef926d5d082deeb445d3edaeb6f21a.jpg)

![4a5ddb8b846bb26a08e18674aa61e8ece2c88611801ae40afa33f19fb4c64566.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/images/4a5ddb8b846bb26a08e18674aa61e8ece2c88611801ae40afa33f19fb4c64566.jpg)

![5b7b0ab20d89d55dc71a7f5f22f8a2257fb36a91b1c23adebf2d3618a870ae59.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/images/5b7b0ab20d89d55dc71a7f5f22f8a2257fb36a91b1c23adebf2d3618a870ae59.jpg)

![9a9b431cb08bfa3dc0af0e53a42c636877344f22206a7937203b94f4f41e736c.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/images/9a9b431cb08bfa3dc0af0e53a42c636877344f22206a7937203b94f4f41e736c.jpg)

![f3c1542adde908b0bb59a4633f577138b76b95f7ab8058e65bf9eda03524a849.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/images/f3c1542adde908b0bb59a4633f577138b76b95f7ab8058e65bf9eda03524a849.jpg)

![f764fab6664bb25aa7737c1a7133555b011ef39026f4fe5c49ca8d02eb7eb744.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/images/f764fab6664bb25aa7737c1a7133555b011ef39026f4fe5c49ca8d02eb7eb744.jpg)

### Tables

![1ad7cdbad9e105d2634505f53f1de3d476fc6a45172522c9b59390049a1afd47.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/1ad7cdbad9e105d2634505f53f1de3d476fc6a45172522c9b59390049a1afd47.jpg)

![228cf6825b113a0b97d505e7e1ceeec1b230fbe6aa640d53fa85711692b4185a.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/228cf6825b113a0b97d505e7e1ceeec1b230fbe6aa640d53fa85711692b4185a.jpg)

![4244d6fdba656eb7a350544c65dffb38e88c4a02734522994dda60f487c059dc.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/4244d6fdba656eb7a350544c65dffb38e88c4a02734522994dda60f487c059dc.jpg)

![42e1ccfe9482efbf1d2545abf25548df0958347f47733f37b95d56f969f261cd.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/42e1ccfe9482efbf1d2545abf25548df0958347f47733f37b95d56f969f261cd.jpg)

![67523dfd0606df80d9be1c702f24dff8ed62c463905b38df303b7c15c21bd33a.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/67523dfd0606df80d9be1c702f24dff8ed62c463905b38df303b7c15c21bd33a.jpg)

![94d136faac8c4045a1545ed5f29cc95dacc9f73f35122eb8bbbfb8b4bb39ca0b.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/94d136faac8c4045a1545ed5f29cc95dacc9f73f35122eb8bbbfb8b4bb39ca0b.jpg)

![9ecffef678901d0fe2377d2ec7e0ea6b9b473207ed570b39be6d7f35c8fff01f.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/9ecffef678901d0fe2377d2ec7e0ea6b9b473207ed570b39be6d7f35c8fff01f.jpg)

![b8556d35b78d2e5c43a3f59109cf33991566b18d015a4b4634fa3ba7bda46581.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/b8556d35b78d2e5c43a3f59109cf33991566b18d015a4b4634fa3ba7bda46581.jpg)

![cc8003cf7727d5fc01fe9e67681a88cbf8cc9b03a5446a3c3d533daec69d7414.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/cc8003cf7727d5fc01fe9e67681a88cbf8cc9b03a5446a3c3d533daec69d7414.jpg)

![d2559fbfd6d92c0fd1f733d734f69137bb0d5f263fa9f1eca092b5ee90f9ae88.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/d2559fbfd6d92c0fd1f733d734f69137bb0d5f263fa9f1eca092b5ee90f9ae88.jpg)

![d9691c05e099a0c117a3c7145bba1e040a18d81136384f4b4eaad5716b43ba8e.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/d9691c05e099a0c117a3c7145bba1e040a18d81136384f4b4eaad5716b43ba8e.jpg)

![ea5ec5959b11d8368646c9572615263fae44cb3ae002fcb0c77c00852bf69df4.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/ea5ec5959b11d8368646c9572615263fae44cb3ae002fcb0c77c00852bf69df4.jpg)

![f202a1a6b95014a7b46620bfdfbfadf8ca0558949dcf84e3165b4a8838c0ccea.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/f202a1a6b95014a7b46620bfdfbfadf8ca0558949dcf84e3165b4a8838c0ccea.jpg)

![fffe3b8f4c893945aeaf337f0c074e91022e61ca7fffa6d3cbfdb6cf8dd842e3.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/fffe3b8f4c893945aeaf337f0c074e91022e61ca7fffa6d3cbfdb6cf8dd842e3.jpg)

## Mitigating Gradient Pathology in PINNs through Aligned Constraint


### Images

![13458a0a7261a76e591081f0dcba5882ba7c655fc76bc7112d8c6c89b2337435.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/images/13458a0a7261a76e591081f0dcba5882ba7c655fc76bc7112d8c6c89b2337435.jpg)

![1e5ed733946fff8270ba821d9563ae1334513f7282b608a8161263286919983a.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/images/1e5ed733946fff8270ba821d9563ae1334513f7282b608a8161263286919983a.jpg)

![245971248c1d2243a6be4d2cd38230d2b6c4f43300e09d97f9a8f4e0c5816ea0.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/images/245971248c1d2243a6be4d2cd38230d2b6c4f43300e09d97f9a8f4e0c5816ea0.jpg)

![2fba4fa6e3fc82ad06783fbdc3c03d65e2ea6c6021ba63fece09d063796a3683.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/images/2fba4fa6e3fc82ad06783fbdc3c03d65e2ea6c6021ba63fece09d063796a3683.jpg)

![3defe255b314747d0fffbc42507cc8de40c4295e87bf0149eeeea319d6b974c7.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/images/3defe255b314747d0fffbc42507cc8de40c4295e87bf0149eeeea319d6b974c7.jpg)

![5989ee31ccab1b6b7f61f60a5e20b5ae60804f682bb55696fcadd59f1acd6f68.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/images/5989ee31ccab1b6b7f61f60a5e20b5ae60804f682bb55696fcadd59f1acd6f68.jpg)

![89d1f5f71ad9a8ad5a6db29805410794bd8e854dbc79216de5e14c3a0c2d2595.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/images/89d1f5f71ad9a8ad5a6db29805410794bd8e854dbc79216de5e14c3a0c2d2595.jpg)

![9d6e4d3f08281599abe15bd21dedecbea796787ca48b8a1203d71e7108225d94.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/images/9d6e4d3f08281599abe15bd21dedecbea796787ca48b8a1203d71e7108225d94.jpg)

![c39a18ae6cf7340194a328ba6fafe04c1900cdd2d9ca93ed3db7f55ce6a2493e.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/images/c39a18ae6cf7340194a328ba6fafe04c1900cdd2d9ca93ed3db7f55ce6a2493e.jpg)

![c7b3ae8f1bbca8509637854bb79d7a53bdc4602e85fd4827d3a1f3c6ae547d19.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/images/c7b3ae8f1bbca8509637854bb79d7a53bdc4602e85fd4827d3a1f3c6ae547d19.jpg)

![e44f2662aae1e4705d4d26f3a89109869c8bdeb8d478797b1cb3bd23dd3d9ff8.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/images/e44f2662aae1e4705d4d26f3a89109869c8bdeb8d478797b1cb3bd23dd3d9ff8.jpg)

![f47daba322d831e1462c3e09a455ae25143455ac4729dab0b289517303c31de8.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/images/f47daba322d831e1462c3e09a455ae25143455ac4729dab0b289517303c31de8.jpg)

![fba61360676bbcf07f566612b8a3bfcdc86287bdda8a4da327e85a8301754716.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/images/fba61360676bbcf07f566612b8a3bfcdc86287bdda8a4da327e85a8301754716.jpg)

### Tables

![07a9df11aed5cdac59d93ea4b07e2889dbed15fcd7f8118698adb5e10730024a.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/07a9df11aed5cdac59d93ea4b07e2889dbed15fcd7f8118698adb5e10730024a.jpg)

![07e4f9abb3b29521ce2d97b45415ac8c6a7e5f99c94d9c8175b0e74d94e20976.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/07e4f9abb3b29521ce2d97b45415ac8c6a7e5f99c94d9c8175b0e74d94e20976.jpg)

![0e39c19785122130f93c245034bd71441a95a52d48cecb328af6cfe8e1e9ebec.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/0e39c19785122130f93c245034bd71441a95a52d48cecb328af6cfe8e1e9ebec.jpg)

![1a11e55112544a1b510b69a845fa60e87bc8303c07b53a689f659338573fb36e.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/1a11e55112544a1b510b69a845fa60e87bc8303c07b53a689f659338573fb36e.jpg)

![1c592fb627b7e91f9ab7b85918364961d13e8f4a95d55aadde7b85349ee2a6f7.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/1c592fb627b7e91f9ab7b85918364961d13e8f4a95d55aadde7b85349ee2a6f7.jpg)

![3380eeb7848b36115ce3426be8b90fc1b0ececfb4203264572d86d3abb95681e.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/3380eeb7848b36115ce3426be8b90fc1b0ececfb4203264572d86d3abb95681e.jpg)

![39f54b3302167bc15dad40f5fc049d7e109b0a82f00edf2ea6cdd407ac4ee909.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/39f54b3302167bc15dad40f5fc049d7e109b0a82f00edf2ea6cdd407ac4ee909.jpg)

![4a8a4ccd0a08f6c9d9f666ac204c02efc624774f0880e5556a267b9dc5d0a79a.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/4a8a4ccd0a08f6c9d9f666ac204c02efc624774f0880e5556a267b9dc5d0a79a.jpg)

![797153156a60889822690d733df44a95eaad2c0210e292a2d8647f439fc40cbe.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/797153156a60889822690d733df44a95eaad2c0210e292a2d8647f439fc40cbe.jpg)

![916311a35360adf9615307503e489158273c2c5e24273c5ce51b80df35d84e49.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/916311a35360adf9615307503e489158273c2c5e24273c5ce51b80df35d84e49.jpg)

![a874bb5cd925b07a5ecd30f98187d3242f802057ce7f39a26136d872ec134150.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/a874bb5cd925b07a5ecd30f98187d3242f802057ce7f39a26136d872ec134150.jpg)

![a876802a12e5dc14ef07cbf560c42aef057d6985b693ec26d28413b2de8afff3.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/a876802a12e5dc14ef07cbf560c42aef057d6985b693ec26d28413b2de8afff3.jpg)

![b61cf92fbbf03ff3be3309ad94e602e3c8dc284b9a446a919e33784eaf7b34be.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/b61cf92fbbf03ff3be3309ad94e602e3c8dc284b9a446a919e33784eaf7b34be.jpg)

![bfc65efaf0007b217b723c4b1a33bbed5c6cadbaaa9137ce0cf86b113e5e41c1.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/bfc65efaf0007b217b723c4b1a33bbed5c6cadbaaa9137ce0cf86b113e5e41c1.jpg)

![c45942234918373013ba0952ae0ea2f131034db68432fc89a14371dfa1fe7636.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/c45942234918373013ba0952ae0ea2f131034db68432fc89a14371dfa1fe7636.jpg)

![c8a20a2c13800549cfc78080d2ab156c8b753ef5b7b52b085ae05c8eadaa866e.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/c8a20a2c13800549cfc78080d2ab156c8b753ef5b7b52b085ae05c8eadaa866e.jpg)

![db69a23cc5556f9a7f285a6c161a4f9f4d05843a5689a1a58f6e3f6861024ddc.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/db69a23cc5556f9a7f285a6c161a4f9f4d05843a5689a1a58f6e3f6861024ddc.jpg)

![e5820f523e7e8d4f86d74ab0f0854e87452fe51213d4943b9ac3cc1ccf5a77e8.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/e5820f523e7e8d4f86d74ab0f0854e87452fe51213d4943b9ac3cc1ccf5a77e8.jpg)

![f45166aba6cb7d6a96b50d73171d4c04f543d040d9e02883306b4bbe0205a749.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/f45166aba6cb7d6a96b50d73171d4c04f543d040d9e02883306b4bbe0205a749.jpg)

![fccd0bd2623e8f9edb05aac7a6f281057f25f79d998ff4cdf8bb83cdf103f0e3.jpg](../icml_results/3340_Mitigating%20Gradient%20Pathology%20in%20PINNs%20through%20Aligned%20Constraint/tables/fccd0bd2623e8f9edb05aac7a6f281057f25f79d998ff4cdf8bb83cdf103f0e3.jpg)

<a id="paper-3302"></a>

## Variance Driven Exploration: A Provable and Efficient Methodology for Pure Exploration in Highly Stochastic Environments


### Images

![0954925a3ae7260084da25363297728c97382b9e93559e0d51bb6dd52083e475.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/images/0954925a3ae7260084da25363297728c97382b9e93559e0d51bb6dd52083e475.jpg)

![38844f6ba3a762a19c2293210491ce395114f54c33ac3db72ff73a12ba803bdd.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/images/38844f6ba3a762a19c2293210491ce395114f54c33ac3db72ff73a12ba803bdd.jpg)

![49ce1900ba7e009a4753d0dcd304fbedb3ce978d735cd4e5db57de523a886337.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/images/49ce1900ba7e009a4753d0dcd304fbedb3ce978d735cd4e5db57de523a886337.jpg)

![55c65d622d2d5fa6f54574077ee81006fbfd011c5d1965ccabd0259aed13d6ed.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/images/55c65d622d2d5fa6f54574077ee81006fbfd011c5d1965ccabd0259aed13d6ed.jpg)

![83c972e1a0123ef6a8aa0cfc63fe92c8665f617d98fbbae20250a9a88a20d4f9.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/images/83c972e1a0123ef6a8aa0cfc63fe92c8665f617d98fbbae20250a9a88a20d4f9.jpg)

![a52a58a1396f48169b7d99e3d396969f70b3d0a2f14ed169f4dc1c1056a1230b.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/images/a52a58a1396f48169b7d99e3d396969f70b3d0a2f14ed169f4dc1c1056a1230b.jpg)

![a5cd166f6cf558c09d656d6e0b3d58d7952c0a70d1cd94f76940615ec6ceeda0.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/images/a5cd166f6cf558c09d656d6e0b3d58d7952c0a70d1cd94f76940615ec6ceeda0.jpg)

![c83b218975232e9afce8b7f91b842f5701d5809ebc0e9b7da968f913b78b82de.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/images/c83b218975232e9afce8b7f91b842f5701d5809ebc0e9b7da968f913b78b82de.jpg)

![d648d1632da5cad56c3a1cf0efff177c5957f72cc1870a9c996068521c5aae7f.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/images/d648d1632da5cad56c3a1cf0efff177c5957f72cc1870a9c996068521c5aae7f.jpg)

![edb75b5c2253bd3efa108d4223d344ddd27b0545e0c0e2425df9bbc527c6e0d4.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/images/edb75b5c2253bd3efa108d4223d344ddd27b0545e0c0e2425df9bbc527c6e0d4.jpg)

### Tables

![15724e02970b84950954b9737e80db78bd17dd508772a4c0034e95078b659c0b.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/tables/15724e02970b84950954b9737e80db78bd17dd508772a4c0034e95078b659c0b.jpg)

![2fbfb94c7f911299f45f85749429dbe1f778e2b687af5021ed0c1fa7056d12d6.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/tables/2fbfb94c7f911299f45f85749429dbe1f778e2b687af5021ed0c1fa7056d12d6.jpg)

![43a882a9d78011b24247224038b96cde0f166f4755871261bbee76572fa67676.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/tables/43a882a9d78011b24247224038b96cde0f166f4755871261bbee76572fa67676.jpg)

![4e90a18f389e4c5baf811fbacbed3a693d295080d5a4f733a335a595dc06bb4b.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/tables/4e90a18f389e4c5baf811fbacbed3a693d295080d5a4f733a335a595dc06bb4b.jpg)

![5cdb05427ed1cbcc330346732b9e5f8cf6670a9f38c64ac2f9cde1d25309034b.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/tables/5cdb05427ed1cbcc330346732b9e5f8cf6670a9f38c64ac2f9cde1d25309034b.jpg)

![6ae4f4b2eca308797387d32d1e3dfa76db55fef3d231ce47e48e214c6fb26395.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/tables/6ae4f4b2eca308797387d32d1e3dfa76db55fef3d231ce47e48e214c6fb26395.jpg)

![c746d4065628f6fa06aecb8217b1213894a30aa8de0a410e95e06c85a87c7c49.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/tables/c746d4065628f6fa06aecb8217b1213894a30aa8de0a410e95e06c85a87c7c49.jpg)

![d9b466ce7c1745de0a9a5a0b5f26a78c007edbb115db7d77051542ffdf15e02c.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/tables/d9b466ce7c1745de0a9a5a0b5f26a78c007edbb115db7d77051542ffdf15e02c.jpg)

![e53ced42f055fcec2371f65f091e18cebf6b04f45691d65e13125c9a64bd664c.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/tables/e53ced42f055fcec2371f65f091e18cebf6b04f45691d65e13125c9a64bd664c.jpg)

![f5461ee10dd8ca804e32e6ffd46ed37d62a5cf678f1f03fb86a1a439b116e75e.jpg](../icml_results/3341_Variance%20Driven%20Exploration_%20A%20Provable%20and%20Efficient%20Methodology%20for%20Pure%20Exploration%20in%20Highly%20Sto/tables/f5461ee10dd8ca804e32e6ffd46ed37d62a5cf678f1f03fb86a1a439b116e75e.jpg)

<a id="paper-3303"></a>

## The Decrypto Benchmark for Multi-Agent Reasoning and Theory of Mind


### Images

![04facf853f8317f472f96d8cac00f40462125c9ac7c2555d345d70bd9bc9ba06.jpg](../icml_results/3342_The%20Decrypto%20Benchmark%20for%20Multi-Agent%20Reasoning%20and%20Theory%20of%20Mind/images/04facf853f8317f472f96d8cac00f40462125c9ac7c2555d345d70bd9bc9ba06.jpg)

![07df23f96edcaa6ec705db677fd6090e893c21c99c8c9701f83cbb47c6b47f4b.jpg](../icml_results/3342_The%20Decrypto%20Benchmark%20for%20Multi-Agent%20Reasoning%20and%20Theory%20of%20Mind/images/07df23f96edcaa6ec705db677fd6090e893c21c99c8c9701f83cbb47c6b47f4b.jpg)

![0d8af1a67fea7a68b43babded16efe6c3819afa936e66fd59635eed649a4a329.jpg](../icml_results/3342_The%20Decrypto%20Benchmark%20for%20Multi-Agent%20Reasoning%20and%20Theory%20of%20Mind/images/0d8af1a67fea7a68b43babded16efe6c3819afa936e66fd59635eed649a4a329.jpg)

![1e77294a121d61e6af1c753866c766301b0c66daad10c59901b6379671fff062.jpg](../icml_results/3342_The%20Decrypto%20Benchmark%20for%20Multi-Agent%20Reasoning%20and%20Theory%20of%20Mind/images/1e77294a121d61e6af1c753866c766301b0c66daad10c59901b6379671fff062.jpg)

![23c0be1dee7486d713e3384138014d10594cb30c179428e8b4ff71ea55c8e740.jpg](../icml_results/3342_The%20Decrypto%20Benchmark%20for%20Multi-Agent%20Reasoning%20and%20Theory%20of%20Mind/images/23c0be1dee7486d713e3384138014d10594cb30c179428e8b4ff71ea55c8e740.jpg)

![2a8e81e2c08666839d1e45fe054e93d6ac62977962a30cc3eddf48b6d8f568bf.jpg](../icml_results/3342_The%20Decrypto%20Benchmark%20for%20Multi-Agent%20Reasoning%20and%20Theory%20of%20Mind/images/2a8e81e2c08666839d1e45fe054e93d6ac62977962a30cc3eddf48b6d8f568bf.jpg)

![35004168c945362bdd5e5c0573adbe37b820307403e7dd83d55420a982458d04.jpg](../icml_results/3342_The%20Decrypto%20Benchmark%20for%20Multi-Agent%20Reasoning%20and%20Theory%20of%20Mind/images/35004168c945362bdd5e5c0573adbe37b820307403e7dd83d55420a982458d04.jpg)

![3559a5f54f128cf224c1e17a4f892ff944e778b1418ec694698e6ffa3b430f68.jpg](../icml_results/3342_The%20Decrypto%20Benchmark%20for%20Multi-Agent%20Reasoning%20and%20Theory%20of%20Mind/images/3559a5f54f128cf224c1e17a4f892ff944e778b1418ec694698e6ffa3b430f68.jpg)

![5dd362a1f0f125a4fcb7942dcd56680cb5858255ff86a47f2e5773875fbdef25.jpg](../icml_results/3342_The%20Decrypto%20Benchmark%20for%20Multi-Agent%20Reasoning%20and%20Theory%20of%20Mind/images/5dd362a1f0f125a4fcb7942dcd56680cb5858255ff86a47f2e5773875fbdef25.jpg)

![83d33c520c5af5d132706b0e15035c9866fef4ab83ed2c4d5c37ea9aa2f7af11.jpg](../icml_results/3342_The%20Decrypto%20Benchmark%20for%20Multi-Agent%20Reasoning%20and%20Theory%20of%20Mind/images/83d33c520c5af5d132706b0e15035c9866fef4ab83ed2c4d5c37ea9aa2f7af11.jpg)

![d17bbd6428e382956eeddc72c715bb8ba7d4180c05fd4f953a00989c073ceeaf.jpg](../icml_results/3342_The%20Decrypto%20Benchmark%20for%20Multi-Agent%20Reasoning%20and%20Theory%20of%20Mind/images/d17bbd6428e382956eeddc72c715bb8ba7d4180c05fd4f953a00989c073ceeaf.jpg)

![fab57e1dbbbc8a0b86a947ea4a6cb8510a54b443bbee3cab97e0f2230ef828d2.jpg](../icml_results/3342_The%20Decrypto%20Benchmark%20for%20Multi-Agent%20Reasoning%20and%20Theory%20of%20Mind/images/fab57e1dbbbc8a0b86a947ea4a6cb8510a54b443bbee3cab97e0f2230ef828d2.jpg)

![fed0afe445a56e9da08a2218c9c3bca6def93532940c3652b9dc5b5ff435a63c.jpg](../icml_results/3342_The%20Decrypto%20Benchmark%20for%20Multi-Agent%20Reasoning%20and%20Theory%20of%20Mind/images/fed0afe445a56e9da08a2218c9c3bca6def93532940c3652b9dc5b5ff435a63c.jpg)

### Tables

![0c440bdcf9681452f146b3a0520fee0842e7a54b610ca28c0a4a6f45293af549.jpg](../icml_results/3342_The%20Decrypto%20Benchmark%20for%20Multi-Agent%20Reasoning%20and%20Theory%20of%20Mind/tables/0c440bdcf9681452f146b3a0520fee0842e7a54b610ca28c0a4a6f45293af549.jpg)

![d3e423df04fa1690bd5443b09a723e78b0372a98c82068eb3a821efd54bcefc6.jpg](../icml_results/3342_The%20Decrypto%20Benchmark%20for%20Multi-Agent%20Reasoning%20and%20Theory%20of%20Mind/tables/d3e423df04fa1690bd5443b09a723e78b0372a98c82068eb3a821efd54bcefc6.jpg)

<a id="paper-3304"></a>

## Directional Neural Collapse Explains Few-Shot Transfer in Self-Supervised Learning


### Images

![0d926838ae69a3b75425a27a22cdf70671e7e93e61a1361f0e7003c4a23ae587.jpg](../icml_results/3343_Directional%20Neural%20Collapse%20Explains%20Few-Shot%20Transfer%20in%20Self-Supervised%20Learning/images/0d926838ae69a3b75425a27a22cdf70671e7e93e61a1361f0e7003c4a23ae587.jpg)

![24bcbd83d15dee06ef3d0cb70424a2de1fcb8862b2bebf088f99011625661622.jpg](../icml_results/3343_Directional%20Neural%20Collapse%20Explains%20Few-Shot%20Transfer%20in%20Self-Supervised%20Learning/images/24bcbd83d15dee06ef3d0cb70424a2de1fcb8862b2bebf088f99011625661622.jpg)

![263dc5df0accbc4f0d238d93c92759509599b402a23a83e257c881997bf2c3e8.jpg](../icml_results/3343_Directional%20Neural%20Collapse%20Explains%20Few-Shot%20Transfer%20in%20Self-Supervised%20Learning/images/263dc5df0accbc4f0d238d93c92759509599b402a23a83e257c881997bf2c3e8.jpg)

![3eeb83476d6aa70e3e49c87ab8e31cbe80b5fe201a609db660a1cb78a506bffc.jpg](../icml_results/3343_Directional%20Neural%20Collapse%20Explains%20Few-Shot%20Transfer%20in%20Self-Supervised%20Learning/images/3eeb83476d6aa70e3e49c87ab8e31cbe80b5fe201a609db660a1cb78a506bffc.jpg)

![5318fe88924eaa0d915f9df6ae887a393c5bf28ff54ea55644be90dec9efee95.jpg](../icml_results/3343_Directional%20Neural%20Collapse%20Explains%20Few-Shot%20Transfer%20in%20Self-Supervised%20Learning/images/5318fe88924eaa0d915f9df6ae887a393c5bf28ff54ea55644be90dec9efee95.jpg)

![6de9d7663dbb8536e0e6e6cab7142dbe2c0e89a7b41d18aa00b7ad874384baad.jpg](../icml_results/3343_Directional%20Neural%20Collapse%20Explains%20Few-Shot%20Transfer%20in%20Self-Supervised%20Learning/images/6de9d7663dbb8536e0e6e6cab7142dbe2c0e89a7b41d18aa00b7ad874384baad.jpg)

![75a8dc215809db8d9f39a217ea0abfc5288789b4cac015de41e85a44af25c538.jpg](../icml_results/3343_Directional%20Neural%20Collapse%20Explains%20Few-Shot%20Transfer%20in%20Self-Supervised%20Learning/images/75a8dc215809db8d9f39a217ea0abfc5288789b4cac015de41e85a44af25c538.jpg)

![c43bc57610497c8101fb7d9b067fd0a4f8b8464d075093befb8bc47a8bf76779.jpg](../icml_results/3343_Directional%20Neural%20Collapse%20Explains%20Few-Shot%20Transfer%20in%20Self-Supervised%20Learning/images/c43bc57610497c8101fb7d9b067fd0a4f8b8464d075093befb8bc47a8bf76779.jpg)

![cc90ca97553ac97bcfe9c481fcac78e21de632e280469e635eaef3a19e35e935.jpg](../icml_results/3343_Directional%20Neural%20Collapse%20Explains%20Few-Shot%20Transfer%20in%20Self-Supervised%20Learning/images/cc90ca97553ac97bcfe9c481fcac78e21de632e280469e635eaef3a19e35e935.jpg)

![e546d23edd20d8bbf3b608a2ec1163ea9178a98524b8f74c073fb5615ce47271.jpg](../icml_results/3343_Directional%20Neural%20Collapse%20Explains%20Few-Shot%20Transfer%20in%20Self-Supervised%20Learning/images/e546d23edd20d8bbf3b608a2ec1163ea9178a98524b8f74c073fb5615ce47271.jpg)

<a id="paper-3305"></a>

## DC-LA: Difference-of-Convex Langevin Algorithm


### Images

![03277d477331b184207340d04ae4f06326ea4e4fa24792654ce73d7219956a76.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/03277d477331b184207340d04ae4f06326ea4e4fa24792654ce73d7219956a76.jpg)

![099792582b67deb5ed6e9aaa4af6da5d9fd76552a6f09963aad73ef23b6273c7.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/099792582b67deb5ed6e9aaa4af6da5d9fd76552a6f09963aad73ef23b6273c7.jpg)

![0cfc6813727f9af7dd7a26b832380317d622c393b6bb43bf0ec329f9b23a44d8.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/0cfc6813727f9af7dd7a26b832380317d622c393b6bb43bf0ec329f9b23a44d8.jpg)

![0dc32669999d2ac60a7316ba6f1a3fba87326ee271ac3924282d0869f5dcc4c8.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/0dc32669999d2ac60a7316ba6f1a3fba87326ee271ac3924282d0869f5dcc4c8.jpg)

![1074ea664d2dd1ee53328e57455fb696bb4fc3847868e181366572f76d8bd627.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/1074ea664d2dd1ee53328e57455fb696bb4fc3847868e181366572f76d8bd627.jpg)

![196526d2e05663503f593fa15518e4ccbe0c04be16b22f40781f85d21c27c950.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/196526d2e05663503f593fa15518e4ccbe0c04be16b22f40781f85d21c27c950.jpg)

![2265424b9617dc622cd8930a267d0030e9ea024f1d1066da814a0600b4fa2b07.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/2265424b9617dc622cd8930a267d0030e9ea024f1d1066da814a0600b4fa2b07.jpg)

![28ad304baa5edf096a123666575abe7b1510f34e4b97cd6a3c433d8987be0e53.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/28ad304baa5edf096a123666575abe7b1510f34e4b97cd6a3c433d8987be0e53.jpg)

![2929decbb8b31b59d006612b5f67c6cb7206bd0ca96786b3e16d17d55e529fcb.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/2929decbb8b31b59d006612b5f67c6cb7206bd0ca96786b3e16d17d55e529fcb.jpg)

![3387544ab19d5448c2da9a8f731bfa70344c7b7d883b0f6a8befec43b125397c.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/3387544ab19d5448c2da9a8f731bfa70344c7b7d883b0f6a8befec43b125397c.jpg)

![36e6f70f25af71a906a10cfc3b20004459d94356b7c1dc5b62b6cb447bfeee7a.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/36e6f70f25af71a906a10cfc3b20004459d94356b7c1dc5b62b6cb447bfeee7a.jpg)

![5c7fac20cbe08437e1e887cb4b7cee1c6db1d0e72ef2627032750667f7b11923.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/5c7fac20cbe08437e1e887cb4b7cee1c6db1d0e72ef2627032750667f7b11923.jpg)

![60ffbf9d07cae0f883e7b7a5f4022f08b93a2875c069466ceb11fcd5bd478034.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/60ffbf9d07cae0f883e7b7a5f4022f08b93a2875c069466ceb11fcd5bd478034.jpg)

![64dcd2d2b03f8dcb89b709b525a73d879694405a94f998e15486f50e72c022f3.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/64dcd2d2b03f8dcb89b709b525a73d879694405a94f998e15486f50e72c022f3.jpg)

![688a9c34b8651db9a2885f41a11d85a72945004b76b150ce676c14aa34827626.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/688a9c34b8651db9a2885f41a11d85a72945004b76b150ce676c14aa34827626.jpg)

![6cd500af9c10df9cb5a685476b9c32509c568ac22a95387483b27a2ea669334b.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/6cd500af9c10df9cb5a685476b9c32509c568ac22a95387483b27a2ea669334b.jpg)

![6f6b76e2effbb06f33f1f3be806edcf713cf5c271cfa1ee82dbea12d88af7b16.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/6f6b76e2effbb06f33f1f3be806edcf713cf5c271cfa1ee82dbea12d88af7b16.jpg)

![714cf5a6ea0e07de55a2c5ea40e8f4fa867cf0e54d5e7e55b693f998417b8c9d.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/714cf5a6ea0e07de55a2c5ea40e8f4fa867cf0e54d5e7e55b693f998417b8c9d.jpg)

![7709a6363442557cd215bd61b3c70a8199dfdb264298b0c70681e9890aef19c3.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/7709a6363442557cd215bd61b3c70a8199dfdb264298b0c70681e9890aef19c3.jpg)

![77119a597341a577d94f7be8966804b083314ce81aad936aae44e42098d12fa8.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/77119a597341a577d94f7be8966804b083314ce81aad936aae44e42098d12fa8.jpg)

![77a6fc15bbb690371d4c6788e88da2fd89f67087eb492d3997cb92da9855521d.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/77a6fc15bbb690371d4c6788e88da2fd89f67087eb492d3997cb92da9855521d.jpg)

![7ade100512deb22084c7b31aa0f9a4193bf2ff831cdb442871145a2045bc3952.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/7ade100512deb22084c7b31aa0f9a4193bf2ff831cdb442871145a2045bc3952.jpg)

![7edf745d0be9852ec375deff8ce7cbe712fdf862dd0f0afa67a07c9c08295222.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/7edf745d0be9852ec375deff8ce7cbe712fdf862dd0f0afa67a07c9c08295222.jpg)

![9d04d4fcb9d18800891522d272285a763db0d81013d6c47b13757d940a9c2b17.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/9d04d4fcb9d18800891522d272285a763db0d81013d6c47b13757d940a9c2b17.jpg)

![c32b546f7a47ca2b2f133ef279e0e1ec4d2900420e1e4d0fa61e204d614d9245.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/c32b546f7a47ca2b2f133ef279e0e1ec4d2900420e1e4d0fa61e204d614d9245.jpg)

![d0a5279cb2a27d3aa8ccf1184452134da1e135fe6bf11130da0c36006a8e137b.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/d0a5279cb2a27d3aa8ccf1184452134da1e135fe6bf11130da0c36006a8e137b.jpg)

![d117cad22034cc25a1bef0232500ce732059fcaffd2ee0230f57227916f089bc.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/d117cad22034cc25a1bef0232500ce732059fcaffd2ee0230f57227916f089bc.jpg)

![d44b0534e79d70b257eeb6c98103c375d2bbcd373934445419d4310aaaf2e1b0.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/d44b0534e79d70b257eeb6c98103c375d2bbcd373934445419d4310aaaf2e1b0.jpg)

![db4d0354902a28c8fda42628b8d777033f171c51c99d921133ab3a5e6f4d994f.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/db4d0354902a28c8fda42628b8d777033f171c51c99d921133ab3a5e6f4d994f.jpg)

![e59873c51b61e8b45c55c38164ab6f4cd235636cc0729c2d4a90ce5e2badd439.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/e59873c51b61e8b45c55c38164ab6f4cd235636cc0729c2d4a90ce5e2badd439.jpg)

![f7c69d1ac2bfd685507c37716f7e74b270da32ecdf520b7ba51839198b58cbcf.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/images/f7c69d1ac2bfd685507c37716f7e74b270da32ecdf520b7ba51839198b58cbcf.jpg)

### Tables

![5de59ea96d4dbb8e2648188add17dc4579f3a83a07ccacafb7d0629103d7db47.jpg](../icml_results/3344_DC-LA_%20Difference-of-Convex%20Langevin%20Algorithm/tables/5de59ea96d4dbb8e2648188add17dc4579f3a83a07ccacafb7d0629103d7db47.jpg)

<a id="paper-3306"></a>

## Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models


### Images

![08f75f454a7d11138dfff128f10df52161a3165018ba22daadd50d0be70ccc81.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/08f75f454a7d11138dfff128f10df52161a3165018ba22daadd50d0be70ccc81.jpg)

![382c4f563590e1cbbf32a01d2829a32bdbef6972f1932f46d8fb079167c81f39.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/382c4f563590e1cbbf32a01d2829a32bdbef6972f1932f46d8fb079167c81f39.jpg)

![3c0c1f000341c59c3957ca59254238e194bac39ac77089d508b0d586a20a5158.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/3c0c1f000341c59c3957ca59254238e194bac39ac77089d508b0d586a20a5158.jpg)

![4c5185e119dcd1956426fc2e34df531d085239abf3e336491e6bf0baaca0adea.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/4c5185e119dcd1956426fc2e34df531d085239abf3e336491e6bf0baaca0adea.jpg)

![61f0148c256f5b2563b822e1b0bfd932136bd2217c64764f8a8a4f36ef171b97.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/61f0148c256f5b2563b822e1b0bfd932136bd2217c64764f8a8a4f36ef171b97.jpg)

![64500c7b67e8ebcf7f973be028a37f259018068d86c497008f007f4213f4759d.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/64500c7b67e8ebcf7f973be028a37f259018068d86c497008f007f4213f4759d.jpg)

![7ccbfe2658b1a1d627117ecc3358c4ca2b658d2390a3e1e3de22190785fafc12.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/7ccbfe2658b1a1d627117ecc3358c4ca2b658d2390a3e1e3de22190785fafc12.jpg)

![8b2654f69a2527775c0a533c68a0479f3887f782b2eb419d8bdf4c41bdeae9e4.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/8b2654f69a2527775c0a533c68a0479f3887f782b2eb419d8bdf4c41bdeae9e4.jpg)

![9f77db0a77ccf273a6ccbd863e2f52793ae1154d587df22d671527621894ea8b.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/9f77db0a77ccf273a6ccbd863e2f52793ae1154d587df22d671527621894ea8b.jpg)

![ab49832d58142d05c445a47924e737aa46ad0fd63bb6acee08f69472f8049b52.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/ab49832d58142d05c445a47924e737aa46ad0fd63bb6acee08f69472f8049b52.jpg)

![ac4f887f206ffddb6dd83f10ea031df107711e0a5e7c147cf614ee057b343137.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/ac4f887f206ffddb6dd83f10ea031df107711e0a5e7c147cf614ee057b343137.jpg)

![ad402799b29aaf6e72b0798240a6a5c819f15f67b9d68d7776837c2701de896d.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/ad402799b29aaf6e72b0798240a6a5c819f15f67b9d68d7776837c2701de896d.jpg)

![bedc67e2a564fdcb48953524c374f5a360d54c351999ea9d814d170769d15ddf.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/bedc67e2a564fdcb48953524c374f5a360d54c351999ea9d814d170769d15ddf.jpg)

![c77c36a3f4db1f959ec8e93e1c887e561181c95aa9939715f57986d2ab54b437.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/c77c36a3f4db1f959ec8e93e1c887e561181c95aa9939715f57986d2ab54b437.jpg)

![ca1fe2f8a5129a3625d91f612bc01e33cbf12bedce7be3cb345533e023d42f5a.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/ca1fe2f8a5129a3625d91f612bc01e33cbf12bedce7be3cb345533e023d42f5a.jpg)

![d3df64b34930a4ce83dc49a2e401e69a35b815d3433ee52ae9b5f9cef8a7247c.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/d3df64b34930a4ce83dc49a2e401e69a35b815d3433ee52ae9b5f9cef8a7247c.jpg)

![da9c2466ee1274c9bafa499b5d2af13668455f9d1e7809f19f420777cf91dfdd.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/da9c2466ee1274c9bafa499b5d2af13668455f9d1e7809f19f420777cf91dfdd.jpg)

![ed28cd3a1657c019da68dc97eea702d31174091f6fc4c80d29c2a22679b8c9aa.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/ed28cd3a1657c019da68dc97eea702d31174091f6fc4c80d29c2a22679b8c9aa.jpg)

![f2362b68f181d83d9cd588e945097c3e861ea4da53d4b9ebd1df71ea5b7b4f1a.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/f2362b68f181d83d9cd588e945097c3e861ea4da53d4b9ebd1df71ea5b7b4f1a.jpg)

![fafbf402a4929f01dfbc4757a614ace6c7c52573c9669f8af95c39a8eec648d4.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/images/fafbf402a4929f01dfbc4757a614ace6c7c52573c9669f8af95c39a8eec648d4.jpg)

### Tables

![038b1786ea15c0b05f4ec5f15c365277f985ba9e52f0e0ad26dfee5da7b5a938.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/038b1786ea15c0b05f4ec5f15c365277f985ba9e52f0e0ad26dfee5da7b5a938.jpg)

![19d8172375b99be188b4e4276c33d484ebd46e05381c32a9dd9bdef6cd8d3567.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/19d8172375b99be188b4e4276c33d484ebd46e05381c32a9dd9bdef6cd8d3567.jpg)

![1f3e961c19a4373b2947c5f7ecd9eed6ce5f0762dd69e6fac9173064352cdfec.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/1f3e961c19a4373b2947c5f7ecd9eed6ce5f0762dd69e6fac9173064352cdfec.jpg)

![2c51e0c9a305dc3c7cf0ca8c4bc7240b643f6aab6ad80b9e33f362f27e873a23.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/2c51e0c9a305dc3c7cf0ca8c4bc7240b643f6aab6ad80b9e33f362f27e873a23.jpg)

![567f285e34b1964d470f3ebe3036676d8b35743db3fa41742e24345252672b25.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/567f285e34b1964d470f3ebe3036676d8b35743db3fa41742e24345252672b25.jpg)

![570b365574d3ff4000676fcff724262bfc813ecfa1f622da34fdb5053dbf91a9.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/570b365574d3ff4000676fcff724262bfc813ecfa1f622da34fdb5053dbf91a9.jpg)

![6c432c35f2c4be506932a0839e7ac5f0ba05720a35c23af587dc767799bfcb40.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/6c432c35f2c4be506932a0839e7ac5f0ba05720a35c23af587dc767799bfcb40.jpg)

![70672231a70a60e5a0d170af5eb036eb7b56920b8d5c3e849a45e4c93487a65c.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/70672231a70a60e5a0d170af5eb036eb7b56920b8d5c3e849a45e4c93487a65c.jpg)

![7496b0ed62219a97162ea2b89fc11bee1776f2bd752ba6aa2cad732eb49b852d.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/7496b0ed62219a97162ea2b89fc11bee1776f2bd752ba6aa2cad732eb49b852d.jpg)

![79deafd3a302a93ec1de4ba5f78d5cd62e43ef660c355b1b396a13e48fd2fcba.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/79deafd3a302a93ec1de4ba5f78d5cd62e43ef660c355b1b396a13e48fd2fcba.jpg)

![a2209a1bd0d63cdb255112d208b6011f874ffa725eff4c87466605febc5afe33.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/a2209a1bd0d63cdb255112d208b6011f874ffa725eff4c87466605febc5afe33.jpg)

![a972d45785a3f95cc6a008fcf79a3b861509daed17512731f22300f73d30a552.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/a972d45785a3f95cc6a008fcf79a3b861509daed17512731f22300f73d30a552.jpg)

![a9caa863b77935a0e27141b69a079351487709a6e6c208cf2b74f0a144f933d4.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/a9caa863b77935a0e27141b69a079351487709a6e6c208cf2b74f0a144f933d4.jpg)

![ad1a3d545d8f7104998ceeb1db03152d5dfb188a1cfa19c58717fafc5066316c.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/ad1a3d545d8f7104998ceeb1db03152d5dfb188a1cfa19c58717fafc5066316c.jpg)

![ae9ff8f15c65dd86dbbf27089fb2e78864f18acbbf38005fa77a70ea46f6e73e.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/ae9ff8f15c65dd86dbbf27089fb2e78864f18acbbf38005fa77a70ea46f6e73e.jpg)

![c16b27b76b1c76023dde1305e15845173c511d9bc657b30b4230a09ab9a92044.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/c16b27b76b1c76023dde1305e15845173c511d9bc657b30b4230a09ab9a92044.jpg)

![cdf9204f61cd0dfc2e55dc6aa5ba405619f889f192e2fa10ed75a6bfe656ee84.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/cdf9204f61cd0dfc2e55dc6aa5ba405619f889f192e2fa10ed75a6bfe656ee84.jpg)

![dbf2b897b9f16813ca6c9230e735d1f98e2acf2187284e893855a2b802288703.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/dbf2b897b9f16813ca6c9230e735d1f98e2acf2187284e893855a2b802288703.jpg)

![eff70cda590b2cf91ec9fa6eb2e5e91d464840a8aa3d3f934715f9ee68812951.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/eff70cda590b2cf91ec9fa6eb2e5e91d464840a8aa3d3f934715f9ee68812951.jpg)

![f029fa5452ed73414f4e07a7d77c2c5d3163d41df39161125fa2cadb9c9b8f77.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/f029fa5452ed73414f4e07a7d77c2c5d3163d41df39161125fa2cadb9c9b8f77.jpg)

![f2fbc55faecb26a0ff2b18cdae8ba7d128764aa1f309cb624ff3eaaeff87ed4e.jpg](../icml_results/3345_Plan%20for%20Speed_%20Dilated%20Scheduling%20for%20Masked%20Diffusion%20Language%20Models/tables/f2fbc55faecb26a0ff2b18cdae8ba7d128764aa1f309cb624ff3eaaeff87ed4e.jpg)

<a id="paper-3307"></a>

## The Hidden Link between RLHF and Contrastive Learning


### Images

![6beaa4137fcd113b7028c932dce1ad99449d303b47eafe6c72c7f288f20f8ad5.jpg](../icml_results/3346_The%20Hidden%20Link%20between%20RLHF%20and%20Contrastive%20Learning/images/6beaa4137fcd113b7028c932dce1ad99449d303b47eafe6c72c7f288f20f8ad5.jpg)

![6ddbd73886ce13bbafe174e40289b34c151999bddec7c11401aabc6f7bd9cf59.jpg](../icml_results/3346_The%20Hidden%20Link%20between%20RLHF%20and%20Contrastive%20Learning/images/6ddbd73886ce13bbafe174e40289b34c151999bddec7c11401aabc6f7bd9cf59.jpg)

![9d7c7d244868c6d195c0c9d6e7c52c5db08726299ef78270f6344a1131886523.jpg](../icml_results/3346_The%20Hidden%20Link%20between%20RLHF%20and%20Contrastive%20Learning/images/9d7c7d244868c6d195c0c9d6e7c52c5db08726299ef78270f6344a1131886523.jpg)

![aeca0e7d2ceded7c47fbd9c332ce55a423cd58cafadcd76bab0bb13bfd287c0f.jpg](../icml_results/3346_The%20Hidden%20Link%20between%20RLHF%20and%20Contrastive%20Learning/images/aeca0e7d2ceded7c47fbd9c332ce55a423cd58cafadcd76bab0bb13bfd287c0f.jpg)

![c36f9538dbdcea60719895f0587a3397fd45485f42eebe33c83dd22660842d45.jpg](../icml_results/3346_The%20Hidden%20Link%20between%20RLHF%20and%20Contrastive%20Learning/images/c36f9538dbdcea60719895f0587a3397fd45485f42eebe33c83dd22660842d45.jpg)

### Tables

![09b24849350e1741d7c950a0a24dc76f5c0f0a40153d2590e63f9a12cf0a2d5c.jpg](../icml_results/3346_The%20Hidden%20Link%20between%20RLHF%20and%20Contrastive%20Learning/tables/09b24849350e1741d7c950a0a24dc76f5c0f0a40153d2590e63f9a12cf0a2d5c.jpg)

![1a8d6bbffdfaa0d0c5923be5de5019eb1238edcef14420629f070bacaf48dea7.jpg](../icml_results/3346_The%20Hidden%20Link%20between%20RLHF%20and%20Contrastive%20Learning/tables/1a8d6bbffdfaa0d0c5923be5de5019eb1238edcef14420629f070bacaf48dea7.jpg)

![5ffe677da733c02e576eab0e70aa38e4bd0b908bc7e92569630c764e194015fe.jpg](../icml_results/3346_The%20Hidden%20Link%20between%20RLHF%20and%20Contrastive%20Learning/tables/5ffe677da733c02e576eab0e70aa38e4bd0b908bc7e92569630c764e194015fe.jpg)

![7f27f0f444968220357f7ea10bafd62d2a1d99a12576aeed068fecd4e6b62557.jpg](../icml_results/3346_The%20Hidden%20Link%20between%20RLHF%20and%20Contrastive%20Learning/tables/7f27f0f444968220357f7ea10bafd62d2a1d99a12576aeed068fecd4e6b62557.jpg)

![b931e33d8fc574a87048e26525ffa95299f1ba4efa031b75ab08634bcb818909.jpg](../icml_results/3346_The%20Hidden%20Link%20between%20RLHF%20and%20Contrastive%20Learning/tables/b931e33d8fc574a87048e26525ffa95299f1ba4efa031b75ab08634bcb818909.jpg)

![da90034774cbd1ec9effbc93527ceec0435f76c9bb0f9d20c4327e6e38be04c7.jpg](../icml_results/3346_The%20Hidden%20Link%20between%20RLHF%20and%20Contrastive%20Learning/tables/da90034774cbd1ec9effbc93527ceec0435f76c9bb0f9d20c4327e6e38be04c7.jpg)

![e5c2b4c0badcdd4b6388d0cde60512b0dcafc15b15f095a2484691dac9e6c55c.jpg](../icml_results/3346_The%20Hidden%20Link%20between%20RLHF%20and%20Contrastive%20Learning/tables/e5c2b4c0badcdd4b6388d0cde60512b0dcafc15b15f095a2484691dac9e6c55c.jpg)

![f45363cc8fd2a1e7f6da1a79052e1fd1b49a1826893dae7461834542e615a93f.jpg](../icml_results/3346_The%20Hidden%20Link%20between%20RLHF%20and%20Contrastive%20Learning/tables/f45363cc8fd2a1e7f6da1a79052e1fd1b49a1826893dae7461834542e615a93f.jpg)

<a id="paper-3308"></a>

## AtomWorld: A Benchmark for Evaluating Spatial Reasoning in Large Language Models on Material Structures


### Images

![0cca75d58315108effbbc9d93efb2a88d2951bd7ea82b0dbbf70c9a79db96a4d.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/images/0cca75d58315108effbbc9d93efb2a88d2951bd7ea82b0dbbf70c9a79db96a4d.jpg)

![2763df0f2d29b3943e8f607bcb46b8911904b07866184d1441ebdf346149e43a.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/images/2763df0f2d29b3943e8f607bcb46b8911904b07866184d1441ebdf346149e43a.jpg)

![27cd11e48df84d09a6b3a64da8b3057e957193ce8c1fd821856cdee43a28c256.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/images/27cd11e48df84d09a6b3a64da8b3057e957193ce8c1fd821856cdee43a28c256.jpg)

![5224eb26c7ea02f547868de1c24fd1930817769c60c862daa4a30982236d6172.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/images/5224eb26c7ea02f547868de1c24fd1930817769c60c862daa4a30982236d6172.jpg)

![76cdda97a226c1df7cbfe108ac41ab68ee174bf9c6901128058a120d25ce13d7.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/images/76cdda97a226c1df7cbfe108ac41ab68ee174bf9c6901128058a120d25ce13d7.jpg)

![9675aa1218686385c69d3fc1287fcadc8f3cf4fd29bec0504fa24d13e3559d34.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/images/9675aa1218686385c69d3fc1287fcadc8f3cf4fd29bec0504fa24d13e3559d34.jpg)

![99856e88207e7ea6f50c5c64ca8c6b2d41fb371fb5f8025b444054a3b2ab5061.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/images/99856e88207e7ea6f50c5c64ca8c6b2d41fb371fb5f8025b444054a3b2ab5061.jpg)

![9ad598e70ba41fe6d802639deb28f81e3fb067ee76dfd3593f6d03ae0a037257.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/images/9ad598e70ba41fe6d802639deb28f81e3fb067ee76dfd3593f6d03ae0a037257.jpg)

![a6894892870c6595daa5e92606bcba2eeeac2db951157bce0816aa8c38245f4f.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/images/a6894892870c6595daa5e92606bcba2eeeac2db951157bce0816aa8c38245f4f.jpg)

![ab83a3dbce9e2219bd90cc59e4ebde576a1a8b73d64d05b153a0c8c876d97763.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/images/ab83a3dbce9e2219bd90cc59e4ebde576a1a8b73d64d05b153a0c8c876d97763.jpg)

![b9671b491cb5160c5089b111d7cbd5c50edb96af26aa5acf5234012707e25cd6.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/images/b9671b491cb5160c5089b111d7cbd5c50edb96af26aa5acf5234012707e25cd6.jpg)

![c436e1d1314623c8a0df65782fdfceb22f8ce53ddcd7461a02c97b8ad79674e5.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/images/c436e1d1314623c8a0df65782fdfceb22f8ce53ddcd7461a02c97b8ad79674e5.jpg)

![cbd270dcdadead47710fe134a1024a8b1870451466b0f8a4d2d7b73d359725d8.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/images/cbd270dcdadead47710fe134a1024a8b1870451466b0f8a4d2d7b73d359725d8.jpg)

![e1209cfc93e5ffb54c4c9471222bd9d5708c360db4efdeab4bd0bf2c61fead4a.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/images/e1209cfc93e5ffb54c4c9471222bd9d5708c360db4efdeab4bd0bf2c61fead4a.jpg)

![f520adfe60fd21789802286a75de91a71e4b79cc0b2c5e0b00ae4742f26a976b.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/images/f520adfe60fd21789802286a75de91a71e4b79cc0b2c5e0b00ae4742f26a976b.jpg)

![f6027b263f98d8b960dffc6965c8e8c0c00cb64f4bcc360813349addf54511e4.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/images/f6027b263f98d8b960dffc6965c8e8c0c00cb64f4bcc360813349addf54511e4.jpg)

### Tables

![06c46cb1b99629fde868d6efe0ffa9c4b07d3d737c445e210d5a91c1add79d60.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/tables/06c46cb1b99629fde868d6efe0ffa9c4b07d3d737c445e210d5a91c1add79d60.jpg)

![4d91566b0878117f794c6d0ee53be16bec64d7da97a411c367535636e56a21f0.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/tables/4d91566b0878117f794c6d0ee53be16bec64d7da97a411c367535636e56a21f0.jpg)

![7e5cc304d8c38eb7582c3a8a810576ae9c2511350c7fb2bb43359ad9c2d99e6e.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/tables/7e5cc304d8c38eb7582c3a8a810576ae9c2511350c7fb2bb43359ad9c2d99e6e.jpg)

![a4f06f7f1ad4ac7b507a16c0137a66b71db444b26b2a1f277088b2dc40afab83.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/tables/a4f06f7f1ad4ac7b507a16c0137a66b71db444b26b2a1f277088b2dc40afab83.jpg)

![a5af1760f163958960b07cf916170a834792c63db0949d2b477833f886a3e437.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/tables/a5af1760f163958960b07cf916170a834792c63db0949d2b477833f886a3e437.jpg)

![bd0588f40ac92b91e4fa3752a351cd80e6647b420352e57ef7803256985a54d2.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/tables/bd0588f40ac92b91e4fa3752a351cd80e6647b420352e57ef7803256985a54d2.jpg)

![ce867ea608027efc0d1c576d0b81784366847aa4989288582de727bc8db76fba.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/tables/ce867ea608027efc0d1c576d0b81784366847aa4989288582de727bc8db76fba.jpg)

![d00a0f37516bb26e21da08e91e723b36ca918f3f7e1f160be57b1ad48ba01baf.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/tables/d00a0f37516bb26e21da08e91e723b36ca918f3f7e1f160be57b1ad48ba01baf.jpg)

![dbb351f31ea0a8207c9c9d2943a6e724a2a0213339bee9f0a9a491c9a8748f58.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/tables/dbb351f31ea0a8207c9c9d2943a6e724a2a0213339bee9f0a9a491c9a8748f58.jpg)

![ee855a9e5a63c05afd7f18484541f5648db84cdb731fc3d4d299fe24111d179f.jpg](../icml_results/3347_AtomWorld_%20A%20Benchmark%20for%20Evaluating%20Spatial%20Reasoning%20in%20Large%20Language%20Models%20on%20Material%20Structu/tables/ee855a9e5a63c05afd7f18484541f5648db84cdb731fc3d4d299fe24111d179f.jpg)

<a id="paper-3309"></a>

## SCalDA: Semantics-Calibrated and Diffusion-Enhanced Data Augmentation


### Images

![5c74747229280c989ad9f44deab4f3a8048b05eba84722771b24ed36b36a0aeb.jpg](../icml_results/3348_SCalDA_%20Semantics-Calibrated%20and%20Diffusion-Enhanced%20Data%20Augmentation/images/5c74747229280c989ad9f44deab4f3a8048b05eba84722771b24ed36b36a0aeb.jpg)

![89287ce120b3d6fb63e4fada4c837889931144b3364c60604e3a3829e2806fc9.jpg](../icml_results/3348_SCalDA_%20Semantics-Calibrated%20and%20Diffusion-Enhanced%20Data%20Augmentation/images/89287ce120b3d6fb63e4fada4c837889931144b3364c60604e3a3829e2806fc9.jpg)

![954273a5a2ca07521d7e1bf4066c58e9265637823206d9dfd9d715d8a87187a8.jpg](../icml_results/3348_SCalDA_%20Semantics-Calibrated%20and%20Diffusion-Enhanced%20Data%20Augmentation/images/954273a5a2ca07521d7e1bf4066c58e9265637823206d9dfd9d715d8a87187a8.jpg)

![a7cb04425826f80a6944ddd87194f8ffc508aec2aa7718673fcd4bd9b01d62d2.jpg](../icml_results/3348_SCalDA_%20Semantics-Calibrated%20and%20Diffusion-Enhanced%20Data%20Augmentation/images/a7cb04425826f80a6944ddd87194f8ffc508aec2aa7718673fcd4bd9b01d62d2.jpg)

![f619250d59140fb2b8ed14f7bc13c5675b2012205ba10a24f60a4a4b17d81976.jpg](../icml_results/3348_SCalDA_%20Semantics-Calibrated%20and%20Diffusion-Enhanced%20Data%20Augmentation/images/f619250d59140fb2b8ed14f7bc13c5675b2012205ba10a24f60a4a4b17d81976.jpg)

### Tables

![3b4813376a055237a88732083a792c9d191adb102bbf307c61b9cbb1715aae53.jpg](../icml_results/3348_SCalDA_%20Semantics-Calibrated%20and%20Diffusion-Enhanced%20Data%20Augmentation/tables/3b4813376a055237a88732083a792c9d191adb102bbf307c61b9cbb1715aae53.jpg)

![4e148710524a003225ce9f289825190dbdc23befa1903f82ad964e6eb8c7642a.jpg](../icml_results/3348_SCalDA_%20Semantics-Calibrated%20and%20Diffusion-Enhanced%20Data%20Augmentation/tables/4e148710524a003225ce9f289825190dbdc23befa1903f82ad964e6eb8c7642a.jpg)

![5d831cea04706f8ef7bb4f6644eb7b1a92ea0be4b1c451d8d7bca5055272bdcd.jpg](../icml_results/3348_SCalDA_%20Semantics-Calibrated%20and%20Diffusion-Enhanced%20Data%20Augmentation/tables/5d831cea04706f8ef7bb4f6644eb7b1a92ea0be4b1c451d8d7bca5055272bdcd.jpg)

![97c1ab872847258d8fdba2803f5559700ad5532350416425daba6722c5ead764.jpg](../icml_results/3348_SCalDA_%20Semantics-Calibrated%20and%20Diffusion-Enhanced%20Data%20Augmentation/tables/97c1ab872847258d8fdba2803f5559700ad5532350416425daba6722c5ead764.jpg)

![ba46b3f7d8193d66d8b939480d334230d07ad1329240be76a79e2908fe805649.jpg](../icml_results/3348_SCalDA_%20Semantics-Calibrated%20and%20Diffusion-Enhanced%20Data%20Augmentation/tables/ba46b3f7d8193d66d8b939480d334230d07ad1329240be76a79e2908fe805649.jpg)

![be57fd6b8a3dae0b7812f0a2add758730e97f28fa23198a11f88096376f8d8bb.jpg](../icml_results/3348_SCalDA_%20Semantics-Calibrated%20and%20Diffusion-Enhanced%20Data%20Augmentation/tables/be57fd6b8a3dae0b7812f0a2add758730e97f28fa23198a11f88096376f8d8bb.jpg)

![ee99307129394beea6f4cf9b100fe46b0478c3566e3b8eafa49c285986dde278.jpg](../icml_results/3348_SCalDA_%20Semantics-Calibrated%20and%20Diffusion-Enhanced%20Data%20Augmentation/tables/ee99307129394beea6f4cf9b100fe46b0478c3566e3b8eafa49c285986dde278.jpg)

![f290b23edeba39604b3a98f25d217ebb4bcab91c4748e8fe3c5881c9c730dea6.jpg](../icml_results/3348_SCalDA_%20Semantics-Calibrated%20and%20Diffusion-Enhanced%20Data%20Augmentation/tables/f290b23edeba39604b3a98f25d217ebb4bcab91c4748e8fe3c5881c9c730dea6.jpg)

![fc79843be8d81b0197a899035249ae51ed912e6a1a221bd780357c231525a8bc.jpg](../icml_results/3348_SCalDA_%20Semantics-Calibrated%20and%20Diffusion-Enhanced%20Data%20Augmentation/tables/fc79843be8d81b0197a899035249ae51ed912e6a1a221bd780357c231525a8bc.jpg)

<a id="paper-3310"></a>

## FLAC: Maximum Entropy RL via Kinetic Energy Regularized Bridge Matching


### Images

![07c26dc7475d5f83702f9d094ec000f2ccc23c852dd17a3bd23076dedd7e4b59.jpg](../icml_results/3349_FLAC_%20Maximum%20Entropy%20RL%20via%20Kinetic%20Energy%20Regularized%20Bridge%20Matching/images/07c26dc7475d5f83702f9d094ec000f2ccc23c852dd17a3bd23076dedd7e4b59.jpg)

![181884a58120c5b62cca97e91a7eaffc50b8d5b073ecb53c1ec407dad4121c4f.jpg](../icml_results/3349_FLAC_%20Maximum%20Entropy%20RL%20via%20Kinetic%20Energy%20Regularized%20Bridge%20Matching/images/181884a58120c5b62cca97e91a7eaffc50b8d5b073ecb53c1ec407dad4121c4f.jpg)

![3fa2cff9a867fac28027164101eaffb1b1a562c2b97f990ecfe27fd5d57f24e0.jpg](../icml_results/3349_FLAC_%20Maximum%20Entropy%20RL%20via%20Kinetic%20Energy%20Regularized%20Bridge%20Matching/images/3fa2cff9a867fac28027164101eaffb1b1a562c2b97f990ecfe27fd5d57f24e0.jpg)

![45d6409220f6ec7d111d24c43d77b0b80bbb75976a0bd8106ebcd1e3cf248b96.jpg](../icml_results/3349_FLAC_%20Maximum%20Entropy%20RL%20via%20Kinetic%20Energy%20Regularized%20Bridge%20Matching/images/45d6409220f6ec7d111d24c43d77b0b80bbb75976a0bd8106ebcd1e3cf248b96.jpg)

![57e2ca0e6e9debf39c59f3095245176e2d1dc11aa8334149cf06113898b52064.jpg](../icml_results/3349_FLAC_%20Maximum%20Entropy%20RL%20via%20Kinetic%20Energy%20Regularized%20Bridge%20Matching/images/57e2ca0e6e9debf39c59f3095245176e2d1dc11aa8334149cf06113898b52064.jpg)

![7e9f6a45a444028db6f62f078b570c1608ed6a00c06bd98c11cfaec79bb842de.jpg](../icml_results/3349_FLAC_%20Maximum%20Entropy%20RL%20via%20Kinetic%20Energy%20Regularized%20Bridge%20Matching/images/7e9f6a45a444028db6f62f078b570c1608ed6a00c06bd98c11cfaec79bb842de.jpg)

![9332d2ce961abd23f1e4ec6f8d8cdfe5f38451471f408ade22b1c129159e96ef.jpg](../icml_results/3349_FLAC_%20Maximum%20Entropy%20RL%20via%20Kinetic%20Energy%20Regularized%20Bridge%20Matching/images/9332d2ce961abd23f1e4ec6f8d8cdfe5f38451471f408ade22b1c129159e96ef.jpg)

![b9f9988ce27ddf3fe905e6e62a09224cf398e40729974ece6fdd39773a9d33c8.jpg](../icml_results/3349_FLAC_%20Maximum%20Entropy%20RL%20via%20Kinetic%20Energy%20Regularized%20Bridge%20Matching/images/b9f9988ce27ddf3fe905e6e62a09224cf398e40729974ece6fdd39773a9d33c8.jpg)

![d37e3a17189ec98e9e7a7282dd20a6e62b9dacb909eb37edf9d476dda2ce9f6f.jpg](../icml_results/3349_FLAC_%20Maximum%20Entropy%20RL%20via%20Kinetic%20Energy%20Regularized%20Bridge%20Matching/images/d37e3a17189ec98e9e7a7282dd20a6e62b9dacb909eb37edf9d476dda2ce9f6f.jpg)

### Tables

![52482a2635bd0b36e6224002641d3c0388e5a2c578649da6e8b861ae086c1e6b.jpg](../icml_results/3349_FLAC_%20Maximum%20Entropy%20RL%20via%20Kinetic%20Energy%20Regularized%20Bridge%20Matching/tables/52482a2635bd0b36e6224002641d3c0388e5a2c578649da6e8b861ae086c1e6b.jpg)

![6ba8cee455c46e70f6aee8734c5098c11def3a9ceba61e39600db407729b1262.jpg](../icml_results/3349_FLAC_%20Maximum%20Entropy%20RL%20via%20Kinetic%20Energy%20Regularized%20Bridge%20Matching/tables/6ba8cee455c46e70f6aee8734c5098c11def3a9ceba61e39600db407729b1262.jpg)

![79ff41895ffd3a55005df1b2ca20b8865c807fd6e93cd7cf2abacb3969dd672f.jpg](../icml_results/3349_FLAC_%20Maximum%20Entropy%20RL%20via%20Kinetic%20Energy%20Regularized%20Bridge%20Matching/tables/79ff41895ffd3a55005df1b2ca20b8865c807fd6e93cd7cf2abacb3969dd672f.jpg)

<a id="paper-3311"></a>

## Spatial-Aware Reduction Framework: Towards Efficient and Faithful Visual State Space Models


### Images

![249a3761363bc155f367ba8ff7d15967090e6b3543a45edbc83092ce0c37992a.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/images/249a3761363bc155f367ba8ff7d15967090e6b3543a45edbc83092ce0c37992a.jpg)

![2f0b773f6825564060038e146a404d6041510001c7bf15c372723f42fbf23028.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/images/2f0b773f6825564060038e146a404d6041510001c7bf15c372723f42fbf23028.jpg)

![3f0fcad89608258de811500ef1c9864724986b0653850056f6ab4127caf14fd2.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/images/3f0fcad89608258de811500ef1c9864724986b0653850056f6ab4127caf14fd2.jpg)

![5bf45fcd2d593481d0226e049c0d184d0ba0ddd111bdffe6cb2fb4ae731d2479.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/images/5bf45fcd2d593481d0226e049c0d184d0ba0ddd111bdffe6cb2fb4ae731d2479.jpg)

![95464efdffa37b1003b81caa3850d48c0e0f7636512055f7548ce714a26c353c.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/images/95464efdffa37b1003b81caa3850d48c0e0f7636512055f7548ce714a26c353c.jpg)

![97a26e1a2a93db2115934004c7c648041f8183f6e608a25e1a331d935d2c004a.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/images/97a26e1a2a93db2115934004c7c648041f8183f6e608a25e1a331d935d2c004a.jpg)

![a3e3d0838c118955521d8b9e137b6b4e1a2cf8055e15acea3398facfe5993dfa.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/images/a3e3d0838c118955521d8b9e137b6b4e1a2cf8055e15acea3398facfe5993dfa.jpg)

![a5ce271cc844f77b0910493afe61ede3b57d117f14543e478c703efb97d297e9.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/images/a5ce271cc844f77b0910493afe61ede3b57d117f14543e478c703efb97d297e9.jpg)

![bb88322bd69964e7325a978cc6519615bf8d44ad76b056c737394d8dadf5d6e8.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/images/bb88322bd69964e7325a978cc6519615bf8d44ad76b056c737394d8dadf5d6e8.jpg)

![d7e52556d080390b0d8e6be12d6a9fc23033ac010ddb5b4d5bf4c1c9d5eb5aaa.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/images/d7e52556d080390b0d8e6be12d6a9fc23033ac010ddb5b4d5bf4c1c9d5eb5aaa.jpg)

![e09da5e910cf4e45df1d6190682a08c9a53c0637e2e1392e2d873125f5f2d91a.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/images/e09da5e910cf4e45df1d6190682a08c9a53c0637e2e1392e2d873125f5f2d91a.jpg)

![f64356d8c40dc9f66f7b58b0fdb9da7ec6ad1befc792cc15e74cb33cd355f70f.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/images/f64356d8c40dc9f66f7b58b0fdb9da7ec6ad1befc792cc15e74cb33cd355f70f.jpg)

### Tables

![37fc2f4cbcd09114ec018f4d232c51156b4b8c77c43d6114fac60eb6c2e78969.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/tables/37fc2f4cbcd09114ec018f4d232c51156b4b8c77c43d6114fac60eb6c2e78969.jpg)

![46b6144285b5948879ad09e87625b7b083cf14cf0096680bc6f2dd4ac31e5320.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/tables/46b6144285b5948879ad09e87625b7b083cf14cf0096680bc6f2dd4ac31e5320.jpg)

![4dc13a3527b4bf2dc5271037bba85887b5f06a2bd78711dd64a4daaabe0a6e09.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/tables/4dc13a3527b4bf2dc5271037bba85887b5f06a2bd78711dd64a4daaabe0a6e09.jpg)

![8d1fcbe52635c3b169ee8151c592851d1621049707b519db6bfd6cbeec36eac9.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/tables/8d1fcbe52635c3b169ee8151c592851d1621049707b519db6bfd6cbeec36eac9.jpg)

![b960265a45d380f8bc447cda6678db56c7f6596ee39354c194603192ec9d0744.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/tables/b960265a45d380f8bc447cda6678db56c7f6596ee39354c194603192ec9d0744.jpg)

![ea3742a56b51eae34a021fc902f8ee80b0e00ee5a17f2f6f498130a927ebde94.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/tables/ea3742a56b51eae34a021fc902f8ee80b0e00ee5a17f2f6f498130a927ebde94.jpg)

![ec97e5440f4f0db130e806ef69762cbf08d48926c75e1f0aa8b731fb4a9a0f0d.jpg](../icml_results/3350_Spatial-Aware%20Reduction%20Framework_%20Towards%20Efficient%20and%20Faithful%20Visual%20State%20Space%20Models/tables/ec97e5440f4f0db130e806ef69762cbf08d48926c75e1f0aa8b731fb4a9a0f0d.jpg)

<a id="paper-3312"></a>

## Beyond Rational Illusion: Behaviorally Realistic Strategic Classification


### Images

![0b94b33edafc94c8ad223c7adce9b75a2da5e76fa80c468a039f0df088248948.jpg](../icml_results/3351_Beyond%20Rational%20Illusion_%20Behaviorally%20Realistic%20Strategic%20Classification/images/0b94b33edafc94c8ad223c7adce9b75a2da5e76fa80c468a039f0df088248948.jpg)

![8b9364f216a175d1e68c5909a60f14df80ba94e0a386647fc5a90a5e27467bf5.jpg](../icml_results/3351_Beyond%20Rational%20Illusion_%20Behaviorally%20Realistic%20Strategic%20Classification/images/8b9364f216a175d1e68c5909a60f14df80ba94e0a386647fc5a90a5e27467bf5.jpg)

![b6121064fc2081eda3c5ef3157ad04251b039c7d053e7e200b7c3962ead74019.jpg](../icml_results/3351_Beyond%20Rational%20Illusion_%20Behaviorally%20Realistic%20Strategic%20Classification/images/b6121064fc2081eda3c5ef3157ad04251b039c7d053e7e200b7c3962ead74019.jpg)

![e7e6240305dde523650396084c76576a870ab72d6163ef156c51618a4afca7a1.jpg](../icml_results/3351_Beyond%20Rational%20Illusion_%20Behaviorally%20Realistic%20Strategic%20Classification/images/e7e6240305dde523650396084c76576a870ab72d6163ef156c51618a4afca7a1.jpg)

![e9629f181a1090cd7ae05fe63aacc6676bd1325d5db6cb2b55c1a8431ba17df1.jpg](../icml_results/3351_Beyond%20Rational%20Illusion_%20Behaviorally%20Realistic%20Strategic%20Classification/images/e9629f181a1090cd7ae05fe63aacc6676bd1325d5db6cb2b55c1a8431ba17df1.jpg)

![ed09657d0e8a35accbe513edfa9584cb39bb23a2cfa06e0a45db2247425b98d6.jpg](../icml_results/3351_Beyond%20Rational%20Illusion_%20Behaviorally%20Realistic%20Strategic%20Classification/images/ed09657d0e8a35accbe513edfa9584cb39bb23a2cfa06e0a45db2247425b98d6.jpg)

### Tables

![048c06bbf0d7e811dcc48b259df9cb97b3279472042035752b9c6aa6b57c6ffe.jpg](../icml_results/3351_Beyond%20Rational%20Illusion_%20Behaviorally%20Realistic%20Strategic%20Classification/tables/048c06bbf0d7e811dcc48b259df9cb97b3279472042035752b9c6aa6b57c6ffe.jpg)

![09a670d25e6c754a4a234e16a72d47677f91a80a46ef660db7fd1e64df646c80.jpg](../icml_results/3351_Beyond%20Rational%20Illusion_%20Behaviorally%20Realistic%20Strategic%20Classification/tables/09a670d25e6c754a4a234e16a72d47677f91a80a46ef660db7fd1e64df646c80.jpg)

![0a60126604b4e5e1481e8bf6ecbeb1e64bc9bc8f7ffacf76680c24378e59cd20.jpg](../icml_results/3351_Beyond%20Rational%20Illusion_%20Behaviorally%20Realistic%20Strategic%20Classification/tables/0a60126604b4e5e1481e8bf6ecbeb1e64bc9bc8f7ffacf76680c24378e59cd20.jpg)

![18b1cda825ae1965130daeb21f15b2342d7411000959c4ea9e657460bf7e88b3.jpg](../icml_results/3351_Beyond%20Rational%20Illusion_%20Behaviorally%20Realistic%20Strategic%20Classification/tables/18b1cda825ae1965130daeb21f15b2342d7411000959c4ea9e657460bf7e88b3.jpg)

![1ad7889739e91e9cc967cba39f0c57f4fb3e9ef3c62cd3ed3537248cbd7dd463.jpg](../icml_results/3351_Beyond%20Rational%20Illusion_%20Behaviorally%20Realistic%20Strategic%20Classification/tables/1ad7889739e91e9cc967cba39f0c57f4fb3e9ef3c62cd3ed3537248cbd7dd463.jpg)

![225bc194dfda51e5a8973b568e1ab7f2e19477bf5840a10cfd14f7c9f14feeb2.jpg](../icml_results/3351_Beyond%20Rational%20Illusion_%20Behaviorally%20Realistic%20Strategic%20Classification/tables/225bc194dfda51e5a8973b568e1ab7f2e19477bf5840a10cfd14f7c9f14feeb2.jpg)

![4bbb80f9131a18a8322642efeb6ce25d879eca60d828f6e4a382bbec717d451a.jpg](../icml_results/3351_Beyond%20Rational%20Illusion_%20Behaviorally%20Realistic%20Strategic%20Classification/tables/4bbb80f9131a18a8322642efeb6ce25d879eca60d828f6e4a382bbec717d451a.jpg)

![8cbaef86051d82f3970c9a805ec31390c806a5db5a5ce2feafc9a467eb3230b2.jpg](../icml_results/3351_Beyond%20Rational%20Illusion_%20Behaviorally%20Realistic%20Strategic%20Classification/tables/8cbaef86051d82f3970c9a805ec31390c806a5db5a5ce2feafc9a467eb3230b2.jpg)

![bf8b7928a7da57ae90bc4bb3b639898cac33a46c14d77ca322daa275d5f9bbd6.jpg](../icml_results/3351_Beyond%20Rational%20Illusion_%20Behaviorally%20Realistic%20Strategic%20Classification/tables/bf8b7928a7da57ae90bc4bb3b639898cac33a46c14d77ca322daa275d5f9bbd6.jpg)

![c8e41f880c9d0f31b735e38385cf8dfaf2188208f52f396545d317d2a4c0140f.jpg](../icml_results/3351_Beyond%20Rational%20Illusion_%20Behaviorally%20Realistic%20Strategic%20Classification/tables/c8e41f880c9d0f31b735e38385cf8dfaf2188208f52f396545d317d2a4c0140f.jpg)

![cd2a88d81b01510374396542d0feeb73f238f077f1c2f2d00599133755e13b82.jpg](../icml_results/3351_Beyond%20Rational%20Illusion_%20Behaviorally%20Realistic%20Strategic%20Classification/tables/cd2a88d81b01510374396542d0feeb73f238f077f1c2f2d00599133755e13b82.jpg)

<a id="paper-3313"></a>

## When Tabular Foundation Models Meet Strategic Tabular Data: A Prior Alignment Approach


### Images

![1558213ac7e36c607601dfdf21c8e9de4f7be21e10b716df876304e730e2a746.jpg](../icml_results/3352_When%20Tabular%20Foundation%20Models%20Meet%20Strategic%20Tabular%20Data_%20A%20Prior%20Alignment%20Approach/images/1558213ac7e36c607601dfdf21c8e9de4f7be21e10b716df876304e730e2a746.jpg)

![217d801d14ab0e7a138f3f8fb7fe435d550bd7601826e410ff68bec2b7a73173.jpg](../icml_results/3352_When%20Tabular%20Foundation%20Models%20Meet%20Strategic%20Tabular%20Data_%20A%20Prior%20Alignment%20Approach/images/217d801d14ab0e7a138f3f8fb7fe435d550bd7601826e410ff68bec2b7a73173.jpg)

![4c16a14e8120eb8351c3c5106b2d2278a4f4b7d1307254f86429f24e1c264eec.jpg](../icml_results/3352_When%20Tabular%20Foundation%20Models%20Meet%20Strategic%20Tabular%20Data_%20A%20Prior%20Alignment%20Approach/images/4c16a14e8120eb8351c3c5106b2d2278a4f4b7d1307254f86429f24e1c264eec.jpg)

![65d156051b4a7bbcf7e39f7e3c207b984cbd76657a33fc35517c7f61032cdb91.jpg](../icml_results/3352_When%20Tabular%20Foundation%20Models%20Meet%20Strategic%20Tabular%20Data_%20A%20Prior%20Alignment%20Approach/images/65d156051b4a7bbcf7e39f7e3c207b984cbd76657a33fc35517c7f61032cdb91.jpg)

![724d55a285d4c179082e78dabc1ad7add8b902971f2f4382ab386d1ab96b313e.jpg](../icml_results/3352_When%20Tabular%20Foundation%20Models%20Meet%20Strategic%20Tabular%20Data_%20A%20Prior%20Alignment%20Approach/images/724d55a285d4c179082e78dabc1ad7add8b902971f2f4382ab386d1ab96b313e.jpg)

![7bb5ecf4bac91680fb05ad66eeda127412d5b6043d05f1d9df4a1638e9820a53.jpg](../icml_results/3352_When%20Tabular%20Foundation%20Models%20Meet%20Strategic%20Tabular%20Data_%20A%20Prior%20Alignment%20Approach/images/7bb5ecf4bac91680fb05ad66eeda127412d5b6043d05f1d9df4a1638e9820a53.jpg)

![954158544ec32031037e9cd6ce01568d62b474e1ce050c4f012945b1e299ee41.jpg](../icml_results/3352_When%20Tabular%20Foundation%20Models%20Meet%20Strategic%20Tabular%20Data_%20A%20Prior%20Alignment%20Approach/images/954158544ec32031037e9cd6ce01568d62b474e1ce050c4f012945b1e299ee41.jpg)

![a70fb2c9d1f5c077b8c9b544147875d4b827a0c6ed79b2fb70438897a0c099c7.jpg](../icml_results/3352_When%20Tabular%20Foundation%20Models%20Meet%20Strategic%20Tabular%20Data_%20A%20Prior%20Alignment%20Approach/images/a70fb2c9d1f5c077b8c9b544147875d4b827a0c6ed79b2fb70438897a0c099c7.jpg)

![e1d7f62f3f6daef19f1346af8f9adace4d7f285fce4d0d9aef221c90d60e9114.jpg](../icml_results/3352_When%20Tabular%20Foundation%20Models%20Meet%20Strategic%20Tabular%20Data_%20A%20Prior%20Alignment%20Approach/images/e1d7f62f3f6daef19f1346af8f9adace4d7f285fce4d0d9aef221c90d60e9114.jpg)

### Tables

![157a1bc6d853635eb89346efa434dd2be5543e78462bff434630086198ea1fa6.jpg](../icml_results/3352_When%20Tabular%20Foundation%20Models%20Meet%20Strategic%20Tabular%20Data_%20A%20Prior%20Alignment%20Approach/tables/157a1bc6d853635eb89346efa434dd2be5543e78462bff434630086198ea1fa6.jpg)

![3c89518c58f5a3322d6c633731b21d0221492debd199dc275ce6b7648f3bf426.jpg](../icml_results/3352_When%20Tabular%20Foundation%20Models%20Meet%20Strategic%20Tabular%20Data_%20A%20Prior%20Alignment%20Approach/tables/3c89518c58f5a3322d6c633731b21d0221492debd199dc275ce6b7648f3bf426.jpg)

![41d5f92ab9152277cf0045466beb7340c16f3e8e5e6fc36a3c5174a2c127d288.jpg](../icml_results/3352_When%20Tabular%20Foundation%20Models%20Meet%20Strategic%20Tabular%20Data_%20A%20Prior%20Alignment%20Approach/tables/41d5f92ab9152277cf0045466beb7340c16f3e8e5e6fc36a3c5174a2c127d288.jpg)

<a id="paper-3314"></a>

## Hierarchical ODE: Learning Continuous-Time Physical Prototypes for Early Link Failure Detection


### Images

![137695bc030eb7d69e3bfbf14b7abc8692f23a395149947d5a969d288dd6825c.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/137695bc030eb7d69e3bfbf14b7abc8692f23a395149947d5a969d288dd6825c.jpg)

![1b82d74263a225f95c1cf47abdede60f06ca7c9e94076bd45ddcea056f8375ae.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/1b82d74263a225f95c1cf47abdede60f06ca7c9e94076bd45ddcea056f8375ae.jpg)

![1eeacddae35588f694c27c0cb8d02005f522b680bb374c0d5e56bb13c90f03b9.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/1eeacddae35588f694c27c0cb8d02005f522b680bb374c0d5e56bb13c90f03b9.jpg)

![21708ef495b1c32d8710be6b927f793142a9edbea0eef585d1fd5743cd543788.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/21708ef495b1c32d8710be6b927f793142a9edbea0eef585d1fd5743cd543788.jpg)

![2d751dcd429b639af9e4350ef084de46b4ea9201912688ca46a24e09cc2e98a0.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/2d751dcd429b639af9e4350ef084de46b4ea9201912688ca46a24e09cc2e98a0.jpg)

![30a242c701faf5f24fe99cfd03ffc2f3071f24b14fa9eee2c277fc328f1f6e30.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/30a242c701faf5f24fe99cfd03ffc2f3071f24b14fa9eee2c277fc328f1f6e30.jpg)

![32b643f0422e3a8f2aadef3d85affd63024b9c1a971e264decc3c0e095fd00a8.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/32b643f0422e3a8f2aadef3d85affd63024b9c1a971e264decc3c0e095fd00a8.jpg)

![356b0c7420687939243f6333e4de3a006dae857886e327fac8ceed37fba9e03e.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/356b0c7420687939243f6333e4de3a006dae857886e327fac8ceed37fba9e03e.jpg)

![3990a9836019b9f1c6490e1e58893a39f0a33915747f8b177c36ca486ed2fe34.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/3990a9836019b9f1c6490e1e58893a39f0a33915747f8b177c36ca486ed2fe34.jpg)

![3eaedc42c081ac1c03a7b9dbc5dfb74cb8b1b5f1ea157e3ceb0729728e890b74.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/3eaedc42c081ac1c03a7b9dbc5dfb74cb8b1b5f1ea157e3ceb0729728e890b74.jpg)

![3ecfd2a141bb64d209b2724974c805df61757e6841a24e4f019a1d053488cff5.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/3ecfd2a141bb64d209b2724974c805df61757e6841a24e4f019a1d053488cff5.jpg)

![40013b559de21381a5559f3c57f0b47da282fd6259b797302f0e3144e414ba99.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/40013b559de21381a5559f3c57f0b47da282fd6259b797302f0e3144e414ba99.jpg)

![40967474e85edbadccad561df8048fecb3586223e925943b8529d6f51bfd9c8f.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/40967474e85edbadccad561df8048fecb3586223e925943b8529d6f51bfd9c8f.jpg)

![44418a67fafba96e9695e83bb5cdf32fe686167a806d6e2645314ca3d924afd6.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/44418a67fafba96e9695e83bb5cdf32fe686167a806d6e2645314ca3d924afd6.jpg)

![45cdd76976510d605675866fe3a7bba81470c751bbdc9619377eaf2b458198e2.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/45cdd76976510d605675866fe3a7bba81470c751bbdc9619377eaf2b458198e2.jpg)

![4a323db3683abb1fc90d59bb895ee1c58140308b5b53369693dfeb75a08d5d1c.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/4a323db3683abb1fc90d59bb895ee1c58140308b5b53369693dfeb75a08d5d1c.jpg)

![4d904056a854cd241704de92fbd16449a961a437625cf3fe1658e0ca1dffb5b3.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/4d904056a854cd241704de92fbd16449a961a437625cf3fe1658e0ca1dffb5b3.jpg)

![58e07640b9f34cd849e89b1ec39f776b6a8b5053db77e38d04461a648cbb2cc6.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/58e07640b9f34cd849e89b1ec39f776b6a8b5053db77e38d04461a648cbb2cc6.jpg)

![5c6ef1e2e87a6ba95f75bc36541e8fd91f52967e01aebb0db08fd0106344bcf7.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/5c6ef1e2e87a6ba95f75bc36541e8fd91f52967e01aebb0db08fd0106344bcf7.jpg)

![6ad106f96211f64b3ebe2506c00bd353b12f0e91a80c77e1bc7a7eec8e400d7f.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/6ad106f96211f64b3ebe2506c00bd353b12f0e91a80c77e1bc7a7eec8e400d7f.jpg)

![6b666cc069d15a5375fc136cded11a03f1adb88dcb758e8b05af89b4e93e139c.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/6b666cc069d15a5375fc136cded11a03f1adb88dcb758e8b05af89b4e93e139c.jpg)

![6ded4cff18e24ad363264ff0513352e842a92ef922fa0b7221285696c85eeee9.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/6ded4cff18e24ad363264ff0513352e842a92ef922fa0b7221285696c85eeee9.jpg)

![71943f049e74b8bd03c70b7fc92aa1985a32ddfbfa114d65556982bdae592804.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/71943f049e74b8bd03c70b7fc92aa1985a32ddfbfa114d65556982bdae592804.jpg)

![71e9e7c033a6e36d99368fa2f5f4a20b6409c6703bf5a4664b02e9beeef28008.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/71e9e7c033a6e36d99368fa2f5f4a20b6409c6703bf5a4664b02e9beeef28008.jpg)

![72bf81bece26dfebf579403b28678832e88334e7178b430cb7d00adbd2f680ef.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/72bf81bece26dfebf579403b28678832e88334e7178b430cb7d00adbd2f680ef.jpg)

![745b3103ad731cf6e1e4c7c8bee5664d1b513d49322e05ec26ad6abe5e4d1915.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/745b3103ad731cf6e1e4c7c8bee5664d1b513d49322e05ec26ad6abe5e4d1915.jpg)

![79c158065b17bee5e3559f273ca72119f2219c21690bc7731b73b896d02c37da.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/79c158065b17bee5e3559f273ca72119f2219c21690bc7731b73b896d02c37da.jpg)

![7d43df4aacba0c396ca3a9b896406fc021125396e9932abce6c8a040a020e81f.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/7d43df4aacba0c396ca3a9b896406fc021125396e9932abce6c8a040a020e81f.jpg)

![80f3c97de57cd44d1d6c3beec265e6bcabc6eae423396515526d5d58dfcc5a87.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/80f3c97de57cd44d1d6c3beec265e6bcabc6eae423396515526d5d58dfcc5a87.jpg)

![8426b825019ed7a430c724b6d66d8af63bc6f63519c7ca7017c93653bc1d054a.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/8426b825019ed7a430c724b6d66d8af63bc6f63519c7ca7017c93653bc1d054a.jpg)

![84d8c20bff53217152fc37e032a6fc963d2c24884507c0cbaaf1db26d27e1829.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/84d8c20bff53217152fc37e032a6fc963d2c24884507c0cbaaf1db26d27e1829.jpg)

![8737b5a5b8c8654480532003a96397dc14dc858a11bdbebcb85923446e71aa0c.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/8737b5a5b8c8654480532003a96397dc14dc858a11bdbebcb85923446e71aa0c.jpg)

![8a687cbc2241b9c3a660ddbf4bcd913a9c6d37a1433af36bb5683dc47a75e62d.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/8a687cbc2241b9c3a660ddbf4bcd913a9c6d37a1433af36bb5683dc47a75e62d.jpg)

![8cc9d7c1ac44bdb37cca5d5e64726b1d47bf86db4dd15c4fa39a2f127645661d.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/8cc9d7c1ac44bdb37cca5d5e64726b1d47bf86db4dd15c4fa39a2f127645661d.jpg)

![8dae99413014891c3ee1881f0d845c18d0b9d0d018c1fe37aacb0baea7e897f5.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/8dae99413014891c3ee1881f0d845c18d0b9d0d018c1fe37aacb0baea7e897f5.jpg)

![9144a3286a17ee03fd7fdf711aa046c6fd64a99bca792dd21ac267220070c025.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/9144a3286a17ee03fd7fdf711aa046c6fd64a99bca792dd21ac267220070c025.jpg)

![920a79c2d3bd085123a8cacd603f4d7198fa9676076469f073a05e1149950fe9.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/920a79c2d3bd085123a8cacd603f4d7198fa9676076469f073a05e1149950fe9.jpg)

![95750b014f4405d0e442deb4720c612a5b13db09e992c8d83a1c1c8fad32a2fc.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/95750b014f4405d0e442deb4720c612a5b13db09e992c8d83a1c1c8fad32a2fc.jpg)

![9c0330e8b7e1ff85b592d872c14d53066c99c9347abdd0088faee41ae24c18ec.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/9c0330e8b7e1ff85b592d872c14d53066c99c9347abdd0088faee41ae24c18ec.jpg)

![9d7ac4e7b175e2960ea081e571e0fea29eadd33a2619eaad87b2c2b874e8ee17.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/9d7ac4e7b175e2960ea081e571e0fea29eadd33a2619eaad87b2c2b874e8ee17.jpg)

![a92878d1129de8b13a2369a35ed03212caae4562e93d6c034dfbfb2b230840f2.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/a92878d1129de8b13a2369a35ed03212caae4562e93d6c034dfbfb2b230840f2.jpg)

![b34b5e498f97baf02471acff23a9bc697d69af8695503e0dc933f51caa4b0e08.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/b34b5e498f97baf02471acff23a9bc697d69af8695503e0dc933f51caa4b0e08.jpg)

![bf908e02c5dff96f4dfc76f535d8f7b81d1e36273956bc4069a48b542b495106.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/bf908e02c5dff96f4dfc76f535d8f7b81d1e36273956bc4069a48b542b495106.jpg)

![cd0661fb78fc37b57a973629d9f7b5588656c2ec9c13d2b23569143c1691744f.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/cd0661fb78fc37b57a973629d9f7b5588656c2ec9c13d2b23569143c1691744f.jpg)

![d85852f166db39df904e8e8d9e3e6f0e6b2d29d35b58aaa4a82b6e0e88dd5f91.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/d85852f166db39df904e8e8d9e3e6f0e6b2d29d35b58aaa4a82b6e0e88dd5f91.jpg)

![da7773b90df53af4683a8a8c2a78f3deff0e231ea57e0339296436db60dcec1c.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/da7773b90df53af4683a8a8c2a78f3deff0e231ea57e0339296436db60dcec1c.jpg)

![eb3f09bd0440dd9403d4cb8f459f494873629a5259e3af53413699f407376430.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/eb3f09bd0440dd9403d4cb8f459f494873629a5259e3af53413699f407376430.jpg)

![f6a99566d5d3bd79ce5a235428f93fe3499f72b3ca05528ac78f87d46723a649.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/images/f6a99566d5d3bd79ce5a235428f93fe3499f72b3ca05528ac78f87d46723a649.jpg)

### Tables

![0d42f6e80816868db503aaf822e03c34770d2e7954e3a0f413024a86191ce359.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/tables/0d42f6e80816868db503aaf822e03c34770d2e7954e3a0f413024a86191ce359.jpg)

![12efee389ce4409e46e54c79f83f20f1b8e9421ef36482537041217c888d43e5.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/tables/12efee389ce4409e46e54c79f83f20f1b8e9421ef36482537041217c888d43e5.jpg)

![d2ddc18c9b8d2ef7650bdd5096027856995d67014ff087b3fabbfb5cbd3c8cac.jpg](../icml_results/3353_Hierarchical%20ODE_%20Learning%20Continuous-Time%20Physical%20Prototypes%20for%20Early%20Link%20Failure%20Detection/tables/d2ddc18c9b8d2ef7650bdd5096027856995d67014ff087b3fabbfb5cbd3c8cac.jpg)

<a id="paper-3315"></a>

## Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention


### Images

![0280635017bad15bcb8f141fd24fd8ddc00167d6f7450acf446670965984875c.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/images/0280635017bad15bcb8f141fd24fd8ddc00167d6f7450acf446670965984875c.jpg)

![0d0d71f23fe1bbd25b0535d6551f0a8ab7259c388ef8aee3a2bb9831244fa0f1.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/images/0d0d71f23fe1bbd25b0535d6551f0a8ab7259c388ef8aee3a2bb9831244fa0f1.jpg)

![3627dbf7f5208c3df09e2fed52da5b894621745d674d03c9f206cae6152d285e.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/images/3627dbf7f5208c3df09e2fed52da5b894621745d674d03c9f206cae6152d285e.jpg)

![5f9b301ca497c53ac6d8e45977f0690b085cd9b27c188e89ad0da632070a3942.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/images/5f9b301ca497c53ac6d8e45977f0690b085cd9b27c188e89ad0da632070a3942.jpg)

![6a3261a0b2637c4af21cdbcb4eae65ac494ca10b86e4887f0d7ecf53c10b3c2c.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/images/6a3261a0b2637c4af21cdbcb4eae65ac494ca10b86e4887f0d7ecf53c10b3c2c.jpg)

![846bad8151ed72b8fdbbb9a56fffca386bda3af945ec6ac08527a3fa20e7d015.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/images/846bad8151ed72b8fdbbb9a56fffca386bda3af945ec6ac08527a3fa20e7d015.jpg)

![93aed6098293085208d5b647702a4f1dc76424b26f105c0113ac28360e0c3478.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/images/93aed6098293085208d5b647702a4f1dc76424b26f105c0113ac28360e0c3478.jpg)

![96dcf54b74b41921e25d01a80c210c5d58f82581cab55835ffc8233adbac0c1a.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/images/96dcf54b74b41921e25d01a80c210c5d58f82581cab55835ffc8233adbac0c1a.jpg)

![adef0bb1e9643749750af3166428f56f25bf28da4ec938e04ea4760e8c142556.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/images/adef0bb1e9643749750af3166428f56f25bf28da4ec938e04ea4760e8c142556.jpg)

![b67fc52211a8c3eb0018f20d587d3c4f4577b488c206e39b6103f7df41e773e0.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/images/b67fc52211a8c3eb0018f20d587d3c4f4577b488c206e39b6103f7df41e773e0.jpg)

### Tables

![0d36c79a68832b6d8175faa60186871188e4c251968a69f445d7c5e5ff910cd3.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/tables/0d36c79a68832b6d8175faa60186871188e4c251968a69f445d7c5e5ff910cd3.jpg)

![20d365c94a82b4f5d01c32009a5cc74b8ed530717b95aca821c021da21be2d2c.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/tables/20d365c94a82b4f5d01c32009a5cc74b8ed530717b95aca821c021da21be2d2c.jpg)

![7349b52c3d53c82ce712aaa94dcccfb1f69dba0ba189123a286ace2edbd1703d.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/tables/7349b52c3d53c82ce712aaa94dcccfb1f69dba0ba189123a286ace2edbd1703d.jpg)

![7b4854ed09c13b12fb21c24ce8b836559526d06ef6136a0a2fab61b41e4f57a2.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/tables/7b4854ed09c13b12fb21c24ce8b836559526d06ef6136a0a2fab61b41e4f57a2.jpg)

![896acd29ef2a8bab796ccb30272030905a70ca5ad6124c35ff9a20ca968a033f.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/tables/896acd29ef2a8bab796ccb30272030905a70ca5ad6124c35ff9a20ca968a033f.jpg)

![9372887b07deb1d42c50c759eeb1aac9c91142136c8353aa00bb7d3b2aace55b.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/tables/9372887b07deb1d42c50c759eeb1aac9c91142136c8353aa00bb7d3b2aace55b.jpg)

![ae52071d5f50c12a039269f38ca8eb33ec8739c0e967e462c6b22a193a4c6539.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/tables/ae52071d5f50c12a039269f38ca8eb33ec8739c0e967e462c6b22a193a4c6539.jpg)

![c1d9b37cb69d9952567860e31f2b2f7eb7c7dd2224fa4eb2421947f2cad14fb3.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/tables/c1d9b37cb69d9952567860e31f2b2f7eb7c7dd2224fa4eb2421947f2cad14fb3.jpg)

![c9b5fdf96b3fbc1ba595ca0980e063c32eaf3a7d40586d4f3017a4ae910316b5.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/tables/c9b5fdf96b3fbc1ba595ca0980e063c32eaf3a7d40586d4f3017a4ae910316b5.jpg)

![cda29d95adc51ad22d8b1072ca10a1361a5fcce2c5f7f74557a6702b37fef34c.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/tables/cda29d95adc51ad22d8b1072ca10a1361a5fcce2c5f7f74557a6702b37fef34c.jpg)

![eb3f0a430e21142d6ecfc8ce83aa0bf38ca802514623ac6ddf9190ee00d223b6.jpg](../icml_results/3354_Light%20Forcing_%20Accelerating%20Autoregressive%20Video%20Diffusion%20via%20Sparse%20Attention/tables/eb3f0a430e21142d6ecfc8ce83aa0bf38ca802514623ac6ddf9190ee00d223b6.jpg)

<a id="paper-3316"></a>

## SHARP-Q: Spectral Hessian Alignment and Rectification for Post-training Quantization


### Images

![12f5899cb6aebadcdff6999607d9b231cf1686179efe139c77854fe60c5d5490.jpg](../icml_results/3355_SHARP-Q_%20Spectral%20Hessian%20Alignment%20and%20Rectification%20for%20Post-training%20Quantization/images/12f5899cb6aebadcdff6999607d9b231cf1686179efe139c77854fe60c5d5490.jpg)

![5ee8654f28eac2dbc2e49fd645ff28810c859180e393255082d0128184a5f940.jpg](../icml_results/3355_SHARP-Q_%20Spectral%20Hessian%20Alignment%20and%20Rectification%20for%20Post-training%20Quantization/images/5ee8654f28eac2dbc2e49fd645ff28810c859180e393255082d0128184a5f940.jpg)

![64f77c17245003f40dce5c2f16a87723a409a1f46dea567c660111c941ff1ec0.jpg](../icml_results/3355_SHARP-Q_%20Spectral%20Hessian%20Alignment%20and%20Rectification%20for%20Post-training%20Quantization/images/64f77c17245003f40dce5c2f16a87723a409a1f46dea567c660111c941ff1ec0.jpg)

![e4e665c5b8cd8adc680a739aa6ee3829118977375a2edfe434b362c6062e1291.jpg](../icml_results/3355_SHARP-Q_%20Spectral%20Hessian%20Alignment%20and%20Rectification%20for%20Post-training%20Quantization/images/e4e665c5b8cd8adc680a739aa6ee3829118977375a2edfe434b362c6062e1291.jpg)

### Tables

![26296194ed3d0a4740d5b12972e76d2b43252950ac5706e5a4387f0c7ae2aac4.jpg](../icml_results/3355_SHARP-Q_%20Spectral%20Hessian%20Alignment%20and%20Rectification%20for%20Post-training%20Quantization/tables/26296194ed3d0a4740d5b12972e76d2b43252950ac5706e5a4387f0c7ae2aac4.jpg)

![38d0617f3d6d3cd11c2c4dba70f34649200ed30e42665425095d2afa10b42193.jpg](../icml_results/3355_SHARP-Q_%20Spectral%20Hessian%20Alignment%20and%20Rectification%20for%20Post-training%20Quantization/tables/38d0617f3d6d3cd11c2c4dba70f34649200ed30e42665425095d2afa10b42193.jpg)

![40a92356fbacf09c28ad2f13a5fa7f9c0262638d150d26bb3ebd75ff8f1d6840.jpg](../icml_results/3355_SHARP-Q_%20Spectral%20Hessian%20Alignment%20and%20Rectification%20for%20Post-training%20Quantization/tables/40a92356fbacf09c28ad2f13a5fa7f9c0262638d150d26bb3ebd75ff8f1d6840.jpg)

![6df7737818421e27e999078379ed25e4fd253d1811a99077625411cb55e43db9.jpg](../icml_results/3355_SHARP-Q_%20Spectral%20Hessian%20Alignment%20and%20Rectification%20for%20Post-training%20Quantization/tables/6df7737818421e27e999078379ed25e4fd253d1811a99077625411cb55e43db9.jpg)

![94d9b7560e009f9e31eecf0edaabdae2f561f4c73726d6798b596967ca58a82e.jpg](../icml_results/3355_SHARP-Q_%20Spectral%20Hessian%20Alignment%20and%20Rectification%20for%20Post-training%20Quantization/tables/94d9b7560e009f9e31eecf0edaabdae2f561f4c73726d6798b596967ca58a82e.jpg)

![abf7ed85127c0293a87289ca9d2800a54365b6a74bb01c709ffb5c7dc809e99e.jpg](../icml_results/3355_SHARP-Q_%20Spectral%20Hessian%20Alignment%20and%20Rectification%20for%20Post-training%20Quantization/tables/abf7ed85127c0293a87289ca9d2800a54365b6a74bb01c709ffb5c7dc809e99e.jpg)

![c97d6fd1ba79ea51d3ef1371a32b560d355a60a226f8b127f200e36136cc60f7.jpg](../icml_results/3355_SHARP-Q_%20Spectral%20Hessian%20Alignment%20and%20Rectification%20for%20Post-training%20Quantization/tables/c97d6fd1ba79ea51d3ef1371a32b560d355a60a226f8b127f200e36136cc60f7.jpg)

![cb2c349af9847842923ed5e7314d75ab79f844b5acec4f45373c3ef2d24c51e5.jpg](../icml_results/3355_SHARP-Q_%20Spectral%20Hessian%20Alignment%20and%20Rectification%20for%20Post-training%20Quantization/tables/cb2c349af9847842923ed5e7314d75ab79f844b5acec4f45373c3ef2d24c51e5.jpg)

![fe36a1af1823e50ec91f18ab555a292a68905378fa0e290fddc1c4313d5b17bc.jpg](../icml_results/3355_SHARP-Q_%20Spectral%20Hessian%20Alignment%20and%20Rectification%20for%20Post-training%20Quantization/tables/fe36a1af1823e50ec91f18ab555a292a68905378fa0e290fddc1c4313d5b17bc.jpg)

<a id="paper-3317"></a>

## Knowing the Unknown: Interpretable Open-World Object Detection via Concept Decomposition Model


### Images

![0c687c1dd780b9b8e891bdfbd86b23d19846676294a1fb7e143a5e722db576f3.jpg](../icml_results/3356_Knowing%20the%20Unknown_%20Interpretable%20Open-World%20Object%20Detection%20via%20Concept%20Decomposition%20Model/images/0c687c1dd780b9b8e891bdfbd86b23d19846676294a1fb7e143a5e722db576f3.jpg)

![319e94510cea965428e2cc24c990bf3cc2196cd91f2d0259ab0d14ad7b9eadf2.jpg](../icml_results/3356_Knowing%20the%20Unknown_%20Interpretable%20Open-World%20Object%20Detection%20via%20Concept%20Decomposition%20Model/images/319e94510cea965428e2cc24c990bf3cc2196cd91f2d0259ab0d14ad7b9eadf2.jpg)

![a1dd824b9e2e8bd86b6371981398ff7452a0081b59d699fb010da5ba09bd715a.jpg](../icml_results/3356_Knowing%20the%20Unknown_%20Interpretable%20Open-World%20Object%20Detection%20via%20Concept%20Decomposition%20Model/images/a1dd824b9e2e8bd86b6371981398ff7452a0081b59d699fb010da5ba09bd715a.jpg)

![c31d5a1259c4b4f249c82978ba476fa297d2cd1211adf25a82d2480c6b5178d6.jpg](../icml_results/3356_Knowing%20the%20Unknown_%20Interpretable%20Open-World%20Object%20Detection%20via%20Concept%20Decomposition%20Model/images/c31d5a1259c4b4f249c82978ba476fa297d2cd1211adf25a82d2480c6b5178d6.jpg)

![de2dd0b97d05f62f124852a7c6f2c8c5f28f1ea1834282deb616dfc3860f9791.jpg](../icml_results/3356_Knowing%20the%20Unknown_%20Interpretable%20Open-World%20Object%20Detection%20via%20Concept%20Decomposition%20Model/images/de2dd0b97d05f62f124852a7c6f2c8c5f28f1ea1834282deb616dfc3860f9791.jpg)

![f314119ebab9efc0dfed2aa1f44990ede1f250d3c0893d8f5730b4f79f5005c4.jpg](../icml_results/3356_Knowing%20the%20Unknown_%20Interpretable%20Open-World%20Object%20Detection%20via%20Concept%20Decomposition%20Model/images/f314119ebab9efc0dfed2aa1f44990ede1f250d3c0893d8f5730b4f79f5005c4.jpg)

### Tables

![075316279eda742bd8481c5c69115d3e17cbe795d3eed08ec99897595be939c2.jpg](../icml_results/3356_Knowing%20the%20Unknown_%20Interpretable%20Open-World%20Object%20Detection%20via%20Concept%20Decomposition%20Model/tables/075316279eda742bd8481c5c69115d3e17cbe795d3eed08ec99897595be939c2.jpg)

![400cfaa8a4ad566561d9228d904755ec5fb0215eeac795fda18406316a1c644e.jpg](../icml_results/3356_Knowing%20the%20Unknown_%20Interpretable%20Open-World%20Object%20Detection%20via%20Concept%20Decomposition%20Model/tables/400cfaa8a4ad566561d9228d904755ec5fb0215eeac795fda18406316a1c644e.jpg)

![4e88c787f6dce73a3aa26241f34c0a35eda1eccaeddde392a9ccc75e29fb1722.jpg](../icml_results/3356_Knowing%20the%20Unknown_%20Interpretable%20Open-World%20Object%20Detection%20via%20Concept%20Decomposition%20Model/tables/4e88c787f6dce73a3aa26241f34c0a35eda1eccaeddde392a9ccc75e29fb1722.jpg)

![51e0dab223a53fad3d34ce5115d7f5e260bbc42e248322f5ed1ed0d54b6bc7c4.jpg](../icml_results/3356_Knowing%20the%20Unknown_%20Interpretable%20Open-World%20Object%20Detection%20via%20Concept%20Decomposition%20Model/tables/51e0dab223a53fad3d34ce5115d7f5e260bbc42e248322f5ed1ed0d54b6bc7c4.jpg)

![730f794fc51e6d27d9efcd9803380b7bd304bc10625567836bba5abc033f1519.jpg](../icml_results/3356_Knowing%20the%20Unknown_%20Interpretable%20Open-World%20Object%20Detection%20via%20Concept%20Decomposition%20Model/tables/730f794fc51e6d27d9efcd9803380b7bd304bc10625567836bba5abc033f1519.jpg)

<a id="paper-3318"></a>

## When Labelers Stay Silent: The Power of Ties in Cost-Effective Preference Learning


### Images

![03776a52482b493ef4845953c562a2a8e7ab3f0fa960c41c0f3279b714ee9554.jpg](../icml_results/3357_When%20Labelers%20Stay%20Silent_%20The%20Power%20of%20Ties%20in%20Cost-Effective%20Preference%20Learning/images/03776a52482b493ef4845953c562a2a8e7ab3f0fa960c41c0f3279b714ee9554.jpg)

![715c78328b697c4b7d729bf232ae27c5de3d058906761d535a44c73ee4a5f4bf.jpg](../icml_results/3357_When%20Labelers%20Stay%20Silent_%20The%20Power%20of%20Ties%20in%20Cost-Effective%20Preference%20Learning/images/715c78328b697c4b7d729bf232ae27c5de3d058906761d535a44c73ee4a5f4bf.jpg)

### Tables

![05036264f1c0865a02ee7e738e5ffe2dc510c3706364e32397781620d1d87464.jpg](../icml_results/3357_When%20Labelers%20Stay%20Silent_%20The%20Power%20of%20Ties%20in%20Cost-Effective%20Preference%20Learning/tables/05036264f1c0865a02ee7e738e5ffe2dc510c3706364e32397781620d1d87464.jpg)

![5bcf2800f4c7b66bd6253508256305caab7dbefb4dad742dc917dddebd5d739a.jpg](../icml_results/3357_When%20Labelers%20Stay%20Silent_%20The%20Power%20of%20Ties%20in%20Cost-Effective%20Preference%20Learning/tables/5bcf2800f4c7b66bd6253508256305caab7dbefb4dad742dc917dddebd5d739a.jpg)

![6b7b4672912ddf14beb10bb726fd336d911dd4b452c66c13eac2d3a69e1b0d93.jpg](../icml_results/3357_When%20Labelers%20Stay%20Silent_%20The%20Power%20of%20Ties%20in%20Cost-Effective%20Preference%20Learning/tables/6b7b4672912ddf14beb10bb726fd336d911dd4b452c66c13eac2d3a69e1b0d93.jpg)

![78ba8640ddb1b600654123f720d8771baa3158f272a5da527536de8dc3876b3a.jpg](../icml_results/3357_When%20Labelers%20Stay%20Silent_%20The%20Power%20of%20Ties%20in%20Cost-Effective%20Preference%20Learning/tables/78ba8640ddb1b600654123f720d8771baa3158f272a5da527536de8dc3876b3a.jpg)

![841ec3cbd8805064687c4d7dd53ba0b56e16515e7cf1769d860596a35ac0beea.jpg](../icml_results/3357_When%20Labelers%20Stay%20Silent_%20The%20Power%20of%20Ties%20in%20Cost-Effective%20Preference%20Learning/tables/841ec3cbd8805064687c4d7dd53ba0b56e16515e7cf1769d860596a35ac0beea.jpg)

![a0b535c378dad8ddfc8a5d5b2cdf1e767f507c8a8a9b9d82550ec1a66f52f558.jpg](../icml_results/3357_When%20Labelers%20Stay%20Silent_%20The%20Power%20of%20Ties%20in%20Cost-Effective%20Preference%20Learning/tables/a0b535c378dad8ddfc8a5d5b2cdf1e767f507c8a8a9b9d82550ec1a66f52f558.jpg)

![b6496d594c465937dd43110ebe748ecca16dd4a6b138214c2a1072f54d23d318.jpg](../icml_results/3357_When%20Labelers%20Stay%20Silent_%20The%20Power%20of%20Ties%20in%20Cost-Effective%20Preference%20Learning/tables/b6496d594c465937dd43110ebe748ecca16dd4a6b138214c2a1072f54d23d318.jpg)

![cbe15217e008348da959abf00ea55a5e0dad96216a5777c2bac8b8dac11e27b4.jpg](../icml_results/3357_When%20Labelers%20Stay%20Silent_%20The%20Power%20of%20Ties%20in%20Cost-Effective%20Preference%20Learning/tables/cbe15217e008348da959abf00ea55a5e0dad96216a5777c2bac8b8dac11e27b4.jpg)

![ed9efb96fa3576952042bbfdf560014ed63b21f6755d3ba89d84e2c14204cda9.jpg](../icml_results/3357_When%20Labelers%20Stay%20Silent_%20The%20Power%20of%20Ties%20in%20Cost-Effective%20Preference%20Learning/tables/ed9efb96fa3576952042bbfdf560014ed63b21f6755d3ba89d84e2c14204cda9.jpg)

![ef41c9addd4427717e57ed485b3e26439772efb2bf769c497aa7e26033cc3a5e.jpg](../icml_results/3357_When%20Labelers%20Stay%20Silent_%20The%20Power%20of%20Ties%20in%20Cost-Effective%20Preference%20Learning/tables/ef41c9addd4427717e57ed485b3e26439772efb2bf769c497aa7e26033cc3a5e.jpg)

![f9cbb044e77af829db2814537707cca473a06bbba9fc22f931c4d16897d4b1b7.jpg](../icml_results/3357_When%20Labelers%20Stay%20Silent_%20The%20Power%20of%20Ties%20in%20Cost-Effective%20Preference%20Learning/tables/f9cbb044e77af829db2814537707cca473a06bbba9fc22f931c4d16897d4b1b7.jpg)

<a id="paper-3319"></a>

## NaviCache: Test-Time Self-Calibration Caching for Video Generation


### Images

![053f51d8c87ca050ba1382c1b4c2312cd9586242f139362f83358b600fa19ee2.jpg](../icml_results/3358_NaviCache_%20Test-Time%20Self-Calibration%20Caching%20for%20Video%20Generation/images/053f51d8c87ca050ba1382c1b4c2312cd9586242f139362f83358b600fa19ee2.jpg)

![335ab398f79601a0b26d26471c5aee5830ee27c74c6929a2f2fc5c740f7978b8.jpg](../icml_results/3358_NaviCache_%20Test-Time%20Self-Calibration%20Caching%20for%20Video%20Generation/images/335ab398f79601a0b26d26471c5aee5830ee27c74c6929a2f2fc5c740f7978b8.jpg)

![3cf44909f22242b82e17995fe52543934271238f6f32dcd66e38baeff663854e.jpg](../icml_results/3358_NaviCache_%20Test-Time%20Self-Calibration%20Caching%20for%20Video%20Generation/images/3cf44909f22242b82e17995fe52543934271238f6f32dcd66e38baeff663854e.jpg)

![9ba87fde1f5c9671915725507faf7d0079dad945626bbd8fceefc1cd919ee684.jpg](../icml_results/3358_NaviCache_%20Test-Time%20Self-Calibration%20Caching%20for%20Video%20Generation/images/9ba87fde1f5c9671915725507faf7d0079dad945626bbd8fceefc1cd919ee684.jpg)

![9e2e373e0df2b284f80505103c9cbbb4c3469cb4ef8e600849f994beaabcd996.jpg](../icml_results/3358_NaviCache_%20Test-Time%20Self-Calibration%20Caching%20for%20Video%20Generation/images/9e2e373e0df2b284f80505103c9cbbb4c3469cb4ef8e600849f994beaabcd996.jpg)

![caf23c4974a67cace08b6c93f89c3f0e89f3d29203747566e878cc1ea6bd2b90.jpg](../icml_results/3358_NaviCache_%20Test-Time%20Self-Calibration%20Caching%20for%20Video%20Generation/images/caf23c4974a67cace08b6c93f89c3f0e89f3d29203747566e878cc1ea6bd2b90.jpg)

![cbb6852401c0fa250c0a9e0684718234ad80d95e61d7430a815c30efd2d4d44c.jpg](../icml_results/3358_NaviCache_%20Test-Time%20Self-Calibration%20Caching%20for%20Video%20Generation/images/cbb6852401c0fa250c0a9e0684718234ad80d95e61d7430a815c30efd2d4d44c.jpg)

### Tables

![0841144da1a3ba94b773e5c6221cf5f1efa6a8d192ae5ecbe5b807a580613fe3.jpg](../icml_results/3358_NaviCache_%20Test-Time%20Self-Calibration%20Caching%20for%20Video%20Generation/tables/0841144da1a3ba94b773e5c6221cf5f1efa6a8d192ae5ecbe5b807a580613fe3.jpg)

![82c26a6b7b2255eff46ba8589dc6c92dc3188a024e0bff542734a8646d178433.jpg](../icml_results/3358_NaviCache_%20Test-Time%20Self-Calibration%20Caching%20for%20Video%20Generation/tables/82c26a6b7b2255eff46ba8589dc6c92dc3188a024e0bff542734a8646d178433.jpg)

![a9991a7ab46c7b3d33096298d64ee8acb9c47dcf82bad9ac8a0a74e799751919.jpg](../icml_results/3358_NaviCache_%20Test-Time%20Self-Calibration%20Caching%20for%20Video%20Generation/tables/a9991a7ab46c7b3d33096298d64ee8acb9c47dcf82bad9ac8a0a74e799751919.jpg)

![c40d75ab7222c5ba8397816ce4ea72ad3fafb6fd64ba2edb71b8079efa31ba25.jpg](../icml_results/3358_NaviCache_%20Test-Time%20Self-Calibration%20Caching%20for%20Video%20Generation/tables/c40d75ab7222c5ba8397816ce4ea72ad3fafb6fd64ba2edb71b8079efa31ba25.jpg)

![d6dc4f2548e677549c638e24327000485db8c3f8874da442e5a08bd953c66fed.jpg](../icml_results/3358_NaviCache_%20Test-Time%20Self-Calibration%20Caching%20for%20Video%20Generation/tables/d6dc4f2548e677549c638e24327000485db8c3f8874da442e5a08bd953c66fed.jpg)

<a id="paper-3320"></a>

## General Covariant Action Modeling: Constructing Generalized Manifolds via Spatio-Temporal Decoupling


### Images

![585df4a8757f40f0ada4a2c7e407e528d27f3d9bd72410d3317b69fdd7286461.jpg](../icml_results/3359_General%20Covariant%20Action%20Modeling_%20Constructing%20Generalized%20Manifolds%20via%20Spatio-Temporal%20Decoupling/images/585df4a8757f40f0ada4a2c7e407e528d27f3d9bd72410d3317b69fdd7286461.jpg)

![5a80db346605d2261fdc5033bd67f300acaa1dbcb4de58467a41794fbd24619b.jpg](../icml_results/3359_General%20Covariant%20Action%20Modeling_%20Constructing%20Generalized%20Manifolds%20via%20Spatio-Temporal%20Decoupling/images/5a80db346605d2261fdc5033bd67f300acaa1dbcb4de58467a41794fbd24619b.jpg)

![6be8ed85ba2ce661a9da000693166ef4cb07f9dcfd9b5c2e0a29fbc762719204.jpg](../icml_results/3359_General%20Covariant%20Action%20Modeling_%20Constructing%20Generalized%20Manifolds%20via%20Spatio-Temporal%20Decoupling/images/6be8ed85ba2ce661a9da000693166ef4cb07f9dcfd9b5c2e0a29fbc762719204.jpg)

![9fa733ae5f7466cd402cef7da4529aecf6655afefef82a242bd2ea9c1d990575.jpg](../icml_results/3359_General%20Covariant%20Action%20Modeling_%20Constructing%20Generalized%20Manifolds%20via%20Spatio-Temporal%20Decoupling/images/9fa733ae5f7466cd402cef7da4529aecf6655afefef82a242bd2ea9c1d990575.jpg)

![b9f8219e9c59f6ec4f07ca4810284f8605561b582d7638c8fa9329b3499dc269.jpg](../icml_results/3359_General%20Covariant%20Action%20Modeling_%20Constructing%20Generalized%20Manifolds%20via%20Spatio-Temporal%20Decoupling/images/b9f8219e9c59f6ec4f07ca4810284f8605561b582d7638c8fa9329b3499dc269.jpg)

![edcd970deff3d4881578fb228332fb0eb3347127345398451d972970b4837317.jpg](../icml_results/3359_General%20Covariant%20Action%20Modeling_%20Constructing%20Generalized%20Manifolds%20via%20Spatio-Temporal%20Decoupling/images/edcd970deff3d4881578fb228332fb0eb3347127345398451d972970b4837317.jpg)

![f21d1035a53eb000207ebe50322df172d69d4139cf22b2339a842ee58579cd16.jpg](../icml_results/3359_General%20Covariant%20Action%20Modeling_%20Constructing%20Generalized%20Manifolds%20via%20Spatio-Temporal%20Decoupling/images/f21d1035a53eb000207ebe50322df172d69d4139cf22b2339a842ee58579cd16.jpg)

### Tables

![0991cca49f923182843378993a6eb4c64990472c0048c4bf60e4e5067b74e779.jpg](../icml_results/3359_General%20Covariant%20Action%20Modeling_%20Constructing%20Generalized%20Manifolds%20via%20Spatio-Temporal%20Decoupling/tables/0991cca49f923182843378993a6eb4c64990472c0048c4bf60e4e5067b74e779.jpg)

![464629cb3f75f5f4286021850faeb912e2a91e36f3f5d3fe55da67abf2208790.jpg](../icml_results/3359_General%20Covariant%20Action%20Modeling_%20Constructing%20Generalized%20Manifolds%20via%20Spatio-Temporal%20Decoupling/tables/464629cb3f75f5f4286021850faeb912e2a91e36f3f5d3fe55da67abf2208790.jpg)

![50727b6b4b20bccf00181938aacf9a3cd7424c1c7d6a6b762c8e2ee42b991e54.jpg](../icml_results/3359_General%20Covariant%20Action%20Modeling_%20Constructing%20Generalized%20Manifolds%20via%20Spatio-Temporal%20Decoupling/tables/50727b6b4b20bccf00181938aacf9a3cd7424c1c7d6a6b762c8e2ee42b991e54.jpg)

![6fb2aadc785780e6e580f7225baa019545afa4ca2dfa0ad51350dcf589a72ad4.jpg](../icml_results/3359_General%20Covariant%20Action%20Modeling_%20Constructing%20Generalized%20Manifolds%20via%20Spatio-Temporal%20Decoupling/tables/6fb2aadc785780e6e580f7225baa019545afa4ca2dfa0ad51350dcf589a72ad4.jpg)

![8fa08d3f59cb097f7ae06de72541202c24d3ef4a59aee8eb1cbf8d1a4a634555.jpg](../icml_results/3359_General%20Covariant%20Action%20Modeling_%20Constructing%20Generalized%20Manifolds%20via%20Spatio-Temporal%20Decoupling/tables/8fa08d3f59cb097f7ae06de72541202c24d3ef4a59aee8eb1cbf8d1a4a634555.jpg)

![922c18c50cf96ff8fe451cc3566f4875c26a34548dde18e6967e2618acd38ae2.jpg](../icml_results/3359_General%20Covariant%20Action%20Modeling_%20Constructing%20Generalized%20Manifolds%20via%20Spatio-Temporal%20Decoupling/tables/922c18c50cf96ff8fe451cc3566f4875c26a34548dde18e6967e2618acd38ae2.jpg)

![c22f06f0ada493a0ff469b1eba891b00f51c62bff2bc4aba6e6bd50a904d7759.jpg](../icml_results/3359_General%20Covariant%20Action%20Modeling_%20Constructing%20Generalized%20Manifolds%20via%20Spatio-Temporal%20Decoupling/tables/c22f06f0ada493a0ff469b1eba891b00f51c62bff2bc4aba6e6bd50a904d7759.jpg)

![ce5bc05a959995ba26de374b3fafdafaaec7caa57216714c532f582ccf856e85.jpg](../icml_results/3359_General%20Covariant%20Action%20Modeling_%20Constructing%20Generalized%20Manifolds%20via%20Spatio-Temporal%20Decoupling/tables/ce5bc05a959995ba26de374b3fafdafaaec7caa57216714c532f582ccf856e85.jpg)

![d50462c4ce558158b656b7101252544df72525e8b8b57116496156157f04396a.jpg](../icml_results/3359_General%20Covariant%20Action%20Modeling_%20Constructing%20Generalized%20Manifolds%20via%20Spatio-Temporal%20Decoupling/tables/d50462c4ce558158b656b7101252544df72525e8b8b57116496156157f04396a.jpg)

<a id="paper-3321"></a>

## LAST: Bridging Vision-Language and Action Manifolds via Gromov-Wasserstein Alignment


### Images

![29cde8c8602564a9d831feced05112ed4d7180441c6571460ffd357bbf1dcfac.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/images/29cde8c8602564a9d831feced05112ed4d7180441c6571460ffd357bbf1dcfac.jpg)

![3f22c9aee093c84e43f97c87412c6efe149eaebba01ef9b32cd81076db1559d7.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/images/3f22c9aee093c84e43f97c87412c6efe149eaebba01ef9b32cd81076db1559d7.jpg)

![5a915a92f7b967e0e3df624b47dd49a2c0449d52d682a894b0f9db4652261bd4.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/images/5a915a92f7b967e0e3df624b47dd49a2c0449d52d682a894b0f9db4652261bd4.jpg)

![6feff84dbcbe2bd8cc7d88a6e6a804c7943cfd56dfd16cb0fe8c50a75cf8f8ac.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/images/6feff84dbcbe2bd8cc7d88a6e6a804c7943cfd56dfd16cb0fe8c50a75cf8f8ac.jpg)

![8cb8c1d6a6b604acc7870b7f2687bd4857e4b0a7c3d7191f2e717ef60ab7be90.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/images/8cb8c1d6a6b604acc7870b7f2687bd4857e4b0a7c3d7191f2e717ef60ab7be90.jpg)

![926e25be62e0f929926b9850a38516123844bf84f47b7a41cd0b7435b0835848.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/images/926e25be62e0f929926b9850a38516123844bf84f47b7a41cd0b7435b0835848.jpg)

![a33087920c89e4a8168b230013505dae2cde54a3b91868a54b7b87a9099fca19.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/images/a33087920c89e4a8168b230013505dae2cde54a3b91868a54b7b87a9099fca19.jpg)

![a4cd04393f425b8061a73b64690aa58aa48611d3afcad295e35489c317b06151.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/images/a4cd04393f425b8061a73b64690aa58aa48611d3afcad295e35489c317b06151.jpg)

![b4283b8d8c663462a18d1d2f1b3dea733bdf86978bae3763229361d69625f847.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/images/b4283b8d8c663462a18d1d2f1b3dea733bdf86978bae3763229361d69625f847.jpg)

![b589b82533d930f5d0d8ba87b14a6b0dfc4d08f21257a9d3cbab545b0fe3e9e2.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/images/b589b82533d930f5d0d8ba87b14a6b0dfc4d08f21257a9d3cbab545b0fe3e9e2.jpg)

![e6254af8663429188046d04b7e0caa6377136453e2f307ee6da4d6416566d4b4.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/images/e6254af8663429188046d04b7e0caa6377136453e2f307ee6da4d6416566d4b4.jpg)

### Tables

![0305484880d2d9113e94476fada09140584e53c62d4160da673d3a37bfe60790.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/tables/0305484880d2d9113e94476fada09140584e53c62d4160da673d3a37bfe60790.jpg)

![1bd67966a9f3d6a66c46716fad6f60116cb0f13afb70c4b97790244c54b3ba1c.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/tables/1bd67966a9f3d6a66c46716fad6f60116cb0f13afb70c4b97790244c54b3ba1c.jpg)

![36fa4f25a8d12ffc91c8426afed8e8d39d73c47f46dd301c254b1cb7a10c1f5a.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/tables/36fa4f25a8d12ffc91c8426afed8e8d39d73c47f46dd301c254b1cb7a10c1f5a.jpg)

![411dfd70be62fbe3a6807998fd92e478c4e9b3894d03576cc6391c3586ba98f3.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/tables/411dfd70be62fbe3a6807998fd92e478c4e9b3894d03576cc6391c3586ba98f3.jpg)

![42fabdb890ccce566f7c580929b7b9ef4791d2f3be99485446bd6fd033f7d4c9.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/tables/42fabdb890ccce566f7c580929b7b9ef4791d2f3be99485446bd6fd033f7d4c9.jpg)

![640666f19ea28271f6785183f48dafc21c6cf4f8f4064c1469045c68ceaa0bbe.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/tables/640666f19ea28271f6785183f48dafc21c6cf4f8f4064c1469045c68ceaa0bbe.jpg)

![7788a11e6ea0a34174533238ffe6a5a00ffe4e3f6afb764ae2ccc40d4273a812.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/tables/7788a11e6ea0a34174533238ffe6a5a00ffe4e3f6afb764ae2ccc40d4273a812.jpg)

![8820bf7dba42acfbd61e4b40e05ad30192c012b289cb74c256340e0631f6b6de.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/tables/8820bf7dba42acfbd61e4b40e05ad30192c012b289cb74c256340e0631f6b6de.jpg)

![93eb2dd1120ffde9808d39561b3af212f7d03b7b8e33f797aa5e5d46bc94ecfc.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/tables/93eb2dd1120ffde9808d39561b3af212f7d03b7b8e33f797aa5e5d46bc94ecfc.jpg)

![9bd6e10782f5af5b2a1a28db5b055c8fe1083ce6cd002b88fd357ef1e47bcf49.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/tables/9bd6e10782f5af5b2a1a28db5b055c8fe1083ce6cd002b88fd357ef1e47bcf49.jpg)

![ac490fc9cd075e05aef15027fef29d3d4698b1a0ecaeb09ed227e5e16d10d42f.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/tables/ac490fc9cd075e05aef15027fef29d3d4698b1a0ecaeb09ed227e5e16d10d42f.jpg)

![b91a90730007cdca1bf0ae963c3dc54b49bdadcebf20ee8561c0c4bd001d34c6.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/tables/b91a90730007cdca1bf0ae963c3dc54b49bdadcebf20ee8561c0c4bd001d34c6.jpg)

![ce54e9a2c6c4c396f8858cb8e32a59915ad94e79410b78085a9bd301994f03a7.jpg](../icml_results/3360_LAST_%20Bridging%20Vision-Language%20and%20Action%20Manifolds%20via%20Gromov-Wasserstein%20Alignment/tables/ce54e9a2c6c4c396f8858cb8e32a59915ad94e79410b78085a9bd301994f03a7.jpg)

<a id="paper-3322"></a>

## QiMeng-ChipV-RTL: Exploiting Information Locality for IP-level Verilog Generation


### Images

![0044809a84ca8d3e87da0b6ad088d85e1e88cf9ad1369d48babe72c4e797465d.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/images/0044809a84ca8d3e87da0b6ad088d85e1e88cf9ad1369d48babe72c4e797465d.jpg)

![091a4f5bf6bd47da13958285ac97b3eab9486f85ed5ad09de68c73a6567b3781.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/images/091a4f5bf6bd47da13958285ac97b3eab9486f85ed5ad09de68c73a6567b3781.jpg)

![5bd97ae0aa3bc8ffab3976dc8815b3a092e1254663ce056da51b56b630dc2553.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/images/5bd97ae0aa3bc8ffab3976dc8815b3a092e1254663ce056da51b56b630dc2553.jpg)

![64b2c06edeef0792080fb526fba1ed64d43fe213728ab6b9c7a5942eead78b64.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/images/64b2c06edeef0792080fb526fba1ed64d43fe213728ab6b9c7a5942eead78b64.jpg)

![68a1e77d1402edd4f69f6e7f1f0e8edec1eddde28093403863b2d891c6929863.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/images/68a1e77d1402edd4f69f6e7f1f0e8edec1eddde28093403863b2d891c6929863.jpg)

![80061d7475929355f0acc601f0c5ba6ce992b178b0c075aeb5e359d9dac081ff.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/images/80061d7475929355f0acc601f0c5ba6ce992b178b0c075aeb5e359d9dac081ff.jpg)

![88b55c4e2908484631b4ab4a0bdd7e30c468228383c3eedc20a104dd178e82d3.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/images/88b55c4e2908484631b4ab4a0bdd7e30c468228383c3eedc20a104dd178e82d3.jpg)

![9b4f39336ae41d83af3418d75d73791e33cc47fc7e5986e6986f35580306643d.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/images/9b4f39336ae41d83af3418d75d73791e33cc47fc7e5986e6986f35580306643d.jpg)

![a6b71dc3541a9a9dceb529515ac1c2193c84911ecb6d8ac5cdf3711a1e7fab8f.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/images/a6b71dc3541a9a9dceb529515ac1c2193c84911ecb6d8ac5cdf3711a1e7fab8f.jpg)

![bf451fddeaa93f8290bec5573912a2079da10a05d80ae05f3c1fe192ac96fa01.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/images/bf451fddeaa93f8290bec5573912a2079da10a05d80ae05f3c1fe192ac96fa01.jpg)

![d816ff3f7b6131a0724ec5bbfb00cd8d61eaae7cd4ac2e91c9e1f810f260214a.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/images/d816ff3f7b6131a0724ec5bbfb00cd8d61eaae7cd4ac2e91c9e1f810f260214a.jpg)

![dab7d0b286e2ecc8afb346766c2749bc6e6061fd2f3ab29cde7da89ebd296123.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/images/dab7d0b286e2ecc8afb346766c2749bc6e6061fd2f3ab29cde7da89ebd296123.jpg)

### Tables

![09076f51fc96d974d3650f3d7150507335f17f39d3347fad5f73a1430906e36d.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/tables/09076f51fc96d974d3650f3d7150507335f17f39d3347fad5f73a1430906e36d.jpg)

![0f81ecfdb30a19f721cd73c0b47b4ca5101942b35259bf37e8f3976678d87848.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/tables/0f81ecfdb30a19f721cd73c0b47b4ca5101942b35259bf37e8f3976678d87848.jpg)

![37cec972910d020892e08f0209852ad8392a0b1a8756feb7770122ea15813a59.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/tables/37cec972910d020892e08f0209852ad8392a0b1a8756feb7770122ea15813a59.jpg)

![4a8972423b1468124713acaefb12892ab62a3dbd2160a03f6b8b6de8b98a5b5a.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/tables/4a8972423b1468124713acaefb12892ab62a3dbd2160a03f6b8b6de8b98a5b5a.jpg)

![6aa6f8548ceb91146c0756ba78a6b92fd44419298d332304641dffa3c9958481.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/tables/6aa6f8548ceb91146c0756ba78a6b92fd44419298d332304641dffa3c9958481.jpg)

![6d8fa4d8eae89dc551ac67904727027b3e5f6053c01b9eb9cc71ef7a56443fef.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/tables/6d8fa4d8eae89dc551ac67904727027b3e5f6053c01b9eb9cc71ef7a56443fef.jpg)

![7a65c56bd6deb09b9cc775972e371fe3d776267e3932dc07a102d341c32f872e.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/tables/7a65c56bd6deb09b9cc775972e371fe3d776267e3932dc07a102d341c32f872e.jpg)

![8c070af4e0cb21e571f8b67c29d765075207ea9bd202aac08ff63b96d1073307.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/tables/8c070af4e0cb21e571f8b67c29d765075207ea9bd202aac08ff63b96d1073307.jpg)

![9e7546c5f30341bbb8d197305cc687d90b8b3d2ffdf05e73728904fe80f444a9.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/tables/9e7546c5f30341bbb8d197305cc687d90b8b3d2ffdf05e73728904fe80f444a9.jpg)

![9f1d1edeee82172b66eef0cd60e111e90cf62d9a89df315e6272432f939627c4.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/tables/9f1d1edeee82172b66eef0cd60e111e90cf62d9a89df315e6272432f939627c4.jpg)

![a9f45859577f4b7c388d120b877e2b7391edf3cc3055889c78a2815548334fb9.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/tables/a9f45859577f4b7c388d120b877e2b7391edf3cc3055889c78a2815548334fb9.jpg)

![b84146eaaa40bed8a67719f2fe16c3cfa258c868706fc0f7c0ea7bc379369a31.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/tables/b84146eaaa40bed8a67719f2fe16c3cfa258c868706fc0f7c0ea7bc379369a31.jpg)

![c2013d44f4a5e1912ecf0fc185576bd7c3be91746a8c2d25cf013f05381eadf6.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/tables/c2013d44f4a5e1912ecf0fc185576bd7c3be91746a8c2d25cf013f05381eadf6.jpg)

![dafaf72c41f864ec6e77e0446f3abff63d0b151584f32a596af2050829a5165a.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/tables/dafaf72c41f864ec6e77e0446f3abff63d0b151584f32a596af2050829a5165a.jpg)

![de0f72bb8a4ce2b7dd4699c2ed7139729d7866bce586e52c6fd65ff8922080f9.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/tables/de0f72bb8a4ce2b7dd4699c2ed7139729d7866bce586e52c6fd65ff8922080f9.jpg)

![ed9bcef9aa43b17e88b666d2eb9fcba50cd6e18559504d740885ee242a6b7b2c.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/tables/ed9bcef9aa43b17e88b666d2eb9fcba50cd6e18559504d740885ee242a6b7b2c.jpg)

![fc6f89044257d0f088130e6086b7bf0551d5c2245ef871925cdb8e6542371551.jpg](../icml_results/3361_QiMeng-ChipV-RTL_%20Exploiting%20Information%20Locality%20for%20IP-level%20Verilog%20Generation/tables/fc6f89044257d0f088130e6086b7bf0551d5c2245ef871925cdb8e6542371551.jpg)

<a id="paper-3323"></a>

## Transformers with RL or SFT Provably Learn Sparse Boolean Functions, But Differently


### Images

![2797e368e232a2e36eec6b8946ad31a763aac22466985cbc33aaef5095148a19.jpg](../icml_results/3362_Transformers%20with%20RL%20or%20SFT%20Provably%20Learn%20Sparse%20Boolean%20Functions%2C%20But%20Differently/images/2797e368e232a2e36eec6b8946ad31a763aac22466985cbc33aaef5095148a19.jpg)

![42260ed84abcde7e14f4f1ce6d02169b6201b8b50a20a4381010509c72ae990b.jpg](../icml_results/3362_Transformers%20with%20RL%20or%20SFT%20Provably%20Learn%20Sparse%20Boolean%20Functions%2C%20But%20Differently/images/42260ed84abcde7e14f4f1ce6d02169b6201b8b50a20a4381010509c72ae990b.jpg)

![8b0bf8760cbb7e1f675beb0d81d1d32375f0d655bb1a5dc83238534c37bf5c96.jpg](../icml_results/3362_Transformers%20with%20RL%20or%20SFT%20Provably%20Learn%20Sparse%20Boolean%20Functions%2C%20But%20Differently/images/8b0bf8760cbb7e1f675beb0d81d1d32375f0d655bb1a5dc83238534c37bf5c96.jpg)

![a3b3ea2c31c02067e295c8375688ad8ec7912abcea91d79a6e602500527a2811.jpg](../icml_results/3362_Transformers%20with%20RL%20or%20SFT%20Provably%20Learn%20Sparse%20Boolean%20Functions%2C%20But%20Differently/images/a3b3ea2c31c02067e295c8375688ad8ec7912abcea91d79a6e602500527a2811.jpg)

![c402653438240e28d60073f4acc1453942c1f9d3726fbb72f460ffd9da7c56c8.jpg](../icml_results/3362_Transformers%20with%20RL%20or%20SFT%20Provably%20Learn%20Sparse%20Boolean%20Functions%2C%20But%20Differently/images/c402653438240e28d60073f4acc1453942c1f9d3726fbb72f460ffd9da7c56c8.jpg)

![d8d4c0e4e86cc7dc7785b0642d82755d06cb2905f1d1abc3848087099fb278b7.jpg](../icml_results/3362_Transformers%20with%20RL%20or%20SFT%20Provably%20Learn%20Sparse%20Boolean%20Functions%2C%20But%20Differently/images/d8d4c0e4e86cc7dc7785b0642d82755d06cb2905f1d1abc3848087099fb278b7.jpg)

![e630051399c668a7062eef2852368180b62abd8d1cb58d60b6f471ff27b4c18c.jpg](../icml_results/3362_Transformers%20with%20RL%20or%20SFT%20Provably%20Learn%20Sparse%20Boolean%20Functions%2C%20But%20Differently/images/e630051399c668a7062eef2852368180b62abd8d1cb58d60b6f471ff27b4c18c.jpg)

![f498393aaebffabfe85aaa005affc2afe62c021c61188fb17011c48413f25ac2.jpg](../icml_results/3362_Transformers%20with%20RL%20or%20SFT%20Provably%20Learn%20Sparse%20Boolean%20Functions%2C%20But%20Differently/images/f498393aaebffabfe85aaa005affc2afe62c021c61188fb17011c48413f25ac2.jpg)

### Tables

![3975745b9f4781c04fd8abfc972c89083710c2197059c08682391887de0a4fe0.jpg](../icml_results/3362_Transformers%20with%20RL%20or%20SFT%20Provably%20Learn%20Sparse%20Boolean%20Functions%2C%20But%20Differently/tables/3975745b9f4781c04fd8abfc972c89083710c2197059c08682391887de0a4fe0.jpg)

![872eb74054688ba9a1645dec5a5ac4ffba65af750152788e5456de3201e2217f.jpg](../icml_results/3362_Transformers%20with%20RL%20or%20SFT%20Provably%20Learn%20Sparse%20Boolean%20Functions%2C%20But%20Differently/tables/872eb74054688ba9a1645dec5a5ac4ffba65af750152788e5456de3201e2217f.jpg)

<a id="paper-3324"></a>

## Efficiently Solving Discounted MDPs via Predictions with Unknown Prediction Errors


### Tables

![68afe9c1ae2c88eea6a049858e837334145a8b320071abfe96db183fd3b5e50e.jpg](../icml_results/3363_Efficiently%20Solving%20Discounted%20MDPs%20via%20Predictions%20with%20Unknown%20Prediction%20Errors/tables/68afe9c1ae2c88eea6a049858e837334145a8b320071abfe96db183fd3b5e50e.jpg)

<a id="paper-3325"></a>

## Debiased Model-based Representations for Sample-efficient Continuous Control


### Images

![0e611c225f5f2c084ad5e58e2fa567bb431e5bc1a4862e9c71666c1a1d8cc8a5.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/images/0e611c225f5f2c084ad5e58e2fa567bb431e5bc1a4862e9c71666c1a1d8cc8a5.jpg)

![13c28ecae0d386a34451bb53668465d3229d4d414b3cb5ff4f839ddfcc6a4990.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/images/13c28ecae0d386a34451bb53668465d3229d4d414b3cb5ff4f839ddfcc6a4990.jpg)

![226f86f57ff1b3817768eca17bf8a7641d98c61a456535ab031af51610c5934e.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/images/226f86f57ff1b3817768eca17bf8a7641d98c61a456535ab031af51610c5934e.jpg)

![246390972d2c754259606aa68651796a756185be3a4955561d795352b163bdcc.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/images/246390972d2c754259606aa68651796a756185be3a4955561d795352b163bdcc.jpg)

![29857e0839a6577e844ff02c6bdeade65770099afc8700d50eeed5a09b3907eb.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/images/29857e0839a6577e844ff02c6bdeade65770099afc8700d50eeed5a09b3907eb.jpg)

![2ed5dea9ab18d79db37cdbdd3942e7ab337b05728eb230b9ec774bc26f45a806.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/images/2ed5dea9ab18d79db37cdbdd3942e7ab337b05728eb230b9ec774bc26f45a806.jpg)

![47f0cbd1cac80c9e43d5cb00d622a6fc2db91b9dbc37d7c9031e2a85e6775017.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/images/47f0cbd1cac80c9e43d5cb00d622a6fc2db91b9dbc37d7c9031e2a85e6775017.jpg)

![4ede78ddc709142d86585975f3f392b72680c2c49b560227e67da775d9ed3975.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/images/4ede78ddc709142d86585975f3f392b72680c2c49b560227e67da775d9ed3975.jpg)

![985143f619641f82e5c4ea6192173d044c8f8531c2a351d17647925486777b0e.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/images/985143f619641f82e5c4ea6192173d044c8f8531c2a351d17647925486777b0e.jpg)

![afbd61eebaac1e0372a2eb4ff39a5d240b394a36f15bb12934f0298014f53299.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/images/afbd61eebaac1e0372a2eb4ff39a5d240b394a36f15bb12934f0298014f53299.jpg)

![c336b37113a332b6aa93f740de22709c35134be78501c32df6ad02ac3b3318d1.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/images/c336b37113a332b6aa93f740de22709c35134be78501c32df6ad02ac3b3318d1.jpg)

![c4513c4cda816aa02159e0fb6d86713fe46e26693d344774fdd55ecef1f08321.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/images/c4513c4cda816aa02159e0fb6d86713fe46e26693d344774fdd55ecef1f08321.jpg)

![ded434c1ffd55c661de156ebcab42bab9dc78cca59ed489bcf2f3add8dce1203.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/images/ded434c1ffd55c661de156ebcab42bab9dc78cca59ed489bcf2f3add8dce1203.jpg)

![e254931d6ab318dc773229b6a30e9b4cf4866a17dfeb1c6b56c71f624b42533b.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/images/e254931d6ab318dc773229b6a30e9b4cf4866a17dfeb1c6b56c71f624b42533b.jpg)

![ee7706839e7b1d139e387ec667ecff9cdad3c66a59e21e0de9fa5a74c7896a7d.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/images/ee7706839e7b1d139e387ec667ecff9cdad3c66a59e21e0de9fa5a74c7896a7d.jpg)

![efc7271a67a83190bca0b6e7248229d040f57eb733cd6a3d1e55b2b74f522140.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/images/efc7271a67a83190bca0b6e7248229d040f57eb733cd6a3d1e55b2b74f522140.jpg)

![f204af0476fb7e07c829bcd1add78ddf18b786622a5c705b632b38bed4ab08f9.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/images/f204af0476fb7e07c829bcd1add78ddf18b786622a5c705b632b38bed4ab08f9.jpg)

### Tables

![13c5dbdfcf0657388c0de2c8919aef1a78dfa08b626d324626da46b2c66a0ad1.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/tables/13c5dbdfcf0657388c0de2c8919aef1a78dfa08b626d324626da46b2c66a0ad1.jpg)

![24619f8d3beb661032d1e00eaafdc007639fd7a27c25de88817251c472754f8c.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/tables/24619f8d3beb661032d1e00eaafdc007639fd7a27c25de88817251c472754f8c.jpg)

![601990ae79e399ba55f2f0050db0d5b94d804ba0993f36a3b71882c393a0211e.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/tables/601990ae79e399ba55f2f0050db0d5b94d804ba0993f36a3b71882c393a0211e.jpg)

![6ed85a1d223ec4a93bdfa6b280f3bbff98a9b1f6255a7963d31b1c050f9b45d7.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/tables/6ed85a1d223ec4a93bdfa6b280f3bbff98a9b1f6255a7963d31b1c050f9b45d7.jpg)

![8bef2d031c6f391cc205063752a87a1e9146cf622a006b256ad0327c14c5c5eb.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/tables/8bef2d031c6f391cc205063752a87a1e9146cf622a006b256ad0327c14c5c5eb.jpg)

![8e99c76e465a4cca8ebe6928ff0a5be150bc8439305d38c031232875dcb0804a.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/tables/8e99c76e465a4cca8ebe6928ff0a5be150bc8439305d38c031232875dcb0804a.jpg)

![a82920395a407a484fe4575756cfcc5dd5742cde4fb3b6b068a86d6c00558cf3.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/tables/a82920395a407a484fe4575756cfcc5dd5742cde4fb3b6b068a86d6c00558cf3.jpg)

![c037ade500c7ab6eaf7d4c20f9518f84fc3c570a49e525a628e05d738b1572c6.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/tables/c037ade500c7ab6eaf7d4c20f9518f84fc3c570a49e525a628e05d738b1572c6.jpg)

![c7e115fe6dbdbb4fe98dd0b3b97fb50fa050f70b172a3e60a6380cc3791ba81e.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/tables/c7e115fe6dbdbb4fe98dd0b3b97fb50fa050f70b172a3e60a6380cc3791ba81e.jpg)

![d29e38697306ac6c105ed4f33bfc408d33a3dfb02add41ea905cfb8d43919591.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/tables/d29e38697306ac6c105ed4f33bfc408d33a3dfb02add41ea905cfb8d43919591.jpg)

![df67a61a7c9a08e61e0bf577d8abf758cc5bb3466a57003d6302eb69045d4a61.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/tables/df67a61a7c9a08e61e0bf577d8abf758cc5bb3466a57003d6302eb69045d4a61.jpg)

![f3f53d516ad57e0199235351409422bd0051aa40c369fc0349aab1afc7c8ae59.jpg](../icml_results/3364_Debiased%20Model-based%20Representations%20for%20Sample-efficient%20Continuous%20Control/tables/f3f53d516ad57e0199235351409422bd0051aa40c369fc0349aab1afc7c8ae59.jpg)

<a id="paper-3326"></a>

## On the Intrinsic Limits of Transformer Image Embeddings in Non-Solvable Spatial Reasoning


### Images

![5c69de760553ce42a198a339afe181f585dd7313db9eb74e0f3bb12c6d069691.jpg](../icml_results/3365_On%20the%20Intrinsic%20Limits%20of%20Transformer%20Image%20Embeddings%20in%20Non-Solvable%20Spatial%20Reasoning/images/5c69de760553ce42a198a339afe181f585dd7313db9eb74e0f3bb12c6d069691.jpg)

![9a207713ff696943ffd3c123fa08a912782f51cbb677e14577eeb1402d7d8c4e.jpg](../icml_results/3365_On%20the%20Intrinsic%20Limits%20of%20Transformer%20Image%20Embeddings%20in%20Non-Solvable%20Spatial%20Reasoning/images/9a207713ff696943ffd3c123fa08a912782f51cbb677e14577eeb1402d7d8c4e.jpg)

![e4395d14296022e46d3d95fd3a767d28d94629e7d96e6f6bf0d76694ab9765ee.jpg](../icml_results/3365_On%20the%20Intrinsic%20Limits%20of%20Transformer%20Image%20Embeddings%20in%20Non-Solvable%20Spatial%20Reasoning/images/e4395d14296022e46d3d95fd3a767d28d94629e7d96e6f6bf0d76694ab9765ee.jpg)

![efa45f58e22d8d42941268876fe2b28cb24b72cbd6ca94540a0c29936d34e51c.jpg](../icml_results/3365_On%20the%20Intrinsic%20Limits%20of%20Transformer%20Image%20Embeddings%20in%20Non-Solvable%20Spatial%20Reasoning/images/efa45f58e22d8d42941268876fe2b28cb24b72cbd6ca94540a0c29936d34e51c.jpg)

### Tables

![3bfaaf15e88ab96cfc01935605810adab276dd1a8aa7ebc786b0de4e7cf81e31.jpg](../icml_results/3365_On%20the%20Intrinsic%20Limits%20of%20Transformer%20Image%20Embeddings%20in%20Non-Solvable%20Spatial%20Reasoning/tables/3bfaaf15e88ab96cfc01935605810adab276dd1a8aa7ebc786b0de4e7cf81e31.jpg)

![683301ec3eb8cc63eb25e540f825e4ef5efe1aa9f6473d5e5a87dbc59f148daf.jpg](../icml_results/3365_On%20the%20Intrinsic%20Limits%20of%20Transformer%20Image%20Embeddings%20in%20Non-Solvable%20Spatial%20Reasoning/tables/683301ec3eb8cc63eb25e540f825e4ef5efe1aa9f6473d5e5a87dbc59f148daf.jpg)

![b81db7be512592a825dfc7005d52ab5ddb5d99f2c5ceff40da0b3f6fe80ebf46.jpg](../icml_results/3365_On%20the%20Intrinsic%20Limits%20of%20Transformer%20Image%20Embeddings%20in%20Non-Solvable%20Spatial%20Reasoning/tables/b81db7be512592a825dfc7005d52ab5ddb5d99f2c5ceff40da0b3f6fe80ebf46.jpg)

![ef9f973a2d7f693eb72f2b9d4454ae1d0413eb1b2ad3290dec32605b8e43c7e6.jpg](../icml_results/3365_On%20the%20Intrinsic%20Limits%20of%20Transformer%20Image%20Embeddings%20in%20Non-Solvable%20Spatial%20Reasoning/tables/ef9f973a2d7f693eb72f2b9d4454ae1d0413eb1b2ad3290dec32605b8e43c7e6.jpg)

<a id="paper-3327"></a>

## Cutting LLM Evaluation Costs with SySRs: A Bandit Algorithm That Provably Exploits Model Similarity


### Images

![028cb6d2720f39961dc5f17c026462eec9f92af9af142e6f40721d6a52204945.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/images/028cb6d2720f39961dc5f17c026462eec9f92af9af142e6f40721d6a52204945.jpg)

![03ddf90829e6506673a731578cccc86982dee8aafecb72777b2f113c17a865c3.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/images/03ddf90829e6506673a731578cccc86982dee8aafecb72777b2f113c17a865c3.jpg)

![1aeb67123125c18a41b8be9551a70acf34c78fe4cd31df3a1be112e631b2add0.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/images/1aeb67123125c18a41b8be9551a70acf34c78fe4cd31df3a1be112e631b2add0.jpg)

![313b864d79a5c90058126e2e7fcf04c163f7d5105f5f13d845b420eb00b0af7e.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/images/313b864d79a5c90058126e2e7fcf04c163f7d5105f5f13d845b420eb00b0af7e.jpg)

![42c7142419ebdfad4db72ab929c55b07c3380ce2ea1d7272e0222988e93ea1b2.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/images/42c7142419ebdfad4db72ab929c55b07c3380ce2ea1d7272e0222988e93ea1b2.jpg)

![7e4f8a8f3c3a2ae8b4fae3a525c0070c1d87785f658f19e19a5f8be673035c60.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/images/7e4f8a8f3c3a2ae8b4fae3a525c0070c1d87785f658f19e19a5f8be673035c60.jpg)

![849c57efd505bab1ca96fe33e84d0629d11af112f1ab50b664cb94bb9b0eb8fb.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/images/849c57efd505bab1ca96fe33e84d0629d11af112f1ab50b664cb94bb9b0eb8fb.jpg)

![aaa9bf28533b6f439a15fcb20e3f57816eea7364c2d8087d3dc16e73435935be.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/images/aaa9bf28533b6f439a15fcb20e3f57816eea7364c2d8087d3dc16e73435935be.jpg)

![ab9986180ecc3a59b74a70b0b16fc8d27d2d13e0ffe3ef642c845668dc1e49e2.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/images/ab9986180ecc3a59b74a70b0b16fc8d27d2d13e0ffe3ef642c845668dc1e49e2.jpg)

![b3a6952b9d7cefb4308db55e380e665d43cf989f6f7e6a2aa175615eade8bfb7.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/images/b3a6952b9d7cefb4308db55e380e665d43cf989f6f7e6a2aa175615eade8bfb7.jpg)

![c2517d21265fe7c73c85265e400c636cf9894358d08005f2cbb7bf8b78528e54.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/images/c2517d21265fe7c73c85265e400c636cf9894358d08005f2cbb7bf8b78528e54.jpg)

![c932ba5d852d1efd8ce31c0f6764fdf761fe5a9d20d102167a1ebd36cb0bd135.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/images/c932ba5d852d1efd8ce31c0f6764fdf761fe5a9d20d102167a1ebd36cb0bd135.jpg)

![d83dac2eeb4d183e473fc8e6cfc1478ec590cab0c6ea3089bfb620ab045a4b15.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/images/d83dac2eeb4d183e473fc8e6cfc1478ec590cab0c6ea3089bfb620ab045a4b15.jpg)

![e192047e9e07738315dc392c89e508743caf848758bcbaba9c0893832ca2618c.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/images/e192047e9e07738315dc392c89e508743caf848758bcbaba9c0893832ca2618c.jpg)

![e935854d197c1a19cc29e52621edc5d92ea2f84e137917d0d0807cee10570d37.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/images/e935854d197c1a19cc29e52621edc5d92ea2f84e137917d0d0807cee10570d37.jpg)

![f78b836e6a1694bf14eec870f817a304ca635ee5d61b962fdadc3f9685b10e97.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/images/f78b836e6a1694bf14eec870f817a304ca635ee5d61b962fdadc3f9685b10e97.jpg)

![f91336fb12dbed3ab16e629ffde73afad16f773dd87f678c0b3d83fd21c3c3ad.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/images/f91336fb12dbed3ab16e629ffde73afad16f773dd87f678c0b3d83fd21c3c3ad.jpg)

### Tables

![06be49ac2a1d278472be6df9b54a03d858656377edf7efaf3a0e5e75d48a9d28.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/tables/06be49ac2a1d278472be6df9b54a03d858656377edf7efaf3a0e5e75d48a9d28.jpg)

![7930f6f59ffedbe38ea8b020ef965515de03c61209a68dbd532c73830fe0f1ce.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/tables/7930f6f59ffedbe38ea8b020ef965515de03c61209a68dbd532c73830fe0f1ce.jpg)

![7b58abd184c18d62d5623a4aa0b87f97be6774e04aff10049b694301d2028ac5.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/tables/7b58abd184c18d62d5623a4aa0b87f97be6774e04aff10049b694301d2028ac5.jpg)

![a9e01f381ea1ad23d6b3b637f66092269d77563dc4a307c481db6a7a5b9091ac.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/tables/a9e01f381ea1ad23d6b3b637f66092269d77563dc4a307c481db6a7a5b9091ac.jpg)

![b88eb45972ebd1c44cfcb8f40a65bd3c3204b5f44544af486e3f1f999d6bc424.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/tables/b88eb45972ebd1c44cfcb8f40a65bd3c3204b5f44544af486e3f1f999d6bc424.jpg)

![e49f4240ec129ac391bdd75407e300f7209f56448ecaedf9328d06ad666326e8.jpg](../icml_results/3366_Cutting%20LLM%20Evaluation%20Costs%20with%20SySRs_%20A%20Bandit%20Algorithm%20That%20Provably%20Exploits%20Model%20Similarity/tables/e49f4240ec129ac391bdd75407e300f7209f56448ecaedf9328d06ad666326e8.jpg)

<a id="paper-3328"></a>

## Explaining Concept Shift with Interpretable Feature Attribution


### Images

![7eaf2efb2d8df72454de3a1d2d5cf15d60d4db7c832980a9114a6cb0e3a76267.jpg](../icml_results/3367_Explaining%20Concept%20Shift%20with%20Interpretable%20Feature%20Attribution/images/7eaf2efb2d8df72454de3a1d2d5cf15d60d4db7c832980a9114a6cb0e3a76267.jpg)

![c034cbd9b01089980b3cc6a8b71fb8ae4de58a5fff4a5bd8bf9f7e16780bbe6f.jpg](../icml_results/3367_Explaining%20Concept%20Shift%20with%20Interpretable%20Feature%20Attribution/images/c034cbd9b01089980b3cc6a8b71fb8ae4de58a5fff4a5bd8bf9f7e16780bbe6f.jpg)

![c78619a30b01f2f517373f0c093cae8b88b3d89c833432fe31247ff7f0a59ade.jpg](../icml_results/3367_Explaining%20Concept%20Shift%20with%20Interpretable%20Feature%20Attribution/images/c78619a30b01f2f517373f0c093cae8b88b3d89c833432fe31247ff7f0a59ade.jpg)

![cffb71b6bf8e6e8c8585229c0c56af4c1b105ad269e65e1ab72723f4ca4c1095.jpg](../icml_results/3367_Explaining%20Concept%20Shift%20with%20Interpretable%20Feature%20Attribution/images/cffb71b6bf8e6e8c8585229c0c56af4c1b105ad269e65e1ab72723f4ca4c1095.jpg)

![d86994d2712f905d5e9bc91e1dd43a6f809c4b3c2ab0647ed4adce31e196baaa.jpg](../icml_results/3367_Explaining%20Concept%20Shift%20with%20Interpretable%20Feature%20Attribution/images/d86994d2712f905d5e9bc91e1dd43a6f809c4b3c2ab0647ed4adce31e196baaa.jpg)

### Tables

![3f619570ccf3ee658d095b284be5490c8044169c7d2f79c4d57800f1434a6938.jpg](../icml_results/3367_Explaining%20Concept%20Shift%20with%20Interpretable%20Feature%20Attribution/tables/3f619570ccf3ee658d095b284be5490c8044169c7d2f79c4d57800f1434a6938.jpg)

![55e71b6edbd362a64eb470b2d42f7b0fe9340b918674c8832780a1eade92e64b.jpg](../icml_results/3367_Explaining%20Concept%20Shift%20with%20Interpretable%20Feature%20Attribution/tables/55e71b6edbd362a64eb470b2d42f7b0fe9340b918674c8832780a1eade92e64b.jpg)

![66d327d1170fed502d19fd8d09468c6ee78d70d151d305d8726683f47b41c86f.jpg](../icml_results/3367_Explaining%20Concept%20Shift%20with%20Interpretable%20Feature%20Attribution/tables/66d327d1170fed502d19fd8d09468c6ee78d70d151d305d8726683f47b41c86f.jpg)

![8e94eba6f407e35896d924d452eef28dbfb834ee5097b2d1c15e2f091f59f36c.jpg](../icml_results/3367_Explaining%20Concept%20Shift%20with%20Interpretable%20Feature%20Attribution/tables/8e94eba6f407e35896d924d452eef28dbfb834ee5097b2d1c15e2f091f59f36c.jpg)

![a70cb990b1a4e86f96931ecbfe81b4608bd49b975795191ac2b90dbd6ad67baa.jpg](../icml_results/3367_Explaining%20Concept%20Shift%20with%20Interpretable%20Feature%20Attribution/tables/a70cb990b1a4e86f96931ecbfe81b4608bd49b975795191ac2b90dbd6ad67baa.jpg)

<a id="paper-3329"></a>

## Student-Centered Distillation Narrows the Agentic Gap Between Small and Large LLMs


### Images

![2eb616636e201af036eb7f95e7cd0459941580197168868e4497082effb1a19b.jpg](../icml_results/3368_Student-Centered%20Distillation%20Narrows%20the%20Agentic%20Gap%20Between%20Small%20and%20Large%20LLMs/images/2eb616636e201af036eb7f95e7cd0459941580197168868e4497082effb1a19b.jpg)

![4f5667b1029caea44090138f29958a884b456031a26eaec48cb4842e3e5b8ce4.jpg](../icml_results/3368_Student-Centered%20Distillation%20Narrows%20the%20Agentic%20Gap%20Between%20Small%20and%20Large%20LLMs/images/4f5667b1029caea44090138f29958a884b456031a26eaec48cb4842e3e5b8ce4.jpg)

![698968c89bd8f032eff43cbcbc9199a949387816ab8518c4ab2adca972d21a85.jpg](../icml_results/3368_Student-Centered%20Distillation%20Narrows%20the%20Agentic%20Gap%20Between%20Small%20and%20Large%20LLMs/images/698968c89bd8f032eff43cbcbc9199a949387816ab8518c4ab2adca972d21a85.jpg)

![78fcc8e513f26f772278762d7da52a9ed9186ed8ae4e61074e967d284f13463b.jpg](../icml_results/3368_Student-Centered%20Distillation%20Narrows%20the%20Agentic%20Gap%20Between%20Small%20and%20Large%20LLMs/images/78fcc8e513f26f772278762d7da52a9ed9186ed8ae4e61074e967d284f13463b.jpg)

![d6502b04a36be07c8ade7f8dd6ef7975ae6f9a07f3bff2a807d58c42b968b1b0.jpg](../icml_results/3368_Student-Centered%20Distillation%20Narrows%20the%20Agentic%20Gap%20Between%20Small%20and%20Large%20LLMs/images/d6502b04a36be07c8ade7f8dd6ef7975ae6f9a07f3bff2a807d58c42b968b1b0.jpg)

### Tables

![7c162101b9dcacd2afcb710d07452ba373bacbb07e09c4fa316c7846f8c019dd.jpg](../icml_results/3368_Student-Centered%20Distillation%20Narrows%20the%20Agentic%20Gap%20Between%20Small%20and%20Large%20LLMs/tables/7c162101b9dcacd2afcb710d07452ba373bacbb07e09c4fa316c7846f8c019dd.jpg)

![8b2c7c23b496fa650fdc19645c15ee309830a07f226ce653ae148488855b92e8.jpg](../icml_results/3368_Student-Centered%20Distillation%20Narrows%20the%20Agentic%20Gap%20Between%20Small%20and%20Large%20LLMs/tables/8b2c7c23b496fa650fdc19645c15ee309830a07f226ce653ae148488855b92e8.jpg)

![993c2718dc5ea7ed50b50d7ec31014df6bb102ab7cf37089e8fa41592729e991.jpg](../icml_results/3368_Student-Centered%20Distillation%20Narrows%20the%20Agentic%20Gap%20Between%20Small%20and%20Large%20LLMs/tables/993c2718dc5ea7ed50b50d7ec31014df6bb102ab7cf37089e8fa41592729e991.jpg)

![9ecb9bba5c3b4513252f6f2895e5d44045006cf2c0fc811c41c2d9837aedbb78.jpg](../icml_results/3368_Student-Centered%20Distillation%20Narrows%20the%20Agentic%20Gap%20Between%20Small%20and%20Large%20LLMs/tables/9ecb9bba5c3b4513252f6f2895e5d44045006cf2c0fc811c41c2d9837aedbb78.jpg)

![ac75199d633acb4f252a788729eff27637b33a2764cb06e465a0e47a54d2a243.jpg](../icml_results/3368_Student-Centered%20Distillation%20Narrows%20the%20Agentic%20Gap%20Between%20Small%20and%20Large%20LLMs/tables/ac75199d633acb4f252a788729eff27637b33a2764cb06e465a0e47a54d2a243.jpg)

![ad415cb5e1f874b0e364a871f1d118711eadc207900c7ff5b2fa9f55bd397e36.jpg](../icml_results/3368_Student-Centered%20Distillation%20Narrows%20the%20Agentic%20Gap%20Between%20Small%20and%20Large%20LLMs/tables/ad415cb5e1f874b0e364a871f1d118711eadc207900c7ff5b2fa9f55bd397e36.jpg)

![cdf449a7806f3cabee61ef76a72898a2e78677c5d58fa42eed52e7eaa93bd22a.jpg](../icml_results/3368_Student-Centered%20Distillation%20Narrows%20the%20Agentic%20Gap%20Between%20Small%20and%20Large%20LLMs/tables/cdf449a7806f3cabee61ef76a72898a2e78677c5d58fa42eed52e7eaa93bd22a.jpg)

![e4467f05a91a5a1e30afe73eefdcb4265da1e50b0f0665a2dd9f748c2ca91601.jpg](../icml_results/3368_Student-Centered%20Distillation%20Narrows%20the%20Agentic%20Gap%20Between%20Small%20and%20Large%20LLMs/tables/e4467f05a91a5a1e30afe73eefdcb4265da1e50b0f0665a2dd9f748c2ca91601.jpg)

<a id="paper-3330"></a>

## The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?


### Images

![07858c302ba4161f3b34ed3d9bb9a0d3910c2a6b8ecd27d7bc8eadd98bf35d03.jpg](../icml_results/3369_The%20Devil%20is%20in%20the%20Condition%20Numbers_%20Why%20is%20GLU%20Better%20than%20non-GLU%20Structure_/images/07858c302ba4161f3b34ed3d9bb9a0d3910c2a6b8ecd27d7bc8eadd98bf35d03.jpg)

![2e2a86d2b1ee13fd265c904fb56bccb7b7ede26f3508b0b72226f92e754792d8.jpg](../icml_results/3369_The%20Devil%20is%20in%20the%20Condition%20Numbers_%20Why%20is%20GLU%20Better%20than%20non-GLU%20Structure_/images/2e2a86d2b1ee13fd265c904fb56bccb7b7ede26f3508b0b72226f92e754792d8.jpg)

![35fc08845912b013cec19fae37b36b8d06943d5ce0cdb5bbf99690ed0573e592.jpg](../icml_results/3369_The%20Devil%20is%20in%20the%20Condition%20Numbers_%20Why%20is%20GLU%20Better%20than%20non-GLU%20Structure_/images/35fc08845912b013cec19fae37b36b8d06943d5ce0cdb5bbf99690ed0573e592.jpg)

![45c8a160a25cd686e6a36130c7f7d5888a086fd2dcecdd4fd23dc662bbf7cccb.jpg](../icml_results/3369_The%20Devil%20is%20in%20the%20Condition%20Numbers_%20Why%20is%20GLU%20Better%20than%20non-GLU%20Structure_/images/45c8a160a25cd686e6a36130c7f7d5888a086fd2dcecdd4fd23dc662bbf7cccb.jpg)

![4c6cd58fadbba5ce34efca81c2f7a2a64edd01c9e0a6c2b9e6a0c186abb814cb.jpg](../icml_results/3369_The%20Devil%20is%20in%20the%20Condition%20Numbers_%20Why%20is%20GLU%20Better%20than%20non-GLU%20Structure_/images/4c6cd58fadbba5ce34efca81c2f7a2a64edd01c9e0a6c2b9e6a0c186abb814cb.jpg)

![58c79348722e98c9ad9e9a11f10bc45339aba0ee5ddc269db6bc6ba789722b28.jpg](../icml_results/3369_The%20Devil%20is%20in%20the%20Condition%20Numbers_%20Why%20is%20GLU%20Better%20than%20non-GLU%20Structure_/images/58c79348722e98c9ad9e9a11f10bc45339aba0ee5ddc269db6bc6ba789722b28.jpg)

![59745e081ce169fd2ea02d57cdfbdd32e92d2e9abbf2985a4a005f29e106fe0b.jpg](../icml_results/3369_The%20Devil%20is%20in%20the%20Condition%20Numbers_%20Why%20is%20GLU%20Better%20than%20non-GLU%20Structure_/images/59745e081ce169fd2ea02d57cdfbdd32e92d2e9abbf2985a4a005f29e106fe0b.jpg)

![662dbcec319050adfd668e5bad0e89f90773219ad2bbe96a1c3665a60828265e.jpg](../icml_results/3369_The%20Devil%20is%20in%20the%20Condition%20Numbers_%20Why%20is%20GLU%20Better%20than%20non-GLU%20Structure_/images/662dbcec319050adfd668e5bad0e89f90773219ad2bbe96a1c3665a60828265e.jpg)

![7fe9c335d894b214a14235a4949f45f565991e0eb600b38c9c98dc2cacadbdf6.jpg](../icml_results/3369_The%20Devil%20is%20in%20the%20Condition%20Numbers_%20Why%20is%20GLU%20Better%20than%20non-GLU%20Structure_/images/7fe9c335d894b214a14235a4949f45f565991e0eb600b38c9c98dc2cacadbdf6.jpg)

![95ccfffb9c14efe883a832612749dbb1ec4bd1602893b0093bab0ddb0d470950.jpg](../icml_results/3369_The%20Devil%20is%20in%20the%20Condition%20Numbers_%20Why%20is%20GLU%20Better%20than%20non-GLU%20Structure_/images/95ccfffb9c14efe883a832612749dbb1ec4bd1602893b0093bab0ddb0d470950.jpg)

![b590791288fc3ffefbc6bc7909a9822d30ec70c79445762b9df66f1e8f721192.jpg](../icml_results/3369_The%20Devil%20is%20in%20the%20Condition%20Numbers_%20Why%20is%20GLU%20Better%20than%20non-GLU%20Structure_/images/b590791288fc3ffefbc6bc7909a9822d30ec70c79445762b9df66f1e8f721192.jpg)

![e05bbfd3431a02c7adeaa0796e093f4e78ca95705a122796b9737e9ff872e81d.jpg](../icml_results/3369_The%20Devil%20is%20in%20the%20Condition%20Numbers_%20Why%20is%20GLU%20Better%20than%20non-GLU%20Structure_/images/e05bbfd3431a02c7adeaa0796e093f4e78ca95705a122796b9737e9ff872e81d.jpg)

### Tables

![5db8917127617323e666352f06d6b51f84b7afa5af239558c05d7b82368e5c33.jpg](../icml_results/3369_The%20Devil%20is%20in%20the%20Condition%20Numbers_%20Why%20is%20GLU%20Better%20than%20non-GLU%20Structure_/tables/5db8917127617323e666352f06d6b51f84b7afa5af239558c05d7b82368e5c33.jpg)

<a id="paper-3331"></a>

## SALAAD: Sparse And Low-Rank Adaptation via ADMM for Large Language Model Inference


### Images

![0893de65a6c68de78827bf9146811505e2e1f35091932c202728cf237d85477f.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/images/0893de65a6c68de78827bf9146811505e2e1f35091932c202728cf237d85477f.jpg)

![1cb3fe320d2ed390c2f286677e0a2930d77dabff72d26eecdd0367287e91e9b5.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/images/1cb3fe320d2ed390c2f286677e0a2930d77dabff72d26eecdd0367287e91e9b5.jpg)

![4552c929e89fee3dcc91076a0d0eba6e302d382d5436733d7bb0103330c7c1c2.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/images/4552c929e89fee3dcc91076a0d0eba6e302d382d5436733d7bb0103330c7c1c2.jpg)

![4afe144327e7f9f8963bfbf4441a61dd32b248f8f28ae42b4569683d7f4bcabd.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/images/4afe144327e7f9f8963bfbf4441a61dd32b248f8f28ae42b4569683d7f4bcabd.jpg)

![4fb8e46eda7963b16a13ef2e2bbcd16c5bd165b5a5b7b4d9d990ea316eedafb5.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/images/4fb8e46eda7963b16a13ef2e2bbcd16c5bd165b5a5b7b4d9d990ea316eedafb5.jpg)

![76346cfa3c2e2b342d90273170d8429c7d07b792e92aa274ec1f2cd5fca61d76.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/images/76346cfa3c2e2b342d90273170d8429c7d07b792e92aa274ec1f2cd5fca61d76.jpg)

![8c6a39e559def00f8c7ec7b276859889eb9ae5c5287360618bbebc3076559311.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/images/8c6a39e559def00f8c7ec7b276859889eb9ae5c5287360618bbebc3076559311.jpg)

![8f934d46f82d4027b0e0d3ba6e4d0e2622dde15359bde936ed1ceea959288927.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/images/8f934d46f82d4027b0e0d3ba6e4d0e2622dde15359bde936ed1ceea959288927.jpg)

![93181632233d3cfdd09145d79f9572312e045a5e35ea483092c93dfc4ea628b2.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/images/93181632233d3cfdd09145d79f9572312e045a5e35ea483092c93dfc4ea628b2.jpg)

![957c1ef339df500bb4755ef2b211dc83058d29c4d0a829a734629362bf9c7f7d.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/images/957c1ef339df500bb4755ef2b211dc83058d29c4d0a829a734629362bf9c7f7d.jpg)

![c0d6a606d8b6db4ab264814e3eca13671de967f4533fda985a9ea349ab9e637e.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/images/c0d6a606d8b6db4ab264814e3eca13671de967f4533fda985a9ea349ab9e637e.jpg)

![c6eca1689339b81de0efc69603b3b1fe89266ff8734f72178d482a801719e344.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/images/c6eca1689339b81de0efc69603b3b1fe89266ff8734f72178d482a801719e344.jpg)

![e107b8e233b4486e14ddfa0fd05593c70947457b524c426dbaa5be56f8fe2d47.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/images/e107b8e233b4486e14ddfa0fd05593c70947457b524c426dbaa5be56f8fe2d47.jpg)

### Tables

![0319c218b033875ece4c78b4d38d6119ac313eda30fb412995496d1ebbf7da92.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/tables/0319c218b033875ece4c78b4d38d6119ac313eda30fb412995496d1ebbf7da92.jpg)

![4f388cbbfd4ffa7b2d0ccfff392499319da8f5b92524aef5ad8f520a937fd965.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/tables/4f388cbbfd4ffa7b2d0ccfff392499319da8f5b92524aef5ad8f520a937fd965.jpg)

![5f17f911c0b1421eb7fa1ea57b4befa4600493814a1ccad98ea3ccd2fd41b426.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/tables/5f17f911c0b1421eb7fa1ea57b4befa4600493814a1ccad98ea3ccd2fd41b426.jpg)

![653bf672bdf898268ec735757a8a29178d2aa65a3e3b7edb4c4375349e98312d.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/tables/653bf672bdf898268ec735757a8a29178d2aa65a3e3b7edb4c4375349e98312d.jpg)

![670d1613992ec9b39511f919be36375c6ce0755bdd9f013106089a4144d74c45.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/tables/670d1613992ec9b39511f919be36375c6ce0755bdd9f013106089a4144d74c45.jpg)

![67bd1c1b7125b6a110c323969ca4d466ebaa1ebeab5677cfc886a618889e7351.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/tables/67bd1c1b7125b6a110c323969ca4d466ebaa1ebeab5677cfc886a618889e7351.jpg)

![9ca3fd2979f22d3b38fc8164c42bce11030cf5318a79b2c806ef550e30c993af.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/tables/9ca3fd2979f22d3b38fc8164c42bce11030cf5318a79b2c806ef550e30c993af.jpg)

![9ec70557129207aaaab61f638ceaa085496e965ee72050ee06fb255113504a11.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/tables/9ec70557129207aaaab61f638ceaa085496e965ee72050ee06fb255113504a11.jpg)

![a6fa55f77174891f76616640502b4c9f629547afe546acc4e7c36f9fd5eb4189.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/tables/a6fa55f77174891f76616640502b4c9f629547afe546acc4e7c36f9fd5eb4189.jpg)

![f0b633799e60069acff45ae3a4df846c08d5a55d39b4d9c926d3e3667f74d6d7.jpg](../icml_results/3370_SALAAD_%20Sparse%20And%20Low-Rank%20Adaptation%20via%20ADMM%20for%20Large%20Language%20Model%20Inference/tables/f0b633799e60069acff45ae3a4df846c08d5a55d39b4d9c926d3e3667f74d6d7.jpg)

<a id="paper-3332"></a>

## $μ$pscaling small models: Principled warm starts and hyperparameter transfer


### Images

![05404de102c308aaf115790c341046e03a7935636c84a786499a3b3ffb7f4908.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/05404de102c308aaf115790c341046e03a7935636c84a786499a3b3ffb7f4908.jpg)

![06a9a9aea90e1ef2634275b929b5ac2096038ac29fcb7dba9e544887665b97bf.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/06a9a9aea90e1ef2634275b929b5ac2096038ac29fcb7dba9e544887665b97bf.jpg)

![08db1a2a9514840bb7a592627ba2963f7ac1a76f7919c957611492d3b2aaeac3.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/08db1a2a9514840bb7a592627ba2963f7ac1a76f7919c957611492d3b2aaeac3.jpg)

![0b5cc5311551c36f1055a2f4cc790e2fccc44514716b9250a0a4c1d8ad92897e.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/0b5cc5311551c36f1055a2f4cc790e2fccc44514716b9250a0a4c1d8ad92897e.jpg)

![44c065b454ba9c977e4cc3cbbfa50c3005a0e363ddb8096c69004c05eb0dbdd9.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/44c065b454ba9c977e4cc3cbbfa50c3005a0e363ddb8096c69004c05eb0dbdd9.jpg)

![47ecabff500fb45f6f5d7ff1c4b48be3ab348ed6eed2492f46a5500c79f52a85.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/47ecabff500fb45f6f5d7ff1c4b48be3ab348ed6eed2492f46a5500c79f52a85.jpg)

![5ffd9aa688bdf454b5a45b11b4ec268ec8f6898fb3c5f375a849972d5150c436.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/5ffd9aa688bdf454b5a45b11b4ec268ec8f6898fb3c5f375a849972d5150c436.jpg)

![646a18067afdb9be51b5900ea415733a3fc3b9e99769119901b4367efdd13410.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/646a18067afdb9be51b5900ea415733a3fc3b9e99769119901b4367efdd13410.jpg)

![9f717d7426c332e4c080b2ffdea32fd3dda82b85453726bf684e303cb17c3000.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/9f717d7426c332e4c080b2ffdea32fd3dda82b85453726bf684e303cb17c3000.jpg)

![a975a2bba4cfa000bd61991d40e207a19ab5d8566507868dee0d5e2164357f36.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/a975a2bba4cfa000bd61991d40e207a19ab5d8566507868dee0d5e2164357f36.jpg)

![a9e9d30d6656e3339fe827dc2f123410d3a1a8bed495d2690dcf8357d71cbe75.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/a9e9d30d6656e3339fe827dc2f123410d3a1a8bed495d2690dcf8357d71cbe75.jpg)

![c3a31eea5eeb9e8c651f321f3b86783b13dcf94280f5d7c2c7651b3de7f7a2c3.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/c3a31eea5eeb9e8c651f321f3b86783b13dcf94280f5d7c2c7651b3de7f7a2c3.jpg)

![e06655e84b84ffb85b490801351a56923c840c08d77ed94f1e7064f5b412d522.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/e06655e84b84ffb85b490801351a56923c840c08d77ed94f1e7064f5b412d522.jpg)

![e59192608c986a809dd838170a2796fe59472be64d9849c20eeb58cb89ce62d9.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/e59192608c986a809dd838170a2796fe59472be64d9849c20eeb58cb89ce62d9.jpg)

![ebed0b48e9bde15c0c2d472603c17c124a3bb199fb776b212e066fbc719de7d7.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/ebed0b48e9bde15c0c2d472603c17c124a3bb199fb776b212e066fbc719de7d7.jpg)

![ef79d13a6cb4630e3c4ccea866f940b5b47257c4195e66bd96327854361ec406.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/ef79d13a6cb4630e3c4ccea866f940b5b47257c4195e66bd96327854361ec406.jpg)

![f11c26e6ff68248048d004f6c84c27156400e384fe31de3d784702859f0f0ac2.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/f11c26e6ff68248048d004f6c84c27156400e384fe31de3d784702859f0f0ac2.jpg)

![fb6aed40a9373ffbf968fb0a2998a0b14516f9f452fcda768b23b33177791455.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/images/fb6aed40a9373ffbf968fb0a2998a0b14516f9f452fcda768b23b33177791455.jpg)

### Tables

![0d70c63b527272a0476902e491d5b42a6047828737b4782e689cb53b3b5c3395.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/tables/0d70c63b527272a0476902e491d5b42a6047828737b4782e689cb53b3b5c3395.jpg)

![26afc296f8b70ea24701b65c26d282c4bbf0669c9ddb1cdef9f165af9647fecc.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/tables/26afc296f8b70ea24701b65c26d282c4bbf0669c9ddb1cdef9f165af9647fecc.jpg)

![53ec974745949870114025c9b24aa4cf3f01fdd5f2e0633a590a155f3de2b0e4.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/tables/53ec974745949870114025c9b24aa4cf3f01fdd5f2e0633a590a155f3de2b0e4.jpg)

![7d6faf71e6e80761c178ff2c9810b08d9d6025270f960fa27c0fc98652c2aefc.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/tables/7d6faf71e6e80761c178ff2c9810b08d9d6025270f960fa27c0fc98652c2aefc.jpg)

![7e5669de8aed1cb62ad2b798d3b51f81b88e102f6c743fec8f0bb4d0b275961f.jpg](../icml_results/3372_%24%CE%BC%24pscaling%20small%20models_%20Principled%20warm%20starts%20and%20hyperparameter%20transfer/tables/7e5669de8aed1cb62ad2b798d3b51f81b88e102f6c743fec8f0bb4d0b275961f.jpg)

<a id="paper-3333"></a>

## Convolutional Learnable-Group Weightless Neural Network

### Images

![135991836d6ff5ce803e9fad0b3f508ac724c779386a64ac04a3a4a2b4a397ac.jpg](../icml_results/3373_Convolutional%20Learnable-Group%20Weightless%20Neural%20Network/images/135991836d6ff5ce803e9fad0b3f508ac724c779386a64ac04a3a4a2b4a397ac.jpg)

![ac8519c15c98c0634076a6e0ecd5283d16737bdf27e68d373856ffd056628156.jpg](../icml_results/3373_Convolutional%20Learnable-Group%20Weightless%20Neural%20Network/images/ac8519c15c98c0634076a6e0ecd5283d16737bdf27e68d373856ffd056628156.jpg)

### Tables

![12003ce23e6f5fa0295c2d072c79e91707c918149edfb41c0e63c35031e9b029.jpg](../icml_results/3373_Convolutional%20Learnable-Group%20Weightless%20Neural%20Network/tables/12003ce23e6f5fa0295c2d072c79e91707c918149edfb41c0e63c35031e9b029.jpg)

![28a11a13fdab3f480d869b056aaaa4a34abf166af73eb441b66c3854f72dc5e9.jpg](../icml_results/3373_Convolutional%20Learnable-Group%20Weightless%20Neural%20Network/tables/28a11a13fdab3f480d869b056aaaa4a34abf166af73eb441b66c3854f72dc5e9.jpg)

![298623a36866a7ffc910f33e13dc73cc5fc2e710c043966c65e04a960e8bee43.jpg](../icml_results/3373_Convolutional%20Learnable-Group%20Weightless%20Neural%20Network/tables/298623a36866a7ffc910f33e13dc73cc5fc2e710c043966c65e04a960e8bee43.jpg)

![3309cf04a33ee74ba1df7e0d8e2386e333d2c36ff23851d4ac0bf5189a76d2b9.jpg](../icml_results/3373_Convolutional%20Learnable-Group%20Weightless%20Neural%20Network/tables/3309cf04a33ee74ba1df7e0d8e2386e333d2c36ff23851d4ac0bf5189a76d2b9.jpg)

![631ae1609711768c5e81ca3f98d5f954e13aa1b2f8080b213f537984d301fc5f.jpg](../icml_results/3373_Convolutional%20Learnable-Group%20Weightless%20Neural%20Network/tables/631ae1609711768c5e81ca3f98d5f954e13aa1b2f8080b213f537984d301fc5f.jpg)

![7149914be65f8cfc30a1b32b9fa131c94ee2914a6c080090124d5aff18df5663.jpg](../icml_results/3373_Convolutional%20Learnable-Group%20Weightless%20Neural%20Network/tables/7149914be65f8cfc30a1b32b9fa131c94ee2914a6c080090124d5aff18df5663.jpg)

![f0c1ac90e6c168bc41a7b19e51e2c8035901bf4302ce67d64cd6560b3a8ae042.jpg](../icml_results/3373_Convolutional%20Learnable-Group%20Weightless%20Neural%20Network/tables/f0c1ac90e6c168bc41a7b19e51e2c8035901bf4302ce67d64cd6560b3a8ae042.jpg)
