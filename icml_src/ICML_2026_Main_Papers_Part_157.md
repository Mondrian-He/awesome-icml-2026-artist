# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 157 of 197

## 目录 (Table of Contents)

1. [Adversarial Latent Embedding Repair for LLM Continual Learning](#paper-5149)
2. [Memora: A Harmonic Memory Representation Balancing Abstraction and Specificity](#paper-5150)
3. [SDiD: Shared Diffusion Prior for Efficient Distribute Stereo Image Compression](#paper-5151)
4. [VIA-SD: Verification via Intra-Model Routing for Speculative Decoding](#paper-5152)
5. [From 2D Grids to 1D Tokens: Reforming Shared Representations for Multimodal Image Fusion](#paper-5153)
6. [Logit-Attention Divergence: Mitigating Position Bias in Multi-Image Retrieval via Attention-Guided Calibration](#paper-5154)
7. [Rethinking Genomic Modeling Through Optical Character Recognition](#paper-5155)
8. [Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility](#paper-5156)
9. [New Wide-Net-Casting Jailbreak Attacks Risk Large Models](#paper-5157)
10. [MOD-SR: Unifying Multimodal Learning and Direct Optimization with Gradient-Guided Diffusion Model for Symbolic Regression](#paper-5158)
11. [Stop the Flip-Flop: Context-Preserving Verification for Fast Revocable Diffusion Decoding](#paper-5159)
12. [Spike Camera Autofocus via Frequency-Domain Spectral-Centroid Migration](#paper-5160)
13. [Seeing the Unseen: Physics-as-Representation for Generalizable Gaze Perception](#paper-5161)
14. [Influence-Disentangled Federated Training: Learning Models That Are Easy to Unlearn](#paper-5162)
15. [Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation](#paper-5163)
16. [PixCLIP: Towards Fine-grained Vision-Language Understanding via Any-granularity Pixel-Text Alignment](#paper-5164)
17. [Drop-in Circulant Structural Priors for Transformer Decoding of Cyclic Codes](#paper-5165)
18. [Functional Attention: From Pairwise Affinities to Functional Correspondences](#paper-5166)
19. [How Do Language Models Speak Languages? A Case Study on Unintended Code-Switching](#paper-5167)
20. [AutoVSR: Automatic Visual-to-Symbolic Reasoning for Symbolic Expression Generation from Circuit Schematic](#paper-5168)
21. [Dynamic Symmetric Point Tracking: Tackling Non-ideal Reference in Analog In-memory Training](#paper-5169)
22. [Capacity-Agnostic Parameter Isolation for Continual Graph Learning](#paper-5170)
23. [Unifying Dataset Pruning and Distillation for Efficient Large-scale Compression](#paper-5171)
24. [Generalized Correctness Models: Learning Calibrated and Cross-Model Correctness Predictors from Historical Patterns](#paper-5172)
25. [Copula-SVI: Vine-Copula Variational Inference with Stein Refining for Instance-Level Correlation Capturing](#paper-5173)
26. [AIR: Improving Agent Safety through Incident Response](#paper-5174)
27. [WebWorld: A Large-Scale World Model for Web Agent Training](#paper-5175)
28. [FedHera: Towards Drift-Resilient Federated Fine-tuning with Heterogeneous Resources](#paper-5176)
29. [Editable Proof Sketch for Automated Theorem Proving](#paper-5177)
30. [Kuramoto Oscillatory Phase Encoding: Neuro-inspired Synchronization for Improved Learning Efficiency](#paper-5178)
31. [LECTOR: Joint Optimization of Scientific Reasoning Graphs and Introduction Generation](#paper-5179)
32. [Memory-Efficient LLM Training with Dynamic Sparsity: From Stability to Practical Scaling](#paper-5180)
33. [CoverPruneGS: Coverage-Preserving Structured Pruning for Compact 3D Gaussian Splatting from Sparse-View Monocular Videos](#paper-5181)

---


<a id="paper-5149"></a>

## Adversarial Latent Embedding Repair for LLM Continual Learning

### Images

![243d6d45d3c108aee8e00394341ec12b6841211706dd15229ef2f4b38d8766ed.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/243d6d45d3c108aee8e00394341ec12b6841211706dd15229ef2f4b38d8766ed.jpg)

![2c06691430daf669e53940f9a7a4ed6bd598b2826711160657bccdce04e9c34b.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/2c06691430daf669e53940f9a7a4ed6bd598b2826711160657bccdce04e9c34b.jpg)

![2e62629c96b655ac1b7eb6e4af3c918264b5e8c69ce0d2f22cbdf63da9e511fb.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/2e62629c96b655ac1b7eb6e4af3c918264b5e8c69ce0d2f22cbdf63da9e511fb.jpg)

![32234d1bed8ffb6b00e1a7a5ae360b654d00488f578fae710a526c2158ba6e23.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/32234d1bed8ffb6b00e1a7a5ae360b654d00488f578fae710a526c2158ba6e23.jpg)

![40f8d52d9fd321283ca8ff7f10546a8e7a4d86d19869d0dff59eb5f3121e6f56.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/40f8d52d9fd321283ca8ff7f10546a8e7a4d86d19869d0dff59eb5f3121e6f56.jpg)

![5312df83daf9854f2d0d3335e70e7f168e66199bdd952ceac9290fb5cd5c1cd8.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/5312df83daf9854f2d0d3335e70e7f168e66199bdd952ceac9290fb5cd5c1cd8.jpg)

![573c49196dee7e4aa524a2695549dde63ea4b559c406a64d24d009939455b0ac.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/573c49196dee7e4aa524a2695549dde63ea4b559c406a64d24d009939455b0ac.jpg)

![5a7650970287dcf22441afdab451eb89422bf531221834209b3dd0ce436aeb0a.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/5a7650970287dcf22441afdab451eb89422bf531221834209b3dd0ce436aeb0a.jpg)

![699c27705821f8082bf99a190c8b5cf3527992f57ceca9218202a8cca90ec570.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/699c27705821f8082bf99a190c8b5cf3527992f57ceca9218202a8cca90ec570.jpg)

![6d6efefd99818e0ecbd16603498fafb917cce3631839cd90f8f82deb36897f62.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/6d6efefd99818e0ecbd16603498fafb917cce3631839cd90f8f82deb36897f62.jpg)

![79daf903415a32b2c6aea4e3a662c364e98d074fb51f33786f033881be78c51c.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/79daf903415a32b2c6aea4e3a662c364e98d074fb51f33786f033881be78c51c.jpg)

![8d1814d4ba203673a054443a55175a5897d5916bc34bc3f1dcf185ab62f0094c.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/8d1814d4ba203673a054443a55175a5897d5916bc34bc3f1dcf185ab62f0094c.jpg)

![a148c87c932071eda5861a5fd0bca297eea7d6421efe778eec9c8e1706a75f3f.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/a148c87c932071eda5861a5fd0bca297eea7d6421efe778eec9c8e1706a75f3f.jpg)

![a37f9d354d4038c106b99fbe8af59a334b3282cb6681ad9ca20a18d763a533bf.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/a37f9d354d4038c106b99fbe8af59a334b3282cb6681ad9ca20a18d763a533bf.jpg)

![a5dff9e81921f5de4c535a314b4c7689ab67736e6042fba5c7a40f8ed58608c7.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/a5dff9e81921f5de4c535a314b4c7689ab67736e6042fba5c7a40f8ed58608c7.jpg)

![a877ad4c69a1059581a30b50fe454e5a071018267493e3a1201d011363bd01a1.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/a877ad4c69a1059581a30b50fe454e5a071018267493e3a1201d011363bd01a1.jpg)

![e981b953afe7ed27dd3cc5aa3825324a37b4154337c4bda0131a5c3be6e70173.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/e981b953afe7ed27dd3cc5aa3825324a37b4154337c4bda0131a5c3be6e70173.jpg)

![eed30645f75afb516a390d78ac90b4d70fac184744576a33f81a2aabd67c8491.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/eed30645f75afb516a390d78ac90b4d70fac184744576a33f81a2aabd67c8491.jpg)

![f9ce066979f0efcf862f3f65a48c4c482e3e56ebb0619f8c46250d07a82fe394.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/images/f9ce066979f0efcf862f3f65a48c4c482e3e56ebb0619f8c46250d07a82fe394.jpg)

### Tables

![1c181fa300dbc0c58437ce3a1a2aefb07deab1b0beee3a658c43ea818fb75927.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/1c181fa300dbc0c58437ce3a1a2aefb07deab1b0beee3a658c43ea818fb75927.jpg)

![25117ed6050a7a2fd439df110c2eef7a985fb9131dda7cc8a06fe59afbef1674.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/25117ed6050a7a2fd439df110c2eef7a985fb9131dda7cc8a06fe59afbef1674.jpg)

![28e141bb53e63285b283533e1f10f010e3a53006750126caff71fac8f53cb9d6.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/28e141bb53e63285b283533e1f10f010e3a53006750126caff71fac8f53cb9d6.jpg)

![2aed140ce7ebd2ee0b879350b0ff8ca8269b6be4bae05d27c7daace36ff937c7.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/2aed140ce7ebd2ee0b879350b0ff8ca8269b6be4bae05d27c7daace36ff937c7.jpg)

![3cf7d9df052e957ac09c124906fa2edc310d37ea296f55fc64990a272bac1b00.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/3cf7d9df052e957ac09c124906fa2edc310d37ea296f55fc64990a272bac1b00.jpg)

![5672012a8319f1cd5293bf1e21c4daf5ec7629564d5f22c467e53ac04d339b17.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/5672012a8319f1cd5293bf1e21c4daf5ec7629564d5f22c467e53ac04d339b17.jpg)

![60a21388704589f7e5ac90ad4c49aeab424fc6ad290cc94819eaa24aa936eaf5.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/60a21388704589f7e5ac90ad4c49aeab424fc6ad290cc94819eaa24aa936eaf5.jpg)

![749c084fa8be72a69175a64ae50e8c66e0e3a1e35d55686426c3ce4d73a59eb5.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/749c084fa8be72a69175a64ae50e8c66e0e3a1e35d55686426c3ce4d73a59eb5.jpg)

![bc0e4aca066c664c92e3b78632842c195841e419a3500c6e1c2a8d3326fa44e0.jpg](../icml_results/5196_Boost%20the%20Identity-Preserving%20Embedding%20for%20Consistent%20Visual%20Generation/tables/bc0e4aca066c664c92e3b78632842c195841e419a3500c6e1c2a8d3326fa44e0.jpg)

## Adversarial Latent Embedding Repair for LLM Continual Learning


### Images

![12fd66c3fd764548f8ba8a99da02ced9b7828acfe252c387f9c3d816fbb064a4.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/images/12fd66c3fd764548f8ba8a99da02ced9b7828acfe252c387f9c3d816fbb064a4.jpg)

![307c986f0266ac373dbced12ef85b3ad9974861883cb5c1796ea8c09980c608d.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/images/307c986f0266ac373dbced12ef85b3ad9974861883cb5c1796ea8c09980c608d.jpg)

![3b0da2e6066ee8e9c6438887fa53e44939c6241d7cdb9196a404d6683a2f2f80.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/images/3b0da2e6066ee8e9c6438887fa53e44939c6241d7cdb9196a404d6683a2f2f80.jpg)

![6bbf5e9a8873c3096ba15920ad302cadc8daf8f5f90ff2bfc5ee922248f2fe86.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/images/6bbf5e9a8873c3096ba15920ad302cadc8daf8f5f90ff2bfc5ee922248f2fe86.jpg)

![90231aa65180930e537607e3fc9591a02673c8305981c015276447a1c6993baf.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/images/90231aa65180930e537607e3fc9591a02673c8305981c015276447a1c6993baf.jpg)

![b6fb0e38d6ca35e5531796ef8e25ad2c3b01a035af70e4a97f66f3d8f3d1cb77.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/images/b6fb0e38d6ca35e5531796ef8e25ad2c3b01a035af70e4a97f66f3d8f3d1cb77.jpg)

![cf0c88d69abb72af36b6a8e48cd0b0d4a3785738d3a566361b88deec16ded89c.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/images/cf0c88d69abb72af36b6a8e48cd0b0d4a3785738d3a566361b88deec16ded89c.jpg)

![e3dd5cdab1312ff8adf07563b6bf44dce305ac791e05cde518ca4ed827dfd5e1.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/images/e3dd5cdab1312ff8adf07563b6bf44dce305ac791e05cde518ca4ed827dfd5e1.jpg)

### Tables

![0620dc7e1689e1a4c9008a60acf049c71a1f8880257638690f85cf546178704f.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/tables/0620dc7e1689e1a4c9008a60acf049c71a1f8880257638690f85cf546178704f.jpg)

![16ad757748ae485f6fd090bd39bb0699fdd1bcff95583483ebcd88b21fa515b6.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/tables/16ad757748ae485f6fd090bd39bb0699fdd1bcff95583483ebcd88b21fa515b6.jpg)

![2ff18e9e5eb49c862f72fff7e7b1b804fb2dafcf20f7c9b8e15a705b8272c87e.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/tables/2ff18e9e5eb49c862f72fff7e7b1b804fb2dafcf20f7c9b8e15a705b8272c87e.jpg)

![517ccabac6ae4cd207563c0428b7b652d1aedbeb645daf8f486ddad4b74b0eb8.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/tables/517ccabac6ae4cd207563c0428b7b652d1aedbeb645daf8f486ddad4b74b0eb8.jpg)

![564b2fdd056ed5a3a8ba5d4ec9847a23a670ffc1b64e305d03f03eb91f3fdb8e.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/tables/564b2fdd056ed5a3a8ba5d4ec9847a23a670ffc1b64e305d03f03eb91f3fdb8e.jpg)

![7618a3a0c249ab95ee825ed9cade8c61d58b389e23dc02fb41322d4c720fc649.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/tables/7618a3a0c249ab95ee825ed9cade8c61d58b389e23dc02fb41322d4c720fc649.jpg)

![a2fd41e869f4deb66abf23037c9a9de37de88f9373e06709335365b2418d4ddc.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/tables/a2fd41e869f4deb66abf23037c9a9de37de88f9373e06709335365b2418d4ddc.jpg)

![c86d1f722ef3cf74e81412bcfb47eaa66c5b0ade182f80f4897b6743189fc747.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/tables/c86d1f722ef3cf74e81412bcfb47eaa66c5b0ade182f80f4897b6743189fc747.jpg)

![ced968ef44083f640c5591d384da0e67c6fbb01dc810a5a8f52dbb477b72f2a3.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/tables/ced968ef44083f640c5591d384da0e67c6fbb01dc810a5a8f52dbb477b72f2a3.jpg)

![ee46e4ac014779347ba639204db3d596da194de9908c8f334ecf7479f7752278.jpg](../icml_results/5197_Adversarial%20Latent%20Embedding%20Repair%20for%20LLM%20Continual%20Learning/tables/ee46e4ac014779347ba639204db3d596da194de9908c8f334ecf7479f7752278.jpg)

<a id="paper-5150"></a>

## Memora: A Harmonic Memory Representation Balancing Abstraction and Specificity


### Images

![1850b18ddd6e061bcf00f3d17c4e029647d2a76f31db86d90bfb7450861335ba.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/images/1850b18ddd6e061bcf00f3d17c4e029647d2a76f31db86d90bfb7450861335ba.jpg)

![736f5e0b503c4051d693f668e0393f32170b9f059e9520967b76c929b1c052e9.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/images/736f5e0b503c4051d693f668e0393f32170b9f059e9520967b76c929b1c052e9.jpg)

![90aefe2ec46450b54d34c0c2a9db3ee847e6d4ef51a98994c4e9bf27a5d185c1.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/images/90aefe2ec46450b54d34c0c2a9db3ee847e6d4ef51a98994c4e9bf27a5d185c1.jpg)

![bd767296c965115ee7748027f321b860bbc930897f7912ffa5df7cbd84742fcf.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/images/bd767296c965115ee7748027f321b860bbc930897f7912ffa5df7cbd84742fcf.jpg)

### Tables

![12233b45dc0cf774db1f5101b8040b6d5a26afa0886f89c3a1fe962f3215a434.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/tables/12233b45dc0cf774db1f5101b8040b6d5a26afa0886f89c3a1fe962f3215a434.jpg)

![2d59b3c49a7846f7962c81f1e1addcab1a70ab4bdecf78147062871a96e58f6f.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/tables/2d59b3c49a7846f7962c81f1e1addcab1a70ab4bdecf78147062871a96e58f6f.jpg)

![31671e5f4fa7e4e9c9fa1dd70ccfc0cef06fccadf6c38146c7ac762d448313ab.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/tables/31671e5f4fa7e4e9c9fa1dd70ccfc0cef06fccadf6c38146c7ac762d448313ab.jpg)

![341111b4614aab4edfe67727d978a63de1b6757bf63d59199c7e1969dc78b580.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/tables/341111b4614aab4edfe67727d978a63de1b6757bf63d59199c7e1969dc78b580.jpg)

![3c24ae4097c5dc35c4b3556a0bc61785985891bca53bf8cc54b09e3ea28cca2a.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/tables/3c24ae4097c5dc35c4b3556a0bc61785985891bca53bf8cc54b09e3ea28cca2a.jpg)

![69da4972dee4057fa38249845463194e2f80d45273593dc175107ba9f64c5e69.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/tables/69da4972dee4057fa38249845463194e2f80d45273593dc175107ba9f64c5e69.jpg)

![745bf35b7eee3f6a442b62c48526bccb3389a22239f0a1f1f0ef13990f864e19.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/tables/745bf35b7eee3f6a442b62c48526bccb3389a22239f0a1f1f0ef13990f864e19.jpg)

![777c4172688013675dd302cb08e61a61e3be0fbe374250e116f03f2de8fcc9e2.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/tables/777c4172688013675dd302cb08e61a61e3be0fbe374250e116f03f2de8fcc9e2.jpg)

![7aee2c07d4ec441775c80c401fd0126f45d4c36928d93e58f240de723cc35cd8.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/tables/7aee2c07d4ec441775c80c401fd0126f45d4c36928d93e58f240de723cc35cd8.jpg)

![a082a7faadbfe0d9758678340ff8eca671aa4baa630cf92e90fc4a4c4a4ba9c9.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/tables/a082a7faadbfe0d9758678340ff8eca671aa4baa630cf92e90fc4a4c4a4ba9c9.jpg)

![a9af6eada5dd7e20fc488079bd6fccda52817253de7823f2efa1b60278edff2b.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/tables/a9af6eada5dd7e20fc488079bd6fccda52817253de7823f2efa1b60278edff2b.jpg)

![c993a97aa2c143f618a44714049873fccdd204b0501bcdabcb7fa45e4df56949.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/tables/c993a97aa2c143f618a44714049873fccdd204b0501bcdabcb7fa45e4df56949.jpg)

![d59960f963c70bf01d42ed6c1d6c88bab3f3308fdfdc67b0c5f43171c85af294.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/tables/d59960f963c70bf01d42ed6c1d6c88bab3f3308fdfdc67b0c5f43171c85af294.jpg)

![d7e3e51bae9cdda9a52f81ec0b7e03bb68237ba7ea2c8f42f900aa7d877e0526.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/tables/d7e3e51bae9cdda9a52f81ec0b7e03bb68237ba7ea2c8f42f900aa7d877e0526.jpg)

![e862ba651cf9891252ee40ae7bb836b4e5fdb6639559dc0731f312d884e54341.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/tables/e862ba651cf9891252ee40ae7bb836b4e5fdb6639559dc0731f312d884e54341.jpg)

![fac85b23506b529e6d6b70dc2ae48868d6e9247855af286d7db82e68e13c8257.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/tables/fac85b23506b529e6d6b70dc2ae48868d6e9247855af286d7db82e68e13c8257.jpg)

![fc33c3458cd769d73ebbbc65a1125d7e921083b4c79acf65cfeb93c9efa44970.jpg](../icml_results/5198_Memora_%20A%20Harmonic%20Memory%20Representation%20Balancing%20Abstraction%20and%20Specificity/tables/fc33c3458cd769d73ebbbc65a1125d7e921083b4c79acf65cfeb93c9efa44970.jpg)

<a id="paper-5151"></a>

## SDiD: Shared Diffusion Prior for Efficient Distribute Stereo Image Compression


### Images

![33b75a8c73633064f99db19c0cb2abe59a3389ae7ede510ed3d4b16fbe477c7e.jpg](../icml_results/5199_SDiD_%20Shared%20Diffusion%20Prior%20for%20Efficient%20Distribute%20Stereo%20Image%20Compression/images/33b75a8c73633064f99db19c0cb2abe59a3389ae7ede510ed3d4b16fbe477c7e.jpg)

![3e3f0ae597dd1d275990ebf1cc06f74545dfa403055c1be2600f375addaf6061.jpg](../icml_results/5199_SDiD_%20Shared%20Diffusion%20Prior%20for%20Efficient%20Distribute%20Stereo%20Image%20Compression/images/3e3f0ae597dd1d275990ebf1cc06f74545dfa403055c1be2600f375addaf6061.jpg)

![60f1e6edf8997c50331b1108150e6484082b39503bd32f66314474746dd6904f.jpg](../icml_results/5199_SDiD_%20Shared%20Diffusion%20Prior%20for%20Efficient%20Distribute%20Stereo%20Image%20Compression/images/60f1e6edf8997c50331b1108150e6484082b39503bd32f66314474746dd6904f.jpg)

![73c6b36040f32794bf6c52dbf0741a48029407f6abfa2be8c0d2f9e37193c2f5.jpg](../icml_results/5199_SDiD_%20Shared%20Diffusion%20Prior%20for%20Efficient%20Distribute%20Stereo%20Image%20Compression/images/73c6b36040f32794bf6c52dbf0741a48029407f6abfa2be8c0d2f9e37193c2f5.jpg)

![c152203658281f352faf08a739eec696a023756874ddb385d49100e05519f9b5.jpg](../icml_results/5199_SDiD_%20Shared%20Diffusion%20Prior%20for%20Efficient%20Distribute%20Stereo%20Image%20Compression/images/c152203658281f352faf08a739eec696a023756874ddb385d49100e05519f9b5.jpg)

![eb78d789d0c8fbf751a5be73f0d3c2e29a97572d39893abae4b3fc6a8a1557d6.jpg](../icml_results/5199_SDiD_%20Shared%20Diffusion%20Prior%20for%20Efficient%20Distribute%20Stereo%20Image%20Compression/images/eb78d789d0c8fbf751a5be73f0d3c2e29a97572d39893abae4b3fc6a8a1557d6.jpg)

### Tables

![0dad70172b50be8c764ecc34d95c3ef82520017ab8178c52f55f37df6a52fadc.jpg](../icml_results/5199_SDiD_%20Shared%20Diffusion%20Prior%20for%20Efficient%20Distribute%20Stereo%20Image%20Compression/tables/0dad70172b50be8c764ecc34d95c3ef82520017ab8178c52f55f37df6a52fadc.jpg)

![54aa71c2f24f8b9b382491f9d0db0c60e311101e441c2444cdb5d6f2c4612dd2.jpg](../icml_results/5199_SDiD_%20Shared%20Diffusion%20Prior%20for%20Efficient%20Distribute%20Stereo%20Image%20Compression/tables/54aa71c2f24f8b9b382491f9d0db0c60e311101e441c2444cdb5d6f2c4612dd2.jpg)

![e365f83dc26727b9777039f96072fec5d2a25af12f69219a92f3e4b5a82cdb8b.jpg](../icml_results/5199_SDiD_%20Shared%20Diffusion%20Prior%20for%20Efficient%20Distribute%20Stereo%20Image%20Compression/tables/e365f83dc26727b9777039f96072fec5d2a25af12f69219a92f3e4b5a82cdb8b.jpg)

<a id="paper-5152"></a>

## VIA-SD: Verification via Intra-Model Routing for Speculative Decoding


### Images

![0bd941c6b062306c0522459ef0e2aca18bb0a4b0594d4ac4e77243c6dc8bd6e7.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/images/0bd941c6b062306c0522459ef0e2aca18bb0a4b0594d4ac4e77243c6dc8bd6e7.jpg)

![164e3d3b7914f76b4e8448ceae49d46cc27279804e23a09e9f63ad318da67d52.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/images/164e3d3b7914f76b4e8448ceae49d46cc27279804e23a09e9f63ad318da67d52.jpg)

![498af1a86b440e521bfbc10f7560110fef1af7da5b78cf52b5e754f70cb60429.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/images/498af1a86b440e521bfbc10f7560110fef1af7da5b78cf52b5e754f70cb60429.jpg)

![4b712d1d9771f5e58706d13cdefa08dfb1b4a0b42e73d4a43de5add4e720afe2.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/images/4b712d1d9771f5e58706d13cdefa08dfb1b4a0b42e73d4a43de5add4e720afe2.jpg)

![bbbbb85f49497089e30d411aa542018e7cceb9c20a90ce862cb8c3472787623f.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/images/bbbbb85f49497089e30d411aa542018e7cceb9c20a90ce862cb8c3472787623f.jpg)

![bcbfec8e9058bad2a7f886de3fc57fdaf4a556466a88a8c45ffb1b502d9d0265.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/images/bcbfec8e9058bad2a7f886de3fc57fdaf4a556466a88a8c45ffb1b502d9d0265.jpg)

![e7c1ec0b3f13328df761e966b08a649004d3fabfd7a08b59b41761cc20bb7b55.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/images/e7c1ec0b3f13328df761e966b08a649004d3fabfd7a08b59b41761cc20bb7b55.jpg)

![fd2e4e7f4d4efaa9f1ed74710af087845481b07cda7750e0a95292f6c5374fad.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/images/fd2e4e7f4d4efaa9f1ed74710af087845481b07cda7750e0a95292f6c5374fad.jpg)

![fdbdda056557da8bdb269c5c4f8636cd4543e5157921a1ca7a5b1f860daf5be5.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/images/fdbdda056557da8bdb269c5c4f8636cd4543e5157921a1ca7a5b1f860daf5be5.jpg)

### Tables

![05933237038d0a8ec8ae6f45c8ef5fe473f34ef57b1aef078e4370484fe93569.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/tables/05933237038d0a8ec8ae6f45c8ef5fe473f34ef57b1aef078e4370484fe93569.jpg)

![0b65d19189c59938da077e8dde3984870aa44fb6db55fe6be7d1c3b2e056b90f.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/tables/0b65d19189c59938da077e8dde3984870aa44fb6db55fe6be7d1c3b2e056b90f.jpg)

![2fc5822a26a1ca34bd37d1688a2a2d6bd7ea559c5fb4bf5afd48a9829f053009.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/tables/2fc5822a26a1ca34bd37d1688a2a2d6bd7ea559c5fb4bf5afd48a9829f053009.jpg)

![35504a2483502fa082b71486da58e9fb1e7f66f89890edd4fea3a59b26df208e.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/tables/35504a2483502fa082b71486da58e9fb1e7f66f89890edd4fea3a59b26df208e.jpg)

![669232458f0efe095db49b41189bc1b11897ef57cd811c7d2ef46954e38df8ea.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/tables/669232458f0efe095db49b41189bc1b11897ef57cd811c7d2ef46954e38df8ea.jpg)

![6c611341938c84fa78f865ed4d5cbd2fa7bc68f207855c1d91d0c87378cd4051.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/tables/6c611341938c84fa78f865ed4d5cbd2fa7bc68f207855c1d91d0c87378cd4051.jpg)

![9e3b53b3e4c0d732367c02b33a1c6292d3d52fa6af347b78c0c530c32787aa50.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/tables/9e3b53b3e4c0d732367c02b33a1c6292d3d52fa6af347b78c0c530c32787aa50.jpg)

![aa541f56a07221a818ece97b805975ce209067f0723b8737b135d193dddfac46.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/tables/aa541f56a07221a818ece97b805975ce209067f0723b8737b135d193dddfac46.jpg)

![bea87e1a69683fa4b445289f446979279265824acecf8dc3a328196f6e6629cc.jpg](../icml_results/5200_VIA-SD_%20Verification%20via%20Intra-Model%20Routing%20for%20Speculative%20Decoding/tables/bea87e1a69683fa4b445289f446979279265824acecf8dc3a328196f6e6629cc.jpg)

<a id="paper-5153"></a>

## From 2D Grids to 1D Tokens: Reforming Shared Representations for Multimodal Image Fusion


### Images

![05208c665900a043105608433fee3e29ca149ea8ab82dfc62b677ff7bd74106f.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/images/05208c665900a043105608433fee3e29ca149ea8ab82dfc62b677ff7bd74106f.jpg)

![2cde6683def5623e8f06f54f0952589bc9e614d4e768694eb4ca6aed05495862.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/images/2cde6683def5623e8f06f54f0952589bc9e614d4e768694eb4ca6aed05495862.jpg)

![63e90213bc8a369d5d3fbd692b12584836842ec4aadfe9f83ec69abec2272707.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/images/63e90213bc8a369d5d3fbd692b12584836842ec4aadfe9f83ec69abec2272707.jpg)

![9aacbdf2ada0b96125ce1352d106420e955ba60ead1dba54ac2426f286b6c65b.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/images/9aacbdf2ada0b96125ce1352d106420e955ba60ead1dba54ac2426f286b6c65b.jpg)

![9b66ea326bdbdf20ac65f52e860e0bebb8a3f83d2ff1aeb957c402ae79b58224.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/images/9b66ea326bdbdf20ac65f52e860e0bebb8a3f83d2ff1aeb957c402ae79b58224.jpg)

![9c1c7c24a038379880e6aab88c3d5f9c0b0f681e1647d78dfde26f23c28aec54.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/images/9c1c7c24a038379880e6aab88c3d5f9c0b0f681e1647d78dfde26f23c28aec54.jpg)

![a209e553c133a11c4041af25bc228243156758806ea6cbef9a7fa0073eca3f28.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/images/a209e553c133a11c4041af25bc228243156758806ea6cbef9a7fa0073eca3f28.jpg)

![bd5cca21aa7528af9a9b2e1133732e0a7d386f14a4cb2e3f8ce556159210a9ef.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/images/bd5cca21aa7528af9a9b2e1133732e0a7d386f14a4cb2e3f8ce556159210a9ef.jpg)

![c01e4d45e9fec84c15255fcc1fafbf592a03209014c6ace6bda5e79135da6255.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/images/c01e4d45e9fec84c15255fcc1fafbf592a03209014c6ace6bda5e79135da6255.jpg)

![ee09ec5314320b1c62378fb5c16f048a2063408bd2b00c37e2a134b3850166a1.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/images/ee09ec5314320b1c62378fb5c16f048a2063408bd2b00c37e2a134b3850166a1.jpg)

![ee843f19789d28d59cfcc0dedc1f47a485cd753235fb0c64e5f45fe5c5c0d708.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/images/ee843f19789d28d59cfcc0dedc1f47a485cd753235fb0c64e5f45fe5c5c0d708.jpg)

![f8f287bd807601ff482629eac93c3188515e7f41e17e9ca1af6cea66d2ef49b8.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/images/f8f287bd807601ff482629eac93c3188515e7f41e17e9ca1af6cea66d2ef49b8.jpg)

### Tables

![058154e92052be4040c76e5f166e1f77abd92f8a7d4c0c2ffd0ccee181b33bc6.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/tables/058154e92052be4040c76e5f166e1f77abd92f8a7d4c0c2ffd0ccee181b33bc6.jpg)

![0bce696a2a798bad2c00fdb5822adc8e0006623c6840d6939d6d06cc650190be.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/tables/0bce696a2a798bad2c00fdb5822adc8e0006623c6840d6939d6d06cc650190be.jpg)

![0db74fd3ea4de31ae19149face1524224469c64e8d434f155235edf5eef07787.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/tables/0db74fd3ea4de31ae19149face1524224469c64e8d434f155235edf5eef07787.jpg)

![3d0cbdf6958b6a2958a021906d346dbb76fa3650f71f01a821de009c37a6d79d.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/tables/3d0cbdf6958b6a2958a021906d346dbb76fa3650f71f01a821de009c37a6d79d.jpg)

![5143567845850b8588d1077df6ac698498237b0237ed4d6134c43b9cc9362abb.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/tables/5143567845850b8588d1077df6ac698498237b0237ed4d6134c43b9cc9362abb.jpg)

![77178702ba152bc31faf7b4e447015d524c5ceebfd8ce8d6de8e9908ca10caa3.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/tables/77178702ba152bc31faf7b4e447015d524c5ceebfd8ce8d6de8e9908ca10caa3.jpg)

![94996ebb984ea69287589a486081c2d51247ea69dc1c1d115aeb9b268787bd60.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/tables/94996ebb984ea69287589a486081c2d51247ea69dc1c1d115aeb9b268787bd60.jpg)

![a607122aa2255eb93750f992e5bf832e6e4c576760f8e82a4f5f999e7c35b93b.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/tables/a607122aa2255eb93750f992e5bf832e6e4c576760f8e82a4f5f999e7c35b93b.jpg)

![a74f9a3516da47955839090e88bbf897db0d9b791bbc2516c096b8e3bf6cf099.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/tables/a74f9a3516da47955839090e88bbf897db0d9b791bbc2516c096b8e3bf6cf099.jpg)

![b32852a04b66e15f81dcd16f1126118255294d664c89aaaed7aa08a177dd1990.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/tables/b32852a04b66e15f81dcd16f1126118255294d664c89aaaed7aa08a177dd1990.jpg)

![c5c9ae45aeeda42ec37d0cd5c8c66df75c89d4a7604bc9718fa7e504affb9a18.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/tables/c5c9ae45aeeda42ec37d0cd5c8c66df75c89d4a7604bc9718fa7e504affb9a18.jpg)

![d9de0c81f81692846d5a56433d06abead0bcad31cea76b353e18b2c22c331eab.jpg](../icml_results/5201_From%202D%20Grids%20to%201D%20Tokens_%20Reforming%20Shared%20Representations%20for%20Multimodal%20Image%20Fusion/tables/d9de0c81f81692846d5a56433d06abead0bcad31cea76b353e18b2c22c331eab.jpg)

<a id="paper-5154"></a>

## Logit-Attention Divergence: Mitigating Position Bias in Multi-Image Retrieval via Attention-Guided Calibration


### Images

![2660ff0ead3a63d697d50a2240e4061ecae04e3a6a26177f9aa33c6d4b6cc434.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/images/2660ff0ead3a63d697d50a2240e4061ecae04e3a6a26177f9aa33c6d4b6cc434.jpg)

![37c3b94800ed65175d2c6d1af9ce217d3db1bb2613b09b04f54321819e6d50b3.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/images/37c3b94800ed65175d2c6d1af9ce217d3db1bb2613b09b04f54321819e6d50b3.jpg)

![5a9507925d55ce5a0b51c0a58ac5bcfe9c289289c99f5b1eb5a87e95075e4612.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/images/5a9507925d55ce5a0b51c0a58ac5bcfe9c289289c99f5b1eb5a87e95075e4612.jpg)

![70cdfba6fe59933e3e54de895ad906b95ae47f30a6d9276cc6f1341340b07a53.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/images/70cdfba6fe59933e3e54de895ad906b95ae47f30a6d9276cc6f1341340b07a53.jpg)

![7bb5072bdb79529ca237fc421a270cb4cda5a85e426c6ff0a3f304a107cd0bb6.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/images/7bb5072bdb79529ca237fc421a270cb4cda5a85e426c6ff0a3f304a107cd0bb6.jpg)

![80305ff95846ee1ec9153e1f5fa7deed51825cea4c4d404317a80fd3d0eab230.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/images/80305ff95846ee1ec9153e1f5fa7deed51825cea4c4d404317a80fd3d0eab230.jpg)

![805dde35752f2e769d603d1aa34826fd22a36c5bda1de3c09bee18e8e4e1cbcf.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/images/805dde35752f2e769d603d1aa34826fd22a36c5bda1de3c09bee18e8e4e1cbcf.jpg)

![8ba5cbeb775bb4c84680b1d825f67e06f43e3545e4ff76b75b6d0b45b977da09.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/images/8ba5cbeb775bb4c84680b1d825f67e06f43e3545e4ff76b75b6d0b45b977da09.jpg)

![c680f1741694085944776af9e6f2f5c37675aa4071f83c2b269c241a6c18c55d.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/images/c680f1741694085944776af9e6f2f5c37675aa4071f83c2b269c241a6c18c55d.jpg)

![c882569d95f2081bcdaaaa4e71bf61169eba5bf22e9a88a61c70802f51b5a2d4.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/images/c882569d95f2081bcdaaaa4e71bf61169eba5bf22e9a88a61c70802f51b5a2d4.jpg)

### Tables

![0ac49cde7b8fe2d8da2f448138fc68fbce90848c958a1386dea59a15270e02b2.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/tables/0ac49cde7b8fe2d8da2f448138fc68fbce90848c958a1386dea59a15270e02b2.jpg)

![2977e41025e36a3bd6107fad30b5d654da729092b3e25aff9f287ff1234ed259.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/tables/2977e41025e36a3bd6107fad30b5d654da729092b3e25aff9f287ff1234ed259.jpg)

![3bae500276d4f1a6799adf3fbc3732a16ca73186436d89133b364ea616c91141.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/tables/3bae500276d4f1a6799adf3fbc3732a16ca73186436d89133b364ea616c91141.jpg)

![4009068298e9e13052b11aba24a7dca81b8b0055b7171cc9fc52f249a7c29143.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/tables/4009068298e9e13052b11aba24a7dca81b8b0055b7171cc9fc52f249a7c29143.jpg)

![8c07c3394b32dcb806ed23dd6ddeb61fbf1cc3b389964e38675509a2fac5f9ba.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/tables/8c07c3394b32dcb806ed23dd6ddeb61fbf1cc3b389964e38675509a2fac5f9ba.jpg)

![998bfef7b87b070b1933fc9a2acc22ca8d5c25e9ee64dd85ddbe5b225715be08.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/tables/998bfef7b87b070b1933fc9a2acc22ca8d5c25e9ee64dd85ddbe5b225715be08.jpg)

![d654b3e55944848fd467b670c5513470f365ef14a020781e95748227a4d7190f.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/tables/d654b3e55944848fd467b670c5513470f365ef14a020781e95748227a4d7190f.jpg)

![ec6370f301d0a933fa24927ce5eac2505933577292f8bead8074a9063596258c.jpg](../icml_results/5202_Logit-Attention%20Divergence_%20Mitigating%20Position%20Bias%20in%20Multi-Image%20Retrieval%20via%20Attention-Guided%20C/tables/ec6370f301d0a933fa24927ce5eac2505933577292f8bead8074a9063596258c.jpg)

<a id="paper-5155"></a>

## Rethinking Genomic Modeling Through Optical Character Recognition


### Images

![17ffca21ce81309748616507187f315c40f41bd67f4ec7264f321144e3cc3f22.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/images/17ffca21ce81309748616507187f315c40f41bd67f4ec7264f321144e3cc3f22.jpg)

![2d4e335efcb0f39dfaf9a878af9d26803490c4c064274934b835142c463a7965.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/images/2d4e335efcb0f39dfaf9a878af9d26803490c4c064274934b835142c463a7965.jpg)

![2e0052aa52d71612655a7bc6254bd11628170dd22e7be586df0e7a35a60898af.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/images/2e0052aa52d71612655a7bc6254bd11628170dd22e7be586df0e7a35a60898af.jpg)

![2e457a372fa3e84a86c4b3e4ae2d400c08f8d6047495e13ef7de88275d9eeaa6.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/images/2e457a372fa3e84a86c4b3e4ae2d400c08f8d6047495e13ef7de88275d9eeaa6.jpg)

![3e48bfe45332dd8d3be989b1c416141b8137090418e8a485cdc1fd06920ec4e8.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/images/3e48bfe45332dd8d3be989b1c416141b8137090418e8a485cdc1fd06920ec4e8.jpg)

![72587d29cbc8b95c119a9274b76e4a1ff4642c7ba89f14b52a627bc959fe6d6c.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/images/72587d29cbc8b95c119a9274b76e4a1ff4642c7ba89f14b52a627bc959fe6d6c.jpg)

![882db69d15407bbab817b9175ac25b45b0a770e7fb78336ea82ad152492ba1e1.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/images/882db69d15407bbab817b9175ac25b45b0a770e7fb78336ea82ad152492ba1e1.jpg)

![8fce1deed3a22e95b3bb42eca383ca0a3b7647bfcf89f171b6ff601d1bbcea70.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/images/8fce1deed3a22e95b3bb42eca383ca0a3b7647bfcf89f171b6ff601d1bbcea70.jpg)

![c68563b8521d6aefccec2a37564d8a2b1d772ad8c4a3dd976c0719db914e92dc.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/images/c68563b8521d6aefccec2a37564d8a2b1d772ad8c4a3dd976c0719db914e92dc.jpg)

![c7b7f08d63ca0ae43f8fb572b4a74f272807edf24fbf2b079e56ef10930681ca.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/images/c7b7f08d63ca0ae43f8fb572b4a74f272807edf24fbf2b079e56ef10930681ca.jpg)

### Tables

![006389bea223f8f5454453c85d236ccfdb3fc8fc2a2e571836d6460fa79a0f2e.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/006389bea223f8f5454453c85d236ccfdb3fc8fc2a2e571836d6460fa79a0f2e.jpg)

![0714a29bae949a1c27e897ad2017c25b7fa67a8c568e8ea95e644713fd870316.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/0714a29bae949a1c27e897ad2017c25b7fa67a8c568e8ea95e644713fd870316.jpg)

![09fdb43f9b54499f9d8b50ae6ed3f25dd1ecbd304ed5a418e37eefff39da4c90.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/09fdb43f9b54499f9d8b50ae6ed3f25dd1ecbd304ed5a418e37eefff39da4c90.jpg)

![10174be9ebef573487cd956264221f6e1e312f7575194736ce3fa5a54b2f5a13.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/10174be9ebef573487cd956264221f6e1e312f7575194736ce3fa5a54b2f5a13.jpg)

![10c4e74a70b6305d6e4d0801eee1784694ea5c6a764d1493db4ea54c2c743143.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/10c4e74a70b6305d6e4d0801eee1784694ea5c6a764d1493db4ea54c2c743143.jpg)

![111e1145577c3d37a15bb0a1d07d7e5da838878996c38fbee38a6817062415fa.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/111e1145577c3d37a15bb0a1d07d7e5da838878996c38fbee38a6817062415fa.jpg)

![12ce559b80ab30243fc55bd9a0c6a11c95e37b20cff6ea07f37fd92426e19301.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/12ce559b80ab30243fc55bd9a0c6a11c95e37b20cff6ea07f37fd92426e19301.jpg)

![16582ee1b800ed1ef0ce2b1f911d8465a6d64c2ff322466c7a009dbfb401337b.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/16582ee1b800ed1ef0ce2b1f911d8465a6d64c2ff322466c7a009dbfb401337b.jpg)

![166cd1a55b3e84165bc4e48a6beb91d63834a30fb5a4f555c3d345d5e7f042c6.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/166cd1a55b3e84165bc4e48a6beb91d63834a30fb5a4f555c3d345d5e7f042c6.jpg)

![233a223480f29b44efd04f26599ffdee8bd62555f3d1bd99533c6f89689213ef.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/233a223480f29b44efd04f26599ffdee8bd62555f3d1bd99533c6f89689213ef.jpg)

![2a41d2a1754eae2442c5538e76661ff82e3218ef5748b27655e2160a2c874077.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/2a41d2a1754eae2442c5538e76661ff82e3218ef5748b27655e2160a2c874077.jpg)

![2bff3743a15d136ffaa5a7a9d3ea3522b4055d734c6aea2e11abe4dd1e5095a6.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/2bff3743a15d136ffaa5a7a9d3ea3522b4055d734c6aea2e11abe4dd1e5095a6.jpg)

![2ccaad711e95bc2691528dd4579d175264adf2fd9e34734de65550631e7d50ee.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/2ccaad711e95bc2691528dd4579d175264adf2fd9e34734de65550631e7d50ee.jpg)

![2d596516b3aad1d61752393b4b86d15a361d6f95fbc2ae28ab5d69364dd39913.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/2d596516b3aad1d61752393b4b86d15a361d6f95fbc2ae28ab5d69364dd39913.jpg)

![2e8daf57314cd89b782d15a14044ffa0af0e6fbd3e615ddeb66c40e15e2c7685.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/2e8daf57314cd89b782d15a14044ffa0af0e6fbd3e615ddeb66c40e15e2c7685.jpg)

![3231cc1b6bc6ef63dec4361b33d14f6254c97598086b4d6c617d7142f67faa26.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/3231cc1b6bc6ef63dec4361b33d14f6254c97598086b4d6c617d7142f67faa26.jpg)

![3eefd9af6f8980be7183a956e5607d1c54638dfa37b963bbd17bd2a970dd5d63.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/3eefd9af6f8980be7183a956e5607d1c54638dfa37b963bbd17bd2a970dd5d63.jpg)

![3f649f0be4602eb39af902efe3d07d36ea2de54349da833ed143a8f62f032417.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/3f649f0be4602eb39af902efe3d07d36ea2de54349da833ed143a8f62f032417.jpg)

![5b36e9b03076622a46f525edf391f4d6817fe4a1040d3b60904ad0d73651ce7a.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/5b36e9b03076622a46f525edf391f4d6817fe4a1040d3b60904ad0d73651ce7a.jpg)

![5f040b6571dc84957c55b91e2eaa68f9e83d853b629905669c53b7d7e0cf99fa.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/5f040b6571dc84957c55b91e2eaa68f9e83d853b629905669c53b7d7e0cf99fa.jpg)

![623b7dc3070b499a275156e6fb526663b5ecd6a83d837dbe57d1cf888233867d.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/623b7dc3070b499a275156e6fb526663b5ecd6a83d837dbe57d1cf888233867d.jpg)

![6655ad40b32f42b656e003766e867417c2d0a5053b75bf00bfb1112f973acf56.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/6655ad40b32f42b656e003766e867417c2d0a5053b75bf00bfb1112f973acf56.jpg)

![70e4f6be2cc9ed152d28d021c552479ff1a1eb07c38445f81b7083675198054f.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/70e4f6be2cc9ed152d28d021c552479ff1a1eb07c38445f81b7083675198054f.jpg)

![714e1f6598b5f29db836ffc341677e97fafcaaf2e92916efac7cdd79766c4192.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/714e1f6598b5f29db836ffc341677e97fafcaaf2e92916efac7cdd79766c4192.jpg)

![77091f7009e4ff4b137c0b197546ef44c1f3891fe70282cfef1df65adae94393.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/77091f7009e4ff4b137c0b197546ef44c1f3891fe70282cfef1df65adae94393.jpg)

![81009741f54e779d53bdaeb2d0eaab1e6c9001bb50017393e6a847966a1c335f.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/81009741f54e779d53bdaeb2d0eaab1e6c9001bb50017393e6a847966a1c335f.jpg)

![855dc0f0ddebaa0997e710afbf75e1fff07f7930513c943ccd0c601372aab838.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/855dc0f0ddebaa0997e710afbf75e1fff07f7930513c943ccd0c601372aab838.jpg)

![887664194e9f994ee8b4495e12c2a9a19acae4be01274087b536a059c56593ff.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/887664194e9f994ee8b4495e12c2a9a19acae4be01274087b536a059c56593ff.jpg)

![8aeede662fc9538e4b4202986e0b2b2ad7c87a5f03b5d8e375cf33df0fa352c8.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/8aeede662fc9538e4b4202986e0b2b2ad7c87a5f03b5d8e375cf33df0fa352c8.jpg)

![9071f4e050ba639017bf8f83bef4a91406c75744e6599f63ca0c2cbe9be0afb0.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/9071f4e050ba639017bf8f83bef4a91406c75744e6599f63ca0c2cbe9be0afb0.jpg)

![98b6b81b72f8596c87a55b9e5871c85d7c24d35da6613d3217b49106a68bd735.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/98b6b81b72f8596c87a55b9e5871c85d7c24d35da6613d3217b49106a68bd735.jpg)

![9f4f317b0332e1a728e2eadd9deafb332d14307987088394f51c8306343ac129.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/9f4f317b0332e1a728e2eadd9deafb332d14307987088394f51c8306343ac129.jpg)

![a76b16933a3ea99902ec9bde700760f4defe4a7d7fb64cbbda8d1ad880a082ce.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/a76b16933a3ea99902ec9bde700760f4defe4a7d7fb64cbbda8d1ad880a082ce.jpg)

![a8df3aa14892714f781fd2b639788867413046222cccd19c0d8b02bb662a0d4e.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/a8df3aa14892714f781fd2b639788867413046222cccd19c0d8b02bb662a0d4e.jpg)

![ac635135c6df6a3eab8b4299275101ede98ac46e76296b49691b6079000f0eff.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/ac635135c6df6a3eab8b4299275101ede98ac46e76296b49691b6079000f0eff.jpg)

![b38a62741233cc1e5fe7a1936631e22a504aac8950541f33742acea0de12c82b.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/b38a62741233cc1e5fe7a1936631e22a504aac8950541f33742acea0de12c82b.jpg)

![b6211c6a30c4e964edb472a44b0656caee4fbfebdda4b1ed802015e30b0e83e4.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/b6211c6a30c4e964edb472a44b0656caee4fbfebdda4b1ed802015e30b0e83e4.jpg)

![b8bef2f7a7c69a732583f1fcbf041bfa8df0e6006d6b6e4a3049b76ec8250d2f.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/b8bef2f7a7c69a732583f1fcbf041bfa8df0e6006d6b6e4a3049b76ec8250d2f.jpg)

![b8cfc7d8b49d091758acd5aab661b75b99c1c9f2d8748019374508505d411852.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/b8cfc7d8b49d091758acd5aab661b75b99c1c9f2d8748019374508505d411852.jpg)

![bdf1d25ee612efb45f56b4a26ce2393d824e676dda519a9acad52bb25c6dae9e.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/bdf1d25ee612efb45f56b4a26ce2393d824e676dda519a9acad52bb25c6dae9e.jpg)

![be10f0cc775c95d046d4ce7bc9e4bcb902c106339ae7644098ccab9ea2253e79.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/be10f0cc775c95d046d4ce7bc9e4bcb902c106339ae7644098ccab9ea2253e79.jpg)

![ca795a46ba964c6ce180ea52e11ec718ed62bd5ca54eb95a4676bd7048d1ffa2.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/ca795a46ba964c6ce180ea52e11ec718ed62bd5ca54eb95a4676bd7048d1ffa2.jpg)

![cfccc6168251aebae17a23844870685389c3c59794fa44e1d173f8316d5d22db.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/cfccc6168251aebae17a23844870685389c3c59794fa44e1d173f8316d5d22db.jpg)

![d0d00837efff0cea6e8243512415da486bb5d9d4b6c5eb373afa68788a6f6143.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/d0d00837efff0cea6e8243512415da486bb5d9d4b6c5eb373afa68788a6f6143.jpg)

![d2ede7b7f0c3e4ffdf075e3e4a8c4c405ed9840ea0289b9e81f6a110e1e36846.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/d2ede7b7f0c3e4ffdf075e3e4a8c4c405ed9840ea0289b9e81f6a110e1e36846.jpg)

![e8bddd7d198bb8ddd8beafac9c3175b4be0cbc271c6c5d6df705a965449c2d76.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/e8bddd7d198bb8ddd8beafac9c3175b4be0cbc271c6c5d6df705a965449c2d76.jpg)

![f6b01d9cdde034e95a015b491f7f3535898dea50bf7e077ff03e74711ef73b21.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/f6b01d9cdde034e95a015b491f7f3535898dea50bf7e077ff03e74711ef73b21.jpg)

![fbe6b49ad0522db88470d9572a8b43e7f70a02f6a467362a8cad3d5085f71528.jpg](../icml_results/5203_Rethinking%20Genomic%20Modeling%20Through%20Optical%20Character%20Recognition/tables/fbe6b49ad0522db88470d9572a8b43e7f70a02f6a467362a8cad3d5085f71528.jpg)

<a id="paper-5156"></a>

## Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility


### Images

![17ec068c47229e9c3a9c0257ab3d9f596602764a80cab48f2d4d4edd6f57d1a6.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/images/17ec068c47229e9c3a9c0257ab3d9f596602764a80cab48f2d4d4edd6f57d1a6.jpg)

![194857109e683594ed3b66849d53c30935cefe932b04e9ee16e793a82dc4748e.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/images/194857109e683594ed3b66849d53c30935cefe932b04e9ee16e793a82dc4748e.jpg)

![3ab777342cd1e157753d601a444a13e9543030bbcf2fb4302b5202dce1b2f4c3.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/images/3ab777342cd1e157753d601a444a13e9543030bbcf2fb4302b5202dce1b2f4c3.jpg)

![5e07ec09c7070e4989f57fb0d79bb520a88d972d2b6f10b6fa5a7d718b096d9e.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/images/5e07ec09c7070e4989f57fb0d79bb520a88d972d2b6f10b6fa5a7d718b096d9e.jpg)

![64ed51333d73642f8381c035e74e6d421f7061fddebb52439f1cc2a51961ec4c.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/images/64ed51333d73642f8381c035e74e6d421f7061fddebb52439f1cc2a51961ec4c.jpg)

![6adbc2547f787f1c33cf6a5742f6abdefde93c262050a6bddae1abc6ee2a3b2a.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/images/6adbc2547f787f1c33cf6a5742f6abdefde93c262050a6bddae1abc6ee2a3b2a.jpg)

![711d3e7fc87df4a439d746febca5a7a5ea2728848bc7769e893008007f7c1524.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/images/711d3e7fc87df4a439d746febca5a7a5ea2728848bc7769e893008007f7c1524.jpg)

![e1d6a23293f0ef8270bb9092f43e6a8c231ca8d432d1147796a5b4c40e894bf2.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/images/e1d6a23293f0ef8270bb9092f43e6a8c231ca8d432d1147796a5b4c40e894bf2.jpg)

![f39aefda3d184ee65fbc0603e03e45066688a5cb1db1e6f761f9b968903572ec.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/images/f39aefda3d184ee65fbc0603e03e45066688a5cb1db1e6f761f9b968903572ec.jpg)

### Tables

![07e26d6992e26184ebb31bd0391e93888153bf5c013e19e3aea9d750db7e37aa.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/tables/07e26d6992e26184ebb31bd0391e93888153bf5c013e19e3aea9d750db7e37aa.jpg)

![14706149a9fb67ad5871a9a6cb4e31f51785fac9817a17093b60555f7bf4949e.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/tables/14706149a9fb67ad5871a9a6cb4e31f51785fac9817a17093b60555f7bf4949e.jpg)

![1da25dee2623dff0fc7e6036ccc55e463b775caf7b4cc54aa62c770677303540.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/tables/1da25dee2623dff0fc7e6036ccc55e463b775caf7b4cc54aa62c770677303540.jpg)

![2ce099140d3d27913c014989a05dd1950486d53ee624a557837b89f3ea900f80.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/tables/2ce099140d3d27913c014989a05dd1950486d53ee624a557837b89f3ea900f80.jpg)

![3212c6c69f86fe737cdccb0bbffcbef085bb81238be45b2d9a4a43aec40bf464.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/tables/3212c6c69f86fe737cdccb0bbffcbef085bb81238be45b2d9a4a43aec40bf464.jpg)

![56fe82cd5dc0a1329e4d964e1d93d03a3ad899b48d9667e55c1bec21c3f75c91.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/tables/56fe82cd5dc0a1329e4d964e1d93d03a3ad899b48d9667e55c1bec21c3f75c91.jpg)

![58a81f878858a3a28d218162a64d03d7180361966f8b87b7ee3730e363280bf6.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/tables/58a81f878858a3a28d218162a64d03d7180361966f8b87b7ee3730e363280bf6.jpg)

![70586e349cd46cc9bcb0322a057fbdf842eb4283002c1040c18ab5564643d4e6.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/tables/70586e349cd46cc9bcb0322a057fbdf842eb4283002c1040c18ab5564643d4e6.jpg)

![7231de3712793e9001a5aebf3952869076cf9f8c0602f00e8081c16c783f4545.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/tables/7231de3712793e9001a5aebf3952869076cf9f8c0602f00e8081c16c783f4545.jpg)

![8a371c6331ac2927dce5fcdc09fd652c46eb3cf978998dc19eb55c551cac6305.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/tables/8a371c6331ac2927dce5fcdc09fd652c46eb3cf978998dc19eb55c551cac6305.jpg)

![8b3fd10f1e86c01545a5330517ec4f277893c15008dea425487e1b26677f3c0c.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/tables/8b3fd10f1e86c01545a5330517ec4f277893c15008dea425487e1b26677f3c0c.jpg)

![c1fae13c9d94f6aa6f2c1b35f19c9bc31b580f275f86abd9514fb4eb6e0caf47.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/tables/c1fae13c9d94f6aa6f2c1b35f19c9bc31b580f275f86abd9514fb4eb6e0caf47.jpg)

![c923707be8edc1a1c304ee86d3bfb7ae07f68059ad53c01bd6ab0f29e0ca8d12.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/tables/c923707be8edc1a1c304ee86d3bfb7ae07f68059ad53c01bd6ab0f29e0ca8d12.jpg)

![cec7e4f157091f381547804deff3474a828350beb10d44a98e9c81b4d98bdd4f.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/tables/cec7e4f157091f381547804deff3474a828350beb10d44a98e9c81b4d98bdd4f.jpg)

![d6588081950eb07ddf9cae5f11848c3dffc9f292f5908c3edbd436a56c2a16f9.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/tables/d6588081950eb07ddf9cae5f11848c3dffc9f292f5908c3edbd436a56c2a16f9.jpg)

![f7d274bf8ab9c2102c628b1b363e536dbe1ace4d4af6293281640ef8314dc0ac.jpg](../icml_results/5204_Multilingual%20Unlearning%20in%20LLMs_%20Transfer%2C%20Dynamics%2C%20and%20Reversibility/tables/f7d274bf8ab9c2102c628b1b363e536dbe1ace4d4af6293281640ef8314dc0ac.jpg)

<a id="paper-5157"></a>

## New Wide-Net-Casting Jailbreak Attacks Risk Large Models


### Images

![1fd0b9093a6899bf01f3d9d4a6cde732bef3fb421685391e1d6a21bd38f09fc7.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/images/1fd0b9093a6899bf01f3d9d4a6cde732bef3fb421685391e1d6a21bd38f09fc7.jpg)

![4273a92b21bd3a561eff2e7c8973d91430017945f5ba8ce3da463dd61fbbf5bb.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/images/4273a92b21bd3a561eff2e7c8973d91430017945f5ba8ce3da463dd61fbbf5bb.jpg)

![51a0d5f1ee2d23a5bfce3c5be789aa3f878e0a46f00dbbcd27e7b0fae46c0a03.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/images/51a0d5f1ee2d23a5bfce3c5be789aa3f878e0a46f00dbbcd27e7b0fae46c0a03.jpg)

![64fdc4604a45dedfdece862f19eb84167762be34149b2591f6391af2e28c7ccf.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/images/64fdc4604a45dedfdece862f19eb84167762be34149b2591f6391af2e28c7ccf.jpg)

![79cdf76842d467b3d2704ab41dcbeff0285660950063aab1585d42ad389b33e3.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/images/79cdf76842d467b3d2704ab41dcbeff0285660950063aab1585d42ad389b33e3.jpg)

![7ba03aa892179f40e6e63c28b228f189fef1a04ca6050d1562a3d01de8b38e60.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/images/7ba03aa892179f40e6e63c28b228f189fef1a04ca6050d1562a3d01de8b38e60.jpg)

### Tables

![074726a17459a98fdffcb3f1215d7e747c723cdedaa89b416b6c991b0d916a78.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/074726a17459a98fdffcb3f1215d7e747c723cdedaa89b416b6c991b0d916a78.jpg)

![14178f7bab5627e40b3cd908a9b45119f64d7d94cae869aa573d9702dc18349c.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/14178f7bab5627e40b3cd908a9b45119f64d7d94cae869aa573d9702dc18349c.jpg)

![291af11fba3aaa1cb3d1cf58abf0b276c5f29a54f73a3cf8340a2744a34b32d5.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/291af11fba3aaa1cb3d1cf58abf0b276c5f29a54f73a3cf8340a2744a34b32d5.jpg)

![2ef066c4b4beb4f1d8071b98556df813d351f31a1ee785304122a6f51cc6db24.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/2ef066c4b4beb4f1d8071b98556df813d351f31a1ee785304122a6f51cc6db24.jpg)

![2f685372ec06bb3a6096e43683e145605577ae6c1ccf8e1e36af5b87db545a26.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/2f685372ec06bb3a6096e43683e145605577ae6c1ccf8e1e36af5b87db545a26.jpg)

![45f260c4a93fdd60386b85c6b34b725dc5d2e3b0635499fa9ae6dd093c3a1db6.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/45f260c4a93fdd60386b85c6b34b725dc5d2e3b0635499fa9ae6dd093c3a1db6.jpg)

![4e648143ead8b58dfe0fb72923b6060ed318a7aa59e3c542add83bd392b48398.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/4e648143ead8b58dfe0fb72923b6060ed318a7aa59e3c542add83bd392b48398.jpg)

![533b30d5fb69732a68b2d73757fb4b51d2ff4433aaa03d98fe313f157a2c56ac.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/533b30d5fb69732a68b2d73757fb4b51d2ff4433aaa03d98fe313f157a2c56ac.jpg)

![5688f26cebd92b546bae7cdf9b1642cac92420ab0a2cb51ffd2a380b74596070.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/5688f26cebd92b546bae7cdf9b1642cac92420ab0a2cb51ffd2a380b74596070.jpg)

![696c12ab0ca8810cee1a81f0c321a505b8ce51a41a66b8f2291184d92b9c33a4.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/696c12ab0ca8810cee1a81f0c321a505b8ce51a41a66b8f2291184d92b9c33a4.jpg)

![6ccd5800b31361c7926d0dd1578a4b9a60660c342fc8c4a25d1b178150d8854f.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/6ccd5800b31361c7926d0dd1578a4b9a60660c342fc8c4a25d1b178150d8854f.jpg)

![6d588514cd03f79031b03cac69c69200e0baba4875fc0bbad5a963a0bbc30b09.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/6d588514cd03f79031b03cac69c69200e0baba4875fc0bbad5a963a0bbc30b09.jpg)

![736b152b6c3aac2fea3633e92df53d08570833ff34d50ab06befe71daa9f9be9.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/736b152b6c3aac2fea3633e92df53d08570833ff34d50ab06befe71daa9f9be9.jpg)

![7502e27535940d01451b0b0bc20187bc1493c3bf1f28ac58f89162ac3b7bdf7f.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/7502e27535940d01451b0b0bc20187bc1493c3bf1f28ac58f89162ac3b7bdf7f.jpg)

![82fac8bd49a740baff4e973f4644a5f7edeae62d56a50f40a67925e6425b961c.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/82fac8bd49a740baff4e973f4644a5f7edeae62d56a50f40a67925e6425b961c.jpg)

![a3ea52f23564b4a7b1ffeb20603d4a8638c8ff35e804d49d183ef5b27e38a88a.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/a3ea52f23564b4a7b1ffeb20603d4a8638c8ff35e804d49d183ef5b27e38a88a.jpg)

![ab97e55d5cb3b9d8d37313dd9e3d9cad91f40a87e3ced9b9f7575d27183b1424.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/ab97e55d5cb3b9d8d37313dd9e3d9cad91f40a87e3ced9b9f7575d27183b1424.jpg)

![c5b6a1105f6d0c84ea03795b547cebaa3f393a51f64703ab3da1d18f45d6a803.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/c5b6a1105f6d0c84ea03795b547cebaa3f393a51f64703ab3da1d18f45d6a803.jpg)

![dd21bb6178dce3b5e24e226b9bfffb3541bf73ad18f57f3bb24d06549ecf9187.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/dd21bb6178dce3b5e24e226b9bfffb3541bf73ad18f57f3bb24d06549ecf9187.jpg)

![e83e64076b96bad5594e60fd76ea102ae2c1c6f151268721ec78858960fe2d2b.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/e83e64076b96bad5594e60fd76ea102ae2c1c6f151268721ec78858960fe2d2b.jpg)

![f14179b5ea6f7a9d4270fcec3bb79267bc39c9323bf59b6c933ffe8276f851f0.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/f14179b5ea6f7a9d4270fcec3bb79267bc39c9323bf59b6c933ffe8276f851f0.jpg)

![f33b6951fed5e5f15e08ab23e58380d0572414308cb92b5076acef1c26f6f90b.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/f33b6951fed5e5f15e08ab23e58380d0572414308cb92b5076acef1c26f6f90b.jpg)

![fc338f3323b6d42ae6d7dec1c069d7baf510544589fa8656f5a9be690ea33534.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/fc338f3323b6d42ae6d7dec1c069d7baf510544589fa8656f5a9be690ea33534.jpg)

![fe658818d75c8c17c70cacf4dd0a5b2f919108929960645b4ad5363138c05668.jpg](../icml_results/5205_New%20Wide-Net-Casting%20Jailbreak%20Attacks%20Risk%20Large%20Models/tables/fe658818d75c8c17c70cacf4dd0a5b2f919108929960645b4ad5363138c05668.jpg)

<a id="paper-5158"></a>

## MOD-SR: Unifying Multimodal Learning and Direct Optimization with Gradient-Guided Diffusion Model for Symbolic Regression


### Images

![0846f5121eb30cf226f63fb2c944c44c30e9a42795fbd45502ce85a9f82a2c9b.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/images/0846f5121eb30cf226f63fb2c944c44c30e9a42795fbd45502ce85a9f82a2c9b.jpg)

![1ee0e62f4e84289bedfeb15991e098df1b96cde484d81c7068fd19b9f34b6fb5.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/images/1ee0e62f4e84289bedfeb15991e098df1b96cde484d81c7068fd19b9f34b6fb5.jpg)

![4b36bdb935cdd5895b39232b3136e52a2553d281f7a96e12a973df125b28d497.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/images/4b36bdb935cdd5895b39232b3136e52a2553d281f7a96e12a973df125b28d497.jpg)

![55649eed666a9e87e39fc329d3faf0587c155460eec010b18554c5eb1b5a6ef5.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/images/55649eed666a9e87e39fc329d3faf0587c155460eec010b18554c5eb1b5a6ef5.jpg)

![5c03e4784a6da2ffe995211caeb71e7b118675646c18e72813b90aa21a42c15c.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/images/5c03e4784a6da2ffe995211caeb71e7b118675646c18e72813b90aa21a42c15c.jpg)

![5ed8183406d952dde08069019e87dca5338b4fb2ab8e8fb827c753ed585f7f93.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/images/5ed8183406d952dde08069019e87dca5338b4fb2ab8e8fb827c753ed585f7f93.jpg)

![64099c0de3567478d0db52a7f7a31151d4a26ac70e22317cb695be73d7330188.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/images/64099c0de3567478d0db52a7f7a31151d4a26ac70e22317cb695be73d7330188.jpg)

![89d8325df4c7970fffa4fa54a012b1b4054d9c5925f6a6160c3b17c0b274ffb7.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/images/89d8325df4c7970fffa4fa54a012b1b4054d9c5925f6a6160c3b17c0b274ffb7.jpg)

![8b72879da5a9d3205ff67caf8225fe80041904ca1340437c6dc7ee7242a21a82.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/images/8b72879da5a9d3205ff67caf8225fe80041904ca1340437c6dc7ee7242a21a82.jpg)

![b45269d6217ed893745fe14e2953885eada49703ad0a66b44ab9ccb93841c243.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/images/b45269d6217ed893745fe14e2953885eada49703ad0a66b44ab9ccb93841c243.jpg)

### Tables

![15ac8b664ca9b3939b646198815ad6b58c984bae25e59f8e52d90311baedd139.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/tables/15ac8b664ca9b3939b646198815ad6b58c984bae25e59f8e52d90311baedd139.jpg)

![3323885ecfdbb63704024d633d25b17eb85ba9c18db786bace3ae98af70d64f7.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/tables/3323885ecfdbb63704024d633d25b17eb85ba9c18db786bace3ae98af70d64f7.jpg)

![396e5aa61206fd7ef7a509c9e1900cc5e46b4d8c8f430dbf69af3b7fc8513813.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/tables/396e5aa61206fd7ef7a509c9e1900cc5e46b4d8c8f430dbf69af3b7fc8513813.jpg)

![40c13cbf3297691329cddefa508371a04d11c79c7ee056825e4082fb35c1244c.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/tables/40c13cbf3297691329cddefa508371a04d11c79c7ee056825e4082fb35c1244c.jpg)

![6223e0fb0699e322b8e3372c82bb418d67ec70d894cd75df9001648b553620b6.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/tables/6223e0fb0699e322b8e3372c82bb418d67ec70d894cd75df9001648b553620b6.jpg)

![65e64b2db7f859d2c2b2c523659b7f8250174880fdafefdf06d35d39203b10ee.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/tables/65e64b2db7f859d2c2b2c523659b7f8250174880fdafefdf06d35d39203b10ee.jpg)

![7ffa17c7c868c0964e77b035764d0161934a82a275b6dd0d67b40bd5f9e11296.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/tables/7ffa17c7c868c0964e77b035764d0161934a82a275b6dd0d67b40bd5f9e11296.jpg)

![95be94ef9bd8cb1b31fa2af02ce3e2c52db3a66c07d1509d00df162e18d90748.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/tables/95be94ef9bd8cb1b31fa2af02ce3e2c52db3a66c07d1509d00df162e18d90748.jpg)

![9ba50d7aa8fafff8274e32baaf44649bf9aa8a57f074f908f44e281522ef1471.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/tables/9ba50d7aa8fafff8274e32baaf44649bf9aa8a57f074f908f44e281522ef1471.jpg)

![a50345742c6d299b455ab3de9a09c10c73e3e8eb5d0c38a9cba31e780251c297.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/tables/a50345742c6d299b455ab3de9a09c10c73e3e8eb5d0c38a9cba31e780251c297.jpg)

![c76b82bbe736f8ac2af366936a4bf35f84d86ca70b6bc4fdd1367dbcbe003309.jpg](../icml_results/5206_MOD-SR_%20Unifying%20Multimodal%20Learning%20and%20Direct%20Optimization%20with%20Gradient-Guided%20Diffusion%20Model%20fo/tables/c76b82bbe736f8ac2af366936a4bf35f84d86ca70b6bc4fdd1367dbcbe003309.jpg)

<a id="paper-5159"></a>

## Stop the Flip-Flop: Context-Preserving Verification for Fast Revocable Diffusion Decoding


### Images

![081f8cde335c2ece1069f5c1a0a14460fe89091471f5a8ae964c5215d2cc9360.jpg](../icml_results/5207_Stop%20the%20Flip-Flop_%20Context-Preserving%20Verification%20for%20Fast%20Revocable%20Diffusion%20Decoding/images/081f8cde335c2ece1069f5c1a0a14460fe89091471f5a8ae964c5215d2cc9360.jpg)

![4b8b213c662285c20f70d6ed292869c9518878ef6b8dea2097a929794a98a8e1.jpg](../icml_results/5207_Stop%20the%20Flip-Flop_%20Context-Preserving%20Verification%20for%20Fast%20Revocable%20Diffusion%20Decoding/images/4b8b213c662285c20f70d6ed292869c9518878ef6b8dea2097a929794a98a8e1.jpg)

![5a71443e99ec1d02f65a490bfbc44bb620da0aec90d777a921c267a7708c7208.jpg](../icml_results/5207_Stop%20the%20Flip-Flop_%20Context-Preserving%20Verification%20for%20Fast%20Revocable%20Diffusion%20Decoding/images/5a71443e99ec1d02f65a490bfbc44bb620da0aec90d777a921c267a7708c7208.jpg)

![dba7bf57cb9affaeb3b487fc1dbb9c710c3dab35e22c7d8a7810692c3c555a65.jpg](../icml_results/5207_Stop%20the%20Flip-Flop_%20Context-Preserving%20Verification%20for%20Fast%20Revocable%20Diffusion%20Decoding/images/dba7bf57cb9affaeb3b487fc1dbb9c710c3dab35e22c7d8a7810692c3c555a65.jpg)

### Tables

![2208fdc06ffdf1766b30982bde3d9bc5a77efe267b08cc15d17c30eea0bc08ed.jpg](../icml_results/5207_Stop%20the%20Flip-Flop_%20Context-Preserving%20Verification%20for%20Fast%20Revocable%20Diffusion%20Decoding/tables/2208fdc06ffdf1766b30982bde3d9bc5a77efe267b08cc15d17c30eea0bc08ed.jpg)

![3ae46ac3d212d63a7d93a2a1319d50db12dcddaa140ee6555af70768a052af7e.jpg](../icml_results/5207_Stop%20the%20Flip-Flop_%20Context-Preserving%20Verification%20for%20Fast%20Revocable%20Diffusion%20Decoding/tables/3ae46ac3d212d63a7d93a2a1319d50db12dcddaa140ee6555af70768a052af7e.jpg)

![3bafffaa1a88cc9a2636c02e62ec17eef53de328f30f08686b6c08bb19c65bc8.jpg](../icml_results/5207_Stop%20the%20Flip-Flop_%20Context-Preserving%20Verification%20for%20Fast%20Revocable%20Diffusion%20Decoding/tables/3bafffaa1a88cc9a2636c02e62ec17eef53de328f30f08686b6c08bb19c65bc8.jpg)

![7d01777e37d2ba372ffa88babbcafb6c3436314107c15a072c31cfdf988b4941.jpg](../icml_results/5207_Stop%20the%20Flip-Flop_%20Context-Preserving%20Verification%20for%20Fast%20Revocable%20Diffusion%20Decoding/tables/7d01777e37d2ba372ffa88babbcafb6c3436314107c15a072c31cfdf988b4941.jpg)

![8ffd5ddd2bdf0ee36d29d3b238e20ddd29b68d55bf38108ce7abe97391d2bd81.jpg](../icml_results/5207_Stop%20the%20Flip-Flop_%20Context-Preserving%20Verification%20for%20Fast%20Revocable%20Diffusion%20Decoding/tables/8ffd5ddd2bdf0ee36d29d3b238e20ddd29b68d55bf38108ce7abe97391d2bd81.jpg)

![cbf53b3f4265f500af1f8a33b608c0287ebf3e931799e4a4b8c91da2dc1a97e2.jpg](../icml_results/5207_Stop%20the%20Flip-Flop_%20Context-Preserving%20Verification%20for%20Fast%20Revocable%20Diffusion%20Decoding/tables/cbf53b3f4265f500af1f8a33b608c0287ebf3e931799e4a4b8c91da2dc1a97e2.jpg)

![e970072e002b39e7102f8b9ec601146452c214c2ec6d5bcc295dcba8d3e4c8d6.jpg](../icml_results/5207_Stop%20the%20Flip-Flop_%20Context-Preserving%20Verification%20for%20Fast%20Revocable%20Diffusion%20Decoding/tables/e970072e002b39e7102f8b9ec601146452c214c2ec6d5bcc295dcba8d3e4c8d6.jpg)

<a id="paper-5160"></a>

## Spike Camera Autofocus via Frequency-Domain Spectral-Centroid Migration


### Images

![0a76957e298112edcfa7873532d11c83ae8ff33837f1d95f684fae3480a8400b.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/images/0a76957e298112edcfa7873532d11c83ae8ff33837f1d95f684fae3480a8400b.jpg)

![0f4e865ebf8b55242a5d7d76eb9b05a7dfcec6f72b71ef0f358cb84633129e5c.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/images/0f4e865ebf8b55242a5d7d76eb9b05a7dfcec6f72b71ef0f358cb84633129e5c.jpg)

![28a0bf48d789fafc95c31a87de64f422ebed323d2e48df5670cc14a833a8846e.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/images/28a0bf48d789fafc95c31a87de64f422ebed323d2e48df5670cc14a833a8846e.jpg)

![3e1809b4ff9236f16737fb5a50c1b5b178888a81a6815cea7106d76a7fa7725b.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/images/3e1809b4ff9236f16737fb5a50c1b5b178888a81a6815cea7106d76a7fa7725b.jpg)

![47a6881941a77709b8d0d12560dc18b26aeaeaf2ef196b7e1b34c22bc690be29.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/images/47a6881941a77709b8d0d12560dc18b26aeaeaf2ef196b7e1b34c22bc690be29.jpg)

![d67eeb6151d1e10f427ca666fefc6ce2e00ffb5160fa1d530e5ce060bb68c5e8.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/images/d67eeb6151d1e10f427ca666fefc6ce2e00ffb5160fa1d530e5ce060bb68c5e8.jpg)

![e5cc5e4700a0190a405af20823f5843f54a54945a32a3ac8c3c9916a44a106a9.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/images/e5cc5e4700a0190a405af20823f5843f54a54945a32a3ac8c3c9916a44a106a9.jpg)

![ff126c58e75191afd402254228649653de53bdb04c170956ab495af40f4e6f9b.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/images/ff126c58e75191afd402254228649653de53bdb04c170956ab495af40f4e6f9b.jpg)

### Tables

![374b812ea08bbe4eab64a7d18caa237af6cae3b69c071c07f661c9d85e890107.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/tables/374b812ea08bbe4eab64a7d18caa237af6cae3b69c071c07f661c9d85e890107.jpg)

![4a2410b0270c4a967023e04526ea3abee6987a81189a499b8063421fdd65d2fb.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/tables/4a2410b0270c4a967023e04526ea3abee6987a81189a499b8063421fdd65d2fb.jpg)

![618b4b4147caf3f9e7c2cae6ca61af77e99614e7aba16ab1bc30f599abf65ac9.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/tables/618b4b4147caf3f9e7c2cae6ca61af77e99614e7aba16ab1bc30f599abf65ac9.jpg)

![644a734028ec79f3bf3211cae27addcd1b01aa51070df2d9b25a2210e8dc0d02.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/tables/644a734028ec79f3bf3211cae27addcd1b01aa51070df2d9b25a2210e8dc0d02.jpg)

![7b64269ab1665e492239ecdda636cecf4dee8d90adad40eb1be5397d3eb1cc82.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/tables/7b64269ab1665e492239ecdda636cecf4dee8d90adad40eb1be5397d3eb1cc82.jpg)

![90402e6833db93a2d3c48bffcda2f5728beb119f78c05c3b63b6d181b47d2552.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/tables/90402e6833db93a2d3c48bffcda2f5728beb119f78c05c3b63b6d181b47d2552.jpg)

![9fa864768d153ba5ed18babb3cd712e86a0fa51f4c9e02088edf233e98b0938c.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/tables/9fa864768d153ba5ed18babb3cd712e86a0fa51f4c9e02088edf233e98b0938c.jpg)

![e62310904c4c7be9a496866b7d85920e3f12552cf2b53a4df38c116c7ab26c8a.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/tables/e62310904c4c7be9a496866b7d85920e3f12552cf2b53a4df38c116c7ab26c8a.jpg)

![e6bef8300f5bb8098f36180f44129c9b4113e6cfbb2521f81a364dfaca0b42de.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/tables/e6bef8300f5bb8098f36180f44129c9b4113e6cfbb2521f81a364dfaca0b42de.jpg)

![f17b84a1751ff73ee47a9e4705b0262c2edcafc58c8709be8784f059c1c76e0e.jpg](../icml_results/5208_Spike%20Camera%20Autofocus%20via%20Frequency-Domain%20Spectral-Centroid%20Migration/tables/f17b84a1751ff73ee47a9e4705b0262c2edcafc58c8709be8784f059c1c76e0e.jpg)

<a id="paper-5161"></a>

## Seeing the Unseen: Physics-as-Representation for Generalizable Gaze Perception


### Images

![13a8c4f0696b5aaedda68eb81f69d7b63a95b2c5d669cf38418b3581b7f266f3.jpg](../icml_results/5209_Seeing%20the%20Unseen_%20Physics-as-Representation%20for%20Generalizable%20Gaze%20Perception/images/13a8c4f0696b5aaedda68eb81f69d7b63a95b2c5d669cf38418b3581b7f266f3.jpg)

![4018646e8136706046c5255f1e7e42c2261db54dfadd9df2738e63ccc698472a.jpg](../icml_results/5209_Seeing%20the%20Unseen_%20Physics-as-Representation%20for%20Generalizable%20Gaze%20Perception/images/4018646e8136706046c5255f1e7e42c2261db54dfadd9df2738e63ccc698472a.jpg)

![56043bce6841b8be18ed3b1828cb7d8e935aa156d069a4f004bbd2e94325e742.jpg](../icml_results/5209_Seeing%20the%20Unseen_%20Physics-as-Representation%20for%20Generalizable%20Gaze%20Perception/images/56043bce6841b8be18ed3b1828cb7d8e935aa156d069a4f004bbd2e94325e742.jpg)

![80f5f8aef65f3b8bc5ec223e6835bd2b04b4cab5d2258441595b6ab2f8981480.jpg](../icml_results/5209_Seeing%20the%20Unseen_%20Physics-as-Representation%20for%20Generalizable%20Gaze%20Perception/images/80f5f8aef65f3b8bc5ec223e6835bd2b04b4cab5d2258441595b6ab2f8981480.jpg)

![a21ce7a7f8c54f0388edcad9a09c9d3e55416e629fe5d1f965ac5b610445d1b3.jpg](../icml_results/5209_Seeing%20the%20Unseen_%20Physics-as-Representation%20for%20Generalizable%20Gaze%20Perception/images/a21ce7a7f8c54f0388edcad9a09c9d3e55416e629fe5d1f965ac5b610445d1b3.jpg)

![b042b7e5f94f22fe45c651112df75fc1494308fcab38f70eee7c6a96da3fa9dc.jpg](../icml_results/5209_Seeing%20the%20Unseen_%20Physics-as-Representation%20for%20Generalizable%20Gaze%20Perception/images/b042b7e5f94f22fe45c651112df75fc1494308fcab38f70eee7c6a96da3fa9dc.jpg)

![bcd2d89a10480b8af615b8fea39261101903a7497bcb1f610a8a6fcbb1e2c7f3.jpg](../icml_results/5209_Seeing%20the%20Unseen_%20Physics-as-Representation%20for%20Generalizable%20Gaze%20Perception/images/bcd2d89a10480b8af615b8fea39261101903a7497bcb1f610a8a6fcbb1e2c7f3.jpg)

![c4827e473fa7db92cd31463f476ea42d18aa5888a0246155e7befa4b157af357.jpg](../icml_results/5209_Seeing%20the%20Unseen_%20Physics-as-Representation%20for%20Generalizable%20Gaze%20Perception/images/c4827e473fa7db92cd31463f476ea42d18aa5888a0246155e7befa4b157af357.jpg)

![ef7d5e25b9d7747a5499819ea51fae2c59392e32aa7a36b4019161a83237bc2b.jpg](../icml_results/5209_Seeing%20the%20Unseen_%20Physics-as-Representation%20for%20Generalizable%20Gaze%20Perception/images/ef7d5e25b9d7747a5499819ea51fae2c59392e32aa7a36b4019161a83237bc2b.jpg)

### Tables

![09111a8786a80da703c3c585f830479d449bd2aee8e1fbdf418cbf61693448ed.jpg](../icml_results/5209_Seeing%20the%20Unseen_%20Physics-as-Representation%20for%20Generalizable%20Gaze%20Perception/tables/09111a8786a80da703c3c585f830479d449bd2aee8e1fbdf418cbf61693448ed.jpg)

![5ea567ec63d6bd3edd0606313cd8e89381846fb04e6cb2dc943f0a4e4433d99d.jpg](../icml_results/5209_Seeing%20the%20Unseen_%20Physics-as-Representation%20for%20Generalizable%20Gaze%20Perception/tables/5ea567ec63d6bd3edd0606313cd8e89381846fb04e6cb2dc943f0a4e4433d99d.jpg)

![60ad7bfd502fadbbd28b55bcfb5b1519426eebf76d16b31b0b0e580cf554ad09.jpg](../icml_results/5209_Seeing%20the%20Unseen_%20Physics-as-Representation%20for%20Generalizable%20Gaze%20Perception/tables/60ad7bfd502fadbbd28b55bcfb5b1519426eebf76d16b31b0b0e580cf554ad09.jpg)

![6266f75a9469cf8db007d90249e09a84c7ccaaff5da678ef9fa5e1dc978fdc8e.jpg](../icml_results/5209_Seeing%20the%20Unseen_%20Physics-as-Representation%20for%20Generalizable%20Gaze%20Perception/tables/6266f75a9469cf8db007d90249e09a84c7ccaaff5da678ef9fa5e1dc978fdc8e.jpg)

![f6d09148d92a26aaf9ccdf47c49fb642556b4326aad9e469d8e554fea0c67007.jpg](../icml_results/5209_Seeing%20the%20Unseen_%20Physics-as-Representation%20for%20Generalizable%20Gaze%20Perception/tables/f6d09148d92a26aaf9ccdf47c49fb642556b4326aad9e469d8e554fea0c67007.jpg)

<a id="paper-5162"></a>

## Influence-Disentangled Federated Training: Learning Models That Are Easy to Unlearn


### Images

![112dc9a617881aadb2063581a5df0bb035a818648beb38cd34162f56d8e65949.jpg](../icml_results/5210_Influence-Disentangled%20Federated%20Training_%20Learning%20Models%20That%20Are%20Easy%20to%20Unlearn/images/112dc9a617881aadb2063581a5df0bb035a818648beb38cd34162f56d8e65949.jpg)

![15db66ef9915b09d016fccffc6b8d5a79e480f156c61c02d5f14fe5f3efda285.jpg](../icml_results/5210_Influence-Disentangled%20Federated%20Training_%20Learning%20Models%20That%20Are%20Easy%20to%20Unlearn/images/15db66ef9915b09d016fccffc6b8d5a79e480f156c61c02d5f14fe5f3efda285.jpg)

![19b7cb10a9eed2ede62da85ac32935c5e9a5feddb8124f562463b9e27693cf2c.jpg](../icml_results/5210_Influence-Disentangled%20Federated%20Training_%20Learning%20Models%20That%20Are%20Easy%20to%20Unlearn/images/19b7cb10a9eed2ede62da85ac32935c5e9a5feddb8124f562463b9e27693cf2c.jpg)

![7cffc7379b3872aaac6340d78e2109ec6e6a23456bafb2e4d77e718792fa9a9b.jpg](../icml_results/5210_Influence-Disentangled%20Federated%20Training_%20Learning%20Models%20That%20Are%20Easy%20to%20Unlearn/images/7cffc7379b3872aaac6340d78e2109ec6e6a23456bafb2e4d77e718792fa9a9b.jpg)

![c82cee474d0f1303afb62fc24ba8e2697c31b60e2886e0b416dafef77d442ea6.jpg](../icml_results/5210_Influence-Disentangled%20Federated%20Training_%20Learning%20Models%20That%20Are%20Easy%20to%20Unlearn/images/c82cee474d0f1303afb62fc24ba8e2697c31b60e2886e0b416dafef77d442ea6.jpg)

![f6ff83de124e80d738fd294da461268f527b8bb3d1a6ee96265d2b75562efecb.jpg](../icml_results/5210_Influence-Disentangled%20Federated%20Training_%20Learning%20Models%20That%20Are%20Easy%20to%20Unlearn/images/f6ff83de124e80d738fd294da461268f527b8bb3d1a6ee96265d2b75562efecb.jpg)

### Tables

![544e6b6dd4c35e0ecb49cbd17f555c8f21aab6fbcc1a5af42f1d5a2c23aa7af8.jpg](../icml_results/5210_Influence-Disentangled%20Federated%20Training_%20Learning%20Models%20That%20Are%20Easy%20to%20Unlearn/tables/544e6b6dd4c35e0ecb49cbd17f555c8f21aab6fbcc1a5af42f1d5a2c23aa7af8.jpg)

![5db593efe9797392c5ba131ceabd89522512bb018d71d5ee68b79250331b271d.jpg](../icml_results/5210_Influence-Disentangled%20Federated%20Training_%20Learning%20Models%20That%20Are%20Easy%20to%20Unlearn/tables/5db593efe9797392c5ba131ceabd89522512bb018d71d5ee68b79250331b271d.jpg)

![6b1259a048b05739bc82b5505f90223d964942edf0a49bc0800540bbb6703564.jpg](../icml_results/5210_Influence-Disentangled%20Federated%20Training_%20Learning%20Models%20That%20Are%20Easy%20to%20Unlearn/tables/6b1259a048b05739bc82b5505f90223d964942edf0a49bc0800540bbb6703564.jpg)

<a id="paper-5163"></a>

## Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation


### Images

![044c53b07d2f06fbd4fb23f4de46cba7314c2c8fe9615bbe3cc3004943f94cbc.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/044c53b07d2f06fbd4fb23f4de46cba7314c2c8fe9615bbe3cc3004943f94cbc.jpg)

![295cad4cacdddfa7443a3404ceb743b3504596fe0385b922c987f17b009d1a0e.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/295cad4cacdddfa7443a3404ceb743b3504596fe0385b922c987f17b009d1a0e.jpg)

![308088679d7137888d5ad4ba106d90f45dfe9dc7f4b41b510dcb26e4e79b7005.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/308088679d7137888d5ad4ba106d90f45dfe9dc7f4b41b510dcb26e4e79b7005.jpg)

![314d91ab0041a84d186379da55dbdf4ef6e14d015e05b2306a0fee98e31adcb6.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/314d91ab0041a84d186379da55dbdf4ef6e14d015e05b2306a0fee98e31adcb6.jpg)

![339f23f40528bb4aae94e23ecd47273acc838ccd7d83558459f600bed453d1c6.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/339f23f40528bb4aae94e23ecd47273acc838ccd7d83558459f600bed453d1c6.jpg)

![3afc1541f2ea70126b884a5e887f6381756ada7297ab2e0143ab0cc1f356b43c.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/3afc1541f2ea70126b884a5e887f6381756ada7297ab2e0143ab0cc1f356b43c.jpg)

![3c9e2f788845f9b9888449d6fbb46b228d10d49eac905bc54b1aee70db70fa24.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/3c9e2f788845f9b9888449d6fbb46b228d10d49eac905bc54b1aee70db70fa24.jpg)

![52020137a5a9f2acf633de76e8c4b8d6b7146ce2185d2050e08223faae7bf50d.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/52020137a5a9f2acf633de76e8c4b8d6b7146ce2185d2050e08223faae7bf50d.jpg)

![54cf17efb61b84572d0174307a88cc7f5dfe53281a8e77e7213ea26cb8f14377.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/54cf17efb61b84572d0174307a88cc7f5dfe53281a8e77e7213ea26cb8f14377.jpg)

![7c5f2bc9fd3ff3037013bae148d9e668cdd888133b31557376425fa99c9cac81.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/7c5f2bc9fd3ff3037013bae148d9e668cdd888133b31557376425fa99c9cac81.jpg)

![8668a820276b09562a5c0965564f7eb0cab7a0406dc35ae8137854edbe677ec3.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/8668a820276b09562a5c0965564f7eb0cab7a0406dc35ae8137854edbe677ec3.jpg)

![876123159024972a1f0d0ca261224f4a0c3c29f971e42fdb9e551ae02a8190ca.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/876123159024972a1f0d0ca261224f4a0c3c29f971e42fdb9e551ae02a8190ca.jpg)

![88447649b44128868706f6626ec2bf6076a40079714f066386a137f558b089ec.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/88447649b44128868706f6626ec2bf6076a40079714f066386a137f558b089ec.jpg)

![9d77a6e50af370a48307a1d63ee4e6d5582ccb58fabaf9bc8bf78fbf1ad3abff.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/9d77a6e50af370a48307a1d63ee4e6d5582ccb58fabaf9bc8bf78fbf1ad3abff.jpg)

![a2efa14f372949bd57ecba6a84003dd38445304f3d8482250a31f926ea3e6d6c.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/a2efa14f372949bd57ecba6a84003dd38445304f3d8482250a31f926ea3e6d6c.jpg)

![aef3d42ec59ad51e50bc018f5865ae715e1a3c26e4ba19b5656753f3854ca220.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/aef3d42ec59ad51e50bc018f5865ae715e1a3c26e4ba19b5656753f3854ca220.jpg)

![e9f0e7dd4869a02fcb75abaf5aa374d6a0f895cc1be5ec157aef9303aad7f978.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/e9f0e7dd4869a02fcb75abaf5aa374d6a0f895cc1be5ec157aef9303aad7f978.jpg)

![ee5035edf3b611a5951e676276e2519f337a3873c221a32f092fde0628bfa649.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/images/ee5035edf3b611a5951e676276e2519f337a3873c221a32f092fde0628bfa649.jpg)

### Tables

![140e56e34b13e83b4afe11d625d2dcb44030ab0db67b66a23001020ca5d55a06.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/tables/140e56e34b13e83b4afe11d625d2dcb44030ab0db67b66a23001020ca5d55a06.jpg)

![1819101ad27a40ea408aff92cda4e0c73c89e265644fde11f1ff9f8f5082b261.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/tables/1819101ad27a40ea408aff92cda4e0c73c89e265644fde11f1ff9f8f5082b261.jpg)

![1af769063c522c7d80064e148d993931f589a243f5ae2b5d36dd8d5a0d651d98.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/tables/1af769063c522c7d80064e148d993931f589a243f5ae2b5d36dd8d5a0d651d98.jpg)

![2478ef55d7bcc2a173d192be3b7e78434665ed9b064d05e7cf0ea668a72b2efa.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/tables/2478ef55d7bcc2a173d192be3b7e78434665ed9b064d05e7cf0ea668a72b2efa.jpg)

![518f379a4b49763c90af3129d333af38d74a9405b404480cd823c23070bbb40d.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/tables/518f379a4b49763c90af3129d333af38d74a9405b404480cd823c23070bbb40d.jpg)

![7a315023d8a2e4629eaaae3c5bea46bd0a8768571449dd4bd7027be23d9388d9.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/tables/7a315023d8a2e4629eaaae3c5bea46bd0a8768571449dd4bd7027be23d9388d9.jpg)

![90592f41feae6877d3b244d0f77bb4e9270799b3ef2b233f2e9aa6441793a280.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/tables/90592f41feae6877d3b244d0f77bb4e9270799b3ef2b233f2e9aa6441793a280.jpg)

![adc05ec2a64a8026fd0aa36adaae9404bcc4c293450403fd71fada89213ba151.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/tables/adc05ec2a64a8026fd0aa36adaae9404bcc4c293450403fd71fada89213ba151.jpg)

![b47f621a90f5a222beba9fb7463812c1fc986967e0418f1a1442b86db560ff96.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/tables/b47f621a90f5a222beba9fb7463812c1fc986967e0418f1a1442b86db560ff96.jpg)

![c8ad201832ef082b25bb34d4929e3fb3606f601af8c2a028e39c362e9222be70.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/tables/c8ad201832ef082b25bb34d4929e3fb3606f601af8c2a028e39c362e9222be70.jpg)

![e6b32aad604dbb62cf819f46f81057c428cf77d371f8aaeabded43b35643c502.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/tables/e6b32aad604dbb62cf819f46f81057c428cf77d371f8aaeabded43b35643c502.jpg)

![f3497c9f4dfde5ea10c704aaab42f3bbd4406ec76059a60d435effacfea3eb0d.jpg](../icml_results/5211_Dive%20into%20the%20Scene_%20Breaking%20the%20Perceptual%20Bottleneck%20in%20Vision-Language%20Decision%20Making%20via%20Focus/tables/f3497c9f4dfde5ea10c704aaab42f3bbd4406ec76059a60d435effacfea3eb0d.jpg)

<a id="paper-5164"></a>

## PixCLIP: Towards Fine-grained Vision-Language Understanding via Any-granularity Pixel-Text Alignment


### Images

![09463f036bd947121f63d9fd0e4c8137b7eb41a590a20d20ae4a28cdd3018e46.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/09463f036bd947121f63d9fd0e4c8137b7eb41a590a20d20ae4a28cdd3018e46.jpg)

![0aabd27ff20395ec7cf140c50d9a19998e1735e970420fc402df66cf2ff151da.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/0aabd27ff20395ec7cf140c50d9a19998e1735e970420fc402df66cf2ff151da.jpg)

![0ae2fb2c8b004a51198ca99bb71777e9a2975aed1cbc32f5887db81249927a5b.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/0ae2fb2c8b004a51198ca99bb71777e9a2975aed1cbc32f5887db81249927a5b.jpg)

![15830b76845b96e3b7dbd6660b6dd47f0d239cbfc18ba22fd213f05c1447a39a.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/15830b76845b96e3b7dbd6660b6dd47f0d239cbfc18ba22fd213f05c1447a39a.jpg)

![1f40fb2f2792a604e93c63cbba79ae7e51a6cfeada7bf93f1bd918aa30dd7dce.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/1f40fb2f2792a604e93c63cbba79ae7e51a6cfeada7bf93f1bd918aa30dd7dce.jpg)

![375bddfb4463ac9fb68f720d7101a2df426d49429551eab9400d8c08824804e9.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/375bddfb4463ac9fb68f720d7101a2df426d49429551eab9400d8c08824804e9.jpg)

![4736f67a3527e2d7d00830ce1da90aa200ba9cccd01dc0d681b23127991a24c8.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/4736f67a3527e2d7d00830ce1da90aa200ba9cccd01dc0d681b23127991a24c8.jpg)

![5c1bfa4a55b30de1ad58a129ab92a9fed3b5a6060e7078e97bbfe1497b506448.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/5c1bfa4a55b30de1ad58a129ab92a9fed3b5a6060e7078e97bbfe1497b506448.jpg)

![6322c14bc00b3f46aebafbc53beeaa41fff84fe93be1732f540041fcc70fb0dc.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/6322c14bc00b3f46aebafbc53beeaa41fff84fe93be1732f540041fcc70fb0dc.jpg)

![663a3a63a359b84ea25d191b834bd17c4cc00b074a58b6647aecc4b55e0c602d.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/663a3a63a359b84ea25d191b834bd17c4cc00b074a58b6647aecc4b55e0c602d.jpg)

![699b61f27ce1b3137b64c492e12a00e070754f5e148f0bbf0fe4f8c083218b78.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/699b61f27ce1b3137b64c492e12a00e070754f5e148f0bbf0fe4f8c083218b78.jpg)

![69dff42768cabe6c898cbbdc272014eade41c36b85c41f946fd8733d6045cdc7.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/69dff42768cabe6c898cbbdc272014eade41c36b85c41f946fd8733d6045cdc7.jpg)

![6f6a415ca90103917d5fe2e64f70f30fe75ab1d5e23d5b5f97d5c3b7f19b603e.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/6f6a415ca90103917d5fe2e64f70f30fe75ab1d5e23d5b5f97d5c3b7f19b603e.jpg)

![77e4217aba512cd259815a22badc34efb90ad7f168cb17ef6349ccc5c42e3043.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/77e4217aba512cd259815a22badc34efb90ad7f168cb17ef6349ccc5c42e3043.jpg)

![8a2c6131cb0a9d2239bbcdbc2f6d47f28e4b5ebd796df122c9a85dbf4d4ef0e5.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/8a2c6131cb0a9d2239bbcdbc2f6d47f28e4b5ebd796df122c9a85dbf4d4ef0e5.jpg)

![c5fd3116a12d706e5b8e47b45470bb65d028a737b80d0263d87836df5af6fe03.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/c5fd3116a12d706e5b8e47b45470bb65d028a737b80d0263d87836df5af6fe03.jpg)

![d5e71d38c0d0e143cc99f30626c220b9431269c516765aa976e281276bf71c6c.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/d5e71d38c0d0e143cc99f30626c220b9431269c516765aa976e281276bf71c6c.jpg)

![d9390d0fd76c0b18f607b0fe8e0833e8f531032c5d9484bc67ae57a465b84b98.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/d9390d0fd76c0b18f607b0fe8e0833e8f531032c5d9484bc67ae57a465b84b98.jpg)

![d985105cd80de6987871dfab2d70f2fc7bbefacda2c6e3d23955bca9d624fd2d.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/d985105cd80de6987871dfab2d70f2fc7bbefacda2c6e3d23955bca9d624fd2d.jpg)

![e17f34831e17f9a806bccd06b399d85c391d70d762c5e626f124bd9556b4b298.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/e17f34831e17f9a806bccd06b399d85c391d70d762c5e626f124bd9556b4b298.jpg)

![e2a40ac6a9bf7842a4c60030d605112171ce2a37b084a6a5e3ead05aa0f0f446.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/e2a40ac6a9bf7842a4c60030d605112171ce2a37b084a6a5e3ead05aa0f0f446.jpg)

![e99c60d9626bc6e8379d8c4e2ff0f4a83dafcf0e7af0727366dc93cfdb86751d.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/e99c60d9626bc6e8379d8c4e2ff0f4a83dafcf0e7af0727366dc93cfdb86751d.jpg)

![f54d0fc79ee30fd5be97b6ecef93417b1c6a0ce40583840b1926f6679d2a0805.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/images/f54d0fc79ee30fd5be97b6ecef93417b1c6a0ce40583840b1926f6679d2a0805.jpg)

### Tables

![16eb7c592e77b39f6f75251cb98f2aaf26ff8d6d2a91fdf75c5f79258dd0eb51.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/tables/16eb7c592e77b39f6f75251cb98f2aaf26ff8d6d2a91fdf75c5f79258dd0eb51.jpg)

![2578f478ade901dd972a324d810f29eee7506f4ebff494e9e10fa77e9b17cbf0.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/tables/2578f478ade901dd972a324d810f29eee7506f4ebff494e9e10fa77e9b17cbf0.jpg)

![400493fdc5aa3d3268a028f6a013d466dedd6c66163a10af459fa2087db2dc46.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/tables/400493fdc5aa3d3268a028f6a013d466dedd6c66163a10af459fa2087db2dc46.jpg)

![42799a26f593cd4a10e3ff49bc7aabd09a5a9235a7b15e09686fd45bc571b570.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/tables/42799a26f593cd4a10e3ff49bc7aabd09a5a9235a7b15e09686fd45bc571b570.jpg)

![4797bd059231e5493e8929d83a0d94f293f0ff5d600d805df7b9bf34a178f4ab.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/tables/4797bd059231e5493e8929d83a0d94f293f0ff5d600d805df7b9bf34a178f4ab.jpg)

![6b965ec3f4110620c65c9aa0e23e4ab44a6d3847920350038d15c9c82243233a.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/tables/6b965ec3f4110620c65c9aa0e23e4ab44a6d3847920350038d15c9c82243233a.jpg)

![7f2cec2a758a661b444f1bfbfdd63397ccfa49bf758c3233205f38c5cae35d94.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/tables/7f2cec2a758a661b444f1bfbfdd63397ccfa49bf758c3233205f38c5cae35d94.jpg)

![81c542fcc57ab644873a19b2c1f3d4fa7a2ad88356f267910f697dbd94c35ba1.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/tables/81c542fcc57ab644873a19b2c1f3d4fa7a2ad88356f267910f697dbd94c35ba1.jpg)

![86200fca3fcec4bb1a92ba533340a54aa82da79721cf1594f1f5c72819160431.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/tables/86200fca3fcec4bb1a92ba533340a54aa82da79721cf1594f1f5c72819160431.jpg)

![a26722e57d8a8b06cce44257b93dc14332b8fd59d543bdfa05905230c36d419e.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/tables/a26722e57d8a8b06cce44257b93dc14332b8fd59d543bdfa05905230c36d419e.jpg)

![a58c987d5453886ef21e6be48682c76852352a3a266f5d810ba8e54504541a09.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/tables/a58c987d5453886ef21e6be48682c76852352a3a266f5d810ba8e54504541a09.jpg)

![f4d7efc18cdfab3d970965005b8b1f3be7c6e64b739fcc79adddeb0e64edfccd.jpg](../icml_results/5212_PixCLIP_%20Towards%20Fine-grained%20Vision-Language%20Understanding%20via%20Any-granularity%20Pixel-Text%20Alignment/tables/f4d7efc18cdfab3d970965005b8b1f3be7c6e64b739fcc79adddeb0e64edfccd.jpg)

<a id="paper-5165"></a>

## Drop-in Circulant Structural Priors for Transformer Decoding of Cyclic Codes


### Images

![024926ca15a645556b9409b33b7a63c74da8c9fd05e2df5a326d6deba4b15411.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/images/024926ca15a645556b9409b33b7a63c74da8c9fd05e2df5a326d6deba4b15411.jpg)

![10fced7c7076bc8aad20a0e4aad8dc55c2c33003f8836b275ff26099db7af620.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/images/10fced7c7076bc8aad20a0e4aad8dc55c2c33003f8836b275ff26099db7af620.jpg)

![14277f3d91fe6e9887f7c3606ae3929383e4290d01dd11ab8625034bd24a1a04.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/images/14277f3d91fe6e9887f7c3606ae3929383e4290d01dd11ab8625034bd24a1a04.jpg)

![14533a79f7f4937132faaf766b39cd1e8e5b7361c2718249365fbf29aa52dad5.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/images/14533a79f7f4937132faaf766b39cd1e8e5b7361c2718249365fbf29aa52dad5.jpg)

![2c69cb9e2b4b2586453e38ae422a3620f283ba6d93e93d15ff046bc5cc60b7c3.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/images/2c69cb9e2b4b2586453e38ae422a3620f283ba6d93e93d15ff046bc5cc60b7c3.jpg)

![3446cebae2b8423657093fef15fe92de449875af07b848f7fe169f73a124c210.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/images/3446cebae2b8423657093fef15fe92de449875af07b848f7fe169f73a124c210.jpg)

![400121c8178ba7457c5c93ea79a53fbf26850fc62ee7b22dc0c9175d247b93ca.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/images/400121c8178ba7457c5c93ea79a53fbf26850fc62ee7b22dc0c9175d247b93ca.jpg)

![485a68da4ea6a213203cf92486960e1191c8e120649ce9121108de00cbabdeea.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/images/485a68da4ea6a213203cf92486960e1191c8e120649ce9121108de00cbabdeea.jpg)

![6466f5ea1737dd3f2d02c77eb59f3667245a4ef4fabcce071c0858fdb03b98e4.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/images/6466f5ea1737dd3f2d02c77eb59f3667245a4ef4fabcce071c0858fdb03b98e4.jpg)

![75ba801f47e867d53a0bec482f29c9b00a0551f95691aa5ae78d6ffe119c9328.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/images/75ba801f47e867d53a0bec482f29c9b00a0551f95691aa5ae78d6ffe119c9328.jpg)

![7aa28a62e56f1f6a719f3d6fddd22a02e16aef71a34d39361da764e58365946c.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/images/7aa28a62e56f1f6a719f3d6fddd22a02e16aef71a34d39361da764e58365946c.jpg)

![9155bea236e2a22d0932e7f9f03729d813dcfd45f1324bf192af71652c8263e0.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/images/9155bea236e2a22d0932e7f9f03729d813dcfd45f1324bf192af71652c8263e0.jpg)

![a55261903d829750dccdc9ece24cdd7a8eff3490a576a1e43328b73291e21a97.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/images/a55261903d829750dccdc9ece24cdd7a8eff3490a576a1e43328b73291e21a97.jpg)

![e3b2863eab847e63e6b763c9f8fb5d773b6d926859738898eaa7e501e824e942.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/images/e3b2863eab847e63e6b763c9f8fb5d773b6d926859738898eaa7e501e824e942.jpg)

![f8980b8a987789d6069b761a9c04873f7fd9fe5dd141b2569c44beadb915946a.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/images/f8980b8a987789d6069b761a9c04873f7fd9fe5dd141b2569c44beadb915946a.jpg)

![fcc90690a398182a0d55727bd23da410c066ee7db71e35f5d09e15ebd4e3aba0.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/images/fcc90690a398182a0d55727bd23da410c066ee7db71e35f5d09e15ebd4e3aba0.jpg)

### Tables

![0165f8def10d8c4ac437a915b3dba5d19d01b817d153a5f1ad12a66675a0b5fd.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/tables/0165f8def10d8c4ac437a915b3dba5d19d01b817d153a5f1ad12a66675a0b5fd.jpg)

![0a5c2181a979e94dde5f3a0b1ed7ef3843e171a1246fb69b0ed38eecc896dadf.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/tables/0a5c2181a979e94dde5f3a0b1ed7ef3843e171a1246fb69b0ed38eecc896dadf.jpg)

![1b2c218d8299ace71f46a9bb312cd19ad671461fa03d35fc76c0b2fd8d3c3801.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/tables/1b2c218d8299ace71f46a9bb312cd19ad671461fa03d35fc76c0b2fd8d3c3801.jpg)

![51e2be017e7e74ca4da846653538c1619bdccb4236ac6748738472fc9921b551.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/tables/51e2be017e7e74ca4da846653538c1619bdccb4236ac6748738472fc9921b551.jpg)

![5428fbd1ad25a4e30d2747c5ff12f56aa44edff7f1b7615e737951891ccfbf80.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/tables/5428fbd1ad25a4e30d2747c5ff12f56aa44edff7f1b7615e737951891ccfbf80.jpg)

![8ece090083b9acbdc8218cdd961b89c98d919f244440ee009d49348cff885632.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/tables/8ece090083b9acbdc8218cdd961b89c98d919f244440ee009d49348cff885632.jpg)

![9f923e88129de4d4d09307abd7379c08c56e58a0e9531ee053274341789a4c1f.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/tables/9f923e88129de4d4d09307abd7379c08c56e58a0e9531ee053274341789a4c1f.jpg)

![b369cace39b69215ecbcc3ff54a2f81e5a8929bd2269eb9071334204571c2aa5.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/tables/b369cace39b69215ecbcc3ff54a2f81e5a8929bd2269eb9071334204571c2aa5.jpg)

![c5f4340732667b6839742af2213c5d97739ff121674093e0c72829fc6495a185.jpg](../icml_results/5213_Drop-in%20Circulant%20Structural%20Priors%20for%20Transformer%20Decoding%20of%20Cyclic%20Codes/tables/c5f4340732667b6839742af2213c5d97739ff121674093e0c72829fc6495a185.jpg)

<a id="paper-5166"></a>

## Functional Attention: From Pairwise Affinities to Functional Correspondences


### Images

![0d27a20ddc3a38f1e373392e95fc39f3ce3b066d5e2279e611b43ca34f0b8211.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/images/0d27a20ddc3a38f1e373392e95fc39f3ce3b066d5e2279e611b43ca34f0b8211.jpg)

![0fa77e1dd850c0e20ac5bf29d3a171536ce83ea462bd621fdb75989802870c7d.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/images/0fa77e1dd850c0e20ac5bf29d3a171536ce83ea462bd621fdb75989802870c7d.jpg)

![10d91bef14d018ea401407d12c0260c6726ef6485c20362dd76c4f60ad698d24.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/images/10d91bef14d018ea401407d12c0260c6726ef6485c20362dd76c4f60ad698d24.jpg)

![26a6cd8bef6f28f1557c99594edfbdb04b22f27340c58452b124cd1842987718.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/images/26a6cd8bef6f28f1557c99594edfbdb04b22f27340c58452b124cd1842987718.jpg)

![273cb93577a858367dc180cf36111d0017b54ab1c5eddf46b3f0922bafed5a76.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/images/273cb93577a858367dc180cf36111d0017b54ab1c5eddf46b3f0922bafed5a76.jpg)

![377cf66c361858f946435c54a6b4d36042387739a1a02980accf7a691c3bacc5.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/images/377cf66c361858f946435c54a6b4d36042387739a1a02980accf7a691c3bacc5.jpg)

![3e5fbf3904c1f65f17627dc870902b9a2c74e807a157e8d70c88f9222c367581.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/images/3e5fbf3904c1f65f17627dc870902b9a2c74e807a157e8d70c88f9222c367581.jpg)

![584851a0980ff7d10069e9c5de3a3ffd5067c0a9d0835329a632f0102ce8044e.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/images/584851a0980ff7d10069e9c5de3a3ffd5067c0a9d0835329a632f0102ce8044e.jpg)

![5ef4e44290307cf51c8558901ddbbb864907ea48bf21f2fd5ec4d18c92112766.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/images/5ef4e44290307cf51c8558901ddbbb864907ea48bf21f2fd5ec4d18c92112766.jpg)

![89dd8a4305eb0b4798286c1370d4c4da52419a6aaa602895597cc569ef59de8d.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/images/89dd8a4305eb0b4798286c1370d4c4da52419a6aaa602895597cc569ef59de8d.jpg)

![d310ff9053fc40ddec42e59f874f988551c01ce328773e422134f472dbc9bc9a.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/images/d310ff9053fc40ddec42e59f874f988551c01ce328773e422134f472dbc9bc9a.jpg)

![e6ed07dff6901221a60889ffcc6dc5ec6b9c2a8505bc55531d90bca48fe66060.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/images/e6ed07dff6901221a60889ffcc6dc5ec6b9c2a8505bc55531d90bca48fe66060.jpg)

### Tables

![12663883c8d69125496a219d223d4df4f566fd50ad2dd22938d69741b32f5cef.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/tables/12663883c8d69125496a219d223d4df4f566fd50ad2dd22938d69741b32f5cef.jpg)

![275a86655eebd61d7596822c67fb7b4daf6b545e2807be1d0927255a754fc45d.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/tables/275a86655eebd61d7596822c67fb7b4daf6b545e2807be1d0927255a754fc45d.jpg)

![340a918cfa5251bbe7974bdedbf96e8b4f09747f46e937d425c28782ec48e7c2.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/tables/340a918cfa5251bbe7974bdedbf96e8b4f09747f46e937d425c28782ec48e7c2.jpg)

![39387af71ae88f2c222b01249420d6e56e303f41bace709dd0992a969140348e.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/tables/39387af71ae88f2c222b01249420d6e56e303f41bace709dd0992a969140348e.jpg)

![5198476485810d3a21a9b5f5f386dd53c01fba7902e8a3a3bc123d2ae2e3a519.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/tables/5198476485810d3a21a9b5f5f386dd53c01fba7902e8a3a3bc123d2ae2e3a519.jpg)

![60660f77750d033f60c1552918c9ebec50ddb29834f77dc89a361586861db3b8.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/tables/60660f77750d033f60c1552918c9ebec50ddb29834f77dc89a361586861db3b8.jpg)

![6de12d0235676abe260e6cb2abc89225254a9fb4c3a4630f0175af8585f989d9.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/tables/6de12d0235676abe260e6cb2abc89225254a9fb4c3a4630f0175af8585f989d9.jpg)

![815ba3c28a6d532bf52e9845c4a13c0972805903ac6b6f714a01c1a61334e729.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/tables/815ba3c28a6d532bf52e9845c4a13c0972805903ac6b6f714a01c1a61334e729.jpg)

![843bc61de6d03222f824bcc0aa1176048e9deb83db09bf963bd45f02e0bd5e27.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/tables/843bc61de6d03222f824bcc0aa1176048e9deb83db09bf963bd45f02e0bd5e27.jpg)

![b338c6fe6eb53686763e2a15beceab0799de312ae10a0d3971d8998488e0f3e2.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/tables/b338c6fe6eb53686763e2a15beceab0799de312ae10a0d3971d8998488e0f3e2.jpg)

![bdd3812419e8ef25bef7ff0aa31e49fe5607fca65e7f7e7310448928dd569470.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/tables/bdd3812419e8ef25bef7ff0aa31e49fe5607fca65e7f7e7310448928dd569470.jpg)

![cea676eade54bf084aeaab12fb9660ab1dd4524a9ad47bc38e06a7b21a8e9b89.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/tables/cea676eade54bf084aeaab12fb9660ab1dd4524a9ad47bc38e06a7b21a8e9b89.jpg)

![e0c4a784d107b68bc79494ef5e27d56291d19dd26ecba2cd0c0252211de41bac.jpg](../icml_results/5214_Functional%20Attention_%20From%20Pairwise%20Affinities%20to%20Functional%20Correspondences/tables/e0c4a784d107b68bc79494ef5e27d56291d19dd26ecba2cd0c0252211de41bac.jpg)

<a id="paper-5167"></a>

## How Do Language Models Speak Languages? A Case Study on Unintended Code-Switching


### Images

![0ff4bc7d4c590c8c54531df99c070af1de4015e72f1055680a6fe257d9ec2d42.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/images/0ff4bc7d4c590c8c54531df99c070af1de4015e72f1055680a6fe257d9ec2d42.jpg)

![399e57923775b492dd8f2115badcd957a5cd49afd1a9c9d37418fc0d38ddb124.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/images/399e57923775b492dd8f2115badcd957a5cd49afd1a9c9d37418fc0d38ddb124.jpg)

![3c873577626a44dbd4d9b8f4e99b6654437e93675be2ba61855e3d72057fdcfc.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/images/3c873577626a44dbd4d9b8f4e99b6654437e93675be2ba61855e3d72057fdcfc.jpg)

![4855e711aa41d5ff27d98e2263693e9b801723a5c196c19ea461c2b007209b1a.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/images/4855e711aa41d5ff27d98e2263693e9b801723a5c196c19ea461c2b007209b1a.jpg)

![5954daa1abda424770d3a45a17600177f8cbf94e24a2698122f0c5ea518dc547.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/images/5954daa1abda424770d3a45a17600177f8cbf94e24a2698122f0c5ea518dc547.jpg)

![7111505cdbc26d709aaafb86177cd685f8b4cd06e71ec923c1ccbfee380f8db7.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/images/7111505cdbc26d709aaafb86177cd685f8b4cd06e71ec923c1ccbfee380f8db7.jpg)

![936107ac0ee373dd9650d65988278e03721974e2bedd5c4e743553d62734a390.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/images/936107ac0ee373dd9650d65988278e03721974e2bedd5c4e743553d62734a390.jpg)

![992e37a154d552351aff3c145a33fdb46341d859561cc5cbd01a8dd349c75635.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/images/992e37a154d552351aff3c145a33fdb46341d859561cc5cbd01a8dd349c75635.jpg)

![a5fe6cf4065c876d501d5bb28dc9a4a83c8da225a950cee8b305ad2959d07da2.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/images/a5fe6cf4065c876d501d5bb28dc9a4a83c8da225a950cee8b305ad2959d07da2.jpg)

![bb43f1b08a4730c154da7d450a554b94cca9fc6c5ae0ab78b6211066026c1009.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/images/bb43f1b08a4730c154da7d450a554b94cca9fc6c5ae0ab78b6211066026c1009.jpg)

![d2221f026ea3f529beff361d0340fb382c5948954034a276b0ff66f5a5e34f5f.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/images/d2221f026ea3f529beff361d0340fb382c5948954034a276b0ff66f5a5e34f5f.jpg)

![d7d2db66d588368fd21283c9156e640962298f17ccf6550bb98a0942e84f64a8.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/images/d7d2db66d588368fd21283c9156e640962298f17ccf6550bb98a0942e84f64a8.jpg)

![d9a4bb2c9159b6af63714aef635430d20f59300beb11265825b8980026767932.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/images/d9a4bb2c9159b6af63714aef635430d20f59300beb11265825b8980026767932.jpg)

![dda06c12bc6e1148dc3e9248eb7211871f4fd83461243dd03c66d191c84e00ef.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/images/dda06c12bc6e1148dc3e9248eb7211871f4fd83461243dd03c66d191c84e00ef.jpg)

![de8f95780d424eb037dc5990f6b6d8b79ae233b5ca998126794c783e41c1d72e.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/images/de8f95780d424eb037dc5990f6b6d8b79ae233b5ca998126794c783e41c1d72e.jpg)

![e7ebdca01ea6a1c4ecfb2cdfd4635250cdbe1c688dac88c569cd97045797cc58.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/images/e7ebdca01ea6a1c4ecfb2cdfd4635250cdbe1c688dac88c569cd97045797cc58.jpg)

![efa6d56c44882840d70ef72fd4f268fc535b658c6d70219a83e97552e9764379.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/images/efa6d56c44882840d70ef72fd4f268fc535b658c6d70219a83e97552e9764379.jpg)

### Tables

![1749bc2d637b99de0dd0d45c778200a2d3a96ff3c3d7540bd82b07492af53d19.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/1749bc2d637b99de0dd0d45c778200a2d3a96ff3c3d7540bd82b07492af53d19.jpg)

![1efe6db76d29500f3517f9b059f8a1fe967d9eaf85d9a906bd7f0e520bfffdf0.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/1efe6db76d29500f3517f9b059f8a1fe967d9eaf85d9a906bd7f0e520bfffdf0.jpg)

![2179589325412784792400df725048de952a21d7d17951e92453df858128e9eb.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/2179589325412784792400df725048de952a21d7d17951e92453df858128e9eb.jpg)

![2a7836921bed04425674fd999fab04d00fe6e396d860051aeff6fe13f5087280.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/2a7836921bed04425674fd999fab04d00fe6e396d860051aeff6fe13f5087280.jpg)

![316e7c67a9f8e653a88064e9eedf19c1745dbd037ee4ffcea6030c9c8efd2735.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/316e7c67a9f8e653a88064e9eedf19c1745dbd037ee4ffcea6030c9c8efd2735.jpg)

![4aacbf88c79438ff525cf0e6f82f6e2b939e5f35335436e36272b474090defe4.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/4aacbf88c79438ff525cf0e6f82f6e2b939e5f35335436e36272b474090defe4.jpg)

![4fcc98c7011bea2ec6371a4668b0e40fe1f0f399343927216882a5e21477e181.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/4fcc98c7011bea2ec6371a4668b0e40fe1f0f399343927216882a5e21477e181.jpg)

![522db5ece3dcb8c9447f141aa2e27da476650824d67dedf8992cc9cb7a983f0d.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/522db5ece3dcb8c9447f141aa2e27da476650824d67dedf8992cc9cb7a983f0d.jpg)

![52b67cc9672f2a5c05ace9fa5612b9bb6a95720da2291605d086220751e36380.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/52b67cc9672f2a5c05ace9fa5612b9bb6a95720da2291605d086220751e36380.jpg)

![62ab752279b69a1efdb5edc1a385dc13d493a6409d84b22715df0be8a28d4994.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/62ab752279b69a1efdb5edc1a385dc13d493a6409d84b22715df0be8a28d4994.jpg)

![7178db7ff759f6cd30dfa8b613a42317ee5128e6d90c6af709c699e5aae83b20.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/7178db7ff759f6cd30dfa8b613a42317ee5128e6d90c6af709c699e5aae83b20.jpg)

![7ccb754bfb4ebfd0fe3af316e3dc92400fce2463cbc14e3a41834e674352e460.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/7ccb754bfb4ebfd0fe3af316e3dc92400fce2463cbc14e3a41834e674352e460.jpg)

![817e755ca006ce784d32b5faeaab6e5b342de4d74dbee280ca6a32be88a4d3fc.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/817e755ca006ce784d32b5faeaab6e5b342de4d74dbee280ca6a32be88a4d3fc.jpg)

![985275cd633963ae96190e65cb04d2c750796484afad213d4c04ffc0bde70b38.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/985275cd633963ae96190e65cb04d2c750796484afad213d4c04ffc0bde70b38.jpg)

![9cc271244b78da540a71c61c9fe1d68b8f2ca1a87a484ccfc766cedbc7c60df4.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/9cc271244b78da540a71c61c9fe1d68b8f2ca1a87a484ccfc766cedbc7c60df4.jpg)

![b3cf1046a9f60bbb1cebe241e969f333e37539b69bb0b2b73ebc513568a46220.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/b3cf1046a9f60bbb1cebe241e969f333e37539b69bb0b2b73ebc513568a46220.jpg)

![c3ef0e2f9e50aeec02dbb160cefbf339ecf7664bf1ff6a7ca5fb3bbb50dc8eb6.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/c3ef0e2f9e50aeec02dbb160cefbf339ecf7664bf1ff6a7ca5fb3bbb50dc8eb6.jpg)

![c6417d00c30d2e780f62020762980f9dc8d1a8636837e107d1b9ea1be85e0ee5.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/c6417d00c30d2e780f62020762980f9dc8d1a8636837e107d1b9ea1be85e0ee5.jpg)

![cdc59034c6602d9b43bb6540c3d3f994acb187685b06015ff1b93cb0a3025da5.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/cdc59034c6602d9b43bb6540c3d3f994acb187685b06015ff1b93cb0a3025da5.jpg)

![d5ffdef45dabbabdb9aa410fb7856bce5919ba0e9aa05f7a0767f8f7d6d4b902.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/d5ffdef45dabbabdb9aa410fb7856bce5919ba0e9aa05f7a0767f8f7d6d4b902.jpg)

![dfc969452b67404e98b99a33cfee3e0d128617d5cd7127115067ec249e57d5b7.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/dfc969452b67404e98b99a33cfee3e0d128617d5cd7127115067ec249e57d5b7.jpg)

![e2565235a10f5f3ee4241f7b6234bd6a61bffeea0df2a17eaf7dbb0681b1315a.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/e2565235a10f5f3ee4241f7b6234bd6a61bffeea0df2a17eaf7dbb0681b1315a.jpg)

![e3f557a2b72cfefe0733f979ceb0992164060f53d776fef428ccf7c8f97f72c0.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/e3f557a2b72cfefe0733f979ceb0992164060f53d776fef428ccf7c8f97f72c0.jpg)

![e5a37f90696236d70a256077816d8108eb3d3857f4920f3799038280cace8c57.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/e5a37f90696236d70a256077816d8108eb3d3857f4920f3799038280cace8c57.jpg)

![f8b5971555b236a3a600ee03bbd2b616d483849389589fd5b8c5b8ff97c69b08.jpg](../icml_results/5215_How%20Do%20Language%20Models%20Speak%20Languages_%20A%20Case%20Study%20on%20Unintended%20Code-Switching/tables/f8b5971555b236a3a600ee03bbd2b616d483849389589fd5b8c5b8ff97c69b08.jpg)

<a id="paper-5168"></a>

## AutoVSR: Automatic Visual-to-Symbolic Reasoning for Symbolic Expression Generation from Circuit Schematic


### Images

![1ee3f6ac874676ec88e00c62a403ec34120ec6f035f766af5f4451c37ca87178.jpg](../icml_results/5216_AutoVSR_%20Automatic%20Visual-to-Symbolic%20Reasoning%20for%20Symbolic%20Expression%20Generation%20from%20Circuit%20Sche/images/1ee3f6ac874676ec88e00c62a403ec34120ec6f035f766af5f4451c37ca87178.jpg)

![651b709ae63c879bdaf8066f2dcfe6b976530118a226510be7f82c08ae12a0d9.jpg](../icml_results/5216_AutoVSR_%20Automatic%20Visual-to-Symbolic%20Reasoning%20for%20Symbolic%20Expression%20Generation%20from%20Circuit%20Sche/images/651b709ae63c879bdaf8066f2dcfe6b976530118a226510be7f82c08ae12a0d9.jpg)

![936db59872619409bd864f4ae5bf15729b865fdb170a273f841be22dba600a81.jpg](../icml_results/5216_AutoVSR_%20Automatic%20Visual-to-Symbolic%20Reasoning%20for%20Symbolic%20Expression%20Generation%20from%20Circuit%20Sche/images/936db59872619409bd864f4ae5bf15729b865fdb170a273f841be22dba600a81.jpg)

### Tables

![387c0f9308011b6305262157cba98ef418172edef84c129a870b3c90cff7e12c.jpg](../icml_results/5216_AutoVSR_%20Automatic%20Visual-to-Symbolic%20Reasoning%20for%20Symbolic%20Expression%20Generation%20from%20Circuit%20Sche/tables/387c0f9308011b6305262157cba98ef418172edef84c129a870b3c90cff7e12c.jpg)

![42f37bd98f6d71ed9d3edf7dbec499efdcfda2b9534053f325cbaf000be86b3f.jpg](../icml_results/5216_AutoVSR_%20Automatic%20Visual-to-Symbolic%20Reasoning%20for%20Symbolic%20Expression%20Generation%20from%20Circuit%20Sche/tables/42f37bd98f6d71ed9d3edf7dbec499efdcfda2b9534053f325cbaf000be86b3f.jpg)

![5198e1cfa3de53219901cce74a166eb35546d1b4b62e9674e94df784abf8814b.jpg](../icml_results/5216_AutoVSR_%20Automatic%20Visual-to-Symbolic%20Reasoning%20for%20Symbolic%20Expression%20Generation%20from%20Circuit%20Sche/tables/5198e1cfa3de53219901cce74a166eb35546d1b4b62e9674e94df784abf8814b.jpg)

![67c7c10ddf2519d37c9fb471fb80f32fdf3efd8c888695adc5b6a4f8bb2631ff.jpg](../icml_results/5216_AutoVSR_%20Automatic%20Visual-to-Symbolic%20Reasoning%20for%20Symbolic%20Expression%20Generation%20from%20Circuit%20Sche/tables/67c7c10ddf2519d37c9fb471fb80f32fdf3efd8c888695adc5b6a4f8bb2631ff.jpg)

![8e12e9d303964a34e5398c9d369f39cbf99b6a120d867e1801e2b599f5afff3c.jpg](../icml_results/5216_AutoVSR_%20Automatic%20Visual-to-Symbolic%20Reasoning%20for%20Symbolic%20Expression%20Generation%20from%20Circuit%20Sche/tables/8e12e9d303964a34e5398c9d369f39cbf99b6a120d867e1801e2b599f5afff3c.jpg)

![9f2d5b1231ced785e1052bbec224a4425a92568ff15e5f0d6214594286c2eef8.jpg](../icml_results/5216_AutoVSR_%20Automatic%20Visual-to-Symbolic%20Reasoning%20for%20Symbolic%20Expression%20Generation%20from%20Circuit%20Sche/tables/9f2d5b1231ced785e1052bbec224a4425a92568ff15e5f0d6214594286c2eef8.jpg)

![b531a1b9ec5a6a39a6154fabb1c5d197e9fae6f6f63eaa1c52dfda2e71968f56.jpg](../icml_results/5216_AutoVSR_%20Automatic%20Visual-to-Symbolic%20Reasoning%20for%20Symbolic%20Expression%20Generation%20from%20Circuit%20Sche/tables/b531a1b9ec5a6a39a6154fabb1c5d197e9fae6f6f63eaa1c52dfda2e71968f56.jpg)

![cf5216e74d9048810954f7dca00b69791b394e805dee25b1691bac17cbb9d683.jpg](../icml_results/5216_AutoVSR_%20Automatic%20Visual-to-Symbolic%20Reasoning%20for%20Symbolic%20Expression%20Generation%20from%20Circuit%20Sche/tables/cf5216e74d9048810954f7dca00b69791b394e805dee25b1691bac17cbb9d683.jpg)

![e462df2f1602e36a6f32628156d577f04c9eadacca39561a67cf1b89b6fcab8e.jpg](../icml_results/5216_AutoVSR_%20Automatic%20Visual-to-Symbolic%20Reasoning%20for%20Symbolic%20Expression%20Generation%20from%20Circuit%20Sche/tables/e462df2f1602e36a6f32628156d577f04c9eadacca39561a67cf1b89b6fcab8e.jpg)

![e8fa6af3007cfae52e77ce6bfce03395e41a0d95b17e5724a9212c77aed6127a.jpg](../icml_results/5216_AutoVSR_%20Automatic%20Visual-to-Symbolic%20Reasoning%20for%20Symbolic%20Expression%20Generation%20from%20Circuit%20Sche/tables/e8fa6af3007cfae52e77ce6bfce03395e41a0d95b17e5724a9212c77aed6127a.jpg)

<a id="paper-5169"></a>

## Dynamic Symmetric Point Tracking: Tackling Non-ideal Reference in Analog In-memory Training


### Images

![17a7d73cdcdb4a8a7a5208d11e45823d332e09b58781aaa75065b06f97a124f5.jpg](../icml_results/5217_Dynamic%20Symmetric%20Point%20Tracking_%20Tackling%20Non-ideal%20Reference%20in%20Analog%20In-memory%20Training/images/17a7d73cdcdb4a8a7a5208d11e45823d332e09b58781aaa75065b06f97a124f5.jpg)

![2397138b815c960748cfd4523356766476e3f756c8bd9ef37c286c4fe83611e2.jpg](../icml_results/5217_Dynamic%20Symmetric%20Point%20Tracking_%20Tackling%20Non-ideal%20Reference%20in%20Analog%20In-memory%20Training/images/2397138b815c960748cfd4523356766476e3f756c8bd9ef37c286c4fe83611e2.jpg)

![310076989d15db0b46ad2a29868200a6a9a9be094663f273e0c17390ae4b3d05.jpg](../icml_results/5217_Dynamic%20Symmetric%20Point%20Tracking_%20Tackling%20Non-ideal%20Reference%20in%20Analog%20In-memory%20Training/images/310076989d15db0b46ad2a29868200a6a9a9be094663f273e0c17390ae4b3d05.jpg)

![a7e552583c7565a3bf413b7def9c4a50f57f4a444b1467b6a10bc855d94da4ee.jpg](../icml_results/5217_Dynamic%20Symmetric%20Point%20Tracking_%20Tackling%20Non-ideal%20Reference%20in%20Analog%20In-memory%20Training/images/a7e552583c7565a3bf413b7def9c4a50f57f4a444b1467b6a10bc855d94da4ee.jpg)

![aca2f424915d0e9868420b0e0b7f73e394082ea0f0e0c46432eb1d2341fd33ad.jpg](../icml_results/5217_Dynamic%20Symmetric%20Point%20Tracking_%20Tackling%20Non-ideal%20Reference%20in%20Analog%20In-memory%20Training/images/aca2f424915d0e9868420b0e0b7f73e394082ea0f0e0c46432eb1d2341fd33ad.jpg)

### Tables

![0875129b31e6c7d4e22487f84360a7317e1f5384dd8dd73abb2c7aa82e2a6244.jpg](../icml_results/5217_Dynamic%20Symmetric%20Point%20Tracking_%20Tackling%20Non-ideal%20Reference%20in%20Analog%20In-memory%20Training/tables/0875129b31e6c7d4e22487f84360a7317e1f5384dd8dd73abb2c7aa82e2a6244.jpg)

![1424f1bc59999a1976f5e87e20274314cd21379ba49f1128aeb8b8cecb96af23.jpg](../icml_results/5217_Dynamic%20Symmetric%20Point%20Tracking_%20Tackling%20Non-ideal%20Reference%20in%20Analog%20In-memory%20Training/tables/1424f1bc59999a1976f5e87e20274314cd21379ba49f1128aeb8b8cecb96af23.jpg)

![23fa8b6a1a1301f111ecc783a6d5c4f6664f48b444bc5d6777f6701deaaf139d.jpg](../icml_results/5217_Dynamic%20Symmetric%20Point%20Tracking_%20Tackling%20Non-ideal%20Reference%20in%20Analog%20In-memory%20Training/tables/23fa8b6a1a1301f111ecc783a6d5c4f6664f48b444bc5d6777f6701deaaf139d.jpg)

![2796e120e8548120c577244c9934e2ed33fac7315da67f32ce07507562076cbe.jpg](../icml_results/5217_Dynamic%20Symmetric%20Point%20Tracking_%20Tackling%20Non-ideal%20Reference%20in%20Analog%20In-memory%20Training/tables/2796e120e8548120c577244c9934e2ed33fac7315da67f32ce07507562076cbe.jpg)

![369d28fcfddcca29299189507439945820f39ab52106c1e2be22d71bd129f6aa.jpg](../icml_results/5217_Dynamic%20Symmetric%20Point%20Tracking_%20Tackling%20Non-ideal%20Reference%20in%20Analog%20In-memory%20Training/tables/369d28fcfddcca29299189507439945820f39ab52106c1e2be22d71bd129f6aa.jpg)

![39dbf213140264e56e01e751a92d6b39201edc3e691e4a53b9387b58fd6201b4.jpg](../icml_results/5217_Dynamic%20Symmetric%20Point%20Tracking_%20Tackling%20Non-ideal%20Reference%20in%20Analog%20In-memory%20Training/tables/39dbf213140264e56e01e751a92d6b39201edc3e691e4a53b9387b58fd6201b4.jpg)

![42b3b753590eab1852eb9a7eca60d3a57c336638fbca86d92ee914e7fbfd5c0f.jpg](../icml_results/5217_Dynamic%20Symmetric%20Point%20Tracking_%20Tackling%20Non-ideal%20Reference%20in%20Analog%20In-memory%20Training/tables/42b3b753590eab1852eb9a7eca60d3a57c336638fbca86d92ee914e7fbfd5c0f.jpg)

![55f65a2e55dcfe5474dfa23d9738c30bc3b0638bc5a100f591533511f41fe286.jpg](../icml_results/5217_Dynamic%20Symmetric%20Point%20Tracking_%20Tackling%20Non-ideal%20Reference%20in%20Analog%20In-memory%20Training/tables/55f65a2e55dcfe5474dfa23d9738c30bc3b0638bc5a100f591533511f41fe286.jpg)

![d1968c7102706074b8bd500a875bcc15410e49bf64efc3d1df4310ac2f92d298.jpg](../icml_results/5217_Dynamic%20Symmetric%20Point%20Tracking_%20Tackling%20Non-ideal%20Reference%20in%20Analog%20In-memory%20Training/tables/d1968c7102706074b8bd500a875bcc15410e49bf64efc3d1df4310ac2f92d298.jpg)

![e08a270c41477c5eb2a3619a6f57c1c24cd8b3b47c878ef475bbb663641e3a0a.jpg](../icml_results/5217_Dynamic%20Symmetric%20Point%20Tracking_%20Tackling%20Non-ideal%20Reference%20in%20Analog%20In-memory%20Training/tables/e08a270c41477c5eb2a3619a6f57c1c24cd8b3b47c878ef475bbb663641e3a0a.jpg)

![eb5b5217df58b5589adabf3405e0b00211c182654f3e372954720fcc0d3c8969.jpg](../icml_results/5217_Dynamic%20Symmetric%20Point%20Tracking_%20Tackling%20Non-ideal%20Reference%20in%20Analog%20In-memory%20Training/tables/eb5b5217df58b5589adabf3405e0b00211c182654f3e372954720fcc0d3c8969.jpg)

<a id="paper-5170"></a>

## Capacity-Agnostic Parameter Isolation for Continual Graph Learning


### Images

![02cf424d808da466a0a3be809cd6d06cf4ce1af0868116f4a369455ed6d6efef.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/02cf424d808da466a0a3be809cd6d06cf4ce1af0868116f4a369455ed6d6efef.jpg)

![039908d003fee8fc1e1775c84d6f6568d0f3941e59ee25e112b912fd296f33cb.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/039908d003fee8fc1e1775c84d6f6568d0f3941e59ee25e112b912fd296f33cb.jpg)

![086e2303bdbc924a1c0f9e196abccffae9ef76330919d891ee3e5bd31f9b78cd.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/086e2303bdbc924a1c0f9e196abccffae9ef76330919d891ee3e5bd31f9b78cd.jpg)

![0d02f4bda7a929ab9188bd07448819a3411761a14bf6edf415460ed0e6a188a4.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/0d02f4bda7a929ab9188bd07448819a3411761a14bf6edf415460ed0e6a188a4.jpg)

![1e3d9a9c168ae117be40bd1212080810090405245822654deb7e5678988b5d94.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/1e3d9a9c168ae117be40bd1212080810090405245822654deb7e5678988b5d94.jpg)

![209c88bf62b55fe007fbd50ecad259f863b9d11abc11bf8a44d3fbc416a5c471.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/209c88bf62b55fe007fbd50ecad259f863b9d11abc11bf8a44d3fbc416a5c471.jpg)

![29f225f6f43c04bc9daabfa758bf2b21195e97ac8a5764df1dc58a5490a3155e.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/29f225f6f43c04bc9daabfa758bf2b21195e97ac8a5764df1dc58a5490a3155e.jpg)

![4a2fbf19cf7c47c3ffcd4aaa226dd4cdfb18188183d032d5e0921acae014eafa.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/4a2fbf19cf7c47c3ffcd4aaa226dd4cdfb18188183d032d5e0921acae014eafa.jpg)

![4ce72ca1831ae48a3760332d7be8012ff14dafbd651f441d6e51e3219d194265.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/4ce72ca1831ae48a3760332d7be8012ff14dafbd651f441d6e51e3219d194265.jpg)

![639292596e758601bb933980929a98ec006fe64f503cfcd533fdb29e4e81a572.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/639292596e758601bb933980929a98ec006fe64f503cfcd533fdb29e4e81a572.jpg)

![6a8d7bf8ba5fa93a01d215c568c6ce9349f7260764bbba722a817826f5edb6c0.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/6a8d7bf8ba5fa93a01d215c568c6ce9349f7260764bbba722a817826f5edb6c0.jpg)

![72a449d0712f57409baf608244cbd3c022b33d19957987aee19061a9aecc4567.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/72a449d0712f57409baf608244cbd3c022b33d19957987aee19061a9aecc4567.jpg)

![7c2b7e21b9afd989ba00bcc69fd10ff0957fc82f9e97978df1dcd9079c5f8c94.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/7c2b7e21b9afd989ba00bcc69fd10ff0957fc82f9e97978df1dcd9079c5f8c94.jpg)

![7efd099516caf64607c056dfd60509e1af5e175e01ea3a6458486139920b753a.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/7efd099516caf64607c056dfd60509e1af5e175e01ea3a6458486139920b753a.jpg)

![9ba8de02dae3335178332d20e7778cf11d1a4a4b8234b0ec4370bd9a20aa66b7.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/9ba8de02dae3335178332d20e7778cf11d1a4a4b8234b0ec4370bd9a20aa66b7.jpg)

![a161972b09caed9b9edc650bd6ac592ab370d0fce24e6821e292955455fffe38.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/a161972b09caed9b9edc650bd6ac592ab370d0fce24e6821e292955455fffe38.jpg)

![a85613be82cf3eb680cae1d0d39b37708fe8f970778f2235d5343c3908125fca.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/a85613be82cf3eb680cae1d0d39b37708fe8f970778f2235d5343c3908125fca.jpg)

![afbff0e4748358c1c59a4da4c287d0696d24f62a7bb5634fce00aee9db99dbb3.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/afbff0e4748358c1c59a4da4c287d0696d24f62a7bb5634fce00aee9db99dbb3.jpg)

![b14bdc65541ddc5661e49b9f62e36a1a0fa61967e16fc5dc8fd74b3ed21aa2c7.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/b14bdc65541ddc5661e49b9f62e36a1a0fa61967e16fc5dc8fd74b3ed21aa2c7.jpg)

![e1bcb919a4b651d83cc1c190432f21752f9d3619c67062011473c4dd2d313aa5.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/e1bcb919a4b651d83cc1c190432f21752f9d3619c67062011473c4dd2d313aa5.jpg)

![e42b0c832dc214d975964f4a61623c586f82065e8773821621f89141793f5bb7.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/e42b0c832dc214d975964f4a61623c586f82065e8773821621f89141793f5bb7.jpg)

![e8bff68319d0cbd457dee5f885a9bcbce8a96458b1d1f463946cdaefd8e221f7.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/e8bff68319d0cbd457dee5f885a9bcbce8a96458b1d1f463946cdaefd8e221f7.jpg)

![f2be4eb47382802af7a0c025376d88f8bb81e1d35308e88af98714574e9811cd.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/f2be4eb47382802af7a0c025376d88f8bb81e1d35308e88af98714574e9811cd.jpg)

![f6ef4b1431ec27cd320cb1c29b98c6fcdd48b269593ad8da6c8d18f973bf37eb.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/images/f6ef4b1431ec27cd320cb1c29b98c6fcdd48b269593ad8da6c8d18f973bf37eb.jpg)

### Tables

![09e61cf12f98febb32e95d8bd9c4463b9270fc31bc13ca8951f675512de97165.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/tables/09e61cf12f98febb32e95d8bd9c4463b9270fc31bc13ca8951f675512de97165.jpg)

![12b3c7bcb8c4a730780aae0cb954558df3c772e0aabc9195e83ecb0cad44bf42.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/tables/12b3c7bcb8c4a730780aae0cb954558df3c772e0aabc9195e83ecb0cad44bf42.jpg)

![1cbffa870ead83493d8359be878f7218e9c63a57ef2ca0063d425bfe2719be46.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/tables/1cbffa870ead83493d8359be878f7218e9c63a57ef2ca0063d425bfe2719be46.jpg)

![78911c6e2920c71b9ea095e1c447a13b4dd53ce3d1a0ff80bcf6e6724f5e0f98.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/tables/78911c6e2920c71b9ea095e1c447a13b4dd53ce3d1a0ff80bcf6e6724f5e0f98.jpg)

![b37b6f5278a5336a31a126990e4b9469b77a94a31a2dfe2006f2869648839eed.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/tables/b37b6f5278a5336a31a126990e4b9469b77a94a31a2dfe2006f2869648839eed.jpg)

![d12e27b34139d494404f5a705e491438f4a804fa35e9f52b707f915b823f0f68.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/tables/d12e27b34139d494404f5a705e491438f4a804fa35e9f52b707f915b823f0f68.jpg)

![e033ee850f89dce01a87f1442d354952074ebf3e6f749826c35b113b4876c2a7.jpg](../icml_results/5218_Capacity-Agnostic%20Parameter%20Isolation%20for%20Continual%20Graph%20Learning/tables/e033ee850f89dce01a87f1442d354952074ebf3e6f749826c35b113b4876c2a7.jpg)

<a id="paper-5171"></a>

## Unifying Dataset Pruning and Distillation for Efficient Large-scale Compression


### Images

![085325ac7ef29fbf39a578b0aaf9a4aa0010abaf68081a1dcba0d65340f6f3bc.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/085325ac7ef29fbf39a578b0aaf9a4aa0010abaf68081a1dcba0d65340f6f3bc.jpg)

![0930626db04931e5030329205ab7013c8ff120ec4a5c759274720acef143f18f.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/0930626db04931e5030329205ab7013c8ff120ec4a5c759274720acef143f18f.jpg)

![332683ce34f62de4d84a0dccd5d78d23a66f8115ec8145b57c2389e1d7979091.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/332683ce34f62de4d84a0dccd5d78d23a66f8115ec8145b57c2389e1d7979091.jpg)

![3a881fc1d7bfd20f7b027d9960d3a85a22a0b8daad99d3326bda060cf832cb77.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/3a881fc1d7bfd20f7b027d9960d3a85a22a0b8daad99d3326bda060cf832cb77.jpg)

![42b9a8390568732adb355fa322f45dc15b8c8010bcf7534eeb5c0e8fe6264a3b.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/42b9a8390568732adb355fa322f45dc15b8c8010bcf7534eeb5c0e8fe6264a3b.jpg)

![58685f13d58c4ad768731a6ed68df9cf96d59ce60573208d90e6cf0d40e18d6f.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/58685f13d58c4ad768731a6ed68df9cf96d59ce60573208d90e6cf0d40e18d6f.jpg)

![5a33d6e08dfd26e41100d01f3a2ab952159a1f0f64bf337dc68173682ec5436c.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/5a33d6e08dfd26e41100d01f3a2ab952159a1f0f64bf337dc68173682ec5436c.jpg)

![6ea23717ade54a6cc6a31ca334c083a145a4cf0423af3ce534b0cb7a63f817b0.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/6ea23717ade54a6cc6a31ca334c083a145a4cf0423af3ce534b0cb7a63f817b0.jpg)

![785063bbd5269651cbdd0da45ce251fefec85dfb9e0982d1ab6abfb0b6e99a9d.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/785063bbd5269651cbdd0da45ce251fefec85dfb9e0982d1ab6abfb0b6e99a9d.jpg)

![79e58a227fda0b99b5fe26ab59ce6e5565b2036c207e2caf97e47b97aed70797.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/79e58a227fda0b99b5fe26ab59ce6e5565b2036c207e2caf97e47b97aed70797.jpg)

![8a285ace0091e8fad95b56dd4aed261b8d296359e1a7cf6affa88b9b53e0e1b6.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/8a285ace0091e8fad95b56dd4aed261b8d296359e1a7cf6affa88b9b53e0e1b6.jpg)

![96e44cf24621857c199291ac8eed31feab02f2b49b4d6cf2ca7cebc425745289.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/96e44cf24621857c199291ac8eed31feab02f2b49b4d6cf2ca7cebc425745289.jpg)

![9bed3c462435524dad9778b92f2b78c3ca5a471f18f311015fdbe655b99b6b91.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/9bed3c462435524dad9778b92f2b78c3ca5a471f18f311015fdbe655b99b6b91.jpg)

![a8f7946b6d069a020daf94bf38a92b61324949607c924b90065ad58220a92089.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/a8f7946b6d069a020daf94bf38a92b61324949607c924b90065ad58220a92089.jpg)

![b5f38210ea1eef1953a52fc1d51f916df73162dc81bc7fae163beca1c4a149ee.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/b5f38210ea1eef1953a52fc1d51f916df73162dc81bc7fae163beca1c4a149ee.jpg)

![c4c5476abb4e41158742be7ff93024b09156dc16213477fd01efdb571fd84e54.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/c4c5476abb4e41158742be7ff93024b09156dc16213477fd01efdb571fd84e54.jpg)

![dad87c0f72ffb405eaefbe4e425c30b6f8f22a274c10f913ca1e7bb585b7c9ce.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/dad87c0f72ffb405eaefbe4e425c30b6f8f22a274c10f913ca1e7bb585b7c9ce.jpg)

![ec515efaae9cde787dc723fc290ee4034036838a935925fac95b4542a42db0d0.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/ec515efaae9cde787dc723fc290ee4034036838a935925fac95b4542a42db0d0.jpg)

![f1625be4a35a67071ed6bdd18cd6e33bbd1a0a6788841cead592058b20773b7d.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/f1625be4a35a67071ed6bdd18cd6e33bbd1a0a6788841cead592058b20773b7d.jpg)

![f79adb0d23f0d3861362b99de598798d984e10adf5b4361bacf2ab6f18eb524d.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/images/f79adb0d23f0d3861362b99de598798d984e10adf5b4361bacf2ab6f18eb524d.jpg)

### Tables

![0f2834a5f01685e4f30f2bdcc8dfaa48e41f48ddb001a69de92d35694ce58b80.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/0f2834a5f01685e4f30f2bdcc8dfaa48e41f48ddb001a69de92d35694ce58b80.jpg)

![15047ea964cbae98f6461cffd1e2b60ad7959f3a53866bc6db4de811c4d39076.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/15047ea964cbae98f6461cffd1e2b60ad7959f3a53866bc6db4de811c4d39076.jpg)

![19ab03cc6a35767cfe65b7880de36d3066dff9fe5a6ea945dd23f6aafc60a943.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/19ab03cc6a35767cfe65b7880de36d3066dff9fe5a6ea945dd23f6aafc60a943.jpg)

![2b0fe5c7e82c1b8b380b53fa3ce9b6923edd804f2250a2ebe4d2db58a6be921d.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/2b0fe5c7e82c1b8b380b53fa3ce9b6923edd804f2250a2ebe4d2db58a6be921d.jpg)

![2d3af685b3b1890aa7597243d8595ac55d6a9326e333fd0fb7e889577b807e31.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/2d3af685b3b1890aa7597243d8595ac55d6a9326e333fd0fb7e889577b807e31.jpg)

![4064c935084f68c4b2be9d2058dbf12f69a613163717f2c402440723b5be3b8e.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/4064c935084f68c4b2be9d2058dbf12f69a613163717f2c402440723b5be3b8e.jpg)

![414ec23c5335fb90a40d38af1c897aeab0aa23df46543970a0bea62464d2f194.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/414ec23c5335fb90a40d38af1c897aeab0aa23df46543970a0bea62464d2f194.jpg)

![456660cef496b891b60d0bf39b6fd535e7627fe7c7dfb2ae26c16af91b13a935.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/456660cef496b891b60d0bf39b6fd535e7627fe7c7dfb2ae26c16af91b13a935.jpg)

![543e4b5b0403b3dcdb9e49ba68b786921c637d49a7f184560596473366665c04.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/543e4b5b0403b3dcdb9e49ba68b786921c637d49a7f184560596473366665c04.jpg)

![673e260cf5a975f0e032ff77e17699b4d9c704c70079c1f46908e13d188d173b.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/673e260cf5a975f0e032ff77e17699b4d9c704c70079c1f46908e13d188d173b.jpg)

![79d03576a8f001019e1f4ba772541c6eb92394be02f4ea2bfa0f43a7037afadf.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/79d03576a8f001019e1f4ba772541c6eb92394be02f4ea2bfa0f43a7037afadf.jpg)

![7e736d1bb52d75857422b5c178d67fd5fa85d8b8b10d4fbb88a2ba3562ca647c.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/7e736d1bb52d75857422b5c178d67fd5fa85d8b8b10d4fbb88a2ba3562ca647c.jpg)

![8d71aa36f85544ce3b1b2c2b5576c693a75b5eda025c5be275100640cdd39923.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/8d71aa36f85544ce3b1b2c2b5576c693a75b5eda025c5be275100640cdd39923.jpg)

![8e11ca5cf1f2daa1122f419e5e78f8c01c856be64841b63e349ce65f2fae29a0.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/8e11ca5cf1f2daa1122f419e5e78f8c01c856be64841b63e349ce65f2fae29a0.jpg)

![9112b26248ea7091c27e5fa55b464352d94d8fd0a6c330bf39d1b8ca3c81f7a6.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/9112b26248ea7091c27e5fa55b464352d94d8fd0a6c330bf39d1b8ca3c81f7a6.jpg)

![a19e9701a68e3a95616a4b385ec7759e6d730e11376baa0d2cc82242583769d2.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/a19e9701a68e3a95616a4b385ec7759e6d730e11376baa0d2cc82242583769d2.jpg)

![a9e8e1c3956077da194692128c2096620f2f3355406ff1d5070f4aefd4351ca7.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/a9e8e1c3956077da194692128c2096620f2f3355406ff1d5070f4aefd4351ca7.jpg)

![bc014461d38ad4ed74e73c2dffda6a0ae547ddb24c840a04d5705340a889df02.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/bc014461d38ad4ed74e73c2dffda6a0ae547ddb24c840a04d5705340a889df02.jpg)

![c1476e4f5e89c2d8a2f170db4eb1d912552718598316c2b669e5c06a2a75aac5.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/c1476e4f5e89c2d8a2f170db4eb1d912552718598316c2b669e5c06a2a75aac5.jpg)

![c8095fad4bde5e5bb80d53073ea68b193f6c8803eccbd83da95733ad7b5bb79a.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/c8095fad4bde5e5bb80d53073ea68b193f6c8803eccbd83da95733ad7b5bb79a.jpg)

![d195d681a22a6c9b7aeddb3463d2a1a4a0e10bf728475f35be3e1414206820e0.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/d195d681a22a6c9b7aeddb3463d2a1a4a0e10bf728475f35be3e1414206820e0.jpg)

![dcb0883b6c33f628010e058b3f2e7dbfa23be13d5607233e3663560710362556.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/dcb0883b6c33f628010e058b3f2e7dbfa23be13d5607233e3663560710362556.jpg)

![e2cc5652146d3774a4b2ba494f46b501c2e5abca5f3297dd82327b92400a7348.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/e2cc5652146d3774a4b2ba494f46b501c2e5abca5f3297dd82327b92400a7348.jpg)

![e5506dd29d480ba48b3d7b9908d2c9e80e5d99d4274ef1d9bde1fb6f9c07495f.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/e5506dd29d480ba48b3d7b9908d2c9e80e5d99d4274ef1d9bde1fb6f9c07495f.jpg)

![e711f4d7bb2199ae7ce4f6cb0f531872079eb75e1dda1ea62e90bcf1c3eaf241.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/e711f4d7bb2199ae7ce4f6cb0f531872079eb75e1dda1ea62e90bcf1c3eaf241.jpg)

![f2142201ad4f5fbf6349b41746842025330efc975554085da460873c98bed0ec.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/f2142201ad4f5fbf6349b41746842025330efc975554085da460873c98bed0ec.jpg)

![f664d571f14129c23388a112bbe7736645e268c887bf08a532cb143d0628b83b.jpg](../icml_results/5219_Unifying%20Dataset%20Pruning%20and%20Distillation%20for%20Efficient%20Large-scale%20Compression/tables/f664d571f14129c23388a112bbe7736645e268c887bf08a532cb143d0628b83b.jpg)

<a id="paper-5172"></a>

## Generalized Correctness Models: Learning Calibrated and Cross-Model Correctness Predictors from Historical Patterns


### Images

![1242a2a0c9089a3759153353942e9e8773c247704b54d091df5cf26f3a90c3ee.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/images/1242a2a0c9089a3759153353942e9e8773c247704b54d091df5cf26f3a90c3ee.jpg)

![2229a2b18b4643a3446726fdf085e08ebd9947e38b15634d72b3f69e1e8eca8a.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/images/2229a2b18b4643a3446726fdf085e08ebd9947e38b15634d72b3f69e1e8eca8a.jpg)

![59f12b7e3c684674d326b927e19625979d8ba965ef6a7e4daa9bef4dbe287e59.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/images/59f12b7e3c684674d326b927e19625979d8ba965ef6a7e4daa9bef4dbe287e59.jpg)

![89267f2bcb1f36c2feb4a7c656696039afc46749c43c4304ed53552c12b1cc91.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/images/89267f2bcb1f36c2feb4a7c656696039afc46749c43c4304ed53552c12b1cc91.jpg)

![e5c49302f639b847fa6f32dd8999fb80678ffb2de229a2892b58ba16e53f5d46.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/images/e5c49302f639b847fa6f32dd8999fb80678ffb2de229a2892b58ba16e53f5d46.jpg)

### Tables

![0528b1698609dde26b5122af4743d825341a29a95dd535b1a8f6838eb35ef3ef.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/0528b1698609dde26b5122af4743d825341a29a95dd535b1a8f6838eb35ef3ef.jpg)

![05cf8338d58ffd913ec5bcd9aed6f000236d142827e28c5c0dbbce72c5cac728.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/05cf8338d58ffd913ec5bcd9aed6f000236d142827e28c5c0dbbce72c5cac728.jpg)

![08d70ba8fdfe17632232781f48856f85a2950f7fe2f824ab7dcd0c5d44647c29.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/08d70ba8fdfe17632232781f48856f85a2950f7fe2f824ab7dcd0c5d44647c29.jpg)

![0a203e015983e7105b4df1f24485ac676308d4041cb2ebaa94e4c2835d0c3298.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/0a203e015983e7105b4df1f24485ac676308d4041cb2ebaa94e4c2835d0c3298.jpg)

![102b982b93deb7ae9156717bb080f9861c20899bdc5cb7f0a0311f98d0671708.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/102b982b93deb7ae9156717bb080f9861c20899bdc5cb7f0a0311f98d0671708.jpg)

![13160856eea48eab1a5e8fe5db709fd412006c765241b4ac3e9efa81ab861ea0.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/13160856eea48eab1a5e8fe5db709fd412006c765241b4ac3e9efa81ab861ea0.jpg)

![1a876c42d87b3d644cecd1e7ef43ebbd51f2def77d03d734be60832c8ff75893.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/1a876c42d87b3d644cecd1e7ef43ebbd51f2def77d03d734be60832c8ff75893.jpg)

![2a2a7f6cf6150d282b8041be330f04c476d2e5332026da2d0d9e5e90365da742.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/2a2a7f6cf6150d282b8041be330f04c476d2e5332026da2d0d9e5e90365da742.jpg)

![2cd764f4b644fb5eb5ea9e4be3d8a08db751ea68d960154459e813cd01a7eafb.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/2cd764f4b644fb5eb5ea9e4be3d8a08db751ea68d960154459e813cd01a7eafb.jpg)

![3727845021e488f10dc384785ee763f55e898304e48286a0005b94387f8543b8.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/3727845021e488f10dc384785ee763f55e898304e48286a0005b94387f8543b8.jpg)

![4117b2c0ac2a08651f601481333128717d4268c689e882016e806d82368a9524.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/4117b2c0ac2a08651f601481333128717d4268c689e882016e806d82368a9524.jpg)

![4437eb290322e1027d7b458c687711b1f3d33e4e6226e625229e5335523d7e64.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/4437eb290322e1027d7b458c687711b1f3d33e4e6226e625229e5335523d7e64.jpg)

![487cc28941c8e2fc346f6aeb55a0ae4f30528e5e1528739656efa4628d58c693.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/487cc28941c8e2fc346f6aeb55a0ae4f30528e5e1528739656efa4628d58c693.jpg)

![4e5f6ddcc5f3706ef426d678da1ee8d0c8dba3c59fd8961219241504092a4e69.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/4e5f6ddcc5f3706ef426d678da1ee8d0c8dba3c59fd8961219241504092a4e69.jpg)

![5e590a2b6772ca2c286b7c87300fc59618968b580cfe57a83e4fdfef51bc0d63.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/5e590a2b6772ca2c286b7c87300fc59618968b580cfe57a83e4fdfef51bc0d63.jpg)

![610c277ce7ece9f9a957d005df4dbd1c1bcb84b2ff793288b17a020b390499ed.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/610c277ce7ece9f9a957d005df4dbd1c1bcb84b2ff793288b17a020b390499ed.jpg)

![73c5d70992f262c90624d8a4efb41b680ce7dca0683ff7fd8e007e01be87035b.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/73c5d70992f262c90624d8a4efb41b680ce7dca0683ff7fd8e007e01be87035b.jpg)

![74851f50a8296606c582f1dea362471a2693a6ac0278a4618c4cb00ac23ed3c8.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/74851f50a8296606c582f1dea362471a2693a6ac0278a4618c4cb00ac23ed3c8.jpg)

![869b6612ee527eec167269a79ac912e2d460a102bdf1d2e260b3cdda53290334.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/869b6612ee527eec167269a79ac912e2d460a102bdf1d2e260b3cdda53290334.jpg)

![8c0893fecc3c0fe62e1f8d9f8244aecc5b84443e2dc861470ad0aa8786740bc3.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/8c0893fecc3c0fe62e1f8d9f8244aecc5b84443e2dc861470ad0aa8786740bc3.jpg)

![a17b44904f6af5b0b3938aabd70f274768540474d189cb5e7637f03e9fad26be.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/a17b44904f6af5b0b3938aabd70f274768540474d189cb5e7637f03e9fad26be.jpg)

![a57553189f0c63245847b07d01ec8127e4ba9330505dec28db87fd9acfe6c03e.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/a57553189f0c63245847b07d01ec8127e4ba9330505dec28db87fd9acfe6c03e.jpg)

![a6a02c7713666b8b4b4ce15050494f54926711f7e59a51a99b5b20a56207c23f.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/a6a02c7713666b8b4b4ce15050494f54926711f7e59a51a99b5b20a56207c23f.jpg)

![b75d3a72fddd185e90cffcb91b8f4412b42c91e9c5f889f5c1d5c29914e24f7f.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/b75d3a72fddd185e90cffcb91b8f4412b42c91e9c5f889f5c1d5c29914e24f7f.jpg)

![be4059e9fab939459bf8a7087d266ec893c79e4db6abcb55d0f8286f962decda.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/be4059e9fab939459bf8a7087d266ec893c79e4db6abcb55d0f8286f962decda.jpg)

![c3e3bb41ea830409a783ffc40e89b26ed1ed3a9837677e084ba79ff87aba87e2.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/c3e3bb41ea830409a783ffc40e89b26ed1ed3a9837677e084ba79ff87aba87e2.jpg)

![d005d6903ed6d51029121a2571a69814336388992726651396ff4502ed0f940a.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/d005d6903ed6d51029121a2571a69814336388992726651396ff4502ed0f940a.jpg)

![de9284cceebfa041660d7e0467872e7dfa33bb548c671ce41e43acef4daf3a9a.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/de9284cceebfa041660d7e0467872e7dfa33bb548c671ce41e43acef4daf3a9a.jpg)

![ea0c269df6c34a2e14283a34405c4467aab8f40e8b9153a3715b84734d1bc56f.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/ea0c269df6c34a2e14283a34405c4467aab8f40e8b9153a3715b84734d1bc56f.jpg)

![ece93f0f2a21e2977ffe51196467cd68ea9a34a00b02d8a23e22be120424370a.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/ece93f0f2a21e2977ffe51196467cd68ea9a34a00b02d8a23e22be120424370a.jpg)

![fd0b2754388f6fbbbb28ef0f3580b269dbbd19b514b3e7aaa53ffbc7a2f274b6.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/fd0b2754388f6fbbbb28ef0f3580b269dbbd19b514b3e7aaa53ffbc7a2f274b6.jpg)

![fd8e50b67f3ac9de90147af62bd222c2d1769a00f25cc6169a77d8626d7b2227.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/fd8e50b67f3ac9de90147af62bd222c2d1769a00f25cc6169a77d8626d7b2227.jpg)

![fe26869321db6f767b961257b414b073560c1bf98056ea20762117249a6f4143.jpg](../icml_results/5220_Generalized%20Correctness%20Models_%20Learning%20Calibrated%20and%20Cross-Model%20Correctness%20Predictors%20from%20Hist/tables/fe26869321db6f767b961257b414b073560c1bf98056ea20762117249a6f4143.jpg)

<a id="paper-5173"></a>

## Copula-SVI: Vine-Copula Variational Inference with Stein Refining for Instance-Level Correlation Capturing


### Images

![1d109d703eabcb20c8f6c3dde3dac8f95c0a4ee00e6c1f64b61c9149412c91df.jpg](../icml_results/5221_Copula-SVI_%20Vine-Copula%20Variational%20Inference%20with%20Stein%20Refining%20for%20Instance-Level%20Correlation%20Cap/images/1d109d703eabcb20c8f6c3dde3dac8f95c0a4ee00e6c1f64b61c9149412c91df.jpg)

![65e679bf72bbe5339a61aa87616e4f42eb32497267c2929529bb0229cd03034c.jpg](../icml_results/5221_Copula-SVI_%20Vine-Copula%20Variational%20Inference%20with%20Stein%20Refining%20for%20Instance-Level%20Correlation%20Cap/images/65e679bf72bbe5339a61aa87616e4f42eb32497267c2929529bb0229cd03034c.jpg)

![af52eb8e6c1dd460d46ef7bed58091b39da4b54eb3a197fff9572ccb7d76b9a8.jpg](../icml_results/5221_Copula-SVI_%20Vine-Copula%20Variational%20Inference%20with%20Stein%20Refining%20for%20Instance-Level%20Correlation%20Cap/images/af52eb8e6c1dd460d46ef7bed58091b39da4b54eb3a197fff9572ccb7d76b9a8.jpg)

![ebd8d869ea635dc1e33ed819d1f1e68f5c3e547d22e1d85ced9ec2ec7d8bd7c6.jpg](../icml_results/5221_Copula-SVI_%20Vine-Copula%20Variational%20Inference%20with%20Stein%20Refining%20for%20Instance-Level%20Correlation%20Cap/images/ebd8d869ea635dc1e33ed819d1f1e68f5c3e547d22e1d85ced9ec2ec7d8bd7c6.jpg)

### Tables

![0217406f50d6fee67870327b4a539513043e0964e208886afec68c67316b072e.jpg](../icml_results/5221_Copula-SVI_%20Vine-Copula%20Variational%20Inference%20with%20Stein%20Refining%20for%20Instance-Level%20Correlation%20Cap/tables/0217406f50d6fee67870327b4a539513043e0964e208886afec68c67316b072e.jpg)

![05afb08713c625002ac87358c2ffdea12a6d2063b687d8ac2725434d8769b714.jpg](../icml_results/5221_Copula-SVI_%20Vine-Copula%20Variational%20Inference%20with%20Stein%20Refining%20for%20Instance-Level%20Correlation%20Cap/tables/05afb08713c625002ac87358c2ffdea12a6d2063b687d8ac2725434d8769b714.jpg)

![12d963ce01a9ef03bd7b05a5ce86ae8f4f9cbaeaad77df42d5ea02c3c6f0a999.jpg](../icml_results/5221_Copula-SVI_%20Vine-Copula%20Variational%20Inference%20with%20Stein%20Refining%20for%20Instance-Level%20Correlation%20Cap/tables/12d963ce01a9ef03bd7b05a5ce86ae8f4f9cbaeaad77df42d5ea02c3c6f0a999.jpg)

![3c29363ee2c9fe9fec23343728e6331793cb908e5adc3edb14776e018a4bccb9.jpg](../icml_results/5221_Copula-SVI_%20Vine-Copula%20Variational%20Inference%20with%20Stein%20Refining%20for%20Instance-Level%20Correlation%20Cap/tables/3c29363ee2c9fe9fec23343728e6331793cb908e5adc3edb14776e018a4bccb9.jpg)

![504779ad7ad131927c92a5d18932954fa656aa30ce0b87c170714c82bcd80125.jpg](../icml_results/5221_Copula-SVI_%20Vine-Copula%20Variational%20Inference%20with%20Stein%20Refining%20for%20Instance-Level%20Correlation%20Cap/tables/504779ad7ad131927c92a5d18932954fa656aa30ce0b87c170714c82bcd80125.jpg)

![53c38ea2a61d298f4c7daafd88d39bb79184d4a13841b3fa1e1941cc26ff64cf.jpg](../icml_results/5221_Copula-SVI_%20Vine-Copula%20Variational%20Inference%20with%20Stein%20Refining%20for%20Instance-Level%20Correlation%20Cap/tables/53c38ea2a61d298f4c7daafd88d39bb79184d4a13841b3fa1e1941cc26ff64cf.jpg)

![5419114ab1fe040a02b6fb6be0f1e199347b421b95a9be0a112063305119996c.jpg](../icml_results/5221_Copula-SVI_%20Vine-Copula%20Variational%20Inference%20with%20Stein%20Refining%20for%20Instance-Level%20Correlation%20Cap/tables/5419114ab1fe040a02b6fb6be0f1e199347b421b95a9be0a112063305119996c.jpg)

![5f2598516ef1d55689dc34d06015a51283bba0ca94be2900aa42107c6ff7b86d.jpg](../icml_results/5221_Copula-SVI_%20Vine-Copula%20Variational%20Inference%20with%20Stein%20Refining%20for%20Instance-Level%20Correlation%20Cap/tables/5f2598516ef1d55689dc34d06015a51283bba0ca94be2900aa42107c6ff7b86d.jpg)

![aab86fc48962fae090d7cddc5d1139846dc7c20139c09f13e5ea079f98289099.jpg](../icml_results/5221_Copula-SVI_%20Vine-Copula%20Variational%20Inference%20with%20Stein%20Refining%20for%20Instance-Level%20Correlation%20Cap/tables/aab86fc48962fae090d7cddc5d1139846dc7c20139c09f13e5ea079f98289099.jpg)

![abb4f3ccc7c841adece7ed50bd261dea2bba5b849e64c9d08d88f972aeb85333.jpg](../icml_results/5221_Copula-SVI_%20Vine-Copula%20Variational%20Inference%20with%20Stein%20Refining%20for%20Instance-Level%20Correlation%20Cap/tables/abb4f3ccc7c841adece7ed50bd261dea2bba5b849e64c9d08d88f972aeb85333.jpg)

![e857b8c1d6154d68582c46e9c98574398d910c64e0dce7b4de3f45379c97bd3f.jpg](../icml_results/5221_Copula-SVI_%20Vine-Copula%20Variational%20Inference%20with%20Stein%20Refining%20for%20Instance-Level%20Correlation%20Cap/tables/e857b8c1d6154d68582c46e9c98574398d910c64e0dce7b4de3f45379c97bd3f.jpg)

<a id="paper-5174"></a>

## AIR: Improving Agent Safety through Incident Response


### Images

![5b401bccc02b117b621cb4c4c07fbc4fd5ea5acdf5361ae76c97be67d3952b05.jpg](../icml_results/5222_AIR_%20Improving%20Agent%20Safety%20through%20Incident%20Response/images/5b401bccc02b117b621cb4c4c07fbc4fd5ea5acdf5361ae76c97be67d3952b05.jpg)

![83b7f2004d029fe4f783dde1c12181b0552a99379bc84d8f43a1efb38fc0a243.jpg](../icml_results/5222_AIR_%20Improving%20Agent%20Safety%20through%20Incident%20Response/images/83b7f2004d029fe4f783dde1c12181b0552a99379bc84d8f43a1efb38fc0a243.jpg)

![c7664181eaf092cc31f4e31bb3c41d17b381338cfd4505b48b388949e97b56cf.jpg](../icml_results/5222_AIR_%20Improving%20Agent%20Safety%20through%20Incident%20Response/images/c7664181eaf092cc31f4e31bb3c41d17b381338cfd4505b48b388949e97b56cf.jpg)

![d2318f9a667cc97603889fec989d5bc4804f8cabace20999acc8cf389ed705ae.jpg](../icml_results/5222_AIR_%20Improving%20Agent%20Safety%20through%20Incident%20Response/images/d2318f9a667cc97603889fec989d5bc4804f8cabace20999acc8cf389ed705ae.jpg)

![d2389d156a9c5b678221799f500e0fda4b7cd98941dbdaf9c71eed89db2a9cb5.jpg](../icml_results/5222_AIR_%20Improving%20Agent%20Safety%20through%20Incident%20Response/images/d2389d156a9c5b678221799f500e0fda4b7cd98941dbdaf9c71eed89db2a9cb5.jpg)

![e6796a01c1ec301fa89e12bad68e28a84fba4d0acd79394b30eb27436e3283e2.jpg](../icml_results/5222_AIR_%20Improving%20Agent%20Safety%20through%20Incident%20Response/images/e6796a01c1ec301fa89e12bad68e28a84fba4d0acd79394b30eb27436e3283e2.jpg)

![f7e49acd63aad4b2313bff8f2e248487233d6affdc409a500ca9e522ab573f6d.jpg](../icml_results/5222_AIR_%20Improving%20Agent%20Safety%20through%20Incident%20Response/images/f7e49acd63aad4b2313bff8f2e248487233d6affdc409a500ca9e522ab573f6d.jpg)

![fd690f3ea4468f1e9266469512c46b0485b1aba5f560893c96bf15ff9097baf8.jpg](../icml_results/5222_AIR_%20Improving%20Agent%20Safety%20through%20Incident%20Response/images/fd690f3ea4468f1e9266469512c46b0485b1aba5f560893c96bf15ff9097baf8.jpg)

### Tables

![129721fc8293bf4e260054cb190a3e8b3b8ca4e554e2e53e5486b19d57c196e3.jpg](../icml_results/5222_AIR_%20Improving%20Agent%20Safety%20through%20Incident%20Response/tables/129721fc8293bf4e260054cb190a3e8b3b8ca4e554e2e53e5486b19d57c196e3.jpg)

![2b57ea1f3a2bc73832b811ddd79849ed7d5bdef29429793df4a5606f97a18721.jpg](../icml_results/5222_AIR_%20Improving%20Agent%20Safety%20through%20Incident%20Response/tables/2b57ea1f3a2bc73832b811ddd79849ed7d5bdef29429793df4a5606f97a18721.jpg)

![32fa5f3f2be3f6b3cd821dce778804a123b7747fc25b37d54125f32e17e0177f.jpg](../icml_results/5222_AIR_%20Improving%20Agent%20Safety%20through%20Incident%20Response/tables/32fa5f3f2be3f6b3cd821dce778804a123b7747fc25b37d54125f32e17e0177f.jpg)

![57de5431081ff018d9379f3f5bb5862a3c06d3299e9206df269770eff6875692.jpg](../icml_results/5222_AIR_%20Improving%20Agent%20Safety%20through%20Incident%20Response/tables/57de5431081ff018d9379f3f5bb5862a3c06d3299e9206df269770eff6875692.jpg)

![67214467e69e3692368e852c8cd800a8860aa314bbf0b7d44f7e65e04b738119.jpg](../icml_results/5222_AIR_%20Improving%20Agent%20Safety%20through%20Incident%20Response/tables/67214467e69e3692368e852c8cd800a8860aa314bbf0b7d44f7e65e04b738119.jpg)

![873f36d1d557882f6555d553e15ea72b542b40f91fa86cafd0c7191775e39e2d.jpg](../icml_results/5222_AIR_%20Improving%20Agent%20Safety%20through%20Incident%20Response/tables/873f36d1d557882f6555d553e15ea72b542b40f91fa86cafd0c7191775e39e2d.jpg)

![f717a1dae475e5b09763729ac2dc118b322b626b683a40ac3d4d039adee8dfef.jpg](../icml_results/5222_AIR_%20Improving%20Agent%20Safety%20through%20Incident%20Response/tables/f717a1dae475e5b09763729ac2dc118b322b626b683a40ac3d4d039adee8dfef.jpg)

<a id="paper-5175"></a>

## WebWorld: A Large-Scale World Model for Web Agent Training


### Images

![4bacf31036ff385116a390922f98164e76f070781aafde100e29a3f2c12f34c3.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/images/4bacf31036ff385116a390922f98164e76f070781aafde100e29a3f2c12f34c3.jpg)

![5c6fc2cecf934ab2bce536a38d841ea542569511062e4e1b52db5db2a58d1b12.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/images/5c6fc2cecf934ab2bce536a38d841ea542569511062e4e1b52db5db2a58d1b12.jpg)

![9f70745c71ad86b541573291e3fb7842ef3af2558a9573507dba0dfff3b0177d.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/images/9f70745c71ad86b541573291e3fb7842ef3af2558a9573507dba0dfff3b0177d.jpg)

![a7d422ce623e55c51708ff50a8c929f313e83a4cf844bb810b6cdf2204732852.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/images/a7d422ce623e55c51708ff50a8c929f313e83a4cf844bb810b6cdf2204732852.jpg)

![acb65732aaac650a7ca54f12112ab909f1760d830342e759082fe8fed3743784.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/images/acb65732aaac650a7ca54f12112ab909f1760d830342e759082fe8fed3743784.jpg)

![d156cb781311513949db4cf90ec9f213f90652cf684ff43b05a278dbb75bf2e8.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/images/d156cb781311513949db4cf90ec9f213f90652cf684ff43b05a278dbb75bf2e8.jpg)

![d955892465ae33500e86173951ce5b7684d4013bd46080f88e86ad04ecb8d072.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/images/d955892465ae33500e86173951ce5b7684d4013bd46080f88e86ad04ecb8d072.jpg)

### Tables

![011e407763f495296931cfc975d6f2c8b8a4adf56eab2178c2aa39a9860971ea.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/tables/011e407763f495296931cfc975d6f2c8b8a4adf56eab2178c2aa39a9860971ea.jpg)

![1d5c3c530ba823b96320c7ea262dbc52e3bf41fdc6324880ec42d81f5cbfa57a.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/tables/1d5c3c530ba823b96320c7ea262dbc52e3bf41fdc6324880ec42d81f5cbfa57a.jpg)

![26a31acdbe7cb475febbded8c51a334a2f7c0d9f6ba2ca34bbad71b6a9e4516f.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/tables/26a31acdbe7cb475febbded8c51a334a2f7c0d9f6ba2ca34bbad71b6a9e4516f.jpg)

![274422158ac7203d89d6d95800707d8e92cea6c3215ece3ef02cbae77ec0e4bc.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/tables/274422158ac7203d89d6d95800707d8e92cea6c3215ece3ef02cbae77ec0e4bc.jpg)

![2ae4a6bc4867365d0780538c5b0a77cf07eee136071acf5b4d51e8d7357b6816.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/tables/2ae4a6bc4867365d0780538c5b0a77cf07eee136071acf5b4d51e8d7357b6816.jpg)

![4cb5efa2628416fd3cd210b7f3039aa52d3de3d7f9ded76f4071d503895e1449.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/tables/4cb5efa2628416fd3cd210b7f3039aa52d3de3d7f9ded76f4071d503895e1449.jpg)

![72f95b87abfe2a56bcf4bf9d3e5f5d671e41e1e191e5bafd6cd483791b36456c.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/tables/72f95b87abfe2a56bcf4bf9d3e5f5d671e41e1e191e5bafd6cd483791b36456c.jpg)

![77f9438da2bd79742148cc681079a29656cc0396ceb60582566a4c13c1430f54.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/tables/77f9438da2bd79742148cc681079a29656cc0396ceb60582566a4c13c1430f54.jpg)

![84f17eac30af849084b69e88ffce1bcc7b8c700477e111624450642c64f53251.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/tables/84f17eac30af849084b69e88ffce1bcc7b8c700477e111624450642c64f53251.jpg)

![964d4166fee5b1af95108f6d0d612b312c39088a39ac8e5f8497318ef9d1fb1a.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/tables/964d4166fee5b1af95108f6d0d612b312c39088a39ac8e5f8497318ef9d1fb1a.jpg)

![98f14c4b5bb4133705ac806630b14370d13a583d787b424df0f22d90ddec90f6.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/tables/98f14c4b5bb4133705ac806630b14370d13a583d787b424df0f22d90ddec90f6.jpg)

![ac324189443f75907df9f3b9ee255f9c978d277d5ba091690ff31f9d788248b6.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/tables/ac324189443f75907df9f3b9ee255f9c978d277d5ba091690ff31f9d788248b6.jpg)

![cff729dc7b430eb011f3b6337e6454ae1795bde6bddf887e6f9457b2428932f8.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/tables/cff729dc7b430eb011f3b6337e6454ae1795bde6bddf887e6f9457b2428932f8.jpg)

![ee4bd55769407d4324c6f31724ed6ddc0811f99f53d5683e381a191938a132b5.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/tables/ee4bd55769407d4324c6f31724ed6ddc0811f99f53d5683e381a191938a132b5.jpg)

![f14229ca78dbee17f6f44b489a316818768a67d0c9e8e5e199b27b67bade91bf.jpg](../icml_results/5223_WebWorld_%20A%20Large-Scale%20World%20Model%20for%20Web%20Agent%20Training/tables/f14229ca78dbee17f6f44b489a316818768a67d0c9e8e5e199b27b67bade91bf.jpg)

<a id="paper-5176"></a>

## FedHera: Towards Drift-Resilient Federated Fine-tuning with Heterogeneous Resources


### Images

![1e9c27a6aafb887bfb19ac8b3899e8cbe8ab49ca560f2ea0480b92d4d8dc06d2.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/images/1e9c27a6aafb887bfb19ac8b3899e8cbe8ab49ca560f2ea0480b92d4d8dc06d2.jpg)

![318a554811f3e0cfe0001f9f2dd533cc76c044b428d862062c258c2d170aa915.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/images/318a554811f3e0cfe0001f9f2dd533cc76c044b428d862062c258c2d170aa915.jpg)

![4e8164bb9ece9dcf2b12a0753290e8ce1abb5146b19d2341d7554cbeeab7e6b5.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/images/4e8164bb9ece9dcf2b12a0753290e8ce1abb5146b19d2341d7554cbeeab7e6b5.jpg)

![5012bdf94dcb004f5e004682b42f9e2030e3cbf63324b9bd3e9816f6a98c7248.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/images/5012bdf94dcb004f5e004682b42f9e2030e3cbf63324b9bd3e9816f6a98c7248.jpg)

![7812f28ad137de3f7e819578901622a91d53fb4fd5309c030f80ee97eef9be76.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/images/7812f28ad137de3f7e819578901622a91d53fb4fd5309c030f80ee97eef9be76.jpg)

![831dcb81b8be3358bcd847c8cf03e6d5457906d7e65c3465167033a15f3d34ba.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/images/831dcb81b8be3358bcd847c8cf03e6d5457906d7e65c3465167033a15f3d34ba.jpg)

![945d765c18c32cfe933c46b126b5e250d6b7c970c904fa58085b3c3a894b86ab.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/images/945d765c18c32cfe933c46b126b5e250d6b7c970c904fa58085b3c3a894b86ab.jpg)

![97d71908d33b644bfae76c69433619481a4f1e75e44ec630e26d0e006df890f3.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/images/97d71908d33b644bfae76c69433619481a4f1e75e44ec630e26d0e006df890f3.jpg)

![ab0251575ef69721684b7300cb78f645534619f06dc77e3020f4b33c295034f3.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/images/ab0251575ef69721684b7300cb78f645534619f06dc77e3020f4b33c295034f3.jpg)

![b6686fec0618f50af8c059fa36b0eae12f04ceb74e9950785a0db01fcdc51f2a.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/images/b6686fec0618f50af8c059fa36b0eae12f04ceb74e9950785a0db01fcdc51f2a.jpg)

### Tables

![01ee7edca2751088c60d7854862cac907b42caf5573dbd65bcdb806b53d8f725.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/tables/01ee7edca2751088c60d7854862cac907b42caf5573dbd65bcdb806b53d8f725.jpg)

![28f1467598f75b5c114652525b32c98ecc40df5fca522984367302a251528d5e.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/tables/28f1467598f75b5c114652525b32c98ecc40df5fca522984367302a251528d5e.jpg)

![322d743e0aa5a76c8e152458a5aae05e0ccce03e9fda6cddc6735efeee3b8745.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/tables/322d743e0aa5a76c8e152458a5aae05e0ccce03e9fda6cddc6735efeee3b8745.jpg)

![329e08787b50710c5e3860ad351834e0838f2c1c8f4f0fecafec3cd40ca23a9d.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/tables/329e08787b50710c5e3860ad351834e0838f2c1c8f4f0fecafec3cd40ca23a9d.jpg)

![4d287099fdba7923b2430f3aff963b65e8e2be217fe2bd529df307eb44b2438c.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/tables/4d287099fdba7923b2430f3aff963b65e8e2be217fe2bd529df307eb44b2438c.jpg)

![5356dfcfa343023929d73e9d8dd3a953f1fec552cccb0dfa7d80cd0f1bc2e8da.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/tables/5356dfcfa343023929d73e9d8dd3a953f1fec552cccb0dfa7d80cd0f1bc2e8da.jpg)

![5d141b2063894cc5a7881676776f79180351c2586b79e056eddb5c2ba2cc4d7f.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/tables/5d141b2063894cc5a7881676776f79180351c2586b79e056eddb5c2ba2cc4d7f.jpg)

![79259f563b0899727fa89732744f26ae4e7d3d83a860257152ca3a5a84ef959f.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/tables/79259f563b0899727fa89732744f26ae4e7d3d83a860257152ca3a5a84ef959f.jpg)

![9d56cc313ae4c7d1f54e91ab8ed03e246dad043fafcd30049ce2e03d002e05c6.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/tables/9d56cc313ae4c7d1f54e91ab8ed03e246dad043fafcd30049ce2e03d002e05c6.jpg)

![a4bb76e6ff79e07a48991b80d60949dd8793d2f2da9ab5b6a61cafd1d5bae0e7.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/tables/a4bb76e6ff79e07a48991b80d60949dd8793d2f2da9ab5b6a61cafd1d5bae0e7.jpg)

![b7fba537bdc72f73a9cac3431f87c8c5ae68914a96647ba460f75ede6387ca40.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/tables/b7fba537bdc72f73a9cac3431f87c8c5ae68914a96647ba460f75ede6387ca40.jpg)

![c10425b4d106b4e0f3a56ee1cd85cfb8704f2fe96f08ef29d5616cba04e43611.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/tables/c10425b4d106b4e0f3a56ee1cd85cfb8704f2fe96f08ef29d5616cba04e43611.jpg)

![e184529606e91b73ce2f580d077354a9387c30048f6a526f144b2feb801953fd.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/tables/e184529606e91b73ce2f580d077354a9387c30048f6a526f144b2feb801953fd.jpg)

![f40e53b88e3ee1933a406d120097fdecb2be68d83673bac45760e951e176f4f5.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/tables/f40e53b88e3ee1933a406d120097fdecb2be68d83673bac45760e951e176f4f5.jpg)

![f70c565c9c53513a6f1cf93a2a934da85fb6dc15246bd2f3aec2faead2cee690.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/tables/f70c565c9c53513a6f1cf93a2a934da85fb6dc15246bd2f3aec2faead2cee690.jpg)

![f96da30287ecc9e082300f44b7963e9aa5c24fdaf3d9de6745fce9dcab9e777d.jpg](../icml_results/5224_FedHera_%20Towards%20Drift-Resilient%20Federated%20Fine-tuning%20with%20Heterogeneous%20Resources/tables/f96da30287ecc9e082300f44b7963e9aa5c24fdaf3d9de6745fce9dcab9e777d.jpg)

<a id="paper-5177"></a>

## Editable Proof Sketch for Automated Theorem Proving


### Images

![00f8a62e5d4e25e54e8e8215c1aa2f3cb687c290c8bff2c331c8d0797ac24894.jpg](../icml_results/5225_Editable%20Proof%20Sketch%20for%20Automated%20Theorem%20Proving/images/00f8a62e5d4e25e54e8e8215c1aa2f3cb687c290c8bff2c331c8d0797ac24894.jpg)

![02b8432a6fd85b2a09d40ae68fc358778873a31b9ab9ff01f78236a8d66c0bef.jpg](../icml_results/5225_Editable%20Proof%20Sketch%20for%20Automated%20Theorem%20Proving/images/02b8432a6fd85b2a09d40ae68fc358778873a31b9ab9ff01f78236a8d66c0bef.jpg)

![650e697d8a4a092ce2020d155be433d8939b27c1949bfbc8a77e508e69bb0c35.jpg](../icml_results/5225_Editable%20Proof%20Sketch%20for%20Automated%20Theorem%20Proving/images/650e697d8a4a092ce2020d155be433d8939b27c1949bfbc8a77e508e69bb0c35.jpg)

![75d0ac1f292567d9e4d992650217cdea7824a32b2fe6750524fe8a2a4473f12a.jpg](../icml_results/5225_Editable%20Proof%20Sketch%20for%20Automated%20Theorem%20Proving/images/75d0ac1f292567d9e4d992650217cdea7824a32b2fe6750524fe8a2a4473f12a.jpg)

![89e7f6b74a7c28541939df2dd9b730fc3b901407b813be125a3e246cb51ae75f.jpg](../icml_results/5225_Editable%20Proof%20Sketch%20for%20Automated%20Theorem%20Proving/images/89e7f6b74a7c28541939df2dd9b730fc3b901407b813be125a3e246cb51ae75f.jpg)

![e7640deece26b655edce42e4dcbfee8818ef2006a92888fa3bdbb965c5614b28.jpg](../icml_results/5225_Editable%20Proof%20Sketch%20for%20Automated%20Theorem%20Proving/images/e7640deece26b655edce42e4dcbfee8818ef2006a92888fa3bdbb965c5614b28.jpg)

![f23046b0c35c225ce7f563085c94db3b6daaa2056fdc64737e96412380cfd1d9.jpg](../icml_results/5225_Editable%20Proof%20Sketch%20for%20Automated%20Theorem%20Proving/images/f23046b0c35c225ce7f563085c94db3b6daaa2056fdc64737e96412380cfd1d9.jpg)

### Tables

![2363bdc407be88821be647f8473355f086f7ef75ea2d373d8c4a34bf91d4b82a.jpg](../icml_results/5225_Editable%20Proof%20Sketch%20for%20Automated%20Theorem%20Proving/tables/2363bdc407be88821be647f8473355f086f7ef75ea2d373d8c4a34bf91d4b82a.jpg)

![28e078425f1d487dc181d28a176589dd43fe3a136b72ff8e4ed550c92be7fbbb.jpg](../icml_results/5225_Editable%20Proof%20Sketch%20for%20Automated%20Theorem%20Proving/tables/28e078425f1d487dc181d28a176589dd43fe3a136b72ff8e4ed550c92be7fbbb.jpg)

![3310b36fec079ddb70bcae88a911de48ef0fbaa9db4a4c137f1136f618c34308.jpg](../icml_results/5225_Editable%20Proof%20Sketch%20for%20Automated%20Theorem%20Proving/tables/3310b36fec079ddb70bcae88a911de48ef0fbaa9db4a4c137f1136f618c34308.jpg)

![b01861d82cde6b6a73d62a717cd7e25f4f0114f81faf001abd6e6a28d5b23fe9.jpg](../icml_results/5225_Editable%20Proof%20Sketch%20for%20Automated%20Theorem%20Proving/tables/b01861d82cde6b6a73d62a717cd7e25f4f0114f81faf001abd6e6a28d5b23fe9.jpg)

<a id="paper-5178"></a>

## Kuramoto Oscillatory Phase Encoding: Neuro-inspired Synchronization for Improved Learning Efficiency


### Images

![1cc7c5d23f3a4f558bbdf406db70c484f96805548762ad4120a3efc7105b93b2.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/images/1cc7c5d23f3a4f558bbdf406db70c484f96805548762ad4120a3efc7105b93b2.jpg)

![262c627a002d5efdb65990541280f3a06ed8f37244f9da1e93cbd65f4197ce0e.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/images/262c627a002d5efdb65990541280f3a06ed8f37244f9da1e93cbd65f4197ce0e.jpg)

![4cdcd19f772b5be386e0d5ccc44011d5cbf113121618e9b5dca003895dee136a.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/images/4cdcd19f772b5be386e0d5ccc44011d5cbf113121618e9b5dca003895dee136a.jpg)

![4f5414045d5b72561b1956daad579330db2a386415f2fffb89d3d12fc34aa123.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/images/4f5414045d5b72561b1956daad579330db2a386415f2fffb89d3d12fc34aa123.jpg)

![5f360ebb9619bb844c9f4999102c67631d8e6ef18e93f655e0d6415f468050a9.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/images/5f360ebb9619bb844c9f4999102c67631d8e6ef18e93f655e0d6415f468050a9.jpg)

![83e3138687280c2b3487668ea0d208b137ed5426952948ac342adbbc43dbe292.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/images/83e3138687280c2b3487668ea0d208b137ed5426952948ac342adbbc43dbe292.jpg)

![86ef99e34cdd991f8564e60fa65692222911b8e5850ab752f81475b8485ca4fe.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/images/86ef99e34cdd991f8564e60fa65692222911b8e5850ab752f81475b8485ca4fe.jpg)

![9717ae504172747f596dbf7023dcbf802827d0634617cd5d4d6723bda17a80f0.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/images/9717ae504172747f596dbf7023dcbf802827d0634617cd5d4d6723bda17a80f0.jpg)

![990b7ee61f099ac61b015da70df98d8bb6b545c3b2af6946f5d3682be7202e8d.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/images/990b7ee61f099ac61b015da70df98d8bb6b545c3b2af6946f5d3682be7202e8d.jpg)

![a31c14361a6179f30c072fce3b96e80853e66076a2a56fa8bfab63c36a8326fb.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/images/a31c14361a6179f30c072fce3b96e80853e66076a2a56fa8bfab63c36a8326fb.jpg)

![edd1bd143d875aa38864f218cb77ae2b33c931209478c942b0f7ba0a0bfc0927.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/images/edd1bd143d875aa38864f218cb77ae2b33c931209478c942b0f7ba0a0bfc0927.jpg)

![f79a3e14db84012a1defe319965be01c6db06d2e4b9445e8834b22754d5fa253.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/images/f79a3e14db84012a1defe319965be01c6db06d2e4b9445e8834b22754d5fa253.jpg)

### Tables

![0a32bfd1262818f487c918005d05447d8814057e44dc03600753cdc904c0f384.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/tables/0a32bfd1262818f487c918005d05447d8814057e44dc03600753cdc904c0f384.jpg)

![1d50e094ec7b7516029641d24b50193f5b42f65495613e096414ceaed01c2931.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/tables/1d50e094ec7b7516029641d24b50193f5b42f65495613e096414ceaed01c2931.jpg)

![268942dd1cca1f53a59ecd414aa4436c4c0b4a055603f1b7a58060e2148f392e.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/tables/268942dd1cca1f53a59ecd414aa4436c4c0b4a055603f1b7a58060e2148f392e.jpg)

![363ba616ced4b9b0b81c8e7b5b91d41ecb00b8e27459839d80d6db51323eb884.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/tables/363ba616ced4b9b0b81c8e7b5b91d41ecb00b8e27459839d80d6db51323eb884.jpg)

![40f8fe2d2721c0bd0bc715764c32c283846cb931a260e79aa01f449f7b6da61a.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/tables/40f8fe2d2721c0bd0bc715764c32c283846cb931a260e79aa01f449f7b6da61a.jpg)

![491496d7ac1dccce8c9641ce31ae60c5f92093dff5ea1825fc08cf5476106427.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/tables/491496d7ac1dccce8c9641ce31ae60c5f92093dff5ea1825fc08cf5476106427.jpg)

![5a25d98d392aeb581676e8bda75c693b8ecbe4d8063a8f9b0d49972fe6bc51d8.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/tables/5a25d98d392aeb581676e8bda75c693b8ecbe4d8063a8f9b0d49972fe6bc51d8.jpg)

![5dde0ec11a8299e82b73c0f4a8b38fa491f65c225bd085302009b393373c6bb7.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/tables/5dde0ec11a8299e82b73c0f4a8b38fa491f65c225bd085302009b393373c6bb7.jpg)

![660b98b0c9d2174c88359eb8f44f48e1f46bdbdb99889deeabc6a1d7a7aeb000.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/tables/660b98b0c9d2174c88359eb8f44f48e1f46bdbdb99889deeabc6a1d7a7aeb000.jpg)

![6f01ade7067846b8baefa41223b27aba4df7af00d130ee12f883b7714b8d2928.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/tables/6f01ade7067846b8baefa41223b27aba4df7af00d130ee12f883b7714b8d2928.jpg)

![77a166b0f5333e9e780ef32cc59061ca1ba392e8efa3364fcccb4bb02cbb4bdd.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/tables/77a166b0f5333e9e780ef32cc59061ca1ba392e8efa3364fcccb4bb02cbb4bdd.jpg)

![ba0c952623d00557c62f2fbc7dcbfd53428207d1f0b4696cddf63bf2446c5255.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/tables/ba0c952623d00557c62f2fbc7dcbfd53428207d1f0b4696cddf63bf2446c5255.jpg)

![c31c923d376b99b81b5713942eded3d43d18f887f835437c919fca16439dd56a.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/tables/c31c923d376b99b81b5713942eded3d43d18f887f835437c919fca16439dd56a.jpg)

![c493577e5a7974a58aab98e169b466b8c219ae5e22592b284815287b97138771.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/tables/c493577e5a7974a58aab98e169b466b8c219ae5e22592b284815287b97138771.jpg)

![ff85351e5c67062f9e232c7d36e50667be4a1f93308f376b4f2d03f46b27d90d.jpg](../icml_results/5226_Kuramoto%20Oscillatory%20Phase%20Encoding_%20Neuro-inspired%20Synchronization%20for%20Improved%20Learning%20Efficiency/tables/ff85351e5c67062f9e232c7d36e50667be4a1f93308f376b4f2d03f46b27d90d.jpg)

<a id="paper-5179"></a>

## LECTOR: Joint Optimization of Scientific Reasoning Graphs and Introduction Generation


### Images

![3cd35bf0205d8d56496fef1412bb83a2f65cc3bc67a8a510cb738fbbc72859c4.jpg](../icml_results/5227_LECTOR_%20Joint%20Optimization%20of%20Scientific%20Reasoning%20Graphs%20and%20Introduction%20Generation/images/3cd35bf0205d8d56496fef1412bb83a2f65cc3bc67a8a510cb738fbbc72859c4.jpg)

![8e57b4734cb8e054cc9bac5280e8ecb18f5296d98c210db338ee54af0f78e7e7.jpg](../icml_results/5227_LECTOR_%20Joint%20Optimization%20of%20Scientific%20Reasoning%20Graphs%20and%20Introduction%20Generation/images/8e57b4734cb8e054cc9bac5280e8ecb18f5296d98c210db338ee54af0f78e7e7.jpg)

### Tables

![774d3688998f96aafd1deda110ec6af85a0aee182c5a07797557d2f48bfe1295.jpg](../icml_results/5227_LECTOR_%20Joint%20Optimization%20of%20Scientific%20Reasoning%20Graphs%20and%20Introduction%20Generation/tables/774d3688998f96aafd1deda110ec6af85a0aee182c5a07797557d2f48bfe1295.jpg)

![93579e1daa7aa010903e1d3fbd09126ba92d386a063ef38d4a078769e3e45ff2.jpg](../icml_results/5227_LECTOR_%20Joint%20Optimization%20of%20Scientific%20Reasoning%20Graphs%20and%20Introduction%20Generation/tables/93579e1daa7aa010903e1d3fbd09126ba92d386a063ef38d4a078769e3e45ff2.jpg)

![a9347bdc9dcf8a850b6c004479112481400db476380fe641de206885c8fdf3af.jpg](../icml_results/5227_LECTOR_%20Joint%20Optimization%20of%20Scientific%20Reasoning%20Graphs%20and%20Introduction%20Generation/tables/a9347bdc9dcf8a850b6c004479112481400db476380fe641de206885c8fdf3af.jpg)

![cda944b9d7d109299f2f53601ae36eb5596ab0fdf04ed8d528d77e5a0196c530.jpg](../icml_results/5227_LECTOR_%20Joint%20Optimization%20of%20Scientific%20Reasoning%20Graphs%20and%20Introduction%20Generation/tables/cda944b9d7d109299f2f53601ae36eb5596ab0fdf04ed8d528d77e5a0196c530.jpg)

![d563cea6184874d837e2a6ad01a1d968ba77d59256a02ac392d3dc591b1c8282.jpg](../icml_results/5227_LECTOR_%20Joint%20Optimization%20of%20Scientific%20Reasoning%20Graphs%20and%20Introduction%20Generation/tables/d563cea6184874d837e2a6ad01a1d968ba77d59256a02ac392d3dc591b1c8282.jpg)

![e1eff988b880621c3d6a6cea4e7679f083308b17ae9850dee562d45aba335045.jpg](../icml_results/5227_LECTOR_%20Joint%20Optimization%20of%20Scientific%20Reasoning%20Graphs%20and%20Introduction%20Generation/tables/e1eff988b880621c3d6a6cea4e7679f083308b17ae9850dee562d45aba335045.jpg)

![e53bcf0ab0bd6600c9057e2b5cc0bc5316498c53ed71c9812db5b563648f5887.jpg](../icml_results/5227_LECTOR_%20Joint%20Optimization%20of%20Scientific%20Reasoning%20Graphs%20and%20Introduction%20Generation/tables/e53bcf0ab0bd6600c9057e2b5cc0bc5316498c53ed71c9812db5b563648f5887.jpg)

![e6282fbbdbf8658e615ca8c9f7de4e35fe6d4dfd42b1b71ab8f259f1f5e1ac2c.jpg](../icml_results/5227_LECTOR_%20Joint%20Optimization%20of%20Scientific%20Reasoning%20Graphs%20and%20Introduction%20Generation/tables/e6282fbbdbf8658e615ca8c9f7de4e35fe6d4dfd42b1b71ab8f259f1f5e1ac2c.jpg)

![ee4f0cc522fd73625a27e94216b1305ac1ae0a771f0f047b9c30cea527b186bc.jpg](../icml_results/5227_LECTOR_%20Joint%20Optimization%20of%20Scientific%20Reasoning%20Graphs%20and%20Introduction%20Generation/tables/ee4f0cc522fd73625a27e94216b1305ac1ae0a771f0f047b9c30cea527b186bc.jpg)

<a id="paper-5180"></a>

## Memory-Efficient LLM Training with Dynamic Sparsity: From Stability to Practical Scaling


### Images

![11e4a0aa508dead6830e931dd40117204a7cb68c6b633ef9e59d3ea1048b5aa1.jpg](../icml_results/5228_Memory-Efficient%20LLM%20Training%20with%20Dynamic%20Sparsity_%20From%20Stability%20to%20Practical%20Scaling/images/11e4a0aa508dead6830e931dd40117204a7cb68c6b633ef9e59d3ea1048b5aa1.jpg)

![1459c2743b21e907191fe2cd915e6ee353484d6fd30fb5f04c857d51b69eaa5c.jpg](../icml_results/5228_Memory-Efficient%20LLM%20Training%20with%20Dynamic%20Sparsity_%20From%20Stability%20to%20Practical%20Scaling/images/1459c2743b21e907191fe2cd915e6ee353484d6fd30fb5f04c857d51b69eaa5c.jpg)

![1e34022dd6462d61a82d6596c1ba772262745596542e2783f1a70c03b2550bf5.jpg](../icml_results/5228_Memory-Efficient%20LLM%20Training%20with%20Dynamic%20Sparsity_%20From%20Stability%20to%20Practical%20Scaling/images/1e34022dd6462d61a82d6596c1ba772262745596542e2783f1a70c03b2550bf5.jpg)

![53235cddfe17ef7df7cbcb180434c534a552d16c5300dc30b2ff34e1c08f2c82.jpg](../icml_results/5228_Memory-Efficient%20LLM%20Training%20with%20Dynamic%20Sparsity_%20From%20Stability%20to%20Practical%20Scaling/images/53235cddfe17ef7df7cbcb180434c534a552d16c5300dc30b2ff34e1c08f2c82.jpg)

![a3f2bcf7b201d409a18995062189b7fabeb8fec9519f61f6c5b944570161c4f4.jpg](../icml_results/5228_Memory-Efficient%20LLM%20Training%20with%20Dynamic%20Sparsity_%20From%20Stability%20to%20Practical%20Scaling/images/a3f2bcf7b201d409a18995062189b7fabeb8fec9519f61f6c5b944570161c4f4.jpg)

![ac1c39dbc8c0840a27ea90022ecdee5aa81d76be04786fb7c1028d64e6b56c7f.jpg](../icml_results/5228_Memory-Efficient%20LLM%20Training%20with%20Dynamic%20Sparsity_%20From%20Stability%20to%20Practical%20Scaling/images/ac1c39dbc8c0840a27ea90022ecdee5aa81d76be04786fb7c1028d64e6b56c7f.jpg)

![e25c39aa3845d3fb331e7f08cf1f9aa1b275b1ac13b88b101d9f476950a0c769.jpg](../icml_results/5228_Memory-Efficient%20LLM%20Training%20with%20Dynamic%20Sparsity_%20From%20Stability%20to%20Practical%20Scaling/images/e25c39aa3845d3fb331e7f08cf1f9aa1b275b1ac13b88b101d9f476950a0c769.jpg)

### Tables

![7017fd8279a30db8772dd894719454e41b2e34a4f3ecdfdd9ca9878afd56f667.jpg](../icml_results/5228_Memory-Efficient%20LLM%20Training%20with%20Dynamic%20Sparsity_%20From%20Stability%20to%20Practical%20Scaling/tables/7017fd8279a30db8772dd894719454e41b2e34a4f3ecdfdd9ca9878afd56f667.jpg)

![794f487cdf98c0b76851ba07e7a13298c1ed1f711bf7c0879a79413a2f7cb7c8.jpg](../icml_results/5228_Memory-Efficient%20LLM%20Training%20with%20Dynamic%20Sparsity_%20From%20Stability%20to%20Practical%20Scaling/tables/794f487cdf98c0b76851ba07e7a13298c1ed1f711bf7c0879a79413a2f7cb7c8.jpg)

![8d9cbb0ff2a2733f2ab5b5be70790bec6e96e76652e8cdc26b944f9f2a707805.jpg](../icml_results/5228_Memory-Efficient%20LLM%20Training%20with%20Dynamic%20Sparsity_%20From%20Stability%20to%20Practical%20Scaling/tables/8d9cbb0ff2a2733f2ab5b5be70790bec6e96e76652e8cdc26b944f9f2a707805.jpg)

![de34f14dfcd51336eeb3f3b8afafa98bf12e1328025f20beec17749f8fca12f5.jpg](../icml_results/5228_Memory-Efficient%20LLM%20Training%20with%20Dynamic%20Sparsity_%20From%20Stability%20to%20Practical%20Scaling/tables/de34f14dfcd51336eeb3f3b8afafa98bf12e1328025f20beec17749f8fca12f5.jpg)

![de75e3c74bc18652cfe1e0cd2769d06d153b68d995b90c4f932537af78961506.jpg](../icml_results/5228_Memory-Efficient%20LLM%20Training%20with%20Dynamic%20Sparsity_%20From%20Stability%20to%20Practical%20Scaling/tables/de75e3c74bc18652cfe1e0cd2769d06d153b68d995b90c4f932537af78961506.jpg)

<a id="paper-5181"></a>

## CoverPruneGS: Coverage-Preserving Structured Pruning for Compact 3D Gaussian Splatting from Sparse-View Monocular Videos

### Images

![2647b59cbf7a1623a7e80a254feedd8257fd2c8afc840cd6d5daa720a535ca9d.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/images/2647b59cbf7a1623a7e80a254feedd8257fd2c8afc840cd6d5daa720a535ca9d.jpg)

![29295f0d4e451e7f2c1b2ad95c796a88f6ceb8a78dac63fbb2425bf2872d8ac0.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/images/29295f0d4e451e7f2c1b2ad95c796a88f6ceb8a78dac63fbb2425bf2872d8ac0.jpg)

![308083b435975649cda25e7c421f29f7d91cc4b9763693d013a575de37dfe344.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/images/308083b435975649cda25e7c421f29f7d91cc4b9763693d013a575de37dfe344.jpg)

![3e592019df1b2eb83dd0ca1edcf9a302a2010ff1d56c6e8494030e13a539db6e.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/images/3e592019df1b2eb83dd0ca1edcf9a302a2010ff1d56c6e8494030e13a539db6e.jpg)

![570065f0c13b3a43e0bc79efb316df83739853ed1c2fd30fadc3a2020f679d82.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/images/570065f0c13b3a43e0bc79efb316df83739853ed1c2fd30fadc3a2020f679d82.jpg)

![b4aa0f621c44eb481a872379512cdc325b4f5c75e362387f0583efee0947fe7e.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/images/b4aa0f621c44eb481a872379512cdc325b4f5c75e362387f0583efee0947fe7e.jpg)

![eeb2e3d76158e2f7becc215a4145c750ce6758eb4696788b4353a43a40aaa04f.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/images/eeb2e3d76158e2f7becc215a4145c750ce6758eb4696788b4353a43a40aaa04f.jpg)

### Tables

![2913f8ec719414fab3c533c44f1ccd774d82ec78d74ebbe28c94f25f6b3dfd22.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/tables/2913f8ec719414fab3c533c44f1ccd774d82ec78d74ebbe28c94f25f6b3dfd22.jpg)

![29aa4f3572f91bc8ea522a00f58d8b01b99a63f188088bdde23626cd410f85ce.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/tables/29aa4f3572f91bc8ea522a00f58d8b01b99a63f188088bdde23626cd410f85ce.jpg)

![2e138a9a17e070f07dc537e4440fa2d59696c4e9fd81ddc341f2c968992ad4bd.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/tables/2e138a9a17e070f07dc537e4440fa2d59696c4e9fd81ddc341f2c968992ad4bd.jpg)

![61c61c262ca281f40982ba1acf36d2c5842f040f33b63cf66d6eb48df459230a.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/tables/61c61c262ca281f40982ba1acf36d2c5842f040f33b63cf66d6eb48df459230a.jpg)

![62230c131a060be8188375eb0a10eb5a7d669a11675eedc85d09dacac52c89d4.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/tables/62230c131a060be8188375eb0a10eb5a7d669a11675eedc85d09dacac52c89d4.jpg)

![700339d17e20f1de90e1b1b3b98ce9f15bc643ba91620d53032342b690ce4e61.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/tables/700339d17e20f1de90e1b1b3b98ce9f15bc643ba91620d53032342b690ce4e61.jpg)

![7f53be45eabe268e317ca4d3620a4997ce3236c466450f87fdf2cd2ab135d88c.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/tables/7f53be45eabe268e317ca4d3620a4997ce3236c466450f87fdf2cd2ab135d88c.jpg)

![c40953857b468408a0ea89a9691768f04abf121e542ba62df7bb6ed971f73a50.jpg](../icml_results/5229_CoverPruneGS_%20Coverage-Preserving%20Structured%20Pruning%20for%20Compact%203D%20Gaussian%20Splatting%20from%20Sparse-V/tables/c40953857b468408a0ea89a9691768f04abf121e542ba62df7bb6ed971f73a50.jpg)
